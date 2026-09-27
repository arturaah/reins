"""Reins tools for the chat model, implemented on the dashboard's own objects.

The dashboard serves these at POST /api/tools/<name> (loopback only, separate tool token) and
tools/reins_mcp.py exposes them to the Claude and Codex CLIs over MCP. Every tool observes,
plans or previews in simulation. None publishes to the robot: there is deliberately no
execute tool and no gesture-button tool. There is no depth estimation, so nothing locates or
supplies metric object positions; camera-guided steps are requested through the shared pipeline.
"""
import copy
import re
import threading
import time

from core.generated_motion import validate_trajectory
from core.tool_specs import CAMERAS, TOOL_NAMES

OBJECT_NAME = re.compile(r"[a-z0-9][a-z0-9 '\-]{0,59}")


class ToolError(ValueError):
    """Shown to the model as the tool's error result."""


class ReinsTools:
    PLAN_TIMEOUT = 420.0
    LOG = 30

    def __init__(self, detector, sources, planner, show_proposal, simulation_status, camera_status=None):
        """sources: camera name -> callable returning (rgb, received_monotonic, frame_id, observation|None)."""
        self.detector, self.sources, self.planner = detector, sources, planner
        self.show_proposal, self.simulation_status = show_proposal, simulation_status
        self.camera_status = camera_status or (lambda: {})
        self.lock = threading.Lock()
        self.log = []
        self.pipeline = None

    # ---- dispatch -------------------------------------------------------------------------
    def call(self, name, arguments):
        if name not in TOOL_NAMES:
            raise ToolError(f'Unknown tool {name!r}')
        if not isinstance(arguments, dict):
            raise ToolError('Tool arguments must be an object')
        started = time.time()
        try:
            result = getattr(self, name)(**arguments)
            # A plan tool that ran but was blocked is shown as a failure, with the planner's reason.
            ok = result.get('state') in ('proposed', 'previewed') if name.startswith('plan_') else True
            self._record(name, arguments, ok, self._summary(name, result), started)
            return result
        except TypeError as exc:
            self._record(name, arguments, False, 'invalid arguments', started)
            raise ToolError(f'Invalid arguments for {name}: {exc}') from None
        except (ValueError, KeyError) as exc:
            self._record(name, arguments, False, str(exc)[:160], started)
            raise ToolError(str(exc)[:800]) from None

    def _record(self, name, arguments, ok, summary, started):
        with self.lock:
            self.log.append({'tool': name, 'ok': ok, 'summary': summary, 'at': round(started, 3),
                             'duration_s': round(time.time() - started, 2),
                             'arguments': {k: v for k, v in arguments.items() if k != 'waypoints'}})
            del self.log[:-self.LOG]

    def recent(self):
        with self.lock:
            return copy.deepcopy(self.log)

    @staticmethod
    def _summary(name, result):
        if name == 'detect_objects':
            return f"{len(result['objects'])} object(s) on {result['camera']}"
        if name == 'plan_hand_path':
            return f"{result['state']}: {result['message'][:100]}"
        if name == 'preview_plan':
            return 'shown in simulation'
        return 'ok'

    # ---- tools ----------------------------------------------------------------------------
    def get_robot_context(self):
        cameras = self.camera_status()
        planner = self.planner.status()
        return {'cameras': cameras,
                'object_positions': 'not available: Reins has no depth estimation, so objects are seen in 2D only '
                                    'and camera-guided steps must not invent metric object coordinates',
                'detector': {'name': getattr(self.detector, 'name', 'unknown'), 'available': bool(self.detector.available),
                             'open_vocabulary': bool(getattr(self.detector, 'open_vocabulary', False))},
                'motion_authoring': self.planner.motion_context(),
                'simulation': {k: self.simulation_status().get(k) for k in ('plan', 'name', 'playing')},
                'planner': {'state': planner['state'], 'message': planner['message']},
                'physical_execution': 'Tools cannot approve or execute. The operator reviews and approves in dashboard or paired glasses.',
                'control': self.pipeline.status() if self.pipeline else None}

    def detect_objects(self, camera, labels=None):
        if camera not in CAMERAS:
            raise ToolError(f'Camera must be one of {", ".join(CAMERAS)}')
        if camera not in self.sources:
            raise ToolError(f'The {camera} camera is not configured on this dashboard.')
        labels = self._labels(labels)
        if not self.detector.available:
            raise ToolError('The object detector is not installed. Run tools/detect_objects.py --download.')
        rgb, received, _, _ = self.sources[camera]()
        objects = self.detector.detect(rgb, labels=labels)[:30]
        h, w = rgb.shape[:2]
        return {'camera': camera, 'detector': getattr(self.detector, 'name', 'unknown'),
                'frame_age_s': round(max(0.0, time.monotonic() - received), 2), 'image_size': [w, h],
                'searched_for': labels or 'default vocabulary',
                'objects': [{'index': i, 'label': o['label'], 'confidence': o['confidence'],
                             'bbox': [round(v, 4) for v in o['bbox']]} for i, o in enumerate(objects)],
                'note': 'bbox is normalized [x0,y0,x1,y1] in the image; it is not a position in metres.'}

    def plan_hand_path(self, name, arm, waypoints, return_to_start):
        draft = validate_trajectory({'name': name, 'arm': arm, 'frame': 'robot_base', 'waypoints': waypoints,
                                     'return_to_start': return_to_start})
        self.planner.submit(draft['name'], 'auto', trajectory=draft)
        return self._await_plan()

    def request_visual_guidance(self, task, arm):
        if not self.pipeline:
            raise ToolError('Visual fallback is not connected')
        if not isinstance(task, str) or not 1 <= len(task) <= 1000 or arm not in ('left', 'right'):
            raise ToolError('Supply a task and a single arm')
        self.pipeline.cfg['robot']['arm'] = arm
        # The primary planner still runs first; missing metric context activates the visual policy.
        self.planner.submit(task, 'auto')
        return {'state': 'planning', 'message': 'Primary planning started; the visual harness can gather more context if needed. Every resulting move needs operator review.'}

    def preview_plan(self, proposal_id):
        if not isinstance(proposal_id, str):
            raise ToolError('proposal_id must be a string')
        if self.pipeline:
            self.pipeline.show_primary(None, proposal_id)
        status = self.planner.show_once(proposal_id, self.show_proposal)
        return {'proposal_id': proposal_id, 'state': status['state'], 'message': status['message']}

    # ---- helpers --------------------------------------------------------------------------
    def _await_plan(self):
        deadline = time.monotonic() + self.PLAN_TIMEOUT
        status = self.planner.status()
        while status['state'] == 'planning':
            if time.monotonic() > deadline:
                self.planner.cancel()
                raise ToolError('Planning took too long and was cancelled.')
            time.sleep(.05)
            status = self.planner.status()
        result = {'proposal_id': status.get('id'), 'state': status['state'], 'message': status['message'],
                  'steps': [e['message'] for e in status.get('events', [])][-8:]}
        if status['state'] == 'blocked' and status.get('retryable'):
            result.update(retryable=True, failures=copy.deepcopy(status.get('failures', [])),
                          next_step='Revise the waypoints using these failures and call plan_hand_path again. '
                                    'Keep the original gesture, arm and return setting; make at most three attempts total.')
        if status['state'] in ('proposed', 'previewed'):
            validation = status.get('validation') or {}
            result.update(target=status.get('target'), duration_s=status.get('duration'),
                          validation={k: v for k, v in validation.items() if not isinstance(v, (list, dict)) or k == 'checks'},
                          execution_allowed=False, requires_operator_approval=True,
                          next_step='Call preview_plan to show it. The human can approve the resolved proposal in the dashboard or paired glasses; tools cannot approve.')
        return result

    @staticmethod
    def _labels(labels):
        if labels is None:
            return None
        if not isinstance(labels, list) or not 1 <= len(labels) <= 30:
            raise ToolError('labels must be a list of 1 to 30 object names')
        return [ReinsTools._object_name(l) for l in labels]

    @staticmethod
    def _object_name(value):
        text = re.sub(r'\s+', ' ', str(value).strip().lower())
        text = re.sub(r'^(?:the|a|an) ', '', text)
        if not OBJECT_NAME.fullmatch(text) or re.search(r'\b(?:then|and|after|before)\b', text):
            raise ToolError('Object names are 1-60 letters, digits, spaces or hyphens, one object at a time')
        return text
