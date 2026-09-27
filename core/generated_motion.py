"""Model-authored hand paths: bounded data, compiled locally with IK, never code."""
import copy
import math

import mujoco
import numpy as np

from core.ik import plan_from_waypoints
from core.motion_validation import slow_acceleration


TRAJECTORY_SCHEMA = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string', 'minLength': 1, 'maxLength': 120},
        'arm': {'type': 'string', 'enum': ['left', 'right']},
        'frame': {'type': 'string', 'enum': ['robot_base']},
        'waypoints': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': {
            'type': 'object', 'properties': {
                'position_m': {'type': 'array', 'minItems': 3, 'maxItems': 3,
                               'items': {'type': 'number', 'minimum': -2, 'maximum': 2}},
                'hold_s': {'type': 'number', 'minimum': 0, 'maximum': 5},
            }, 'required': ['position_m', 'hold_s'], 'additionalProperties': False}},
        'return_to_start': {'type': 'boolean'},
    },
    'required': ['name', 'arm', 'frame', 'waypoints', 'return_to_start'],
    'additionalProperties': False,
}


class TrajectoryRejected(ValueError):
    """A valid draft whose geometry or timing could not be compiled."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def _number(value, lo, hi):
    return type(value) in (int, float) and lo <= value <= hi and math.isfinite(value)


def validate_trajectory(draft):
    """Validate provider/browser data before allocating IK work or retaining history."""
    if not isinstance(draft, dict) or set(draft) != set(TRAJECTORY_SCHEMA['required']):
        raise ValueError('Invalid generated trajectory fields')
    if not isinstance(draft['name'], str) or not 1 <= len(draft['name'].strip()) <= 120:
        raise ValueError('Generated trajectory needs a name of 1–120 characters')
    if draft['arm'] not in ('left', 'right') or draft['frame'] != 'robot_base':
        raise ValueError('Generated trajectory must select one arm in robot_base coordinates')
    if type(draft['return_to_start']) is not bool:
        raise ValueError('Invalid return-to-start setting')
    points = draft['waypoints']
    if not isinstance(points, list) or not 1 <= len(points) <= 16:
        raise ValueError('Generated trajectory needs 1–16 waypoints')
    for point in points:
        if not isinstance(point, dict) or set(point) != {'position_m', 'hold_s'}:
            raise ValueError('Invalid generated waypoint fields')
        position = point['position_m']
        if (not isinstance(position, list) or len(position) != 3 or
                not all(_number(v, -2, 2) for v in position) or position[2] < 0):
            raise ValueError('Waypoint position must be three finite metres above the floor')
        if not _number(point['hold_s'], 0, 5):
            raise ValueError('Waypoint hold must be between 0 and 5 seconds')
    result = copy.deepcopy(draft)
    result['name'] = result['name'].strip()
    return result


def validate_pose(ik, pose, measured=False):
    if measured:
        required = {ik.model.joint(i).name for i in range(ik.model.njnt)
                    if 'wrist_pitch' not in ik.model.joint(i).name
                    and 'wrist_yaw' not in ik.model.joint(i).name
                    and ik.model.joint(i).name != 'waist_pitch_joint'}
        if not required.issubset(pose):
            raise ValueError('Observation is missing measured model joints: ' + ', '.join(sorted(required-set(pose))))
    for name, value in pose.items():
        joint = ik.model.joint(name)
        if not _number(value, -math.pi*2, math.pi*2) or (joint.limited and not joint.range[0] <= value <= joint.range[1]):
            raise ValueError('Invalid starting joint pose: ' + name)


def motion_context(ik, pose):
    """Geometry comes from the same model as the preview, not a language-model guess."""
    validate_pose(ik, pose)
    data = mujoco.MjData(ik.model)
    for name, value in pose.items():
        data.qpos[ik.model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(ik.model, data)
    torso = ik.model.body('torso_link').id
    head = data.xpos[torso] + data.xmat[torso].reshape(3, 3) @ np.array([0., 0., .35])
    return {
        'frame': 'robot_base: metres; x forward, y left, z up; origin at floor under pelvis',
        'pose_source': 'current simulation pose, not a measured physical robot pose',
        'head_envelope_center_m': head.round(4).tolist(),
        'minimum_head_clearance_m': .17,
        'arms': {side: {
            'hand_position_m': data.site_xpos[ik.model.site(f'{side}_hand_preview').id].round(4).tolist(),
            'shoulder_position_m': data.xpos[ik.model.body(f'{side}_shoulder_roll_link').id].round(4).tolist(),
        } for side in ('left', 'right')},
        'capabilities': 'New single-arm hand paths and compound gestures, 1–16 waypoints, optional pauses and return. '
                        'Other joints stay fixed. No finger articulation, wrist orientation, walking, grasping or contact. '
                        'Keep paths outside the head and torso; IK and full-path collision checks may reject them.',
    }


def compile_trajectory(ik, draft, pose, progress=None):
    """Interpolate each authored segment from the supplied pose, retaining pauses."""
    draft = validate_trajectory(draft)
    validate_pose(ik, pose)
    side = draft['arm']
    current = dict(pose)
    points = list(draft['waypoints'])
    if draft['return_to_start']:
        start, _ = ik.fk(side, pose, pose)
        points.append({'position_m': start.tolist(), 'hold_s': 0})
    frames, targets, errors = [], [], []
    for i, point in enumerate(points):
        if progress:
            progress(f'Solving generated waypoint {i+1}/{len(points)}.')
        try:
            segment, _ = plan_from_waypoints(ik, side, [point['position_m']], pose=current,
                                             name=draft['name'], max_vel=.3, hold_s=point['hold_s'])
        except ValueError as exc:
            returning = i == len(draft['waypoints'])
            start, _ = ik.fk(side, current, current)
            label = 'Return to start' if returning else f'Authored waypoint {i+1}'
            raise TrajectoryRejected(f'{label}: {exc}', {
                'waypoint_number': i+1, 'return_segment': returning,
                'segment_start_m': start.round(4).tolist(),
                'target_position_m': point['position_m'],
            }) from exc
        offset = frames[-1]['time_s'] if frames else 0.
        for frame in segment['keyframes'][1 if frames else 0:]:
            frame['time_s'] = round(frame['time_s'] + offset, 6)
            frames.append(frame)
        current.update(frames[-1]['joint_targets_rad'])
        targets.extend(segment['ik']['targets_m'])
        errors.extend(segment['ik']['position_error_mm'])
        if frames[-1]['time_s'] > 180:
            raise TrajectoryRejected('Generated trajectory exceeds the 180-second preview limit')
    plan = {'schema_version': 1, 'name': draft['name'], 'duration_s': frames[-1]['time_s'],
            'keyframes': frames, 'held_joints_rad': {k: v for k, v in pose.items() if k not in ik.joint_names(side)},
            'ik': {'side': side, 'frame': 'robot_base', 'targets_m': targets, 'position_error_mm': errors}}
    slow_acceleration(plan)
    if plan['duration_s'] > 180:
        raise TrajectoryRejected('Generated trajectory exceeds the 180-second preview limit')
    return plan
