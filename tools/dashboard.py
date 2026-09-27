"""Local R1 dashboard: live views, one-time generated previews and firmware gesture buttons.

Run: .venv/bin/python tools/dashboard.py [--iface en6]
Explicit operator approval sends validated trajectories through the shared robot bridge.
The dashboard does not record or replay trajectories.
"""
from __future__ import annotations
import argparse
import errno
import io
import json
import math
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Select before importing MuJoCo. Override for systems that need OSMesa/GLFW.
os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
import mujoco
import numpy as np
from PIL import Image
from sim.preview import prepare_plan
from spectacles.plan_feed import hand_paths
from core.prompt_planner import PromptPlanner
from core.object_detection import DEFAULT_MODEL, make_detector
from core.detection_stream import DetectionStream
from core.perception import Observation
from core.dashboard_chat import DashboardChat
from core.r1_gestures import GestureController
from core.reins_tools import ReinsTools, ToolError
from core.codex_chat import ToolLink
from core.robot_pipeline import RobotPipeline
from core.glasses_bridge import GlassesBridge

ASSETS = ROOT / 'tools/dashboard'


class CameraFeed:
    """One upstream reader per feed, bounded memory and stale-frame expiry."""
    def __init__(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.jpg = b''
        self.updated = 0.0
        self.error = 'Not configured' if not url else 'Connecting'
        if url:
            threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=4) as response:
                    buffer = b''
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        buffer += chunk
                        while True:
                            start = buffer.find(b'\xff\xd8')
                            end = buffer.find(b'\xff\xd9', max(0, start))
                            if start < 0 or end < 0:
                                break
                            jpg, buffer = buffer[start:end + 2], buffer[end + 2:]
                            # Validate the payload so connection status means a usable image.
                            with Image.open(io.BytesIO(jpg)) as im:
                                im.verify()
                            with self.lock:
                                self.jpg = jpg
                                self.updated = time.monotonic()
                            self.error = ''
                        if len(buffer) > 8 * 1024 * 1024:
                            raise ValueError('Stream frame exceeded 8 MB')
                self.error = 'Stream ended; reconnecting'
            except Exception:
                self.error = 'Feed unavailable; retrying'
            time.sleep(1)

    def detection_snapshot(self):
        with self.lock:
            jpg, received_at = self.jpg, self.updated
        if not jpg or time.monotonic()-received_at >= 3:
            raise ValueError('Camera offline; waiting for a current frame.')
        with Image.open(io.BytesIO(jpg)) as image:
            if max(image.size) > 4096:
                raise ValueError('Detection supports camera images up to 4096 pixels per axis')
            rgb = np.array(image.convert('RGB'))
        return rgb, received_at, str(received_at), None

    def status(self):
        age = time.monotonic() - self.updated if self.updated else None
        return {'configured': bool(self.url), 'online': age is not None and age < 3,
                'age': round(age, 1) if age is not None else None, 'detail': self.error}


class Simulation:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'sim/models/r1/scene_fixed_base.xml'))
        # A navy-black stage (matching the page) keeps the model and the neon hand paths readable.
        for index in range(self.model.ntex):
            start = self.model.tex_adr[index]
            size = self.model.tex_width[index] * self.model.tex_height[index] * self.model.tex_nchannel[index]
            pixels = self.model.tex_data[start:start + size].reshape(-1, self.model.tex_nchannel[index])
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, index) == 'groundplane':
                shade = pixels[:, :3].mean(axis=1, keepdims=True) * .28
                pixels[:, :3] = np.clip(shade + np.array([14, 16, 26]), 0, 255).astype(np.uint8)
            else:
                pixels[:, :3] = [7, 8, 14]
        self.model.vis.rgba.haze[:] = [.035, .04, .07, 1]
        self.model.mat_reflectance[:] = .04
        self.model.vis.global_.offwidth = 1000
        self.model.vis.global_.offheight = 760
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.05, 0, .72]
        self.camera.distance = 2.35
        self.camera.azimuth = 145
        self.camera.elevation = -16
        self.jpg = b''
        self.error = ''
        self.updated = 0.0
        self.position = 0.0
        self.playing = False
        self.show_paths = True
        self.paths = {'left': [], 'right': []}
        self.key = None
        self.plan = {'name':'No preview', 'duration_s':0, 'held_joints_rad':{},
                     '_joint_ids':{}, '_times':[], 'keyframes':[],
                     'prompt_proposal':{'source':'idle'}, 'scene_boxes':[]}

    def show_proposal(self, source, proposal_id):
        if source.get('preview_only') is not True or source.get('prompt_proposal',{}).get('id') != proposal_id:
            raise ValueError('Only a validated generated proposal can be shown.')
        plan = prepare_plan(self.model, source)
        duration = float(plan['duration_s'])
        if not math.isfinite(duration) or duration <= 0 or not all(math.isfinite(t) for t in plan['_times']):
            raise ValueError('Invalid preview timing')
        for name, value in plan.get('held_joints_rad', {}).items():
            joint = self.model.joint(name)
            if not math.isfinite(float(value)) or (joint.limited and not joint.range[0] <= value <= joint.range[1]):
                raise ValueError('Invalid held joint')
        paths = hand_paths(self.model, plan, 90)
        with self.lock:
            if self.key == proposal_id:
                raise ValueError('This preview has already been shown.')
            self.plan, self.key, self.paths = plan, proposal_id, paths
            self.position, self.playing = 0., True

    def advance(self, elapsed):
        with self.lock:
            if self.playing:
                self.position = min(self.plan['duration_s'], self.position + max(0., elapsed))
                if self.position >= self.plan['duration_s']:
                    self.playing = False

    def control(self, command):
        with self.lock:
            if command.get('action') == 'stop':
                self.playing = False
            else:
                raise ValueError('Only stopping the current simulation preview is supported.')

    def status(self):
        with self.lock:
            return {'plan':self.key, 'name':self.plan['name'], 'time':self.position,
                    'duration':self.plan['duration_s'], 'playing':self.playing,
                    'ready':bool(self.jpg) and time.monotonic()-self.updated<3, 'error':self.error}

    def run(self):
        try:
            # Renderer and GL context live on the same thread for CGL/EGL portability.
            with mujoco.Renderer(self.model, height=540, width=960) as renderer:
                last = time.monotonic()
                while True:
                    start = time.monotonic()
                    with self.lock:
                        self.advance(start - last)
                        last = start
                        mujoco.mj_resetData(self.model, self.data)
                        for name, value in self.plan.get('held_joints_rad', {}).items():
                            self.data.qpos[self.model.joint(name).qposadr[0]] = value
                        for name, jid in self.plan['_joint_ids'].items():
                            self.data.qpos[self.model.jnt_qposadr[jid]] = np.interp(self.position, self.plan['_times'],
                                [f['joint_targets_rad'][name] for f in self.plan['keyframes']])
                        mujoco.mj_forward(self.model, self.data)
                        renderer.update_scene(self.data, camera=self.camera)
                        if self.plan.get('prompt_proposal'):
                            # Replace demonstration props with this proposal's collision scene.
                            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0:
                                    if self.model.geom(geom.objid).name in ('pickup_table', 'pickup_cube_geom'):
                                        geom.rgba[3] = 0
                            for box in self.plan.get('scene_boxes', []):
                                if renderer.scene.ngeom >= renderer.scene.maxgeom: break
                                lo, hi = np.array(box['min']), np.array(box['max'])
                                geom = renderer.scene.geoms[renderer.scene.ngeom]
                                color = [.5,.58,.48,.45] if box['name'] == 'table' else [.85,.95,.6,.8]
                                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, (hi-lo)/2, (hi+lo)/2,
                                                   np.eye(3).ravel(), np.array(color,dtype=np.float32))
                                renderer.scene.ngeom += 1
                            for key, color in [('surface_m',[1,.5,.2,1]),('goal_m',[.4,1,.8,1])]:
                                if self.plan['prompt_proposal'].get(key) is None: continue
                                if renderer.scene.ngeom >= renderer.scene.maxgeom: break
                                geom = renderer.scene.geoms[renderer.scene.ngeom]
                                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.012,0,0]),
                                                   np.array(self.plan['prompt_proposal'][key]), np.eye(3).ravel(), np.array(color,dtype=np.float32))
                                renderer.scene.ngeom += 1
                        if self.show_paths:
                            # colours match --left-hand / --right-hand in dashboard/style.css
                            for side, rgba in [('left', [.10, .85, 1, .9]), ('right', [1, .25, .71, .9])]:
                                for a, b in zip(self.paths[side], self.paths[side][1:]):
                                    if np.linalg.norm(np.array(b) - a) < 1e-5 or renderer.scene.ngeom >= renderer.scene.maxgeom:
                                        continue
                                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
                                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .004, a, b)
                                    renderer.scene.ngeom += 1
                        frame = renderer.render()
                    out = io.BytesIO()
                    Image.fromarray(frame).save(out, 'JPEG', quality=88)
                    self.jpg = out.getvalue()
                    self.updated = time.monotonic()
                    time.sleep(max(0, 1 / 15 - (time.monotonic() - start)))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            print('MuJoCo renderer:', self.error, flush=True)


class TextPoller:
    """Polls a small plain-text status URL (the twin server's /status) once a second."""
    def __init__(self, url):
        self.url, self.text = url, ''
        if url:
            threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=2) as response:
                    self.text = response.read(512).decode(errors='replace').strip()
            except Exception:
                self.text = ''
            time.sleep(1)



def bind_dashboard_server(port=None):
    """Reserve the listener before loading models or starting background workers.

    A default launch tries 8090–8099. An explicit port never silently changes;
    port 0 asks the OS for a free port. The real handler is installed before serving.
    """
    candidates = range(8090, 8100) if port is None else (port,)
    for candidate in candidates:
        try:
            return ThreadingHTTPServer(('127.0.0.1', candidate), BaseHTTPRequestHandler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if port is not None:
                raise OSError(errno.EADDRINUSE,
                    f'Port {port} is already in use. Open http://localhost:{port} to check the existing service, '
                    'or run .venv/bin/python tools/dashboard.py --port 0 to start on a free port.') from None
    raise OSError(errno.EADDRINUSE,
        'Ports 8090–8099 are already in use. Run .venv/bin/python tools/dashboard.py --port 0 to choose a free port.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=None, help='HTTP port; default tries 8090–8099, or use 0 for any free port')
    parser.add_argument('--head', default='http://127.0.0.1:8081/cam')
    parser.add_argument('--left-wrist', default='http://127.0.0.1:8080/cam/0')
    parser.add_argument('--right-wrist', default='http://127.0.0.1:8080/cam/2')
    parser.add_argument('--glasses', default='', help='Glasses MJPEG/JPEG video URL, if available')
    parser.add_argument('--twin', default='http://127.0.0.1:8082/twin', help='Live twin MJPEG URL (tools/cockpit.py); empty to disable')
    parser.add_argument('--iface', default='en6', help='Network interface connected to the R1 gesture service')
    parser.add_argument('--observation', type=Path, help='Atomically updated calibrated RGB/depth observation NPZ')
    parser.add_argument('--sim', action='store_true', help='Disable hardware control, robot feeds and calibrated observations')
    parser.add_argument('--voice-url', default='', help='Optional local voice service, e.g. http://127.0.0.1:8770/')
    parser.add_argument('--detector', choices=('auto', 'omdet', 'nanodet'), default='auto',
                        help='auto: open-vocabulary OmDet-Turbo if installed, else NanoDet (80 COCO classes)')
    parser.add_argument('--detector-model', type=Path, default=DEFAULT_MODEL, help='Verified NanoDet ONNX model (fallback)')
    parser.add_argument('--detection-confidence', type=float, default=None, help='Local detection threshold (0.1–0.95; default 0.4)')
    parser.add_argument('--no-chat-tools', action='store_true', help='Do not give the chat model the Reins tools (detection, planning, preview)')
    parser.add_argument('--chat-backend', choices=('claude', 'codex', 'openai'), default=os.environ.get('REINS_CHAT_BACKEND', 'codex'), help='Assistant that answers first (default: signed-in Codex CLI); switch any time in the chat panel')
    parser.add_argument('--harness-config', type=Path, help='Shared harness configuration')
    parser.add_argument('--glasses-port', type=int, default=8765, help='Authenticated AR review port (0 chooses a free port)')
    parser.add_argument('--glasses-host', default='0.0.0.0', help='AR listener; paired clients only')
    args = parser.parse_args()
    if args.voice_url:
        voice = urlparse(args.voice_url)
        if (voice.scheme != 'http' or voice.hostname not in ('127.0.0.1', 'localhost')
                or voice.username or voice.password or voice.path not in ('', '/')
                or voice.query or voice.fragment):
            parser.error('--voice-url must be a local http://localhost:PORT/ address')
    if args.sim:
        args.head = args.left_wrist = args.right_wrist = args.glasses = args.twin = ''
        args.observation = None
    for url in (args.head, args.left_wrist, args.right_wrist, args.glasses, args.twin):
        if url and urlparse(url).scheme not in ('http', 'https'):
            parser.error('Feed URLs must use http:// or https://')
    if args.port is not None and not 0 <= args.port <= 65535:
        parser.error('Port must be between 0 and 65535')
    requested_port = args.port
    try:
        server = bind_dashboard_server(requested_port)
    except OSError as exc:
        parser.exit(2, f'Dashboard could not start: {exc.strerror or exc}\n')
    args.port = server.server_address[1]
    if requested_port is None and args.port != 8090:
        print(f'Port 8090 is occupied; using http://localhost:{args.port} for this dashboard.', flush=True)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sim = Simulation()
    def preview_pose():
        with sim.lock:
            return {sim.model.joint(i).name: float(sim.data.qpos[sim.model.jnt_qposadr[i]]) for i in range(sim.model.njnt)}
    detector = make_detector(args.detector, args.detection_confidence, nanodet_model=args.detector_model)
    print(f'Object detection: {detector.name}' + ('' if detector.available else ' (model not installed: tools/detect_objects.py --download)'), flush=True)
    prompt_planner = PromptPlanner(args.observation, preview_pose=preview_pose, detector=detector)
    feeds = {name: CameraFeed(url) for name, url in {'head': args.head, 'left': args.left_wrist,
             'right': args.right_wrist, 'glasses': args.glasses, 'twin': args.twin}.items()}
    detection_sources = {name: feed.detection_snapshot for name, feed in feeds.items() if name != 'twin'}
    if args.observation:
        def observation_snapshot():
            observation = Observation.load(args.observation)
            received_at = time.monotonic() - (time.time() - observation.captured_at)
            return observation.rgb, received_at, str(observation.captured_at), observation
        detection_sources['observation'] = observation_snapshot
    detections = DetectionStream(detector, detection_sources)
    token = secrets.token_urlsafe(24)
    # The twin server reports rt/lowstate health next to its stream.
    robot_status = TextPoller(args.twin.rsplit('/', 1)[0] + '/status' if args.twin else '')

    gestures = GestureController(args.iface)
    if not args.sim:
        gestures.command({'action':'refresh'})  # Read-only firmware discovery; no gesture at startup.

    def chat_context():
        simulation = sim.status()
        planning = prompt_planner.status()
        detection = detections.status()
        result = detection.get('result')
        return {'simulation': {k: simulation.get(k) for k in ('plan', 'playing', 'ready')},
                'robot_gestures': gestures.status(),
                'motion_authoring': prompt_planner.motion_context(),
                'cameras': {k: {'configured': v.status()['configured'], 'online': v.status()['online']} for k, v in feeds.items() if k != 'twin'},
                'planner': {'state': planning['state'], 'message': planning['message'],
                            'configured': planning['configured'], 'context': planning.get('context'),
                            'prompt': planning.get('prompt'), 'events': planning.get('events', [])[-3:]},
                'detections': {'source': detection['source'], 'ready': detection['ready'],
                    'age_s': result['age_s'] if result else None,
                    'objects': [{'label': o['label'], 'confidence': o['confidence']} for o in result['objects'][:20]] if result else []},
                'execution_of_generated_plans': 'Human approval in dashboard or paired glasses; model tools cannot approve',
                'control': pipeline.status()}
    # Tools for the chat model (served below at /api/tools/<name>, reached through tools/reins_mcp.py).
    # They observe, plan and preview only; nothing here publishes to the robot.
    tool_token = secrets.token_urlsafe(24)
    reins_tools = ReinsTools(detector, detection_sources, prompt_planner, sim.show_proposal, sim.status,
                             camera_status=lambda: {k: v.status()['online'] for k, v in feeds.items() if k != 'twin'})
    tool_link = None if args.no_chat_tools else ToolLink(f'http://127.0.0.1:{args.port}/api/tools', tool_token)
    chat = DashboardChat(context=chat_context, backend=args.chat_backend, tools=tool_link)
    from harness.config import load as load_harness_config
    pipeline = RobotPipeline(prompt_planner, sim, feeds, args.iface, cfg=load_harness_config(args.harness_config))
    pipeline.provider = args.chat_backend
    prompt_planner.reviser_factory = chat.motion_reviser
    reins_tools.pipeline = pipeline
    reins_tools.show_proposal = pipeline.show_primary
    glasses_bridge = GlassesBridge(pipeline, args.glasses_host, args.glasses_port)


    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, mime='application/json', code=200):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed(self):
            return self.headers.get('Host') in (f'127.0.0.1:{args.port}', f'localhost:{args.port}')

        def do_GET(self):
            if not self.allowed():
                return self.send({'error': 'Local access only'}, code=403)
            path = self.path.split('?')[0]
            if path == '/api/status':
                return self.send({'simulation': sim.status(), 'feeds': {k: v.status() for k, v in feeds.items()},
                                  'mode': 'sim' if args.sim else 'hardware', 'voice_url': args.voice_url,
                                  'gestures': gestures.status(), 'robot': robot_status.text, 'prompt': prompt_planner.status(), 'detection': detections.status(), 'chat': chat.status(),
                                  'tools': reins_tools.recent()[-8:], 'pipeline': pipeline.status()})
            if path == '/api/robot':
                return self.send(pipeline.status())
            if path == '/api/glasses':
                return self.send({**pipeline.glasses, 'token': pipeline.glasses_token})
            if path == '/api/chat':
                return self.send(chat.status())
            if path == '/api/detection':
                return self.send(detections.status())
            if path == '/frame/detection':
                frame_id = parse_qs(urlparse(self.path).query).get('id', [''])[0]
                jpg = detections.image(frame_id)
                return self.send(jpg, 'image/jpeg') if jpg else self.send({'error': 'Detection frame expired'}, code=503)
            if path == '/api/prompt':
                return self.send(prompt_planner.status())
            if path == '/frame/grounding':
                jpg = prompt_planner.image
                return self.send(jpg, 'image/jpeg') if jpg else self.send({'error': 'No grounded image'}, code=404)
            if path == '/api/session':
                return self.send({'token': token})
            if path == '/api/gestures':
                return self.send(gestures.status())
            if path.startswith('/frame/'):
                key = path.removeprefix('/frame/')
                if key == 'simulation':
                    jpg = sim.jpg if sim.status()['ready'] else b''
                else:
                    feed = feeds.get(key)
                    jpg = feed.jpg if feed and feed.status()['online'] else b''
                return self.send(jpg, 'image/jpeg', 200) if jpg else self.send({'error': 'No current frame'}, code=503)
            files = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css'),
                     '/reins-mark.png': ('reins-mark.png', 'image/png'), '/favicon.ico': ('reins-mark.png', 'image/png')}
            if path in files:
                name, mime = files[path]
                return self.send((ASSETS / name).read_bytes(), mime)
            return self.send({'error': 'Not found'}, code=404)

        def do_POST(self):
            if not self.allowed():
                return self.send({'error': 'Invalid local session'}, code=403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= (32768 if self.path == '/api/chat' else 4096):
                    raise ValueError('Invalid request size')
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise ValueError('Expected object')
            except (ValueError, TypeError) as exc:
                return self.send({'error': str(exc)}, code=400)
            if self.path.startswith('/api/tools/'):
                # The chat model's tools. Own token, never the browser's; loopback only (checked above).
                if not secrets.compare_digest(self.headers.get('X-Reins-Tool-Token', ''), tool_token):
                    return self.send({'error': 'Invalid tool token'}, code=403)
                try:
                    return self.send(reins_tools.call(self.path.removeprefix('/api/tools/'), command))
                except ToolError as exc:
                    return self.send({'error': str(exc)}, code=400)
                except Exception as exc:
                    return self.send({'error': f'Tool failed: {type(exc).__name__}'}, code=500)
            if self.headers.get('X-Reins-Token') != token:
                return self.send({'error': 'Invalid local session'}, code=403)
            try:
                if self.path == '/api/robot':
                    if args.sim and command.get('action') == 'connect':
                        raise ValueError('Hardware control is disabled in simulation mode')
                    if command.get('action') == 'connect' and gestures.status()['busy']:
                        raise ValueError('Wait for the firmware gesture to finish.')
                    return self.send(pipeline.command(command))
                if self.path == '/api/chat':
                    action = command.get('action', 'send')
                    if action == 'send':
                        return self.send(chat.send(command.get('message')))
                    if action == 'retry':
                        return self.send(chat.retry())
                    if action == 'clear':
                        return self.send(chat.clear())
                    if action == 'cancel':
                        return self.send(chat.cancel())
                    if action == 'backend':
                        return self.send(chat.set_backend(command.get('backend')))
                    raise ValueError('Unknown chat action')
                if self.path == '/api/detection':
                    return self.send(detections.configure(command.get('enabled'), command.get('source')))
                if self.path == '/api/prompt':
                    action = command.get('action', 'submit')
                    if action == 'submit':
                        if args.sim and command.get('source') == 'camera':
                            raise ValueError('Camera prompts are disabled in simulation mode')
                        if 'chat_message_id' in command:
                            suggestion = chat.motion_request(command['chat_message_id'])
                            return self.send(prompt_planner.submit(suggestion['prompt'], command.get('source', 'auto'),
                                                                   trajectory=suggestion['trajectory'],
                                                                   reviser=chat.motion_reviser() if suggestion['trajectory'] is not None else None))
                        return self.send(prompt_planner.submit(command.get('prompt'), command.get('source', 'auto')))
                    if action == 'cancel':
                        if sim.status()['plan'] == prompt_planner.status().get('id'):
                            sim.control({'action':'stop'})
                        pipeline.stop()
                        return self.send(prompt_planner.status())
                    if action == 'preview':
                        pipeline.show_primary(None, command.get('id'))
                        return self.send(prompt_planner.show_once(command.get('id'), pipeline.show_primary))
                    raise ValueError('Unknown prompt action')
                if self.path == '/api/control':
                    sim.control(command)
                    return self.send(sim.status())
                if self.path == '/api/gestures':
                    if args.sim:
                        raise ValueError('Hardware gestures are disabled in simulation mode')
                    if command.get('action') == 'gesture' and (pipeline.connected or pipeline.status()['busy']):
                        raise ValueError('Release robot control before using a firmware gesture.')
                    return self.send(gestures.command(command))
                return self.send({'error': 'Not found'}, code=404)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                self.send({'error': str(exc)}, code=400)

    server.RequestHandlerClass = Handler
    threading.Thread(target=sim.run, daemon=True).start()
    print(f'Reins Observatory → http://localhost:{args.port}', flush=True)
    print(f'Robot control on {args.iface} requires an explicit connection and per-motion review. Pair glasses in Connections.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        glasses_bridge.close()
        pipeline.close()
        prompt_planner.cancel()
        chat.close()
        detections.close()
        gestures.close()
        server.server_close()


if __name__ == '__main__':
    main()
