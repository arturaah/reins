"""Local Reins observatory: MuJoCo plan preview, camera monitors and trajectory control.

Run: .venv/bin/python tools/dashboard.py [--iface en6]
The preview itself never touches the robot. Dry run, Execute and Abort run
tools/arm_lift.py as a subprocess, exactly like tools/reins_ui.py, so the tool's
own gates apply (limits, speed cap, FSM 4/811, tracking-error abort). On top of
that, Execute is only accepted for the plan and settings of a dry run that
succeeded in the last few minutes, and only with the UI's confirmation. Abort
sends the tool SIGINT, which ramps the arm weight down. No remote services are
started.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Select before importing MuJoCo. Override for systems that need OSMesa/GLFW.
os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
import mujoco
import numpy as np
from PIL import Image
from sim.preview import load_plan
from spectacles.plan_feed import hand_paths
from core.prompt_planner import PromptPlanner

ASSETS = ROOT / 'tools/dashboard'
DRYRUN_PLAN = 'sim/plans/arm_lift_dryrun.json'   # written by every arm_lift dry run


class CameraFeed:
    """One upstream reader per feed, bounded memory and stale-frame expiry."""
    def __init__(self, url):
        self.url = url
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
                            self.jpg = jpg
                            self.updated = time.monotonic()
                            self.error = ''
                        if len(buffer) > 8 * 1024 * 1024:
                            raise ValueError('Stream frame exceeded 8 MB')
                self.error = 'Stream ended; reconnecting'
            except Exception:
                self.error = 'Feed unavailable; retrying'
            time.sleep(1)

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
        self.speed = 1.0
        self.show_paths = True
        self.files = {}
        self.plans = []
        self.library = 0          # bumped on every rescan so the browser refetches the list
        self.scan()
        self.select('tools/plans/cup_grab_right.json' if 'tools/plans/cup_grab_right.json' in self.files else next(iter(self.files)))

    def scan(self):
        """(Re)build the plan library from disk; malformed files are left out."""
        files, plans = {}, []
        for folder in ('sim/plans', 'tools/plans', 'recordings'):
            for path in sorted((ROOT / folder).glob('*.json')):
                try:
                    plan = load_plan(self.model, path)
                    duration = float(plan['duration_s'])
                    if not math.isfinite(duration) or duration <= 0 or not all(math.isfinite(t) for t in plan['_times']):
                        continue
                    key = str(path.relative_to(ROOT))
                    kind = 'Dry run' if key == DRYRUN_PLAN else 'Recording' if folder == 'recordings' else 'Plan'
                    files[key] = path
                    plans.append({'id': key, 'name': str(plan.get('name', path.stem)).replace('_', ' '),
                                  'duration': duration, 'joints': len(plan['_joint_ids']), 'kind': kind, 'preview_only': bool(plan.get('preview_only'))})
                except (ValueError, KeyError, TypeError, OSError):
                    continue
        with self.lock:
            self.files, self.plans = files, plans
            self.library += 1

    def select(self, key):
        if key not in self.files:
            raise ValueError('Unknown plan')
        plan = load_plan(self.model, self.files[key])
        held = plan.get('held_joints_rad', {})
        for name, value in held.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or not math.isfinite(float(value)):
                raise ValueError('Invalid held joint')
        self.paths = hand_paths(self.model, plan, 90)
        self.plan = plan
        self.key = key
        self.position = 0.0
        self.playing = False

    def control(self, command):
        with self.lock:
            action = command.get('action')
            if action == 'plan':
                self.select(command.get('id'))
            elif action == 'play':
                if self.position >= self.plan['duration_s']:
                    self.position = 0
                self.playing = True
            elif action == 'pause':
                self.playing = False
            elif action == 'seek':
                value = float(command['time'])
                if not math.isfinite(value):
                    raise ValueError('Invalid time')
                self.position = max(0, min(float(self.plan['duration_s']), value))
            elif action == 'speed':
                value = float(command['value'])
                if value not in (.25, .5, 1, 1.5, 2):
                    raise ValueError('Invalid speed')
                self.speed = value
            elif action == 'paths':
                self.show_paths = bool(command['value'])
            elif action == 'view':
                views = {'perspective': (145, -16, 2.35), 'front': (180, -10, 2.3), 'side': (90, -10, 2.3)}
                self.camera.azimuth, self.camera.elevation, self.camera.distance = views[command['value']]
            else:
                raise ValueError('Unknown action')

    def status(self):
        with self.lock:
            return {'plan': self.key, 'library': self.library, 'time': self.position, 'duration': self.plan['duration_s'],
                    'playing': self.playing, 'speed': self.speed, 'paths': self.show_paths,
                    'ready': bool(self.jpg) and time.monotonic() - self.updated < 3, 'error': self.error}

    def run(self):
        try:
            # Renderer and GL context live on the same thread for CGL/EGL portability.
            with mujoco.Renderer(self.model, height=540, width=960) as renderer:
                last = time.monotonic()
                while True:
                    start = time.monotonic()
                    with self.lock:
                        if self.playing:
                            self.position = min(self.plan['duration_s'], self.position + (start - last) * self.speed)
                            if self.position >= self.plan['duration_s']:
                                self.playing = False
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


class Runner:
    """Runs tools/arm_lift.py for one plan at a time, the same way tools/reins_ui.py does.

    Execute is accepted only for the plan, speed and kp scale of the last dry run
    that exited 0 less than DRY_RUN_VALID_S ago, and only with confirm=True.
    """
    DRY_RUN_VALID_S = 300
    SPEED = (0.1, 2.0)
    KP_SCALE = (0.5, 2.0)
    MAX_LINES = 4000

    def __init__(self, iface, tool=ROOT / 'tools/arm_lift.py', python=sys.executable, on_exit=None):
        self.iface, self.tool, self.python, self.on_exit = iface, Path(tool), python, on_exit
        self.lock = threading.Lock()
        self.proc = None
        self.job = None           # {'kind', 'plan', 'speed', 'kp_scale', 'started'}
        self.exit = None
        self.cleared = None       # last successful dry run: {'plan', 'speed', 'kp_scale', 'at', 'fsm'}
        self.fsm = None           # (id, name) from the last run's "fsm id:" line
        self.source = None        # plan id of the last dry run, which the resolved dry-run file stands for
        self.lines = []           # (seq, text)
        self.seq = 0

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def _log(self, text):
        with self.lock:
            self.seq += 1
            self.lines.append((self.seq, text))
            del self.lines[:-self.MAX_LINES]

    @staticmethod
    def _number(value, bounds, name):
        value = float(value)
        if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise ValueError(f'{name} must be between {bounds[0]} and {bounds[1]}')
        return round(value, 3)

    def _valid_cleared(self):
        c = self.cleared
        return c if c and time.time() - c['at'] < self.DRY_RUN_VALID_S else None

    def start(self, kind, plan, plan_path, speed=1.0, kp_scale=1.0, confirm=False):
        if self.running():
            raise ValueError('A run is still active; abort it first')
        if kind == 'dry':
            speed, kp_scale = self._number(speed, self.SPEED, 'Speed'), self._number(kp_scale, self.KP_SCALE, 'kp scale')
        elif kind == 'execute':
            cleared = self._valid_cleared()
            if not cleared:
                raise ValueError('Execute needs a successful dry run from the last 5 minutes')
            if not confirm:
                raise ValueError('Execute needs confirmation')
            # Always the dry-run settings, never whatever the client sends now.
            plan, plan_path, speed, kp_scale = cleared['plan'], cleared['path'], cleared['speed'], cleared['kp_scale']
        else:
            raise ValueError('Unknown run')
        if json.loads(Path(plan_path).read_text()).get('preview_only'):
            raise ValueError('Generated prompt plans are preview-only: calibration, contact control and a validated execution adapter are required.')
        cmd = [self.python, str(self.tool), self.iface, '--plan', str(plan_path),
               '--speed', str(speed), '--kp-scale', str(kp_scale)] + (['--execute'] if kind == 'execute' else [])
        with self.lock:
            self.lines, self.exit = [], None
            self.job = {'kind': kind, 'plan': plan, 'path': plan_path, 'speed': speed, 'kp_scale': kp_scale, 'started': time.time()}
            if kind == 'dry':
                self.source = plan
            if kind == 'execute':
                self.cleared = None   # one execute per dry run
        self._log('$ ' + ' '.join(['arm_lift.py'] + cmd[3:]))
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._pump, args=(self.proc, dict(self.job)), daemon=True).start()

    def _pump(self, proc, job):
        for line in proc.stdout:
            line = line.rstrip('\n')
            if 'take sample error' in line:   # CycloneDDS noise, as in reins_ui
                continue
            if line.startswith('fsm id:'):
                try:
                    fid = int(line.split(':')[1].split('=')[0])
                    self.fsm = (fid, line.split('=', 1)[1].split('fsm mode')[0].strip())
                except (ValueError, IndexError):
                    pass
            self._log(line)
        proc.stdout.close()
        code = proc.wait()
        with self.lock:
            self.exit = code
            if job['kind'] == 'dry' and code == 0:
                self.cleared = {**job, 'at': time.time(), 'fsm': self.fsm}
        self._log(f'[exit {code}]')
        if self.on_exit:
            try:
                self.on_exit(job, code)
            except Exception as exc:
                self._log(f'(dashboard: {exc})')

    def abort(self):
        if self.running():
            self.proc.send_signal(signal.SIGINT)
            self._log('[abort sent: the tool releases the arm]')
            return True
        return False

    def shutdown(self, wait=5.0):
        if self.abort():
            try:
                self.proc.wait(wait)
            except subprocess.TimeoutExpired:
                pass

    def status(self, since=0):
        with self.lock:
            job = {k: v for k, v in self.job.items() if k != 'path'} if self.job else None
            c = self._valid_cleared()
            return {'running': self.running(), 'job': job, 'exit': self.exit, 'iface': self.iface, 'source': self.source,
                    'fsm': {'id': self.fsm[0], 'name': self.fsm[1], 'ok': self.fsm[0] in (4, 811)} if self.fsm else None,
                    'cleared': {'plan': c['plan'], 'speed': c['speed'], 'kp_scale': c['kp_scale'],
                                'expires_in': round(self.DRY_RUN_VALID_S - (time.time() - c['at']))} if c else None,
                    'seq': self.seq, 'lines': [t for n, t in self.lines if n > since]}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--head', default='http://127.0.0.1:8081/cam')
    parser.add_argument('--left-wrist', default='http://127.0.0.1:8080/cam/0')
    parser.add_argument('--right-wrist', default='http://127.0.0.1:8080/cam/2')
    parser.add_argument('--glasses', default='', help='Glasses MJPEG/JPEG video URL, if available')
    parser.add_argument('--twin', default='http://127.0.0.1:8082/twin', help='Live twin MJPEG URL (tools/cockpit.py); empty to disable')
    parser.add_argument('--iface', default='en6', help='Network interface passed to tools/arm_lift.py')
    parser.add_argument('--observation', type=Path, help='Atomically updated calibrated RGB/depth observation NPZ')
    args = parser.parse_args()
    for url in (args.head, args.left_wrist, args.right_wrist, args.glasses, args.twin):
        if url and urlparse(url).scheme not in ('http', 'https'):
            parser.error('Feed URLs must use http:// or https://')
    # Abort relies on SIGINT reaching arm_lift.py. A shell that starts us in the
    # background (`&`, nohup) sets SIGINT to ignored, and children inherit that.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sim = Simulation()
    def preview_pose():
        with sim.lock:
            return {sim.model.joint(i).name: float(sim.data.qpos[sim.model.jnt_qposadr[i]]) for i in range(sim.model.njnt)}
    prompt_planner = PromptPlanner(args.observation, preview_pose=preview_pose)
    feeds = {name: CameraFeed(url) for name, url in {'head': args.head, 'left': args.left_wrist,
             'right': args.right_wrist, 'glasses': args.glasses, 'twin': args.twin}.items()}
    token = secrets.token_urlsafe(24)
    # The twin server reports rt/lowstate health next to its stream.
    robot_status = TextPoller(args.twin.rsplit('/', 1)[0] + '/status' if args.twin else '')

    def after_run(job, code):
        sim.scan()                         # new recording or dry-run plan
        if job['kind'] == 'dry' and code == 0 and DRYRUN_PLAN in sim.files:
            sim.control({'action': 'plan', 'id': DRYRUN_PLAN})   # preview what the robot would do
    runner = Runner(args.iface, on_exit=after_run)

    def run(command):
        action = command.get('action')
        if action == 'abort':
            return runner.abort()
        if action == 'dry':
            plan = command.get('plan')
            if plan == DRYRUN_PLAN:            # the resolved file stands for the plan it came from
                plan = runner.source
            if plan not in sim.files or plan == DRYRUN_PLAN:
                raise ValueError('Choose a plan or recording to dry run')
            runner.start('dry', plan, sim.files[plan], command.get('speed', 1.0), command.get('kp_scale', 1.0))
        elif action == 'execute':
            runner.start('execute', None, None, confirm=command.get('confirm') is True)
        else:
            raise ValueError('Unknown run action')
        return True

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
                try:
                    since = int(self.path.partition('since=')[2] or 0)
                except ValueError:
                    since = 0
                return self.send({'simulation': sim.status(), 'feeds': {k: v.status() for k, v in feeds.items()},
                                  'run': runner.status(since), 'robot': robot_status.text, 'prompt': prompt_planner.status()})
            if path == '/api/prompt':
                return self.send(prompt_planner.status())
            if path == '/frame/grounding':
                jpg = prompt_planner.image
                return self.send(jpg, 'image/jpeg') if jpg else self.send({'error': 'No grounded image'}, code=404)
            if path == '/api/plans':
                return self.send({'plans': sim.plans, 'library': sim.library, 'token': token})
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
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size')
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise ValueError('Expected object')
            except (ValueError, TypeError) as exc:
                return self.send({'error': str(exc)}, code=400)
            if self.path == '/api/abort-beacon':
                # sendBeacon cannot set headers, so the token travels in the body. Abort only.
                if command.get('token') != token:
                    return self.send({'error': 'Invalid local session'}, code=403)
                return self.send({'aborted': runner.abort()})
            if self.headers.get('X-Reins-Token') != token:
                return self.send({'error': 'Invalid local session'}, code=403)
            try:
                if self.path == '/api/prompt':
                    action = command.get('action', 'submit')
                    if action == 'submit':
                        if runner.running():
                            raise ValueError('Wait for the active robot run to finish before planning')
                        return self.send(prompt_planner.submit(command.get('prompt'), command.get('source', 'auto')))
                    if action == 'cancel':
                        prompt_planner.cancel()
                        return self.send(prompt_planner.status())
                    if action == 'preview':
                        if runner.running():
                            raise ValueError('Cannot change the preview during an active robot run')
                        path = prompt_planner.preview(command.get('id'))
                        sim.scan()
                        sim.control({'action': 'plan', 'id': str(path.relative_to(ROOT))})
                        return self.send(prompt_planner.status())
                    raise ValueError('Unknown prompt action')
                if self.path == '/api/control':
                    sim.control(command)
                    return self.send(sim.status())
                if self.path == '/api/run':
                    run(command)
                    return self.send(runner.status())
                return self.send({'error': 'Not found'}, code=404)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.send({'error': str(exc)}, code=400)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=sim.run, daemon=True).start()
    print(f'Reins Observatory → http://localhost:{args.port}', flush=True)
    print(f'Preview is local; Dry run / Execute run tools/arm_lift.py on {args.iface}. Ctrl-C to stop (aborts a run).', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
