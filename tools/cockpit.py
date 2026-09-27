"""Cockpit: the three cameras and a live MuJoCo twin of the robot on one page.

Subscribes to rt/lowstate, poses Unitree's free-standing R1 model with the live
joint angles and IMU orientation, renders it offscreen and streams it as MJPEG.
The page embeds the head camera (tools/headcam.py, port 8081) and the wrist
cameras (tools/camstream.py on the Jetson, forwarded to port 8080). Read-only.

Trajectory preview: GET /preview?file=sim/plans/arm_lift_dryrun.json loads the
resolved plan (what arm_lift streams, lead-in and return included) and draws both
hands' full paths as lines, each ending in an arrowhead at the destination; a translucent ghost of the arms plays the plan in a
loop with a progress caption. GET /preview/stop ends it. With &hold=1 the preview
loops until stopped and stays on top even while the arm topic is live: that is
how the window's AI pane shows a proposed move while the harness streamer holds
the arms, until the operator presses Accept (the window then stops the preview,
so the yellow SENDING ghost shows the real motion) or Reject. A plan with base_keyframes (a
whole-body step from the harness, or spectacles/make_walk_plan.py) walks a translucent ghost of
the whole robot along its floor path, drawn as a line with the end pose marked.
Whenever anything publishes on rt/arm_sdk with weight > 0 (arm_lift --execute,
teach.py, teleop), the ghost switches to the commanded joint targets read off
that topic, in yellow, captioned SENDING: the twin then shows exactly what is
being sent next to what the robot measures. The desktop window loads the plan
after every passing dry run and again the moment an execute starts.

    .venv/bin/python tools/cockpit.py en6 [--port 8082] [--fps 15]
    open http://localhost:8082/
"""
import argparse, io, json, os, threading, time, urllib.parse
os.environ.setdefault("MUJOCO_GL", "cgl")          # macOS offscreen context, no window needed
import mujoco, numpy as np
from PIL import Image, ImageDraw
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "sim/models/r1/scene.xml")
# controller slot -> MuJoCo joint (unitree_sdk2/include/unitree/dds_wrapper/robots/r1/defines.h)
SLOT_TO_JOINT = {
    0: "left_hip_pitch_joint", 1: "left_hip_roll_joint", 2: "left_hip_yaw_joint", 3: "left_knee_joint",
    4: "left_ankle_pitch_joint", 5: "left_ankle_roll_joint",
    6: "right_hip_pitch_joint", 7: "right_hip_roll_joint", 8: "right_hip_yaw_joint", 9: "right_knee_joint",
    10: "right_ankle_pitch_joint", 11: "right_ankle_roll_joint",
    12: "waist_roll_joint", 13: "waist_yaw_joint",
    15: "left_shoulder_pitch_joint", 16: "left_shoulder_roll_joint", 17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint", 19: "left_wrist_roll_joint",
    22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint", 24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint", 26: "right_wrist_roll_joint",
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("iface"); ap.add_argument("--port", type=int, default=8082); ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--head", default="http://localhost:8081/cam"); ap.add_argument("--wrists", default="http://localhost:8080")
ap.add_argument("--domain", type=int, default=0, help="DDS domain (0 = the robot; 1 with lo0 for loopback tests)")
a = ap.parse_args()

model = mujoco.MjModel.from_xml_path(SCENE); data = mujoco.MjData(model)
qadr = {s: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for s, j in SLOT_TO_JOINT.items()}
state = {"q": None, "quat": None, "n": 0, "mode": None, "t": 0.0, "cmd_q": None, "cmd_w": 0, "cmd_t": 0.0, "cmd_n": 0}

def on_msg(m: LowState_):
    state["q"] = [m.motor_state[s].q for s in range(35)]
    state["quat"] = list(m.imu_state.quaternion)          # w, x, y, z
    state["mode"] = m.mode_machine; state["n"] += 1; state["t"] = time.time()

def on_cmd(m: LowCmd_):                                    # what any publisher is sending on the arm topic
    state["cmd_q"] = [m.motor_cmd[s].q for s in range(35)]; state["cmd_w"] = int(m.mode_pr)
    state["cmd_t"] = time.time(); state["cmd_n"] += 1

ChannelFactoryInitialize(a.domain, a.iface)
sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(on_msg, 10)
sub_cmd = ChannelSubscriber("rt/arm_sdk", LowCmd_); sub_cmd.Init(on_cmd, 10)
ARM_SLOTS = [s for s in qadr if s in (13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26)]   # slots the arm topic drives

renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.0, 0.0, 0.72]; cam.distance = 2.4; cam.azimuth = 155; cam.elevation = -12   # 180 = facing the camera
floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
latest = {"jpg": b""}

# ---- trajectory preview: a ghost of the arms following a resolved plan ---------------------------
def body_name(i): return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
def descendants(root):
    out = set()
    for b in range(model.nbody):
        k = b
        while k > 0:
            if k == root: out.add(b); break
            k = int(model.body_parentid[k])
    return out
ARM_BODIES = set().union(*(descendants(b) for b in range(model.nbody) if "shoulder_pitch" in body_name(b)))
WRIST = {"left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link"),
         "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")}
TIP = np.array([0.13, 0.0, 0.0])                                    # hand tip in the wrist roll frame (as the fixed-base model's sites)
PATH_RGBA = {"left": (0.0, 1.0, 1.0, 0.9), "right": (1.0, 0.4, 0.0, 0.9)}
GHOST_RGBA = (0.35, 0.85, 1.0, 0.45)                                # preview playback
CMD_RGBA = (1.0, 0.9, 0.2, 0.55)                                    # commanded targets read off rt/arm_sdk
CMD_STALE_S = 0.5
ghost = mujoco.MjData(model); scratch = mujoco.MjData(model)
vopt = mujoco.MjvOption(); pert = mujoco.MjvPerturb()
preview = {"plan": None, "t0": 0.0, "name": "", "duration": 0.0, "loops": 0, "was_sending": False, "hold": False}
PAUSE_S, MAX_LOOPS = 1.0, 3      # a dry-run preview plays 3 times and clears; after real streaming the ghost clears at once

def load_plan(path):
    """Resolved plan (sim contract) -> dict with times, per-keyframe qpos values of the named joints, held joints, hand paths in the base frame."""
    src = json.loads(open(path).read())
    kfs = sorted(src["keyframes"], key=lambda f: f["time_s"])
    adr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in kfs[0]["joint_targets_rad"]}
    held = {int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]): float(v) for n, v in src.get("held_joints_rad", {}).items()}
    times = np.array([float(f["time_s"]) for f in kfs])
    frames = np.array([[float(f["joint_targets_rad"][n]) for n in adr] for f in kfs])
    gaps = np.diff(times); linear = len(gaps) > 0 and float(np.median(gaps)) < 0.25
    plan = {"times": times, "frames": frames, "adr": list(adr.values()), "held": held, "linear": linear, "base": None}
    bk = src.get("base_keyframes")
    if bk:                                                          # planar base path: origin = the robot's current base pose
        bt = np.array([float(f["time_s"]) for f in bk]); bx = np.array([[float(f["x_m"]), float(f["y_m"]), float(f["yaw_rad"])] for f in bk])
        plan["base"] = lambda t: np.array([np.interp(t, bt, bx[:, 0]), np.interp(t, bt, bx[:, 1]), np.interp(t, bt, np.unwrap(bx[:, 2]))])
        plan["base_end"] = bx[-1]
    def pose(t):
        i = int(np.searchsorted(times, t, side="right") - 1); i = max(0, min(i, len(times) - 2))
        t0, t1 = times[i], times[i + 1]; x = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        x = min(max(x, 0.0), 1.0); r = x if linear else 0.5 - 0.5 * np.cos(np.pi * x)
        return frames[i] + (frames[i + 1] - frames[i]) * r
    plan["pose"] = pose
    # hand paths in the pelvis frame, sampled every 0.1 s: base at the origin, live legs do not matter for the arms
    scratch.qpos[:] = 0.0; scratch.qpos[3] = 1.0
    for k, v in held.items(): scratch.qpos[k] = v
    paths = {side: [] for side in WRIST}
    floor = []
    for t in np.arange(0.0, times[-1] + 1e-9, 0.1):
        q = pose(t)
        for k, v in zip(plan["adr"], q): scratch.qpos[k] = v
        mujoco.mj_forward(model, scratch)
        base = plan["base"](t) if plan["base"] else None
        for side, b in WRIST.items():
            p = scratch.xpos[b] + scratch.xmat[b].reshape(3, 3) @ TIP
            if base is not None:                                   # carried along by the walking base
                x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
                p = np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])
            paths[side].append(p)
        if base is not None: floor.append([base[0], base[1], 0.01])
    plan["paths"] = {side: np.array(p) for side, p in paths.items()}
    plan["floor"] = np.array(floor) if floor else None
    return plan, src.get("name", os.path.basename(path)), float(times[-1])

def add_line(scn, p0, p1, rgba, width=4.0):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(p0, dtype=float), np.asarray(p1, dtype=float))
    scn.ngeom += 1

def add_sphere(scn, p, rgba, r=0.02):
    if scn.ngeom >= scn.maxgeom: return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, dtype=float),
                        np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    scn.ngeom += 1

def arrow_segments(pts, length=0.05, radius=0.015):
    """Wireframe pyramid at a path's end pointing along its final motion, as (p0, p1) line pairs; empty for a static path.
    The direction comes from the last `length` of arc rather than the last two samples, so sample noise cannot flip it."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2: return []
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); total = float(seg.sum())
    if total < 0.01: return []
    length = min(length, total / 2); radius = min(radius, length * 0.3)
    k, acc = len(pts) - 1, 0.0
    while k > 0 and acc < length: acc += seg[k - 1]; k -= 1
    tip = pts[-1]; d = tip - pts[k]; n = np.linalg.norm(d)
    if n < 1e-6: return []
    d = d / n; side = np.cross(d, [0.0, 0.0, 1.0])
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([0.0, 1.0, 0.0])
    up = np.cross(d, side); base = tip - d * length
    corners = [base + side * radius, base + up * radius, base - side * radius, base - up * radius]
    return [(c, tip) for c in corners] + list(zip(corners, corners[1:] + corners[:1]))

def draw_preview(scn):
    """Ghost arms (commanded targets while the arm topic is live, else the plan's playback) plus the plan's hand paths,
    all in the live robot's base frame. Returns the caption, or None when there is nothing to show."""
    plan = preview["plan"]
    sending = state["cmd_q"] is not None and time.time() - state["cmd_t"] < CMD_STALE_S and state["cmd_w"] > 0
    show_plan = plan is not None and (preview["hold"] or not sending)     # a held preview stays on top of the live topic
    if not show_plan and not sending: return None
    ghost.qpos[:] = data.qpos
    if not show_plan:
        for s in ARM_SLOTS: ghost.qpos[qadr[s]] = state["cmd_q"][s]
        rgba = CMD_RGBA
        caption = f"SENDING {preview['name'] if plan else ''}   weight {state['cmd_w']}%"
        preview["was_sending"] = True
    else:
        if preview["was_sending"] and not preview["hold"]:           # streaming just ended: show only the real robot again
            preview.update(plan=None, was_sending=False); return None
        cycle = preview["duration"] + PAUSE_S
        el = time.time() - preview["t0"]; t = min(el % cycle, preview["duration"]); preview["loops"] = int(el // cycle) + 1
        if preview["loops"] > MAX_LOOPS and not preview["hold"]:
            preview["plan"] = None; return None
        for k, v in plan["held"].items(): ghost.qpos[k] = v
        for k, v in zip(plan["adr"], plan["pose"](t)): ghost.qpos[k] = v
        rgba = GHOST_RGBA
        caption = (f"PROPOSED {preview['name'][:60]}   {t:4.1f} / {preview['duration']:.1f} s   Accept or Reject in the window" if preview["hold"]
                   else f"PREVIEW {preview['name']}   {t:4.1f} / {preview['duration']:.1f} s   loop {preview['loops']}")
    mujoco.mj_forward(model, ghost)
    n0 = scn.ngeom
    mujoco.mjv_addGeoms(model, ghost, vopt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
    for i in range(n0, scn.ngeom):
        g = scn.geoms[i]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and int(model.geom_bodyid[g.objid]) in ARM_BODIES: g.rgba[:] = rgba
        else: g.rgba[3] = 0.0                                        # the rest of the ghost coincides with the live robot: hide it
    R = ghost.xmat[1].reshape(3, 3) if model.nbody > 1 else np.eye(3); P = ghost.xpos[1]   # body 1 = pelvis (floating base)
    if plan:
        for side, pts in plan["paths"].items():
            w = pts @ R.T + P
            for p0, p1 in list(zip(w[:-1], w[1:])) + arrow_segments(w): add_line(scn, p0, p1, PATH_RGBA[side])
    for side in WRIST:
        add_sphere(scn, ghost.xpos[WRIST[side]] + ghost.xmat[WRIST[side]].reshape(3, 3) @ TIP, PATH_RGBA[side])
    return caption

def start_preview(rel, hold=False):
    path = os.path.realpath(os.path.join(ROOT, rel))
    if not path.startswith(ROOT + os.sep) or not path.endswith(".json"): raise ValueError("plan must be a .json inside the repo")
    plan, name, duration = load_plan(path)
    preview.update(plan=plan, name=name, duration=duration, t0=time.time(), loops=0, was_sending=False, hold=bool(hold))
    return f"previewing {name}: {duration:.1f} s, " + ("until stopped" if hold else f"{MAX_LOOPS} times")

def render_loop():
    while True:
        t = time.time()
        q, quat = state["q"], state["quat"]
        if q is not None:
            data.qpos[0:3] = [0.0, 0.0, 0.78]
            if quat and abs(sum(v * v for v in quat) - 1) < 0.1:
                w, x, y, z = quat                                  # keep roll and pitch, drop heading so the view stays robot-relative
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                q_unyaw = np.zeros(4); mujoco.mju_mulQuat(q_unyaw, np.array([np.cos(yaw / 2), 0, 0, -np.sin(yaw / 2)]), np.array(quat, dtype=float))
                data.qpos[3:7] = q_unyaw
            else:
                data.qpos[3:7] = [1, 0, 0, 0]
            for s, adr in qadr.items(): data.qpos[adr] = q[s]
            mujoco.mj_forward(model, data)
            zmin = min(data.geom_xpos[g][2] for g in range(model.ngeom) if g != floor)   # put the lowest point on the floor
            data.qpos[2] += 0.02 - zmin
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            try:
                caption = draw_preview(renderer.scene)
            except Exception as e:                                   # a bad plan must not kill the live twin
                preview["plan"] = None; caption = f"preview failed: {e}"[:80]
            im = Image.fromarray(renderer.render())
            if caption:
                ImageDraw.Draw(im).text((8, 6), caption, fill=(255, 255, 255))
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=75)
            latest["jpg"] = buf.getvalue()
        time.sleep(max(0.0, 1.0 / a.fps - (time.time() - t)))

PAGE = f"""<title>R1 cockpit</title>
<body style="margin:0;background:#0f1419;color:#c9d1d9;font:13px system-ui;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto;gap:6px;padding:6px;height:100vh;box-sizing:border-box">
<div><div style="padding:2px 6px">Head camera (controller)</div><img src="{a.head}" style="width:100%;background:#000"></div>
<div><div style="padding:2px 6px">Live twin (MuJoCo from rt/lowstate) <span id=st></span></div><img src="/twin" style="width:100%;background:#000"></div>
<div><div style="padding:2px 6px">Left wrist (Jetson)</div><img src="{a.wrists}/cam/0" style="width:100%;background:#000"></div>
<div><div style="padding:2px 6px">Right wrist (Jetson)</div><img src="{a.wrists}/cam/2" style="width:100%;background:#000"></div>
<script>setInterval(()=>fetch('/status').then(r=>r.text()).then(t=>document.getElementById('st').textContent=' · '+t),1000)</script>
</body>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        if self.path == "/":
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(PAGE.encode()); return
        if self.path == "/status":
            age = time.time() - state["t"]
            txt = f"mode_machine {state['mode']}, {state['n']} msgs, last {age:.1f}s ago" if state["n"] else "no rt/lowstate yet"
            if state["cmd_q"] is not None and time.time() - state["cmd_t"] < CMD_STALE_S: txt += f" · SENDING weight {state['cmd_w']}%"
            elif preview["plan"] is not None: txt += f" · preview {preview['name'][:40]} loop {preview['loops']}" + (" (held)" if preview["hold"] else "")
            if preview["plan"] is not None and preview["hold"]: txt += " · PROPOSAL shown"
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(txt.encode()); return
        if self.path.startswith("/preview"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                if self.path.startswith("/preview/stop"): preview.update(plan=None, hold=False); txt = "preview stopped"
                else: txt = start_preview(q.get("file", ["sim/plans/arm_lift_dryrun.json"])[0], q.get("hold", ["0"])[0] not in ("0", "", "false"))
                code = 200
            except Exception as e:
                txt, code = f"preview failed: {e}", 400
            self.send_response(code); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(txt.encode()); return
        if self.path == "/twin":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            try:
                while True:
                    if (jpg := latest["jpg"]):
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")
                    time.sleep(1.0 / a.fps)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(404); self.end_headers()

class Server(ThreadingMixIn, HTTPServer): daemon_threads = True
threading.Thread(target=Server(("127.0.0.1", a.port), H).serve_forever, daemon=True).start()
print(f"cockpit on http://localhost:{a.port}/  (Ctrl-C to stop)", flush=True)
render_loop()
