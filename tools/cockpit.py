"""Cockpit: the three cameras and a live MuJoCo twin of the robot on one page.

Subscribes to rt/lowstate, poses Unitree's free-standing R1 model with the live
joint angles and IMU orientation, renders it offscreen and streams it as MJPEG.
The page embeds the head camera (tools/headcam.py, port 8081) and the wrist
cameras (tools/camstream.py on the Jetson, forwarded to port 8080). Read-only.

    .venv/bin/python tools/cockpit.py en6 [--port 8082] [--fps 15]
    open http://localhost:8082/
"""
import argparse, io, os, threading, time
os.environ.setdefault("MUJOCO_GL", "cgl")          # macOS offscreen context, no window needed
import mujoco, numpy as np
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

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
a = ap.parse_args()

model = mujoco.MjModel.from_xml_path(SCENE); data = mujoco.MjData(model)
qadr = {s: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for s, j in SLOT_TO_JOINT.items()}
state = {"q": None, "quat": None, "n": 0, "mode": None, "t": 0.0}

def on_msg(m: LowState_):
    state["q"] = [m.motor_state[s].q for s in range(35)]
    state["quat"] = list(m.imu_state.quaternion)          # w, x, y, z
    state["mode"] = m.mode_machine; state["n"] += 1; state["t"] = time.time()

ChannelFactoryInitialize(0, a.iface)
sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(on_msg, 10)

renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.0, 0.0, 0.72]; cam.distance = 2.4; cam.azimuth = 155; cam.elevation = -12   # 180 = facing the camera
floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
latest = {"jpg": b""}

def render_loop():
    while True:
        t = time.time()
        q, quat = state["q"], state["quat"]
        if q is not None:
            data.qpos[0:3] = [0.0, 0.0, 0.78]
            data.qpos[3:7] = quat if quat and abs(sum(v * v for v in quat) - 1) < 0.1 else [1, 0, 0, 0]
            for s, adr in qadr.items(): data.qpos[adr] = q[s]
            mujoco.mj_forward(model, data)
            zmin = min(data.geom_xpos[g][2] for g in range(model.ngeom) if g != floor)   # put the lowest point on the floor
            data.qpos[2] += 0.02 - zmin
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            buf = io.BytesIO(); Image.fromarray(renderer.render()).save(buf, "JPEG", quality=75)
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
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(txt.encode()); return
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
