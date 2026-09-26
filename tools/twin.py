"""Live MuJoCo twin of the R1: the fixed-base model mirrors rt/lowstate. Subscribe-only.

Reads the measured joint angles from rt/lowstate (legs, waist, arms) and poses the
model with them, no physics. When something streams rt/arm_sdk with weight above 0
(tools/arm_lift.py --execute, teach.py, xr_teleoperate), the commanded hand
positions are drawn as small spheres, so command and reality can be compared.
--plan overlays a plan's hand paths (the review preview) on top of the live robot.
Publishes nothing; safe to run next to any other tool.

    .venv/bin/python tools/twin.py en6
    .venv/bin/python tools/twin.py en6 --plan sim/plans/arm_lift_dryrun.json
    .venv/bin/python tools/twin.py ws://MAC:8766     # via tools/relay.py; needs only mujoco + websockets

The pelvis is pinned, so balance shifts show up as leg motion rather than body sway.
Joints the A5 lacks (waist pitch, wrist pitch/yaw) stay at 0; the model has no head.
Keep --domain at 0 for the robot. Never run Unitree's MuJoCo bridge on domain 0.
"""
import argparse, json, sys, time
from pathlib import Path
import mujoco, mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim/models/r1/scene_fixed_base.xml"
HIDDEN = ("pickup_table", "pickup_cube_geom")   # props of the pickup demo
# controller slot -> MuJoCo joint, from unitree_sdk2/include/unitree/dds_wrapper/robots/r1/defines.h
SLOTS = {
    0: "left_hip_pitch_joint", 1: "left_hip_roll_joint", 2: "left_hip_yaw_joint",
    3: "left_knee_joint", 4: "left_ankle_pitch_joint", 5: "left_ankle_roll_joint",
    6: "right_hip_pitch_joint", 7: "right_hip_roll_joint", 8: "right_hip_yaw_joint",
    9: "right_knee_joint", 10: "right_ankle_pitch_joint", 11: "right_ankle_roll_joint",
    12: "waist_roll_joint", 13: "waist_yaw_joint",
    15: "left_shoulder_pitch_joint", 16: "left_shoulder_roll_joint", 17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint", 19: "left_wrist_roll_joint",
    22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint", 24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint", 26: "right_wrist_roll_joint",
}
ARM_SLOTS = [s for s, n in SLOTS.items() if n.startswith(("left_shoulder", "left_elbow", "left_wrist",
                                                          "right_shoulder", "right_elbow", "right_wrist", "waist_yaw"))]
RELAY_PORT = 8766
COLORS = {"left": (0, 1, 1), "right": (1, .35, .1)}


class RobotState:
    """Latest robot snapshot keyed by MuJoCo joint name, filled from DDS or from a relay."""
    def __init__(self):
        self.q, self.t, self.count = None, 0.0, 0
        self.weight, self.cmd_q, self.cmd_t = 0.0, None, 0.0
    def set_state(self, q):
        self.q, self.t, self.count = q, time.time(), self.count + 1
    def set_cmd(self, weight, q):
        self.weight, self.cmd_q, self.cmd_t = weight, q, time.time()
    def commanding(self, now):
        return self.cmd_q is not None and now - self.cmd_t < 0.5 and self.weight > 0
    def to_json(self):
        cmd = ({"weight": round(self.weight, 3), "q": {n: round(v, 5) for n, v in self.cmd_q.items()}}
               if self.commanding(time.time()) else None)
        return json.dumps({"type": "r1_state", "version": 1, "t": round(self.t, 4),
                           "q": {n: round(v, 5) for n, v in self.q.items()}, "cmd": cmd})


def listen_dds(state, iface, domain):
    """Subscribe to rt/lowstate and rt/arm_sdk. Subscribe-only."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    ChannelFactoryInitialize(domain, iface)
    subs = [ChannelSubscriber("rt/lowstate", LowState_), ChannelSubscriber("rt/arm_sdk", LowCmd_)]
    subs[0].Init(lambda m: state.set_state({n: float(m.motor_state[s].q) for s, n in SLOTS.items()}), 10)
    subs[1].Init(lambda c: state.set_cmd(c.mode_pr / 100.0, {SLOTS[s]: float(c.motor_cmd[s].q) for s in ARM_SLOTS}), 10)
    print(f"listening on {iface}, domain {domain} (subscribe-only)", flush=True)
    return subs   # keep the readers alive


def listen_relay(state, url):
    """Read tools/relay.py's feed in a background thread, reconnecting as needed."""
    import asyncio, threading
    from websockets.asyncio.client import connect

    async def run():
        while True:
            try:
                async with connect(url) as ws:
                    print(f"connected to relay {url}", flush=True)
                    async for text in ws:
                        d = json.loads(text)
                        if d.get("type") != "r1_state" or d.get("version") != 1:
                            continue
                        if d.get("cmd"):
                            state.set_cmd(float(d["cmd"]["weight"]), d["cmd"]["q"])
                        state.set_state(d["q"])
            except Exception as exc:
                print(f"relay {url}: {exc}; retrying in 2 s", flush=True)
            await asyncio.sleep(2)

    threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()


def pose(model, data, adr, q_by_name):
    data.qpos[:] = 0.0
    for n, q in q_by_name.items():
        if n in adr:
            data.qpos[adr[n]] = q
    mujoco.mj_kinematics(model, data)


def draw(viewer, paths, commanded):
    """Plan paths as lines, commanded hand tips as spheres, into the viewer's user scene."""
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0
        for side, pts in paths.items():
            rgba = np.array((*COLORS[side], .5), dtype=np.float32)
            for a, b in zip(pts, pts[1:]):
                if scn.ngeom >= scn.maxgeom:
                    return
                if np.linalg.norm(b - a) < 1e-4:
                    continue
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 6.0, a, b)
                scn.ngeom += 1
        for side, p in commanded.items():
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.025, 0, 0]),
                                p, np.eye(3).ravel(), np.array((*COLORS[side], .8), dtype=np.float32))
            scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="robot network interface (en6 on the Mac, eth10 on the Jetson), "
                                   "or a relay URL such as ws://MAC:8766 from tools/relay.py")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain; the robot is 0")
    ap.add_argument("--plan", type=Path, help="plan JSON whose hand paths to overlay (re-read when it changes)")
    ap.add_argument("--fps", type=float, default=60.0)
    a = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    for name in HIDDEN:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = 0.0
    adr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in SLOTS.values()}
    sites = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_preview") for side in COLORS}
    data, cmd_data = mujoco.MjData(model), mujoco.MjData(model)

    feed = None
    if a.plan:
        sys.path.insert(0, str(ROOT / "spectacles"))
        from plan_feed import Feed
        feed = Feed(model, a.plan.resolve(), 200)

    state = RobotState()
    if a.source.startswith(("ws://", "wss://")):
        listen_relay(state, a.source)
    else:
        readers = listen_dds(state, a.source, a.domain)

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = [0.1, 0.0, 0.8]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 150, -15
    paths, plan_text, last_report, last_count = {}, None, time.time(), 0
    try:
        while viewer.is_running():
            now = time.time()
            if feed and (text := feed.current()) is not plan_text:
                plan_text = text
                paths = {s: np.array(p) for s, p in json.loads(text)["hands"].items()} if text else {}
            q = state.q
            if q is not None:
                with viewer.lock():
                    pose(model, data, adr, q)
            commanded = {}
            if q is not None and state.commanding(now):
                pose(model, cmd_data, adr, {**q, **state.cmd_q})
                commanded = {side: cmd_data.site_xpos[site].copy() for side, site in sites.items()}
            draw(viewer, paths, commanded)
            viewer.sync()
            if now - last_report >= 2.0:
                hz = (state.count - last_count) / (now - last_report)
                age = now - state.t if state.t else float("inf")
                arm = f"arm_sdk weight {state.weight:.0%}" if state.commanding(now) else "arm_sdk idle"
                print(f"state {hz:6.0f} Hz" + ("" if age < 0.5 else f"  STALE ({age:.1f} s since last)") + f"  {arm}", flush=True)
                last_report, last_count = now, state.count
            time.sleep(max(0.0, 1.0 / a.fps - (time.time() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
