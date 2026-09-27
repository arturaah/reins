"""Log the R1 arm joints from rt/lowstate into a replayable plan. Subscribe-only.

Captures whatever the arms are doing, from any source: the remote's gestures,
teleop, a hand-guided teach, or our own tool. Stops after --seconds or Ctrl-C.
The head and wrist camera streams are sampled at 3 Hz meanwhile and tiled into
recordings/NAME.sheet.jpg at the motion's key moments (tools/framelog.py).

    .venv/bin/python tools/record.py en6 NAME [--seconds 20] [--hz 20]
    .venv/bin/python tools/arm_lift.py en6 --plan recordings/NAME.json [--execute]   # replay

Replaying commands the measured angles, so an arm that drooped under gravity
while recording will droop a little more on replay. Fine for gestures; for
precise work record with tools/arm_lift.py, which logs the commanded targets.
"""
import argparse, json, sys, time
from pathlib import Path
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from arm_lift import JOINTS, State, ROOT
from framelog import FrameLogger, save_sheet

ARMS = [(s, mj) for s, n, mj, *_ in JOINTS if mj and n.startswith(("left", "right"))]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("iface"); ap.add_argument("name")
ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--hz", type=float, default=20.0)
ap.add_argument("--no-frames", action="store_true", help="do not sample the camera streams for the contact sheet")
a = ap.parse_args()

ChannelFactoryInitialize(0, a.iface)
st = State(); sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(st.on_msg, 10)
t0 = time.time()
while st.msg is None and time.time() - t0 < 3.0:
    time.sleep(0.05)
if st.msg is None:
    sys.exit(f"no rt/lowstate on {a.iface}")
print(f"recording {a.name} at {a.hz:g} Hz for up to {a.seconds:g} s, Ctrl-C to stop early")
rec, t0 = [], time.time()
frames = None if a.no_frames else FrameLogger().start(t0)
try:
    while (t := time.time() - t0) < a.seconds:
        rec.append((0.0 if not rec else round(t, 3), {mj: round(st.msg.motor_state[s].q, 4) for s, mj in ARMS}))
        time.sleep(1.0 / a.hz)
except KeyboardInterrupt:
    pass
sub.Close()
if frames: frames.stop.set()
path = ROOT / "recordings" / f"{a.name}.json"; path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps({"schema_version": 1, "name": a.name, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "source": "rt/lowstate measured", "duration_s": rec[-1][0],
    "keyframes": [{"time_s": t, "joint_targets_rad": q} for t, q in rec]}, indent=1) + "\n")
span = {mj: max(q[mj] for _, q in rec) - min(q[mj] for _, q in rec) for _, mj in ARMS}
print(f"saved {len(rec)} samples over {rec[-1][0]} s to {path.relative_to(ROOT)}")
moved = ", ".join(f"{mj.replace('_joint','')}={v:.2f}" for mj, v in span.items() if v > 0.02)
print(f"range of motion (rad): {moved}" if moved else "range of motion: none, the arms were still")
if frames:
    print(save_sheet(path, rec, frames.finish()))
