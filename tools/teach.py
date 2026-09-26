"""Kinesthetic teaching: make both arms compliant, move them by hand, record, replay later.

While this runs, the arm topic is streamed at 50 Hz with low stiffness and a
"clutch": a joint's target follows the measured angle while it is being moved
(error over 0.05 rad and speed over 0.15 rad/s) and holds where it was left
otherwise, so a released arm settles instead of sagging away under gravity.
Expect a few degrees of droop at full extension with the soft gains. Waist and head
are held at their measured pose with normal gains. Both arms are recorded at
20 Hz into recordings/NAME.json in the sim contract format.

    .venv/bin/python tools/teach.py en6 NAME [--seconds 30] [--kp 15] [--kd 1]
    .venv/bin/python tools/arm_lift.py en6 --plan recordings/NAME.json [--speed 0.5] [--execute]   # replay

Ends after --seconds or Ctrl-C, then ramps the weight down over 2 s: the
built-in controller takes the arms back to its own pose during that ramp, so
hands off the arms when the tool says "releasing". Requires FSM 4 or 811.
Move slowly: the replay gate caps joint speed at 0.5 rad/s (about 30 deg/s);
--speed on replay slows a recording that was taught faster.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
from arm_lift import JOINTS, State, ROOT, FSM_NAMES, FSM_ARM_OK, RATE_HZ

ARM = [j for j in JOINTS if j[1].startswith(("left", "right"))]
OTHER = [j for j in JOINTS if j not in ARM]
CLUTCH, V_MOVE = 0.05, 0.15

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("iface"); ap.add_argument("name")
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--kp", type=float, default=20.0, help="arm stiffness while teaching (Unitree's normal is 30-50)")
ap.add_argument("--kd", type=float, default=1.5)
a = ap.parse_args()

ChannelFactoryInitialize(0, a.iface)
st = State(); sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(st.on_msg, 10)
t0 = time.time()
while st.count < 10 and time.time() - t0 < 3.0:
    time.sleep(0.05)
if st.msg is None:
    sys.exit(f"no rt/lowstate on {a.iface}")
lc = LocoClient(); lc.SetTimeout(3.0); lc.Init()
code, data = lc._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "")
fsm = json.loads(data)["data"] if code == 0 and data else None
print(f"fsm {fsm} = {FSM_NAMES.get(fsm, 'unknown')}")
if fsm not in FSM_ARM_OK:
    sys.exit(f"refusing: teach mode needs FSM {sorted(FSM_ARM_OK)}")

q = {s: st.msg.motor_state[s].q for s, *_ in JOINTS}      # targets, start at measured
pub = ChannelPublisher("rt/arm_sdk", LowCmd_); pub.Init()
crc = CRC(); cmd = unitree_hg_msg_dds__LowCmd_()
for s, n, _, kp, kd in JOINTS:
    mc = cmd.motor_cmd[s]; mc.q, mc.dq, mc.tau = q[s], 0.0, 0.0
    mc.kp, mc.kd = (a.kp, a.kd) if (s, n, _, kp, kd) in ARM else (kp, kd)

def send(weight):
    cmd.mode_pr = int(round(np.clip(weight, 0.0, 1.0) * 100))
    for s in q: cmd.motor_cmd[s].q = float(q[s])
    cmd.crc = crc.Crc(cmd); pub.Write(cmd)

def release(seconds=2.0):
    print("releasing: hands off the arms, the controller takes them back")
    t_r = time.time()
    while (el := time.time() - t_r) < seconds:
        send(1.0 - el / seconds); time.sleep(1.0 / RATE_HZ)
    send(0.0)

dt = 1.0 / RATE_HZ
print("ramping weight up (arms go soft in 1 s)")
t_start = time.time()
while (t := time.time() - t_start) < 1.0:
    send(t); time.sleep(dt)
print(f"TEACH: move the arms by hand, slowly. Recording {a.name} for up to {a.seconds:g} s, Ctrl-C to finish early.")
rec, rec_last, t_start = [], -1.0, time.time()
try:
    while (t := time.time() - t_start) < a.seconds:
        now = time.time()
        if now - st.t_last > 0.5:
            print("ABORT: lowstate stale"); release(1.0); sys.exit(2)
        for s, *_ in ARM:
            ms = st.msg.motor_state[s]
            if abs(ms.q - q[s]) > CLUTCH and abs(ms.dq) > V_MOVE: q[s] = ms.q     # being moved: follow
        send(1.0)
        if t - rec_last >= 0.05:
            rec_last = 0.0 if not rec else t
            rec.append((round(rec_last, 3), {mj: round(st.msg.motor_state[s].q, 4) for s, n, mj, *_ in ARM}))
        if int(t / 5) != int((t - dt) / 5):
            print(f"  {t:4.0f} s")
        time.sleep(max(0.0, dt - (time.time() - now)))
except KeyboardInterrupt:
    print()
release()
sub.Close()
if len(rec) < 2:
    sys.exit("nothing recorded")
path = ROOT / "recordings" / f"{a.name}.json"; path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps({"schema_version": 1, "name": a.name, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "source": "kinesthetic teach, measured", "duration_s": rec[-1][0],
    "keyframes": [{"time_s": t, "joint_targets_rad": qq} for t, qq in rec]}, indent=1) + "\n")
span = {mj: max(qq[mj] for _, qq in rec) - min(qq[mj] for _, qq in rec) for _, _, mj, *_ in ARM}
moved = ", ".join(f"{mj.replace('_joint','')}={v:.2f}" for mj, v in span.items() if v > 0.02)
print(f"saved {len(rec)} samples over {rec[-1][0]} s to {path.relative_to(ROOT)}")
print(f"range of motion (rad): {moved}" if moved else "range of motion: none, the arms were not moved")
print(f"replay: .venv/bin/python tools/arm_lift.py {a.iface} --plan {path.relative_to(ROOT)}   (add --speed 0.5 if the dry run says too fast)")
