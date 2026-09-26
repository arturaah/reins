"""Lift one R1 arm joint a little and put it back, through rt/arm_sdk.

Dry run (default): subscribes to rt/lowstate, asks the controller its FSM id,
builds the trajectory from the measured pose, checks joint limits and speed,
writes a plan for sim/preview.py and runs the headless MuJoCo preview.
Publishes nothing.

--execute: streams the same trajectory at 50 Hz with the blend weight ramped
0 -> 1 before and 1 -> 0 after, watching the measured joint the whole time.

    .venv/bin/python tools/arm_lift.py en6                 # dry run
    .venv/bin/python tools/arm_lift.py en6 --execute       # moves the robot
    options: --joint left_shoulder_pitch --delta -0.25 --move-s 2 --hold-s 1

Protocol facts (from the vendored C++ SDK, robots/r1/r1_pub.h and defines.h):
the message is the hg LowCmd on topic rt/arm_sdk, the weight is mode_pr in
0..100, and joints use the controller's 35-slot layout, not the 26-motor one.
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
ROBOT_API_ID_LOCO_GET_FSM_MODE = 7002   # in the C++ r1_loco_api.hpp, missing from the Python file

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "sim/models/r1/R1_fixed_base.xml"

# slot, name, mujoco joint (None = not in the sim model), kp, kd  -- order and gains as in
# unitree_sdk2/example/r1/high_level/r1_arm_sdk_dds_example.cpp
JOINTS = [
    (15, "left_shoulder_pitch",  "left_shoulder_pitch_joint",  50.0, 2.0),
    (16, "left_shoulder_roll",   "left_shoulder_roll_joint",   50.0, 2.0),
    (17, "left_shoulder_yaw",    "left_shoulder_yaw_joint",    40.0, 2.0),
    (18, "left_elbow",           "left_elbow_joint",           40.0, 2.0),
    (19, "left_wrist_roll",      "left_wrist_roll_joint",      30.0, 2.0),
    (22, "right_shoulder_pitch", "right_shoulder_pitch_joint", 50.0, 2.0),
    (23, "right_shoulder_roll",  "right_shoulder_roll_joint",  50.0, 2.0),
    (24, "right_shoulder_yaw",   "right_shoulder_yaw_joint",   40.0, 2.0),
    (25, "right_elbow",          "right_elbow_joint",          40.0, 2.0),
    (26, "right_wrist_roll",     "right_wrist_roll_joint",     30.0, 2.0),
    (13, "waist_yaw",            "waist_yaw_joint",            50.0, 3.0),
    (29, "head_pitch",           None,                         15.0, 1.0),
    (30, "head_yaw",             None,                         15.0, 1.0),
]
BY_NAME = {j[1]: j for j in JOINTS}
RATE_HZ, RAMP_S, MAX_VEL, MAX_ERR, LIMIT_MARGIN = 50.0, 1.0, 0.5, 0.6, 0.05
# FSM ids from unitree_sdk2/include/unitree/robot/r1/loco/r1_loco_client.hpp
FSM_NAMES = {0: "ZeroTorque (motors unpowered)", 1: "Damp", 4: "StandUp (position lock)", 811: "Start (balance control)"}
FSM_ARM_OK = {4, 811}   # states in which the built-in controller drives the arms; extend only with evidence


def ease(x):  # cosine ease 0..1 -> 0..1
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


class State:
    def __init__(self):
        self.msg, self.count, self.t_last = None, 0, 0.0
    def on_msg(self, m: LowState_):
        self.msg, self.count, self.t_last = m, self.count + 1, time.time()


def target_at(t, q0, q1, move_s, hold_s):
    if t < move_s:                 return q0 + (q1 - q0) * ease(t / move_s)
    if t < move_s + hold_s:        return q1
    if t < 2 * move_s + hold_s:    return q1 + (q0 - q1) * ease((t - move_s - hold_s) / move_s)
    return q0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iface")
    ap.add_argument("--joint", default="left_shoulder_pitch", choices=sorted(BY_NAME))
    ap.add_argument("--delta", type=float, default=-0.25, help="radians to add to the measured angle")
    ap.add_argument("--move-s", type=float, default=2.0)
    ap.add_argument("--hold-s", type=float, default=1.0)
    ap.add_argument("--execute", action="store_true", help="actually publish to rt/arm_sdk")
    a = ap.parse_args()
    slot, name, mj_name, kp, kd = BY_NAME[a.joint]

    ChannelFactoryInitialize(0, a.iface)
    st = State()
    sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(st.on_msg, 10)
    t0 = time.time()
    while st.count < 10 and time.time() - t0 < 3.0:
        time.sleep(0.05)
    if st.msg is None:
        sys.exit(f"no rt/lowstate on {a.iface}; nothing done")
    q_meas = {s: st.msg.motor_state[s].q for s, *_ in JOINTS}
    print(f"lowstate: {st.count} msgs, mode_machine={st.msg.mode_machine}")

    fsm = mode = None
    try:
        lc = LocoClient(); lc.SetTimeout(3.0); lc.Init(); lc._RegistApi(ROBOT_API_ID_LOCO_GET_FSM_MODE, 0)
        code, data = lc._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "")
        fsm = json.loads(data)["data"] if code == 0 and data else None
        code2, data2 = lc._Call(ROBOT_API_ID_LOCO_GET_FSM_MODE, "")
        mode = json.loads(data2)["data"] if code2 == 0 and data2 else None
        print(f"fsm id: {fsm} = {FSM_NAMES.get(fsm, 'unknown')}   fsm mode: {mode}   (rpc codes {code}, {code2})")
    except Exception as e:
        print(f"fsm query failed: {e}")

    # plan + checks
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    q0 = q_meas[slot]; q1 = q0 + a.delta
    if mj_name:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
        lo, hi = model.jnt_range[jid]
        if not (lo + LIMIT_MARGIN <= q1 <= hi - LIMIT_MARGIN):
            sys.exit(f"ABORT: target {q1:.3f} outside {mj_name} range [{lo:.3f}, {hi:.3f}] with margin")
    vel = abs(a.delta) / a.move_s
    if vel > MAX_VEL:
        sys.exit(f"ABORT: {vel:.2f} rad/s exceeds cap {MAX_VEL}")
    total = 2 * a.move_s + a.hold_s
    print(f"\nplan: {name} (slot {slot}) {q0:+.3f} -> {q1:+.3f} rad over {a.move_s}s, hold {a.hold_s}s, back over {a.move_s}s; "
          f"peak {vel:.2f} rad/s; {RATE_HZ:.0f} Hz; weight ramp {RAMP_S}s each side; total {total + 2*RAMP_S:.1f}s")
    print("held joints (slot name measured):")
    for s, n, _, k, d in JOINTS:
        tag = "  <-- moves" if s == slot else ""
        print(f"  {s:2d} {n:20s} {q_meas[s]:+.3f}  kp={k:g} kd={d:g}{tag}")

    # preview plan for sim/preview.py (mujoco joint names; head is not in the sim model)
    times = [0.0, a.move_s, a.move_s + a.hold_s, total]
    frames = []
    for t in times:
        tgt = {}
        for s, n, mj, *_ in JOINTS:
            if mj: tgt[mj] = round(float(target_at(t, q0, q1, a.move_s, a.hold_s) if s == slot else q_meas[s]), 6)
        frames.append({"time_s": t, "joint_targets_rad": tgt})
    plan_path = ROOT / "sim/plans/arm_lift_dryrun.json"
    plan_path.write_text(json.dumps({"schema_version": 1, "name": f"{name} {a.delta:+.2f} rad lift", "duration_s": total, "keyframes": frames}, indent=1) + "\n")
    # forward kinematics at the start and peak poses (no physics, no transient)
    data = mujoco.MjData(model)
    def hand_at(q_slot):
        data.qpos[:] = 0.0
        for s_, n_, mj, *_ in JOINTS:
            if mj:
                data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj)]] = q_slot if s_ == slot else q_meas[s_]
        mujoco.mj_forward(model, data)
        side = "left" if name.startswith("left") else "right"
        return data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_preview")].copy()
    h0, h1 = hand_at(q0), hand_at(q1)
    print(f"\nkinematic check: hand xyz {h0.round(3)} -> {h1.round(3)} m; rise {100*(h1[2]-h0[2]):+.1f} cm, forward {100*(h1[0]-h0[0]):+.1f} cm")
    print(f"plan written to {plan_path.relative_to(ROOT)}  (view: mjpython sim/preview.py --plan {plan_path.relative_to(ROOT)} --preview-only)")

    if fsm not in FSM_ARM_OK:
        print(f"\nNOTE: controller is in FSM {fsm} = {FSM_NAMES.get(fsm, 'unknown')}. The arm topic only takes effect in "
              f"{sorted(FSM_ARM_OK)}; --execute is refused in this state.")
        if a.execute: sys.exit(3)

    if not a.execute:
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_pr = 100; mc = cmd.motor_cmd[slot]; mc.q, mc.dq, mc.tau, mc.kp, mc.kd = q1, 0.0, 0.0, kp, kd
        print(f"\nDRY RUN, nothing published. At peak the message would carry: mode_pr={cmd.mode_pr} (weight 1.0), "
              f"motor_cmd[{slot}] q={mc.q:+.3f} dq=0 tau=0 kp={mc.kp:g} kd={mc.kd:g}; other 12 joints held at measured q.")
        sub.Close(); return

    # ---- execute ----
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_); pub.Init()
    crc = CRC(); cmd = unitree_hg_msg_dds__LowCmd_()
    for s, n, _, k, d in JOINTS:
        mc = cmd.motor_cmd[s]; mc.q, mc.dq, mc.tau, mc.kp, mc.kd = q_meas[s], 0.0, 0.0, k, d

    def send(weight, q_cmd):
        cmd.mode_pr = int(round(np.clip(weight, 0.0, 1.0) * 100))
        cmd.motor_cmd[slot].q = float(q_cmd)
        cmd.crc = crc.Crc(cmd); pub.Write(cmd)

    def release(q_cmd, seconds=RAMP_S):
        t_r = time.time()
        while (el := time.time() - t_r) < seconds:
            send(1.0 - el / seconds, q_cmd); time.sleep(1.0 / RATE_HZ)
        send(0.0, q_cmd)

    print("\nEXECUTE: ramping weight up")
    dt = 1.0 / RATE_HZ; t_start = time.time(); err_since = None; q_cmd = q0
    try:
        while True:
            now = time.time(); t = now - t_start
            if t < RAMP_S:
                w, q_cmd = t / RAMP_S, q0
            elif t < RAMP_S + total:
                w, q_cmd = 1.0, target_at(t - RAMP_S, q0, q1, a.move_s, a.hold_s)
            else:
                break
            send(w, q_cmd)
            q_now = st.msg.motor_state[slot].q
            if now - st.t_last > 0.5:
                print("ABORT: lowstate stale"); release(q_cmd, 0.5); sys.exit(2)
            if abs(q_now - q_cmd) > MAX_ERR:
                err_since = err_since or now
                if now - err_since > 0.3:
                    print(f"ABORT: tracking error {q_now - q_cmd:+.2f} rad"); release(q_cmd, 0.5); sys.exit(2)
            else:
                err_since = None
            if int(t / 0.5) != int((t - dt) / 0.5):
                print(f"  t={t:4.1f}s weight={w:.2f} cmd={q_cmd:+.3f} meas={q_now:+.3f}")
            time.sleep(max(0.0, dt - (time.time() - now)))
        print("ramping weight down"); release(q_cmd)
        print(f"done. final measured {st.msg.motor_state[slot].q:+.3f} rad (started {q0:+.3f})")
    except KeyboardInterrupt:
        print("\ninterrupted: releasing"); release(q_cmd, 0.5); sys.exit(130)
    finally:
        sub.Close()


if __name__ == "__main__":
    main()
