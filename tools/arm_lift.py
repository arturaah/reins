"""Move R1 arm joints through rt/arm_sdk: one joint, or a keyframe plan.

Dry run (default): subscribes to rt/lowstate, asks the controller its FSM id,
builds the trajectory from the measured pose, checks joint limits and speed,
prints the hand path per keyframe from forward kinematics, and writes the
resolved plan for sim/preview.py. Publishes nothing.

--execute: streams the trajectory at 50 Hz with the blend weight ramped 0 -> 1
before and 1 -> 0 after, watching every moving joint the whole time.

    .venv/bin/python tools/arm_lift.py en6                          # one-joint dry run
    .venv/bin/python tools/arm_lift.py en6 --execute                # moves the robot
    .venv/bin/python tools/arm_lift.py en6 --plan tools/plans/cup_grab_right.json [--execute]
    one-joint options: --joint left_shoulder_pitch --delta -0.25 --move-s 2 --hold-s 1
Every --execute run is logged to recordings/<timestamp>_<name>.json with the
commanded and measured trajectory; replay one with --plan recordings/<file>.json.
--record PATH overrides the file name.

Plan files use the sim contract (schema_version 1, keyframes with MuJoCo joint
names in radians, first keyframe at t=0). The t=0 values are replaced by the
measured pose so every plan starts where the arm actually is, and a return to
the measured pose is appended over --return-s seconds. Joints a plan does not
name are held at their measured angle.

Protocol facts (vendored C++ SDK, robots/r1/r1_pub.h and defines.h): hg LowCmd
on rt/arm_sdk, weight = mode_pr in 0..100, controller's 35-slot joint layout.
"""
import argparse, json, signal, sys, time
from pathlib import Path
import numpy as np
import mujoco

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
ROBOT_API_ID_LOCO_GET_FSM_MODE = 7002   # in the C++ r1_loco_api.hpp, missing from the Python file

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "sim/models/r1/R1_fixed_base.xml"

# slot, name, mujoco joint (None = not in the sim model), kp, kd -- order and gains as in
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
BY_MJ = {j[2]: j for j in JOINTS if j[2]}
RATE_HZ, RAMP_S, MAX_VEL, MAX_ERR, LIMIT_MARGIN = 50.0, 1.0, 1.5, 0.6, 0.05   # MAX_VEL was 0.5 until Artur raised it (2026-09-26 evening) so hand-taught takes replay at real speed
APPROACH_VEL = 0.25   # rad/s for the lead-in from the measured pose to a recording's first sample
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


class Plan:
    """times[i] and frames[i] = {slot: q}; every frame names the same slots.
    Sparse, hand-authored keyframes ease in and out per segment; dense ones
    (recordings, median spacing under 0.25 s) interpolate linearly, otherwise
    every segment would stop and restart."""
    def __init__(self, times, frames, linear=None):
        self.times, self.frames, self.slots = times, frames, sorted(frames[0])
        gaps = np.diff(times)
        self.linear = (len(gaps) > 0 and float(np.median(gaps)) < 0.25) if linear is None else linear
    @property
    def duration(self): return self.times[-1]
    def at(self, t):
        if t >= self.duration: return dict(self.frames[-1])
        i = max(k for k, tk in enumerate(self.times) if tk <= t)
        t0, t1 = self.times[i], self.times[i + 1]
        x = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        r = x if self.linear else ease(x)
        return {s: self.frames[i][s] + (self.frames[i + 1][s] - self.frames[i][s]) * r for s in self.slots}


def build_plan(a, q_meas):
    if a.plan:
        src = json.loads(Path(a.plan).read_text())
        assert src.get("schema_version") == 1, "plan schema_version must be 1"
        kfs = sorted(src["keyframes"], key=lambda f: f["time_s"])
        assert kfs[0]["time_s"] == 0.0, "first keyframe must be at t=0"
        names = set(kfs[0]["joint_targets_rad"])
        for f in kfs:
            assert set(f["joint_targets_rad"]) == names, "keyframes must name the same joints"
        unknown = names - set(BY_MJ)
        assert not unknown, f"joints not on the arm topic or not in the sim model: {sorted(unknown)}"
        slots = {n: BY_MJ[n][0] for n in names}
        times = [float(f["time_s"]) / a.speed for f in kfs]
        frames = [{slots[n]: float(v) for n, v in f["joint_targets_rad"].items()} for f in kfs]
        start = {s: q_meas[s] for s in slots.values()}
        src_gaps = np.diff([float(f["time_s"]) for f in kfs])
        recorded = len(src_gaps) > 0 and float(np.median(src_gaps)) < 0.25
        if recorded:
            # A recording's first sample is a real pose (often drooped a little from where the controller holds the
            # arm now). Keep it and approach it from the measured pose slowly, instead of within one sample interval.
            jump = max(abs(frames[0][s] - start[s]) for s in start)
            lead = max(0.5, jump / APPROACH_VEL)
            if jump > 0.05:
                print(f"note: the arm is {jump:.2f} rad from the recording's first pose; {lead:.1f} s lead-in added")
            times = [0.0] + [lead + t for t in times]
            frames = [start] + frames
        else:
            frames[0] = start                                          # hand-authored plans start where the arm is
        times.append(times[-1] + a.return_s); frames.append(dict(start))   # and come back
        label = src.get("name", Path(a.plan).name)
        return Plan(times, frames, linear=recorded), label
    else:
        slot = BY_NAME[a.joint][0]
        q0 = q_meas[slot]; q1 = a.to if a.to is not None else q0 + a.delta
        times = [0.0, a.move_s, a.move_s + a.hold_s, 2 * a.move_s + a.hold_s]
        frames = [{slot: q0}, {slot: q1}, {slot: q1}, {slot: q0}]
        label = f"{a.joint} to {q1:+.2f} rad"
    return Plan(times, frames), label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iface")
    ap.add_argument("--plan", help="keyframe plan JSON (sim contract, MuJoCo joint names)")
    ap.add_argument("--return-s", type=float, default=2.5, help="seconds for the appended return to the measured pose")
    ap.add_argument("--speed", type=float, default=1.0, help="time scale for --plan: 0.5 plays it at half speed")
    ap.add_argument("--kp-scale", type=float, default=1.0, help="multiply Unitree's arm kp (stiffer replay = less gravity droop)")
    ap.add_argument("--joint", default="left_shoulder_pitch", choices=sorted(BY_NAME))
    ap.add_argument("--delta", type=float, default=-0.25, help="radians to add to the measured angle (one-joint mode)")
    ap.add_argument("--to", type=float, help="absolute target in radians (one-joint mode); overrides --delta")
    ap.add_argument("--move-s", type=float, default=2.0)
    ap.add_argument("--hold-s", type=float, default=1.0)
    ap.add_argument("--execute", action="store_true", help="actually publish to rt/arm_sdk")
    ap.add_argument("--record", help="override the recording path (default recordings/<timestamp>_<name>.json)")
    ap.add_argument("--brief", action="store_true", help="print only what a person acts on (the desktop window uses this)")
    a = ap.parse_args()
    detail = (lambda *x: None) if a.brief else print     # per-joint tables and protocol facts: CLI only

    if a.plan and json.loads(Path(a.plan).read_text()).get('preview_only'):
        sys.exit('ABORT: this generated prompt plan is preview-only; physical execution is not enabled.')

    ChannelFactoryInitialize(0, a.iface)
    st = State()
    sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(st.on_msg, 10)
    t0 = time.time()
    while st.count < 10 and time.time() - t0 < 3.0:
        time.sleep(0.05)
    if st.msg is None:
        sys.exit(f"no rt/lowstate on {a.iface}; nothing done")
    q_meas = {s: st.msg.motor_state[s].q for s, *_ in JOINTS}
    detail(f"lowstate: {st.count} msgs, mode_machine={st.msg.mode_machine}")

    fsm = mode = None
    try:
        lc = LocoClient(); lc.SetTimeout(3.0); lc.Init(); lc._RegistApi(ROBOT_API_ID_LOCO_GET_FSM_MODE, 0)
        code, data = lc._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "")
        fsm = json.loads(data)["data"] if code == 0 and data else None
        code2, data2 = lc._Call(ROBOT_API_ID_LOCO_GET_FSM_MODE, "")
        mode = json.loads(data2)["data"] if code2 == 0 and data2 else None
        if a.brief: print(f"robot: FSM {fsm} = {FSM_NAMES.get(fsm, 'unknown')}")
        else: print(f"fsm id: {fsm} = {FSM_NAMES.get(fsm, 'unknown')}   fsm mode: {mode}   (rpc codes {code}, {code2})")
    except Exception as e:
        print(f"fsm query failed: {e}")

    plan, label = build_plan(a, q_meas)
    moving = plan.slots
    names = {s: n for s, n, *_ in JOINTS}

    # limits and speed
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    for s in moving:
        mj = BY_NAME[names[s]][2]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj)
        lo, hi = model.jnt_range[jid]
        for t, f in zip(plan.times, plan.frames):
            if not (lo <= f[s] <= hi):
                print(f"ABORT before sending anything: {names[s]} = {f[s]:+.3f} at t={t:.1f} s is outside the joint range [{lo:.3f}, {hi:.3f}]"); sys.exit(4)
            clamped = float(np.clip(f[s], lo + LIMIT_MARGIN, hi - LIMIT_MARGIN))   # inside the range but in the margin band: pull to the band edge
            if clamped != f[s]:
                print(f"note: {names[s]} {f[s]:+.3f} at t={t}s pulled to {clamped:+.3f} to keep the {LIMIT_MARGIN} rad margin from the limit")
                f[s] = clamped
    ts = np.arange(0.0, plan.duration + 1e-9, 0.01)
    qs = np.array([[plan.at(t)[s] for s in moving] for t in ts])
    vel = np.abs(np.diff(qs, axis=0)) / 0.01 if len(ts) > 1 else np.zeros((1, len(moving)))
    peak = float(vel.max()) if vel.size else 0.0
    if peak > MAX_VEL:
        j, k = np.unravel_index(vel.argmax(), vel.shape)
        if ts[j] >= plan.times[-2] - 1e-6:                              # the appended return leg: --speed does not scale it
            fix = f"The return to the start pose is what is too fast: end the recording nearer to where it began, or raise --return-s to {a.return_s * peak / MAX_VEL * 1.05:.1f}."
        else:
            fix = f"Lower the speed to {int(a.speed * MAX_VEL / peak * 100 * 0.97) / 100:.2f} or less."
        print(f"ABORT before sending anything: {names[moving[k]]} would move at {peak:.2f} rad/s at t={ts[j]:.1f} s, over the {MAX_VEL} rad/s cap. {fix}"); sys.exit(4)
    if a.brief:
        print(f"plan '{label}': {plan.duration:.1f} s, {len(moving)} joints move, peak speed {peak:.2f} rad/s (cap {MAX_VEL})")
    else:
        print(f"\nplan '{label}': {len(plan.times)} keyframes ({'linear' if plan.linear else 'eased'}), {plan.duration:.1f} s, {len(moving)} joints move, "
              f"peak {peak:.2f} rad/s; {RATE_HZ:.0f} Hz; weight ramp {RAMP_S}s each side; total {plan.duration + 2*RAMP_S:.1f}s")

    # forward kinematics per keyframe (no physics)
    data = mujoco.MjData(model)
    sides = [sd for sd in ("left", "right") if any(names[s].startswith(sd) for s in moving)] or ["left"]
    side = sides[0]                                   # the detailed table shows one hand; the brief line reports every moving hand
    sites = {sd: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sd}_hand_preview") for sd in sides}
    def fk(frame):
        data.qpos[:] = 0.0
        for s_, n_, mj, *_ in JOINTS:
            if mj:
                data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj)]] = frame.get(s_, q_meas[s_])
        mujoco.mj_forward(model, data)
        return {sd: data.site_xpos[k].copy() for sd, k in sites.items()}, data.ncon
    h0, ncon0 = fk(plan.frames[0])
    detail(f"keyframes ({side} hand xyz in m, relative to start; contacts vs {ncon0} at rest):")
    detail("   t    " + "  ".join(f"{names[s][:14]:>14s}" for s in moving) + "     dx     dy     dz  contacts")
    reach, contact_t = {sd: 0.0 for sd in sides}, None
    for t, f in zip(plan.times, plan.frames):
        h, ncon = fk(f)
        d = h[side] - h0[side]
        for sd in sides: reach[sd] = max(reach[sd], float(np.linalg.norm(h[sd] - h0[sd])))
        if ncon > ncon0 and contact_t is None: contact_t = t
        detail(f"  {t:4.1f}  " + "  ".join(f"{f[s]:+14.3f}" for s in moving) + f"  {d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}  {ncon:5d}" + ("  <-- new contacts!" if ncon > ncon0 else ""))
    detail("held joints: " + ", ".join(f"{names[s]}={q_meas[s]:+.2f}" for s, *_ in JOINTS if s not in moving))
    if a.brief:
        print(", ".join(f"{sd} hand travels up to {reach[sd]:.2f} m" for sd in sides) + " from where it is now" +
              (f"; WARNING: the model shows the arm touching something from t={contact_t:.1f} s" if contact_t is not None else "; no self-contact in the model"))

    resolved = ROOT / "sim/plans/arm_lift_dryrun.json"
    resolved.write_text(json.dumps({"schema_version": 1, "name": label, "duration_s": plan.duration,
        "keyframes": [{"time_s": t, "joint_targets_rad": {BY_NAME[names[s]][2]: round(f[s], 6) for s in moving}}
                      for t, f in zip(plan.times, plan.frames)],
        # measured pose of the joints that stay put, so spectacles/plan_feed.py draws both hands where they are
        "held_joints_rad": {mj: round(q_meas[s], 6) for s, _, mj, *_ in JOINTS if mj and s not in moving}},
        indent=1) + "\n")
    detail(f"resolved plan written to {resolved.relative_to(ROOT)} (view: mjpython sim/preview.py --plan {resolved.relative_to(ROOT)} --preview-only)")

    if fsm not in FSM_ARM_OK:
        print(f"\nNOTE: controller is in FSM {fsm} = {FSM_NAMES.get(fsm, 'unknown')}. The arm topic only takes effect in "
              f"{sorted(FSM_ARM_OK)}; --execute is refused in this state.")
        if a.execute: sys.exit(3)
    if not a.execute:
        if a.brief: print("checks passed. Dry run: nothing was sent to the robot.")
        else: print("\nDRY RUN, nothing published. Messages would carry mode_pr=100 (weight 1.0) and, per joint, q from the plan, dq=0, tau=0, kp/kd from Unitree's example.")
        sub.Close(); return

    # ---- execute ----
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_); pub.Init()
    crc = CRC(); cmd = unitree_hg_msg_dds__LowCmd_()
    for s, n, _, k, d in JOINTS:
        mc = cmd.motor_cmd[s]; mc.q, mc.dq, mc.tau, mc.kp, mc.kd = q_meas[s], 0.0, 0.0, k * a.kp_scale, d

    def send(weight, targets):
        cmd.mode_pr = int(round(np.clip(weight, 0.0, 1.0) * 100))
        for s, q in targets.items(): cmd.motor_cmd[s].q = float(q)
        cmd.crc = crc.Crc(cmd); pub.Write(cmd)

    def release(targets, seconds=RAMP_S):
        """Ramp the weight to 0. A Ctrl-C while this runs is ignored: the ramp must finish."""
        signal.signal(signal.SIGINT, lambda *_: print("(already releasing: the weight ramps down first)"))
        t_r = time.time()
        try:
            while (el := time.time() - t_r) < seconds:
                send(1.0 - el / seconds, targets); time.sleep(1.0 / RATE_HZ)
        finally:
            send(0.0, targets)

    rec, rec_last = [], -1.0
    def save_recording():
        if not rec: return
        slug = "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")[:40]
        path = Path(a.record) if a.record else ROOT / "recordings" / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}.json"
        mj = {s: BY_NAME[names[s]][2] for s in moving}
        out = {"schema_version": 1, "name": f"recording of {label}", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "duration_s": round(rec[-1][0], 3),
               "keyframes": [{"time_s": round(t, 3),
                              "joint_targets_rad": {mj[s]: round(c[s], 4) for s in moving},
                              "measured_rad": {mj[s]: round(m[s], 4) for s in moving}} for t, c, m in rec]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(f"recorded {len(rec)} keyframes over {out['duration_s']} s to {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")

    print("\nEXECUTE: ramping weight up")
    dt = 1.0 / RATE_HZ; t_start = time.time(); err_since = None; targets = plan.at(0.0); ticks = 0; lag_max = (0.0, None)
    try:
        while True:
            now = time.time(); t = now - t_start
            if t < RAMP_S:
                w, targets = t / RAMP_S, plan.at(0.0)
            elif t < RAMP_S + plan.duration:
                w, targets = 1.0, plan.at(t - RAMP_S)
            else:
                break
            send(w, targets); ticks += 1
            if t >= RAMP_S and (t - RAMP_S) - rec_last >= 0.05:
                rec_last = 0.0 if not rec else t - RAMP_S          # first sample is stamped exactly 0
                rec.append((rec_last, dict(targets), {s: st.msg.motor_state[s].q for s in moving}))
            errs = {s: st.msg.motor_state[s].q - targets[s] for s in moving}
            worst = max(errs, key=lambda s: abs(errs[s]))
            if abs(errs[worst]) > lag_max[0]: lag_max = (abs(errs[worst]), worst)
            if now - st.t_last > 0.5:
                print("ABORT: lowstate stale"); release(targets, 0.5); sys.exit(2)
            if abs(errs[worst]) > MAX_ERR:
                err_since = err_since or now
                if now - err_since > 0.3:
                    print(f"ABORT: {names[worst]} lags by {errs[worst]:+.2f} rad"); release(targets, 0.5); sys.exit(2)
            else:
                err_since = None
            if a.brief:
                if t >= RAMP_S and int((t - RAMP_S) / 2.0) != int((t - RAMP_S - dt) / 2.0):
                    print(f"  {t - RAMP_S:4.1f} / {plan.duration:.1f} s   lag {abs(errs[worst]):.2f} rad ({names[worst]})")
            elif int(t / 0.5) != int((t - dt) / 0.5):
                print(f"  t={t:4.1f}s w={w:.2f}  worst lag {names[worst]} {errs[worst]:+.3f}  " +
                      " ".join(f"{names[s][:8]}={targets[s]:+.2f}" for s in moving[:5]))
            time.sleep(max(0.0, dt - (time.time() - now)))
        detail(f"loop: {ticks} ticks in {t:.1f} s = {ticks / t:.0f} Hz (target {RATE_HZ:.0f})")
        print("ramping weight down"); release(targets); save_recording()
        back = max(abs(st.msg.motor_state[s].q - q_meas[s]) for s in moving)
        if a.brief:
            print(f"done. Worst lag during the motion {lag_max[0]:.2f} rad ({names[lag_max[1]] if lag_max[1] is not None else '-'}); "
                  f"arm back within {back:.2f} rad of where it started")
        else:
            print("done. final vs start: " + ", ".join(f"{names[s]} {st.msg.motor_state[s].q:+.3f}/{q_meas[s]:+.3f}" for s in moving))
    except KeyboardInterrupt:
        print("\ninterrupted: releasing")
        try: release(targets, 0.5)
        finally: save_recording()
        sys.exit(130)
    finally:
        sub.Close()


if __name__ == "__main__":
    main()
