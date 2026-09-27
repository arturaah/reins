"""Legacy read-only R1 plan inspection from measured joint state.

This utility subscribes to telemetry, queries the controller state, prints a
kinematic inspection and writes sim/plans/arm_lift_dryrun.json. It cannot publish
arm commands. Its historical lead-in/return calculations are diagnostics, not the
shared dashboard validator or an authorization to execute the resulting file.

    .venv/bin/python tools/arm_lift.py IFACE --plan PATH

For motion, start tools/dashboard.py and submit a task there, or use
`python -m tools.reins prompt "wave with the right arm"`. Review and approve the
complete proposal in the dashboard or paired glasses. --execute is retired.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import mujoco

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
    def on_msg(self, m):
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
    ap.add_argument("--joint", default="left_shoulder_pitch", choices=sorted(BY_NAME))
    ap.add_argument("--delta", type=float, default=-0.25, help="radians to add to the measured angle (one-joint mode)")
    ap.add_argument("--to", type=float, help="absolute target in radians (one-joint mode); overrides --delta")
    ap.add_argument("--move-s", type=float, default=2.0)
    ap.add_argument("--hold-s", type=float, default=1.0)
    ap.add_argument("--execute", action="store_true", help="retired: use the dashboard complete-motion review")
    ap.add_argument("--brief", action="store_true", help="print only what a person acts on (the desktop window uses this)")
    a = ap.parse_args()
    if a.execute:
        ap.error("--execute is retired. Submit a complete motion through tools/dashboard.py or python -m tools.reins prompt; approve in dashboard/glasses.")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
    from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
    detail = (lambda *x: None) if a.brief else print     # per-joint tables and protocol facts: CLI only

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
    resolved.write_text(json.dumps({"schema_version": 1, "preview_only": True, "name": label, "duration_s": plan.duration,
        "keyframes": [{"time_s": t, "joint_targets_rad": {BY_NAME[names[s]][2]: round(f[s], 6) for s in moving}}
                      for t, f in zip(plan.times, plan.frames)],
        # measured pose of the joints that stay put, so spectacles/plan_feed.py draws both hands where they are
        "held_joints_rad": {mj: round(q_meas[s], 6) for s, _, mj, *_ in JOINTS if mj and s not in moving}},
        indent=1) + "\n")
    detail(f"resolved plan written to {resolved.relative_to(ROOT)} (view: mjpython sim/preview.py --plan {resolved.relative_to(ROOT)} --preview-only)")

    print("Dry-run inspection complete; no arm commands were sent. Submit motion through the dashboard for full validation and approval.")
    sub.Close()


if __name__ == "__main__":
    main()
