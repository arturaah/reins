"""One resolved trajectory format for preview, approval and streaming."""
import copy
import hashlib
import json
import math

import numpy as np
from core.ik import ArmIK
from core.motion_validation import MotionValidator, slow_acceleration

ARM_JOINTS = {s: [f"{s}_{j}_joint" for j in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
              for s in ("left", "right")}


def digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frame_plan(arm, start, frames, dt, held, name="Arm motion"):
    if arm not in ARM_JOINTS or not math.isfinite(dt) or not .005 <= dt <= .1:
        raise ValueError("Invalid arm or command interval")
    values = np.asarray([start, *frames], dtype=float)
    if values.ndim != 2 or values.shape[1] != 5 or len(values) < 2 or len(values) > 36001 or not np.isfinite(values).all():
        raise ValueError("Expected a bounded sequence of five finite joint targets")
    names = ARM_JOINTS[arm]
    return {"schema_version": 1, "name": name, "duration_s": (len(values)-1)*dt,
            "held_joints_rad": {n: float(q) for n, q in held.items() if n not in names},
            "keyframes": [{"time_s": round(i*dt, 6), "joint_targets_rad": dict(zip(names, q.tolist()))}
                          for i, q in enumerate(values)]}


def resolve(plan, arm, pose, rate=50):
    """Resolve all five moving joints, resample once, then validate these exact samples."""
    if arm not in ARM_JOINTS:
        raise ValueError("Choose one arm")
    result = copy.deepcopy(plan)
    names = ARM_JOINTS[arm]
    ts = np.asarray([k["time_s"] for k in result["keyframes"]], float)
    if len(ts) < 2 or not np.isfinite(ts).all() or ts[0] != 0 or np.any(np.diff(ts) <= 0) or not 0 < ts[-1] <= 180:
        raise ValueError("Trajectory times must increase from zero and fit within 180 seconds")
    # Do not silently ignore any other moving joints.
    if any(set(k["joint_targets_rad"]) - set(names) for k in result["keyframes"]):
        raise ValueError("A proposal may move only its selected arm")
    for k in result["keyframes"]:
        k["joint_targets_rad"] = {n: float(k["joint_targets_rad"].get(n, pose[n])) for n in names}
    result["held_joints_rad"] = {n: float(q) for n, q in pose.items() if n not in names}
    if any(abs(result["keyframes"][0]["joint_targets_rad"][n] - pose[n]) > .001 for n in names):
        raise ValueError("Plan starts at a different pose; regenerate it from the current robot state")
    qs = np.array([list(k["joint_targets_rad"].values()) for k in result["keyframes"]])
    peak = float(np.max(np.abs(np.diff(qs, axis=0) / np.diff(ts)[:, None])))
    scale = max(1., peak/.35)
    for k in result["keyframes"]:
        k["time_s"] *= scale
    # Slow before resampling; interpolation onto the command grid must also meet acceleration limits.
    for _ in range(8):
        slow_acceleration(result, limit=1.2)
        ts = [k["time_s"] for k in result["keyframes"]]
        if ts[-1] > 180:
            raise ValueError("Resolved trajectory exceeds 180 seconds")
        times = np.arange(math.ceil(ts[-1]*rate)+1) / rate
        q = np.asarray([[k["joint_targets_rad"][n] for n in names] for k in result["keyframes"]])
        samples = np.column_stack([np.interp(times, ts, q[:, i]) for i in range(5)])
        result["keyframes"] = [{"time_s": round(float(t), 6), "joint_targets_rad": dict(zip(names, row.tolist()))}
                              for t, row in zip(times, samples)]
        result["duration_s"] = float(times[-1])
        velocity = np.diff(samples, axis=0)*rate
        acceleration = float(np.max(np.abs(np.diff(velocity, axis=0))))*rate if len(velocity)>1 else 0.
        if acceleration <= 1.4:
            break
    else:
        raise ValueError("Could not resolve a smooth command trajectory")
    return result


def validate(plan, arm, model=None, obstacles=None):
    model = model or ArmIK(backend="mujoco").model
    return MotionValidator(model).check(plan, arm, obstacles or [])


def start_pose(plan):
    return {**plan.get("held_joints_rad", {}), **plan["keyframes"][0]["joint_targets_rad"]}


def require_start(plan, measured, tolerance=.025):
    expected = start_pose(plan)
    if any(n not in measured or not math.isfinite(float(measured[n])) or abs(measured[n]-q) > tolerance
           for n, q in expected.items()):
        raise ValueError("Robot pose changed since review. Regenerate and review a new proposal.")


def frames(plan, arm):
    names = ARM_JOINTS[arm]
    keys = plan["keyframes"]
    dt = float(keys[1]["time_s"])
    if any(abs(k["time_s"] - i*dt) > 1e-5 for i, k in enumerate(keys)):
        raise ValueError("Execution requires uniformly timed resolved samples")
    return [[float(k["joint_targets_rad"][n]) for n in names] for k in keys[1:]], dt
