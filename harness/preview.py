"""A proposed joint trajectory as a plan file in the sim contract, for the twin's ghost preview. No SDK.

tools/cockpit.py loads the file with GET /preview?file=...&hold=1 and plays it as translucent arms
over the live robot until the operator answers; the desktop window does that call when the harness
prints a PROPOSAL line (harness.__main__.make_confirm writes the file just before).
"""
import json
from pathlib import Path

from .kinematics import ARM_JOINTS, OTHER_JOINTS, ROOT


def write_walk_plan(path, arm, dx, dy, dyaw, duration, joints, name):
    """A whole-body step as a plan: the arms hold where they are, and base_keyframes (spectacles/plan_feed.py's planar base
    path, origin = the robot's current base pose, x forward, y left, yaw positive = left) go from 0 to (dx, dy, dyaw) over
    the step's duration. The twin walks a translucent ghost along it; the glasses feed transforms the hand paths by it."""
    names = ARM_JOINTS[arm]
    other = ARM_JOINTS["left" if arm == "right" else "right"] + OTHER_JOINTS
    hold = {n: round(float(joints[n]), 6) for n in names if n in joints}
    held = {n: float(joints[n]) for n in other if n in joints}
    duration = round(max(float(duration), 0.1), 4)
    obj = {"schema_version": 1, "name": name, "source": "harness walk proposal", "duration_s": duration, "held_joints_rad": held,
           "keyframes": [{"time_s": 0.0, "joint_targets_rad": hold}, {"time_s": duration, "joint_targets_rad": dict(hold)}],
           "base_keyframes": [{"time_s": 0, "x_m": 0, "y_m": 0, "yaw_rad": 0},
                              {"time_s": duration, "x_m": round(float(dx), 4), "y_m": round(float(dy), 4), "yaw_rad": round(float(dyaw), 5)}]}
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) + "\n")
    return p


def write_plan(path, arm, q_now, frames, dt, joints, name):
    """path: file to write. q_now: the arm's 5 joints now; frames: the vetted 5-joint frames, dt apart;
    joints: every joint the backend reports (the other arm and the waist are held there)."""
    names = ARM_JOINTS[arm]
    other = ARM_JOINTS["left" if arm == "right" else "right"] + OTHER_JOINTS
    held = {n: float(joints[n]) for n in other if n in joints}
    kfs = [{"time_s": 0.0, "joint_targets_rad": {n: round(float(v), 6) for n, v in zip(names, q_now)}}]
    kfs += [{"time_s": round((i + 1) * dt, 4), "joint_targets_rad": {n: round(float(v), 6) for n, v in zip(names, f)}}
            for i, f in enumerate(frames)]
    obj = {"schema_version": 1, "name": name, "source": "harness proposal", "duration_s": kfs[-1]["time_s"],
           "held_joints_rad": held, "keyframes": kfs}
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) + "\n")
    return p
