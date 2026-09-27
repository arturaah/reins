"""A proposed joint trajectory as a plan file in the sim contract, for the twin's ghost preview. No SDK.

tools/cockpit.py loads the file with GET /preview?file=...&hold=1 and plays it as translucent arms
over the live robot until the operator answers; the desktop window does that call when the harness
prints a PROPOSAL line (harness.__main__.make_confirm writes the file just before).
"""
import json
from pathlib import Path

from .kinematics import ARM_JOINTS, OTHER_JOINTS, ROOT


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
