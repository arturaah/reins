#!/usr/bin/env python3
"""Prepend a planned straight walk to an arm dry run for AR preview only.

This writes a combined visual plan; it never commands R1 locomotion or arms.
The map origin is the robot's base pose when the shoulder tags are calibrated.
"""
import argparse
import json
from pathlib import Path


def combine(arm, distance_m, walk_s):
    if arm.get("schema_version") != 1 or len(arm.get("keyframes", [])) < 2:
        raise ValueError("arm plan needs schema_version 1 and at least two keyframes")
    if not (0 < distance_m <= 5 and 0 < walk_s <= 30):
        raise ValueError("distance_m must be 0..5 m and walk_s must be 0..30 s")
    frames = arm["keyframes"]
    if float(frames[0]["time_s"]) != 0:
        raise ValueError("arm plan must start at time 0")
    if any(float(b["time_s"]) <= float(a["time_s"]) for a, b in zip(frames, frames[1:])):
        raise ValueError("arm keyframe times must increase")
    duration = float(frames[-1]["time_s"])
    initial = dict(frames[0]["joint_targets_rad"])
    shifted = [{"time_s": round(walk_s + float(f["time_s"]), 5),
                "joint_targets_rad": dict(f["joint_targets_rad"])} for f in frames[1:]]
    return {
        "schema_version": 1,
        "preview_only": True,
        "name": f"walk {distance_m:g} m, then {arm.get('name', 'arm action')}",
        "duration_s": round(walk_s + duration, 5),
        "keyframes": [{"time_s": 0, "joint_targets_rad": initial},
                      {"time_s": walk_s, "joint_targets_rad": initial}, *shifted],
        "held_joints_rad": dict(arm.get("held_joints_rad", {})),
        "base_keyframes": [
            {"time_s": 0, "x_m": 0, "y_m": 0, "yaw_rad": 0},
            {"time_s": walk_s, "x_m": distance_m, "y_m": 0, "yaw_rad": 0},
            {"time_s": round(walk_s + duration, 5), "x_m": distance_m, "y_m": 0, "yaw_rad": 0},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_plan", type=Path, help="resolved arm plan from arm_lift.py dry run")
    parser.add_argument("--distance-m", type=float, required=True)
    parser.add_argument("--walk-s", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("sim/plans/walk_then_arm.json"))
    args = parser.parse_args()
    plan = combine(json.loads(args.arm_plan.read_text()), args.distance_m, args.walk_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote visual plan {args.output}: {plan['name']} ({plan['duration_s']} s)")


if __name__ == "__main__":
    main()
