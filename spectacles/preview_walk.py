#!/usr/bin/env python3
"""Plot the exact kinematic walk-and-arm paths served to Spectacles.

This is a path preview, not a balanced walking simulation or robot command.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from plan_feed import MJCF, base_path, hand_paths


def render(plan_path, output, points=200):
    plan = json.loads(plan_path.read_text())
    if "base_keyframes" not in plan:
        raise ValueError("preview_walk needs a plan with base_keyframes")
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    hands = hand_paths(model, plan, points)
    duration = float(plan["keyframes"][-1]["time_s"])
    base = base_path(plan, np.linspace(0, duration, points))
    fig, (top, side) = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, color in (("left", "#00bcd4"), ("right", "#ff7a34")):
        path = np.asarray(hands[name])
        top.plot(path[:, 0], path[:, 1], color=color, label=f"{name} hand")
        side.plot(path[:, 0], path[:, 2], color=color, label=f"{name} hand")
        top.scatter(path[0, 0], path[0, 1], color=color, s=35)
    top.plot(base[:, 0], base[:, 1], "k--", linewidth=1, label="robot base")
    top.set(xlabel="forward x (m)", ylabel="left y (m)", title="Top view")
    side.set(xlabel="forward x (m)", ylabel="height z (m)", title="Side view")
    for axis in (top, side):
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
    top.legend(loc="best")
    fig.suptitle(plan.get("name", plan_path.stem))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/walk-plan.png"))
    args = parser.parse_args()
    render(args.plan, args.output)


if __name__ == "__main__":
    main()
