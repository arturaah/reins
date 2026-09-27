"""Room mode: a model walks the R1 around a simulated room and hands the arm to the arm policy.

    mjpython -m harness.room "Put the red cube on the counter"                   # Claude via Claude Code, viewer
    python -m harness.room --headless "Bring the blue bottle to the counter"
    python -m harness.room --brain demo --headless --auto-approve --gif runs/room.gif

For each plan: read it in the terminal (and the viewer, or the preview PNG), then type y
to run it, n to decline, or any other text to decline with that feedback, which goes back
to the model. While the robot moves, press Enter to stop it.

Two models: the top-level brain (walking, looking, deciding what to do) and the arm policy
(harness.loop, one small hand move per step during a manipulate). The brain runs through
the local Claude Code CLI by default (--brain api: the Anthropic API; --brain demo: no
model). The arm policy is harness/config.yaml's vlm.provider, claude-cli here; --arm-vlm
picks another. --review-moves asks before every arm move, as `python -m harness live` does.

The models get only what the real robot has: joint readings, drifting odometry, and the
head camera (image and depth), plus the wrist camera for the arm policy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from harness.__main__ import parse_overrides
from harness.room.agent import AutoApprove, Harness, SessionLog, TerminalReviewer
from harness.room.brains import AnthropicBrain, ClaudeCodeBrain
from harness.room.demo import PointingDemoBrain, demo_arm_vlm
from harness.room.display import Display, Lines, ViewerDisplay
from harness.room.skills import Skills, room_config
from harness.room.world import REPO, SimWorld

DEFAULT_TASK = "Pick up the red cube from the table and put it on the counter."


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.room", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--brain", choices=["claude-code", "api", "demo"], default="claude-code")
    parser.add_argument("--model", help="top-level model id; defaults to Claude Code's model, or claude-opus-5 for --brain api")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--arm-vlm", help="arm policy provider: claude-cli | anthropic | chat | scripted (default: claude-cli)")
    parser.add_argument("--review-moves", action="store_true", help="ask before every arm move, not only every plan")
    parser.add_argument("--config", help="harness config file (default harness/config.yaml)")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value")
    parser.add_argument("--headless", action="store_true", help="no viewer window")
    parser.add_argument("--auto-approve", action="store_true", help="approve every plan without asking")
    parser.add_argument("--gif", type=Path, help="record the run as an animated GIF")
    parser.add_argument("--out", type=Path, default=REPO / "runs", help="where session logs and previews go")
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--no-drift", action="store_true", help="perfect odometry (it drifts by default)")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    if args.arm_vlm:
        overrides["vlm.provider"] = args.arm_vlm
    cfg = room_config(overrides) if not args.config else _load(args.config, overrides)

    world = SimWorld(odom_drift=not args.no_drift)
    arm_vlm = None
    if args.brain == "demo":
        brain = PointingDemoBrain(world)
        arm_vlm = demo_arm_vlm(world, brain)
    elif args.brain == "api":
        brain = AnthropicBrain(model=args.model or "claude-opus-5", effort=args.effort)
    else:
        brain = ClaudeCodeBrain(model=args.model, effort=args.effort, max_turns=args.max_turns)
    out = args.out / f"room_{time.strftime('%Y%m%d_%H%M%S')}"
    lines = None if args.auto_approve and args.headless else Lines()
    display = (Display if args.headless else ViewerDisplay)(world, out, gif=args.gif, lines=lines)
    reviewer = AutoApprove() if args.auto_approve else TerminalReviewer(lines)
    confirm = reviewer.confirm if args.review_moves and not args.auto_approve else None
    skills = Skills(world, cfg, arm_vlm=arm_vlm, confirm_moves=confirm)
    log = SessionLog(out / "session.jsonl")
    harness = Harness(skills, brain, reviewer, display, log, max_turns=args.max_turns)

    arm_name = "demo" if args.brain == "demo" else cfg["vlm"]["provider"]
    print(f"Task: {args.task}\nBrain: {brain.name}\nArm policy: {arm_name}\nSession log and plan previews: {out}")
    if lines:
        print("While the robot moves, press Enter to stop it.")
    try:
        harness.run(args.task)
        if not args.headless and isinstance(display, ViewerDisplay):
            print("\nClose the viewer to exit.")
            while display.viewer.is_running():
                time.sleep(0.1)
    finally:
        display.close()
    return 0


def _load(path, overrides):
    from harness import config
    return config.load(path, {"hand.type": "virtual", "vlm.provider": "claude-cli", **overrides})


if __name__ == "__main__":
    sys.exit(main())
