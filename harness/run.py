"""Let an LLM drive the R1 in simulation, with every plan reviewed first.

    mjpython harness/run.py "put the red cube on the counter"           # Claude via Claude Code, live viewer
    python3 harness/run.py --headless "bring the blue bottle to the counter"
    python3 harness/run.py --brain demo --headless --auto-approve --gif harness/demo.gif

For each plan: read it in the terminal (and the viewer, or the preview PNG),
then type y to run it, n to decline, or any other text to decline with that
feedback, which goes back to the model. While the robot moves, press Enter to
stop it. The default brain runs Claude through the local Claude Code CLI, on
your Claude Code login. `--brain api` calls the Anthropic API directly and needs
ANTHROPIC_API_KEY. `--brain demo` runs the pick-and-place with no model: it uses
the same tools, and only cheats at choosing which pixel to point at.

The model gets only what the real robot has: joint readings, drifting odometry,
and the head camera (image and depth). Nothing tells it what's in the room.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reins_harness.agent import AutoApprove, Harness, SessionLog, TerminalReviewer  # noqa: E402
from reins_harness.brains import AnthropicBrain, ClaudeCodeBrain  # noqa: E402
from reins_harness.demo import PointingDemoBrain  # noqa: E402
from reins_harness.display import Display, Lines, ViewerDisplay  # noqa: E402
from reins_harness.skills import Skills  # noqa: E402
from reins_harness.world import SimWorld  # noqa: E402

DEFAULT_TASK = "Pick up the red cube from the table and put it on the counter."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--brain", choices=["claude-code", "api", "demo"], default="claude-code")
    parser.add_argument("--model", help="model id; defaults to Claude Code's model, or claude-opus-5 for --brain api")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--headless", action="store_true", help="no viewer window")
    parser.add_argument("--auto-approve", action="store_true", help="approve every plan without asking")
    parser.add_argument("--gif", type=Path, help="record the run as an animated GIF")
    parser.add_argument("--out", type=Path, default=HERE / "sessions", help="where logs and previews go")
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--no-drift", action="store_true", help="perfect odometry (it drifts by default)")
    args = parser.parse_args()

    world = SimWorld(odom_drift=not args.no_drift)
    if args.brain == "demo":
        brain = PointingDemoBrain(world)
    elif args.brain == "api":
        brain = AnthropicBrain(model=args.model or "claude-opus-5", effort=args.effort)
    else:
        brain = ClaudeCodeBrain(model=args.model, effort=args.effort, max_turns=args.max_turns)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out / stamp
    lines = None if args.auto_approve and args.headless else Lines()
    display = (Display if args.headless else ViewerDisplay)(world, out, gif=args.gif, lines=lines)
    reviewer = AutoApprove() if args.auto_approve else TerminalReviewer(lines)
    log = SessionLog(out / "session.jsonl")
    harness = Harness(Skills(world), brain, reviewer, display, log, max_turns=args.max_turns)

    print(f"Task: {args.task}\nBrain: {brain.name}\nSession log and plan previews: {out}")
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


if __name__ == "__main__":
    sys.exit(main())
