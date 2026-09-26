"""Walk the R1 through a plan in MuJoCo, with approval first.

    mjpython loco/run_sim.py                                   # around the table
    mjpython loco/run_sim.py --to 1.8 1.8 1.5708               # straight line; hits the table
    python3 loco/run_sim.py --headless --gif loco/walk.gif     # no window, auto-approve

Approve or decline in the terminal: type y (approve) or n (decline) and press
Enter. While walking, press Enter in the terminal to abort. Closing the viewer
also declines or aborts. (The viewer's own keys can't be used: MuJoCo keeps
Enter, Backspace and most letters for itself.) The orange line is the plan,
cyan is where the robot went.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reins_loco.follower import FollowResult  # noqa: E402
from reins_loco.sim import SimLoco, walk_preview  # noqa: E402
from reins_loco.skills import execute_walk_step, plan_walk_to  # noqa: E402

DEFAULT_PLAN = Path(__file__).resolve().parents[1] / "sim" / "plans" / "walk_around_table.json"
TRACE_RGBA = np.array([0.2, 0.9, 1.0, 1.0], dtype=np.float32)


def step_from_plan_file(sim: SimLoco, path: Path) -> dict:
    """A walk_preview waypoint plan becomes one contract walk step."""
    waypoints = json.loads(path.read_text())["waypoints"]
    *via, goal = waypoints
    return plan_walk_to(sim.pose(), goal["x"], goal["y"], yaw=goal.get("yaw"),
                        via=[[w["x"], w["y"]] for w in via[1:]],
                        description=goal.get("label") and f"Walk to the {goal['label']}")


def decorate(scene: mujoco.MjvScene, sim: SimLoco, step: dict, trace: list) -> None:
    pts = step["path"]["points"]
    goal = step["goal"]
    wps = [walk_preview.Waypoint(x, y) for x, y in pts[:-1]]
    wps.append(walk_preview.Waypoint(goal["x"], goal["y"], goal.get("yaw"), "goal"))
    if len(wps) >= 2 and walk_preview.Trajectory(wps).length > 0:
        walk_preview.draw_path(scene, walk_preview.Trajectory(wps))
    for (x0, y0), (x1, y1) in zip(trace[::3], trace[3::3]):
        walk_preview._segment(scene, (x0, y0, 0.03), (x1, y1, 0.03), 0.012, TRACE_RGBA)


def camera(sim: SimLoco, step: dict) -> mujoco.MjvCamera:
    pts = np.array(step["path"]["points"] + [[ob.x, ob.y] for ob in sim.obstacles])
    lo, hi = pts.min(0), pts.max(0)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (*((lo + hi) / 2), 0.3)
    cam.distance = 1.3 * float(np.linalg.norm(hi - lo)) + 2.0
    cam.azimuth, cam.elevation = 65, -60
    return cam


def run_headless(sim: SimLoco, step: dict, gif: Path | None) -> FollowResult:
    frames, trace = [], []
    renderer = mujoco.Renderer(sim.model, 360, 640) if gif else None
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0
    cam = camera(sim, step)

    def tick(dt: float) -> None:
        sim.advance(dt)
        trace.append((sim.pose().x, sim.pose().y))
        if renderer and len(trace) % 2 == 0:
            renderer.update_scene(sim.data, cam, opt)
            decorate(renderer.scene, sim, step, trace)
            frames.append(renderer.render())

    result = execute_walk_step(step, sim, tick=tick)
    if renderer:
        from PIL import Image
        images = [Image.fromarray(f) for f in frames]
        images[0].save(gif, save_all=True, append_images=images[1:], duration=200, loop=0)
        renderer.close()
        print(f"Animation: {gif}")
    return result


def _terminal_lines() -> "queue.Queue[str]":
    """Lines typed in the terminal, read on a background thread so the viewer keeps drawing."""
    lines: "queue.Queue[str]" = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            lines.put(line.strip().lower())

    threading.Thread(target=read, daemon=True).start()
    return lines


def run_viewer(sim: SimLoco, step: dict) -> FollowResult | None:
    from mujoco import viewer as mjviewer
    viewer = mjviewer.launch_passive(sim.model, sim.data)
    lines = _terminal_lines()
    trace: list = []
    with viewer:
        viewer.opt.geomgroup[0] = 0
        cam = camera(sim, step)
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation

        def refresh() -> None:
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                decorate(viewer.user_scn, sim, step, trace)
            viewer.sync()

        print("Approve this plan? Type y or n in this terminal, then Enter.")
        answer = None
        while viewer.is_running() and answer is None:
            refresh()
            try:
                line = lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if line in ("y", "yes", "n", "no"):
                answer = line.startswith("y")
            else:
                print("Type y or n, then Enter.")
        if not answer:
            print("Declined.")
            return None
        print("Approved. Walking. Press Enter here to abort.")

        def tick(dt: float) -> None:
            end = time.perf_counter() + dt
            sim.advance(dt)
            trace.append((sim.pose().x, sim.pose().y))
            refresh()
            time.sleep(max(0.0, end - time.perf_counter()))

        result = execute_walk_step(step, sim, tick=tick,
                                   should_stop=lambda: not lines.empty() or not viewer.is_running())
        report(sim, result)
        print("Close the viewer to exit.")
        while viewer.is_running():
            sim.advance(0.05)  # let it settle to a standstill
            refresh()
            time.sleep(0.05)
    return result


def report(sim: SimLoco, result: FollowResult) -> None:
    p = result.pose
    print(f"{'Reached' if result.reached else 'Did not reach'} goal ({result.reason}): "
          f"x={p.x:.2f} y={p.y:.2f} yaw={math.degrees(p.yaw):.0f} deg, "
          f"{sim.time:.1f} s, collisions={sim.collisions}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="walk_preview waypoint plan")
    parser.add_argument("--to", type=float, nargs="+", metavar=("X", "Y"), help="goal x y [yaw] instead of a plan")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gif", type=Path, help="with --headless, save an animation")
    args = parser.parse_args()

    sim = SimLoco()
    if args.to:
        x, y, *yaw = args.to
        step = plan_walk_to(sim.pose(), x, y, yaw=yaw[0] if yaw else None)
    else:
        step = step_from_plan_file(sim, args.plan)
    print("Proposed step:", json.dumps(step))
    if args.headless:
        report(sim, run_headless(sim, step, args.gif))
    else:
        run_viewer(sim, step)


if __name__ == "__main__":
    main()
