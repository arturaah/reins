#!/usr/bin/env python3
"""Serve a Reins arm plan to the Spectacles Lens as robot_base hand paths.

Reads a plan file (schema_version 1, keyframes of MuJoCo joint names in
radians: sim/plans, tools/plans, recordings/, or the resolved plan that
`tools/arm_lift.py IFACE --plan ...` writes on every dry run), poses the R1
fixed-base model at each sample (forward kinematics only, no physics), and
streams both hand-tip paths in the Lens's `trajectory` format. Opens nothing
towards the robot.

    .venv/bin/python spectacles/plan_feed.py sim/plans/arm_lift_dryrun.json
    .venv/bin/python spectacles/plan_feed.py tools/plans/cup_grab_right.json --print

The plan file is re-read when it changes, so a new arm_lift dry run shows up
on the glasses without restarting. The Lens falls back to its mock after 1.5 s
of silence, so the path is resent every --period seconds.

Frame: the fixed-base model pins the pelvis at 0.74 m above the world origin
with x forward, y left, z up, which is the Lens's robot_base (origin on the
floor under the pelvis). Joints a plan does not name are taken from its
optional `held_joints_rad` map (written by arm_lift's dry run from the
measured pose), else 0.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "sim/models/r1/R1_fixed_base.xml"
MAX_POINTS = 512   # the Lens rejects longer paths


def hand_paths(model, plan, points):
    frames = sorted(plan["keyframes"], key=lambda f: f["time_s"])
    times = np.array([float(f["time_s"]) for f in frames])
    names = list(frames[0]["joint_targets_rad"])
    held = {**plan.get("held_joints_rad", {})}
    adr = {}
    for name in set(names) | set(held):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint not in the R1 model: {name}")
        adr[name] = model.jnt_qposadr[jid]
    q = {n: np.array([float(f["joint_targets_rad"][n]) for f in frames]) for n in names}
    sites = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_preview")
             for side in ("left", "right")}
    data = mujoco.MjData(model)
    hands = {"left": [], "right": []}
    # Sample evenly in time: joint-space interpolation traces the same path
    # whether arm_lift eases the segments or not.
    for t in np.linspace(times[0], times[-1], points):
        data.qpos[:] = 0.0
        for n, v in held.items():
            data.qpos[adr[n]] = v
        for n in names:
            data.qpos[adr[n]] = np.interp(t, times, q[n])
        mujoco.mj_kinematics(model, data)
        for side, site in sites.items():
            hands[side].append([round(float(v), 4) for v in data.site_xpos[site]])
    return hands


def message(model, path, points):
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1 or len(plan.get("keyframes", [])) < 2:
        raise ValueError("plan needs schema_version 1 and at least two keyframes")
    return {
        "type": "trajectory",
        "version": 1,
        "id": f"{plan.get('name', path.stem)}@{int(path.stat().st_mtime)}",
        "frame": "robot_base",
        "units": "m",
        "duration_s": float(plan.get("duration_s", plan["keyframes"][-1]["time_s"])),
        "hands": hand_paths(model, plan, points),
    }


class Feed:
    def __init__(self, model, path, points):
        self.model, self.path, self.points = model, path, points
        self.mtime, self.text = None, None

    def current(self):
        """The latest message, re-reading the plan when its file changes."""
        try:
            mtime = self.path.stat().st_mtime
            if mtime != self.mtime:
                self.mtime = mtime
                msg = message(self.model, self.path, self.points)
                self.text = json.dumps(msg)
                span = {s: np.ptp(np.array(p), axis=0).round(3).tolist() for s, p in msg["hands"].items()}
                print(f"loaded {self.path.name}: {msg['duration_s']:.1f} s, "
                      f"{self.points} points per hand, extent (m) {span}", file=sys.stderr, flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"cannot use {self.path}: {exc}; keeping the previous path", file=sys.stderr, flush=True)
        return self.text


async def serve_feed(feed, host, port, period):
    from websockets.asyncio.server import serve

    async def handler(websocket):
        print(f"Lens connected: {websocket.remote_address}", file=sys.stderr, flush=True)
        try:
            while True:
                text = feed.current()
                if text:
                    await websocket.send(text)
                await asyncio.sleep(period)
        except Exception as exc:
            print(f"Lens disconnected: {exc}", file=sys.stderr, flush=True)

    async with serve(handler, host, port):
        print(f"Plan feed for {feed.path}: ws://{host}:{port}", file=sys.stderr, flush=True)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", type=Path, help="plan JSON (schema_version 1, MuJoCo joint names)")
    ap.add_argument("--points", type=int, default=200, help=f"samples per hand, 2..{MAX_POINTS}")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--period", type=float, default=0.25, help="seconds between resends (Lens times out at 1.5)")
    ap.add_argument("--print", action="store_true", help="print one message and exit, no server")
    a = ap.parse_args()
    if not 2 <= a.points <= MAX_POINTS:
        ap.error(f"--points must be 2..{MAX_POINTS}")
    feed = Feed(mujoco.MjModel.from_xml_path(str(MJCF)), a.plan.resolve(), a.points)
    if a.print:
        text = feed.current()
        if text is None:
            raise SystemExit(1)
        print(text)
        return
    asyncio.run(serve_feed(feed, a.host, a.port, a.period))


if __name__ == "__main__":
    main()
