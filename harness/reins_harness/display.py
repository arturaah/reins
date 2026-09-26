"""Showing plans and motion: offscreen (PNG previews, GIF) or in the MuJoCo viewer.

Both draw the same overlay: the planned walking path in orange with a ghost of
the robot where it will stop, planned hand paths in cyan (left) and orange
(right), targets in yellow, and where the robot actually walked in blue.
"""
from __future__ import annotations

import math
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from .skills import Overlay
from .world import SimWorld, walk_preview

HAND_RGBA = {"left_hand": np.array([0.0, 1.0, 1.0, 0.7], np.float32),
             "right_hand": np.array([1.0, 0.35, 0.1, 0.7], np.float32)}
MARK_RGBA = np.array([1.0, 0.9, 0.1, 0.9], np.float32)
TRACE_RGBA = np.array([0.2, 0.6, 1.0, 0.8], np.float32)


class Lines:
    """Lines typed in the terminal, read on a background thread so drawing never blocks."""

    def __init__(self):
        self.queue: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in sys.stdin:
            self.queue.put(line.rstrip("\n"))
        self.queue.put(None)  # EOF

    def get(self, timeout: float) -> str | None:
        """The next line, "" on timeout, or None at end of input."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return ""

    def pending(self) -> bool:
        """True (and consumes the line) if something was typed."""
        try:
            self.queue.get_nowait()
            return True
        except queue.Empty:
            return False


class Display:
    """Offscreen. Subclassed by `ViewerDisplay` for the live window."""

    def __init__(self, world: SimWorld, out_dir: Path, gif: Path | None = None,
                 lines: Lines | None = None):
        self.world = world
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lines = lines
        self.overlay = Overlay()
        self.trace: list[tuple[float, float]] = []
        self.gif = gif
        self.frames: list[np.ndarray] = []
        self._ghost = mujoco.MjData(world.model)
        self._ghost_poser = walk_preview.Poser(world.model)
        self._previews = 0

    # --- the harness calls these -----------------------------------------------

    def show(self, overlay: Overlay) -> None:
        """Draw a plan. Plans are in the robot's odom frame; the scene is the true world, so
        they're mapped through the current odometry error: a drifted robot's plan is drawn
        where the robot would really go."""
        self.overlay = self._to_true(overlay)
        self.trace = []

    def _to_true(self, ov: Overlay) -> Overlay:
        w = self.world
        dyaw = w.true_pose().yaw - w.odom_pose().yaw

        def xy(p):
            return [float(v) for v in w.odom_to_true([p[0], p[1], 0.0])[:2]]

        walk = []
        for step in ov.walk:
            goal = dict(step["goal"])
            goal["x"], goal["y"] = xy([goal["x"], goal["y"]])
            if "yaw" in goal:
                goal["yaw"] = goal["yaw"] + dyaw
            walk.append({**step, "goal": goal, "path": {**step["path"], "points": [xy(q) for q in step["path"]["points"]]}})
        return Overlay(walk=walk, hands={h: [w.odom_to_true(q) for q in pts] for h, pts in ov.hands.items()},
                       marks=[w.odom_to_true(m) for m in ov.marks])

    def preview(self, overlay: Overlay) -> Path:
        """Save a PNG of the plan drawn over the current scene; return its path."""
        self.show(overlay)
        from PIL import Image
        self._previews += 1
        path = self.out_dir / f"plan-{self._previews:02d}.png"
        Image.fromarray(self.world.render(self.camera(), 960, 720, self.decorate)).save(path)
        return path

    def tick(self, dt: float) -> None:
        self.world.advance(dt)
        p = self.world.true_pose()
        self.trace.append((p.x, p.y))
        if self.gif and len(self.trace) % 5 == 0:
            self.frames.append(self.world.render(self.camera(), 640, 480, self.decorate))

    def should_stop(self) -> bool:
        return bool(self.lines and self.lines.pending())

    def close(self) -> None:
        if self.gif and self.frames:
            from PIL import Image
            images = [Image.fromarray(f) for f in self.frames]
            images[0].save(self.gif, save_all=True, append_images=images[1:], duration=100, loop=0)
            print(f"Animation: {self.gif}")
        self.world.close()

    # --- drawing ---------------------------------------------------------------

    def camera(self) -> mujoco.MjvCamera:
        """Above and behind the robot, framing it and everything in the plan."""
        pose = self.world.true_pose()
        pts = [np.array([pose.x, pose.y])]
        for step in self.overlay.walk:
            pts += [np.array(p) for p in step.get("path", {}).get("points", [])]
            pts.append(np.array([step["goal"]["x"], step["goal"]["y"]]))
        for path in self.overlay.hands.values():
            pts += [np.asarray(p)[:2] for p in path]
        cam = mujoco.MjvCamera()
        if self.overlay.hands and not self.overlay.walk:
            # Arm plans: from the front quarter, close in, or the body hides the hands.
            hand_pts = np.concatenate([np.asarray(p) for p in self.overlay.hands.values()])
            cam.lookat[:] = hand_pts.mean(0)
            cam.distance = 1.4
            cam.azimuth = math.degrees(pose.yaw) + 215
            cam.elevation = -35
            return cam
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        cam.lookat[:] = (*((lo + hi) / 2), 0.6)
        cam.distance = 2.0 + 1.2 * float(np.linalg.norm(hi - lo))
        cam.azimuth = math.degrees(pose.yaw) + 35  # behind, a little to the right
        cam.elevation = -40
        return cam

    def decorate(self, scene: mujoco.MjvScene) -> None:
        pose = self.world.true_pose()
        for step in self.overlay.walk:
            goal = step["goal"]
            pts = step.get("path", {}).get("points") or [[pose.x, pose.y]]
            wps = [walk_preview.Waypoint(x, y) for x, y in pts[:-1]]
            wps.append(walk_preview.Waypoint(goal["x"], goal["y"], goal.get("yaw"), "goal"))
            if len(wps) >= 2 and walk_preview.Trajectory(wps).length > 0.01:
                walk_preview.draw_path(scene, walk_preview.Trajectory(wps))
            elif goal.get("yaw") is not None:  # a turn in place: just the new heading
                base = np.array([goal["x"], goal["y"], 0.02])
                tip = base + 0.45 * np.array([math.cos(goal["yaw"]), math.sin(goal["yaw"]), 0])
                walk_preview._segment(scene, base, tip, 0.03, walk_preview.WAYPOINT_RGBA,
                                      kind=mujoco.mjtGeom.mjGEOM_ARROW)
        if self.overlay.walk:
            goal = self.overlay.walk[-1]["goal"]
            yaw = goal.get("yaw")
            if yaw is None:
                pts = self.overlay.walk[-1]["path"]["points"]
                yaw = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
            walk_preview.draw_ghost(scene, self.world.model, self._ghost_poser, self._ghost,
                                    (goal["x"], goal["y"], yaw), walk_preview.GOAL_GHOST_RGBA)
        for hand, path in self.overlay.hands.items():
            for a, b in zip(path, path[1:]):
                if np.linalg.norm(np.asarray(b) - np.asarray(a)) > 1e-4:
                    walk_preview._segment(scene, a, b, 0.008, HAND_RGBA[hand])
        for mark in self.overlay.marks:
            walk_preview._add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.02, 0, 0), mark, MARK_RGBA)
        for (x0, y0), (x1, y1) in zip(self.trace[::4], self.trace[4::4]):
            walk_preview._segment(scene, (x0, y0, 0.03), (x1, y1, 0.03), 0.012, TRACE_RGBA)


class ViewerDisplay(Display):
    """The live MuJoCo viewer. Run under `mjpython` on macOS."""

    def __init__(self, world: SimWorld, out_dir: Path, gif: Path | None = None,
                 lines: Lines | None = None):
        super().__init__(world, out_dir, gif, lines)
        from mujoco import viewer as mjviewer
        self.viewer = mjviewer.launch_passive(world.model, world.data)
        self.viewer.opt.geomgroup[0] = 0
        self._frame_camera()
        self.refresh()

    def _frame_camera(self) -> None:
        cam = self.camera()
        self.viewer.cam.lookat[:] = cam.lookat
        self.viewer.cam.distance = cam.distance
        self.viewer.cam.azimuth, self.viewer.cam.elevation = cam.azimuth, cam.elevation

    def refresh(self) -> None:
        if not self.viewer.is_running():
            return
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            self.decorate(self.viewer.user_scn)
        self.viewer.sync()

    def show(self, overlay: Overlay) -> None:
        super().show(overlay)
        self._frame_camera()
        self.refresh()

    def tick(self, dt: float) -> None:
        end = time.perf_counter() + dt
        super().tick(dt)
        self.refresh()
        time.sleep(max(0.0, end - time.perf_counter()))

    def should_stop(self) -> bool:
        return super().should_stop() or not self.viewer.is_running()

    def close(self) -> None:
        super().close()
        self.viewer.close()
