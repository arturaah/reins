"""Preview a planned R1 walking path before anything moves.

A walk plan is a list of floor waypoints. The preview draws the path, puts
translucent ghosts of the robot at each waypoint and the goal, and walks the
robot along it. Posing is kinematic: the gait is cosmetic, not a locomotion
controller, so this shows where the robot intends to go, not whether it can.

This module never opens the Unitree SDK or sends commands to physical hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "models" / "r1" / "scene_walk.xml"
DEFAULT_PLAN = ROOT / "plans" / "walk_around_table.json"
DEFAULT_OUTPUT = ROOT / "preview.json"
SAMPLE_PERIOD = 0.04

# Pelvis height with all joints at zero and the foot contact spheres touching the floor.
STAND_HEIGHT = 0.743
STRIDE = 0.6  # m per full left+right gait cycle, cosmetic only
WALK_SPEED = 0.5  # m/s, for playback and export timing
END_PAUSE = 1.0  # s to hold at the goal before looping

PATH_RGBA = np.array([1.0, 0.6, 0.1, 1.0], dtype=np.float32)
WAYPOINT_RGBA = np.array([1.0, 0.85, 0.2, 1.0], dtype=np.float32)
START_RGBA = np.array([0.3, 0.9, 0.4, 1.0], dtype=np.float32)
GHOST_RGBA = np.array([0.4, 0.8, 1.0, 0.25], dtype=np.float32)
GOAL_GHOST_RGBA = np.array([0.4, 0.8, 1.0, 0.45], dtype=np.float32)


# --- plan -------------------------------------------------------------------

@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    yaw: float | None = None  # radians; None faces the direction of travel
    label: str | None = None


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Trajectory:
    """A piecewise-linear path through floor waypoints, parameterised by arc length."""

    def __init__(self, waypoints: list[Waypoint]):
        if len(waypoints) < 2:
            raise ValueError("A walk plan needs at least two waypoints")
        self.waypoints = list(waypoints)
        self.points = np.array([(w.x, w.y) for w in waypoints], dtype=float)
        if not np.all(np.isfinite(self.points)):
            raise ValueError("Waypoints must be finite")
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cumlen = np.concatenate([[0.0], np.cumsum(seg)])
        self.yaws = self._resolve_yaws()

    @property
    def length(self) -> float:
        return float(self.cumlen[-1])

    def _resolve_yaws(self) -> np.ndarray:
        headings = []
        for i in range(len(self.points)):
            j, k = (i, i + 1) if i + 1 < len(self.points) else (i - 1, i)
            d = self.points[k] - self.points[j]
            headings.append(np.arctan2(d[1], d[0]))
        return np.array([w.yaw if w.yaw is not None else h
                         for w, h in zip(self.waypoints, headings)])

    def pose_at(self, s: float) -> tuple[float, float, float]:
        """(x, y, yaw) at arc length s, clamped to the path."""
        s = float(np.clip(s, 0.0, self.length))
        i = min(int(np.searchsorted(self.cumlen, s, side="right") - 1), len(self.points) - 2)
        span = self.cumlen[i + 1] - self.cumlen[i]
        t = 0.0 if span == 0 else (s - self.cumlen[i]) / span
        x, y = (1 - t) * self.points[i] + t * self.points[i + 1]
        yaw = self.yaws[i] + t * _wrap(self.yaws[i + 1] - self.yaws[i])
        return float(x), float(y), float(_wrap(yaw))

    def sample(self, n: int) -> np.ndarray:
        return np.array([self.pose_at(s) for s in np.linspace(0, self.length, n)])


def load_plan(path: Path) -> Trajectory:
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1:
        raise ValueError("Plan schema_version must be 1")
    return Trajectory([Waypoint(**w) for w in plan["waypoints"]])


# --- robot pose ---------------------------------------------------------------

def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class Poser:
    """Writes a base pose plus an optional cosmetic gait into qpos and runs FK."""

    JOINTS = ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "shoulder_roll", "elbow")

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.qadr = {f"{side}_{j}": model.jnt_qposadr[model.joint(f"{side}_{j}_joint").id]
                     for side in ("left", "right") for j in self.JOINTS}
        self.base = model.jnt_qposadr[model.joint("floating_base_joint").id]

    def set(self, data: mujoco.MjData, x: float, y: float, yaw: float,
            gait_phase: float | None = None) -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        q, a = data.qpos, self.qadr
        # Relaxed arms. Elbow zero on the R1 holds the forearm horizontal.
        q[a["left_shoulder_roll"]], q[a["right_shoulder_roll"]] = 0.15, -0.15
        q[a["left_elbow"]] = q[a["right_elbow"]] = 1.0

        z = STAND_HEIGHT
        if gait_phase is not None:
            s = np.sin(2 * np.pi * gait_phase)
            for side, sign in (("left", 1.0), ("right", -1.0)):
                fwd = sign * s  # >0 while this leg swings forward
                hip = -0.2 - 0.35 * fwd  # hip pitch is negative for forward flexion
                knee = 0.4 + 0.5 * max(fwd, 0.0)
                q[a[f"{side}_hip_pitch"]] = hip
                q[a[f"{side}_knee"]] = knee
                q[a[f"{side}_ankle_pitch"]] = -(hip + knee)  # keep the sole level
                q[a[f"{side}_shoulder_pitch"]] = 0.4 * fwd  # arms counter-swing
            z -= 0.05

        q[self.base:self.base + 3] = (x, y, z)
        q[self.base + 3:self.base + 7] = yaw_quat(yaw)
        mujoco.mj_forward(self.model, data)


# --- overlay ------------------------------------------------------------------
# These append decoration geoms to any mjvScene, so they serve both the viewer
# (viewer.user_scn) and offscreen rendering (Renderer.scene after update_scene).

def _add_geom(scene: mujoco.MjvScene, kind, size, pos, rgba) -> mujoco.MjvGeom | None:
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel(), rgba)
    scene.ngeom += 1
    return geom


def _segment(scene, start, end, width, rgba, kind=mujoco.mjtGeom.mjGEOM_CAPSULE) -> None:
    geom = _add_geom(scene, kind, np.zeros(3), np.zeros(3), rgba)
    if geom is not None:
        mujoco.mjv_connector(geom, kind, width, np.asarray(start, float), np.asarray(end, float))


def draw_path(scene: mujoco.MjvScene, traj: Trajectory, progress: float | None = None,
              z: float = 0.02) -> None:
    """Path line, waypoint markers and heading arrows. Travelled part is dimmed."""
    samples = traj.sample(max(2, int(traj.length / 0.05)))
    arc = np.linspace(0, traj.length, len(samples))
    done_rgba = PATH_RGBA * np.array([1, 1, 1, 0.3], dtype=np.float32)
    for (x0, y0, _), (x1, y1, _), s in zip(samples, samples[1:], arc[1:]):
        rgba = done_rgba if progress is not None and s <= progress else PATH_RGBA
        _segment(scene, (x0, y0, z), (x1, y1, z), 0.025, rgba)
    for i, (w, yaw) in enumerate(zip(traj.waypoints, traj.yaws)):
        pos = np.array([w.x, w.y, z])
        rgba = START_RGBA if i == 0 else WAYPOINT_RGBA
        marker = _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.05, 0, 0), pos, rgba)
        if marker is not None:
            marker.label = w.label or ""
        tip = pos + 0.35 * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        _segment(scene, pos, tip, 0.03, rgba, kind=mujoco.mjtGeom.mjGEOM_ARROW)


def draw_ghost(scene: mujoco.MjvScene, model: mujoco.MjModel, poser: Poser,
               ghost: mujoco.MjData, pose: tuple[float, float, float], rgba: np.ndarray) -> None:
    """A translucent copy of the robot at pose (x, y, yaw)."""
    poser.set(ghost, *pose)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1  # visual meshes only
    start = scene.ngeom
    mujoco.mjv_addGeoms(model, ghost, opt, mujoco.MjvPerturb(),
                        mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    for geom in scene.geoms[start:scene.ngeom]:
        geom.rgba = rgba
        geom.objtype, geom.objid, geom.segid = mujoco.mjtObj.mjOBJ_UNKNOWN, -1, -1


# --- preview ------------------------------------------------------------------

class WalkPreview:
    def __init__(self, traj: Trajectory):
        self.traj = traj
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.ghost = mujoco.MjData(self.model)
        self.poser = Poser(self.model)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0  # hide collision geoms; the floor is in group 2

    def pose_robot(self, s: float) -> None:
        moving = 0.0 < s < self.traj.length
        self.poser.set(self.data, *self.traj.pose_at(s),
                       gait_phase=s / STRIDE if moving else None)

    def decorate(self, scene: mujoco.MjvScene, progress: float | None) -> None:
        draw_path(scene, self.traj, progress)
        for i in range(1, len(self.traj.waypoints) - 1):
            draw_ghost(scene, self.model, self.poser, self.ghost,
                       self.traj.pose_at(self.traj.cumlen[i]), GHOST_RGBA)
        draw_ghost(scene, self.model, self.poser, self.ghost,
                   self.traj.pose_at(self.traj.length), GOAL_GHOST_RGBA)

    def camera(self) -> mujoco.MjvCamera:
        """Overview from behind and above the start, looking along the travel direction."""
        cam = mujoco.MjvCamera()
        lo, hi = self.traj.points.min(0), self.traj.points.max(0)
        cam.lookat[:] = (*((lo + hi) / 2), 0.4)
        cam.distance = max(3.0, 1.4 * float(np.linalg.norm(hi - lo)) + 1.5)
        d = self.traj.points[-1] - self.traj.points[0]
        cam.azimuth = (np.degrees(np.arctan2(d[1], d[0])) if np.any(d) else 0.0) + 25
        cam.elevation = -40
        return cam

    def timeline(self, fps: float) -> list[float]:
        """Arc-length values for one playback pass, including the pause at the goal."""
        n = max(2, int(self.traj.length / WALK_SPEED * fps))
        return list(np.linspace(0, self.traj.length, n)) + [self.traj.length] * int(END_PAUSE * fps)

    def export(self, source_plan: str) -> dict:
        """Floor-projected base path in the same frame and units as preview.py's export."""
        duration = self.traj.length / WALK_SPEED
        samples = []
        for t in np.append(np.arange(0.0, duration, SAMPLE_PERIOD), duration):
            x, y, yaw = self.traj.pose_at(t * WALK_SPEED)
            samples.append({"time_s": round(float(t), 4),
                            "base_xyz_m": [round(x, 5), round(y, 5), 0.0],
                            "yaw_rad": round(yaw, 5)})
        return {
            "schema_version": 1,
            "source_plan": source_plan,
            "robot": "Unitree R1 kinematic walk preview",
            "frame": "mujoco_world",
            "position_units": "m",
            "angle_units": "rad",
            "tracked": ["base"],
            "waypoints": [{"xyz_m": [w.x, w.y, 0.0], "yaw_rad": round(float(yaw), 5),
                           "label": w.label} for w, yaw in zip(self.traj.waypoints, self.traj.yaws)],
            "samples": samples,
        }

    def render(self, renderer: mujoco.Renderer, s: float) -> np.ndarray:
        self.pose_robot(s)
        renderer.update_scene(self.data, self.camera(), self.opt)
        self.decorate(renderer.scene, s)
        return renderer.render()

    def save_image(self, path: Path, width: int = 960, height: int = 540, fps: float = 20) -> Path:
        """.png: still with the robot at the start. .gif: the full walk."""
        from PIL import Image
        with mujoco.Renderer(self.model, height, width) as renderer:
            if path.suffix.lower() == ".gif":
                frames = [Image.fromarray(self.render(renderer, s)) for s in self.timeline(fps)]
                frames[0].save(path, save_all=True, append_images=frames[1:],
                               duration=int(1000 / fps), loop=0)
            else:
                Image.fromarray(self.render(renderer, 0.0)).save(path)
        return path

    def view(self) -> None:
        from mujoco import viewer as mjviewer
        try:
            viewer = mjviewer.launch_passive(self.model, self.data)
        except RuntimeError:
            if sys.platform == "darwin":
                sys.exit("On macOS the viewer must run under mjpython: mjpython sim/walk_preview.py")
            raise
        fps = 60.0
        with viewer:
            viewer.opt.geomgroup[0] = 0
            cam = self.camera()
            viewer.cam.lookat[:] = cam.lookat
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                cam.distance, cam.azimuth, cam.elevation)
            frames = self.timeline(fps)
            i = 0
            while viewer.is_running():
                tick = time.perf_counter()
                s = frames[i % len(frames)]
                with viewer.lock():
                    self.pose_robot(s)
                    viewer.user_scn.ngeom = 0
                    self.decorate(viewer.user_scn, s)
                viewer.sync()
                i += 1
                time.sleep(max(0.0, 1 / fps - (time.perf_counter() - tick)))


def run(plan_path: Path, output_path: Path, headless: bool, gif: Path | None = None) -> None:
    preview = WalkPreview(load_plan(plan_path))
    output_path.write_text(json.dumps(preview.export(plan_path.name), indent=2) + "\n")
    print(f"Preview: {len(preview.traj.waypoints)} waypoints, {preview.traj.length:.3f} m of walking")
    print(f"Saved: {output_path}")
    print(f"Visual: {preview.save_image(output_path.with_suffix('.png'))}")
    if gif:
        print(f"Animation: {preview.save_image(gif, 640, 360)}")
    if not headless:
        print("Orange line is the planned path; blue ghosts mark waypoints and the goal.")
        preview.view()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gif", type=Path, help="Also render the walk to this GIF")
    args = parser.parse_args()
    run(args.plan, args.output, args.headless, args.gif)
