"""Drawing a planned trajectory into a MuJoCo scene.

Everything here appends decoration geoms to an ``mjvScene``, so the same code
serves both the interactive viewer (``viewer.user_scn``) and offscreen
rendering (``Renderer.scene`` after ``update_scene``).
"""

import mujoco
import numpy as np

from .model import Poser
from .trajectory import Trajectory

PATH_RGBA = np.array([1.0, 0.6, 0.1, 1.0], dtype=np.float32)
WAYPOINT_RGBA = np.array([1.0, 0.85, 0.2, 1.0], dtype=np.float32)
START_RGBA = np.array([0.3, 0.9, 0.4, 1.0], dtype=np.float32)
GHOST_RGBA = np.array([0.4, 0.8, 1.0, 0.25], dtype=np.float32)
GOAL_GHOST_RGBA = np.array([0.4, 0.8, 1.0, 0.45], dtype=np.float32)

_IDENTITY = np.eye(3).flatten()


def _next_geom(scn: mujoco.MjvScene) -> mujoco.MjvGeom | None:
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    return g


def _segment(scn, a, b, width, rgba, kind=mujoco.mjtGeom.mjGEOM_CAPSULE):
    g = _next_geom(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), _IDENTITY, rgba)
    mujoco.mjv_connector(g, kind, width, np.asarray(a, float), np.asarray(b, float))


def _sphere(scn, pos, radius, rgba, label=""):
    g = _next_geom(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, float), _IDENTITY, rgba)
    g.label = label


def draw_path(scn: mujoco.MjvScene, traj: Trajectory, z: float = 0.02,
              progress: float | None = None) -> None:
    """Path line, waypoint markers and heading arrows.

    If ``progress`` (arc length) is given, the part already travelled is dimmed.
    """
    samples = traj.sample(max(2, int(traj.length / 0.05)))
    s_samples = np.linspace(0, traj.length, len(samples))
    done_rgba = PATH_RGBA * np.array([1, 1, 1, 0.3], dtype=np.float32)
    for (x0, y0, _), (x1, y1, _), s in zip(samples[:-1], samples[1:], s_samples[1:]):
        rgba = done_rgba if progress is not None and s <= progress else PATH_RGBA
        _segment(scn, (x0, y0, z), (x1, y1, z), 0.025, rgba)

    for i, (w, yaw) in enumerate(zip(traj.waypoints, traj.yaws)):
        pos = np.array([w.x, w.y, z])
        rgba = START_RGBA if i == 0 else WAYPOINT_RGBA
        _sphere(scn, pos, 0.05, rgba, label=w.label or "")
        tip = pos + 0.35 * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        _segment(scn, pos, tip, 0.03, rgba, kind=mujoco.mjtGeom.mjGEOM_ARROW)


def draw_ghost(scn: mujoco.MjvScene, model: mujoco.MjModel, poser: Poser,
               ghost_data: mujoco.MjData, pose: tuple[float, float, float],
               rgba: np.ndarray = GHOST_RGBA) -> None:
    """A translucent copy of the robot at ``pose`` (x, y, yaw)."""
    poser.set(ghost_data, *pose)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1  # visual meshes only
    start = scn.ngeom
    mujoco.mjv_addGeoms(model, ghost_data, opt, mujoco.MjvPerturb(),
                        mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
    for i in range(start, scn.ngeom):
        scn.geoms[i].rgba = rgba
        # Ghosts are decoration; keep them out of mouse selection.
        scn.geoms[i].objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        scn.geoms[i].objid = -1
        scn.geoms[i].segid = -1


def draw_plan(scn: mujoco.MjvScene, model: mujoco.MjModel, poser: Poser,
              ghost_data: mujoco.MjData, traj: Trajectory,
              progress: float | None = None) -> None:
    """Full plan overlay: path, plus ghosts at intermediate waypoints and the goal."""
    draw_path(scn, traj, progress=progress)
    for i in range(1, len(traj.waypoints) - 1):
        draw_ghost(scn, model, poser, ghost_data,
                   traj.pose_at(traj.cumlen[i]), GHOST_RGBA)
    draw_ghost(scn, model, poser, ghost_data, traj.pose_at(traj.length), GOAL_GHOST_RGBA)
