"""MuJoCo backend for `Loco`: the R1 moves by the commanded velocity.

Kinematic, not physics: the base follows the commanded body velocity through
an acceleration limit, and the legs play a cosmetic gait. That matches how
Reins uses the real robot, where the onboard policy owns balance and we only
send velocities. What it models from the real API: walking only in FSM 811,
commands expiring after `duration`, and the limits in `Limits`.

What it adds: geoms named `obstacle*` in the scene are solid. A step that
would put the robot's footprint into one is refused, and the next
`set_velocity` returns ERR_BLOCKED. The real robot would just walk into it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from .base import (ERR_BLOCKED, ERR_NOT_WALKING, FSM_DAMP, FSM_STANCE, FSM_WALK, OK,
                   Limits, Pose2, wrap)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
import walk_preview  # noqa: E402  (Poser and overlay live with the sim)

SCENE_ROOM = REPO / "sim" / "models" / "r1" / "scene_room.xml"
FOOTPRINT_RADIUS = 0.25  # m, conservative circle around the R1's feet and arms
SUBSTEP = 0.01  # s
LINEAR_ACCEL = 1.0  # m/s^2
ANGULAR_ACCEL = 2.0  # rad/s^2
STRIDE = walk_preview.STRIDE


class SimLoco:
    def __init__(self, scene: Path = SCENE_ROOM, start: Pose2 = Pose2(0, 0, 0),
                 limits: Limits = Limits()):
        self.limits = limits
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.poser = walk_preview.Poser(self.model)
        self.obstacles = _obstacles(self.model)
        self.time = 0.0
        self._fsm = FSM_DAMP
        self._pose = start
        self._vel = np.zeros(3)
        self._cmd = np.zeros(3)
        self._cmd_until = 0.0
        self._gait = 0.0
        self._blocked = False
        self.collisions = 0
        self._update_model()

    # --- Loco interface ---------------------------------------------------

    def damp(self) -> int:
        return self._set_fsm(FSM_DAMP)

    def stance(self) -> int:
        return self._set_fsm(FSM_STANCE)

    def start(self) -> int:
        return self._set_fsm(FSM_WALK)

    def set_velocity(self, vx: float, vy: float, wz: float, duration: float = 1.0) -> int:
        if self._fsm != FSM_WALK:
            return ERR_NOT_WALKING
        if self._blocked:
            self._blocked = False
            self._cmd[:] = 0.0
            return ERR_BLOCKED
        self._cmd[:] = self.limits.clamp(vx, vy, wz)
        self._cmd_until = self.time + duration
        return OK

    def stop(self) -> int:
        self._cmd[:] = 0.0
        return OK

    def fsm_id(self) -> int:
        return self._fsm

    def pose(self) -> Pose2:
        return self._pose

    def pose_is_estimated(self) -> bool:
        return False

    # --- simulation -------------------------------------------------------

    def advance(self, dt: float) -> None:
        """Step simulated time forward by dt seconds."""
        for _ in range(max(1, round(dt / SUBSTEP))):
            self._substep(SUBSTEP)
        self._update_model()

    def velocity(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self._vel)

    def clearance(self, pose: Pose2 | None = None) -> float:
        """Distance from the footprint edge to the nearest obstacle (inf if none)."""
        p = pose or self._pose
        return min((ob.distance(p.x, p.y) for ob in self.obstacles), default=math.inf) - FOOTPRINT_RADIUS

    def _set_fsm(self, fsm: int) -> int:
        self._fsm = fsm
        if fsm != FSM_WALK:
            self._cmd[:] = 0.0
            self._vel[:] = 0.0
        return OK

    def _substep(self, dt: float) -> None:
        self.time += dt
        target = self._cmd if self.time <= self._cmd_until else np.zeros(3)
        step = np.array([LINEAR_ACCEL, LINEAR_ACCEL, ANGULAR_ACCEL]) * dt
        self._vel += np.clip(target - self._vel, -step, step)
        vx, vy, wz = (float(v) for v in self._vel)
        p = self._pose
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        nxt = Pose2(p.x + (c * vx - s * vy) * dt, p.y + (s * vx + c * vy) * dt, wrap(p.yaw + wz * dt))
        if self.clearance(nxt) < 0 and self.clearance(nxt) < self.clearance(p):
            self._vel[:] = 0.0
            if not self._blocked:
                self.collisions += 1
            self._blocked = True
            return
        self._gait += (math.hypot(vx, vy) + 0.15 * abs(wz)) * dt / STRIDE
        self._pose = nxt

    def _update_model(self) -> None:
        moving = float(np.abs(self._vel).max()) > 0.02
        self.poser.set(self.data, self._pose.x, self._pose.y, self._pose.yaw,
                       gait_phase=self._gait if moving else None)


class _Box:
    def __init__(self, x, y, yaw, hx, hy):
        self.x, self.y, self.yaw, self.hx, self.hy = x, y, yaw, hx, hy

    def distance(self, px: float, py: float) -> float:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx, ly = c * (px - self.x) + s * (py - self.y), -s * (px - self.x) + c * (py - self.y)
        dx, dy = max(abs(lx) - self.hx, 0.0), max(abs(ly) - self.hy, 0.0)
        return math.hypot(dx, dy)


class _Circle:
    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r

    def distance(self, px: float, py: float) -> float:
        return max(math.hypot(px - self.x, py - self.y) - self.r, 0.0)


_ROUND = {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_SPHERE),
          int(mujoco.mjtGeom.mjGEOM_CAPSULE)}


def _obstacles(model: mujoco.MjModel) -> list:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    found = []
    for g in range(model.ngeom):
        name = model.geom(g).name
        if not name.startswith("obstacle"):
            continue
        x, y = (float(v) for v in data.geom_xpos[g][:2])
        xmat = data.geom_xmat[g].reshape(3, 3)
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])
        size = [float(v) for v in model.geom_size[g]]
        kind = int(model.geom_type[g])
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            found.append(_Box(x, y, yaw, size[0], size[1]))
        elif kind in _ROUND:
            found.append(_Circle(x, y, size[0]))
        else:
            raise ValueError(f"obstacle geom {name!r}: only box, cylinder, sphere, capsule supported")
    return found
