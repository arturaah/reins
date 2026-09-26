"""The high-level walking interface, shaped like Unitree's R1 LocoClient.

Every backend (MuJoCo sim, real robot) implements `Loco`. Code above this line,
such as the path follower, the skills and the VLM tools, only talks to `Loco`,
so it runs unchanged on either.

Semantics copied from unitree_sdk2 `r1::LocoClient` (service "sport"):
- `set_velocity(vx, vy, wz, duration)` is body-frame m/s and rad/s. The robot
  keeps that velocity for `duration` seconds, then stops by itself. Re-sending
  every ~0.1 s with a short duration doubles as a dead-man switch.
- FSM ids: 0 zero torque, 1 damp, 4 stance, 811 start (walking),
  701 lie-to-stand, 702 stand-to-lie. The robot only walks in 811.
- Calls return an int code, 0 on success.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

FSM_ZERO_TORQUE = 0
FSM_DAMP = 1
FSM_STANCE = 4
FSM_WALK = 811
FSM_LIE_TO_STAND = 701
FSM_STAND_TO_LIE = 702

OK = 0
ERR_NOT_WALKING = 7400  # Reins-side code: velocity sent while not in FSM_WALK
ERR_BLOCKED = 7401  # Reins-side code: sim stopped the robot before a collision


@dataclass(frozen=True)
class Limits:
    """Command limits. Unverified on hardware; start conservative and raise with care."""
    vx_forward: float = 0.6
    vx_backward: float = 0.3
    vy: float = 0.3
    wz: float = 0.8

    def clamp(self, vx: float, vy: float, wz: float) -> tuple[float, float, float]:
        return (min(max(vx, -self.vx_backward), self.vx_forward),
                min(max(vy, -self.vy), self.vy),
                min(max(wz, -self.wz), self.wz))


@dataclass(frozen=True)
class Pose2:
    """Planar base pose in the map frame: metres, radians."""
    x: float
    y: float
    yaw: float

    def to_body(self, x: float, y: float) -> tuple[float, float]:
        """A map-frame point expressed in this pose's body frame (x forward, y left)."""
        dx, dy = x - self.x, y - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Loco(Protocol):
    limits: Limits

    def damp(self) -> int: ...
    def stance(self) -> int: ...
    def start(self) -> int: ...
    def set_velocity(self, vx: float, vy: float, wz: float, duration: float = 1.0) -> int: ...
    def stop(self) -> int: ...
    def fsm_id(self) -> int: ...

    def pose(self) -> Pose2:
        """Best available base pose. Exact in sim; dead-reckoned on the real robot."""
        ...

    def pose_is_estimated(self) -> bool:
        """True when pose() is dead reckoning rather than measured."""
        ...
