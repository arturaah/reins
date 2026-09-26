"""Turn a planned path into velocity commands for any `Loco` backend.

`PathFollower` is a pure controller: give it the current pose, get back a
body-frame velocity. `drive()` runs it against a backend with a tick function,
which is `time.sleep` on the real robot and `SimLoco.advance` in sim.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

from .base import OK, Limits, Loco, Pose2, wrap

COMMAND_PERIOD = 0.1  # s between velocity commands
COMMAND_DURATION = 0.5  # s each command stays valid; the robot stops if we go quiet


@dataclass
class FollowResult:
    reached: bool
    reason: str
    pose: Pose2
    trace: list[tuple[float, float]] = field(default_factory=list)


class PathFollower:
    """Pure pursuit along a polyline, then turn in place to the goal heading.

    Walks forward when the lookahead point is roughly ahead, turns in place when
    it's behind, and slows down approaching the goal.
    """

    def __init__(self, points: list[tuple[float, float]], goal_yaw: float | None,
                 limits: Limits, lookahead: float = 0.4,
                 pos_tolerance: float = 0.08, yaw_tolerance: float = 0.08):
        if len(points) < 2:
            raise ValueError("path needs at least two points")
        self.points = [(float(x), float(y)) for x, y in points]
        self.goal_yaw = goal_yaw
        self.limits = limits
        self.lookahead = lookahead
        self.pos_tolerance = pos_tolerance
        self.yaw_tolerance = yaw_tolerance
        self._dense = _densify(self.points, 0.02)
        self._progress = 0  # index into _dense of the closest point so far; only moves forward
        self._at_position = False

    @property
    def goal(self) -> tuple[float, float]:
        return self.points[-1]

    def _lookahead_point(self, pose: Pose2) -> tuple[float, float]:
        # Closest dense point, searching a short window ahead so the path can't be
        # "short-cut" where it doubles back near itself.
        window = self._dense[self._progress:self._progress + 100]
        nearest = min(range(len(window)),
                      key=lambda i: math.hypot(window[i][0] - pose.x, window[i][1] - pose.y))
        self._progress += nearest
        for px, py in self._dense[self._progress:]:
            if math.hypot(px - pose.x, py - pose.y) >= self.lookahead:
                return px, py
        return self.goal

    def command(self, pose: Pose2) -> tuple[float, float, float] | None:
        """Body-frame (vx, vy, wz), or None when the goal is reached."""
        gx, gy = self.goal
        dist = math.hypot(gx - pose.x, gy - pose.y)
        if dist < self.pos_tolerance:
            self._at_position = True
        if self._at_position:
            if self.goal_yaw is None:
                return None
            err = wrap(self.goal_yaw - pose.yaw)
            if abs(err) < self.yaw_tolerance:
                return None
            return self.limits.clamp(0.0, 0.0, 1.5 * err)

        tx, ty = self._lookahead_point(pose)
        bx, by = pose.to_body(tx, ty)
        heading_err = math.atan2(by, bx)
        wz = 2.0 * heading_err
        if abs(heading_err) > math.radians(50):
            return self.limits.clamp(0.0, 0.0, wz)  # face the path before walking
        speed = min(self.limits.vx_forward, 0.8 * dist + 0.1) * math.cos(heading_err)
        return self.limits.clamp(speed, 0.0, wz)


def _densify(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    dense = [points[0]]
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        n = max(1, math.ceil(math.hypot(bx - ax, by - ay) / spacing))
        dense += [(ax + (bx - ax) * i / n, ay + (by - ay) * i / n) for i in range(1, n + 1)]
    return dense


def drive(loco: Loco, follower: PathFollower, tick: Callable[[float], None] = time.sleep,
          timeout: float = 120.0, should_stop: Callable[[], bool] = lambda: False) -> FollowResult:
    """Run the follower until the goal, a failure, `timeout` or `should_stop()`.

    Always leaves the robot with a zero-velocity command.
    """
    trace = []
    elapsed = 0.0
    try:
        while elapsed < timeout:
            pose = loco.pose()
            trace.append((pose.x, pose.y))
            if should_stop():
                return FollowResult(False, "stopped", pose, trace)
            cmd = follower.command(pose)
            if cmd is None:
                return FollowResult(True, "reached", pose, trace)
            code = loco.set_velocity(*cmd, duration=COMMAND_DURATION)
            if code != OK:
                return FollowResult(False, f"loco error {code}", pose, trace)
            tick(COMMAND_PERIOD)
            elapsed += COMMAND_PERIOD
        return FollowResult(False, "timeout", loco.pose(), trace)
    finally:
        loco.stop()
