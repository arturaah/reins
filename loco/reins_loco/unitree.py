"""Real-robot backend for `Loco`, over unitree_sdk2_python's R1 LocoClient.

Dry run by default: calls are logged, not sent. Pass `live=True` only once
Artur has approved sending commands to the robot (see CLAUDE.md, "Rule for
agents"). Needs `pip install unitree_sdk2py` (CycloneDDS 0.10.2, Python 3.8-3.10
on macOS).

The R1 SDK exposes no odometry, so `pose()` is dead reckoning from the
commanded velocities. It drifts. Fine for short moves the operator is watching;
real localization is a separate piece of work.
"""
from __future__ import annotations

import logging
import math
import time

from .base import FSM_DAMP, FSM_STANCE, FSM_WALK, OK, Limits, Pose2, wrap

log = logging.getLogger("reins.loco")


class UnitreeLoco:
    def __init__(self, interface: str = "en6", live: bool = False,
                 limits: Limits = Limits(), timeout_s: float = 2.0, domain: int = 0):
        self.limits = limits
        self.live = live
        self._fsm = FSM_DAMP
        self._pose = Pose2(0.0, 0.0, 0.0)
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_until = 0.0
        self._last = time.monotonic()
        self._client = None
        if live:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
            ChannelFactoryInitialize(domain, interface)
            self._client = LocoClient()
            self._client.SetTimeout(timeout_s)
            self._client.Init()
        log.info("UnitreeLoco %s on %s", "LIVE" if live else "dry run", interface)

    # The SDK's Damp()/Stance()/Start() helpers drop the return code, so call SetFsmId.
    def _set_fsm(self, fsm: int) -> int:
        self._integrate()
        code = self._client.SetFsmId(fsm) if self._client else OK
        log.info("SetFsmId(%d) -> %d%s", fsm, code, "" if self.live else " [dry run]")
        if code == OK:
            self._fsm = fsm
            if fsm != FSM_WALK:
                self._cmd = (0.0, 0.0, 0.0)
        return code

    def damp(self) -> int:
        return self._set_fsm(FSM_DAMP)

    def stance(self) -> int:
        return self._set_fsm(FSM_STANCE)

    def start(self) -> int:
        return self._set_fsm(FSM_WALK)

    def set_velocity(self, vx: float, vy: float, wz: float, duration: float = 1.0) -> int:
        vx, vy, wz = self.limits.clamp(vx, vy, wz)
        self._integrate()
        code = self._client.SetVelocity(vx, vy, wz, duration) if self._client else OK
        log.debug("SetVelocity(%.2f, %.2f, %.2f, %.2f) -> %d", vx, vy, wz, duration, code)
        if code == OK:
            self._cmd = (vx, vy, wz)
            self._cmd_until = time.monotonic() + duration
        return code

    def stop(self) -> int:
        return self.set_velocity(0.0, 0.0, 0.0)

    def fsm_id(self) -> int:
        if self._client and hasattr(self._client, "GetFsmId"):
            code, fsm = self._client.GetFsmId()
            if code == OK:
                self._fsm = fsm
        return self._fsm

    def pose(self) -> Pose2:
        self._integrate()
        return self._pose

    def pose_is_estimated(self) -> bool:
        return True

    def reset_pose(self, pose: Pose2 = Pose2(0.0, 0.0, 0.0)) -> None:
        """Declare where the robot is now, e.g. after the operator places it."""
        self._integrate()
        self._pose = pose

    def _integrate(self) -> None:
        now = time.monotonic()
        # Integrate up to the moment the last command expired, not past it.
        end = min(now, self._cmd_until)
        dt = max(0.0, end - self._last)
        vx, vy, wz = self._cmd
        p = self._pose
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        self._pose = Pose2(p.x + (c * vx - s * vy) * dt, p.y + (s * vx + c * vy) * dt, wrap(p.yaw + wz * dt))
        self._last = now
