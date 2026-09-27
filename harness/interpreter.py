"""Semantic action -> proposed hand setpoint. Pure numpy, no SDK.

State per arm: hand tip position p (m) and wrist roll (rad) in the robot base frame, both
re-synced from forward kinematics after every step (harness.executor does the sync).

Translation: p += sigma * R_view @ d, where d is the unit vector of the semantic direction in
the context-camera view frame (forward, left, up) and R_view maps view directions into the base
frame (config frames.view_forward / view_left; up is gravity).
Rotation: this R1 A5 arm has one orientation degree of freedom, wrist roll about the forearm.
ROTATE_CW/CCW turn the roll by theta; requests about base x/y/z are reported as unavailable.
GRASP/RELEASE do not touch the pose; the executor runs the hand preset.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .actions import Action


@dataclass
class ArmState:
    p: np.ndarray                      # hand tip, base frame, m
    roll: float                        # wrist roll joint, rad
    hand_closed: bool = False
    q: dict = field(default_factory=dict)   # joint name -> rad, the pose this state was read from


@dataclass
class Proposal:
    kind: str                          # move | rotate | hand | still | done | unavailable | walk
    p: np.ndarray                      # requested hand tip (before the safety gate)
    roll: float
    mode: str = "unit"                 # unit | param  (which per-step cap applies)
    hand_closed: Optional[bool] = None # for kind == hand
    note: str = ""                     # feedback text for the model when nothing moves
    action: Optional[Action] = None
    walk: Optional[tuple] = None       # kind == walk: (dx, dy, dyaw) in the body frame (m, m, rad)


class Interpreter:
    def __init__(self, view_forward, view_left, up=(0.0, 0.0, 1.0)):
        f = np.asarray(view_forward, float); l = np.asarray(view_left, float); u = np.asarray(up, float)
        for v, n in ((f, "view_forward"), (l, "view_left"), (u, "up")):
            if abs(np.linalg.norm(v) - 1.0) > 1e-6:
                raise ValueError(f"{n} must be a unit vector")
        self.R_view = np.column_stack([f, l, u])   # columns: forward, left, up in the base frame

    def direction(self, axis, sign):
        idx = {"forward": 0, "left": 1, "up": 2}[axis]
        return sign * self.R_view[:, idx]

    def propose(self, state, action, sigma_m, theta_rad, walk_m=0.2, turn_rad=0.35):
        """state: ArmState. sigma_m/theta_rad: the current unit step sizes; walk_m/turn_rad: one WALK_* / TURN_* step.
        Walks are in the body frame (forward = +x, left = +y), not the camera view: the camera rides on the body."""
        p, roll = np.array(state.p, float), float(state.roll)
        if action.name == "WALK":
            dist = walk_m if action.amount is None else action.amount
            d = (dist * action.sign, 0.0) if action.axis == "forward" else (0.0, dist * action.sign)
            return Proposal("walk", p, roll, action.mode, action=action, walk=(float(d[0]), float(d[1]), 0.0))
        if action.name == "TURN":
            ang = turn_rad if action.amount is None else action.amount
            return Proposal("walk", p, roll, action.mode, action=action, walk=(0.0, 0.0, float(action.sign * ang)))
        if action.name == "MOVE":
            dist = sigma_m if action.amount is None else action.amount
            return Proposal("move", p + dist * self.direction(action.axis, action.sign), roll, action.mode, action=action)
        if action.name == "ROTATE":
            if action.axis != "roll":
                return Proposal("unavailable", p, roll, action.mode, action=action,
                                note=f"rotation about base {action.axis} is not available on this 5-joint arm; "
                                     f"only ROTATE_CW / ROTATE_CCW (wrist roll) can be executed")
            ang = theta_rad if action.amount is None else action.amount
            return Proposal("rotate", p, roll + action.sign * ang, action.mode, action=action)
        if action.name == "POINT":
            return Proposal("unavailable", p, roll, action.mode, action=action,
                            note="POINT presets are not reachable on this 5-joint arm; the hand orientation follows from its position")
        if action.name == "GRASP":
            return Proposal("hand", p, roll, action.mode, hand_closed=True, action=action)
        if action.name == "RELEASE":
            return Proposal("hand", p, roll, action.mode, hand_closed=False, action=action)
        if action.name == "STILL":
            return Proposal("still", p, roll, action.mode, action=action)
        if action.name == "DONE":
            return Proposal("done", p, roll, action.mode, action=action)
        raise ValueError(f"unhandled action {action}")


def step_size(cfg_steps, wrist_visible):
    """(sigma_m, theta_rad) for the configured profile and the wrist-visibility flag."""
    import math
    if cfg_steps["profile"] == "precision":
        return float(cfg_steps["precision"]["step_m"]), math.radians(float(cfg_steps["precision"]["rotate_deg"]))
    sigma = float(cfg_steps["fine_m"]) if wrist_visible else float(cfg_steps["coarse_m"])
    return sigma, math.radians(float(cfg_steps["rotate_deg"]))
