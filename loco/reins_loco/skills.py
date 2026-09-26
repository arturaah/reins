"""High-level walking skills: what the VLM asks for.

Two halves, matching Reins' review-before-action loop:

- `plan_*` turns a skill call into a contract `walk` step (goal plus path, in
  the map frame) for preview and approval. Nothing moves.
- `execute_walk_step` runs an approved step on any `Loco` backend.

`TOOLS` are the tool definitions to hand the VLM harness. `plan_tool_call`
dispatches one tool call to the matching `plan_*` function.
"""
from __future__ import annotations

import math
import time
from typing import Callable

from .base import FSM_STANCE, FSM_WALK, OK, Loco, Pose2, wrap
from .follower import FollowResult, PathFollower, drive

TOOLS = [
    {
        "name": "walk_to",
        "description": "Walk to a position on the floor, optionally through via points, and "
                       "optionally end facing a heading. Map frame unless frame is 'robot' "
                       "(x forward, y left of where the robot stands now).",
        "input_schema": {
            "type": "object",
            "required": ["x", "y"],
            "properties": {
                "x": {"type": "number", "description": "metres"},
                "y": {"type": "number", "description": "metres"},
                "yaw": {"type": "number", "description": "final heading, radians; omit to keep the walking direction"},
                "via": {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                        "minItems": 2, "maxItems": 2}, "description": "[x, y] points to pass through, same frame"},
                "frame": {"enum": ["map", "robot"], "default": "map"},
                "description": {"type": "string", "description": "one sentence for the human reviewer"},
            },
        },
    },
    {
        "name": "walk",
        "description": "Walk a distance relative to the robot's current heading.",
        "input_schema": {
            "type": "object",
            "required": ["forward_m"],
            "properties": {
                "forward_m": {"type": "number", "description": "metres forward (negative: backward)"},
                "left_m": {"type": "number", "default": 0, "description": "metres to the left"},
                "description": {"type": "string"},
            },
        },
    },
    {
        "name": "turn",
        "description": "Turn in place. Positive is counter-clockwise (left).",
        "input_schema": {
            "type": "object",
            "required": ["angle_rad"],
            "properties": {"angle_rad": {"type": "number"}, "description": {"type": "string"}},
        },
    },
    {
        "name": "stop",
        "description": "Stop walking immediately.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _to_map(pose: Pose2, x: float, y: float) -> tuple[float, float]:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return pose.x + c * x - s * y, pose.y + s * x + c * y


def plan_walk_to(pose: Pose2, x: float, y: float, yaw: float | None = None,
                 via: list[list[float]] | None = None, frame: str = "map",
                 description: str | None = None, step_id: str = "s1") -> dict:
    points = [*(via or []), [x, y]]
    if frame == "robot":
        points = [list(_to_map(pose, px, py)) for px, py in points]
        yaw = None if yaw is None else wrap(pose.yaw + yaw)
    elif frame != "map":
        raise ValueError(f"unknown frame {frame!r}")
    path = [[round(pose.x, 4), round(pose.y, 4)]] + [[round(px, 4), round(py, 4)] for px, py in points]
    gx, gy = path[-1]
    goal = {"frame": "map", "x": gx, "y": gy}
    if yaw is not None:
        goal["yaw"] = round(yaw, 4)
    return {"step_id": step_id, "kind": "walk",
            "description": description or f"Walk to ({gx:.2f}, {gy:.2f})",
            "goal": goal, "corridor_half_width_m": 0.4,
            "path": {"frame": "map", "points": path}}


def plan_walk(pose: Pose2, forward_m: float, left_m: float = 0.0,
              description: str | None = None, step_id: str = "s1") -> dict:
    return plan_walk_to(pose, forward_m, left_m, yaw=0.0, frame="robot", step_id=step_id,
                        description=description or f"Walk {forward_m:.2f} m forward, {left_m:.2f} m left")


def plan_turn(pose: Pose2, angle_rad: float, description: str | None = None,
              step_id: str = "s1") -> dict:
    # A turn is a walk step whose path doesn't go anywhere.
    step = plan_walk_to(pose, pose.x, pose.y, yaw=wrap(pose.yaw + angle_rad), step_id=step_id,
                        description=description or f"Turn {math.degrees(angle_rad):.0f} degrees")
    return step


def plan_tool_call(pose: Pose2, name: str, arguments: dict, step_id: str = "s1") -> dict | None:
    """Contract walk step for one VLM tool call; None for `stop`, which needs no approval."""
    if name == "walk_to":
        return plan_walk_to(pose, step_id=step_id, **arguments)
    if name == "walk":
        return plan_walk(pose, step_id=step_id, **arguments)
    if name == "turn":
        return plan_turn(pose, step_id=step_id, **arguments)
    if name == "stop":
        return None
    raise ValueError(f"unknown tool {name!r}")


def ensure_walking(loco: Loco) -> int:
    """Stance, then start walking mode, if the robot isn't walking already."""
    if loco.fsm_id() == FSM_WALK:
        return OK
    if loco.fsm_id() != FSM_STANCE:
        code = loco.stance()
        if code != OK:
            return code
    return loco.start()


def execute_walk_step(step: dict, loco: Loco, tick: Callable[[float], None] = time.sleep,
                      timeout: float = 120.0,
                      should_stop: Callable[[], bool] = lambda: False) -> FollowResult:
    """Walk an approved contract `walk` step."""
    if step.get("kind") != "walk":
        raise ValueError("not a walk step")
    if step["goal"]["frame"] != "map" or step.get("path", {}).get("frame", "map") != "map":
        raise ValueError("walk steps must be in the map frame by execution time")
    code = ensure_walking(loco)
    if code != OK:
        return FollowResult(False, f"loco error {code} entering walk mode", loco.pose())
    points = step.get("path", {}).get("points") or [[loco.pose().x, loco.pose().y]]
    start = loco.pose()
    points = [[start.x, start.y], *points[1:]] if len(points) > 1 else [[start.x, start.y], points[0]]
    goal = step["goal"]
    points[-1] = [goal["x"], goal["y"]]
    follower = PathFollower(points, goal.get("yaw"), loco.limits)
    return drive(loco, follower, tick=tick, timeout=timeout, should_stop=should_stop)
