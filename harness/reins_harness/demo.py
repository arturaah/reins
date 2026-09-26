"""A stand-in for the vision model, for demos without a model and for tests.

`PointingDemoBrain` does the red-cube-to-counter task through exactly the
tools a model gets. It cheats at one thing only: choosing which pixel to point
at, which it does by projecting the sim's true object position into the last
camera image. That is the part a vision model does by looking. Everything after
the pixel (depth, locate, approach, IK, grasp) runs as it would for a model,
so this exercises the whole perception chain end to end.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .brains import ToolCall, ToolResult, Turn
from .world import SimWorld


class PointingDemoBrain:
    name = "demo (pointing from sim truth)"

    def __init__(self, world: SimWorld, obj: str = "red_cube", surface: str = "counter"):
        self.world, self.obj, self.surface = world, obj, surface

    def start(self, system: str, tools: list[dict], task: str) -> None:
        self.stage = 0
        self.point = None
        self._n = 0

    def _call(self, name: str, args: dict, text: str = "") -> Turn:
        self._n += 1
        return Turn(text, [ToolCall(f"demo-{self._n}", name, args)])

    def _pixel(self, xyz_world) -> list[float]:
        """Where a true world point appears in the last image: the vision model's job."""
        cap = self.world.last_capture
        p_robot = self.world._true_to_robot(xyz_world)
        p_cam = (np.linalg.inv(cap.T_robot_cam) @ np.r_[p_robot, 1.0])[:3]
        u, v = cap.rgb_model.project(p_cam)
        return [round(float(u), 1), round(float(v), 1)]

    def _object_top(self) -> np.ndarray:
        return self.world.object_pos(self.obj) + [0, 0, self.world.object_half_height(self.obj)]

    def _surface_spot(self) -> np.ndarray:
        s = self.world.surfaces[self.surface]
        p = self.world.true_pose()
        x, y = s.nearest_point(p.x, p.y, 0.08)
        return np.array([x, y, s.top])

    def step(self, results: list[ToolResult]) -> Turn:
        for r in results:
            if r.is_error or r.text.startswith("Operator"):
                return Turn(f"Stopping: the demo can't adapt. Last result: {r.text}")
            if r.text.startswith("[{"):
                found, _ = json.JSONDecoder().raw_decode(r.text)
                if "error" in found[0]:
                    return Turn(f"Stopping: locate failed: {found[0]['error']}")
                self.point = found[0]["robot"]
        self.stage += 1
        x, y, z = self.point or (0, 0, 0)
        target = {"x": x, "y": y, "z": z}
        label = self.obj.replace("_", " ")
        stages = {
            1: lambda: self._call("look", {}, "Looking around."),
            2: lambda: self._call("locate", {"pixels": [self._pixel(self._object_top())]}, f"The {label} is there."),
            3: lambda: self._call("approach", target),
            4: lambda: self._call("locate", {"pixels": [self._pixel(self._object_top())]}, "Closer look."),
            5: lambda: self._call("pick_up", {**target, "label": label}),
            6: lambda: self._call("turn", {"angle_rad": round(self._bearing(self._surface_spot()), 3)},
                                  f"Turning to find the {self.surface}."),
            7: lambda: self._call("locate", {"pixels": [self._pixel(self._surface_spot())]}),
            8: lambda: self._call("approach", target),
            9: lambda: self._call("locate", {"pixels": [self._pixel(self._surface_spot())]}),
            10: lambda: self._call("place", target),
            11: lambda: self._call("arm_home", {}),
        }
        if self.stage in stages:
            return stages[self.stage]()
        return Turn(f"Done: the {label} should be on the {self.surface}.")

    def _bearing(self, xyz_world) -> float:
        p = self.world.true_pose()
        ang = math.atan2(xyz_world[1] - p.y, xyz_world[0] - p.x) - p.yaw
        return (ang + math.pi) % (2 * math.pi) - math.pi
