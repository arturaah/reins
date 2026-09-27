"""Stand-ins for both models, for demos without a model and for tests.

Both do only what a vision model would, and cheat only at seeing: they read
the sim's truth where a model would read the camera image.

- `PointingDemoBrain` is the top-level model. It does the object-to-counter
  task through exactly the room tools, picking which pixels to point at by
  projecting the true positions into the last camera image.
- `ServoDemoVLM` is the arm policy's model. It answers the controller prompts of
  `harness.loop.Episode` with the same JSON and action vocabulary a real VLM
  must use (MV_FWD ... GRASP, DONE), choosing each move from the true hand and
  target positions. Everything it drives, from the parser and the safety gate to
  the executor and the backend, is the real code path.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from ..vlm.base import VLM, VLMResponse
from .brains import ToolCall, ToolResult, Turn
from .world import SimWorld

NEAR = 0.012  # m: the servo demo counts the hand as on target within this


class PointingDemoBrain:
    name = "demo (pointing from sim truth)"

    def __init__(self, world: SimWorld, obj: str = "red_cube", surface: str = "counter"):
        self.world, self.obj, self.surface = world, obj, surface

    def start(self, system: str, tools: list[dict], task: str) -> None:
        self.stage = 0
        self.found: list[dict] = []
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

    def object_top(self) -> np.ndarray:
        return self.world.object_pos(self.obj) + [0, 0, self.world.object_half_height(self.obj)]

    def beside_object(self) -> np.ndarray:
        """A point on the table next to the object, toward the robot: to read the table height."""
        p, pose = self.world.object_pos(self.obj), self.world.true_pose()
        toward = np.array([pose.x - p[0], pose.y - p[1]])
        toward /= max(float(np.linalg.norm(toward)), 1e-6)
        _, top = self.world.surface_under(p[0], p[1], p[2])
        return np.array([p[0] + 0.06 * toward[0], p[1] + 0.06 * toward[1], top])

    def surface_spot(self) -> np.ndarray:
        s = self.world.surfaces[self.surface]
        p = self.world.true_pose()
        x, y = s.nearest_point(p.x, p.y, 0.08)
        return np.array([x, y, s.top])

    def step(self, results: list[ToolResult]) -> Turn:
        for r in results:
            if r.is_error or r.text.startswith(("Operator", "Failed")):
                return Turn(f"Stopping: the demo can't adapt. Last result: {r.text}")
            if r.text.startswith("[{"):
                self.found, _ = json.JSONDecoder().raw_decode(r.text)
                if any("error" in f for f in self.found):
                    return Turn(f"Stopping: locate failed: {self.found}")
        self.stage += 1
        pt = lambda i: {k: v for k, v in zip("xyz", self.found[i]["robot"])} if self.found else {}  # noqa: E731
        label = self.obj.replace("_", " ")
        stages = {
            1: lambda: self._call("look", {}, "Looking around."),
            2: lambda: self._call("locate", {"pixels": [self._pixel(self.object_top())]}, f"The {label} is there."),
            3: lambda: self._call("approach", pt(0)),
            4: lambda: self._call("locate", {"pixels": [self._pixel(self.object_top()),
                                                        self._pixel(self.beside_object())]}, "Closer look."),
            5: lambda: self._call("manipulate", {"task": f"pick up the {label}", "surface_z": self.found[1]["robot"][2]}),
            6: lambda: self._call("turn", {"angle_rad": round(self._bearing(self.surface_spot()), 3)},
                                  f"Turning to find the {self.surface}."),
            7: lambda: self._call("locate", {"pixels": [self._pixel(self.surface_spot())]}),
            8: lambda: self._call("approach", pt(0)),
            9: lambda: self._call("locate", {"pixels": [self._pixel(self.surface_spot())]}),
            10: lambda: self._call("manipulate", {"task": f"put the held {label} down on the {self.surface}",
                                                  "surface_z": self.found[0]["robot"][2]}),
            11: lambda: self._call("arm_home", {}),
        }
        if self.stage in stages:
            return stages[self.stage]()
        return Turn(f"Done: the {label} should be on the {self.surface}.")

    def _bearing(self, xyz_world) -> float:
        p = self.world.true_pose()
        ang = math.atan2(xyz_world[1] - p.y, xyz_world[0] - p.x) - p.yaw
        return (ang + math.pi) % (2 * math.pi) - math.pi


class ServoDemoVLM(VLM):
    """Answers Artur's planner and controller prompts from the sim's truth.

    Pick tasks get the stages GRASP, LIFT; put-down tasks get MOVE, RELEASE, RETREAT,
    following the planner rules in harness.prompts.
    """
    name = "servo-demo"

    def __init__(self, world: SimWorld, hand: str, task: str, demo: PointingDemoBrain):
        self.world, self.hand, self.task, self.demo = world, hand, task, demo
        self.pick = "pick" in task.lower()
        self._stage, self._base_z, self._released = None, None, False

    def plan(self, prompt, images, schema=None):
        if self.pick:
            stages = [("grasp_object", "GRASP", "hand closed on the object"),
                      ("lift_object", "LIFT", "object lifted clear of the table")]
        else:
            stages = [("move_over_spot", "MOVE", "held object above the spot"),
                      ("release_object", "RELEASE", "object resting on the surface"),
                      ("retreat", "RETREAT", "hand lifted clear")]
        return VLMResponse(json.dumps({"subgoals": [
            {"id": i, "target": "object" if self.pick else "spot", "affordance": "main body", "motion": m,
             "description": d, "completion": d} for i, m, d in stages]}), "servo-demo")

    def act(self, prompt, images, schema=None, retry_note=None):
        stage = re.search(r"^STAGE: (\S+)", prompt, re.M).group(1)
        if stage != self._stage:
            self._stage, self._base_z = stage, self.world.hand_in_robot(self.hand)[2]
        tip = self.world.hand_in_robot(self.hand)
        if stage == "GRASP":
            if self.world.held_by(self.hand):
                return self._say("DONE", "the object is in the hand")
            target = self.world._true_to_robot(self.world.object_pos(self.demo.obj))
            return self._toward(tip, target, "GRASP")
        if stage in ("LIFT", "RETREAT"):
            return self._say("MV_UP" if tip[2] < self._base_z + 0.08 else "DONE", "lifting clear")
        if stage == "MOVE":
            # Hover so the held object's bottom clears the surface by a few centimetres.
            clear = self.world.object_half_height(self.demo.obj) + 0.04
            target = self.world._true_to_robot(self.demo.surface_spot()) + [0, 0, clear]
            return self._toward(tip, target, "DONE")
        if stage == "RELEASE":
            if not self._released:
                self._released = True
                return self._say("RELEASE", "the object is just above the spot")
            return self._say("DONE", "the object is on the surface")
        return self._say("DONE", f"unknown stage {stage}")

    def _toward(self, tip, target, arrived: str) -> VLMResponse:
        d = target - tip
        if float(np.linalg.norm(d)) < NEAR:
            return self._say(arrived, "the hand is on the target")
        axis = int(np.argmax(np.abs(d)))
        move = [("MV_FWD", "MV_BACK"), ("MV_LEFT", "MV_RIGHT"), ("MV_UP", "MV_DOWN")][axis][0 if d[axis] > 0 else 1]
        return self._say(move, "moving toward the target")

    @staticmethod
    def _say(decision: str, why: str) -> VLMResponse:
        return VLMResponse(json.dumps({"decision": decision, "reasoning": f"WRIST: NO {why}"}), "servo-demo")


def demo_arm_vlm(world: SimWorld, demo: PointingDemoBrain):
    """The Skills.arm_vlm factory for the demo."""
    return lambda task, hand: ServoDemoVLM(world, hand, task, demo)
