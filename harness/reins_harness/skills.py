"""What the LLM can ask the robot to do, and how each request becomes a plan.

The model gets only what the real R1 can sense (see world.py):

- **Sensing** answers straight away and moves nothing: `robot_state` (joint
  angles, IMU, odometry, hands), `look` (the head camera image), and `locate`
  (the model points at pixels in the last image; depth turns them into 3D
  points).
- **Acting** (`walk_to`, `walk`, `turn`, `approach`, `reach`, `pick_up`,
  `place`, `arm_home`) takes coordinates the model worked out from what it
  saw, and returns a `Proposal`: contract steps (`walk`, `arm`, `grip`) plus
  what to draw. Only an approved proposal reaches `execute`.

No tool knows what objects exist or where they are. Planning checks what the
robot itself could check: IK must reach every hand waypoint, and walking paths
must clear the obstacles its depth camera has seen.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from reins_loco.base import FSM_WALK, Pose2, wrap
from reins_loco.skills import execute_walk_step, plan_turn, plan_walk, plan_walk_to

from . import planner
from .camera import H_FOV, V_FOV, annotate
from .ik import IK_JOINTS, ArmIK, Unreachable
from .world import HANDS, HOME, SimWorld, arm_joint_names, png_bytes

# How far in front of the pelvis to stand from a target, best first. The hands reach
# about 0.38 m at table height, so nearer leaves room for walking error.
STAND_OFF = (0.33, 0.36, 0.38)
REACH_SIDE = 0.15  # m to the hand's side
PREGRASP_HEIGHT = 0.10  # m above the object before descending
LIFT_HEIGHT = 0.12  # m above where the object was picked up
# The camera sees an object's top and near side, not its middle: reach a bit past the
# seen point, and down to halfway between it and the surface it stands on.
GRASP_DEPTH = 0.02  # m further along the camera's horizontal line of sight
GRASP_DROP = 0.015  # m lower, when the surface below can't be seen
RELEASE_HEIGHT = 0.10  # m above the surface point where the hand lets go
ARM_TICK = 0.02  # s between arm updates when executing
WALK_TOLERANCE = 0.03  # m; the hands need the robot to stop close to where it was planned
ERR_BLOCKED = 7401


class PlanError(ValueError):
    """The request can't be planned; the message tells the model why."""


@dataclass
class Overlay:
    """What a reviewer sees drawn in the scene, in the odom frame."""
    walk: list[dict] = field(default_factory=list)  # contract walk steps
    hands: dict[str, list[np.ndarray]] = field(default_factory=dict)  # hand -> points
    marks: list[np.ndarray] = field(default_factory=list)  # targets


@dataclass
class Proposal:
    summary: str
    steps: list[dict]
    overlay: Overlay


@dataclass
class Observation:
    text: str
    image_png: bytes | None = None


def _num(description: str) -> dict:
    return {"type": "number", "description": description}


FRAME = {"enum": ["robot", "odom"], "default": "robot",
         "description": "robot: x forward, y left, z up from the floor under the robot, as it stands now. "
                        "odom: fixed where the robot started (dead-reckoned, drifts)"}
HAND_AUTO = {"enum": ["left_hand", "right_hand", "auto"], "default": "auto"}
XYZ = {"x": _num("metres"), "y": _num("metres"), "z": _num("metres above the floor")}
DESCRIPTION = {"type": "string", "description": "one sentence for the human reviewer"}

TOOLS = [
    {"name": "robot_state",
     "description": "The robot's own sensing: joint angles, IMU, odometry (where it thinks it is "
                    "relative to where it started), hand positions from its joint angles, and "
                    "whether each hand is closed. Nothing moves.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "look",
     "description": "Take a picture with the head camera: a wide fisheye (150° across, 124° "
                    "tall) looking forward and a little down, 960×816 pixels with pixel "
                    "coordinates marked along the edges. Nothing moves. The robot has no neck: "
                    "turn to look elsewhere.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "locate",
     "description": "Get 3D positions for pixels in the most recent camera image, from the "
                    "camera's depth. Point at the middle of an object's visible top, or at the "
                    "spot on a surface where something should go. Returns positions in the "
                    "robot and odom frames, or says there's no depth there.",
     "input_schema": {"type": "object", "required": ["pixels"], "properties": {
         "pixels": {"type": "array", "minItems": 1, "maxItems": 20,
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "description": "[u, v] pairs: u from the left edge, v from the top, in image pixels"}}}},
    {"name": "walk_to",
     "description": "Walk to a point on the floor, planning around obstacles the camera has seen. "
                    "Obstacles never seen aren't avoided: look first.",
     "input_schema": {"type": "object", "required": ["x", "y"], "properties": {
         "x": _num("metres"), "y": _num("metres"),
         "yaw": _num("final heading in radians, same frame; omit to keep the walking direction"),
         "frame": FRAME, "description": DESCRIPTION}}},
    {"name": "walk",
     "description": "Walk a short distance in a straight line, relative to where the robot faces.",
     "input_schema": {"type": "object", "required": ["forward_m"], "properties": {
         "forward_m": _num("metres forward, negative for backward"),
         "left_m": _num("metres to the left"), "description": DESCRIPTION}}},
    {"name": "turn",
     "description": "Turn in place. Positive is counter-clockwise (left). Use it to look around.",
     "input_schema": {"type": "object", "required": ["angle_rad"], "properties": {
         "angle_rad": _num("radians"), "description": DESCRIPTION}}},
    {"name": "approach",
     "description": "Walk to where one hand can work at a point (an object to pick up, or a spot "
                    "to put something), facing it. Use a point from locate.",
     "input_schema": {"type": "object", "required": ["x", "y", "z"], "properties": {
         **XYZ, "frame": FRAME, "hand": HAND_AUTO, "description": DESCRIPTION}}},
    {"name": "reach",
     "description": "Move one hand to a point and hold it there (pointing, gestures).",
     "input_schema": {"type": "object", "required": ["hand", "x", "y", "z"], "properties": {
         "hand": {"enum": ["left_hand", "right_hand"]}, **XYZ, "frame": FRAME, "description": DESCRIPTION}}},
    {"name": "pick_up",
     "description": "Pick up the object at a point from locate (the middle of its visible top): "
                    "hand above it, down onto it, close, lift. It must be within reach, so "
                    "approach it first. Look afterwards to check it's in the hand.",
     "input_schema": {"type": "object", "required": ["x", "y", "z"], "properties": {
         **XYZ, "frame": FRAME, "hand": HAND_AUTO,
         "label": {"type": "string", "description": "what you're picking up, for the reviewer"},
         "description": DESCRIPTION}}},
    {"name": "place",
     "description": "Put down what a hand holds at a surface point from locate: hand above it, "
                    "down to just above the surface, open, lift clear.",
     "input_schema": {"type": "object", "required": ["x", "y", "z"], "properties": {
         **XYZ, "frame": FRAME, "hand": HAND_AUTO, "description": DESCRIPTION}}},
    {"name": "arm_home",
     "description": "Return one or both arms to the relaxed pose, keeping hold of anything held.",
     "input_schema": {"type": "object", "properties": {
         "hand": {"enum": ["left_hand", "right_hand", "both"], "default": "both"},
         "description": DESCRIPTION}}},
]

SENSING = {"robot_state", "look", "locate"}


class Skills:
    def __init__(self, world: SimWorld):
        self.world = world
        self.ik = ArmIK(world)

    # --- dispatch --------------------------------------------------------------

    def call(self, name: str, args: dict) -> Observation | Proposal:
        """Run a sensing tool, or plan an acting one. Raises PlanError if it can't be planned."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise PlanError(f"unknown tool {name!r}")
        args = {k: v for k, v in args.items() if v is not None}
        try:
            return handler(**args)
        except (Unreachable, planner.NoPath) as e:
            raise PlanError(str(e)) from e
        except TypeError as e:
            raise PlanError(f"bad arguments for {name}: {e}") from e

    # --- sensing ---------------------------------------------------------------

    def _tool_robot_state(self) -> Observation:
        return Observation(json.dumps(self.world.proprioception(), indent=1))

    def _tool_look(self) -> Observation:
        return self.snapshot("Head camera.")

    def snapshot(self, text: str) -> Observation:
        """A fresh camera frame (which also updates the obstacle map), as a tool result."""
        cap = self.world.capture()
        w, h = cap.rgb_model.width, cap.rgb_model.height
        return Observation(f"{text} Fisheye, {math.degrees(H_FOV):.0f}°×{math.degrees(V_FOV):.0f}°, "
                           f"{w}×{h} px, (0, 0) top-left; image right is the robot's right. "
                           f"Straight lines look curved near the edges.", png_bytes(annotate(cap.rgb)))

    def _tool_locate(self, pixels: list) -> Observation:
        cap, cap_pose = self.world.last_capture, self.world.last_capture_odom
        if cap is None:
            raise PlanError("no camera image yet; call look first")
        out = []
        for uv in pixels:
            u, v = (float(c) for c in uv)
            if not (0 <= u < cap.rgb_model.width and 0 <= v < cap.rgb_model.height):
                out.append({"pixel": [u, v], "error": "outside the image"})
                continue
            p = cap.point(u, v)
            if p is None:
                out.append({"pixel": [u, v], "error": "no depth there (sky, too far, or too close)"})
                continue
            odom = self.world.to_odom(p, cap_pose)
            robot = self.world.to_robot(odom)
            out.append({"pixel": [u, v], "robot": _r(robot), "odom": _r(odom),
                        "distance_m": round(float(np.hypot(robot[0], robot[1])), 3)})
        moved = cap_pose != self.world.odom_pose()
        note = " The robot has moved since that image: robot-frame values are converted to where it stands now." \
            if moved else ""
        return Observation(json.dumps(out) + note)

    # --- walking -------------------------------------------------------------

    def _to_odom(self, x: float, y: float, z: float, frame: str) -> np.ndarray:
        if frame == "robot":
            return self.world.to_odom([x, y, z])
        if frame == "odom":
            return np.array([x, y, z], float)
        raise PlanError(f"frame must be robot or odom, got {frame!r}")

    def _clearance(self, x: float, y: float) -> float:
        return self.world.obstacles.clearance(x, y)

    def _walk_steps(self, goal: tuple[float, float], yaw: float | None, description: str) -> list[dict]:
        """A planned walk, preceded by a turn in place if the path starts well off the heading.

        The path follower only turns in place until the path is within 50 degrees
        of ahead, then walks while turning, which swings the robot sideways into
        whatever it is standing next to.
        """
        pose = self.world.odom_pose()
        path = planner.plan_path(self._clearance, (pose.x, pose.y), goal,
                                 known=[tuple(c) for c in self.world.obstacles.cells()[::20]])
        steps = []
        (x0, y0), (x1, y1) = path[0], path[1]
        if math.hypot(x1 - x0, y1 - y0) > 0.05:
            turn = wrap(math.atan2(y1 - y0, x1 - x0) - pose.yaw)
            if abs(turn) > math.radians(30):
                steps.append(plan_turn(pose, turn, f"Turn {math.degrees(turn):.0f} degrees to face the path",
                                       step_id="s1"))
                pose = Pose2(pose.x, pose.y, wrap(pose.yaw + turn))
        steps.append(plan_walk_to(pose, goal[0], goal[1], yaw=yaw, via=[list(p) for p in path[1:-1]],
                                  description=description, step_id=f"s{len(steps) + 1}"))
        return steps

    def _tool_walk_to(self, x: float, y: float, yaw: float | None = None, frame: str = "robot",
                      description: str | None = None) -> Proposal:
        gx, gy, _ = self._to_odom(x, y, 0.0, frame)
        if yaw is not None and frame == "robot":
            yaw = wrap(self.world.odom_pose().yaw + yaw)
        steps = self._walk_steps((float(gx), float(gy)), yaw, description or f"Walk to ({x:.2f}, {y:.2f}) {frame}")
        return Proposal(steps[-1]["description"], steps, Overlay(walk=steps))

    def _tool_walk(self, forward_m: float, left_m: float = 0.0, description: str | None = None) -> Proposal:
        step = plan_walk(self.world.odom_pose(), forward_m, left_m, description)
        end = step["goal"]
        if self._clearance(end["x"], end["y"]) < 0:
            raise PlanError("that would end inside an obstacle the camera has seen")
        return Proposal(step["description"], [step], Overlay(walk=[step]))

    def _tool_turn(self, angle_rad: float, description: str | None = None) -> Proposal:
        step = plan_turn(self.world.odom_pose(), angle_rad, description)
        return Proposal(step["description"], [step], Overlay(walk=[step]))

    def _tool_approach(self, x: float, y: float, z: float, frame: str = "robot", hand: str = "auto",
                       description: str | None = None) -> Proposal:
        pose = self.world.odom_pose()
        tx, ty, tz = self._to_odom(x, y, z, frame)
        hands = list(HANDS) if hand == "auto" else [hand]
        candidates = []
        for h in hands:
            side = REACH_SIDE if h == "left_hand" else -REACH_SIDE
            for rank, forward in enumerate(STAND_OFF):
                for k in range(72):
                    yaw = 2 * math.pi * k / 72
                    c, s = math.cos(yaw), math.sin(yaw)
                    px = float(tx - (c * forward - s * side))
                    py = float(ty - (s * forward + c * side))
                    if self._clearance(px, py) < 0.0:
                        continue
                    turn = abs(wrap(yaw - math.atan2(ty - pose.y, tx - pose.x)))
                    cost = 10 * rank + math.hypot(px - pose.x, py - pose.y) + 0.2 * turn
                    candidates.append((cost, h, px, py, yaw))
        candidates.sort()
        for _, h, px, py, yaw in candidates[:20]:
            try:
                steps = self._walk_steps((px, py), yaw, description or
                                         f"Walk up to the target so the {h.replace('_', ' ')} can reach it")
            except planner.NoPath:
                continue
            return Proposal(steps[-1]["description"], steps,
                            Overlay(walk=steps, marks=[np.array([tx, ty, tz])]))
        raise PlanError("no clear spot to stand within reach of that point: the hands reach only about "
                        "0.18 m past the robot's front, so on a table or counter choose a spot close to "
                        "the edge nearest the robot")

    # --- arms ------------------------------------------------------------------

    def _arm_step(self, hand: str, times, rows, points, description: str, step_id: str, pose: Pose2) -> dict:
        names = arm_joint_names(hand)[:IK_JOINTS]
        return {"step_id": step_id, "kind": "arm", "description": description,
                "trajectory": {"joint_names": names,
                               "times_s": [round(t, 4) for t in times],
                               "positions_rad": [[round(r[n], 5) for n in names] for r in rows]},
                "preview": {"effector_paths": {hand: {
                    "frame": "robot",
                    "points": [[round(float(v), 4) for v in self.world.to_robot(p, pose)] for p in points],
                    "times_s": [round(t, 4) for t in times]}}}}

    def _pick_hand(self, hand: str, target_odom, want_closed: bool) -> list[str]:
        if hand != "auto":
            return [hand]
        y = self.world.to_robot(target_odom)[1]
        order = ["left_hand", "right_hand"] if y >= 0 else ["right_hand", "left_hand"]
        return [h for h in order if (self.world.grip_command[h] == "closed") == want_closed] or order

    def _tool_reach(self, hand: str, x: float, y: float, z: float, frame: str = "robot",
                    description: str | None = None) -> Proposal:
        target = self._to_odom(x, y, z, frame)
        times, rows, points = self.ik.cartesian_path(hand, [target])
        desc = description or f"Move the {hand.replace('_', ' ')} to ({x:.2f}, {y:.2f}, {z:.2f}) {frame}"
        step = self._arm_step(hand, times, rows, points, desc, "s1", self.world.odom_pose())
        return Proposal(desc, [step], Overlay(hands={hand: points}, marks=[target]))

    def _tool_pick_up(self, x: float, y: float, z: float, frame: str = "robot", hand: str = "auto",
                      label: str | None = None, description: str | None = None) -> Proposal:
        seen = self._to_odom(x, y, z, frame)
        pose = self.world.odom_pose()
        # Push the target past the visible surface, along the line of sight from the robot.
        sight = seen[:2] - np.array([pose.x, pose.y])
        sight = sight / max(float(np.linalg.norm(sight)), 1e-6)
        target = seen + np.r_[GRASP_DEPTH * sight, 0.0]
        below = self._surface_below(seen)
        target[2] = max((seen[2] + below) / 2, below + 0.02) if below is not None else seen[2] - GRASP_DROP
        name = label or "object"
        errors = []
        for h in self._pick_hand(hand, target, want_closed=False):
            if self.world.grip_command[h] == "closed":
                errors.append(f"{h} is closed (holding something?)")
                continue
            try:
                t1, r1, p1 = self.ik.cartesian_path(h, [target + [0, 0, PREGRASP_HEIGHT], target])
                lift = target + [0, 0, LIFT_HEIGHT]
                back = self.world.to_odom(self.world.to_robot(lift) - [0.08, 0, 0])
                t2, r2, p2 = self.ik.cartesian_path(h, [lift, back], start_q=r1[-1])
            except Unreachable as e:
                errors.append(str(e))
                continue
            hn = h.replace("_", " ")
            steps = [self._arm_step(h, t1, r1, p1, f"Reach the {hn} above the {name} and down onto it", "s1", pose),
                     {"step_id": "s2", "kind": "grip", "description": f"Close the {hn} on the {name}",
                      "effector": h, "action": "close", **({"object": label} if label else {})},
                     self._arm_step(h, t2, r2, p2, f"Lift the {name}", "s3", pose)]
            return Proposal(description or f"Pick up the {name} with the {hn}", steps,
                            Overlay(hands={h: p1 + p2}, marks=[seen]))
        dist = float(np.hypot(*self.world.to_robot(target)[:2]))
        raise PlanError(f"can't pick up at that point from here ({dist:.2f} m away; approach it first). "
                        + "; ".join(errors))

    def _surface_below(self, seen_odom: np.ndarray) -> float | None:
        """Height of what an object stands on, from the depth around it in the last image."""
        cap, cap_pose = self.world.last_capture, self.world.last_capture_odom
        if cap is None:
            return None
        cloud = cap.cloud(stride=1)
        cloud = np.array([self.world.to_odom(q, cap_pose) for q in cloud[::3]])
        ring = np.hypot(cloud[:, 0] - seen_odom[0], cloud[:, 1] - seen_odom[1])
        near = cloud[(ring > 0.06) & (ring < 0.16) & (cloud[:, 2] < seen_odom[2] - 0.015)]
        if len(near) < 10:
            return None
        return float(np.median(near[:, 2]))

    def _tool_place(self, x: float, y: float, z: float, frame: str = "robot", hand: str = "auto",
                    description: str | None = None) -> Proposal:
        spot = self._to_odom(x, y, z, frame)
        hands = [h for h in self._pick_hand(hand, spot, want_closed=True) if self.world.grip_command[h] == "closed"]
        if not hands:
            raise PlanError("no hand is closed, so there's nothing to put down")
        h = hands[0]
        pose = self.world.odom_pose()
        release = spot + [0, 0, RELEASE_HEIGHT]
        try:
            t1, r1, p1 = self.ik.cartesian_path(h, [release + [0, 0, PREGRASP_HEIGHT], release])
            t2, r2, p2 = self.ik.cartesian_path(h, [release + [0, 0, PREGRASP_HEIGHT]], start_q=r1[-1])
        except Unreachable as e:
            raise PlanError(f"can't put it down at that point from here; approach it first. {e}") from e
        hn = h.replace("_", " ")
        steps = [self._arm_step(h, t1, r1, p1, f"Lower the {hn} to just above the spot", "s1", pose),
                 {"step_id": "s2", "kind": "grip", "description": f"Open the {hn}", "effector": h, "action": "open"},
                 self._arm_step(h, t2, r2, p2, f"Lift the {hn} clear", "s3", pose)]
        return Proposal(description or f"Put down what the {hn} holds", steps,
                        Overlay(hands={h: p1 + p2}, marks=[spot]))

    def _tool_arm_home(self, hand: str = "both", description: str | None = None) -> Proposal:
        hands = list(HANDS) if hand == "both" else [hand]
        pose = self.world.odom_pose()
        steps, paths = [], {}
        for h in hands:
            goal = {n: HOME.get(n, 0.0) for n in arm_joint_names(h)[:IK_JOINTS]}
            if max(abs(self.world.arm_q[n] - q) for n, q in goal.items()) < 0.01:
                continue  # already relaxed
            times, rows, points = self.ik.joint_path(h, goal)
            steps.append(self._arm_step(h, times, rows, points,
                                        f"Relax the {h.replace('_', ' ')}", f"s{len(steps) + 1}", pose))
            paths[h] = points
        if not steps:
            raise PlanError("the arms are already relaxed")
        return Proposal(description or "Relax the arms", steps, Overlay(hands=paths))

    # --- execution -------------------------------------------------------------

    def execute(self, steps: list[dict], tick: Callable[[float], None],
                should_stop: Callable[[], bool] = lambda: False) -> tuple[bool, str]:
        """Run approved contract steps in order. Returns (succeeded, what the robot knows happened)."""
        log = []
        for step in steps:
            kind = step["kind"]
            if should_stop():
                return False, "; ".join(log + ["stopped by the operator"])
            if kind == "walk":
                result = execute_walk_step(step, self.world.walker, tick=tick, should_stop=should_stop,
                                           pos_tolerance=WALK_TOLERANCE)
                self._settle(tick)
                p = result.pose
                if not result.reached:
                    why = {"stopped": "stopped by the operator",
                           f"loco error {ERR_BLOCKED}": "bumped into something and stopped"}.get(
                        result.reason, result.reason)
                    return False, "; ".join(log + [f"walk ended early ({why}) at odom "
                                                   f"({p.x:.2f}, {p.y:.2f}, yaw {p.yaw:.2f})"])
                log.append(f"walked to odom ({p.x:.2f}, {p.y:.2f}, yaw {p.yaw:.2f})")
            elif kind == "arm":
                ok, msg = self._play_arm(step, tick, should_stop)
                log.append(msg)
                if not ok:
                    return False, "; ".join(log)
            elif kind == "grip":
                log.append(self.world.grip(step["effector"], step["action"]))
            else:
                return False, f"can't execute a {kind!r} step"
        return True, "; ".join(log)

    def _settle(self, tick) -> None:
        """Wait for the base to stop. The gait lowers the pelvis, so arms plan from a standstill."""
        for _ in range(60):
            if max(abs(v) for v in self.world.loco.velocity()) < 0.01:
                break
            tick(0.05)
        tick(0.05)  # one more update so the model shows the standing pose

    def _play_arm(self, step: dict, tick, should_stop) -> tuple[bool, str]:
        tr = step["trajectory"]
        names, times, rows = tr["joint_names"], tr["times_s"], np.array(tr["positions_rad"])
        drift = max(abs(self.world.arm_q[n] - rows[0][i]) for i, n in enumerate(names))
        if drift > 0.05:
            return False, f"arm moved since planning ({drift:.2f} rad); plan is stale"
        if self.world.loco.fsm_id() == FSM_WALK:
            self.world.loco.stop()
        t = 0.0
        while True:
            t = min(t + ARM_TICK, times[-1])
            self.world.arm_q.update({n: float(np.interp(t, times, rows[:, i])) for i, n in enumerate(names)})
            tick(ARM_TICK)
            if should_stop():
                return False, f"arm stopped by the operator at t={t:.2f} s"
            if t >= times[-1]:
                break
        hand = next(iter(step["preview"]["effector_paths"]))
        return True, f"{step['description'].lower()} ({times[-1]:.1f} s); {hand} at " \
                     f"{_r(self.world.hand_in_robot(hand))} (robot)"


def _r(values) -> list[float]:
    return [round(float(v), 3) for v in values]
