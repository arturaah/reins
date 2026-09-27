"""What the top-level model can ask the robot to do in the room, and how each request becomes a plan.

The model gets only what the real R1 can sense (see world.py):

- **Sensing** answers straight away and moves nothing: `robot_state` (joint
  angles, IMU, odometry, hands), `look` (the head camera image), and `locate`
  (the model points at pixels in the last image; depth turns them into 3D
  points).
- **Acting** returns a `Proposal`: contract steps plus what to draw. Only an
  approved proposal reaches `execute`.
  - `walk_to`, `walk`, `turn`, `approach` plan `walk` steps around the
    obstacles the depth camera has seen.
  - `manipulate` hands one arm to the closed-loop arm policy in the rest of
    `harness` (see arm.py) for a bounded task such as "pick up the red cube".
    Its plan is a `servo` step: the reviewer approves the task and the box the
    hand must stay in, not a precomputed path, because the policy decides each
    move from what it sees. Artur's safety gate enforces that box.
  - `arm_home` relaxes the arms along a joint-space `arm` step.

No tool knows what objects exist or where they are.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from reins_loco.base import FSM_WALK, Pose2, wrap
from reins_loco.skills import execute_walk_step, plan_turn, plan_walk, plan_walk_to

from ..kinematics import ARM_JOINTS, ArmKinematics
from . import planner
from .camera import H_FOV, V_FOV, annotate
from .world import HANDS, HOME, SimWorld, png_bytes

# How far in front of the pelvis to stand from a target, best first. The arm policy starts
# with the hand about 0.30 m ahead and the arm reaches about 0.46 m from the shoulder.
STAND_OFF = (0.33, 0.36, 0.38)
REACH_SIDE = 0.15  # m to the hand's side
ARM_TICK = 0.02  # s between arm updates when executing
WALK_TOLERANCE = 0.03  # m; the hands need the robot to stop close to where it was planned
HOME_SPEED = 0.8  # rad/s for the joint-space move to the relaxed pose
ERR_BLOCKED = 7401


class PlanError(ValueError):
    """The request can't be planned; the message tells the model why."""


@dataclass
class Overlay:
    """What a reviewer sees drawn in the scene, in the odom frame."""
    walk: list[dict] = field(default_factory=list)  # contract walk steps
    hands: dict[str, list[np.ndarray]] = field(default_factory=dict)  # hand -> points
    marks: list[np.ndarray] = field(default_factory=list)  # targets
    boxes: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)  # (centre, half sizes, yaw)


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
                    "camera's depth. Point at an object's visible top, or at a spot on a surface. "
                    "Returns positions in the robot and odom frames, or says there's no depth there.",
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
                    "to put something), facing it. Use a point from locate. auto sets up for the "
                    "hand that's holding something, else whichever is closer.",
     "input_schema": {"type": "object", "required": ["x", "y", "z"], "properties": {
         **XYZ, "frame": FRAME, "hand": HAND_AUTO, "description": DESCRIPTION}}},
    {"name": "manipulate",
     "description": "Hand one arm to the arm policy for a short task within reach, such as "
                    "\"pick up the green mug\" or \"put the held mug down on the shelf\". A "
                    "vision model then steers the hand in small steps using the head and wrist "
                    "cameras until the task is done, inside a safety box in front of the robot. "
                    "Approach first. The result says how the episode ended; look to check.",
     "input_schema": {"type": "object", "required": ["task", "surface_z"], "properties": {
         "task": {"type": "string", "description": "one short instruction for the arm policy, naming "
                                                   "the object and what to do with it"},
         "hand": {**HAND_AUTO, "description": "auto: the hand that is closed, else the one approach set up for"},
         "surface_z": _num("height of the table or counter top the work happens on, from locate; "
                           "the hand is kept 2 cm above it"),
         "description": DESCRIPTION}}},
    {"name": "arm_home",
     "description": "Return one or both arms to the relaxed pose, keeping hold of anything held.",
     "input_schema": {"type": "object", "properties": {
         "hand": {"enum": ["left_hand", "right_hand", "both"], "default": "both"},
         "description": DESCRIPTION}}},
]

SENSING = {"robot_state", "look", "locate"}


def room_config(overrides: dict | None = None):
    """harness/config.yaml with the room's differences: a virtual hand, and the claude-cli policy."""
    from .. import config
    return config.load(None, {"hand.type": "virtual", "vlm.provider": "claude-cli", **(overrides or {})})


class Skills:
    def __init__(self, world: SimWorld, cfg=None, arm_vlm: Callable[[str, str], object] | None = None,
                 confirm_moves: Callable[[str], bool] | None = None):
        """arm_vlm(task, hand) -> a harness.vlm VLM for one manipulation episode (default: from cfg).
        confirm_moves: asked before every move inside an episode; None trusts the approved box."""
        self.world = world
        self.cfg = cfg or room_config()
        self.arm_vlm = arm_vlm or (lambda task, hand: _make_vlm(self.cfg))
        self.confirm_moves = confirm_moves
        self._kin = {arm: ArmKinematics(self.cfg["robot"]["model"], arm) for arm in ("left", "right")}
        self._approach_hand: str | None = None
        self.log = print

    # --- dispatch --------------------------------------------------------------

    def call(self, name: str, args: dict) -> Observation | Proposal:
        """Run a sensing tool, or plan an acting one. Raises PlanError if it can't be planned."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise PlanError(f"unknown tool {name!r}")
        args = {k: v for k, v in args.items() if v is not None}
        try:
            return handler(**args)
        except planner.NoPath as e:
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
        closed = [h for h in HANDS if self.world.grip_command[h] == "closed"]
        hands = [hand] if hand != "auto" else closed or list(HANDS)  # set up for the hand that's holding
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
            self._approach_hand = h
            return Proposal(steps[-1]["description"], steps,
                            Overlay(walk=steps, marks=[np.array([tx, ty, tz])]))
        raise PlanError("no clear spot to stand within reach of that point: the hands reach only about "
                        "0.18 m past the robot's front, so on a table or counter choose a spot close to "
                        "the edge nearest the robot")

    # --- arms ------------------------------------------------------------------

    def _tool_manipulate(self, task: str, surface_z: float, hand: str = "auto",
                         description: str | None = None) -> Proposal:
        if hand == "auto":
            closed = [h for h in HANDS if self.world.grip_command[h] == "closed"]
            hand = closed[0] if closed else (self._approach_hand or "right_hand")
        if hand not in HANDS:
            raise PlanError(f"hand must be left_hand or right_hand, got {hand!r}")
        ws, st, lp = self.cfg["workspace"], self.cfg["steps"], self.cfg["loop"]
        lo = np.array(ws["box_min_m"], float)
        hi = np.array(ws["box_max_m"], float)
        floor = float(surface_z) + float(ws["table_margin_m"])
        lo[2] = max(lo[2], floor)
        if lo[2] >= hi[2]:
            raise PlanError(f"surface_z {surface_z:.2f} m leaves no room for the hand inside the safety box")
        step = {"step_id": "s1", "kind": "servo", "description": description or task,
                "effector": hand, "task": task,
                "bounds": {"frame": "robot", "box_min_m": _r(lo), "box_max_m": _r(hi),
                           "max_step_m": float(st["max_translation_m"]), "max_steps": int(lp["max_steps"]),
                           "max_joint_vel_rad_s": float(self.cfg["limits"]["max_joint_vel_rad_s"])}}
        pose = self.world.odom_pose()
        centre = self.world.to_odom((lo + hi) / 2, pose)
        return Proposal(f"{hand.replace('_', ' ').capitalize()}: {task}", [step],
                        Overlay(boxes=[(centre, (hi - lo) / 2, pose.yaw)]))

    def _tool_arm_home(self, hand: str = "both", description: str | None = None) -> Proposal:
        hands = list(HANDS) if hand == "both" else [hand]
        pose = self.world.odom_pose()
        steps, paths = [], {}
        for h in hands:
            arm = h.split("_")[0]
            names = ARM_JOINTS[arm]
            q0 = np.array([self.world.arm_q[n] for n in names])
            q1 = np.array([HOME.get(n, 0.0) for n in names])
            span = float(np.abs(q1 - q0).max())
            if span < 0.01:
                continue  # already relaxed
            duration = max(span / HOME_SPEED, 0.3)
            times = np.linspace(0, duration, 13)
            rows = [q0 + (q1 - q0) * t / duration for t in times]
            points = [self.world.to_odom(self._kin[arm].fk(q)[0], pose) for q in rows]
            steps.append({"step_id": f"s{len(steps) + 1}", "kind": "arm",
                          "description": f"Relax the {h.replace('_', ' ')}",
                          "trajectory": {"joint_names": names, "times_s": [round(float(t), 4) for t in times],
                                         "positions_rad": [[round(float(v), 5) for v in r] for r in rows]},
                          "preview": {"effector_paths": {h: {
                              "frame": "robot", "points": [_r(self.world.to_robot(p, pose)) for p in points],
                              "times_s": [round(float(t), 4) for t in times]}}}})
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
            elif kind == "servo":
                ok, msg = self._servo(step, tick, should_stop)
                log.append(msg)
                if not ok:
                    return False, "; ".join(log)
            else:
                return False, f"can't execute a {kind!r} step"
        return True, "; ".join(log)

    def _servo(self, step: dict, tick, should_stop) -> tuple[bool, str]:
        from .arm import run_episode
        hand, task = step["effector"], step["task"]
        if self.world.loco.fsm_id() == FSM_WALK:
            self.world.loco.stop()
        surface_z = step["bounds"]["box_min_m"][2] - float(self.cfg["workspace"]["table_margin_m"])
        summary = run_episode(self.world, self.cfg, self.arm_vlm(task, hand), hand, task, surface_z,
                              tick, should_stop, self.confirm_moves, log=self.log)
        stopped = should_stop() or summary.get("reason") == "e-stop"
        text = (f"arm policy {'finished' if summary.get('success') else 'stopped'}: {summary.get('reason')} "
                f"after {summary.get('steps')} steps; {hand} is {self.world.grip_command[hand]}")
        if stopped:
            return False, text + "; stopped by the operator"
        return bool(summary.get("success")), text

    def _settle(self, tick) -> None:
        """Wait for the base to stop. The gait lowers the pelvis, so arms work from a standstill."""
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


def _make_vlm(cfg):
    from ..vlm import base
    return base.make(cfg)


def _r(values) -> list[float]:
    return [round(float(v), 3) for v in values]
