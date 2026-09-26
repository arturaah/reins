"""The harness loop: the model proposes, a human reviews, the sim acts.

    task ─► brain ─► tool call ─► Skills.call ─┬─ sensing: result straight back
                ▲                              └─ acting: Proposal
                │                                   │ plan_proposed + preview image
                │                              reviewer ── decline + feedback ──┐
                │                                   │ approve                  │
                │                              Skills.execute (Enter aborts)   │
                └──────────── tool result ◄─────────┴──────────────────────────┘

Every session is written as contract messages (contract/README.md) and each
one is validated against the schema as it's written, so the log can be
replayed into any other Reins surface.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from reins_contract import validate

from .brains import Brain, ToolCall, ToolResult
from .display import Display, Lines
from .skills import SENSING, TOOLS, Observation, PlanError, Proposal, Skills


MAX_JOINT_VEL = 1.0  # rad/s, the limit plans declare; generated trajectories aim for 0.8

SYSTEM = """\
You control a Unitree R1 humanoid robot through the tools provided. It walks with \
its own balance controller, has two arms, and can pick up small objects with \
either hand. This run is a simulation, but you get exactly what the real robot \
would: its own joint readings and odometry, and its head camera. Nobody tells \
you what's in the room or where; find out by looking.

Seeing:
- look takes a picture with the head camera: a wide fisheye (150° across, 124° \
tall), pointing forward and a little down, with pixel coordinates marked along \
the edges. The robot has no neck, so turn to see elsewhere.
- locate turns pixels in the latest image into 3D positions using the camera's \
depth. To find an object, point at the middle of its visible top. To put \
something down, point at the spot on the surface. Then use those coordinates.
- robot_state gives joint angles, odometry and whether each hand is closed. The \
robot can't feel whether it's holding something: look to check.

Acting:
- Every motion tool proposes a plan. A human reviews each plan, drawn in the \
scene, before anything moves. If they approve, it runs, and you get back what \
happened plus a fresh camera image. If they decline, you get their feedback: \
take it seriously and adjust, don't resubmit the same plan.
- If a tool says it can't plan something, the message says why. Change the \
approach (stand somewhere else, use the other hand, look again) rather than \
repeating the call.
- To pick something up: look, locate it, approach that point, look and locate it \
again from close up (more accurate), then pick_up. To put it down: find the spot \
the same way, approach it, then place.
- Hands reach roughly 0.25-0.5 m in front of the pelvis, between about 0.6 and \
1.0 m above the floor. Walking plans only avoid obstacles the camera has seen.

Frames, metres and radians: `robot` has its origin on the floor under the \
pelvis, x forward, y left, z up, and moves with the robot. `odom` is fixed where \
the robot started; it comes from dead reckoning and drifts, so prefer fresh \
robot-frame positions from locate.

The simulation is kinematic: a grasp holds an object if the hand closes within \
3 cm of its middle, and nothing tests balance or grip force. Say so if the human \
asks whether this would work on the real robot.

Keep text between tool calls short. When the task is done, or you're stuck, stop \
calling tools and say what happened in a sentence or two."""


@dataclass
class Decision:
    approve: bool
    feedback: str | None = None


class Reviewer(Protocol):
    def review(self, plan: dict, preview: Path) -> Decision: ...


class AutoApprove:
    """Approves everything. For headless runs and tests only."""

    def review(self, plan: dict, preview: Path) -> Decision:
        print("  [auto-approved]")
        return Decision(True)


class TerminalReviewer:
    def __init__(self, lines: Lines):
        self.lines = lines

    def review(self, plan: dict, preview: Path) -> Decision:
        print(f"  Preview image: {preview}")
        print("  Approve? y = run it, n = decline, or type feedback to decline with it:")
        while True:
            line = self.lines.get(timeout=0.1)
            if line is None:
                return Decision(False, "operator closed the input")
            answer = line.strip()
            if not answer:
                continue
            if answer.lower() in ("y", "yes"):
                return Decision(True)
            if answer.lower() in ("n", "no"):
                return Decision(False)
            return Decision(False, answer)


class SessionLog:
    """Contract messages for one session, validated and appended to a JSONL file."""

    def __init__(self, path: Path | None):
        self.path = path
        self.messages: list[dict] = []
        self.state = "idle"
        self._n = 0
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def emit(self, kind: str, **fields) -> dict:
        self._n += 1
        message = {"type": kind, "id": f"h-{self._n}", "t": round(time.time(), 3), **fields}
        validate(message)
        self.messages.append(message)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(message) + "\n")
        return message

    def set_state(self, state: str, **fields) -> None:
        if state != self.state or fields:
            self.state = state
            self.emit("state", state=state, **fields)


class Harness:
    def __init__(self, skills: Skills, brain: Brain, reviewer: Reviewer, display: Display,
                 log: SessionLog, max_turns: int = 40):
        self.skills, self.brain, self.reviewer = skills, brain, reviewer
        self.display, self.log, self.max_turns = display, log, max_turns
        self._plans = 0

    def run(self, task: str, command_id: str = "c-1") -> str:
        self.command_id = command_id
        self.log.emit("command", command_id=command_id, text=task)
        self.log.set_state("planning")
        self.brain.start(SYSTEM, TOOLS, task)
        results: list[ToolResult] = []
        for _ in range(self.max_turns):
            turn = self.brain.step(results)
            if turn.text:
                print(f"\n{self.brain.name}: {turn.text}")
            if turn.done:
                self.log.set_state("idle")
                return turn.text
            results = []
            blocked = None  # once a plan is declined or fails, the rest of this turn's calls are skipped
            for call in turn.calls:
                if blocked:
                    results.append(ToolResult(call.id, f"Skipped: {blocked}", is_error=True))
                    continue
                result = self.handle(call)
                results.append(result)
                if call.name not in SENSING and (result.is_error or result.text.startswith("Operator")):
                    blocked = "an earlier call in the same turn was declined or failed; re-plan from its result."
        self.log.set_state("idle")
        return f"Stopped after {self.max_turns} turns."

    def handle(self, call: ToolCall) -> ToolResult:
        print(f"\n→ {call.name}({json.dumps(call.args)})")
        try:
            out = self.skills.call(call.name, call.args)
        except PlanError as e:
            print(f"  can't plan: {e}")
            return ToolResult(call.id, f"Can't plan that: {e}", is_error=True)
        if isinstance(out, Observation):
            print(f"  {out.text[:300]}{'…' if len(out.text) > 300 else ''}")
            return ToolResult(call.id, out.text, out.image_png)
        return self._review_and_run(call, out)

    def _review_and_run(self, call: ToolCall, proposal: Proposal) -> ToolResult:
        self._plans += 1
        plan = {"plan_id": f"p-{self._plans}", "revision": 1, "command_id": self.command_id,
                "summary": proposal.summary, "source": {"planner": "reins-harness", "model": self.brain.name},
                "limits": {"max_joint_vel_rad_s": MAX_JOINT_VEL}, "steps": proposal.steps}
        ref = {"plan_id": plan["plan_id"], "revision": 1}
        self.log.set_state("planning")
        self.log.emit("plan_proposed", plan=plan)
        self.log.set_state("proposed", **ref)
        print(describe(plan, self.skills))
        decision = self.reviewer.review(plan, self.display.preview(proposal.overlay))
        self.log.emit("decision", **ref, decision="approve" if decision.approve else "decline",
                      **({"feedback": decision.feedback} if decision.feedback else {}))
        if not decision.approve:
            self.log.set_state("planning")
            text = "Operator declined this plan." + (f" Feedback: {decision.feedback}" if decision.feedback
                                                    else " No feedback given.")
            print(f"  {text}")
            return ToolResult(call.id, text)

        self.log.set_state("executing", **ref)
        self.log.emit("execute", plan=plan)
        ok, detail = self.skills.execute(plan["steps"], self.display.tick, self.display.should_stop)
        stopped = "stopped by the operator" in detail
        self.log.emit("done", **ref, outcome="succeeded" if ok else "halted" if stopped else "failed",
                      detail=detail)
        if stopped:
            self.log.set_state("halting")
        self.log.set_state("idle")
        self.log.set_state("planning")
        status = "Done" if ok else "Operator stopped it" if stopped else "Failed"
        print(f"  {status}: {detail}")
        after = self.skills.snapshot("Head camera, just after.")  # let the model see the outcome
        return ToolResult(call.id, f"{status}: {detail}\n{after.text}", after.image_png,
                          is_error=not ok and not stopped)


def describe(plan: dict, skills: Skills) -> str:
    """The plan in words, for the terminal reviewer."""
    lines = [f"\nPLAN {plan['plan_id']}: {plan['summary']}"]
    for step in plan["steps"]:
        kind = step["kind"]
        if kind == "walk":
            pts = step["path"]["points"]
            length = sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(pts, pts[1:]))
            g = step["goal"]
            yaw = f", facing {np.degrees(g['yaw']):.0f}°" if "yaw" in g else ""
            lines.append(f"  {step['step_id']} walk  {step['description']}: {length:.2f} m "
                         f"to ({g['x']:.2f}, {g['y']:.2f}){yaw}")
        elif kind == "arm":
            (hand, path), = step["preview"]["effector_paths"].items()
            pts = np.array(path["points"])
            travel = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            lines.append(f"  {step['step_id']} arm   {step['description']}: {hand} travels {travel:.2f} m "
                         f"in {step['trajectory']['times_s'][-1]:.1f} s, ends at "
                         f"{np.round(pts[-1], 2).tolist()} (robot)")
        elif kind == "grip":
            lines.append(f"  {step['step_id']} grip  {step['description']}")
    return "\n".join(lines)
