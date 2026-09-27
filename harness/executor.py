"""Setpoint -> gate -> IK -> interpolated joint frames -> backend, plus execution feedback.

ArmExecutor is the only caller of a backend's stream(): it hands over frames that came out of
harness.safety.SafetyGate.vet and were re-checked by check_trajectory. Backends:
  harness.sim.mock_robot.MockBackend       kinematic mock, optional rendering (Mac, no hardware)
  harness.robot.dry_run.DryRunBackend      real joint state, prints what would be sent, publishes nothing
  harness.robot.arm_client.ArmClientBackend client of the arm_sdk streamer process (the real robot)
"""
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from core import trajectory

from .interpreter import ArmState


def ease(x):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0.0), 1.0))


def interpolate(q_from, q_to, duration_s, rate_hz):
    """Cosine-eased joint frames from q_from to q_to, one per 1/rate_hz, last one exactly q_to."""
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    n = max(1, int(round(duration_s * rate_hz)))
    return [q_from + (q_to - q_from) * ease(i / n) for i in range(1, n + 1)]


@dataclass
class ExecResult:
    ok: bool
    feedback: str                      # what the model is told
    p_before: Optional[np.ndarray] = None
    p_after: Optional[np.ndarray] = None
    requested_dp: Optional[np.ndarray] = None
    achieved_dp: Optional[np.ndarray] = None
    roll_before: float = 0.0
    roll_after: float = 0.0
    q_target: Optional[np.ndarray] = None
    q_after: Optional[np.ndarray] = None
    ik_fail: bool = False
    timeout: bool = False
    clamped: bool = False
    hand_closed: Optional[bool] = None
    empty_grasp: bool = False
    duration_s: float = 0.0
    notes: list = field(default_factory=list)
    declined: bool = False             # the operator rejected the move before it was sent
    operator_note: str = ""            # the note they typed with their Accept or Reject, when they gave one
    asked: bool = False                # an operator was asked about this move (confirm was set)
    walk: Optional[tuple] = None       # a whole-body step that was executed: (dx, dy, dyaw) achieved (odometry) or commanded
    moves: list = field(default_factory=list)      # a trajectory: the per-move ExecResults, in order, up to the first stop
    waypoints: list = field(default_factory=list)  # a trajectory: the planned hand-tip targets the operator saw
    planned: int = 0                   # a trajectory: how many moves were proposed (0 = a single move)


class Backend:
    """What a robot (or its mock) must provide. All joints by MuJoCo name."""
    name = "backend"
    dry_run = False
    def joints(self) -> dict: raise NotImplementedError
    def velocities(self) -> dict: raise NotImplementedError
    def stream(self, arm, frames, dt): raise NotImplementedError      # blocking, frames: list of 5-vectors
    def walk(self, vx, vy, vyaw, duration): raise NotImplementedError  # blocking; -> odometry {dx, dy, dyaw} in the pre-walk body frame, or None
    def hand(self, arm, closed) -> str: raise NotImplementedError      # returns feedback text
    def hand_state(self, arm): return None                              # True closed, False open, None unknown
    def engage(self): pass
    def release(self): pass
    def freeze(self): pass


class ArmExecutor:
    def __init__(self, cfg, kin, gate, backend, arm):
        self.cfg, self.kin, self.gate, self.backend, self.arm = cfg, kin, gate, backend, arm
        self.rate = float(cfg["robot"]["command_rate_hz"])
        # optional callable(text, preview) asked before every motion: True sends; False or a str (the operator's
        # reason) rejects. preview = {arm, q_now, frames, dt, joints}: what would be streamed, for a ghost preview.
        self.confirm = None

    # -- state ------------------------------------------------------------------------------------
    def others(self, joints):
        return {n: v for n, v in joints.items() if n not in self.kin.joint_names}

    def sync(self):
        j = self.backend.joints()
        q = self.kin.q_from_dict(j)
        p, _ = self.kin.fk(q, self.others(j))
        return ArmState(p, float(q[4]), bool(self.backend.hand_state(self.arm) or False), dict(j))

    # -- motion -----------------------------------------------------------------------------------
    def execute(self, proposal, state):
        """proposal: harness.interpreter.Proposal; state: ArmState read just before. Returns ExecResult."""
        if proposal.kind in ("still", "done"):
            return ExecResult(True, "", state.p, state.p, np.zeros(3), np.zeros(3), state.roll, state.roll)
        if proposal.kind == "unavailable":
            return ExecResult(False, proposal.note, state.p, state.p, np.zeros(3), np.zeros(3), state.roll, state.roll)
        if proposal.kind == "hand":
            return self._hand(proposal, state)
        if proposal.kind == "walk":
            return self._walk(proposal, state)
        j = self.backend.joints()
        q_now = self.kin.q_from_dict(j)
        others = self.others(j)
        v = self.gate.vet(state.p, state.roll, proposal.p, proposal.roll, q_now, others, proposal.mode)
        requested = np.asarray(proposal.p, float) - state.p
        if not v.ok:
            return ExecResult(False, v.feedback, state.p, state.p, requested, np.zeros(3), state.roll, state.roll,
                              ik_fail=v.reason.startswith("IK_FAIL"), clamped=bool(v.clamped), notes=v.clamped)
        dt = 1.0 / self.rate
        frames = interpolate(q_now, v.q_target, v.duration_s, self.rate)
        try:
            plan = trajectory.resolve(trajectory.frame_plan(self.arm, q_now, frames, dt, j), self.arm, j, self.rate)
            frames, dt = trajectory.frames(plan, self.arm)
            v.duration_s = plan["duration_s"]
        except ValueError as exc:
            return ExecResult(False, str(exc))
        bad = self.gate.check_trajectory(frames, dt, q_now, others)
        if bad:
            return ExecResult(False, f"REJECTED by the trajectory check: {bad}", state.p, state.p, requested, np.zeros(3),
                              state.roll, state.roll, clamped=bool(v.clamped), notes=v.clamped)
        note, asked = "", False
        if self.confirm is not None:
            dq = v.q_target - q_now
            text = (f"{proposal.action.raw if proposal.action else proposal.kind}: hand {state.p.round(3).tolist()} -> {v.p.round(3).tolist()} m, "
                    f"roll {math.degrees(state.roll):.0f} -> {math.degrees(v.roll):.0f} deg, {v.duration_s:.1f} s, "
                    f"largest joint change {math.degrees(float(np.abs(dq).max())):.0f} deg" +
                    (f"; {'; '.join(v.clamped)}" if v.clamped else ""))
            ok, note = self._ask(text, {"arm": self.arm, "q_now": q_now, "frames": frames, "dt": dt, "joints": j}); asked = True
            if not ok:
                return self._declined(note, state, requested)
        if self.gate.estop.is_set():
            return ExecResult(False, "ESTOP: nothing moves")
        try:
            trajectory.require_start(plan, self.backend.joints())
        except ValueError as exc:
            return ExecResult(False, str(exc))
        t0 = time.time()
        self.backend.stream(self.arm, frames, dt)
        timeout = self._settle()
        after = self.sync()
        achieved = after.p - state.p
        fb = self._feedback(proposal, requested, achieved, v, timeout)
        return ExecResult(True, fb, state.p, after.p, requested, achieved, state.roll, after.roll, v.q_target,
                          self.kin.q_from_dict(after.q), False, timeout, bool(v.clamped), None, False, time.time() - t0, v.clamped,
                          asked=asked, operator_note=note)

    def plan_sequence(self, actions, state, propose):
        """Chain the actions from the current state: each proposal is made from, and vetted against, the predicted end of
        the previous move. propose(state, action) -> Proposal (the loop's interpreter with the current step sizes).
        -> (legs, note): legs = [{proposal, v, frames, p_from, roll_from, q_from}] up to the first action the gate refuses
        or that is not an arm move / rotation; note says which and why ('' when every action went in)."""
        from .interpreter import ArmState
        j = self.backend.joints()
        q, others = self.kin.q_from_dict(j), self.others(j)
        p, roll = np.asarray(state.p, float), float(state.roll)
        legs, note, dt = [], "", 1.0 / self.rate
        for a in actions:
            pr = propose(ArmState(p, roll, state.hand_closed, {}), a)
            if pr.kind not in ("move", "rotate"):
                note = f"{a.raw.upper()} cannot be part of a trajectory" + (f": {pr.note}" if pr.note else ""); break
            v = self.gate.vet(p, roll, pr.p, pr.roll, q, others, pr.mode)
            if not v.ok:
                note = f"{a.raw.upper()}: {v.feedback}"; break
            frames = interpolate(q, v.q_target, v.duration_s, self.rate)
            try:
                predicted_pose = {**j, **dict(zip(self.kin.joint_names, q))}
                plan = trajectory.resolve(trajectory.frame_plan(self.arm, q, frames, dt, predicted_pose), self.arm, predicted_pose, self.rate)
                frames, dt = trajectory.frames(plan, self.arm)
                v.duration_s = plan["duration_s"]
            except ValueError as exc:
                note = f"{a.raw.upper()}: {exc}"; break
            bad = self.gate.check_trajectory(frames, dt, q, others)
            if bad:
                note = f"{a.raw.upper()}: REJECTED by the trajectory check: {bad}"; break
            legs.append({"proposal": pr, "v": v, "frames": frames, "p_from": p, "roll_from": roll, "q_from": q})
            p, roll, q = np.asarray(v.p, float), float(v.roll), np.asarray(v.q_target, float)
        return legs, note

    def execute_sequence(self, actions, state, propose):
        """Several arm moves as ONE proposal: planned from the current state (plan_sequence), shown and confirmed as a
        whole (the preview carries every frame, so the twin and the glasses draw the entire path), then executed move by
        move, each re-vetted from the measured pose toward its planned target, so the arm follows the waypoints the
        operator saw. Stops at the first move that fails or stalls. -> one ExecResult for the trajectory (ok when at
        least one move ran) with .moves, .waypoints and .planned set."""
        legs, note = self.plan_sequence(actions, state, propose)
        zero = np.zeros(3)
        if not legs:                                                   # the first move is refused: the single-move path says why
            first = propose(state, actions[0])
            if first.kind in ("move", "rotate"):
                return self.execute(first, state)
            return ExecResult(False, note, state.p, state.p, zero, zero, state.roll, state.roll)
        tokens = [l["proposal"].action.raw.upper() for l in legs]
        dropped = f"; the plan's move {len(legs) + 1} ({actions[len(legs)].raw.upper()}) was dropped: {note}" if note else ""
        frames = [f for l in legs for f in l["frames"]]
        dt = 1.0 / self.rate
        total_s = sum(l["v"].duration_s for l in legs)
        clamped = [c for l in legs for c in l["v"].clamped]
        dq = max(float(np.abs(np.asarray(l["v"].q_target) - l["q_from"]).max()) for l in legs)
        p_end = np.asarray(legs[-1]["v"].p, float)
        waypoints = [np.asarray(l["v"].p, float) for l in legs]
        op_note, asked = "", False
        if self.confirm is not None:
            text = (f"TRAJECTORY of {len(legs)} moves ({', '.join(tokens)}): hand {state.p.round(3).tolist()} -> {p_end.round(3).tolist()} m "
                    f"over {total_s:.1f} s, largest joint change {math.degrees(dq):.0f} deg" + (f"; {'; '.join(clamped)}" if clamped else "") + dropped)
            ok, op_note = self._ask(text, {"arm": self.arm, "q_now": legs[0]["q_from"], "frames": frames, "dt": dt, "joints": self.backend.joints()})
            asked = True
            if not ok:
                r = self._declined(op_note, state, p_end - state.p)
                r.waypoints, r.planned = waypoints, len(legs)
                return r
        saved, self.confirm = self.confirm, None                      # accepted as a whole: no question per move
        moves, stop = [], ""
        try:
            for k, l in enumerate(legs):
                st = self.sync()
                r = self.execute(l["proposal"], st)
                moves.append(r)
                if not r.ok:
                    stop = f"stopped before move {k + 1} ({tokens[k]}): {r.feedback}"; break
                want = np.asarray(r.requested_dp, float)
                if np.linalg.norm(want) > 1e-6 and float(np.asarray(r.achieved_dp) @ want) < 0.3 * float(np.linalg.norm(want)) ** 2:
                    stop = f"stopped after move {k + 1} ({tokens[k]}): blocked or in contact, the rest was not sent"; break
        finally:
            self.confirm = saved
        done = [r for r in moves if r.ok]
        after = self.sync()
        parts = [f"{tokens[k]}: {r.feedback or 'ok'}" for k, r in enumerate(moves)]
        fb = (f"trajectory: {len(done)} of {len(legs)} moves executed" + (f" ({'; '.join(parts)})" if parts else "")
              + (f"; {stop}" if stop else "") + dropped)
        planned_dp = (waypoints[len(done) - 1] - state.p) if done else zero
        res = ExecResult(bool(done), fb, state.p, after.p, planned_dp, after.p - state.p, state.roll, after.roll,
                         q_target=done[-1].q_target if done else None, q_after=self.kin.q_from_dict(after.q),
                         timeout=any(r.timeout for r in moves), clamped=bool(clamped), duration_s=sum(r.duration_s for r in moves),
                         notes=clamped, asked=asked, operator_note=op_note)
        res.moves, res.waypoints, res.planned = moves, waypoints, len(legs)
        return res

    def _walk(self, proposal, state):
        """A whole-body step: gate (caps, budget), confirmation, the backend's loco call, odometry feedback."""
        v = self.gate.vet_walk(*proposal.walk)
        zero = np.zeros(3)
        if not v.ok:
            return ExecResult(False, v.feedback, state.p, state.p, zero, zero, state.roll, state.roll, clamped=bool(v.clamped), notes=v.clamped)
        what = (f"walk {math.hypot(v.dx, v.dy) * 100:.0f} cm {'forward' if v.dx > 0 else 'back' if v.dx < 0 else 'left' if v.dy > 0 else 'right'}"
                if (v.dx or v.dy) else f"turn {math.degrees(abs(v.dyaw)):.0f} deg {'left' if v.dyaw > 0 else 'right'}")
        text = (f"{proposal.action.raw if proposal.action else 'walk'}: the WHOLE ROBOT steps: {what} at {max(abs(v.vx), abs(v.vy)):.2f} m/s, "
                f"{math.degrees(abs(v.vyaw)):.0f} deg/s over {v.duration_s:.1f} s" + (f"; {'; '.join(v.clamped)}" if v.clamped else ""))
        note, asked = "", False
        if self.confirm is not None:
            ok, note = self._ask(text, {"walk": (v.dx, v.dy, v.dyaw), "duration": v.duration_s, "arm": self.arm, "joints": self.backend.joints()}); asked = True
            if not ok:
                return self._declined(note, state, zero)
        t0 = time.time()
        try:
            odom = self.backend.walk(v.vx, v.vy, v.vyaw, v.duration_s)
        except RuntimeError as e:                                    # the streamer refused (FSM, caps): the model is told, the episode goes on
            self.gate.walked_m -= math.hypot(v.dx, v.dy); self.gate.turned_rad -= abs(v.dyaw)
            return ExecResult(False, f"the step was refused: {e}", state.p, state.p, zero, zero, state.roll, state.roll, asked=asked, operator_note=note)
        after = self.sync()
        if odom:
            fb = (f"walked {odom['dx'] * 100:.0f} cm forward, {odom['dy'] * 100:.0f} cm left, turned {math.degrees(odom['dyaw']):.0f} deg (odometry); "
                  "the view has changed, judge the target again")
            done = (float(odom["dx"]), float(odom["dy"]), float(odom["dyaw"]))
        else:
            fb = f"{what} commanded (no odometry available); the view has changed, judge the target again"
            done = (v.dx, v.dy, v.dyaw)
        if v.clamped:
            fb += "; clamped: " + "; ".join(v.clamped)
        return ExecResult(True, fb, state.p, after.p, zero, zero, state.roll, after.roll, duration_s=time.time() - t0,
                          clamped=bool(v.clamped), notes=v.clamped, asked=asked, operator_note=note, walk=done)

    def _ask(self, text, preview):
        """-> (accepted, note). confirm may answer True / False, a str (a rejection carrying that note) or (accepted, note).
        preview: what would move ({arm, q_now, frames, dt, joints} for an arm move, {walk, duration} for a step)."""
        ans = self.confirm(text, preview)
        if isinstance(ans, tuple):
            ok, note = bool(ans[0]), str(ans[1] or "")
        elif isinstance(ans, str):
            ok, note = False, ans
        else:
            ok, note = bool(ans), ""
        return ok, note.strip()

    @staticmethod
    def _declined(note, state, requested):
        return ExecResult(False, "the operator rejected this move" + (f": {note}" if note else ""), state.p, state.p, requested,
                          np.zeros(3), state.roll, state.roll, declined=True, operator_note=note, asked=True)

    def _settle(self):
        lim = self.cfg["limits"]
        t0 = time.time()
        thresh, timeout = float(lim["settle_vel_rad_s"]), float(lim["settle_timeout_s"])
        while time.time() - t0 < timeout:
            vel = self.backend.velocities()
            if all(abs(vel.get(n, 0.0)) < thresh for n in self.kin.joint_names):
                return False
            time.sleep(0.02)
        return True

    def _feedback(self, proposal, requested, achieved, v, timeout):
        parts = []
        if proposal.kind == "move":
            want, got = float(np.linalg.norm(requested)), float(np.dot(achieved, requested) / max(np.linalg.norm(requested), 1e-9))
            parts.append(f"moved {got * 100:.1f} of {want * 100:.1f} cm")
            if got < 0.3 * want:
                parts.append("-> blocked or in contact, do NOT repeat this move")
            elif got < 0.75 * want:
                parts.append("(the arm sags under its own weight at these gains; the rest may come with one more step)")
        elif proposal.kind == "rotate":
            parts.append(f"wrist rolled to {math.degrees(v.roll):.0f} deg")
        if v.clamped:
            parts.append("clamped: " + "; ".join(v.clamped))
        if timeout:
            parts.append("the arm was still moving after the settle timeout")
        return " ".join(parts)

    def _hand(self, proposal, state):
        note, asked = "", False
        if self.confirm is not None and self.cfg["hand"]["type"] != "none":        # a real hand moves: ask like any motion
            j = self.backend.joints()
            q_now = self.kin.q_from_dict(j)
            text = f"{'close' if proposal.hand_closed else 'open'} the {self.arm} hand ({self.cfg['hand']['type']}); the arm holds still"
            ok, note = self._ask(text, {"arm": self.arm, "q_now": q_now, "frames": [q_now, q_now], "dt": 1.0 / self.rate, "joints": j}); asked = True
            if not ok:
                return self._declined(note, state, np.zeros(3))
        fb = self.backend.hand(self.arm, proposal.hand_closed)
        closed = self.backend.hand_state(self.arm)
        empty = False
        if proposal.hand_closed and closed is not None and self.cfg["hand"]["type"] != "none":
            empty = fb.startswith("EMPTY")
        return ExecResult(True, fb, state.p, state.p, np.zeros(3), np.zeros(3), state.roll, state.roll,
                          hand_closed=closed, empty_grasp=empty, asked=asked, operator_note=note)

    def go_to_joints(self, q_target, label="joint move"):
        """Joint-space move (start pose, home step). Still gated: e-stop, limits, speed cap, self-collision, confirmation."""
        if self.gate.estop.is_set():
            return ExecResult(False, "ESTOP: nothing moves")
        state = self.sync()
        j = self.backend.joints()
        q_now = self.kin.q_from_dict(j)
        q_t = self.kin.clamp(np.asarray(q_target, float))
        bad = self.kin.joint_violations(q_t)
        if bad:
            return ExecResult(False, f"{label} refused: {bad} outside the joint range", state.p, state.p)
        if self.gate.contacts_baseline is not None and self.kin.contacts(q_t, self.others(j)) > self.gate.contacts_baseline:
            return ExecResult(False, f"{label} refused: the model shows a self-collision at the target", state.p, state.p)
        lim = self.cfg["limits"]
        duration = max(float(lim["min_move_s"]), float(np.abs(q_t - q_now).max()) / float(lim["max_joint_vel_rad_s"]) * math.pi / 2)
        frames = interpolate(q_now, q_t, duration, self.rate)
        try:
            plan = trajectory.resolve(trajectory.frame_plan(self.arm, q_now, frames, 1.0/self.rate, j), self.arm, j, self.rate)
            frames, dt = trajectory.frames(plan, self.arm)
            duration = plan["duration_s"]
        except ValueError as exc:
            return ExecResult(False, str(exc))
        bad = self.gate.check_trajectory(frames, dt, q_now, self.others(j))
        if bad:
            return ExecResult(False, f"{label} refused by the trajectory check: {bad}", state.p, state.p)
        p_t, _ = self.kin.fk(q_t, self.others(j))
        note, asked = "", False
        if self.confirm is not None:
            text = (f"{label}: hand {state.p.round(3).tolist()} -> {p_t.round(3).tolist()} m over {duration:.1f} s, "
                    f"largest joint change {math.degrees(float(np.abs(q_t - q_now).max())):.0f} deg")
            ok, note = self._ask(text, {"arm": self.arm, "q_now": q_now, "frames": frames, "dt": 1.0 / self.rate, "joints": j}); asked = True
            if not ok:
                r = self._declined(note, state, p_t - state.p); r.feedback = f"the operator rejected the {label}" + (f": {note}" if note else "")
                return r
        if self.gate.estop.is_set():
            return ExecResult(False, "ESTOP: nothing moves")
        try:
            trajectory.require_start(plan, self.backend.joints())
        except ValueError as exc:
            return ExecResult(False, str(exc))
        t0 = time.time()
        self.backend.stream(self.arm, frames, dt)
        timeout = self._settle()
        after = self.sync()
        return ExecResult(True, f"{label} done", state.p, after.p, p_t - state.p, after.p - state.p, state.roll, after.roll,
                          q_t, self.kin.q_from_dict(after.q), False, timeout, duration_s=time.time() - t0, asked=asked, operator_note=note)

    def home_step(self, state, q_home, fraction=0.3):
        """One step toward the home joint pose after repeated IK failures."""
        q_now = self.kin.q_from_dict(self.backend.joints())
        q_t = q_now + (np.asarray(q_home, float) - q_now) * fraction
        r = self.go_to_joints(q_t, "home step")
        if r.ok:
            r.feedback = f"moved {fraction:.0%} of the way toward the home pose"
        return r
