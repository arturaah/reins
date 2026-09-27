"""Dashboard-owned proposal, review and execution pipeline.

Model tools can submit proposals, never approve them. Both the gesture planner
and the visual policy use the same compiler, validator, approval and executor.
"""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from core.ik import ArmIK, plan_from_waypoints
from core.generated_motion import compile_trajectory
from core import trajectory
from harness.config import load
from harness.executor import ArmExecutor, ExecResult, interpolate
from harness.interpreter import ArmState
from harness.kinematics import ArmKinematics
from harness.safety import SafetyGate

ROOT = Path(__file__).resolve().parents[1]


class PreviewBackend:
    name = "mock"
    dry_run = True

    def __init__(self, pose):
        self.q = dict(pose)

    def joints(self): return dict(self.q)
    def velocities(self): return {n: 0. for n in self.q}
    def hand_state(self, arm): return None
    def hand(self, arm, closed): return "No physical gripper is configured."
    def freeze(self): pass
    def release(self): pass
    def close(self): pass


class ReviewedExecutor(ArmExecutor):
    """Harness policy adapter: core IK and full-path checks replace its local planner."""
    def __init__(self, pipeline, arm):
        self.pipeline = pipeline
        kin = ArmKinematics(str(ROOT / "sim/models/r1/R1_fixed_base.xml"), arm)
        gate = SafetyGate(pipeline.cfg, kin, pipeline.cfg["workspace"]["table_z_m"],
                          live=pipeline.mode == "live")
        super().__init__(pipeline.cfg, kin, gate, pipeline.backend, arm)
        gate.estop = pipeline.cancelled

    def execute(self, proposal, state):
        if self.gate.estop.is_set():
            return ExecResult(False, "Stopped")
        if proposal.kind in ("still", "done", "hand", "unavailable"):
            return ExecResult(proposal.kind in ("still", "done"), "No arm motion requested")
        try:
            pose = self.pipeline.planning_pose()
            p, _ = self.pipeline.ik.fk(self.arm, pose, pose)
            q = [pose[n] for n in trajectory.ARM_JOINTS[self.arm]]
            target, roll, notes = self.gate.clamp_setpoint(p, q[4], proposal.p, proposal.roll, proposal.mode)
            if proposal.kind == "rotate":
                dest = np.array(q); dest[4] = roll
                duration = max(.6, abs(roll-q[4])/.25*math.pi/2)
                plan = trajectory.frame_plan(self.arm, q, interpolate(q, dest, duration, 50), .02, pose, "Wrist roll")
            else:
                plan, _ = plan_from_waypoints(self.pipeline.ik, self.arm, [target], pose=pose,
                                               name=proposal.action.raw if proposal.action else "Visual step", max_vel=.25)
            result = self.pipeline.review_and_run(plan, self.arm, "visual", pose)
            after = self.sync()
            return ExecResult(result, "Move completed" if result else "Operator declined or proposal expired",
                              state.p, after.p, np.asarray(proposal.p)-state.p, after.p-state.p,
                              state.roll, after.roll, declined=not result, asked=True, clamped=bool(notes), notes=notes,
                              operator_note=self.pipeline.last_review_note)
        except ValueError as exc:
            return ExecResult(False, str(exc), ik_fail="unreachable" in str(exc).lower())

    def go_to_joints(self, q_target, label="Home pose"):
        pose = self.pipeline.planning_pose()
        q = [pose[n] for n in trajectory.ARM_JOINTS[self.arm]]
        duration = max(1., float(np.max(np.abs(np.asarray(q_target)-q)))/.25*math.pi/2)
        plan = trajectory.frame_plan(self.arm, q, interpolate(q, q_target, duration, 50), .02, pose, label)
        try:
            ok = self.pipeline.review_and_run(plan, self.arm, "manual", pose)
            return ExecResult(ok, "Move completed" if ok else "Move declined", declined=not ok)
        except ValueError as exc:
            self.pipeline.event("blocked", str(exc))
            return ExecResult(False, str(exc))


class RobotPipeline:
    REVIEW_SECONDS = 120
    OPERATOR_TIMEOUT = 10.0

    def __init__(self, planner, simulation, cameras, iface="en6", cfg=None, run_dir=None,
                 backend_factory=None, visual_factory=None):
        self.planner, self.sim, self.cameras = planner, simulation, cameras
        self.iface = iface
        self.cfg = copy.deepcopy(cfg or load())
        self.cfg["limits"]["max_joint_vel_rad_s"] = .4
        self.cfg["loop"]["chunk_max"] = 1
        self.cfg["loop"]["max_steps"] = min(20, self.cfg["loop"]["max_steps"])
        self.ik = ArmIK(backend="mujoco")
        self.lock = threading.RLock()
        self.log_lock = threading.Lock()
        self.cancelled = threading.Event()
        self.decision_event = threading.Event()
        self.generation = 0
        self.worker = None
        self.visual = None
        self.streamer = None
        self.streamer_log = None
        self.pending_backend = None
        self.backend_factory = backend_factory
        self.visual_factory = visual_factory
        self.mode = "sim"
        self.backend = PreviewBackend(self._simulation_pose())
        self.auto_fallback = True
        self.provider = "codex"
        self.proposal = None
        self.primary_id = None
        self.plan = None
        self.decision = None
        self.last_review_note = ""
        self.paths = None
        self.state = "idle"
        self.message = "Ask for a motion, review its path, then approve it here or in the glasses."
        self.events = []
        self.revision = 0
        self.connected = False
        self.run_dir = Path(run_dir or ROOT / "runs/dashboard") / uuid.uuid4().hex
        self.glasses = {"connected": 0, "error": "", "port": None}
        self.glasses_token = secrets.token_urlsafe(24)
        self.last_operator = time.monotonic()
        self.closed = threading.Event()
        self.watchdog = threading.Thread(target=self._operator_watchdog, daemon=True)
        self.watchdog.start()
        self.planner.before_submit = self.before_submit
        self.planner.on_complete = self.planned
        self.planner.preview_pose = self.planning_pose
        self.planner.pose_label = lambda: "measured robot joints and held command targets" if self.mode == "live" else "approved simulation pose"

    def operator_seen(self):
        self.last_operator = time.monotonic()

    def _operator_watchdog(self):
        while not self.closed.wait(.25):
            if self.connected and (time.monotonic()-self.last_operator > self.OPERATOR_TIMEOUT
                                   or not getattr(self.backend, "alive", True)):
                self.stop()
                self.event("stopped", "Operator or robot connection lost. Arm control released.")

    def _simulation_pose(self):
        with self.sim.lock:
            return {self.sim.model.joint(i).name: float(self.sim.data.qpos[self.sim.model.jnt_qposadr[i]])
                    for i in range(self.sim.model.njnt)}

    def planning_pose(self):
        if self.mode == "live":
            if not self.connected:
                raise ValueError("Connect robot control before preparing a live proposal.")
            state = self.backend.snapshot()
            # All held command targets are modeled; require measured joints to agree before motion.
            pose = {**state["joints"], **state.get("targets", {})}
        else:
            pose = self.backend.joints()
        return {n: float(v) for n, v in pose.items() if self.ik.model.joint(n).id >= 0}

    def status(self):
        with self.lock:
            proposal = copy.deepcopy(self.proposal)
            if proposal:
                proposal["expired"] = time.time() >= proposal["expires_at"]
            return {"state": self.state, "message": self.message, "mode": self.mode,
                    "connected": self.connected, "busy": bool(self.worker and self.worker.is_alive()),
                    "proposal": proposal, "events": copy.deepcopy(self.events[-20:]),
                    "auto_fallback": self.auto_fallback, "provider": self.provider,
                    "iface": self.iface, "table_z_m": self.cfg["workspace"]["table_z_m"],
                    "glasses": dict(self.glasses), "run_id": self.run_dir.name}

    def event(self, stage, message):
        with self.lock:
            self.state, self.message = stage, str(message)[:800]
            entry = {"stage": stage, "message": self.message, "at": time.time()}
            self.events.append(entry)
            self.events = self.events[-100:]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.log_lock, (self.run_dir / "events.jsonl").open("a") as file:
            file.write(json.dumps(entry, allow_nan=False)+"\n")

    def before_submit(self):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("Finish or stop the current robot task before submitting another.")
            self.generation += 1
            self.cancelled.clear()
            self.proposal = self.plan = self.paths = None
            self.primary_id = None
            self.decision_event.clear()
            self.state, self.message = "planning", "Generating a trajectory with the primary planner."

    def planned(self, status, plan):
        with self.lock:
            if self.cancelled.is_set() or status["state"] == "cancelled":
                return
            generation = self.generation
            self.primary_id = status.get("id")
        if plan is not None:
            def ready():
                if generation != self.generation or self.cancelled.is_set(): return
                arm = status["target"]["arm"]
                pose = self.planning_pose()
                # Recompile authored geometry if the measured starting state has changed while planning.
                if plan.get("generated_trajectory"):
                    compiled = compile_trajectory(self.ik, plan["generated_trajectory"], pose)
                else:
                    if self.mode == "live" and status.get("source") == "demo":
                        raise ValueError("Simulation fixtures cannot authorize physical motion.")
                    compiled = plan
                self.review_and_run(compiled, arm, "trajectory", pose)
            self._start(ready)
        elif status["state"] == "blocked":
            if status.get("arm") in ("left", "right"):
                self.cfg["robot"]["arm"] = status["arm"]
            self.event("context" if self.auto_fallback and status.get("source") != "demo" else "blocked", status["message"])
            if self.auto_fallback and status.get("source") != "demo":
                self._start(lambda: self._fallback(status["prompt"], status["message"]))

    def show_primary(self, source, proposal_id):
        # Primary previews are automatically resolved and displayed by this pipeline.
        # Never replace a reviewed plan with the earlier, differently timed draft.
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            with self.lock:
                if (self.primary_id is not None and proposal_id != self.primary_id) or self.cancelled.is_set():
                    raise ValueError("Proposal is no longer available")
                if self.plan is not None and self.proposal is not None:
                    return
                if self.state in ("blocked", "needs_context", "stopped", "expired"):
                    raise ValueError(self.message)
            time.sleep(.02)
        raise ValueError("Proposal validation is still running")

    def _start(self, work):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("A task is already running")
            def run():
                try:
                    work()
                except Exception as exc:
                    if not self.cancelled.is_set():
                        self.event("blocked", str(exc))
            self.worker = threading.Thread(target=run, daemon=True)
            self.worker.start()

    def _obstacles(self):
        z = self.cfg["workspace"]["table_z_m"]
        if self.mode != "live" or z is None:
            return []
        # Table plane over the configured forward work area; full arm volume is checked.
        lo, hi = self.cfg["workspace"]["box_min_m"], self.cfg["workspace"]["box_max_m"]
        return [{"name": "measured table workspace", "min": [lo[0], lo[1], -.1],
                 "max": [hi[0], hi[1], float(z)]}]

    def offer(self, plan, arm, source, pose, generation=None):
        self.event("validating", "Checking the exact trajectory, both arms, joint limits, speed and acceleration.")
        resolved = trajectory.resolve(plan, arm, pose)
        report = trajectory.validate(resolved, arm, self.ik.model, self._obstacles())
        if self.mode == "live":
            report["coverage"] = "Robot model and measured table workspace; no depth obstacle map"
        from spectacles.plan_feed import hand_paths
        paths = hand_paths(self.ik.model, resolved, 160)
        with self.lock:
            if self.cancelled.is_set() or (generation is not None and generation != self.generation):
                raise ValueError("Proposal cancelled")
            self.revision += 1
            proposal_id = uuid.uuid4().hex
            # Keep legacy executors from loading a file without this runtime's approval.
            resolved["preview_only"] = True
            resolved["prompt_proposal"] = {"id": proposal_id, "revision": self.revision, "source": source}
            self.plan, self.paths = resolved, paths
            self.proposal = {"id": proposal_id, "revision": self.revision, "digest": trajectory.digest(resolved),
                             "name": resolved.get("name", "Arm motion"), "arm": arm, "source": source,
                             "mode": self.mode, "duration_s": resolved["duration_s"], "validation": report,
                             "expires_at": time.time()+self.REVIEW_SECONDS}
            self.decision = None
            self.decision_event.clear()
            self.sim.show_proposal(copy.deepcopy(resolved), proposal_id)
            self.event("review", "Review the path in simulation or glasses. Approve this motion or reject it.")
        return copy.deepcopy(self.proposal)

    def decide(self, proposal_id, digest, decision, note=""):
        self.operator_seen()
        if decision not in ("approve", "decline"):
            raise ValueError("Choose approve or decline")
        with self.lock:
            p = self.proposal
            if self.state != "review" or not p or p["id"] != proposal_id or p["digest"] != digest:
                raise ValueError("Proposal changed; review the current revision")
            if self.cancelled.is_set() or time.time() >= p["expires_at"] or self.decision is not None:
                raise ValueError("Proposal expired or already decided")
            self.decision = (decision, str(note)[:500])
            self.last_review_note = str(note)[:500]
            self.decision_event.set()
            waiting = bool(self.worker and self.worker.is_alive())
            self.state = "approved" if decision == "approve" else "declined"
        self.event(self.state, "Approved in "+self.mode+" mode." if decision == "approve" else "Proposal declined. "+str(note)[:500])
        if not waiting:
            if decision == "approve":
                self._start(self._execute)
            else:
                with self.lock:
                    self.proposal = self.plan = self.paths = None
        return self.status()

    def review_and_run(self, plan, arm, source, pose):
        p = self.offer(plan, arm, source, pose)
        try:
            while not self.cancelled.is_set() and time.time() < p["expires_at"]:
                if self.decision_event.wait(.1):
                    if self.decision and self.decision[0] == "approve":
                        self._execute()
                        return not self.cancelled.is_set()
                    return False
            if not self.cancelled.is_set():
                self.event("expired", "Approval expired. Prepare a new proposal.")
            return False
        finally:
            with self.lock:
                if self.proposal and self.proposal["id"] == p["id"]:
                    self.proposal = self.plan = self.paths = None

    def _execute(self):
        with self.lock:
            if not self.proposal or not self.decision or self.decision[0] != "approve" or self.cancelled.is_set():
                raise ValueError("No current approval")
            p, plan = copy.deepcopy(self.proposal), copy.deepcopy(self.plan)
        if time.time() >= p["expires_at"] or trajectory.digest(plan) != p["digest"]:
            raise ValueError("Approval expired or plan changed")
        backend, mode = self.backend, self.mode
        measured = backend.joints()
        trajectory.require_start(plan, measured)
        trajectory.validate(plan, p["arm"], self.ik.model, self._obstacles())
        if self.cancelled.is_set():
            raise ValueError("Motion stopped")
        self.event("executing", "Executing approved "+p["name"]+"." if self.mode == "live" else "Applying approved motion in simulation.")
        if p["source"] == "visual" and mode == "live" and not self.cameras.get("head").status()["online"]:
            raise ValueError("Head camera became unavailable. Gather fresh context before execution.")
        if mode == "live":
            result = backend.stream_plan(plan, p["arm"])
            after = result["joints"]
            expected = plan["keyframes"][-1]["joint_targets_rad"]
            if any(abs(after[n]-v) > .12 for n, v in expected.items()):
                raise ValueError("Motion ended with a tracking error. Inspect the robot before continuing.")
        else:
            backend.q.update(plan["keyframes"][-1]["joint_targets_rad"])
            after = backend.joints()
        if self.cancelled.is_set():
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / (p["id"]+".json")).write_text(json.dumps(
            {"proposal": p, "plan": plan, "measured_after": after, "decision": self.decision}, allow_nan=False)+"\n")
        with self.lock:
            self.proposal = self.plan = self.paths = None
        self.event("completed", "Motion completed. "+("Measured robot feedback recorded." if self.mode == "live" else "Simulation only."))

    def _fallback(self, task, failure):
        self.event("context", "Primary trajectory failed. Gathering camera context for the visual harness.")
        from core.visual_policy import DashboardPerception, make_visual
        per = DashboardPerception(self.cameras, self.cfg["robot"]["arm"])
        packet = per.capture()
        if not packet.images:
            self.event("needs_context", "No fresh head camera frame. Connect the camera, then choose Retry with cameras.")
            return
        self.visual = self.visual_factory(self.provider) if self.visual_factory else make_visual(self.provider, self.cfg)
        from harness.loop import Episode
        from harness.recorder import Recorder
        ex = ReviewedExecutor(self, self.cfg["robot"]["arm"])
        rec = Recorder(self.cfg, self.mode, task, root=self.run_dir)
        from harness.feedback import FeedbackStore
        feedback = FeedbackStore(self.cfg["feedback"]["path"], self.run_dir.name, self.cfg["feedback"]["max_in_prompt"])
        episode = Episode(self.cfg, self.visual, ex, per, rec, feedback=feedback,
                          log=lambda message: self.event("context", message))
        result = episode.run(task+"\nThe full-path planner rejected the request: "+failure+
                             "\nUse fresh visual context and small non-contact actions; do not repeat the rejected trajectory.")
        if self.visual and hasattr(self.visual, "close"):
            self.visual.close()
        if not self.cancelled.is_set():
            self.event("completed" if result.get("success") else "blocked",
                       "Visual policy finished; "+str(result.get("reason", "")))

    def connect(self, table_z=None):
        if self.backend_factory:
            backend = self.backend_factory()
        else:
            from harness.robot.arm_client import ArmClientBackend
            try:
                backend = ArmClientBackend(self.cfg)
            except ConnectionRefusedError:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                self.streamer_log = (self.run_dir / "streamer.log").open("a")
                # Explicit browser connection starts only the existing arm bridge, never a shell.
                config_file = self.run_dir / "harness-config.yaml"
                import yaml
                config_file.write_text(yaml.safe_dump(self.cfg))
                self.streamer = subprocess.Popen([sys.executable, "-m", "harness.robot.arm_stream", self.iface,
                                                  "--config", str(config_file)], cwd=ROOT,
                                                 stdin=subprocess.DEVNULL, stdout=self.streamer_log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                backend = None
                for _ in range(40):
                    if self.cancelled.wait(.25) or self.streamer.poll() is not None:
                        break
                    try:
                        backend = ArmClientBackend(self.cfg)
                        break
                    except (OSError, RuntimeError):
                        continue
                if backend is None:
                    raise ValueError("Robot bridge could not connect. Check the interface and SDK; see the session streamer log.")
        try:
            self.pending_backend = backend
            if self.cancelled.is_set():
                raise ValueError("Connection cancelled")
            backend.engage()
            if self.cancelled.is_set():
                backend.release()
                raise ValueError("Connection cancelled")
            backend.joints()
        except Exception:
            backend.close()
            self.pending_backend = None
            raise
        with self.lock:
            if self.cancelled.is_set():
                backend.close()
                self.pending_backend = None
                raise ValueError("Connection cancelled")
            self.backend, self.mode, self.connected = backend, "live", True
            self.pending_backend = None
        self.event("idle", "Robot connected; arms held at their current pose. New motions require approval.")

    def command(self, command):
        self.operator_seen()
        action = command.get("action")
        if action == "heartbeat":
            return {"ok": True}
        if action in ("stop", "release"):
            self.stop()
            return self.status()
        if action == "decision":
            return self.decide(command.get("id"), command.get("digest"), command.get("decision"), command.get("note", ""))
        if action == "settings":
            with self.lock:
                if self.worker and self.worker.is_alive() or self.proposal:
                    raise ValueError("Finish or stop the current task before changing settings")
                if command.get("provider", self.provider) not in ("codex", "claude", "openai", "anthropic"):
                    raise ValueError("Unknown visual provider")
                self.provider = command.get("provider", self.provider)
                if "auto_fallback" in command:
                    if type(command["auto_fallback"]) is not bool: raise ValueError("Invalid fallback setting")
                    self.auto_fallback = command["auto_fallback"]
                if command.get("arm", self.cfg["robot"]["arm"]) not in ("left", "right"):
                    raise ValueError("Choose left or right arm")
                self.cfg["robot"]["arm"] = command.get("arm", self.cfg["robot"]["arm"])
            return self.status()
        if action not in ("connect", "fallback", "jog", "home", "roll"):
            raise ValueError("Unknown robot action")
        if self.worker and self.worker.is_alive() or self.planner.status()["state"] == "planning":
            raise ValueError("A task is already running; stop it first")
        self.before_submit()
        if action == "connect":
            if self.connected:
                raise ValueError("Robot already connected")
            z = command.get("table_z_m", self.cfg["workspace"]["table_z_m"])
            if type(z) not in (float, int) or not math.isfinite(z) or not 0 <= z <= 1.2:
                raise ValueError("Enter the measured table height in robot-base metres (0–1.2).")
            self.cfg["workspace"]["table_z_m"] = float(z)
            self.event("connecting", "Connecting and taking arm control at the current pose.")
            self._start(self.connect)
        elif action == "fallback":
            task = command.get("prompt") or self.planner.status().get("prompt")
            if not isinstance(task, str) or not 1 <= len(task) <= 1000:
                raise ValueError("Describe the task first")
            self._start(lambda: self._fallback(task, self.planner.status().get("message", "")))
        elif action in ("jog", "home", "roll"):
            arm = command.get("arm", self.cfg["robot"]["arm"])
            if arm not in trajectory.ARM_JOINTS:
                raise ValueError("Choose left or right arm")
            if action == "roll":
                sign = command.get("sign")
                if type(sign) is not int or sign not in (-1, 1):
                    raise ValueError("Choose clockwise or counterclockwise")
                def roll():
                    pose = self.planning_pose()
                    q = [pose[n] for n in trajectory.ARM_JOINTS[arm]]
                    q[4] += sign*math.radians(5)
                    ReviewedExecutor(self, arm).go_to_joints(q, "Roll wrist "+("+" if sign>0 else "−")+"5°")
                self._start(roll)
            elif action == "home":
                self._start(lambda: ReviewedExecutor(self, arm).go_to_joints(self.cfg["robot"]["start_pose_rad"][arm]))
            else:
                direction = command.get("direction")
                deltas = {"forward": [.02, 0, 0], "back": [-.02, 0, 0], "left": [0, .02, 0],
                          "right": [0, -.02, 0], "up": [0, 0, .02], "down": [0, 0, -.02]}
                if direction not in deltas:
                    raise ValueError("Choose a nudge direction")
                def jog():
                    pose = self.planning_pose()
                    tip, _ = self.ik.fk(arm, pose, pose)
                    draft = {"name": "Nudge "+direction, "arm": arm, "frame": "robot_base",
                             "waypoints": [{"position_m": (tip+deltas[direction]).tolist(), "hold_s": 0}],
                             "return_to_start": False}
                    self.review_and_run(compile_trajectory(self.ik, draft, pose), arm, "manual", pose)
                self._start(jog)
        else:
            raise ValueError("Unknown robot action")
        return self.status()

    def stop(self):
        self.cancelled.set()
        self.planner.cancel()
        self.decision_event.set()
        with self.lock:
            self.generation += 1
            self.proposal = self.plan = self.paths = None
        if self.pending_backend:
            self.pending_backend.close()
            self.pending_backend = None
        if self.visual and hasattr(self.visual, "close"):
            self.visual.close()
        self.sim.control({"action": "stop"})
        if self.mode == "live":
            try:
                self.backend.freeze()
                self.backend.release()
            except (OSError, RuntimeError):
                pass  # a disconnected streamer releases independently
            finally:
                self.backend.close()
                self.connected = False
                self.mode = "sim"
                self.backend = PreviewBackend(self._simulation_pose())
        self.event("stopped", "Stopped. Robot arm control released; pending approval cancelled.")

    def close(self):
        self.closed.set()
        self.stop()
        self.watchdog.join(2)
        if self.streamer and self.streamer.poll() is None:
            self.streamer.terminate()
            try: self.streamer.wait(timeout=4)
            except subprocess.TimeoutExpired: self.streamer.kill()
        if self.streamer_log:
            self.streamer_log.close()

    def glasses_message(self):
        with self.lock:
            p = self.proposal
            if not p or not self.paths or self.state not in ("review", "approved", "executing"):
                return {"type": "trajectory", "version": 1, "id": "idle", "frame": "robot_base",
                        "units": "m", "clear": True, "hands": {"left": [], "right": []}}
            return {"type": "trajectory", "version": 1, "id": p["id"], "frame": "robot_base",
                    "units": "m", "duration_s": p["duration_s"], "hands": copy.deepcopy(self.paths),
                    "review": {"id": p["id"], "digest": p["digest"], "revision": p["revision"],
                               "text": p["name"], "mode": p["mode"]}
                    if self.state == "review" and time.time() < p["expires_at"] else None}
