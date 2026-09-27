"""One draft/review/execution authority for dashboard, model tools and glasses.

Planning never engages actuators. Only an immutable, human-approved complete motion
is forwarded over the private control transport. Visual reasoning is part of the
agent's planning loop, not a second physical step-by-step executor.
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
import tempfile
import threading
import time
import uuid

import numpy as np
from contract.runtime import digest, validate_motion, validate_approval
from core import trajectory
from core.generated_motion import TrajectoryRejected, compile_trajectory, validate_trajectory
from core.experience import ExperienceMemory
from core.ik import ArmIK
from core.motion_policy import base_path, check_waypoints, table_obstacles, walking_payload
from harness.config import load

ROOT = Path(__file__).resolve().parents[1]


class PreviewBackend:
    name = "mock"
    dry_run = True
    alive = True

    def __init__(self, pose):
        self.q, self.hands = dict(pose), {"left": False, "right": False}

    def joints(self): return dict(self.q)
    def velocities(self): return {n: 0. for n in self.q}
    def snapshot(self): return {"joints": self.joints(), "velocities": self.velocities(), "lowstate_age_s": 0., "engaged": False}
    def hand_state(self, arm): return self.hands[arm]
    def freeze(self): pass
    def release(self): pass
    def close(self): pass


class RobotPipeline:
    REVIEW_SECONDS = 120
    DRAFT_SECONDS = 300
    OBSERVATION_SECONDS = 120
    OPERATOR_TIMEOUT = 10.

    def __init__(self, planner, simulation, cameras, iface="en6", cfg=None, run_dir=None,
                 backend_factory=None, visual_factory=None, simulation_only=False):
        self.planner, self.sim, self.cameras, self.iface = planner, simulation, cameras, iface
        self.cfg = copy.deepcopy(cfg or load())
        self.ik = ArmIK(backend="mujoco")
        self.simulation_only = simulation_only
        self.lock, self.log_lock = threading.RLock(), threading.Lock()
        self.cancelled, self.closed = threading.Event(), threading.Event()
        self.generation, self.revision = 0, 0
        self.session_id = uuid.uuid4().hex
        self.experience = ExperienceMemory(self.cfg, root=Path(run_dir) if run_dir else ROOT)
        self.experience_snapshot = None
        self.worker = self.streamer = self.streamer_log = self.pending_backend = None
        self.hand_server = self.hand_log = None
        self.backend_factory = backend_factory
        self.mode, self.connected = "sim", False
        self.backend = PreviewBackend(self._simulation_pose())
        self.robot_state = self.backend.snapshot()
        self.drafts, self.requests, self.results, self.observations = {}, {}, {}, {}
        self.draft = self.proposal = self.plan = self.paths = self.decision = None
        self.primary_id = None
        self.primary_generation = None
        self.last_result = None
        self.on_result = None
        self.on_stop = None
        self.firmware_pending = None
        self.on_planning_blocked = None
        self.state, self.message = "idle", "Describe a motion. Planning and preview are automatic; Accept runs the complete motion."
        self.events = []
        self.run_dir = Path(run_dir or ROOT / "runs/dashboard") / self.session_id
        self.control_dir = None
        self.glasses = {"connected": 0, "error": "", "port": None}
        self.last_operator = time.monotonic()
        self.walked_m = self.turned_rad = 0.
        self.planner.before_submit = self.before_submit
        self.planner.on_complete = self.planned
        self.planner.validate_candidate = self.validate_candidate
        self.planner.preview_pose = self.planning_pose
        self.planner.pose_label = lambda: "measured robot joints" if self.mode == "live" else "approved simulation state"
        self.watchdog = threading.Thread(target=self._watchdog, daemon=True)
        self.watchdog.start()

    def _simulation_pose(self):
        with self.sim.lock:
            return {self.sim.model.joint(i).name: float(self.sim.data.qpos[self.sim.model.jnt_qposadr[i]])
                    for i in range(self.sim.model.njnt)}

    def operator_seen(self):
        self.last_operator = time.monotonic()

    def _watchdog(self):
        while not self.closed.wait(.25):
            if self.connected and (time.monotonic()-self.last_operator > self.OPERATOR_TIMEOUT or not self.backend.alive):
                self.stop("Operator or robot connection lost; control released.")
            elif self.connected:
                try: self._update_robot_state(self.backend.snapshot())
                except (OSError, RuntimeError): self.stop("Robot telemetry unavailable; control released.")
            with self.lock:
                if self.proposal and self.state == "review" and time.time() >= self.proposal["expires_at"]:
                    self._finish("expired", "Approval expired. Observe and prepare a new proposal.")

    def planning_pose(self):
        state = self.backend.snapshot()
        if self.mode == "live" and state.get("lowstate_age_s", 999) > .5:
            raise ValueError("Robot telemetry is stale")
        self._update_robot_state(state)
        pose = dict(state["joints"])
        # Context and compilation use measured telemetry. Desired held targets
        # remain separately visible in robot_state and must not masquerade as pose.
        known = {self.ik.model.joint(i).name for i in range(self.ik.model.njnt)}
        return {n: float(q) for n, q in pose.items() if n in known}

    def _update_robot_state(self, state):
        self.robot_state = copy.deepcopy(state)
        # The streamer reserves the entire requested displacement, including an
        # interrupted walk. Reconnecting must not reset that conservative budget.
        self.walked_m = max(self.walked_m, float(state.get("walked_m", 0.)))
        self.turned_rad = max(self.turned_rad, float(state.get("turned_rad", 0.)))

    def capabilities(self):
        hands = self.cfg["hand"]["type"] in ("revo2", "virtual")
        return {"walking": bool(self.cfg.get("locomotion", {}).get("enabled")),
                "walking_note": self.cfg.get("locomotion", {}).get("note", ""),
                "hands": {"left": hands, "right": hands}, "hardware_allowed": not self.simulation_only,
                "hand_type": self.cfg["hand"]["type"], "arm_motion": True,
                "depth": False, "contact_planning": False, "dual_arm": False}

    def status(self):
        with self.lock:
            return copy.deepcopy({"state": self.state, "message": self.message, "mode": self.mode,
                "connected": self.connected, "busy": bool(self.worker and self.worker.is_alive()),
                "draft": self.draft, "proposal": self.proposal, "last_result": self.last_result,
                "robot_state": self.robot_state, "capabilities": self.capabilities(),
                "events": self.events[-20:], "iface": self.iface, "table_z_m": self.cfg["workspace"]["table_z_m"],
                "glasses": self.glasses, "experience": self.experience.status(),
                "run_id": self.session_id, "session_id": self.session_id})

    def event(self, stage, message):
        with self.lock:
            self.state, self.message = stage, str(message)[:800]
            entry = {"stage": stage, "message": self.message, "at": time.time(), "session_id": self.session_id}
            self.events.append(entry); self.events = self.events[-100:]
        self._log(entry)

    def _log(self, entry):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.log_lock, (self.run_dir / "events.jsonl").open("a") as file:
            file.write(json.dumps(entry, allow_nan=False)+"\n")

    def _policy_digest(self):
        return digest({k: self.cfg[k] for k in ("workspace", "robot", "limits", "locomotion", "hand")})

    def _available(self):
        if self.proposal or (self.worker and self.worker.is_alive() and self.worker is not threading.current_thread()):
            raise ValueError("Finish, decline or stop the current motion first")
        if self.cancelled.is_set():
            raise ValueError("Task stopped. Start a new planning request")

    def reset_planning(self, cancelled=False):
        """Supersede unsent agent work without disturbing a submitted human review."""
        job_id = self.planner.status().get("id")
        with self.lock:
            if self.proposal or (self.worker and self.worker.is_alive()):
                return
            self.generation += 1
            self.drafts.clear()
            self.draft = self.paths = self.primary_id = None
            if cancelled:
                self.cancelled.set()
                self.state, self.message = "cancelled", "Agent planning cancelled. Start a new request to continue."
            else:
                self.cancelled.clear()
                self.state, self.message = "idle", "Agent planning started. No motion has been submitted for review."
        if job_id is not None:
            self.planner.cancel(job_id=job_id)

    def supersede_review(self, proposal_id):
        """A new motion request can replace a pending review, never an accepted run."""
        with self.lock:
            if not self.proposal:
                return
            if (self.proposal["id"] != proposal_id or self.state != "review" or self.decision
                    or (self.worker and self.worker.is_alive())):
                raise ValueError("The current motion was already accepted or changed; wait for it or use Stop")
            self._finish("cancelled", "Pending proposal replaced by a new motion request.")
            self.sim.control({"action": "stop"})
        self.reset_planning()

    def _planning_generation(self, generation=None):
        self._available()
        if generation is not None and generation != self.generation:
            raise ValueError("Planning request was superseded")
        return self.generation

    def before_submit(self):
        with self.lock:
            if self.proposal or (self.worker and self.worker.is_alive()):
                raise ValueError("Finish, decline or stop the current task first")
            self.generation += 1
            self.primary_generation = self.generation
            self.cancelled.clear()
            self.draft = self.paths = None
            self.state, self.message = "planning", "Planning and validating the motion for automatic preview."
            return self.generation

    def validate_candidate(self, plan, draft, pose, generation=None):
        """Apply execution workspace policy inside the bounded path-revision loop."""
        with self.lock:
            self._planning_generation(generation)
        try:
            check_waypoints(self.cfg, draft["waypoints"], self.mode == "live")
            resolved = trajectory.resolve(plan, draft["arm"], pose)
            trajectory.validate(resolved, draft["arm"], self.ik.model, table_obstacles(self.cfg, self.mode == "live"))
        except ValueError as exc:
            raise TrajectoryRejected(str(exc), {"stage": "workspace_policy", "workspace": copy.deepcopy(self.cfg["workspace"])}) from exc

    def _planning_blocked(self, status, message, generation):
        with self.lock:
            if generation != self.generation or self.cancelled.is_set():
                return
            self.event("blocked", message)
            callback = self.on_planning_blocked
            report = {**copy.deepcopy(status), "state": "blocked", "stage": "blocked", "message": message}
        if callback:
            callback(report)

    def planned(self, status, plan):
        with self.lock:
            generation = status.get("pipeline_generation", self.primary_generation)
            if (self.cancelled.is_set() or status["state"] == "cancelled"
                    or generation != self.generation
                    or "prompt:" + str(status.get("id")) in self.requests):
                return
            self.primary_id = status.get("id")
        try:
            if plan is None:
                self._planning_blocked(status, status["message"], generation)
                return
            pose = self.planning_pose()
            if plan.get("generated_trajectory"):
                draft = plan["generated_trajectory"]
                check_waypoints(self.cfg, draft["waypoints"], self.mode == "live")
                plan = compile_trajectory(self.ik, draft, pose)
            elif self.mode == "live" and status.get("source") == "demo":
                raise ValueError("Demonstration fixtures cannot authorize physical motion")
            draft = self.prepare_arm(plan, status["target"]["arm"], pose=pose, generation=generation)
            self.propose_motion(draft["id"], "prompt:" + str(status.get("id")), task=status.get("prompt"))
        except (ValueError, RuntimeError) as exc:
            self._planning_blocked(status, f"Motion planning blocked: {exc}", generation)

    def _start(self, work):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("A task is already running")
            def run():
                try:
                    work()
                except Exception as exc:
                    if not self.cancelled.is_set():
                        if self.proposal: self._finish("failed", str(exc))
                        else: self.event("blocked", str(exc))
            self.worker = threading.Thread(target=run, daemon=True)
            self.worker.start()

    def register_observation(self, metadata):
        with self.lock:
            self.observations[metadata["id"]] = copy.deepcopy(metadata)
            while len(self.observations) > 32: self.observations.pop(next(iter(self.observations)))

    def _check_observation(self, observation_id):
        if not observation_id:
            return None
        item = self.observations.get(observation_id)
        if not item or time.time()-item["observed_at"] > self.OBSERVATION_SECONDS:
            raise ValueError("Observation expired; observe again and regenerate the motion")
        if self.mode == "live":
            for name in item["cameras"]:
                feed = self.cameras.get(name)
                if not feed or not feed.status()["online"]:
                    raise ValueError(f"{name} camera became stale; observe again")
        return copy.deepcopy(item)

    def compile_hand_path(self, draft, observation_id=None, *, generation=None):
        with self.lock: generation = self._planning_generation(generation)
        draft = validate_trajectory(draft)
        pose = self.planning_pose()
        check_waypoints(self.cfg, draft["waypoints"], self.mode == "live")
        plan = compile_trajectory(self.ik, draft, pose)
        return self.prepare_arm(plan, draft["arm"], pose, observation_id, generation=generation)

    def prepare_arm(self, plan, arm, pose=None, observation_id=None, *, generation=None):
        with self.lock: generation = self._planning_generation(generation)
        pose = pose or self.planning_pose()
        resolved = trajectory.resolve(plan, arm, pose)
        report = trajectory.validate(resolved, arm, self.ik.model, table_obstacles(self.cfg, self.mode == "live"))
        report["coverage"] = "Robot geometry and measured table; no depth obstacle map" if self.mode == "live" else "Robot geometry in simulation; physical scene is not measured"
        resolved["preview_only"] = True
        return self._store_draft({"kind": "arm", "arm": arm, "plan": resolved}, resolved.get("name", "Arm motion"), report, pose, observation_id, generation)

    def prepare_walk(self, dx, dy, dyaw, *, generation=None):
        with self.lock: generation = self._planning_generation(generation)
        pose = self.planning_pose()
        payload = walking_payload(self.cfg, dx, dy, dyaw)
        if self.walked_m+math.hypot(dx, dy) > self.cfg["locomotion"]["max_total_m"] or self.turned_rad+abs(dyaw) > math.radians(self.cfg["locomotion"]["max_total_turn_deg"]):
            raise ValueError("Session walking budget exhausted")
        return self._store_draft(payload, f"Move base {dx:.2f} m forward, {dy:.2f} m left, turn {math.degrees(dyaw):.0f}°",
            {"coverage": "Bounded velocity/time and controller FSM; no obstacle or balance model", "checks": ["distance", "speed", "duration", "session budget"]}, pose, generation=generation)

    def prepare_hand(self, arm, closed, *, generation=None):
        with self.lock: generation = self._planning_generation(generation)
        if not self.capabilities()["hands"].get(arm):
            raise ValueError("No hand configured for this arm")
        if self.mode == "live" and self.cfg["hand"]["type"] != "revo2":
            raise ValueError("Virtual hands cannot command the robot")
        return self._store_draft({"kind": "hand", "arm": arm, "closed": closed}, ("Close" if closed else "Open")+f" {arm} hand",
            {"coverage": "Configured Revo2 pose and fresh hand telemetry; contact/force not modeled", "checks": ["hand capability", "command shape"]}, self.planning_pose(), generation=generation)

    def _store_draft(self, payload, name, report, pose, observation_id=None, generation=None):
        validate_motion(payload)
        observation = self._check_observation(observation_id)
        with self.lock:
            self._available()
            if generation is not None and generation != self.generation:
                raise ValueError("Planning request was superseded")
            ident = uuid.uuid4().hex
            duration = payload["plan"]["duration_s"] if payload["kind"] == "arm" else payload.get("duration_s", self.cfg["hand"]["pause_s"])
            public = {"id": ident, "plan_id": ident, "state": "draft", "name": name, "kind": payload["kind"],
                "arm": payload.get("arm"), "duration_s": duration, "digest": digest(payload), "validation": report,
                "mode": self.mode, "expires_at": time.time()+self.DRAFT_SECONDS, "observation_id": observation_id}
            self.drafts[ident] = {"public": public, "payload": copy.deepcopy(payload), "pose": dict(pose),
                "policy": self._policy_digest(), "observation": observation, "submitted": None, "generation": self.generation}
            while len(self.drafts)>32: self.drafts.pop(next(iter(self.drafts)))
            self.draft = copy.deepcopy(public)
            self.paths = None
            self.event("draft", "Path validated. Finishing the complete proposal for automatic preview.")
            return copy.deepcopy(public)

    def _get_draft(self, plan_id):
        item = self.drafts.get(plan_id)
        if not item or time.time() >= item["public"]["expires_at"] or item["generation"] != self.generation:
            raise ValueError("Draft expired or is no longer available")
        if item["public"]["mode"] != self.mode or item["policy"] != self._policy_digest():
            raise ValueError("Robot mode or policy changed; regenerate the draft")
        if digest(item["payload"]) != item["public"]["digest"]:
            raise ValueError("Draft content changed; regenerate it")
        return item

    def _display(self, item, display_id):
        payload, pose = item["payload"], item["pose"]
        if payload["kind"] == "arm":
            preview = copy.deepcopy(payload["plan"])
        else:
            duration = item["public"]["duration_s"]
            preview = {"schema_version": 1, "name": item["public"]["name"], "duration_s": duration,
                "held_joints_rad": pose, "keyframes": [{"time_s": t, "joint_targets_rad": {"right_elbow_joint": pose["right_elbow_joint"]}} for t in (0., duration)]}
            if payload["kind"] == "walk":
                preview["base_keyframes"] = [{"time_s":p["time_s"],"x_m":p["position_m"][0],"y_m":p["position_m"][1],"yaw_rad":p["yaw_rad"]} for p in base_path(payload)]
        preview["preview_only"] = True
        preview["prompt_proposal"] = {"id": display_id, "source": "draft", "kind": payload["kind"]}
        preview["scene_boxes"] = table_obstacles(self.cfg, self.mode == "live") if payload["kind"] == "arm" else []
        from spectacles.plan_feed import hand_paths
        self.paths = hand_paths(self.ik.model, preview, 120)
        self.sim.show_proposal(preview, display_id)

    def preview_plan(self, plan_id):
        with self.lock:
            self._available()
            item = self._get_draft(plan_id)
            self.draft = copy.deepcopy(item["public"])
            self._display(item, uuid.uuid4().hex)
            self.event("draft", "Draft preview. No approval is active and nothing can execute.")
        return {"state": "previewed", "plan_id": plan_id, "requires_operator_approval": True}

    def show_primary(self, source, proposal_id):
        if proposal_id != self.primary_id or not self.draft:
            raise ValueError("Draft is no longer available")
        return self.preview_plan(self.draft["id"])

    def propose_motion(self, plan_id, request_id=None, *, task=None):
        request_id = request_id or plan_id
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("Use a bounded request_id")
        with self.lock:
            if request_id in self.requests:
                prior = self.requests[request_id]
                if prior["plan_id"] != plan_id: raise ValueError("request_id already identifies a different motion")
                return self.motion_result(prior["proposal_id"])
            self._available()
            item = self._get_draft(plan_id)
            if item["submitted"]:
                return self.motion_result(item["submitted"])
            self._check_observation(item["public"]["observation_id"])
            pose = self.planning_pose()
            trajectory.require_start({"keyframes": [{"joint_targets_rad": item["pose"]}]}, pose)
            ident = uuid.uuid4().hex
            self.revision += 1
            p = {**copy.deepcopy(item["public"]), "id": ident, "session_id": self.session_id,
                "revision": self.revision, "state": "review", "expires_at": min(item["public"]["expires_at"], time.time()+self.REVIEW_SECONDS),
                "source": "agent"}
            self.proposal, self.plan, self.decision = p, copy.deepcopy(item["payload"]), None
            self.requests[request_id] = {"plan_id": plan_id, "proposal_id": ident}
            item["submitted"] = ident
            self._display(item, ident)
            try:
                self.experience_snapshot = self.experience.capture(task, p, self.plan, item["pose"], self.paths, item["observation"])
            except Exception as exc:
                self.experience_snapshot = None
                self.experience.error = str(exc)[:180]
            self.event("review", "Previewing the complete motion. Accept once in the dashboard or pinch in the glasses to run it.")
            self._log({"type": "proposal", "proposal": p, "payload": self.plan})
            return self.motion_result(ident)

    def motion_result(self, proposal_id):
        with self.lock:
            if proposal_id in self.results: return copy.deepcopy(self.results[proposal_id])
            if self.proposal and self.proposal["id"] == proposal_id:
                return {"proposal_id": proposal_id, "state": self.state, "outcome": None, "proposal": copy.deepcopy(self.proposal)}
            raise ValueError("Unknown proposal in this session")

    def decide(self, proposal_id, payload_digest, decision, note=""):
        self.operator_seen()
        with self.lock:
            p = self.proposal
            if decision not in ("approve", "decline"): raise ValueError("Choose approve or decline")
            if not p or self.state != "review" or p["id"] != proposal_id or p["digest"] != payload_digest or self.decision:
                raise ValueError("Proposal changed or was already decided")
            if self.cancelled.is_set() or time.time() >= p["expires_at"]:
                raise ValueError("Proposal expired or cancelled")
            self.decision = {"decision": decision, "note": str(note)[:500], "at": time.time()}
            if decision == "decline":
                self._finish("declined", "Proposal declined. "+str(note)[:500])
            else:
                self.event("approved", "Complete motion accepted in "+p["mode"]+" mode.")
                self._start(self._execute)
        return self.status()

    def _execute(self):
        with self.lock:
            firmware = self.proposal and self.proposal.get("kind") == "firmware"
        if firmware:
            return self._execute_firmware()
        with self.lock:
            if not self.proposal or not self.decision or self.decision["decision"] != "approve":
                raise ValueError("No human approval")
            p, payload = copy.deepcopy(self.proposal), copy.deepcopy(self.plan)
            item = self._get_draft(p["plan_id"])
            self._check_observation(p["observation_id"])
            approval = {"proposal_id": p["id"], "revision": p["revision"], "digest": p["digest"], "expires_at": p["expires_at"]}
            validate_approval(approval, digest(payload)); validate_motion(payload)
            backend, mode, generation = self.backend, self.mode, self.generation
        measured = backend.joints()
        trajectory.require_start({"keyframes": [{"joint_targets_rad": item["pose"]}]}, measured)
        if payload["kind"] == "arm":
            trajectory.validate(payload["plan"], payload["arm"], self.ik.model, table_obstacles(self.cfg, mode == "live"))
        with self.lock:
            if self.cancelled.is_set() or generation != self.generation:
                return
            self.event("executing", ("Executing " if mode == "live" else "Simulating ")+p["name"])
            if mode == "sim":
                # Acceptance runs the exact reviewed motion even when its automatic
                # preview has already finished. There is no replay control/API.
                self._display(item, p["id"]+":accepted")
        result = {}
        try:
            if payload["kind"] == "walk":
                self.walked_m += math.hypot(payload["vx"],payload["vy"])*payload["duration_s"]
                self.turned_rad += abs(payload["vyaw"])*payload["duration_s"]
            if mode == "live":
                result = backend.execute_motion(payload, approval)
                after = result.get("joints") or backend.joints()
            else:
                if payload["kind"] == "arm": backend.q.update(payload["plan"]["keyframes"][-1]["joint_targets_rad"])
                elif payload["kind"] == "hand": backend.hands[payload["arm"]] = payload["closed"]
                else: result["odom"] = {"dx": base_path(payload)[-1]["position_m"][0], "dy": base_path(payload)[-1]["position_m"][1], "dyaw": payload["vyaw"]*payload["duration_s"], "predicted": True}
                after = backend.joints()
            tracking = None
            if payload["kind"] == "arm":
                expected = payload["plan"]["keyframes"][-1]["joint_targets_rad"]
                tracking = max(abs(after[n]-q) for n,q in expected.items())
                if tracking > .12: raise ValueError("Motion ended with excessive tracking error")
            if not self.cancelled.is_set() and generation == self.generation:
                self.robot_state = {**self.robot_state, "joints": after}
                self._finish("executed", "Approved motion completed." if mode == "live" else "Approved motion completed in simulation.",
                    measured_end_pose=after, tracking_error=tracking, feedback=result)
        except Exception as exc:
            if mode == "live":
                try:
                    try: backend.freeze()
                    finally: backend.release()
                except (OSError, RuntimeError): pass
                finally:
                    try: backend.close()
                    except (OSError, RuntimeError): pass
                    finally:
                        with self.lock:
                            if self.backend is backend:
                                self.connected, self.mode = False, "sim"
                                self.backend = PreviewBackend(self.robot_state.get("joints", self._simulation_pose()))
            if not self.cancelled.is_set(): self._finish("failed", str(exc))

    def _finish(self, outcome, message, **feedback):
        with self.lock:
            p = self.proposal
            if not p: return
            result = {"proposal_id": p["id"], "plan_id": p["plan_id"], "outcome": outcome, "state": outcome,
                "mode": p["mode"], "name": p["name"], "digest": p["digest"], "at": time.time(),
                "message": message, "measured_end_pose": None, "tracking_error": None,
                "decision": copy.deepcopy(self.decision), **feedback}
            self.results[p["id"]] = result
            self.last_result = result
            try:
                self.experience.record(self.experience_snapshot, result)
            except Exception as exc:
                # Diagnostic storage must not change an approval or motion outcome.
                self.experience.error = str(exc)[:180]
            self.experience_snapshot = None
            self.proposal = self.plan = self.paths = self.draft = self.firmware_pending = None
            self._log({"type": "outcome", **result})
            self.event("completed" if outcome == "executed" else outcome, message)
            callback = self.on_result
        if callback:
            # Do not acquire a chat/UI lock while an outer coordinator lock is held.
            def notify():
                try: callback(copy.deepcopy(result))
                except Exception: pass
            threading.Thread(target=notify, daemon=True).start()

    def _private_config(self):
        """Materialize private transport configuration; never contains model keys."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.control_dir: self.control_dir = tempfile.TemporaryDirectory(prefix="reins-control-")
        token_file = self.cfg["streamer"].get("control_token_file")
        if not token_file:
            token_file = str(Path(self.control_dir.name)/"control.token")
            if not Path(token_file).exists():
                fd = os.open(token_file, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
                with os.fdopen(fd,"w") as f: f.write(secrets.token_urlsafe(32))
            self.cfg["streamer"]["control_token_file"] = token_file
        self.cfg["hand"]["revo2"].setdefault("control_token_file", token_file)
        import yaml
        config_file = Path(self.control_dir.name)/"config.yaml"
        config_file.write_text(yaml.safe_dump(json.loads(json.dumps(self.cfg))))
        return config_file, token_file

    def _connect_hands(self, backend):
        from harness.robot.hand_client import Revo2Client
        try:
            hand = Revo2Client(self.cfg, log=lambda _: None)
        except RuntimeError as exc:
            # Only an absent listener permits starting our own bridge. An existing
            # incompatible/private bridge is never replaced or treated as ready.
            if not isinstance(exc.__cause__, ConnectionRefusedError): raise
            config_file, _ = self._private_config()
            hand_cfg = self.cfg["hand"]["revo2"]
            if self.hand_log: self.hand_log.close()
            self.hand_log = (self.run_dir/"hands.log").open("a")
            self.hand_server = subprocess.Popen([sys.executable,"-m","harness.robot.revo2",hand_cfg.get("iface") or self.iface,
                "serve","--config",str(config_file),"--control-token-file",hand_cfg["control_token_file"]],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=self.hand_log,stderr=subprocess.STDOUT,start_new_session=True)
            hand = None
            for _ in range(40):
                if self.cancelled.wait(.25) or self.hand_server.poll() is not None: break
                try: hand = Revo2Client(self.cfg, log=lambda _: None); break
                except (OSError,RuntimeError): continue
            if hand is None: raise ValueError("Private hand bridge unavailable. Check hand interface and session hands.log")
        if not hand.authenticated or self.cancelled.is_set():
            hand.close()
            raise ValueError("Hand bridge is not authorized for this session, or connection was cancelled")
        backend.hands = hand

    def connect(self, table_z=None):
        if self.simulation_only: raise ValueError("Hardware is disabled in simulation-only mode")
        if self.backend_factory:
            backend = self.backend_factory()
        else:
            from harness.robot.arm_client import ArmClientBackend
            try:
                backend = ArmClientBackend(self.cfg)
            except ConnectionRefusedError:
                config_file, token_file = self._private_config()
                if self.streamer_log: self.streamer_log.close()
                self.streamer_log = (self.run_dir/"streamer.log").open("a")
                self.streamer = subprocess.Popen([sys.executable,"-m","harness.robot.arm_stream",self.iface,"--config",str(config_file),"--control-token-file",str(token_file)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=self.streamer_log, stderr=subprocess.STDOUT, start_new_session=True)
                backend = None
                for _ in range(40):
                    if self.cancelled.wait(.25) or self.streamer.poll() is not None: break
                    try: backend = ArmClientBackend(self.cfg); break
                    except (OSError, RuntimeError): continue
                if backend is None: raise ValueError("Private robot bridge unavailable. Check interface, SDK and session streamer.log")
        self.pending_backend = backend
        try:
            state = backend.snapshot()  # Deliberately no engage: telemetry only.
            if not getattr(backend,"authenticated",False):
                raise ValueError("This bridge belongs to another session. Configure its private capability or start it through this dashboard.")
            if not self.backend_factory and self.cfg["hand"]["type"] == "revo2":
                self._connect_hands(backend)
            with self.lock:
                if self.cancelled.is_set(): raise ValueError("Connection cancelled")
                self.backend, self.mode, self.connected = backend, "live", True
                self._update_robot_state(state)
                self.pending_backend = None
                self.drafts.clear(); self.draft = self.paths = None
            self.event("idle", "Robot telemetry connected. Actuators remain unchanged until a motion is approved.")
        except Exception:
            backend.close(); self.pending_backend = None
            raise

    def command(self, command):
        self.operator_seen()
        action = command.get("action")
        if action == "heartbeat": return {"ok": True}
        if action in ("stop", "release"): self.stop(); return self.status()
        if action == "decision": return self.decide(command.get("id"), command.get("digest"), command.get("decision"), command.get("note", ""))
        if action == "propose":
            self.propose_motion(command.get("plan_id"), command.get("request_id")); return self.status()
        if action == "preview":
            self.preview_plan(command.get("plan_id")); return self.status()
        if action == "settings":
            with self.lock:
                self._available()
                if "arm" in command:
                    if command["arm"] not in ("left","right"): raise ValueError("Choose one arm")
                    self.cfg["robot"]["arm"] = command["arm"]
                    self.draft = self.paths = None
                    self.drafts.clear()
            return self.status()
        if action not in ("connect","jog","roll","home","walk","hand"):
            raise ValueError("Unknown robot action")
        if self.planner.status()["state"] == "planning": raise ValueError("Wait for the current draft")
        self.before_submit()
        if action == "connect":
            if self.connected: raise ValueError("Robot already connected")
            z = command.get("table_z_m", self.cfg["workspace"]["table_z_m"])
            if type(z) not in (int,float) or not math.isfinite(z) or not 0 <= z <= 1.2:
                raise ValueError("Enter measured table height in robot-base metres (0–1.2)")
            self.cfg["workspace"]["table_z_m"] = float(z)
            self._start(lambda: self.connect(z))
        else:
            arm = command.get("arm", self.cfg["robot"]["arm"])
            if arm not in ("left","right"): raise ValueError("Choose one arm")
            def prepare():
                if action == "walk": draft = self.prepare_walk(command.get("dx",0),command.get("dy",0),command.get("dyaw",0))
                elif action == "hand": draft = self.prepare_hand(arm,command.get("closed"))
                elif action == "jog":
                    deltas = {"forward":[.02,0,0],"back":[-.02,0,0],"left":[0,.02,0],"right":[0,-.02,0],"up":[0,0,.02],"down":[0,0,-.02]}
                    direction = command.get("direction")
                    if direction not in deltas: raise ValueError("Choose a nudge direction")
                    pose = self.planning_pose(); tip,_ = self.ik.fk(arm,pose,pose)
                    draft = self.compile_hand_path({"name":"Nudge "+direction,"arm":arm,"frame":"robot_base",
                        "waypoints":[{"position_m":(tip+deltas[direction]).tolist(),"hold_s":0}],"return_to_start":False})
                else:
                    pose = self.planning_pose(); q = np.array([pose[n] for n in trajectory.ARM_JOINTS[arm]])
                    target = np.array(self.cfg["robot"]["start_pose_rad"][arm]) if action == "home" else q.copy()
                    if action == "roll":
                        sign = command.get("sign")
                        if type(sign) not in (int,float) or sign not in (-1,1): raise ValueError("Choose a wrist-roll direction")
                        target[4] += math.radians(5)*sign
                    duration = max(1.,float(np.max(abs(target-q)))/.25*math.pi/2)
                    times = np.linspace(0,1,math.ceil(duration*50)+1)[1:]
                    frames = [q+(target-q)*(.5-.5*math.cos(math.pi*t)) for t in times]
                    plan = trajectory.frame_plan(arm,q,frames,.02,pose,"Home pose" if action == "home" else "Wrist roll")
                    draft = self.prepare_arm(plan,arm,pose)
                self.propose_motion(draft["id"])
            self._start(prepare)
        return self.status()

    def firmware(self, command, gestures):
        with self.lock:
            if self.simulation_only: raise ValueError("Firmware is disabled in simulation-only mode")
            if self.connected or self.proposal or self.state == "planning" or (self.worker and self.worker.is_alive()):
                raise ValueError("Release trajectory control and finish planning before using firmware presets")
            if command.get("action") == "refresh":
                return gestures.command(command)  # Read-only discovery needs no approval.
            if command.get("action") != "gesture":
                raise ValueError("Unknown gesture command")
            state = gestures.status()
            action_id = command.get("id")
            preset = next((a for a in state["actions"] if type(action_id) is int and a["id"] == action_id), None)
            if not state["connected"] or state["busy"] or preset is None:
                raise ValueError("Connect and choose an available R1 gesture")
            # Opaque firmware presets are human-only. They never enter the model's
            # validated trajectory transport or masquerade as a kinematic preview.
            self.generation += 1
            self.cancelled.clear()
            self.drafts.clear()
            self.draft = None
            self.revision += 1
            ident = uuid.uuid4().hex
            payload = {"kind": "firmware", "action_id": action_id,
                       "name": preset["name"], "iface": state["iface"]}
            self.proposal = {"id": ident, "plan_id": ident, "session_id": self.session_id,
                "revision": self.revision, "state": "review", "kind": "firmware", "mode": "live",
                "name": preset["label"], "digest": digest(payload), "expires_at": time.time()+self.REVIEW_SECONDS,
                "source": "operator", "duration_s": None, "observation_id": None,
                "description": "Onboard preset; its path and duration are unavailable for simulation preview.",
                "validation": {"coverage": "Firmware preset identity only; onboard path is opaque"}}
            self.plan, self.decision = payload, None
            self.firmware_pending = {"controller": gestures, "payload": copy.deepcopy(payload), "dispatched": False}
            self.paths = {"left": [], "right": []}
            self.sim.control({"action": "stop"})
            self.event("review", "Onboard preset selected. Accept once to run on the robot; no trajectory preview is available.")
            self._log({"type": "proposal", "proposal": self.proposal, "payload": payload})
            return gestures.status()

    def _execute_firmware(self):
        with self.lock:
            p, pending = self.proposal, self.firmware_pending
            if not p or not pending or not self.decision or self.decision["decision"] != "approve":
                raise ValueError("No human acceptance for this preset")
            if self.simulation_only or self.connected or self.cancelled.is_set():
                raise ValueError("Firmware ownership changed; select the preset again")
            payload = copy.deepcopy(self.plan)
            if payload != pending["payload"]:
                raise ValueError("Accepted preset changed")
            validate_approval({"proposal_id": p["id"], "revision": p["revision"],
                              "digest": p["digest"], "expires_at": p["expires_at"]}, digest(payload))
            controller = pending["controller"]
            state = controller.status()
            if (state["iface"] != payload["iface"] or not state["connected"] or state["busy"] or
                not any(a["id"] == payload["action_id"] and a["name"] == payload["name"] for a in state["actions"])):
                raise ValueError("Firmware preset availability changed; refresh and select it again")
            self.event("executing", "Running accepted onboard preset. Use the Unitree controller to interrupt an onboard gesture.")
            if self.cancelled.is_set():
                return
            controller.command({"action": "gesture", "id": payload["action_id"]})
            pending["dispatched"] = True
            proposal_id = p["id"]
        while controller.status()["busy"]:
            if self.closed.wait(.05):
                return
        with self.lock:
            if not self.proposal or self.proposal["id"] != proposal_id:
                return
            state = controller.status()
            self._finish("failed" if state["error"] else "executed",
                         state["error"] or "Accepted onboard preset completed (firmware reported).",
                         feedback={"source": "firmware_rpc", "action_id": payload["action_id"]})

    def stop(self, message="Stopped. Control released and pending motion cancelled."):
        self.cancelled.set(); self.planner.cancel()
        if self.on_stop: self.on_stop()
        with self.lock:
            self.generation += 1
            if self.firmware_pending and self.firmware_pending["dispatched"]:
                message = "Onboard preset was already dispatched. Use the Unitree controller to halt it; dashboard approval is cancelled."
            if self.proposal: self._finish("cancelled", message)
            self.drafts.clear(); self.draft = self.paths = self.plan = None
            pending, self.pending_backend = self.pending_backend, None
            backend, mode = self.backend, self.mode
        if pending: pending.close()
        self.sim.control({"action":"stop"})
        if mode == "live":
            try: backend.freeze(); backend.release()
            except (OSError,RuntimeError): pass
            finally:
                backend.close()
                with self.lock:
                    self.connected, self.mode = False, "sim"
                    self.backend = PreviewBackend(self.robot_state.get("joints", self._simulation_pose()))
        self.event("stopped", message)

    def close(self):
        self.closed.set(); self.stop(); self.watchdog.join(2)
        if self.worker and self.worker is not threading.current_thread(): self.worker.join(3)
        for process in (self.streamer,self.hand_server):
            if process and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=4)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        for log in (self.streamer_log,self.hand_log):
            if log: log.close()
        if self.control_dir: self.control_dir.cleanup()

    def glasses_message(self):
        with self.lock:
            p = self.proposal or self.draft
            if not p or self.paths is None:
                return {"type":"trajectory","version":1,"id":"idle","frame":"robot_base","units":"m","clear":True,"hands":{"left":[],"right":[]},"phase":"idle","review":None}
            item = self.drafts.get(p["plan_id"])
            result = {"type":"trajectory","version":1,"id":p["id"],"frame":"robot_base","units":"m",
                "duration_s":p["duration_s"],"hands":copy.deepcopy(self.paths),"phase":"review" if self.state=="review" else self.state,"review":None}
            if item and item["payload"]["kind"] == "walk":
                result["base_path"] = [[p["position_m"][0], p["position_m"][1], p["yaw_rad"]] for p in base_path(item["payload"])]
                result["frame"] = "map"
            if self.proposal and self.state=="review" and time.time()<p["expires_at"]:
                result["review"] = {"id":p["id"],"digest":p["digest"],"revision":p["revision"],"text":p["name"],"mode":p["mode"],"expires_at":p["expires_at"]}
            return result
