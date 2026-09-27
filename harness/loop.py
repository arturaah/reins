"""perceive -> reason -> act with subtask tracking, chunking, adaptive step, history and recovery.

One Episode drives one arm. Every motion goes through ArmExecutor.execute, which goes through
the SafetyGate: the loop never touches joints. The VLM sees only the active stage.
"""
import math
import re
import time

import numpy as np

from .actions import ActionError, OUTPUT_SCHEMA, parse_decision, parse_action
from .demos import demo_images, demos_block
from .executor import ExecResult
from .interpreter import Interpreter, step_size
from .perception import height_above_table_cm
from .prompts import PLAN_SCHEMA, RECOVERY_NOTES, controller_prompt, parse_plan, planner_prompt, proprio_text


class Episode:
    def __init__(self, cfg, vlm, executor, perception, recorder=None, log=print, demos=None, feedback=None, stats=None, experience=None):
        self.cfg, self.vlm, self.ex, self.per, self.rec, self.log = cfg, vlm, executor, perception, recorder, log
        self.arm = executor.arm
        self.exp = experience                              # harness.experience.ExperienceStore: cards of answered proposals (optional)
        self.task = ""
        self.demos = list(demos or [])                    # harness.demos.Demo: shown before the images in every call
        self.demo_text = demos_block(self.demos)
        self.demo_images = demo_images(self.demos, int((cfg.get("demos") or {}).get("max_width_px", 1568)))
        self.feedback, self.stats = feedback, stats       # harness.feedback.FeedbackStore, harness.stats.InferenceLog (optional)
        self.fb_text = ""
        self.interp = Interpreter(cfg["frames"]["view_forward"], cfg["frames"]["view_left"])
        self.limits = {"param_max_translation_m": cfg["steps"]["param_max_translation_m"],
                       "param_max_rotation_deg": cfg["steps"]["param_max_rotation_deg"]}
        self.table_z = executor.gate.table_z
        self.stop_reason = None
        lo = cfg.get("locomotion") or {}
        self.loco_cfg = bool(lo.get("enabled", False))
        self.loco = False                                  # decided per task in run(): enabled AND the task says "walk"
        self.walk_m, self.turn_rad = float(lo.get("step_m", 0.2)), math.radians(float(lo.get("turn_deg", 20.0)))
        self.limits.update({"param_max_walk_m": lo.get("param_max_walk_m", 0.4), "param_max_turn_deg": lo.get("param_max_turn_deg", 45.0)})
        self.holding = False                              # the last hand command was a GRASP that closed on something

    # -- planning -------------------------------------------------------------------------------
    def make_plan(self, task, packet):
        prompt = self.with_context(planner_prompt(task, self.cfg, self.arm, locomotion=self.loco, pose_view=any("POSE" in l for l, _ in packet.images)))
        images = self.demo_images + self.exp_images() + packet.images
        resp = self.vlm.plan(prompt, images, PLAN_SCHEMA); self.stat("plan", resp, prompt, images)
        stages, err = None, resp.error
        if not err:
            try:
                stages = parse_plan(resp.text)
            except (ValueError, KeyError) as e:
                err = f"plan rejected: {e}"
        if stages is None:
            prompt2 = prompt + f"\n\nYour previous plan was rejected: {err}. Return the JSON plan only."
            resp2 = self.vlm.plan(prompt2, images, PLAN_SCHEMA); self.stat("plan", resp2, prompt2, images)
            try:
                stages = parse_plan(resp2.text) if not resp2.error else None
            except (ValueError, KeyError):
                stages = None
            resp = resp2
        if self.rec:
            self.rec.plan(prompt, resp, stages, images)
        return stages, resp

    # -- the loop ------------------------------------------------------------------------------------
    @staticmethod
    def task_allows_walking(task):
        """The operator's explicit consent to whole-body motion is the word walk in the task text itself."""
        return bool(re.search(r"\bwalk(s|ed|ing)?\b", task or "", re.IGNORECASE))

    @staticmethod
    def is_walking_stage(stage):
        m = (stage.get("motion") or stage.get("id") or "").upper()
        return "APPROACH" in m or "WALK" in m or "TURN" in m

    def run(self, task, start_pose=None):
        """start_pose: joint targets for the arm's start pose, executed (gated, confirmed) right before the first stage that
        is not a walking stage; None = the caller did it already (or does not want it)."""
        cfg, lp = self.cfg, self.cfg["loop"]
        self.task = task
        self.loco = self.loco_cfg and self.task_allows_walking(task)
        if self.loco_cfg and not self.loco:
            self.log("walking stays off: the task text does not say 'walk'")
        self.start_pose = None if start_pose is None else np.asarray(start_pose, float)
        state = self.ex.sync()
        self.ex.gate.set_baseline(self.ex.kin.q_from_dict(state.q), self.ex.others(state.q))
        self.fb_text = self.feedback.block(task) if self.feedback is not None else ""
        packet = self.per.capture(state.p, joints=state.q)
        if getattr(self.per, "require_context", False) and "CONTEXT VIEW" in packet.missing:
            return self.finish({"success": False, "reason": "camera missing: CONTEXT VIEW", "steps": 0})
        stages, presp = self.make_plan(task, packet)
        if not stages:
            return self.finish({"success": False, "reason": f"no plan: {presp.error or 'unparseable'}", "steps": 0})
        self.log(f"plan ({presp.model}, {presp.latency_s:.1f} s): " + " -> ".join(f"{s['id']}" for s in stages))
        stage_i, history, recovery, op_notes = 0, [], None, []                # op_notes: operator notes since the last prompt
        ik_fails, last_result, failed_steps, stage_steps = 0, None, 0, 0
        q_home = np.asarray(cfg["robot"]["start_pose_rad"][self.arm], float)
        for step in range(int(lp["max_steps"])):
            if self.ex.gate.estop.is_set():
                return self.finish({"success": False, "reason": "e-stop", "steps": step})
            stage = stages[stage_i]
            if self.start_pose is not None and not self.is_walking_stage(stage):
                r = self.ex.go_to_joints(self.start_pose, "start pose"); self.start_pose = None
                self.log(f"start pose: {r.feedback}")
                if not r.ok and not self.ex.backend.dry_run:
                    return self.finish({"success": False, "reason": f"start pose: {r.feedback}", "steps": step})
            state = self.ex.sync()
            aim = None                                              # where the last move aimed, for the pose view
            if last_result is not None and last_result.ok and last_result.requested_dp is not None and np.linalg.norm(last_result.requested_dp) > 1e-6:
                aim = np.asarray(last_result.p_before, float) + np.asarray(last_result.requested_dp, float)
            packet = self.per.capture(state.p, joints=state.q, last_target=aim)
            wrist_missing = any("WRIST" in m for m in packet.missing)
            fatal = [m for m in packet.missing if "WRIST" not in m or not self.per.wrist_optional]
            if fatal and (getattr(self.per, "require_context", False) or (not self.ex.backend.dry_run and self.ex.backend.name != "mock")):
                return self.finish({"success": False, "reason": f"camera missing: {fatal}", "steps": step})
            prompt = self.build_prompt(task, stage, state, history, " ".join(n for n in [recovery, *op_notes] if n) or None,
                                       last_result, wrist_missing, pose_view=any("POSE" in l for l, _ in packet.images))
            op_notes = []
            decision, resp = self.ask(prompt, packet.images)
            if decision is None:
                failed_steps += 1
                history.insert(0, "INVALID"); history = history[:int(lp["history_len"])]
                self.record(step, packet, prompt, resp, stage, state, None, None, {"failed": True, "error": resp.error})
                self.log(f"step {step}: invalid answer ({resp.error}); counted as a failed step")
                if failed_steps >= 3:
                    return self.finish({"success": False, "reason": "three invalid answers", "steps": step + 1})
                continue
            action = decision.action(self.arm); wrist = decision.wrist_visible
            self.log(f"step {step} [{stage['id']}] {resp.model} {resp.latency_s:.1f}s: {action.raw}"
                     + (f" + {len(decision.plan) - 1} more" if len(decision.plan) > 1 else "") + f"  ({decision.reasoning[:90]})")
            sigma, theta = step_size(cfg["steps"], bool(wrist))
            if action.name == "DONE":
                stage_i += 1; recovery = None; history = []; stage_steps = 0
                self.record(step, packet, prompt, resp, stage, state, action, None, {"stage_done": True})
                if stage_i >= len(stages):
                    return self.finish({"success": True, "reason": "all stages done", "steps": step + 1})
                self.log(f"stage done -> {stages[stage_i]['id']}")
                continue
            if history and action.opposite_of is not None and self.same_token(history[0], action.opposite_of):
                recovery = RECOVERY_NOTES["oscillation"]
            def propose(st, a, sigma=sigma, theta=theta):
                return self.interp.propose(st, a, sigma, theta, self.walk_m, self.turn_rad)
            proposal = propose(state, action)
            chain = [a for a in decision.plan[:int(lp["chunk_max"])] if a.name != "STILL"] if decision.plan else []
            if proposal.kind == "walk" and not self.loco:
                why = ("walking is not enabled for this session: the robot cannot move its body; use the arm" if not self.loco_cfg else
                       "walking is only allowed when the task text itself says 'walk'; this task does not, so the body stays put: use the arm")
                result = ExecResult(False, why, state.p, state.p, np.zeros(3), np.zeros(3), state.roll, state.roll)
            elif len(chain) > 1 and proposal.kind in ("move", "rotate"):
                result = self.ex.execute_sequence(chain, state, propose)     # the plan is ONE proposal: the whole trajectory
            else:
                result = self.ex.execute(proposal, state)
            planned = chain[:result.planned] if result.planned else [action]
            label = ", ".join(a.raw.upper() for a in planned)              # what was proposed: for the notes, the stores, the cards
            last_result = result
            if not result.ok:
                recovery = "The previous action was rejected: " + result.feedback
            extra = {}
            if result.ik_fail:
                ik_fails += 1; recovery = RECOVERY_NOTES["ik_fail"]
                if ik_fails >= int(lp["ik_fail_home_after"]):
                    hr = self.ex.home_step(self.ex.sync(), q_home)
                    recovery = RECOVERY_NOTES["home_step"] + " " + hr.feedback
                    ik_fails = 0; extra["home_step"] = hr.feedback
            elif result.ok:
                ik_fails = 0
            if result.asked and self.feedback is not None and not self.ex.gate.estop.is_set():   # every Accept / Reject is kept (an e-stop is not an answer)
                self.feedback.add(task, stage["id"], label, not result.declined, result.operator_note,
                                  hand_tip=state.p, height_cm=height_above_table_cm(state.p, self.table_z), mode=self.ex.backend.name)
            if result.ok and result.walk is not None and result.feedback:                          # after a step: what the odometry says
                op_notes.append(result.feedback)
            if not result.ok and not result.declined and not result.ik_fail and result.feedback:   # refused for another reason: say why
                op_notes.append(f"Your last action ({label}) was NOT executed: {result.feedback}")
            elif result.moves and ("stopped" in result.feedback or "dropped" in result.feedback):  # a trajectory cut short: say where and why
                op_notes.append(f"Your last trajectory ({label}): {result.feedback}")
            if result.declined:                                     # the operator said no: tell the model
                op_notes.append(RECOVERY_NOTES["rejected"].format(token=label, why=f' with the note "{result.operator_note}"' if result.operator_note else ""))
            elif result.operator_note:                              # accepted with a note: the model reads it at its next call
                op_notes.append(RECOVERY_NOTES["accepted_note"].format(token=label, note=result.operator_note))
            if proposal.kind == "hand" and result.ok and not result.declined:
                self.holding = bool(proposal.hand_closed) and not result.empty_grasp
            if result.empty_grasp:                                  # open again (Show-Harness recovery), note, roll back
                recovery = RECOVERY_NOTES["empty_grasp"]
                stage_i = self.grasp_stage(stages, stage_i)
                self.ex.backend.hand(self.arm, False); extra["auto_release"] = True
            if result.moves:                                        # a trajectory: one history token per move that was sent
                for a, r in zip(planned, result.moves):
                    history.insert(0, a.raw.upper() + ("(unreachable)" if r.ik_fail else "" if r.ok else "(failed)"))
            else:
                history.insert(0, label + ("(empty)" if result.empty_grasp else "") + ("(unreachable)" if result.ik_fail else "")
                               + ("(rejected)" if result.declined else ""))
            history = history[:int(lp["history_len"])]
            self.log(f"   -> {result.feedback or 'ok'}")
            self.record(step, packet, prompt, resp, stage, state, action, result, extra, planned)
            if result.asked and self.exp is not None and not self.ex.gate.estop.is_set():
                self.remember(task, stage, label, proposal, result, state, packet)
