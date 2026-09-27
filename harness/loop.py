"""perceive -> reason -> act with subtask tracking, chunking, adaptive step, history and recovery.

One Episode drives one arm. Every motion goes through ArmExecutor.execute, which goes through
the SafetyGate: the loop never touches joints. The VLM sees only the active stage.
"""
import math
import re
import time

import numpy as np

from .actions import ActionError, OUTPUT_SCHEMA, parse_decision
from .demos import demo_images, demos_block
from .executor import ExecResult
from .interpreter import Interpreter, step_size
from .perception import height_above_table_cm
from .prompts import PLAN_SCHEMA, RECOVERY_NOTES, controller_prompt, parse_plan, planner_prompt, proprio_text


class Episode:
    def __init__(self, cfg, vlm, executor, perception, recorder=None, log=print, demos=None, feedback=None, stats=None):
        self.cfg, self.vlm, self.ex, self.per, self.rec, self.log = cfg, vlm, executor, perception, recorder, log
        self.arm = executor.arm
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
        images = self.demo_images + packet.images
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
        self.loco = self.loco_cfg and self.task_allows_walking(task)
        if self.loco_cfg and not self.loco:
            self.log("walking stays off: the task text does not say 'walk'")
        self.start_pose = None if start_pose is None else np.asarray(start_pose, float)
        state = self.ex.sync()
        self.ex.gate.set_baseline(self.ex.kin.q_from_dict(state.q), self.ex.others(state.q))
        self.fb_text = self.feedback.block(task) if self.feedback is not None else ""
        packet = self.per.capture(state.p, joints=state.q)
        stages, presp = self.make_plan(task, packet)
        if not stages:
            return self.finish({"success": False, "reason": f"no plan: {presp.error or 'unparseable'}", "steps": 0})
        self.log(f"plan ({presp.model}, {presp.latency_s:.1f} s): " + " -> ".join(f"{s['id']}" for s in stages))
        stage_i, history, recovery, queue, op_notes = 0, [], None, [], []      # op_notes: operator notes since the last prompt
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
            if fatal and not self.ex.backend.dry_run and self.ex.backend.name != "mock":
                return self.finish({"success": False, "reason": f"camera missing: {fatal}", "steps": step})
            decision, resp, prompt, action = None, None, None, None
            if queue:                                                   # open-loop chunk, no VLM call
                action = queue.pop(0); wrist = False
                self.log(f"step {step}: chunk -> {action.raw}")
            else:
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
                self.log(f"step {step} [{stage['id']}] {resp.model} {resp.latency_s:.1f}s: {action.raw}  ({decision.reasoning[:90]})")
            sigma, theta = step_size(cfg["steps"], bool(wrist))
            if action.name == "DONE":
                stage_i += 1; queue = []; recovery = None; history = []; stage_steps = 0
                self.record(step, packet, prompt, resp, stage, state, action, None, {"stage_done": True})
                if stage_i >= len(stages):
                    return self.finish({"success": True, "reason": "all stages done", "steps": step + 1})
                self.log(f"stage done -> {stages[stage_i]['id']}")
                continue
            if history and action.opposite_of is not None and self.same_token(history[0], action.opposite_of):
                recovery = RECOVERY_NOTES["oscillation"]
            proposal = self.interp.propose(state, action, sigma, theta, self.walk_m, self.turn_rad)
            if proposal.kind == "walk" and not self.loco:
                why = ("walking is not enabled for this session: the robot cannot move its body; use the arm" if not self.loco_cfg else
                       "walking is only allowed when the task text itself says 'walk'; this task does not, so the body stays put: use the arm")
                result = ExecResult(False, why, state.p, state.p, np.zeros(3), np.zeros(3), state.roll, state.roll)
            else:
                result = self.ex.execute(proposal, state)
            last_result = result
            extra = {}
            if result.ik_fail:
                ik_fails += 1; recovery = RECOVERY_NOTES["ik_fail"]; queue = []
                if ik_fails >= int(lp["ik_fail_home_after"]):
                    hr = self.ex.home_step(self.ex.sync(), q_home)
                    recovery = RECOVERY_NOTES["home_step"] + " " + hr.feedback
                    ik_fails = 0; extra["home_step"] = hr.feedback
            elif result.ok:
                ik_fails = 0
            if result.asked and self.feedback is not None and not self.ex.gate.estop.is_set():   # every Accept / Reject is kept (an e-stop is not an answer)
                self.feedback.add(task, stage["id"], action.raw.upper(), not result.declined, result.operator_note,
                                  hand_tip=state.p, height_cm=height_above_table_cm(state.p, self.table_z), mode=self.ex.backend.name)
            if result.ok and result.walk is not None and result.feedback:                          # after a step: what the odometry says
                op_notes.append(result.feedback)
            if not result.ok and not result.declined and not result.ik_fail and result.feedback:   # refused for another reason: say why
                op_notes.append(f"Your last action ({action.raw.upper()}) was NOT executed: {result.feedback}")
            if result.declined:                                     # the operator said no: tell the model, drop the chunk
                queue = []
                op_notes.append(RECOVERY_NOTES["rejected"].format(token=action.raw.upper(), why=f' with the note "{result.operator_note}"' if result.operator_note else ""))
            elif result.operator_note:                              # accepted with a note: the model reads it at its next call
                op_notes.append(RECOVERY_NOTES["accepted_note"].format(token=action.raw.upper(), note=result.operator_note))
            if proposal.kind == "hand" and result.ok and not result.declined:
                self.holding = bool(proposal.hand_closed) and not result.empty_grasp
            if result.empty_grasp:                                  # open again (Show-Harness recovery), note, roll back
                recovery = RECOVERY_NOTES["empty_grasp"]; queue = []
                stage_i = self.grasp_stage(stages, stage_i)
                self.ex.backend.hand(self.arm, False); extra["auto_release"] = True
            token = action.raw.upper() + ("(empty)" if result.empty_grasp else "") + ("(unreachable)" if result.ik_fail else "") + ("(rejected)" if result.declined else "")
            history.insert(0, token); history = history[:int(lp["history_len"])]
            self.log(f"   -> {result.feedback or 'ok'}")
            self.record(step, packet, prompt, resp, stage, state, action, result, extra)
            stalled = (result.ok and result.requested_dp is not None and np.linalg.norm(result.requested_dp) > 1e-6
                       and float(result.achieved_dp @ result.requested_dp) < 0.3 * float(np.linalg.norm(result.requested_dp)) ** 2)
            if decision is not None and decision.plan and wrist is False and not result.ik_fail and not result.clamped and not result.declined and not stalled:
                queue = decision.plan[1:int(lp["chunk_max"])]
            stage_steps += 1
        return self.finish({"success": False, "reason": "max steps", "steps": int(lp["max_steps"])})

    # -- pieces ------------------------------------------------------------------------------------------
    def build_prompt(self, task, stage, state, history, recovery, last, wrist_missing=False, pose_view=False):
        sigma, _ = step_size(self.cfg["steps"], False)
        h = height_above_table_cm(state.p, self.table_z)
        stall = clamped = ik = None
        if last is not None:
            if last.ok and last.requested_dp is not None and np.linalg.norm(last.requested_dp) > 1e-6:
                want = np.linalg.norm(last.requested_dp); got = float(last.achieved_dp @ last.requested_dp / want)
                if got < 0.3 * want:
                    stall = f"Last move achieved {got * 100:.1f} of {want * 100:.1f} cm -> already in contact, do NOT repeat it."
            if last.clamped:
                clamped = "; ".join(last.notes)
            if last.ik_fail:
                ik = "The last target was unreachable (IK failed); the arm did not move."
        hand = "no hand" if self.cfg["hand"]["type"] == "none" else ("closed" if state.hand_closed else "open")
        hand_note = last.feedback if last is not None and last.hand_closed is not None and not last.declined else None
        pro = proprio_text(h, sigma * 100, hand, stall, clamped, ik, holding=state.hand_closed and self.holding, hand_note=hand_note)
        return self.with_context(controller_prompt(task, stage, pro, history, recovery, self.cfg, self.arm, wrist_missing=wrist_missing,
                                                   locomotion=self.loco, pose_view=pose_view))

    def with_context(self, prompt):
        """Demonstrations, then earlier sessions' operator feedback, then the prompt itself."""
        return "\n\n".join([p for p in (self.demo_text, self.fb_text) if p] + [prompt])

    with_demos = with_context

    def stat(self, kind, resp, prompt, images):
        if self.stats is not None:
            self.stats.record(kind, resp, prompt, images)

    def ask(self, prompt, images):
        images = self.demo_images + list(images)
        resp = self.vlm.act(prompt, images, OUTPUT_SCHEMA); self.stat("act", resp, prompt, images)
        for attempt in range(2):
            if resp.error:
                return None, resp
            try:
                return parse_decision(resp.text, (self.arm,), self.limits), resp
            except ActionError as e:
                if attempt == 1 or not self.cfg["loop"]["reprompt_once"]:
                    resp.error = f"invalid answer: {e}"
                    return None, resp
                resp = self.vlm.act(prompt, images, OUTPUT_SCHEMA, retry_note=str(e)); self.stat("act", resp, prompt, images)
        return None, resp

    @staticmethod
    def same_token(hist_token, action):
        return hist_token.split("(")[0] == action.raw.upper() if action.raw else False

    @staticmethod
    def grasp_stage(stages, i):
        for k in range(i, -1, -1):
            if "GRASP" in stages[k].get("motion", "").upper():
                return k
        return i

    def record(self, step, packet, prompt, resp, stage, state, action, result, extra):
        if not self.rec:
            return
        rec = {"stage": stage["id"], "action": action.raw if action else None,
               "hand_tip_before": state.p, "roll_before": state.roll, "joints_before": state.q,
               "images_missing": packet.missing, **extra}
        if result is not None:
            rec.update({"ok": result.ok, "feedback": result.feedback, "hand_tip_after": result.p_after,
                        "requested_dp": result.requested_dp, "achieved_dp": result.achieved_dp,
                        "q_target": result.q_target, "q_after": result.q_after, "ik_fail": result.ik_fail,
                        "timeout": result.timeout, "clamped": result.clamped, "notes": result.notes, "duration_s": result.duration_s})
        self.rec.step(step, rec, packet, prompt, resp)

    def finish(self, summary):
        self.log(f"episode: {summary}")
        if self.rec:
            summary["run_dir"] = str(self.rec.finish(summary))
        return summary
