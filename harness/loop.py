"""perceive -> reason -> act with subtask tracking, chunking, adaptive step, history and recovery.

One Episode drives one arm. Every motion goes through ArmExecutor.execute, which goes through
the SafetyGate: the loop never touches joints. The VLM sees only the active stage.
"""
import math
import time

import numpy as np

from .actions import ActionError, OUTPUT_SCHEMA, parse_decision
from .interpreter import Interpreter, step_size
from .perception import height_above_table_cm
from .prompts import PLAN_SCHEMA, RECOVERY_NOTES, controller_prompt, parse_plan, planner_prompt, proprio_text


class Episode:
    def __init__(self, cfg, vlm, executor, perception, recorder=None, log=print):
        self.cfg, self.vlm, self.ex, self.per, self.rec, self.log = cfg, vlm, executor, perception, recorder, log
        self.arm = executor.arm
        self.interp = Interpreter(cfg["frames"]["view_forward"], cfg["frames"]["view_left"])
        self.limits = {"param_max_translation_m": cfg["steps"]["param_max_translation_m"],
                       "param_max_rotation_deg": cfg["steps"]["param_max_rotation_deg"]}
        self.table_z = executor.gate.table_z
        self.stop_reason = None

    # -- planning -------------------------------------------------------------------------------
    def make_plan(self, task, packet):
        prompt = planner_prompt(task, self.cfg, self.arm)
        images = packet.images[:1] + packet.images[1:]
        resp = self.vlm.plan(prompt, images, PLAN_SCHEMA)
        stages, err = None, resp.error
        if not err:
            try:
                stages = parse_plan(resp.text)
            except (ValueError, KeyError) as e:
                err = f"plan rejected: {e}"
        if stages is None:
            resp2 = self.vlm.plan(prompt + f"\n\nYour previous plan was rejected: {err}. Return the JSON plan only.", images, PLAN_SCHEMA)
            try:
                stages = parse_plan(resp2.text) if not resp2.error else None
            except (ValueError, KeyError):
                stages = None
            resp = resp2
        if self.rec:
            self.rec.plan(prompt, resp, stages, images)
        return stages, resp

    # -- the loop ------------------------------------------------------------------------------------
    def run(self, task):
        cfg, lp = self.cfg, self.cfg["loop"]
        state = self.ex.sync()
        self.ex.gate.set_baseline(self.ex.kin.q_from_dict(state.q), self.ex.others(state.q))
        packet = self.per.capture(state.p)
        stages, presp = self.make_plan(task, packet)
        if not stages:
            return self.finish({"success": False, "reason": f"no plan: {presp.error or 'unparseable'}", "steps": 0})
        self.log(f"plan ({presp.model}, {presp.latency_s:.1f} s): " + " -> ".join(f"{s['id']}" for s in stages))
        stage_i, history, recovery, queue = 0, [], None, []
        ik_fails, last_result, failed_steps, stage_steps = 0, None, 0, 0
        q_home = np.asarray(cfg["robot"]["start_pose_rad"][self.arm], float)
        for step in range(int(lp["max_steps"])):
            if self.ex.gate.estop.is_set():
                return self.finish({"success": False, "reason": "e-stop", "steps": step})
            stage = stages[stage_i]
            state = self.ex.sync()
            packet = self.per.capture(state.p)
            if packet.missing and not self.ex.backend.dry_run and self.ex.backend.name != "mock":
                return self.finish({"success": False, "reason": f"camera missing: {packet.missing}", "steps": step})
            decision, resp, prompt, action = None, None, None, None
            if queue:                                                   # open-loop chunk, no VLM call
                action = queue.pop(0); wrist = False
                self.log(f"step {step}: chunk -> {action.raw}")
            else:
                prompt = self.build_prompt(task, stage, state, history, recovery, last_result)
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
            proposal = self.interp.propose(state, action, sigma, theta)
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
            if result.empty_grasp:                                  # open again (Show-Harness recovery), note, roll back
                recovery = RECOVERY_NOTES["empty_grasp"]; queue = []
                stage_i = self.grasp_stage(stages, stage_i)
                self.ex.backend.hand(self.arm, False); extra["auto_release"] = True
            token = action.raw.upper() + ("(empty)" if result.empty_grasp else "") + ("(unreachable)" if result.ik_fail else "")
            history.insert(0, token); history = history[:int(lp["history_len"])]
            self.log(f"   -> {result.feedback or 'ok'}")
            self.record(step, packet, prompt, resp, stage, state, action, result, extra)
            if decision is not None and decision.plan and wrist is False and not result.ik_fail and not result.clamped:
                queue = decision.plan[1:int(lp["chunk_max"])]
            stage_steps += 1
        return self.finish({"success": False, "reason": "max steps", "steps": int(lp["max_steps"])})

    # -- pieces ------------------------------------------------------------------------------------------
    def build_prompt(self, task, stage, state, history, recovery, last):
        sigma, _ = step_size(self.cfg["steps"], False)
        h = height_above_table_cm(state.p, self.table_z)
        stall = clamped = ik = None
        if last is not None:
            if last.ok and last.requested_dp is not None and np.linalg.norm(last.requested_dp) > 1e-6:
                want = np.linalg.norm(last.requested_dp); got = float(last.achieved_dp @ last.requested_dp / want)
                if got < 0.7 * want:
                    stall = f"Last move achieved {got * 100:.1f} of {want * 100:.1f} cm -> already in contact, do NOT repeat it."
            if last.clamped:
                clamped = "; ".join(last.notes)
            if last.ik_fail:
                ik = "The last target was unreachable (IK failed); the arm did not move."
        hand = "no hand" if self.cfg["hand"]["type"] == "none" else ("closed" if state.hand_closed else "open")
        pro = proprio_text(h, sigma * 100, hand, stall, clamped, ik, holding=state.hand_closed)
        return controller_prompt(task, stage, pro, history, recovery, self.cfg, self.arm)

    def ask(self, prompt, images):
        resp = self.vlm.act(prompt, images, OUTPUT_SCHEMA)
        for attempt in range(2):
            if resp.error:
                return None, resp
            try:
                return parse_decision(resp.text, (self.arm,), self.limits), resp
            except ActionError as e:
                if attempt == 1 or not self.cfg["loop"]["reprompt_once"]:
                    resp.error = f"invalid answer: {e}"
                    return None, resp
                resp = self.vlm.act(prompt, images, OUTPUT_SCHEMA, retry_note=str(e))
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
