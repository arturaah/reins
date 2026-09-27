"""Claude plans and looks, rarely; a System One model (TypeSafe Jev) decides every step from text.

Claude (the VLM, any harness.vlm provider) makes the stage plan exactly as in harness.loop and, for each stage, a scene
SNAPSHOT from the camera images: where the hand tip has to go to finish the stage, as an offset from where it is now
(cm along forward / left / up), the stage's done condition restated in terms of that offset, whether the wrist camera
sees the target, and hazards. Between snapshots the goal is dead-reckoned: it stays fixed in the robot frame while the
hand's own motion is known exactly from forward kinematics, so code recomputes the remaining gap after every move and
writes it as words ("the goal is 6 cm below the hand tip (near)"). Jev does semantic judgement, not arithmetic, so the
numbers stay in code (docs.typesafe.ai/model-jaggedness/jev-1.13).

Each step the decider (harness.decider: jev | scripted) answers, in one call:
  action      Choice over MV_* / STILL / DONE (+ GRASP / RELEASE with a hand) / LOOK, with probabilities
  stage_done  Noul: is the restated done condition met?
  needs_look  Noul: does the situation need fresh eyes?
With executor.mover = geometric, code picks the move (close the largest gap) and Jev only answers the two Nouls.

Claude is called again only when:
  code says so    no snapshot for this stage yet, N moves or M cm since the last one, the last move was unreachable,
                  stalled, rejected or carried an operator note, the last snapshot's confidence was low
  the hand says   a GRASP that closed on an object ends a GRASP stage, a RELEASE ends a RELEASE stage, without asking
                  Claude: the hand's own report beats the cameras, where a held object and one just past the fist look alike.
                  In any other stage a GRASP / RELEASE makes Claude look at the stage next step
  Jev says so     it chose LOOK, its action confidence is under executor.min_action_confidence, needs_look is high
                  (ignored when Claude looked this very step: nothing fresher exists),
                  or the call failed; the step is re-asked once with the fresh snapshot, and if it is still unsure
                  Claude decides that step itself from the images (the harness.loop controller prompt)
  a stage ends    DONE from Jev is a claim: Claude confirms it from the images before the stage advances
Every move still goes through ArmExecutor.execute and the SafetyGate, and asks the operator exactly as before.
"""
import json
from dataclasses import dataclass

import numpy as np

from .actions import Action, parse_action
from .decider.base import Choice, Noul
from .decider.scripted import geometric_token
from .loop import Episode
from .interpreter import step_size
from .perception import height_above_table_cm
from .prompts import IMAGES_LINE, IMAGES_LINE_NO_WRIST, RECOVERY_NOTES, mem_text, robot_description

AXES = (("forward", "forward", "back"), ("left", "left", "right"), ("up", "up", "down"))
MOVE_TOKEN = {("forward", 1): "MV_FWD", ("forward", -1): "MV_BACK", ("left", 1): "MV_LEFT",
              ("left", -1): "MV_RIGHT", ("up", 1): "MV_UP", ("up", -1): "MV_DOWN"}

SNAPSHOT = """ROLE: SceneSnapshot
You are the eyes of a fast text-only controller. It cannot see. Until your next look it moves the hand tip in small
steps using only what you report now plus the hand's own measured motion, so report the geometry carefully.

TASK: {task}
ROBOT: {robot_desc}
STAGE: {motion} ({stage_id})
TARGET: {target}
AFFORD: {affordance}
Stage goal: {description}
DONE WHEN: {completion}
Hand now: {hand_state}. The hand tip is {height_cm:.1f} cm above the table; each controller step moves about {step_cm:.0f} cm.
{mem_text}
{notes}
Why you are asked now: {why}

{images_line}

DIRECTIONS in the robot frame, as seen in the CONTEXT VIEW: forward = away from the robot = higher in the context image;
left = the image's left; up = against gravity.

Report:
- goal_offset_cm: where the hand tip must be to finish THIS stage, relative to where it is now, in cm along forward,
  left and up. When the stage ends at a position relative to an object (above it, touching it, beside it), give that
  position, not the object's centre: hovering 3 cm above a block whose top is 10 cm forward and 8 cm lower than the
  hand tip is forward 10, left 0, up -5. For LIFT or RETREAT give the clearance point. Judge distances from the grid,
  the wrist view and the known hand height. 0 on an axis means already aligned on it.
- done_when: DONE WHEN restated so it can be judged from those offsets alone (e.g. "the hand tip is within 1 cm of the
  goal on every axis"). Preserve any non-position requirements: GRASP also requires holding the object, RELEASE
  requires opening the hand, and rotation requires the requested orientation; alignment alone cannot finish them.
- stage_complete: true only if DONE WHEN is already visible in the images.
- wrist_sees_target, target_visible (in any view), hazards (anything the hand could hit on the way, or "none"),
  confidence in the offsets (low | medium | high), reasoning (one visual sentence).
{next_block}Return JSON only."""

NEXT_BLOCK = """NEXT STAGE: {motion} -- {description} TARGET: {target}; AFFORD: {affordance}; DONE WHEN: {completion}
Only if stage_complete is true, also report next_goal_offset_cm and next_done_when for the NEXT stage, the same way
(offsets from where the hand tip is now), so the controller can go on without another look. Otherwise leave them out.
"""

SNAPSHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_offset_cm": {"type": "object", "properties": {k: {"type": "number"} for k in ("forward", "left", "up")},
                           "required": ["forward", "left", "up"], "additionalProperties": False},
        "done_when": {"type": "string"},
        "stage_complete": {"type": "boolean"},
        "wrist_sees_target": {"type": "boolean"},
        "target_visible": {"type": "boolean"},
        "hazards": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
        "next_goal_offset_cm": {"type": "object", "properties": {k: {"type": "number"} for k in ("forward", "left", "up")},
                                "required": ["forward", "left", "up"], "additionalProperties": False},
        "next_done_when": {"type": "string"},
    },
    "required": ["goal_offset_cm", "done_when", "stage_complete", "wrist_sees_target", "target_visible", "hazards",
                 "confidence", "reasoning"],
    "additionalProperties": False,
}


DEFAULT_DONE = "the hand tip is within 1 cm of the goal on every axis"


@dataclass
class Snapshot:
    goal: np.ndarray          # where the hand tip must go for this stage, robot base frame, m
    p_at: np.ndarray          # hand tip when the snapshot was taken
    step: int                 # loop step it was taken at
    offset_cm: dict           # as Claude reported it (after clipping)
    done_when: str
    stage_complete: bool
    wrist_sees_target: bool
    target_visible: bool
    hazards: str
    confidence: str
    reasoning: str
    next: "Snapshot" = None   # the next stage's goal, reported with stage_complete, so the next stage starts without a look

    def summary(self):
        return {**({"next_goal_offset_cm": self.next.offset_cm} if self.next else {}),"goal_offset_cm": self.offset_cm, "stage_complete": self.stage_complete, "done_when": self.done_when,
                "wrist_sees_target": self.wrist_sees_target, "confidence": self.confidence, "hazards": self.hazards,
                "reasoning": self.reasoning}


def parse_snapshot(text, p_now, R_view, step, max_cm=60.0):
    """Claude's snapshot JSON -> Snapshot. Raises ValueError when it is off-contract."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e.msg}") from None
    if not isinstance(obj, dict):
        raise ValueError("the snapshot must be a JSON object")
    p_now = np.asarray(p_now, float)

    def offset(key, required):
        off = obj.get(key)
        if off is None and not required:
            return None
        if not isinstance(off, dict):
            raise ValueError(f'missing "{key}" {{forward, left, up}}')
        try:
            v = [off[k] for k in ("forward", "left", "up")]
            if any(type(x) not in (int, float) or not np.isfinite(x) for x in v):
                raise ValueError("offsets must be finite numbers")
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'"{key}" needs numbers for forward, left and up') from None
        return [max(-max_cm, min(max_cm, x)) for x in v]

    def goal(v):
        return p_now + np.asarray(R_view, float) @ (np.asarray(v) / 100.0)

    v = offset("goal_offset_cm", True)
    for key in ("stage_complete", "wrist_sees_target", "target_visible"):
        if type(obj.get(key)) is not bool:
            raise ValueError(f'"{key}" must be a boolean')
    for key in ("done_when", "hazards", "reasoning"):
        if not isinstance(obj.get(key), str):
            raise ValueError(f'"{key}" must be a string')
    conf = obj.get("confidence")
    if conf not in ("low", "medium", "high"):
        raise ValueError('"confidence" must be low, medium or high')
    snap = Snapshot(goal(v), p_now.copy(), step, dict(zip(("forward", "left", "up"), v)),
                    str(obj.get("done_when") or DEFAULT_DONE), bool(obj.get("stage_complete", False)),
                    bool(obj.get("wrist_sees_target", False)), bool(obj.get("target_visible", True)),
                    str(obj.get("hazards") or "none"), conf, str(obj.get("reasoning", "")))
    nv = offset("next_goal_offset_cm", False) if snap.stage_complete else None
    if nv is not None:
        snap.next = Snapshot(goal(nv), p_now.copy(), step, dict(zip(("forward", "left", "up"), nv)),
                             str(obj.get("next_done_when") or DEFAULT_DONE), False, False, snap.target_visible,
                             snap.hazards, conf, "reported with the previous stage's completion")
    return snap


def gap_cm(snap, p, R_view):
    """Remaining goal - hand tip in the view frame (forward, left, up), cm."""
    return np.asarray(R_view, float).T @ (snap.goal - np.asarray(p, float)) * 100.0


def bucket(cm):
    return "close" if cm < 3 else ("near" if cm < 8 else "far")


def describe_gap(g, aligned_cm):
    """Gap (forward, left, up cm) -> ({axis: sentence}, largest-gap direction word or None, its move token or None)."""
    words, best, best_v = {}, None, 0.0
    for (axis, pos, neg), v in zip(AXES, g):
        key = f"{pos}_{neg}"
        if abs(v) <= aligned_cm:
            words[key] = f"aligned (within {aligned_cm:.0f} cm)"
            continue
        d = pos if v > 0 else neg
        where = {"forward": "forward of", "back": "behind", "left": "to the left of", "right": "to the right of",
                 "up": "above", "down": "below"}[d]
        words[key] = f"the goal is {abs(v):.0f} cm {where} the hand tip ({bucket(abs(v))})"
        if abs(v) > best_v:
            best, best_v = (axis, 1 if v > 0 else -1, d), abs(v)
    if best is None:
        return words, None, None
    return words, best[2], MOVE_TOKEN[(best[0], best[1])]


def last_result_text(last, stall_ratio=0.7):
    if last is None:
        return "no move yet in this session"
    if last.declined:
        return "the operator rejected the last proposal; it was not executed" + (f' (note: "{last.operator_note}")' if last.operator_note else "")
    if last.ik_fail:
        return "the last target was out of reach; the arm did not move"
    if not last.ok:
        return "the last move was blocked: " + last.feedback
    if last.timeout:
        return "the last move did not settle before the timeout: " + last.feedback
    if last.empty_grasp:
        return "the last grasp closed on nothing and the hand was opened again"
    parts = []
    if last.ok and last.requested_dp is not None and np.linalg.norm(last.requested_dp) > 1e-6:
        want = np.linalg.norm(last.requested_dp); got = float(last.achieved_dp @ last.requested_dp / want)
        if got < stall_ratio * want:
            parts.append(f"the last move stalled after {got * 100:.1f} of {want * 100:.1f} cm: the hand is probably touching something")
    if last.clamped:
        parts.append("the last move was shortened by the safety box")
    return "; ".join(parts) or "the last move completed normally"


def stalled(last, stall_ratio=0.7):
    if last is None or not last.ok or last.requested_dp is None or np.linalg.norm(last.requested_dp) < 1e-6:
        return False
    want = np.linalg.norm(last.requested_dp)
    return float(last.achieved_dp @ last.requested_dp / want) < stall_ratio * want


def decision_questions(hand_type, mover):
    qs = {}
    if mover == "jev":
        opts = {
            "MV_FWD": "Move the hand tip one step forward, away from the robot. Right when the goal is forward of the hand tip.",
            "MV_BACK": "Move the hand tip one step back, toward the robot. Right when the goal is behind the hand tip.",
            "MV_LEFT": "Move the hand tip one step to the left. Right when the goal is to the left of the hand tip.",
            "MV_RIGHT": "Move the hand tip one step to the right. Right when the goal is to the right of the hand tip.",
            "MV_UP": "Move the hand tip one step up. Right when the goal is above the hand tip.",
            "MV_DOWN": "Move the hand tip one step down. Right when the goal is below the hand tip.",
            "ROTATE_CW": "Roll the wrist clockwise one step when the stage explicitly requires clockwise rotation.",
            "ROTATE_CCW": "Roll the wrist counterclockwise one step when the stage explicitly requires counterclockwise rotation.",
        }
        if hand_type != "none":
            opts["GRASP"] = "Close the hand. Only in a GRASP stage, with the goal aligned on every axis and nothing held."
            opts["RELEASE"] = "Open the hand. Only in a RELEASE stage, while holding an object, with the goal aligned on every axis."
        opts["STILL"] = "Hold still for one step. Only when every move would make things worse."
        opts["DONE"] = ("This stage is finished: every axis is aligned AND stage.completion and stage.done_when are met. "
                        "Never DONE for GRASP while holding_object is no, or RELEASE while holding_object is yes.")
        opts["LOOK"] = ("The text is not enough to choose safely: the last move was blocked, stalled or rejected, an operator "
                        "note contradicts the state, or the goal may have moved. Ask for a fresh camera look.")
        qs["action"] = Choice(
            "Which one move should the hand tip make next to finish `stage`? The gap still to close is in "
            "`goal_relative_to_hand_tip`; close `goal_relative_to_hand_tip.largest_gap` first. "
            "Once aligned, GRASP if this is a GRASP stage and holding_object is no; RELEASE if this is a RELEASE "
            "stage and holding_object is yes. Those actions are required before DONE. "
            "Take `last_move_result` and `operator_notes` into account.", opts)
    qs["stage_done"] = Noul("Are BOTH stage.completion and stage.done_when already met? Check the gap AND hand.holding_object.",
                            {"true": "the completion condition is satisfied; for GRASP the object is held; for RELEASE the hand is open",
                             "false": "some gap remains, or a GRASP still needs to close, or a RELEASE still needs to open, "
                                      "or the completion condition is not met"})
    qs["needs_look"] = Noul("Should the robot take a fresh camera look before its next move?",
                            {"true": "`last_move_result` says the move was blocked, stalled or rejected, or `operator_notes` "
                                     "say something the rest of the state does not show",
                             "false": "the state is current and the next move is clear"})
    return qs


def decision_state(task, stage, snap, p, R_view, height_cm, step_cm, aligned_cm, hand_type, holding, history, last,
                   op_notes, steps_since, moved_cm, fresh=False, floor_cm=None):
    """-> (state for Jev: words only where numbers would be judged, facts: the numbers behind them).
    floor_cm: the lowest hand height the safety gate allows; at it, a goal further down counts as reached."""
    g = gap_cm(snap, p, R_view)
    at_floor = floor_cm is not None and height_cm <= floor_cm + 0.5 and g[2] < 0
    if at_floor:
        g = np.array([g[0], g[1], 0.0])
    words, largest, token = describe_gap(g, aligned_cm)
    if at_floor:
        words["up_down"] = "aligned: the hand tip is at the table and cannot go lower"
    words["largest_gap"] = largest or "none: the hand tip is at the goal"
    state = {
        "task": task,
        "stage": {"name": stage.get("motion") or stage.get("id", ""), "target": stage.get("target", ""),
                  "description": stage.get("description", ""), "completion": stage.get("completion", ""),
                  "goal_for_hand_tip": stage.get("affordance", ""), "done_when": snap.done_when},
        "hand": {"height_above_table": f"{height_cm:.0f} cm",
                 "holding_object": "no hand fitted" if hand_type == "none" else ("yes" if holding else "no")},
        "goal_relative_to_hand_tip": words,
        "step_size": f"each move is about {step_cm:.0f} cm",
        "scene": {"wrist_camera_sees_target": "yes" if snap.wrist_sees_target else "no", "hazards": snap.hazards,
                  "target_visible": "yes" if snap.target_visible else "no", "confidence": snap.confidence,
                  "last_camera_look": "just now, after the last move" if fresh else
                                      f"{steps_since} move(s) ago; the hand has moved {moved_cm:.0f} cm since"},
        "recent_moves_newest_first": history or ["none"],
        "last_move_result": last_result_text(last) + ("; a camera look was taken after it and `goal_relative_to_hand_tip` "
                                                       "already accounts for it" if fresh and last is not None else ""),
        "operator_notes": op_notes or ["none"],
    }
    facts = {"gap_cm": [round(float(x), 2) for x in g], "aligned_cm": aligned_cm, "largest": token,
             "motion": (stage.get("motion") or "").upper(), "holding": bool(holding), "hand": hand_type}
    return state, facts


class SplitEpisode(Episode):
    def __init__(self, cfg, vlm, decider, executor, perception, **kw):
        super().__init__(cfg, vlm, executor, perception, **kw)
        self.decider = decider
        x = cfg["executor"]
        self.mover = x.get("mover", "jev")
        self.min_conf = float(x["min_action_confidence"])
        self.done_p, self.look_p = float(x["stage_done_threshold"]), float(x["needs_look_threshold"])
        self.every, self.after_cm = int(x["look_every_steps"]), float(x["look_after_move_cm"])
        self.aligned_min_cm = float(x["aligned_cm"])
        self.claude_fallback, self.confirm_done = bool(x["claude_fallback"]), bool(x["confirm_done_with_claude"])
        self.max_offset_cm = float(x.get("max_goal_offset_cm", 60.0))
        self.snap = None
        self.confirm_why = None     # set after a GRASP / RELEASE that did something: Claude checks the stage next step
        self.counts = {"claude_looks": 0, "decider_calls": 0, "claude_steps": 0, "decider_steps": 0}

    # -- Claude's look -------------------------------------------------------------------------------------------
    def look(self, task, stage, state, packet, history, notes, wrist_missing, step, why, next_stage=None):
        h = height_above_table_cm(state.p, self.table_z)
        sigma, _ = step_size(self.cfg["steps"], self.snap.wrist_sees_target if self.snap else False)
        hand = "no hand" if self.cfg["hand"]["type"] == "none" else (
            "closed and HOLDING an object (the hand senses a real hold; a close on nothing reopens by itself). An object "
            "attached to or sticking out of the closed hand is the held object, not a target to reach for"
            if state.hand_closed else "open, holding nothing")
        images_line = (IMAGES_LINE_NO_WRIST if wrist_missing else IMAGES_LINE).format(
            wrist_label=f"{self.arm.upper()} WRIST VIEW", arm=self.arm,
            ee_desc="the point 13 cm beyond the wrist" if self.cfg["hand"]["type"] == "none" else "between the fingers")
        prompt = self.with_context(SNAPSHOT.format(
            task=task, robot_desc=robot_description(self.cfg, self.arm), motion=stage.get("motion", ""), stage_id=stage["id"],
            target=stage.get("target", ""), affordance=stage.get("affordance", ""), description=stage.get("description", ""),
            completion=stage.get("completion", ""), hand_state=hand, height_cm=h, step_cm=sigma * 100,
            mem_text=mem_text(history), notes=("Notes: " + " ".join(notes)) if notes else "", why=why, images_line=images_line,
            next_block=NEXT_BLOCK.format(**{k: next_stage.get(k, "") for k in ("motion", "description", "target", "affordance",
                                                                                "completion")}) if next_stage else ""))
        images = self.demo_images + packet.images
        snap, err, resp = None, "", None
        for attempt in range(2):
            resp = self.vlm.act(prompt, images, SNAPSHOT_SCHEMA, retry_note=err or None)
            self.stat("snapshot", resp, prompt, images); self.counts["claude_looks"] += 1
            if resp.error:
                err = resp.error
                continue
            try:
                snap = parse_snapshot(resp.text, state.p, self.interp.R_view, step, self.max_offset_cm)
                break
            except ValueError as e:
                err = f"snapshot rejected: {e}"
        if self.rec:
            self.rec.attach(step, "look_prompt.txt", prompt)
            self.rec.attach(step, "look_response.txt", (resp.text or resp.error) if resp else err)
        if snap is not None:
            self.snap = snap
            o = snap.offset_cm
            self.log(f"step {step}: LOOK ({why}) {resp.model} {resp.latency_s:.1f}s: goal fwd {o['forward']:+.0f} left {o['left']:+.0f} "
                     f"up {o['up']:+.0f} cm, {snap.confidence}{', STAGE COMPLETE' if snap.stage_complete else ''}  ({snap.reasoning[:80]})")
        else:
            self.snap = None
            self.log(f"step {step}: LOOK failed ({err})")
        return snap, err

    def look_reason(self, step, state, last):
        if self.confirm_why:                                   # the goal geometry cannot tell whether a grasp finished the stage
            why, self.confirm_why = self.confirm_why, None
            return why
        if self.snap is None:
            return "first look at this stage"
        if last is not None:
            if last.ik_fail:
                return "the last target was out of reach"
            if last.declined:
                return "the operator rejected the last move"
            if last.operator_note:
                return "the operator added a note"
            if not last.ok:
                return "the last move was blocked: " + last.feedback
            if last.timeout:
                return "the last move did not settle"
            if stalled(last):
                return "the last move stalled"
        n = step - self.snap.step
        if n >= self.every:
            return f"{n} moves since the last look"
        moved = float(np.linalg.norm(np.asarray(state.p) - self.snap.p_at)) * 100
        if moved >= self.after_cm:
            return f"the hand moved {moved:.0f} cm since the last look"
        if self.snap.confidence == "low" and n >= 2:
            return "the last look was low confidence"
        return None

    # -- the decider -------------------------------------------------------------------------------------------------
    def consult(self, task, stage, state, history, last, op_notes, step):
        """One decider call. -> (token or None, escalation reason or None, done_claim, record dict, prompt text, resp)"""
        sigma, _ = step_size(self.cfg["steps"], self.snap.wrist_sees_target)
        aligned = max(self.aligned_min_cm, 0.6 * sigma * 100)
        moved = float(np.linalg.norm(np.asarray(state.p) - self.snap.p_at)) * 100
        fresh = self.snap.step == step                          # Claude looked this step: nothing fresher is available
        st, facts = decision_state(task, stage, self.snap, state.p, self.interp.R_view, height_above_table_cm(state.p, self.table_z),
                                   sigma * 100, aligned, self.cfg["hand"]["type"], state.hand_closed, history, last,
                                   op_notes, step - self.snap.step, moved, fresh,
                                   float(self.cfg["workspace"]["table_margin_m"]) * 100)
        qs = decision_questions(self.cfg["hand"]["type"], self.mover)
        prompt = json.dumps({"state": st, "questions": {k: {"type": q.type, "instructions": q.instructions,
                                                            "criteria": q.criteria} for k, q in qs.items()}}, indent=1)
        answers, resp = self.decider.ask(st, qs, facts)
        self.stat("decider", resp, prompt, []); self.counts["decider_calls"] += 1
        rec = {"facts": facts, "answers": {k: vars(a) for k, a in answers.items()}}
        if resp.error:
            return None, f"decider error: {resp.error}", False, rec, prompt, resp
        done_p, look_p = answers["stage_done"].noul, answers["needs_look"].noul
        if self.mover == "jev":
            a = answers["action"]
            token = a.choice
            if token == "LOOK":
                return None, "the decider asked for a look", False, rec, prompt, resp
            if a.confidence < self.min_conf:
                return None, f"decider unsure ({token} at confidence {a.confidence:.2f})", False, rec, prompt, resp
        else:
            token = geometric_token(facts)
        if look_p >= self.look_p and not fresh:
            return None, f"the decider wants a look (needs_look {look_p:.2f})", False, rec, prompt, resp
        done = token == "DONE" or done_p >= self.done_p
        # Position alone cannot complete a hand operation, regardless of model probabilities.
        pending_hand = (facts["hand"] != "none" and
                        (("GRASP" in facts["motion"] and not facts["holding"]) or
                         ("RELEASE" in facts["motion"] and facts["holding"])))
        if pending_hand:
            done = False
            if token == "DONE":
                return None, "the hand operation has not completed", False, rec, prompt, resp
        return ("DONE" if done else token), None, done, rec, prompt, resp

    # -- the loop ----------------------------------------------------------------------------------------------------
    def run(self, task):
        cfg, lp = self.cfg, self.cfg["loop"]
        state = self.ex.sync()
        self.ex.gate.set_baseline(self.ex.kin.q_from_dict(state.q), self.ex.others(state.q))
        self.fb_text = self.feedback.block(task) if self.feedback is not None else ""
        packet = self.per.capture(state.p)
        stages, presp = self.make_plan(task, packet)
        if not stages:
            return self.finish({"success": False, "reason": f"no plan: {presp.error or 'unparseable'}", "steps": 0})
        self.log(f"plan ({presp.model}, {presp.latency_s:.1f} s): " + " -> ".join(s["id"] for s in stages)
                 + f"   [per step: {self.decider.name}, mover {self.mover}]")
        stage_i, history, op_notes, recovery = 0, [], [], None
        ik_fails, last, failed_steps, done_denied = 0, None, 0, 0
        q_home = np.asarray(cfg["robot"]["start_pose_rad"][self.arm], float)
        for step in range(int(lp["max_steps"])):
            if self.ex.gate.estop.is_set():
                return self.finish({"success": False, "reason": "e-stop", "steps": step})
            stage = stages[stage_i]
            state = self.ex.sync()
            packet = self.per.capture(state.p)
            wrist_missing = any("WRIST" in m for m in packet.missing)
            fatal = [m for m in packet.missing if "WRIST" not in m or not self.per.wrist_optional]
            if fatal and not self.ex.backend.dry_run and self.ex.backend.name != "mock":
                return self.finish({"success": False, "reason": f"camera missing: {fatal}", "steps": step})
            notes = [n for n in [recovery, *op_notes] if n]
            extra, looked = {}, False

            def take_look(why):
                nonlocal looked
                nxt = stages[stage_i + 1] if stage_i + 1 < len(stages) else None
                snap, err = self.look(task, stage, state, packet, history, notes, wrist_missing, step, why, nxt)
                looked = True
                extra.setdefault("looks", []).append({"why": why, **(snap.summary() if snap else {"error": err})})
                return snap

            why = self.look_reason(step, state, last)
            if why and take_look(why) is None:
                failed_steps += 1
                self.record(step, packet, None, None, stage, state, None, None, {"failed": True, **extra})
                if failed_steps >= 3:
                    return self.finish({"success": False, "reason": "three failed looks", "steps": step + 1})
                continue
            if looked and self.snap.stage_complete:
                stage_i, advanced = self.advance(stages, stage_i, step, packet, stage, state, extra, "planner saw it complete")
                history, recovery, op_notes, done_denied = [], None, [], 0
                if advanced is not None:
                    return advanced
                continue

            token, esc, done, drec, prompt, resp = self.consult(task, stage, state, history, last, op_notes, step)
            extra["decider"] = drec
            if esc and not looked:                                  # fresh eyes, then ask the fast model once more
                if take_look(esc) is not None:
                    if self.snap.stage_complete:
                        stage_i, advanced = self.advance(stages, stage_i, step, packet, stage, state, extra, "planner saw it complete")
                        history, recovery, op_notes, done_denied = [], None, [], 0
                        if advanced is not None:
                            return advanced
                        continue
                    token, esc2, done, drec, prompt, resp = self.consult(task, stage, state, history, last, op_notes, step)
                    extra["decider_after_look"] = drec
                    esc = esc2 and f"{esc}; after the look: {esc2}"
            by = self.decider.name if self.mover == "jev" else "geometry"
            if done and not esc and self.confirm_done:              # DONE is a claim until Claude has seen it
                if looked or take_look("the fast model says the stage is done; confirm from the images") is not None:
                    if self.snap.stage_complete:
                        stage_i, advanced = self.advance(stages, stage_i, step, packet, stage, state, extra, f"{by} said done, planner confirmed")
                        history, recovery, op_notes, done_denied = [], None, [], 0
                        if advanced is not None:
                            return advanced
                        continue
                    done_denied += 1
                    esc = "the planner did not confirm the stage as done" if done_denied >= 2 else None
                    if esc is None:                                 # the fresh look is the new goal: decide again from it
                        token, esc, done, drec, prompt, resp = self.consult(task, stage, state, history, last, op_notes, step)
                        extra["decider_after_denied_done"] = drec
                        if done:
                            esc = "the fast model still says done; the planner does not"
                else:
                    esc = "the stage completion camera look failed"
            elif done and not esc:
                stage_i, advanced = self.advance(stages, stage_i, step, packet, stage, state, extra, f"{by} said done")
                history, recovery, op_notes, done_denied = [], None, [], 0
                if advanced is not None:
                    return advanced
                continue

            action = None
            if esc:
                extra["escalation"] = esc
                if not self.claude_fallback:
                    action = parse_action("STILL"); by = "hold (no fallback)"
                else:                                               # Claude decides this step from the images
                    prompt = self.build_prompt(task, stage, state, history, " ".join(notes) or None, last, wrist_missing)
                    decision, resp = self.ask(prompt, packet.images)
                    self.counts["claude_steps"] += 1; by = self.vlm.name
                    if decision is None:
                        failed_steps += 1
                        history.insert(0, "INVALID"); history = history[:int(lp["history_len"])]
                        self.record(step, packet, prompt, resp, stage, state, None, None, {"failed": True, "error": resp.error, **extra})
                        self.log(f"step {step}: {esc}; planner's answer was invalid ({resp.error})")
                        if failed_steps >= 3:
                            return self.finish({"success": False, "reason": "three invalid answers", "steps": step + 1})
                        continue
                    action = decision.action(self.arm)
                    if action.name == "DONE":                       # Claude judged it from the images: no confirmation needed
                        self.log(f"step {step}: {esc}; {self.vlm.name}: DONE ({decision.reasoning[:80]})")
                        stage_i, advanced = self.advance(stages, stage_i, step, packet, stage, state, extra, "planner said done")
                        history, recovery, op_notes, done_denied = [], None, [], 0
                        if advanced is not None:
                            return advanced
                        continue
            else:
                action = parse_action(token)
                self.counts["decider_steps"] += 1
            done_denied = 0 if action.name != "STILL" else done_denied
            extra["decided_by"] = by
            self.log(f"step {step} [{stage['id']}] {by} {resp.latency_s:.2f}s: {action.raw}" + (f"  (escalated: {esc})" if esc else ""))
            op_notes = []; recovery = None

            # -- execute, then the same bookkeeping as harness.loop --------------------------------------------------
            sigma, theta = step_size(cfg["steps"], bool(self.snap and self.snap.wrist_sees_target))
            proposal = self.interp.propose(state, action, sigma, theta)
            result = self.ex.execute(proposal, state)
            last = result
            if result.stream_error:
                self.record(step, packet, prompt, resp, stage, state, action, result, {"stream_error": result.stream_error, **extra})
                return self.finish({"success": False, "reason": f"move not sent: {result.stream_error}", "steps": step + 1})
            if result.ik_fail:
                ik_fails += 1; recovery = RECOVERY_NOTES["ik_fail"]
                if ik_fails >= int(lp["ik_fail_home_after"]):
                    hr = self.ex.home_step(self.ex.sync(), q_home)
                    if hr.stream_error:
                        return self.finish({"success": False, "reason": f"home step not sent: {hr.stream_error}", "steps": step + 1})
                    recovery = RECOVERY_NOTES["home_step"] + " " + hr.feedback
                    ik_fails = 0; extra["home_step"] = hr.feedback; self.snap = None
            elif result.ok:
                ik_fails = 0
            if result.asked and self.feedback is not None and not self.ex.gate.estop.is_set():
                self.feedback.add(task, stage["id"], action.raw.upper(), not result.declined, result.operator_note,
                                  hand_tip=state.p, height_cm=height_above_table_cm(state.p, self.table_z), mode=self.ex.backend.name)
            if result.declined:
                op_notes.append(RECOVERY_NOTES["rejected"].format(token=action.raw.upper(), why=f' with the note "{result.operator_note}"' if result.operator_note else ""))
            elif result.operator_note:
                op_notes.append(RECOVERY_NOTES["accepted_note"].format(token=action.raw.upper(), note=result.operator_note))
            sensed = None                                           # the hand's own report ends a GRASP / RELEASE stage
            hand_confirmed = result.hand_closed is (action.name == "GRASP")
            if (action.name in ("GRASP", "RELEASE") and result.ok and hand_confirmed
                    and not (result.empty_grasp or result.declined or result.ik_fail)):
                if action.name in (stage.get("motion") or stage["id"]).upper():
                    sensed = "the hand sensed a hold" if action.name == "GRASP" else "the hand opened"
                else:                                               # a grasp in another stage: let Claude judge the stage
                    self.confirm_why = f"the hand just {'closed on an object' if action.name == 'GRASP' else 'opened'}; is the stage complete?"
            if result.empty_grasp:
                recovery = RECOVERY_NOTES["empty_grasp"]
                stage_i = self.grasp_stage(stages, stage_i); self.snap = None
                self.ex.backend.hand(self.arm, False); extra["auto_release"] = True
            tok = action.raw.upper() + ("(empty)" if result.empty_grasp else "") + ("(unreachable)" if result.ik_fail else "") + ("(rejected)" if result.declined else "")
            history.insert(0, tok); history = history[:int(lp["history_len"])]
            self.log(f"   -> {result.feedback or 'ok'}")
            self.record(step, packet, prompt, resp, stage, state, action, result, {**extra, **({"stage_done": True, "why": sensed} if sensed else {})})
            if sensed:
                self.snap = None
                if stage_i + 1 >= len(stages):
                    return self.finish({"success": True, "reason": "all stages done", "steps": step + 1})
                stage_i += 1; history, recovery, op_notes, done_denied = [], None, [], 0
                self.log(f"stage done ({sensed}) -> {stages[stage_i]['id']}")
        return self.finish({"success": False, "reason": "max steps", "steps": int(lp["max_steps"])})

    def advance(self, stages, stage_i, step, packet, stage, state, extra, why):
        """Stage done: record it, move on, forget the snapshot. -> (new stage index, finished summary or None)"""
        self.record(step, packet, None, None, stage, state, Action("DONE", raw="DONE"), None, {"stage_done": True, "why": why, **extra})
        seen = self.snap is not None and self.snap.stage_complete and self.snap.step == step
        self.snap = self.snap.next if seen else None           # Claude's look already located the next stage's goal
        if stage_i + 1 >= len(stages):
            return stage_i, self.finish({"success": True, "reason": "all stages done", "steps": step + 1})
        self.log(f"stage done ({why}) -> {stages[stage_i + 1]['id']}")
        return stage_i + 1, None

    def finish(self, summary):
        # Keep the original claude_* metrics for consumers of older recordings.
        summary = {**summary, **self.counts, "planner_provider": self.vlm.name,
                   "planner_looks": self.counts["claude_looks"], "planner_steps": self.counts["claude_steps"]}
        return super().finish(summary)
