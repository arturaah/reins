"""Planner and controller prompts plus the plugin fragments. Pure text, no SDK.

Adapted from Show-Harness (arXiv 2609.10522, Appendix 7.4 and the repo's prompts/ and plugins/)
with RoboDawn's (arXiv 2609.22966) feedback phrasing. Deviations for this robot are marked in
DESIGN.md: the Unitree R1 A5 has 5-joint arms with no hand, so the end effector is the bare
hand tip, GRASP/RELEASE are pauses unless a hand is fitted, and the only rotation is wrist roll.
Direction conventions are repeated in EVERY controller call: they carry more grounding than the
action names (Show-Harness Fig. 10).
"""
import json

PLANNER = """ROLE: SubgoalPlanner
TASK: {task}

ROBOT: {robot_desc}

Return an ordered JSON plan:
{{"subgoals": [{{"id": "short_snake_case_id", "target": "object or destination",
  "affordance": "visible part or placement region", "motion": "semantic stage label",
  "description": "visual strategy for this stage",
  "completion": "visible condition that means this stage is complete"}}]}}

### STRICT RULES
#### 1. Stage Segmentation
- Break the task down into meaningful visual milestones (e.g., GRASP, LIFT, MOVE, PLACE, RELEASE, RETREAT)
- MERGE: Do NOT split immediate pre-grasp steps. Combine approach, align, lower, and close into a single 'GRASP' stage
- SEPARATE: Keep lift/clearance after a successful grasp as a separate 'LIFT' stage
- RETREAT: After every 'RELEASE', add a 'RETREAT' stage that lifts the hand up
{hand_rules}
#### 2. Affordance Selection
- ONE specific part -- never alternatives like "left end or middle"
- Must be visible in the CONTEXT VIEW
- Containers/Hollow objects (cups, bowls): Target left/right rim or edge.
- Simple Solid objects (blocks): main body
#### 3. Completion Criteria
- ALL completion conditions MUST be strictly judgeable from raw 2D images
- Movement stages: End with a stable visual spatial relation, NOT a hand event
- Set-aside placements: If moving an object to another location, ensure it is placed away from the origin point on the table
- MUST distinguish among similar objects
Return JSON ONLY."""

PLANNER_HAND_RULES_NONE = """- THIS ROBOT HAS NO HAND OR GRIPPER. It can only reach, touch, push and hover. Never plan a GRASP, LIFT or
  RELEASE stage; plan REACH / HOVER / TOUCH / PUSH stages whose completion is a visible spatial relation."""
PLANNER_LOCOMOTION = """- THE ROBOT CAN WALK for this task. The arm reaches about 45 cm from the shoulder; when the target is farther
  than that, plan an APPROACH stage first (motion label APPROACH) whose completion is "the target is within arm's reach
  in the context view", then the arm stages.
- If the task itself is a walking instruction with no arm work (e.g. "walk forward", "turn around", "go to the door"),
  plan ONLY walking stages (motion label WALK or APPROACH) with the distance or direction in the description and a
  completion the images can show (e.g. "the robot has advanced about 1 m: the near objects look clearly closer"), and
  no arm stages at all."""
LOCOMOTION_RULES = """LOCOMOTION: the whole robot can step. WALK_FWD / WALK_BACK / WALK_LEFT / WALK_RIGHT move the body {walk_cm:.0f} cm (the
hand comes along; its position relative to the body does not change), TURN_LEFT / TURN_RIGHT turn the body {turn_deg:.0f} deg,
or WALK <forward|back|left|right> <cm> (up to {walk_max_cm:.0f} cm in one command) and TURN <deg> (positive = left, up to
{turn_max_deg:.0f} deg). When the task or the stage names a distance or an angle, take it in ONE sized command ("walk 1 m
forward" -> WALK forward 100), not as a series of small steps. Use them ONLY when the TARGET is
out of the arm's reach: more than about 40 cm from the hand tip, or the arm keeps reporting unreachable targets. Face the
TARGET with turns, then WALK_FWD until it is within reach; WALK_BACK when too close. A walk is a single action, never in a
plan chunk, and the images change afterwards: judge again before the next action. Never walk while the hand is near an
object or a surface. IN A STAGE WHOSE MOTION IS APPROACH OR WALK, choose ONLY WALK_* / TURN_* / sized WALK or TURN, or DONE
when its completion is visible; never an arm move there."""

PLANNER_HAND_RULES_REVO2 = """- HAND: a five-finger hand that only opens or closes all fingers at once (a power grasp around the palm). Good for
  objects roughly 3 to 9 cm across (blocks, bottles, cups by the body); thin flat objects lying on the table cannot be
  picked up, plan PUSH stages for them instead. One object at a time.
- The GRASP stage's completion is the object visibly inside the closed fingers; LIFT's is the object clear of the table."""

CONTROLLER = """TASK: {task}
STAGE: {stage}
TARGET: {target}
AFFORD: {affordance}
Stage goal: {description}
DONE WHEN: {completion}
Hand now: {hand_state}
{mem_text}
{recovery}
{proprio}

{images_line}

DIRECTION (these conventions matter more than the action names):
Is TARGET inside the {wrist_label}?
A) YES -> the wrist view is the primary guide. Judge AFFORD's position vs the hand tip and take the direction of LARGEST deviation:
{wrist_rules}
B) NO -> the CONTEXT VIEW is the primary guide. Judge TARGET's position vs the hand tip and take the direction of LARGEST deviation:
- TARGET to the left in the image   -> MV_LEFT
- TARGET to the right in the image  -> MV_RIGHT
- TARGET lower in the image (nearer the robot) -> MV_BACK
- TARGET higher in the image (farther from the robot) -> MV_FWD
- TARGET below the hand tip -> MV_DOWN ; hand tip lower than the target or under the table edge -> MV_UP
C) MV_UP when:
- Need to lift the object
- too low to reach TARGET
- retreating after a RELEASE
{rotation}
ATTENTION:
{mem_rules}
- DONE only when "DONE WHEN" is already visible in the images; DONE ends this stage, not the task
{hand_block}
{locomotion}
{variable_step}
{action_chunk}
Think one visual sentence, then commit.
{output_contract}"""

WRIST_RULES_DEFAULT = """- AFFORD to the left in the wrist image  -> MV_LEFT
- AFFORD to the right in the wrist image -> MV_RIGHT
- AFFORD near the wrist image bottom (close to the hand) -> MV_DOWN
- AFFORD near the wrist image top (far from the hand)    -> MV_FWD
- AFFORD roughly centred and large -> the hand tip is over it: MV_DOWN"""

ROTATION_FRAGMENT = """- If the hand needs to spin about the forearm to align with the TARGET -> ROTATE_CW / ROTATE_CCW (wrist roll, the only rotation this arm has)"""

MEM_RULES = """- If recent moves show GRASP(empty), do not GRASP in place again; prioritize MV_UP, MV_BACK, MV_DOWN, or MV_FWD
- NEVER OSCILLATE: Do NOT choose the opposite of the newest recent move (Pairs: MV_LEFT/MV_RIGHT, MV_FWD/MV_BACK)
- When opposite directions appear in recent moves, prioritize MV_DOWN or MV_UP"""

HAND_BLOCK_GRIPPER = """HAND:
- GRASP when BOTH the CONTEXT VIEW and the wrist view confirm the {affordance} is clearly between the fingers
- RELEASE only when the held object is above its destination and lowered onto it"""
HAND_BLOCK_REVO2 = """HAND (five fingers, GRASP closes all of them, RELEASE opens all of them; nothing in between):
- The object is grasped against the palm: before GRASP the {affordance} must be between the open fingers and the
  palm in BOTH views, at palm height, not below the fingertips. If it is only near the fingertips -> move closer first
- If "Hand now" is closed and nothing is held while you still have to approach -> RELEASE first to open the hand
- After GRASP read "Last hand command": "closed on an object" = held, go on (usually MV_UP); GRASP(empty) = missed
- RELEASE only when the held object is above its destination and lowered onto it"""
HAND_BLOCK_NONE = """HAND: this robot has no hand. GRASP and RELEASE only pause the arm; do not use them."""

WRIST_MARKER = "WRIST CHECK: begin your reasoning with `WRIST: YES` if the TARGET is visible in the wrist view, else `WRIST: NO`."
ACTION_CHUNK = ("TRAJECTORY: give your next moves as \"plan\": [M1, M2, ...] (up to {n} arm moves: MV_*, sized MOVE, ROTATE_*) and set "
                "decision to M1. The operator sees the WHOLE trajectory drawn in the twin and accepts or rejects it as one, so make it a "
                "purposeful stretch of motion toward the stage goal (several moves, a sized MOVE where the way is free), not one small "
                "step; plan it from the height and the step sizes so it does not overshoot (never more MV_DOWN than the height above the "
                "table allows). When WRIST: YES and the target is within a few cm, one or two moves are enough.")

OUTPUT_CONTRACT = """Choose exactly one action:
{vocab}
Return JSON only: {{"decision":"ONE_ACTION","reasoning":"one visual sentence"}}"""
SIZED_MOVE_RULES = ("""STEP SIZE: MV_* moves ~{step_cm:.0f} cm. When the TARGET is clearly far from the hand tip and the way is free, take ONE sized
move instead of many small ones: MOVE <forward|back|left|right|up|down> <cm>, up to {cap_cm:.0f} cm (e.g. MOVE forward 15).
Near the target, or with anything in the way, use the small MV_* moves.""")
OUTPUT_CONTRACT_DUAL = """For EACH arm choose exactly one action from:
{vocab}
Return JSON only: {{"decision":{{"left":"ONE_ACTION","right":"ONE_ACTION"}},"reasoning":"one visual sentence"}}"""

RECOVERY_NOTES = {
    "empty_grasp": "Empty close; do not retry on an edge/corner. Recenter on the body and confirm depth.",
    "unverified_grasp": "Grasp not verified; continue GRASP until the hand reading and the images show a real hold.",
    "lost_grasp": "Grasp lost; return to GRASP, recenter the object body, then confirm depth.",
    "ik_fail": "The last target was out of reach for the arm; the arm did NOT move. Do not retry the same direction: "
               "come back toward the robot, lower or raise the hand, or try another direction.",
    "home_step": "After repeated unreachable targets the arm was moved part of the way toward its home pose.",
    "oscillation": "Opposite moves in a row: re-judge the images instead of hunting; prefer MV_UP or MV_DOWN.",
    "rejected": "The operator rejected your last proposal ({token}){why}. It was NOT executed: propose something different "
                "(another direction, a smaller step, or DONE if the stage is already complete).",
    "accepted_note": "The operator accepted your last move ({token}) and added the note: \"{note}\". Take it into account now.",
}


def robot_description(cfg, arm):
    hand = cfg["hand"]["type"]
    return (f"Unitree R1 humanoid, {arm} arm only (5 joints: shoulder pitch/roll/yaw, elbow, wrist roll). "
            + ("No hand or gripper is fitted: the end effector is the bare hand tip." if hand == "none"
               else f"A BrainCo Revo2 five-finger hand on the {arm} wrist, used open/close only: GRASP closes all fingers "
                    "around what is between them (a power grasp), RELEASE opens them." if hand == "revo2"
               else "A simple hand: open/close only.")
            + " The legs and balance are handled by the robot itself and are not controllable.")


def planner_prompt(task, cfg, arm, locomotion=None, pose_view=False):
    """locomotion: offer walking; None = whatever the config says (the loop passes its per-task decision).
    pose_view: the images include the ROBOT POSE VIEW rendering."""
    hand_rules = {"none": PLANNER_HAND_RULES_NONE, "revo2": PLANNER_HAND_RULES_REVO2}.get(cfg["hand"]["type"], "")
    if (cfg.get("locomotion") or {}).get("enabled", False) if locomotion is None else locomotion:
        hand_rules = (hand_rules + "\n" if hand_rules else "") + PLANNER_LOCOMOTION
    desc = robot_description(cfg, arm)
    if pose_view:
        desc += " IMAGES: the camera views, plus a " + POSE_LINE.format(arm=arm)
    return PLANNER.format(task=task, robot_desc=desc, hand_rules=hand_rules)


def mem_text(history):
    """history: newest first, list of strings like 'MV_FWD' or 'GRASP(empty)'."""
    return "Recent moves, newest first: " + (", ".join(history) if history else "none") if history is not None else ""


IMAGES_LINE = ("IMAGES: CONTEXT VIEW = the fixed camera on the robot's head looking at the workspace; {wrist_label} = the camera on\n"
               "the moving arm. The end effector is the {arm} hand tip ({ee_desc}).")
POSE_LINE = ("ROBOT POSE VIEW = a rendering of the robot's OWN current configuration from its measured joints (not a camera), seen\n"
             "from the front right: cyan sphere = the {arm} hand tip, yellow sphere = where the last move aimed, white box = the\n"
             "reachable workspace, grey plane = the table height. Use it to locate your hand when the cameras do not show it and to\n"
             "check the last move went where it aimed; the cameras remain the only source for where the TARGET is.")
IMAGES_LINE_NO_WRIST = ("IMAGES: only the CONTEXT VIEW (the fixed camera on the robot's head) is available this step; there is NO wrist\n"
                        "view, so judge everything from the context view, answer WRIST: NO, and take the direction of LARGEST deviation\n"
                        "of the hand tip from the TARGET. The end effector is the {arm} hand tip ({ee_desc}).")


def controller_prompt(task, stage, proprio, history, recovery, cfg, arm, dual=False, vocab=None, wrist_missing=False, locomotion=False,
                      pose_view=False):
    """stage: dict with target, affordance, motion, description, completion. proprio: text. history: list newest first.
    locomotion: the WALK_* / TURN_* tokens and their rules are offered (locomotion.enabled)."""
    hand = cfg["hand"]["type"]
    chunk = cfg["loop"]["chunk_max"]
    lo = cfg.get("locomotion") or {}
    wrist_label = f"{arm.upper()} WRIST VIEW"
    ee_desc = "the point 13 cm beyond the wrist" if hand == "none" else "between the fingers"
    images_line = (IMAGES_LINE_NO_WRIST if wrist_missing else IMAGES_LINE).format(wrist_label=wrist_label, arm=arm, ee_desc=ee_desc)
    if pose_view:
        images_line += "\n" + POSE_LINE.format(arm=arm)
    cap_cm = float(cfg["steps"].get("param_max_translation_m", 0.2)) * 100
    vocab = vocab or ("MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN, MOVE <forward|back|left|right|up|down> <cm>, "
                      "ROTATE_CW, ROTATE_CCW, STILL, DONE") + \
        ("" if hand == "none" else ", GRASP, RELEASE") + \
        (", WALK_FWD, WALK_BACK, WALK_LEFT, WALK_RIGHT, TURN_LEFT, TURN_RIGHT" if locomotion else "")
    loco_text = LOCOMOTION_RULES.format(walk_cm=float(lo.get("step_m", 0.2)) * 100, turn_deg=float(lo.get("turn_deg", 20.0)),
                                        walk_max_cm=float(lo.get("param_max_walk_m", 0.4)) * 100,
                                        turn_max_deg=float(lo.get("param_max_turn_deg", 45.0))) if locomotion else ""
    contract = (OUTPUT_CONTRACT_DUAL if dual else OUTPUT_CONTRACT).format(vocab=vocab)
    return CONTROLLER.format(
        task=task, stage=stage.get("motion") or stage.get("id", ""), target=stage.get("target", ""),
        affordance=stage.get("affordance", ""), description=stage.get("description", ""),
        completion=stage.get("completion", ""), hand_state=proprio.get("hand_state", "no hand"),
        mem_text=mem_text(history), recovery=("Recovery: " + recovery) if recovery else "",
        proprio=proprio.get("text", ""), wrist_label=wrist_label, arm=arm, images_line=images_line,
        wrist_rules=WRIST_RULES_DEFAULT, rotation=ROTATION_FRAGMENT, mem_rules=MEM_RULES,
        hand_block=HAND_BLOCK_NONE if hand == "none" else (HAND_BLOCK_REVO2 if hand == "revo2" else HAND_BLOCK_GRIPPER).format(affordance=stage.get("affordance", "")),
        locomotion=loco_text,
        variable_step=SIZED_MOVE_RULES.format(step_cm=float(cfg["steps"]["coarse_m"] if cfg["steps"]["profile"] != "precision" else cfg["steps"]["precision"]["step_m"]) * 100,
                                              cap_cm=cap_cm) + "\n" + WRIST_MARKER,
        action_chunk=ACTION_CHUNK.format(n=chunk) if chunk > 1 and not dual else "",
        output_contract=contract)


def proprio_text(height_cm, step_cm, hand_state, stall=None, clamped=None, ik_fail=None, holding=False, high_cm=8.0, hand_note=None):
    """Show-Harness proprioception block with RoboDawn-style outcome notes."""
    parts = [f"The hand tip is {height_cm:.1f} cm above the table; each step moves ~{step_cm:.0f} cm."]
    if holding:
        parts.append("Holding an object: lift until clear of the table; descend only to place.")
    elif height_cm > high_cm:
        parts.append(f"If height > {high_cm:.0f} cm, MV_DOWN first when the hand is over the target.")
    if stall:
        parts.append(stall)
    if clamped:
        parts.append("Last setpoint was clamped by the safety box: " + clamped + ".")
    if ik_fail:
        parts.append(ik_fail)
    if hand_note:
        parts.append(f"Last hand command: {hand_note}.")
    return {"text": " ".join(parts), "hand_state": hand_state}


def parse_plan(text):
    """Planner JSON -> list of subgoal dicts. Raises ValueError on a malformed plan."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    obj = json.loads(t)
    goals = obj.get("subgoals") if isinstance(obj, dict) else None
    if not isinstance(goals, list) or not goals:
        raise ValueError("plan must be {\"subgoals\": [...]} with at least one stage")
    out = []
    for i, g in enumerate(goals):
        if not isinstance(g, dict) or not all(k in g for k in ("target", "completion")):
            raise ValueError(f"subgoal {i} needs target and completion")
        g = dict(g); g.setdefault("id", f"stage_{i + 1}"); g.setdefault("motion", g["id"].upper())
        g.setdefault("affordance", g["target"]); g.setdefault("description", "")
        out.append(g)
    return out


PLAN_SCHEMA = {
    "type": "object",
    "properties": {"subgoals": {"type": "array", "items": {
        "type": "object",
        "properties": {k: {"type": "string"} for k in ("id", "target", "affordance", "motion", "description", "completion")},
        "required": ["id", "target", "affordance", "motion", "description", "completion"],
        "additionalProperties": False}}},
    "required": ["subgoals"], "additionalProperties": False,
}
