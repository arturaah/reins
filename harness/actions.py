"""Action vocabulary, the JSON output contract and a strict parser. Pure Python.

Unit actions (one per arm and step):
  MV_FWD MV_BACK MV_LEFT MV_RIGHT MV_UP MV_DOWN   translate one step size
  ROTATE_CW [axis] ROTATE_CCW [axis]              rotate one angle step (this arm: wrist roll only)
  GRASP RELEASE STILL DONE
  WALK_FWD WALK_BACK WALK_LEFT WALK_RIGHT TURN_LEFT TURN_RIGHT   whole-body steps (only when locomotion is enabled)
Parameterized form (later experiments):
  MOVE <forward|back|left|right|up|down> <cm>     clipped to param_max_translation
  ROTATE <axis> <deg>                             signed, clipped to param_max_rotation
  POINT <down|forward|down45>                     orientation preset (not reachable on a 5-joint arm)
  WALK <forward|back|left|right> <cm>             clipped to locomotion.param_max_walk_m
  TURN <deg>                                      positive = left, clipped to locomotion.param_max_turn_deg

Output contract, JSON only:
  {"decision": "<ONE_ACTION>", "reasoning": "<one visual sentence>", "plan": ["MV_FWD", ...]}
Dual-arm mode: "decision": {"left": "<ACTION>", "right": "<ACTION>"}.
The reasoning may start with "WRIST: YES" or "WRIST: NO" (target visible in the active wrist camera).
"""
import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

MOVES = {"MV_FWD": ("forward", 1), "MV_BACK": ("forward", -1), "MV_LEFT": ("left", 1),
         "MV_RIGHT": ("left", -1), "MV_UP": ("up", 1), "MV_DOWN": ("up", -1)}
PARAM_DIRS = {"forward": ("forward", 1), "back": ("forward", -1), "left": ("left", 1),
              "right": ("left", -1), "up": ("up", 1), "down": ("up", -1)}
ROTATE_AXES = ("roll", "x", "y", "z")
POINT_PRESETS = ("down", "forward", "down45")
SIMPLE = ("GRASP", "RELEASE", "STILL", "DONE")
# what a model says for the hand when it forgets the vocabulary; the Action keeps its raw token for the history
HAND_ALIASES = {"GRAB": "GRASP", "CLOSE": "GRASP", "CLOSE_HAND": "GRASP", "GRIP": "GRASP", "PICK": "GRASP",
                "OPEN": "RELEASE", "OPEN_HAND": "RELEASE", "LET_GO": "RELEASE", "DROP": "RELEASE", "UNGRASP": "RELEASE"}
UNIT_VOCAB = list(MOVES) + ["ROTATE_CW", "ROTATE_CCW"] + list(SIMPLE)
WALKS = {"WALK_FWD": ("forward", 1), "WALK_BACK": ("forward", -1), "WALK_LEFT": ("left", 1), "WALK_RIGHT": ("left", -1)}
TURNS = {"TURN_LEFT": 1, "TURN_RIGHT": -1}
WALK_VOCAB = list(WALKS) + list(TURNS)


class ActionError(ValueError):
    """The model's output does not follow the contract. Reported back to the model once, then a failed step."""


@dataclass(frozen=True)
class Action:
    name: str                       # MOVE | ROTATE | GRASP | RELEASE | STILL | DONE | POINT | WALK | TURN
    axis: Optional[str] = None      # MOVE: forward|left|up ; ROTATE: roll|x|y|z ; POINT: preset
    sign: int = 0                   # MOVE/ROTATE direction, +1 or -1
    amount: Optional[float] = None  # MOVE: metres, ROTATE: radians (unsigned); None = one step size
    mode: str = "unit"              # unit | param
    raw: str = ""

    @property
    def key(self): return (self.name, self.axis, self.sign, self.amount, self.mode)
    @property
    def is_move(self): return self.name == "MOVE"
    @property
    def is_rotate(self): return self.name == "ROTATE"
    @property
    def opposite_of(self):
        """The action that undoes this one, for the anti-oscillation rule."""
        if self.name in ("MOVE", "ROTATE", "WALK", "TURN"):
            return Action(self.name, self.axis, -self.sign, self.amount, self.mode)
        return None

    @property
    def is_walk(self): return self.name in ("WALK", "TURN")

    def same_direction(self, other):
        return other is not None and self.name == other.name and self.axis == other.axis and self.sign == other.sign


@dataclass
class Decision:
    actions: dict                          # arm -> Action ("right": ..) ; single-arm mode has one key
    reasoning: str = ""
    plan: list = field(default_factory=list)   # chunk of Actions, first one equals the decision
    wrist_visible: Optional[bool] = None       # parsed from "WRIST: YES/NO"
    stage_complete: bool = False
    raw: str = ""

    def action(self, arm):
        return self.actions[arm]


def parse_action(token, limits=None):
    """One action string -> Action. limits: {"param_max_translation_m", "param_max_rotation_deg"}."""
    if not isinstance(token, str):
        raise ActionError(f"action must be a string, got {type(token).__name__}")
    parts = token.strip().upper().split()
    if not parts:
        raise ActionError("empty action")
    head, args = parts[0], parts[1:]
    head = HAND_ALIASES.get(head, head)
    lim = limits or {}
    if head in MOVES:
        if args:
            raise ActionError(f"{head} takes no arguments")
        axis, sign = MOVES[head]
        return Action("MOVE", axis, sign, None, "unit", token)
    if head in ("ROTATE_CW", "ROTATE_CCW"):
        if len(args) > 1:
            raise ActionError(f"{head} takes at most one axis")
        axis = args[0].lower() if args else "roll"
        if axis not in ROTATE_AXES:
            raise ActionError(f"unknown rotation axis {axis!r}; use roll, x, y or z")
        return Action("ROTATE", axis, 1 if head == "ROTATE_CW" else -1, None, "unit", token)
    if head in SIMPLE:
        if args:
            raise ActionError(f"{head} takes no arguments")
        return Action(head, raw=token)
    if head == "MOVE":
        if len(args) != 2 or args[0].lower() not in PARAM_DIRS:
            raise ActionError("MOVE needs <forward|back|left|right|up|down> <cm>")
        axis, sign = PARAM_DIRS[args[0].lower()]
        cm = _number(args[1], "cm")
        cap = float(lim.get("param_max_translation_m", 0.20))
        return Action("MOVE", axis, sign * (1 if cm >= 0 else -1), min(abs(cm) / 100.0, cap), "param", token)
    if head == "ROTATE":
        if len(args) != 2 or args[0].lower() not in ROTATE_AXES:
            raise ActionError("ROTATE needs <roll|x|y|z> <deg>")
        deg = _number(args[1], "deg")
        cap = math.radians(float(lim.get("param_max_rotation_deg", 90.0)))
        return Action("ROTATE", args[0].lower(), 1 if deg >= 0 else -1, min(abs(math.radians(deg)), cap), "param", token)
    if head == "POINT":
        if len(args) != 1 or args[0].lower() not in POINT_PRESETS:
            raise ActionError("POINT needs <down|forward|down45>")
        return Action("POINT", args[0].lower(), 0, None, "param", token)
    if head in WALKS:
        if args:
            raise ActionError(f"{head} takes no arguments")
        axis, sign = WALKS[head]
        return Action("WALK", axis, sign, None, "unit", token)
    if head in TURNS:
        if args:
            raise ActionError(f"{head} takes no arguments")
        return Action("TURN", "yaw", TURNS[head], None, "unit", token)
    if head == "WALK":
        if len(args) != 2 or args[0].lower() not in ("forward", "back", "left", "right"):
            raise ActionError("WALK needs <forward|back|left|right> <cm>")
        axis, sign = PARAM_DIRS[args[0].lower()]
        cm = _number(args[1], "cm")
        cap = float(lim.get("param_max_walk_m", 0.40))
        return Action("WALK", axis, sign * (1 if cm >= 0 else -1), min(abs(cm) / 100.0, cap), "param", token)
    if head == "TURN":
        if len(args) != 1:
            raise ActionError("TURN needs <deg> (positive = left)")
        deg = _number(args[0], "deg")
        cap = math.radians(float(lim.get("param_max_turn_deg", 45.0)))
        return Action("TURN", "yaw", 1 if deg >= 0 else -1, min(abs(math.radians(deg)), cap), "param", token)
    raise ActionError(f"unknown action {head!r}")


def _number(s, unit):
    try:
        value = float(s)
        if not math.isfinite(value):
            raise ValueError()
        return value
    except ValueError:
        raise ActionError(f"expected a number of {unit}, got {s!r}") from None


_WRIST = re.compile(r"^\s*WRIST\s*:\s*(YES|NO)\b", re.IGNORECASE)


def parse_decision(text, arms=("right",), limits=None, allow_plan=True):
    """Model output (JSON text, possibly fenced) -> Decision. Raises ActionError on anything off-contract."""
    body = _strip_fence(text)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise ActionError(f"not valid JSON: {e.msg}") from None
    if not isinstance(obj, dict):
        raise ActionError("output must be a JSON object")
    unknown = set(obj) - {"decision", "reasoning", "plan", "stage_complete"}
    if unknown:
        raise ActionError(f"unexpected keys: {sorted(unknown)}")
    if "decision" not in obj:
        raise ActionError('missing "decision"')
    dec = obj["decision"]
    actions = {}
    if len(arms) == 1:
        if isinstance(dec, dict):
            raise ActionError('single-arm mode: "decision" must be one action string')
        actions[arms[0]] = parse_action(dec, limits)
    else:
        if not isinstance(dec, dict) or set(dec) != set(arms):
            raise ActionError(f'dual-arm mode: "decision" must be an object with keys {list(arms)}')
        for arm in arms:
            actions[arm] = parse_action(dec[arm], limits)
    reasoning = obj.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ActionError('"reasoning" must be a string')
    m = _WRIST.match(reasoning)
    wrist = (m.group(1).upper() == "YES") if m else None
    plan = []
    if "plan" in obj and obj["plan"] is not None:
        if not allow_plan:
            raise ActionError('"plan" is not allowed on this step')
        if not isinstance(obj["plan"], list) or not all(isinstance(p, str) for p in obj["plan"]):
            raise ActionError('"plan" must be a list of action strings')
        if len(arms) != 1:
            raise ActionError('"plan" is only allowed in single-arm mode')
        plan = [parse_action(p, limits) for p in obj["plan"]]
        if plan and plan[0].key != actions[arms[0]].key:
            raise ActionError('"plan"[0] must equal "decision"')
        for p in plan:
            if p.name in ("GRASP", "RELEASE", "DONE", "WALK", "TURN"):
                raise ActionError('"plan" may only contain arm moves, rotations and STILL (a walk needs a fresh look each time)')
    stage_complete = bool(obj.get("stage_complete", False))
    return Decision(actions, reasoning, plan, wrist, stage_complete, text)


def _strip_fence(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Start with 'WRIST: YES' or 'WRIST: NO', then one visual sentence."},
        "decision": {"type": "string", "description": "Exactly one action from the vocabulary."},
        "plan": {"type": "array", "items": {"type": "string"},
                 "description": "The next moves as one trajectory, starting with the decision (arm moves and rotations only)."},
