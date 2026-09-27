"""Validation for Reins contract messages (see README.md).

JSON Schema covers shape. This module adds the checks a schema can't express:
trajectory rows match their joint names, times strictly increase, numbers are
finite, and core state transitions are legal. Import it from the core, the
streamer or tests:

    from reins_contract import validate, ContractError
    validate(message)  # raises ContractError
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from jsonschema import Draft202012Validator

PROTOCOL = "reins/0.1"
SCHEMA = json.loads((Path(__file__).resolve().parent / "reins.schema.json").read_text())

# Legal core state transitions. Any state may also go to "halting" on abort.
TRANSITIONS = {
    "idle": {"planning"},
    "planning": {"proposed", "idle"},
    "proposed": {"executing", "planning", "idle"},
    "executing": {"idle"},
    "halting": {"idle"},
}


class ContractError(ValueError):
    pass


def validate(message: dict) -> None:
    """Check one message against the schema and the semantic rules."""
    kind = message.get("type")
    if kind not in MESSAGE_TYPES:
        raise ContractError(f"unknown message type: {kind!r}")
    # Validate against that type's definition alone so errors name the right field.
    errors = sorted(_validator_for(kind).iter_errors(message), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "(message)"
        raise ContractError(f"{kind}: {where}: {e.message}")
    _check_finite(message, kind)
    if "plan" in message:
        check_plan(message["plan"])


def check_plan(plan: dict) -> None:
    step_ids = [s["step_id"] for s in plan["steps"]]
    if len(set(step_ids)) != len(step_ids):
        raise ContractError(f"plan {plan['plan_id']}: duplicate step_id")
    for step in plan["steps"]:
        where = f"plan {plan['plan_id']} step {step['step_id']}"
        if step["kind"] == "servo":
            lo, hi = step["bounds"]["box_min_m"], step["bounds"]["box_max_m"]
            if any(a >= b for a, b in zip(lo, hi)):
                raise ContractError(f"{where}: servo box_min_m must be below box_max_m on every axis")
        if step["kind"] == "arm":
            tr = step["trajectory"]
            _check_times(tr["times_s"], where)
            if len(tr["positions_rad"]) != len(tr["times_s"]):
                raise ContractError(f"{where}: positions_rad has {len(tr['positions_rad'])} rows "
                                    f"for {len(tr['times_s'])} times")
            width = len(tr["joint_names"])
            if any(len(row) != width for row in tr["positions_rad"]):
                raise ContractError(f"{where}: every positions_rad row needs {width} values")
            for effector, path in step.get("preview", {}).get("effector_paths", {}).items():
                if "times_s" in path:
                    _check_times(path["times_s"], f"{where} {effector} path", start_at_zero=False)
                    if len(path["times_s"]) != len(path["points"]):
                        raise ContractError(f"{where}: {effector} path times_s and points differ in length")


def check_session(messages: Iterable[dict]) -> None:
    """Validate a recorded session in order, including core state and revision rules."""
    state = "idle"
    proposed: dict[str, dict] = {}  # plan_id -> latest proposed plan
    current: tuple[str, int] | None = None  # (plan_id, revision) awaiting decision
    pending_stale: str | None = None  # id of a stale decision awaiting its error
    for message in messages:
        validate(message)
        kind = message["type"]
        if pending_stale and kind != "heartbeat":
            if kind != "error" or message.get("ref") != pending_stale or message["code"] != "stale_revision":
                raise ContractError(f"stale decision {pending_stale} was not answered with stale_revision")
            pending_stale = None
            continue
        if kind == "state":
            new = message["state"]
            if new != "halting" and new not in TRANSITIONS[state]:
                raise ContractError(f"illegal transition {state} -> {new} ({message['id']})")
            state = new
            # Only a plan in "proposed" can be decided on.
            current = (message["plan_id"], message["revision"]) if new == "proposed" else None
        elif kind == "plan_proposed":
            proposed[message["plan"]["plan_id"]] = message["plan"]
        elif kind == "decision":
            if state != "proposed" or current != (message["plan_id"], message["revision"]):
                pending_stale = message["id"]
        elif kind == "execute":
            plan = message["plan"]
            if state != "executing":
                raise ContractError(f"execute {message['id']} sent while core is {state}")
            if proposed.get(plan["plan_id"]) != plan:
                raise ContractError(f"execute {message['id']} carries a plan that was never proposed as-is")
    if pending_stale:
        raise ContractError(f"stale decision {pending_stale} was not answered")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


MESSAGE_TYPES = frozenset(entry["$ref"].rsplit("/", 1)[1] for entry in SCHEMA["oneOf"])
_validators: dict[str, Draft202012Validator] = {}


def _validator_for(kind: str) -> Draft202012Validator:
    if kind not in _validators:
        schema = {key: value for key, value in SCHEMA.items() if key != "oneOf"}
        schema["$ref"] = f"#/$defs/{kind}"
        _validators[kind] = Draft202012Validator(schema)
    return _validators[kind]


def _check_times(times: list[float], where: str, start_at_zero: bool = True) -> None:
    if start_at_zero and times[0] != 0:
        raise ContractError(f"{where}: times_s must start at 0")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ContractError(f"{where}: times_s must strictly increase")


def _check_finite(value, where: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{where}: non-finite number")
    if isinstance(value, dict):
        for v in value.values():
            _check_finite(v, where)
    elif isinstance(value, list):
        for v in value:
            _check_finite(v, where)
