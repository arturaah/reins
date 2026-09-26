import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

CONTRACT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CONTRACT))

from reins_contract import (  # noqa: E402
    MESSAGE_TYPES, SCHEMA, ContractError, check_session, read_jsonl, validate)

SESSIONS = sorted((CONTRACT / "examples").glob("*.jsonl"))


def test_schema_is_valid_json_schema():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("path", SESSIONS, ids=lambda p: p.name)
def test_example_sessions(path):
    messages = read_jsonl(path)
    for message in messages:
        # The top-level oneOf must accept every message too; other languages use that.
        Draft202012Validator(SCHEMA).validate(message)
    check_session(messages)


def test_examples_cover_every_message_type():
    seen = {m["type"] for path in SESSIONS for m in read_jsonl(path)}
    assert seen == MESSAGE_TYPES


def _plan_message():
    return copy.deepcopy(next(m for m in read_jsonl(CONTRACT / "examples/session_reach.jsonl")
                              if m["type"] == "plan_proposed"))


def test_rejects_ragged_trajectory():
    m = _plan_message()
    m["plan"]["steps"][0]["trajectory"]["positions_rad"][1].append(0.1)
    with pytest.raises(ContractError, match="row"):
        validate(m)


def test_rejects_non_increasing_times():
    m = _plan_message()
    m["plan"]["steps"][0]["trajectory"]["times_s"] = [0.0, 1.0, 1.0]
    with pytest.raises(ContractError, match="increase"):
        validate(m)


def test_rejects_motor_indices_instead_of_joint_names():
    m = _plan_message()
    m["plan"]["steps"][0]["trajectory"]["joint_names"] = ["15", "18"]
    with pytest.raises(ContractError, match="joint_names"):
        validate(m)


def test_rejects_nan():
    m = json.loads(json.dumps(_plan_message()).replace("0.275", "NaN"))
    with pytest.raises(ContractError, match="non-finite"):
        validate(m)


def test_approving_a_stale_revision_must_be_refused():
    messages = read_jsonl(CONTRACT / "examples/session_reach.jsonl")
    without_error = [m for m in messages if m["type"] != "error"]
    with pytest.raises(ContractError, match="stale"):
        check_session(without_error)


def test_execute_must_carry_the_proposed_plan_unchanged():
    messages = read_jsonl(CONTRACT / "examples/session_reach.jsonl")
    execute = next(m for m in messages if m["type"] == "execute")
    execute["plan"]["steps"][0]["trajectory"]["positions_rad"][2] = [-1.2, 0.2]
    with pytest.raises(ContractError, match="never proposed"):
        check_session(messages)
