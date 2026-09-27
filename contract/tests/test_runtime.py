"""The live motion examples and schema agree with executable boundary checks."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator, ValidationError

from contract.runtime import digest, validate_approval, validate_motion

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT/"motion.schema.json").read_text())
EXAMPLES = sorted((ROOT/"runtime_examples").glob("*.json"))
NOW = 1800000000.


def test_motion_schema_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p:p.stem)
def test_examples_match_schema_payload_and_receipt(path):
    request = json.loads(path.read_text())
    Draft202012Validator(SCHEMA).validate(request)
    validate_motion(request["payload"])
    with patch("contract.runtime.time.time", return_value=NOW):
        validate_approval(request["approval"], digest(request["payload"]))


def test_examples_cover_all_motion_kinds():
    assert {json.loads(path.read_text())["payload"]["kind"] for path in EXAMPLES} == {"arm","walk","hand"}


def test_unknown_payload_fields_and_nonboolean_hand_are_refused():
    request = json.loads((ROOT/"runtime_examples/hand.json").read_text())
    for mutation in ({"closed":1}, {"execute_without_review":True}):
        bad = copy.deepcopy(request)
        bad["payload"].update(mutation)
        with pytest.raises(ValidationError): Draft202012Validator(SCHEMA).validate(bad)
        with pytest.raises(ValueError): validate_motion(bad["payload"])


def test_coupled_walk_limits_remain_runtime_checks():
    request = json.loads((ROOT/"runtime_examples/walk.json").read_text())
    for changes in ({"vx":.3,"vy":.3}, {"vx":.2,"duration_s":4}, {"vyaw":.5,"duration_s":2}):
        bad = copy.deepcopy(request)
        bad["payload"].update(changes)
        # Each scalar is legal; the vector/product limit requires runtime policy.
        Draft202012Validator(SCHEMA).validate(bad)
        with pytest.raises(ValueError): validate_motion(bad["payload"])


def test_changed_payload_and_expiry_cannot_reuse_receipt():
    request = json.loads((ROOT/"runtime_examples/hand.json").read_text())
    changed = {**request["payload"], "closed":True}
    with patch("contract.runtime.time.time", return_value=NOW):
        with pytest.raises(ValueError, match="changed"):
            validate_approval(request["approval"], digest(changed))
        for expiry in (NOW-1, NOW+181):
            with pytest.raises(ValueError, match="expir"):
                validate_approval({**request["approval"],"expires_at":expiry}, digest(request["payload"]))


def test_arm_timing_and_nested_nonfinite_values_are_rejected():
    payload = json.loads((ROOT/"runtime_examples/arm.json").read_text())["payload"]
    for field in ("time_s", "joint_targets_rad"):
        bad = copy.deepcopy(payload)
        if field == "time_s": bad["plan"]["keyframes"][1][field] = 0.
        else: bad["plan"]["keyframes"][0][field]["right_elbow_joint"] = float("nan")
        with pytest.raises(ValueError): validate_motion(bad)
