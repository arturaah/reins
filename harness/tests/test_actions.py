import json
import math

import pytest

from harness.actions import Action, ActionError, parse_action, parse_decision

LIM = {"param_max_translation_m": 0.20, "param_max_rotation_deg": 90.0}


def dec(**kw):
    return json.dumps(kw)


def test_unit_moves():
    a = parse_action("MV_FWD")
    assert (a.name, a.axis, a.sign, a.amount, a.mode) == ("MOVE", "forward", 1, None, "unit")
    assert parse_action("mv_down").sign == -1 and parse_action("mv_down").axis == "up"
    assert parse_action("MV_RIGHT").axis == "left" and parse_action("MV_RIGHT").sign == -1


def test_rotate_default_axis_is_roll():
    a = parse_action("ROTATE_CW")
    assert (a.name, a.axis, a.sign) == ("ROTATE", "roll", 1)
    assert parse_action("ROTATE_CCW z").axis == "z"
    with pytest.raises(ActionError):
        parse_action("ROTATE_CW q")


def test_param_forms_are_clipped():
    a = parse_action("MOVE left 35", LIM)
    assert a.mode == "param" and a.axis == "left" and a.sign == 1 and a.amount == pytest.approx(0.20)
    b = parse_action("MOVE down -3", LIM)
    assert b.axis == "up" and b.sign == 1 and b.amount == pytest.approx(0.03)      # down of -3 cm is up 3 cm
    r = parse_action("ROTATE roll -120", LIM)
    assert r.sign == -1 and r.amount == pytest.approx(math.radians(90))
    assert parse_action("POINT down").name == "POINT"


@pytest.mark.parametrize("bad", ["", "FLY", "MV_FWD MV_UP", "MV_FWD 3", "MOVE x", "MOVE sideways 3", "ROTATE roll",
                                 "ROTATE roll ten", "GRASP now", "POINT up"])
def test_bad_tokens(bad):
    with pytest.raises(ActionError):
        parse_action(bad, LIM)


def test_decision_single_arm():
    d = parse_decision(dec(decision="MV_UP", reasoning="WRIST: NO. The block is far below the hand."))
    assert d.action("right").name == "MOVE" and d.wrist_visible is False and not d.plan


def test_decision_fenced_and_wrist_yes():
    txt = "```json\n" + dec(decision="GRASP", reasoning="WRIST: YES, the block is centred.") + "\n```"
    d = parse_decision(txt)
    assert d.action("right").name == "GRASP" and d.wrist_visible is True


def test_decision_plan_chunk():
    d = parse_decision(dec(decision="MV_FWD", reasoning="WRIST: NO", plan=["MV_FWD", "MV_FWD", "MV_DOWN"]))
    assert [p.raw for p in d.plan] == ["MV_FWD", "MV_FWD", "MV_DOWN"]
    with pytest.raises(ActionError):                                            # plan must start with the decision
        parse_decision(dec(decision="MV_FWD", reasoning="x", plan=["MV_UP"]))
    with pytest.raises(ActionError):                                            # no GRASP inside a chunk
        parse_decision(dec(decision="MV_FWD", reasoning="x", plan=["MV_FWD", "GRASP"]))


@pytest.mark.parametrize("bad", ["not json", "[1,2]", '{"reasoning": "x"}', '{"decision": ["MV_FWD", "MV_UP"]}',
                                 '{"decision": "MV_FWD", "extra": 1}', '{"decision": {"left": "MV_FWD"}}',
                                 '{"decision": "MV_FWD MV_UP"}', '{"decision": "JUMP"}'])
def test_decision_rejects(bad):
    with pytest.raises(ActionError):
        parse_decision(bad)


def test_decision_dual_arm():
    d = parse_decision(dec(decision={"left": "STILL", "right": "MV_LEFT"}, reasoning="x"), arms=("left", "right"))
    assert d.action("left").name == "STILL" and d.action("right").axis == "left"
    with pytest.raises(ActionError):
        parse_decision(dec(decision="MV_LEFT", reasoning="x"), arms=("left", "right"))


def test_opposite():
    assert parse_action("MV_FWD").opposite_of.same_direction(parse_action("MV_BACK"))
    assert parse_action("GRASP").opposite_of is None
