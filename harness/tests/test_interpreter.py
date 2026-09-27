import math

import numpy as np
import pytest

from harness.actions import parse_action
from harness.interpreter import ArmState, Interpreter, step_size

SIGMA, THETA = 0.02, math.radians(15)


def state(p=(0.3, -0.1, 0.8), roll=0.0):
    return ArmState(np.array(p, float), roll)


def test_unit_moves_in_view_frame_identity():
    it = Interpreter([1, 0, 0], [0, 1, 0])
    s = state()
    for tok, d in [("MV_FWD", [1, 0, 0]), ("MV_BACK", [-1, 0, 0]), ("MV_LEFT", [0, 1, 0]),
                   ("MV_RIGHT", [0, -1, 0]), ("MV_UP", [0, 0, 1]), ("MV_DOWN", [0, 0, -1])]:
        pr = it.propose(s, parse_action(tok), SIGMA, THETA)
        assert pr.kind == "move"
        assert np.allclose(pr.p - s.p, SIGMA * np.array(d))
        assert pr.roll == s.roll


def test_view_frame_rotates_directions():
    # a context camera facing the robot: image-forward is -x, image-left is -y
    it = Interpreter([-1, 0, 0], [0, -1, 0])
    pr = it.propose(state(), parse_action("MV_LEFT"), SIGMA, THETA)
    assert np.allclose(pr.p - state().p, [0, -SIGMA, 0])
    with pytest.raises(ValueError):
        Interpreter([2, 0, 0], [0, 1, 0])


def test_param_move_uses_its_amount():
    it = Interpreter([1, 0, 0], [0, 1, 0])
    pr = it.propose(state(), parse_action("MOVE up 7"), SIGMA, THETA)
    assert np.allclose(pr.p - state().p, [0, 0, 0.07]) and pr.mode == "param"


def test_rotation_composes_on_roll_only():
    it = Interpreter([1, 0, 0], [0, 1, 0])
    s = state(roll=0.1)
    pr = it.propose(s, parse_action("ROTATE_CW"), SIGMA, THETA)
    assert pr.kind == "rotate" and pr.roll == pytest.approx(0.1 + THETA) and np.allclose(pr.p, s.p)
    pr2 = it.propose(ArmState(pr.p, pr.roll), parse_action("ROTATE_CCW"), SIGMA, THETA)
    assert pr2.roll == pytest.approx(0.1)
    un = it.propose(s, parse_action("ROTATE_CW z"), SIGMA, THETA)
    assert un.kind == "unavailable" and "5-joint" in un.note


def test_hand_still_done():
    it = Interpreter([1, 0, 0], [0, 1, 0])
    assert it.propose(state(), parse_action("GRASP"), SIGMA, THETA).hand_closed is True
    assert it.propose(state(), parse_action("RELEASE"), SIGMA, THETA).hand_closed is False
    assert it.propose(state(), parse_action("STILL"), SIGMA, THETA).kind == "still"
    assert it.propose(state(), parse_action("DONE"), SIGMA, THETA).kind == "done"


def test_step_size_switching(cfg):
    st = dict(cfg["steps"]); st["profile"] = "coarse_fine"
    assert step_size(st, wrist_visible=False)[0] == pytest.approx(0.06)     # steps.coarse_m (raised 2026-09-27)
    assert step_size(st, wrist_visible=True)[0] == pytest.approx(0.03)      # steps.fine_m
    assert step_size(st, None)[0] == pytest.approx(0.06)
    st["profile"] = "precision"
    assert step_size(st, True) == (pytest.approx(0.01), pytest.approx(math.radians(5)))
