import numpy as np
import pytest

from harness.actions import parse_action
from harness.executor import ArmExecutor, interpolate
from harness.interpreter import Interpreter
from harness.kinematics import ArmKinematics
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend


@pytest.fixture
def rig(cfg):
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    gate = SafetyGate(cfg, kin, None, live=False)
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, gate, backend, "right")
    it = Interpreter(cfg["frames"]["view_forward"], cfg["frames"]["view_left"])
    return cfg, kin, gate, backend, ex, it


def test_interpolate_endpoints_and_velocity():
    fr = interpolate([0, 0, 0, 0, 0], [0.4, 0, 0, 0, 0], 1.0, 50)
    assert len(fr) == 50 and np.allclose(fr[-1], [0.4, 0, 0, 0, 0])
    v = max(abs(b[0] - a[0]) / 0.02 for a, b in zip(fr, fr[1:]))
    assert v <= 0.4 * np.pi / 2 + 1e-6


def test_move_round_trip_on_mock(rig):
    cfg, kin, gate, backend, ex, it = rig
    s = ex.sync()
    pr = it.propose(s, parse_action("MV_FWD"), 0.02, 0.1)
    r = ex.execute(pr, s)
    assert r.ok, r.feedback
    assert np.linalg.norm(r.achieved_dp - [0.02, 0, 0]) < 0.005
    assert "moved 2.0 of 2.0 cm" in r.feedback
    assert backend.frames_sent >= 20                                   # at least min_move_s at 50 Hz


def test_velocity_cap_respected_on_mock(rig):
    cfg, kin, gate, backend, ex, it = rig
    s = ex.sync()
    pr = it.propose(s, parse_action("MOVE up 15"), 0.02, 0.1)
    r = ex.execute(pr, s)
    assert r.ok, r.feedback
    n = backend.frames_sent
    assert r.duration_s >= 0.0 and n >= 20
    # every streamed frame stayed inside limits and under the speed cap (re-check the executor's own trajectory)
    fr = interpolate(kin.q_from_dict(s.q), r.q_target, n / 50.0, 50)
    assert gate.check_trajectory(fr, 0.02) == ""


def test_ik_fail_does_not_move(rig):
    cfg, kin, gate, backend, ex, it = rig
    s = ex.sync()
    s.p = np.array([0.58, -0.58, 1.20]); pr = it.propose(s, parse_action("MOVE forward 20"), 0.02, 0.1)
    before = backend.frames_sent
    r = ex.execute(pr, s)
    assert not r.ok and r.ik_fail and backend.frames_sent == before


def test_clamp_flag_and_table_floor(rig):
    cfg, kin, gate, backend, ex, it = rig
    s = ex.sync()
    # walk down until the floor clamps
    for _ in range(12):
        pr = it.propose(s, parse_action("MOVE down 5"), 0.02, 0.1)
        r = ex.execute(pr, s)
        s = ex.sync()
        if r.clamped:
            break
    assert r.clamped and "above the table" in r.feedback
    assert s.p[2] >= gate.floor_z - 0.006
    assert s.p[2] <= gate.floor_z + 0.006


def test_hand_none_and_virtual(rig):
    cfg, kin, gate, backend, ex, it = rig
    s = ex.sync()
    r = ex.execute(it.propose(s, parse_action("GRASP"), 0.02, 0.1), s)
    assert r.ok and "no hand" in r.feedback
    cfg["hand"]["type"] = "virtual"
    r = ex.execute(it.propose(s, parse_action("GRASP"), 0.02, 0.1), s)
    assert r.empty_grasp and r.feedback.startswith("EMPTY")


def test_settle_timeout_flag(rig, monkeypatch):
    cfg, kin, gate, backend, ex, it = rig
    monkeypatch.setattr(backend, "velocities", lambda: {n: 1.0 for n in kin.joint_names})
    cfg["limits"]["settle_timeout_s"] = 0.1
    s = ex.sync()
    r = ex.execute(it.propose(s, parse_action("MV_UP"), 0.02, 0.1), s)
    assert r.ok and r.timeout and "still moving" in r.feedback


@pytest.mark.parametrize("token", ["GRASP", "RELEASE"])
def test_hand_obeys_review_and_estop(rig, monkeypatch, token):
    cfg, kin, gate, backend, ex, it = rig
    calls = []
    monkeypatch.setattr(backend, "hand", lambda *args: calls.append(args) or "ok")
    s = ex.sync()
    proposal = it.propose(s, parse_action(token), 0.02, 0.1)
    ex.confirm = lambda *_: "wait"
    result = ex.execute(proposal, s)
    assert result.declined and result.asked and result.operator_note == "wait" and not calls
    ex.confirm = lambda *_: (True, "ready")
    result = ex.execute(proposal, s)
    assert result.ok and result.asked and result.operator_note == "ready" and len(calls) == 1
    gate.estop.set()
    result = ex.execute(proposal, s)
    assert not result.ok and len(calls) == 1
