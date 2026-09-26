import math

import numpy as np
import pytest

from harness.kinematics import ArmKinematics
from harness.safety import SafetyGate

REST = np.array([0.16, -0.02, 0.53, 1.45, -0.05])


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics("sim/models/r1/scene_fixed_base.xml", "right")


@pytest.fixture
def gate(cfg, kin):
    return SafetyGate(cfg, kin, None, live=False)


def test_live_gate_needs_table_height(cfg, kin):
    with pytest.raises(RuntimeError):
        SafetyGate(cfg, kin, None, live=True)
    SafetyGate(cfg, kin, 0.70, live=True)


def test_translation_and_rotation_caps(gate):
    p0 = np.array([0.35, -0.15, 0.80])
    p, roll, notes = gate.clamp_setpoint(p0, 0.0, p0 + [0.20, 0, 0], math.radians(45))
    assert np.linalg.norm(p - p0) == pytest.approx(0.05)
    assert roll == pytest.approx(math.radians(20))
    assert any("translation capped" in n for n in notes) and any("rotation capped" in n for n in notes)
    p, roll, notes = gate.clamp_setpoint(p0, 0.0, p0 + [0.15, 0, 0], 0.0, mode="param")
    assert np.linalg.norm(p - p0) == pytest.approx(0.15) and not notes


def test_box_and_table_floor(gate):
    p0 = np.array([0.35, -0.15, 0.70])
    p, _, notes = gate.clamp_setpoint(p0, 0.0, p0 + [0, 0, -0.05], 0.0)
    assert p[2] == pytest.approx(gate.floor_z) and any("above the table" in n for n in notes)
    edge = np.array([0.58, -0.15, 0.80])
    p, _, notes = gate.clamp_setpoint(edge, 0.0, edge + [0.05, 0, 0], 0.0)
    assert p[0] == pytest.approx(gate.box_max[0]) and any("workspace box" in n for n in notes)


def test_vet_returns_joint_target_for_reachable_step(gate, kin):
    p0, _ = kin.fk(REST)
    v = gate.vet(p0, REST[4], p0 + [0.02, 0, 0], REST[4], REST)
    assert v.ok and v.q_target is not None and v.duration_s >= 0.4
    assert v.duration_s >= float(np.abs(v.q_target - REST).max()) / 0.8


def test_vet_ik_fail_and_estop(gate, kin):
    p0, _ = kin.fk(REST)
    v = gate.vet(p0, 0.0, p0 + [0.05, 0, 0], 0.0, REST, mode="param")   # reachable
    assert v.ok
    far = gate.vet(np.array([0.55, -0.55, 1.2]), 0.0, np.array([0.60, -0.60, 1.25]), 0.0, REST, mode="param")
    assert not far.ok and far.reason.startswith("IK_FAIL") and far.q_target is None
    gate.estop.set()
    assert gate.vet(p0, 0.0, p0, 0.0, REST).reason.startswith("ESTOP")


def test_trajectory_check_catches_speed(gate):
    frames = [REST, REST + [0.5, 0, 0, 0, 0]]
    assert "cap" in gate.check_trajectory(frames, 0.02)
    assert gate.check_trajectory([REST, REST + [0.01, 0, 0, 0, 0]], 0.02) == ""
