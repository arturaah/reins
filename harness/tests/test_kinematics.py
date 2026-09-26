import numpy as np
import pytest

from harness.kinematics import ArmKinematics

REST = np.array([0.16, -0.02, 0.53, 1.45, -0.05])     # the controller's standing pose, right arm
START = np.array([-0.05, -0.23, 0.13, 0.0, 0.0])      # harness start pose: forearm forward, 75% reach


@pytest.fixture(scope="module")
def kin(cfg=None):
    return ArmKinematics("sim/models/r1/scene_fixed_base.xml", "right")


def test_fk_matches_model_zero_pose(kin):
    p, R = kin.fk(np.zeros(5))
    assert np.allclose(p, [0.291, -0.139, 0.771], atol=2e-3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_wrist_roll_does_not_move_the_tip(kin):
    p0, _ = kin.fk(REST)
    p1, _ = kin.fk(REST + [0, 0, 0, 0, 0.8])
    assert np.allclose(p0, p1, atol=1e-9)


@pytest.mark.parametrize("delta", [[0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02], [-0.01, 0.01, 0.01], [0.05, 0, 0]])
def test_ik_round_trip_small_steps(kin, delta):
    p0, _ = kin.fk(START)
    res = kin.ik(p0 + delta, START[4], START)
    assert res.ok, res.reason
    p1, _ = kin.fk(res.q)
    assert np.linalg.norm(p1 - (p0 + delta)) < 0.005
    assert np.abs(res.q[:4] - START[:4]).max() < 0.4             # small Cartesian step, small joint step
    assert res.q[4] == pytest.approx(START[4])


def test_rest_pose_cannot_go_lower(kin):
    """The standing pose hangs the arm nearly straight: 2 cm straight down is out of reach (DESIGN.md)."""
    p0, _ = kin.fk(REST)
    assert not kin.ik(p0 + [0, 0, -0.02], REST[4], REST).ok


def test_ik_respects_limits_and_reports_failure(kin):
    res = kin.ik([1.5, -0.1, 0.8], 0.0, REST)                     # far outside reach
    assert not res.ok and res.err_m > 0.1
    assert not kin.joint_violations(res.q)                          # still inside the limits
    with_margin = kin.within_limits(kin.limits[:, 0], margin=0.0)
    assert with_margin == []


def test_waist_yaw_moves_the_hand(kin):
    p0, _ = kin.fk(REST)
    p1, _ = kin.fk(REST, {"waist_yaw_joint": 0.3})
    assert np.linalg.norm(p1 - p0) > 0.03
