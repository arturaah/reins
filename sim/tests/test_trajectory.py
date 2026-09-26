import math

import pytest

from reins_sim.trajectory import Trajectory, Waypoint


def test_length_and_endpoints():
    t = Trajectory([Waypoint(0, 0), Waypoint(3, 0), Waypoint(3, 4)])
    assert t.length == pytest.approx(7.0)
    assert t.pose_at(0)[:2] == pytest.approx((0, 0))
    assert t.pose_at(t.length)[:2] == pytest.approx((3, 4))
    assert t.pose_at(99)[:2] == pytest.approx((3, 4))


def test_heading_defaults_to_direction_of_travel():
    t = Trajectory([Waypoint(0, 0), Waypoint(0, 2)])
    assert t.pose_at(1.0)[2] == pytest.approx(math.pi / 2)


def test_explicit_yaw_interpolates_the_short_way():
    t = Trajectory([Waypoint(0, 0, yaw=math.radians(170)), Waypoint(1, 0, yaw=math.radians(-170))])
    _, _, yaw = t.pose_at(0.5)
    assert abs(yaw) == pytest.approx(math.pi)


def test_needs_two_waypoints():
    with pytest.raises(ValueError):
        Trajectory([Waypoint(0, 0)])


def test_example_plans_render(tmp_path):
    from pathlib import Path

    from reins_sim.preview import Preview

    for plan in (Path(__file__).parent.parent / "examples").glob("*.json"):
        Preview(Trajectory.load(plan)).save(tmp_path / f"{plan.stem}.png", 320, 180, fps=10)
        assert (tmp_path / f"{plan.stem}.png").stat().st_size > 0
