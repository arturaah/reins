import json
import math
import sys
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIM))

import walk_preview  # noqa: E402
from walk_preview import Trajectory, Waypoint  # noqa: E402


def test_length_and_endpoints():
    t = Trajectory([Waypoint(0, 0), Waypoint(3, 0), Waypoint(3, 4)])
    assert t.length == pytest.approx(7.0)
    assert t.pose_at(0)[:2] == pytest.approx((0, 0))
    assert t.pose_at(99)[:2] == pytest.approx((3, 4))


def test_heading_defaults_to_direction_of_travel():
    t = Trajectory([Waypoint(0, 0), Waypoint(0, 2)])
    assert t.pose_at(1.0)[2] == pytest.approx(math.pi / 2)


def test_explicit_yaw_interpolates_the_short_way():
    t = Trajectory([Waypoint(0, 0, yaw=math.radians(170)), Waypoint(1, 0, yaw=math.radians(-170))])
    assert abs(t.pose_at(0.5)[2]) == pytest.approx(math.pi)


def test_needs_two_waypoints():
    with pytest.raises(ValueError):
        Trajectory([Waypoint(0, 0)])


@pytest.mark.parametrize("plan", sorted((SIM / "plans").glob("walk_*.json")), ids=lambda p: p.stem)
def test_walk_plans_export_and_render(plan, tmp_path):
    out = tmp_path / "preview.json"
    walk_preview.run(plan, out, headless=True)
    export = json.loads(out.read_text())
    assert export["frame"] == "mujoco_world"
    first, last = export["samples"][0], export["samples"][-1]
    assert first["base_xyz_m"][:2] == pytest.approx(export["waypoints"][0]["xyz_m"][:2])
    assert last["base_xyz_m"][:2] == pytest.approx(export["waypoints"][-1]["xyz_m"][:2])
    assert out.with_suffix(".png").stat().st_size > 0
