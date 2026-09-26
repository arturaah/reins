import math
import sys
from pathlib import Path

import pytest

LOCO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LOCO))
sys.path.insert(0, str(LOCO.parent / "contract"))

from reins_loco import unitree  # noqa: E402
from reins_loco.base import ERR_BLOCKED, ERR_NOT_WALKING, FSM_WALK, OK, Limits, Pose2  # noqa: E402
from reins_loco.follower import drive, PathFollower  # noqa: E402
from reins_loco.sim import SimLoco  # noqa: E402
from reins_loco.skills import (TOOLS, execute_walk_step, plan_tool_call,  # noqa: E402
                               plan_walk_to)


def run(sim, step):
    return execute_walk_step(step, sim, tick=sim.advance, timeout=60)


def test_velocity_needs_walk_mode_and_expires():
    sim = SimLoco()
    assert sim.set_velocity(0.5, 0, 0) == ERR_NOT_WALKING
    sim.stance(), sim.start()
    assert sim.set_velocity(0.5, 0, 0, duration=1.0) == OK
    sim.advance(3.0)
    assert sim.velocity() == (0.0, 0.0, 0.0), "command must expire after its duration"
    assert 0.3 < sim.pose().x < 0.6


def test_limits_clamp_commands():
    sim = SimLoco(limits=Limits(vx_forward=0.2))
    sim.stance(), sim.start()
    sim.set_velocity(5.0, 0, 0, duration=10)
    sim.advance(3.0)
    assert sim.velocity()[0] == pytest.approx(0.2)


def test_walks_around_the_table_without_touching_it():
    sim = SimLoco()
    clearances = []
    step = plan_walk_to(sim.pose(), 1.8, 1.8, yaw=math.pi / 2, via=[[1.2, 0], [1.8, 0.8]])
    result = execute_walk_step(step, sim, tick=lambda dt: (sim.advance(dt), clearances.append(sim.clearance())))
    assert result.reached
    assert math.hypot(result.pose.x - 1.8, result.pose.y - 1.8) < 0.1
    assert abs(result.pose.yaw - math.pi / 2) < 0.1
    assert sim.collisions == 0 and min(clearances) > 0.1


def test_straight_line_through_the_table_is_stopped():
    sim = SimLoco()
    result = run(sim, plan_walk_to(sim.pose(), 1.8, 1.8))
    assert not result.reached and result.reason == f"loco error {ERR_BLOCKED}"
    assert sim.collisions == 1 and sim.clearance() >= -1e-6


def test_robot_frame_walk_and_turn_in_place():
    sim = SimLoco(start=Pose2(0, 0, math.pi / 2))
    assert run(sim, plan_tool_call(sim.pose(), "walk", {"forward_m": 1.0})).reached
    assert sim.pose().x == pytest.approx(0, abs=0.1) and sim.pose().y == pytest.approx(1.0, abs=0.1)
    assert run(sim, plan_tool_call(sim.pose(), "turn", {"angle_rad": -math.pi / 2})).reached
    assert sim.pose().yaw == pytest.approx(0, abs=0.1)


def test_abort_stops_the_robot():
    sim = SimLoco()
    step = plan_walk_to(sim.pose(), 3, 0)  # clear of all obstacles
    result = execute_walk_step(step, sim, tick=sim.advance, should_stop=lambda: sim.time > 1.0)
    assert result.reason == "stopped"
    sim.advance(2.0)
    assert sim.velocity() == (0.0, 0.0, 0.0)


def test_skill_steps_satisfy_the_contract():
    from reins_contract import validate
    pose = Pose2(0.2, -0.1, 0.3)
    calls = [("walk_to", {"x": 1, "y": 2, "yaw": 1.0, "via": [[0.5, 1]]}),
             ("walk_to", {"x": 1, "y": 0, "frame": "robot"}),
             ("walk", {"forward_m": 0.5, "left_m": 0.2}),
             ("turn", {"angle_rad": 1.2})]
    steps = [plan_tool_call(pose, name, args, step_id=f"s{i}") for i, (name, args) in enumerate(calls)]
    validate({"type": "plan_proposed", "id": "m-1", "t": 0.0,
              "plan": {"plan_id": "p-1", "revision": 1, "command_id": "c-1", "summary": "walk around", "steps": steps}})
    assert plan_tool_call(pose, "stop", {}) is None
    assert {t["name"] for t in TOOLS} == {"walk_to", "walk", "turn", "stop"}


def test_real_backend_dry_run_dead_reckons(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(unitree.time, "monotonic", lambda: clock[0])
    robot = unitree.UnitreeLoco(live=False)
    assert robot.set_velocity(0.5, 0, 0, duration=0.5) == OK  # dry run accepts, sends nothing
    assert robot.stance() == OK and robot.start() == OK and robot.fsm_id() == FSM_WALK
    robot.set_velocity(0.5, 0, 0, duration=1.0)
    clock[0] += 3.0  # command expired after 1 s
    assert robot.pose().x == pytest.approx(0.5)
    assert robot.pose_is_estimated()


def test_same_follower_drives_the_real_backend(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(unitree.time, "monotonic", lambda: clock[0])
    robot = unitree.UnitreeLoco(live=False)
    robot.stance(), robot.start()

    def tick(dt):
        clock[0] += dt

    result = drive(robot, PathFollower([(0, 0), (1, 0)], None, robot.limits), tick=tick)
    assert result.reached and robot.pose().x == pytest.approx(1.0, abs=0.1)
