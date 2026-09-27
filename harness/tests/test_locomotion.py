"""Whole-body steps: tokens, proposals, the walk gate, the mock scene shift, the loop's use of them, and the streamer's command."""
import math
import threading
import time

import numpy as np
import pytest

from harness.actions import ActionError, parse_action, parse_decision
from harness.executor import ArmExecutor
from harness.interpreter import ArmState, Interpreter
from harness.kinematics import ArmKinematics
from harness.loop import Episode
from harness.perception import MockCameras, Perception
from harness.prompts import controller_prompt, planner_prompt
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from harness.vlm.scripted import ScriptedVLM

LIM = {"param_max_walk_m": 0.4, "param_max_turn_deg": 45.0}


def test_parse_walk_and_turn_tokens():
    a = parse_action("WALK_FWD"); assert (a.name, a.axis, a.sign, a.amount, a.mode) == ("WALK", "forward", 1, None, "unit") and a.is_walk
    a = parse_action("WALK_RIGHT"); assert (a.axis, a.sign) == ("left", -1)
    a = parse_action("TURN_LEFT"); assert (a.name, a.axis, a.sign) == ("TURN", "yaw", 1)
    a = parse_action("WALK back 30", LIM); assert (a.axis, a.sign, a.amount, a.mode) == ("forward", -1, 0.3, "param")
    assert parse_action("WALK forward 80", LIM).amount == 0.4                                   # capped
    a = parse_action("TURN -30", LIM); assert a.sign == -1 and abs(a.amount - math.radians(30)) < 1e-9
    assert parse_action("TURN 90", LIM).amount == math.radians(45)
    assert parse_action("WALK_FWD").opposite_of.raw == "" and parse_action("WALK_FWD").opposite_of.sign == -1
    with pytest.raises(ActionError): parse_action("WALK_FWD 20")
    with pytest.raises(ActionError): parse_action("WALK up 20")
    with pytest.raises(ActionError):                                                            # never inside a chunk
        parse_decision('{"decision": "WALK_FWD", "reasoning": "WRIST: NO", "plan": ["WALK_FWD", "WALK_FWD"]}')
    d = parse_decision('{"decision": "TURN_RIGHT", "reasoning": "WRIST: NO"}')
    assert d.action("right").name == "TURN"


def test_interpreter_walk_proposals():
    it = Interpreter([1, 0, 0], [0, 1, 0])
    st = ArmState(np.array([0.3, -0.15, 0.78]), 0.0)
    p = it.propose(st, parse_action("WALK_FWD"), 0.04, 0.2, walk_m=0.2, turn_rad=math.radians(20))
    assert p.kind == "walk" and p.walk == (0.2, 0.0, 0.0) and np.allclose(p.p, st.p)
    assert it.propose(st, parse_action("WALK_LEFT"), 0.04, 0.2, 0.2, 0.35).walk == (0.0, 0.2, 0.0)
    assert it.propose(st, parse_action("TURN_RIGHT"), 0.04, 0.2, 0.2, 0.35).walk == (0.0, 0.0, -0.35)
    assert it.propose(st, parse_action("WALK back 10", LIM), 0.04, 0.2).walk == (-0.1, 0.0, 0.0)


def test_walk_gate_caps_budget_and_enable(cfg):
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    g = SafetyGate(cfg, kin, None, live=False)
    assert not g.vet_walk(0.2, 0, 0).ok and "not enabled" in g.vet_walk(0.2, 0, 0).reason
    cfg["locomotion"]["enabled"] = True
    g = SafetyGate(cfg, kin, None, live=False)
    v = g.vet_walk(0.2, 0.0, 0.0)
    assert v.ok and abs(v.vx - 0.25) < 1e-9 and abs(v.duration_s - 0.8) < 1e-9 and v.vy == 0 and v.vyaw == 0
    v = g.vet_walk(1.0, 0.0, 0.0)
    assert v.ok and abs(v.dx - 0.4) < 1e-9 and "capped" in v.clamped[0]                       # per-command cap
    v = g.vet_walk(0.0, 0.0, math.radians(20))
    assert v.ok and abs(v.duration_s - math.radians(20) / 0.4) < 1e-9 and abs(v.vyaw * v.duration_s - math.radians(20)) < 1e-9
    v = g.vet_walk(0.0, 0.0, -math.radians(90))
    assert v.ok and abs(v.dyaw + math.radians(45)) < 1e-9
    g.walked_m = 2.9
    assert not g.vet_walk(0.2, 0, 0).ok and "budget" in g.vet_walk(0.2, 0, 0).reason
    g.estop.set()
    assert g.vet_walk(0.0, 0.0, 0.1).reason.startswith("ESTOP")


def test_mock_walk_moves_the_scene(cfg):
    cfg["hand"]["type"] = "virtual"
    b = MockBackend(cfg, render=False)
    c0 = b.cube_pos().copy(); t0 = b.tip("right").copy()
    od = b.walk(0.25, 0.0, 0.0, 0.8)
    assert od == {"dx": 0.2, "dy": 0.0, "dyaw": 0.0}
    assert np.allclose(b.cube_pos(), c0 - [0.2, 0, 0]) and np.allclose(b.tip("right"), t0)     # the scene came closer, the arm did not move
    assert np.allclose(b.base, [0.2, 0, 0])
    c1 = b.cube_pos().copy()
    b.walk(0.0, 0.0, math.pi / 2 / 1.0, 1.0)                                                   # turn left 90 deg
    assert np.allclose(b.cube_pos(), [c1[1], -c1[0], c1[2]], atol=1e-9)                        # what was ahead-left is now ahead-right... rotated
    assert np.allclose(b.base, [0.2, 0, math.pi / 2])


@pytest.fixture
def rig(cfg, tmp_path):
    cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path); cfg["locomotion"]["enabled"] = True
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    return cfg, backend, ex, Perception(cfg, "right", MockCameras(backend, "right", 160, 90))


def test_episode_walks_when_allowed_and_prompts_say_so(rig):
    cfg, backend, ex, per = rig
    asked = []
    ex.confirm = lambda text, preview: (asked.append((text, preview)), True)[1]
    plan = {"subgoals": [{"id": "approach", "target": "the block", "affordance": "block", "motion": "APPROACH",
                         "description": "walk until the block is within reach", "completion": "block within reach"},
                        {"id": "hover", "target": "the block", "affordance": "block top", "motion": "REACH",
                         "description": "hand above the block", "completion": "hand above the block"}]}
    vlm = ScriptedVLM(plan=plan, decisions=[{"decision": "TURN_LEFT", "reasoning": "WRIST: NO"}, {"decision": "WALK_FWD", "reasoning": "WRIST: NO"},
                                            {"decision": "WALK forward 50", "reasoning": "WRIST: NO"}, {"decision": "DONE", "reasoning": "WRIST: NO"},
                                            {"decision": "MV_UP", "reasoning": "WRIST: YES"}, {"decision": "DONE", "reasoning": "WRIST: YES"}])
    c0 = backend.cube_pos().copy()
    s = Episode(cfg, vlm, ex, per, None, log=lambda *_: None).run("walk to the block")
    assert s["success"]
    assert "THE ROBOT CAN WALK" in vlm.calls[0][1] and "APPROACH" in vlm.calls[0][1]
    prompts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert all("LOCOMOTION: the whole robot can step" in p and "WALK_FWD" in p for p in prompts)
    assert asked[0][0].startswith("TURN_LEFT: the WHOLE ROBOT steps: turn 20 deg left") and asked[0][1]["walk"][2] > 0
    assert asked[1][0].startswith("WALK_FWD: the WHOLE ROBOT steps: walk 20 cm forward at 0.25 m/s") and "frames" not in asked[1][1]
    assert "walk 40 cm forward" in asked[2][0]                                                # the parser already caps WALK <cm>
    assert abs(backend.base[2] - math.radians(20)) < 1e-9 and abs(np.hypot(*backend.base[:2]) - 0.6) < 1e-9
    assert "walked 20 cm forward" in prompts[2] or "Recent moves, newest first: WALK_FWD" in prompts[2]
    assert "the view has changed" in prompts[2]
    assert ex.gate.walked_m == pytest.approx(0.6)


def test_walking_needs_the_word_walk_in_the_task(rig):
    """Config enabled, but the task does not say walk: no WALK tokens, and a walk the model tries anyway is refused and explained."""
    cfg, backend, ex, per = rig
    logs = []
    vlm = ScriptedVLM(decisions=[{"decision": "WALK_FWD", "reasoning": "WRIST: NO"}, {"decision": "DONE", "reasoning": "WRIST: NO"}])
    Episode(cfg, vlm, ex, per, None, log=logs.append).run("go to the block")
    assert np.allclose(backend.base, 0) and ex.gate.walked_m == 0
    assert "walking stays off" in logs[0]
    assert "WALK_FWD" not in vlm.calls[0][1] and "THE ROBOT CAN WALK" not in vlm.calls[0][1]
    acts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert "WALK_FWD" not in acts[0].split("Choose exactly one action")[1]
    assert "only allowed when the task text itself says 'walk'" in acts[1]
    assert Episode.task_allows_walking("Walk to the chair") and Episode.task_allows_walking("please walk, then touch it")
    assert not Episode.task_allows_walking("walkway inspection") and not Episode.task_allows_walking("hand over the sidewalk sign")


def test_prompts_hide_walking_when_disabled(cfg):
    stage = {"id": "s", "target": "t", "affordance": "a", "motion": "REACH", "description": "d", "completion": "c"}
    pro = {"text": "x", "hand_state": "no hand"}
    assert "WALK_FWD" not in controller_prompt("task", stage, pro, [], None, cfg, "right")
    assert "WALK" not in planner_prompt("task", cfg, "right")
    cfg["locomotion"]["enabled"] = True
    p = controller_prompt("task", stage, pro, [], None, cfg, "right", locomotion=True)
    assert "WALK_FWD, WALK_BACK, WALK_LEFT, WALK_RIGHT, TURN_LEFT, TURN_RIGHT" in p and "move the body 20 cm" in p
    assert "THE ROBOT CAN WALK" in planner_prompt("task", cfg, "right")
    assert "THE ROBOT CAN WALK" not in planner_prompt("task", cfg, "right", locomotion=False)


def test_walk_refused_when_disabled_is_reported_to_the_model(cfg, tmp_path):
    cfg["recorder"]["root"] = str(tmp_path)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    per = Perception(cfg, "right", MockCameras(backend, "right", 160, 90))
    vlm = ScriptedVLM(decisions=[{"decision": "WALK_FWD", "reasoning": "WRIST: NO"}, {"decision": "DONE", "reasoning": "WRIST: NO"}])
    Episode(cfg, vlm, ex, per, None, log=lambda *_: None).run("walk to x")          # the task consents, the config does not
    assert np.allclose(backend.base, 0)
    assert "walking is not enabled" in [c[1] for c in vlm.calls if c[0] == "act"][1]


# ---- the streamer's side, with fakes for the loco service and the odometry topic
pytest.importorskip("unitree_sdk2py")
from harness.robot import arm_stream as am
from harness.tests.test_streamer_watchdog import FakePub, FakeReader


class FakeLoco:
    calls = []
    def SetVelocity(self, vx, vy, w, duration): FakeLoco.calls.append(("vel", vx, vy, w, duration))
    def StopMove(self): FakeLoco.calls.append(("stop",))


def test_streamer_walk_command(cfg, monkeypatch):
    monkeypatch.setattr(am, "LowStateReader", FakeReader); monkeypatch.setattr(am, "ChannelPublisher", FakePub)
    monkeypatch.setattr(am, "_odom_sub", lambda cb: None); monkeypatch.setattr(am, "_loco", lambda: FakeLoco())
    monkeypatch.setattr(am, "query_fsm", lambda: (811, "Start (balance control)"))
    cfg["locomotion"]["settle_s"] = 0.0
    st = am.Streamer(cfg, "lo0", log=lambda *a: None)
    r = st.dispatch("walk", {"vx": 0.25, "vy": 0.0, "vyaw": 0.0, "duration": 0.2})
    assert r["ok"] is False and "disabled" in r["error"] and FakeLoco.calls == []
    cfg["locomotion"]["enabled"] = True
    st = am.Streamer(cfg, "lo0", log=lambda *a: None)
    r = st.dispatch("walk", {"vx": 0.25, "vy": 0.0, "vyaw": 0.0, "duration": 0.2})
    assert r["ok"] and r["odom"] is None and FakeLoco.calls == [("vel", 0.25, 0.0, 0.0, 0.2), ("stop",)]     # no odometry topic: still stops
    FakeLoco.calls.clear()
    r = st.dispatch("walk", {"vx": 0.6, "vy": 0.0, "vyaw": 0.0, "duration": 0.2})
    assert r["ok"] is False and "over the cap" in r["error"] and FakeLoco.calls == []
    r = st.dispatch("walk", {"vx": 0.25, "vy": 0.0, "vyaw": 0.0, "duration": 5.0})
    assert r["ok"] is False and "duration" in r["error"]
    monkeypatch.setattr(am, "query_fsm", lambda: (4, "StandUp"))
    r = st.dispatch("walk", {"vx": 0.25, "vy": 0.0, "vyaw": 0.0, "duration": 0.2})
    assert r["ok"] is False and "FSM 811" in r["error"]
    # odometry: a walk of 0.2 m along the heading at yaw 90 deg -> dx 0.2 in the body frame
    monkeypatch.setattr(am, "query_fsm", lambda: (811, "Start"))
    st.odom = {"pos": [1.0, 1.0, 0.0], "yaw": math.pi / 2, "t": time.time()}
    def moved(): st.odom = {"pos": [1.0, 1.2, 0.0], "yaw": math.pi / 2 + 0.05, "t": time.time()}
    threading.Timer(0.1, moved).start()
    r = st.dispatch("walk", {"vx": 0.25, "vy": 0.0, "vyaw": 0.0, "duration": 0.3})
    assert r["ok"] and abs(r["odom"]["dx"] - 0.2) < 1e-6 and abs(r["odom"]["dy"]) < 1e-6 and abs(r["odom"]["dyaw"] - 0.05) < 1e-9
    FakeLoco.calls.clear(); st.walking = True; st.dispatch("freeze", {}); assert FakeLoco.calls == [("stop",)]  # e-stop stops a walk
