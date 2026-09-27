"""A plan is ONE proposal: the whole trajectory is planned from the current state, shown and confirmed as one, executed
move by move, stopped where the arm is blocked, and every waypoint reaches the recording; a rejection drops it whole."""
import pytest

from harness.executor import ArmExecutor
from harness.kinematics import ArmKinematics
from harness.loop import Episode
from harness.perception import MockCameras, Perception
from harness.recorder import Recorder
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from harness.vlm.scripted import ScriptedVLM


@pytest.fixture
def rig(cfg, tmp_path):
    cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    return cfg, backend, ex, Perception(cfg, "right", MockCameras(backend, "right", 160, 90))


def test_plan_is_one_trajectory_proposal(rig, tmp_path):
    cfg, backend, ex, per = rig
    asked = []
    ex.confirm = lambda text, preview: (asked.append((text, preview)), True)[1]
    vlm = ScriptedVLM(decisions=[{"decision": "MV_UP", "reasoning": "WRIST: NO", "plan": ["MV_UP", "MV_UP", "MOVE forward 10"]},
                                 {"decision": "DONE", "reasoning": "WRIST: YES"}])
    p0 = backend.tip("right").copy()
    rec = Recorder(cfg, "sim", "traj")
    s = Episode(cfg, vlm, ex, per, rec, log=lambda *_: None).run("traj")
    assert s["success"] and len(asked) == 1                                       # one question for the three moves
    text, pv = asked[0]
    assert text.startswith("TRAJECTORY of 3 moves (MV_UP, MV_UP, MOVE FORWARD 10): hand ")
    assert len(pv["frames"]) >= 60 and pv["arm"] == "right"                       # the preview carries the whole path
    tip = backend.tip("right")
    assert tip[2] > p0[2] + 0.10 and tip[0] > p0[0] + 0.08                         # 2 x 6 cm up, 10 cm forward
    prompts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert "Recent moves, newest first: MOVE FORWARD 10, MV_UP, MV_UP" in prompts[1]
    assert "trajectory: 3 of 3 moves executed" in prompts[1] or "TRAJECTORY:" in prompts[1]
    assert "The operator sees the WHOLE trajectory" in prompts[0]
    path, msg = rec.export_recording("right", tmp_path)                            # every waypoint is in the recording
    assert path is not None and "3 accepted move(s)" in msg


def test_rejected_trajectory_moves_nothing(rig):
    cfg, backend, ex, per = rig
    asked = []
    ex.confirm = lambda text, preview: (asked.append(text), "too high")[1]
    vlm = ScriptedVLM(decisions=[{"decision": "MV_UP", "reasoning": "WRIST: NO", "plan": ["MV_UP", "MV_UP", "MV_UP"]},
                                 {"decision": "DONE", "reasoning": "WRIST: YES"}])
    p0 = backend.tip("right").copy()
    Episode(cfg, vlm, ex, per, None, log=lambda *_: None).run("traj")
    assert len(asked) == 1 and abs(backend.tip("right")[2] - p0[2]) < 1e-6
    prompts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert 'rejected your last proposal (MV_UP, MV_UP, MV_UP) with the note "too high"' in prompts[1]
    assert "Recent moves, newest first: MV_UP, MV_UP, MV_UP(rejected)" in prompts[1]


def test_trajectory_stops_where_the_arm_is_blocked(rig):
    """The second move does not happen (the mock arm stays put): the third is not sent and the model is told."""
    cfg, backend, ex, per = rig
    ex.confirm = lambda text, preview: True
    calls = {"n": 0}; real = backend.stream
    def stream(arm, frames, dt):
        calls["n"] += 1
        if calls["n"] != 2:
            real(arm, frames, dt)
    backend.stream = stream
    vlm = ScriptedVLM(decisions=[{"decision": "MV_UP", "reasoning": "WRIST: NO", "plan": ["MV_UP", "MV_UP", "MV_UP"]},
                                 {"decision": "DONE", "reasoning": "WRIST: YES"}])
    Episode(cfg, vlm, ex, per, None, log=lambda *_: None).run("traj")
    assert calls["n"] == 2                                                         # the third move was never sent
    prompts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert "stopped after move 2 (MV_UP): blocked or in contact" in prompts[1] and "2 of 3 moves executed" in prompts[1]


def test_unreachable_first_move_is_reported_like_a_single_move(rig):
    cfg, backend, ex, per = rig
    ex.confirm = lambda text, preview: True
    vlm = ScriptedVLM(decisions=[{"decision": "MOVE forward 20", "reasoning": "WRIST: NO", "plan": ["MOVE forward 20", "MOVE forward 20", "MOVE forward 20"]},
                                 {"decision": "DONE", "reasoning": "WRIST: YES"}])
    Episode(cfg, vlm, ex, per, None, log=lambda *_: None).run("traj")
    prompts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert "dropped" in prompts[1] or "unreachable" in prompts[1] or "clamped" in prompts[1]    # the gate had a say, in words
