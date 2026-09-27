"""Accept / reject before every move: the operator's answer and note reach the model, a rejected chunk is dropped."""
import json
import os
import subprocess
import sys

import pytest

from harness.executor import ArmExecutor
from harness.kinematics import ArmKinematics, ROOT
from harness.loop import Episode
from harness.perception import MockCameras, Perception
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


def test_reject_with_note_reaches_the_model_and_drops_the_chunk(rig):
    cfg, backend, ex, per = rig
    asked = []
    answers = ["too far left", True, True]
    def confirm(text, preview):
        asked.append((text, preview)); return answers.pop(0)
    ex.confirm = confirm
    vlm = ScriptedVLM(decisions=[
        {"decision": "MV_LEFT", "reasoning": "WRIST: NO", "plan": ["MV_LEFT", "MV_LEFT", "MV_LEFT"]},   # rejected: the chunk must not run
        {"decision": "MV_UP", "reasoning": "WRIST: YES"},
        {"decision": "DONE", "reasoning": "WRIST: YES"}])
    p0 = backend.tip("right").copy()
    s = Episode(cfg, vlm, ex, per, None, log=lambda *_: None).run("x")
    assert s["success"]
    assert len(asked) == 2 and asked[0][0].startswith("MV_LEFT") and asked[1][0].startswith("MV_UP")
    pv = asked[0][1]
    assert pv["arm"] == "right" and len(pv["frames"]) >= 20 and len(pv["frames"][0]) == 5 and "left_elbow_joint" in pv["joints"]
    prompts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert "rejected your last proposal (MV_LEFT)" in prompts[1] and 'note "too far left"' in prompts[1]
    assert "Recent moves, newest first: MV_LEFT(rejected)" in prompts[1]
    assert "rejected your last proposal" not in prompts[2]                        # the note is shown once; the history token stays
    tip = backend.tip("right")
    assert abs(tip[1] - p0[1]) < 1e-3 and tip[2] > p0[2] + 0.015                 # no left move happened, the up move did


def test_start_pose_rejection_is_reported(rig):
    cfg, backend, ex, per = rig
    ex.confirm = lambda text, preview: False
    r = ex.go_to_joints(cfg["robot"]["start_pose_rad"]["right"], "start pose")
    assert not r.ok and r.declined and r.feedback == "the operator rejected the start pose"
    ex.confirm = lambda text, preview: "not now"
    r = ex.go_to_joints(cfg["robot"]["start_pose_rad"]["right"], "start pose")
    assert r.declined and r.operator_note == "not now" and "not now" in r.feedback


def run_cli(args, stdin, tmp_path):
    cmd = [sys.executable, "-m", "harness", "--set", f"recorder.root={tmp_path}", *args]
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=180, cwd=ROOT,
                          env={**os.environ, "MUJOCO_GL": "cgl"})


def test_cli_proposal_protocol(tmp_path):
    """What the desktop window consumes: a PROPOSAL line per move, the preview plan file, Enter / n <note> / x answers."""
    preview = tmp_path / "preview.json"
    r = run_cli(["sim", "reach", "--vlm", "scripted", "--confirm", "--start-pose", "--preview", str(preview)], "n keep still\n", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PROPOSAL: start pose: hand" in r.stdout and "rejected the start pose: keep still" in r.stdout
    plan = json.loads(preview.read_text())
    assert plan["schema_version"] == 1 and plan["keyframes"][0]["time_s"] == 0.0 and len(plan["keyframes"]) > 20
    assert set(plan["keyframes"][0]["joint_targets_rad"]) == {f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")}
    assert "left_elbow_joint" in plan["held_joints_rad"] and plan["name"].startswith("start pose")
    r = run_cli(["sim", "reach", "--vlm", "scripted", "--confirm", "--start-pose", "--preview", str(preview)], "\n", tmp_path)
    assert r.returncode == 0 and "start pose: start pose done" in r.stdout, r.stdout + r.stderr
    r = run_cli(["sim", "reach", "--vlm", "scripted", "--confirm", "--start-pose"], "x\n", tmp_path)
    assert r.returncode == 0 and "E-STOP set" in r.stdout and "'reason': 'e-stop'" in r.stdout, r.stdout + r.stderr
