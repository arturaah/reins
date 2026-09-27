"""ROBOT POSE VIEW: a rendering of the robot's own configuration goes with every call, next to the camera images."""
import io
import os

import numpy as np
import pytest
from PIL import Image

from harness.executor import ArmExecutor
from harness.kinematics import ArmKinematics
from harness.loop import Episode
from harness.perception import MockCameras, Perception
from harness.poseview import PoseView
from harness.prompts import controller_prompt, planner_prompt
from harness.recorder import Recorder, load_step
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from harness.vlm.scripted import ScriptedVLM

os.environ.setdefault("MUJOCO_GL", "cgl")


def test_pose_view_renders_the_measured_pose(cfg):
    pv = PoseView(cfg, "right", table_z=0.665, width=320, height=180, props=False)
    hang = {n: 0.0 for n in pv.adr}
    raised = dict(hang); raised.update(zip(["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"],
                                          cfg["robot"]["start_pose_rad"]["right"]))
    a, b = pv.render(hang), pv.render(raised, last_target=np.array([0.3, -0.15, 0.85]))
    if a is None:
        pytest.skip(f"no GL context: {pv.error}")
    assert a.size == (320, 180) and b.size == (320, 180)
    changed = (np.abs(np.asarray(a, int) - np.asarray(b, int)).sum(axis=2) > 30).mean()
    assert changed > 0.005                                                          # a different pose changes a visible part of the picture
    assert np.allclose(pv.tip(raised), ArmKinematics(cfg["robot"]["model"], "right").fk(cfg["robot"]["start_pose_rad"]["right"])[0], atol=1e-6)


def test_perception_adds_the_pose_view_and_prompts_explain_it(cfg):
    backend = MockBackend(cfg, render=False)
    per = Perception(cfg, "right", MockCameras(backend, "right", 160, 90), pose_view=PoseView(cfg, "right", 0.665, 160, 90, props=True))
    pk = per.capture(backend.tip("right"), joints=backend.joints(), last_target=backend.tip("right") + [0, 0, 0.04])
    labels = [l for l, _ in pk.images]
    if "ROBOT POSE VIEW" not in labels:
        pytest.skip("no GL context for the pose view")
    assert labels == ["CONTEXT VIEW", "RIGHT WRIST VIEW", "ROBOT POSE VIEW"]
    with Image.open(io.BytesIO(pk.images[-1][1])) as im:
        assert im.width == cfg["perception"]["width_px"]
    assert per.capture(backend.tip("right")).images[-1][0] != "ROBOT POSE VIEW"      # no joints given: no pose view
    stage = {"id": "s", "target": "t", "affordance": "a", "motion": "REACH", "description": "d", "completion": "c"}
    pro = {"text": "x", "hand_state": "no hand"}
    p = controller_prompt("task", stage, pro, [], None, cfg, "right", pose_view=True)
    assert "ROBOT POSE VIEW = a rendering of the robot's OWN current configuration" in p and "cyan sphere = the right hand tip" in p
    assert "ROBOT POSE VIEW" not in controller_prompt("task", stage, pro, [], None, cfg, "right")
    assert "ROBOT POSE VIEW" in planner_prompt("task", cfg, "right", pose_view=True) and "ROBOT POSE VIEW" not in planner_prompt("task", cfg, "right")


def test_episode_records_the_pose_view(cfg, tmp_path):
    cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    per = Perception(cfg, "right", MockCameras(backend, "right", 160, 90), pose_view=PoseView(cfg, "right", 0.665, 160, 90, props=True))
    if per.pose_view.render(backend.joints()) is None:
        pytest.skip("no GL context for the pose view")
    vlm = ScriptedVLM(decisions=[{"decision": "MV_UP", "reasoning": "WRIST: YES"}, {"decision": "DONE", "reasoning": "WRIST: YES"}])
    rec = Recorder(cfg, "sim", "pose")
    s = Episode(cfg, vlm, ex, per, rec, log=lambda *_: None).run("pose")
    assert s["success"]
    assert (rec.dir / "step_000" / "robot.jpg").exists() and (rec.dir / "plan_robot.jpg").exists()
    _, prompt, images = load_step(rec.dir, 1)
    assert [l for l, _ in images] == ["CONTEXT VIEW", "RIGHT WRIST VIEW", "ROBOT POSE VIEW"] and "ROBOT POSE VIEW" in prompt
    for kind, p, *_ in vlm.calls:
        assert "ROBOT POSE VIEW" in p
