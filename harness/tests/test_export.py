"""An episode's accepted moves become a recording with a contact sheet, usable as a demonstration and by arm_lift."""
import json

import numpy as np

from harness.demos import load_demos
from harness.executor import ArmExecutor
from harness.kinematics import ArmKinematics
from harness.loop import Episode
from harness.perception import MockCameras, Perception
from harness.recorder import Recorder
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from harness.vlm.scripted import ScriptedVLM


def test_export_recording_from_sim_episode(cfg, tmp_path):
    cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    per = Perception(cfg, "right", MockCameras(backend, "right", 160, 90))
    ex.confirm = lambda text, preview: True                                             # accept everything
    vlm = ScriptedVLM(decisions=[{"decision": "MV_UP", "reasoning": "WRIST: YES"}, {"decision": "MOVE forward 12", "reasoning": "WRIST: YES"},
                                 {"decision": "MV_LEFT", "reasoning": "WRIST: YES"}, {"decision": "DONE", "reasoning": "WRIST: YES"}])
    rec = Recorder(cfg, "sim", "lift and push")
    s = Episode(cfg, vlm, ex, per, rec, log=lambda *_: None).run("lift and push")
    assert s["success"]
    path, msg = rec.export_recording("right", tmp_path / "recordings")
    assert path.name.startswith("ai_sim_lift_and_push_") and "3 accepted move(s)" in msg and "contact sheet" in msg
    d = json.loads(path.read_text())
    kfs = d["keyframes"]
    assert d["schema_version"] == 1 and d["arm"] == "right" and len(kfs) >= 3 * (8 + 10) and kfs[0]["time_s"] == 0.0   # 20 Hz: moves + holds
    assert all(set(k["joint_targets_rad"]) == set(kfs[0]["joint_targets_rad"]) for k in kfs)       # arm_lift needs the same joints in every frame
    assert "left_elbow_joint" in kfs[0]["joint_targets_rad"] and "waist_yaw_joint" in kfs[0]["joint_targets_rad"]
    assert "waist_roll_joint" not in kfs[0]["joint_targets_rad"]                                     # not on the arm topic: arm_lift refuses it
    times = [k["time_s"] for k in kfs]
    assert times == sorted(times) and times[-1] >= 3 * cfg["limits"]["min_move_s"] + 2 * 0.5           # three moves at least min_move_s each, holds between
    assert kfs[8]["joint_targets_rad"]["right_elbow_joint"] != kfs[0]["joint_targets_rad"]["right_elbow_joint"]   # the first move changed the arm
    steps_rad = [max(abs(a["joint_targets_rad"][k] - b["joint_targets_rad"][k]) for k in a["joint_targets_rad"]) for a, b in zip(kfs[:-1], kfs[1:])]
    assert max(steps_rad) / 0.05 <= cfg["limits"]["max_joint_vel_rad_s"] * 1.6 + 1e-6                # eased: peak speed pi/2 x the cap's mean
    assert d["sheet"]["cameras"] == ["context", "right wrist"] and (tmp_path / "recordings" / (path.stem + ".sheet.jpg")).exists()
    demo = load_demos([path], cfg)[0]
    assert demo.image is not None and "right arm moved" in demo.text and "Contact sheet columns" in demo.text
    # replay compatibility with tools/arm_lift.py's loader: dense samples (median gap under 0.25 s) get the lead-in treatment
    assert np.median(np.diff(times)) < 0.25


def test_export_without_moves(cfg, tmp_path):
    cfg["recorder"]["root"] = str(tmp_path)
    rec = Recorder(cfg, "sim", "nothing")
    assert rec.export_recording("right", tmp_path / "r") == (None, "no executed moves to export")
