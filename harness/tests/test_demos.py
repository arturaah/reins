"""Demonstrations: key moments, text summaries, the contact sheet as an image, and their place in every prompt."""
import io
import json

import numpy as np
import pytest
from PIL import Image

from harness.demos import Demo, demo_images, demos_block, key_moments, load_demos, smart_moments
from harness.executor import ArmExecutor
from harness.kinematics import ArmKinematics
from harness.loop import Episode
from harness.perception import MockCameras, Perception
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from harness.vlm.scripted import ScriptedVLM

RIGHT = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"]
LEFT = [n.replace("right", "left") for n in RIGHT]


def recording(path, n=40, name="reach_forward", sheet=None):
    """A right-arm reach: shoulder pitch swings -0.05 -> -0.9 over 4 s, the left arm holds still."""
    kfs = []
    for i in range(n):
        t = 4.0 * i / (n - 1)
        q = {**{k: 0.0 for k in LEFT}, **{k: 0.0 for k in RIGHT}}
        q["left_shoulder_roll_joint"] = 0.23; q["right_shoulder_roll_joint"] = -0.23
        q["right_shoulder_pitch_joint"] = -0.05 - 0.85 * (i / (n - 1))
        kfs.append({"time_s": round(t, 3), "joint_targets_rad": q})
    d = {"schema_version": 1, "name": name, "source": "kinesthetic teach, measured", "duration_s": 4.0, "keyframes": kfs}
    if sheet:
        d["sheet"] = sheet
    path.write_text(json.dumps(d))
    return path


def test_key_moments_ends_and_spacing():
    t = np.linspace(0, 10, 101)
    q = np.zeros((101, 3)); q[:, 0] = np.where(t < 5, 0.0, (t - 5) / 5)          # still for 5 s, then moves
    idx = key_moments(t, q, 6)
    assert idx[0] == 0 and idx[-1] == 100 and len(idx) <= 6 and idx == sorted(idx)
    assert all(i >= 50 for i in idx[1:])                                          # moments follow the motion, not the clock
    still = key_moments(t, np.zeros((101, 3)), 4)
    assert still[0] == 0 and still[-1] == 100 and len(still) == 4               # nothing moved: spread in time
    assert key_moments([0, 1, 2], np.zeros((3, 2)), 6) == [0, 1, 2]


def test_smart_moments_follow_the_path_shape():
    t = np.arange(50) * 0.1
    straight = np.zeros((50, 2)); straight[:, 0] = np.linspace(0, 1, 50)
    assert smart_moments(t, straight, 0.08, 8) == [0, 25, 49]                     # ends plus the arc-length middle
    corner = np.zeros((50, 2)); corner[:, 0] = 0.4 * np.minimum(np.arange(50), 25) / 25; corner[:, 1] = 0.4 * np.maximum(0, np.arange(50) - 25) / 24
    assert smart_moments(t, corner, 0.08, 8) == [0, 25, 49]                       # the turn is a moment
    zig = np.zeros((50, 2)); zig[:, 0] = np.linspace(0, 1, 50); zig[:, 1] = 0.3 * np.sin(np.arange(50) * 1.3)
    assert len(smart_moments(t, zig, 0.08, 5)) <= 5 and smart_moments(t, zig, 0.08, 5)[-1] == 49    # capped
    assert smart_moments(t, np.random.default_rng(0).normal(0, 0.002, (50, 2)), 0.08, 8) == [0, 24, 49]   # noise only: time spread
    assert smart_moments([0, 1], np.zeros((2, 3))) == [0, 1]


def test_demo_images_stacks_every_sheet_into_one(tmp_path):
    def jpg(w, h):
        b = io.BytesIO(); Image.new("RGB", (w, h), (5, 5, 5)).save(b, "JPEG"); return b.getvalue()
    demos = [Demo("a", "", ("DEMO_1 contact sheet of 'a'", jpg(800, 100)), label="DEMO_1"),
             Demo("b", "", None, label="DEMO_2"),
             Demo("c", "", ("DEMO_3 contact sheet of 'c'", jpg(400, 100)), label="DEMO_3")]
    out = demo_images(demos, max_w=1568)
    assert len(out) == 1 and out[0][0].startswith("DEMOS")
    with Image.open(io.BytesIO(out[0][1])) as im:
        assert im.width == 800 and im.height == (20 + 100 + 4) + (20 + 200)      # the narrow sheet is scaled to the common width
    assert demo_images([demos[1]]) == []
    wide = demo_images([Demo("w", "", ("x", jpg(3000, 300)), label="DEMO_1")], max_w=1568)
    with Image.open(io.BytesIO(wide[0][1])) as im:
        assert im.width == 1568


def test_load_demo_text_only(cfg, tmp_path):
    d = load_demos([recording(tmp_path / "reach.json")], cfg)[0]
    assert d.image is None and d.name == "reach_forward"
    assert "DEMO_1" in d.text and "right arm moved" in d.text and "left arm still" in d.text
    assert "text only" in d.text
    lines = [l for l in d.text.splitlines() if l.strip().startswith("t=")]
    assert 3 <= len(lines) <= 6 and lines[0].strip().startswith("t=0.0s") and lines[-1].strip().startswith("t=4.0s")
    assert "forward" in lines[1] or "up" in lines[1]                                # the swing carries the hand forward and up


def test_load_demo_with_sheet(cfg, tmp_path):
    sheet = tmp_path / "reach.sheet.jpg"; Image.new("RGB", (640, 200), (10, 20, 30)).save(sheet, "JPEG")
    meta = {"file": "reach.sheet.jpg", "moments": [0, 10, 20, 40], "times_s": [0.0, 1.0, 2.0, 4.0], "cameras": ["context", "left wrist"]}
    path = recording(tmp_path / "reach.json", n=41, sheet=meta)
    d = load_demos([path], cfg)[0]
    assert d.image is not None and d.image[0].startswith("DEMO_1") and d.image[1] == sheet.read_bytes()
    assert "Contact sheet columns: t=0.0s, t=1.0s, t=2.0s, t=4.0s; rows: context, left wrist" in d.text
    assert sum(1 for l in d.text.splitlines() if l.strip().startswith("t=")) == 4         # the sheet's own moments
    block = demos_block([d])
    assert block.startswith("DEMONSTRATIONS: 1 motion") and "DEMO_1:" in block and "stacked in ONE image labelled DEMOS" in block
    assert demos_block([]) == ""


def test_episode_shows_demos_in_every_call(cfg, tmp_path):
    cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path)
    sheet = tmp_path / "a.sheet.jpg"; Image.new("RGB", (320, 100)).save(sheet, "JPEG")
    demos = load_demos([recording(tmp_path / "a.json", sheet={"file": "a.sheet.jpg", "moments": [0, 39], "times_s": [0, 4], "cameras": ["context"]}),
                        recording(tmp_path / "b.json", name="second")], cfg)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    per = Perception(cfg, "right", MockCameras(backend, "right", 160, 90))
    seen = []
    vlm = ScriptedVLM(decisions=[{"decision": "MV_UP", "reasoning": "WRIST: YES"}, {"decision": "DONE", "reasoning": "WRIST: YES"}])
    orig = vlm.act
    def act(prompt, images, schema=None, retry_note=None):
        seen.append((prompt, [l for l, _ in images])); return orig(prompt, images, schema, retry_note)
    vlm.act = act
    s = Episode(cfg, vlm, ex, per, None, log=lambda *_: None, demos=demos).run("reach")
    assert s["success"]
    plan_prompt = vlm.calls[0][1]
    assert plan_prompt.startswith("DEMONSTRATIONS: 2 motion") and "DEMO_2: \"second\"" in plan_prompt and "ROLE: SubgoalPlanner" in plan_prompt
    assert len(seen) == 2
    for prompt, labels in seen:
        assert prompt.startswith("DEMONSTRATIONS") and "TASK: reach" in prompt
        assert labels[0].startswith("DEMOS") and labels[1] == "CONTEXT VIEW"        # one stacked sheet image before the live images
