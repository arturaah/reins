"""Every answered proposal becomes a picture card that later calls, in this session and the next ones, see."""
import io
import os

import numpy as np
import pytest
from PIL import Image

from harness.executor import ArmExecutor
from harness.experience import ExperienceStore, make_card
from harness.kinematics import ArmKinematics
from harness.loop import Episode
from harness.perception import MockCameras, Perception
from harness.poseview import PoseView
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from harness.vlm.scripted import ScriptedVLM

os.environ.setdefault("MUJOCO_GL", "cgl")


@pytest.fixture
def rig(cfg, tmp_path):
    cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    return cfg, backend, ex, Perception(cfg, "right", MockCameras(backend, "right", 160, 90))


def test_cards_are_kept_and_shown_to_later_calls(rig, tmp_path):
    cfg, backend, ex, per = rig
    ex.confirm = lambda text, preview: (True, "good") if text.startswith("MV_UP") else "not that way"
    exp = ExperienceStore(tmp_path / "exp", session="s1", max_in_prompt=3)
    vlm = ScriptedVLM(decisions=[{"decision": "MV_LEFT", "reasoning": "WRIST: NO"}, {"decision": "MV_UP", "reasoning": "WRIST: YES"},
                                 {"decision": "DONE", "reasoning": "WRIST: YES"}])
    Episode(cfg, vlm, ex, per, None, log=lambda *_: None, experience=exp).run("reach the block")
    e = exp.entries()
    assert [(x["actions"], x["accepted"], x["note"]) for x in e] == [("MV_LEFT", False, "not that way"), ("MV_UP", True, "good")]
    assert all((tmp_path / "exp" / x["image"]).exists() for x in e) and e[1]["outcome"].startswith("moved")
    prompts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert "EXPERIENCE" not in prompts[0]                                          # nothing to show at the first call
    assert "EXP_1 ✗ this task, stage hover_block: MV_LEFT — \"not that way\"" in prompts[1]
    assert "EXP_1 ✓ this task" in prompts[2] and "EXP_2 ✗" in prompts[2]         # newest first, the running session's included
    assert any(l.startswith("EXPERIENCE") for l, _ in vlm.last_images)
    # a later session sees them too, this task's cards first
    later = ExperienceStore(tmp_path / "exp", session="s2", max_in_prompt=3)
    later.add("other task", "s", "MV_FWD", True, "", "moved 6.0 of 6.0 cm", None, None, "sim")
    assert [x["actions"] for x in later.select("reach the block")] == ["MV_UP", "MV_LEFT", "MV_FWD"]
    assert [x["actions"] for x in later.select("other task")] == ["MV_FWD", "MV_UP", "MV_LEFT"]
    label, jpg = later.images("reach the block")[0]
    with Image.open(io.BytesIO(jpg)) as im:
        assert im.width == 800 and im.height > 3 * 200
    assert "EXP_k" in label and "(outcome: moved 6.0 of 6.0 cm)" in later.block("other task")
    # a card whose file is gone is not offered
    os.remove(tmp_path / "exp" / e[0]["image"])
    assert [x["actions"] for x in later.select("reach the block")] == ["MV_UP", "MV_FWD"]


def test_card_layout_and_path_drawing(cfg, tmp_path):
    ctx = Image.new("RGB", (640, 360), (10, 120, 10))
    card = make_card("✓ ACCEPTED", ["proposal: MV_UP"], None, ctx, 800, True)
    assert card.width == 800 and card.height > 200
    pv = PoseView(cfg, "right", table_z=0.665, width=320, height=180)
    q = {n: 0.0 for n in pv.adr}
    tip = pv.tip(q)
    a = pv.render(q)
    if a is None:
        pytest.skip(f"no GL context: {pv.error}")
    b = pv.render(q, path=[tip, tip + [0.1, 0.0, 0.0], tip + [0.1, 0.0, 0.1]])
    changed = (np.abs(np.asarray(a, int) - np.asarray(b, int)).sum(axis=2) > 30).mean()
    assert changed > 0.002                                                          # the path is drawn


def test_trajectory_card_carries_the_planned_waypoints(rig, tmp_path):
    cfg, backend, ex, per = rig
    per.pose_view = PoseView(cfg, "right", 0.665, 320, 180, props=True)
    if per.pose_view.render(backend.joints()) is None:
        pytest.skip("no GL context for the pose view")
    ex.confirm = lambda text, preview: True
    exp = ExperienceStore(tmp_path / "exp", session="s1")
    vlm = ScriptedVLM(decisions=[{"decision": "MV_UP", "reasoning": "WRIST: NO", "plan": ["MV_UP", "MV_UP", "MV_UP"]},
                                 {"decision": "DONE", "reasoning": "WRIST: YES"}])
    Episode(cfg, vlm, ex, per, None, log=lambda *_: None, experience=exp).run("up")
    e = exp.entries()
    assert len(e) == 1 and e[0]["actions"] == "MV_UP, MV_UP, MV_UP" and e[0]["accepted"]
    with Image.open(tmp_path / "exp" / e[0]["image"]) as im:
        assert im.width == 800
