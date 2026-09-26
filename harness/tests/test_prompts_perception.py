import io
import json

import numpy as np
import pytest
from PIL import Image

from harness.perception import Packet, Perception, draw_grid, project, resize
from harness.prompts import controller_prompt, parse_plan, planner_prompt, proprio_text
from harness.recorder import Recorder, load_step
from harness.vlm.base import VLMResponse


def test_parse_plan_and_defaults():
    st = parse_plan('```json\n{"subgoals": [{"target": "cup", "completion": "cup lifted"}]}\n```')
    assert st[0]["id"] == "stage_1" and st[0]["affordance"] == "cup"
    with pytest.raises(ValueError):
        parse_plan('{"subgoals": []}')
    with pytest.raises((ValueError, KeyError)):
        parse_plan('{"subgoals": [{"target": "cup"}]}')


def test_controller_prompt_carries_conventions_every_call(cfg):
    stage = {"id": "reach", "target": "the red block", "affordance": "block top", "motion": "REACH",
             "description": "hover", "completion": "hand above the block"}
    pro = proprio_text(9.5, 1.0, "no hand", stall="Last move achieved 0.3 of 2.0 cm -> already in contact, do NOT repeat it.")
    p = controller_prompt("hover over the block", stage, pro, ["MV_FWD", "MV_UP"], "note", cfg, "right")
    for must in ("MV_LEFT", "TARGET to the left in the image", "WRIST: YES", "Recent moves, newest first: MV_FWD, MV_UP",
                 "Recovery: note", "9.5 cm above the table", "do NOT repeat it", "no hand", "RIGHT WRIST VIEW", "plan"):
        assert must in p, must
    assert "GRASP when BOTH" not in p                       # no hand on this robot
    assert "NO HAND" in planner_prompt("x", cfg, "right")


def test_grid_and_projection():
    im = draw_grid(Image.new("RGB", (640, 360)), 8, 6)
    assert im.size == (640, 360)
    cam = {"pos": [0, 0, 1.0], "forward": [1, 0, 0], "up": [0, 0, 1], "fx": 300, "fy": 300, "cx": 320, "cy": 180}
    assert project(cam, [1.0, 0, 1.0]) == pytest.approx((320, 180))          # straight ahead -> image centre
    u, v = project(cam, [1.0, 0.5, 1.0]);  assert u < 320                    # +y (robot left) is image left
    u, v = project(cam, [1.0, 0, 1.5]);    assert v < 180                    # up is image up
    assert project(cam, [-1.0, 0, 1.0]) is None                              # behind the camera
    assert resize(Image.new("RGB", (1280, 720)), 640).size == (640, 360)


class FakeCams:
    def frames(self):
        return {"CONTEXT VIEW": Image.new("RGB", (1280, 720), (10, 20, 30)), "RIGHT WRIST VIEW": None}


def test_perception_packet_labels_and_missing(cfg):
    cfg["perception"]["wrist_optional"] = False
    per = Perception(cfg, "right", FakeCams())
    pk = per.capture(np.array([0.3, -0.1, 0.8]))
    assert [l for l, _ in pk.images] == ["CONTEXT VIEW", "RIGHT WRIST VIEW"]
    assert pk.missing == ["RIGHT WRIST VIEW"]
    assert Image.open(io.BytesIO(pk.images[0][1])).size == (640, 360)
    cfg["perception"]["wrist_optional"] = True
    pk = Perception(cfg, "right", FakeCams()).capture(np.array([0.3, -0.1, 0.8]))
    assert [l for l, _ in pk.images] == ["CONTEXT VIEW"] and pk.missing == ["RIGHT WRIST VIEW"]
    p = controller_prompt("t", {"id": "s", "target": "x", "completion": "y"}, {"text": "", "hand_state": "no hand"}, [], None,
                          cfg, "right", wrist_missing=True)
    assert "NO wrist" in p and "RIGHT WRIST VIEW = the camera" not in p


def test_recorder_round_trip(cfg, tmp_path):
    cfg["recorder"]["root"] = str(tmp_path)
    rec = Recorder(cfg, "sim", "test task")
    pk = Packet([("CONTEXT VIEW", b"\xff\xd8jpg"), ("RIGHT WRIST VIEW", b"\xff\xd8jpg2")])
    rec.step(0, {"action": "MV_FWD", "hand_tip_before": np.array([0.3, 0.1, 0.8])}, pk, "PROMPT", VLMResponse("{}", "m", 1.2, 10, 5))
    rec.finish({"success": True})
    r, prompt, images = load_step(rec.dir, 0)
    assert r["action"] == "MV_FWD" and r["latency_s"] == 1.2 and prompt == "PROMPT"
    assert [l for l, _ in images] == ["CONTEXT VIEW", "RIGHT WRIST VIEW"]
    assert json.loads((rec.dir / "meta.json").read_text())["summary"]["success"] is True
