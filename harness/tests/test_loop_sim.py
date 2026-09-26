"""End-to-end episode on the mock robot with an oracle policy standing in for the VLM.

Exercises: planner parse, controller parse, chunking when WRIST: NO, fine/coarse step switching,
empty-grasp recovery with rollback to the GRASP stage, stage advancement on DONE, RELEASE, RETREAT.
"""
import json
import re

import numpy as np
import pytest

from harness.executor import ArmExecutor
from harness.kinematics import ArmKinematics
from harness.loop import Episode
from harness.perception import MockCameras, Perception
from harness.recorder import Recorder
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from harness.vlm.scripted import ScriptedVLM

PLAN = {"subgoals": [
    {"id": "grasp_block", "target": "the orange block", "affordance": "block body", "motion": "GRASP",
     "description": "centre the hand over the block, lower, close", "completion": "the block is held"},
    {"id": "lift_block", "target": "the block", "affordance": "block body", "motion": "LIFT",
     "description": "lift straight up", "completion": "a clear gap under the block"},
    {"id": "move_to_plate", "target": "the plate", "affordance": "plate centre", "motion": "MOVE",
     "description": "carry the block over the plate", "completion": "block above the plate centre"},
    {"id": "release_block", "target": "the plate", "affordance": "plate centre", "motion": "RELEASE",
     "description": "lower and open", "completion": "block resting on the plate"},
    {"id": "retreat", "target": "free space", "affordance": "above the plate", "motion": "RETREAT",
     "description": "lift the hand clear", "completion": "hand well above the plate"}]}


class Oracle:
    """Cheating policy: reads the mock's geometry and answers like the VLM should. Counts what it exercised."""
    def __init__(self, backend):
        self.b = backend; self.grasps = 0; self.chunks = 0; self.seen = set()

    def __call__(self, prompt, images):
        stage = re.search(r"STAGE: (\w+)", prompt).group(1)
        self.seen.add(stage)
        tip, cube, plate = self.b.tip("right"), self.b.cube_pos(), self.b.plate
        if stage == "GRASP":
            return self.approach(tip, cube, grasp=True)
        if stage == "LIFT":
            return self.dec("MV_UP", True) if tip[2] < self.b.cube_start[2] + 0.10 else self.dec("DONE", True)
        if stage == "MOVE":
            return self.approach(tip, plate + [0, 0, 0.12], grasp=False, xy_only=True)
        if stage == "RELEASE":
            if tip[2] - plate[2] > 0.06:
                return self.dec("MV_DOWN", True)
            return self.dec("RELEASE", True) if self.b.hand_closed["right"] else self.dec("DONE", True)
        if stage == "RETREAT":
            return self.dec("MV_UP", False) if tip[2] < plate[2] + 0.12 else self.dec("DONE", False)
        return self.dec("DONE", False)

    def approach(self, tip, goal, grasp, xy_only=False):
        d = goal - tip
        far = np.linalg.norm(d[:2]) > 0.05
        if abs(d[0]) > 0.015 or abs(d[1]) > 0.015:
            ax = 0 if abs(d[0]) >= abs(d[1]) else 1
            tok = ("MV_FWD" if d[0] > 0 else "MV_BACK") if ax == 0 else ("MV_LEFT" if d[1] > 0 else "MV_RIGHT")
            if far:
                self.chunks += 1
                return {"decision": tok, "reasoning": "WRIST: NO. target far in the context view", "plan": [tok, tok]}
            return self.dec(tok, True)
        if xy_only:
            return self.dec("DONE", True)
        if self.grasps == 0 and tip[2] - goal[2] <= 0.07:   # first grasp deliberately too high -> empty -> recovery
            self.grasps += 1; return self.dec("GRASP", True)
        if tip[2] - goal[2] > 0.02:
            return self.dec("MV_DOWN", True)
        if not self.b.hand_closed["right"]:
            self.grasps += 1; return self.dec("GRASP", True)
        return self.dec("DONE", True)

    @staticmethod
    def dec(tok, wrist):
        return {"decision": tok, "reasoning": f"WRIST: {'YES' if wrist else 'NO'}. oracle"}


@pytest.fixture
def rig(cfg, tmp_path):
    cfg["hand"]["type"] = "virtual"; cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    gate = SafetyGate(cfg, kin, None, live=False)
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, gate, backend, "right")
    per = Perception(cfg, "right", MockCameras(backend, "right", 320, 180))
    return cfg, backend, ex, per


def test_pick_and_place_episode(rig):
    cfg, backend, ex, per = rig
    oracle = Oracle(backend)
    vlm = ScriptedVLM(plan=PLAN, on_act=oracle)
    rec = Recorder(cfg, "sim", "pick up the block and place it on the plate")
    logs = []
    ep = Episode(cfg, vlm, ex, per, rec, log=logs.append)
    summary = ep.run("pick up the block and place it on the plate")
    assert summary["success"], (summary, logs[-5:])
    assert oracle.grasps >= 2 and oracle.chunks >= 1
    assert oracle.seen >= {"GRASP", "LIFT", "MOVE", "RELEASE", "RETREAT"}
    cube = backend.cube_pos()
    assert np.linalg.norm(cube[:2] - backend.plate[:2]) < 0.05, cube
    assert backend.holding["right"] is None
    steps = [json.loads(l) for l in (rec.dir / "steps.jsonl").read_text().splitlines()]
    assert any(s.get("feedback", "").startswith("EMPTY") for s in steps)          # recovery path ran
    assert any("empty" in " ".join(vlm.calls[i][1].splitlines()).lower() for i in range(len(vlm.calls)) if vlm.calls[i][0] == "act")
    acts = [c for c in vlm.calls if c[0] == "act"]
    assert len(acts) < summary["steps"]                                           # chunked steps made no VLM call
    assert (rec.dir / "plan.json").exists() and (rec.dir / "step_000" / "prompt.txt").exists()


def test_invalid_answers_end_the_episode(rig):
    cfg, backend, ex, per = rig
    vlm = ScriptedVLM(plan=PLAN, decisions=["garbage", "{}", '{"decision": "FLY"}', "x", "y", "z"])
    ep = Episode(cfg, vlm, ex, per, None, log=lambda *_: None)
    s = ep.run("anything")
    assert not s["success"] and "invalid" in s["reason"]
    assert any(c[2] for c in vlm.calls if c[0] == "act")                        # the re-prompt carried the error


def test_ik_failures_trigger_home_step(rig):
    cfg, backend, ex, per = rig
    cfg["loop"]["max_steps"] = 8
    # park the hand at the far corner of the box so forward moves are unreachable
    ex.go_to_joints([-1.2, -0.05, 0.2, 1.3, 0.0], "park")
    vlm = ScriptedVLM(plan=PLAN, decisions=[{"decision": "MOVE forward 20", "reasoning": "WRIST: NO"}] * 8)
    logs = []
    ep = Episode(cfg, vlm, ex, per, None, log=logs.append)
    s = ep.run("reach far")
    assert not s["success"]
    prompts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert any("unreachable" in p for p in prompts) and any("home pose" in p for p in prompts)
