"""Claude as planner and eyes, a System One decider per step (harness/split.py, harness/decider/).

Units: geometry -> words, snapshot parsing, the Jev HTTP client against a loopback fake of api.typesafe.ai.
Episodes on the mock robot: Claude is a ScriptedVLM oracle that answers snapshot prompts from the mock's geometry (and
controller prompts when the loop falls back to it); the decider is the scripted geometric stand-in or a queue of
injected answers. Nothing here reaches the network or the robot.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from harness.decider.base import Answer, Choice, Noul
from harness.decider.jev import JevDecider, parse_answers
from harness.decider.scripted import ScriptedDecider
from harness.executor import ArmExecutor, ExecResult
from harness.kinematics import ArmKinematics
from harness.perception import MockCameras, Perception
from harness.recorder import Recorder
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from harness.split import (SplitEpisode, decision_questions, decision_state, describe_gap, last_result_text, parse_snapshot)
from harness.vlm.scripted import ScriptedVLM

from .test_loop_sim import PLAN

I3 = np.eye(3)


# -- units ----------------------------------------------------------------------------------------------------------------
def test_describe_gap_words_and_largest():
    words, largest, token = describe_gap(np.array([6.0, -0.5, -12.0]), 1.0)
    assert words["forward_back"] == "the goal is 6 cm forward of the hand tip (near)"
    assert words["left_right"].startswith("aligned")
    assert words["up_down"] == "the goal is 12 cm below the hand tip (far)"
    assert (largest, token) == ("down", "MV_DOWN")
    words, largest, token = describe_gap(np.array([0.4, 0.2, -0.9]), 1.0)
    assert largest is None and token is None and all(w.startswith("aligned") for w in words.values())
    assert describe_gap(np.array([0.0, -2.0, 0.0]), 1.0)[2] == "MV_RIGHT"


def test_parse_snapshot_goal_in_robot_frame():
    text = json.dumps({"goal_offset_cm": {"forward": 10, "left": -4, "up": 200}, "done_when": "over it", "stage_complete": False,
                       "wrist_sees_target": True, "target_visible": True, "hazards": "none", "confidence": "high", "reasoning": "r"})
    s = parse_snapshot(text, [0.3, 0.0, 0.8], I3, step=3, max_cm=60)
    assert np.allclose(s.goal, [0.4, -0.04, 1.4]) and s.offset_cm["up"] == 60 and s.step == 3
    # a rotated view: image-forward is robot +y
    R = np.column_stack([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(parse_snapshot(text, [0, 0, 0], R, 0).goal[:2], [0.04, 0.10])
    for bad in ("nope", "[]", json.dumps({"goal_offset_cm": {"forward": 1}}), json.dumps({"goal_offset_cm": {"forward": "x", "left": 0, "up": 0}})):
        with pytest.raises(ValueError):
            parse_snapshot(bad, [0, 0, 0], I3, 0)


def test_state_is_words_and_facts_are_numbers():
    s = parse_snapshot(json.dumps({"goal_offset_cm": {"forward": 0, "left": 5, "up": -3}, "done_when": "d", "stage_complete": False,
                                   "wrist_sees_target": False, "target_visible": True, "hazards": "a cup on the left",
                                   "confidence": "medium", "reasoning": "r"}), [0.3, 0, 0.8], I3, 0)
    st, facts = decision_state("t", PLAN["subgoals"][0], s, [0.3, 0.01, 0.8], I3, 12.0, 4.0, 2.4, "none", False, ["MV_LEFT"],
                               None, ["the operator says: slower"], 1, 1.0)
    assert st["goal_relative_to_hand_tip"]["largest_gap"] == "left"             # 4 cm left beats 3 cm down
    assert st["goal_relative_to_hand_tip"]["up_down"] == "the goal is 3 cm below the hand tip (near)"
    assert facts["gap_cm"] == [0.0, 4.0, -3.0] and facts["largest"] == "MV_LEFT"
    assert st["scene"]["hazards"] == "a cup on the left" and st["operator_notes"] == ["the operator says: slower"]
    assert "gap_cm" not in json.dumps(st)                                       # Jev never gets the raw numbers
    qs = decision_questions("none", "jev")
    assert set(qs) == {"action", "stage_done", "needs_look"} and "GRASP" not in qs["action"].criteria and "LOOK" in qs["action"].criteria
    assert set(decision_questions("virtual", "geometric")) == {"stage_done", "needs_look"}
    assert "GRASP" in decision_questions("virtual", "jev")["action"].criteria


# -- the Jev client, against a loopback fake --------------------------------------------------------------------------------
class FakeJev(BaseHTTPRequestHandler):
    requests, script = [], []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeJev.requests.append((dict(self.headers), body))
        status, obj = FakeJev.script.pop(0) if FakeJev.script else (200, None)
        if obj is None:
            obj = {"model": "jev-1.13.0", "usage": {"input_tokens": 300, "output_tokens": 20}, "answers": {
                "action": {"type": "choice", "choice": "MV_DOWN", "probabilities": {"MV_DOWN": 0.9, "DONE": 0.1}, "confidence": 0.85},
                "stage_done": {"type": "noul", "noul": 0.1}}}
        data = json.dumps(obj).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        if status == 529:
            self.send_header("retry-after", "0.01")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_jev(cfg, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeJev)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeJev.requests.clear(); FakeJev.script.clear()
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    cfg["executor"]["jev_url"] = f"http://127.0.0.1:{srv.server_port}/v1/systemone"
    yield JevDecider(cfg)
    srv.shutdown()


QS = {"action": Choice("which?", {"MV_DOWN": "down", "DONE": "done"}), "stage_done": Noul("done?", {"true": "y", "false": "n"})}


def test_jev_request_and_answers(fake_jev):
    answers, resp = fake_jev.ask({"task": "t"}, QS)
    headers, body = FakeJev.requests[0]
    assert headers["Authorization"] == "Bearer test-key"
    assert body == {"state": {"task": "t"}, "model": "jev-1.13.0", "questions": {
        "action": {"type": "choice", "instructions": "which?", "criteria": {"MV_DOWN": "down", "DONE": "done"}},
        "stage_done": {"type": "noul", "instructions": "done?", "criteria": {"true": "y", "false": "n"}}}}
    assert not resp.error and resp.model == "jev-1.13.0" and resp.input_tokens == 300
    assert answers["action"].choice == "MV_DOWN" and answers["action"].confidence == 0.85 and answers["stage_done"].noul == 0.1


def test_jev_retries_overload_and_reports_errors(fake_jev):
    FakeJev.script[:] = [(529, {"error": "overloaded"})]
    answers, resp = fake_jev.ask("s", QS)
    assert len(FakeJev.requests) == 2 and not resp.error and answers["action"].choice == "MV_DOWN"
    FakeJev.script[:] = [(401, {"error": "bad key"})]
    answers, resp = fake_jev.ask("s", QS)
    assert answers == {} and "401" in resp.error
    FakeJev.script[:] = [(200, {"model": "jev", "answers": {"action": {"type": "choice", "choice": "FLY", "confidence": 1}}})]
    answers, resp = fake_jev.ask("s", QS)
    assert "action" not in answers and "stage_done" in resp.error


def test_jev_needs_a_key(cfg, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevDecider(cfg)


def test_parse_answers_types():
    out, missing = parse_answers({"answers": {"action": {"type": "noul", "noul": 1}, "stage_done": {"type": "noul", "noul": "x"}}}, QS)
    assert out == {} and missing == ["action", "stage_done"]


@pytest.mark.parametrize("obj", [None, [], "bad", {"answers": []}, {"answers": "bad"}])
def test_jev_malformed_envelope_escalates(fake_jev, obj):
    # None selects the fake server's default, so use a non-null envelope on the wire.
    if obj is None:
        assert parse_answers(obj, QS) == ({}, list(QS))
        return
    FakeJev.script[:] = [(200, obj)]
    answers, resp = fake_jev.ask("s", QS)
    assert not answers and resp.error


@pytest.mark.parametrize("bad", [None, True, "0.9", "x", float("nan"), float("inf"), -0.1, 1.1])
def test_jev_rejects_invalid_probabilities(bad):
    raw = {"action": {"type": "choice", "choice": "MV_DOWN", "confidence": bad,
                      "probabilities": {"MV_DOWN": 0.9, "DONE": 0.1}},
           "stage_done": {"type": "noul", "noul": bad}}
    assert parse_answers({"answers": raw}, QS) == ({}, list(QS))


@pytest.mark.parametrize("probs", [None, [], {"MV_DOWN": 1}, {"MV_DOWN": "x", "DONE": 0},
                                      {"MV_DOWN": 0.2, "DONE": 0.8}, {"MV_DOWN": 0.9, "DONE": 0.9}])
def test_jev_rejects_invalid_choice_distribution(probs):
    raw = {"action": {"type": "choice", "choice": "MV_DOWN", "confidence": 0.9, "probabilities": probs}}
    out, missing = parse_answers({"answers": raw}, QS)
    assert not out and "action" in missing


def test_jev_accepts_api_rounded_distribution():
    raw = {"action": {"type": "choice", "choice": "MV_DOWN", "confidence": 0.9,
                      "probabilities": {"MV_DOWN": 0.94, "DONE": 0.05}}}
    out, missing = parse_answers({"answers": raw}, QS)
    assert out["action"].choice == "MV_DOWN" and missing == ["stage_done"]


@pytest.mark.parametrize("changes", [
    {"stage_complete": "false"}, {"wrist_sees_target": 1}, {"target_visible": None},
    {"confidence": "certain"}, {"done_when": []},
    *[{"goal_offset_cm": {"forward": x, "left": 0, "up": 0}}
      for x in (True, "10", float("nan"), float("inf"))],
])
def test_snapshot_rejects_malformed_model_output(changes):
    obj = {"goal_offset_cm": {"forward": 0, "left": 0, "up": 0}, "done_when": "at goal",
           "stage_complete": False, "wrist_sees_target": False, "target_visible": True,
           "hazards": "none", "confidence": "high", "reasoning": "test", **changes}
    with pytest.raises(ValueError):
        parse_snapshot(json.dumps(obj), [0, 0, 0], I3, 0)


# -- episodes on the mock robot ------------------------------------------------------------------------------------------
class Eyes:
    """Claude stand-in. Snapshot prompts: the stage's hand-tip goal from the mock's geometry. Controller prompts (the
    fallback): one move toward that goal. Counts both."""
    def __init__(self, backend, fallback=None):
        self.b = backend; self.looks = 0; self.controller = 0; self.fallback = fallback; self.deny_done = 0

    def goal(self, motion):
        tip, cube, plate = self.b.tip("right"), self.b.cube_pos(), self.b.plate
        return {"GRASP": cube, "LIFT": np.array([tip[0], tip[1], self.b.cube_start[2] + 0.12]),
                "MOVE": plate + [0, 0, 0.12], "RELEASE": plate + [0, 0, 0.05], "RETREAT": plate + [0, 0, 0.14]}[motion]

    def complete(self, motion, tip, goal):
        held = self.b.hand_closed["right"]
        if motion == "GRASP":
            return self.b.holding["right"] is not None
        if motion == "MOVE":
            return np.linalg.norm((goal - tip)[:2]) < 0.025
        if motion == "RELEASE":
            return not held and np.linalg.norm(self.b.cube_pos()[:2] - self.b.plate[:2]) < 0.05
        return tip[2] >= goal[2] - 0.015

    def __call__(self, prompt, images):
        motion = prompt.split("STAGE: ", 1)[1].split()[0]
        tip = self.b.tip("right"); goal = self.goal(motion)
        if "ROLE: SceneSnapshot" in prompt:
            self.looks += 1
            done = self.complete(motion, tip, goal)
            if done and self.deny_done:
                self.deny_done -= 1; done = False
            off = (goal - tip) * 100
            nxt = {}
            if done and "NEXT STAGE: " in prompt:
                nxt = {"next_goal_offset_cm": dict(zip(("forward", "left", "up"),
                                                       map(float, (self.goal(prompt.split("NEXT STAGE: ", 1)[1].split()[0]) - tip) * 100))),
                       "next_done_when": "within 1 cm of the goal"}
            return {**nxt, "goal_offset_cm": dict(zip(("forward", "left", "up"), map(float, off))), "done_when": "within 1 cm of the goal",
                    "stage_complete": bool(done), "wrist_sees_target": bool(np.linalg.norm(off) < 8), "target_visible": True,
                    "hazards": "none", "confidence": "high", "reasoning": "oracle"}
        self.controller += 1
        if self.fallback:
            return self.fallback
        return {"decision": "STILL", "reasoning": "WRIST: NO. fallback"}


@pytest.fixture
def rig(cfg, tmp_path):
    cfg["hand"]["type"] = "virtual"; cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path)
    cfg["executor"]["per_step"] = "split"
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    per = Perception(cfg, "right", MockCameras(backend, "right", 320, 180))
    return cfg, backend, ex, per


@pytest.mark.parametrize("mover", ["jev", "geometric"])
def test_pick_and_place_claude_looks_rarely(rig, mover):
    cfg, backend, ex, per = rig
    cfg["executor"]["mover"] = mover
    eyes = Eyes(backend)
    vlm, decider = ScriptedVLM(plan=PLAN, on_act=eyes), ScriptedDecider()
    rec = Recorder(cfg, "sim", "pick and place")
    logs = []
    s = SplitEpisode(cfg, vlm, decider, ex, per, recorder=rec, log=logs.append).run("pick up the block and place it on the plate")
    assert s["success"], (s, logs[-8:])
    assert np.linalg.norm(backend.cube_pos()[:2] - backend.plate[:2]) < 0.05 and backend.holding["right"] is None
    assert eyes.controller == 0 and s["claude_steps"] == 0                    # Claude never had to pick a move
    assert s["decider_steps"] > 2 * eyes.looks, (s, eyes.looks)                 # most steps were the fast model's
    steps = [json.loads(l) for l in (rec.dir / "steps.jsonl").read_text().splitlines()]
    assert sum(1 for x in steps if x.get("stage_done")) == 5
    assert any((rec.dir / f"step_{x['step']:03d}" / "look_prompt.txt").exists() for x in steps)
    assert all(x.get("decided_by") in (None, "scripted", "geometry") for x in steps)
    sensed = [x["why"] for x in steps if x.get("stage_done") and x.get("action") in ("GRASP", "RELEASE")]
    assert sensed == ["the hand sensed a hold", "the hand opened"], sensed    # the hand's report ended those stages


def test_a_goal_under_the_table_counts_as_reached():
    s = parse_snapshot(json.dumps({"goal_offset_cm": {"forward": 0, "left": 0, "up": -3}, "done_when": "d", "stage_complete": False,
                                   "wrist_sees_target": True, "target_visible": True, "hazards": "none", "confidence": "high",
                                   "reasoning": "r"}), [0.3, 0, 0.7], I3, 0)
    args = ("t", PLAN["subgoals"][0], s, [0.3, 0, 0.7], I3)
    st, facts = decision_state(*args, 2.1, 1.0, 1.0, "virtual", False, [], None, [], 0, 0.0, True, 2.0)
    assert facts["largest"] is None and "cannot go lower" in st["goal_relative_to_hand_tip"]["up_down"]
    st, facts = decision_state(*args, 8.0, 1.0, 1.0, "virtual", False, [], None, [], 0, 0.0, True, 2.0)
    assert facts["largest"] == "MV_DOWN"                                        # above the floor: go down as usual


def test_a_fresh_look_is_not_asked_for_again(rig):
    cfg, backend, ex, per = rig
    cfg["loop"]["max_steps"] = 1
    wants_look = {"action": Answer("choice", choice="MV_DOWN", confidence=0.9, probabilities={}),
                  "stage_done": Answer("noul", noul=0.05), "needs_look": Answer("noul", noul=0.95)}
    eyes = Eyes(backend)
    decider = ScriptedDecider(queue=[wants_look])
    s = SplitEpisode(cfg, ScriptedVLM(plan=PLAN, on_act=eyes), decider, ex, per, log=lambda *_: None).run("pick")
    assert eyes.looks == 1 and eyes.controller == 0 and s["decider_steps"] == 1   # step 0's look is fresh: the move goes ahead
    assert decider.calls[0][0]["scene"]["last_camera_look"].startswith("just now")


def test_unsure_decider_gets_a_look_then_claude_decides(rig):
    cfg, backend, ex, per = rig
    cfg["loop"]["max_steps"] = 3
    unsure = {"action": Answer("choice", choice="MV_DOWN", confidence=0.2, probabilities={}),
              "stage_done": Answer("noul", noul=0.1), "needs_look": Answer("noul", noul=0.1)}
    look = {"action": Answer("choice", choice="LOOK", confidence=0.9, probabilities={}),
            "stage_done": Answer("noul", noul=0.1), "needs_look": Answer("noul", noul=0.1)}
    eyes = Eyes(backend, fallback={"decision": "MV_UP", "reasoning": "WRIST: NO. up"})
    decider = ScriptedDecider(queue=[unsure, look])
    logs = []
    s = SplitEpisode(cfg, ScriptedVLM(plan=PLAN, on_act=eyes), decider, ex, per, log=logs.append).run("pick")
    # step 0: first look, unsure -> second look + re-ask -> LOOK -> Claude picks MV_UP from the images
    assert eyes.looks == 2 and eyes.controller == 1 and s["claude_steps"] == 1
    assert any("escalated" in l and "MV_UP" in l for l in logs), logs


def test_done_claim_needs_claudes_eyes(rig):
    cfg, backend, ex, per = rig
    cfg["loop"]["max_steps"] = 2
    down = {"action": Answer("choice", choice="MV_DOWN", confidence=0.9, probabilities={}),
            "stage_done": Answer("noul", noul=0.05), "needs_look": Answer("noul", noul=0.0)}
    done = {"action": Answer("choice", choice="DONE", confidence=0.95, probabilities={}),
            "stage_done": Answer("noul", noul=0.95), "needs_look": Answer("noul", noul=0.0)}
    eyes = Eyes(backend)
    decider = ScriptedDecider(queue=[down, done])                             # a wrong DONE on the second step
    logs = []
    plan = {"subgoals": [PLAN["subgoals"][1]]}  # LIFT: completion requires camera confirmation
    s = SplitEpisode(cfg, ScriptedVLM(plan=plan, on_act=eyes), decider, ex, per, log=logs.append).run("lift")
    assert not any("stage done" in l for l in logs), logs                      # Claude looked, said no: no stage advance
    assert eyes.looks == 2 and s["decider_steps"] == 2                          # step 1: confirm look, re-ask, a move
    assert "confirm from the images" in " ".join(logs)


def test_decider_error_holds_without_fallback(rig):
    cfg, backend, ex, per = rig
    cfg["loop"]["max_steps"] = 1; cfg["executor"]["claude_fallback"] = False
    eyes = Eyes(backend)

    class Broken(ScriptedDecider):
        def ask(self, state, questions, facts=None):
            a, r = {}, super().ask(state, questions, facts)[1]; r.error = "Jev HTTP 529"; return a, r
    logs = []
    SplitEpisode(cfg, ScriptedVLM(plan=PLAN, on_act=eyes), Broken(), ex, per, log=logs.append).run("pick")
    assert eyes.controller == 0 and any("STILL" in l and "529" in l for l in logs), logs


def test_failed_done_confirmation_holds_and_invalidates_snapshot(rig):
    cfg, backend, ex, per = rig
    cfg["loop"]["max_steps"] = 3
    cfg["executor"]["claude_fallback"] = False
    eyes = Eyes(backend)
    looks = []
    def fail_refresh(prompt, images):
        if "ROLE: SceneSnapshot" in prompt:
            looks.append(prompt)
            if len(looks) > 1:
                return "invalid snapshot"
        return eyes(prompt, images)
    done = {"action": Answer("choice", choice="DONE", confidence=0.9),
            "stage_done": Answer("noul", noul=0.9), "needs_look": Answer("noul", noul=0)}
    decider = ScriptedDecider()
    move = decider.policy(decision_questions("virtual", "jev"), {"largest": "MV_DOWN"})
    decider.queue = [move, done]
    logs = []
    plan = {"subgoals": [PLAN["subgoals"][1]]}
    ep = SplitEpisode(cfg, ScriptedVLM(plan=plan, on_act=fail_refresh), decider, ex, per, log=logs.append)
    summary = ep.run("pick")
    assert not summary["success"] and ep.snap is None
    assert len(decider.calls) == 2    # failed refresh on step 2 prevents another decision from stale state
    assert any("STILL" in line and "camera look failed" in line for line in logs)


def test_rejected_move_reaches_decider_after_fresh_look(rig):
    cfg, backend, ex, per = rig
    cfg["loop"]["max_steps"] = 2
    ex.confirm = lambda *_: "keep away from the cup"
    before = backend.frames_sent
    eyes, decider = Eyes(backend), ScriptedDecider()
    SplitEpisode(cfg, ScriptedVLM(plan=PLAN, on_act=eyes), decider, ex, per, log=lambda *_: None).run("pick")
    assert backend.frames_sent == before and eyes.looks == 2
    state = decider.calls[-1][0]
    assert "rejected" in state["last_move_result"]
    assert "cup" in " ".join(state["operator_notes"])


@pytest.mark.parametrize("result", [ExecResult(False, "trajectory rejected"), ExecResult(True, "still moving", timeout=True)])
def test_blocked_or_unsettled_move_requires_look(rig, result):
    cfg, backend, ex, per = rig
    ep = SplitEpisode(cfg, ScriptedVLM(), ScriptedDecider(), ex, per, log=lambda *_: None)
    state = ex.sync()
    text = Eyes(backend)("STAGE: GRASP\nROLE: SceneSnapshot", [])
    ep.snap = parse_snapshot(json.dumps(text), state.p, I3, 0)
    assert ep.look_reason(1, state, result)
    assert "completed normally" not in last_result_text(result)


@pytest.mark.parametrize("result", [ExecResult(False, "ESTOP"), ExecResult(True, "no hand fitted")])
def test_grasp_without_hand_confirmation_does_not_finish_stage(rig, monkeypatch, result):
    cfg, backend, ex, per = rig
    cfg["loop"]["max_steps"] = 1
    monkeypatch.setattr(ex, "execute", lambda *_: result)
    grasp = {"action": Answer("choice", choice="GRASP", confidence=0.9),
             "stage_done": Answer("noul", noul=0), "needs_look": Answer("noul", noul=0)}
    plan = {"subgoals": [PLAN["subgoals"][0]]}
    summary = SplitEpisode(cfg, ScriptedVLM(plan=plan, on_act=Eyes(backend)), ScriptedDecider(queue=[grasp]),
                           ex, per, log=lambda *_: None).run("pick")
    assert not summary["success"]


@pytest.mark.parametrize("token", ["GRASP", "DONE"])
def test_high_done_probability_cannot_skip_pending_grasp(rig, token):
    cfg, backend, ex, per = rig
    cfg["executor"]["confirm_done_with_claude"] = False
    answers = {"action": Answer("choice", choice=token, confidence=0.9),
               "stage_done": Answer("noul", noul=0.99), "needs_look": Answer("noul", noul=0)}
    ep = SplitEpisode(cfg, ScriptedVLM(), ScriptedDecider(queue=[answers]), ex, per, log=lambda *_: None)
    state = ex.sync()
    text = Eyes(backend)("STAGE: GRASP\nROLE: SceneSnapshot", [])
    ep.snap = parse_snapshot(json.dumps(text), state.p, I3, 0)
    action, escalation, done, *_ = ep.consult("pick", PLAN["subgoals"][0], state, [], None, [], 0)
    assert not done
    assert (action == "GRASP" and not escalation) if token == "GRASP" else (action is None and escalation)
