import json
from types import SimpleNamespace

import numpy as np
import pytest

from harness.room import planner  # first: puts contract/ and loco/ on the path
from harness.room.agent import AutoApprove, Decision, Harness, SessionLog
from harness.room.brains import AnthropicBrain, ToolResult, Turn
from harness.room.demo import PointingDemoBrain, ServoDemoVLM, demo_arm_vlm
from harness.room.display import Display
from harness.room.skills import SENSING, TOOLS, PlanError, Skills, room_config
from harness.room.world import SimWorld
from reins_contract import check_session, validate


@pytest.fixture
def world():
    w = SimWorld()
    yield w
    w.close()


def make_harness(world, brain, reviewer, tmp_path, display=None, arm_vlm=None):
    display = display or Display(world, tmp_path)
    log = SessionLog(tmp_path / "session.jsonl")
    skills = Skills(world, room_config({"recorder.root": str(tmp_path / "runs")}), arm_vlm=arm_vlm)
    skills.log = lambda *a: None
    return Harness(skills, brain, reviewer, display, log), log


def true_pixel(world, xyz_world):
    """Where a true point appears in the last image (tests only)."""
    cap = world.last_capture
    p_cam = (np.linalg.inv(cap.T_robot_cam) @ np.r_[world._true_to_robot(xyz_world), 1.0])[:3]
    return [float(c) for c in cap.rgb_model.project(p_cam)]


# --- the model only gets what the real robot has ---------------------------------------

def test_no_tool_exposes_ground_truth():
    names = {t["name"] for t in TOOLS}
    assert "observe" not in names and SENSING == {"robot_state", "look", "locate"}
    from harness.room.agent import SYSTEM
    text = json.dumps(TOOLS) + SYSTEM
    for secret in ("red_cube", "blue_bottle", "plant", "cube", "bottle"):  # what's in this scene
        assert secret not in text


def test_robot_state_is_proprioception_only(world):
    state = json.loads(Skills(world).call("robot_state", {}).text)
    assert set(state) == {"odometry", "imu", "walking", "hands", "joint_positions_rad"}
    assert "red_cube" not in json.dumps(state)
    assert state["hands"]["left_hand"]["grip"] == "open"


def test_grip_results_do_not_say_what_was_caught(world):
    assert world.grip("left_hand", "close") == "left_hand closed"  # nothing there; the robot can't tell
    assert world.held_by("left_hand") is None


def test_odometry_drifts_from_the_truth(world, tmp_path):
    skills = Skills(world)
    skills.execute(skills.call("walk", {"forward_m": 1.0}).steps, world.advance)
    odom, true = world.odom_pose(), world.true_pose()
    assert odom.x == pytest.approx(1.0, abs=0.04)  # the robot believes it walked the metre
    assert abs(true.x - odom.x) > 0.01  # but it really walked a bit less


# --- seeing ----------------------------------------------------------------------------

def test_look_is_a_fisheye_image_and_locate_finds_the_cube_by_depth(world):
    skills = Skills(world)
    look = skills.call("look", {})
    assert look.image_png.startswith(b"\x89PNG")
    top = world.object_pos("red_cube") + [0, 0, 0.025]
    found = json.loads(skills.call("locate", {"pixels": [true_pixel(world, top)]}).text)[0]
    assert np.linalg.norm(np.array(found["robot"]) - world._true_to_robot(top)) < 0.05


def test_locate_before_look_and_off_image_are_explained(world):
    skills = Skills(world)
    with pytest.raises(PlanError, match="look first"):
        skills.call("locate", {"pixels": [[10, 10]]})
    skills.call("look", {})
    sky, off = json.loads(skills.call("locate", {"pixels": [[480, 60], [5000, 5]]}).text)
    assert "no depth" in sky["error"] and "outside" in off["error"]


def test_obstacle_map_comes_from_depth_and_excludes_the_robot(world):
    assert world.obstacles.clearance(1.5, 0.0) == float("inf")  # nothing seen yet
    Skills(world).call("look", {})
    assert world.obstacles.clearance(1.3, 0.0) < 0  # the table, once seen
    assert world.obstacles.clearance(0.0, 0.0) > 0.3  # its own body isn't an obstacle


def test_walk_to_plans_around_the_table_once_seen(world):
    skills = Skills(world)
    skills.call("look", {})
    proposal = skills.call("walk_to", {"x": 2.3, "y": 0.0})  # straight through the table
    points = [p for step in proposal.steps for p in step["path"]["points"]]
    assert len(points) > 2
    for a, b in zip(points, points[1:]):
        for t in np.linspace(0, 1, 20):
            assert world.obstacles.clearance(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) > 0


def test_walking_into_an_unseen_table_bumps(world):
    skills = Skills(world)  # never looked, so the table isn't on the map
    ok, detail = skills.execute(skills.call("walk", {"forward_m": 2.0}).steps, world.advance)
    assert not ok and "bumped into something" in detail


# --- the whole loop ------------------------------------------------------------------------

def test_demo_walks_there_and_the_arm_policy_moves_the_cube_to_the_counter(world, tmp_path):
    """Room layer to walk, Artur's closed-loop arm policy (harness.loop) to pick up and put down."""
    brain = PointingDemoBrain(world)
    harness, log = make_harness(world, brain, AutoApprove(), tmp_path, arm_vlm=demo_arm_vlm(world, brain))
    assert harness.run("put the cube on the counter").startswith("Done")
    truth = world.ground_truth()["objects"]["red_cube"]
    assert truth["on"] == "counter" and truth["held_by"] is None
    assert world.loco.collisions == 0
    check_session(log.messages)  # every message valid, every state transition legal
    kinds = [s["kind"] for m in log.messages if m["type"] == "plan_proposed" for s in m["plan"]["steps"]]
    assert kinds.count("servo") == 2
    assert len(list((tmp_path / "runs").glob("room_*/steps.jsonl"))) == 2  # the arm policy's own run logs


def test_manipulate_proposes_a_bounded_servo_step(world):
    skills = Skills(world)
    proposal = skills.call("manipulate", {"task": "pick up the red cube", "surface_z": 0.72, "hand": "left_hand"})
    (step,) = proposal.steps
    assert step["kind"] == "servo" and step["effector"] == "left_hand"
    assert step["bounds"]["box_min_m"][2] == pytest.approx(0.74)  # 2 cm above the surface
    validate({"type": "plan_proposed", "id": "m1", "t": 0.0, "plan": {
        "plan_id": "p-1", "revision": 1, "command_id": "c-1", "summary": proposal.summary, "steps": proposal.steps}})
    assert proposal.overlay.boxes  # the reviewer sees the box the hand must stay in
    with pytest.raises(PlanError, match="no room"):
        skills.call("manipulate", {"task": "x", "surface_z": 1.3})


def test_the_arm_policy_is_refused_moves_outside_the_approved_box(world, tmp_path):
    """The episode runs through Artur's SafetyGate: a policy pushing down into the table is clamped."""
    from harness.room.arm import run_episode

    class PushDown(ServoDemoVLM):
        def act(self, prompt, images, schema=None, retry_note=None):
            return self._say("MV_DOWN", "pushing down")

    cfg = room_config({"recorder.root": str(tmp_path), "loop.max_steps": 15})
    summary = run_episode(world, cfg, PushDown(world, "right_hand", "pick up the cube", None), "right_hand",
                          "pick up the cube", 0.72, world.advance, log=lambda *a: None)
    assert summary["reason"] == "max steps"
    assert world.hand_in_robot("right_hand")[2] >= 0.74 - 0.005  # never below surface + 2 cm


class RecordingBrain:
    """Makes one call, then reports what came back."""
    name = "recording"

    def __init__(self, call):
        self.call, self.seen = call, []

    def start(self, system, tools, task):
        self.turns = 0

    def step(self, results):
        self.seen += results
        self.turns += 1
        return Turn("go", [SimpleNamespace(id="c1", name=self.call[0], args=self.call[1])]) \
            if self.turns == 1 else Turn("ok")


class DeclineOnce:
    def review(self, plan, preview):
        assert preview.exists()
        return Decision(False, "use the other hand")


def test_decline_feedback_reaches_the_model_and_nothing_moves(world, tmp_path):
    brain = RecordingBrain(("walk_to", {"x": 0.5, "y": 0.5}))
    harness, log = make_harness(world, brain, DeclineOnce(), tmp_path)
    harness.run("go somewhere")
    assert brain.seen[0].text == "Operator declined this plan. Feedback: use the other hand"
    assert (world.true_pose().x, world.true_pose().y) == (0.0, 0.0)
    assert not any(m["type"] == "execute" for m in log.messages)
    check_session(log.messages)


def test_executed_plans_come_back_with_a_fresh_image(world, tmp_path):
    brain = RecordingBrain(("turn", {"angle_rad": 0.5}))
    harness, _ = make_harness(world, brain, AutoApprove(), tmp_path)
    harness.run("turn")
    assert brain.seen[0].text.startswith("Done") and brain.seen[0].image_png.startswith(b"\x89PNG")


class StopAfter(Display):
    def __init__(self, world, out, ticks):
        super().__init__(world, out)
        self.left = ticks

    def should_stop(self):
        self.left -= 1
        return self.left < 0


def test_operator_stop_halts_mid_walk(world, tmp_path):
    brain = RecordingBrain(("walk_to", {"x": -1.0, "y": 0.0}))
    harness, log = make_harness(world, brain, AutoApprove(), tmp_path, StopAfter(world, tmp_path, 10))
    harness.run("walk")
    assert brain.seen[0].text.startswith("Operator stopped it")
    done = next(m for m in log.messages if m["type"] == "done")
    assert done["outcome"] == "halted"
    check_session(log.messages)


def test_bad_calls_are_plan_errors_not_crashes(world):
    skills = Skills(world)
    with pytest.raises(PlanError, match="unknown tool"):
        skills.call("observe", {})
    with pytest.raises(PlanError, match="bad arguments"):
        skills.call("walk_to", {"x": 1.0})
    with pytest.raises(PlanError, match="unknown tool"):
        skills.call("pick_up", {"x": 0.3, "y": 0.0, "z": 0.8})  # manipulation goes through manipulate


# --- brains ----------------------------------------------------------------------------------

def test_anthropic_brain_round_trip_without_network():
    """Tool calls come out, results (with images) go back in the Messages API shape."""
    sent = []

    def create(**kwargs):
        sent.append({**kwargs, "messages": list(kwargs["messages"])})
        if len(sent) == 1:
            content = [SimpleNamespace(type="text", text="Looking."),
                       SimpleNamespace(type="tool_use", id="tu1", name="look", input={})]
            return SimpleNamespace(content=content, stop_reason="tool_use")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="A red cube.")],
                               stop_reason="end_turn")

    brain = AnthropicBrain.__new__(AnthropicBrain)
    brain.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    brain.model, brain.effort, brain.max_tokens, brain.name = "claude-opus-5", "high", 1000, "t"
    brain.start("sys", [{"name": "look"}], "what's there?")
    turn = brain.step([])
    assert turn.calls[0].name == "look" and not turn.done
    assert sent[0]["fallbacks"] == "default" and sent[0]["thinking"] == {"type": "adaptive"}
    turn = brain.step([ToolResult("tu1", "Head camera image.", b"\x89PNG fake")])
    assert turn.done and turn.text == "A red cube."
    result = sent[1]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu1"
    assert [c["type"] for c in result["content"]] == ["text", "image"]


def test_claude_code_brain_bridges_sdk_tool_calls_to_the_harness(monkeypatch):
    """The SDK calls tools on its own loop; each call must reach step() and wait for its result."""
    import claude_agent_sdk as sdk
    from harness.room.brains import ClaudeCodeBrain

    seen = {}

    class FakeClient:
        def __init__(self, options):
            seen["options"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def query(self, task):
            seen["task"] = task

        async def receive_response(self):
            tools = {t.name: t for t in seen["options"].mcp_servers["reins"]}
            yield sdk.AssistantMessage(content=[sdk.TextBlock("Looking first.")], model="m")
            seen["look"] = await tools["look"].handler({})
            yield sdk.AssistantMessage(content=[sdk.TextBlock("Done.")], model="m")
            yield sdk.ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                                    num_turns=2, session_id="s")

    monkeypatch.setattr(sdk, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(sdk, "create_sdk_mcp_server", lambda name, tools: tools)
    monkeypatch.setattr("harness.room.brains._leave_host_session", lambda: None)
    brain = ClaudeCodeBrain()
    brain.start("sys", [{"name": "look", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
                "what's there?")
    turn = brain.step([])
    assert turn.text == "Looking first." and turn.calls[0].name == "look"
    turn = brain.step([ToolResult(turn.calls[0].id, "Head camera image.", b"\x89PNG fake")])
    assert turn.done and turn.text == "Done."
    assert [c["type"] for c in seen["look"]["content"]] == ["text", "image"]
    options = seen["options"]
    assert options.tools == [] and options.setting_sources == []  # robot tools only, no user config
    assert options.allowed_tools == ["mcp__reins__look"]


def test_planner_prefers_a_straight_line_when_clear():
    assert planner.plan_path(lambda x, y: float("inf"), (0, 0), (2, 1)) == [(0, 0), (2, 1)]
