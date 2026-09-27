"""Real planning/review pipeline behind HTTP tools; model and actuators are fake."""
import base64
import copy
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import numpy as np
from PIL import Image

from contract.runtime import digest
from core.codex_chat import ToolLink
from core.dashboard_chat import DashboardChat
from core.prompt_planner import PromptPlanner
from core.reins_tools import ReinsTools
from core.robot_pipeline import RobotPipeline
from core.test_generated_motion import DRAFT
from core.test_robot_pipeline import FakeRobot, Feed
from tools.dashboard import Simulation


def wait_until(condition, description, seconds=15):
    deadline = time.monotonic()+seconds
    while time.monotonic()<deadline:
        value = condition()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("Timed out waiting for "+description)


def output_of(payload, call_id):
    return next(item["output"] for item in payload["input"]
                if item.get("type") == "function_call_output" and item["call_id"] == call_id)


def test_scripted_agent_observes_repairs_previews_and_proposes_once(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-no-network-key")
    monkeypatch.setenv("REINS_CHAT_MODEL", "scripted-test-model")
    monkeypatch.setenv("REINS_CODEX_BIN", "/not-installed-for-this-test")
    monkeypatch.setenv("REINS_CLAUDE_BIN", "/not-installed-for-this-test")
    sim, planner = Simulation(), PromptPlanner(output_dir=tmp_path/"plans")
    feed = Feed()
    pipe = RobotPipeline(planner, sim, {"head": feed}, run_dir=tmp_path/"runs")
    pipe.cfg["perception"]["pose_view"] = False
    robot = FakeRobot(pipe.planning_pose())
    pipe.backend_factory = lambda: robot
    pipe.command({"action": "connect", "table_z_m": .6})
    wait_until(lambda: pipe.connected and not pipe.status()["busy"], "fake read-only connection")
    before = robot.joints()
    frame = np.full((48, 64, 3), [30, 90, 150], dtype=np.uint8)
    detector = Mock(available=True, open_vocabulary=True, name="fake-detector")
    registry = ReinsTools(detector, {"head": lambda: (frame, time.monotonic(), "head-frame-1", None)},
                          planner, sim.show_proposal, sim.status, lambda: {"head": {"online": True}})
    registry.pipeline = pipe
    dispatched, provider_inputs = [], []
    token = "fake-local-tool-token"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            if self.headers.get("X-Reins-Tool-Token") != token:
                self.send_error(403); return
            name = self.path.removeprefix("/api/tools/")
            args = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                result = registry.call(name, args)
                dispatched.append((name, copy.deepcopy(args), copy.deepcopy(result)))
                encoded, status = json.dumps(result, allow_nan=False).encode(), 200
            except ValueError as exc:
                encoded, status = json.dumps({"error": str(exc)}).encode(), 400
            self.send_response(status); self.send_header("Content-Length", str(len(encoded))); self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    chat = DashboardChat(context=registry.get_robot_context, backend="openai",
                         tools=ToolLink(f"http://127.0.0.1:{server.server_port}/api/tools", token))
    chat.before_turn, chat.on_cancel = registry.begin_turn, registry.cancel
    pipe.on_result = chat.record_motion_result
    stage, observation_id, draft_id = 0, None, None

    def model(payload, timeout):
        nonlocal stage, observation_id, draft_id
        provider_inputs.append(copy.deepcopy(payload))
        stage += 1
        if stage == 1:
            name, arguments = "get_robot_context", {}
        elif stage == 2:
            name, arguments = "observe", {"cameras": ["head"]}
        elif stage == 3:
            observed = output_of(payload, "step2")
            metadata = json.loads(observed[0]["text"])
            observation_id = metadata["observation"]["id"]
            image = next(item for item in observed if item["type"] == "input_image")
            decoded = Image.open(io.BytesIO(base64.b64decode(image["image_url"].split(",", 1)[1])))
            assert decoded.size == (64, 48)
            assert pipe.proposal is None and robot.sent == []
            # Within configured workspace, but unreachable by the selected right arm.
            name, arguments = "plan_hand_path", {"name": "Blow a kiss", "arm": "right", "return_to_start": True,
                "waypoints": [{"position_m": [.59, .59, 1.24], "hold_s": 0}], "observation_id": observation_id}
        elif stage == 4:
            rejection = json.loads(output_of(payload, "step3"))
            assert rejection["state"] == "blocked" and rejection["retryable"]
            assert rejection["failures"] and "unreachable" in rejection["message"].lower()
            assert pipe.proposal is None and robot.sent == []
            name, arguments = "plan_hand_path", {**{key: copy.deepcopy(value) for key, value in DRAFT.items() if key != "frame"},
                                                   "observation_id": observation_id}
        elif stage == 5:
            draft = json.loads(output_of(payload, "step4"))
            assert draft["state"] == "draft", draft
            assert not draft["execution_allowed"] and pipe.proposal is None and robot.sent == []
            draft_id = draft["plan_id"]
            name, arguments = "preview_plan", {"plan_id": draft_id}
        elif stage == 6:
            preview = json.loads(output_of(payload, "step5"))
            assert preview["state"] == "previewed" and sim.playing
            assert pipe.proposal is None and pipe.glasses_message()["review"] is None and robot.sent == []
            name, arguments = "propose_motion", {"plan_id": draft_id, "request_id": "one-complete-kiss"}
        else:
            if stage == 7:
                proposal = json.loads(output_of(payload, "step6"))
                assert proposal["state"] == "review" and payload["tool_choice"] == "none"
                reply = "The complete non-contact gesture is ready for your review."
            else:
                assert stage == 8
                encoded = json.dumps(payload["input"])
                assert "Reins runtime outcome (not model judgment)" in encoded
                assert '"outcome": "executed"' in encoded.replace('\\"', '"')
                assert "measured_end_pose" in encoded and "tracking_error" in encoded
                reply = "The runtime reports the approved motion completed with measured feedback."
            return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps({"reply": reply, "robot_request": None, "trajectory": None})}]}]}
        return {"status": "completed", "output": [{"type": "function_call", "name": name, "call_id": "step"+str(stage),
                                                       "status": "completed", "arguments": json.dumps(arguments)}]}

    def checked_model(payload, timeout):
        try:
            return model(payload, timeout)
        except AssertionError as exc:
            raise ValueError(f"Scripted model round {stage}: {exc}") from exc
    chat.responder.request = checked_model
    try:
        chat.send("Observe and prepare a non-contact blow-a-kiss gesture with the right arm.")
        wait_until(lambda: not chat.status()["busy"], "scripted planning turn")
        assert chat.status()["error"] is None, chat.status()["error"]
        assert [entry[0] for entry in dispatched] == ["get_robot_context", "observe", "plan_hand_path", "plan_hand_path", "preview_plan", "propose_motion"]
        assert pipe.revision == 1 and robot.sent == [] and robot.joints() == before
        proposal = copy.deepcopy(pipe.proposal)
        reviewed_payload = copy.deepcopy(pipe.plan)
        assert proposal["observation_id"] == observation_id
        assert pipe.glasses_message()["review"]["digest"] == proposal["digest"]
        # The only human decision in the entire observe/replan/preview/run workflow.
        pipe.decide(proposal["id"], proposal["digest"], "approve")
        wait_until(lambda: pipe.last_result and pipe.last_result["outcome"] == "executed", "approved fake execution")
        wait_until(lambda: any("runtime_result" in m for m in chat.status()["messages"]), "runtime conversation feedback")
        assert robot.sent == [reviewed_payload] and digest(robot.sent[0]) == proposal["digest"]
        result = pipe.motion_result(proposal["id"])
        assert result["measured_end_pose"] == robot.joints() and result["tracking_error"] == 0.
        assert pipe.propose_motion(draft_id, "one-complete-kiss")["outcome"] == "executed"
        assert len(robot.sent) == 1
        chat.send("What happened?")
        wait_until(lambda: not chat.status()["busy"], "next turn sees authoritative outcome")
        assert chat.status()["error"] is None and stage == 8
        assert len(robot.sent) == 1
    finally:
        chat.close(); pipe.close(); server.shutdown(); server.server_close(); thread.join(2)
