"""API provider parity through fake Responses/registry transports; no paid calls."""
import base64
import copy
import io
import json
import os
import threading
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from jsonschema import Draft202012Validator

from core.codex_chat import ToolLink
from core.dashboard_chat import DashboardChat, INSTRUCTIONS, SCHEMA, configuration, validate_reply
from core.openai_chat import Cancelled, OpenAIResponder, function_output, tool_definitions
from core.tool_specs import TOOL_NAMES, TOOL_SPECS

ENV = {"OPENAI_API_KEY": "fake-provider-secret", "REINS_CHAT_MODEL": "fake-model"}
LINK = ToolLink("http://127.0.0.1:8090/api/tools", "fake-registry-secret")
ANSWER = {"reply": "The complete motion is ready for your review.", "robot_request": None, "trajectory": None}


def completed(*items):
    return {"status": "completed", "output": list(items)}


def call(name, arguments=None, ident="call_001"):
    return {"type": "function_call", "name": name, "call_id": ident, "status": "completed", "arguments": json.dumps(arguments or {})}


def final():
    return completed({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(ANSWER)}]})


def responder(responses, tool_call):
    sent = []
    results = iter(responses)
    def request(payload, timeout):
        sent.append(copy.deepcopy(payload))
        return next(results)
    agent = OpenAIResponder(INSTRUCTIONS, SCHEMA, LINK, configuration, validate_reply, request, tool_call)
    return agent, sent


@pytest.fixture(autouse=True)
def config(monkeypatch):
    for key, value in ENV.items(): monkeypatch.setenv(key, value)


def test_tool_definitions_are_canonical_and_strict():
    definitions = tool_definitions()
    assert [d["name"] for d in definitions] == TOOL_NAMES
    assert not {"approve", "execute_motion", "firmware", "decision"}.intersection(TOOL_NAMES)
    for definition in definitions:
        schema = definition["parameters"]
        assert definition["strict"] and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    observe = next(d["parameters"] for d in definitions if d["name"] == "observe")
    Draft202012Validator(observe).validate({"cameras": None})
    # Registry schemas must retain their MCP optional/default behavior.
    assert next(d for d in TOOL_SPECS if d["name"] == "observe")["inputSchema"]["required"] == []


def test_complete_tool_loop_revises_and_preserves_images_and_reasoning():
    image = io.BytesIO(); Image.new("RGB", (3, 2), "blue").save(image, "JPEG")
    encoded = base64.b64encode(image.getvalue()).decode()
    reasoning = {"type": "reasoning", "id": "reasoning_1", "summary": [], "encrypted_content": "opaque-encrypted-value"}
    first = {"name": "Wave", "arm": "right", "waypoints": [{"position_m": [.2, -.2, .8], "hold_s": 0}], "return_to_start": False, "observation_id": "observe1"}
    revised = {**first, "waypoints": [{"position_m": [.25, -.2, .9], "hold_s": 0}]}
    inputs = [completed(reasoning, call("get_robot_context")),
              completed(call("observe", {"cameras": None}, "call_002")),
              completed(call("plan_hand_path", first, "call_003")),
              completed(call("plan_hand_path", revised, "call_004")),
              completed(call("preview_plan", {"plan_id": "draft001"}, "call_005")),
              completed(call("propose_motion", {"plan_id": "draft001", "request_id": "request1"}, "call_006")), final()]
    dispatched = []
    def tools(name, args, timeout):
        dispatched.append((name, copy.deepcopy(args)))
        if name == "observe":
            return {"observation": {"id": "observe1"}, "content_blocks": [
                {"type": "text", "text": "head camera: real frame"},
                {"type": "image", "data": encoded, "mimeType": "image/jpeg"}]}
        if name == "plan_hand_path" and len([d for d in dispatched if d[0] == name]) == 1:
            return {"state": "blocked", "message": "head envelope; revise path"}
        if name == "propose_motion": return {"state": "review", "proposal_id": "proposal1"}
        return {"state": "draft", "plan_id": "draft001"}
    agent, sent = responder(inputs, tools)
    assert agent([{"role": "user", "text": "Prepare a wave"}], {}) == ANSWER
    assert [d[0] for d in dispatched] == ["get_robot_context", "observe", "plan_hand_path", "plan_hand_path", "preview_plan", "propose_motion"]
    assert dispatched[1][1] == {}  # strict null becomes the registry's default argument
    assert dispatched[2][1]["waypoints"] != dispatched[3][1]["waypoints"]
    assert reasoning in sent[-1]["input"]
    observation = next(i for i in sent[2]["input"] if i.get("type") == "function_call_output" and i["call_id"] == "call_002")
    assert observation["output"][-1]["type"] == "input_image"
    assert observation["output"][-1]["image_url"] == "data:image/jpeg;base64,"+encoded
    assert encoded not in observation["output"][0]["text"]
    assert sent[-1]["tool_choice"] == "none"
    assert all(p["store"] is False and not p["parallel_tool_calls"] for p in sent)
    assert all("previous_response_id" not in p for p in sent)
    assert "fake-registry-secret" not in json.dumps(sent)
    assert "fake-provider-secret" not in json.dumps(sent)


@pytest.mark.parametrize("item", [call("approve_motion"), call("plan_hand_action", {"arm": "right", "closed": "yes"}),
                                   {**call("observe"), "arguments": "[]"}, {**call("observe"), "arguments": "{broken"},
                                   {**call("observe"), "status": "in_progress"}])
def test_invalid_calls_return_failure_without_dispatch(item):
    tools = Mock()
    agent, sent = responder([completed(item), final()], tools)
    assert agent([], {}) == ANSWER
    tools.assert_not_called()
    output = sent[1]["input"][-1]
    assert output["type"] == "function_call_output" and json.loads(output["output"])["state"] == "blocked"


def test_repeated_call_id_is_never_executed_twice():
    tools = Mock(return_value={"state": "ok"})
    agent, _ = responder([completed(call("get_robot_context")), completed(call("get_robot_context"))], tools)
    with pytest.raises(ValueError, match="repeated"): agent([], {})
    assert tools.call_count == 1


def test_no_additional_tools_after_proposal():
    tools = Mock(return_value={"state": "review", "proposal_id": "proposal1"})
    agent, _ = responder([completed(call("propose_motion", {"plan_id": "draft001", "request_id": "req1"})),
                          completed(call("get_robot_context", ident="call_002"))], tools)
    with pytest.raises(ValueError, match="planning phase"): agent([], {})
    assert tools.call_count == 1


def test_cancellation_after_network_reply_prevents_late_proposal():
    tools = Mock()
    agent, _ = responder([], tools)
    def request(payload, timeout):
        agent.cancel()
        return completed(call("propose_motion", {"plan_id": "draft001", "request_id": "req1"}))
    agent.request = request
    with pytest.raises(Cancelled): agent([], {})
    tools.assert_not_called()


def test_cancellation_during_tool_prevents_next_round():
    agent, sent = responder([completed(call("get_robot_context"))], None)
    def tools(*args): agent.cancel(); return {"state": "ok"}
    agent.call_tool = tools
    with pytest.raises(Cancelled): agent([], {})
    assert len(sent) == 1


def test_incomplete_response_never_dispatches_tools():
    tools = Mock()
    agent, _ = responder([{**completed(call("get_robot_context")), "status": "incomplete"}], tools)
    with pytest.raises(ValueError, match="did not finish"): agent([], {})
    tools.assert_not_called()


def test_round_and_call_budgets_are_bounded():
    tools = Mock(return_value={"state": "ok"})
    agent, sent = responder([completed(call("get_robot_context", ident="call_"+str(i))) for i in range(3)], tools)
    agent.MAX_ROUNDS = 2
    with pytest.raises(ValueError, match="round budget"): agent([], {})
    assert len(sent) == tools.call_count == 2
    tools.reset_mock()
    agent, _ = responder([completed(*(call("get_robot_context", ident="call_"+str(i)) for i in range(3)))], tools)
    agent.MAX_TOOL_CALLS = 2
    with pytest.raises(ValueError, match="tool-call budget"): agent([], {})
    tools.assert_not_called()


def http_response(value):
    reply = Mock()
    reply.__enter__ = Mock(return_value=reply); reply.__exit__ = Mock(return_value=False)
    reply.read.return_value = json.dumps(value).encode()
    return reply


def test_http_tools_use_only_the_local_registry_token():
    agent = OpenAIResponder(INSTRUCTIONS, SCHEMA, LINK, configuration, validate_reply)
    with patch("urllib.request.urlopen", return_value=http_response({"observation": {"id": "observe1"}})) as network:
        result = agent._call_tool("observe", {}, 2)
    request = network.call_args.args[0]
    assert request.full_url == LINK.url+"/observe"
    assert request.get_header("X-reins-tool-token") == LINK.token
    assert request.get_header("Authorization") is None
    assert result["observation"]["id"] == "observe1"
    agent.tools = ToolLink("http://example.org/api/tools", LINK.token)
    with patch("urllib.request.urlopen") as network:
        with pytest.raises(ValueError, match="local dashboard"): agent._call_tool("observe", {}, 2)
        network.assert_not_called()


def test_api_backend_has_tools_and_independent_revision_transport():
    with patch("core.codex_chat.CodexResponder._configuration", return_value={"configured": False}), \
         patch("core.claude_chat.ClaudeResponder._configuration", return_value={"configured": False}):
        chat = DashboardChat(backend="openai", tools=LINK)
        assert isinstance(chat.responder, OpenAIResponder) and chat.status()["tools"]
        revision = chat.motion_reviser()
        separate = revision.factory()
        assert separate is not chat.responder and separate.tools is None
        chat.responder.prepare(); chat.responder.cancel()
        assert chat.responder.cancelled.is_set() and not separate.cancelled.is_set()
        chat.close(); separate.close()


def test_invalid_images_are_not_silently_replaced_by_captions():
    with pytest.raises(ValueError, match="encoding"):
        function_output({"content_blocks": [{"type": "image", "mimeType": "image/jpeg", "data": "bad-base64"}]})
