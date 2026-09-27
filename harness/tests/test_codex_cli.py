import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.actions import OUTPUT_SCHEMA
from harness.split import SNAPSHOT_SCHEMA
from harness.vlm import base
from harness.vlm.codex_cli import CodexCliVLM, strict_schema


def test_codex_images_schema_output_and_credentials(cfg, monkeypatch):
    monkeypatch.setattr("harness.vlm.codex_cli.shutil.which", lambda _: "/fake/codex")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-secret")
    folders = []
    def run(cmd, **kw):
        folders.append(kw["cwd"])
        assert cmd[cmd.index("--model") + 1] == "gpt-5.6-sol"
        assert 'model_reasoning_effort="low"' in cmd
        assert "--ephemeral" in cmd and "--ignore-user-config" in cmd
        assert cmd[cmd.index("--sandbox") + 1] == "read-only" and "features.shell_tool=false" in cmd
        assert "TYPESAFE_API_KEY" not in kw["env"]
        assert "Image 1: CONTEXT VIEW" in kw["input"] and "rejected: retry test" in kw["input"]
        assert Path(cmd[cmd.index("--image") + 1]).read_bytes() == b"jpeg"
        schema = json.loads(Path(cmd[cmd.index("--output-schema") + 1]).read_text())
        assert "plan" in schema["required"] and schema["properties"]["plan"]["anyOf"][-1] == {"type": "null"}
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text('{"decision":"MV_UP","reasoning":"WRIST: NO","plan":null}')
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 42, "output_tokens": 12, "cached_input_tokens": 10}}))
    monkeypatch.setattr("harness.vlm.codex_cli.subprocess.run", run)
    result = base.make(cfg, "codex-cli").act("P", [("CONTEXT VIEW", b"jpeg")], OUTPUT_SCHEMA, retry_note="retry test")
    assert not result.error and result.model == "gpt-5.6-sol" and result.input_tokens == 42
    assert json.loads(result.text)["decision"] == "MV_UP"
    assert result.raw["cached_input_tokens"] == 10 and not Path(folders[0]).exists()


def test_codex_strict_schema_preserves_original():
    schema = strict_schema(SNAPSHOT_SCHEMA)
    assert "next_goal_offset_cm" not in SNAPSHOT_SCHEMA["required"]
    assert "next_goal_offset_cm" in schema["required"]
    assert schema["properties"]["next_goal_offset_cm"]["anyOf"][0]["required"] == ["forward", "left", "up"]


@pytest.mark.parametrize("mode", ["failed", "exit", "empty", "timeout", "message"])
def test_codex_errors_and_message_fallback(cfg, monkeypatch, mode):
    monkeypatch.setattr("harness.vlm.codex_cli.shutil.which", lambda _: "/fake/codex")
    def run(cmd, **kw):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        event = {"type": "turn.failed", "error": {"message": "limit reached"}} if mode == "failed" else {
            "type": "item.completed", "item": {"type": "agent_message", "text": '{"decision":"DONE"}'}}
        return SimpleNamespace(returncode=1 if mode == "exit" else 0, stderr="exit test",
                               stdout="" if mode == "empty" else json.dumps(event))
    monkeypatch.setattr("harness.vlm.codex_cli.subprocess.run", run)
    result = CodexCliVLM(cfg).plan("P", [])
    assert bool(result.error) == (mode != "message")
    if mode == "message":
        assert json.loads(result.text)["decision"] == "DONE"
