"""The claude-cli provider parses the CLI's JSON envelope; a fake `claude` on PATH stands in for the real one."""
import json
import os
import stat

from harness.vlm.claude_cli import ClaudeCliVLM


def fake_claude(tmp_path, envelope):
    script = tmp_path / "claude"
    script.write_text("#!/bin/sh\n" + f"cat <<'EOF'\n{json.dumps(envelope)}\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_structured_output_and_result(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    fake_claude(tmp_path, {"result": "ignored", "structured_output": {"decision": "MV_UP", "reasoning": "WRIST: NO"},
                           "model": "claude-fable-5-1", "usage": {"input_tokens": 12, "output_tokens": 34}, "total_cost_usd": 0.1})
    v = ClaudeCliVLM(cfg)
    r = v.act("P", [("CONTEXT VIEW", b"\xff\xd8")], {"type": "object"})
    assert json.loads(r.text)["decision"] == "MV_UP" and r.input_tokens == 12 and not r.error
    fake_claude(tmp_path, {"result": '{"decision": "DONE", "reasoning": "x"}', "is_error": False})
    assert json.loads(ClaudeCliVLM(cfg).plan("P", []).text)["decision"] == "DONE"
    fake_claude(tmp_path, {"result": "boom", "is_error": True})
    assert "error" in ClaudeCliVLM(cfg).plan("P", []).error
