"""Codex CLI vision provider using the existing Codex login, attached images and structured JSON.

Each call runs ephemerally in a temporary directory with read-only permissions and shell tools
disabled. The model returns a plan/snapshot; only the harness executor can actuate the robot.
"""
import copy
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .base import VLM, VLMResponse


def strict_schema(schema):
    """OpenAI structured output requires every property; represent optional ones as nullable."""
    obj = copy.deepcopy(schema)
    def visit(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            props = node.get("properties", {})
            required = node.get("required", [])
            for key, child in props.items():
                visit(child)
                if key not in required:
                    props[key] = {"anyOf": [child, {"type": "null"}]}
            node["required"] = list(props)
            node["additionalProperties"] = False
        if "items" in node:
            visit(node["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            for child in node.get(key, []):
                visit(child)
    visit(obj)
    return obj


class CodexCliVLM(VLM):
    name = "codex-cli"

    def __init__(self, cfg):
        v = cfg["vlm"]
        self.model = v.get("codex_model", "gpt-5.6-sol")
        self.effort = v.get("codex_effort", "low")
        self.timeout = float(v.get("timeout_s", 120))
        self.key_env = cfg.get("executor", {}).get("jev_key_env", "TYPESAFE_API_KEY")
        self.bin = shutil.which("codex")
        if not self.bin:
            raise RuntimeError("the `codex` CLI is not on PATH")

    def _call(self, prompt, images, schema):
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="harness_codex_") as folder:
            root = Path(folder)
            output = root / "answer.json"
            cmd = [self.bin, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--json", "--color", "never", "--model", self.model,
                   "-c", f'model_reasoning_effort={json.dumps(self.effort)}',
                   "-c", 'approval_policy="never"', "-c", "features.shell_tool=false",
                   "-c", "features.multi_agent=false", "-c", 'web_search="disabled"',
                   "--output-last-message", str(output)]
            labels = []
            for i, (label, jpg) in enumerate(images):
                path = root / f"image_{i}.jpg"
                path.write_bytes(jpg)
                cmd += ["--image", str(path)]
                labels.append(f"Image {i + 1}: {label}")
            if schema:
                path = root / "schema.json"
                path.write_text(json.dumps(strict_schema(schema)))
                cmd += ["--output-schema", str(path)]
            cmd.append("-")
            head = ("You are the vision and planning component of a robot harness. Analyze the attached images in order. "
                    "Return only the requested JSON. Do not use tools or execute actions.\n" + "\n".join(labels) + "\n\n")
            # Jev's credential is unrelated to the planner and must not enter its subprocess.
            env = {k: v for k, v in os.environ.items() if k not in (self.key_env, "TYPESAFE_API_KEY")}
            try:
                result = subprocess.run(cmd, input=head + prompt, capture_output=True, text=True,
                                        timeout=self.timeout, cwd=folder, env=env)
            except subprocess.TimeoutExpired:
                return VLMResponse("", self.model, time.monotonic() - t0, error=f"codex exec timed out after {self.timeout:.0f} s")
            events = []
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.append(event)
                except json.JSONDecodeError:
                    pass
            usage, error, text = {}, "", ""
            for event in events:
                if event.get("type") == "turn.completed":
                    usage = event.get("usage") or {}
                if event.get("type") == "turn.failed":
                    error = str(event.get("error", "Codex turn failed"))
                item = event.get("item") or {}
                if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                    text = item.get("text", "")
            if output.exists():
                text = output.read_text().strip()
            if result.returncode:
                error = f"codex exec exit {result.returncode}: {error or (result.stderr or result.stdout)[-500:]}"
            elif not text and not error:
                error = "codex exec returned no final answer"
            return VLMResponse(text, self.model, time.monotonic() - t0,
                               int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                               "end_turn", error=error, raw={"cached_input_tokens": usage.get("cached_input_tokens", 0)})

    def plan(self, prompt, images, schema=None):
        return self._call(prompt, images, schema)

    def act(self, prompt, images, schema=None, retry_note=None):
        if retry_note:
            prompt += f"\nYour previous answer was rejected: {retry_note}. Follow the JSON contract exactly."
        return self._call(prompt, images, schema)
