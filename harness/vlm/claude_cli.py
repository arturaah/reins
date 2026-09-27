"""Claude Code CLI provider: `claude -p` under the machine's Claude login, no API key.

Each call writes the images to a temp folder, asks Claude Code to Read them and answer with JSON
only, and parses the CLI's JSON envelope. Slower than the API (CLI start-up per call) but runs
the loop unattended on the subscription. Flags per the Claude Code docs: --output-format json,
--json-schema, --allowed-tools Read, --permission-mode dontAsk, --max-turns, --model.
"""
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .base import VLM, VLMResponse


class ClaudeCliVLM(VLM):
    name = "claude-cli"

    def __init__(self, cfg):
        v = cfg["vlm"]
        self.model = v.get("model", "claude-fable-5-1")
        self.timeout = float(v.get("timeout_s", 120))
        self.bin = shutil.which("claude")
        if not self.bin:
            raise RuntimeError("the `claude` CLI is not on PATH")

    def _call(self, prompt, images, schema):
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="harness_vlm_") as d:
            paths = []
            for label, jpg in images:
                p = Path(d) / (label.split()[0].lower() + ".jpg"); p.write_bytes(jpg); paths.append((label, str(p)))
            head = "Look at these camera images first, using the Read tool on each file in this order:\n" + \
                   "".join(f"- {label}: {path}\n" for label, path in paths) + \
                   "Then answer the prompt below. Output the JSON object only, no other text.\n\n"
            cmd = [self.bin, "-p", head + prompt, "--output-format", "json", "--allowed-tools", "Read",
                   "--permission-mode", "dontAsk", "--max-turns", str(len(paths) + 3), "--model", self.model]
            if schema:
                cmd += ["--json-schema", json.dumps(schema)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=d)
            except subprocess.TimeoutExpired:
                return VLMResponse("", self.model, time.time() - t0, error=f"claude -p timed out after {self.timeout:.0f} s")
        lat = time.time() - t0
        if r.returncode != 0:
            return VLMResponse("", self.model, lat, error=f"claude -p exit {r.returncode}: {(r.stderr or r.stdout)[-300:]}")
        try:
            env = json.loads(r.stdout)
        except json.JSONDecodeError:
            return VLMResponse(r.stdout.strip(), self.model, lat, error="claude -p did not return a JSON envelope")
        out = env.get("structured_output")
        text = json.dumps(out) if isinstance(out, dict) else str(env.get("result", ""))
        usage = env.get("usage") or {}
        resp = VLMResponse(text, env.get("model") or self.model, lat, int(usage.get("input_tokens", 0) or 0),
                           int(usage.get("output_tokens", 0) or 0), "end_turn",
                           raw={**{k: env.get(k) for k in ("total_cost_usd", "num_turns", "session_id")},
                                **{k: usage.get(k) for k in ("cache_read_input_tokens", "cache_creation_input_tokens")}})
        if env.get("is_error"):
            resp.error = f"claude -p error: {text[:200]}"
        return resp

    def plan(self, prompt, images, schema=None):
        return self._call(prompt, images, schema)

    def act(self, prompt, images, schema=None, retry_note=None):
        if retry_note:
            prompt += f"\n\nYour previous answer was rejected: {retry_note}. Answer again following the contract exactly."
        return self._call(prompt, images, schema)
