"""The model side of the harness, behind one small interface.

A `Brain` is started with a system prompt, the tool definitions and the task,
then stepped: each step takes the results of the tool calls it asked for last
time and returns its next turn. Each brain keeps its own conversation in its
own provider's format, so the harness never depends on one vendor's message
shapes. That's what keeps Reins model-agnostic: adding a provider is one class.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    id: str
    text: str
    image_png: bytes | None = None
    is_error: bool = False


@dataclass
class Turn:
    text: str
    calls: list[ToolCall] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return not self.calls


class Brain(Protocol):
    name: str

    def start(self, system: str, tools: list[dict], task: str) -> None: ...
    def step(self, results: list[ToolResult]) -> Turn: ...


class AnthropicBrain:
    """Claude through the Anthropic Messages API, with a manual tool loop.

    The loop is manual rather than the SDK's tool runner because the harness
    owns execution: every acting call goes through human review first.
    """

    def __init__(self, model: str = "claude-opus-5", effort: str = "high", max_tokens: int = 16000):
        import anthropic  # only needed for this brain
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic/{model}"
        self.messages: list[dict] = []

    def start(self, system: str, tools: list[dict], task: str) -> None:
        self.system, self.tools = system, tools
        self.messages = [{"role": "user", "content": task}]

    def step(self, results: list[ToolResult]) -> Turn:
        if results:
            self.messages.append({"role": "user", "content": [_anthropic_result(r) for r in results]})
        extra = {}
        if self.model in ("claude-opus-5", "claude-fable-5-1"):
            # On a safety-classifier decline, the API re-runs the turn on a fallback model.
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=self.system, tools=self.tools,
            messages=self.messages, thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # the growing history, images included, is re-sent every turn
            **extra)
        # Keep every block (thinking, fallback markers) so the next request replays them unchanged.
        self.messages.append({"role": "assistant", "content": response.content})
        text = "\n".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "refusal":
            details = response.stop_details
            return Turn(f"[model declined: {getattr(details, 'category', None) or 'refusal'}] {text}".strip())
        if response.stop_reason == "max_tokens":
            return Turn(f"[stopped: hit max_tokens] {text}".strip())
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        return Turn(text, calls)


def _anthropic_result(r: ToolResult) -> dict:
    content: list[dict] = [{"type": "text", "text": r.text}]
    if r.image_png:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                    "data": base64.standard_b64encode(r.image_png).decode()}})
    return {"type": "tool_result", "tool_use_id": r.id, "content": content, "is_error": r.is_error}


class ClaudeCodeBrain:
    """Claude through the local Claude Code CLI, so it runs on your Claude Code login.

    Uses the Claude Agent SDK, which starts `claude` as a subprocess. The
    harness tools are offered as in-process MCP tools, Claude Code's own tools
    (Bash, file edits, web) are switched off, and your Claude Code settings,
    hooks and CLAUDE.md files aren't loaded, so the model can only act through
    the robot tools.

    The SDK drives its own loop and calls tools from its event loop. A
    background thread bridges that to `step()`: each tool call is handed to the
    harness as a one-call Turn, and the tool waits there until the harness
    (review, execution) passes back the result.
    """

    def __init__(self, model: str | None = None, effort: str = "high", max_turns: int = 60):
        import claude_agent_sdk  # noqa: F401  (fail at startup if it's missing)
        self.model, self.effort, self.max_turns = model, effort, max_turns
        self.name = f"claude-code/{model or 'default'}"

    def start(self, system: str, tools: list[dict], task: str) -> None:
        import queue
        import threading
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._pending: dict[str, tuple] = {}  # call id -> (future, loop)
        self._text: list[str] = []
        self._ids = 0
        threading.Thread(target=self._run, args=(system, tools, task), daemon=True).start()

    def step(self, results: list[ToolResult]) -> Turn:
        import queue
        for r in results:
            future, loop = self._pending.pop(r.id)
            loop.call_soon_threadsafe(future.set_result, r)
        while True:
            try:
                kind, payload, text = self._events.get(timeout=0.5)  # timeout keeps Ctrl-C working
                break
            except queue.Empty:
                continue
        if kind == "error":
            raise payload
        return Turn(text, [payload]) if kind == "call" else Turn(text)

    def _take_text(self) -> str:
        text, self._text = "\n".join(self._text), []
        return text

    def _run(self, system: str, tools: list[dict], task: str) -> None:
        import asyncio
        try:
            asyncio.run(self._session(system, tools, task))
        except Exception as e:  # surfaced to the harness on its next step()
            self._events.put(("error", e, ""))

    async def _session(self, system: str, tools: list[dict], task: str) -> None:
        import asyncio

        from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage,
                                      TextBlock, create_sdk_mcp_server, tool)

        def handler_for(name: str):
            async def handler(args: dict) -> dict:
                self._ids += 1
                call = ToolCall(f"cc-{self._ids}", name, dict(args))
                future = asyncio.get_running_loop().create_future()
                self._pending[call.id] = (future, asyncio.get_running_loop())
                self._events.put(("call", call, self._take_text()))
                result: ToolResult = await future
                content: list[dict] = [{"type": "text", "text": result.text}]
                if result.image_png:
                    content.append({"type": "image", "mimeType": "image/png",
                                    "data": base64.standard_b64encode(result.image_png).decode()})
                return {"content": content, "is_error": result.is_error}
            return handler

        server = create_sdk_mcp_server("reins", tools=[
            tool(t["name"], t["description"], t["input_schema"])(handler_for(t["name"])) for t in tools])
        options = ClaudeAgentOptions(
            tools=[],  # no built-in Claude Code tools: the robot tools are all it gets
            mcp_servers={"reins": server}, strict_mcp_config=True,
            allowed_tools=[f"mcp__reins__{t['name']}" for t in tools],  # review happens in the harness
            system_prompt=system, setting_sources=[],
            model=self.model, effort=self.effort, max_turns=self.max_turns)
        _leave_host_session()
        final = ""
        async with ClaudeSDKClient(options) as client:
            await client.query(task)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    self._text += [b.text for b in message.content if isinstance(b, TextBlock) and b.text]
                elif isinstance(message, ResultMessage):
                    if message.is_error or "Failed to authenticate" in (message.result or ""):
                        final = f"[Claude Code ended with {message.subtype}] {message.result or ''}".strip()
                        if "authenticate" in final:
                            final += ("\nLog the Claude Code CLI in once from a normal terminal: "
                                      "claude auth login")
        self._events.put(("done", None, final or self._take_text()))


def _leave_host_session() -> None:
    """Drop the variables a parent Claude Code session puts in the environment.

    Launched from inside Claude Code (its terminal, or the desktop app), this
    process inherits the host session's proxy URL and auth hand-off variables.
    The `claude` subprocess would then try to borrow the host's login and fail,
    so it should use its own. Outside a Claude Code session this does nothing.
    """
    import os
    if "CLAUDECODE" not in os.environ and "CLAUDE_CODE_HOST_SESSION_ID" not in os.environ:
        return
    for key in list(os.environ):
        if key == "CLAUDECODE" or key.startswith(("CLAUDE_CODE_", "CLAUDE_AGENT_SDK_")) or key in (
                "ANTHROPIC_BASE_URL", "CLAUDE_PID", "CLAUDE_EFFORT") or key.startswith("CLAUDE_PREVIEW_"):
            del os.environ[key]


class ScriptedBrain:
    """Plays back a fixed list of tool calls. For demos without an API key, and tests.

    Stops early, reporting why, if a call errors or the operator declines.
    """

    def __init__(self, calls: list[tuple[str, dict]], name: str = "scripted"):
        self.script = list(calls)
        self.name = name

    def start(self, system: str, tools: list[dict], task: str) -> None:
        self.index = 0

    def step(self, results: list[ToolResult]) -> Turn:
        for r in results:
            if r.is_error or r.text.startswith("Operator declined"):
                return Turn(f"Stopping: the script can't adapt. Last result: {r.text}")
        if self.index >= len(self.script):
            return Turn("Script finished.")
        name, args = self.script[self.index]
        self.index += 1
        return Turn(f"Step {self.index} of {len(self.script)}.", [ToolCall(f"call-{self.index}", name, args)])
