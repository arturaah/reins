"""VLM adapter interface. plan(task, images) -> Plan, act(context, images) -> Decision text + metadata.

Images are (label, jpeg_bytes) pairs, sent before the text (Show-Harness "frontier mode").
Adapters return raw text plus call metadata; parsing lives in harness.actions / harness.prompts so
a parse failure can be re-prompted once with the error message.
"""
from dataclasses import dataclass, field


@dataclass
class VLMResponse:
    text: str
    model: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    error: str = ""
    raw: dict = field(default_factory=dict)


class VLM:
    name = "vlm"

    def plan(self, prompt, images, schema=None) -> VLMResponse:
        raise NotImplementedError

    def act(self, prompt, images, schema=None, retry_note=None) -> VLMResponse:
        """retry_note: the parse error of the previous attempt, appended once."""
        raise NotImplementedError


def make(cfg, name=None):
    name = name or cfg["vlm"]["provider"]
    if name == "anthropic":
        from .anthropic_client import AnthropicVLM
        return AnthropicVLM(cfg)
    if name == "openai":
        from .openai_client import OpenAIVLM
        return OpenAIVLM(cfg)
    if name == "scripted":
        from .scripted import ScriptedVLM
        return ScriptedVLM()
    if name == "chat":
        from .chat import ChatVLM
        return ChatVLM(cfg)
    raise ValueError(f"unknown vlm provider {name!r}")
