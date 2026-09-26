"""OpenAI adapter (optional, gpt-6-astra). Same interface as the Claude adapter.

Requires the `openai` package and OPENAI_API_KEY. Uses the Responses API with JSON schema output.
Untested against a live key in this repo; kept minimal so the harness stays model-agnostic.
"""
import base64
import json
import time

from .base import VLM, VLMResponse


class OpenAIVLM(VLM):
    name = "openai"

    def __init__(self, cfg):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("pip install openai to use vlm.provider: openai") from e
        v = cfg["vlm"]
        self.model = v.get("openai_model", "gpt-6-astra")
        self.max_tokens = int(v.get("max_tokens", 2000))
        self.client = openai.OpenAI(timeout=float(v.get("timeout_s", 120)), max_retries=int(v.get("retries", 2)))

    def _content(self, prompt, images):
        parts = []
        for label, jpeg in images:
            parts.append({"type": "input_text", "text": f"Image: {label}"})
            parts.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode()})
        parts.append({"type": "input_text", "text": prompt})
        return parts

    def _call(self, prompt, images, schema):
        t0 = time.time()
        kwargs = dict(model=self.model, max_output_tokens=self.max_tokens,
                      input=[{"role": "user", "content": self._content(prompt, images)}])
        if schema:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "harness", "schema": schema, "strict": True}}
        try:
            resp = self.client.responses.create(**kwargs)
        except Exception as e:                       # the openai package's error classes are not imported at module level
            return VLMResponse("", self.model, time.time() - t0, error=f"{type(e).__name__}: {e}")
        usage = getattr(resp, "usage", None)
        return VLMResponse(getattr(resp, "output_text", "") or "", getattr(resp, "model", self.model), time.time() - t0,
                           getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0, "end_turn")

    def plan(self, prompt, images, schema=None):
        return self._call(prompt, images, schema)

    def act(self, prompt, images, schema=None, retry_note=None):
        if retry_note:
            prompt += f"\n\nYour previous answer was rejected: {retry_note}. Answer again following the contract exactly."
        return self._call(prompt, images, schema)
