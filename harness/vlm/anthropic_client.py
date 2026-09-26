"""Claude adapter: structured JSON output, retries, timeouts, latency and token logging.

Model: cfg vlm.model (default claude-fable-5-1). Thinking is always on for this model, so no
thinking parameter is sent; depth comes from output_config.effort. A refusal stop reason is
returned as an error (a failed step) unless vlm.fallbacks is "default", which lets the API
re-run the request on another model server-side; the served model is logged either way.
"""
import base64
import time

import anthropic

from .base import VLM, VLMResponse

BETA_FALLBACK = "server-side-fallback-2026-07-01"


class AnthropicVLM(VLM):
    name = "anthropic"

    def __init__(self, cfg):
        v = cfg["vlm"]
        self.model = v["model"]
        self.effort = v.get("effort", "medium")
        self.max_tokens = int(v.get("max_tokens", 2000))
        self.retries = int(v.get("retries", 2))
        self.fallbacks = v.get("fallbacks")
        self.client = anthropic.Anthropic(timeout=float(v.get("timeout_s", 120)), max_retries=self.retries)

    def _content(self, prompt, images):
        blocks = []
        for label, jpeg in images:
            blocks.append({"type": "text", "text": f"Image: {label}"})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.standard_b64encode(jpeg).decode()}})
        blocks.append({"type": "text", "text": prompt})
        return blocks

    def _call(self, prompt, images, schema):
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role": "user", "content": self._content(prompt, images)}],
                      output_config={"effort": self.effort, **({"format": {"type": "json_schema", "schema": schema}} if schema else {})})
        t0 = time.time()
        try:
            if self.fallbacks == "default":
                resp = self.client.beta.messages.create(betas=[BETA_FALLBACK], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            return VLMResponse("", self.model, time.time() - t0, error=f"rate limited: {e.message}")
        except anthropic.APIStatusError as e:
            return VLMResponse("", self.model, time.time() - t0, error=f"API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return VLMResponse("", self.model, time.time() - t0, error=f"connection error: {e}")
        lat = time.time() - t0
        usage = getattr(resp, "usage", None)
        out = VLMResponse("", getattr(resp, "model", self.model), lat,
                          getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0,
                          getattr(resp, "stop_reason", "") or "")
        if resp.stop_reason == "refusal":
            det = getattr(resp, "stop_details", None)
            out.error = "refusal" + (f" ({det.category}: {det.explanation})" if det else "")
            return out
        out.text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if resp.stop_reason == "max_tokens":
            out.error = "max_tokens reached before the JSON was complete"
        return out

    def plan(self, prompt, images, schema=None):
        return self._call(prompt, images, schema)

    def act(self, prompt, images, schema=None, retry_note=None):
        if retry_note:
            prompt = prompt + f"\n\nYour previous answer was rejected: {retry_note}. Answer again following the contract exactly."
        return self._call(prompt, images, schema)
