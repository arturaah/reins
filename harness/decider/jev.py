"""TypeSafe Jev over its HTTP API (docs.typesafe.ai/api), standard library only.

POST {url} with {"state", "model", "questions"}; Authorization: Bearer $TYPESAFE_API_KEY. Answers come back under the
question ids with calibrated probabilities (Choice also carries a confidence). 429 and 529 are retried with backoff,
honouring retry-after. Jev is text only (no images) and weak at arithmetic, so the state it gets is words that code
computed from the geometry (harness.split). The response's `model` is the versioned id that answered; it is logged.
"""
import json
import math
import os
import time
import urllib.error
import urllib.request

from ..vlm.base import VLMResponse
from .base import Answer, Decider, question_json

RETRY_STATUS = (429, 500, 502, 503, 529)


class JevDecider(Decider):
    name = "jev"

    def __init__(self, cfg):
        e = cfg["executor"]
        self.url = e.get("jev_url", "https://api.typesafe.ai/v1/systemone")
        self.model = e.get("jev_model", "jev-latest")
        self.timeout = float(e.get("jev_timeout_s", 5.0))
        self.retries = int(e.get("jev_retries", 2))
        self.key = os.environ.get(e.get("jev_key_env", "TYPESAFE_API_KEY"), "")
        if not self.key:
            raise RuntimeError(f"Jev needs an API key in ${e.get('jev_key_env', 'TYPESAFE_API_KEY')} (console.typesafe.ai/keys)")

    def body(self, state, questions):
        return {"state": state, "model": self.model, "questions": {k: question_json(q) for k, q in questions.items()}}

    def ask(self, state, questions, facts=None):
        data = json.dumps(self.body(state, questions)).encode()
        req = urllib.request.Request(self.url, data=data, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        t0, err = time.time(), ""
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    obj = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                err = f"Jev HTTP {e.code}: {detail}"
                if e.code not in RETRY_STATUS or attempt == self.retries:
                    return {}, VLMResponse("", self.model, time.time() - t0, error=err)
                wait = e.headers.get("retry-after") if e.headers else None
                time.sleep(min(float(wait), 5.0) if wait and wait.replace(".", "", 1).isdigit() else 0.25 * 2 ** attempt)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = f"Jev connection error: {getattr(e, 'reason', e)}"
                if attempt == self.retries:
                    return {}, VLMResponse("", self.model, time.time() - t0, error=err)
                time.sleep(0.25 * 2 ** attempt)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}, VLMResponse("", self.model, time.time() - t0, error="Jev returned something that is not JSON")
        lat = time.time() - t0
        answers, missing = parse_answers(obj, questions)
        payload = obj if isinstance(obj, dict) else {}
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        def tokens(key):
            value = usage.get(key)
            return value if type(value) is int and value >= 0 else 0
        resp = VLMResponse(json.dumps(payload.get("answers", {})), payload.get("model") or self.model, lat,
                           tokens("input_tokens"), tokens("output_tokens"), "end_turn",
                           error=f"Jev answered without {missing}" if missing else "")
        return answers, resp


def parse_answers(obj, questions):
    """Response JSON -> ({id: Answer}, [missing ids]). Unknown or malformed answers count as missing."""
    out, missing = {}, []
    raw = obj.get("answers") if isinstance(obj, dict) else None
    if not isinstance(raw, dict):
        return {}, list(questions)

    def probability(value):
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError("probabilities must be finite numbers in [0, 1]")
        return float(value)

    for k, q in questions.items():
        a = raw.get(k)
        if not isinstance(a, dict) or a.get("type") != q.type:
            missing.append(k); continue
        try:
            if q.type == "choice":
                choice, probs = a.get("choice"), a.get("probabilities")
                if not isinstance(choice, str) or choice not in q.criteria:
                    raise ValueError("unknown choice")
                if not isinstance(probs, dict) or set(probs) != set(q.criteria):
                    raise ValueError("missing option probabilities")
                probs = {o: probability(p) for o, p in probs.items()}
                # The API rounds each option to hundredths (observed sums include 0.99).
                # Allow half a rounding unit per option, retaining the reported probabilities.
                if not math.isclose(sum(probs.values()), 1.0, abs_tol=0.005 * len(probs) + 1e-9):
                    raise ValueError("probabilities do not sum to one")
                if probs[choice] < max(probs.values()):
                    raise ValueError("choice is not the highest probability option")
                out[k] = Answer("choice", choice=choice, probabilities=probs, confidence=probability(a["confidence"]))
            else:
                out[k] = Answer("noul", noul=probability(a["noul"]))
        except (KeyError, TypeError, ValueError):
            missing.append(k)
    return out, missing
