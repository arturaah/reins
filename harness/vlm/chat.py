"""Chat provider: the model is the Claude session driving this repo, not an API call.

Each planner or controller call is written to an inbox folder (prompt.txt, the JPEG images,
request.json) and the loop blocks until answer.json appears there, written by whoever is acting
as the model (a Claude Code session reading the images, or a person). Nothing else changes: the
same prompts, parser, gate and recorder run. Timeout: vlm.chat_timeout_s (default 15 min); on
the real robot the streamer keeps holding the arms while we wait, heartbeats included.

Protocol, per request folder runs/chat_inbox/<NNN>_<plan|act>/:
  request.json  {"kind": "plan"|"act", "prompt": "prompt.txt", "images": ["context.jpg", ...], "retry_note": ...}
  answer.json   the model's JSON answer, exactly as it would come back from the API (a JSON object)
"""
import json
import time
from pathlib import Path

from ..kinematics import ROOT
from .base import VLM, VLMResponse


class ChatVLM(VLM):
    name = "chat"

    def __init__(self, cfg, log=print):
        v = cfg["vlm"]
        inbox = Path(v.get("chat_inbox", "runs/chat_inbox"))
        self.inbox = inbox if inbox.is_absolute() else ROOT / inbox
        self.timeout = float(v.get("chat_timeout_s", 900))
        self.log = log
        self.n = 0
        self.inbox.mkdir(parents=True, exist_ok=True)
        for old in self.inbox.iterdir():                      # a fresh episode starts with an empty inbox
            if old.is_dir():
                for f in old.iterdir():
                    f.unlink()
                old.rmdir()

    def _request(self, kind, prompt, images, retry_note=None):
        self.n += 1
        d = self.inbox / f"{self.n:03d}_{kind}"
        d.mkdir()
        (d / "prompt.txt").write_text(prompt)
        names = []
        for label, jpg in images:
            name = label.split()[0].lower() + ".jpg"
            (d / name).write_bytes(jpg); names.append(name)
        (d / "request.json").write_text(json.dumps({"kind": kind, "prompt": "prompt.txt", "images": names,
                                                     "retry_note": retry_note, "answer": "answer.json"}, indent=1) + "\n")
        answer = d / "answer.json"
        self.log(f"CHAT: {kind} request {self.n} waiting for {answer}")
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            if answer.exists():
                time.sleep(0.2)                                # let the writer finish
                text = answer.read_text().strip()
                if text:
                    return VLMResponse(text, "chat", time.time() - t0)
            time.sleep(0.5)
        return VLMResponse("", "chat", time.time() - t0, error=f"no answer within {self.timeout:.0f} s")

    def plan(self, prompt, images, schema=None):
        return self._request("plan", prompt, images)

    def act(self, prompt, images, schema=None, retry_note=None):
        if retry_note:
            prompt = prompt + f"\n\nYour previous answer was rejected: {retry_note}. Answer again following the contract exactly."
        return self._request("act", prompt, images, retry_note)
