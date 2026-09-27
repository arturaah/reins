"""File mailbox linking a harness proposal to Spectacles' read-only path feed.

The feed only writes a decision for the current proposal. The harness remains
the sole owner of the confirmation gate and robot command path.
"""
import hashlib
import json
import os
import secrets
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(json.dumps(value) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ReviewMailbox:
    def __init__(self, path):
        self.path = Path(path)
        self.answer = self.path.with_name(self.path.stem + "_answer.json")

    def propose(self, plan_path, text, mode):
        plan_path = Path(plan_path)
        proposal = {"id": secrets.token_hex(16), "text": str(text)[:180],
                    "mode": mode, "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "created_at": time.time()}
        self.answer.unlink(missing_ok=True)
        atomic_json(self.path, proposal)
        return proposal

    def pending(self, plan_path):
        try:
            proposal = json.loads(self.path.read_text())
            if time.time() - float(proposal["created_at"]) > 300:
                return None
            if proposal["mode"] not in ("live", "dry-run", "sim"):
                return None
            if hashlib.sha256(Path(plan_path).read_bytes()).hexdigest() != proposal["plan_sha256"]:
                return None
            if self.answer.exists():
                return None
            return {**{k: proposal[k] for k in ("id", "text", "mode")},
                    "digest": proposal["plan_sha256"], "revision": 1,
                    "expires_at": float(proposal["created_at"])+300}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def decide(self, plan_path, proposal_id, decision):
        if decision not in ("approve", "decline"):
            return False
        pending = self.pending(plan_path)
        if not pending or pending["id"] != proposal_id:
            return False
        atomic_json(self.answer, {"id": proposal_id, "decision": decision,
                                  "created_at": time.time()})
        return True

    def take(self, proposal_id, plan_path):
        try:
            proposal = json.loads(self.path.read_text())
            if (time.time() - float(proposal.get("created_at", 0)) > 300 or proposal.get("id") != proposal_id or
                hashlib.sha256(Path(plan_path).read_bytes()).hexdigest() != proposal.get("plan_sha256")):
                return None
            answer = json.loads(self.answer.read_text())
            if answer.get("id") == proposal_id and answer.get("decision") in ("approve", "decline"):
                self.answer.unlink(missing_ok=True)
                return answer["decision"]
        except (OSError, ValueError, TypeError):
            pass
        return None

    def clear(self, proposal_id):
        try:
            if json.loads(self.path.read_text()).get("id") == proposal_id:
                self.path.unlink(missing_ok=True)
                self.answer.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError):
            pass
