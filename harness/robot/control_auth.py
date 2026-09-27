"""Local controller capability and one-use review receipts.

The file is deliberately separate from logs/model context. This protects the supported
socket surface, not arbitrary code running as the same OS user or vendor DDS clients.
"""
import hmac
import os
from pathlib import Path
import stat
import threading
import time


def read_token(path):
    if not path:
        return None
    p = Path(path)
    info = p.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("Control token must be a private file (mode 0600)")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError("Control token must belong to the current user")
    token = p.read_text().strip()
    if len(token) < 32 or len(token) > 512:
        raise ValueError("Invalid control capability")
    return token


def matches(token, candidate):
    return bool(token and isinstance(candidate, str) and hmac.compare_digest(token, candidate))


class ReviewLedger:
    """An accepted receipt is consumed before any actuator call, including failed ones."""
    def __init__(self):
        self.lock = threading.Lock()
        self.used = {}

    def consume(self, payload, approval):
        from contract.runtime import digest, validate_approval, validate_motion
        validate_motion(payload)
        receipt = validate_approval(approval, digest(payload))
        key = receipt["proposal_id"]
        with self.lock:
            if key in self.used:
                raise ValueError("Proposal approval has already been consumed")
            # Expired receipts cannot pass validation, so bounded retention is safe.
            now = time.time()
            self.used = {k: expiry for k, expiry in self.used.items() if expiry > now}
            self.used[key] = float(receipt["expires_at"])
        return receipt
