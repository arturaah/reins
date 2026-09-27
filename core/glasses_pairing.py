"""Revocable Spectacles credentials; secrets never enter repository files or status."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time
from contextlib import contextmanager


class PairingStore:
    def __init__(self, path=None):
        state = Path(os.environ.get("REINS_STATE_DIR", "~/.local/state/reins")).expanduser()
        self.path = Path(path) if path is not None else state / "glasses_pairing.json"
        self.lock = threading.RLock()

    def _read(self):
        try:
            value = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot read glasses pairing store") from exc
        if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("devices"), dict):
            raise ValueError("Invalid glasses pairing store")
        if any(not isinstance(device, dict) or not isinstance(device.get("token_hash"), str)
               for device in value["devices"].values()):
            raise ValueError("Invalid glasses pairing entry")
        return value["devices"]

    @contextmanager
    def _write_access(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            fd = os.open(str(self.path)+".lock", os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield self._read()
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _write(self, devices):
        fd, temporary = tempfile.mkstemp(prefix=".glasses-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump({"version": 1, "devices": devices}, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _public(device_id, device):
        return {"device_id": device_id, **{key: device.get(key) for key in ("label", "created_at", "revoked_at")}}

    def list_devices(self):
        with self.lock:
            return [self._public(key, value) for key, value in self._read().items()]

    def create_device(self, label):
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 80:
            raise ValueError("Give the glasses a name of 1–80 characters")
        token, device_id = secrets.token_urlsafe(32), secrets.token_hex(16)
        device = {"label": " ".join(label.split()), "created_at": time.time(), "revoked_at": None,
                  "token_hash": hashlib.sha256(token.encode()).hexdigest()}
        with self._write_access() as devices:
            devices[device_id] = device
            self._write(devices)
        return {**self._public(device_id, device), "token": token}

    def revoke_device(self, device_id):
        if not isinstance(device_id, str) or not 1 <= len(device_id) <= 80:
            raise ValueError("Invalid paired device ID")
        with self._write_access() as devices:
            if device_id not in devices:
                return False
            devices[device_id]["revoked_at"] = time.time()
            self._write(devices)
        return True

    def authenticate(self, device_id, token):
        if not isinstance(device_id, str) or not isinstance(token, str) or not 1 <= len(token) <= 200:
            return None
        fingerprint = hashlib.sha256(token.encode()).hexdigest()
        return fingerprint if self.active(device_id, fingerprint) else None

    def active(self, device_id, fingerprint):
        with self.lock:
            device = self._read().get(device_id)
            return bool(isinstance(device, dict) and device.get("revoked_at") is None and
                        isinstance(device.get("token_hash"), str) and
                        secrets.compare_digest(device["token_hash"], fingerprint))
