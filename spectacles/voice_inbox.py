"""Single pending Spectacles speech command shared by plan_feed and the desktop UI."""
import json
import os
from pathlib import Path
from uuid import uuid4


class VoiceInbox:
    def __init__(self, path):
        self.path = Path(path)

    def enqueue(self, command_id, text):
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 80:
            return False
        if not isinstance(text, str):
            return False
        text = " ".join(text.split())
        if not 1 <= len(text) <= 500 or self.path.exists():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + "." + uuid4().hex + ".tmp")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as out:
                json.dump({"id": command_id, "text": text}, out)
                out.flush()
                os.fsync(out.fileno())
            # Linking exposes a complete file and fails if a command is already queued.
            os.link(temporary, self.path)
            return True
        except FileExistsError:
            return False
        finally:
            temporary.unlink(missing_ok=True)

    def take(self):
        """Claim a queued command once. Leave it queued while the UI is busy."""
        claimed = self.path.with_name(self.path.name + "." + uuid4().hex + ".claimed")
        try:
            self.path.rename(claimed)
        except FileNotFoundError:
            return None
        try:
            return json.loads(claimed.read_text())
        finally:
            claimed.unlink(missing_ok=True)
