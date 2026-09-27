"""Cross-process ownership for the local R1 command paths (no SDK imports)."""
import fcntl
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RobotLease:
    def __init__(self, owner, path=None):
        self.owner = owner
        state = Path(os.environ.get("REINS_STATE_DIR", "~/.local/state/reins")).expanduser()
        self.path = Path(path or os.environ.get("REINS_ROBOT_LOCK", str(state / "robot-control.lock")))
        self.file = None

    def acquire(self):
        if self.file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise ValueError("Robot control is owned by another Reins process. Release it first.") from None
        stream.seek(0); stream.truncate(); stream.write(self.owner); stream.flush()
        self.file = stream

    def release(self):
        if self.file is not None:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
