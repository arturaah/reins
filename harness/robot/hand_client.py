"""The harness side of the Revo2 hands: talks to `python -m harness.robot.revo2 IFACE serve` over a local socket. No SDK
import here.

GRASP sends hand.revo2.close, RELEASE sends hand.revo2.open, then waits until the fingers stop (at most settle_s) and
reads them back. A close whose fingers (index..pinky) got within empty_reach of the close pose had nothing to stop
them: that is reported as "EMPTY grasp", which the loop already answers by opening and rolling back to the GRASP
stage. A dry run reads the hands but never sends a set; it pretends the command happened so an episode walks through.
"""
import json
import socket
import threading
import time

FINGERS = slice(2, 6)              # index, middle, ring, pinky; the thumb and its rotation close by design less far


class Revo2Client:
    def __init__(self, cfg, log=print, dry_run=False):
        self.h = cfg["hand"]["revo2"]
        self.log, self.dry_run = log, dry_run
        self.lock = threading.Lock()
        try:
            self.sock = socket.create_connection((self.h["host"], int(self.h["port"])), timeout=5.0)
        except OSError as e:
            raise RuntimeError(f"no Revo2 hand server on {self.h['host']}:{self.h['port']} ({e}); start it with "
                               "`python -m harness.robot.revo2 IFACE serve` (IFACE = the link that sees the Jetson)") from e
        self.file = self.sock.makefile("rb")
        self.closed = {}           # arm -> last commanded (or, in a dry run, pretended) state
        hands = self.call({"cmd": "hello"})["hands"]
        self.log("Revo2 hands: " + ", ".join(f"{s} {'ok' if st else 'NO STATE'}" for s, st in hands.items())
                 + ("; dry run, no hand command will be sent" if dry_run else ""))

    def call(self, req):
        with self.lock:
            self.sock.sendall((json.dumps(req) + "\n").encode())
            line = self.file.readline()
        if not line:
            raise RuntimeError("the Revo2 hand server closed the connection")
        return json.loads(line)

    def read(self, arm):
        return self.call({"cmd": "state"})["hands"].get(arm)

    def hand(self, arm, closed):
        word = "close" if closed else "open"
        target = [float(v) for v in self.h[word]]
        if self.dry_run:
            st = self.read(arm)
            self.closed[arm] = closed
            return (f"dry run: would {word} the {arm} hand"
                    + ("" if st else f" (and it would fail: no {arm} hand state)"))
        r = self.call({"cmd": "set", "side": arm, "q": target, "speed": float(self.h["speed"])})
        if not r.get("ok"):
            return f"the {arm} hand did not move: {r.get('error')}"
        self.closed[arm] = closed
        q = self._settle(arm)
        if q is None:
            return f"{word} sent but the {arm} hand state stopped arriving"
        reach = sum(q[FINGERS]) / max(sum(target[FINGERS]), 1e-6)
        if not closed:
            return "hand opened" if max(q[FINGERS]) < 0.2 else f"hand opening, fingers still at {max(q[FINGERS]):.0%} closed"
        if reach >= float(self.h["empty_reach"]):
            return f"EMPTY grasp: the hand closed fully ({reach:.0%} of the close pose), nothing between the fingers"
        return f"hand closed on an object: the fingers stopped at {reach:.0%} of the close pose"

    def _settle(self, arm):
        """Poll until the fingers stop moving or settle_s runs out; -> the last q or None."""
        t0, prev, q = time.time(), None, None
        time.sleep(0.2)
        while time.time() - t0 < float(self.h["settle_s"]):
            st = self.read(arm)
            if st is None:
                return None
            q = st["q"]
            if prev is not None and max(abs(a - b) for a, b in zip(q, prev)) < 0.005:
                break
            prev = q
            time.sleep(0.1)
        return q

    def hand_state(self, arm):
        """The last command once one was sent; before that the measured fingers (closed when index..pinky are over
        half way), so the model is not told "open" about a hand that starts closed."""
        st = self.read(arm)
        if st is None:
            return None
        if arm in self.closed:
            return self.closed[arm]
        return sum(st["q"][FINGERS]) / 4 > 0.5

    def close(self):
        self.sock.close()
