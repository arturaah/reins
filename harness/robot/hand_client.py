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

from .control_auth import read_token

FINGERS = slice(2, 6)              # index, middle, ring, pinky; the thumb and its rotation close by design less far


class Revo2Client:
    def __init__(self, cfg, log=print, dry_run=False):
        self.h = cfg["hand"]["revo2"]
        self.log, self.dry_run = log, dry_run
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.external_cancel = None
        self.alive = True
        self.authenticated = False
        self.hb_s = float(cfg.get("streamer", {}).get("heartbeat_s", .1))
        self.control_token = read_token(self.h.get("control_token_file") or cfg.get("streamer", {}).get("control_token_file"))
        try:
            self.sock = socket.create_connection((self.h["host"], int(self.h["port"])), timeout=5.0)
        except OSError as e:
            raise RuntimeError(f"no Revo2 hand server on {self.h['host']}:{self.h['port']} ({e}); start it with "
                               "`python -m harness.robot.revo2 IFACE serve` (IFACE = the link that sees the Jetson)") from e
        self.sock.settimeout(3.)
        self.file = self.sock.makefile("rb")
        self.closed = {}           # arm -> last commanded (or, in a dry run, pretended) state
        hello = self.call({"cmd": "hello"})
        if hello.get("control_protocol") != 2:
            self.close()
            raise RuntimeError("Outdated hand bridge; launch the private reviewed bridge")
        if self.control_token and not self.dry_run:
            response = self.call({"cmd": "authenticate", "token": self.control_token})
            if not response.get("ok"):
                self.close()
                raise RuntimeError(response.get("error", "Hand authentication refused"))
            self.authenticated = True
            threading.Thread(target=self._heartbeat, daemon=True).start()
        hands = hello["hands"]
        self.log("Revo2 hands: " + ", ".join(f"{s} {'ok' if st else 'NO STATE'}" for s, st in hands.items())
                 + ("; dry run, no hand command will be sent" if dry_run else ""))

    def call(self, req):
        with self.lock:
            if req.get("cmd") == "execute_motion" and (self.cancelled.is_set() or self.external_cancel is not None and self.external_cancel.is_set()):
                raise RuntimeError("Hand motion stopped before submission")
            self.sock.sendall((json.dumps(req, allow_nan=False) + "\n").encode())
            line = self.file.readline()
        if not line:
            raise RuntimeError("the Revo2 hand server closed the connection")
        return json.loads(line)

    def read(self, arm):
        return self.call({"cmd": "state"})["hands"].get(arm)

    def _heartbeat(self):
        while self.alive:
            try:
                with self.lock:
                    self.sock.sendall(b'{"cmd":"heartbeat"}\n')
            except OSError:
                self.alive = False
                return
            time.sleep(self.hb_s)

    def hand(self, arm, closed):
        if not self.dry_run:
            raise RuntimeError("Unreviewed hand commands are retired; propose through RobotPipeline")
        st = self.read(arm)
        self.closed[arm] = closed
        return (f"dry run: would {'close' if closed else 'open'} the {arm} hand"
                + ("" if st else f" (and it would fail: no {arm} hand state)"))

    def execute_motion(self, payload, approval, cancelled=None):
        self.external_cancel = cancelled
        from contract.runtime import digest, validate_approval, validate_motion
        validate_motion(payload)
        validate_approval(approval, digest(payload))
        if payload["kind"] != "hand" or self.dry_run or not self.authenticated:
            raise RuntimeError("Reviewed hand execution needs a private live hand connection")
        self.cancelled.clear()
        arm, closed = payload["arm"], payload["closed"]
        response = self.call({"cmd": "execute_motion", "payload": payload, "approval": approval})
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Hand command refused"))
        self.closed[arm] = closed
        q = self._settle(arm)
        if self.cancelled.is_set() or cancelled is not None and cancelled.is_set():
            raise RuntimeError("Hand motion stopped")
        if q is None:
            raise RuntimeError("Hand state stopped arriving after the command")
        target = [float(v) for v in self.h["close" if closed else "open"]]
        reach = sum(q[FINGERS]) / max(sum(target[FINGERS]), 1e-6)
        if not closed:
            feedback = "hand opened" if max(q[FINGERS]) < .2 else f"hand opening, fingers still at {max(q[FINGERS]):.0%} closed"
        elif reach >= float(self.h["empty_reach"]):
            feedback = f"EMPTY grasp: the hand closed fully ({reach:.0%} of the close pose), nothing between the fingers"
        else:
            feedback = f"hand closed on an object: the fingers stopped at {reach:.0%} of the close pose"
        return {"ok": True, "hand_feedback": feedback, "hands": self.call({"cmd": "state"})["hands"]}

    def freeze(self):
        self.cancelled.set()
        if self.authenticated and self.alive:
            return self.call({"cmd": "freeze"})
        return {"ok": True}

    def _settle(self, arm):
        """Poll until the fingers stop moving or settle_s runs out; -> the last q or None."""
        t0, prev, q = time.time(), None, None
        if self.cancelled.wait(.2):
            return None
        while time.time() - t0 < float(self.h["settle_s"]):
            st = self.read(arm)
            if st is None or st["age"] > float(self.h["state_max_age_s"]):
                return None
            q = st["q"]
            if prev is not None and max(abs(a - b) for a, b in zip(q, prev)) < 0.005:
                break
            prev = q
            if self.cancelled.wait(.1):
                return None
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
        self.cancelled.set()
        self.alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
