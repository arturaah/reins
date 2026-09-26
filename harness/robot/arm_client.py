"""Backend that talks to the arm_sdk streamer process over a local socket. No SDK import here.

A heartbeat thread writes to the socket every streamer.heartbeat_s; when this process dies the
socket closes and the streamer ramps the weight down on its own.
"""
import json
import socket
import threading
import time

from ..executor import Backend


class ArmClientBackend(Backend):
    name = "arm_sdk"
    dry_run = False

    def __init__(self, cfg, log=print):
        s = cfg["streamer"]
        self.log = log
        self.sock = socket.create_connection((s["host"], int(s["port"])), timeout=5.0)
        self.file = self.sock.makefile("rb")
        self.lock = threading.Lock()
        self.hb_s = float(s["heartbeat_s"])
        self.alive = True
        threading.Thread(target=self._heartbeat, daemon=True).start()
        hello = self.call({"cmd": "hello"})
        self.fsm, self.fsm_name = hello.get("fsm"), hello.get("fsm_name")
        self.log(f"streamer: FSM {self.fsm} = {self.fsm_name}, weight {hello.get('weight', 0):.2f}")
        self._joints, self._vel = hello["joints"], hello["velocities"]

    def _heartbeat(self):
        while self.alive:
            try:
                with self.lock:
                    self.sock.sendall(b'{"cmd": "heartbeat"}\n')
            except OSError:
                return
            time.sleep(self.hb_s)

    def call(self, req, timeout=None):
        with self.lock:
            self.sock.settimeout(timeout)
            self.sock.sendall((json.dumps(req) + "\n").encode())
            line = self.file.readline()
        if not line:
            raise RuntimeError("streamer closed the connection")
        resp = json.loads(line)
        if "joints" in resp:
            self._joints, self._vel = resp["joints"], resp["velocities"]
        return resp

    def joints(self):
        self.call({"cmd": "state"}); return dict(self._joints)

    def velocities(self):
        self.call({"cmd": "state"}); return dict(self._vel)

    def engage(self):
        r = self.call({"cmd": "engage"}, timeout=15.0)
        if not r.get("ok"):
            raise RuntimeError(f"engage refused: {r.get('error')}")

    def release(self):
        try:
            self.call({"cmd": "release"}, timeout=15.0)
        finally:
            self.alive = False
            self.sock.close()

    def freeze(self):
        self.call({"cmd": "freeze"})

    def stream(self, arm, frames, dt):
        r = self.call({"cmd": "frames", "arm": arm, "frames": [[float(v) for v in f] for f in frames], "dt": dt},
                      timeout=len(frames) * dt + 10.0)
        if not r.get("ok"):
            raise RuntimeError(f"streamer refused the frames: {r.get('error')}")

    def hand(self, arm, closed):
        return "this robot has no hand: nothing to grasp with, the arm paused"
