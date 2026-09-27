"""Multiplexed loopback client; stop and heartbeat never wait behind a motion."""
import json
import socket
import threading
import time
import uuid

from .control_auth import read_token


class ArmClientBackend:
    name = "arm_sdk"
    dry_run = False
    hands = None                   # private Revo2Client when configured by RobotPipeline

    def __init__(self, cfg, log=print):
        s = cfg["streamer"]
        self.log = log
        self.cfg = cfg
        self.hands = None
        self.control_token = read_token(s.get("control_token_file"))
        self.authenticated = False
        self.motion_cancel = threading.Event()
        self.sock = socket.create_connection((s["host"], int(s["port"])), timeout=3)
        self.sock.settimeout(None)
        self.lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.alive = True
        self.hb_s = float(s["heartbeat_s"])
        self.latest = {}
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        try:
            hello = self.call({"cmd": "hello"}, timeout=5)
            self.fsm, self.fsm_name = hello.get("fsm"), hello.get("fsm_name")
            if hello.get("control_protocol") != 2:
                raise RuntimeError("Outdated robot bridge; stop it and launch the reviewed dashboard bridge")
            if self.control_token:
                self.call({"cmd": "authenticate", "token": self.control_token})
                self.authenticated = True
        except Exception:
            self.close()
            raise

    def _send(self, req):
        with self.lock:
            if not self.alive:
                raise RuntimeError("Robot connection closed")
            if req.get("cmd") == "execute_motion" and self.motion_cancel.is_set():
                raise RuntimeError("Motion stopped before submission")
            self.sock.sendall((json.dumps(req, allow_nan=False)+"\n").encode())

    def _read(self):
        try:
            with self.sock.makefile("rb") as file:
                while self.alive:
                    line = file.readline(8*1024*1024)
                    if not line:
                        break
                    answer = json.loads(line)
                    with self.pending_lock:
                        item = self.pending.get(answer.get("request_id"))
                        if item:
                            item[1].update(answer); item[0].set()
        except (OSError, ValueError):
            pass
        finally:
            self.alive = False
            with self.pending_lock:
                for event, response in self.pending.values():
                    response.update(ok=False, error="Robot connection lost"); event.set()

    def _heartbeat(self):
        while self.alive:
            try:
                self._send({"cmd": "heartbeat"})
            except (OSError, RuntimeError):
                break
            time.sleep(self.hb_s)

    def call(self, req, timeout=5):
        key = uuid.uuid4().hex
        event, response = threading.Event(), {}
        with self.pending_lock:
            self.pending[key] = (event, response)
        try:
            self._send({**req, "request_id": key})
            if not event.wait(timeout):
                raise RuntimeError("Robot command timed out")
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "Robot refused the command")
            if "joints" in response:
                self.latest = dict(response)
            return response
        finally:
            with self.pending_lock:
                self.pending.pop(key, None)

    def snapshot(self):
        state = self.call({"cmd": "state"})
        if state["lowstate_age_s"] > .5:
            raise RuntimeError("Robot telemetry is stale")
        return state

    def joints(self):
        return dict(self.snapshot()["joints"])

    def velocities(self):
        return dict(self.snapshot()["velocities"])

    def engage(self):
        raise RuntimeError("Engage is owned by RobotPipeline after a complete motion is approved")

    def release(self):
        self.motion_cancel.set()
        try:
            return self.call({"cmd": "release"}, timeout=15) if self.authenticated else self.snapshot()
        finally:
            if self.hands:
                self.hands.freeze()

    def freeze(self):
        self.motion_cancel.set()
        try:
            return self.call({"cmd": "freeze"}, timeout=3) if self.authenticated else self.snapshot()
        finally:
            if self.hands:
                self.hands.freeze()

    def close(self):
        self.motion_cancel.set()
        self.alive = False
        try:
            if self.hands:
                self.hands.close()
        finally:
            try: self.sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            self.sock.close()

    def stream(self, arm, frames, dt):
        raise RuntimeError("Raw frames are retired; propose a complete motion through RobotPipeline")

    def execute_motion(self, payload, approval):
        # Stop stays latched for this connection. Clearing it here could erase a
        # stop that arrived after the coordinator's final check, before dispatch.
        if self.motion_cancel.is_set():
            raise RuntimeError("Motion stopped; reconnect before submitting another motion")
        from contract.runtime import digest, validate_approval, validate_motion
        validate_motion(payload)
        validate_approval(approval, digest(payload))
        if not self.authenticated:
            raise RuntimeError("Read-only robot connection; private controller capability required")
        if payload["kind"] == "hand":
            if self.cfg.get("hand", {}).get("type") != "revo2":
                raise RuntimeError("Revo2 hands are not configured")
            if self.hands is None:
                from .hand_client import Revo2Client
                self.hands = Revo2Client(self.cfg, log=self.log)
            if not self.alive or self.motion_cancel.is_set():
                self.hands.close()
                raise RuntimeError("Hand motion stopped before submission")
            result = self.hands.execute_motion(payload, approval, cancelled=self.motion_cancel)
            return {**self.snapshot(), **result}
        duration = payload["plan"]["duration_s"] if payload["kind"] == "arm" else payload["duration_s"]
        return self.call({"cmd": "execute_motion", "payload": payload, "approval": approval}, timeout=duration+20)

    def stream_plan(self, plan, arm, approval=None):
        return self.execute_motion({"kind": "arm", "arm": arm, "plan": plan}, approval)

    def walk(self, vx, vy, vyaw, duration, approval=None):
        result = self.execute_motion({"kind": "walk", "vx": float(vx), "vy": float(vy),
                                      "vyaw": float(vyaw), "duration_s": float(duration)}, approval)
        return result.get("odom")

    def hand(self, arm, closed):
        raise RuntimeError("Unreviewed hand commands are retired; propose through RobotPipeline")

    def hand_state(self, arm):
        return self.hands.hand_state(arm) if self.hands else None
