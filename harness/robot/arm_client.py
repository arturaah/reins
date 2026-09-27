"""Multiplexed loopback client; stop and heartbeat never wait behind a motion."""
import json
import socket
import threading
import time
import uuid

from ..executor import Backend


class ArmClientBackend(Backend):
    name = "arm_sdk"
    dry_run = False
    hands = None                   # hand_client.Revo2Client when hand.type is revo2 (set by harness.__main__.build)

    def __init__(self, cfg, log=print):
        s = cfg["streamer"]
        self.log = log
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
        except Exception:
            self.close()
            raise

    def _send(self, req):
        with self.lock:
            if not self.alive:
                raise RuntimeError("Robot connection closed")
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
        state = self.call({"cmd": "engage"}, timeout=15)
        if not state.get("engaged"):
            raise RuntimeError("Robot released while engaging")

    def release(self):
        return self.call({"cmd": "release"}, timeout=15)

    def freeze(self):
        return self.call({"cmd": "freeze"}, timeout=3)

    def close(self):
        self.alive = False
        try: self.sock.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        self.sock.close()

    def stream(self, arm, frames, dt):
        return self.call({"cmd": "frames", "arm": arm, "frames": [[float(v) for v in f] for f in frames], "dt": dt},
                         timeout=len(frames)*dt+15)

    def stream_plan(self, plan, arm):
        from core.trajectory import digest
        return self.call({"cmd": "plan", "arm": arm, "plan": plan, "digest": digest(plan)},
                         timeout=plan["duration_s"]+15)

    def walk(self, vx, vy, vyaw, duration):
        r = self.call({"cmd": "walk", "vx": float(vx), "vy": float(vy), "vyaw": float(vyaw), "duration": float(duration)},
                      timeout=float(duration) + 15.0)
        if not r.get("ok"):
            raise RuntimeError(f"streamer refused the walk: {r.get('error')}")
        return r.get("odom")

    def hand(self, arm, closed):
        if self.hands:
            return self.hands.hand(arm, closed)
        return "this robot has no hand: nothing to grasp with, the arm paused"

    def hand_state(self, arm):
        return self.hands.hand_state(arm) if self.hands else None
