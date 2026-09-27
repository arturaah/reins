"""BrainCo Revo2 hands over DDS, and the hand server the harness talks to. Imports the SDK.

On the robot, brainco_hand_server (unitreerobotics/brainco_hand_service, runs on the Jetson next to the hands' USB
serial ports) bridges each hand to DDS: it subscribes MotorCmds_ on rt/brainco/{left,right}/cmd and publishes
MotorStates_ on rt/brainco/{left,right}/state. 6 motors per hand in the order thumb, thumb_aux, index, middle, ring,
pinky; q is normalized, 0 = open .. 1 = closed; dq is the speed 0..1; tau_est is the normalized motor current -1..1.
This is the same contract xr_teleoperate's --ee brainco uses.

This process is the only publisher on the hand topics, as arm_stream is on rt/arm_sdk. It publishes once per set
command (the hand firmware holds the last target, nothing streams) and refuses a set while that hand's state is not
arriving. The harness talks to it over a local socket (hand_client.py), so the loop needs no DDS participant of its
own and the hands can be reached through a different interface than the body cable (the Jetson link).

  python -m harness.robot.revo2 IFACE state [--watch]            subscribe-only
  python -m harness.robot.revo2 IFACE serve                       the hand server on hand.revo2.host:port
  python -m harness.robot.revo2 lo0 fake --domain 1               fake hands for loopback tests (never on domain 0)

Legacy raw set commands are rejected. The controller authenticates using its private token and submits
execute_motion with a canonical hand payload and a one-use review receipt. Telemetry stays public.
Legacy protocol reference, one JSON object per line: {"cmd": "hello"} {"cmd": "state"} {"cmd": "set", "side": "right", "q": [6 floats], "speed": 1.0}
Every reply carries "hands": {"left": {"q": [6], "tau": [6], "age": s} or null, "right": ...}.
"""
import argparse
import json
import math
import socket
import signal
import threading
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_go_msg_dds__MotorState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

from .lowstate import init_dds
from .control_auth import ReviewLedger, matches, read_token

SIDES = ("left", "right")
MOTORS = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
N = len(MOTORS)


class Revo2Dds:
    """Subscribes both hands' state; with publish=True also owns the command publishers."""

    def __init__(self, iface, domain=0, prefix="rt/brainco", publish=True):
        init_dds(iface, domain)
        self.prefix = prefix
        self.lock = threading.Lock()
        self.msgs = {s: None for s in SIDES}
        self.t_last = {s: 0.0 for s in SIDES}
        self.subs = {}
        for s in SIDES:
            self.subs[s] = ChannelSubscriber(f"{prefix}/{s}/state", MotorStates_)
            self.subs[s].Init(lambda m, s=s: self._on_state(s, m), 10)
        self.pubs = {}
        if publish:
            for s in SIDES:
                self.pubs[s] = ChannelPublisher(f"{prefix}/{s}/cmd", MotorCmds_)
                self.pubs[s].Init()

    def _on_state(self, side, m):
        with self.lock:
            self.msgs[side], self.t_last[side] = m, time.time()

    def state(self, side):
        """{"q": [6], "tau": [6], "age": s} or None before the first message."""
        with self.lock:
            m, t = self.msgs[side], self.t_last[side]
        if m is None or len(m.states) < N:
            return None
        return {"q": [round(float(m.states[i].q), 4) for i in range(N)],
                "tau": [round(float(m.states[i].tau_est), 4) for i in range(N)], "age": round(time.time() - t, 3)}

    def set(self, side, q, speed):
        if side not in SIDES or len(q) != N or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in q):
            raise ValueError("Hand target must contain six finite normalized joint values")
        if not math.isfinite(float(speed)) or not 0 <= float(speed) <= 1:
            raise ValueError("Hand speed must be finite and normalized")
        msg = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(N)])
        for i, v in enumerate(q):
            msg.cmds[i].q = min(max(float(v), 0.0), 1.0)
            msg.cmds[i].dq = min(max(float(speed), 0.0), 1.0)
        self.pubs[side].Write(msg)

    def close(self):
        for sub in self.subs.values():
            sub.Close()


class HandServer:
    def __init__(self, dds, cfg, log=print):
        self.dds, self.cfg, self.log = dds, cfg["hand"]["revo2"], log
        self.control_token = read_token(self.cfg.get("control_token_file") or cfg.get("streamer", {}).get("control_token_file"))
        self.reviews = ReviewLedger()
        self.owner_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.owner = None
        self.active = set()
        self.cancelled = threading.Event()
        self.stop = threading.Event()
        self.last_client = 0.
        self.watchdog_s = float(cfg.get("streamer", {}).get("watchdog_s", .5))

    def snapshot(self):
        return {"hands": {s: self.dds.state(s) for s in SIDES}, "control_protocol": 2,
                "control_available": self.control_token is not None}

    def freeze(self):
        """Hold measured finger positions, where telemetry is fresh enough to do so."""
        self.cancelled.set()
        errors = []
        with self.command_lock:
            for side in list(self.active):
                st = self.dds.state(side)
                if st is None or st["age"] > float(self.cfg["state_max_age_s"]):
                    errors.append(f"Cannot hold {side} fingers: hand telemetry is stale")
                else:
                    q = st["q"]
                    if len(q) != N or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in q):
                        errors.append(f"Cannot hold {side} fingers: invalid measured state")
                    else:
                        try:
                            self.dds.set(side, q, 0.)
                        except Exception as exc:
                            errors.append(f"Cannot hold {side} fingers: {exc}")
                self.active.discard(side)
        return {"ok": not errors, "error": "; ".join(errors), **self.snapshot()}

    def dispatch(self, req, *, authorized=False):
        cmd = req.get("cmd")
        if cmd in ("hello", "state"):
            return {"ok": True, **self.snapshot()}
        if not authorized:
            return {"ok": False, "error": "Read-only hand connection: private controller capability required"}
        if cmd == "freeze":
            return self.freeze()
        if cmd != "execute_motion":
            return {"ok": False, "error": "Raw hand set commands are retired; submit one reviewed motion through RobotPipeline"}
        payload, approval = req["payload"], req["approval"]
        if payload.get("kind") != "hand":
            raise ValueError("The hand bridge accepts only hand motions")
        self.reviews.consume(payload, approval)
        side = payload["arm"]
        q = list(self.cfg["close" if payload["closed"] else "open"])
        speed = float(self.cfg["speed"])
        if len(q) != N or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in q):
            raise ValueError("Configured hand target must have six finite normalized values")
        if not math.isfinite(speed) or not 0 < speed <= 1:
            raise ValueError("Invalid configured hand speed")
        with self.command_lock:
            if self.cancelled.is_set() or self.stop.is_set():
                return {"ok": False, "error": "Hand motion stopped; reconnect before submitting another motion"}
            st = self.dds.state(side)
            max_age = float(self.cfg["state_max_age_s"])
            if st is None or st["age"] > max_age:
                return {"ok": False, "error": f"no {side} hand state on {self.dds.prefix}/{side}/state in the last {max_age:.1f} s: "
                                              "is brainco_hand_server running on the Jetson and the hand bound?", **self.snapshot()}
            self.active.add(side)
            self.dds.set(side, q, speed)
        self.log(f"reviewed {side} hand {'close' if payload['closed'] else 'open'}")
        return {"ok": True, **self.snapshot()}

    def close(self):
        """Latch shutdown before holding fingers; no connection may rearm the server."""
        self.stop.set()
        self.freeze()

    def serve(self, host, port):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Hand control must bind to loopback")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port)); srv.listen(4); srv.settimeout(.2)
            self.log(f"Revo2 hand bridge on {host}:{port}; commands require the private controller and human review")
            while not self.stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        conn.settimeout(.1)
        buf, authenticated = b"", False
        try:
            while not self.stop.is_set():
                try:
                    chunk = conn.recv(16384)
                except socket.timeout:
                    if authenticated and self.active and time.time()-self.last_client > self.watchdog_s:
                        self.freeze()
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 65536:
                    break
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        req = json.loads(line)
                        if not isinstance(req, dict):
                            raise ValueError("Invalid hand command")
                        if req.get("cmd") == "authenticate":
                            with self.owner_lock:
                                if (matches(self.control_token, req.get("token")) and self.owner in (None, conn)
                                        and not self.stop.is_set()):
                                    if self.owner is None:
                                        self.cancelled.clear()
                                    self.owner, authenticated = conn, True
                                    self.last_client = time.time()
                                    resp = {"ok": True, "control_protocol": 2}
                                else:
                                    resp = {"ok": False, "error": "Hand controller authentication refused or already owned"}
                        else:
                            if authenticated:
                                self.last_client = time.time()
                            if req.get("cmd") == "heartbeat":
                                continue
                            resp = self.dispatch(req, authorized=authenticated)
                    except Exception as exc:
                        resp = {"ok": False, "error": str(exc)}
                    conn.sendall((json.dumps(resp, allow_nan=False)+"\n").encode())
        except OSError:
            pass
        finally:
            if authenticated:
                self.freeze()
                with self.owner_lock:
                    if self.owner is conn:
                        self.owner = None
            conn.close()


def run_fake(iface, domain, prefix, block_at):
    """Fake hands: follow each command at 1.0/s, fingers (index..pinky) stop at block_at as if closing on an object
    (block_at >= 1 means an empty hand). Publishes state at 100 Hz."""
    if domain == 0:
        raise SystemExit("refusing to run fake hands on domain 0 (the robot's)")
    init_dds(iface, domain)
    q = {s: [0.0] * N for s in SIDES}
    target = {s: [0.0] * N for s in SIDES}
    lock = threading.Lock()

    def on_cmd(side, m):
        with lock:
            target[side] = [float(c.q) for c in m.cmds[:N]]
    pubs, subs = {}, {}
    for s in SIDES:
        pubs[s] = ChannelPublisher(f"{prefix}/{s}/state", MotorStates_); pubs[s].Init()
        subs[s] = ChannelSubscriber(f"{prefix}/{s}/cmd", MotorCmds_); subs[s].Init(lambda m, s=s: on_cmd(s, m), 10)
    print(f"fake Revo2 hands on {iface} domain {domain}, fingers block at {block_at}")
    dt = 0.01
    while True:
        with lock:
            for s in SIDES:
                for i in range(N):
                    goal = min(target[s][i], block_at) if i >= 2 else target[s][i]
                    q[s][i] += max(-dt, min(dt, goal - q[s][i]))
                msg = MotorStates_([unitree_go_msg_dds__MotorState_() for _ in range(N)])
                for i in range(N):
                    msg.states[i].q = q[s][i]
                pubs[s].Write(msg)
        time.sleep(dt)


def main():
    from ..config import load
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iface")
    p.add_argument("action", choices=["state", "open", "close", "serve", "fake"])
    p.add_argument("--side", choices=SIDES, default="right")
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--config")
    p.add_argument("--control-token-file", help="private coordinator capability (0600); omitted means read-only")
    p.add_argument("--block-at", type=float, default=0.5, help="fake: where the fingers stop (1.0 = nothing in the hand)")
    a = p.parse_args()
    cfg = load(a.config)
    if a.control_token_file:
        cfg["hand"]["revo2"]["control_token_file"] = a.control_token_file
    if a.action in ("open", "close"):
        raise SystemExit("Direct hand commands are retired. Review a hand motion in the dashboard.")
    h = cfg["hand"]["revo2"]
    if a.action == "fake":
        return run_fake(a.iface, a.domain, h["topic_prefix"], a.block_at)
    dds = Revo2Dds(a.iface, a.domain, h["topic_prefix"], publish=a.action != "state")
    if a.action == "serve":
        server = HandServer(dds, cfg)
        def interrupt(*_):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupt)
        try:
            return server.serve(h["host"], int(h["port"]))
        except KeyboardInterrupt:
            print("hand server stopped")
            return
        finally:
            server.close()
            dds.close()
    time.sleep(1.0)
    while True:
        for s in SIDES:
            st = dds.state(s)
            print(f"{s:5s} " + ("no state" if st is None else
                  " ".join(f"{m}={v:.2f}" for m, v in zip(MOTORS, st["q"])) + f"  current {max(abs(t) for t in st['tau']):.2f}  age {st['age']:.2f} s"))
        if not a.watch:
            break
        time.sleep(0.5)


if __name__ == "__main__":
    main()
