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
  python -m harness.robot.revo2 IFACE open|close --side right     one publish
  python -m harness.robot.revo2 IFACE serve                       the hand server on hand.revo2.host:port
  python -m harness.robot.revo2 lo0 fake --domain 1               fake hands for loopback tests (never on domain 0)

Protocol, one JSON object per line: {"cmd": "hello"} {"cmd": "state"} {"cmd": "set", "side": "right", "q": [6 floats], "speed": 1.0}
Every reply carries "hands": {"left": {"q": [6], "tau": [6], "age": s} or null, "right": ...}.
"""
import argparse
import json
import socket
import threading
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_go_msg_dds__MotorState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

from .lowstate import init_dds

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

    def snapshot(self):
        return {"hands": {s: self.dds.state(s) for s in SIDES}}

    def dispatch(self, req):
        cmd = req.get("cmd")
        if cmd in ("hello", "state"):
            return {"ok": True, **self.snapshot()}
        if cmd == "set":
            side, q = req.get("side"), req.get("q")
            if side not in SIDES or not isinstance(q, list) or len(q) != N:
                return {"ok": False, "error": f"set needs side in {SIDES} and q with {N} values ({', '.join(MOTORS)})", **self.snapshot()}
            st = self.dds.state(side)
            max_age = float(self.cfg["state_max_age_s"])
            if st is None or st["age"] > max_age:
                return {"ok": False, "error": f"no {side} hand state on {self.dds.prefix}/{side}/state in the last {max_age:.1f} s: "
                                              "is brainco_hand_server running on the Jetson and the hand bound?", **self.snapshot()}
            speed = float(req.get("speed", self.cfg["speed"]))
            self.dds.set(side, q, speed)
            self.log(f"{side} hand -> {[round(float(v), 2) for v in q]} speed {speed:.2f}")
            return {"ok": True, **self.snapshot()}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    def serve(self, host, port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port)); srv.listen(4)
        self.log(f"Revo2 hand server listening on {host}:{port}; publishes on {self.dds.prefix}/{{left,right}}/cmd only on a set")
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        with conn, conn.makefile("rb") as f:
            for line in f:
                try:
                    resp = self.dispatch(json.loads(line))
                except Exception as e:
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                conn.sendall((json.dumps(resp) + "\n").encode())


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
    p.add_argument("--block-at", type=float, default=0.5, help="fake: where the fingers stop (1.0 = nothing in the hand)")
    a = p.parse_args()
    cfg = load()
    h = cfg["hand"]["revo2"]
    if a.action == "fake":
        return run_fake(a.iface, a.domain, h["topic_prefix"], a.block_at)
    dds = Revo2Dds(a.iface, a.domain, h["topic_prefix"], publish=a.action != "state")
    if a.action == "serve":
        try:
            return HandServer(dds, cfg).serve(h["host"], int(h["port"]))
        except KeyboardInterrupt:                    # the window's Stop / close: nothing is published on the way out
            print("hand server stopped")
            return
    time.sleep(1.0)
    if a.action in ("open", "close"):
        r = HandServer(dds, cfg).dispatch({"cmd": "set", "side": a.side, "q": list(h[a.action])})
        if not r["ok"]:
            raise SystemExit(r["error"])
        time.sleep(float(h["settle_s"]))
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
