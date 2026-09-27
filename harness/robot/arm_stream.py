"""The arm_sdk streamer process: the ONLY publisher on rt/arm_sdk in the harness.

Runs on its own (a Terminal on the Mac, or the Jetson), holds the arms at their commanded targets at
50 Hz with the blend weight at 1, and accepts short JSON-line commands from harness.robot.arm_client
over a local TCP socket. Safety it enforces by itself, whatever the client says:
  - refuses to engage outside FSM 4/811; queries the FSM read-only first
  - weight ramps 0->1 over robot.weight_ramp_s on engage and 1->0 on release, abort, Ctrl-C and loss of client
  - watchdog: no bytes from the client for streamer.watchdog_s while engaged -> ramp down (the loop died); it pauses
    while a command from that client is being served (engage ramp, frames), since the client is then waiting on the
    socket and cannot heartbeat, and a dead client shows up as a closed socket instead
  - tracking error over limits.tracking_abort_rad for 0.3 s, or rt/lowstate stale for 0.5 s -> ramp down
  - per-frame joint speed re-checked against limits.max_joint_vel_rad_s; a faster frame is refused
  - waist yaw and head pitch/yaw are held at their measured values with Unitree's gains (robot.hold_head)
Commands (one JSON object per line):  {"cmd": "hello"} {"cmd": "state"} {"cmd": "engage"}
  {"cmd": "frames", "arm": "right", "frames": [[q1..q5], ...], "dt": 0.02}   (blocks until streamed)
  {"cmd": "freeze"} {"cmd": "release"} {"cmd": "heartbeat"} (no reply)
Every other command gets {"ok": false, "error": ...}.

    .venv/bin/python -m harness.robot.arm_stream en6 [--port 8790]
"""
import argparse
import json
import signal
import socket
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

from ..config import load
from ..kinematics import ARM_JOINTS
from .lowstate import FSM_ARM_OK, JOINT_TO_SLOT, LowStateReader, query_fsm

# gains as in unitree_sdk2/example/r1/high_level/r1_arm_sdk_dds_example.cpp
GAINS = {"shoulder_pitch": (50.0, 2.0), "shoulder_roll": (50.0, 2.0), "shoulder_yaw": (40.0, 2.0),
         "elbow": (40.0, 2.0), "wrist_roll": (30.0, 2.0)}
WAIST_YAW, HEAD = 13, (29, 30)


class Streamer:
    def __init__(self, cfg, iface, log=print):
        self.cfg, self.log = cfg, log
        self.rate = float(cfg["robot"]["command_rate_hz"]); self.dt = 1.0 / self.rate
        self.ramp_s = float(cfg["robot"]["weight_ramp_s"])
        self.vmax = float(cfg["limits"]["max_joint_vel_rad_s"])
        self.err_abort = float(cfg["limits"]["tracking_abort_rad"])
        self.watchdog_s = float(cfg["streamer"]["watchdog_s"])
        self.reader = LowStateReader(iface)
        if not self.reader.wait():
            sys.exit(f"no rt/lowstate on {iface}")
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.pub.Init()
        self.crc = CRC(); self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.slots = {}                                     # slot -> (kp, kd)
        for side in ("left", "right"):
            for n in ARM_JOINTS[side]:
                kp, kd = GAINS[n[len(side) + 1:-len("_joint")]]
                self.slots[JOINT_TO_SLOT[n]] = (kp, kd)
        self.slots[WAIST_YAW] = (50.0, 3.0)
        if cfg["robot"]["hold_head"]:
            for s in HEAD: self.slots[s] = (15.0, 1.0)
        self.lock = threading.Lock()
        self.targets = {}                                   # slot -> q
        self.weight = 0.0
        self.engaged = False
        self.last_client = time.time()
        self.serving = False                                # a client command is in dispatch: its silence is expected
        self.err_since = None
        self.stop = threading.Event()
        self.reason = ""
        self.frames_sent = 0

    # -- DDS side ---------------------------------------------------------------------------------
    def measured(self):
        m = self.reader.msg
        return {s: float(m.motor_state[s].q) for s in self.slots}

    def send(self):
        with self.lock:
            w, targets = self.weight, dict(self.targets)
        self.cmd.mode_pr = int(round(np.clip(w, 0.0, 1.0) * 100))
        for s, (kp, kd) in self.slots.items():
            mc = self.cmd.motor_cmd[s]
            mc.q, mc.dq, mc.tau, mc.kp, mc.kd = float(targets.get(s, 0.0)), 0.0, 0.0, kp, kd
        self.cmd.crc = self.crc.Crc(self.cmd); self.pub.Write(self.cmd)
        self.frames_sent += 1

    def ramp(self, to, seconds):
        t0 = time.time()
        with self.lock:
            w0 = self.weight
        while (el := time.time() - t0) < seconds:
            with self.lock:
                self.weight = w0 + (to - w0) * el / seconds
            self.send(); time.sleep(self.dt)
        with self.lock:
            self.weight = to
        self.send()

    def engage(self):
        fsm, name = query_fsm()
        if fsm not in FSM_ARM_OK:
            return f"refused: FSM {fsm} = {name}; the arm topic only takes effect in {sorted(FSM_ARM_OK)}"
        with self.lock:
            self.targets = self.measured()                  # hold everything where it is
            self.engaged = True
        self.err_since = None
        self.log(f"engage: FSM {fsm} = {name}; ramping weight up over {self.ramp_s} s")
        self.ramp(1.0, self.ramp_s)
        return ""

    def release(self, reason="release", seconds=None):
        if not self.engaged and self.weight == 0.0:
            return
        self.log(f"{reason}: ramping weight down")
        self.ramp(0.0, seconds or self.ramp_s)
        with self.lock:
            self.engaged = False
        self.reason = reason

    def hold_loop(self):
        """Background: keep publishing the current targets while engaged, run the watchdogs."""
        while not self.stop.is_set():
            t = time.time()
            if self.engaged:
                if self.reader.age() > 0.5:
                    self.release("ABORT: rt/lowstate stale", 0.5)
                elif not self.serving and time.time() - self.last_client > self.watchdog_s:
                    self.release("ABORT: client heartbeat lost", 1.0)
                else:
                    m = self.measured()
                    with self.lock:
                        errs = {s: abs(m[s] - self.targets[s]) for s in self.targets if s in m}
                    worst = max(errs.values()) if errs else 0.0
                    if worst > self.err_abort:
                        self.err_since = self.err_since or t
                        if t - self.err_since > 0.3:
                            self.release(f"ABORT: tracking error {worst:.2f} rad", 0.5)
                    else:
                        self.err_since = None
                    if not self.streaming:
                        self.send()
            time.sleep(max(0.0, self.dt - (time.time() - t)))

    streaming = False

    def stream_frames(self, arm, frames, dt):
        if not self.engaged:
            return "not engaged"
        names = ARM_JOINTS[arm]; slots = [JOINT_TO_SLOT[n] for n in names]
        frames = [np.asarray(f, float) for f in frames]
        for i, f in enumerate(frames):
            if f.shape != (5,) or not np.all(np.isfinite(f)):
                return f"frame {i}: expected 5 finite joint values"
        with self.lock:
            prev = np.array([self.targets[s] for s in slots])
        # The client plans from the measured joints; the hold target differs from them by the gravity droop (about
        # 0.01 rad), which at 50 Hz would be a 0.5 rad/s jump on the first frame. Approach the first frame from the
        # current target at the cap first, then play the frames.
        step = self.vmax * 0.8 * dt
        jump = float(np.abs(frames[0] - prev).max())
        k = int(np.ceil(jump / step)) - 1 if jump > step else 0
        frames = [prev + (frames[0] - prev) * (i + 1) / (k + 1) for i in range(k)] + frames
        for i, f in enumerate(frames):
            v = float(np.abs(f - prev).max() / dt)
            if v > self.vmax * 1.05:
                return f"frame {i}: {v:.2f} rad/s over the {self.vmax} rad/s cap; refused before sending"
            prev = f
        self.streaming = True
        try:
            t0 = time.time()
            for i, f in enumerate(frames):
                if not self.engaged:
                    return f"released during streaming ({self.reason})"
                with self.lock:
                    for s, q in zip(slots, f):
                        self.targets[s] = float(q)
                self.send()
                time.sleep(max(0.0, t0 + (i + 1) * dt - time.time()))
        finally:
            self.streaming = False
        return ""

    def state(self):
        m = self.reader.joints(); v = self.reader.velocities()
        with self.lock:
            return {"ok": True, "joints": m, "velocities": v, "weight": self.weight, "engaged": self.engaged,
                    "lowstate_age_s": self.reader.age(), "frames_sent": self.frames_sent, "reason": self.reason}

    # -- socket side ------------------------------------------------------------------------------------
    def serve(self, host, port):
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port)); srv.listen(1)
        self.log(f"arm streamer listening on {host}:{port}; publishes on rt/arm_sdk only while engaged (Ctrl-C releases)")
        while not self.stop.is_set():
            srv.settimeout(0.5)
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            self.log("client connected"); self.last_client = time.time()
            try:
                self.handle(conn)
            finally:
                conn.close()
                self.log("client gone")
                if self.engaged:
                    self.release("client disconnected", 1.0)

    def handle(self, conn):
        conn.settimeout(0.2)
        buf = b""
        while not self.stop.is_set():
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return
            self.last_client = time.time()
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    conn.sendall(b'{"ok": false, "error": "bad json"}\n'); continue
                cmd = req.get("cmd")
                if cmd == "heartbeat":
                    continue
                self.serving = True
                try:
                    resp = self.dispatch(cmd, req)
                except Exception as e:
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                finally:
                    self.last_client = time.time(); self.serving = False
                conn.sendall((json.dumps(resp) + "\n").encode())

    def dispatch(self, cmd, req):
        if cmd == "hello":
            fsm, name = query_fsm()
            return {"ok": True, "fsm": fsm, "fsm_name": name, **self.state()}
        if cmd == "state":
            return self.state()
        if cmd == "engage":
            err = self.engage()
            return {**self.state(), "ok": not err, "error": err}       # the flag last: state() carries its own ok
        if cmd == "frames":
            n = len(req.get("frames") or []); t0 = time.time()
            err = self.stream_frames(req["arm"], req["frames"], float(req["dt"]))
            self.log(f"frames: {req.get('arm')} arm, {n} frames over {n * float(req['dt']):.2f} s -> " + (f"REFUSED: {err}" if err else f"streamed in {time.time() - t0:.2f} s"))
            return {**self.state(), "ok": not err, "error": err}
        if cmd == "freeze":
            with self.lock:
                self.targets = self.measured() if not self.engaged else dict(self.targets)
            return {"ok": True, **self.state()}
        if cmd == "release":
            self.release(); return {"ok": True, **self.state()}
        return {"ok": False, "error": f"unknown command {cmd!r}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iface"); ap.add_argument("--port", type=int); ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load(a.config)
    st = Streamer(cfg, a.iface)
    threading.Thread(target=st.hold_loop, daemon=True).start()

    def on_sigint(*_):
        if st.engaged:
            signal.signal(signal.SIGINT, lambda *_: print("(already releasing: the weight ramps down first)"))
            st.release("Ctrl-C")
        st.stop.set()
    signal.signal(signal.SIGINT, on_sigint)
    try:
        st.serve(cfg["streamer"]["host"], a.port or int(cfg["streamer"]["port"]))
    finally:
        if st.engaged:
            st.release("exit")
        st.reader.close()


if __name__ == "__main__":
    main()
