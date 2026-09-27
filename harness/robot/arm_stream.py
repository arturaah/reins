"""The arm_sdk streamer process: the ONLY publisher on rt/arm_sdk in the harness.

Runs on its own (a Terminal on the Mac, or the Jetson), holds the arms at their commanded targets at
50 Hz with the blend weight at 1, and accepts short JSON-line commands from harness.robot.arm_client
over a local TCP socket. Safety it enforces by itself, whatever the client says:
  - refuses to engage outside FSM 4/811; queries the FSM read-only first
  - weight ramps 0->1 over robot.weight_ramp_s on engage and 1->0 on release, abort, Ctrl-C and loss of client
  - watchdog remains active during commands; receive and heartbeat processing never wait behind a stream
  - stop/release preempts streaming; disconnect cancels frames before ramping weight down
  - full-path joint, speed, acceleration and collision validation; no implicit lead-in frames
  - tracking error over limits.tracking_abort_rad for 0.3 s, or rt/lowstate stale for 0.5 s -> ramp down
  - per-frame joint speed re-checked against limits.max_joint_vel_rad_s; a faster frame is refused
  - waist yaw and head pitch/yaw are held at their measured values with Unitree's gains (robot.hold_head); with
    robot.head_pitch_rad the head pitch is commanded there on engage instead, so the head camera sees the hand
  - arm gains are Unitree's example gains times robot.arm_kp_scale (gravity droop at 1.0 was 2 to 3 cm per 4 cm step)
  - walk: {"cmd": "walk", "vx", "vy", "vyaw", "duration"} hands the arms back to the controller first (any weight > 0 on
    rt/arm_sdk puts the R1 into FSM 816 ArmSdkLoco, in which every walk was refused; it is back in 811 at weight 0), waits
    for an allowed FSM (locomotion.fsm_ok, fsm_wait_s), asks the loco service for that velocity for that long, logs the
    controller's answer code, sends an explicit stop whatever happens, and takes the arms back; only with
    locomotion.enabled in its own config, velocities and duration capped by the config on its own; odometry from
    rt/odommodestate comes back as {dx, dy, dyaw} in the pre-walk frame
Commands (one JSON object per line):  {"cmd": "hello"} {"cmd": "state"} {"cmd": "engage"}
  {"cmd": "frames", "arm": "right", "frames": [[q1..q5], ...], "dt": 0.02}   (blocks until streamed)
  {"cmd": "plan", "arm": "right", "plan": {...}, "digest": "..."} (resolved reviewed path)
  {"cmd": "freeze"} {"cmd": "release"} {"cmd": "heartbeat"} (no reply)
Every other command gets {"ok": false, "error": ...}.

    .venv/bin/python -m harness.robot.arm_stream en6 [--port 8790]
"""
import argparse
import json
import math
import signal
import socket
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

from core.robot_lease import RobotLease
from core import trajectory
from ..config import load
from ..kinematics import ARM_JOINTS
from .lowstate import FSM_ARM_OK, JOINT_TO_SLOT, LowStateReader, query_fsm

# gains as in unitree_sdk2/example/r1/high_level/r1_arm_sdk_dds_example.cpp
GAINS = {"shoulder_pitch": (50.0, 2.0), "shoulder_roll": (50.0, 2.0), "shoulder_yaw": (40.0, 2.0),
         "elbow": (40.0, 2.0), "wrist_roll": (30.0, 2.0)}
WAIST_YAW, HEAD = 13, (29, 30)
# answers to a velocity command that the SDK's error lists do not name
CODE_HINTS = {127: " (127 is in none of the SDK's error lists; another R1 EDU on ai_sport 1.0.2.154 gets it for every velocity "
                   "command in every state, unitreerobotics/xr_teleoperate#319, which points at SDK locomotion being switched off "
                   "in that firmware)"}


def _loco():
    """A loco client (SetVelocity / StopMove). Separate so tests can replace it."""
    from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
    lc = LocoClient(); lc.SetTimeout(3.0); lc.Init()
    return lc


def _odom_sub(cb):
    """Subscribe to the controller's odometry (position, yaw) for walk feedback: rt/odommodestate is what the R1 publishes
    (rt/sportmodestate exists but was silent on 2026-09-27); both are tried. None when the type is unavailable."""
    try:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        subs = []
        for topic in ("rt/odommodestate", "rt/sportmodestate"):
            sub = ChannelSubscriber(topic, SportModeState_); sub.Init(cb, 10); subs.append(sub)
        return subs
    except Exception:
        return None


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
        scale = float(cfg["robot"].get("arm_kp_scale", 1.0))     # >1 stiffens the arms against gravity droop
        for side in ("left", "right"):
            for n in ARM_JOINTS[side]:
                kp, kd = GAINS[n[len(side) + 1:-len("_joint")]]
                self.slots[JOINT_TO_SLOT[n]] = (kp * scale, kd)
        self.head_pitch = cfg["robot"].get("head_pitch_rad")          # None: hold the head where it is; else look there on engage
        self.loco = dict(cfg.get("locomotion") or {})
        self.walking = False
        self.odom = {"pos": None, "yaw": None, "t": 0.0}
        self.odom_sub = _odom_sub(self._on_odom)
        self.slots[WAIST_YAW] = (50.0, 3.0)
        if cfg["robot"]["hold_head"]:
            for s in HEAD: self.slots[s] = (15.0, 1.0)
        self.lock = threading.Lock()
        self.publish_lock = threading.Lock()
        self.targets = {}                                   # slot -> q
        self.weight = 0.0
        self.engaged = False
        self.last_client = time.time()
        self.serving = False                                # a client command is in dispatch: its silence is expected
        self.err_since = None
        self.stop = threading.Event()
        self.reason = ""
        self.frames_sent = 0
        self.motion_cancel = threading.Event()
        self.release_lock = threading.Lock()
        self.lease = RobotLease("arm streamer")

    # -- DDS side ---------------------------------------------------------------------------------
    def _on_odom(self, m):
        try:
            self.odom = {"pos": [float(v) for v in m.position[:3]], "yaw": float(m.imu_state.rpy[2]), "t": time.time()}
        except Exception:
            pass

    def _odom_now(self):
        o = self.odom
        return (list(o["pos"]), o["yaw"]) if o["pos"] is not None and time.time() - o["t"] < 1.0 else None

    def _await_fsm(self, ok, wait_s):
        """Poll the FSM until it is one of ok, for up to wait_s (the R1 leaves 816 the moment the arm weight is 0)."""
        t0 = time.time()
        while True:
            fsm, name = query_fsm()
            if fsm in ok or time.time() - t0 >= wait_s:
                return fsm, name
            if self.motion_cancel.wait(0.25) or self.stop.is_set():
                return fsm, name

    def _retake(self, held):
        """After a step: the arms back under the streamer when it held them before; '' or why not."""
        if not held:
            return ""
        if self.motion_cancel.is_set() or self.stop.is_set():
            return "arm retake cancelled"
        err = self.engage()
        if err:
            self.log(f"walk: the arms are NOT held again: {err}")
            return f"the arms could not be taken back after the step: {err}"
        return ""

    def walk(self, vx, vy, vyaw, duration):
        """-> (error, odometry). The streamer checks everything itself: enabled, caps, an allowed FSM. It hands the arms
        back to the controller for the step (any weight > 0 on rt/arm_sdk puts the R1 into FSM 816 ArmSdkLoco, in which
        every walk was refused; it is back in 811 at weight 0), asks the loco service for the velocity for duration
        seconds, sends an explicit stop whatever happens, then takes the arms back if it held them before."""
        lo = self.loco
        if not all(math.isfinite(v) for v in (vx, vy, vyaw, duration)):
            return "walk parameters must be finite", None
        if self.motion_cancel.is_set():
            return "walk cancelled", None
        if not lo.get("enabled", False):
            return "walking is disabled in the streamer's config (locomotion.enabled)", None
        v, w = float(lo["speed_mps"]), float(lo["turn_speed_rps"])
        if abs(vx) > v * 1.05 or abs(vy) > v * 1.05 or abs(vyaw) > w * 1.05:
            return f"refused: velocity over the cap ({v} m/s, {w} rad/s)", None
        t_max = max(float(lo["param_max_walk_m"]) / v, math.radians(float(lo["param_max_turn_deg"])) / w) * 1.05 + 0.3
        if not (0.0 < duration <= t_max):
            return f"refused: duration {duration:.1f} s over the cap ({t_max:.1f} s)", None
        fsm_ok = {int(x) for x in (lo.get("fsm_ok") or [811])}      # which FSM ids may walk: the operator's list (config), 811 by default
        held = self.engaged
        if held:
            self.release("walk: the arms go back to the controller for the step (the arm topic puts the R1 into FSM 816, which did not walk)", cancel_motion=False)
        fsm, name = self._await_fsm(fsm_ok, float(lo.get("fsm_wait_s", 2.0)))
        if self.motion_cancel.is_set() or self.stop.is_set():
            return "walk cancelled", None
        if fsm not in fsm_ok:
            back = self._retake(held)
            return (f"refused: walking is allowed in FSM {sorted(fsm_ok)} (locomotion.fsm_ok); the robot is in {fsm} = {name}"
                    + (" even with the arm topic released" if held else "") + (f"; {back}" if back else "")), None
        before = self._odom_now()
        lc = _loco()
        self.walking = True
        self.log(f"walk: vx={vx:+.2f} vy={vy:+.2f} m/s yaw={vyaw:+.2f} rad/s for {duration:.1f} s in FSM {fsm}")
        err = ""
        try:
            code = lc.SetVelocity(float(vx), float(vy), float(vyaw), float(duration))
            self.log(f"walk: the controller answered {code} to the velocity command")
            if code not in (0, None):
                err = f"the controller refused the velocity command (code {code}){CODE_HINTS.get(code, '')}; nothing moved"
            elif self.motion_cancel.wait(float(duration)):
                return "walk cancelled", None
        finally:
            try:
                lc.StopMove()
            finally:
                self.walking = False
        odom = None
        if not err:
            if self.motion_cancel.wait(float(lo.get("settle_s", 1.0))) or self.stop.is_set():
                return "walk cancelled", None
            after = self._odom_now()
            if before is not None and after is not None:
                (p0, y0), (p1, y1) = before, after
                dxw, dyw = p1[0] - p0[0], p1[1] - p0[1]
                c, s = math.cos(-y0), math.sin(-y0)
                dyaw = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
                odom = {"dx": c * dxw - s * dyw, "dy": s * dxw + c * dyw, "dyaw": dyaw}
        back = self._retake(held)
        if back:
            err = (err + "; " if err else "the step was sent, but ") + back
        return err, odom

    def stop_walking(self):
        if self.walking:
            try:
                _loco().StopMove()
            except Exception as e:
                self.log(f"stop failed: {e}")

    def measured(self):
        m = self.reader.msg
        return {s: float(m.motor_state[s].q) for s in self.slots}

    def send(self):
        with self.publish_lock:
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
            if to > 0 and self.motion_cancel.is_set():
                return
            with self.lock:
                self.weight = w0 + (to - w0) * el / seconds
            self.send(); time.sleep(self.dt)
        with self.lock:
            self.weight = to
        self.send()

    def engage(self):
        fsm, name = query_fsm()
        ok = {int(v) for v in (self.cfg["robot"].get("fsm_ok_arms") or FSM_ARM_OK)}
        if fsm not in ok:
            return f"refused: FSM {fsm} = {name}; the arm topic is used only in {sorted(ok)} (robot.fsm_ok_arms)"
        if self.motion_cancel.is_set():
            return "engage cancelled"
        self.lease.acquire()
        with self.lock:
            self.targets = self.measured()                  # hold everything where it is
            if self.head_pitch is not None and HEAD[0] in self.slots:
                self.targets[HEAD[0]] = float(self.head_pitch)   # the head camera looks at the workspace (positive = down)
            self.engaged = True
        self.err_since = None
        self.log(f"engage: FSM {fsm} = {name}; ramping weight up over {self.ramp_s} s")
        self.ramp(1.0, self.ramp_s)
        return ""

    def release(self, reason="release", seconds=None, *, cancel_motion=True):
        if cancel_motion:
            self.motion_cancel.set()
        self.stop_walking()
        with self.release_lock:
            if not self.engaged and self.weight == 0.0:
                self.lease.release()
                return
            self.engaged = False
            self.reason = reason
            self.log(f"{reason}: ramping weight down")
            try:
                self.ramp(0.0, seconds or self.ramp_s)
            finally:
                self.lease.release()

    def hold_loop(self):
        """Background: keep publishing the current targets while engaged, run the watchdogs."""
        while not self.stop.is_set():
            t = time.time()
            if self.walking and (self.reader.age() > .5 or time.time() - self.last_client > self.watchdog_s):
                self.release("ABORT: walk telemetry or heartbeat lost", .5)
            if self.engaged:
                if self.reader.age() > 0.5:
                    self.release("ABORT: rt/lowstate stale", 0.5)
                elif time.time() - self.last_client > self.watchdog_s:
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
        if arm not in ARM_JOINTS or not np.isfinite(dt) or not .005 <= dt <= .1:
            return "invalid arm or command interval"
        names = ARM_JOINTS[arm]; slots = [JOINT_TO_SLOT[n] for n in names]
        if not 1 <= len(frames) <= 9000 or len(frames)*dt > 180:
            return "invalid trajectory size"
        frames = [np.asarray(f, float) for f in frames]
        for i, f in enumerate(frames):
            if f.shape != (5,) or not np.all(np.isfinite(f)):
                return f"frame {i}: expected 5 finite joint values"
        with self.lock:
            prev = np.array([self.targets[s] for s in slots])
        # Never alter an approved trajectory to reconcile a stale starting target.
        measured = self.reader.joints()
        plan = trajectory.frame_plan(arm, prev, frames, dt, measured)
        try:
            trajectory.validate(plan, arm)
        except (ValueError, KeyError) as exc:
            return str(exc)
        for i, f in enumerate(frames):
            v = float(np.abs(f - prev).max() / dt)
            if v > self.vmax * 1.05:
                return f"frame {i}: {v:.2f} rad/s over the {self.vmax} rad/s cap; refused before sending"
            prev = f
        if self.motion_cancel.is_set():
            return "motion stopped; engage again before a new motion"
        self.streaming = True
        try:
            t0 = time.time()
            for i, f in enumerate(frames):
                if not self.engaged or self.motion_cancel.is_set():
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
                    "targets": {n: self.targets[JOINT_TO_SLOT[n]] for names in ARM_JOINTS.values() for n in names if JOINT_TO_SLOT[n] in self.targets},
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
        send_lock = threading.Lock()
        worker = None

        def answer(req, resp):
            try:
                with send_lock:
                    conn.sendall((json.dumps({**resp, "request_id": req.get("request_id")}) + "\n").encode())
            except OSError:
                self.motion_cancel.set()

        def run(req):
            try:
                answer(req, self.dispatch(req.get("cmd"), req))
            except Exception as exc:
                answer(req, {"ok": False, "error": str(exc)})
            finally:
                self.serving = False

        try:
            while not self.stop.is_set():
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                self.last_client = time.time()
                buf += chunk
                if len(buf) > 8*1024*1024:
                    break
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        req = json.loads(line)
                        if not isinstance(req, dict): raise ValueError()
                    except ValueError:
                        answer({}, {"ok": False, "error": "invalid command"}); continue
                    cmd = req.get("cmd")
                    if cmd == "heartbeat":
                        continue
                    if cmd in ("frames", "plan", "engage", "walk"):
                        if worker and worker.is_alive():
                            answer(req, {"ok": False, "error": "motion command already active"})
                        else:
                            if cmd == "engage": self.motion_cancel.clear()
                            self.serving = True
                            worker = threading.Thread(target=run, args=(req,), daemon=True)
                            worker.start()
                    else:
                        # Stop/release/state are received even while frames are streaming.
                        try: answer(req, self.dispatch(cmd, req))
                        except Exception as exc: answer(req, {"ok": False, "error": str(exc)})
        finally:
            self.motion_cancel.set()
            self.stop_walking()
            if self.engaged:
                self.release("client disconnected", .5)
            if worker:
                worker.join(3)

    def dispatch(self, cmd, req):
        if cmd == "hello":
            fsm, name = query_fsm()
            return {"ok": True, "fsm": fsm, "fsm_name": name, **self.state()}
        if cmd == "state":
            return self.state()
        if cmd == "engage":
            err = self.engage()
            return {**self.state(), "ok": not err, "error": err}       # the flag last: state() carries its own ok
        if cmd == "plan":
            plan, arm = req["plan"], req["arm"]
            if trajectory.digest(plan) != req.get("digest"):
                return {"ok": False, "error": "plan digest mismatch"}
            measured = self.reader.joints()
            trajectory.require_start(plan, measured)
            trajectory.validate(plan, arm)
            frames, dt = trajectory.frames(plan, arm)
            # The streamer verifies the exact reviewed starting targets too.
            with self.lock:
                targets = {n: self.targets[JOINT_TO_SLOT[n]] for n in ARM_JOINTS[arm]}
            if any(abs(targets[n]-plan["keyframes"][0]["joint_targets_rad"][n]) > .002 for n in targets):
                return {"ok": False, "error": "hold target changed; regenerate the proposal"}
            err = self.stream_frames(arm, frames, dt)
            return {**self.state(), "ok": not err, "error": err}
        if cmd == "frames":
            n = len(req.get("frames") or []); t0 = time.time()
            err = self.stream_frames(req["arm"], req["frames"], float(req["dt"]))
            self.log(f"frames: {req.get('arm')} arm, {n} frames over {n * float(req['dt']):.2f} s -> " + (f"REFUSED: {err}" if err else f"streamed in {time.time() - t0:.2f} s"))
            return {**self.state(), "ok": not err, "error": err}
        if cmd == "walk":
            err, odom = self.walk(float(req.get("vx", 0.0)), float(req.get("vy", 0.0)), float(req.get("vyaw", 0.0)), float(req.get("duration", 0.0)))
            self.log("walk -> " + (f"REFUSED: {err}" if err else f"done, odometry {odom}"))
            return {**self.state(), "ok": not err, "error": err, "odom": odom}
        if cmd == "freeze":
            self.motion_cancel.set()
            self.stop_walking()
            with self.lock:
                self.targets = self.measured() if not self.engaged else dict(self.targets)
            return {"ok": True, **self.state()}
        if cmd == "release":
            self.release(); return {"ok": True, **self.state()}
        return {"ok": False, "error": f"unknown command {cmd!r}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iface"); ap.add_argument("--port", type=int); ap.add_argument("--config")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="config override, e.g. locomotion.enabled=true")
    a = ap.parse_args()
    over = {}
    for it in a.set or []:
        k, _, v = it.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        over[k] = v
    cfg = load(a.config, over)
    st = Streamer(cfg, a.iface)
    threading.Thread(target=st.hold_loop, daemon=True).start()

    def on_sigint(*_):
        st.stop_walking()
        if st.engaged:
            signal.signal(signal.SIGINT, lambda *_: print("(already releasing: the weight ramps down first)"))
            st.release("Ctrl-C")
        st.stop.set()
    signal.signal(signal.SIGINT, on_sigint)
    try:
        st.serve(cfg["streamer"]["host"], a.port or int(cfg["streamer"]["port"]))
    finally:
        st.motion_cancel.set()
        st.stop_walking()
        if st.engaged:
            st.release("exit")
        st.reader.close()


if __name__ == "__main__":
    main()
