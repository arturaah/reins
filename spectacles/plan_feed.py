#!/usr/bin/env python3
"""Serve a Reins plan to Spectacles as robot-base or room-fixed hand paths.

Reads a plan file (schema_version 1, keyframes of MuJoCo joint names in
radians: sim/plans, tools/plans, recordings/, or the resolved plan that
`tools/arm_lift.py IFACE --plan ...` writes on every dry run), poses the R1
fixed-base model at each sample (forward kinematics only, no physics), and
streams both hand-tip paths in the Lens's `trajectory` format. Opens nothing
towards the robot.

    .venv/bin/python spectacles/plan_feed.py sim/plans/arm_lift_dryrun.json --state-url ws://ROBOT_MAC:8766
    .venv/bin/python spectacles/plan_feed.py tools/plans/cup_grab_right.json --print

The plan file is re-read when it changes. With --state-url, measured joints
from tools/relay.py advance the remaining path while rt/arm_sdk is active.
With --robot-iface, subscribe directly on the robot-side Mac instead of
running a separate relay. Both modes are read-only.
The Lens falls back to its mock after 1.5 s of silence, so the path is
resent every --period seconds. No command is sent to the robot.

Frame: the fixed-base model pins the pelvis at 0.74 m above the world origin
with x forward, y left, z up, which is the Lens's robot_base (origin on the
floor under the pelvis). Joints a plan does not name are taken from its
optional `held_joints_rad` map (written by arm_lift's dry run from the
measured pose), else 0.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
try:
    from .review import ReviewMailbox
    from .voice_inbox import VoiceInbox
except ImportError:  # also runnable as `python spectacles/plan_feed.py`
    from review import ReviewMailbox
    from voice_inbox import VoiceInbox

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "sim/models/r1/R1_fixed_base.xml"
MAX_POINTS = 512   # the Lens rejects longer paths


def base_path(plan, times):
    """Optional planar base poses in a map whose origin is the initial robot base."""
    frames = plan.get("base_keyframes")
    if frames is None:
        return None
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("base_keyframes needs at least two poses")
    values = np.asarray([[float(f[k]) for k in ("time_s", "x_m", "y_m", "yaw_rad")]
                         for f in frames], dtype=float)
    if not np.all(np.isfinite(values)) or values[0, 0] != 0 or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("base_keyframes must be finite, start at 0, and increase in time")
    if np.any(np.abs(values[0, 1:]) > 1e-6):
        raise ValueError("the map origin must be the initial robot base pose [0, 0, 0]")
    if values[-1, 0] < times[-1] - 1e-6:
        raise ValueError("base_keyframes must cover the full plan duration")
    yaw = np.unwrap(values[:, 3])
    return np.column_stack((np.interp(times, values[:, 0], values[:, 1]),
                            np.interp(times, values[:, 0], values[:, 2]),
                            np.interp(times, values[:, 0], yaw)))


def sample_plan(model, plan, points):
    frames = sorted(plan["keyframes"], key=lambda f: f["time_s"])
    times = np.array([float(f["time_s"]) for f in frames])
    names = list(frames[0]["joint_targets_rad"])
    held = {**plan.get("held_joints_rad", {})}
    adr = {}
    for name in set(names) | set(held):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint not in the R1 model: {name}")
        adr[name] = model.jnt_qposadr[jid]
    q = {n: np.array([float(f["joint_targets_rad"][n]) for f in frames]) for n in names}
    sites = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_preview")
             for side in ("left", "right")}
    data = mujoco.MjData(model)
    hands = {"left": [], "right": []}
    joint_samples = []
    sample_times = np.linspace(times[0], times[-1], points)
    bases = base_path(plan, sample_times)
    # Sample evenly in time: joint-space interpolation traces the same path
    # whether arm_lift eases the segments or not.
    for index, t in enumerate(sample_times):
        data.qpos[:] = 0.0
        for n, v in held.items():
            data.qpos[adr[n]] = v
        for n in names:
            data.qpos[adr[n]] = np.interp(t, times, q[n])
        joint_samples.append([float(data.qpos[adr[n]]) for n in names])
        mujoco.mj_kinematics(model, data)
        for side, site in sites.items():
            p = data.site_xpos[site]
            if bases is not None:
                x, y, yaw = bases[index]
                c, s = np.cos(yaw), np.sin(yaw)
                p = (x + c*p[0] - s*p[1], y + s*p[0] + c*p[1], p[2])
            hands[side].append([round(float(v), 4) for v in p])
    return hands, names, np.asarray(joint_samples)


def hand_paths(model, plan, points):
    return sample_plan(model, plan, points)[0]


def message(model, path, points):
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1 or len(plan.get("keyframes", [])) < 2:
        raise ValueError("plan needs schema_version 1 and at least two keyframes")
    return {
        "type": "trajectory",
        "version": 1,
        "id": f"{plan.get('name', path.stem)}@{int(path.stat().st_mtime)}",
        "frame": "map" if "base_keyframes" in plan else "robot_base",
        "units": "m",
        "duration_s": float(plan.get("duration_s", plan["keyframes"][-1]["time_s"])),
        "hands": hand_paths(model, plan, points),
    }


class Feed:
    def __init__(self, model, path, points):
        self.model, self.path, self.points = model, path, points
        self.mtime, self.text, self.live_text = None, None, None
        self.message, self.names, self.joints, self.bases = None, None, None, None
        self.progress = None
        self.site = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_preview")
                     for side in ("left", "right")}
        self.joint_addr = {name: int(model.jnt_qposadr[i]) for i in range(model.njnt)
                           if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))}
        self.live_data = mujoco.MjData(model)

    def remaining(self, state, base_pose=None):
        """Use measured robot joints to advance a plan; never infer progress from AR."""
        map_plan = self.message is not None and self.message["frame"] == "map"
        if map_plan and base_pose is None:
            # rt/lowstate provides arm joints, not the robot's measured map pose.
            return self.text
        if not state or not state.fresh() or self.message is None:
            return self.live_text or self.text
        q = state.q
        if any(n not in q for n in self.names):
            return self.live_text or self.text
        try:
            measured = np.array([q[n] for n in self.names], dtype=float)
        except (TypeError, ValueError):
            return self.live_text or self.text
        if not np.all(np.isfinite(measured)):
            return self.live_text or self.text
        self.live_data.qpos[:] = 0.0
        for name, value in q.items():
            if name in self.joint_addr and isinstance(value, (int, float)) and np.isfinite(value):
                self.live_data.qpos[self.joint_addr[name]] = value
        mujoco.mj_kinematics(self.model, self.live_data)
        current_hands = {side: [round(float(v), 4) for v in self.live_data.site_xpos[self.site[side]]]
                         for side in ("left", "right")}
        if map_plan:
            x, y, yaw = (base_pose[k] for k in ("x_m", "y_m", "yaw_rad"))
            c, s = np.cos(yaw), np.sin(yaw)
            current_hands = {side: [round(x+c*p[0]-s*p[1], 4),
                                    round(y+s*p[0]+c*p[1], 4), p[2]]
                             for side, p in current_hands.items()}
        base_moved = (map_plan and
                      (np.hypot(base_pose["x_m"], base_pose["y_m"]) > 0.03 or
                       abs(base_pose["yaw_rad"]) > 0.05))
        if self.progress is None and not state.commanding and not base_moved:
            # Show the proposed path before execution, but draw an idle hand
            # as its measured position rather than a zero-length polyline.
            hands = {}
            for side in ("left", "right"):
                path = self.message["hands"][side]
                moving = max(np.linalg.norm(np.asarray(p) - path[0]) for p in path) >= 0.01
                hands[side] = path if moving else [current_hands[side]]
            self.live_text = json.dumps({**self.message, "hands": hands,
                                         "progress_source": "measured_joints"})
            return self.live_text
        start = self.progress or 0
        # Search only at or after the last measured position. Progress never
        # moves backwards, and a viewer can join during an ongoing execution.
        errors = np.sqrt(np.mean((self.joints[start:] - measured) ** 2, axis=1))
        if map_plan:
            delta = self.bases[start:] - np.array([base_pose[k] for k in
                                                    ("x_m", "y_m", "yaw_rad")])
            delta[:, 2] = (delta[:, 2] + np.pi) % (2*np.pi) - np.pi
            errors = np.sqrt(errors**2 + np.sum(delta[:, :2]**2, axis=1) +
                             (0.2*delta[:, 2])**2)
        best_error = float(np.min(errors))
        if best_error > 0.25:
            # A command unrelated to this plan must not leave an old path
            # apparently attached to a moving robot.
            self.live_text = json.dumps({**self.message,
                                         "hands": {s: [p] for s, p in current_hands.items()},
                                         "progress_source": "measured_joints"})
            return self.live_text
        # A dry-run can revisit the same joint pose on its return leg (or
        # hold it across many samples). Measurement noise can make a much
        # later sample marginally closer and otherwise erase the whole path.
        # Keep the earliest sample whose fit is effectively as good.
        candidate = start + int(np.flatnonzero(errors <= best_error + 0.01)[0])
        if self.progress is None:
            self.progress = candidate
        else:
            self.progress = max(self.progress, candidate)
        hands = {}
        for side in ("left", "right"):
            tail = self.message["hands"][side][self.progress + 1:]
            current = current_hands[side]
            # A single measured point keeps a completed or stationary hand
            # visible; only moving hands retain a path.
            hands[side] = ([current] + tail if tail and
                           max(np.linalg.norm(np.asarray(p) - current) for p in tail) >= 0.01 else [current])
        self.live_text = json.dumps({**self.message, "hands": hands,
                                     "progress_source": "measured_joints"})
        return self.live_text

    def current(self, state=None, base_pose=None):
        """The latest message, re-reading the plan when its file changes."""
        try:
            mtime = self.path.stat().st_mtime_ns
            if mtime != self.mtime:
                plan = json.loads(self.path.read_text())
                if plan.get("schema_version") != 1 or len(plan.get("keyframes", [])) < 2:
                    raise ValueError("plan needs schema_version 1 and at least two keyframes")
                hands, names, joints = sample_plan(self.model, plan, self.points)
                frames = sorted(plan["keyframes"], key=lambda f: f["time_s"])
                bases = base_path(plan, np.linspace(float(frames[0]["time_s"]),
                                                     float(frames[-1]["time_s"]), self.points))
                msg = {"type": "trajectory", "version": 1,
                       "id": f"{plan.get('name', self.path.stem)}@{mtime}",
                       "frame": "map" if "base_keyframes" in plan else "robot_base", "units": "m",
                       "duration_s": float(plan.get("duration_s", plan["keyframes"][-1]["time_s"])),
                       "hands": hands}
                self.message, self.names, self.joints, self.bases = msg, names, joints, bases
                self.progress, self.live_text = None, None
                self.text = json.dumps(msg)
                self.mtime = mtime
                span = {s: np.ptp(np.array(p), axis=0).round(3).tolist() for s, p in msg["hands"].items()}
                print(f"loaded {self.path.name}: {msg['duration_s']:.1f} s, "
                      f"{self.points} points per hand, extent (m) {span}", file=sys.stderr, flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"cannot use {self.path}: {exc}; keeping the previous path", file=sys.stderr, flush=True)
        return self.remaining(state, base_pose) if state else self.text


class RelayState:
    def __init__(self):
        self.q, self.commanding, self.received_at = None, False, 0.0
        self.source_t, self.source_change_at = None, 0.0

    def update(self, message):
        if message.get("type") != "r1_state" or message.get("version") != 1:
            return
        q = message.get("q")
        if not isinstance(q, dict):
            return
        cmd = message.get("cmd")
        self.q = q
        weight = cmd.get("weight", 0) if isinstance(cmd, dict) else 0
        self.commanding = isinstance(weight, (int, float)) and weight > 0
        self.received_at = time.monotonic()
        source_t = message.get("t")
        if isinstance(source_t, (int, float)) and source_t != self.source_t:
            self.source_t, self.source_change_at = source_t, self.received_at

    def fresh(self):
        now = time.monotonic()
        return (self.q is not None and now - self.received_at < 0.5 and
                (self.source_t is None or now - self.source_change_at < 0.5))


class DirectRobotState:
    """Adapt the existing DDS subscriber to the feed's measured-state interface."""
    def __init__(self, iface, domain=0):
        sys.path.insert(0, str(ROOT / "tools"))
        from twin import RobotState, listen_dds
        self.robot = RobotState()
        self.readers = listen_dds(self.robot, iface, domain)

    @property
    def q(self):
        return self.robot.q

    @property
    def commanding(self):
        return self.robot.commanding(time.time())

    def fresh(self):
        return self.robot.q is not None and time.time() - self.robot.t < 0.5


class BasePoseFile:
    """Read a measured planar pose supplied by the robot-side controller."""
    def __init__(self, path):
        self.path = path

    def read(self):
        try:
            if time.time() - self.path.stat().st_mtime > 1.0:
                return None
            pose = json.loads(self.path.read_text())
            if pose.get("frame") != "map":
                return None
            values = {key: float(pose[key]) for key in ("x_m", "y_m", "yaw_rad")}
            return values if all(np.isfinite(v) for v in values.values()) else None
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None


async def listen_relay(state, url):
    from websockets.asyncio.client import connect
    while True:
        try:
            async with connect(url) as websocket:
                print(f"robot state connected: {url}", file=sys.stderr, flush=True)
                async for text in websocket:
                    state.update(json.loads(text))
        except Exception as exc:
            print(f"robot state unavailable: {exc}; retrying", file=sys.stderr, flush=True)
        await asyncio.sleep(2)


async def serve_feed(feed, host, port, period, state=None, base_pose_source=None, review=None, voice=None):
    from websockets.asyncio.server import serve

    async def handler(websocket):
        print(f"Lens connected: {websocket.remote_address}", file=sys.stderr, flush=True)
        last_review_id = None
        async def send_paths():
            nonlocal last_review_id
            while True:
                payload = feed.current(state, base_pose_source.read() if base_pose_source else None)
                if payload:
                    if review and (pending := review.pending(feed.path)):
                        # A proposal has not been approved yet. Always show its
                        # complete path; earlier rt/arm_sdk traffic may otherwise
                        # make joint matching collapse it to a hand point.
                        message = json.loads(feed.text)
                        message["review"] = pending
                        payload = json.dumps(message)
                        if pending["id"] != last_review_id:
                            print(f"Spectacles review pending: {pending['id']} ({pending['mode']})",
                                  file=sys.stderr, flush=True)
                            last_review_id = pending["id"]
                    elif last_review_id:
                        print(f"Spectacles review cleared: {last_review_id}", file=sys.stderr, flush=True)
                        last_review_id = None
                    await websocket.send(payload)
                await asyncio.sleep(period)

        async def receive_decisions():
            async for raw in websocket:
                try:
                    if not isinstance(raw, str) or len(raw) > 1024:
                        continue
                    msg = json.loads(raw)
                    if msg.get("type") == "voice_command" and msg.get("version") == 1:
                        accepted = bool(voice and voice.enqueue(msg.get("id"), msg.get("text")))
                        print(f"Spectacles voice command: {'queued' if accepted else 'ignored'}: "
                              f"{str(msg.get('text', ''))[:100]}", file=sys.stderr, flush=True)
                        await websocket.send(json.dumps({"type": "voice_ack", "version": 1,
                                                         "id": msg.get("id"), "accepted": accepted}))
                        continue
                    if msg.get("type") != "review_decision" or msg.get("version") != 1:
                        continue
                    accepted = bool(review and review.decide(feed.path, msg.get("id"), msg.get("decision")))
                    print(f"Spectacles review decision: {msg.get('decision')} for {msg.get('id')}: "
                          f"{'accepted' if accepted else 'ignored (stale or unmatched)'}",
                          file=sys.stderr, flush=True)
                    await websocket.send(json.dumps({"type": "review_ack", "version": 1,
                                                     "id": msg.get("id"), "accepted": accepted}))
                except (ValueError, TypeError, AttributeError, OSError) as exc:
                    print(f"bad Spectacles review decision: {exc}", file=sys.stderr, flush=True)
        try:
            sender = asyncio.create_task(send_paths())
            receiver = asyncio.create_task(receive_decisions())
            done, pending_tasks = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
            for task in pending_tasks:
                task.cancel()
            for task in done:
                task.result()
        except Exception as exc:
            print(f"Lens disconnected: {exc}", file=sys.stderr, flush=True)

    async with serve(handler, host, port):
        print(f"Plan feed for {feed.path}: ws://{host}:{port}", file=sys.stderr, flush=True)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", type=Path, help="plan JSON (schema_version 1, MuJoCo joint names)")
    ap.add_argument("--points", type=int, default=200, help=f"samples per hand, 2..{MAX_POINTS}")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--period", type=float, default=0.25, help="seconds between resends (Lens times out at 1.5)")
    ap.add_argument("--print", action="store_true", help="print one message and exit, no server")
    state_source = ap.add_mutually_exclusive_group()
    state_source.add_argument("--state-url", help="read-only relay, e.g. ws://ARTUR_MAC:8766")
    state_source.add_argument("--robot-iface", help="subscribe to R1 DDS directly on this Mac, e.g. en6")
    ap.add_argument("--base-pose-file", type=Path,
                    help="fresh measured map pose JSON from the walking controller; never inferred from glasses")
    ap.add_argument("--review-file", type=Path,
                    help="Spectacles accept/reject mailbox written by the harness; serve the same --preview plan")
    ap.add_argument("--voice-inbox", type=Path, default=Path("runs/spectacles_voice.json"),
                    help="mailbox for Spectacles speech; the desktop UI consumes it as a dry-run Claude task")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain for --robot-iface (default: 0)")
    a = ap.parse_args()
    if not 2 <= a.points <= MAX_POINTS:
        ap.error(f"--points must be 2..{MAX_POINTS}")
    if a.base_pose_file and not (a.state_url or a.robot_iface):
        ap.error("--base-pose-file also needs measured joints via --robot-iface or --state-url")
    feed = Feed(mujoco.MjModel.from_xml_path(str(MJCF)), a.plan.resolve(), a.points)
    if a.print:
        text = feed.current()
        if text is None:
            raise SystemExit(1)
        print(text)
        return
    async def run():
        if a.review_file:
            print(f"Spectacles review mailbox: {a.review_file.resolve()}", file=sys.stderr, flush=True)
        state = DirectRobotState(a.robot_iface, a.domain) if a.robot_iface else (RelayState() if a.state_url else None)
        if a.state_url:
            asyncio.create_task(listen_relay(state, a.state_url))
        await serve_feed(feed, a.host, a.port, a.period, state,
                         BasePoseFile(a.base_pose_file) if a.base_pose_file else None,
                         ReviewMailbox(a.review_file.resolve()) if a.review_file else None,
                         VoiceInbox(a.voice_inbox.resolve()))
    asyncio.run(run())


if __name__ == "__main__":
    main()
