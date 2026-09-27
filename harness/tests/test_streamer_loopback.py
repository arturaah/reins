"""The executor, the arm client and the real streamer together over the local socket, with a fake robot that droops
under gravity. This is the path that failed in the 2026-09-27 11:32 and 11:46 live sessions: every move was planned
from the measured joints, refused by the streamer for the droop jump, reported as ok, and the model was told the arm
was blocked. DDS is replaced by fakes; nothing is published."""
import socket
import threading
import time
import types

import numpy as np
import pytest

pytest.importorskip("unitree_sdk2py")
from harness.actions import parse_action
from harness.executor import ArmExecutor, StreamError
from harness.interpreter import Interpreter
from harness.kinematics import ARM_JOINTS, ArmKinematics
from harness.robot import arm_stream as am
from harness.robot.arm_client import ArmClientBackend
from harness.robot.lowstate import SLOT_TO_JOINT
from harness.safety import SafetyGate

DROOP = 0.02            # rad the fake arm sags below every command: 1 rad/s if jumped in one 50 Hz frame, over the cap


class FakeReader:
    def __init__(self, iface=None):
        self.msg = types.SimpleNamespace(motor_state=[types.SimpleNamespace(q=0.0, dq=0.0) for _ in range(35)])
    def wait(self): return True
    def age(self): return 0.0
    def joints(self): return {n: float(self.msg.motor_state[s].q) for s, n in SLOT_TO_JOINT.items()}
    def velocities(self): return {n: 0.0 for n in SLOT_TO_JOINT.values()}
    def close(self): pass


class FakePub:
    def __init__(self, *a): pass
    def Init(self): pass
    def Write(self, cmd): pass


@pytest.fixture
def rig(cfg, monkeypatch):
    monkeypatch.setattr(am, "LowStateReader", FakeReader)
    monkeypatch.setattr(am, "ChannelPublisher", FakePub)
    monkeypatch.setattr(am, "query_fsm", lambda: (811, "Start (balance control)"))
    probe = socket.socket(); probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]; probe.close()
    cfg["robot"]["weight_ramp_s"] = 0.2; cfg["streamer"]["port"] = port
    st = am.Streamer(cfg, "lo0", log=lambda *a: None)
    arm_slots = {am.JOINT_TO_SLOT[n] for side in ("left", "right") for n in ARM_JOINTS[side]}
    for n, q in zip(ARM_JOINTS["right"], cfg["robot"]["start_pose_rad"]["right"]):
        st.reader.msg.motor_state[am.JOINT_TO_SLOT[n]].q = q
    def droop(cmd):                                     # the fake arm settles DROOP below each arm command at once
        if cmd.mode_pr > 0:
            for s in range(35):
                st.reader.msg.motor_state[s].q = cmd.motor_cmd[s].q - (DROOP if s in arm_slots else 0.0)
    st.pub.Write = droop
    threads = [threading.Thread(target=st.hold_loop, daemon=True),
               threading.Thread(target=st.serve, args=("127.0.0.1", port), daemon=True)]
    for t in threads: t.start()
    for _ in range(50):
        try:
            backend = ArmClientBackend(cfg, log=lambda *a: None); break
        except OSError:
            time.sleep(0.05)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    sent = []
    real = backend.stream
    backend.stream = lambda arm, frames, dt: (sent.append(np.array(frames)), real(arm, frames, dt))
    yield st, backend, ex, Interpreter(cfg["frames"]["view_forward"], cfg["frames"]["view_left"]), sent
    try:
        backend.release()
    except OSError:
        pass
    st.stop.set()
    for t in threads: t.join(timeout=2)


def test_move_streams_from_the_hold_target_despite_droop(rig):
    st, backend, ex, it, sent = rig
    backend.engage()
    s = ex.sync()
    held = ex.q_start(s.q); measured = ex.kin.q_from_dict(s.q)
    assert np.allclose(held - measured, DROOP, atol=1e-6)                      # the client sees the droop
    r = ex.execute(it.propose(s, parse_action("MV_UP"), 0.04, 0.1), s)
    assert r.ok and not r.stream_error, r.feedback
    first = sent[-1][0]
    assert np.abs(first - held).max() / 0.02 <= st.vmax * 0.8                  # no droop jump: no lead-in needed
    assert "blocked" not in r.feedback and r.achieved_dp[2] > 0.03, r.feedback


def test_refusal_stops_instead_of_blocked(rig):
    st, backend, ex, it, sent = rig
    backend.engage()
    st.release("ABORT: test", 0.05)                                            # the streamer let go on its own
    s = ex.sync()
    r = ex.execute(it.propose(s, parse_action("MV_UP"), 0.04, 0.1), s)
    assert not r.ok and r.stream_error and r.feedback.startswith("NOT SENT"), r.feedback
    assert "blocked" not in r.feedback


def test_start_pose_refusal_is_not_done(rig):
    st, backend, ex, it, sent = rig
    backend.engage()
    st.release("ABORT: test", 0.05)
    r = ex.go_to_joints(np.asarray(ex.cfg["robot"]["start_pose_rad"]["right"]) + 0.1, "start pose")
    assert not r.ok and r.stream_error and "NOT SENT" in r.feedback


def test_client_catches_an_instant_ok():
    """A streamer that says ok but returns far sooner than the frames last did not play them."""
    b = ArmClientBackend.__new__(ArmClientBackend)
    b.call = lambda req, timeout=None: {"ok": True, "engaged": True}
    with pytest.raises(StreamError, match="not streamed"):
        b.stream("right", [np.zeros(5)] * 50, 0.02)
