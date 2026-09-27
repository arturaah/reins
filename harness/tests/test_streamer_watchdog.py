"""Hardware-free streamer fault injection: fake telemetry, publisher and CRC."""
import json
import socket
import threading
import time
import types

import numpy as np
import pytest
from harness.robot import arm_stream as am
from harness.executor import interpolate
from core.robot_lease import RobotLease


class FakeReader:
    def __init__(self, iface=None):
        self.msg = types.SimpleNamespace(motor_state=[types.SimpleNamespace(q=0., dq=0.) for _ in range(35)])
    def wait(self): return True
    def age(self): return 0.
    def joints(self): return {n: self.msg.motor_state[slot].q for n, slot in am.JOINT_TO_SLOT.items()}
    def velocities(self): return {n: 0. for n in am.JOINT_TO_SLOT}
    def close(self): pass


class FakePub:
    def __init__(self, *args): self.n = 0
    def Init(self): pass
    def Write(self, cmd): self.n += 1


@pytest.fixture
def streamer(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(am, "LowStateReader", FakeReader)
    monkeypatch.setattr(am, "ChannelPublisher", FakePub)
    monkeypatch.setattr(am, "_odom_sub", lambda cb: None)
    monkeypatch.setattr(am, "CRC", lambda: types.SimpleNamespace(Crc=lambda cmd: 0))
    monkeypatch.setattr(am, "query_fsm", lambda: (811, "Start"))
    cfg["robot"]["arm_kp_scale"] = 1.5
    cfg["robot"]["head_pitch_rad"] = .35
    cfg["robot"]["weight_ramp_s"] = .02
    cfg["streamer"]["watchdog_s"] = .2
    st = am.Streamer(cfg, "unused", log=lambda *args: None)
    st.lease = RobotLease("test", tmp_path/"robot.lock")
    st.sent = []
    def follow(cmd):
        for slot in range(35):
            st.reader.msg.motor_state[slot].q = cmd.motor_cmd[slot].q
        st.sent.append((time.monotonic(), [cmd.motor_cmd[s].q for s in (22, 23, 24, 25, 26)]))
    st.pub.Write = follow
    assert st.engage() == ""
    hold = threading.Thread(target=st.hold_loop, daemon=True); hold.start()
    yield st
    st.stop.set(); st.motion_cancel.set()
    st.release(seconds=.02)
    hold.join(2)


def start_command(st, frames=None):
    server, client = socket.socketpair()
    def serve():
        try: st.handle(server)
        finally: server.close()
    worker = threading.Thread(target=serve, daemon=True); worker.start()
    frames = frames or interpolate(np.zeros(5), np.full(5, .03), 2, 50)
    request = {"cmd": "frames", "request_id": "motion", "arm": "right",
               "frames": [list(f) for f in frames], "dt": .02}
    client.sendall((json.dumps(request)+"\n").encode())
    deadline = time.monotonic()+2
    while not st.streaming and time.monotonic()<deadline:
        client.sendall(b'{"cmd":"heartbeat"}\n'); time.sleep(.02)
    assert st.streaming
    return client, worker


def test_disconnect_interrupts_active_frames(streamer):
    st = streamer
    client, worker = start_command(st)
    before = len(st.sent)
    client.close()
    worker.join(2)
    assert not worker.is_alive()
    assert not st.streaming and not st.engaged
    # Release may continue publishing the fixed last pose during weight ramp-down.
    after = np.array([q for _, q in st.sent[before+2:]])
    assert len(after) < 50
    if len(after)>1: assert np.max(np.abs(np.diff(after, axis=0))) < .002


def test_freeze_preempts_frames_on_same_socket(streamer):
    st = streamer
    client, worker = start_command(st)
    client.sendall(b'{"cmd":"freeze","request_id":"stop"}\n')
    client.settimeout(2)
    file = client.makefile("rb")
    responses = {}
    while "stop" not in responses or "motion" not in responses:
        response = json.loads(file.readline())
        responses[response["request_id"]] = response
    assert responses["stop"]["ok"]
    assert not responses["motion"]["ok"]
    assert "released during streaming" in responses["motion"]["error"]
    assert not st.streaming
    file.close(); client.close(); worker.join(2)


def test_watchdog_remains_active_during_command(streamer):
    st = streamer
    client, worker = start_command(st)
    # Keep the socket open but stop heartbeats.
    deadline = time.monotonic()+1
    while st.engaged and time.monotonic()<deadline: time.sleep(.02)
    assert not st.engaged
    assert "heartbeat" in st.reason
    client.close(); worker.join(2)


def test_no_unreviewed_lead_in_and_malformed_frames_are_rejected(streamer):
    st = streamer
    # Move the commanded start away from the caller's first target.
    with st.lock:
        for n in am.ARM_JOINTS["right"]:
            st.targets[am.JOINT_TO_SLOT[n]] = .08
    before = len(st.sent)
    result = st.dispatch("frames", {"arm": "right", "frames": [[0.]*5, [.001]*5], "dt": .02})
    assert not result["ok"] and ("velocity" in result["error"] or "acceleration" in result["error"])
    assert not st.streaming
    assert len(st.sent)-before < 5  # only background holding, no generated lead-in
    for dt in (0, -1, float("nan")):
        assert st.stream_frames("right", [[0.]*5], dt)


def test_gain_scale_and_head_pitch(streamer):
    st = streamer
    assert st.slots[am.JOINT_TO_SLOT["right_shoulder_pitch_joint"]] == (75.0, 2.0)
    assert st.slots[am.JOINT_TO_SLOT["left_wrist_roll_joint"]] == (45.0, 2.0)
    assert st.slots[am.WAIST_YAW] == (50.0, 3.0) and st.slots[am.HEAD[0]] == (15.0, 1.0)
    assert st.targets[am.HEAD[0]] == .35 and st.targets[am.HEAD[1]] == 0.0
