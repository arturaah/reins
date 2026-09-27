"""Hardware-free private-channel, review and stop fault injection."""
import json
import socket
import threading
import time
import types
import uuid

import numpy as np
import pytest
from contract.runtime import digest
from core import trajectory
from core.robot_lease import RobotLease
from harness.executor import interpolate
from harness.robot import arm_stream as am


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
    cfg["robot"].update(arm_kp_scale=1.5, head_pitch_rad=.35, weight_ramp_s=.02)
    cfg["streamer"]["watchdog_s"] = .2
    cfg["workspace"]["table_z_m"] = .1
    path = tmp_path/"control.token"; path.write_text("test-controller-capability"*3); path.chmod(0o600)
    cfg["streamer"]["control_token_file"] = str(path)
    st = am.Streamer(cfg, "unused", log=lambda *args: None)
    st.lease = RobotLease("test", tmp_path/"robot.lock")
    st.sent = []
    def follow(cmd):
        for slot in range(35):
            st.reader.msg.motor_state[slot].q = cmd.motor_cmd[slot].q
        st.sent.append((time.monotonic(), [cmd.motor_cmd[s].q for s in (22, 23, 24, 25, 26)]))
    st.pub.Write = follow
    hold = threading.Thread(target=st.hold_loop, daemon=True); hold.start()
    yield st
    st.stop.set(); st.motion_cancel.set()
    st.release(seconds=.02)
    hold.join(2)


def receipt(payload):
    return {"proposal_id": uuid.uuid4().hex, "revision": 1, "digest": digest(payload), "expires_at": time.time()+60}


def arm_payload(st, seconds=2):
    frames = interpolate(np.zeros(5), np.full(5, .03), seconds, 50)
    plan = trajectory.frame_plan("right", np.zeros(5), frames, .02, st.reader.joints())
    return {"kind": "arm", "arm": "right", "plan": plan}


def connect(st, auth=True):
    server, client = socket.socketpair()
    worker = threading.Thread(target=st.handle, args=(server,), daemon=True); worker.start()
    client.settimeout(3)
    file = client.makefile("rb")
    if auth:
        send(client, {"cmd": "authenticate", "token": st.control_token})
        assert json.loads(file.readline())["ok"]
    return client, file, worker


def send(client, req):
    client.sendall((json.dumps(req)+"\n").encode())


def start_command(st, payload=None):
    client, file, worker = connect(st)
    payload = payload or arm_payload(st)
    send(client, {"cmd": "execute_motion", "request_id": "motion", "payload": payload, "approval": receipt(payload)})
    deadline = time.monotonic()+2
    while not (st.streaming or st.walking) and time.monotonic()<deadline:
        send(client, {"cmd": "heartbeat"}); time.sleep(.01)
    assert st.streaming or st.walking
    return client, file, worker


def test_read_only_connect_cannot_publish_or_keep_control_alive(streamer):
    st = streamer
    client, file, worker = connect(st, auth=False)
    send(client, {"cmd": "hello"}); assert json.loads(file.readline())["control_protocol"] == 2
    payload = arm_payload(st)
    for req in ({"cmd": "engage"}, {"cmd": "frames"}, {"cmd": "walk"},
                {"cmd": "execute_motion", "payload": payload, "approval": receipt(payload)}):
        send(client, req); assert not json.loads(file.readline())["ok"]
    assert not st.engaged and st.sent == []
    file.close(); client.close(); worker.join(1)
    assert not st.motion_cancel.is_set()


def test_authenticated_raw_commands_still_cannot_publish(streamer):
    st = streamer
    for cmd in ("engage", "frames", "plan", "walk"):
        assert not st.dispatch(cmd, {}, authorized=True)["ok"]
    assert st.sent == []


def test_disconnect_interrupts_active_frames(streamer):
    st = streamer
    client, file, worker = start_command(st)
    before = len(st.sent)
    file.close(); client.close(); worker.join(2)
    assert not worker.is_alive() and not st.streaming and not st.engaged
    after = np.array([q for _, q in st.sent[before+2:]])
    assert len(after) < 50
    if len(after)>1: assert np.max(np.abs(np.diff(after, axis=0))) < .002


def test_freeze_preempts_frames_on_same_socket(streamer):
    st = streamer
    client, file, worker = start_command(st)
    send(client, {"cmd": "freeze", "request_id": "stop"})
    responses = {}
    while "stop" not in responses or "motion" not in responses:
        response = json.loads(file.readline()); responses[response["request_id"]] = response
    assert responses["stop"]["ok"] and not responses["motion"]["ok"]
    assert "released during streaming" in responses["motion"]["error"]
    assert not st.streaming
    file.close(); client.close(); worker.join(2)


def test_watchdog_remains_active_during_command(streamer):
    st = streamer
    client, file, worker = start_command(st)
    deadline = time.monotonic()+1
    while st.engaged and time.monotonic()<deadline: time.sleep(.02)
    assert not st.engaged and "heartbeat" in st.reason
    file.close(); client.close(); worker.join(2)


def test_bad_review_and_stale_start_publish_nothing(streamer):
    st = streamer; payload = arm_payload(st)
    for approval in (None, {**receipt(payload), "digest": "changed"}, {**receipt(payload), "expires_at": time.time()-1}):
        with pytest.raises(ValueError): st.execute_motion(payload, approval)
    payload["plan"]["keyframes"][0]["joint_targets_rad"][am.ARM_JOINTS["right"][0]] = .1
    with pytest.raises(ValueError): st.execute_motion(payload, receipt(payload))
    assert not st.engaged and st.sent == []


def test_receipt_consumed_before_failure_and_retry(streamer):
    st = streamer; payload = arm_payload(st); review = receipt(payload)
    st.reader.msg.motor_state[22].q = .1
    with pytest.raises(ValueError, match="pose changed"): st.execute_motion(payload, review)
    st.reader.msg.motor_state[22].q = 0
    with pytest.raises(ValueError, match="already been consumed"): st.execute_motion(payload, review)
    assert st.sent == []


def test_no_unreviewed_lead_in_and_malformed_frames(streamer):
    st = streamer
    assert st.engage() == ""
    with st.lock:
        for n in am.ARM_JOINTS["right"]: st.targets[am.JOINT_TO_SLOT[n]] = .08
    payload = arm_payload(st)
    with pytest.raises(ValueError): st.execute_motion(payload, receipt(payload))
    assert not st.streaming
    for dt in (0, -1, float("nan")): assert st.stream_frames("right", [[0.]*5], dt)


def test_gain_scale_and_engage_has_no_hidden_head_motion(streamer):
    st = streamer
    assert st.slots[am.JOINT_TO_SLOT["right_shoulder_pitch_joint"]] == (75.0, 2.0)
    assert st.slots[am.JOINT_TO_SLOT["left_wrist_roll_joint"]] == (45.0, 2.0)
    assert st.slots[am.WAIST_YAW] == (50.0, 3.0) and st.slots[am.HEAD[0]] == (15.0, 1.0)
    assert st.engage() == ""
    assert st.targets[am.HEAD[0]] == 0.0 and st.targets[am.HEAD[1]] == 0.0


@pytest.mark.parametrize("stop_kind", ["freeze", "disconnect", "watchdog"])
def test_walk_is_interruptible_without_engaged_arms(streamer, monkeypatch, stop_kind):
    st = streamer; calls = []
    st.loco.update(enabled=True, settle_s=0)
    class Loco:
        def SetVelocity(self, *args): calls.append("velocity")
        def StopMove(self): calls.append("stop")
    monkeypatch.setattr(am, "_loco", Loco)
    payload = {"kind": "walk", "vx": .2, "vy": 0., "vyaw": 0., "duration_s": 2.}
    client, file, worker = start_command(st, payload)
    began = time.monotonic()
    if stop_kind == "freeze": send(client, {"cmd": "freeze"})
    elif stop_kind == "disconnect": file.close(); client.close()
    while st.walking and time.monotonic()-began < 1: time.sleep(.01)
    assert not st.walking and "stop" in calls and not st.engaged and st.sent == []
    file.close(); client.close(); worker.join(2)


def test_observer_disconnect_does_not_cancel_controller(streamer):
    st = streamer
    client, file, worker = start_command(st)
    observer, stream, observer_worker = connect(st, auth=False)
    stream.close(); observer.close(); observer_worker.join(1)
    assert not st.motion_cancel.is_set() and st.engaged
    send(client, {"cmd": "freeze"})
    file.close(); client.close(); worker.join(2)


def test_missing_table_is_rejected_before_engage(streamer):
    st = streamer; st.cfg["workspace"]["table_z_m"] = None
    payload = arm_payload(st)
    with pytest.raises(ValueError, match="table height"):
        st.execute_motion(payload, receipt(payload))
    assert not st.sent and not st.engaged


def test_walking_budget_is_independent_of_caller(streamer, monkeypatch):
    st = streamer; calls = []
    st.loco.update(enabled=True, settle_s=0, max_total_m=.03)
    class Loco:
        def SetVelocity(self, *args): calls.append("velocity")
        def StopMove(self): calls.append("stop")
    monkeypatch.setattr(am, "_loco", Loco)
    payload = {"kind": "walk", "vx": .2, "vy": 0., "vyaw": 0., "duration_s": .2}
    result = st.execute_motion(payload, receipt(payload))
    assert not result["ok"] and "budget" in result["error"]
    assert not calls


def test_multiplexed_client_stops_without_waiting_for_motion(streamer, monkeypatch):
    from harness.robot.arm_client import ArmClientBackend
    st = streamer
    server, client = socket.socketpair()
    worker = threading.Thread(target=st.handle, args=(server,), daemon=True); worker.start()
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: client)
    backend = ArmClientBackend(st.cfg, log=lambda *a: None)
    assert not st.engaged and not st.sent
    payload = arm_payload(st); failures = []
    def move():
        try: backend.execute_motion(payload, receipt(payload))
        except RuntimeError as exc: failures.append(str(exc))
    mover = threading.Thread(target=move); mover.start()
    deadline = time.monotonic()+2
    while not st.streaming and time.monotonic()<deadline: time.sleep(.01)
    assert st.streaming
    began = time.monotonic(); backend.freeze(); mover.join(1)
    assert time.monotonic()-began < .5 and not mover.is_alive() and failures
    backend.close(); worker.join(2)
