"""The streamer's client watchdog must not fire while the streamer itself is busy serving a command (the engage ramp, a
frames command): no client bytes can arrive then, because the handle thread is inside dispatch and the client's
heartbeat waits behind its call lock. This is what ended the first live attempt on 2026-09-27 (ABORT: client heartbeat
lost, during engage). DDS is replaced by fakes; nothing is published."""
import threading
import time
import types

import numpy as np
import pytest

pytest.importorskip("unitree_sdk2py")
from harness.robot import arm_stream as am


class FakeReader:
    def __init__(self, iface=None):
        self.msg = types.SimpleNamespace(motor_state=[types.SimpleNamespace(q=0.0, dq=0.0) for _ in range(35)])
    def wait(self): return True
    def age(self): return 0.0
    def joints(self): return {n: 0.0 for n in am.JOINT_TO_SLOT}
    def velocities(self): return {n: 0.0 for n in am.JOINT_TO_SLOT}
    def close(self): pass


class FakePub:
    def __init__(self, *a): self.n = 0
    def Init(self): pass
    def Write(self, cmd): self.n += 1


@pytest.fixture
def streamer(cfg, monkeypatch):
    monkeypatch.setattr(am, "LowStateReader", FakeReader)
    monkeypatch.setattr(am, "ChannelPublisher", FakePub)
    monkeypatch.setattr(am, "query_fsm", lambda: (811, "Start (balance control)"))
    cfg["robot"]["weight_ramp_s"] = 0.6; cfg["streamer"]["watchdog_s"] = 0.2
    st = am.Streamer(cfg, "lo0", log=lambda *a: None)
    st.sent = []
    def follow(cmd):                                    # the fake robot tracks every command perfectly and keeps a log
        for s in range(35): st.reader.msg.motor_state[s].q = cmd.motor_cmd[s].q
        st.sent.append([cmd.motor_cmd[s].q for s in (22, 23, 24, 25, 26)])
    st.pub.Write = follow
    th = threading.Thread(target=st.hold_loop, daemon=True); th.start()
    yield st
    st.stop.set(); th.join(timeout=2)


def test_watchdog_waits_while_a_command_is_served(streamer):
    st = streamer
    st.last_client = time.time()
    st.serving = True                                   # as handle() does around dispatch
    assert st.engage() == ""                            # the ramp takes 3x the watchdog
    assert st.engaged and st.weight == 1.0
    frames = [np.zeros(5) + 0.001 * i for i in range(30)]   # 0.6 s of frames, 0.05 rad/s
    assert st.stream_frames("right", frames, 0.02) == ""
    assert st.engaged and st.reason == ""
    st.serving = False; st.last_client = time.time()  # dispatch finished: the client now has to heartbeat again
    t0 = time.time()
    while st.engaged and time.time() - t0 < 3.0:      # a real silence still releases (after its 1 s ramp down)
        time.sleep(0.05)
    assert not st.engaged and st.reason.startswith("ABORT: client heartbeat lost") and time.time() - t0 < 2.0


def test_refusal_is_reported_and_droop_gets_a_lead_in(streamer):
    """2026-09-27 11:32 live: the first frame sat 0.008 rad from the hold target (measured vs commanded), so frame 0 was over
    the cap and the move was refused, while dispatch reported ok because state() carries its own ok."""
    st = streamer
    st.last_client = time.time(); st.serving = True
    assert st.engage() == ""
    slots = [am.JOINT_TO_SLOT[n] for n in am.ARM_JOINTS["right"]]
    with st.lock:
        for s in slots: st.targets[s] = 0.008                                   # the hold target, a droop away from the measured 0
        st.targets[slots[3]] = 0.03                                              # the elbow droops more: 1.5 rad/s if jumped in one frame
    from harness.executor import interpolate
    frames = interpolate(np.zeros(5), np.full(5, 0.84), 1.65, 50.0)             # what the executor sends for a 48 deg move at the cap
    st.sent.clear()
    r = st.dispatch("frames", {"arm": "right", "frames": [list(f) for f in frames], "dt": 0.02})
    assert r["ok"] and r["error"] == "" and r["engaged"], r
    q = np.array(st.sent)
    assert len(q) > len(frames) and abs(q[-1, 0] - 0.84) < 1e-9                  # lead-in frames, then the move to its end
    assert np.abs(np.diff(q, axis=0)).max() / 0.02 <= st.vmax * 1.05             # never over the cap, lead-in included
    assert 0.0 < q[0, 3] < 0.03 and abs(q[0, 3] - 0.03) <= st.vmax * 0.8 * 0.02 + 1e-9   # the elbow leaves its hold target gradually
    too_fast = [np.zeros(5), np.full(5, 0.5)]
    r = st.dispatch("frames", {"arm": "right", "frames": [list(f) for f in too_fast], "dt": 0.02})
    assert r["ok"] is False and "over the 0.8 rad/s cap" in r["error"]           # a real violation is still refused, and said so
    r = st.dispatch("engage", {})
    assert r["ok"] is True and r["engaged"]


def test_release_during_frames_is_reported(streamer):
    st = streamer
    st.last_client = time.time(); st.serving = True
    assert st.engage() == ""
    threading.Timer(0.15, lambda: st.release("ABORT: test", 0.05)).start()
    err = st.stream_frames("right", [np.zeros(5)] * 50, 0.02)
    assert err.startswith("released during streaming") and not st.engaged
