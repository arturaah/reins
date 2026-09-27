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


def test_release_during_frames_is_reported(streamer):
    st = streamer
    st.last_client = time.time(); st.serving = True
    assert st.engage() == ""
    threading.Timer(0.15, lambda: st.release("ABORT: test", 0.05)).start()
    err = st.stream_frames("right", [np.zeros(5)] * 50, 0.02)
    assert err.startswith("released during streaming") and not st.engaged
