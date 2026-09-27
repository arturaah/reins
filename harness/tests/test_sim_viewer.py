from contextlib import nullcontext
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from harness.executor import StreamError
from harness.kinematics import ARM_JOINTS
from harness.sim.mock_robot import MockBackend


class FakeViewer:
    def __init__(self):
        self.cam = SimpleNamespace(lookat=np.zeros(3))
        self.running = True
        self.syncs = 0
    def lock(self): return nullcontext()
    def sync(self): self.syncs += 1
    def is_running(self): return self.running
    def close(self): self.running = False


def test_live_viewer_updates_and_stops_closed_window(cfg, monkeypatch):
    import mujoco.viewer
    viewer = FakeViewer()
    callbacks = []
    def launch(model, data, **kw):
        callbacks.append(kw["key_callback"])
        return viewer
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    backend = MockBackend(cfg, render=False)
    stop = threading.Event()
    backend.open_viewer(stop)
    q = np.array([backend.q[n] for n in ARM_JOINTS["right"]])
    backend.stream("right", [q, q + 0.001], 0.02)
    assert viewer.syncs >= 3 and backend.frames_sent == 2
    callbacks[0](88)
    assert stop.is_set()
    stop.clear()
    viewer.close()
    with pytest.raises(StreamError, match="simulation stopped"):
        backend.stream("right", [q], 0.02)
    assert stop.is_set() and backend.frames_sent == 2
    backend.close_viewer()
    assert backend._viewer is None
