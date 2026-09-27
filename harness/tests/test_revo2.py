"""Revo2 hands: the hand server and the harness client over a real local socket, DDS replaced by a fake that moves the
fingers toward each command and stops them where an object would. Nothing is published."""
import socket
import threading
import time

import numpy as np
import pytest

from harness.executor import ArmExecutor
from harness.interpreter import Proposal
from harness.kinematics import ArmKinematics
from harness.robot.hand_client import Revo2Client
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend

revo2 = pytest.importorskip("harness.robot.revo2")


class FakeDds:
    prefix = "rt/brainco"

    def __init__(self, block_at=1.0, sides=("left", "right")):
        self.q = {s: [0.0] * 6 for s in sides}
        self.block_at, self.sets = block_at, []

    def state(self, side):
        return {"q": list(self.q[side]), "tau": [0.0] * 6, "age": 0.01} if side in self.q else None

    def set(self, side, q, speed):
        self.sets.append((side, list(q)))
        self.q[side] = [min(v, self.block_at) if i >= 2 else v for i, v in enumerate(q)]   # fingers stop at the object


@pytest.fixture
def server(cfg):
    def start(dds):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        cfg["hand"]["type"] = "revo2"
        cfg["hand"]["revo2"].update(port=port, settle_s=0.5)
        threading.Thread(target=revo2.HandServer(dds, cfg, log=lambda *a: None).serve, args=("127.0.0.1", port), daemon=True).start()
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port)).close(); break
            except OSError:
                time.sleep(0.02)
        return cfg
    return start


def test_close_on_nothing_is_an_empty_grasp(server):
    dds = FakeDds(block_at=1.0)
    c = Revo2Client(server(dds), log=lambda *a: None)
    assert c.hand("right", True).startswith("EMPTY")
    assert dds.sets == [("right", [0.98, 0.7, 0.98, 0.98, 0.98, 0.98])]
    assert c.hand_state("right") is True
    assert c.hand("right", False) == "hand opened" and c.hand_state("right") is False


def test_close_on_an_object(server):
    c = Revo2Client(server(FakeDds(block_at=0.5)), log=lambda *a: None)
    fb = c.hand("left", True)
    assert fb.startswith("hand closed on an object") and "51%" in fb


def test_no_state_means_no_command(server):
    dds = FakeDds(sides=("left",))
    c = Revo2Client(server(dds), log=lambda *a: None)
    fb = c.hand("right", True)
    assert "did not move" in fb and "brainco_hand_server" in fb
    assert dds.sets == [] and c.hand_state("right") is None


def test_dry_run_never_sends(server):
    dds = FakeDds()
    c = Revo2Client(server(dds), log=lambda *a: None, dry_run=True)
    assert c.hand("right", True) == "dry run: would close the right hand"
    assert dds.sets == [] and c.hand_state("right") is True


def test_hand_command_is_confirmed_first(cfg):
    cfg["hand"]["type"] = "revo2"                        # the sim mock treats it as the virtual hand
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    asked = []
    ex.confirm = lambda text, preview: asked.append(text) or (False, "not yet")
    r = ex.execute(Proposal(kind="hand", p=np.zeros(3), roll=0.0, hand_closed=True), ex.sync())
    assert r.declined and r.operator_note == "not yet" and "close the right hand" in asked[0]
    assert backend.hand_closed["right"] is False
    ex.confirm = lambda text, preview: True
    r = ex.execute(Proposal(kind="hand", p=np.zeros(3), roll=0.0, hand_closed=True), ex.sync())
    assert r.ok and r.asked and r.hand_closed is True
