"""Revo2 hands: the hand server and the harness client over a real local socket, DDS replaced by a fake that moves the
fingers toward each command and stops them where an object would. Nothing is published."""
import json
import socket
import uuid
import threading
import time

import numpy as np
import pytest

from contract.runtime import digest
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
def server(cfg, tmp_path):
    token = tmp_path/"hand.token"; token.write_text("private-hand-controller-capability"*2); token.chmod(0o600)
    cfg["streamer"]["control_token_file"] = str(token)
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


def approved_hand(client, arm, closed):
    payload = {"kind": "hand", "arm": arm, "closed": closed}
    approval = {"proposal_id": uuid.uuid4().hex, "revision": 1, "digest": digest(payload), "expires_at": time.time()+60}
    return client.execute_motion(payload, approval)["hand_feedback"]


def test_close_on_nothing_is_an_empty_grasp(server):
    dds = FakeDds(block_at=1.0)
    c = Revo2Client(server(dds), log=lambda *a: None)
    assert approved_hand(c, "right", True).startswith("EMPTY")
    assert dds.sets == [("right", [0.98, 0.7, 0.98, 0.98, 0.98, 0.98])]
    assert c.hand_state("right") is True
    assert approved_hand(c, "right", False) == "hand opened" and c.hand_state("right") is False


def test_close_on_an_object(server):
    c = Revo2Client(server(FakeDds(block_at=0.5)), log=lambda *a: None)
    fb = approved_hand(c, "left", True)
    assert fb.startswith("hand closed on an object") and "51%" in fb


def test_no_state_means_no_command(server):
    dds = FakeDds(sides=("left",))
    c = Revo2Client(server(dds), log=lambda *a: None)
    with pytest.raises(RuntimeError, match="brainco_hand_server"):
        approved_hand(c, "right", True)
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


def test_hand_words_parse_as_grasp_and_release():
    from harness.actions import parse_action
    assert [parse_action(t).name for t in ("GRAB", "close", "OPEN", "let_go")] == ["GRASP", "GRASP", "RELEASE", "RELEASE"]


def test_hand_state_is_measured_until_the_first_command(server):
    dds = FakeDds(); dds.q["right"] = [0.9] * 6                  # starts closed
    c = Revo2Client(server(dds), log=lambda *a: None)
    assert c.hand_state("right") is True and c.hand_state("left") is False


def test_sim_pick_and_place_with_the_revo2_prompts(cfg, tmp_path):
    """The whole loop with hand.type revo2 (the mock grasps like the virtual hand): the model is told about the
    five-finger hand, sees each hand command's result, and the episode succeeds through the empty-grasp recovery."""
    from harness.loop import Episode
    from harness.perception import MockCameras, Perception
    from harness.recorder import Recorder
    from harness.tests.test_loop_sim import PLAN, Oracle
    from harness.vlm.scripted import ScriptedVLM
    cfg["hand"]["type"] = "revo2"; cfg["steps"]["profile"] = "coarse_fine"; cfg["recorder"]["root"] = str(tmp_path)
    kin = ArmKinematics(cfg["robot"]["model"], "right")
    backend = MockBackend(cfg, render=False)
    ex = ArmExecutor(cfg, kin, SafetyGate(cfg, kin, None, live=False), backend, "right")
    vlm = ScriptedVLM(plan=PLAN, on_act=Oracle(backend))
    summary = Episode(cfg, vlm, ex, Perception(cfg, "right", MockCameras(backend, "right", 320, 180)),
                      Recorder(cfg, "sim", "pick"), log=lambda *a: None).run("pick up the block and place it on the plate")
    assert summary["success"], summary
    plans = [c[1] for c in vlm.calls if c[0] == "plan"]
    acts = [c[1] for c in vlm.calls if c[0] == "act"]
    assert "BrainCo Revo2" in plans[0] and "power grasp" in plans[0]
    assert all("HAND (five fingers" in p and "GRASP, RELEASE" in p for p in acts)
    assert any("Last hand command: EMPTY grasp" in p for p in acts)
    assert any("Last hand command: hand closed on the object" in p and "Holding an object" in p for p in acts)


def test_public_state_but_no_raw_hand_actuation(server):
    dds = FakeDds(); cfg = server(dds)
    public_cfg = {**cfg, "streamer": {k: v for k, v in cfg["streamer"].items() if k != "control_token_file"}}
    client = Revo2Client(public_cfg, log=lambda *a: None)
    assert client.read("right") is not None
    assert not client.call({"cmd": "set", "side": "right", "q": [.9]*6})["ok"]
    with pytest.raises(RuntimeError, match="Unreviewed"):
        client.hand("right", True)
    assert not dds.sets
    client.close()


def test_hand_digest_expiry_and_duplicate_are_rejected(server):
    dds = FakeDds(); client = Revo2Client(server(dds), log=lambda *a: None)
    payload = {"kind": "hand", "arm": "right", "closed": True}
    approval = {"proposal_id": uuid.uuid4().hex, "revision": 1, "digest": digest(payload), "expires_at": time.time()+60}
    for bad in ({**approval, "digest": "wrong"}, {**approval, "expires_at": 1}):
        with pytest.raises(ValueError): client.execute_motion(payload, bad)
    assert not dds.sets
    client.execute_motion(payload, approval)
    count = len(dds.sets)
    with pytest.raises(RuntimeError, match="consumed"):
        client.execute_motion(payload, approval)
    assert len(dds.sets) == count
    client.close()


def test_hand_disconnect_holds_measured_fingers(server):
    dds = FakeDds(block_at=.5); client = Revo2Client(server(dds), log=lambda *a: None)
    approved_hand(client, "right", True)
    client.close()
    for _ in range(50):
        if len(dds.sets)>1: break
        time.sleep(.01)
    assert dds.sets[-1][1][2:] == [.5]*4
    assert len(dds.sets) == 2


def test_hand_nonfinite_configuration_never_publishes(server):
    dds = FakeDds(); cfg = server(dds); cfg["hand"]["revo2"]["close"][2] = float("nan")
    client = Revo2Client(cfg, log=lambda *a: None)
    with pytest.raises(RuntimeError, match="finite"):
        approved_hand(client, "right", True)
    assert not dds.sets
    client.close()
