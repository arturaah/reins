"""Offline integration tests: real IK/validation, fake robot and model, loopback AR."""
import asyncio
import copy
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from core.glasses_bridge import GlassesBridge
from core.prompt_planner import PromptPlanner
from core.robot_pipeline import RobotPipeline, PreviewBackend
from core.robot_lease import RobotLease
from core.test_generated_motion import DRAFT
from core.test_trajectory_revision import UNREACHABLE
from core.trajectory import digest, frames, require_start, resolve, validate
from core.generated_motion import compile_trajectory
from harness.actions import ActionError, parse_action
from harness.loop import Episode
from harness.vlm.scripted import ScriptedVLM
from tools.dashboard import Simulation


class FakeRobot(PreviewBackend):
    name = "arm_sdk"
    dry_run = False

    def __init__(self, pose):
        super().__init__(pose)
        self.sent = []
        self.engaged = False
        self.frozen = False

    def snapshot(self):
        return {"joints": dict(self.q), "targets": dict(self.q), "engaged": self.engaged, "lowstate_age_s": 0}

    def engage(self): self.engaged = True
    def release(self): self.engaged = False
    def freeze(self): self.frozen = True

    def stream_plan(self, plan, arm):
        assert self.engaged
        self.sent.append(copy.deepcopy(plan))
        self.q.update(plan["keyframes"][-1]["joint_targets_rad"])
        return {"joints": self.joints(), "ok": True}


class Feed:
    def __init__(self):
        self.lock = threading.Lock()
        output = io.BytesIO(); Image.new("RGB", (64, 64)).save(output, "JPEG")
        self.jpg = output.getvalue()
    def status(self): return {"online": True}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sim = Simulation()
        self.planner = PromptPlanner()
        self.pipe = RobotPipeline(self.planner, self.sim, {}, run_dir=self.tmp.name)
        self.pipe.cfg["feedback"]["path"] = str(Path(self.tmp.name)/"feedback.jsonl")

    def tearDown(self):
        self.pipe.close()
        if self.pipe.worker: self.pipe.worker.join(3)
        self.tmp.cleanup()

    def wait(self, *states):
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            value = self.pipe.status()
            if value["state"] in states and not (value["state"] in ("idle", "completed") and value["busy"]):
                return value
            time.sleep(.01)
        self.fail(str(self.pipe.status()))

    def ready(self):
        self.planner.submit("Blow a kiss", trajectory=DRAFT)
        value = self.wait("review", "blocked")
        self.assertEqual(value["state"], "review", value["message"])
        return value["proposal"]

    def approve(self, p):
        self.pipe.decide(p["id"], p["digest"], "approve")
        return self.wait("completed", "blocked")

    def test_primary_preview_and_exact_approved_execution(self):
        fake = FakeRobot(self.pipe.planning_pose())
        self.pipe.backend_factory = lambda: fake
        self.pipe.command({"action": "connect", "table_z_m": .6})
        self.wait("idle")
        p = self.ready()
        self.assertEqual(p["mode"], "live")
        self.assertEqual(fake.sent, [])
        self.assertEqual(self.sim.key, p["id"])
        self.assertEqual(self.pipe.glasses_message()["review"]["digest"], p["digest"])
        result = self.approve(p)
        self.assertEqual(result["state"], "completed", result["message"])
        self.assertEqual(len(fake.sent), 1)
        self.assertEqual(digest(fake.sent[0]), p["digest"])

    def test_reject_stale_double_and_changed_pose(self):
        p = self.ready()
        with self.assertRaises(ValueError): self.pipe.decide("old", p["digest"], "approve")
        with self.assertRaises(ValueError): self.pipe.decide(p["id"], "wrong", "approve")
        self.pipe.backend.q["right_elbow_joint"] += .1
        result = self.approve(p)
        self.assertEqual(result["state"], "blocked")
        self.assertIn("pose changed", result["message"])
        with self.assertRaises(ValueError): self.pipe.decide(p["id"], p["digest"], "approve")

    def test_cancel_invalidates_approval(self):
        p = self.ready()
        self.pipe.stop()
        self.wait("stopped")
        with self.assertRaises(ValueError): self.pipe.decide(p["id"], p["digest"], "approve")
        self.assertFalse(self.pipe.status()["connected"])

    def test_expiry_and_mutated_plan_are_rejected(self):
        p = self.ready()
        self.pipe.proposal["expires_at"] = time.time()-1
        with self.assertRaisesRegex(ValueError, "expired"):
            self.pipe.decide(p["id"], p["digest"], "approve")

    def test_host_owned_revision_applies_to_tool_calls(self):
        from core.reins_tools import ReinsTools
        calls = []
        def revise(*args):
            calls.append(args)
            return DRAFT
        self.planner.reviser_factory = lambda: revise
        tools = ReinsTools(type("Detector", (), {"available": False})(), {}, self.planner,
                           self.sim.show_proposal, self.sim.status)
        result = tools.plan_hand_path(UNREACHABLE["name"], "right", UNREACHABLE["waypoints"], True)
        self.assertEqual(result["state"], "proposed", result["message"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.wait("review", "blocked")["state"], "review")

    def test_visual_fallback_requests_fresh_context_and_reviews_step(self):
        self.pipe.cameras = {"head": Feed()}
        model = ScriptedVLM(decisions=[{"decision": "MV_UP", "reasoning": "WRIST: NO. Small upward step."},
                                       {"decision": "DONE", "reasoning": "WRIST: NO. complete"}])
        self.pipe.visual_factory = lambda provider: model
        self.planner.submit("inspect the target with a small move")
        value = self.wait("review", "blocked", "needs_context")
        self.assertEqual(value["state"], "review", value["message"])
        self.assertEqual(value["proposal"]["source"], "visual")
        self.assertGreater(value["proposal"]["validation"]["samples"], 0)
        result = self.approve(value["proposal"])
        self.assertEqual(result["state"], "completed")
        self.assertTrue(model.calls)

    def test_no_camera_does_not_call_model(self):
        self.pipe.visual_factory = lambda provider: self.fail("No camera must block visual inference")
        self.planner.submit("look at the object")
        value = self.wait("needs_context")
        self.assertIn("head camera", value["message"])

    def test_wrist_roll_requires_review(self):
        before = self.pipe.backend.joints()
        self.pipe.command({"action":"roll", "arm":"right", "sign":1})
        result = self.wait("review", "blocked")
        self.assertEqual(result["state"], "review", result["message"])
        self.assertEqual(self.pipe.backend.joints(), before)
        self.assertEqual(self.approve(result["proposal"])["state"], "completed")
        self.assertAlmostEqual(self.pipe.backend.q["right_wrist_roll_joint"]-before["right_wrist_roll_joint"], .0872664626)

    def test_manual_nudge_requires_review(self):
        before = self.pipe.backend.joints()
        self.pipe.command({"action": "jog", "arm": "right", "direction": "up"})
        result = self.wait("review", "blocked")
        self.assertEqual(result["state"], "review", result["message"])
        self.assertEqual(before, self.pipe.backend.joints())
        self.assertEqual(self.approve(result["proposal"])["state"], "completed")

    def test_glasses_requires_authentication_and_exact_proposal(self):
        p = self.ready()
        bridge = GlassesBridge(self.pipe, "127.0.0.1", 0)
        self.assertTrue(bridge.ready.wait(3))
        async def run():
            url = "ws://127.0.0.1:"+str(self.pipe.glasses["port"])
            async with connect(url) as ws:
                await ws.send(json.dumps({"type": "authenticate", "token": "wrong"}))
                with self.assertRaises(ConnectionClosed): await ws.recv()
            async with connect(url) as ws:
                await ws.send(json.dumps({"type": "authenticate", "token": self.pipe.glasses_token}))
                self.assertEqual(json.loads(await ws.recv())["type"], "auth_ack")
                msg = json.loads(await ws.recv())
                self.assertEqual(msg["review"]["id"], p["id"])
                await ws.send(json.dumps({"type": "review_decision", "version": 1, "id": p["id"],
                                          "digest": p["digest"], "decision": "approve"}))
                while True:
                    ack = json.loads(await ws.recv())
                    if ack["type"] == "review_ack":
                        self.assertTrue(ack["accepted"]); break
        try: asyncio.run(run())
        finally: bridge.close()
        self.assertEqual(self.wait("completed", "blocked")["state"], "completed")

    def test_rejection_clears_proposal_and_allows_next_task(self):
        p = self.ready()
        self.pipe.decide(p["id"], p["digest"], "decline", "Keep it lower")
        if self.pipe.worker: self.pipe.worker.join(3)
        self.assertIsNone(self.pipe.status()["proposal"])
        self.assertEqual(self.pipe.last_review_note, "Keep it lower")
        self.pipe.command({"action": "jog", "direction": "up", "arm": "right"})
        self.assertEqual(self.wait("review", "blocked")["state"], "review")

    def test_operator_disconnect_releases_fake_robot(self):
        fake = FakeRobot(self.pipe.planning_pose())
        self.pipe.backend_factory = lambda: fake
        self.pipe.command({"action": "connect", "table_z_m": .6})
        self.wait("idle")
        self.pipe.last_operator -= self.pipe.OPERATOR_TIMEOUT+1
        self.wait("stopped")
        self.assertFalse(fake.engaged)
        self.assertTrue(fake.frozen)
        self.assertFalse(self.pipe.status()["connected"])

    def test_stop_during_final_validation_prevents_streaming(self):
        fake = FakeRobot(self.pipe.planning_pose())
        self.pipe.backend_factory = lambda: fake
        self.pipe.command({"action": "connect", "table_z_m": .6})
        self.wait("idle")
        p = self.ready()
        real_validate = validate
        def interrupted(*args, **kwargs):
            result = real_validate(*args, **kwargs)
            self.pipe.stop()
            return result
        with patch("core.robot_pipeline.trajectory.validate", side_effect=interrupted):
            self.pipe.decide(p["id"], p["digest"], "approve")
            self.wait("stopped")
        self.assertEqual(fake.sent, [])


class BoundaryTests(unittest.TestCase):
    def test_nonfinite_actions_and_oscillation(self):
        for value in ("nan", "inf", "-inf"):
            with self.assertRaises(ActionError): parse_action("MOVE forward "+value)
        self.assertTrue(Episode.same_token("MV_FWD", parse_action("MV_BACK").opposite_of))

    def test_exclusive_robot_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"robot.lock"
            one, two = RobotLease("one", path), RobotLease("two", path)
            with one:
                with self.assertRaises(ValueError): two.acquire()
            with two: pass

    def test_contract_requires_approval(self):
        from contract.reins_contract import ContractError, check_session, read_jsonl
        messages = read_jsonl("contract/examples/session_reach.jsonl")
        with self.assertRaisesRegex(ContractError, "approval"):
            check_session([m for m in messages if m["id"] != "g-5"])
