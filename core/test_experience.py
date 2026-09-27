"""Historical memory tests use local JPEGs, fake operators and no robot/model calls."""
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np
from PIL import Image

from contract.runtime import digest
from core.experience import ExperienceMemory, HISTORY, verdict


def proposal(outcome="executed", mode="live", task="Blow a kiss", decision="approve"):
    ident = uuid.uuid4().hex
    payload = {"kind": "hand", "arm": "right", "closed": False}
    public = {"id": ident, "name": task, "digest": digest(payload), "mode": mode,
              "session_id": uuid.uuid4().hex, "revision": 1, "expires_at": time.time() + 120,
              "observation_id": "observed-head"}
    result = {"proposal_id": ident, "digest": public["digest"], "mode": mode, "outcome": outcome,
              "decision": {"decision": decision, "note": "Keep it away from the face"} if decision else None,
              "at": time.time(), "message": "Actual runtime result", "measured_end_pose": {"elbow": .5}}
    return public, payload, result


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = {"experience": {"dir": "experience", "max_in_prompt": 3}, "perception": {"pose_view": False}}
        self.memory = ExperienceMemory(self.cfg, self.tmp.name)

    def add(self, *, task="Blow a kiss", **kwargs):
        p, payload, result = proposal(task=task, **kwargs)
        captured = self.memory.capture(task, p, payload, {"elbow": .4}, {"right": [[.2, -.2, .8]]},
                                       {"id": "observed-head", "observed_at": time.time() - 60})
        self.memory.record(captured, result)
        return p, payload, result

    def test_exact_immutable_proposal_and_actual_outcome_survive_restart(self):
        p, payload, result = proposal(outcome="failed")
        captured = self.memory.capture("Blow a kiss", p, payload, {"elbow": .4}, {}, None)
        payload["closed"] = True  # caller changes cannot mutate the captured proposal
        self.memory.record(captured, result)
        self.memory.record(captured, result)
        loaded = ExperienceMemory(self.cfg, self.tmp.name)
        self.assertEqual(loaded.status()["count"], 1)
        row = json.loads((loaded.dir / (p["id"] + ".json")).read_text())
        self.assertFalse(row["payload"]["closed"])
        self.assertEqual(row["proposal"], p)
        self.assertEqual(row["result"], result)
        self.assertEqual(row["start_pose"], {"elbow": .4})
        self.assertEqual(verdict(row), "APPROVED; FAILED; COMPLETION UNCONFIRMED")
        self.assertNotIn("EXECUTION COMPLETED", row["verdict"])
        self.assertEqual((loaded.dir / (p["id"] + ".json")).stat().st_mode & 0o777, 0o600)
        self.assertEqual(loaded.dir.stat().st_mode & 0o777, 0o700)

    def test_verdict_distinguishes_approval_simulation_cancellation_and_decline(self):
        cases = [("executed", "sim", "approve", "SIMULATION COMPLETED"),
                 ("executed", "live", "approve", "EXECUTION COMPLETED"),
                 ("cancelled", "live", "approve", "COMPLETION UNCONFIRMED"),
                 ("declined", "live", "decline", "DECLINED; NOT EXECUTED"),
                 ("expired", "live", None, "NOT APPROVED")]
        for outcome, mode, decision, expected in cases:
            _, _, result = proposal(outcome, mode, decision=decision)
            self.assertIn(expected, verdict({"result": result}))

    def test_wrong_digest_or_unapproved_completion_cannot_enter_memory(self):
        p, payload, result = proposal()
        captured = self.memory.capture("test", p, payload, {}, {}, None)
        result["digest"] = "bad"
        with self.assertRaisesRegex(ValueError, "match"): self.memory.record(captured, result)
        result["digest"] = p["digest"]; result["decision"] = None
        with self.assertRaisesRegex(ValueError, "approval"): self.memory.record(captured, result)
        self.assertEqual(self.memory.status()["count"], 0)

    def test_retrieval_returns_real_picture_cards_and_historical_untrusted_labels(self):
        output = io.BytesIO(); Image.new("RGB", (64, 48), "red").save(output, "JPEG")
        self.memory.observe("observed-head", "head", output.getvalue())
        self.add(task="Move elsewhere", outcome="declined", decision="decline")
        p, _, _ = self.add(task="Blow a kiss", mode="sim")
        result = self.memory.recall("Blow a kiss", "sim")
        self.assertEqual(result["experience"]["selected"][0]["proposal_id"], p["id"])
        self.assertEqual(result["experience"]["selected"][0]["context_camera"], "head")
        text = result["content_blocks"][0]["text"]
        self.assertIn(HISTORY, text)
        self.assertIn("SIMULATION COMPLETED", text)
        image = result["content_blocks"][1]
        decoded = Image.open(io.BytesIO(base64.b64decode(image["data"])))
        self.assertEqual(decoded.width, 800)
        self.assertGreater(decoded.height, 200)
        self.assertNotIn("payload", result["experience"]["selected"][0])
        self.assertNotIn("expires_at", result["experience"]["selected"][0])

    def test_count_limit_prunes_images_and_exact_payloads_together(self):
        self.memory.max_entries = 2
        old, _, _ = self.add(task="Old")
        self.memory.recall("Old", "live")
        self.add(task="Middle"); self.add(task="New")
        self.assertEqual(self.memory.status()["count"], 2)
        self.assertEqual(list(self.memory.dir.glob(old["id"] + ".*")), [])
        self.assertEqual(len(self.memory.recall("", "live")["experience"]["selected"]), 2)

    def test_disk_byte_limit_and_disabled_mode(self):
        self.memory.max_bytes = 1024
        p, payload, result = proposal(task="x" * 3000)
        with self.assertRaisesRegex(ValueError, "storage limit"):
            self.memory.record(self.memory.capture(p["name"], p, payload, {}, {}, None), result)
        self.memory.enabled = False
        self.assertIsNone(self.memory.capture("unused", p, payload, {}, {}, None))
        self.memory.record(None, result)
        self.assertFalse(self.memory.recall("", "sim")["experience"]["enabled"])
        self.assertFalse(self.memory.dir.exists())

    def test_multiple_process_instances_do_not_lose_records(self):
        other = ExperienceMemory(self.cfg, self.tmp.name)
        def write(memory):
            p, payload, result = proposal()
            memory.record(memory.capture(p["name"], p, payload, {}, {}, None), result)
        threads = [threading.Thread(target=write, args=(self.memory if i % 2 else other,)) for i in range(8)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(5)
        self.assertEqual(self.memory.status()["count"], 8)

    def test_forget_removes_images_and_memory_but_leaves_legacy_history(self):
        self.add()
        self.memory.recall("", "live")
        legacy = self.memory.dir.parent / "index.jsonl"; legacy.write_text("historical offline data")
        result = self.memory.clear()
        self.assertEqual(result["count"], 0)
        self.assertEqual(legacy.read_text(), "historical offline data")
        self.assertEqual(sorted(p.name for p in self.memory.dir.iterdir()), [".lock", "index.json"])
        self.assertEqual(self.memory.recall("", "live")["content_blocks"], [])

    def test_corrupt_or_missing_memory_does_not_block_context(self):
        self.add()
        self.memory.index.write_text("corrupt")
        result = self.memory.recall("", "live")
        self.assertTrue(result["experience"]["error"])
        self.assertEqual(result["experience"]["selected"], [])
        self.assertEqual(self.memory.clear()["count"], 0)

    def test_corrupt_index_metadata_is_skipped_before_relevance_sorting(self):
        self.add()
        rows = json.loads(self.memory.index.read_text())
        rows[0]["task"] = {"unexpected": "object"}
        self.memory.index.write_text(json.dumps(rows))
        result = self.memory.recall("Blow a kiss", "live")
        self.assertEqual(result["experience"]["selected"], [])
        self.assertTrue(result["experience"]["error"])


class ExperiencePipelineTests(unittest.TestCase):
    def test_observe_review_decline_then_retrieve_without_actuation(self):
        from core.prompt_planner import PromptPlanner
        from core.reins_tools import ReinsTools
        from core.robot_pipeline import RobotPipeline
        from core.test_generated_motion import DRAFT
        from harness.config import load
        from tools.dashboard import Simulation
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load(overrides={"perception.pose_view": False})
            planner, sim = PromptPlanner(output_dir=tmp), Simulation()
            pipe = RobotPipeline(planner, sim, {}, cfg=cfg, run_dir=tmp, simulation_only=True)
            try:
                detector = Mock(available=False); detector.name = "test"
                tools = ReinsTools(detector, {"head": lambda: (np.zeros((48, 64, 3), np.uint8), time.monotonic(), "frame-1", None)},
                                   planner, sim.show_proposal, sim.status)
                tools.pipeline = pipe; tools.begin_turn("Blow a kiss safely")
                observation = tools.call("observe", {})["observation"]
                draft = pipe.compile_hand_path(DRAFT, observation["id"])
                p = tools.call("propose_motion", {"plan_id": draft["id"], "request_id": "review-1"})["proposal"]
                self.assertEqual(pipe.experience.status()["count"], 0)
                pipe.decide(p["id"], p["digest"], "decline", "Keep hand farther away")
                self.assertEqual(pipe.status()["last_result"]["outcome"], "declined")
                remembered = tools.call("get_robot_context", {})
                card = remembered["experience"]["selected"][0]
                self.assertEqual(card["task"], "Blow a kiss safely")
                self.assertEqual(card["outcome"], "declined")
                self.assertEqual(card["observation"]["cameras"]["head"]["frame_id"], "frame-1")
                self.assertEqual(remembered["content_blocks"][1]["type"], "image")
                self.assertIsNone(pipe.proposal)
                self.assertFalse(pipe.connected)
            finally:
                pipe.close()

    def test_memory_failure_cannot_change_terminal_motion_result(self):
        from core.prompt_planner import PromptPlanner
        from core.robot_pipeline import RobotPipeline
        from harness.config import load
        from tools.dashboard import Simulation
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load(overrides={"hand.type": "virtual", "perception.pose_view": False})
            pipe = RobotPipeline(PromptPlanner(output_dir=tmp), Simulation(), {}, cfg=cfg, run_dir=tmp, simulation_only=True)
            try:
                draft = pipe.prepare_hand("right", False)
                p = pipe.propose_motion(draft["id"])["proposal"]
                with patch.object(pipe.experience, "record", side_effect=OSError("disk full")):
                    pipe.decide(p["id"], p["digest"], "decline")
                self.assertEqual(pipe.motion_result(p["id"])["outcome"], "declined")
                self.assertEqual(pipe.experience.status()["error"], "disk full")
                self.assertIsNone(pipe.proposal)
            finally:
                pipe.close()
