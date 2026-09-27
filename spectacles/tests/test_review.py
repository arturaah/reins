"""A Spectacles decision must apply only to the exact pending proposal."""
import asyncio
import json
import queue
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness.__main__ import make_confirm, stdin_lines
from spectacles.review import ReviewMailbox
from spectacles.plan_feed import Feed, MJCF, RelayState, serve_feed
import mujoco
from websockets.asyncio.client import connect


class ReviewTests(unittest.TestCase):
    def setUp(self):
        while True:
            try:
                stdin_lines.get_nowait()
            except queue.Empty:
                break

    def test_plan_hash_and_proposal_id_gate_decision(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / "preview.json"
            plan.write_text('{"keyframes":1}')
            mailbox = ReviewMailbox(Path(folder) / "review.json")
            old = mailbox.propose(plan, "first", "live")
            self.assertFalse(mailbox.decide(plan, "wrong", "approve"))
            plan.write_text('{"keyframes":2}')
            self.assertIsNone(mailbox.pending(plan))
            self.assertFalse(mailbox.decide(plan, old["id"], "approve"))
            current = mailbox.propose(plan, "second", "live")
            self.assertFalse(mailbox.decide(plan, old["id"], "approve"))
            self.assertTrue(mailbox.decide(plan, current["id"], "decline"))
            plan.write_text('{"keyframes":3}')
            self.assertIsNone(mailbox.take(current["id"], plan))
            plan.write_text('{"keyframes":2}')
            self.assertEqual(mailbox.take(current["id"], plan), "decline")
            mailbox.clear(current["id"])
            self.assertIsNone(mailbox.pending(plan))

    def test_harness_gate_accepts_spectacles_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            preview = Path(folder) / "preview.json"
            review = Path(folder) / "review.json"
            result = []
            confirm = make_confirm(None, None, preview, review, "dry-run")
            sample = {"arm": "right", "q_now": [0] * 5,
                      "frames": [[0.1] * 5], "dt": 0.1,
                      "joints": {}}
            thread = threading.Thread(target=lambda: result.append(confirm("move right", sample)))
            thread.start()
            mailbox = ReviewMailbox(review)
            deadline = time.time() + 3
            while not mailbox.pending(preview) and time.time() < deadline:
                time.sleep(0.01)
            proposal = mailbox.pending(preview)
            self.assertIsNotNone(proposal)
            self.assertEqual(proposal["mode"], "dry-run")
            self.assertTrue(mailbox.decide(preview, proposal["id"], "approve"))
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [True])
            self.assertFalse(review.exists())


class WebSocketReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_decision_roundtrip_rejects_wrong_id(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / "preview.json"
            plan.write_text(json.dumps({"schema_version": 1, "keyframes": [
                {"time_s": 0, "joint_targets_rad": {"right_shoulder_pitch_joint": 0}},
                {"time_s": 1, "joint_targets_rad": {"right_shoulder_pitch_joint": 0.2}},
            ]}))
            mailbox = ReviewMailbox(Path(folder) / "review.json")
            proposal = mailbox.propose(plan, "move right", "dry-run")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            feed = Feed(mujoco.MjModel.from_xml_path(str(MJCF)), plan, 20)
            state = RelayState()
            state.update({"type": "r1_state", "version": 1,
                          "q": {"right_shoulder_pitch_joint": 0.2},
                          "cmd": {"weight": 1}})
            server = asyncio.create_task(serve_feed(feed, "127.0.0.1", port, 0.05,
                                                    state=state, review=mailbox))
            try:
                for attempt in range(50):
                    try:
                        client = await connect(f"ws://127.0.0.1:{port}")
                        break
                    except OSError:
                        await asyncio.sleep(0.02)
                else:
                    self.fail("feed did not start")
                async with client as websocket:
                    first = json.loads(await asyncio.wait_for(websocket.recv(), 2))
                    self.assertEqual(first["review"]["id"], proposal["id"])
                    self.assertEqual(len(first["hands"]["right"]), 20)
                    for proposal_id, accepted in (("wrong", False), (proposal["id"], True)):
                        await websocket.send(json.dumps({"type": "review_decision", "version": 1,
                                                         "id": proposal_id, "decision": "decline"}))
                        while True:
                            reply = json.loads(await asyncio.wait_for(websocket.recv(), 2))
                            if reply.get("type") == "review_ack":
                                break
                        self.assertEqual(reply["accepted"], accepted)
                    self.assertEqual(mailbox.take(proposal["id"], plan), "decline")
            finally:
                server.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await server


if __name__ == "__main__":
    unittest.main()
