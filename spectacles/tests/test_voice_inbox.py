import asyncio
import json
import socket
import tempfile
import unittest
from pathlib import Path

import mujoco
from websockets.asyncio.client import connect

from spectacles.plan_feed import Feed, MJCF, serve_feed
from spectacles.voice_inbox import VoiceInbox


class VoiceInboxTest(unittest.TestCase):
    def test_queues_once_and_normalizes_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            inbox = VoiceInbox(Path(tmp) / "runs" / "voice.json")
            self.assertTrue(inbox.enqueue("command-1", " walk  one metre\nforward "))
            self.assertFalse(inbox.enqueue("command-2", "turn around"))
            self.assertEqual(inbox.take(), {"id": "command-1", "text": "walk one metre forward"})
            self.assertIsNone(inbox.take())
            self.assertTrue(inbox.enqueue("command-2", "turn around"))

    def test_rejects_empty_and_oversized_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            inbox = VoiceInbox(Path(tmp) / "voice.json")
            self.assertFalse(inbox.enqueue("id", "   "))
            self.assertFalse(inbox.enqueue("id", "x" * 501))
            self.assertFalse(inbox.enqueue("", "move"))
            self.assertIsNone(inbox.take())


class VoiceWebSocketTest(unittest.IsolatedAsyncioTestCase):
    async def test_command_ack_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "preview.json"
            plan.write_text(json.dumps({"schema_version": 1, "keyframes": [
                {"time_s": 0, "joint_targets_rad": {"right_shoulder_pitch_joint": 0}},
                {"time_s": 1, "joint_targets_rad": {"right_shoulder_pitch_joint": 0.2}},
            ]}))
            inbox = VoiceInbox(Path(tmp) / "voice.json")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            feed = Feed(mujoco.MjModel.from_xml_path(str(MJCF)), plan, 20)
            server = asyncio.create_task(serve_feed(feed, "127.0.0.1", port, 0.05, voice=inbox, review_token="test-voice-token"))
            try:
                for _ in range(50):
                    try:
                        client = await connect(f"ws://127.0.0.1:{port}")
                        break
                    except OSError:
                        await asyncio.sleep(0.02)
                else:
                    self.fail("feed did not start")
                async with client as websocket:
                    await websocket.send(json.dumps({"type":"authenticate", "token":"test-voice-token"}))
                    auth = json.loads(await websocket.recv())
                    self.assertEqual(auth["type"], "auth_ack")
                    for command_id, expected in (("one", True), ("two", False)):
                        await websocket.send(json.dumps({"type": "voice_command", "version": 1,
                                                         "id": command_id, "text": "move right", "session":auth["session"]}))
                        while True:
                            reply = json.loads(await asyncio.wait_for(websocket.recv(), 2))
                            if reply.get("type") == "voice_ack":
                                break
                        self.assertEqual(reply["accepted"], expected)
                    self.assertEqual(inbox.take(), {"id": "one", "text": "move right"})
            finally:
                server.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await server
