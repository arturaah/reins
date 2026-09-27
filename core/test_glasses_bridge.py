"""No robot/model: persistent pairing and fault-injected loopback AR protocol."""
import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from core.glasses_bridge import GlassesBridge, require_tracking
from core.glasses_pairing import PairingStore


class PairingTests(unittest.TestCase):
    def test_restart_revocation_and_no_plaintext_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"pairing.json"
            store = PairingStore(path)
            self.assertEqual(store.list_devices(), [])
            self.assertFalse(path.exists())
            device = store.create_device("Lab glasses")
            other = PairingStore(path)
            self.assertTrue(other.authenticate(device["device_id"], device["token"]))
            self.assertNotIn(device["token"], path.read_text())
            self.assertNotIn("token", json.dumps(store.list_devices()))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertIsNone(other.authenticate(device["device_id"], "wrong"))
            self.assertTrue(other.revoke_device(device["device_id"]))
            self.assertIsNone(store.authenticate(device["device_id"], device["token"]))
            self.assertIsNotNone(store.list_devices()[0]["revoked_at"])

    def test_corrupt_store_does_not_reset_trust(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"pairing.json"
            path.write_text('{broken')
            with self.assertRaises(ValueError): PairingStore(path).create_device("test")
            self.assertEqual(path.read_text(), '{broken')

    def test_tracking_rejects_missing_nonfinite_or_stale_registration(self):
        for value in (None, {}, {"registered":False,"age_s":0}, {"registered":True,"age_s":float("nan")},
                      {"registered":True,"age_s":4}, {"registered":True,"age_s":-1}, {"registered":True,"age_s":True}):
            with self.subTest(value=value), self.assertRaises(ValueError): require_tracking(value)
        require_tracking({"registered":True,"age_s":0.5})


class FakePipeline:
    def __init__(self):
        self.glasses = {"connected":0, "error":"", "port":None}
        self.session_id = "dashboard-run"
        self.seen = 0
        self.decisions = []
        self.message = {"type":"trajectory", "version":1, "id":"a"*32, "phase":"review",
                        "frame":"robot_base", "units":"m", "hands":{"left":[], "right":[[.1,0,1],[.2,0,1]]},
                        "review":{"id":"a"*32, "digest":"b"*64, "revision":2, "mode":"sim",
                                  "text":"Wave", "expires_at":time.time()+60}}
    def operator_seen(self): self.seen += 1
    def glasses_message(self): return copy.deepcopy(self.message)
    def decide(self, *args):
        if not self.message.get("review"): raise ValueError("No review")
        self.decisions.append(args)
        self.message["review"] = None


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PairingStore(Path(self.tmp.name)/"pairing.json")
        self.device = self.store.create_device("test glasses")
        self.pipeline = FakePipeline()
        self.voices = []
        def voice(text, command_id, device_id):
            self.voices.append((text, command_id, device_id))
            return {"accepted":True, "task_id":"task"}
        self.bridge = GlassesBridge(self.pipeline, "127.0.0.1", 0, pairing=self.store, on_voice=voice)
        self.assertTrue(await asyncio.to_thread(self.bridge.ready.wait, 3))
        self.assertEqual(self.pipeline.glasses["error"], "")
        self.url = "ws://127.0.0.1:"+str(self.pipeline.glasses["port"])

    async def asyncTearDown(self):
        await asyncio.to_thread(self.bridge.close)
        self.tmp.cleanup()

    async def paired(self):
        socket = await connect(self.url)
        await socket.send(json.dumps({"type":"authenticate", "device_id":self.device["device_id"], "token":self.device["token"]}))
        ack = json.loads(await socket.recv())
        self.assertTrue(ack["accepted"])
        return socket, ack["session"]

    async def response(self, socket, kind):
        while True:
            message = json.loads(await asyncio.wait_for(socket.recv(), 2))
            if message.get("type") == kind: return message

    def decision(self, current_session, **changes):
        return {"type":"review_decision", "version":1, "session":current_session, "id":"a"*32,
                "digest":"b"*64, "revision":2, "decision":"approve", "tracking":{"registered":True,"age_s":0}, **changes}

    async def test_unauthenticated_client_receives_no_paths(self):
        async with connect(self.url) as socket:
            await socket.send(json.dumps({"type":"authenticate", "device_id":self.device["device_id"], "token":"wrong"}))
            with self.assertRaises(ConnectionClosed): await socket.recv()
        self.assertEqual(self.pipeline.decisions, [])

    async def test_exact_session_revision_and_fresh_tracking_required(self):
        socket, session = await self.paired()
        async with socket:
            draft = await self.response(socket, "trajectory")
            self.assertEqual(draft["review"]["session"], session)
            for changes in ({"session":"old"}, {"digest":"c"*64}, {"revision":1},
                            {"tracking":{"registered":True,"age_s":8}}, {"tracking":None}):
                await socket.send(json.dumps(self.decision(session, **changes)))
                self.assertFalse((await self.response(socket, "review_ack"))["accepted"])
            self.assertEqual(self.pipeline.decisions, [])
            await socket.send(json.dumps(self.decision(session)))
            self.assertTrue((await self.response(socket, "review_ack"))["accepted"])
            await socket.send(json.dumps(self.decision(session)))
            self.assertFalse((await self.response(socket, "review_ack"))["accepted"])
        self.assertEqual(len(self.pipeline.decisions), 1)

    async def test_decline_works_with_stale_tracking(self):
        socket, session = await self.paired()
        async with socket:
            await socket.send(json.dumps(self.decision(session, decision="decline", tracking=None)))
            self.assertTrue((await self.response(socket, "review_ack"))["accepted"])

    async def test_draft_is_visible_but_cannot_be_approved(self):
        self.pipeline.message.update(phase="draft", review=None)
        socket, session = await self.paired()
        async with socket:
            message = await self.response(socket, "trajectory")
            self.assertEqual(message["phase"], "draft")
            self.assertIsNone(message["review"])
            await socket.send(json.dumps(self.decision(session)))
            self.assertFalse((await self.response(socket, "review_ack"))["accepted"])
        self.assertEqual(self.pipeline.decisions, [])

    async def test_revoke_closes_existing_and_denies_reconnect(self):
        socket, session = await self.paired()
        async with socket:
            self.store.revoke_device(self.device["device_id"])
            with self.assertRaises(ConnectionClosed):
                while True: await asyncio.wait_for(socket.recv(), 2)
        async with connect(self.url) as socket:
            await socket.send(json.dumps({"type":"authenticate", **{"device_id":self.device["device_id"], "token":self.device["token"]}}))
            with self.assertRaises(ConnectionClosed): await socket.recv()

    async def test_voice_is_authenticated_deduplicated_and_subordinate_to_review(self):
        socket, session = await self.paired()
        command = {"type":"voice_command", "version":1, "id":"voice-1", "text":"  wave  please "}
        async with socket:
            await socket.send(json.dumps({**command, "session":session}))
            self.assertFalse((await self.response(socket, "voice_ack"))["accepted"])
            self.pipeline.message.update(phase="draft", review=None)
            await socket.send(json.dumps({**command, "session":"old"}))
            self.assertFalse((await self.response(socket, "voice_ack"))["accepted"])
            for _ in range(2):
                await socket.send(json.dumps({**command, "session":session}))
                self.assertTrue((await self.response(socket, "voice_ack"))["accepted"])
        second, new_session = await self.paired()
        self.assertNotEqual(new_session, session)
        async with second:
            await second.send(json.dumps({**command, "session":new_session}))
            self.assertTrue((await self.response(second, "voice_ack"))["accepted"])
            await second.send(json.dumps({**command, "session":new_session, "text":"something different"}))
            self.assertFalse((await self.response(second, "voice_ack"))["accepted"])
        self.assertEqual(self.voices, [("wave please", "voice-1", self.device["device_id"])])
