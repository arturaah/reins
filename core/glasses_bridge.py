"""Paired, session-bound review and voice input for the dashboard coordinator."""
import asyncio
from collections import OrderedDict
import copy
import json
import math
import secrets
import threading
import time

from core.glasses_pairing import PairingStore

TRACKING_MAX_AGE = 3.0


def require_tracking(value):
    """Reported registration freshness, not an assertion of calibrated accuracy."""
    age = value.get("age_s") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("registered") is not True or
            type(age) not in (float, int) or not math.isfinite(age) or not 0 <= age <= TRACKING_MAX_AGE):
        raise ValueError("Scan both shoulder tags again before approving")


class GlassesBridge:
    def __init__(self, pipeline, host="0.0.0.0", port=8765, pairing=None, on_voice=None,
                 live_voice_url=None, relay_factory=None):
        self.pipeline, self.host, self.port = pipeline, host, port
        self.pairing = pairing if pairing is not None else PairingStore()
        self.on_voice = on_voice
        if live_voice_url:
            from spectacles.live_voice import LiveVoiceRelay, local_voice_url
            self.live_voice_url = local_voice_url(live_voice_url)
            self.relay_factory = relay_factory or LiveVoiceRelay
        else:
            self.live_voice_url, self.relay_factory = None, None
        self.loop = None
        self.stop_event = None
        self.ready = threading.Event()
        self.connections = {}
        self.voice_jobs = OrderedDict()
        self.pipeline.glasses.setdefault("devices", [])
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            self.pipeline.glasses["error"] = str(exc)
        finally:
            self.ready.set()

    async def _serve(self):
        from websockets.asyncio.server import serve
        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        async with serve(self.handler, self.host, self.port, max_size=4096, max_queue=8) as server:
            self.pipeline.glasses["port"] = server.sockets[0].getsockname()[1]
            self.ready.set()
            await self.stop_event.wait()

    def _connected(self):
        self.pipeline.glasses["connected"] = len(self.connections)
        self.pipeline.glasses["devices"] = sorted(set(self.connections.values()))

    async def _voice(self, device_id, msg):
        command_id, text = msg.get("id"), msg.get("text")
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 80 or not isinstance(text, str):
            raise ValueError("Invalid voice command")
        text = " ".join(text.split())
        if not 1 <= len(text) <= 500:
            raise ValueError("Voice command must contain 1–500 characters")
        key = (device_id, command_id)
        if key in self.voice_jobs:
            previous, task = self.voice_jobs[key]
            if previous != text:
                raise ValueError("Voice command ID was already used for different text")
            return await asyncio.shield(task)
        if self.pipeline.glasses_message().get("review"):
            raise ValueError("Finish reviewing the current motion before speaking a new task")
        if self.on_voice is None:
            raise ValueError("Voice input is not connected to the dashboard agent")

        async def submit():
            try:
                result = await asyncio.to_thread(self.on_voice, text, command_id, device_id)
                return result if isinstance(result, dict) else {"accepted": bool(result)}
            except (ValueError, RuntimeError) as exc:
                return {"accepted": False, "message": str(exc)[:300]}
        task = asyncio.create_task(submit())
        self.voice_jobs[key] = (text, task)
        while len(self.voice_jobs) > 256:
            old_key, (_, old_task) = next(iter(self.voice_jobs.items()))
            if not old_task.done():
                break
            del self.voice_jobs[old_key]
        return await asyncio.shield(task)

    async def handler(self, websocket):
        from websockets.exceptions import ConnectionClosed
        try:
            auth = json.loads(await asyncio.wait_for(websocket.recv(), 5))
            device_id = auth.get("device_id") if isinstance(auth, dict) else None
            fingerprint = (self.pairing.authenticate(device_id, auth.get("token"))
                           if isinstance(auth, dict) and auth.get("type") == "authenticate" else None)
            if not fingerprint:
                await websocket.close(code=1008, reason="Pair this device in dashboard Connections")
                return
        except (ValueError, asyncio.TimeoutError, ConnectionClosed):
            await websocket.close(code=1008, reason="Authentication required")
            return
        session = secrets.token_urlsafe(24)
        await websocket.send(json.dumps({"type": "auth_ack", "accepted": True, "session": session,
                                         "run_id": getattr(self.pipeline, "session_id", None), "live_voice": bool(self.live_voice_url)}))
        self.connections[session] = device_id
        self._connected()
        self.pipeline.operator_seen()

        class SessionLens:
            async def send(_, raw):
                if not self.pairing.active(device_id, fingerprint):
                    return
                message = json.loads(raw)
                message["session"] = session
                await websocket.send(json.dumps(message))

        live = self.relay_factory(SessionLens(), self.live_voice_url) if self.relay_factory else None

        async def send():
            while True:
                if not self.pairing.active(device_id, fingerprint):
                    await websocket.close(code=1008, reason="Device pairing revoked")
                    return
                message = copy.deepcopy(self.pipeline.glasses_message())
                message["session"] = session
                if message.get("review"):
                    if live:
                        await live.stop()
                    message["review"]["session"] = session
                    expiry = message["review"].get("expires_at")
                    if expiry is not None:
                        message["review"]["expires_in_s"] = max(0, expiry-time.time())
                await websocket.send(json.dumps(message, allow_nan=False))
                await asyncio.sleep(.2)

        async def receive():
            async for raw in websocket:
                msg = None
                try:
                    if not self.pairing.active(device_id, fingerprint):
                        await websocket.close(code=1008, reason="Device pairing revoked")
                        return
                    if isinstance(raw, bytes):
                        if live and not self.pipeline.glasses_message().get("review"):
                            await live.audio(raw)
                        continue
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("Expected a message object")
                    if msg.get("session") != session:
                        raise ValueError("Connection session changed; reconnect and review again")
                    if msg.get("type") == "heartbeat":
                        self.pipeline.operator_seen()
                        continue
                    if msg.get("version") != 1:
                        raise ValueError("Unsupported glasses message version")
                    if msg.get("type") in ("voice_start", "voice_stop"):
                        identity = msg.get("id")
                        if not isinstance(identity, str) or not 1 <= len(identity) <= 80:
                            raise ValueError("Invalid voice session ID")
                        if not live:
                            raise ValueError("Live voice is not configured; use ASR or start the dashboard with --voice-url")
                        if msg["type"] == "voice_start":
                            if self.pipeline.glasses_message().get("review"):
                                raise ValueError("Finish reviewing the current motion before starting voice")
                            if msg.get("sample_rate") != 16000:
                                raise ValueError("Live voice requires 16 kHz mono PCM16")
                            await live.start(identity)
                        elif identity == live.identity:
                            await live.stop()
                    elif msg.get("type") == "voice_command":
                        if live and live.task and not live.task.done():
                            raise ValueError("Stop live voice before sending a separate transcript")
                        result = await self._voice(device_id, msg)
                        await websocket.send(json.dumps({**result, "type": "voice_ack", "version": 1,
                                                         "id": msg.get("id"), "session": session}))
                    elif msg.get("type") == "review_decision":
                        review = self.pipeline.glasses_message().get("review")
                        if not review or any(msg.get(key) != review.get(key) for key in ("id", "digest", "revision")):
                            raise ValueError("Proposal changed; review the current complete motion")
                        if review.get("expires_at", 0) <= time.time():
                            raise ValueError("Proposal expired; request a new motion")
                        if msg.get("decision") == "approve":
                            require_tracking(msg.get("tracking"))
                        self.pipeline.decide(msg.get("id"), msg.get("digest"), msg.get("decision"), "Spectacles review")
                        await websocket.send(json.dumps({"type": "review_ack", "version": 1, "id": msg.get("id"),
                                                         "accepted": True, "session": session}))
                except (ValueError, TypeError, AttributeError) as exc:
                    if isinstance(msg, dict) and msg.get("type") in ("voice_start", "voice_stop"):
                        await websocket.send(json.dumps({"type":"voice_event", "version":1, "id":msg.get("id"),
                            "session":session, "event":{"type":"error", "text":str(exc)[:300]}}))
                        continue
                    kind = "voice_ack" if isinstance(msg, dict) and msg.get("type") == "voice_command" else "review_ack"
                    await websocket.send(json.dumps({"type": kind, "version": 1,
                                                      "id": msg.get("id") if isinstance(msg, dict) else None,
                                                      "accepted": False, "error": str(exc)[:300], "session": session}))

        tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except ConnectionClosed:
            pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if live:
                await live.stop()
            self.connections.pop(session, None)
            self._connected()

    def close(self):
        if self.loop and self.stop_event and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(3)
