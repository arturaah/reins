"""Authenticated Spectacles adapter for the same proposals used by the browser."""
import asyncio
import json
import secrets
import threading


class GlassesBridge:
    def __init__(self, pipeline, host="0.0.0.0", port=8765):
        self.pipeline, self.host, self.port = pipeline, host, port
        self.loop = None
        self.stop_event = None
        self.ready = threading.Event()
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
        async with serve(self.handler, self.host, self.port, max_size=4096) as server:
            self.pipeline.glasses["port"] = server.sockets[0].getsockname()[1]
            self.ready.set()
            await self.stop_event.wait()

    async def handler(self, websocket):
        try:
            auth = json.loads(await asyncio.wait_for(websocket.recv(), 5))
            if not isinstance(auth, dict) or auth.get("type") != "authenticate" or not isinstance(auth.get("token"), str) or not secrets.compare_digest(auth["token"], self.pipeline.glasses_token):
                await websocket.close(code=1008, reason="Pair the glasses with this dashboard session")
                return
        except (ValueError, asyncio.TimeoutError):
            await websocket.close(code=1008, reason="Authentication required")
            return
        await websocket.send(json.dumps({"type": "auth_ack", "accepted": True}))
        self.pipeline.glasses["connected"] += 1
        self.pipeline.operator_seen()

        async def send():
            while True:
                await websocket.send(json.dumps(self.pipeline.glasses_message(), allow_nan=False))
                await asyncio.sleep(.2)

        sender = asyncio.create_task(send())
        try:
            async for raw in websocket:
                msg = None
                try:
                    msg = json.loads(raw)
                    if msg.get("type") == "heartbeat":
                        self.pipeline.operator_seen()
                        continue
                    if msg.get("type") == "review_decision" and msg.get("version") == 1:
                        self.pipeline.decide(msg.get("id"), msg.get("digest"), msg.get("decision"), "Spectacles review")
                        await websocket.send(json.dumps({"type": "review_ack", "version": 1, "id": msg.get("id"), "accepted": True}))
                except (ValueError, TypeError, AttributeError) as exc:
                    await websocket.send(json.dumps({"type": "review_ack", "version": 1,
                                                      "id": msg.get("id") if isinstance(msg, dict) else None,
                                                      "accepted": False, "error": str(exc)}))
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            self.pipeline.glasses["connected"] -= 1

    def close(self):
        if self.loop and self.stop_event and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(3)
