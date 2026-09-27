"""Local browser audio transport. Simulation only; never imports the robot SDK."""
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import secrets
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from .audio import MAX_SECONDS, RATE, robotic_pcm
from .providers import demo_session

ASSETS = Path(__file__).parent / 'web'
INPUT_RATE = 16000
MAX_INPUT = MAX_SECONDS * INPUT_RATE * 2


async def conversation(ws, provider):
    """One bounded turn at a time; input and Stop remain responsive during replies."""
    task = None
    recording = False
    busy = False
    awaiting_playback = False
    audio = bytearray()
    started = 0.0

    async def respond(*, pcm=None, text=None):
        nonlocal awaiting_playback, busy
        await ws.send_json({'type': 'thinking'})
        reply = await asyncio.wait_for(provider.reply(pcm=pcm, text=text), 120)
        if reply.heard:
            await ws.send_json({'type': 'transcript', 'role': 'you', 'text': reply.heard[:8000]})
        if reply.said:
            await ws.send_json({'type': 'transcript', 'role': 'reins', 'text': reply.said[:8000]})
        if reply.details:
            await ws.send_json({'type': 'result', **reply.details})
        if not reply.pcm:
            busy = False
            await ws.send_json({'type': 'ready'})
            return
        output = await asyncio.to_thread(robotic_pcm, reply.pcm)
        # Ordered frames: metadata then one bounded PCM buffer. Client acknowledges onended.
        awaiting_playback = True
        await ws.send_json({'type': 'audio', 'sample_rate': RATE})
        await ws.send_bytes(output)

    await ws.send_json({'type': 'ready'})
    try:
        while True:
            timeout = max(.01, MAX_SECONDS + 2 - (time.monotonic() - started)) if recording else 120
            receive = asyncio.create_task(ws.receive())
            try:
                pending = [receive] + ([task] if task and not task.done() else [])
                if task and task.done():
                    task.result()
                done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    raise ValueError('Recording or session timed out; reconnect to continue')
                if task and task in done:
                    task.result()
                    if receive not in done:
                        continue
                message = receive.result()
            finally:
                if not receive.done():
                    receive.cancel()
                    await asyncio.gather(receive, return_exceptions=True)
            if message['type'] == 'websocket.disconnect':
                return
            if message.get('bytes') is not None:
                chunk = message['bytes']
                if not recording:
                    raise ValueError('Audio received outside a recording')
                if not chunk or len(chunk) % 2 or len(chunk) > 16384 or len(audio) + len(chunk) > MAX_INPUT:
                    raise ValueError('Invalid audio or 30-second recording limit exceeded')
                audio.extend(chunk)
                if hasattr(provider, 'input_chunk'):
                    endpoint = await provider.input_chunk(chunk)
                    if endpoint and auto_end:
                        auto_end = False
                        await ws.send_json({'type': 'endpoint'})
                continue
            raw = message.get('text', '')
            if len(raw) > 4096:
                raise ValueError('Message too large')
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise ValueError('Expected a voice action')
            action = event.get('type')
            if action == 'stop':
                return
            if action == 'played':
                if not awaiting_playback or not task or not task.done():
                    raise ValueError('No completed reply to acknowledge')
                task.result()
                awaiting_playback = busy = False
                task = None
                await ws.send_json({'type': 'ready'})
            elif action == 'start':
                if busy or recording:
                    raise ValueError('Wait for the current turn')
                if type(event.get('sample_rate')) is not int or event['sample_rate'] != INPUT_RATE:
                    raise ValueError('Microphone must provide 16 kHz mono PCM16')
                audio.clear()
                auto_end = event.get('auto_end') is True
                if hasattr(provider, 'begin_input'):
                    await provider.begin_input()
                recording = True
                started = time.monotonic()
                await ws.send_json({'type': 'recording'})
            elif action == 'end':
                if not recording or not audio:
                    raise ValueError('Record some audio before sending')
                recording = False
                busy = True
                task = asyncio.create_task(respond(pcm=bytes(audio)))
                audio.clear()
            elif action in ('hello', 'text'):
                if busy or recording:
                    raise ValueError('Wait for the current turn')
                text = 'Hello. I am Reins. Ready to talk.' if action == 'hello' else event.get('text')
                if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
                    raise ValueError('Enter 1 to 1000 characters')
                busy = True
                task = asyncio.create_task(respond(text=text.strip()))
            else:
                raise ValueError('Unknown voice action')
    finally:
        audio.clear()
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def create_app(*, port=8770, provider='demo', session_factory=demo_session, public_config=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    token = secrets.token_urlsafe(32)
    owner = asyncio.Lock()
    hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
    origins = {f'http://{host}' for host in hosts}

    @app.middleware('http')
    async def local_only(request, call_next):
        if request.headers.get('host') not in hosts:
            return JSONResponse({'error': 'Local access only'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; "
            "style-src 'self'; script-src 'self'; media-src 'self' blob:; "
            "frame-ancestors http://127.0.0.1:* http://localhost:*; object-src 'none'; base-uri 'none'"
        )
        return response

    @app.get('/config')
    async def config():
        return {'mode': 'sim', 'provider': provider,
                'voice': 'Charon', 'token': token, 'input_rate': INPUT_RATE, **(public_config or {})}

    @app.get('/')
    async def index():
        return FileResponse(ASSETS / 'index.html')

    @app.get('/{asset}')
    async def asset(asset: str):
        allowed = {'app.js': 'text/javascript', 'mic.js': 'text/javascript', 'style.css': 'text/css'}
        if asset not in allowed:
            return JSONResponse({'error': 'Not found'}, status_code=404)
        return FileResponse(ASSETS / asset, media_type=allowed[asset])

    @app.websocket('/voice')
    async def voice(ws: WebSocket):
        if ws.headers.get('host') not in hosts or ws.headers.get('origin') not in origins:
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            hello = await asyncio.wait_for(ws.receive_json(), 5)
            supplied = hello.get('token') if isinstance(hello, dict) else None
            if not isinstance(supplied, str) or not secrets.compare_digest(supplied, token):
                await ws.close(code=1008)
                return
            if owner.locked():
                await ws.send_json({'type': 'error', 'text': 'Another voice session is connected. Stop it first.'})
                return
            async with owner:
                await ws.send_json({'type': 'connecting'})
                async def run():
                    context = session_factory()
                    session = await asyncio.wait_for(context.__aenter__(), 15)
                    try:
                        await conversation(ws, session)
                    finally:
                        await context.__aexit__(None, None, None)
                # Bounds stalled provider setup as well as idle connections.
                await asyncio.wait_for(run(), 15 * 60)
        except WebSocketDisconnect:
            pass
        except (ValueError, asyncio.TimeoutError):
            with suppress(Exception):
                await ws.send_json({'type': 'error', 'text': 'Voice turn failed or timed out. Reconnect and try a shorter recording.'})
        except Exception:
            # Do not return provider payloads, transcripts, URLs, or credentials in errors/logs.
            with suppress(Exception):
                await ws.send_json({'type': 'error', 'text': 'Voice provider unavailable. Check the server key, model access and internet connection.'})
        finally:
            with suppress(Exception):
                await ws.close()
    return app
