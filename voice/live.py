"""GPT-Live voice frontend with client-owned reasoning and optional R1 audio output.

Run separately from the cascade: python -m voice.live --key-file /path/to/.env
RobotBackend is the integration boundary; this module never imports motion tools.
"""
import argparse
import asyncio
import base64
from collections import deque
from contextlib import asynccontextmanager, suppress
import json
import os
import time
from pathlib import Path
from typing import Protocol

from websockets.asyncio.client import connect

from .acoustics import Acoustics, FOCUS_MODEL
from .audio import RoboticStream
from .errors import SpeechError, provider_error
from .insight import make_live_tyto
from .server import create_app

LIVE_MODEL = 'gpt-live-1'
LIVE_VOICE = 'cedar'
CONVERSATION_INSTRUCTIONS = '''You are Reins, a friendly robot voice for a live demo.
Speak naturally in short, direct sentences, with a low masculine voice and a
lightly robotic, clipped delivery. Match the caller's language. Listen before
answering; do not backchannel while the caller speaks. Handle conversation
directly, without delegating. You have no tools, camera, motion controls or
planner. Never claim to see the room or to move the robot. If asked to move,
explain briefly that this demo is conversation only. Do not introduce model
names or implementation details unless asked.'''
INSTRUCTIONS = '''You are Reins, a robot voice interface in a local simulation.
Speak in short, direct sentences, with a low masculine voice and a lightly robotic,
clipped delivery. Match the caller's language. You cannot see or move a robot.
No simulator is connected either. If acknowledging delegated work, say only
"Let me ask the backend." Never claim to be checking a robot or simulator.
Backchannel policy: No backchannels while the caller is speaking.
Interruption policy: Pause and listen if the caller corrects you.
Delegation policy:
Backend tools:
- Robot reasoning: a text-only simulation backend for spatial questions, robot
  capabilities and requested movements. It cannot execute actions.
Delegate to the backend when:
- The user requests a robot movement, spatial reasoning, or explicitly asks the backend.
- A correction changes a pending request.
Do not delegate to the backend when:
- Greeting, small talk, or clarifying what the user wants.
Delegate before answering a robot task. Do not guess results while waiting.
Never say an action happened unless the backend confirms it. This test has no
executor: requested movements are discussion only. Acknowledge a wait at most once.
'''


DASHBOARD_INSTRUCTIONS = '''You are Reins, a voice interface for the dashboard robot agent.
Speak briefly and naturally. Handle greetings and conversation yourself.
Delegate robot tasks, movement requests, spatial questions and corrections to the
backend before answering. You have no camera or robot controls yourself. The
backend submits tasks to the dashboard agent and returns submission status, not
planning or execution results. Report only what it confirms. Do not invent what
the agent sees or claim a motion succeeded. Draft previews may be revised freely;
a completed proposal always needs separate operator approval in the dashboard or
paired glasses. Spoken instructions never approve a motion. Ask the caller to
review the dashboard/glasses after submitting. Keep quiet while the caller speaks.
'''


class RobotBackend(Protocol):
    """Team-owned adapter: a bounded transcript snapshot in, verified facts out."""
    model: str
    async def respond(self, messages: list[dict[str, str]]) -> str: ...
    async def close(self) -> None: ...


class ConversationBackend:
    """A local response if Live delegates unexpectedly; no API or task dispatch."""
    model = 'Conversation only'

    async def respond(self, messages):
        return 'This demo is conversation only. No action was queued or executed. Continue talking directly.'

    async def close(self):
        pass


class TestRobotBackend:
    """Temporary GPT-5-mini adapter; no functions, harness, or robot connection."""
    __test__ = False

    def __init__(self, key, model='gpt-5-mini'):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=key, max_retries=0, timeout=20)
        self.model = model

    async def respond(self, messages):
        result = await self.client.responses.create(
            model=self.model, store=False, input=messages, max_output_tokens=700,
            reasoning={'effort': 'minimal'}, text={'verbosity': 'low'},
            instructions=('You are the text backend for a robot voice simulation. '
                          'The conversation is a possibly imperfect voice transcript. '
                          'Use the latest request and corrections. You have no tools, '
                          'camera, robot connection, motion executor, or simulator state. '
                          'Never claim an action was executed or describe an observed room. '
                          'Return concise useful reasoning or one clarification, at most '
                          '600 characters. For a movement request explicitly say no movement '
                          'was executed. Do not add a greeting.'))
        answer = result.output_text.strip()
        if result.status != 'completed' or not 1 <= len(answer) <= 800:
            raise SpeechError('invalid_response')
        return answer

    async def close(self):
        await self.client.close()


class LiveSession:
    def __init__(self, key, backend: RobotBackend, acoustics, *, voice=LIVE_VOICE, metallic=True,
                 speaker=None, instructions=INSTRUCTIONS):
        self.key, self.backend, self.acoustics, self.voice = key, backend, acoustics, voice
        self.upstream = self.ws = self.focus = self.insight = None
        self.output_effect = RoboticStream(rate=16000) if metallic else None
        self.speaker, self.instructions = speaker, instructions
        self.tasks = set()
        self.backend_task = None
        self.history = deque(maxlen=40)
        self.seen_delegations = set()
        self.revision = 0
        self.listening = self.speaker_busy = self.blocked = self.closing = False
        self.nudge_waiting_speech = False
        self.backend_count = 0
        self.closed = asyncio.Event()

    async def send(self, event):
        await asyncio.wait_for(self.upstream.send(json.dumps(event)), 5)

    async def emit(self, stage, status, **details):
        if self.ws and not self.closing:
            await self.ws.send_json({'type': 'log', 'timestamp': time.time(),
                                     'stage': stage, 'status': status, **details})
        if stage == 'tyto' and status == 'nudge' and not self.closing:
            self.blocked = True
            self.nudge_waiting_speech = True
            self.revision += 1
            self.history.clear()  # Do not carry unclear transcript fragments into a later delegation.
            if self.backend_task: self.backend_task.cancel()
            await self.ws.send_json({'type': 'clear_audio'})
            if self.speaker:
                await self.speaker.clear()
            # Live may paraphrase. Exact rendered nudge wording remains a cascade feature.
            await self.send({'type': 'session.instructions.append', 'delegation_id': None,
                             'content': 'Stop the current answer. Ignore the unclear request. '
                             'Say this clarification now, then listen: ' + details['text']})
            self.spawn(self.nudge_timeout())

    def spawn(self, call):
        task = asyncio.create_task(call)
        self.tasks.add(task)
        return task

    async def nudge_timeout(self):
        await asyncio.sleep(15)
        if self.blocked:
            raise SpeechError('timeout')

    async def prepare(self):
        try:
            if self.speaker:
                self.speaker.on_busy = self.set_speaker
                await self.speaker.start()
            self.upstream = await connect(
                'wss://api.openai.com/v1/live/sessions',
                additional_headers={'Authorization': 'Bearer ' + self.key},
                open_timeout=10, close_timeout=2, max_size=262144, max_queue=16)
            await self.send({'type': 'session.start', 'session': {
                'model': LIVE_MODEL, 'store': False, 'instructions': self.instructions,
                'audio': {'format': {'type': 'audio/pcm', 'rate': 16000},
                          'output': {'voice': self.voice}},
                'delegation': {'type': 'client'}}})
            event = json.loads(await asyncio.wait_for(self.upstream.recv(), 10))
            if event.get('type') == 'error': self.raise_error(event)
            if event.get('type') != 'session.started': raise SpeechError('invalid_response')
        except SpeechError:
            raise
        except Exception as error:
            raise provider_error(error) from None
        finally:
            self.key = None

    @staticmethod
    def raise_error(event):
        code = event.get('error', {}).get('code', '')
        raise SpeechError({'invalid_api_key': 'authentication', 'model_not_found': 'model_access',
                           'rate_limit_exceeded': 'rate_limit', 'insufficient_quota': 'rate_limit'
                           }.get(code, 'provider_error'))

    def remember(self, role, delta):
        if not isinstance(delta, str) or len(delta) > 4000: raise SpeechError('invalid_response')
        if role == 'user': self.revision += 1
        if self.history and self.history[-1]['role'] == role:
            self.history[-1]['content'] = (self.history[-1]['content'] + delta)[-4000:]
        else:
            self.history.append({'role': role, 'content': delta})

    def snapshot(self):
        messages, size = [], 0
        for message in reversed(self.history):
            size += len(message['content'])
            if size > 12000: break
            messages.append(dict(message))
        return list(reversed(messages))

    async def delegate(self, delegation):
        identity = delegation.get('id')
        if delegation.get('target') != 'client' or not isinstance(identity, str) or not 1 <= len(identity) <= 200:
            raise SpeechError('invalid_response')
        if identity in self.seen_delegations: return
        if len(self.seen_delegations) >= 60: raise SpeechError('rate_limit')
        self.seen_delegations.add(identity)
        if self.blocked or not any(m['role'] == 'user' for m in self.history):
            await self.send({'type': 'session.thinking.append', 'delegation_id': identity,
                             'content': 'No clear accepted request is available. Ask the caller to repeat it.'})
            return
        if self.backend_task:
            self.backend_task.cancel()
            await asyncio.gather(self.backend_task, return_exceptions=True)
        self.backend_task = self.spawn(self.run_backend(identity))

    async def run_backend(self, identity):
        revision = self.revision
        started = time.monotonic()
        self.backend_count += 1
        await self.emit('backend', 'started', model=self.backend.model, text=getattr(self.backend, 'description', 'Simulation reasoning; no action executor.'))
        try:
            result = await asyncio.wait_for(self.backend.respond(self.snapshot()), 20)
            if self.closing or self.blocked or revision != self.revision:
                await self.emit('backend', 'discarded', text='New speech or audio clarification superseded this result.')
                return
            if not isinstance(result, str) or not 1 <= len(result) <= 800:
                raise SpeechError('invalid_response')
            await self.send({'type': 'session.commentary.append', 'delegation_id': identity, 'content': result})
            await self.emit('backend', 'completed', model=self.backend.model,
                            elapsed_s=round(time.monotonic()-started, 3), text=result)
        except asyncio.CancelledError:
            await self.emit('backend', 'cancelled')
            raise
        except Exception:
            failure = getattr(self.backend, 'failure_message',
                'The simulation backend could not complete the request. No action was executed.')
            await self.emit('backend', 'error', text=failure)
            await self.send({'type': 'session.commentary.append', 'delegation_id': identity, 'content': failure})

    async def receive_upstream(self):
        async for raw in self.upstream:
            event = json.loads(raw)
            kind = event.get('type')
            if kind == 'error': self.raise_error(event)
            if kind == 'session.closed':
                self.closed.set()
                await self.emit('live', 'closed')
                return
            if self.closing: continue
            if kind == 'session.output_audio.delta':
                pcm = base64.b64decode(event['delta'], validate=True)
                if not pcm or len(pcm) % 2 or len(pcm) > 64000: raise SpeechError('invalid_response')
                if self.output_effect: pcm = self.output_effect.process(pcm)
                if self.speaker:
                    await self.speaker.play(pcm)
                else:
                    await self.ws.send_bytes(pcm)
            elif kind in ('session.input_transcript.delta', 'session.output_transcript.delta'):
                role = 'user' if kind == 'session.input_transcript.delta' else 'assistant'
                if role == 'user' and self.blocked: continue
                self.remember(role, event['delta'])
                await self.ws.send_json({'type': 'transcript_delta', 'role': role, 'text': event['delta']})
            elif kind == 'session.delegation.created':
                await self.delegate(event['delegation'])
            elif kind == 'session.usage.updated':
                await self.emit('live', 'usage', text='Voice session remains active; Stop ends billing.')

    async def set_speaker(self, busy):
        if busy == self.speaker_busy: return
        self.speaker_busy = busy
        await self.emit('playback', 'started' if busy else 'completed')
        if busy:
            self.nudge_waiting_speech = False
            if self.insight: await self.insight.pause()
        else:
            if self.blocked and self.nudge_waiting_speech: return
            self.blocked = False
            if self.focus: self.focus.close()
            self.focus = self.acoustics.stream()
            if self.insight: await self.insight.reset()

    async def receive_browser(self):
        while True:
            message = await asyncio.wait_for(self.ws.receive(), 120)
            if message['type'] == 'websocket.disconnect': return
            pcm = message.get('bytes')
            if pcm is not None:
                if not self.listening or not pcm or len(pcm) % 2 or len(pcm) > 4096:
                    raise ValueError('Invalid live microphone frame')
                if self.speaker_busy or self.blocked:
                    enhanced = bytes(len(pcm))
                else:
                    if self.insight: self.insight.feed(pcm)
                    enhanced = self.focus.feed(pcm)
                if enhanced:
                    await self.send({'type': 'session.input_audio.append',
                                     'audio': base64.b64encode(enhanced).decode('ascii')})
                continue
            raw = message.get('text', '')
            if len(raw) > 4096: raise ValueError('Message too large')
            event = json.loads(raw)
            if not isinstance(event, dict): raise ValueError('Expected a live action')
            kind = event.get('type')
            if kind == 'stop': return
            if kind == 'listen' and not self.listening and event.get('sample_rate') == 16000:
                self.listening = True
                self.focus = self.acoustics.stream()
                if self.acoustics.tyto:
                    self.insight = make_live_tyto(self.acoustics, self.emit)
                    await self.insight.reset()
                await self.emit('voice_focus', 'enabled' if self.acoustics.focus else 'disabled',
                                model=FOCUS_MODEL, enhancement_level=self.acoustics.focus_level)
                await self.ws.send_json({'type': 'listening'})
            elif kind == 'speaker' and self.listening and type(event.get('busy')) is bool:
                if not self.speaker:  # The R1 output owns its own playback gate.
                    await self.set_speaker(event['busy'])
            else:
                raise ValueError('Unknown live action')

    async def run(self, ws):
        self.ws = ws
        await ws.send_json({'type': 'ready'})
        await self.emit('voice_effect', 'enabled' if self.output_effect else 'disabled',
                        text='Streaming metallic tone · no full-reply buffering.' if self.output_effect else 'Natural voice.')
        receiver = self.spawn(self.receive_upstream())
        browser = self.spawn(self.receive_browser())
        if self.speaker:
            self.spawn(self.speaker.monitor())
        try:
            while True:
                done, _ = await asyncio.wait(self.tasks, timeout=.1, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    self.tasks.discard(task)
                    if not task.cancelled(): task.result()
                if receiver in done or browser in done: break
        finally:
            self.closing = True
            for task in self.tasks:
                if task is not receiver: task.cancel()
            if self.speaker:
                await self.speaker.close()
            if self.upstream and not receiver.done():
                with suppress(Exception):
                    await self.send({'type': 'session.close'})
                    await asyncio.wait_for(self.closed.wait(), 3)
            for task in self.tasks: task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)

    async def close(self):
        self.closing = True
        for task in self.tasks: task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        try:
            if self.insight: await self.insight.close()
        finally:
            if self.speaker: await self.speaker.close()
            if self.focus: self.focus.close()
            if self.upstream: await self.upstream.close()
            await self.backend.close()
            self.history.clear()


async def live_conversation(ws, session):
    await session.run(ws)


def main():
    from .config import load_keys, ROOT
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--backend-model', default='gpt-5-mini')
    parser.add_argument('--backend', choices=['test', 'spectacles', 'conversation', 'dashboard'], default='test')
    from .dashboard_backend import dashboard_url
    parser.add_argument('--dashboard-url', type=dashboard_url, default='http://127.0.0.1:8090',
                        help='Local Reins dashboard used by --backend dashboard')
    parser.add_argument('--voice-inbox', type=Path, help='Retired: use --backend dashboard')
    parser.add_argument('--output', choices=['browser', 'r1'], default='browser')
    parser.add_argument('--robot-iface', help='R1 body Ethernet interface; required with --output r1')
    parser.add_argument('--robot-python', default=str(ROOT / '.venv/bin/python'),
                        help='Python with unitree_sdk2py installed; separate from the voice environment')
    parser.add_argument('--voice', default=LIVE_VOICE)
    parser.add_argument('--metallic', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply the cascade metallic tone to streamed audio without buffering replies')
    parser.add_argument('--voice-focus', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tyto', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.voice_inbox is not None:
        parser.error('--voice-inbox is retired; use --backend dashboard --dashboard-url URL')
    if args.backend == 'spectacles':
        print('--backend spectacles now uses the dashboard agent; use --backend dashboard --dashboard-url URL.', flush=True)
        args.backend = 'dashboard'
    if not 1024 <= args.port <= 65535: parser.error('Invalid port')
    if args.output == 'r1' and not args.robot_iface:
        parser.error('--output r1 requires --robot-iface')
    try:
        load_keys(args.key_file)
    except ValueError as error:
        parser.error(str(error))
    if not os.environ.get('OPENAI_API_KEY'): parser.error('Set OPENAI_API_KEY or provide --key-file')
    acoustics = Acoustics(focus=args.voice_focus, tyto=args.tyto, vad='webrtc',
                          license_key=os.environ.get('AIC_SDK_LICENSE', ''))

    @asynccontextmanager
    async def factory():
        key = os.environ['OPENAI_API_KEY']
        from .dashboard_backend import DashboardBackend
        from .robot_speaker import RobotSpeaker
        if args.backend == 'conversation':
            backend = ConversationBackend()
        elif args.backend == 'dashboard':
            backend = DashboardBackend(args.dashboard_url)
        else:
            backend = TestRobotBackend(key, args.backend_model)
        instructions = INSTRUCTIONS
        if args.backend == 'conversation':
            instructions = CONVERSATION_INSTRUCTIONS
        if args.backend == 'dashboard':
            instructions = DASHBOARD_INSTRUCTIONS
        speaker = RobotSpeaker(args.robot_python, args.robot_iface) if args.output == 'r1' else None
        session = LiveSession(key, backend, acoustics, voice=args.voice, metallic=args.metallic,
                              speaker=speaker, instructions=instructions)
        try:
            await session.prepare()
            yield session
        finally:
            await session.close()

    app = create_app(port=args.port, provider='gpt-live', session_factory=factory,
                     session_runner=live_conversation, index_asset='live.html', public_config={
                         'live_model': LIVE_MODEL, 'backend_model': {'test': args.backend_model, 'dashboard': 'Reins dashboard agent', 'conversation': 'Conversation only'}[args.backend], 'voice': args.voice,
                         'output': args.output, 'backend': args.backend,
                         'metallic': args.metallic,
                         'voice_focus': acoustics.focus, 'enhancement_level': acoustics.focus_level,
                         'tyto': acoustics.tyto, 'barge_in': False})
    print(f'Reins GPT-Live → http://127.0.0.1:{args.port} · output: {args.output} · no motion executor', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning', ws_max_size=32768)


if __name__ == '__main__': main()
