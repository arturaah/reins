import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

from voice.live import LiveSession, live_conversation, ConversationBackend
from voice.tests.test_voice import ORIGIN, connect, authenticate
from voice.server import create_app

class Backend:
    model = 'test-backend'
    def __init__(self): self.calls=[]; self.release=None; self.closed=False
    async def respond(self, messages):
        self.calls.append(messages)
        if self.release: await self.release.wait()
        return 'Simulation only. No movement was executed.'
    async def close(self): self.closed=True


def test_conversation_demo_delegation_is_local_and_never_dispatches_a_task():
    async def run():
        backend = ConversationBackend()
        result = await backend.respond([{'role':'user','content':'Walk forward'}])
        assert 'No action was queued or executed' in result
        assert not hasattr(backend, 'client') and not hasattr(backend, 'inbox')
        await backend.close()
    asyncio.run(run())

class Socket:
    def __init__(self): self.events=[]
    async def send_json(self, event): self.events.append(event)


def fixture():
    backend=Backend()
    session=LiveSession('not-a-real-key',backend,SimpleNamespace(focus=False,tyto=False))
    session.ws=Socket();sent=[]
    async def send(event): sent.append(event)
    session.send=send
    return session,backend,sent


def test_client_delegation_matches_id_and_never_repeats_duplicate():
    async def run():
        session,backend,sent=fixture()
        session.remember('user','Can you walk forward?')
        event={'id':'delegation-1','target':'client'}
        await session.delegate(event);await session.backend_task
        await session.delegate(event)
        assert backend.calls==[[{'role':'user','content':'Can you walk forward?'}]]
        assert sent==[{'type':'session.commentary.append','delegation_id':'delegation-1',
                       'content':'Simulation only. No movement was executed.'}]
        await session.close();assert backend.closed
    asyncio.run(run())


def test_new_speech_discards_old_backend_result():
    async def run():
        session,backend,sent=fixture();backend.release=asyncio.Event()
        session.remember('user','Walk forward.')
        await session.delegate({'id':'old','target':'client'});await asyncio.sleep(0)
        session.remember('user',' Actually, stop.')
        backend.release.set();await session.backend_task
        assert not sent
        assert any(e['status']=='discarded' for e in session.ws.events)
        await session.close()
    asyncio.run(run())


def test_unclear_audio_blocks_backend_and_interrupts_with_nudge():
    async def run():
        session,backend,sent=fixture()
        session.remember('user','Walk forward.')
        await session.emit('tyto','nudge',text='Please reduce background noise.')
        await session.delegate({'id':'held','target':'client'})
        assert session.blocked and not backend.calls
        assert not session.history
        assert sent[0]['type']=='session.instructions.append'
        assert sent[0]['delegation_id'] is None
        assert sent[1]['type']=='session.thinking.append'
        assert any(e['type']=='clear_audio' for e in session.ws.events)
        # Clearing an old spoken answer must not reopen the gate before the nudge speaks.
        session.speaker_busy=True
        await session.set_speaker(False)
        assert session.blocked
        await session.close()
    asyncio.run(run())


def test_stop_cancels_pending_backend_without_sending_late_result():
    async def run():
        session,backend,sent=fixture();backend.release=asyncio.Event()
        session.remember('user','Think about my movement request.')
        await session.delegate({'id':'pending','target':'client'});await asyncio.sleep(0)
        await session.close();backend.release.set()
        assert session.backend_task.cancelled() and not sent
    asyncio.run(run())


def test_context_and_delegations_are_bounded():
    async def run():
        session,backend,sent=fixture()
        for n in range(100):session.remember('user' if n%2 else 'assistant','x'*1000)
        assert len(session.history)==40
        assert sum(len(m['content']) for m in session.snapshot())<=12000
        await session.delegate({'id':'one','target':'client'});await session.backend_task
        assert len(backend.calls[0])==12
        await session.close()
    asyncio.run(run())


def test_live_page_uses_existing_auth_origin_and_session_cleanup():
    closed=[]
    class Session:
        async def run(self,ws):
            await ws.send_json({'type':'ready'})
            assert (await ws.receive_json())['type']=='stop'
    @asynccontextmanager
    async def factory():
        try:yield Session()
        finally:closed.append(True)
    app=create_app(session_factory=factory,session_runner=live_conversation,index_asset='live.html',provider='gpt-live')
    with TestClient(app,base_url=ORIGIN) as client:
        assert 'Live voice trial' in client.get('/').text
        assert client.get('/live.js').status_code==200
        assert client.get('/live-mic.js').status_code==200
        with connect(client) as ws:
            authenticate(client,ws);ws.send_json({'type':'stop'})
            assert ws.receive()['type']=='websocket.close'
        assert closed==[True]
        with pytest.raises(WebSocketDisconnect):
            with connect(client,origin='https://example.org'):pytest.fail('Cross-origin connection accepted')


def test_speaker_gate_sends_silence_without_enhancement_or_tyto():
    async def run():
        session,backend,sent=fixture();session.listening=True;session.speaker_busy=True
        class Browser(Socket):
            def __init__(self):super().__init__();self.messages=iter([
                {'type':'websocket.receive','bytes':b'\x01\x02'*320},
                {'type':'websocket.receive','text':'{"type":"stop"}'}])
            async def receive(self):return next(self.messages)
        session.ws=Browser()
        # Missing focus/insight is intentional: suppressed input must bypass both.
        await session.receive_browser()
        import base64
        assert base64.b64decode(sent[0]['audio'])==bytes(640)
        await session.close()
    asyncio.run(run())


def test_backend_failure_returns_only_safe_message():
    async def run():
        session,backend,sent=fixture()
        async def broken(messages):raise RuntimeError('secret-private-provider-payload')
        backend.respond=broken;session.remember('user','Ask the backend.')
        await session.delegate({'id':'failed','target':'client'});await session.backend_task
        assert 'No action was executed' in sent[-1]['content']
        assert 'secret' not in str(sent)+str(session.ws.events)
        await session.close()
    asyncio.run(run())
