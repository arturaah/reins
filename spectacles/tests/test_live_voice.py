import asyncio
import json
import socket

import pytest
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
from types import SimpleNamespace

from spectacles import live_voice
from spectacles.plan_feed import serve_feed


@pytest.mark.parametrize('url', ['https://example.com', 'http://127.0.0.1@evil.test', 'http://localhost:8770/path'])
def test_relay_only_connects_to_local_voice_service(url):
    with pytest.raises(ValueError): live_voice.local_voice_url(url)


def test_relay_authenticates_privately_streams_pcm_and_closes_on_stop(monkeypatch):
    async def run():
        received = []; lens_events = []; disconnected = asyncio.Event()
        class Lens:
            async def send(self, text): lens_events.append(json.loads(text))
        async def provider(ws):
            received.append(json.loads(await ws.recv()))
            await ws.send(json.dumps({'type':'ready'}))
            received.append(json.loads(await ws.recv()))
            await ws.send(json.dumps({'type':'listening'}))
            try:
                async for raw in ws: received.append(raw)
            finally: disconnected.set()
        monkeypatch.setattr(live_voice, 'read_config', lambda url: {
            'provider':'gpt-live', 'output':'r1', 'token':'private-local-token'})
        async with serve(provider, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            relay = live_voice.LiveVoiceRelay(Lens(), f'http://127.0.0.1:{port}')
            await relay.start()
            for _ in range(100):
                if relay.listening: break
                await asyncio.sleep(.01)
            assert relay.listening
            pcm = b'\x01\x02' * 320
            await relay.audio(pcm)
            for _ in range(100):
                if pcm in received: break
                await asyncio.sleep(.01)
            assert received == [{'token':'private-local-token'}, {'type':'listen','sample_rate':16000}, pcm]
            assert 'private-local-token' not in json.dumps(lens_events)
            await relay.stop()
            await asyncio.wait_for(disconnected.wait(), 1)
            assert not relay.listening and relay.upstream is None
    asyncio.run(run())


def test_relay_rejects_browser_only_output_instead_of_losing_reply_audio(monkeypatch):
    async def run():
        events = []
        class Lens:
            async def send(self, text): events.append(json.loads(text)['event'])
        monkeypatch.setattr(live_voice, 'read_config', lambda url: {'provider':'gpt-live','output':'browser'})
        relay = live_voice.LiveVoiceRelay(Lens(), 'http://127.0.0.1:8770')
        await relay.start(); await relay.task
        assert events[0]['type'] == 'error'
        assert '--output r1' in events[0]['text']
        assert events[-1]['type'] == 'stopped'
    asyncio.run(run())


def test_plan_feed_multiplexes_audio_with_trajectories_and_does_not_enqueue_legacy_text(monkeypatch, tmp_path):
    async def run():
        audio = asyncio.Event(); disconnected = asyncio.Event()
        async def provider(ws):
            assert json.loads(await ws.recv()) == {'token':'local-test-token'}
            await ws.send(json.dumps({'type':'ready'}))
            assert json.loads(await ws.recv())['type'] == 'listen'
            await ws.send(json.dumps({'type':'listening'}))
            try:
                async for raw in ws:
                    assert raw == b'\x01\x02'*320
                    audio.set()
            finally: disconnected.set()
        class Inbox:
            def enqueue(self, *args): raise AssertionError('Legacy text bypassed GPT-Live')
        monkeypatch.setattr(live_voice, 'read_config', lambda url: {
            'provider':'gpt-live','output':'r1','token':'local-test-token'})
        feed = SimpleNamespace(path=tmp_path/'preview.json', current=lambda *args: '{"type":"trajectory"}')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        async with serve(provider,'127.0.0.1',0) as upstream:
            url=f'http://127.0.0.1:{upstream.sockets[0].getsockname()[1]}'
            task=asyncio.create_task(serve_feed(feed,'127.0.0.1',port,.01,voice=Inbox(),live_voice_url=url))
            try:
                for _ in range(100):
                    try: lens=await connect(f'ws://127.0.0.1:{port}'); break
                    except OSError: await asyncio.sleep(.01)
                async with lens:
                    await lens.send(json.dumps({'type':'voice_start','version':1,'sample_rate':16000,'id':'call-1'}))
                    while True:
                        event=json.loads(await asyncio.wait_for(lens.recv(),2))
                        if event.get('event',{}).get('type')=='listening':
                            assert event['id']=='call-1'; break
                    await lens.send(b'\x01\x02'*320); await asyncio.wait_for(audio.wait(),1)
                    await lens.send(json.dumps({'type':'voice_command','version':1,'id':'legacy','text':'move'}))
                    while True:
                        event=json.loads(await asyncio.wait_for(lens.recv(),2))
                        if event['type']=='voice_ack':
                            assert not event['accepted']; break
                    await lens.send(json.dumps({'type':'voice_stop','version':1,'id':'call-1'}))
                    await asyncio.wait_for(disconnected.wait(),2)
            finally:
                task.cancel(); await asyncio.gather(task,return_exceptions=True)
    asyncio.run(run())
