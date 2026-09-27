import asyncio
from contextlib import asynccontextmanager
import threading
from pathlib import Path
import ast
import numpy as np
import pytest
from fastapi.testclient import TestClient
from voice.audio import robotic_pcm
from voice.providers import Demo, Reply
from voice.server import create_app

ORIGIN = 'http://127.0.0.1:8770'


def client_for(factory=None):
    options = {'session_factory': factory} if factory else {}
    return TestClient(create_app(provider='demo', **options), base_url=ORIGIN)


def connect(client, token=None, origin=ORIGIN):
    ws = client.websocket_connect(ORIGIN.replace('http:', 'ws:') + '/voice', headers={'origin': origin})
    return ws


def authenticate(client, ws):
    ws.send_json({'token': client.get('/config').json()['token']})
    assert ws.receive_json()['type'] == 'connecting'
    assert ws.receive_json()['type'] == 'ready'


def read_reply(ws):
    messages = []
    while True:
        item = ws.receive()
        if 'bytes' in item:
            return messages, item['bytes']
        messages.append(__import__('json').loads(item['text']))
        assert messages[-1]['type'] != 'error', messages


def test_demo_two_turns_and_explicit_playback_ack():
    with client_for() as client, connect(client) as ws:
        authenticate(client, ws)
        for action in ({'type':'hello'}, {'type':'text', 'text':'Move your arm'}):
            ws.send_json(action)
            messages, pcm = read_reply(ws)
            assert messages[0]['type'] == 'thinking'
            assert messages[-1] == {'type':'audio', 'sample_rate':24000}
            assert len(pcm) == 24000
            assert any('not an AI reply' in x.get('text','') for x in messages)
            ws.send_json({'type':'played'})
            assert ws.receive_json()['type'] == 'ready'
        ws.send_json({'type':'stop'})
        assert ws.receive()['type'] == 'websocket.close'


def test_microphone_delivered_only_after_send_and_provider_closed():
    calls, closed = [], threading.Event()
    @asynccontextmanager
    async def factory(*_):
        class Provider(Demo):
            async def reply(self, **kwargs):
                calls.append(kwargs)
                return await super().reply(**kwargs)
        try: yield Provider()
        finally: closed.set()
    with client_for(factory) as client, connect(client) as ws:
        authenticate(client, ws)
        ws.send_json({'type':'start', 'sample_rate':16000})
        assert ws.receive_json()['type'] == 'recording'
        ws.send_bytes(b'\x10\0' * 1600)
        assert calls == []
        ws.send_json({'type':'end'})
        read_reply(ws)
        assert calls == [{'pcm':b'\x10\0' * 1600, 'text':None}]
        ws.send_json({'type':'stop'})
        ws.receive()
    assert closed.wait(2)


@pytest.mark.parametrize('action,chunk', [
    ({'type':'execute'}, None),
    ({'type':'played'}, None),
    ({'type':'start','sample_rate':48000}, None),
    ({'type':'end'}, None),
    (None, b'\0\0'),
    ({'type':'start','sample_rate':16000}, b'\0'),
    ({'type':'start','sample_rate':16000}, b'\0' * 16386),
])
def test_invalid_turns_never_reach_provider(action, chunk):
    @asynccontextmanager
    async def factory(*_):
        class NoCalls:
            async def reply(self, **_): pytest.fail('Invalid input reached provider')
        yield NoCalls()
    with client_for(factory) as client, connect(client) as ws:
        authenticate(client, ws)
        if action:
            ws.send_json(action)
            if action == {'type':'start','sample_rate':16000}:
                assert ws.receive_json()['type'] == 'recording'
        if chunk is not None: ws.send_bytes(chunk)
        assert ws.receive_json()['type'] == 'error'
        assert ws.receive()['type'] == 'websocket.close'


def test_recording_size_bound():
    with client_for() as client, connect(client) as ws:
        authenticate(client, ws)
        ws.send_json({'type':'start','sample_rate':16000}); ws.receive_json()
        for _ in range(60): ws.send_bytes(b'\0' * 16384)
        assert ws.receive_json()['type'] == 'error'


def test_stop_cancels_provider_and_releases_single_session():
    entered, cancelled, closed = threading.Event(), threading.Event(), threading.Event()
    @asynccontextmanager
    async def factory(*_):
        class Slow:
            async def reply(self, **_):
                entered.set()
                try: await asyncio.Event().wait()
                finally: cancelled.set()
        try: yield Slow()
        finally: closed.set()
    with client_for(factory) as client:
        with connect(client) as ws:
            authenticate(client, ws); ws.send_json({'type':'hello'})
            assert ws.receive_json()['type'] == 'thinking'
            assert entered.wait(2)
            with connect(client) as second:
                second.send_json({'token':client.get('/config').json()['token']})
                assert 'Another' in second.receive_json()['text']
            ws.send_json({'type':'stop'}); ws.receive()
        assert cancelled.wait(2) and closed.wait(2)
        with connect(client) as next_ws:
            authenticate(client, next_ws)
            next_ws.send_json({'type':'stop'})


def test_foreign_origin_host_and_wrong_token_rejected_before_provider():
    @asynccontextmanager
    async def forbidden(*_):
        pytest.fail('Unauthenticated provider access')
        yield
    with client_for(forbidden) as client:
        assert client.get('/config', headers={'host':'evil.example'}).status_code == 403
        with pytest.raises(Exception):
            with connect(client, origin='https://evil.example'): pass
        with connect(client) as ws:
            ws.send_json({'token':'wrong'})
            assert ws.receive()['code'] == 1008
        assert client.get('/config').json()['mode'] == 'sim'
        assert 'key' not in client.get('/config').json()


def test_provider_failure_does_not_leak_credentials():
    @asynccontextmanager
    async def factory(*_):
        class Broken:
            async def reply(self, **_): raise RuntimeError('secret-key?transcript=private')
        yield Broken()
    with client_for(factory) as client, connect(client) as ws:
        authenticate(client, ws); ws.send_json({'type':'hello'})
        assert ws.receive_json()['type'] == 'thinking'
        error = ws.receive_json()
        assert error['type'] == 'error' and 'secret' not in str(error) and 'private' not in str(error)


def test_effect_preserves_duration_and_reduces_peak():
    tone = (np.sin(2*np.pi*1000*np.arange(24000)/24000)*30000).astype('<i2')
    output = robotic_pcm(tone.tobytes())
    assert len(output) == len(tone)*2
    peak = np.abs(np.frombuffer(output,dtype='<i2')).max()
    assert 1000 < peak < 16000
    for invalid in (b'', b'\0', b'\0'*(30*24000*2+2)):
        with pytest.raises(ValueError): robotic_pcm(invalid)


def test_conversation_has_no_motion_or_sdk_dependencies():
    root = Path(__file__).parents[1]
    for path in root.glob('*.py'):
        tree = ast.parse(path.read_text())
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        assert not any(name.startswith(('unitree','cyclonedds','harness','core','tools','subprocess')) for name in imports)


def test_transcription_without_playback_rearms_connection():
    @asynccontextmanager
    async def factory():
        class Transcribe:
            async def reply(self, **_): return Reply(b'', '', 'Wave your right hand.')
        yield Transcribe()
    with client_for(factory) as client, connect(client) as ws:
        authenticate(client, ws)
        for _ in range(2):
            ws.send_json({'type':'start','sample_rate':16000}); ws.receive_json()
            ws.send_bytes(b'\0'*3200); ws.send_json({'type':'end'})
            assert ws.receive_json()['type']=='thinking'
            assert ws.receive_json()=={'type':'transcript','role':'you','text':'Wave your right hand.'}
            assert ws.receive_json()['type']=='ready'
        ws.send_json({'type':'stop'})
