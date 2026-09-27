import asyncio
import json
import socket
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from spectacles import pairing, plan_feed

TOKEN = 'wireless-test-token-1234567890abcd'


@pytest.mark.parametrize('host', ['0.0.0.0', '192.168.1.5', '::'])
def test_lan_live_voice_requires_separate_pairing_token(host):
    with pytest.raises(ValueError, match='--pairing-file'):
        pairing.validate_bind(host, 'http://127.0.0.1:8770', None)
    pairing.validate_bind(host, 'http://127.0.0.1:8770', TOKEN)


def test_loopback_usb_remains_compatible():
    pairing.validate_bind('127.0.0.1', 'http://127.0.0.1:8770', None)
    pairing.validate_bind('0.0.0.0', None, None)  # existing trajectory-only feed


@pytest.mark.parametrize('content', ['', 'short', TOKEN + '!', 'é'*32, 'x'*131])
def test_invalid_token_files_fail_without_echoing_contents(tmp_path, content):
    path = tmp_path/'token'
    path.write_text(content)
    with pytest.raises(ValueError, match='Invalid pairing file') as error:
        pairing.load_token(path)
    assert not content or content not in str(error.value)


def test_token_creation_is_private_and_reused(tmp_path, monkeypatch, capsys):
    path = tmp_path/'token'
    monkeypatch.setattr('sys.argv', ['pairing', '--file', str(path)])
    pairing.main()
    token = pairing.load_token(path)
    assert len(token) == 32
    assert path.stat().st_mode & 0o777 == 0o600
    assert token in capsys.readouterr().out  # explicitly displayed for Lens pairing only
    pairing.main()
    assert pairing.load_token(path) == token
    capsys.readouterr()


@pytest.mark.parametrize('message', [
    {'type':'pair','version':1,'token':'wrong-but-long-enough-token-123456'},
    {'type':'pair','version':1,'token':''},
    {'type':'pair','version':2,'token':TOKEN},
    {'type':'voice_start','version':1,'id':'bypass','sample_rate':16000},
    {'type':'review_decision','version':1,'id':'bypass','decision':'approve'},
    {'type':'pair','version':1,'token':'é'*32},
    [], None, b'\x00\x00'*320,
])
def test_unpaired_clients_cannot_read_paths_or_open_voice_relay(tmp_path, monkeypatch, message):
    async def run():
        calls = []
        def forbidden(*args):
            calls.append(args)
            raise AssertionError('Unpaired client accessed the feed or voice')
        monkeypatch.setattr(plan_feed, 'LiveVoiceRelay', forbidden)
        feed = SimpleNamespace(path=tmp_path/'preview.json', current=forbidden)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        task = asyncio.create_task(plan_feed.serve_feed(feed, '127.0.0.1', port, .01,
            live_voice_url='http://127.0.0.1:8770', pairing_token=TOKEN))
        try:
            for _ in range(100):
                try:
                    lens = await connect(f'ws://127.0.0.1:{port}')
                    break
                except OSError:
                    await asyncio.sleep(.01)
            async with lens:
                assert json.loads(await lens.recv())['type'] == 'pairing_required'
                await lens.send(message if isinstance(message, bytes) else json.dumps(message))
                assert json.loads(await lens.recv()) == {'type':'pairing_result','version':1,'accepted':False}
                with pytest.raises(ConnectionClosed) as error:
                    await lens.recv()
                assert error.value.rcvd.code == 1008
                assert calls == []
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_silent_client_times_out_and_each_reconnection_must_pair_again():
    async def run():
        results = []
        async def handler(ws):
            results.append(await pairing.authenticate(ws, TOKEN, timeout=.05))
        async with serve(handler, '127.0.0.1', 0) as server:
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
            for pair in (True, False):
                async with connect(url) as lens:
                    assert json.loads(await lens.recv())['type'] == 'pairing_required'
                    if pair:
                        await lens.send(json.dumps({'type':'pair','version':1,'token':TOKEN}))
                    assert json.loads(await asyncio.wait_for(lens.recv(), 1))['accepted'] is pair
                    with pytest.raises(ConnectionClosed):
                        await lens.recv()
        assert results == [True, False]
    asyncio.run(run())
