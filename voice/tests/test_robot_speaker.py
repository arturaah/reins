import asyncio
import base64
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from voice.errors import SpeechError
from voice.inbox_backend import InboxBackend
from voice.r1_worker import R1PCMClient
from voice.robot_speaker import RobotSpeaker
from voice.tests.test_live import fixture


def test_sdk_worker_only_uses_pcm_and_own_stream_stop(monkeypatch):
    calls = []
    class Client:
        def __init__(self, *args): calls.append(('client', args))
        def SetTimeout(self, value): pass
        def _SetApiVerson(self, value): pass
        def _RegistApi(self, api, priority): calls.append(('register', api))
        def _Call(self, api, params): calls.append((api, json.loads(params))); return 0, '{}'
        def _CallRequestWithParamAndBin(self, api, params, data):
            calls.append((api, json.loads(params), bytes(data))); return 0, '{}'
    monkeypatch.setitem(sys.modules, 'unitree_sdk2py.core.channel', SimpleNamespace(
        ChannelFactoryInitialize=lambda domain, iface: calls.append(('interface', iface))))
    monkeypatch.setitem(sys.modules, 'unitree_sdk2py.rpc.client', SimpleNamespace(Client=Client))
    client = R1PCMClient('test-interface')
    client.play(b'\x01\x02' * 320)
    client.stop(); client.stop()
    assert [c[1] for c in calls if c[0] == 'register'] == [1003, 1004, 1005]
    writes = [c for c in calls if c[0] in (1003, 1004)]
    assert len(writes) == 2
    assert writes[0][2] == b'\x01\x02' * 320
    assert writes[0][1]['app_name'] == writes[1][1]['app_name'] == client.app


def test_robot_playback_gates_immediately_does_not_stall_on_silence_and_clears():
    async def run():
        speaker = RobotSpeaker('unused', 'unused')
        events = []
        async def request(event): events.append(event)
        async def gate(busy): events.append(busy)
        speaker._request = request; speaker.on_busy = gate
        await speaker.play(bytes(640))
        assert not events
        pcm = np.full(320, 2000, '<i2').tobytes()
        await speaker.play(pcm)
        assert events[0] is True and base64.b64decode(events[1]['audio']) == pcm
        until = speaker.until
        await speaker.play(bytes(640))
        assert speaker.until == until
        await speaker.clear()
        assert events[-2:] == [{'type': 'stop'}, False]
        assert speaker.until == 0
    asyncio.run(run())


def test_robot_playback_stops_instead_of_building_unbounded_delay():
    async def run():
        speaker = RobotSpeaker('unused', 'unused')
        async def request(event): pass
        speaker._request = request
        pcm = np.full(16000, 2000, '<i2').tobytes()
        await speaker.play(pcm); await speaker.play(pcm)
        with pytest.raises(SpeechError, match='fell behind'):
            await speaker.play(pcm)
    asyncio.run(run())


def test_idle_robot_playback_reopens_input_after_tail():
    async def run():
        speaker = RobotSpeaker('unused', 'unused')
        speaker.busy = True
        cleared = asyncio.Event(); requests = []
        async def request(event): requests.append(event)
        async def gate(busy):
            if not busy: cleared.set()
        speaker._request = request; speaker.on_busy = gate
        monitor = asyncio.create_task(speaker.monitor())
        try:
            await asyncio.wait_for(cleared.wait(), 1)
            assert requests == [{'type':'stop'}] and not speaker.busy
        finally:
            monitor.cancel(); await asyncio.gather(monitor, return_exceptions=True)
    asyncio.run(run())


def test_live_output_goes_to_robot_and_nudge_stops_it():
    async def run():
        session, backend, sent = fixture()
        played = []
        class Speaker:
            async def play(self, pcm): played.append(pcm)
            async def clear(self): played.append('stop')
            async def close(self): played.append('closed')
        session.speaker = Speaker()
        pcm = np.full(320, 2000, '<i2').tobytes()
        async def upstream():
            yield json.dumps({'type':'session.output_audio.delta', 'delta':base64.b64encode(pcm).decode()})
        session.upstream = upstream()
        await session.receive_upstream()
        assert len(played) == 1 and len(played[0]) == len(pcm)
        await session.emit('tyto', 'nudge', text='Please reduce background noise.')
        assert played[-1] == 'stop'
        session.upstream = None
        await session.close()
        assert played[-1] == 'closed'
    asyncio.run(run())


def test_delegated_task_uses_existing_dry_run_inbox_and_deduplicates(tmp_path):
    async def run():
        backend = InboxBackend(tmp_path / 'inbox.json')
        messages = [{'role':'user','content':'Walk one metre forward.'}]
        assert 'queued' in await backend.respond(messages)
        command = backend.inbox.take()
        assert command['text'] == messages[0]['content']
        assert 'already queued' in await backend.respond(messages)
        assert backend.inbox.take() is None
        assert 'Nothing was queued' in await backend.respond([{'role':'user','content':'x'*501}])
    asyncio.run(run())
