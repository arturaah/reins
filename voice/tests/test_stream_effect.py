import asyncio
import base64
import json

import numpy as np
import pytest

from voice.audio import RoboticStream, robotic_pcm
from voice.tests.test_live import fixture


@pytest.mark.parametrize('rate',[16000,24000])
def test_stream_effect_is_independent_of_chunk_boundaries_and_never_buffers(rate):
    pcm=np.random.default_rng(4).integers(-30000,30000,rate*3+71,dtype=np.int16).tobytes()
    whole=RoboticStream(rate=rate).process(pcm)
    stream=RoboticStream(rate=rate);parts=[]
    for offset in range(0,len(pcm),634):
        chunk=pcm[offset:offset+634];output=stream.process(chunk)
        assert len(output)==len(chunk)
        parts.append(output)
    np.testing.assert_allclose(np.frombuffer(b''.join(parts),'<i2'),np.frombuffer(whole,'<i2'),atol=1)


def test_stream_effect_matches_existing_charon_processing_at_24khz():
    t=np.arange(48000)/24000
    pcm=(np.sin(2*np.pi*180*t)*22000+np.sin(2*np.pi*2200*t)*5000).astype('<i2').tobytes()
    np.testing.assert_allclose(np.frombuffer(RoboticStream().process(pcm),'<i2'),
                               np.frombuffer(robotic_pcm(pcm),'<i2'),atol=1)


@pytest.mark.parametrize('metallic',[True,False])
def test_live_relay_delivers_first_audio_before_later_chunks_arrive(metallic):
    async def run():
        session,backend,sent=fixture()
        if not metallic:session.output_effect=None
        chunk=np.arange(-3200,3200,dtype=np.int16).tobytes()
        proceed=asyncio.Event();delivered=asyncio.Event();received=[]
        async def upstream():
            yield json.dumps({'type':'session.output_audio.delta','delta':base64.b64encode(chunk).decode()})
            await proceed.wait()
            yield json.dumps({'type':'session.closed'})
        async def send_bytes(pcm):received.append(pcm);delivered.set()
        session.ws.send_bytes=send_bytes;session.upstream=upstream()
        task=asyncio.create_task(session.receive_upstream())
        await asyncio.wait_for(delivered.wait(),1)
        assert not task.done() and len(received)==1 and len(received[0])==len(chunk)
        assert (received[0]!=chunk) is metallic
        proceed.set();await task
        session.upstream=None;await session.close()
    asyncio.run(run())
