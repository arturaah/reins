import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import numpy as np
import pytest
from voice.errors import SpeechError, provider_error, validate_transcript
from voice.live_stt import PCM24k, OpenAILiveSTT
from voice.acoustics import FocusStream
from voice.cascade import AudioPipeline
from voice.tests.test_pipeline import STT, TTS, LLM, Acoustics, End
from voice.tests.test_voice import client_for, connect, authenticate


@pytest.mark.parametrize('text,code',[('', 'empty_transcript'),(' \n ', 'empty_transcript'),('a'*1001,'transcript_too_long'),(None,'invalid_response')])
def test_transcription_diagnostics_distinguish_no_speech_and_bad_responses(text,code):
    with pytest.raises(SpeechError) as result: validate_transcript(text)
    assert result.value.code==code


def test_provider_errors_have_safe_actionable_codes():
    for status,code in [(401,'authentication'),(403,'model_access'),(404,'model_access'),(429,'rate_limit'),(500,'provider_error')]:
        error=RuntimeError('sk-secret-provider-payload'); error.status_code=status
        public=provider_error(error)
        assert public.code==code and 'secret' not in str(public)


def test_resampling_is_continuous_independent_of_capture_chunks():
    pcm=np.random.default_rng(2).integers(-30000,30000,16001,dtype=np.int16).tobytes()
    whole=PCM24k().feed(pcm)
    stream=PCM24k()
    chunks=b''.join(stream.feed(pcm[i:i+514]) for i in range(0,len(pcm),514))
    assert len(chunks)==len(whole)==48004
    np.testing.assert_allclose(np.frombuffer(chunks,'<i2'),np.frombuffer(whole,'<i2'),atol=1)


def test_focus_stream_retains_partial_block_tail_and_levels():
    stream=FocusStream(SimpleNamespace(focus=False,focus_level=None))
    pcm=np.arange(-15000,13001,dtype=np.int16).tobytes()
    output=b''.join(stream.feed(pcm[i:i+514]) for i in range(0,len(pcm),514))+stream.finish()
    assert output==pcm
    assert stream.levels()['input_rms_dbfs']==stream.levels()['output_rms_dbfs']
    stream.close()


def test_empty_transcript_keeps_socket_ready_and_never_prompts_llm():
    calls=[]
    @asynccontextmanager
    async def factory():
        stt, llm=STT(),LLM()
        async def transcribe(pcm): calls.append('stt'); raise SpeechError('empty_transcript')
        stt.transcribe=transcribe
        pipeline=AudioPipeline(stt,TTS(),Acoustics(),llm=llm)
        async def begin(): pipeline.endpoint=End()
        async def chunk(pcm): return False
        pipeline.begin_input=begin; pipeline.input_chunk=chunk
        try: yield pipeline
        finally: assert not llm.calls; await pipeline.close()
    with client_for(factory) as client,connect(client) as ws:
        authenticate(client,ws)
        for _ in range(2):
            ws.send_json({'type':'start','sample_rate':16000}); assert ws.receive_json()['type']=='recording'
            ws.send_bytes(b'\0'*3200); ws.send_json({'type':'end'})
            events=[]
            while True:
                event=ws.receive_json(); events.append(event)
                assert event['type']!='error'
                if event['type']=='ready':break
            assert any(e.get('code')=='empty_transcript' for e in events)
            assert any(e.get('notice') and e.get('blocked')=='empty_transcript' for e in events)
        assert calls==['stt','stt']
        ws.send_json({'type':'stop'})


class StreamingSTT:
    streaming=True
    model='fixture-live-stt'
    def __init__(self): self.chunks=[]; self.commits=self.aborts=0
    def set_partial_sink(self,sink): self.sink=sink
    async def start(self): pass
    async def append(self,pcm): self.chunks.append(pcm)
    async def finish(self): self.commits+=1; return 'Hello from live speech.'
    async def abort(self): self.aborts+=1


class StreamingAcoustics(Acoustics):
    focus=False; focus_level=None; vad='webrtc'
    def endpoint(self): return SimpleNamespace(speech_ms=500,feed=lambda pcm:False,close=lambda:None)
    def stream(self): return FocusStream(self)


def test_streaming_audio_flows_before_send_but_llm_waits_for_final():
    async def run():
        stt,llm=StreamingSTT(),LLM()
        pipeline=AudioPipeline(stt,TTS(),StreamingAcoustics(),llm=llm)
        await pipeline.begin_input()
        pcm=b'\1\0'*1600
        await pipeline.input_chunk(pcm)
        assert any(stt.chunks) and stt.commits==0 and not llm.calls
        result=await pipeline.reply(pcm=pcm)
        assert result.heard=='Hello from live speech.' and stt.commits==1
        assert b''.join(stt.chunks)==pcm
        assert llm.calls==['Hello from live speech.']
        await pipeline.close()
    asyncio.run(run())


def test_nudge_discards_streamed_transcription_before_commit_or_llm():
    async def run():
        stt,llm=StreamingSTT(),LLM()
        pipeline=AudioPipeline(stt,TTS(),StreamingAcoustics(),llm=llm)
        await pipeline.begin_input(); await pipeline.input_chunk(b'\1\0'*1600)
        async def pause(): pass
        pipeline.insight=SimpleNamespace(line='Please reduce the noise.',cause='noise',latest={},pause=pause,close=pause)
        result=await pipeline.reply(pcm=b'\1\0'*1600)
        assert stt.aborts==1 and stt.commits==0 and not llm.calls
        assert result.said=='Please reduce the noise.'
        await pipeline.close()
    asyncio.run(run())


def test_live_final_is_matched_to_committed_item_even_when_completion_arrives_first():
    async def run():
        stt=OpenAILiveSTT.__new__(OpenAILiveSTT)
        stt.fault=None;stt.waiters={};stt.completed={'old-item':'Wrong text'};stt.active=True
        async def commit():
            stt.completed['new-item']='Correct text'
            stt._resolve('input_audio_buffer.committed',SimpleNamespace(item_id='new-item'))
        stt.connection=SimpleNamespace(input_audio_buffer=SimpleNamespace(commit=commit))
        assert await stt.finish()=='Correct text' and stt.active is False
    asyncio.run(run())
