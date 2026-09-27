import asyncio
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
import numpy as np
import pytest
from voice.insight import LiveTyto, SCORES
from voice.tyto_nudge import TytoNudger, CAUSES
from voice.cascade import AudioPipeline
from voice.tests.test_pipeline import STT, TTS, LLM, Acoustics, End
from voice.tests.test_voice import client_for, connect, authenticate, read_reply


def reading(**values):
    return {**dict.fromkeys(SCORES, 0.), **values}


def test_supplied_policy_smoothing_episodes_cooldown_and_nonfixable_causes():
    now = [100.]
    policy = TytoNudger(clock=lambda: now[0])
    clean, noisy = reading(), reading(risk_score=.9, noise=.9)
    assert policy.on_result(clean) is None
    assert policy.on_result(noisy) is None  # One spike after clean audio is smoothed out.
    assert policy.on_result(clean) is None
    lines = [policy.on_result(noisy) for _ in range(8)]
    assert [line for line in lines if line] == [CAUSES['noise'][1]]
    for _ in range(10): assert policy.on_result(clean) is None
    for _ in range(10): assert policy.on_result(noisy) is None  # Same cause still on cooldown.
    policy.reset()  # Playback resets smoothing, but not the cause's cooldown.
    assert policy.on_result(noisy) is None
    now[0] += 31
    assert policy.on_result(noisy) == CAUSES['noise'][1]
    for cause in ('codec_degradation', 'speaker_reverb', 'speaker_loudness'):
        policy.reset()
        assert policy.on_result(reading(risk_score=.99, **{cause:1})) is None
    policy.reset()
    assert policy.on_result(reading(risk_score=.2, noise=1)) is None


class Collector:
    def __init__(self): self.blocks=[]
    def buffer(self, block): self.blocks.append(block.copy())


class Analyzer:
    def __init__(self, result=None):
        self.result=SimpleNamespace(**(result or reading()))
        self.calls=self.resets=self.closed=0
    def analyze_buffered(self): self.calls+=1; return self.result
    def reset(self): self.resets+=1
    def terminate_session(self): self.closed+=1


def test_collector_exact_blocks_warmup_pause_reset_and_original_samples():
    async def run():
        events=[]
        async def emit(stage,status,**data): events.append(dict(stage=stage,status=status,**data))
        collector, analyzer = Collector(), Analyzer()
        insight = LiveTyto(collector,analyzer,240,emit)
        await insight.reset()
        pcm = (np.arange(80160) % 30000).astype('<i2').tobytes()
        for offset in range(0,159998,3074):
            insight.feed(pcm[offset:min(offset+3074,159998)])
        assert insight.task is None
        insight.feed(pcm[159998:])
        await insight.pause()
        assert analyzer.calls==1
        assert all(block.shape==(240,) and block.dtype==np.float32 for block in collector.blocks)
        np.testing.assert_array_equal(np.concatenate(collector.blocks),np.frombuffer(pcm,'<i2')/32768)
        assert events[0]['status']=='reading' and events[0]['smoothed']['risk_score']==0
        before=len(collector.blocks)
        insight.feed(pcm)
        assert len(collector.blocks)==before  # Agent playback must not enter the collector.
        await insight.playback_finished()
        assert insight.samples==0 and insight.nudger.smoothed is None and not insight.active
        await insight.reset()
        insight.feed(pcm[:32000]); assert insight.task is None or insight.task.done()
        await insight.close(); assert analyzer.closed==1
    asyncio.run(run())


def test_analysis_never_queues_and_close_waits_for_native_worker():
    async def run():
        entered, release = threading.Event(), threading.Event()
        events=[]
        async def emit(*args,**kwargs): events.append((args,kwargs))
        class Slow(Analyzer):
            def analyze_buffered(self):
                entered.set(); release.wait(2); return super().analyze_buffered()
        analyzer=Slow(); insight=LiveTyto(Collector(),analyzer,240,emit)
        await insight.reset(); insight.feed(b'\0'*160320)
        assert await asyncio.to_thread(entered.wait,1)
        task=insight.task
        insight.feed(b'\0'*320640)
        assert insight.task is task
        closing=asyncio.create_task(insight.close())
        await asyncio.sleep(.01); assert not closing.done() and analyzer.closed==0
        release.set(); await closing
        assert analyzer.calls==1 and analyzer.closed==1
    asyncio.run(run())


def test_stage_logs_arrive_before_downstream_completion_and_sanitize_errors():
    async def run():
        events=[]; entered=asyncio.Event(); release=asyncio.Event()
        async def emit(event): events.append(event)
        class SlowLLM(LLM):
            async def respond(self,text): entered.set(); await release.wait(); return await super().respond(text)
        pipeline=AudioPipeline(STT(),TTS(),Acoustics(),llm=SlowLLM()); pipeline.endpoint=End()
        pipeline.set_event_sink(emit)
        task=asyncio.create_task(pipeline.reply(pcm=b'original'))
        await entered.wait()
        assert any(e['stage']=='stt' and e['status']=='completed' and e['text']=='Wave your right hand.' for e in events)
        assert not any(e['stage']=='tts' for e in events)
        release.set(); await task
        assert [(e['stage'],e['status']) for e in events][-2:]==[('tts','started'),('tts','completed')]
        assert events[-1]['audio_s']==.1
        async def broken(_): raise RuntimeError('secret-key and private-provider-payload')
        pipeline.tts.speak=broken; events.clear()
        with pytest.raises(RuntimeError): await pipeline.reply(text='Exact line')
        assert events[-1]['status']=='error'
        assert 'secret-key' not in json.dumps(events) and 'private-provider' not in json.dumps(events)
    asyncio.run(run())


def test_failed_analysis_closes_vad_and_never_logs_provider_payload():
    async def run():
        events=[]; closed=[]
        async def emit(stage,status,**data): events.append(dict(stage=stage,status=status,**data))
        class Broken(Analyzer):
            def analyze_buffered(self): raise RuntimeError('secret-license and private payload')
        analyzer=Broken(); insight=LiveTyto(Collector(),analyzer,240,emit)
        pipeline=AudioPipeline(STT(),TTS(),Acoustics())
        pipeline.insight=insight; pipeline.endpoint=SimpleNamespace(close=lambda:closed.append(True))
        await insight.reset(); insight.feed(b'\0'*160320)
        with pytest.raises(RuntimeError): await pipeline.close()
        assert closed==[True] and analyzer.closed==1
        assert events[-1]['status']=='error' and insight.latest is None
        assert 'secret-license' not in json.dumps(events) and 'private payload' not in json.dumps(events)
    asyncio.run(run())


def test_live_nudge_stops_before_stt_or_llm_and_speaks_verbatim_over_websocket():
    stt,tts,llm=STT(),TTS(),LLM()
    remembered=[]; llm.remember_spoken=remembered.append
    @asynccontextmanager
    async def factory():
        pipeline=AudioPipeline(stt,tts,Acoustics(),llm=llm)
        pipeline.insight=LiveTyto(Collector(),Analyzer(reading(risk_score=.9,noise=.9)),240,pipeline.emit)
        async def begin(): pipeline.endpoint=End(); await pipeline.insight.reset()
        async def chunk(pcm): pipeline.insight.feed(pcm); return False
        pipeline.begin_input=begin; pipeline.input_chunk=chunk
        try: yield pipeline
        finally: await pipeline.close()
    with client_for(factory) as client,connect(client) as ws:
        authenticate(client,ws)
        ws.send_json({'type':'start','sample_rate':16000}); assert ws.receive_json()['type']=='recording'
        for _ in range(10): ws.send_bytes(b'\1\0'*8016)
        events=[]
        while True:
            event=ws.receive_json(); events.append(event)
            if event['type']=='nudge': break
            assert event['type']!='error'
        assert events[0]['stage']=='tyto' and events[0]['status']=='reading'
        assert events[0]['turn']==1 and events[0]['timestamp']>0
        # Already-in-flight capture frames are discarded, then browser confirms capture ended.
        ws.send_bytes(b'\1\0'*2048); ws.send_json({'type':'end'})
        messages,pcm=read_reply(ws)
        assert not stt.calls and not llm.calls
        assert tts.calls==remembered==[CAUSES['noise'][1]]
        assert any(e.get('blocked')=='tyto_nudge' for e in messages)
        ws.send_json({'type':'played'}); assert ws.receive_json()['type']=='ready'
        ws.send_json({'type':'stop'})
