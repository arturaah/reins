import asyncio
import json
from types import SimpleNamespace
import numpy as np
import httpx
import pytest
from voice.acoustics import Endpoint, NoisePolicy
from voice.cascade import AudioPipeline
from voice.speech import CartesiaTTS, GeminiTTS, decode_wav, wav_bytes


class STT:
    model = 'fixture-stt'
    def __init__(self): self.calls = []
    async def transcribe(self, pcm): self.calls.append(pcm); return 'Wave your right hand.'


class TTS:
    model = 'fixture-tts'
    def __init__(self): self.calls = []
    async def speak(self, text): self.calls.append(text); return b'\0'*4800


class Acoustics:
    def __init__(self, interference=0, noise=0): self.interference, self.noise = interference, noise
    def process(self, pcm): return b'enhanced', {'interfering_speech':self.interference,'noise':self.noise}


class End:
    speech_ms = 500
    def close(self): pass


def test_stt_returns_text_without_a_conversational_call_or_tts():
    async def run():
        stt, tts = STT(), TTS()
        pipeline = AudioPipeline(stt,tts,Acoustics()); pipeline.endpoint = End()
        result = await pipeline.reply(pcm=b'original')
        assert result.heard == 'Wave your right hand.' and result.pcm == b'' and result.said == ''
        assert stt.calls == [b'enhanced'] and tts.calls == []
        assert set(result.details['timings']) == {'acoustics_s','stt_s'}
    asyncio.run(run())


def test_tts_reads_the_exact_existing_reply_without_stt():
    async def run():
        stt, tts = STT(), TTS()
        text = 'Preview ready. Please review the trajectory.'
        result = await AudioPipeline(stt,tts,None).reply(text=text)
        assert result.said == text and result.pcm
        assert tts.calls == [text] and not stt.calls
    asyncio.run(run())


@pytest.mark.parametrize('interference,noise,reason',[(.8,.9,'competing_speech'),(.1,.9,'background_noise')])
def test_clarification_never_transcribes_bad_audio_even_on_cooldown(interference,noise,reason):
    async def run():
        clock, stt, tts = [100], STT(), TTS()
        pipeline = AudioPipeline(stt,tts,Acoustics(interference,noise),NoisePolicy(clock=lambda:clock[0]))
        for expected in (True,False,True):
            pipeline.endpoint = End()
            previous = len(tts.calls)
            result = await pipeline.reply(pcm=b'original')
            assert result.details['blocked'] == reason and not result.heard
            assert bool(result.pcm) == expected
            assert len(tts.calls)-previous == expected
            clock[0] += 16
        assert not stt.calls
    asyncio.run(run())


def test_silence_never_transcribes_or_speaks():
    async def run():
        stt,tts=STT(),TTS()
        result=await AudioPipeline(stt,tts,Acoustics()).reply(pcm=b'\0'*16000)
        assert result.details['blocked'] == 'no_speech'
        assert not result.heard and not stt.calls and not tts.calls
    asyncio.run(run())


def test_endpoint_requires_speech_then_silence_and_handles_partial_chunks():
    endpoint=Endpoint()
    decisions=iter([True]*8+[False]*25)
    endpoint.engine=SimpleNamespace(is_speech=lambda *_:next(decisions))
    assert not endpoint.feed(b'\0'*959)
    assert not endpoint.feed(b'\0')
    assert not endpoint.feed(b'\0'*(960*31))
    assert endpoint.feed(b'\0'*960)
    assert endpoint.speech_ms==240 and endpoint.silence_ms==750


def test_cartesia_contract_and_output_bounds():
    async def run():
        captured=[]
        def handle(request):
            captured.append(json.loads(request.content))
            assert request.headers['Cartesia-Version']=='2026-08-14'
            return httpx.Response(200,content=b'\0'*4800)
        tts=CartesiaTTS('fixture-key','fixture-voice')
        await tts.client.aclose()
        tts.client=httpx.AsyncClient(transport=httpx.MockTransport(handle),headers={'Cartesia-Version':'2026-08-14'})
        try:
            assert len(await tts.speak('Exact text.'))==4800
            assert captured[0]['transcript']=='Exact text.' and captured[0]['voice']=='fixture-voice'
            assert captured[0]['output_format']=={'container':'raw','encoding':'pcm_s16le','sample_rate':24000}
        finally: await tts.close()
        for data in (b'',b'\0',b'\0'*(30*24000*2+2)):
            tts=CartesiaTTS('fixture-key','fixture-voice');await tts.client.aclose()
            tts.client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,content=data)))
            try:
                with pytest.raises(ValueError): await tts.speak('test')
            finally: await tts.close()
    asyncio.run(run())


def test_gemini_tts_sends_text_verbatim_and_decodes_expected_wav():
    async def run():
        import base64
        calls=[]
        async def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_audio=SimpleNamespace(data=base64.b64encode(wav_bytes(b'\0'*4800,24000)).decode()))
        tts=GeminiTTS.__new__(GeminiTTS)
        tts.model='fixture-tts'
        tts.client=SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create)))
        assert len(await tts.speak('Exact reply.'))==4800
        assert calls[0]['input'][0]['content'][0]['text']=='Exact reply.'
        assert calls[0]['generation_config']['speech_config']==[{'voice':'Charon'}]
        with pytest.raises(ValueError): decode_wav(wav_bytes(b'\0'*1000,16000))
    asyncio.run(run())


class LLM:
    model = 'gpt-5-mini'
    def __init__(self): self.calls=[]
    async def respond(self,text): self.calls.append(text); return 'Hello. We can test a conversation.'


def test_optional_chat_handles_speech_and_text_but_literal_tts_stays_literal():
    async def run():
        stt,tts,llm=STT(),TTS(),LLM()
        pipeline=AudioPipeline(stt,tts,Acoustics(),llm=llm)
        pipeline.endpoint=End()
        spoken=await pipeline.reply(pcm=b'original')
        assert spoken.heard=='Wave your right hand.'
        assert spoken.said=='Hello. We can test a conversation.'
        assert spoken.details['llm_model']=='gpt-5-mini'
        assert set(spoken.details['timings'])=={'acoustics_s','stt_s','llm_s','tts_s'}
        typed=await pipeline.chat('Hello')
        assert typed.heard=='Hello' and typed.said==spoken.said
        literal=await pipeline.reply(text='Read this exactly.')
        assert literal.said=='Read this exactly.'
        assert llm.calls==['Wave your right hand.','Hello']
        assert tts.calls[-1]=='Read this exactly.'
    asyncio.run(run())


def test_optional_chat_never_receives_flagged_audio_or_silence():
    async def run():
        for interference in (0,.9):
            llm=LLM()
            pipeline=AudioPipeline(STT(),TTS(),Acoustics(interference),llm=llm)
            result=await pipeline.reply(pcm=b'original')
            assert result.details['blocked'] in ('no_speech','competing_speech')
            assert llm.calls==[]
        with pytest.raises(ValueError): await AudioPipeline(STT(),TTS(),None).chat('Hello')
    asyncio.run(run())


def test_gpt5_mini_request_and_bounded_session_history():
    from voice.chat import OpenAIChat
    async def run():
        calls=[]
        async def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(status='completed',output_text='A short answer.')
        async def close(): pass
        chat=OpenAIChat.__new__(OpenAIChat)
        chat.model='gpt-5-mini';chat.history=[]
        chat.client=SimpleNamespace(responses=SimpleNamespace(create=create),close=close)
        for i in range(9): await chat.respond(f'Question {i}')
        assert len(chat.history)==12
        assert len(calls[-1]['input'])==13
        assert calls[-1]['input'][-3]['content']=='Question 7'
        assert all(c['model']=='gpt-5-mini' and c['store'] is False and 'tools' not in c for c in calls)
        assert calls[-1]['reasoning']=={'effort':'minimal'}
        assert 'Your configured language model is gpt-5-mini.' in calls[-1]['instructions']
        await chat.close()
        assert chat.history==[]
    asyncio.run(run())
