"""STT/TTS endpoints around the team's text UI. No conversational LLM or robot control."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import time
from .acoustics import NoisePolicy
from .providers import Reply

CLARIFICATIONS = {
    'competing_speech': 'I hear competing voices. Could one person speak at a time so I know what to do next?',
    'background_noise': 'The background noise is making it hard to hear you. Could you move closer or reduce the noise?',
}


class AudioPipeline:
    """Inject async transcribe(pcm) and speak(text) adapters. Audio capture feeds VAD separately."""
    def __init__(self, stt, tts, acoustics, noise=None):
        self.stt, self.tts, self.acoustics = stt, tts, acoustics
        self.noise = noise or NoisePolicy()
        self.endpoint = None

    async def begin_input(self):
        await self.end_input()
        self.endpoint = await asyncio.to_thread(self.acoustics.endpoint)

    async def input_chunk(self, pcm):
        return await asyncio.to_thread(self.endpoint.feed, pcm)

    async def end_input(self):
        if self.endpoint:
            endpoint, self.endpoint = self.endpoint, None
            await asyncio.to_thread(endpoint.close)

    async def reply(self, *, pcm=None, text=None):
        timings, info = {}, {}
        if pcm is not None:
            speech_ms = self.endpoint.speech_ms if self.endpoint else 0
            await self.end_input()
            t = time.monotonic()
            pcm, info = await asyncio.to_thread(self.acoustics.process, pcm)
            timings['acoustics_s'] = round(time.monotonic() - t, 3)
            reason, speak = self.noise.check(info)
            if reason:
                said = CLARIFICATIONS[reason]
                t = time.monotonic()
                output = await self.tts.speak(said) if speak else b''
                timings['tts_s'] = round(time.monotonic() - t, 3)
                return Reply(output, said if speak else 'Waiting for clearer audio; spoken prompt is on cooldown.',
                             details={'blocked': reason, 'acoustics': info, 'timings': timings})
            if speech_ms < 240:
                return Reply(b'', 'No clear speech detected. Please try again.',
                             details={'blocked': 'no_speech', 'acoustics': info, 'timings': timings})
            t = time.monotonic()
            heard = await self.stt.transcribe(pcm)
            if not isinstance(heard, str) or not 1 <= len(heard.strip()) <= 1000:
                raise ValueError('Empty or oversized transcription')
            timings['stt_s'] = round(time.monotonic() - t, 3)
            return Reply(b'', '', heard,
                         {'timings': timings, 'acoustics': info, 'stt_model': self.stt.model})
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError('Expected 1–1000 characters of speakable text')
        t = time.monotonic()
        output = await self.tts.speak(text)
        timings['tts_s'] = round(time.monotonic() - t, 3)
        return Reply(output, text, details={'timings': timings, 'tts_model': self.tts.model})


def session_factory(*, make_stt, make_tts, acoustics, noise_threshold=.6, background_threshold=.8):
    @asynccontextmanager
    async def session():
        async with AsyncExitStack() as stack:
            adapters = []
            for make in (make_stt, make_tts):
                adapter = make()
                stack.push_async_callback(adapter.close)
                adapters.append(adapter)
            pipeline = AudioPipeline(*adapters, acoustics,
                                     NoisePolicy(threshold=noise_threshold, noise_threshold=background_threshold))
            stack.push_async_callback(pipeline.end_input)
            yield pipeline
    return session
