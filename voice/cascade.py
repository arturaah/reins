"""STT/TTS endpoints with optional test conversation; no robot control."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import time
from .acoustics import NoisePolicy, FOCUS_MODEL, TYTO_MODEL
from .providers import Reply
from .errors import SpeechError, validate_transcript

CLARIFICATIONS = {
    'competing_speech': 'I hear competing voices. Could one person speak at a time so I know what to do next?',
    'background_noise': 'The background noise is making it hard to hear you. Could you move closer or reduce the noise?',
}


class AudioPipeline:
    """Inject async transcribe(pcm) and speak(text) adapters. Audio capture feeds VAD separately."""
    def __init__(self, stt, tts, acoustics, noise=None, llm=None, *, live_tyto=False):
        self.stt, self.tts, self.acoustics = stt, tts, acoustics
        self.noise = noise or NoisePolicy()
        self.endpoint = None
        self.llm = llm
        self.live_tyto_enabled, self.insight, self.event_sink = live_tyto, None, None
        self.focus_stream = None

    def set_event_sink(self, sink):
        self.event_sink = sink

    async def emit(self, stage, status, **details):
        if self.event_sink:
            await self.event_sink({'stage': stage, 'status': status, **details})

    @asynccontextmanager
    async def stage(self, name, **details):
        started, result = time.monotonic(), {}
        await self.emit(name, 'started', **details)
        try:
            yield result
        except SpeechError as error:
            await self.emit(name, 'no_speech' if error.code == 'empty_transcript' else 'error',
                            code=error.code, text=error.public_text,
                            elapsed_s=round(time.monotonic() - started, 3))
            raise
        except Exception as error:
            await self.emit(name, 'error', error_kind=type(error).__name__,
                            text=f'{name.upper()} failed; check provider access and try again.')
            raise
        else:
            result['elapsed_s'] = round(time.monotonic() - started, 3)
            await self.emit(name, 'completed', **details, **result)

    async def begin_input(self):
        await self.end_input()
        await self.close_focus()
        if self.live_tyto_enabled:
            if self.insight is None:
                from .insight import make_live_tyto
                self.insight = await asyncio.to_thread(make_live_tyto, self.acoustics, self.emit)
            await self.insight.reset()
            await self.emit('tyto', 'warming_up', model=TYTO_MODEL, text='Needs 5 seconds of original microphone audio.')
        self.endpoint = await asyncio.to_thread(self.acoustics.endpoint)
        if getattr(self.stt, 'streaming', False):
            self.focus_stream = await asyncio.to_thread(self.acoustics.stream)
            self.stt.set_partial_sink(self.partial_transcript)
            await self.stt.start()
            await self.emit('voice_focus', 'started', model=FOCUS_MODEL, enabled=self.acoustics.focus,
                            enhancement_level=self.acoustics.focus_level, text='Enhancing microphone chunks as they arrive.')
            await self.emit('stt', 'streaming', model=self.stt.model, text='Transcribing while you speak; waiting for the final accepted turn.')

    async def partial_transcript(self, text):
        # Preview only. Neither the planner bridge nor LLM receives provisional text.
        await self.emit('stt', 'partial', model=self.stt.model, text=text)

    async def input_chunk(self, pcm):
        if self.insight:
            self.insight.feed(pcm)
        endpoint = await asyncio.to_thread(self.endpoint.feed, pcm)
        if self.focus_stream and not (self.insight and self.insight.line):
            enhanced = await asyncio.to_thread(self.focus_stream.feed, pcm)
            await self.stt.append(enhanced)
        return endpoint

    async def end_input(self):
        try:
            if self.insight:
                await self.insight.pause()
        finally:
            if self.endpoint:
                endpoint, self.endpoint = self.endpoint, None
                await asyncio.to_thread(endpoint.close)

    async def playback_finished(self):
        if self.insight:
            await self.insight.playback_finished()

    async def close(self):
        try:
            await self.end_input()
        finally:
            try:
                await self.close_focus()
            finally:
                if self.insight:
                    await self.insight.close()

    async def close_focus(self):
        if self.focus_stream:
            stream, self.focus_stream = self.focus_stream, None
            await asyncio.to_thread(stream.close)

    async def speak(self, text):
        async with self.stage('tts', model=self.tts.model, text=text) as log:
            output = await self.tts.speak(text)
            log.update(audio_s=round(len(output) / 48000, 3), bytes=len(output))
        return output, log['elapsed_s']

    async def reply(self, *, pcm=None, text=None):
        timings, info = {}, {}
        if pcm is not None:
            speech_ms = self.endpoint.speech_ms if self.endpoint else 0
            await self.end_input()
            if self.insight and self.insight.line:
                if self.focus_stream:
                    await self.stt.abort()
                    await self.close_focus()
                    await self.emit('stt', 'discarded', text='Tyto nudge: provisional transcription discarded; no LLM request.')
                said = self.insight.line
                output, timings['tts_s'] = await self.speak(said)
                if self.llm and hasattr(self.llm, 'remember_spoken'):
                    self.llm.remember_spoken(said)
                return Reply(output, said, details={'blocked': 'tyto_nudge', 'cause': self.insight.cause,
                             'acoustics': {'tyto': 'scored', **self.insight.latest}, 'timings': timings})
            streaming = self.focus_stream is not None
            if streaming:
                tail = await asyncio.to_thread(self.focus_stream.finish)
                await self.stt.append(tail)
                timings['acoustics_s'] = round(self.focus_stream.processing_s, 3)
                info = {'voice_focus': self.acoustics.focus, 'enhancement_level': self.acoustics.focus_level,
                        'vad': self.acoustics.vad, **self.focus_stream.levels()}
                await self.emit('voice_focus', 'completed', model=FOCUS_MODEL, enabled=self.acoustics.focus,
                                enhancement_level=self.acoustics.focus_level,
                                elapsed_s=timings['acoustics_s'], **self.focus_stream.levels())
                await self.close_focus()
            else:
                t = time.monotonic()
                focus = getattr(self.acoustics, 'focus', False)
                async with self.stage('voice_focus', model=FOCUS_MODEL, enabled=focus,
                                      enhancement_level=getattr(self.acoustics, 'focus_level', None)) as focus_log:
                    kwargs = {'score_tyto': False} if self.insight else {}
                    pcm, info = await asyncio.to_thread(self.acoustics.process, pcm, **kwargs)
                    focus_log.update({key:info[key] for key in ('input_rms_dbfs','output_rms_dbfs') if key in info})
                timings['acoustics_s'] = round(time.monotonic() - t, 3)
            if self.insight:
                info.update(tyto='scored' if self.insight.latest else 'needs_5_seconds', **(self.insight.latest or {}))
                if not self.insight.latest:
                    await self.emit('tyto', 'unscored', text='Turn ended before a full 5-second window.')
            elif info.get('tyto') == 'scored':
                await self.emit('tyto', 'reading', model=TYTO_MODEL,
                                scores={k: info[k] for k in ('risk_score', 'noise', 'interfering_speech')},
                                text='Legacy end-of-turn analysis; live nudge disabled.')
            # The live nudge policy replaces the legacy raw-score interruption rule.
            reason, speak = (None, False) if self.insight else self.noise.check(info)
            if reason:
                said = CLARIFICATIONS[reason]
                await self.emit('tyto', 'blocked', cause=reason, text=said if speak else 'Spoken clarification on cooldown.')
                output, timings['tts_s'] = await self.speak(said) if speak else (b'', 0)
                return Reply(output, said if speak else 'Waiting for clearer audio; spoken prompt is on cooldown.',
                             details={'blocked': reason, 'acoustics': info, 'timings': timings})
            if speech_ms < 240:
                if streaming: await self.stt.abort()
                await self.emit('vad', 'no_speech', text='No clear speech; STT skipped.')
                return Reply(b'', 'No clear speech detected. Please try again.',
                             details={'blocked': 'no_speech', 'acoustics': info, 'timings': timings})
            try:
                async with self.stage('stt', model=self.stt.model, audio_s=round(len(pcm) / 32000, 3)) as log:
                    heard = validate_transcript(await self.stt.finish() if streaming else await self.stt.transcribe(pcm))
                    log['text'] = heard
            except SpeechError as error:
                if not error.recoverable:
                    raise
                return Reply(b'', '', details={'blocked': error.code, 'notice': error.public_text,
                             'timings': timings, 'acoustics': info, 'stt_model': self.stt.model})
            timings['stt_s'] = log['elapsed_s']
            if self.llm is not None:
                result = await self.chat(heard)
                result.details['timings'] = {**timings, **result.details['timings']}
                result.details.update(acoustics=info, stt_model=self.stt.model)
                return result
            return Reply(b'', '', heard,
                         {'timings': timings, 'acoustics': info, 'stt_model': self.stt.model})
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError('Expected 1–1000 characters of speakable text')
        output, timings['tts_s'] = await self.speak(text)
        return Reply(output, text, details={'timings': timings, 'tts_model': self.tts.model})

    async def chat(self, text):
        if self.llm is None:
            raise ValueError('Conversation mode is not enabled')
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError('Expected 1–1000 characters of conversation input')
        async with self.stage('llm', model=self.llm.model) as log:
            answer = await self.llm.respond(text)
        llm_s = log['elapsed_s']
        result = await self.reply(text=answer)  # Literal TTS stays independent of the model.
        result.heard = text
        result.details['llm_model'] = self.llm.model
        result.details['timings'] = {'llm_s': llm_s, **result.details['timings']}
        return result


def session_factory(*, make_stt, make_tts, acoustics, noise_threshold=.6, background_threshold=.8,
                    make_llm=None, live_tyto=False):
    @asynccontextmanager
    async def session():
        async with AsyncExitStack() as stack:
            adapters = []
            for make in (make_stt, make_tts):
                adapter = make()
                stack.push_async_callback(adapter.close)
                if hasattr(adapter, 'prepare'):
                    await adapter.prepare()
                adapters.append(adapter)
            llm = make_llm() if make_llm else None
            if llm is not None:
                stack.push_async_callback(llm.close)
            pipeline = AudioPipeline(*adapters, acoustics,
                                     NoisePolicy(threshold=noise_threshold, noise_threshold=background_threshold), llm,
                                     live_tyto=live_tyto)
            stack.push_async_callback(pipeline.close)
            yield pipeline
    return session
