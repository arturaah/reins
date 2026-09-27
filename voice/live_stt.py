"""Streaming transcription only: no conversational model or robot tools."""
import asyncio
import base64
import numpy as np
from scipy.signal import firwin, lfilter
from .errors import SpeechError, provider_error, validate_transcript

MODEL = 'gpt-live-transcribe'


class PCM24k:
    """Continuous 16 → 24 kHz FIR resampling; state crosses capture chunk boundaries."""
    def __init__(self):
        self.kernel = firwin(61, 1/3, window=('kaiser',5.))*3
        self.state = np.zeros(60)
        self.phase = 0

    def feed(self, pcm):
        if not pcm: return b''
        samples = np.frombuffer(pcm, '<i2').astype(np.float64)
        up = np.zeros(len(samples)*3); up[::3] = samples
        filtered, self.state = lfilter(self.kernel, [1.], up, zi=self.state)
        output = filtered[self.phase::2]
        self.phase = (self.phase-len(up)) % 2
        return np.clip(np.rint(output), -32768, 32767).astype('<i2').tobytes()


class OpenAILiveSTT:
    streaming = True

    def __init__(self, key, model=MODEL):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=key, max_retries=0)
        self.model, self.connection, self.reader = model, None, None
        self.waiters, self.partials, self.completed = {}, {}, {}
        self.active, self.partial_sink = False, None
        self.fault = None

    def set_partial_sink(self, sink):
        self.partial_sink = sink

    async def prepare(self):
        try:
            self.connection = await self.client.realtime.connect(
                extra_query={'intent':'transcription'}, max_retries=0,
                websocket_connection_options={'open_timeout':10, 'max_size':262144}).enter()
            self.reader = asyncio.create_task(self._read())
            update = self._expect('session.updated')
            await self.connection.session.update(session={'type':'transcription', 'audio':{'input':{
                'format':{'type':'audio/pcm', 'rate':24000},
                'transcription':{'model':self.model, 'delay':'minimal'},
                'turn_detection':None, 'noise_reduction':None}}})
            await self._wait(update)
        except Exception as error:
            raise provider_error(error) from None

    def _expect(self, kind):
        future = asyncio.get_running_loop().create_future()
        # Retrieve exceptions even when cancellation happens before the owner awaits this future.
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        self.waiters[kind] = future
        return future

    async def _wait(self, future):
        if self.fault: raise self.fault
        try:
            return await asyncio.wait_for(future, 10)
        except TimeoutError:
            raise SpeechError('timeout') from None

    def _resolve(self, kind, value):
        future = self.waiters.pop(kind, None)
        if future and not future.done(): future.set_result(value)

    async def _read(self):
        try:
            async for event in self.connection:
                kind = event.type
                if kind == 'error' or kind == 'conversation.item.input_audio_transcription.failed':
                    code = getattr(getattr(event, 'error', None), 'code', '')
                    public = {'invalid_api_key':'authentication', 'model_not_found':'model_access',
                              'rate_limit_exceeded':'rate_limit', 'insufficient_quota':'rate_limit'}.get(code, 'provider_error')
                    raise SpeechError(public)
                if kind in ('session.updated', 'input_audio_buffer.cleared', 'input_audio_buffer.committed'):
                    self._resolve(kind, event)
                elif kind == 'conversation.item.input_audio_transcription.delta' and self.active:
                    item = event.item_id
                    self.partials[item] = (self.partials.get(item,'') + event.delta)[:1000]
                    if self.partial_sink:
                        await self.partial_sink(self.partials[item])
                elif kind == 'conversation.item.input_audio_transcription.completed' and self.active:
                    self.completed[event.item_id] = event.transcript
                    self._resolve('final:'+event.item_id, event.transcript)
                    if len(self.completed)>4: raise SpeechError('invalid_response')
            raise SpeechError('connection')
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.fault = provider_error(error)
            for future in self.waiters.values():
                if not future.done(): future.set_exception(self.fault)

    async def start(self):
        if self.fault: raise self.fault
        self.active = False
        self.partials.clear(); self.completed.clear()
        clear = self._expect('input_audio_buffer.cleared')
        await self.connection.input_audio_buffer.clear()
        await self._wait(clear)
        self.resampler = PCM24k()
        self.active = True

    async def append(self, pcm):
        if self.fault: raise self.fault
        if not self.active or not pcm: return
        data = self.resampler.feed(pcm)
        try:
            await asyncio.wait_for(self.connection.input_audio_buffer.append(
                audio=base64.b64encode(data).decode()), 5)
        except Exception as error:
            raise provider_error(error) from None

    async def finish(self):
        if self.fault: raise self.fault
        committed = self._expect('input_audio_buffer.committed')
        await self.connection.input_audio_buffer.commit()
        item = (await self._wait(committed)).item_id
        try:
            transcript = self.completed.get(item)
            if transcript is None:
                transcript = await self._wait(self._expect('final:'+item))
            return validate_transcript(transcript)
        finally:
            self.active = False

    async def abort(self):
        self.active = False
        self.partials.clear(); self.completed.clear()
        # Acknowledged clear at the next start prevents old audio joining the next turn.
        if self.connection and not self.fault:
            clear = self._expect('input_audio_buffer.cleared')
            try:
                await self.connection.input_audio_buffer.clear()
                await self._wait(clear)
            except Exception as error:
                self.fault = provider_error(error)

    async def close(self):
        self.active = False
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.connection:
            await self.connection.close()
        await self.client.close()
