"""Per-session Tyto collector and bounded analysis worker, on original microphone audio."""
import asyncio
import math
import time
import numpy as np
from .acoustics import RATE, TYTO_MODEL
from .tyto_nudge import TytoNudger

SCORES = ('risk_score', 'noise', 'interfering_speech', 'packet_loss',
          'codec_degradation', 'speaker_loudness', 'speaker_reverb')


class LiveTyto:
    def __init__(self, collector, analyzer, block_size, emit, *, nudger=None, clock=time.monotonic):
        self.collector, self.analyzer, self.block_size, self.emit = collector, analyzer, block_size, emit
        self.nudger = nudger or TytoNudger()
        self.clock, self.interval, self.task = clock, 5.0, None
        self.active = False
        self._clear()

    def _clear(self):
        self.residual = bytearray()
        self.samples = self.last_samples = 0
        self.next_at = 0
        self.latest = self.line = self.cause = None
        self.nudger.reset()

    async def reset(self):
        await self.pause()
        await asyncio.to_thread(self.analyzer.reset)
        self._clear()
        self.active = True

    def feed(self, pcm):
        if self.task and self.task.done():
            self.task.result()  # Fail visibly instead of using stale scores after an SDK error.
        if not self.active or self.line:
            return
        self.residual.extend(pcm)
        size = self.block_size * 2
        while len(self.residual) >= size:
            block = np.frombuffer(bytes(self.residual[:size]), '<i2').astype(np.float32) / 32768
            del self.residual[:size]
            self.collector.buffer(block)
            self.samples += self.block_size
        if (self.samples >= 5 * RATE and self.samples - self.last_samples >= 5 * RATE
                and self.clock() >= self.next_at and (self.task is None or self.task.done())):
            self.last_samples = self.samples
            self.next_at = self.clock() + self.interval
            self.task = asyncio.create_task(self._analyze())

    async def _analyze(self):
        started = self.clock()
        try:
            result = await asyncio.to_thread(self.analyzer.analyze_buffered)
            raw = {key: float(getattr(result, key)) for key in SCORES}
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in raw.values()):
                raise ValueError('Invalid Tyto scores')
            elapsed = self.clock() - started
            if elapsed > self.interval:
                self.interval = elapsed + .1  # Never queue inference jobs when the worker falls behind.
                self.next_at = self.clock() + self.interval
            self.latest = raw
            self.line = self.nudger.on_result(raw)
            self.cause = self.nudger.strongest_cause(self.nudger.smoothed)
            decision = ('Nudge requested.' if self.line else
                        'Below the smoothed risk gate.' if self.nudger.smoothed['risk_score'] < self.nudger.gate else
                        'No qualifying fixable cause.' if self.cause is None else
                        'Cause already addressed or on cooldown.')
            await self.emit('tyto', 'reading', model=TYTO_MODEL, scores=raw,
                            smoothed=dict(self.nudger.smoothed), elapsed_s=round(elapsed, 3),
                            interval_s=round(self.interval, 3), audio_s=round(self.samples / RATE, 3), text=decision)
            if self.line:
                self.active = False
                await self.emit('tyto', 'nudge', cause=self.cause, text=self.line,
                                risk_score=self.nudger.smoothed['risk_score'])
        except Exception:
            await self.emit('tyto', 'error', text='Tyto analysis failed; reconnect to try again.')
            raise

    async def pause(self):
        self.active = False
        if self.task:
            # Do not reset/terminate native state while its worker still uses it.
            try:
                await asyncio.shield(self.task)
            except asyncio.CancelledError:
                await asyncio.shield(self.task)
                raise

    async def playback_finished(self):
        await self.reset()
        self.active = False  # Next capture starts only after playback has ended and the UI rearms.

    async def close(self):
        try:
            await self.pause()
        finally:
            await asyncio.to_thread(self.analyzer.terminate_session)


def make_live_tyto(acoustics, emit):
    import aic_sdk as aic
    model = acoustics.models[TYTO_MODEL]
    collector, analyzer = aic.analyzer_pair(model, acoustics.key)
    try:
        config = aic.ProcessorConfig.optimal(model, sample_rate=RATE)
        collector.initialize(config)
        return LiveTyto(collector, analyzer, config.block_size, emit)
    except BaseException:
        analyzer.terminate_session()
        raise
