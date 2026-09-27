"""Original-signal VAD/Tyto and optional ai-coustics Voice Focus enhancement."""
from pathlib import Path
import time
import numpy as np

RATE = 16000
FOCUS_MODEL = 'quail-vf-2.2-l-16khz'
VAD_MODEL = 'vad-vf-2.0-s-16khz'
TYTO_MODEL = 'tyto-1.1-l-16khz'


def load_model(name, cache):
    import aic_sdk as aic
    return aic.Model.from_file(aic.Model.download(name, Path(cache)))


class Endpoint:
    """Conservative end-of-utterance: >= 240 ms speech, then >= 750 ms silence."""
    def __init__(self, vad='webrtc', license_key='', model=None):
        self.mode = vad
        self.buffer = bytearray()
        self.speech_ms = self.silence_ms = 0
        self.done = False
        if vad == 'aic':
            import aic_sdk as aic
            config = aic.ProcessorConfig.optimal(model, sample_rate=RATE)
            self.engine = aic.Vad(model, license_key, config)
            self.samples = config.block_size
        else:
            import webrtcvad
            self.engine = webrtcvad.Vad(2)
            self.samples = 480

    def feed(self, pcm):
        self.buffer.extend(pcm)
        size, duration = self.samples * 2, self.samples * 1000 / RATE
        while len(self.buffer) >= size and not self.done:
            block = bytes(self.buffer[:size]); del self.buffer[:size]
            if self.mode == 'aic':
                self.engine.process(np.frombuffer(block, '<i2').astype(np.float32) / 32768)
                speech = self.engine.get_context().is_speech_detected()
            else:
                speech = self.engine.is_speech(block, RATE)
            if speech:
                self.speech_ms += duration
                self.silence_ms = 0
            elif self.speech_ms >= 240:
                self.silence_ms += duration
            self.done = self.speech_ms >= 240 and self.silence_ms >= 750
        return self.done

    def close(self):
        if self.mode == 'aic':
            self.engine.terminate_session()


class Acoustics:
    def __init__(self, *, focus=False, tyto=False, vad='webrtc', license_key='', cache='.voice-cache/models'):
        self.focus, self.tyto, self.vad, self.key = focus, tyto, vad, license_key
        self.models = {}
        for enabled, name in ((focus, FOCUS_MODEL), (tyto, TYTO_MODEL), (vad == 'aic', VAD_MODEL)):
            if enabled:
                if not license_key:
                    raise ValueError('ai-coustics options require AIC_SDK_LICENSE')
                self.models[name] = load_model(name, cache)

    def endpoint(self):
        return Endpoint(self.vad, self.key, self.models.get(VAD_MODEL))

    def process(self, pcm):
        samples = np.frombuffer(pcm, '<i2').astype(np.float32) / 32768
        info = {'voice_focus': self.focus, 'vad': self.vad, 'tyto': 'off',
                'models': [model.get_id() for model in self.models.values()]}
        if self.models:
            import aic_sdk as aic
            info['sdk'] = aic.get_sdk_version()
        if self.tyto:
            if len(samples) < 5 * RATE:
                info['tyto'] = 'needs_5_seconds'
            else:
                analyzer = aic.FileAnalyzer(self.models[TYTO_MODEL], self.key)
                results = analyzer.analyze(np.ascontiguousarray(samples), RATE, RATE)
                # Average original-channel scores; no speaker count or identity inference.
                scores = {name: float(np.mean([getattr(r, name) for r in results]))
                          for name in ('interfering_speech', 'noise', 'risk_score')}
                if not all(np.isfinite(x) and 0 <= x <= 1 for x in scores.values()):
                    raise ValueError('Invalid Tyto score')
                info.update(tyto='scored', **scores)
        if self.focus:
            config = aic.ProcessorConfig.optimal(self.models[FOCUS_MODEL], sample_rate=RATE)
            processor = aic.Processor(self.models[FOCUS_MODEL], self.key)
            try:
                processor.initialize(config)
                delay = processor.get_context().get_audio_delay()
                size = ((len(samples) + delay + config.block_size - 1) // config.block_size) * config.block_size
                padded = np.zeros(size, dtype=np.float32); padded[:len(samples)] = samples
                output = np.empty_like(padded)
                for offset in range(0, size, config.block_size):
                    output[offset:offset+config.block_size] = processor.process(padded[offset:offset+config.block_size])
                samples = output[delay:delay+len(samples)]
                if not np.isfinite(samples).all():
                    raise ValueError('Invalid enhanced audio')
            finally:
                processor.terminate_session()
        return (np.clip(samples, -1, 32767/32768)*32768).astype('<i2').tobytes(), info


class NoisePolicy:
    """Experimental score threshold, with a spoken-prompt cooldown; always blocks flagged input."""
    def __init__(self, threshold=.6, noise_threshold=.8, cooldown=30, clock=time.monotonic):
        self.threshold, self.noise_threshold = threshold, noise_threshold
        self.cooldown, self.clock = cooldown, clock
        self.last = float('-inf')

    def check(self, info):
        reason = ('competing_speech' if info.get('interfering_speech', 0) >= self.threshold else
                  'background_noise' if info.get('noise', 0) >= self.noise_threshold else None)
        speak = reason is not None and self.clock() - self.last >= self.cooldown
        if speak:
            self.last = self.clock()
        return reason, speak
