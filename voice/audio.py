"""PCM validation and the Charon metallic voice effect, shared by transports."""
import numpy as np
from scipy.signal import butter, sosfilt

RATE = 24000
MAX_SECONDS = 30


def robotic_pcm(pcm: bytes, gain: float = 0.65) -> bytes:
    """24 kHz mono PCM16 in/out. No pitch shift; preserve speech duration."""
    if not pcm or len(pcm) % 2 or len(pcm) > MAX_SECONDS * RATE * 2:
        raise ValueError('Invalid or oversized reply audio')
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float64) / 32768
    for cutoff, kind in ((100, 'highpass'), (4200, 'lowpass')):
        samples = sosfilt(butter(2, cutoff, btype=kind, fs=RATE, output='sos'), samples)
    samples *= .84 + .16 * np.sin(2 * np.pi * 34 * np.arange(len(samples)) / RATE)
    amplitude = np.abs(samples)
    threshold = 10 ** (-12 / 20)
    compressed = np.where(amplitude > threshold, threshold * (amplitude / threshold) ** .25, amplitude)
    return (np.clip(np.sign(samples) * compressed * gain, -1, 1) * 32767).astype('<i2').tobytes()
