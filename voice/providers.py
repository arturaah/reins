"""Small transport result and an offline audio check; no motion dependencies."""
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import numpy as np
from .audio import RATE


@dataclass
class Reply:
    pcm: bytes
    said: str
    heard: str = ''
    details: dict = field(default_factory=dict)


class Demo:
    async def reply(self, *, pcm=None, text=None):
        t = np.arange(RATE // 2) / RATE
        tone = np.sin(2 * np.pi * 440 * t) * np.sin(np.pi * t / .5) ** 2 * 7000
        return Reply(tone.astype('<i2').tobytes(),
                     'Offline audio check complete. This tone is not an AI reply.',
                     'Microphone audio received locally.' if pcm else text or '')


@asynccontextmanager
async def demo_session():
    yield Demo()
