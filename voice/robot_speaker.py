"""Stream Live PCM to an isolated audio-only SDK process, with bounded buffering."""
import asyncio
import base64
from contextlib import suppress
import json
import os
from pathlib import Path
import time

import numpy as np

from .errors import SpeechError


class RobotSpeaker:
    def __init__(self, python, iface):
        self.python, self.iface = python, iface
        self.process = None
        self.lock = asyncio.Lock()
        self.until = 0.
        self.busy = False
        self.on_busy = None

    async def start(self):
        # The SDK worker needs no cloud credentials.
        env = {k: v for k, v in os.environ.items()
               if not any(word in k.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'LICENSE', 'PASSWORD'))}
        self.process = await asyncio.create_subprocess_exec(
            self.python, '-u', str(Path(__file__).with_name('r1_worker.py')), '--iface', self.iface,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=env)
        await self._ack()

    async def _ack(self):
        line = await asyncio.wait_for(self.process.stdout.readline(), 5)
        if not line or json.loads(line).get('ok') is not True:
            raise SpeechError('speaker_unavailable')

    async def _request(self, event):
        if not self.process or self.process.returncode is not None:
            raise SpeechError('speaker_unavailable')
        self.process.stdin.write((json.dumps(event) + '\n').encode())
        await asyncio.wait_for(self.process.stdin.drain(), 3)
        await self._ack()

    async def _busy(self, value):
        if value != self.busy:
            self.busy = value
            if self.on_busy:
                await self.on_busy(value)

    async def play(self, pcm):
        if not pcm or len(pcm) % 2 or len(pcm) > 64000:
            raise ValueError('Invalid robot speaker PCM')
        audible = bool(np.any(np.abs(np.frombuffer(pcm, '<i2').astype(np.int32)) > 100))
        async with self.lock:
            if not audible:
                return
            now = time.monotonic()
            end = max(self.until, now) + len(pcm) / 32000
            if end - now > 2:
                raise SpeechError('speaker_backlog')
            if audible:
                await self._busy(True)
            await self._request({'type': 'pcm', 'audio': base64.b64encode(pcm).decode('ascii')})
            self.until = max(self.until, time.monotonic()) + len(pcm) / 32000

    async def monitor(self):
        while True:
            await asyncio.sleep(.05)
            async with self.lock:
                if self.process and self.process.returncode is not None:
                    raise SpeechError('speaker_unavailable')
                if self.busy and time.monotonic() >= self.until + .4:
                    await self._request({'type': 'stop'})
                    self.until = 0.
                    await self._busy(False)

    async def clear(self):
        async with self.lock:
            await self._request({'type': 'stop'})
            self.until = 0.
            await self._busy(False)

    async def close(self):
        self.on_busy = None
        if self.process:
            # EOF also stops this worker's own stream. Let an in-flight RPC finish
            # before terminating it; never stop another application's playback.
            with suppress(Exception):
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            self.process = None
