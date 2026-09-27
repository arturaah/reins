"""Relay glasses PCM through the existing authenticated loopback voice service.

The glasses never receive cloud keys or the local service token. The plan feed
accepts USB/ADB on loopback or separately paired clients over a private LAN.
"""
import asyncio
from contextlib import suppress
import json
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from websockets.asyncio.client import connect


def local_voice_url(value):
    url = urlsplit(value)
    if (url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1')
            or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment):
        raise ValueError('Live voice URL must be a loopback HTTP origin')
    return value.rstrip('/')


def read_config(url):
    with build_opener(ProxyHandler({})).open(url + '/config', timeout=3) as response:
        return json.loads(response.read(16384))


class LiveVoiceRelay:
    def __init__(self, lens, url):
        self.lens, self.url = lens, local_voice_url(url)
        self.task = self.upstream = None
        self.listening = False
        self.last_audio = 0.
        self.identity = ''

    async def event(self, event):
        await self.lens.send(json.dumps({'type': 'voice_event', 'version': 1,
                                        'id': self.identity, 'event': event}))

    async def start(self, identity=''):
        if self.task and not self.task.done():
            return
        self.identity = identity
        self.task = asyncio.create_task(self.run())

    async def run(self):
        try:
            config = await asyncio.to_thread(read_config, self.url)
            if config.get('provider') != 'gpt-live' or config.get('output') != 'r1':
                await self.event({'type': 'error', 'text': 'Start GPT-Live with --output r1 on the Mac.'})
                return
            async with connect(self.url.replace('http:', 'ws:', 1) + '/voice', origin=self.url,
                               max_size=65536, max_queue=8, open_timeout=5, close_timeout=2) as upstream:
                self.upstream = upstream
                await upstream.send(json.dumps({'token': config['token']}))
                while True:
                    try:
                        raw = await asyncio.wait_for(upstream.recv(), 1)
                    except asyncio.TimeoutError:
                        if self.listening and time.monotonic() - self.last_audio > 5:
                            raise ValueError('Glasses microphone stopped sending')
                        continue
                    if not isinstance(raw, str):
                        raise ValueError('Expected robot playback, received browser audio')
                    event = json.loads(raw)
                    if event.get('type') == 'ready':
                        await upstream.send(json.dumps({'type': 'listen', 'sample_rate': 16000}))
                    if event.get('type') == 'listening':
                        self.listening = True
                        self.last_audio = time.monotonic()
                    if self.listening and time.monotonic() - self.last_audio > 5:
                        raise ValueError('Glasses microphone stopped sending')
                    await self.event(event)
                    if event.get('type') == 'error':
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            with suppress(Exception):
                await self.event({'type': 'error', 'text': 'Live voice connection failed. Check the Mac voice service.'})
        finally:
            self.listening = False
            self.upstream = None
            with suppress(Exception):
                await self.event({'type': 'stopped'})

    async def audio(self, pcm):
        if not pcm or len(pcm) % 2 or len(pcm) > 4096:
            await self.stop()
            return
        if self.upstream and self.listening:
            self.last_audio = time.monotonic()
            try:
                await asyncio.wait_for(self.upstream.send(pcm), 1)
            except Exception:
                await self.stop()

    async def stop(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
