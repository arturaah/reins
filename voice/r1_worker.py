"""Audio-only Unitree worker; run with the robot SDK's Python (often 3.10).

Private stdin/stdout protocol for robot_speaker.py. No network listener, API keys,
volume changes, TTS, locomotion or joint publishers. PCM is mono 16 kHz int16.
"""
import argparse
import base64
from contextlib import redirect_stdout
import json
import sys
from uuid import uuid4


class R1PCMClient:
    def __init__(self, iface):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.rpc.client import Client
        ChannelFactoryInitialize(0, iface)
        self.client = Client('voice', False)
        self.client.SetTimeout(2.0)
        self.client._SetApiVerson('1.0.0.0')
        for api in (1003, 1004, 1005):
            self.client._RegistApi(api, 0)
        self.app = 'reins_live_' + uuid4().hex
        self.stream = uuid4().hex
        self.active = False
        code, _ = self.client._Call(1005, '{}')
        if code:
            raise RuntimeError('R1 speaker is unavailable')

    def play(self, pcm):
        self.active = True  # Also stop if the RPC fails after reaching the robot.
        code, _ = self.client._CallRequestWithParamAndBin(
            1003, json.dumps({'app_name': self.app, 'stream_id': self.stream}), list(pcm))
        if code:
            raise RuntimeError('R1 playback was rejected')

    def stop(self):
        if self.active:
            code, _ = self.client._Call(1004, json.dumps({'app_name': self.app}))
            if code:
                raise RuntimeError('R1 playback stop was rejected')
            self.active = False
        self.stream = uuid4().hex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iface', required=True)
    args = parser.parse_args()
    client = None
    def reply(ok):
        print(json.dumps({'ok': ok}), flush=True)
    try:
        with redirect_stdout(sys.stderr):
            client = R1PCMClient(args.iface)
        reply(True)
        while True:
            line = sys.stdin.buffer.readline(100000)
            if not line:
                break
            if not line.endswith(b'\n'):
                raise ValueError('Oversized speaker message')
            event = json.loads(line)
            with redirect_stdout(sys.stderr):
                if event.get('type') == 'stop':
                    client.stop()
                elif event.get('type') == 'pcm':
                    pcm = base64.b64decode(event['audio'], validate=True)
                    if not pcm or len(pcm) % 2 or len(pcm) > 64000:
                        raise ValueError('Invalid PCM')
                    client.play(pcm)
                else:
                    raise ValueError('Unknown speaker message')
            reply(True)
    except Exception:
        reply(False)  # Do not return SDK exceptions or subprocess environment.
    finally:
        if client:
            with redirect_stdout(sys.stderr):
                client.stop()


if __name__ == '__main__':
    main()
