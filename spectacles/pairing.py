"""Pair a Lens with the LAN plan/voice feed, separately from cloud credentials."""
import argparse
import asyncio
import hmac
import json
import os
from pathlib import Path
import re
import secrets

DEFAULT_FILE = Path(__file__).resolve().parents[1] / '.spectacles-pairing-token'
TOKEN_PATTERN = re.compile(r'[A-Za-z0-9_-]{32,128}')


def load_token(path):
    try:
        with Path(path).open() as stream:
            token = stream.read(130).strip()
        if not TOKEN_PATTERN.fullmatch(token):
            raise ValueError
        return token
    except (OSError, UnicodeError, ValueError):
        raise ValueError('Invalid pairing file; create it with python -m spectacles.pairing') from None


def validate_bind(host, live_voice_url, token):
    if token is not None and not TOKEN_PATTERN.fullmatch(token):
        raise ValueError('Invalid Spectacles pairing token')
    if live_voice_url and host not in ('127.0.0.1', 'localhost', '::1') and not token:
        raise ValueError('Wireless live voice requires --pairing-file; USB uses --host 127.0.0.1')


async def authenticate(websocket, token, timeout=5.0):
    """Complete pairing before opening a relay or reading/sending any feed data."""
    from websockets.exceptions import ConnectionClosed
    try:
        await websocket.send(json.dumps({'type': 'pairing_required', 'version': 1}))
        raw = await asyncio.wait_for(websocket.recv(), timeout)
        message = json.loads(raw) if isinstance(raw, str) and len(raw) <= 1024 else None
        candidate = message.get('token') if isinstance(message, dict) else None
        accepted = bool(isinstance(message, dict) and message.get('type') == 'pair'
                        and message.get('version') == 1 and isinstance(candidate, str)
                        and TOKEN_PATTERN.fullmatch(candidate)
                        and hmac.compare_digest(candidate, token))
    except (asyncio.TimeoutError, ValueError, ConnectionClosed):
        accepted = False
    try:
        await websocket.send(json.dumps({'type': 'pairing_result', 'version': 1, 'accepted': accepted}))
        if not accepted:
            await websocket.close(code=1008, reason='Pairing required')
    except ConnectionClosed:
        return False
    return accepted


def main():
    parser = argparse.ArgumentParser(description='Create or show the local Spectacles pairing token.')
    parser.add_argument('--file', type=Path, default=DEFAULT_FILE)
    args = parser.parse_args()
    try:
        fd = os.open(args.file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'w') as stream:
            stream.write(secrets.token_urlsafe(24) + '\n')
    print('Copy this token into the R1Trajectory pairingToken field in Lens Studio:')
    print(load_token(args.file))
    print('Keep this file and the paired Lens private. Cloud API keys stay on the Mac.')


if __name__ == '__main__':
    main()
