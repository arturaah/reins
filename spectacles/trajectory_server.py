#!/usr/bin/env python3
"""Mock R1 hand trajectories over WebSocket. Python 3.10+ and websockets."""

import argparse
import asyncio
import json
import math
import time

from websockets.asyncio.server import serve


def trajectory(phase: float = 0.0) -> dict:
    # Robot base frame: x forward, y left, z up; all distances in metres.
    # Approximate standing R1 hands begin near x=0.08, y=+/-0.23, z=0.68.
    hands = {}
    for name, side in (("left", 1), ("right", -1)):
        hands[name] = [
            [
                round(0.08 + 0.42 * t, 4),
                round(side * (0.23 + 0.10 * t), 4),
                round(0.68 + 0.22 * math.sin(math.pi * t * 0.7 + phase)
                      + (0.04 if name == "right" else 0.0), 4),
            ]
            for t in (i / 29 for i in range(30))
        ]
    return {
        "type": "trajectory",
        "version": 1,
        "id": "mock-r1-hands",
        "frame": "robot_base",
        "units": "m",
        "timestamp_ms": int(time.time() * 1000),
        "hands": hands,
    }


async def handler(websocket):
    print(f"Lens connected: {websocket.remote_address}", flush=True)
    phase = 0.0
    try:
        while True:
            await websocket.send(json.dumps(trajectory(phase)))
            phase += 0.05
            await asyncio.sleep(0.25)
    except Exception as exc:
        print(f"Lens disconnected: {exc}", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    async with serve(handler, args.host, args.port):
        print(f"Mock R1 trajectory feed: ws://{args.host}:{args.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
