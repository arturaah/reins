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
    # Exact neutral hand preview sites from the fixed-base R1 MuJoCo model.
    # Keep the first point fixed so changing plans never fake hand motion.
    hands = {}
    for name, side in (("left", 1), ("right", -1)):
        wave = phase + (math.pi if name == "right" else 0.0)
        reach = 0.24 + 0.07 * math.sin(wave)
        lateral = 0.06 + 0.035 * math.cos(wave)
        rise = 0.18 + 0.07 * math.cos(wave + 0.5)
        hands[name] = [
            [
                round(0.2909 + reach * t, 4),
                round(side * (0.1386 + lateral * t), 4),
                round(0.771 + rise * t + 0.025 * math.sin(math.pi * t) * t, 4),
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


async def handler(websocket, static=False):
    print(f"Lens connected: {websocket.remote_address}", flush=True)
    phase = 0.0
    try:
        while True:
            await websocket.send(json.dumps(trajectory(phase)))
            if not static:
                phase += 0.18
            await asyncio.sleep(0.25)
    except Exception as exc:
        print(f"Lens disconnected: {exc}", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static", action="store_true", help="hold fixed neutral-hand paths for alignment checks")
    args = parser.parse_args()
    async with serve(lambda websocket: handler(websocket, args.static), args.host, args.port):
        mode = "static" if args.static else "animated"
        print(f"{mode} R1 trajectory feed: ws://{args.host}:{args.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
