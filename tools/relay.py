"""Share the R1's live joint state with other computers over WebSocket. Subscribe-only.

Runs on the machine cabled to the robot. It reads rt/lowstate (measured angles) and
rt/arm_sdk (commanded arm angles and blend weight) and sends them to every connected
client as JSON, by MuJoCo joint name. Anything clients send is ignored; nothing goes
back to the robot. Other computers run the twin against it:

    .venv/bin/python tools/relay.py en6                       # on the robot-side Mac
    .venv/bin/python tools/twin.py ws://MAC_ADDRESS:8766      # on any other computer

Message, at --hz (default 50):
    {"type": "r1_state", "version": 1, "t": unix_s, "q": {joint: rad, ...},
     "cmd": {"weight": 0..1, "q": {joint: rad, ...}} or null when nothing is streaming}

Clients must reach the Mac's normal network (Wi-Fi, Tailscale), not the robot's
192.168.123.x link. The feed is read-only but unauthenticated: anyone who can reach
the port sees the joint angles.
"""
import argparse, asyncio, time
from websockets.asyncio.server import serve
from twin import RELAY_PORT, RobotState, listen_dds


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iface", help="interface on the robot network (en6 on the Mac, eth10 on the Jetson)")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain; the robot is 0")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=RELAY_PORT)
    ap.add_argument("--hz", type=float, default=50.0, help="messages per second to each client")
    a = ap.parse_args()
    state = RobotState()
    readers = listen_dds(state, a.iface, a.domain)
    clients = set()

    async def handler(ws):
        clients.add(ws)
        print(f"client connected: {ws.remote_address} ({len(clients)} total)", flush=True)
        try:
            while True:
                if state.q is not None:
                    await ws.send(state.to_json())
                await asyncio.sleep(1.0 / a.hz)
        except Exception:
            pass
        finally:
            clients.discard(ws)
            print(f"client left: {ws.remote_address} ({len(clients)} total)", flush=True)

    async with serve(handler, a.host, a.port):
        print(f"relay on ws://{a.host}:{a.port}", flush=True)
        last, count = time.time(), state.count
        while True:
            await asyncio.sleep(5)
            now = time.time()
            age = now - state.t if state.t else float("inf")
            print(f"lowstate {(state.count - count) / (now - last):5.0f} Hz"
                  + ("" if age < 0.5 else f"  STALE ({age:.1f} s)")
                  + f"  {'streaming' if state.commanding(now) else 'arm_sdk idle'}  {len(clients)} client(s)", flush=True)
            last, count = now, state.count


if __name__ == "__main__":
    asyncio.run(main())
