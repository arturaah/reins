"""Listen to the R1's onboard speech recognition and, optionally, speak the result back.

The robot publishes recognized speech as strings on rt/audio_msg
(unitree_sdk2/example/r1/audio/r1_audio_client_example.cpp, asr_handler). Without --speak
this is subscribe-only: it prints every phrase heard during the window and publishes nothing.
With --speak it sends each phrase to the voice service's TtsMaker (a DDS publish, needs
Artur's yes), choosing the Chinese voice when the text contains CJK characters.

    .venv/bin/python tools/echo.py en6                    # print what the robot hears for 15 s
    .venv/bin/python tools/echo.py en6 --speak            # echo the first phrase, then exit
    .venv/bin/python tools/echo.py en6 --speak --loop     # keep echoing until the window ends
"""
import argparse, sys, time
from pathlib import Path
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

sys.path.insert(0, str(Path(__file__).resolve().parent))
from say import R1AudioClient  # noqa: E402


def has_cjk(s):
    return any("一" <= ch <= "鿿" for ch in s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iface", help="network interface facing the robot (en6 on the Mac, eth10 on the Jetson)")
    ap.add_argument("--seconds", type=float, default=15.0, help="how long to listen")
    ap.add_argument("--speak", action="store_true", help="speak each recognized phrase back (publishes)")
    ap.add_argument("--loop", action="store_true", help="with --speak: keep echoing instead of stopping after the first")
    a = ap.parse_args()

    heard = []
    def on_msg(msg: String_):
        text = msg.data.strip()
        if text:
            heard.append(text)
            print(f"[{time.time() - t0:5.1f} s] heard: {text!r}", flush=True)

    ChannelFactoryInitialize(0, a.iface)
    client = None
    if a.speak:
        client = R1AudioClient(); client.SetTimeout(5.0); client.Init()
    sub = ChannelSubscriber("rt/audio_msg", String_)
    sub.Init(on_msg, 10)
    t0 = time.time()
    print(f"listening on rt/audio_msg via {a.iface} for {a.seconds:.0f} s{' and echoing' if a.speak else ''}", flush=True)
    spoken = 0
    while time.time() - t0 < a.seconds:
        time.sleep(0.1)
        if client is not None and spoken < len(heard):
            text = heard[spoken]; spoken += 1
            code = client.TtsMaker(text, 0 if has_cjk(text) else 1)
            print(f"        said back {text!r}: code={code}", flush=True)
            if not a.loop:
                break
    sub.Close()
    if not heard:
        print(f"nothing recognized in {a.seconds:.0f} s (the robot's ASR may need a wake word or a closer speaker)")
        sys.exit(1)


if __name__ == "__main__":
    main()
