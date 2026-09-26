"""Make the R1 speak a sentence through its onboard text-to-speech.

Calls the robot's "voice" RPC service (unitree_sdk2/include/unitree/robot/r1/audio/audio_api.hpp):
reads the current volume, optionally sets it, then sends one TtsMaker request.
The robot synthesizes and plays the text itself. No joint command is sent.

    .venv/bin/python tools/say.py en6 "six seven"              # English voice
    .venv/bin/python tools/say.py en6 "你好" --speaker 0        # Chinese voice
    .venv/bin/python tools/say.py en6 "hello" --volume 60      # set volume (0..100) first

This publishes an RPC request on DDS, so it needs Artur's yes before each run.
The Python SDK's G1 AudioClient shares the R1's service name and API ids but never
increments its TTS index (g1_audio_client.py, "self.tts_index += self.tts_index"),
so the client is redefined here with a counting index.
"""
import argparse, json, sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.rpc.client import Client
from unitree_sdk2py.g1.audio.g1_audio_api import (
    AUDIO_SERVICE_NAME, AUDIO_API_VERSION,
    ROBOT_API_ID_AUDIO_TTS, ROBOT_API_ID_AUDIO_GET_VOLUME, ROBOT_API_ID_AUDIO_SET_VOLUME)


class R1AudioClient(Client):
    def __init__(self):
        super().__init__(AUDIO_SERVICE_NAME, False)
        self.tts_index = 0

    def Init(self):
        self._SetApiVerson(AUDIO_API_VERSION)
        for api in (ROBOT_API_ID_AUDIO_TTS, ROBOT_API_ID_AUDIO_GET_VOLUME, ROBOT_API_ID_AUDIO_SET_VOLUME):
            self._RegistApi(api, 0)

    def TtsMaker(self, text, speaker_id):
        p = {"index": self.tts_index, "text": text, "speaker_id": speaker_id}
        self.tts_index += 1
        code, _ = self._Call(ROBOT_API_ID_AUDIO_TTS, json.dumps(p))
        return code

    def GetVolume(self):
        code, data = self._Call(ROBOT_API_ID_AUDIO_GET_VOLUME, json.dumps({}))
        return code, (json.loads(data).get("volume") if code == 0 and data else None)

    def SetVolume(self, volume):
        code, _ = self._Call(ROBOT_API_ID_AUDIO_SET_VOLUME, json.dumps({"volume": int(volume)}))
        return code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iface", help="network interface facing the robot (en6 on the Mac, eth10 on the Jetson)")
    ap.add_argument("text")
    ap.add_argument("--speaker", type=int, default=1, help="TTS voice: 1 English (default), 0 Chinese")
    ap.add_argument("--volume", type=int, help="set speaker volume 0..100 before speaking")
    ap.add_argument("--timeout", type=float, default=5.0, help="RPC timeout in seconds")
    a = ap.parse_args()
    if a.volume is not None and not 0 <= a.volume <= 100:
        sys.exit("--volume must be 0..100")

    ChannelFactoryInitialize(0, a.iface)
    c = R1AudioClient(); c.SetTimeout(a.timeout); c.Init()

    code, vol = c.GetVolume()
    print(f"GetVolume: code={code} volume={vol}")
    if code != 0:
        sys.exit(f"voice service did not answer on {a.iface} (code {code}); nothing spoken")
    if a.volume is not None:
        code = c.SetVolume(a.volume)
        print(f"SetVolume({a.volume}): code={code}")
    code = c.TtsMaker(a.text, a.speaker)
    print(f"TtsMaker({a.text!r}, speaker={a.speaker}): code={code}")
    sys.exit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()
