# Spectacles → dashboard → GPT-Live → R1 speaker

The Lens can send raw 16 kHz mono PCM from its microphone through the paired
dashboard connection to the local GPT-Live service. Voice Focus enhances input;
Tyto analyzes original audio. Generated audio and the metallic effect stream to
the R1 `PlayStream` service through an isolated audio-only SDK worker. Robot
microphone capture and the R1 built-in TTS voice are not used.

Conversation stays in GPT-Live. With `--backend dashboard`, robot requests are
submitted to the **same dashboard agent** used by text chat. Speech returns
submission status only; planning results and complete proposals appear in the
dashboard and glasses. The operator must approve a proposal separately. Stopping
voice ends capture/playback; use dashboard **Stop** to cancel already submitted
planning or motion. The old `--backend spectacles` name now aliases `dashboard`.

## Start the optional services

Keep the robot SDK in `.venv`. Use a separate Python 3.12 voice environment:

```sh
python3.12 -m venv .venv-voice
.venv-voice/bin/python -m pip install -r voice/requirements-aic.txt
```

Put `OPENAI_KEY` and `AIC_KEY` in the repository's gitignored `.env` (canonical
`OPENAI_API_KEY` and `AIC_SDK_LICENSE` also work). Environment values take
precedence. `--key-file PATH` selects another file. Cloud keys and the local
voice-service token stay on the computer, never in the Lens. For a baseline
without licensed enhancement, install `voice/requirements.txt` and add
`--no-voice-focus --no-tyto`.

Start the dashboard on the desired port, using `--sim` for software-only motion
preview. Supplying `--voice-url` also enables the paired glasses PCM relay:

```sh
.venv/bin/python tools/dashboard.py --sim --port 8090 \
  --voice-url http://127.0.0.1:8770/
```

To use the robot speaker, substitute the actual **body Ethernet interface** for
`en8` below. Speaker playback is explicitly enabled by `--output r1` and starts
only when a voice session starts. The worker reads existing volume without
changing it. This is a physical speaker command even if the dashboard uses
simulation for motion:

```sh
.venv-voice/bin/python -m voice.live --backend dashboard \
  --dashboard-url http://127.0.0.1:8090 --output r1 \
  --robot-iface en8 --robot-python .venv/bin/python
```

For laptop microphone/speaker testing leave `--output` at `browser`; the glasses
relay requires R1 output and reports an explicit error for browser output.
`--backend conversation` handles conversation without submitting tasks; the
default `--backend test` uses the standalone text-model experiment.

## Pair and talk

Use **Connections → Glasses motion review** in the dashboard to create a device.
Copy its `deviceId` and one-time `reviewToken` into the Lens Inspector. Leave the
standalone `pairingToken` empty. Set `websocketUrl` to `ws://MAC_WIFI_IP:8765` on
the same private Wi-Fi/hotspot; allow that Python listener through the firewall.
For USB use `adb reverse tcp:8765 tcp:8765` and `ws://127.0.0.1:8765` instead.
Both paths require dashboard pairing. Keep `useLiveVoice` enabled, save and resend
the Lens, then allow microphone/network permissions. Do not commit credentials
in the scene.

Double right pinch starts a conversation. Wait for the listening label, speak
normally and pause. Double right or left pinch stops microphone and playback.
A review card stops voice and takes priority; restart talking after the review.
Both tags are needed for fresh motion approval, not for conversation. Reconnect
or revocation stops the relay and requires a new authenticated connection.

The voice service accepts one caller at a time. Stop the browser voice session
before using glasses. While the robot speaks, input is silenced with a 400 ms
tail. Stop, disconnect or missing microphone frames closes the voice session.
Playback has a two-second backlog limit. Robot RPC acceptance is not measured
speaker completion; timing is estimated from PCM duration. Room acoustics and
network timing need on-device tests.

## Standalone conversation compatibility

The upstream read-only plan feed still supports paired live audio without a
dashboard, for isolated conversation experiments. It does not provide a motion
execution authority. Create a token with `python -m spectacles.pairing`, then:

```sh
.venv/bin/python spectacles/plan_feed.py runs/preview.json \
  --host 0.0.0.0 --pairing-file .spectacles-pairing-token \
  --live-voice-url http://127.0.0.1:8770
```

Use `--backend conversation` in the voice service. The preview file may be absent
when voice starts. Set the Lens `pairingToken` to the generated value and clear
`deviceId`/`reviewToken`. USB loopback-only voice can omit the pairing file;
network voice requires it. Do not run this listener alongside the dashboard's
port 8765 listener. Tokens control access; local `ws://` is unencrypted.

## Verification

Automated tests use fake providers and SDK clients. They cover authenticated
session multiplexing, stale/revoked clients, review priority, PCM encoding,
readiness, speaker gating, bounded playback, stop/disconnect cleanup, request
deduplication and the dashboard's chat-only submission boundary. They never
call models or hardware. Lens Studio compilation, microphone permissions,
wireless connectivity and audible R1 playback require on-device verification.

References: [Snap microphone audio](https://developers.snap.com/lens-studio/api/lens-scripting/classes/Built-In.MicrophoneAudioProvider),
[Snap WebSockets](https://developers.snap.com/lens-studio/api/lens-scripting/classes/Built-In.WebSocket.html),
[Snap WebSocket setup and local-network requirements](https://developers.snap.com/spectacles/about-spectacles-features/apis/web-socket),
[GPT-Live client delegation](https://developers.openai.com/api/docs/guides/live-delegation).
R1 PCM API: `unitree_sdk2/include/unitree/robot/r1/audio/audio_client.hpp`.
