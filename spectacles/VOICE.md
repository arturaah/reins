# Spectacles → GPT-Live → R1 speaker

The Lens sends raw 16 kHz mono PCM from its microphone through the plan feed to
the local GPT-Live service. Voice Focus enhances it and Tyto analyzes the original
input. GPT-Live's generated audio, including the metallic effect, streams to the
R1's `PlayStream` service. It does not use the robot's built-in TTS voice.

Ordinary conversation stays in GPT-Live. With `--backend spectacles`, delegated
robot requests enter the existing desktop dry-run inbox. The desktop UI must be
running to consume it. The spoken result confirms queuing only; planner results
and accept/reject decisions remain in the existing desktop/glasses review flow.
Voice does not approve proposals or execute motion. Stopping voice does not
cancel a task already handed off to the desktop; use its Stop control.

## Setup on the robot-connected Mac

Keep the existing `.venv` with the Unitree SDK. Install the voice dependencies in
a separate Python 3.12 environment so the robot SDK environment stays intact:

```sh
python3.12 -m venv .venv-voice
.venv-voice/bin/python -m pip install -r voice/requirements-aic.txt
```

Put these two entries in the repository's gitignored `.env`:

```dotenv
OPENAI_KEY=your_openai_key
AIC_KEY=your_ai_coustics_sdk_license
```

The services load that file automatically, regardless of working directory.
`OPENAI_API_KEY` and `AIC_SDK_LICENSE` also work. Environment values take precedence,
including the short aliases. `--key-file PATH` selects a different file. Credentials
and the local voice token stay on the Mac; no keys belong in the Lens.

Start the voice service, substituting the actual **body Ethernet interface** for
`en8`. This explicitly enables speaker playback when a voice session starts;
it reads the existing volume and never changes it:

```sh
.venv-voice/bin/python -m voice.live --backend spectacles --output r1 \
  --robot-iface en8 --robot-python .venv/bin/python
```

The separate SDK worker exposes PCM playback and stopping its own stream only.
Leave `--backend` at `test` for the GPT-5-mini conversation test without queuing
anything to the desktop. Leave `--output` at `browser` for the standalone Mac
microphone/speaker test; that output mode cannot serve the glasses relay.

Connect Spectacles by USB and start the existing plan feed **with** the voice
relay. Use the same preview/review files as the desktop UI:

```sh
adb reverse tcp:8765 tcp:8765
.venv/bin/python spectacles/plan_feed.py runs/ui_preview.json \
  --host 127.0.0.1 --robot-iface en8 \
  --review-file runs/spectacles_review.json \
  --live-voice-url http://127.0.0.1:8770
```

The preview file appears when the desktop proposes a plan; voice can connect
before a preview exists. The live relay requires loopback binding and USB/ADB.
It is not an authenticated wireless voice endpoint. Keep the browser voice
session stopped while using the glasses; the service allows one active caller.

## Lens and controls

Open `spectacles/R1 Hand Path Preview.esproj` in Lens Studio. The scene includes **Voice Microphone**
and has `useLiveVoice` enabled. Set `websocketUrl` to `ws://127.0.0.1:8765`, save,
and **resend the Lens to the glasses**. An already-installed Lens retains the old
ASR-to-text behavior. Allow microphone/network permissions on the glasses.

- Double right pinch starts the conversation. Wait for the GPT-Live listening label.
- Speak normally and pause; GPT-Live handles turn timing. Hear replies on the R1.
- Double right or double left pinch stops the conversation and microphone.
- A pending trajectory review takes priority and stops voice capture; the existing
  accept/reject gestures still apply. Start voice again after review.

Capture is silenced while the robot is speaking, with a 400 ms tail. Stop,
disconnect, or loss of microphone frames closes the voice session and stops this
worker's stream. Playback has a two-second backlog limit. The robot reports RPC
acceptance, not a measured speaker completion time; timing is estimated from PCM
duration. Echo and packet timing still need testing in the actual room.

If the Lens reports “Start the plan feed with --live-voice-url”, restart the feed
with the command above. If it requests `--output r1`, restart the voice service
with robot output. If the speaker is unavailable, check the body interface, SDK
Python and robot connectivity; no fallback changes the robot volume or voice.

## Validation

Automated tests cover PCM encoding, session readiness, playback input gating,
gesture priority, transport authentication, stop/disconnect cleanup, bounded
playback, SDK API selection, inbox deduplication and credential aliases. They use
fake providers/SDK clients, never hardware. Lens Studio compilation, microphone
permissions and audible R1 playback require an on-device check; they have not
been verified by these tests.

References: [Snap microphone audio](https://developers.snap.com/lens-studio/api/lens-scripting/classes/Built-In.MicrophoneAudioProvider),
[Snap WebSockets](https://developers.snap.com/lens-studio/api/lens-scripting/classes/Built-In.WebSocket.html),
[GPT-Live client delegation](https://developers.openai.com/api/docs/guides/live-delegation).
R1 PCM API: `unitree_sdk2/include/unitree/robot/r1/audio/audio_client.hpp`.
