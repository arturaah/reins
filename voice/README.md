# Reins voice I/O

Speech input and output around the **existing text UI**. This package has no
conversational LLM, harness, motion catalog or robot SDK dependency. The team can
keep changing its LLMs without changing the audio pipeline.

```text
Computer microphone, 16 kHz mono PCM16
  ├─ original audio → Voice Focus VAD → end of utterance
  ├─ original audio → Tyto 1.1 → interference/noise policy
  └─ Quail Voice Focus → OpenAI STT → existing prompt box

Existing reply text → Gemini or Cartesia TTS → metallic effect → computer speaker
```

Dictation inserts text at the prompt box's selection. The user submits it using
the existing **Generate plan** button. **Read reply** speaks the existing status
or reply text. Speech does not submit a plan, approve it, or execute a motion.
The standalone page also provides a literal text-to-speech test.

## Run locally

Python 3.12 is recommended. Use a separate virtual environment if teammates are
using the repository environment:

```sh
python3 -m venv .venv-voice
.venv-voice/bin/python -m pip install -r voice/requirements-aic.txt
# Only needed for the existing simulation dashboard:
.venv-voice/bin/python -m pip install mujoco Pillow
```

Set `OPENAI_API_KEY`, `GEMINI_API_KEY` and `AIC_SDK_LICENSE`, or put them in a local
`.env.voice` and pass `--key-file .env.voice`. Environment values take precedence.
Keys stay server-side; do not paste them into the browser or commit them.

From the repository root, in two terminals:

```sh
.venv-voice/bin/python -m voice --key-file .env.voice
.venv-voice/bin/python tools/dashboard.py --sim --port 8091 --voice-url http://127.0.0.1:8770/
```

Open **http://127.0.0.1:8091/**. Click **Connect**, then **Test greeting**. Click
**Talk**, allow microphone access and speak. A pause after speech ends the turn;
**Send** also works manually. Review the dictated text in the existing prompt box.
After submitting it normally, click **Read reply** to hear the response.

The voice lab alone is at **http://127.0.0.1:8770/**. On that page dictation is
shown as text, and **Read text** speaks exactly what you enter. It does not ask
an LLM to answer. Only one voice connection can be active at a time.

Use **Stop**, Escape, closing or hiding the page to release the microphone and
stop playback. Ctrl-C stops each server. The browser releases its microphone
before inference and playback, preventing its own output from becoming another
input turn. Continuous listening, wake words and barge-in are not implemented.

Both servers bind to loopback. If changing the dashboard port or using
`localhost` instead of `127.0.0.1`, pass the exact `--dashboard-origin` to voice.
Changing the voice port also requires the matching dashboard `--voice-url`.

`--sim` rejects hardware run requests on the server, hides hardware controls,
and disables camera/twin readers and calibrated observations. The existing
MuJoCo preview and planner are unchanged. The dashboard now uses a neutral theme
with the colourful Reins logo retained.

## Speech adapters

Default STT: OpenAI `gpt-transcribe` (`--stt-model` selects a compatible model).
Default TTS: Gemini's dedicated `gemini-3.8-flash-lite-tts`, **Charon** voice.
This is exact-text TTS, not Gemini Live speech-to-speech. Output passes through
a restrained metallic effect: 100–4200 Hz filtering, 34 Hz modulation at 16%
depth, compression and 65% gain. Use the device's normal volume control.

To use Cartesia, set `CARTESIA_API_KEY` and `CARTESIA_VOICE_ID`, then:

```sh
.venv-voice/bin/python -m voice --tts cartesia --key-file .env.voice
```

Or pass `--tts-voice YOUR_VOICE_ID`. The default model is `sonic-3.6`; `--tts-model`
selects a compatible model explicitly. Choose a male voice in Cartesia's library
for the requested character. Cartesia does not require a Gemini key. The adapter
uses the official bytes endpoint, API version `2026-08-14`, raw PCM16 at 24 kHz.

STT and TTS are independent classes in `speech.py`. A different provider only
needs async `transcribe(pcm_16khz) -> str` or `speak(text) -> pcm_24khz`, a `model`
label and async `close()`. `AudioPipeline` in `cascade.py` accepts these adapters.
It never imports or chooses an LLM. Fish Audio can be added at this same boundary.

## Voice Focus, VAD and Tyto 1.1

The default licensed configuration uses:

- `quail-vf-2.2-l-16khz`: primary speaker enhancement before STT.
- `vad-vf-2.0-s-16khz`: Voice Focus VAD on the **original** input.
- `tyto-1.1-l-16khz`: interference, noise and overall risk scores on original input.

Models download to gitignored `.voice-cache/models` on first launch. Processing
runs locally; model download, SDK license authorization and usage telemetry
require network access. SDK/core versions and resolved model IDs accompany the
quality results. There is no fallback that silently bypasses a failed SDK call.

VAD requires at least 240 ms of speech, then 750 ms of silence. Tyto uses full
5-second windows at 1-second steps and averages their scores. Turns shorter than
5 seconds are explicitly marked **unscored**; VAD/STT still work. Do not interpret
an unscored turn as evidence that no competing speaker was present.

The experimental policy holds audio before STT if interfering speech averages
at least `0.6`, or background noise at least `0.8`. It speaks a fixed request for
one speaker or clearer audio. Interference takes priority. A 30-second cooldown
limits repeated spoken prompts, but flagged audio remains blocked throughout.
Use `--interference-threshold` and `--noise-threshold` to calibrate with real room
audio. Scores are quality indicators, not a speaker count or a guarantee of
recognition accuracy. Synthetic tests do not establish real-room performance.

For an unlicensed baseline, install `voice/requirements.txt` and run:

```sh
.venv-voice/bin/python -m voice --no-voice-focus --no-tyto --vad webrtc --key-file .env.voice
```

For an entirely offline transport check:

```sh
.venv-voice/bin/python -m voice --demo
```

The demo returns a clearly labelled tone. It does not transcribe, synthesize
speech, load ai-coustics models, or make cloud calls.

## Team integration boundary

The dashboard bridge is deliberately text-only. It checks the configured origin
and the iframe window on both sides:

- Voice emits `reins-voice-transcript` with `{text}`. The current UI inserts it
  into `actionPrompt`, preserving its normal submission and review workflow.
- The team can call `window.ReinsVoice.speak(replyText)` with up to 1,000 characters.
  It returns `false` when voice is disconnected, recording or playing, or input
  is invalid. A successful call only requests speech; it does not confirm playback.
- Internally, `reins-voice-speak` carries the text to the audio iframe.
  `reins-voice-ready` reports whether another TTS request can be accepted.

The loopback voice WebSocket requires a same-origin session token from `/config`.
After authenticating with `{token}`, it accepts `start` (16 kHz PCM16), binary
chunks, `end`, literal `text`, `hello`, `played`, and `stop`. STT-only turns return
transcripts/results then `ready`; TTS turns return bounded PCM and wait for the
client's `played` acknowledgement. No LLM or motion commands exist in this protocol.

Audio input is capped at 30 seconds (browser sends at 29); replies at 30 seconds.
A turn times out at 120 seconds, idle connections after two minutes, and sessions
after 15 minutes. Reconnect when a session expires. Audio/transcripts remain in
memory; this app neither stores them nor logs provider payloads. Cloud providers
receive audio for STT or text for TTS and apply their own data handling policies.

## Robot audio

This PR uses the computer microphone and speaker to test with R1 disconnected.
It does **not** claim support for R1 microphone capture. An audio-device adapter
can later supply the same 16 kHz input and play the resampled output through R1's
speaker, without changing STT/TTS or the team's LLM integration. No robot voice
mode, persistent setting or movement is changed by this package.

## Validation

```sh
.venv-voice/bin/python -m pip install pytest
.venv-voice/bin/python -m pytest voice/tests -q
.venv-voice/bin/python -m unittest tools.test_dashboard core.test_ik
.venv-voice/bin/python -m pytest contract -q
```

Automated tests use fake speech providers. They cover audio/turn bounds, origin
and token checks, Stop cancellation, STT rearming without playback, exact-text
TTS contracts, interference/noise cooldowns and server-side simulation guards.
Live Cartesia validation requires a configured key and voice ID.

References: [ai-coustics Python SDK](https://docs.ai-coustics.com/reference/sdk/language-bindings/python),
[Tyto](https://docs.ai-coustics.com/models/audio-insight/tyto),
[OpenAI STT](https://developers.openai.com/api/docs/guides/speech-to-text),
[Gemini TTS](https://ai.google.dev/gemini-api/docs/speech-generation),
[Cartesia TTS](https://docs.cartesia.ai/api-reference/tts/bytes).
