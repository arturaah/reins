# Cascaded speech adapters

Optional reference path for dictation into an existing text UI and literal reply playback.
For the primary speech-to-speech conversation, use [GPT-Live](README.md).
Run one voice service per port.

Speech input and output around the **existing text UI**. By default, this package has no
conversational LLM. It has no harness, motion catalog or robot SDK dependency. The team can
keep changing its LLMs without changing the audio pipeline.

```text
Computer microphone, 16 kHz mono PCM16
  ├─ original audio → Voice Focus VAD → end of utterance
  ├─ original audio → live Tyto 1.1 → smoothed nudge policy → direct TTS
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
shown as text, and **Read text** speaks exactly what you enter. In the default mode it does not ask
an LLM to answer. Only one voice connection can be active at a time.

In the optional conversation test, **Keep listening after replies** is on by
default. Click **Connect**, then **Start conversation** once. Each reply finishes,
the server resets the audio state, and capture resumes after a 300 ms speaker
tail. The microphone is released during response generation and playback:
barge-in is off. Tyto nudges and recoverable no-speech results rearm the same loop.
The loop preserves the connection and recent conversational context across turns.

Turn off **Keep listening after replies** for one turn per **Talk** click; it is
off by default in plain speech I/O mode. **Test greeting**, typed messages and
Connect alone never start microphone capture. Disabling the loop finishes the
current turn and prevents another automatic recording. Normal VAD behavior is
unchanged; without auto-end, click Send to finish each turn.

Use **Stop**, Escape, closing or hiding the page to cancel the loop, release the
microphone and stop playback. Pending restarts and late microphone permission
results are also cancelled. Ctrl-C stops each server. Wake words and barge-in
are not implemented. Sessions still have the existing 15-minute limit.

## Diagnostics

The configuration cards show whether Voice Focus 2.2, Tyto 1.1 and Tyto nudge
are enabled. Enhancement level is read from the initialized SDK processor, not
assumed: the tested Quail VF 2.2 model defaults to **0.8 / 80%**. This is the
suppression-strength parameter, not a measured percentage of noise removed.

The **Pipeline log** receives events while stages run, including STT transcripts,
literal TTS input, models, processing times and audio durations. Tyto entries show
all seven raw scores plus the four smoothed policy fields; nudge entries include
the timestamp, risk, cause and exact line. Voice Focus entries show the active
enhancement level; playback start/end events are separate from TTS generation.
Provider errors are sanitized and never include keys or raw provider payloads.

Logs are bounded to 200 entries in the current browser page. **Clear** removes the
logs; reloading clears both logs and the on-page transcript. They are not written
to disk or local storage. Stop clears the server's conversational context but
leaves the page's transcript and logs visible for inspection.

Both servers bind to loopback. If changing the dashboard port or using
`localhost` instead of `127.0.0.1`, pass the exact `--dashboard-origin` to voice.
Changing the voice port also requires the matching dashboard `--voice-url`.

`--sim` rejects hardware run requests on the server, hides hardware controls,
and disables camera/twin readers and calibrated observations. The existing
MuJoCo preview and planner are unchanged. The dashboard now uses a neutral theme
with the colourful Reins logo retained.

## Optional conversation test

To test the full STT → LLM → TTS conversation locally before connecting the team's
harness, restart the voice service with:

```sh
.venv-voice/bin/python -m voice --chat-model gpt-5-mini --key-file .env.voice
```

Reload the voice page and click **Connect**. Speak through **Talk**, or type a
message. With the loop enabled, the button is labelled **Start conversation**
and listening resumes after each reply. GPT-5 mini answers through the existing TTS voice. This uses OpenAI's
hosted Responses API with the existing `OPENAI_API_KEY`; the app runs locally.
`chat.py` has no tools or robot access. Input flagged by the noise policy never
reaches the conversation model. Conversation transcripts do not fill the
planner's prompt box. The **Test greeting** and dashboard **Read reply** still
use literal TTS.

The last six text exchanges remain in session memory for follow-up questions;
Stop/disconnect clears them. API requests use `store=False`. A separate `chat`
WebSocket action handles typed conversation so literal `text` requests retain
their existing meaning. Start without `--chat-model` to return to plain voice I/O.
The model must support Responses with `minimal` reasoning effort.

## Faster speech input

Use streaming transcription to move enhancement and STT into the time spent
speaking, instead of uploading and processing the entire recording after Send:

```sh
.venv-voice/bin/python -m voice --stt openai-live --chat-model gpt-5-mini --key-file .env.voice
```

This selects `gpt-live-transcribe` with `delay=minimal`. The server opens the
transcription connection during Connect; every capture chunk passes through a
continuous Voice Focus processor and a stateful 16→24 kHz resampler. Local VAD
still determines the turn boundary. Partial transcripts are previewed in the UI;
only the final transcript is forwarded to the LLM or existing prompt box. There
are no automatic provider retries or hidden fallback calls.

Tyto still scores the original input. Unlike file STT, streaming STT may already
have received audio when a nudge fires; the provisional text is then discarded,
the buffer is cleared and the turn never reaches the LLM. The raw-score legacy
Tyto policy is not supported with streaming STT. The default `--stt openai`
retains file transcription for comparison or troubleshooting.

The log's STT duration in live mode measures the final wait after Send, not the
recording duration. Voice Focus time is CPU processing spread across capture.
The browser shows the selected microphone and current input level; the Voice
Focus log shows input/output RMS levels in dBFS. An empty transcript is a
recoverable no-speech result: the connection returns to Ready, no text goes to
the LLM, and Talk can be used again. Authentication, model-access, quota,
connection and timeout failures have separate sanitized messages.

A 13.44-second synthetic file transcribed successfully with the configured key
in 1.59 s; silence returned an empty transcript rather than an authorization
failure. With streaming, the probe produced a first partial at 0.56 s and a final
transcript 0.75 s after commit. The integrated 10-second clean-speech run finished
STT 0.80 s after Send. These are individual measurements, not latency guarantees;
LLM and TTS time are additional. Real microphone quality still needs checking.
The streamed noisy-audio test discarded provisional text, made no LLM request,
and completed one exact spoken nudge. Gemini TTS took roughly 5–8 seconds in these
tests and one request failed before a later check succeeded; synthesis remains a
separate latency/reliability limit. No automatic retry hides those delays.

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
The default pipeline does not choose an LLM. Fish Audio can be added at this same boundary.

## Voice Focus, VAD and Tyto 1.1

The default licensed configuration uses:

- `quail-vf-2.2-l-16khz`: primary speaker enhancement before STT.
- `vad-vf-2.0-s-16khz`: Voice Focus VAD on the **original** input.
- `tyto-1.1-l-16khz`: live quality readings on original input, before enhancement.

Models download to gitignored `.voice-cache/models` on first launch. Processing
runs locally; model download, SDK license authorization and usage telemetry
require network access. SDK/core versions and resolved model IDs accompany the
quality results. There is no fallback that silently bypasses a failed SDK call.

VAD requires at least 240 ms of speech, then 750 ms of silence. Live Tyto uses one
collector/analyzer pair per connection. The collector receives exact float32
blocks with a residual buffer; inference runs on a worker, at most once per five
seconds of new audio. It never queues overlapping analysis jobs and increases its
interval if analysis takes too long. It waits for five seconds of real input
before its first reading. Shorter turns are **unscored**, not certified clean.

`tyto_nudge.py` is the user's supplied policy, copied unchanged. It uses EMA alpha
0.3, a smoothed risk gate of 0.40, episode clearing below 0.30, and a 30-second
cooldown per cause. Only noise, packet loss and interfering speech may produce a
line. Codec degradation, loudness and reverb remain informational. The first
reading seeds the EMA; a high first window can therefore cause a nudge. Later
isolated spikes after clean audio are smoothed. These are experimental defaults.

When the policy returns a line, the UI releases the microphone, in-flight frames
are ignored, and the unfinished turn never reaches the LLM. File STT is skipped;
streaming STT is aborted and its provisional text discarded. Direct TTS speaks the
line verbatim and the optional test LLM remembers it as an assistant message.
The half-duplex UI cannot have an agent reply active during capture, so no
concurrent reply needs interrupting. Analysis is paused through playback; the
`played` acknowledgement resets the analyzer, EMA and warm-up count. Per-cause
cooldown survives this reset. The next capture starts a fresh window, whether
started by Talk or automatically by the active conversation loop.

The live nudge policy replaces the previous raw-score interruption rule. Use
`--no-tyto-nudge` for the legacy end-of-turn policy: full five-second windows at
one-second steps, with mean interference >=0.6 or noise >=0.8 holding the input.
`--interference-threshold` and `--noise-threshold` apply only to that legacy mode.
`--no-tyto` disables both analysis and nudges. Scores are quality indicators,
not a speaker count or proof of recognition accuracy. Synthetic verification
does not establish real-room performance.

Integration map for this custom stack:

- Raw inbound frames: `AudioPipeline.input_chunk` in `cascade.py`; `insight.py` owns the collector/worker.
- Playback start/end: `playback.start` / `playback.onended` in `web/app.js`; `played` resets analysis.
- Reply cancellation: `conversation` cleanup in `server.py`; browser cancellation: `stop` in `web/app.js`.
- Literal speech: `AudioPipeline.speak` in `cascade.py`, bypassing the optional LLM.

Verification used Python aic-sdk 3.2.0/core 0.24.0. The policy and wiring tests
cover clean readings, sustained noise (one nudge), a spike after clean audio,
high risk with only non-actionable causes, cooldown, five-second warm-up, exact
collector block sizes, playback reset, worker shutdown, and sanitized failures.
A paced 10-second synthetic clean-speech stream produced no nudge and completed
STT → GPT-5 mini → TTS. A noisy fixture with risk 0.56/noise 0.91 produced one
exact TTS nudge after roughly five seconds and no file STT/LLM request. Another noisy
fixture with risk 0.32 stayed below the policy gate and did not nudge. A live
quiet-room/video/fan microphone test is still needed before tuning thresholds.

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
client's `played` acknowledgement. The optional `chat` action is available only when a conversation model is configured.
No motion commands exist in this protocol.

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
node --test voice/tests/test_browser_loop.cjs
node --test voice/tests/test_live_browser.cjs
.venv-voice/bin/python -m unittest tools.test_dashboard core.test_ik
.venv-voice/bin/python -m pytest contract -q
```

Automated tests use fake speech providers. They cover audio/turn bounds, origin
and token checks, Stop cancellation, STT rearming without playback, exact-text
TTS contracts, interference/noise cooldowns and server-side simulation guards.
The browser lifecycle tests execute the actual page script with simulated media
and sockets: repeated turns, playback-gated restart, Tyto/no-speech recovery,
Stop during microphone permission, loop toggling and page hiding.
GPT-Live tests cover delegation IDs, deduplication, bounded context, stale-result
discarding, Tyto cancellation, Stop, streamed playback and microphone gating.
Live Cartesia validation requires a configured key and voice ID.

References: [ai-coustics Python SDK](https://docs.ai-coustics.com/reference/sdk/language-bindings/python),
[Tyto](https://docs.ai-coustics.com/models/audio-insight/tyto),
[OpenAI STT](https://developers.openai.com/api/docs/guides/speech-to-text),
[OpenAI streaming transcription](https://developers.openai.com/api/docs/guides/realtime-transcription),
[Gemini TTS](https://ai.google.dev/gemini-api/docs/speech-generation),
[Cartesia TTS](https://docs.cartesia.ai/api-reference/tts/bytes).
