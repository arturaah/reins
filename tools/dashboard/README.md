# Reins Observatory

A local browser workspace for the R1 MuJoCo preview, robot cameras, the
Spectacles wearer view, and trajectory control (Dry run, Execute, Abort through
`tools/arm_lift.py`, the same as the Tk window `tools/reins_ui.py`). Graphite glass
look (near-black canvas, hairline-bordered cards, one orange-red accent),
labelled sidebar, Ctrl/⌘+K trajectory search, status cards for the robot link,
controller FSM, cameras and glasses, responsive panels, fullscreen video, a
searchable motion library, and a playback timeline.

From the repository root:

```sh
.venv/bin/python tools/dashboard.py
```

Open **http://localhost:8090** in a desktop browser. Dependencies are `mujoco`,
`numpy`, and `Pillow`; these are already present in this workspace. On a fresh
virtual environment install them with `python -m pip install mujoco numpy Pillow`.
No JavaScript build step, external fonts, or hosted services are required.

The R1 is rendered by MuJoCo, not an illustration. Playback poses the fixed-base
model along the plan using joint interpolation and shows both predicted hand
paths. It is a **kinematic review**, not a physics/contact or balance simulation.
The cube is a scene prop; this view does not simulate a grasp. Plans are loaded
from `sim/plans/`, `tools/plans/`, and `recordings/` at startup. Malformed or
unsupported files are excluded. Restart after adding new files.

Use the play button or Space, drag the timeline, choose a playback speed, switch
between three camera angles, or choose another trajectory from the library.
These controls affect only the preview and are disabled while viewing the live twin.
To monitor measured robot movement instead, supply the live twin URL and select
**Live robot twin** in the panel.

## Trajectory control

The **Trajectory control** panel runs `tools/arm_lift.py IFACE --plan PLAN
--speed S --kp-scale K` as a subprocess, like `tools/reins_ui.py`. The tool's
own gates still apply: joint limits, the 0.5 rad/s speed cap, FSM 4/811 only,
and the tracking-error abort. Start the dashboard with the interface that
reaches the robot (default `en6`):

```sh
.venv/bin/python tools/dashboard.py --iface en6
```

1. **Preview.** The target is the trajectory in the preview. Choose another
   with **Change** or the library.
2. **Dry run.** The tool reads the pose and FSM, checks the plan, and publishes
   nothing. Its output streams into the console, and the FSM chip shows the
   controller state. On success the preview switches to the resolved plan
   (`sim/plans/arm_lift_dryrun.json`). It starts from the measured pose, so it
   shows what the robot would actually do.
3. **Execute.** This unlocks only after a successful dry run of the same target,
   speed and kp scale in the last 5 minutes. It asks for confirmation in a
   dialog that shows exactly what will run. The server always executes the dry
   run's settings, and each dry run allows only one execute.

**Abort** (or **Esc**) sends the tool SIGINT, which ramps the arm weight down
and saves the recording. Closing or reloading the tab aborts a running move, as
closing the Tk window does. Stopping the dashboard with Ctrl-C also aborts. The
remote stays the primary stop: Abort depends on the tool being responsive.
New recordings appear in the library automatically.

The header chips show `rt/lowstate` health from the twin server's `/status`
(`tools/cockpit.py` on port 8082), the FSM from the last dry run, and the
interface.

## Robot video

Existing services are reused. The dashboard does not start them or configure the
robot network:

- Head: `http://127.0.0.1:8081/cam` (from `tools/headcam.py YOUR_INTERFACE`).
- Left wrist: `http://127.0.0.1:8080/cam/0`.
- Right wrist: `http://127.0.0.1:8080/cam/2`.
- Live twin: `http://127.0.0.1:8082/twin` (from `tools/cockpit.py`); `--twin ''` disables it.

The wrist cameras require the existing Jetson camera service and forwarding
setup described in `CLAUDE.md`. Override URLs when services run elsewhere:

```sh
.venv/bin/python tools/dashboard.py \
  --head http://HOST:8081/cam \
  --left-wrist http://HOST:8080/cam/0 \
  --right-wrist http://HOST:8080/cam/2 \
  --twin http://HOST:8082/twin
```

The **All** tab shows the head camera and both wrists at once. The server reads JPEG/MJPEG inputs once per source and shares the latest frame
with browser clients. Offline streams retry automatically; images older than
three seconds are hidden. Status reflects arrival of valid JPEGs, not physical
capture timestamps from the camera.

## Glasses video

The current Spectacles Lens sends/receives trajectory data, **not video**. This UI
does not add a camera-video encoder to the Lens. Supply either:

1. A mirrored glasses window from your existing glasses tooling. Click **Share
   glasses window** and select that window in the browser picker. The selected
   content stays in the local video element; the dashboard does not upload it.
   Stop sharing with the panel button or browser sharing indicator. Choosing
   a regular desktop window does not make it a glasses feed—select the actual
   mirror you want to see.
2. An existing HTTP JPEG/MJPEG endpoint:

   ```sh
   .venv/bin/python tools/dashboard.py --glasses http://HOST:PORT/stream
   ```

A trajectory WebSocket URL, RTSP URL, or ordinary HTML page is not a video input.
Window capture requires a supported desktop browser and your explicit source
selection. Actual glasses capture must be verified with the connected device.

## Rendering and scope

Linux defaults to EGL; macOS defaults to CGL. If required by your graphics stack,
set `MUJOCO_GL=osmesa` or `MUJOCO_GL=glfw` before launch. Errors appear in the UI
and terminal. On Linux, offscreen EGL needs working system graphics drivers.

The dashboard binds to loopback only and never imports the robot SDK itself.
The preview never touches the robot. Only the trajectory control panel does,
and only by running `tools/arm_lift.py`. Every state-changing request needs the
per-process token, and Execute also needs the dry-run gate and the confirmation.
It starts no remote services. The Tk window `tools/reins_ui.py` offers the same
controls without a browser.

## Natural-language prompts

The **Describe the next move** panel accepts a request and shows observation,
localization, IK and validation progress. Choose **Simulation demo** to try
`touch the bottle` without cameras or API credentials. Choose **Calibrated robot
camera** with `--observation PATH.npz`, `OPENAI_API_KEY`, and
`REINS_VISION_MODEL` for OpenAI visual grounding on measured depth.

Generated approaches stop short of contact, are offered for explicit MuJoCo
preview, and cannot be executed through the legacy robot runner. See
[the perception/planning guide](../../core/README.md) for observation schemas,
stereo conversion, validation limits and remaining hardware integration.

**Auto context** now routes known gestures without visual recognition. Try
`wave your right hand` or `raise your left arm`. Object-directed commands such
as `point at the bottle` require calibrated object context. A fresh identical
observation can reuse its target grounding; the panel explains each decision.
Without measured context, gestures use the current simulation pose for preview
and explicitly report unknown physical clearance. Generated plans stay locked
against physical execution.

## Simulation-only voice workspace

Start the optional [voice service](../../voice/README.md), then run:

```sh
.venv/bin/python tools/dashboard.py --sim --port 8091 --voice-url http://127.0.0.1:8770/
```

`--sim` disables hardware runs on the server, camera/twin readers and calibrated
observations. The MuJoCo preview and local demo planner still work. The optional
voice panel uses computer audio. GPT-Live handles conversation and delegates
robot questions through its backend adapter; it does not submit the plan form.
The optional [cascaded mode](../../voice/CASCADE.md) instead dictates into the
existing prompt and exposes `window.ReinsVoice.speak(text)` for reply playback.
Neither mode authorizes motion. Omit `--voice-url` to keep the existing layout.
