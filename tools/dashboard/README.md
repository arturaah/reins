# Reins dashboard and reviewed robot control

The dashboard is the main operator interface. It connects the trajectory planner,
visual harness and Spectacles to one proposal/review/execution pipeline.

## Start

From the repository root:

```sh
python3 -m pip --python .venv/bin/python install -r requirements.txt
.venv/bin/python tools/dashboard.py --iface YOUR_ROBOT_INTERFACE
```

The server prints its local browser URL (normally http://localhost:8090).
An occupied default HTTP port advances to 8091–8099. Use `--port 0` for a free port.
The browser API stays on loopback. No JavaScript build step is needed.

The dashboard starts in **Simulation** mode. Startup only discovers firmware
gestures; it does not engage the trajectory controller or run a preset.

## One motion workflow

1. Describe a motion in chat, or prepare a 2 cm nudge, 5° wrist roll or home pose in **Robot control**.
2. The core planner compiles the path and checks joint limits, speed,
   acceleration and the swept arm volume, including the other arm.
3. The resolved proposal appears automatically in MuJoCo and paired glasses.
4. Approve or reject that exact proposal in either interface. The approval button
   explicitly says whether it applies to simulation or the physical robot.
5. The runtime checks the current pose again, sends the reviewed samples unchanged,
   and records the result. A new move requires a new approval.

The model can propose paths and request context. It has no approval or execution
tool. Approvals expire after two minutes and cannot be reused for another revision.
Pose changes invalidate execution rather than adding an unseen approach motion.

There is no replay library or recording workflow. Session diagnostics under
`runs/dashboard/<session>/` retain proposals, decisions and execution feedback;
they are not offered as executable recordings.

### New gestures and visual fallback

A new gesture does not need a predefined trajectory. Chat can author a single-arm
path with intermediate waypoints, pauses and a return. Both the chat tool and
Generate preview button use the same host-managed budget: an initial draft plus
up to two revisions.

If primary planning fails or lacks object context, the visual harness gathers
fresh head/wrist camera frames and optional wearer frames. It proposes small,
non-contact steps through the same core compiler and validation. Each step is
reviewed separately; open-loop action chunks are disabled in the dashboard.

The **Visual fallback** settings select Codex CLI, Claude CLI, OpenAI API or
Anthropic API. Codex and Claude reuse the isolated, cancellable dashboard CLI
transport. API providers require their optional SDK and account configuration.
For OpenAI, set `OPENAI_API_KEY` and `REINS_CHAT_MODEL` (or `REINS_VISION_MODEL`).
For Anthropic, install `anthropic`, set `ANTHROPIC_API_KEY`, and configure
`vlm.model` in the harness config. There are no model calls until requested work
needs them. The selected provider receives the camera frames used for visual planning.

A missing head-camera frame produces **Needs context**, with a **Retry with
cameras** button. The wearer view is supplementary; an uncalibrated wearer camera
does not define metric object positions or robot-relative movement directions.
A browser-shared glasses window is display-only and is not sent to the visual policy.

## Physical robot control

Enter the measured table height in metres in the robot-base frame, then click
**Connect & hold arms**. This is an explicit physical action: the bridge takes
arm control and holds the current pose. The robot must already be in a supported
controller state (FSM 4 or 811).

The dashboard connects to the local harness streamer or starts one if needed.
The optional `--harness-config FILE` configures the streamer address, workspace,
model-provider settings and home poses. Camera endpoints are configured on the
dashboard command line.

The arm path is checked against the robot model and a table volume covering the
configured forward workspace. RGB images and human review provide context; they
are not a calibrated obstacle map. The fixed-base model does not validate balance
or walking. This pipeline supports one arm, including wrist roll for incremental
actions, but no finger articulation, grasping or physical contact.

**Stop / release** cancels planning, invalidates approvals, interrupts trajectory
streaming and releases this arm controller. Escape also stops connected arm
control. Loss of both authenticated operator heartbeats for ten seconds releases
control. The streamer separately handles client disconnects, telemetry staleness
and tracking failures.

A cross-process lease prevents Reins firmware presets, the legacy arm tool and
the streamer from taking local control simultaneously. Release trajectory control
before using a firmware preset. An onboard preset is run by the firmware; the
trajectory Stop button does not cancel it. Its separate **Release arms** preset
is available after the firmware action finishes.

## Spectacles review

The dashboard includes the AR WebSocket service; a separate `plan_feed.py`
process is not needed for this workflow.

1. Open **Connections → Glasses motion review**.
2. Set the Lens's `websocketUrl` to
   `ws://DASHBOARD_COMPUTER_LAN_IP:8765`.
3. Copy the session's review token into the Lens's `reviewToken` input.
4. Scan the shoulder markers to anchor the robot frame.
5. Double-pinch right to approve or left to reject the displayed proposal.

Use the updated `spectacles/Assets/R1Trajectory.js`. The Lens authenticates,
sends operator heartbeats, and includes the exact proposal digest in its decision.
An idle/cancelled pipeline clears the AR proposal. A paired Lens does not replace
missing live data with a mock trajectory. Prototype mock paths require explicitly enabling the Lens demoMode input.

`--glasses-port NUMBER` changes the AR port; zero chooses a free one. The default
listener is `0.0.0.0` for access from the glasses. Only authenticated clients
receive proposal paths or submit decisions. Pairing tokens change when the
dashboard restarts. Use the local robot network; the default WebSocket transport
is not encrypted.

The older standalone feed remains available for development. A review-enabled
feed now requires `--review-token-file FILE`; read-only visualization does not.
It uses the existing proposal ID/hash mailbox. Dashboard review uses the in-memory
runtime directly.

## Cameras and views

Existing camera services are reused:

- Head: `http://127.0.0.1:8081/cam` from `tools/headcam.py YOUR_INTERFACE`.
- Left/right wrist: `http://127.0.0.1:8080/cam/0` and `/cam/2`.
- Optional live twin: `http://127.0.0.1:8082/twin`.
- Optional wearer feed: `--glasses http://HOST:PORT/stream`.

Override with `--head`, `--left-wrist`, `--right-wrist`, `--twin`, and
`--glasses`. Empty URLs disable sources. These are JPEG/MJPEG inputs, not
trajectory WebSockets or RTSP. Camera status reflects receipt of a usable frame,
not synchronized physical capture timestamps.

The browser can also display a mirrored glasses window using its screen-sharing
picker. This does not add video capture to the Lens or make the window available
to server-side planning.

Linux rendering defaults to EGL and macOS to CGL; set `MUJOCO_GL` before startup
if the graphics environment requires another backend.

## Implementation and checks

- `core/robot_pipeline.py`: task state, primary/fallback routing, shared review,
  pose freshness, execution and operator liveness.
- `core/trajectory.py`: five-joint resolution, smooth command-grid resampling,
  plan digest and starting-state checks.
- `core/ik.py` and `core/motion_validation.py`: geometry and full-path validation.
- `core/visual_policy.py`: harness policy adapters and dashboard camera packets.
- `core/glasses_bridge.py`: authenticated review and heartbeat protocol.
- `harness/robot/arm_stream.py`: cancellable hardware streaming and independent checks.
- `core/robot_lease.py`: local cross-process ownership.

```sh
.venv/bin/python -m pytest -q harness/tests contract/tests spectacles/tests core \
  tools/test_dashboard.py tools/test_dashboard_http.py
```

Tests use scripted models, fake telemetry/publishers and local sockets. Browser
checks can use the same fake backend; they do not qualify the physical robot,
camera calibration, network or Lens tracking.

The action schema requires all properties, using an empty plan array for a
single step, following the [official Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

## Simulation lockout and voice

Run `--sim` to disable hardware controls, robot feeds and calibrated observations.
Firmware discovery is also disabled. Generated motions stay in the local preview.

Pass `--voice-url http://127.0.0.1:8770/` to embed the optional local voice service.
Only a local HTTP origin is accepted. Dictation inserts text into the chat field;
review and send it yourself. The **Read reply** button sends the reply to the
voice panel for playback. Voice input never approves a motion.
