# Reins dashboard

The dashboard is the operator interface for one reviewed motion pipeline. The
assistant can observe cameras, detect objects, compile paths with core IK,
inspect rejection reasons and revise drafts. A complete motion becomes a
proposal only when it is submitted for human review.

## Start locally

```sh
python3 -m pip --python .venv/bin/python install -r requirements.txt
.venv/bin/python tools/dashboard.py --sim
```

Open the printed browser URL, normally http://localhost:8090. `--sim` disables
hardware connection, firmware discovery, physical controls and robot feeds.
No model request runs until you send a task. Use a configured assistant in the
chat selector. The optional provider SDKs and credentials depend on that choice.

The HTTP API listens on loopback. An occupied default port advances through
8091–8099; `--port 0` chooses a free one. No JavaScript build is required.

## Prompt → preview → approve and run

1. Describe a task in chat. The assistant chooses observations and planning tools;
   planning and revisions do not need approvals.
2. Inspect the draft in MuJoCo or paired glasses. A draft cannot be approved.
3. The assistant submits the complete motion with `propose_motion`. For a manual
   chat suggestion, **Generate preview** creates a draft and **Submit complete
   motion for review** submits it. Robot control buttons submit their short motion
   directly for review.
4. Approve or reject once, in the dashboard or glasses. The decision binds the
   complete motion, revision and digest. Any further motion needs a new proposal.
5. The coordinator rechecks the current pose and validation, runs the approved
   motion, and returns measured feedback. The last outcome remains visible.

There is no per-step camera fallback or permission to keep steering after
approval. Camera context and planning errors go back to the same assistant
before it submits a new complete motion. Model tools cannot approve, execute a
preset or alter a pending approved path.

Approvals expire after two minutes. A changed start pose invalidates execution;
no unseen approach motion is added. **Stop / release** cancels the current work
and invalidates its proposal. Escape also stops connected control.

There is no replay library or recording UI. Diagnostic proposals, decisions and
measured results are retained by the coordinator for debugging.

The optional operator CLI uses the running dashboard's same entry point:
`python -m tools.reins prompt "wave with the right arm"`. Its `status` and `stop`
commands share the same session; it has no approval or direct execution command.
The old Tk and live harness launchers now lead to the dashboard. Teaching and
direct legacy `arm_lift --execute` are retired from the supported control path.

## Connect the physical robot

Run without `--sim`, with the interface connected to the R1:

```sh
.venv/bin/python tools/dashboard.py --iface YOUR_ROBOT_INTERFACE
```

Enter the measured table height in metres in the robot-base frame and click
**Connect robot**. Connection reads telemetry; it does not engage the arms.
An approved arm motion takes control through the single arm streamer. The
operator must select the supported onboard controller state using the remote.
The configuration is selected with `--harness-config FILE`.

The manual controls propose 2 cm nudges, 5° wrist turns and a home pose. Walking
and Revo2 hand controls are enabled only when the coordinator reports support.
Walking previews show planned base displacement, not simulated gait, balance or
foot contact. Hand actions show their description; the A5 model has no animated
finger joints. Disabled capabilities cannot be enabled by model text.

Firmware presets remain separate human buttons and are coordinated by the same
pipeline ownership gate. They run onboard; the dashboard does not possess their
joint trajectory. Their **Release arms** preset is distinct from stopping a
reviewed trajectory. Follow the current controller state reported by the robot.

Arm validation checks sampled motion, robot geometry and the configured table
volume. RGB detections locate objects in an image, not in 3D. Image-guided motions
are approximate; this pipeline does not claim a measured obstacle map or contact
control. Hardware behavior still requires commissioning on the actual robot.

## Pair Spectacles

The dashboard starts its AR bridge; do not start a second `plan_feed.py` on the
same port.

1. Open **Connections → Glasses motion review**, name the device and select
   **Pair a device**.
2. Set the Lens `websocketUrl` to `ws://DASHBOARD_COMPUTER_LAN_IP:8765`.
3. Save the displayed `deviceId` and `reviewToken` in the updated
   `spectacles/Assets/R1Trajectory.js` Inspector inputs. The token is displayed
   once; the dashboard stores its hash.
4. Scan the shoulder markers. Draft paths are labelled as drafts; only the
   complete proposal can be approved or rejected.
5. Double-pinch right to approve or left to reject the current complete proposal.

Pairing survives restarts. Revoke a device in Connections to close its active
session and remove its authority. Pairing metadata is stored under
`REINS_STATE_DIR`, or `~/.local/state/reins/` by default; keep that directory when
restarting the dashboard. `--glasses-port NUMBER` changes the AR port; zero picks
a free port. The default listener is `0.0.0.0`; the default WebSocket connection
is not encrypted, so use the local robot network.

Authenticated Lens voice tasks go to the same assistant as browser prompts.
A voice task never approves a motion. The bridge deduplicates command IDs and
requires the current authenticated session. Camera-window sharing is separate
from the AR trajectory and voice protocol.

## Cameras, measured state and voice

The dashboard reuses these JPEG/MJPEG camera endpoints:

- Head: `http://127.0.0.1:8081/cam` from `tools/headcam.py YOUR_INTERFACE`.
- Left/right wrist: `http://127.0.0.1:8080/cam/0` and `/cam/2`.
- Optional wearer camera: `--glasses http://HOST:PORT/stream`.
- Optional existing live twin: `--twin http://127.0.0.1:8082/twin`.

Override with `--head`, `--left-wrist`, `--right-wrist`, `--glasses`, or `--twin`;
empty values disable camera sources; an empty `--twin` selects the local measured view. Receipt freshness is checked, but capture times
are not synchronized. The selected model receives frames when using observation
tools. Browser window sharing is display-only and is not a server camera source.

The Robot control panel exposes measured state and the last motion's measured
end pose. The default Live robot twin renders cached coordinator telemetry locally, with a fixed base and no global localization. It needs no cockpit or relay process and remains unavailable until connected. `--twin URL` can select an external subscribe-only view.
Linux rendering defaults to EGL and macOS to CGL; override `MUJOCO_GL` if needed.

For the existing local voice service, pass
`--voice-url http://127.0.0.1:8770/`. Only a local HTTP origin is accepted.
Dictation inserts text in the chat field for review and submission; **Read reply**
plays an assistant reply through the voice panel. It cannot approve motions.

## Offline verification

```sh
.venv/bin/python -m pytest -q core tools/test_dashboard.py tools/test_dashboard_http.py \
  harness/tests contract/tests spectacles/tests
```

Tests use scripted models, fake robot state and loopback sockets. They do not
publish on robot DDS, start Jetson services or qualify physical robot operation.
