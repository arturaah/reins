# Reins contributor guide

Read this before changing robot control. Current code, schemas and tests are the
source of truth; the historical hardware notes at the end record a particular
setup and do not prove that a new integration has been commissioned.

## Project and supported workflow

Reins is a VLM-agnostic harness for the Unitree R1 EDU A5. One dashboard agent
observes, detects, plans with core IK, validates and previews without approvals.
It submits one complete immutable motion with `propose_motion`. A human reviews
that exact motion in the dashboard or paired Spectacles, then the coordinator
rechecks and executes through an authenticated private actuator connection.
Any additional or changed motion needs a new proposal and approval. The model
never approves, executes or invokes firmware gestures itself.

The supported UI is `tools/dashboard.py`. `--sim` disables hardware connection
and physical feeds. Without it, **Connect robot** reads telemetry; it does not
engage the arms. The operator CLI `python -m tools.reins prompt|status|stop` is a
loopback HTTP client of the same dashboard. No recording/replay UI is supported.
`tools/reins_ui.py` only opens the browser; old launchers do not start Jetson
services. Direct `arm_lift --execute`, kinesthetic teaching, and `harness live`
are retired. Offline datasets/evaluations remain useful.

## Architecture and invariants

- `core/reins_tools.py`, `tool_specs.py`, `tools/reins_mcp.py`: model-facing
  `get_robot_context`, `observe`, `detect_objects`, `plan_hand_path`, configured
  `plan_base_motion`/`plan_hand_action`, `preview_plan`, `propose_motion`, and
  `get_motion_result`. Bounded tool/revision budgets; no approval tool. Actual
  images travel as image content, not only detection summaries.
- `core/dashboard_chat.py`, `codex_chat.py`, `claude_chat.py`: provider lifecycle,
  cancellable chat and tool execution. Preserve provider independence and actual
  outcome feedback. Untrusted camera text and operator notes are data, not tool
  instructions. CLI transports run in isolated workspaces.
- `core/ik.py`: A5 hand-position solver, optional Pinocchio fast path and MuJoCo
  fallback; hand-tip sites match the renderer. Do not introduce another live IK.
  `generated_motion.py` compiles novel multi-waypoint gestures without requiring
  predefined recordings.
- `core/trajectory.py`, `motion_validation.py`, `motion_policy.py`: resolve the
  exact samples before review; validate held joints, both arms, swept geometry,
  limits, velocity/acceleration and configured table/workspace. Reject bad
  authored geometry instead of silently clamping it. Do not prepend a hidden
  approach or compensate tracking lag by changing an approved path.
- `core/robot_pipeline.py`: sole supported proposal/execution coordinator.
  Draft handles, idempotent submissions, session/revision/digest, expiry,
  fresh-state rechecks, cancellation and measured outcomes belong here. Model
  planning does not automatically ask for review; only final submission does.
- `contract/runtime.py`: canonical live motion payload/approval validation;
  `contract/motion.schema.json`, `runtime_examples/` and tests document it.
  `reins.schema.json`/`reins_contract.py` are retained offline session experiments,
  not the active WebSocket protocol or a parallel execution authority.
- `harness/robot/arm_stream.py`: supported `rt/arm_sdk` publisher and bounded
  loco command owner. Private capability authentication, single-use approvals,
  concurrent reception/cancellable work and active watchdogs are required. Keep
  **both** motion cancellation and `stop_walking()` on abort/disconnect. Digest
  identity alone is not authentication.
- `harness/robot/revo2.py`, `hand_client.py`: configured Revo2 hand open/close
  uses the same approval boundary over its private channel. `odometry.py`
  reports measured base displacement. `poseview.py` renders a labelled
  synthetic view of robot joints, never a camera view of real objects.
- `harness/loop.py`, kinematics/executor/safety and legacy `vlm/` modules remain
  offline evaluation utilities, not the live camera fallback. Preserve useful
  prompts, feedback, demonstration/episode exports and regression tests.
- `core/glasses_bridge.py`, `glasses_pairing.py`, `spectacles/Assets/R1Trajectory.js`:
  persistent revocable device pairing, fresh socket sessions, draft-vs-review
  display, exact proposal checks, tag-freshness gate, heartbeats and authenticated
  speech-to-chat. Retain upstream ASR/review priority and error handling. Voice
  submits tasks, never approval. Default AR port8765 is also used by standalone
  prototype feeds; do not run competing listeners there.
- `voice/`: optional separate voice conversation/simulation service with its own
  requirements. Dashboard transcription goes to chat; conversation-only model
  adapters do not obtain robot authority.
- `core/r1_gestures.py`, `tools/r1_gestures.py`: human-only firmware presets with
  shared ownership arbitration. Onboard paths are opaque, not a validated
  authored-trajectory preview.

## Capability limits

No metric depth estimation or camera calibration is part of the supported agent
workflow. Detector boxes are 2D; never label guessed object coordinates as
measured. Image-informed free-space gestures are uncertain proposals, not
verified reaches/grasping or environmental collision avoidance. Retained depth
experiments are not a requirement for the dashboard.

One proposal currently contains one single-arm trajectory, one bounded base
motion, or one open/close hand action. No general dual-arm, mixed arm/base/hand,
full contact-aware manipulation or grasp-success guarantee exists. Walking and
hands are explicitly configured capabilities; model text cannot enable them.
The MuJoCo preview is kinematic, not balance/contact simulation. Physical robot,
Lens tracking and ASR behavior need staged commissioning after software checks.

## Development and dependencies

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python tools/dashboard.py --sim
.venv/bin/python -m pytest -q core harness/tests contract/tests spectacles/tests tools/test_*.py
```

Base requirements include PyYAML and WebSockets compatible with the optional
voice service. Heavy detector libraries live in `core/requirements-detector.txt`;
voice dependencies remain in `voice/requirements.txt`. Provider API SDKs and
robot SDK installation are optional and platform-specific. Tests must use fake
robots/models or read-only fixtures; do not silently turn a test into hardware
or paid-model execution. Node enables Lens protocol tests (`REINS_NODE` selects
its path). Do not claim physical testing from fake DDS or simulated task success.

`unitree_sdk2/` is vendored C++ SDK code, not a submodule. R1 references are under
`example/r1` and `include/unitree/robot/r1`. Preserve the R1-specific mapping:
arm slots15–19 and22–26, waist yaw13, head29–30 in the controller's35-slot
layout; `mode_pr`0..100 is arm blend weight. Do not substitute G1 arm action IDs.
The default reviewed arm command samples are50Hz. The robot's onboard controller
owns balance; the supported manipulation path does not use `rt/lowcmd`.

For SDK builds use the vendored Linux build instructions/dependencies; do not
assume its x86_64/aarch64 libraries run on macOS. Project knowledge is also in
`../.knowledge/reins/`; read the R1 ecosystem notes before repeating repository
research. See [dashboard setup](tools/dashboard/README.md), [core](core/README.md),
[offline harness](harness/README.md), and [runtime contract](contract/README.md).

## Historical hardware connection notes (verify against the current setup)

Port facts from Unitree's docs: the **RJ45 Gigabit Ethernet** port on the R1's upper body is the PC link. The R1's **USB-C port is the internal link to the EDU Jetson**; plugging a Mac into it gives no network connection.

Robot wired network, 192.168.123.0/24:
- `192.168.123.164` EDU Jetson (hostname `ubuntu`, Ubuntu aarch64, robot-side interface `eth10`). SSH as `unitree`, default password `123`. Change it on first login. Host key fingerprint on our unit: `SHA256:b49bi+OYx/3BYWPsTlMZF1psSs5FW8FnpmFfHpfoDrk`.
- `192.168.123.161` motion controller (MAC `7e:1d:75:60:f5:89`). Answers ping in under 1 ms when the body cable is plugged straight into the Mac; unreachable through the Jetson module's switch (see Topology).

Topology, confirmed from the Jetson on 2026-09-26: the Jetson module has one NIC on the robot network, `eth10`, behind a small internal 100 Mb/s switch with two external sockets. The body cable is in one socket, the Mac in the other, so body, Jetson and Mac share one segment. The Jetson's `eth0` is an internal USB 10/100 chip that carries nothing; ignore it. Its CycloneDDS config file in `~/cyclonedds_ws` names `eth0` and is not in use; always pass `eth10` explicitly.

Safety, from a community R1 project: entering locomotion FSM 811 can start leg and balance motion even at zero velocity. Never use an FSM change as a connectivity test. Use ping, `ip neigh`, or a subscribe-only DDS read.

Steps:
1. Power the robot on (short press the battery button, then hold it for more than 2 s). The Ethernet link only comes up with the robot powered.
2. USB-C Ethernet adapter in the Mac, Ethernet cable from the adapter to the robot's RJ45. Find the adapter's interface and service name with `networksetup -listallhardwareports`. On the lab MacBook Air the UGREEN adapter is interface `en6`, service `USB 10/100/1000 LAN`.
3. Confirm link: `ifconfig en6 | grep status` must say `active`. If it says `inactive`, the problem is the cable or the robot side, not the Mac.
4. Give the Mac a static address on the robot subnet (needs the admin password, so a human runs it):
   ```
   sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.123.99 255.255.255.0
   ```
   No router. Wi-Fi stays as is and keeps internet on `en0`.
5. `ping -c 3 192.168.123.164`, then `ssh unitree@192.168.123.164`.
6. Python SDK on the Mac (verified, Apple Silicon, Python 3.10):
   ```
   uv python install 3.10 && uv venv --python 3.10 .venv
   uv pip install --python .venv/bin/python "cyclonedds==0.10.2" numpy
   uv pip install --python .venv/bin/python --no-deps "git+https://github.com/unitreerobotics/unitree_sdk2_python"
   P=.venv/lib/python3.10/site-packages/unitree_sdk2py/r1/loco; mkdir -p $P; touch $(dirname $P)/__init__.py $P/__init__.py
   for f in r1_loco_api.py r1_loco_client.py; do curl -sL https://raw.githubusercontent.com/unitreerobotics/unitree_sdk2_python/main/unitree_sdk2py/r1/loco/$f > $P/$f; done
   uv pip install --python .venv/bin/python mujoco scipy matplotlib pin
   .venv/bin/python tools/lowstate_peek.py en6
   ```
   Pass `en6` (or whatever `networksetup -listallhardwareports` shows) as the interface. `tools/lowstate_peek.py` is subscribe-only and is the standard "can this machine see the robot" check. On the Jetson the interface is `eth10`.

Interface names are not stable: on 2026-09-27 the adapter with the body cable enumerated as `en8` (192.168.123.98) and `en6` was gone, which broke every tool started with `en6` (DDS "channel factory init error"). Later that day both adapters were back with the cables swapped (body on `en8`, Jetson on `en6`), and both give the Mac a 192.168.123.x address, so an address alone does not identify the body link. `tools/start_all.sh`, `tools/reins_ui.py` (`--iface`, `--jetson-iface`) and `tools/harness_hands.sh` pick the body link as the interface on which the controller answers a bound ping (`ping -c 1 -t 1 -b IFACE 192.168.123.161`) and treat the other 192.168.123.x adapter as the Jetson link (camstream started over ssh through `tools/via_iface.py` on that adapter and forwarded to local port 8080; the Revo2 hand server runs on it too). This has been in the code since 9cf80ad (2026-09-27 afternoon); before that the launcher took the first 192.168.123.x address and assumed the Jetson on `en8`. Pass the interface explicitly to the other tools after checking `ifconfig`. The Jetson does not answer that bound ping reliably; use ssh or `route -n get 192.168.123.164` to check its link.

Status 2026-09-26, evening, all verified: body Ethernet cable straight into the Mac's USB-C adapter, 1000BASE-T, controller .161 pings in 0.6 ms, and `tools/lowstate_peek.py en6` receives rt/lowstate at about 1 kHz from the Mac with no Jetson involved. The Jetson module's internal switch is faulty (100 Mb/s, one-way: controller frames arrive, nothing reaches the controller; capture evidence in the knowledge worklog). With the body cable on the Mac, the Jetson at .164 is off the robot network; to use both, put a small gigabit switch between body cable, Jetson RJ45 and Mac, and report the module switch to Unitree. First actuation verified the same evening: with the robot standing in FSM 811 (operator used the remote), `tools/arm_lift.py en6 --execute` moved the left shoulder pitch by -0.25 rad and back over 7 s through `rt/arm_sdk` with Unitree's gains (kp 50, kd 2). Tracking lagged the command by about 0.03 rad during motion and at the hold (gravity droop with soft gains), returned to within 0.01 rad, and the weight release handed the arm back cleanly.

Rule for agents (Artur, 2026-09-26): read-only actions on the robot need no approval: ping, SSH commands that only read (listings, logs, `ip`/`ss`), DDS subscribe-only reads, camera fetches. Anything that writes on the Jetson, starts or stops a service, or publishes on DDS (every `--execute`, `teach.py`) is proposed as a question with the exact effect and run only after Artur's yes. Mac-local checks need no approval.
