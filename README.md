<p align="center"><img src="reins.png" alt="Reins logo" width="480"></p>

# Reins

**Any VLM. Your hands on the reins.**

Reins is a model-independent harness for the Unitree R1. The assistant observes,
plans and revises with tools. The operator reviews a complete motion before the
coordinator can send it to the robot.

## One planning and control pipeline

```text
Dashboard / Spectacles voice / operator CLI
                    ↓
Agent → observe / detect / plan / validate / preview
                    ↓
Immutable complete motion → human approval in dashboard or glasses
                    ↓
Fresh-state recheck → authenticated actuator bridge → measured outcome
```

Planning and simulation previews need no approvals. `propose_motion` submits a
finished draft for one human decision. Changing a path or requesting another
motion creates a new proposal; approval never lets the model keep steering.
The model has no approval, execution or firmware-preset tool.

Novel gestures use core IK and whole-path validation; they do not need a recorded
or predefined trajectory. Camera images, planning failures and measured outcomes
return to the same agent. There is no separate per-step visual fallback, replay
library, or recording UI.

The coordinator also remembers reviewed proposals as bounded local picture cards:
the original pose/path, an observed camera image when available, the operator's
decision and the actual outcome. Relevant examples return to the agent as
historical context; approval, physical completion and simulation stay distinct.
Connections shows the saved count and a **Forget saved examples** control.

## Start in simulation

Use Python 3.10 or newer with a working MuJoCo renderer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/dashboard.py --sim
```

Open the printed local URL. Choose an installed/signed-in Codex or Claude CLI,
or configure an optional API provider. No model call happens until a task is
submitted. Optional 2D detector dependencies and weights are described in the
[core guide](core/README.md).

The operator CLI uses the same running dashboard:

```sh
.venv/bin/python -m tools.reins prompt "Blow a kiss with the right arm, then return"
.venv/bin/python -m tools.reins status
.venv/bin/python -m tools.reins stop
```

Use `--url http://127.0.0.1:PORT` before the command for a different port.
Approvals remain in the dashboard or paired glasses.

## Robot, glasses and voice

For robot telemetry, start without `--sim` and supply `--iface YOUR_INTERFACE`.
**Connect robot** reads state; it does not engage the arms. An approved motion
uses the private coordinator-to-streamer channel. Cameras and any robot-side
services are configured explicitly, not started through the old Tk launcher.

The dashboard provides reviewed arm nudges, wrist turns and home motion. Bounded
base movement and Revo2 open/close are available only when configured. Upstream
odometry, robot-pose rendering, speech input and hand adapters are retained.
Human-only firmware gesture buttons invoke the onboard service; their internal
paths are not generated or collision-checked by Reins.

Pair Spectacles in **Connections → Glasses motion review**. Device pairing
survives restarts and can be revoked. Drafts show no approval controls; complete
proposals bind the decision to their revision and digest. Spectacles speech
submits a task to the same agent, never an approval. See the
[dashboard setup guide](tools/dashboard/README.md) and [Lens guide](spectacles/README.md).

The optional [voice service](voice/README.md) runs in its own environment.
Use `--backend dashboard --dashboard-url http://127.0.0.1:8090` to submit spoken
tasks to the same agent, or retain the standalone test/conversation modes.
Dashboard `--voice-url http://127.0.0.1:8770/` embeds the browser voice page and
allows paired Spectacles to relay live PCM. [R1 speaker output](spectacles/VOICE.md)
is separately opt-in with `--output r1`; speech never approves motion. ASR
transcripts remain available without a live voice service.

## Current limits

- A complete proposal contains one single-arm trajectory, one bounded base
  motion, or one hand open/close action. There is no coordinated dual-arm or
  mixed arm/base/hand task execution under one approval.
- Camera detections are 2D. There is no metric depth, calibrated object location,
  measured obstacle map or contact planner. Image-informed free-space motions
  are hypotheses, not verified reaches or grasps.
- MuJoCo is a kinematic preview. It does not prove balance, gait, friction,
  clearance against unmodeled objects, or successful grasping. AR alignment and
  these integrated hardware paths still need commissioning on the actual R1.
- Diagnostic proposal/decision/outcome logs are kept. Offline evaluation and
  dataset utilities remain in `harness/`; their former live CLI and direct
  teaching/replay execution paths are retired.

## Development

`core/robot_pipeline.py` owns the proposal lifecycle; `contract/runtime.py` owns
shared motion/approval validation; `harness/robot/arm_stream.py` is the supported
arm publisher. See the [runtime contract](contract/README.md) and
[offline harness guide](harness/README.md).

```sh
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q core harness/tests contract/tests spectacles/tests tools/test_*.py
```

The suite uses fake robots/models and loopback transports. Lens protocol tests
use Node when available (`REINS_NODE` can select it); real headset, ASR and robot
operation are separate physical commissioning tasks.
