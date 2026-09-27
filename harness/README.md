# Offline harness utilities and shared robot adapters

The supported operator workflow is the [dashboard](../tools/dashboard/README.md).
Its model plans with tools, submits one complete motion and waits for one human
approval. `core/robot_pipeline.py` is the execution coordinator. The former
per-step live harness and Tk control window are retired.

This package retains useful offline evaluation, demonstrations, perception,
robot-pose rendering, outcome feedback, episode logs and robot transport code.
Those modules are not a second supported live execution path.

## Supported entry points

```sh
# Current operator interface; use --sim for no hardware connection:
.venv/bin/python tools/dashboard.py --sim
.venv/bin/python -m tools.reins prompt "wave with the right arm"
.venv/bin/python -m tools.reins status
.venv/bin/python -m tools.reins stop

# Offline scripted evaluation, with no robot or model service:
.venv/bin/python -m harness sim "move your hand above the block" --vlm scripted

# Read-only real joints/cameras plus the selected model (no actuator commands):
.venv/bin/python -m harness dry-run YOUR_INTERFACE "inspect the task" --vlm claude-cli

# Capture observations or re-query an existing recorded model step:
.venv/bin/python -m harness packet --sim
.venv/bin/python -m harness replay RUN_DIR --step 0 --vlm scripted
.venv/bin/python -m harness measure-table YOUR_INTERFACE --watch
```

Provider-backed offline commands can make external model calls. The simulation
is a mock environment, and its success flag is an evaluation result, not proof
of physical task success. `replay` above re-queries a saved prompt/image packet;
it does not replay commands onto a robot.

`python -m harness live` refuses before creating a robot backend. `--no-confirm`
live operation is unavailable. `tools/arm_lift.py` retains read-only diagnostic
helpers, but `--execute` refuses before importing the SDK. `tools/teach.py` is a
retired command; making joints compliant is not represented by complete-motion
approval. `tools/reins_ui.py` only opens the dashboard. `tools/start_all.sh` starts
the dashboard without starting services on the Jetson. `tools/harness_live.sh`
submits a task through the operator HTTP client.

## What remains useful

| Area | Current use |
|---|---|
| `prompts.py`, `actions.py`, `interpreter.py`, `loop.py` | Offline incremental-policy experiments, recovery and behavior tests |
| `kinematics.py`, `safety.py`, `executor.py` | Offline policy evaluation; shared trajectory checks are reused where applicable |
| `perception.py`, `poseview.py` | Head/wrist observations and a labelled synthetic view of robot configuration |
| `feedback.py`, `demos.py`, `stats.py` | Operator notes, optional demonstration summaries/contact sheets and inference metrics |
| `recorder.py` | Episode artifacts and offline dataset exports, separate from the dashboard UI |
| `vlm/` | Legacy experiment providers, scripted fixtures and recorded-response evaluation |
| `robot/arm_client.py`, `robot/arm_stream.py` | Private authenticated coordinator connection and supported arm/base actuator owner |
| `robot/odometry.py` | Measured base displacement from robot state; predictions remain separately labelled |
| `robot/hand_client.py`, `robot/revo2.py` | Capability-gated, reviewed Revo2 open/close transport |
| `robot/dry_run.py`, `robot/lowstate.py`, `sim/mock_robot.py` | Read-only telemetry and simulation adapters |

Demonstrations are optional context, not a required library of predefined skills.
Retained datasets and contact sheets can be evaluated without restoring a
recording/replay control panel. Offline `--confirm` reviews each *simulated*
evaluation step and remains useful for tests; this is not the dashboard's review
protocol or permission to send physical actions.

## Runtime boundary

The dashboard supplies the authenticated private bridge capability and a fresh
approval receipt for the exact motion payload. The bridge checks the receipt,
consumes each proposal once, and independently validates commands. An ID or
SHA-256 digest alone is not authorization. See the
[runtime contract](../contract/README.md).

Connecting the dashboard reads telemetry. Arms are engaged only for an approved
arm motion, with the supported onboard controller state selected by the operator.
Stream handling remains responsive to stop, disconnect and heartbeats during a
motion; telemetry/tracking watchdogs remain active. Walking sends an explicit
stop on completion or cancellation. Direct unreviewed frame, walk and hand
commands are not supported public controls.

Revo2 support and bounded walking are disabled unless configured. A hand closing
is not proof of a grasp. Odometry measures movement, not obstacle clearance or
navigation success. A base preview does not simulate balance or foot contacts.
The model must not infer permission to move the base from a request about an arm
or an object.

## Configuration and tests

`config.yaml` holds robot/model references, workspace/table limits, capability
settings and defaults for retained offline experiments. The dashboard accepts
`--harness-config FILE`. Measure workspace geometry independently; the retired
teaching command is not a setup step. Interface names can change after a cable
replug; confirm the robot-side connection before hardware work.

```sh
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q harness/tests contract/tests core/test_robot_pipeline.py
```

The tests use fake DDS publishers, loopback bridges and scripted models. Existing
hardware observations in repository history do not establish that the newly
integrated dashboard/AR/control workflow has been physically commissioned.
