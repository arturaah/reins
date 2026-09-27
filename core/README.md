# Core planning and reviewed control

The dashboard runs one agent/tool loop. The model chooses when to observe,
detect objects, compile a path and revise from rejection reasons within a bounded
budget. A completed validated motion automatically becomes a proposal and plays
in MuJoCo and paired glasses. One operator **Accept** runs it in the selected mode.
See the [dashboard guide](../tools/dashboard/README.md).

## Components

| Module | Responsibility |
|---|---|
| `tool_specs.py`, `reins_tools.py`, `tools/reins_mcp.py` | Model-visible tools, bounded planning budget, image observations and provider-independent MCP transport |
| `dashboard_chat.py`, `codex_chat.py`, `claude_chat.py` | Chat/provider lifecycle, cancellation and outcome feedback |
| `ik.py`, `generated_motion.py` | A5 hand-position IK and authored multi-waypoint gestures |
| `trajectory.py`, `motion_validation.py`, `motion_policy.py` | Exact sample resolution, joint/dynamics/model collision checks, workspace/table and base-motion policy |
| `robot_pipeline.py` | Immutable drafts, automatic proposal preview, one human Accept, rechecks, execution and results |
| `glasses_bridge.py`, `glasses_pairing.py` | Paired AR review and voice input with session/revision binding and revocation |
| `object_detection.py`, `detection_stream.py` | Local 2D detection and freshness-labelled image results |
| `contract/runtime.py` | Shared arm/base/hand payload and coordinator-approval validation |

`observe` returns actual camera images, frame IDs and receipt timestamps alongside
measured joints. It may include an explicitly synthetic robot-pose rendering.
`detect_objects` can use the same observation ID. Neither an image box nor a
pose rendering gives an object's measured position in robot-base metres.

## Draft and proposal lifecycle

1. `plan_hand_path` compiles all waypoints, pauses and any return using core IK.
   It validates the resolved single-arm trajectory and returns a server-owned
   draft ID. Failures return details for revision; they never relax constraints.
2. The model calls `propose_motion(plan_id, request_id)` when the complete motion
   is ready. The host also finalizes its latest valid draft after a successful
   model turn if the model omitted submission. An intermediate draft is not an
   approval request; a later failed revision prevents automatic submission of an
   older valid draft. `preview_plan` remains an optional model inspection tool.
3. Submission freezes the motion for review and automatically plays its preview
   in MuJoCo and paired glasses. Retrying a request ID is idempotent. Any different
   motion needs a new draft/proposal. No Generate, Show once or Submit click exists.
4. A single dashboard **Accept** or glasses acceptance pinch binds the revision and
   digest. Execution rechecks expiry, cancellation, measured starting pose,
   observation availability and deterministic validation before sending the
   same payload through the private actuator connection.
5. `get_motion_result` returns the outcome and available measured feedback.
   A claimed task success from the model is not physical completion evidence.

Configured `plan_base_motion` and `plan_hand_action` follow the same proposal
boundary. A base motion is one bounded velocity/duration command with a predicted
path and odometry feedback; it is not obstacle-aware navigation. Revo2 actions
use configured open/close poses, not contact-aware grasp synthesis. Firmware
presets are operator-only and separate from the model tools.

The `prompt_planner.py` adapter supports simple local gestures and final chat
replies containing `trajectory` or `robot_request`. These are submitted to planning
automatically; validated output becomes a previewing proposal awaiting Accept.
Planning failures appear in chat and drive bounded automatic revision without
extra user clicks. The old incremental `visual_policy.py` adapter
is retained for reference/offline work and is not the live dashboard fallback.

## Motion checks and limits

The shared arm path models both arms and held joints, samples intervening motion,
and checks joint margins, velocity, acceleration, native contacts, conservative
body envelopes and the configured measured table volume. Validation covers the
robot model and supplied geometry, not unseen environmental objects. The default
A5 authoring path controls hand position plus separately reviewed wrist roll;
there is no general full wrist-pose, dual-arm or contact controller.

Plans are resolved before review. Start-pose drift invalidates the proposal;
the streamer does not add an unseen lead-in or change the path after approval.
There is no metric depth estimation in the supported agent workflow. The older
calibrated RGB/depth modules (`perception.py`, `stereo.py` and related fixtures)
remain offline experiments; they are not required setup or the source of object
positions for current tools.

## Optional local detector

The dashboard runs without downloaded detector weights. For the open-vocabulary
OmDet-Turbo adapter, install its separate dependencies, then download the pinned
weights:

```sh
.venv/bin/python -m pip install -r core/requirements-detector.txt
.venv/bin/python tools/detect_objects.py --download
```

OmDet-Turbo uses PyTorch/Transformers and chooses CUDA, MPS or CPU. NanoDet is a
smaller OpenCV fallback when available. Detection returns normalized image boxes,
labels and scores. It does not provide persistent object identity, depth or
metric clearance. Ordinary MJPEG freshness measures receipt age, not synchronized
sensor capture time. Use `--detector` and the camera settings described by
`tools/dashboard.py --help`.

## Historical experience

`core/experience.py` adapts the team's `harness/experience.py` card renderer and
pose/path rendering to the coordinator. Each terminal proposal retains its exact
canonical payload and review identity, the operator decision, measured outcome,
starting pose, hand path and the selected observed image when present. Approval
is separate from completion: failed or cancelled approved proposals never become
success examples, and simulation is labelled explicitly.

`get_robot_context` retrieves relevant examples as historical text and real image
tool content. Images and operator notes are untrusted past data, never current
perception or authority to execute. Stored payloads have no replay endpoint.
Cards render lazily, so image rendering cannot delay an execution outcome.

The `experience` settings in `harness/config.yaml` apply to this adapter:
`enabled`, `dir`, `max_in_prompt` (capped at 6), and `card_width_px`. Optional
`max_entries` defaults to 128 and `max_bytes` to 64 MiB. The adapter stores its
private files under `experience.dir/reviewed`, apart from legacy offline cards,
and evicts oldest records when limits are exceeded. Tests with a custom relative
run directory keep this memory there. **Connections → Forget saved examples**
deletes the adapter's memory and saved images. Existing conversation context and
diagnostic session logs are separate; forgetting does not recall data already
sent to a model. There is no motion recording or replay UI.

## Tests

```sh
.venv/bin/python -m pytest -q core tools/test_dashboard.py tools/test_dashboard_http.py
```

Tests use real local geometry with fake robot/model adapters. Physical R1 motion,
camera alignment and Lens tracking require a separate commissioning run.
