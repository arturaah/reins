# Shared planning and reviewed execution

The current operator workflow is described in the
[dashboard guide](../tools/dashboard/README.md). `robot_pipeline.py` coordinates
the core trajectory planner, visual harness fallback, dashboard/glasses approval
and the hardware bridge. `trajectory.py` resolves and validates the exact samples
that are reviewed and streamed.

The calibrated-object preview implementation documented below is an optional
planning source. Its saved files remain preview-only for legacy executors.
Physical execution is authorized only by the current dashboard proposal, with
fresh-state checks and a human decision.

# Prompt-to-approach preview

The dashboard now accepts natural-language requests, grounds a target in a
calibrated RGB/depth observation, uses the existing R1 A5 IK solver, and validates
an approach before offering it for MuJoCo review.

**This is a preview implementation, not a commissioned autonomous robot
controller.** Generated plans explicitly stop short of contact and carry
`preview_only: true`. The dashboard only shows them in simulation; `tools/arm_lift.py` also rejects
them. Physical dashboard buttons use the R1 firmware preset service.

## Try it without robot hardware

```sh
.venv/bin/python tools/dashboard.py
```

Use the signed-in Codex chat to request a motion, click **Generate preview**,
then **Show once in simulation**. No recording is saved. For the object demo,
choose **Motion preview → Preview context → Simulation object fixture** before
generating a request such as `touch the bottle`. The known table and object
geometry supply the target without a camera or vision API call. Orange marks
the surface and green the stand-off goal. The fixture supports a bottle or cube;
it is not AI object recognition. The planner can also be exercised directly
without any model using the local tests.

## OpenAI vision configuration

Install stereo processing dependencies if absent:

```sh
uv pip install --python .venv/bin/python -r core/requirements.txt
```

Set `OPENAI_API_KEY` and `REINS_VISION_MODEL` in the server environment, using a
vision-capable model with Responses API structured-output support that your
account can access. Keys are never entered in the web page or returned by its
API. No model is selected automatically and no API call occurs in demo mode.

```sh
export REINS_VISION_MODEL=YOUR_VISION_MODEL_ID
# Set OPENAI_API_KEY securely in your environment, not in a committed file.
.venv/bin/python tools/dashboard.py --observation /path/to/latest_observation.npz
```

When local detection cannot resolve the requested object, the planner sends the
prompt and the observation's RGB image to OpenAI's
Responses API (`store: false`). The returned normalized box identifies the
candidate image region; it never supplies trusted metric coordinates. Ambiguous
or missing targets block planning. The selected box is displayed with the
proposal for operator review. Current localization uses a robust depth cluster
inside the box, not a full object segmentation model, so clutter/transparent
objects can remain ambiguous. Do not interpret a returned box as verification.

## Calibrated observation interface

A sensor adapter must atomically replace the NPZ at `--observation`. No automatic
pairing of the existing independent MJPEG feeds is performed. NPZ uses plain
arrays/scalars, `allow_pickle=False`, with these fields:

| Field | Meaning |
|---|---|
| `rgb` | RGB uint8 array, H×W×3, rectified/aligned with depth |
| `depth_m` | H×W metric depth along camera Z; invalid pixels NaN |
| `K` | 3×3 intrinsics for that exact image resolution |
| `T_robot_camera` | Rigid 4×4 transform from rectified camera to `robot_base` |
| `captured_at` | Unix capture timestamp, not server receipt time |
| `pose_at` | Unix timestamp of measured robot joint state |
| `pose_json` | Scalar JSON string: MuJoCo joint names → measured radians |
| `calibration_id` | Nonempty identifier/version of the measured calibration |
| `uncertainty_m` | Calibrated uncertainty allowance, 0.005–0.05 m; default 0.02 |

The observation must be at most 3 seconds old when acquired by the planner. Pose
and image timestamps must differ by at most 50 ms; clocks must be synchronized.
Include all physical joints in the fixed-base model; dummy waist-pitch and
wrist-pitch/yaw joints are excluded. Supply the transform for that measured head
or camera pose. The fixed-base preview assumes the documented nominal pelvis
placement; it does not estimate whole-body balance or base drift.

Model calls can take longer than a camera frame interval. The proposal remains
a historical snapshot preview, never a live authorization. Captured observations
are not kept fresh by changing their timestamps. Physical execution will need a
fresh-scene/state check and tracking during motion.

## Stereo reconstruction

If the camera driver supplies calibrated metric depth, use it directly. Otherwise
`tools/stereo_depth.py` supports a synchronized horizontal stereo capture:

```sh
.venv/bin/python tools/stereo_depth.py capture.npz calibration.json observation.npz
```

`capture.npz` fields: `left_rgb`, `right_rgb`, `left_at`, `right_at`, `pose_at`,
`pose_json`, and `T_robot_left_camera` (unrectified left camera to robot).

Factory/measured calibration JSON fields:
`image_size: [width,height]`, `K_left`, `K_right`, `dist_left`, `dist_right`,
`R_right_left`, `t_right_left_m`, `calibration_id`, `uncertainty_m`.
The stereo transform maps left-camera coordinates into right-camera coordinates;
a conventional right camera has negative X translation in this transform.

The converter rectifies using OpenCV, computes StereoSGBM disparity, rejects
left/right-inconsistent matches, reconstructs metric depth, and adjusts the
camera-to-robot rotation for rectification. It preserves capture timestamps and
rejects pair skew above 15 ms. This threshold is an initial policy, not proof of
hardware synchronization. Rolling shutter, motion, calibration error and minimum
usable depth must be measured on the actual cameras. Native sensor acquisition
and factory calibration discovery are still hardware integration work.

## Motion checks and limitations

- Existing `core/ik.py` solves A5 arm position with measured held joints, joint
  limit margins and warm-started Cartesian waypoints. Unreachable or discontinuous
  solutions reject the request. This implementation does not search around
  obstacles; a blocked straight-line approach is rejected.
- Joint velocity is capped at 0.4 rad/s (generated plans target 0.3). Time scaling
  limits finite-difference sampled acceleration to 1.5 rad/s². Piecewise-linear
  playback is not a jerk-limited hardware controller.
- Upper-arm/forearm envelopes use approximate 5 cm radii plus 1.5 cm clearance.
  The full arm is checked against the opposite arm, floor, scene boxes and depth;
  a torso-adjacent mount segment is excluded only from its torso-envelope check.
  Native MuJoCo arm contacts and conservative torso/head envelopes are checked.
- Motion is sampled at joint increments no larger than 0.01 rad. These checks
  are conservative prototype filters, **not continuous collision certification**.
  Actual link/hand geometry and controller tracking must be validated on the
  correct hardware variant before any physical execution.
- Depth visibility checks project each swept arm sphere into the depth image.
  Any missing depth, out-of-view volume, occlusion, or obstacle blocks the plan.
  Unknown space is not treated as free. A single head camera can therefore reject
  many valid reaches that would need additional calibrated views or a map.
- Approaches stop 10 cm plus the depth uncertainty before the observed surface.
  Actual contact, contact forces and retreat are not implemented. No collision
  exception is granted to the bottle.
- Preview IDs are immutable per request; stale/dismissed IDs cannot be loaded.
  Loading a proposal does not execute it. Plans cannot pass through the legacy
  executor, which replaces their start pose and appends an unvalidated return.

The remaining physical integration is: verify the stereo driver/calibration,
measure the hand and collision envelopes, obtain calibrated observations with
measured base/camera poses, add contact feedback/compliance, and integrate an
executor that validates and runs the exact approved path with scene/state
freshness checks. Existing manual Execute is not evidence those checks exist.

## Tests

```sh
.venv/bin/python -m unittest core.test_reins_tools core.test_claude_chat core.test_codex_chat core.test_dashboard_chat core.test_generated_motion core.test_trajectory_revision core.test_r1_gestures core.test_object_detection core.test_prompt_planner core.test_ik tools.test_dashboard tools.test_dashboard_http
python3 -m pytest contract
```

Tests include synthetic stereo metric depth, timestamp/transform validation,
missing-depth rejection, known-scene prompt-to-IK, obstacle/unreachable target
rejection, provider ambiguity, and preview-only execution guards. They require
no API key, physical cameras, or robot connection.

## Auto context

**Auto context** is the default. Each skill declares its requirements in
`core/action_context.py`; every plan still passes motion validation.

- `wave`, `wave your left hand`, and `raise your right arm` use local gesture
  skills. They do not send images or make a model API call. If configured,
  calibrated depth supplies measured pose and workspace clearance without visual
  recognition. Otherwise the dashboard's current simulated pose supplies an
  explicitly simulation-only preview; physical clearance remains unknown.
- `point at the bottle`, `touch the bottle`, and `approach the cube` require an
  object position. Auto reuses grounding only for the same fresh observation,
  calibration, transform and pose; new or changed observations require fresh detection or visual grounding.
  `Refresh object vision` bypasses this grounding cache.
- `touch it` / `point at it` resolve the last object selector from the same
  source. A stale target's position is never reused; a new observation is
  grounded again. Without an earlier target the request is rejected.
- Pointing chooses a reachable hand pose whose forearm axis aims toward the
  target (within 10 degrees); the hand need not reach the object. The complete
  motion must still pass collision checks. Some nearby targets cannot satisfy
  this constrained A5 gesture and are rejected.
- The local router handles bounded object commands and built-in gestures. Use
  dashboard chat to author new gestures or compound single-arm sequences, then
  generate it with **Generate preview** on the chat reply. Directed waving (`wave at the
  person`) is not silently treated as a generic wave or supplied an invented
  person location. Object actions still require grounded context.

The dashboard displays the selected skill, its context requirements, whether
vision was needed or reused, and the validation coverage. All new skill plans
retain the existing preview-only execution lock. No robot motion is authorized
by choosing Auto context.


## Local object detection

Install the checksum-pinned models once (OmDet-Turbo needs `torch` and
`transformers` from `core/requirements.txt`; without them only NanoDet is installed):

```sh
uv pip install --python .venv/bin/python -r core/requirements.txt
.venv/bin/python tools/detect_objects.py --download
.venv/bin/python tools/dashboard.py            # prints "Object detection: OmDet-Turbo · mps" (or cuda/cpu)
```

**Scene understanding → Start detection** runs the local detector on a selected head,
wrist or glasses JPEG/MJPEG stream. The default is **OmDet-Turbo** (swin-tiny,
Apache-2.0), an open-vocabulary detector: it looks for COCO's 80 classes plus 24
tabletop objects COCO lacks (plate, drinking glass, jar, box, lid, tray, pan, tape,
screwdriver, cable and more; `HOUSEHOLD` in `object_detection.py`), and any other
name can be passed to `detect(labels=...)` or `tools/detect_objects.py --labels`.
It runs on CUDA, Apple MPS or CPU (chosen automatically). If PyTorch or its
weights are missing, **NanoDet** (80 COCO classes, OpenCV CPU) is used instead;
force either with `--detector omdet|nanodet`. No API key or depth is needed for 2D
boxes.

Measured 2026-09-27 on 500 random COCO val2017 images (none used in training),
pycocotools, same images for both:

| | NanoDet (fallback) | OmDet-Turbo (default) |
|---|---|---|
| COCO mAP / AP50 | 22.2 / 38.2 | **43.0 / 58.9** |
| small objects AP | 8.3 | **28.8** |
| cup / fork / knife / remote AP | 16 / 10 / 0.6 / 4 | **57 / 48 / 22 / 37** |
| precision / recall at confidence 0.4 | 71% / 34% | 67% / **50%** |
| 30 household classes outside COCO (LVIS labels) | 0 (cannot name them) | **26.5 mAP** |
| time per image | 60 ms (CPU) | 78 ms (RTX 4060, 104 names), 1.2 s (CPU) |

OpenCV Zoo's own reference code scores NanoDet at 21.5 on the same images, so the
wrapper is faithful; the model is the limit. For comparison YOLO11n scored 40.5
but Ultralytics models are AGPL-3.0, so they were not adopted. OmDet-Turbo is weak
on thin or tiny parts (hooks, magnets, lightbulbs, doorknobs, handles: AP < 10).
Mac speed on MPS has not been measured yet; on CPU it is too slow for live video,
so check `Object detection:` at dashboard start shows `mps`. Browser window sharing stays in the browser and is not a detection source.
The worker starts paused, samples only the latest frame at up to about 3 Hz,
serializes inference with the planner and never accumulates a frame queue.
Labels and confidence are baked into the exact inferred frame. Results expire
three seconds after receipt (or capture for calibrated observations); ordinary
MJPEG receipt time does not establish sensor capture time. No persistent object
tracking or identity across frames is claimed. The minimum confidence defaults
to 0.40 for both detectors, configurable with `--detection-confidence 0.5` at dashboard launch.

With `--observation PATH.npz`, the panel also offers **Calibrated RGB + depth**.
Only this source can display measured 3D surfaces in robot coordinates. Missing
or mixed depth remains unknown. Detecting a bottle in a head/glasses stream does
not automatically associate that box with a different camera's depth image.

For object prompts, the planner detects on the RGB image inside its own fresh,
synchronized observation. A single matching class supplies the box without an
OpenAI call. Multiple matches ask for a more specific description. Qualifiers
such as `red bottle` use the configured OpenAI model; missed or unsupported
classes also fall back to it. With no configured model, an unresolved target
blocks with setup instructions. Detection confidence is not a safety score, and
absence of a detection is never evidence of free space. Recognition can miss
small/occluded objects, confuse classes and mix background into a bounding box.
No detection supplies metric coordinates or relaxes the IK, depth-clearance,
joint-limit or collision checks. A snapshot that expires during recognition is
rejected before localization, including slow model requests. All generated plans
remain preview-only. Gestures never invoke the detector; independently enabled
camera monitoring may continue while a gesture is planned.

To test a local image without cameras:

```sh
.venv/bin/python tools/detect_objects.py --input photo.jpg --output detected.jpg
```

The CLI prints normalized `xyxy` boxes, labels and scores. See
[model provenance and license](models/README.md). No model weights or calibration
are downloaded at dashboard startup, and weights are excluded from Git.


## Conversational dashboard assistant

`core/dashboard_chat.py` owns bounded session history and read-only dashboard
context. Its transports are switchable at runtime from the chat panel
(`DashboardChat.set_backend`): `core/codex_chat.py` runs the signed-in Codex CLI (the
default), `core/claude_chat.py` the signed-in Claude CLI (`claude -p` with tools, MCP,
skills and user settings disabled), both ephemeral with structured JSON output and no
API key; `openai` uses the Responses API with `OPENAI_API_KEY` and `REINS_CHAT_MODEL`
(or `REINS_VISION_MODEL`). `--chat-backend claude|codex|openai` picks the first one.

The CLI bridge disables command execution and external tools, runs outside the
repository in a temporary directory, and never attaches to existing Codex threads.
Messages go through stdin; executable/flags are server-controlled argument lists.
Process groups are terminated and reaped on cancel, timeout, excessive output or
shutdown. Only validated assistant replies reach the UI, not stderr or reasoning.
Each request carries the bounded dashboard history rather than resuming a CLI
thread. CLI authentication remains managed by each CLI.

Replies may contain either a grounded object command handled by `route_intent`
or a newly authored `trajectory`. `core/generated_motion.py` defines the draft:
a name, one arm, up to 16 hand positions in `robot_base` metres, per-waypoint
pauses, and an optional return to the starting hand position. The assistant
receives model-derived shoulder/hand positions and the head envelope. New
gestures and compound arm sequences need no named skill or predefined plan.
Drafts and planner failure feedback remain available in follow-up conversation.

**Generate preview** on a chat reply compiles the exact server-held draft with IK
from the current simulation or measured pose, interpolates and times each segment,
then checks joint limits, velocity,
acceleration, swept collisions and available depth clearance. When an authored
path fails IK, timing or collision checks, the planner requests up to two revised
drafts from the selected chat backend (three attempts total). Each revision gets
the failed paths, waypoint/collision details and model geometry, then goes through
the same checks from the original starting pose. The preview shows progress and
can be cancelled while recalculating. CLI revision sessions are isolated from
normal chat; switching chat backends does not switch an in-flight revision.
When chat uses `plan_hand_path` directly, the tool returns the same structured
failure feedback and a retryable flag; the calling assistant is instructed to
revise and call it again, up to three attempts, without a nested model request.
Missing/stale observations, invalid starting poses and provider errors stop
planning instead of weakening validation. Explicit camera
mode requires a calibrated observation; Auto without one is simulation-only.
Malformed, nonfinite, oversized drafts and stale chat suggestions are rejected.
**Show once in simulation** passes the validated proposal to MuJoCo in memory.
The animation runs once, with an optional Stop preview button. No file is saved,
and the proposal cannot be replayed. The separate legacy export helper remains
available to code outside the dashboard.

Only one arm's hand position is authored; other joints stay fixed. There is no
finger articulation, independent wrist orientation, coordinated two-arm motion,
walking, grasping or contact. Near-face gestures use non-contact approximations
and may still fail reach/collision checks. Generated paths remain preview-only
and cannot enter the firmware gesture service. Chatting alone never invokes a
planner or commands hardware. See the [chat guide](../tools/dashboard/README.md#chat-with-the-dashboard-assistant)
for configuration, memory, retry, cancellation and data handling.
