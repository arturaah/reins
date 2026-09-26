# Prompt-to-approach preview

The dashboard now accepts natural-language requests, grounds a target in a
calibrated RGB/depth observation, uses the existing R1 A5 IK solver, and validates
an approach before offering it for MuJoCo review.

**This is a preview implementation, not a commissioned autonomous robot
controller.** Generated plans explicitly stop short of contact and carry
`preview_only: true`. Both the dashboard runner and `tools/arm_lift.py` reject
them. The older manual trajectory workflow remains separate.

## Try it without hardware or API credentials

```sh
.venv/bin/python tools/dashboard.py
```

In **Describe the next move**, choose **Simulation demo · known geometry**, enter
`touch the bottle`, and click **Generate approach**, then **Load into MuJoCo**.
The fixture contains a table and an object represented by its collision box.
Orange marks the estimated surface, green the stand-off goal. Use the existing
playback controls to review the motion. The demo supports `touch`, `approach`,
and `reach for` a `bottle` or `cube`, optionally `with the right/left hand`.
This small local grammar uses known geometry; it is not AI object recognition.
Unsupported requests are rejected, never silently mapped to a canned motion.

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

Real-camera mode sends the prompt and the observation's RGB image to OpenAI's
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
.venv/bin/python -m unittest core.test_prompt_planner core.test_ik tools.test_dashboard
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
  calibration, transform and pose; new or changed observations require vision.
  `Refresh object vision` bypasses this grounding cache.
- `touch it` / `point at it` resolve the last object selector from the same
  source. A stale target's position is never reused; a new observation is
  grounded again. Without an earlier target the request is rejected.
- Pointing chooses a reachable hand pose whose forearm axis aims toward the
  target (within 10 degrees); the hand need not reach the object. The complete
  motion must still pass collision checks. Some nearby targets cannot satisfy
  this constrained A5 gesture and are rejected.
- Unknown and compound commands are rejected. Directed waving (`wave at the
  person`) is not silently treated as a generic wave. The local intent router
  deliberately accepts a bounded vocabulary; it is not an unrestricted language
  interpreter. OpenAI performs object grounding only when required.

The dashboard displays the selected skill, its context requirements, whether
vision was needed or reused, and the validation coverage. All new skill plans
retain the existing preview-only execution lock. No robot motion is authorized
by choosing Auto context.
