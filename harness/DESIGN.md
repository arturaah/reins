# DESIGN: assumptions and deviations

Semantic end-effector-delta harness after Show-Harness (arXiv 2609.10522) and RoboDawn
(arXiv 2609.22966), built on this repo's `rt/arm_sdk` streaming path. Status 2026-09-26: sim and
tests pass; `dry-run en6` verified against the robot in FSM 0 (real joints, real head camera, nothing
published); the streamer and `live` have not run on the robot yet.

## Robot facts that shaped the design

- **Unitree R1 EDU, A5 arms: 5 joints** (shoulder pitch/roll/yaw, elbow, wrist roll). The hand's
  pointing direction is fixed by its position; the only free orientation is roll about the forearm.
  So the setpoint is **position (3) + wrist roll (1)**, not a 6-DoF pose. `ROTATE_CW/CCW` roll the
  wrist; `ROTATE_* x|y|z` and `POINT` presets parse but are answered "not available" without moving.
- **No hand or gripper on this unit.** `GRASP`/`RELEASE` are pauses (`hand.type: none`); the
  planner is told there is no hand and plans reach/hover/touch/push stages. `hand.type: virtual`
  exists for the sim (kinematic grasp of the scene's cube within 3 cm) so the grasp/recovery logic
  is tested. If a hand is fitted, add its presets to the backend's `hand()` and set the type.
- **End effector** = wrist roll link + 0.13 m along its x, the model's `*_hand_preview` site
  (`robot.ee_offset_m`). What is physically at the wrist flange is unverified: check before a
  touch task and change the offset if needed.
- **Reach** is about 0.46 m from the shoulder (shoulder at z 0.99 in the robot frame). The
  controller's standing pose hangs the arm nearly straight down (hand at z 0.54, 98 % extended):
  from there the hand cannot move down at all and barely forward. Episodes therefore start from
  `robot.start_pose_rad` (forearm horizontal forward, hand at (0.30, -0.15, 0.78), 75 % reach), a
  gated joint-space move. Executed only in sim so far.
- **The head camera is the context camera and it moves**: head pitch/yaw are on the arm topic, so
  the streamer holds them at their measured values (`robot.hold_head`). Torso sway under the
  balance controller is not compensated. No intrinsics or extrinsics exist; the hand-tip marker
  is off for the real cameras until `perception.context_camera` is calibrated (it works in sim).
- **Frames**: robot frame = floor under the pelvis, x forward, y left, z up (contract/README.md).
  The fixed-base model pins the pelvis at 0.74 m. `workspace.table_z_m` is measured in this frame
  with the robot's own arm (`measure-table`), so standing-height error cancels.
- **View conventions** (`frames.view_forward/left`): the head camera looks forward, so image-left =
  +y, image-up (far) = +x. Up/down is gravity. These are repeated in every controller prompt.
- **Mac as host**: the Python streamer reaches 42 to 50 Hz (pure-Python CRC). Commands are
  cosine-eased joint frames; the onboard PD (kp 50/40/30, kd 2) smooths between them. Gravity
  droop with these gains is 1 to 3 cm at the hand, which shows up in the "achieved" feedback.
- **The arms must be held for the whole episode**: releasing the blend weight between steps hands
  the arms back to the controller, which returns them to its own pose. Hence the separate
  streamer process that holds the last target at weight 1 while the VLM thinks, with a heartbeat.

## Numbers (config.yaml)

| item | value | source |
|---|---|---|
| unit step, precision profile | 1 cm, 5 deg | spec; Show-Harness 1 cm for stacking |
| coarse / fine step | 4 cm / 2 cm, by WRIST: YES/NO | Show-Harness |
| per-command hard cap (unit) | 5 cm, 20 deg | spec |
| parameterized clip | 20 cm, 90 deg | RoboDawn |
| joint speed cap | 0.8 rad/s (arm_lift uses 1.5) | conservative for VLM steps |
| joint limit margin | 0.05 rad | tools/arm_lift.py |
| tracking abort | 0.6 rad for 0.3 s | tools/arm_lift.py |
| settle | joint speed < 0.05 rad/s or 3 s | spec |
| heartbeat / watchdog | 100 ms / 500 ms | contract/README.md safety rules |
| workspace box | x 0.15..0.60, y -0.60..0.60, z 0.55..1.25 | generous; IK reach is the real limit |
| table floor | table_z + 2 cm | spec |
| episode cap | 60 steps, 5 history, chunks of 3 | spec / Show-Harness |
| sim table top | 0.665 m (scene_fixed_base.xml) | model |

## The one gate

`harness/safety.py` `SafetyGate.vet` is the only producer of joint targets: e-stop, box and
table floor (applied before the per-command cap so a clamp can never lengthen a move), the caps,
IK reachability, joint limits with margin, speed-derived duration, self-collision veto (more
model contacts than at the start pose). `ArmExecutor` re-checks the interpolated trajectory,
asks the operator when a confirm callback is set, then hands frames to a backend. Joint-space
moves (start pose, home step) go through `ArmExecutor.go_to_joints` with the same checks minus
the Cartesian clamp. The streamer process re-checks joint speed a third time, independently.

## Deviations from the papers

- Show-Harness assumes a gripper and a wrist camera between the fingers. Here the wrist-view
  rules refer to the bare hand tip and the wrist camera's mounting is unverified: the
  `WRIST_RULES_DEFAULT` block in `prompts.py` must be checked against a real wrist image.
- `DONE` ends the active stage (the loop advances); the last stage's DONE ends the episode.
- Recovery: empty grasp rolls back to the latest GRASP stage; three consecutive IK failures move
  30 % toward the start pose and tell the model; oscillation injects the anti-hunting note but
  the move is still executed (Show-Harness bans it in the prompt only).
- RoboDawn's grid overlay is drawn on the context image (A..H x 1..6, labels only, no metric
  table grid because there is no calibration); their in-context demonstrations are not used yet.
- Structured JSON output is requested from the API; the parser still validates every field and
  re-prompts once with the error, then counts a failed step; three failed steps end the episode.
- No self-collision model with the table exists in the MJCF (table geom has contype 0); the
  contact veto only covers the arm against the body.

## Not done yet

Real cameras seen by this package (the dry run will show it), table measurement, the start-pose
move on the robot, calibrated hand marker, dual-arm mode (parser and prompts support it, the loop
drives one arm), OpenAI adapter untested, `sim` renders need `MUJOCO_GL=cgl` on macOS.
