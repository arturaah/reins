# R1 trajectory preview

This simulation-only prototype implements Reins' review-before-action loop for Unitree's R1 MuJoCo model. It does not connect to the robot or publish DDS commands.

## Requirements

- macOS with Homebrew Python 3.14 and `python3 -m pip install mujoco scipy matplotlib pillow`, or another Python with those packages (`pytest` for the tests)
- Run GUI commands from a normal Terminal session with `mjpython`. A Codex-launched viewer process crashed during macOS AppKit registration on this Mac; headless execution works.

## Run

From the repository root:

```bash
python3 sim/preview.py --headless
mjpython sim/preview.py
```

The default plan is `sim/plans/left_reach.json`. The script displays the predicted paths, waits three seconds, then moves the R1 through the plan in MuJoCo. It writes `sim/preview.json` and `sim/preview.png`. Add `--preview-only` to show paths without moving the robot.

For the cube pickup demonstration, generate the inverse-kinematics plan and preview it before running:

```bash
python3 sim/plan_pick.py
mjpython sim/preview.py --plan sim/plans/pick_cube.json
```

The preview writes both JSON and a PNG. The MuJoCo viewer shows only the left and right hand trajectories, as thick translucent cyan and orange lines. Both arms then move through their planned paths. The cube sits on a visual table. At the grasp time, the simulator checks that the left wrist-tip site is within 3 cm, then moves the cube with the left hand at a fixed offset. The model has no fingers, contact grasp, or grasp force, so this is a **kinematic grasp proxy** rather than a verified physical pickup.

To run the plan without a window:

```bash
python3 sim/preview.py --headless
```

Use `--plan path/to/plan.json` for another plan and `--output path/to/preview.json` to choose the export path.

## Walking path preview

Arm plans above keep the pelvis fixed. For where the robot will *walk*, give `preview.py` a waypoint plan, or run `walk_preview.py` directly:

```bash
mjpython sim/preview.py --plan sim/plans/walk_around_table.json
mjpython sim/walk_preview.py --plan sim/plans/walk_straight.json
python3 sim/walk_preview.py --headless --gif sim/walk.gif
```

![walk preview](docs/walk_preview.png)

The orange line is the planned path on the floor, arrows show the heading at each waypoint, and translucent blue ghosts mark the robot at each waypoint and the goal. The robot then walks the path on a loop, with the travelled part dimmed. The floor grid in `scene_walk.xml` is 1 m squares.

A walk plan has `schema_version: 1` and at least two `waypoints`, each with `x` and `y` in metres in the `mujoco_world` frame, an optional `yaw` in radians (default: face the direction of travel) and an optional `label`. The export replaces hand samples with a floor-projected `base_xyz_m` and `yaw_rad` every 0.04 s at 0.5 m/s, plus the resolved `waypoints`, so an AR client can draw the path on the floor. The PNG is a MuJoCo render with the robot at the start.

This is **kinematic only**. It uses the free-base model, the gait is a cosmetic leg swing rather than a locomotion controller, and nothing is stepped through physics. It shows where the robot intends to go, not whether it can balance its way there. Turning in place (same position, new yaw) is not animated.

Tests: `python3 -m pytest sim/tests`.

## Plan and preview contract

A plan contains `schema_version: 1`, `duration_s`, and time-ordered keyframes. Each keyframe has `time_s` and a map of **MuJoCo joint names** to target angles in radians. Keyframes must name the same actuated joints, start at zero, and end at `duration_s`. The example plan moves the left shoulder and elbow. All other joints are held at zero by a simple PD controller. Targets interpolate linearly between keyframes.

The export has timed `samples` with `joint_targets_rad`, `hands_xyz_m`, and `cube_xyz_m`. Coordinates are in the `mujoco_world` frame and meters. An AR client must calibrate and apply the world-to-glasses transform before drawing the points. The preview follows sites on both wrists, approximating the hand tips; this R1 model does not include a full hand model.

## Scope

The checked-in model is copied from Unitree's `unitree_mujoco` R1 files; its license is in `sim/models/r1/LICENSE`. `scene_fixed_base.xml` anchors the pelvis so arm tests do not require a balance controller. The original `scene.xml` is retained for future free-standing work; `scene_walk.xml` wraps the same free-base robot for the walking preview, with collision geoms hidden and a metric floor grid. The MuJoCo model has 29 actuators; the R1 EDU A5 hardware map described by the vendored SDK has 26 motors. Do not map joint indices directly between them.

This is a deterministic trajectory preview and simulation check, not a balance policy, grasp planner, collision validator, or hardware command path.
