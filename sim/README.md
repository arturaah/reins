# R1 trajectory preview

This simulation-only prototype implements Reins' review-before-action loop for Unitree's R1 MuJoCo model. It does not connect to the robot or publish DDS commands.

## Requirements

- macOS with Homebrew Python 3.14 and `python3 -m pip install mujoco scipy matplotlib`, or another Python with those packages
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

## Plan and preview contract

A plan contains `schema_version: 1`, `duration_s`, and time-ordered keyframes. Each keyframe has `time_s` and a map of **MuJoCo joint names** to target angles in radians. Keyframes must name the same actuated joints, start at zero, and end at `duration_s`. The example plan moves the left shoulder and elbow. All other joints are held at zero by a simple PD controller. Targets interpolate linearly between keyframes.

The export has timed `samples` with `joint_targets_rad`, `hands_xyz_m`, and `cube_xyz_m`. Coordinates are in the `mujoco_world` frame and meters. An AR client must calibrate and apply the world-to-glasses transform before drawing the points. The preview follows sites on both wrists, approximating the hand tips; this R1 model does not include a full hand model.

## Scope

The checked-in model is copied from Unitree's `unitree_mujoco` R1 files; its license is in `sim/models/r1/LICENSE`. `scene_fixed_base.xml` anchors the pelvis so arm tests do not require a balance controller. The original `scene.xml` and robot XML are retained for future free-standing work. The MuJoCo model has 29 actuators; the R1 EDU A5 hardware map described by the vendored SDK has 26 motors. Do not map joint indices directly between them.

This is a deterministic trajectory preview and simulation check, not a balance policy, grasp planner, collision validator, or hardware command path.
