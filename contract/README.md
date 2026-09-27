# Reins contract v0.1

The messages that connect the parts of Reins: the VLM harness, IK and the core,
the review surfaces (MuJoCo preview, Spectacles) and the DDS streamer. It maps
onto the architecture in `CLAUDE.md` ("Chosen control method"):

```
operator ──command──► core ──command──► planner (VLM harness)
                        ▲                  │
                        └──── goals ───────┘   end-effector goals
                      core: IK → q(t) → preview
                        │
                        ├──plan_proposed──► review surfaces (MuJoCo, Spectacles)
                        │◄─────decision──── operator
                        │
                        └──execute (approved q(t))──► streamer ──rt/arm_sdk 250 Hz──► R1
```

**Out of scope:** the robot-side DDS topics (`rt/arm_sdk`, `rt/lowstate`). Those
are Unitree's contract and stay as they are. This contract ends where the
streamer receives an approved plan.

The machine-readable source of truth is [`reins.schema.json`](reins.schema.json).
[`examples/`](examples/) holds full example sessions that validate against it.
Where this page and the schema disagree, the schema wins; fix the page.

## Transport

- WebSocket, text frames, one JSON object per frame. The core is the server,
  default `ws://<mac>:8765/reins`.
- Every client opens with `hello`. The core answers `welcome` with the current
  state, so a client that reconnects mid-plan can catch up.
- Camera images for the VLM are **not** carried here. The planner gets them its own way.

## Conventions

| | |
|---|---|
| Units | metres, radians, seconds |
| Timestamps | `t`: sender's Unix time in seconds (float). `time_s`/`times_s`: seconds from the start of a step |
| Quaternions | `[w, x, y, z]` (MuJoCo order) |
| Joint names | MuJoCo/URDF joint names, e.g. `left_shoulder_pitch_joint`. Never motor indices |
| Frames | `robot`: origin on the floor under the pelvis, x forward, y left, z up. Moves with the robot. `map`: fixed world frame. In sim it is `mujoco_world` |

Geometry always says its frame. Review surfaces that track the robot (Spectacles
with a marker on the robot) can draw `robot`-frame geometry directly, with no
room calibration.

**Joint names, not indices.** The MuJoCo model has 29 actuators; the R1 EDU A5
hardware has 26 motors. The streamer owns the name→`rt/arm_sdk` index map and
**rejects** any plan that names a joint it cannot map. No component besides the
streamer uses indices.

## Envelope

Every message has:

```json
{"type": "plan_proposed", "id": "m-0192", "t": 1790000000.12, ...}
```

`id` is unique per sender. Replies that refer to a message put its id in `ref`.

## Roles

Sent in `hello`. The core enforces who may send what.

| role | who | may send |
|---|---|---|
| `operator` | Spectacles, MuJoCo review window, CLI | `command`, `decision`, `abort` |
| `viewer` | live twin, dashboards | `abort` |
| `planner` | VLM harness | `goals`, `abort` |
| `executor` | DDS streamer (or the sim standing in for it) | `progress`, `robot_state`, `done`, `abort` |

Anyone may send `abort`. Only `operator` may send `decision`.

## State machine (core)

```
          command            goals ok           approve (matching revision)
  idle ───────────► planning ────────► proposed ─────────────────────► executing
   ▲                  │  ▲               │  │                              │
   │     plan_rejected│  └───decline + ──┘  │ decline, no feedback         │ done
   │    (after retries)     feedback        ▼                              ▼
   └──────────────────────────────────── idle ◄────────────────────────────┘

  abort from any state → halting → idle
```

- The core broadcasts `state` on every transition.
- **Revisions.** Each plan has `plan_id` and an integer `revision`. Anything that
  changes what the robot will do creates a new revision. A `decision` must name
  the exact revision it approves. A decision for a stale revision is refused
  with `error` `stale_revision`. No one can approve a plan they haven't seen.
- **Decline with feedback** ("use the other hand") goes back to `planning`. The
  core forwards the feedback to the planner with the original command.
  Declining without feedback returns to `idle`.

## Safety rules

These are requirements, not suggestions.

1. The core sends `heartbeat` at least every 100 ms. The executor **halts** if
   it hears nothing from the core for 500 ms.
2. On `halt`, or on losing the core, the executor ramps the `rt/arm_sdk` blend
   weight to 0 over at most 1 s. The robot's own controller keeps balance. The
   core may then have the loco client `Damp` as the session safety wrapper.
3. The executor only runs a plan received in `execute`, and only if its
   `plan_id`/`revision` equals the latest `state` it saw with `state: executing`.
4. The core validates every trajectory before proposing it: known joints, finite
   values, within URDF limits, velocity within `limits.max_joint_vel_rad_s`. The
   streamer validates again before streaming. Two independent checks.
5. The streamer never extrapolates past the last sample. It interpolates linearly
   to 250 Hz and holds the final pose until the blend ramps out.

## Messages

### Session

| type | from → to | fields |
|---|---|---|
| `hello` | client → core | `role`, `name`, `protocol` (`"reins/0.1"`) |
| `welcome` | core → client | `protocol`, `session_id`, `state` (a `state` payload) |
| `heartbeat` | core → all | none |
| `error` | core → client | `code`, `message`, optional `ref` |

Error codes: `bad_message`, `not_allowed` (wrong role), `stale_revision`,
`invalid_plan`, `busy` (a command arrived while a plan is active).

### Planning

| type | from → to | fields |
|---|---|---|
| `command` | operator → core, core → planner | `command_id`, `text`, optional `feedback` (with the rejected plan id) |
| `goals` | planner → core | `command_id`, `summary`, `steps`: end-effector goals per step (below) |
| `plan_rejected` | core → planner | `command_id`, `reasons` (e.g. IK unreachable), so the planner can retry |
| `plan_proposed` | core → all | `plan` |

The planner speaks in **end-effector goals**, and the core turns them into joint
trajectories with IK. A goal step:

```json
{"step_id": "s1", "kind": "reach", "description": "Reach over the cup",
 "goals": [{"effector": "left_hand", "time_s": 2.0,
            "position_m": [0.35, 0.16, 0.78], "quat_wxyz": [1, 0, 0, 0], "frame": "robot"}]}
```

Effectors: `left_hand`, `right_hand`, `head`. `quat_wxyz` is optional; without it
IK solves for position only. A planner may also ask for a `preset` step, which
runs a named arm action.

### Review

| type | from → to | fields |
|---|---|---|
| `decision` | operator → core | `plan_id`, `revision`, `decision` (`approve`/`decline`), optional `feedback` |
| `state` | core → all | `state`, `plan_id`, `revision`, optional `step_id` |

### Execution

| type | from → to | fields |
|---|---|---|
| `execute` | core → executor | `plan` (full, self-contained) |
| `halt` | core → executor | `reason` |
| `abort` | anyone → core | `reason` |
| `progress` | executor → core → all | `plan_id`, `revision`, `step_id`, `time_s` |
| `robot_state` | executor → core → all | `joint_positions_rad` (name → value), optional `effectors` (positions), `blend_weight`. At most 30 Hz. For clients that aren't on DDS, such as the Spectacles |
| `done` | executor → core → all | `plan_id`, `revision`, `outcome` (`succeeded`/`halted`/`failed`), optional `detail` |

## The plan

What reviewers see and what the streamer runs. Both get the same object.

```json
{
  "plan_id": "p-7", "revision": 2, "command_id": "c-3",
  "summary": "Pick up the cup with the left hand",
  "source": {"planner": "vlm-harness", "model": "claude-opus-5-5"},
  "limits": {"max_joint_vel_rad_s": 1.5},
  "steps": [{
    "step_id": "s1", "kind": "arm", "description": "Reach over the cup",
    "trajectory": {
      "joint_names": ["left_shoulder_pitch_joint", "left_elbow_joint"],
      "times_s": [0.0, 1.0, 2.0],
      "positions_rad": [[0.0, 0.0], [0.28, 0.23], [0.55, 0.45]]
    },
    "preview": {
      "effector_paths": {"left_hand": {"frame": "robot",
        "points": [[0.16, 0.17, 0.78], [0.25, 0.17, 0.80], [0.35, 0.16, 0.78]],
        "times_s": [0.0, 1.0, 2.0]}}
    }
  }]
}
```

- `trajectory` is q(t) as samples. `times_s` starts at 0 and strictly increases,
  and each `positions_rad` row matches `joint_names`. Samples may be sparse
  keyframes or dense; the streamer interpolates linearly to 250 Hz. Joints not
  named are held where they are.
- `preview.effector_paths` are the predicted hand paths from the MuJoCo replay:
  the lines people see in the preview and the glasses.
- Step kinds: `arm` (q(t) on `rt/arm_sdk`), `preset` (a named arm action),
  and `walk` and `servo` (provisional, below).

### `walk` steps (provisional)

The control decision in `CLAUDE.md` uses the loco client only as a safety
wrapper, so walking isn't in the approved architecture yet. The locomotion work
is in progress, so this shape is reserved for it and may change:

```json
{"step_id": "s0", "kind": "walk", "description": "Walk to the counter",
 "goal": {"frame": "map", "x": 1.8, "y": 1.8, "yaw": 1.5708},
 "corridor_half_width_m": 0.4,
 "path": {"frame": "map", "points": [[0, 0], [1.2, 0], [1.8, 0.8], [1.8, 1.8]]}}
```

With `path_update` (core → all: `plan_id`, `revision`, `step_id`, `path`,
`within_corridor`), the path can be replanned live around obstacles, the
"self-driving line". Replans that stay within `corridor_half_width_m` of the
approved path and keep the same goal need no new approval. Anything else halts
the robot and proposes a new revision.

### `servo` steps (provisional)

Hand one arm to a closed-loop policy, such as the VLM arm controller in
`harness/`, for a task. The policy picks each move from the cameras as it
goes, so there is no path to preview. What the reviewer approves is the task
and the bounds, and the policy's safety gate enforces them:

```json
{"step_id": "s1", "kind": "servo", "description": "Pick up the red cube",
 "effector": "right_hand", "task": "pick up the red cube",
 "bounds": {"frame": "robot", "box_min_m": [0.15, -0.6, 0.74], "box_max_m": [0.6, 0.6, 1.25],
            "max_step_m": 0.05, "max_steps": 60, "max_joint_vel_rad_s": 0.8}}
```

The hand stays inside the box. Each move is at most `max_step_m` and joints
stay under `max_joint_vel_rad_s`. The episode ends after `max_steps` moves at
the latest. A reviewer who wants to see every move can still be asked before
each one; that is the policy's own confirmation, outside this contract.

## Changing the contract

Bump `protocol` for breaking changes. Update the schema, this page and the
examples in the same commit, and run `python3 -m pytest contract`.
