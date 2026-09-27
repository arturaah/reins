# Loco: high-level walking

Walk the R1 by velocity commands, the way Unitree's `LocoClient` does, with the
same code running in MuJoCo now and on the robot later. The onboard policy owns
balance; Reins only decides where to go.

```
VLM tool call ─► skills.plan_* ─► contract walk step ─► preview + approval
                                                            │ approved
                         skills.execute_walk_step ◄─────────┘
                                   │
                          PathFollower + drive()   (10 Hz, 0.5 s command lease)
                                   │
                         Loco interface (base.py)
                        ┌──────────┴──────────┐
                   SimLoco (MuJoCo)     UnitreeLoco (unitree_sdk2py, dry run by default)
```

## Run it

Needs `mujoco numpy pillow` (and `pytest`, `jsonschema` for tests).

```bash
mjpython loco/run_sim.py                           # around the table: type y/n in the terminal; Enter aborts while walking
mjpython loco/run_sim.py --to 1.8 1.8 1.5708       # straight line: the table stops it
python3 loco/run_sim.py --headless --gif walk.gif  # no window, auto-approve
python3 -m pytest loco/tests
```

The scene is `sim/models/r1/scene_room.xml`: the walk scene plus a table, a
counter and a plant. Any geom named `obstacle*` is solid to the sim.

## Pieces

- **`base.py`: the `Loco` interface.** A copy of R1 `LocoClient` semantics:
  `set_velocity(vx, vy, wz, duration)` in the body frame, and the robot stops by
  itself when `duration` runs out. FSM ids: 1 damp, 4 stance, 811 walk. Calls
  return 0 on success. `Limits` clamps commands; the defaults (0.6 m/s forward,
  0.3 back and sideways, 0.8 rad/s) are **unverified on hardware**.
- **`follower.py`: `PathFollower` and `drive()`.** Pure pursuit along the path,
  turning in place when the path is behind the robot, then aligning to the goal
  heading. `drive()` re-sends a 0.5 s command every 0.1 s, so if the Mac hangs
  the robot stops within half a second. It always ends with a stop.
- **`skills.py`: what the VLM uses.** `TOOLS` are tool definitions
  (`walk_to`, `walk`, `turn`, `stop`) to give the VLM harness.
  `plan_tool_call()` turns a call into a contract `walk` step
  (`contract/README.md`) for review. `execute_walk_step()` runs an approved one.
- **`sim.py`: `SimLoco`.** Kinematic: the base follows the commanded velocity
  through an acceleration limit, and the legs play a cosmetic gait. It walks
  only in FSM 811, commands expire like on the robot, and a step into an
  obstacle is refused with `ERR_BLOCKED` (7401). `advance(dt)` steps sim time,
  so tests run faster than real time.
- **`unitree.py`: `UnitreeLoco`.** The real robot over `unitree_sdk2py`'s R1
  `LocoClient` (service `sport`). It calls `SetFsmId` directly because the SDK's
  `Damp()`/`Start()` helpers drop the return code.

## Going to the real robot

```python
from reins_loco.unitree import UnitreeLoco
robot = UnitreeLoco(interface="en6")             # dry run: logs, sends nothing
robot = UnitreeLoco(interface="en6", live=True)  # only after Artur approves (CLAUDE.md)
execute_walk_step(step, robot)                   # same call as in sim
```

Known gaps before that's safe:

1. **No position feedback.** The R1 SDK exposes no odometry, so `pose()` is dead
   reckoning from commanded velocities and will drift. It's OK for short moves
   under supervision. Longer ones need localization, or a check on the robot for
   an odometry topic (sending anything to the robot needs Artur's approval).
2. **Limits and acceleration** in `base.py`/`sim.py` are guesses. Measure them
   on the robot, hung in its frame first as Unitree's example says, and update
   both files.
3. **No obstacle sensing on the robot.** The sim knows where the furniture is;
   the real robot doesn't. Blocked-path detection is sim-only until perception
   exists.
4. **No path planning.** Paths are the straight lines between the VLM's via
   points. The sim catches a bad path, but nothing finds a good one yet. A grid
   planner over the sim's obstacles is the natural next step.
5. **Walking vs the control decision.** `CLAUDE.md` currently uses the loco
   client only as a safety wrapper; walking steps are provisional in the
   contract until the team agrees.
