# harness: the VLM is the policy

A frontier vision-language model controls one R1 arm zero-shot: each step it sees the head camera
and the wrist camera plus a short text state, and answers with ONE discrete hand action. Code turns
that into a bounded hand-tip setpoint, IK turns the setpoint into arm joints, the safety gate vets
everything, and only then does anything stream to `rt/arm_sdk`. Design and assumptions: [DESIGN.md](DESIGN.md).

Layers: algorithm (`actions`, `interpreter`, `kinematics`, `safety`, `executor`, `prompts`,
`perception`, `loop`, `recorder`, `sim/`) never imports the robot SDK and runs on the Mac;
`robot/` holds the DDS side; `vlm/` holds the API adapters. `tests/test_layering.py` fails if the
algorithm layer ever imports `unitree_sdk2py`.

## The model

`vlm.provider` picks who decides. Default `chat`: the Claude session driving this repo (no API
key). Each planner or controller call lands in `runs/chat_inbox/<NNN>_<plan|act>/` as
`prompt.txt`, `context.jpg`, `right.jpg` and `request.json`; the loop blocks until `answer.json`
is written there with the model's JSON answer, then continues. `claude-cli` runs `claude -p` under the
Mac's Claude login for unattended loops (about 20 to 25 s per step, images read by the CLI's Read
tool, JSON schema enforced). `anthropic` calls the API with `ANTHROPIC_API_KEY`; `scripted` is the
test stand-in. First chat-driven sim episode 2026-09-26: 3 steps, one chunk of two MV_DOWN, DONE,
success.

## Run

Everything uses the repo venv (`.venv/bin/python`, see CLAUDE.md step 6) plus `anthropic`,
`pytest`, `pyyaml`. All numbers live in [config.yaml](config.yaml); override any with `--set key=value`.

```
.venv/bin/python -m pytest harness                       # 76 tests, no hardware, about 6 s

# simulation: kinematic mock on the MuJoCo scene, rendered cameras
MUJOCO_GL=cgl .venv/bin/python -m harness sim "move your hand above the block"        # chat provider
MUJOCO_GL=cgl .venv/bin/python -m harness --set hand.type=virtual sim "pick up the block and place it on the plate"
.venv/bin/python -m harness sim "..." --vlm scripted     # no API calls

# real robot, nothing published: real joints from rt/lowstate, real cameras, real VLM calls, IK, gate
.venv/bin/python -m harness dry-run en6 "move your hand above the red block"

# real robot. Terminal 1: the streamer (the only publisher). Terminal 2: the loop.
.venv/bin/python -m harness.robot.arm_stream en6
.venv/bin/python -m harness live en6 "move your hand above the red block"

.venv/bin/python -m harness packet --sim | --iface en6   # one perception packet to runs/packet_*/
.venv/bin/python -m harness replay runs/<run> --step 7   # rebuild that step's prompt, re-query, print the decision
.venv/bin/python -m harness measure-table en6 --watch    # hand tip z from rt/lowstate; set workspace.table_z_m
```

Cameras come from the stream servers the cockpit already uses: `tools/headcam.py en6` (port 8081,
head camera = CONTEXT VIEW) and the Jetson's `camstream.py` forwarded to port 8080 (wrists).

## Demonstrations as context (`--demos`)

`--demos recordings/a.json recordings/b.json` puts the selected recordings in front of every
planner and controller call, as `DEMO_k` blocks ([demos.py](demos.py)): a text summary (source,
duration, which arm moved, the hand tip at the motion's key moments in cm in the robot frame with
the change between moments in the same forward/left/up words the actions use) and, when the
recording has one, its contact sheet image `recordings/<name>.sheet.jpg`. `tools/teach.py` and
`tools/record.py` make that sheet since 2026-09-27: they sample the head and wrist streams at 3 Hz
while recording (`tools/framelog.py`) and tile the frames at 6 key moments (equal joint-space
arc-length fractions, first and last always; one row per camera, one column per moment, time
labels) and note the moments in the JSON under `"sheet"`. Older recordings work text-only. The
prompt tells the model the demonstrations are references, not scripts.

## Accept / Reject before every move

With `--confirm` (live does it by default) the loop prints `PROPOSAL: <what would move>` before
every motion, including the start pose, and reads one line: Enter sends it, `n` rejects it,
`n <note>` rejects it and the model reads the note in its next call ("The operator rejected your
last proposal (MV_LEFT) with the note ..."; a rejected chunk is dropped; the history shows
`MV_LEFT(rejected)`), `x` is the e-stop. `--preview FILE` writes each proposal as a plan file in
the sim contract first (moving arm at 50 Hz, other arm and waist held), which the twin server plays
as ghost arms: `GET /preview?file=runs/ui_preview.json&hold=1` keeps it on top until
`/preview/stop`, even while the streamer holds the arms with weight 1.

## Desktop window: the AI pane

`tools/reins_ui.py` (started by `tools/start_all.sh`) has this loop under the cameras: task, dry
run | live, arm, step profile, floor z (`workspace.table_z_m`), a multi-select list of recordings
as context (✓ = has a contact sheet), Run / Stop, the current proposal with Accept / Reject and a
note field, and the session log. It runs exactly
`python -m harness --arm A --profile P --set workspace.table_z_m=Z live|dry-run IFACE TASK --vlm claude-cli --confirm --preview runs/ui_preview.json --demos ...`
as a subprocess, so the VLM is `claude -p` on the Mac's Claude login. Every `PROPOSAL` line
enables the buttons and loads the held preview in the twin pane; Accept stops the preview (the
yellow SENDING ghost then shows the real motion) and sends Enter; Reject sends `n <note>`; Stop
sends Ctrl-C (the session releases the arms). Live mode starts `harness.robot.arm_stream` itself
when port 8790 is closed (log `/tmp/harness_stream.log`) and asks once before the arms are
engaged; the ENGAGE prompt is answered by the window. Dry run reads real joints and cameras and
publishes nothing, so the whole Accept / Reject flow can be rehearsed on the standing robot.

## Before the first live run

1. Measure the table: run `tools/teach.py en6 measure --seconds 60` in one terminal (arms go soft),
   rest the hand tip on the table, and read `measure-table --watch` in another. Put the z in
   `workspace.table_z_m`. The live gate refuses to start without it.
2. Check the workspace box in `config.yaml` against the real table and the hand's reach
   (about 0.46 m from the shoulder at z 0.99).
3. Robot standing in FSM 811 (operator, remote), body cable on `en6`, head camera server up.
4. `dry-run` first: same prompts, same IK, same gate, prints every frame it would send.
5. `live`: the loop asks before ENGAGE, before the start-pose move and before every step
   (Enter sends, `n [note]` rejects, `x` is the e-stop). Ctrl-C in either terminal ramps the weight down.

## What the streamer guarantees on its own

`harness/robot/arm_stream.py` is the single publisher. It refuses to engage outside FSM 4/811,
ramps the blend weight over 1 s both ways, holds waist yaw and head at their measured values,
re-checks every frame's joint speed, and ramps down by itself on: client heartbeat lost for 0.5 s,
client disconnect (the loop process died), `rt/lowstate` stale for 0.5 s, tracking error over
0.6 rad for 0.3 s, Ctrl-C. When the weight is 0 the robot's own controller has the arms.

## Runs

Every episode writes `runs/<mode>_<timestamp>_<task>/`: `plan_*`, `steps.jsonl`, and per step the
two images, the full prompt, the raw response and `step.json` (action, setpoint before/after,
joint target, feedback, timings, tokens). `sim_*`, `dry-run_*` and `packet_*` are gitignored;
`live_*` runs are kept in the repo.
