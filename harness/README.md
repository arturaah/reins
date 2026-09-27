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
tool, JSON schema enforced). `codex-cli` runs `codex exec` under the existing Codex login with
images attached directly and structured JSON output. It defaults to `gpt-5.6-sol` with `low`
reasoning effort (the Sol model available to this CLI account; GPT-6 Sol was rejected by the service).
Override `vlm.codex_model` / `vlm.codex_effort` through `--set`. Calls are ephemeral, use a temporary
directory, ignore user CLI configuration, and disable shell tools. `anthropic` calls the API with `ANTHROPIC_API_KEY`; `scripted` is the
test stand-in. First chat-driven sim episode 2026-09-26: 3 steps, one chunk of two MV_DOWN, DONE,
success.

## Run

Everything uses the repo venv (`.venv/bin/python`, see CLAUDE.md step 6) plus `anthropic`,
`pytest`, `pyyaml`. All numbers live in [config.yaml](config.yaml); override any with `--set key=value`.

```
.venv/bin/python -m pytest harness                       # 105 tests, no hardware, about 16 s

# simulation: kinematic mock on the MuJoCo scene, rendered cameras
MUJOCO_GL=cgl .venv/bin/python -m harness sim "move your hand above the block"        # chat provider
MUJOCO_GL=cgl .venv/bin/python -m harness --set hand.type=virtual sim "pick up the block and place it on the plate"
.venv/bin/python -m harness sim "..." --vlm scripted     # no API calls

# live 3D simulation window: run from Terminal on macOS; TYPESAFE_API_KEY must be set
MUJOCO_GL=cgl .venv/bin/python tools/mjpython.py -m harness sim "hover 5 cm above the orange block" --vlm codex-cli --executor jev --viewer

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

`sim --viewer` opens a live MuJoCo window and automatically uses real-time motion. Drag to orbit,
scroll to zoom. Add `--confirm` to approve each proposed move in Terminal. The window stays open
after the episode so you can inspect the final pose. X or Esc requests a stop; closing the window
stops the simulation (an in-flight model call can finish before the loop exits). On macOS launch
with `.venv/bin/python tools/mjpython.py` from Terminal. This wraps MuJoCo's required `mjpython`
launcher and supplies the Python shared-library path for uv-managed environments. The desktop AI pane's
`live` mode controls hardware and is not simulation.

## The planner looks occasionally, Jev decides each step (`--executor jev`)

`harness/split.py`. The chosen `--vlm` (Codex CLI or Claude, for example) makes the stage plan, and per stage it takes a
**look**: from the camera images it reports where the hand tip must go to finish the stage, as an offset
from where it is now (cm forward / left / up), the done condition restated in terms of that offset,
whether the wrist camera sees the target, and hazards. Between looks the goal is dead-reckoned: it stays
fixed in the robot frame while the hand's own motion is known exactly from forward kinematics. Code
recomputes the remaining gap after every move and writes it as words ("the goal is 6 cm below the hand
tip (near)"), because Jev is text only and weak at arithmetic.

Every step, [TypeSafe Jev](https://docs.typesafe.ai/api) (`harness/decider/jev.py`, plain HTTP)
answers three typed questions in one call: `action` (a Choice over MV_* / ROTATE_CW / ROTATE_CCW / STILL / DONE, GRASP /
RELEASE with a hand, and LOOK, with probabilities), `stage_done` and `needs_look` (Nouls). With
`--mover geometric`, code picks the move that closes the largest gap and Jev only answers the two Nouls.

The planner is called again only when code or Jev asks for it:
- **Code** asks for a new look: when a stage starts, after `executor.look_every_steps` moves or
  `look_after_move_cm` of travel, after an unreachable, stalled or rejected move or an operator note,
  and when the last look was low confidence.
- **Jev** asks for a look: it chose LOOK, its confidence is under `min_action_confidence`, `needs_look`
  is high, or the call failed. The step is then asked again with the fresh look. If Jev is still unsure,
  the planner decides that one step from the images, using the normal controller prompt. With
  `claude_fallback: false`, the arm holds STILL instead.
- **A stage ends**: a DONE from Jev is only a claim. The planner confirms it from the images before the
  stage advances. That same look also reports the next stage's goal, so the next stage starts without
  another call. A successful GRASP / RELEASE can also finish its matching stage when the hand reports
  the expected closed / open state; an unavailable or rejected hand action cannot finish a stage.

Every move still goes through the executor, the safety gate and the operator's Accept / Reject, exactly
as before. Run records mark each step with `decided_by`. Jev's state, questions and answers go in the
step's `prompt.txt` / `response.txt`, and planner looks in `look_prompt.txt` / `look_response.txt`.
The inference plot shows looks (purple squares) and decider calls (green diamonds).
Gripper actions also honor operator review and e-stop. Invalid snapshots and malformed Jev answers
(including non-finite probabilities, unknown choices, or incomplete distributions) trigger the
same retry / fallback path. A failed camera refresh invalidates the old snapshot; a failed completion
check cannot advance a stage. Safety rejections and settle timeouts request a fresh look.

On the mock pick-and-place (`tests/test_split.py`, with scripted eyes and decider), an episode takes
one plan, 9 camera looks, 25 decider calls and 20 execution steps, with no planner fallback actions.
These counts demonstrate the control flow; they do not measure real Claude or Jev accuracy or latency.

```
export TYPESAFE_API_KEY=...                              # console.typesafe.ai/keys
MUJOCO_GL=cgl .venv/bin/python -m harness sim "hover above the orange block" --vlm codex-cli --executor jev --confirm
.venv/bin/python -m harness --set hand.type=virtual sim "pick up the block and place it on the plate" --vlm claude-cli --executor jev --confirm
.venv/bin/python -m harness dry-run en6 "touch the red block" --vlm claude-cli --executor jev
```
Add `--mover geometric` to have code choose translations and Jev judge completion / escalation.
For a fully offline simulation test, with no API key or hardware, run
`.venv/bin/python -m pytest harness/tests/test_split.py -q`.
The desktop AI pane defaults to **Planner → codex-cli**, **per-step → jev**. Select `planner`
for the VLM to decide every step, or `claude-cli` to use Claude as the planner.
`executor.claude_fallback` and `confirm_done_with_claude` are legacy config names that also apply to
Codex. Summaries expose `planner_provider`, `planner_looks`, and `planner_steps`, retaining the old
`claude_*` counts for compatibility.
The thresholds in `config.yaml` under `executor:` are initial guesses, not calibrated for hardware.
The config pins `jev-1.13.0` so a new release cannot shift them without notice. On 2026-09-27, real Jev
completed the mock pick-and-place with a scripted plan and scene snapshots, with planner fallback
disabled. The API rounds individual probabilities, so validation allows the corresponding rounding
error in their sum. Completion checks preserve the hand requirement: positional alignment alone
cannot complete a pending grasp or release.
An attempted combined Claude CLI + Jev simulation stopped before planning because Claude reported
its monthly spend limit. Thus real Claude scene estimation with Jev remains unverified. Validation
records are under `runs/jev_validation/`; automated tests use a loopback fake API and scripted models.
The subsequent real **Codex CLI (GPT-5.6 Sol, low) + Jev** hover simulation completed with one plan,
three camera looks, eight Jev calls and six executed moves, with no planner fallback actions.
Ground-truth inspection found a 2.6 cm horizontal error and 3.8 cm clearance above the block top
against the requested 5 cm, despite the planner reporting completion. This verifies the integration,
not precise visual positioning. See the run's `validation.json` for measured geometry and latency.
Between looks, the controller assumes the target stays fixed. Moving objects require a new camera
look; the text decider cannot detect them from joint state alone.

## Demonstrations as context (`--demos`)

`--demos recordings/a.json recordings/b.json` puts the selected recordings in front of every
planner and controller call, as `DEMO_k` blocks ([demos.py](demos.py)): a text summary (source,
duration, which arm moved, the hand tip at the motion's key moments in cm in the robot frame with
the change between moments in the same forward/left/up words the actions use) and, when the
recording has one, its contact sheet image `recordings/<name>.sheet.jpg`. `tools/teach.py` and
`tools/record.py` make that sheet since 2026-09-27: they sample the head and wrist streams at 3 Hz
while recording (`tools/framelog.py`) and tile the frames at the motion's key moments, one row per
camera, one column per moment, time labels, and note the moments in the JSON under `"sheet"`.
Sampling is meant to keep the image context small: the moments are Douglas-Peucker samples of the
joint-space path (`demos.tolerance_rad`, at most `demos.max_moments`), so a straight reach gets 3
frames and a motion with turns up to 8; the context row is cropped to the region of the image that
changed during the recording (where the arm and the objects moved), with one full-view tile that
shows the crop box; the sheet is never wider than `demos.max_width_px` (1568, the model's long-edge
limit), and `"sheet.tokens_est"` records what it costs. All selected sheets are stacked into ONE
image labelled DEMOS, one Read for `claude -p` instead of one per demo. Older recordings work
text-only. The prompt tells the model the demonstrations are references, not scripts.

## Accept / Reject before every move

With `--confirm` (live does it by default) the loop prints `PROPOSAL: <what would move>` before
every motion, including the start pose, and reads one line: Enter sends it, `y <note>` sends it and
the model reads the note in its next call, `n` rejects it, `n <note>` rejects it with the note
("The operator rejected your last proposal (MV_LEFT) with the note ..."; a rejected chunk is
dropped; the history shows `MV_LEFT(rejected)`), `x` is the e-stop. Every answer is appended to
`feedback.path` (`runs/operator_feedback.jsonl`: task, stage, action, accepted, note, hand height)
and later sessions get it back as an OPERATOR FEEDBACK block before the prompt: entries for the
same task first, newest first, rejections and noted accepts listed (up to `feedback.max_in_prompt`),
bare accepts only counted. `--preview FILE` writes each proposal as a plan file in
the sim contract first (moving arm at 50 Hz, other arm and waist held), which the twin server plays
as ghost arms: `GET /preview?file=runs/ui_preview.json&hold=1` keeps it on top until
`/preview/stop`, even while the streamer holds the arms with weight 1.

## Every session becomes a recording

When an episode ends (any mode), its accepted, executed moves are exported to
`recordings/ai_<mode>_<task>_<time>.json` in the sim contract (`recorder.export_dir`): commanded
joint targets of both arms and the waist, each move taking the duration the gate gave it, then
0.5 s, with the thinking time removed; a contact sheet is built from the images the model saw
before each move, so the recording shows up with ✓ in the window's context list and can be
selected as a demonstration for the next session, dry-run it and replayed with `tools/arm_lift.py`
like a taught skill. A dry-run export is the pretend trajectory (what would have been sent).

## Inference time vs context

Every VLM call appends a line to `stats.path` (`runs/inference_log.jsonl`): latency, the tokens the
provider reported (with `claude -p` almost all input is a cache read, so `input_tokens` alone is
tiny), and an estimated context that is comparable across providers: prompt characters / 4 plus
image pixels / 750. After each call `stats.plot` (`runs/inference_stats.png`) is redrawn: x =
estimated context tokens, y = seconds; this session in colour (plan = triangle, act = dot), earlier
sessions grey. The window shows it under the twin and reloads it whenever the file changes.

## Desktop window: the AI pane

`tools/reins_ui.py` (started by `tools/start_all.sh`) has this loop under the cameras: task, dry
run | live, arm, step profile, floor z (`workspace.table_z_m`), a click-to-toggle list of recordings
as context (✓ = has a contact sheet), Run / Stop, the current proposal with Accept / Reject and a
feedback field that goes to the model with either answer, the session log, and under the twin the
inference-time-vs-context plot. It runs exactly
`python -m harness --arm A --profile P --set workspace.table_z_m=Z live|dry-run IFACE TASK --vlm claude-cli --confirm --preview runs/ui_preview.json --demos ...`
as a subprocess, so the VLM is `claude -p` on the Mac's Claude login. Every `PROPOSAL` line
enables the buttons and loads the held preview in the twin pane; Accept stops the preview (the
yellow SENDING ghost then shows the real motion) and sends Enter or `y <note>`; Reject sends `n <note>`; Stop
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
re-checks every frame's joint speed, and ramps down by itself on: client heartbeat lost for 0.5 s
(paused while it is serving that client's command, since the client cannot heartbeat then),
client disconnect (the loop process died), `rt/lowstate` stale for 0.5 s, tracking error over
0.6 rad for 0.3 s, Ctrl-C. When the weight is 0 the robot's own controller has the arms.

## Runs

Every episode writes `runs/<mode>_<timestamp>_<task>/`: `plan_*`, `steps.jsonl`, and per step the
two images, the full prompt, the raw response and `step.json` (action, setpoint before/after,
joint target, feedback, timings, tokens). `sim_*`, `dry-run_*` and `packet_*` are gitignored;
`live_*` runs are kept in the repo.
