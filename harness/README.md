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
.venv/bin/python -m pytest harness                       # 97 tests, no hardware, about 14 s

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

## Revo2 hands (`hand.type: revo2`)

BrainCo Revo2 five-finger hands on the R1's wrists, driven open/close: GRASP sends `hand.revo2.close`,
RELEASE sends `hand.revo2.open` ([config.yaml](config.yaml); motors thumb, thumb_aux, index, middle,
ring, pinky, 0 = open .. 1 = closed). After a close the fingers are read back: index..pinky within
`empty_reach` of the close pose means nothing stopped them, reported as `EMPTY grasp`, which the loop
answers as before (open, note, roll back to the GRASP stage). With `--confirm` every hand command is a
PROPOSAL like a motion.

What the model is told ([prompts.py](prompts.py)): the planner gets the hand (five fingers, open/close
only, a power grasp for objects about 3 to 9 cm, flat things are pushed) and plans GRASP / LIFT / MOVE /
RELEASE / RETREAT stages; the controller gets GRASP and RELEASE in its vocabulary plus a HAND block (the
object must sit between the open fingers and the palm in both views before GRASP; RELEASE first if the
hand is closed and empty; release only when lowered onto the destination), `Hand now: open|closed` and,
after a hand command, `Last hand command: <result>` ("hand closed on an object ..." or "EMPTY grasp ...").
"Hand now" is measured from the fingers until the first command, so a hand that starts closed is not
reported open; "Holding an object" is only said after a GRASP that closed on something. GRAB, CLOSE,
OPEN, LET_GO and similar words are accepted as GRASP / RELEASE. `test_revo2.py` runs a whole simulated
pick and place with `hand.type: revo2` and checks these prompts.

Chain: `brainco_hand_server` on the Jetson (unitreerobotics/brainco_hand_service, the same bridge
xr_teleoperate's `--ee brainco` uses; hands on USB serial, DDS `MotorCmds_` on
`rt/brainco/{left,right}/cmd`, `MotorStates_` on `.../state`) -> the hand server on the Mac
(`harness/robot/revo2.py serve`, localhost:8791, the only publisher on the hand topics, publishes once
per accepted command and refuses while that hand's state is not arriving) -> `hand_client.py` in the
loop (no SDK import). The server is separate from the arm streamer because the hands' DDS comes from
the Jetson, which with the dual-link setup is a different Mac adapter than the body cable.

```
.venv/bin/python -m harness.robot.revo2 IFACE state --watch        # subscribe-only: are both hands publishing?
.venv/bin/python -m harness.robot.revo2 IFACE close --side right   # one publish (needs the operator's yes on the robot)
tools/harness_hands.sh [IFACE]                                     # the hand server, next to tools/harness_stream.sh
.venv/bin/python -m harness --set hand.type=revo2 live en6 "pick up the red cube"
.venv/bin/python -m harness --set hand.type=revo2 dry-run en6 "..."   # reads the hands, never commands them
MUJOCO_GL=cgl .venv/bin/python -m harness --set hand.type=revo2 sim "..."   # sim: same as virtual, no server needed
```

Loopback rehearsal without the robot: `python -m harness.robot.revo2 lo0 fake --domain 1 --block-at 0.5`
(fake hands, fingers stop at 0.5 as if on an object) and `python -m harness.robot.revo2 lo0 serve --domain 1`;
verified 2026-09-27. Not yet verified on the robot: the server on the Jetson side, which adapter
carries its DDS, the `empty_reach` threshold, and the end-effector offset with a hand on the flange
(`robot.ee_offset_m` is still the bare-wrist 0.13 m; the grasp point between the fingers is likely
further out).

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

The same proposal can be reviewed from Spectacles when the harness is started with
`--preview runs/ui_preview.json --spectacles-review runs/spectacles_review.json` and
`spectacles/plan_feed.py` serves that preview with `--review-file runs/spectacles_review.json`.
The desktop AI pane supplies these harness flags automatically. A double right-hand
pinch accepts and a double left-hand pinch rejects on the glasses; the harness still
consumes the decision through this confirmation callback and runs all existing checks.
See `spectacles/README.md` for the setup and network limits.

## Every session becomes a recording

When an episode ends (any mode), its accepted, executed moves are exported to
`recordings/ai_<mode>_<task>_<time>.json` in the sim contract (`recorder.export_dir`): commanded
joint targets of both arms and the waist, each move taking the duration the gate gave it, then
0.5 s, with the thinking time removed; a contact sheet is built from the images the model saw
before each move, so the recording shows up with ✓ in the window's context list and can be
selected as a demonstration for the next session, dry-run it and replayed with `tools/arm_lift.py`
like a taught skill. A dry-run export is the pretend trajectory (what would have been sent).

## Step sizes

Unit moves are `steps.coarse_m` (4 cm) or `steps.fine_m` (2 cm, wrist camera sees the target),
1 cm in the precision profile. The controller prompt also offers `MOVE <dir> <cm>` up to
`steps.param_max_translation_m` (20 cm) and tells the model to take one sized move when the target
is far and the way is free, small moves near it. The gate caps unit and sized moves separately.

## Robot pose view (a second reading of the state)

With `perception.pose_view` (default on) every planner and controller call also gets a ROBOT POSE
VIEW: the fixed-base MuJoCo model posed from the measured joints, rendered from the front right
with the hand tip (cyan), where the last move aimed (yellow, with a line to where the hand actually
is), the workspace box (white) and the table height (plane), captioned with the tip position
(`harness/poseview.py`). It is not a camera, and the prompt says so: it tells the model where its
hand is when the cameras do not show it and whether the last move went where it aimed, next to the
joint numbers in the text; the cameras stay the only source for where the target is. Recorded as
`robot.jpg` per step and replayed by `harness replay`. Costs one more image (about 300 tokens, one
more Read with `claude -p`).

## Locomotion (whole-body steps)

Off by default, and on only when the task text itself contains the word "walk": the window then starts the
streamer and the loop with `locomotion.enabled=true`, and the loop refuses a walk otherwise. That adds WALK_FWD /
WALK_BACK / WALK_LEFT / WALK_RIGHT (30 cm), TURN_LEFT / TURN_RIGHT (20 deg), `WALK <dir> <cm>` (up
to 60 cm) and `TURN <deg>` (up to 45 deg) to the vocabulary, tells the planner to add an APPROACH
stage when the target is beyond the arm's reach, and tells the controller to walk only when the
target is out of reach and to look again afterwards. A step never sits in a plan chunk. The gate
(`SafetyGate.vet_walk`) caps each step, keeps a per-episode budget (5 m, 360 deg) and turns the
step into a velocity and a duration; the streamer's `walk` command checks enabled, FSM 811, speed
and duration again on its own, asks the loco service for that velocity for that long, then sends an
explicit stop (also on e-stop and Ctrl-C), and reports odometry from `rt/sportmodestate` as the
achieved (dx, dy, dyaw) in the pre-step frame. Every step is a PROPOSAL behind Accept, shown as
text (the twin's ghost has nothing to play). In sim the mock shifts the scene under the fixed base.

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
