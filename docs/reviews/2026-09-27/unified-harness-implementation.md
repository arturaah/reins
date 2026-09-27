# Unified harness implementation

Implemented on `fix/dashboard-trajectory-retries`, rebased onto `origin/main`
`f917db9` (wireless Spectacles). Main was unchanged by this work.

## Supported flow

Dashboard chat, the operator HTTP CLI and paired Spectacles submit tasks to the
same agent. Codex CLI, Claude CLI and the OpenAI Responses adapter expose the
same planning tools. The agent can receive actual camera images, detect 2D
objects, compile a novel single-arm waypoint path, inspect a rejection, revise,
and preview without an approval. Planning has bounded time/call/revision budgets.

`propose_motion` submits one immutable complete motion to `RobotPipeline`.
Dashboard or glasses approve its exact digest/revision once. The coordinator
rechecks current pose, observation freshness, policy and path, then sends an
authenticated, single-use receipt to the private actuator bridge. Stop remains
latched until reconnect. There is no per-step model execution loop or model
approval/firmware tool. The measured outcome returns to the conversation.

Drafts remain editable through new planning calls; changed paths cannot inherit
an earlier approval. Manual arm, base and hand controls use the same review gate.
Robot connection reads telemetry without engaging arms. The measured robot view
renders coordinator telemetry locally, without a separate cockpit process.

## Integrated teammate functionality

- Walking retains the arm-weight handoff, cancellable FSM wait, controller error
  feedback and odometry from main. Arms remain released after walking. The
  configured note about firmware refusing SDK locomotion is visible in the GUI.
- Revo2 uses a private hand bridge and the same reviewed payload boundary. The
  dashboard manages its local process when configured; physical services on the
  robot are not started by the old launchers.
- Persistent glasses pairing, revocation, socket sessions, tracking freshness and
  review expiry coexist with wireless microphone streaming and ASR retries.
- Live voice can use `--backend dashboard`; its adapter submits chat tasks only.
  Standalone conversation/test modes and explicit R1 speaker output remain.
- Historical proposal cards reuse the upstream renderer, with exact payload,
  observed image, verdict and measured outcome. Approved-but-failed, simulated
  and completed motions remain distinct. Bounded private memory is returned as
  historical text/images and can be forgotten in Connections. It is not replay
  authority or model training.
- Offline harness simulation, evaluations, experience experiments and episode
  exports remain. Direct live teaching, raw-frame entry points and the former
  Tk executor are retired; the supported launcher opens the dashboard.

The live motion contract is `contract/runtime.py` plus its schema, examples and
tests. The older session contract remains an offline experiment.

## Verification

Final combined run:

```sh
REINS_NODE=/path/to/node .venv/bin/python -m pytest -q \
  core tools harness/tests contract spectacles/tests voice/tests
```

**512 passed, 44 subtests passed.** One third-party Starlette/httpx deprecation
warning remains. The three standalone Node suites also passed 19 tests.

A scripted model exercised actual image output, invalid-path feedback, revised
IK compilation, preview, one proposal, one human decision and measured outcome
feedback through the real HTTP tool boundary, using fake actuators. Fault tests
cover stale state, changed/expired approvals, cancellation during planning and
streaming, disconnect, duplicate submission, pairing revocation and voice review
priority. The installed Codex/Claude CLI help and Codex feature inventory support
the configured launch flags; no live model request was made.

Headless Chrome exercised the actual simulation dashboard on desktop and at
390px width: draft/review separation, rejection notes, wrist/base/hand approval,
pairing/revocation, memory count/forget, and hardware lockout. No page errors or
horizontal overflow occurred. Temporary test servers were stopped.

## Remaining physical limits

No hardware actuation or Jetson service change was performed. Physical
commissioning with the operator remains necessary. The supported proposal is a
single-arm trajectory, one base movement, or one hand action; mixed coordinated
motions and dual-arm execution are not implemented. Camera observations have no
metric depth or calibrated object location. Collision checks cover modeled
geometry and configured table clearance, not arbitrary scene obstacles; preview
does not establish balance, contact or grasp success. The repository's current
robot note reports controller code 127 for SDK walking until firmware enables it.
