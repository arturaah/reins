# Live motion boundary and historical session experiments

[`runtime.py`](runtime.py) is the shared live payload/approval validator used by
`RobotPipeline` and the private actuator transports. Its structural companion is
[`motion.schema.json`](motion.schema.json); examples are in
[`runtime_examples/`](runtime_examples/). This is the active execution boundary.

`reins.schema.json`, `reins_contract.py` and the older `examples/*.jsonl` describe
an earlier offline multi-role session design. They retain their regression tests,
but do **not** govern the dashboard, Spectacles WebSocket or streamer. In
particular their `hello/welcome`, 250 Hz description, preset steps and provisional
corridor replanning are not promises of the current runtime.

## Supported flow

```text
Agent tools → immutable validated draft → draft preview
                               ↓ propose_motion
                  complete proposal, ID/revision/digest
                               ↓ one human decision
             fresh-state/expiry/cancellation/validation recheck
                               ↓ authenticated private connection
                      execute_motion → measured outcome
```

Planning does not confer execution authority. Model tools cannot create an
approval receipt. A digest identifies content; it is not a secret, operator
identity or permission to execute. An approved path cannot change afterward.
A new physical observation requiring another motion needs a new proposal.

## Payloads

The private newline-delimited JSON request has this shape:

```json
{
  "cmd": "execute_motion",
  "request_id": "transport-request-id",
  "payload": {"kind": "hand", "arm": "right", "closed": false},
  "approval": {
    "proposal_id": "human-reviewed-proposal-id",
    "revision": 1,
    "digest": "64 lowercase hexadecimal SHA-256 characters",
    "expires_at": 1800000120.0
  }
}
```

The snippet illustrates field meanings; the files in `runtime_examples/` contain
valid digests. Their fixed timestamps are test fixtures, not usable approvals.

| Kind | Exact payload fields | Meaning |
|---|---|---|
| `arm` | `kind`, `arm`, `plan` | One resolved schema-version1 single-arm trajectory; default command samples50Hz |
| `walk` | `kind`, `vx`, `vy`, `vyaw`, `duration_s` | One bounded constant body-velocity motion; explicit stop afterward |
| `hand` | `kind`, `arm`, `closed` | One configured Revo2 open/close action |

Positions use metres; angles use radians; times use seconds. Robot-base axes
are x forward, y left, z up, with origin on the floor below the pelvis. Arm plans
name model joints, not DDS motor slots. `held_joints_rad` carries the assumed
other-joint configuration. The streamer owns the mapping to R1 hardware slots.

The coordinator hashes the **entire payload**, including plan metadata, with
sorted JSON object keys, compact separators and non-finite values prohibited.
The receipt contains exactly `proposal_id`, integer `revision`, matching `digest`
and `expires_at`. Runtime validation checks that expiry is in the future and no
more than180 seconds away; the dashboard normally offers120 seconds for review.

## Checks that JSON Schema cannot replace

The JSON schema checks structure and individual bounds. Live execution must
also use `validate_motion`, `validate_approval`, the trajectory/policy checks and
an authenticated transport:

- All numbers must be finite. Arm times increase from zero to the stated
  duration, with2–36,001 keyframes and duration at most180 seconds.
- Walk duration is at most15 seconds; combined linear speed at most0.3 m/s,
  angular speed at most0.5 rad/s, distance at most0.6 m and turn at most45°.
  Configuration may tighten these limits; disabled capabilities stay disabled.
- The arm validator checks joints, limits, velocity/acceleration, held pose and
  swept model geometry; the coordinator adds the configured table/workspace.
- Before actuation, fresh measured state must still match the reviewed starting
  assumptions. No unreviewed lead-in, return, online steering or path repair is
  permitted. A changed path needs a new draft and review.
- The private connection must authenticate using its coordinator capability.
  The receiver consumes each approved proposal ID once; replaying a receipt
  cannot execute it again. Stop/disconnect invalidates active motion and keeps
  watchdogs effective while command execution is in progress.

The actual arm/base and hand bridge envelopes also carry transport request IDs
for concurrent response matching. Read-only `hello`/state traffic does not grant
authority. Unreviewed old `frames`, `walk` and hand-set commands are not supported
operator interfaces.

## Operator surfaces and outcomes

The dashboard uses its local browser session for human decisions. Spectacles
uses revocable paired-device credentials plus a fresh connection session, exact
proposal ID/digest/revision and registration freshness. AR `trajectory` messages
carry `phase`; drafts have `review:null` and cannot be approved. The AR wire
adapter is in `core/glasses_bridge.py`, not the old session schema.

`propose_motion(plan_id, request_id)` is idempotent and returns immediately.
`get_motion_result(proposal_id)` reads the current state or a terminal outcome:
`executed`, `declined`, `expired`, `cancelled`, `blocked` or `failed`. Available
measured end pose, tracking error and base/hand feedback are recorded separately
from predicted preview geometry. Model claims of task success are not outcomes.

This boundary does not prove environmental clearance, grasp success, AR accuracy,
walking balance or physical task completion. Camera observations are2D, without
metric object depth. Those capability limits remain visible to the model and
operator.

## Changing the boundary

Change runtime validation, `motion.schema.json`, examples and regression tests
together. Retain the historical contract tests while those fixtures remain.

```sh
.venv/bin/python -m pytest -q contract/tests
```

Runtime tests validate examples against both the schema and executable checks,
freeze time for their example receipts, and reject mutated payloads, stale
receipts, non-finite values and excessive combined walking limits. Streamer and
pipeline tests separately cover authentication, one-time use and fault injection.
