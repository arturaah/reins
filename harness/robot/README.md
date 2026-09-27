# Reviewed robot transports

The dashboard's `RobotPipeline` owns motion approval. Connecting a bridge reads
telemetry; it does not engage the arms or move the head. After human approval,
`ArmClientBackend.execute_motion(payload, approval)` sends exactly one complete
arm, bounded base, or hand motion.

The dashboard starts its local arm bridge with a temporary, private capability
file and passes the same file to a configured Revo2 hand bridge. The file must
belong to the current user, have mode `0600`, and contain at least 32 characters.
Its contents never belong in prompts, browser responses, command-line arguments,
or logs. Only the file path is passed with `--control-token-file`.

A manually launched bridge without a capability is telemetry-only. A bridge that
already belongs to another controller must be released by that controller before
reuse. Both bridges bind to loopback. When the DDS bridge runs on another host,
use a private tunnel; this protocol does not provide network encryption.

## Protocol

Each line is JSON. `hello` and `state` are public read-only commands. A private
controller first sends `authenticate` with its token. There is one authenticated
owner, while read-only observers can connect independently. Observers cannot keep
the control watchdog alive, and disconnecting an observer does not stop the owner.

The only motion command is `execute_motion`, carrying a canonical payload from
`contract/runtime.py` plus a review receipt containing `proposal_id`, `revision`,
`digest`, and `expires_at`. Receipts are consumed before actuation, including when
validation or delivery fails. Raw `frames`, `plan`, `walk`, `engage`, and hand
`set` commands are rejected. The model has no interface that issues receipts.

Arm execution checks the complete path, measured table, start pose, command timing,
joint limits, acceleration, speed, and robot geometry before holding the current
pose and streaming the reviewed samples. No head movement or approach path is
inserted. Walking also checks fresh robot telemetry, the configured FSM, velocity,
duration, distance, turn limits, and cumulative controller-session budget.
Interrupted or uncertain walks conservatively consume their reserved budget. Before
walking, the bridge releases arm weight and waits for an allowed balance-controller
FSM. It leaves the arms released afterwards; taking arm control again requires a
separately approved arm motion.

`freeze`, `release`, disconnect, missed controller heartbeats, and process shutdown
cancel active arm and base motion. Walking remains interruptible even if the arms
were never engaged. SIGTERM follows the same release path as Ctrl-C. Stop stays
latched in both client and bridge until a new controller connection is authenticated;
re-authenticating the existing socket does not clear it. Successful motions can be
followed by another approved motion on the same connection.

## Revo2 hands

Set `hand.type: revo2` and configure `hand.revo2.iface` when the hands' DDS interface
is different from the body interface. The configured open/close targets retain
upstream's normalized six-motor poses and speed limits. Every hand motion needs
its own complete proposal and human approval. The bridge rejects stale hand
telemetry and invalid or nonfinite targets.

On stop or owner disconnect, the hand bridge requests a hold at measured finger
positions with zero requested speed when fresh state is available. It reports an
error if fresh state is unavailable; it does not invent a finger target. This
firmware behavior still requires hardware commissioning. Finger feedback indicates
whether the close target was reached or fingers stopped early; it is not independent
proof that an object was grasped.

Direct `revo2 open` and `revo2 close` are retired. `revo2 IFACE state` remains
subscribe-only. Fake hands remain restricted to a nonzero DDS domain.

These controls protect the supported Reins command surfaces. They do not prevent
an arbitrary program running as the same OS user, or an external vendor controller,
from issuing its own robot commands.

## Offline verification

```sh
.venv/bin/python -m pytest -q harness/tests/test_streamer_watchdog.py harness/tests/test_revo2.py harness/tests/test_locomotion.py
```

These tests replace DDS, robot telemetry, loco RPCs, and hand publishers with fakes.
They do not connect to or actuate the robot.
