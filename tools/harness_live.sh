#!/bin/sh
# Run the VLM harness on the robot from a Terminal window, so the per-step confirmations are typed here.
# Usage: tools/harness_live.sh "task text" [more harness options...]
# Needs: robot standing in FSM 811, tools/headcam.py en6 running, and the streamer
#   .venv/bin/python -m harness.robot.arm_stream en6   (started separately; it publishes only after ENGAGE is confirmed here)
cd "$(dirname "$0")/.." || exit 1
TASK="$1"; shift
echo "harness live: $TASK"
echo "Nothing moves until you press Enter at each SEND? prompt. n skips a move, x is the e-stop, Ctrl-C releases the arms."
.venv/bin/python -m harness "$@" live en6 "$TASK" 2>&1 | tee /tmp/harness_live.log
echo; echo "harness finished. This window can be closed."
read -r _
