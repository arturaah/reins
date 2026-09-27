#!/bin/sh
# Run the VLM harness on the robot from a Terminal window, so the per-step confirmations are typed here.
# Usage: tools/harness_live.sh "task text" [more harness options...]
# Needs: robot standing in FSM 811, tools/headcam.py "$IFACE" running, and the streamer
#   .venv/bin/python -m harness.robot.arm_stream "$IFACE"   (started separately; it publishes only after ENGAGE is confirmed here)
cd "$(dirname "$0")/.." || exit 1
# the adapter carrying the robot cable: whichever interface holds a 192.168.123.x address (it re-enumerates after a re-plug)
IFACE=$(ifconfig | awk '/^[a-z0-9]+:/{i=$1; sub(":","",i)} /inet 192\.168\.123\./{print i; exit}')
[ -n "$IFACE" ] || { echo "no interface with a 192.168.123.x address: is the robot cable plugged in?"; exit 1; }
TASK="$1"; shift
echo "harness live: $TASK"
echo "Nothing moves until you press Enter at each SEND? prompt. n skips a move, x is the e-stop, Ctrl-C releases the arms."
.venv/bin/python -m harness "$@" live "$IFACE" "$TASK" 2>&1 | tee /tmp/harness_live.log
echo; echo "harness finished. This window can be closed."
read -r _
