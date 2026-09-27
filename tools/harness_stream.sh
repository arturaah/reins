#!/bin/sh
# The harness's only rt/arm_sdk publisher, in its own Terminal window. Publishes nothing until the live loop
# asks to ENGAGE and you confirm there. Ctrl-C here ramps the weight down and hands the arms back.
cd "$(dirname "$0")/.." || exit 1
# the adapter carrying the robot cable: whichever interface holds a 192.168.123.x address (it re-enumerates after a re-plug)
IFACE=$(ifconfig | awk '/^[a-z0-9]+:/{i=$1; sub(":","",i)} /inet 192\.168\.123\./{print i; exit}')
[ -n "$IFACE" ] || { echo "no interface with a 192.168.123.x address: is the robot cable plugged in?"; exit 1; }
.venv/bin/python -m harness.robot.arm_stream "$IFACE" "$@" 2>&1 | tee /tmp/harness_stream.log
