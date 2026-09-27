#!/bin/sh
# The harness's Revo2 hand server (the only publisher on rt/brainco/*/cmd), in its own Terminal window. It publishes
# only when the live loop sends a GRASP/RELEASE you accepted there. Needs brainco_hand_server running on the Jetson.
# Usage: tools/harness_hands.sh [IFACE]   IFACE = the adapter on which the Jetson's DDS arrives. Default: with the
#   dual-link setup the 192.168.123.x adapter on which the motion controller (.161) does NOT answer a bound ping (the
#   body cable is the one where it does); with a single link, that adapter.
cd "$(dirname "$0")/.." || exit 1
IFACE=$1
if [ -z "$IFACE" ]; then
  BODY=""
  for i in $(ifconfig | awk '/^[a-z0-9]+:/{i=$1; sub(":","",i)} /inet 192\.168\.123\./{print i}'); do
    if ping -c 1 -t 1 -b "$i" 192.168.123.161 >/dev/null 2>&1; then BODY=$i; else IFACE=${IFACE:-$i}; fi
  done
  IFACE=${IFACE:-$BODY}
fi
[ -n "$IFACE" ] || { echo "no interface with a 192.168.123.x address: is the robot cable plugged in?"; exit 1; }
echo "Revo2 hand server on $IFACE"
.venv/bin/python -m harness.robot.revo2 "$IFACE" state || exit 1
.venv/bin/python -u -m harness.robot.revo2 "$IFACE" serve 2>&1 | tee /tmp/harness_hands.log
