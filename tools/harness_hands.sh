#!/bin/sh
# The harness's Revo2 hand server (the only publisher on rt/brainco/*/cmd), in its own Terminal window. It publishes
# only when the live loop sends a GRASP/RELEASE you accepted there. Needs brainco_hand_server running on the Jetson.
# Usage: tools/harness_hands.sh [IFACE]   IFACE = the adapter on which the Jetson's DDS arrives (default: the first
#   interface with a 192.168.123.x address; with the dual-link setup pass the Jetson adapter explicitly)
cd "$(dirname "$0")/.." || exit 1
IFACE=${1:-$(ifconfig | awk '/^[a-z0-9]+:/{i=$1; sub(":","",i)} /inet 192\.168\.123\./{print i; exit}')}
[ -n "$IFACE" ] || { echo "no interface with a 192.168.123.x address: is the robot cable plugged in?"; exit 1; }
.venv/bin/python -m harness.robot.revo2 "$IFACE" state || exit 1
.venv/bin/python -u -m harness.robot.revo2 "$IFACE" serve 2>&1 | tee /tmp/harness_hands.log
