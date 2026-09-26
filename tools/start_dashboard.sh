#!/bin/sh
# Browser dashboard: preview, cameras, and Dry run / Execute / Abort via tools/arm_lift.py.
# Existing camera services reconnect automatically. Pass --iface IFACE for the robot link.
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python tools/dashboard.py "$@"
