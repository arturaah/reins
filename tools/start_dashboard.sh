#!/bin/sh
# Unified dashboard; use --sim for offline operation, --iface for robot telemetry.
# Hardware ownership and motion require explicit connection and complete review.
set -eu
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python tools/dashboard.py "$@"
