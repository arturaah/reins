#!/bin/sh
# Compatibility launcher: one dashboard; no automatic robot/Jetson service setup.
set -eu
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python tools/dashboard.py "$@"
