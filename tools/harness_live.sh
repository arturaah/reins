#!/bin/sh
# Compatibility task submission. Approval happens only in dashboard/glasses.
set -eu
cd "$(dirname "$0")/.." || exit 1
if [ "$#" -ne 1 ]; then
  echo 'Usage: tools/harness_live.sh "task text" (start dashboard first)' >&2
  echo 'Use python -m tools.reins --url http://127.0.0.1:8090 prompt "task" for another port.' >&2
  exit 2
fi
exec .venv/bin/python -m tools.reins prompt "$1"
