#!/bin/sh
# The harness's only rt/arm_sdk publisher, in its own Terminal window. Publishes nothing until the live loop
# asks to ENGAGE and you confirm there. Ctrl-C here ramps the weight down and hands the arms back.
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python -m harness.robot.arm_stream en6 "$@" 2>&1 | tee /tmp/harness_stream.log
