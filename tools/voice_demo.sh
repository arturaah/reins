#!/bin/bash
# Laptop microphone → GPT-Live + Voice Focus/Tyto → R1 or laptop speakers.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$PWD"
VOICE_PYTHON="${VOICE_PYTHON:-$REPO_DIR/.venv-voice/bin/python}"
ROBOT_PYTHON="${ROBOT_PYTHON:-$REPO_DIR/.venv/bin/python}"
DEMO_PORT="${DEMO_PORT:-8770}"
DEMO_IFACE="${1:-}"
if [ "$DEMO_IFACE" = "--help" ]; then
  echo 'Usage: bash tools/voice_demo.sh [--laptop | ROBOT_INTERFACE]'
  echo 'No argument: find the robot body link, otherwise use laptop speakers.'
  exit 0
fi

if [ ! -x "$VOICE_PYTHON" ]; then
  echo 'Preparing a separate Python 3.12 voice environment…'
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$REPO_DIR/.venv-voice"
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$REPO_DIR/.venv-voice"
  else
    echo 'Python 3.12 is needed. Install it, then run this command again.' >&2
    exit 1
  fi
fi
if ! "$VOICE_PYTHON" -c 'import aic_sdk, fastapi, uvicorn, openai, websockets, webrtcvad, dotenv, numpy, scipy' >/dev/null 2>&1; then
  echo 'Installing voice dependencies…'
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VOICE_PYTHON" -r voice/requirements-aic.txt
  else
    "$VOICE_PYTHON" -m pip install -r voice/requirements-aic.txt
  fi
fi
"$VOICE_PYTHON" - <<'PY'
from voice.config import load_keys
import os
load_keys()
if not os.environ.get('OPENAI_API_KEY') or not os.environ.get('AIC_SDK_LICENSE'):
    raise SystemExit('Add OPENAI_KEY and AIC_KEY to the project .env, then run again.')
PY

# Match the existing Mac launcher: probe the controller on each robot-subnet link.
if [ -z "$DEMO_IFACE" ] && command -v ifconfig >/dev/null 2>&1; then
  for candidate in $(ifconfig | awk '/^[a-z0-9]+:/{i=$1; sub(":","",i)} /inet 192\.168\.123\./{print i}'); do
    if ping -c 1 -t 1 -b "$candidate" 192.168.123.161 >/dev/null 2>&1; then
      DEMO_IFACE="$candidate"; break
    fi
  done
fi
args=(--backend conversation --port "$DEMO_PORT" --voice-focus --tyto)
if [ -n "$DEMO_IFACE" ] && [ "$DEMO_IFACE" != "--laptop" ]; then
  if ! "$ROBOT_PYTHON" -c 'from unitree_sdk2py.rpc.client import Client' >/dev/null 2>&1; then
    echo 'Robot SDK Python unavailable. Set ROBOT_PYTHON or run with --laptop.' >&2
    exit 1
  fi
  args+=(--output r1 --robot-iface "$DEMO_IFACE" --robot-python "$ROBOT_PYTHON")
  echo "Microphone: laptop. Speaker: R1 on $DEMO_IFACE."
else
  args+=(--output browser)
  echo 'Microphone: laptop. Speaker: laptop. Robot connection not required.'
fi
echo "Open http://127.0.0.1:$DEMO_PORT/ and click Start conversation. Allow the laptop microphone."
echo 'GPT-Live + Voice Focus + Tyto. Conversation only. Stop ends audio; Ctrl-C stops the service.'
exec "$VOICE_PYTHON" -m voice.live "${args[@]}"
