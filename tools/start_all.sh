#!/bin/sh
# Start the stream servers if they are not up, replace any running Reins window with a fresh one.
# Run it from Terminal (open -a Terminal tools/start_all.sh) so macOS remembers the privacy grants.
# The window's output goes to /tmp/reins_ui.log; a crash also lands in /tmp/reins_ui_crash.log.
cd "$(dirname "$0")/.." || exit 1
up() { curl -s -m 2 -o /dev/null "http://localhost:$1/"; }
up 8081 || { nohup .venv/bin/python tools/headcam.py en6 > /tmp/reins_headcam.log 2>&1 & }
if ! up 8080; then
  if ifconfig en8 >/dev/null 2>&1; then
    ssh -o BatchMode=yes -o ConnectTimeout=6 jetson-en8 'ss -ltn | grep -q 8080 || (setsid nohup python3 ~/camstream.py --devices 0,2 --width 640 --fps 15 > ~/camstream.log 2>&1 < /dev/null &)' \
      && nohup .venv/bin/python tools/via_iface.py en8 192.168.123.164 8080 --listen 8080 > /tmp/reins_forward.log 2>&1 &
  else
    echo "no en8 (second USB Ethernet adapter to the Jetson): wrist cameras stay off"
  fi
fi
up 8082 || { nohup .venv/bin/python tools/cockpit.py en6 > /tmp/reins_cockpit.log 2>&1 & }
if pgrep -f "python tools/reins_ui.py" >/dev/null; then
  echo "closing the previous Reins window"; pkill -f "python tools/reins_ui.py"; sleep 1
fi
sleep 3
echo "opening the Reins window (log: /tmp/reins_ui.log)"
.venv/bin/python tools/reins_ui.py "$@" 2>&1 | tee /tmp/reins_ui.log
