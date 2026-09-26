#!/bin/sh
# Start the stream servers if they are not up, then open the Reins window.
cd "$(dirname "$0")/.." || exit 1
up() { curl -s -m 2 -o /dev/null "http://localhost:$1/"; }
up 8081 || { nohup .venv/bin/python tools/headcam.py en6 > /tmp/reins_headcam.log 2>&1 & }
up 8080 || { ssh -o BatchMode=yes -o ConnectTimeout=6 jetson-en8 'ss -ltn | grep -q 8080 || (setsid nohup python3 ~/camstream.py --devices 0,2 --width 640 --fps 15 > ~/camstream.log 2>&1 < /dev/null &)' \
             && nohup .venv/bin/python tools/via_iface.py en8 192.168.123.164 8080 --listen 8080 > /tmp/reins_forward.log 2>&1 & }
up 8082 || { nohup .venv/bin/python tools/cockpit.py en6 > /tmp/reins_cockpit.log 2>&1 & }
sleep 4
exec .venv/bin/python tools/reins_ui.py "$@"
