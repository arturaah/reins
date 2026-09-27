#!/bin/sh
# Start the stream servers if they are not up, replace any running Reins window with a fresh one.
# Run it from Terminal (open -a Terminal tools/start_all.sh) so macOS remembers the privacy grants.
# The window's output goes to /tmp/reins_ui.log; a crash also lands in /tmp/reins_ui_crash.log.
cd "$(dirname "$0")/.." || exit 1
# The adapters re-enumerate after a re-plug (en6 one day, en8 the next) and both hold a 192.168.123.x address, so the
# body cable is the one on which the motion controller (.161) answers a ping bound to it; with the dual-link setup the
# other adapter is the Jetson module (wrist cameras, the Revo2 hands' DDS).
IFACE=""; JETSON=""
for i in $(ifconfig | awk '/^[a-z0-9]+:/{i=$1; sub(":","",i)} /inet 192\.168\.123\./{print i}'); do
  if ping -c 1 -t 1 -b "$i" 192.168.123.161 >/dev/null 2>&1; then IFACE=$i; else JETSON=${JETSON:-$i}; fi
done
[ -n "$IFACE" ] || IFACE=$JETSON            # one link through the module: everything on that adapter
[ -n "$IFACE" ] || { echo "no interface with a 192.168.123.x address: is the robot cable plugged in?"; exit 1; }
echo "robot interface: $IFACE${JETSON:+   Jetson adapter: $JETSON}"
up() { curl -s -m 2 -o /dev/null "http://localhost:$1/"; }
up 8081 || { nohup .venv/bin/python tools/headcam.py "$IFACE" > /tmp/reins_headcam.log 2>&1 & }
if ! up 8080; then
  if [ -n "$JETSON" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new \
        -o ProxyCommand="$PWD/.venv/bin/python $PWD/tools/via_iface.py $JETSON %h %p" unitree@192.168.123.164 \
        'ss -ltn | grep -q 8080 || (setsid nohup python3 ~/camstream.py --devices 0,2 --width 640 --fps 15 > ~/camstream.log 2>&1 < /dev/null &)' \
      && nohup .venv/bin/python tools/via_iface.py "$JETSON" 192.168.123.164 8080 --listen 8080 > /tmp/reins_forward.log 2>&1 &
  else
    echo "no second USB Ethernet adapter to the Jetson: wrist cameras and the Revo2 hands stay off"
  fi
fi
up 8082 || { nohup .venv/bin/python tools/cockpit.py "$IFACE" > /tmp/reins_cockpit.log 2>&1 & }
if pgrep -f "python tools/reins_ui.py" >/dev/null; then
  echo "closing the previous Reins window"; pkill -f "python tools/reins_ui.py"; sleep 1
fi
sleep 3
echo "opening the Reins window (log: /tmp/reins_ui.log)"
.venv/bin/python tools/reins_ui.py --iface "$IFACE" ${JETSON:+--jetson-iface "$JETSON"} "$@" 2>&1 | tee /tmp/reins_ui.log
