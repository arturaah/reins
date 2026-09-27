"""Compatibility launcher for the Reins browser dashboard.

The former Tk controller is retired. This launcher opens the running local
operator dashboard; it starts no robot, camera, hand or Jetson services. Use
`tools/start_dashboard.sh --iface IFACE` to start it, or --sim for simulation.
"""
import argparse
import os
import sys
from pathlib import Path
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.reins import dashboard_url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("REINS_DASHBOARD_URL", "http://127.0.0.1:8090"))
    parser.add_argument("--iface", help="migration hint; configure this on tools/dashboard.py")
    parser.add_argument("--jetson-iface", help="migration hint; camera/hand services are managed explicitly")
    parser.add_argument("--no-open", action="store_true", help="print the dashboard URL without launching a browser")
    args = parser.parse_args(argv)
    try:
        url = dashboard_url(args.url)
    except ValueError as exc:
        parser.error(str(exc))
    if args.iface or args.jetson_iface:
        print("Robot interfaces are configured on the dashboard server; this launcher only opens its page.")
    print(url)
    print("Start the server with tools/start_dashboard.sh if it is not already running.")
    if not args.no_open:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
