"""Thin DDS helper for the dashboard's R1 preset gesture buttons."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.robot_lease import RobotLease
from contextlib import nullcontext
from core.r1_gestures import create_client, request_firmware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('iface')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true')
    action.add_argument('--action', type=int)
    args = parser.parse_args()
    try:
        with RobotLease("firmware gesture") if args.action is not None else nullcontext():
            result = request_firmware(create_client(args.iface), args.action)
    except ImportError:
        result = {'error':'Unitree SDK2 Python is not installed in the dashboard environment.'}
    except ValueError as exc:
        result = {'error':str(exc)}
    except Exception:
        result = {'error':'Could not reach the R1 arm controller. Check the robot and network interface.'}
    print(json.dumps(result), flush=True)
    return 1 if 'error' in result else 0


if __name__ == '__main__':
    raise SystemExit(main())
