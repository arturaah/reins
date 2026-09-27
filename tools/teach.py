"""Retired kinesthetic teaching entry point.

Making joints compliant is interactive physical control, not a complete reviewed
trajectory. This former independent publisher is intentionally unavailable.
Existing recordings and passive dataset tools remain usable offline.
"""
import sys


def main(argv=None):
    print("Kinesthetic teaching is retired from the supported Reins pipeline. "
          "It cannot run through complete-motion approval. Use the dashboard for "
          "reviewed motions; tools/record.py remains a separate passive dataset utility.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
