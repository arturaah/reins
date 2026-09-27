"""Room layer: a top-level model walks the R1 around a room and hands the arm to the
arm policy in the rest of `harness`, with human review of every plan. See README.md here."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _dir in ("loco", "contract"):  # sibling packages in this repo, not installed
    if str(_REPO / _dir) not in sys.path:
        sys.path.insert(0, str(_REPO / _dir))
