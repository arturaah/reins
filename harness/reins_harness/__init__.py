"""An LLM drives the R1 in simulation, and a human reviews every plan first. See harness/README.md."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _dir in ("loco", "contract"):  # sibling packages in this repo, not installed
    if str(_REPO / _dir) not in sys.path:
        sys.path.insert(0, str(_REPO / _dir))
