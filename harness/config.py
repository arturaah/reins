"""Load harness/config.yaml (or another file) into a dict with attribute access."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path(__file__).with_name("config.yaml")


class Cfg(dict):
    """dict with attribute access, nested. cfg.steps.coarse_m == cfg["steps"]["coarse_m"]."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v
    def __setattr__(self, k, v):
        self[k] = v


def load(path=None, overrides=None):
    cfg = yaml.safe_load(Path(path or DEFAULT).read_text())
    for dotted, v in (overrides or {}).items():
        d = cfg
        *keys, last = dotted.split(".")
        for k in keys:
            d = d.setdefault(k, {})
        d[last] = v
    return Cfg(copy.deepcopy(cfg))
