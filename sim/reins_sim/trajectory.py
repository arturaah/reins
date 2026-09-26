"""Planned base trajectories: the thing a VLM proposes and a human reviews."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    # Heading in radians. None means "face the direction of travel".
    yaw: float | None = None
    label: str | None = None


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Trajectory:
    """A piecewise-linear path through planar waypoints, parameterised by arc length."""

    def __init__(self, waypoints: list[Waypoint]):
        if len(waypoints) < 2:
            raise ValueError("a trajectory needs at least two waypoints")
        self.waypoints = list(waypoints)
        self.points = np.array([(w.x, w.y) for w in waypoints], dtype=float)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cumlen = np.concatenate([[0.0], np.cumsum(seg)])
        self.yaws = self._resolve_yaws()

    @property
    def length(self) -> float:
        return float(self.cumlen[-1])

    def _resolve_yaws(self) -> np.ndarray:
        pts = self.points
        headings = []
        for i in range(len(pts)):
            j, k = (i, i + 1) if i + 1 < len(pts) else (i - 1, i)
            d = pts[k] - pts[j]
            headings.append(np.arctan2(d[1], d[0]))
        return np.array([
            w.yaw if w.yaw is not None else h for w, h in zip(self.waypoints, headings)
        ])

    def pose_at(self, s: float) -> tuple[float, float, float]:
        """(x, y, yaw) at arc length s, clamped to the path."""
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.cumlen, s, side="right") - 1)
        i = min(i, len(self.points) - 2)
        span = self.cumlen[i + 1] - self.cumlen[i]
        t = 0.0 if span == 0 else (s - self.cumlen[i]) / span
        x, y = (1 - t) * self.points[i] + t * self.points[i + 1]
        dyaw = _wrap(self.yaws[i + 1] - self.yaws[i])
        return float(x), float(y), float(_wrap(self.yaws[i] + t * dyaw))

    def sample(self, n: int) -> np.ndarray:
        """n evenly spaced (x, y, yaw) poses from start to end."""
        return np.array([self.pose_at(s) for s in np.linspace(0, self.length, n)])

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls([Waypoint(**w) for w in d["waypoints"]])

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        return cls.from_dict(json.loads(Path(path).read_text()))
