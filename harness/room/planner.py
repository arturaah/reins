"""Find a walkable path around obstacles the robot has seen.

A* on a 5 cm grid over a clearance function (the robot's depth-built
`ObstacleMap`, not the sim's true furniture), then shortcut to as few straight
segments as keep a safety margin. The margin matters because the path follower
cuts corners by up to its lookahead. Near the start and goal, which may sit
close to furniture on purpose (standing at a table to reach it), the margin
shrinks to whatever clearance those poses have.
"""
from __future__ import annotations

import heapq
import math

from typing import Callable

CELL = 0.05  # m
MARGIN = 0.12  # m of clearance wanted along the path
PREFERRED = 0.3  # m; closer than this to an obstacle costs extra


class NoPath(ValueError):
    pass


Clearance = Callable[[float, float], float]  # (x, y) -> distance from footprint edge to nearest obstacle


def plan_path(clearance: Clearance, start: tuple[float, float], goal: tuple[float, float],
              known: list[tuple[float, float]] = ()) -> list[tuple[float, float]]:
    """Points from start to goal (both included), clear of obstacles.

    `known` are obstacle locations, only used to size the search area."""
    c_start, c_goal = clearance(*start), clearance(*goal)
    if c_goal < 0:
        raise NoPath(f"goal ({goal[0]:.2f}, {goal[1]:.2f}) is inside an obstacle's footprint")
    floor = min(MARGIN, max(c_start, 0.0), c_goal)
    if _segment_clear(clearance, start, goal, floor):
        return [start, goal]

    xs = [start[0], goal[0]] + [k[0] for k in known]
    ys = [start[1], goal[1]] + [k[1] for k in known]
    x0, y0 = min(xs) - 1.5, min(ys) - 1.5
    nx, ny = int((max(xs) + 1.5 - x0) / CELL) + 1, int((max(ys) + 1.5 - y0) / CELL) + 1

    def cell(p):
        return round((p[0] - x0) / CELL), round((p[1] - y0) / CELL)

    def centre(c):
        return x0 + c[0] * CELL, y0 + c[1] * CELL

    free_cache: dict[tuple[int, int], float] = {}

    def cost_at(c) -> float | None:
        if c not in free_cache:
            free_cache[c] = clearance(*centre(c))
        cl = free_cache[c]
        if cl < min(floor, 0.02) and c not in (s, g):
            return None
        return 1.0 + 3.0 * max(0.0, PREFERRED - cl) / PREFERRED

    s, g = cell(start), cell(goal)
    frontier = [(0.0, s)]
    came = {s: None}
    spent = {s: 0.0}
    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == g:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                continue
            w = cost_at(nxt)
            if w is None:
                continue
            new = spent[cur] + w * math.hypot(dx, dy)
            if new < spent.get(nxt, math.inf):
                spent[nxt], came[nxt] = new, cur
                heapq.heappush(frontier, (new + math.hypot(g[0] - nxt[0], g[1] - nxt[1]), nxt))
    if g not in came:
        raise NoPath(f"no walkable path from ({start[0]:.2f}, {start[1]:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f})")
    cells = []
    c = g
    while c is not None:
        cells.append(c)
        c = came[c]
    raw = [start] + [centre(c) for c in reversed(cells[1:-1])] + [goal]
    return _shortcut(clearance, raw, floor)


def path_length(points) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _shortcut(clearance, points, floor) -> list[tuple[float, float]]:
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not _segment_clear(clearance, points[i], points[j], floor):
            j -= 1
        out.append(points[j])
        i = j
    return [(round(x, 3), round(y, 3)) for x, y in out]


def _segment_clear(clearance, a, b, floor) -> bool:
    n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.03))
    for k in range(n + 1):
        x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
        # Full margin in the middle, relaxing to `floor` within 0.4 m of either end.
        near_end = min(math.hypot(x - a[0], y - a[1]), math.hypot(x - b[0], y - b[1]))
        need = floor + (MARGIN - floor) * min(1.0, near_end / 0.4)
        if clearance(x, y) < need - 1e-9:
            return False
    return True
