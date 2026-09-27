"""Shared deterministic workspace and locomotion policy; no model or actuator calls."""
import math
import numpy as np
from contract.runtime import finite, validate_motion


def table_obstacles(cfg, live):
    ws = cfg["workspace"]
    height = ws["table_z_m"] if live else None
    if live and height is None:
        raise ValueError("Measure and enter table height before preparing a live arm motion")
    if height is None:
        return []
    height = finite(height, "Table height")
    lo, hi = ws["box_min_m"], ws["box_max_m"]
    return [{"name": "measured table workspace", "min": [lo[0], lo[1], -.1],
             "max": [hi[0], hi[1], height + float(ws["table_margin_m"])]}]


def check_waypoints(cfg, waypoints, live):
    """Never silently clamp authored geometry: return a useful rejection for revision."""
    ws = cfg["workspace"]
    lo, hi = np.array(ws["box_min_m"]), np.array(ws["box_max_m"])
    for i, waypoint in enumerate(waypoints):
        p = np.asarray(waypoint["position_m"], float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError(f"Waypoint {i+1} must contain three finite metres")
        if np.any(p < lo) or np.any(p > hi):
            raise ValueError(f"Waypoint {i+1} outside configured workspace {lo.tolist()} to {hi.tolist()}")
        if live and ws["table_z_m"] is not None and p[2] < ws["table_z_m"] + ws["table_margin_m"]:
            raise ValueError(f"Waypoint {i+1} is below the table clearance")


def walking_payload(cfg, dx, dy, dyaw):
    lo = cfg.get("locomotion") or {}
    if not lo.get("enabled", False):
        raise ValueError("Walking is disabled in this session's configuration")
    dx, dy, dyaw = [finite(v, n) for v, n in zip((dx, dy, dyaw), ("dx", "dy", "dyaw"))]
    distance = math.hypot(dx, dy)
    if distance > min(.6, float(lo["param_max_walk_m"])) or abs(dyaw) > min(math.pi/4, math.radians(lo["param_max_turn_deg"])):
        raise ValueError("Requested base motion exceeds the configured distance/turn bounds")
    speed = min(.3, finite(lo["speed_mps"], "Walking speed"))
    angular = min(.5, finite(lo["turn_speed_rps"], "Turn speed"))
    if speed <= 0 or angular <= 0:
        raise ValueError("Walking speeds must be positive")
    duration = max(.3, distance/speed, abs(dyaw)/angular)
    return validate_motion({"kind": "walk", "vx": dx/duration, "vy": dy/duration,
                            "vyaw": dyaw/duration, "duration_s": duration})


def base_path(payload, samples=60):
    """Predicted constant body-velocity path; odometry is reported separately."""
    vx, vy, w, duration = [payload[k] for k in ("vx", "vy", "vyaw", "duration_s")]
    result = []
    for t in np.linspace(0, duration, samples):
        yaw = w*t
        x = (vx*math.sin(yaw)+vy*(math.cos(yaw)-1))/w if abs(w)>1e-8 else vx*t
        y = (vx*(1-math.cos(yaw))+vy*math.sin(yaw))/w if abs(w)>1e-8 else vy*t
        result.append({"time_s": float(t), "position_m": [float(x), float(y), 0.], "yaw_rad": float(yaw)})
    return result
