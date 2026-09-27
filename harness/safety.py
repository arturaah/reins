"""THE safety gate. Every hand motion, sim or real, is vetted here and nowhere else.

Order of checks for a proposed setpoint:
  1. e-stop set -> refuse.
  2. per-command translation and rotation caps (unit mode: steps.max_*; param mode: steps.param_max_*)
  3. workspace box, then table floor (table_z + margin); clamps, never refuses, and flags it
  4. IK: reachable within limits.ik_tol_m, else IK_FAIL (nothing moves)
  5. joint limits with margin and per-joint velocity for the interpolated move -> duration
  6. self-collision veto: more contacts in the model than at the start pose -> refuse
A Verdict is the only object the executor accepts; the backend never sees raw joint targets.
The live gate refuses to start without a measured table height (workspace.table_z_m).
"""
import math
import threading
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Verdict:
    ok: bool
    reason: str = ""                       # why not ok (IK_FAIL, ESTOP, LIMITS, COLLISION, ...)
    p: Optional[np.ndarray] = None         # clamped setpoint actually used
    roll: float = 0.0
    q_target: Optional[np.ndarray] = None  # 5 arm joints
    duration_s: float = 0.0
    clamped: list = field(default_factory=list)   # human-readable clamp notes
    ik_err_m: float = 0.0

    @property
    def feedback(self):
        parts = []
        if not self.ok:
            parts.append(self.reason)
        if self.clamped:
            parts.append("clamped: " + "; ".join(self.clamped))
        return " ".join(parts)


@dataclass
class WalkVerdict:
    ok: bool
    reason: str = ""
    dx: float = 0.0
    dy: float = 0.0
    dyaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    duration_s: float = 0.0
    clamped: list = field(default_factory=list)

    @property
    def feedback(self):
        parts = []
        if not self.ok:
            parts.append(self.reason)
        if self.clamped:
            parts.append("clamped: " + "; ".join(self.clamped))
        return " ".join(parts)


class SafetyGate:
    def __init__(self, cfg, kin, table_z, live):
        """cfg: the whole config. kin: ArmKinematics. table_z: metres in the robot frame or None. live: real robot."""
        self.cfg = cfg
        self.kin = kin
        self.live = live
        self.estop = threading.Event()
        self.dry_run = not live
        ws = cfg["workspace"]
        if table_z is None:
            if live:
                raise RuntimeError("workspace.table_z_m is not set: measure the table before running on the robot")
            table_z = float(ws["sim_table_z_m"])
        self.table_z = float(table_z)
        self.floor_z = self.table_z + float(ws["table_margin_m"])
        self.box_min = np.asarray(ws["box_min_m"], float)
        self.box_max = np.asarray(ws["box_max_m"], float)
        if np.any(self.box_min >= self.box_max):
            raise ValueError("workspace box_min must be below box_max on every axis")
        self.contacts_baseline = None
        self.loco = dict(cfg.get("locomotion") or {})
        self.walked_m, self.turned_rad = 0.0, 0.0            # per-episode budget

    def vet_walk(self, dx, dy, dyaw):
        """A whole-body step (body frame, m, m, rad) -> velocities and duration for the loco service, or a refusal.
        Order: e-stop -> enabled -> per-command caps -> episode budget -> speeds."""
        lo = self.loco
        if self.estop.is_set():
            return WalkVerdict(False, "ESTOP: nothing moves")
        if not lo.get("enabled", False):
            return WalkVerdict(False, "walking is not enabled for this session: the robot cannot move its body; use the arm only")
        notes = []
        cap_m, cap_yaw = float(lo["param_max_walk_m"]), math.radians(float(lo["param_max_turn_deg"]))
        dist = math.hypot(dx, dy)
        if dist > cap_m:
            dx, dy = dx * cap_m / dist, dy * cap_m / dist; notes.append(f"walk capped to {cap_m * 100:.0f} cm"); dist = cap_m
        if abs(dyaw) > cap_yaw:
            dyaw = math.copysign(cap_yaw, dyaw); notes.append(f"turn capped to {math.degrees(cap_yaw):.0f} deg")
        if self.walked_m + dist > float(lo["max_total_m"]):
            return WalkVerdict(False, f"walking budget for this episode is used up ({float(lo['max_total_m']):.1f} m)", clamped=notes)
        if self.turned_rad + abs(dyaw) > math.radians(float(lo["max_total_turn_deg"])):
            return WalkVerdict(False, f"turning budget for this episode is used up ({float(lo['max_total_turn_deg']):.0f} deg)", clamped=notes)
        v, w = float(lo["speed_mps"]), float(lo["turn_speed_rps"])
        duration = max(dist / v if dist > 0 else 0.0, abs(dyaw) / w if dyaw else 0.0, 0.3)
        vx, vy, vyaw = dx / duration, dy / duration, dyaw / duration
        self.walked_m += dist; self.turned_rad += abs(dyaw)
        return WalkVerdict(True, "", dx, dy, dyaw, vx, vy, vyaw, duration, notes)

    def set_baseline(self, q5, others=None):
        self.contacts_baseline = self.kin.contacts(q5, others)

    # -- 2, 3: setpoint clamping (pure geometry) ---------------------------------------------------
    def clamp_setpoint(self, p_from, roll_from, p_to, roll_to, mode="unit"):
        st = self.cfg["steps"]
        cap_t = float(st["max_translation_m"] if mode == "unit" else st["param_max_translation_m"])
        cap_r = math.radians(float(st["max_rotation_deg"] if mode == "unit" else st["param_max_rotation_deg"]))
        notes = []
        p_from, p_to = np.asarray(p_from, float), np.asarray(p_to, float)
        # box and floor first, then the per-command cap: a target pulled back into the box can never
        # turn into a longer move than the cap allows
        boxed = np.clip(p_to, self.box_min, self.box_max)
        if np.any(np.abs(boxed - p_to) > 1e-9):
            axes = "".join(a for a, b, c in zip("xyz", boxed, p_to) if abs(b - c) > 1e-9)
            notes.append(f"hand kept inside the workspace box ({axes})")
            p_to = boxed
        if p_to[2] < self.floor_z:
            notes.append(f"hand kept {self.floor_z * 100 - self.table_z * 100:.0f} mm above the table")
            p_to = p_to.copy(); p_to[2] = self.floor_z
        d = p_to - p_from
        n = float(np.linalg.norm(d))
        if n > cap_t + 1e-9:
            p_to = p_from + d * (cap_t / n)
            notes.append(f"translation capped from {n * 100:.1f} to {cap_t * 100:.1f} cm")
        dr = roll_to - roll_from
        if abs(dr) > cap_r + 1e-9:
            roll_to = roll_from + math.copysign(cap_r, dr)
            notes.append(f"rotation capped to {math.degrees(cap_r):.0f} deg")
        return p_to, float(roll_to), notes

    # -- the gate -----------------------------------------------------------------------------------
    def vet(self, p_from, roll_from, p_to, roll_to, q_now, others=None, mode="unit"):
        """q_now: 5 measured arm joints. Returns a Verdict; only ok verdicts carry q_target."""
        if self.estop.is_set():
            return Verdict(False, "ESTOP: e-stop is set, nothing moves")
        lim = self.cfg["limits"]
        p_c, roll_c, notes = self.clamp_setpoint(p_from, roll_from, p_to, roll_to, mode)
        q_now = np.asarray(q_now, float)
        res = self.kin.ik(p_c, roll_c, q_now, others, iters=int(lim["ik_iters"]), tol=float(lim["ik_tol_m"]))
        if not res.ok:
            return Verdict(False, f"IK_FAIL: {res.reason}", p_c, roll_c, None, 0.0, notes, res.err_m)
        bad = self.kin.joint_violations(res.q)
        if bad:
            return Verdict(False, f"LIMITS: {', '.join(bad)} outside the joint range", p_c, roll_c, None, 0.0, notes, res.err_m)
        dq = np.abs(res.q - q_now)
        vmax = float(lim["max_joint_vel_rad_s"])
        # cosine-eased interpolation peaks at pi/2 times the mean speed
        duration = max(float(lim["min_move_s"]), float(dq.max()) / vmax * math.pi / 2)
        if self.contacts_baseline is not None and self.kin.contacts(res.q, others) > self.contacts_baseline:
            return Verdict(False, "COLLISION: the model shows the arm touching something at the target", p_c, roll_c, None, 0.0, notes, res.err_m)
        return Verdict(True, "", p_c, roll_c, res.q, duration, notes, res.err_m)

    def check_trajectory(self, frames, dt):
        """Independent re-check of an interpolated joint trajectory (list of 5-vectors): limits and speed."""
        vmax = float(self.cfg["limits"]["max_joint_vel_rad_s"])
        for i, q in enumerate(frames):
            bad = self.kin.joint_violations(q)
            if bad:
                return f"frame {i}: {bad} outside limits"
            if i:
                v = float(np.abs(np.asarray(q) - np.asarray(frames[i - 1])).max() / dt)
                if v > vmax * 1.05:
                    return f"frame {i}: {v:.2f} rad/s over the {vmax} rad/s cap"
        return ""
