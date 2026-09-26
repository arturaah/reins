"""Inverse kinematics for the R1 arms, in the frame and on the hand tips every preview uses.

Solver: prioritised damped least squares in numpy (no CasADi). Kinematics come
from Pinocchio on the A5 URDF when it is installed (`pin` from PyPI, core only,
no CasADi bindings needed): measured 2.1 us per FK + Jacobian against 13 us for
MuJoCo, 0.06 ms against 0.1 ms per warm-started path point, and IPOPT/fatrop
through CasADi were slower or failed more often (benchmark notes in CLAUDE.md).
MuJoCo on sim/models/r1/R1_fixed_base.xml stays the reference: the URDF chain
is aligned to it at torso_link (the waist-yaw body both models share), the hand
tip is the MuJoCo `{side}_hand_preview` site, and at start-up the two are
cross-checked on random poses. If Pinocchio is missing or they disagree, the
solver falls back to MuJoCo kinematics. Either way a solution lands exactly
where sim/preview.py, spectacles/plan_feed.py and tools/arm_lift.py draw it.

Frame: robot_base = the MuJoCo model's world frame (floor under the pelvis,
pelvis pinned at 0.74 m; x forward, y left, z up), metres and radians.

The A5 arm has five joints on rt/arm_sdk: shoulder pitch/roll/yaw, elbow,
wrist roll. The hand-tip site lies on the wrist-roll axis, so wrist roll moves
neither the tip nor its pointing axis (the site's x axis, along the forearm).
Position is therefore solved with the other four joints, which leaves one
redundant degree of freedom (elbow swivel). That freedom is spent, in the
null space of the position task, on an optional pointing direction and then
on staying close to the seed. Wrist roll keeps its value from `pose`.

    .venv/bin/python core/ik.py --side right --to 0.35 -0.15 0.85
    .venv/bin/python core/ik.py --side right --to 0.30 -0.20 0.80 --to 0.40 -0.12 0.90 \\
        --start sim/plans/arm_lift_dryrun.json --out tools/plans/reach.json
    .venv/bin/python core/ik.py --bench 2000 [--backend mujoco]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "sim/models/r1/R1_fixed_base.xml"
URDF = ROOT / "sim/models/r1/r1_a5.urdf"
ALIGN_BODY = "torso_link"   # driven by waist_yaw_joint in both models; the URDF chain is anchored here
LIMIT_MARGIN = 0.05     # rad; same margin tools/arm_lift.py keeps from every joint limit
MAX_VEL = 0.4           # rad/s peak for generated plans; arm_lift.py refuses above 0.5
SOLVED_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
SIDES = ("left", "right")
AGREE_M = 1e-5          # start-up cross-check tolerance between the backends


@dataclass
class Solution:
    q: dict             # MuJoCo joint name -> angle for the solved joints
    position: np.ndarray
    direction: np.ndarray
    position_error: float       # metres
    direction_error_deg: float | None
    iterations: int
    ok: bool


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _dpinv(J, damping):
    """Damped pseudo-inverse J^T (J J^T + damping^2 I)^-1."""
    return J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(J.shape[0]))


def _arm_names(side):
    return [f"{side}_{j}_joint" for j in SOLVED_JOINTS]


class MujocoKinematics:
    """Hand-tip FK and Jacobians from MuJoCo (reference backend)."""
    name = "mujoco"

    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        self.qadr, self.dofadr, self.site = {}, {}, {}
        for side in SIDES:
            ids = [model.joint(n).id for n in _arm_names(side)]
            self.qadr[side] = np.array([model.jnt_qposadr[i] for i in ids])
            self.dofadr[side] = np.array([model.jnt_dofadr[i] for i in ids])
            self.site[side] = model.site(f"{side}_hand_preview").id
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def set_pose(self, pose):
        self.data.qpos[:] = 0.0
        for name, value in (pose or {}).items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = value

    def eval(self, side, q, direction=True):
        """(tip position, pointing axis, position Jacobian, axis Jacobian); the last two
        axis terms are None when direction=False."""
        m, d, site = self.model, self.data, self.site[side]
        d.qpos[self.qadr[side]] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, self._jacp, self._jacr, site)
        pos = d.site_xpos[site].copy()
        Jp = self._jacp[:, self.dofadr[side]].copy()
        if not direction:
            return pos, None, Jp, None
        axis = d.site_xmat[site].reshape(3, 3)[:, 0].copy()
        # d(axis)/dt = omega x axis = -[axis]_x omega
        Jd = -_skew(axis) @ self._jacr[:, self.dofadr[side]]
        return pos, axis, Jp, Jd


class PinocchioKinematics:
    """Hand-tip FK and Jacobians from Pinocchio on the A5 URDF, expressed in the MuJoCo world.

    Joints the URDF lacks but that sit between the MuJoCo world and torso_link
    (waist roll/pitch) are handled by re-anchoring at torso_link once per pose.
    A pose that sets a joint the URDF lacks further down an arm (MuJoCo's wrist
    pitch/yaw) cannot be represented; set_pose raises and ArmIK uses MuJoCo.
    """
    name = "pinocchio"

    def __init__(self, mj_model, urdf=URDF):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.mj = mj_model
        self.mjd = mujoco.MjData(mj_model)
        self.align_body = mj_model.body(ALIGN_BODY).id
        self.align_frame = self.model.getFrameId(ALIGN_BODY if self.model.existFrame(ALIGN_BODY) else "waist_yaw_link")
        self.urdf_joints = {self.model.names[j]: self.model.joints[j].idx_q for j in range(1, self.model.njoints)}
        self.cols, self.qidx, self.tip = {}, {}, {}
        for side in SIDES:
            idx = [self.model.joints[self.model.getJointId(n)].idx_v for n in _arm_names(side)]
            qdx = [self.urdf_joints[n] for n in _arm_names(side)]
            assert idx == list(range(idx[0], idx[0] + len(idx))) and idx == qdx, "arm joints must be contiguous"
            self.cols[side] = slice(idx[0], idx[0] + len(idx))
            self.qidx[side] = self.cols[side]
            # Hand tip: the MuJoCo site's placement on its body, reproduced on the URDF link of the same name.
            sid = mj_model.site(f"{side}_hand_preview").id
            body = mj_model.body(mj_model.site_bodyid[sid]).name
            quat = mj_model.site_quat[sid]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, quat)
            parent = self.model.getFrameId(body)
            frame = pin.Frame(f"{side}_tip", self.model.frames[parent].parentJoint, parent,
                              self.model.frames[parent].placement * pin.SE3(R.reshape(3, 3), mj_model.site_pos[sid].copy()),
                              pin.FrameType.OP_FRAME)
            self.tip[side] = self.model.addFrame(frame)
            self.data = self.model.createData()
        # MuJoCo joints the URDF cannot express on each arm's chain below torso_link
        self.unsupported = {}
        for side in SIDES:
            b = mj_model.site_bodyid[mj_model.site(f"{side}_hand_preview").id]
            chain = []
            while b != self.align_body and b > 0:
                chain.append(b); b = mj_model.body_parentid[b]
            self.unsupported[side] = {mj_model.joint(j).name for j in range(mj_model.njnt)
                                      if mj_model.jnt_bodyid[j] in chain and mj_model.joint(j).name not in self.urdf_joints}
        self.q = self.pin.neutral(self.model)
        self.R = np.eye(3); self.t = np.zeros(3); self.identity = True
        self.pose = None

    def supports(self, side, pose):
        return not any(abs((pose or {}).get(n, 0.0)) > 1e-12 for n in self.unsupported[side])

    def set_pose(self, pose):
        pose = pose or {}
        if pose == self.pose:
            return      # eval() only ever writes the solved arm's joints, so nothing else changed
        self.pose = dict(pose)
        self.q = self.pin.neutral(self.model)
        for name, value in pose.items():
            if name in self.urdf_joints:
                self.q[self.urdf_joints[name]] = value
        # T_mjworld_urdfroot = T_mj(torso_link) * T_urdf(torso_link)^-1 for this pose
        d = self.mjd
        d.qpos[:] = 0.0
        for name, value in pose.items():
            d.qpos[self.mj.joint(name).qposadr[0]] = value
        mujoco.mj_kinematics(self.mj, d)
        self.pin.framesForwardKinematics(self.model, self.data, self.q)
        anchor = self.data.oMf[self.align_frame]
        R_mj = d.xmat[self.align_body].reshape(3, 3)
        self.R = R_mj @ anchor.rotation.T
        self.t = d.xpos[self.align_body] - self.R @ anchor.translation
        self.identity = bool(np.allclose(self.R, np.eye(3), atol=1e-12))   # true unless waist roll/pitch are set

    def eval(self, side, q, direction=True):
        pin, cols = self.pin, self.cols[side]
        self.q[cols] = q
        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.tip[side], pin.LOCAL_WORLD_ALIGNED)
        oMf = self.data.oMf[self.tip[side]]
        if self.identity:
            pos, Jp = oMf.translation + self.t, J[:3, cols].copy()
        else:
            pos, Jp = self.R @ oMf.translation + self.t, self.R @ J[:3, cols]
        if not direction:
            return pos, None, Jp, None
        # copy: oMf.rotation is a view into Pinocchio's data, overwritten by the next eval
        axis = oMf.rotation[:, 0].copy() if self.identity else self.R @ oMf.rotation[:, 0]
        Jr = J[3:, cols] if self.identity else self.R @ J[3:, cols]
        return pos, axis, Jp, -_skew(axis) @ Jr


def _cross_check(ref, fast, rng, samples=40):
    """Largest hand-tip / Jacobian disagreement between two backends on random poses (waist included)."""
    worst = 0.0
    m = ref.model
    for i in range(samples):
        side = SIDES[i % 2]
        pose = {n: rng.uniform(*m.joint(n).range) for n in
                ["waist_yaw_joint", f"{side}_wrist_roll_joint"] + _arm_names("left" if side == "right" else "right")}
        q = np.array([rng.uniform(*m.joint(n).range) for n in _arm_names(side)])
        ref.set_pose(pose); fast.set_pose(pose)
        a, b = ref.eval(side, q), fast.eval(side, q)
        worst = max(worst, float(np.linalg.norm(a[0] - b[0])), float(np.abs(a[2] - b[2]).max()),
                    float(np.abs(a[1] - b[1]).max()) * 0.1)
    return worst


class ArmIK:
    def __init__(self, model: mujoco.MjModel | None = None, margin: float = LIMIT_MARGIN, backend: str = "auto"):
        """backend: "auto" (Pinocchio if installed and consistent, else MuJoCo), "pinocchio" or "mujoco"."""
        self.model = model or mujoco.MjModel.from_xml_path(str(MJCF))
        self.margin = margin
        self.reference = MujocoKinematics(self.model)
        self.kin = self.reference
        self.fallback_reason = None
        if backend in ("auto", "pinocchio"):
            try:
                fast = PinocchioKinematics(self.model)
                worst = _cross_check(self.reference, fast, np.random.default_rng(0))
                if worst > AGREE_M:
                    raise RuntimeError(f"URDF and MuJoCo models disagree by {worst:.2e}")
                self.kin = fast
            except Exception as exc:      # ImportError, missing URDF, mismatch
                if backend == "pinocchio":
                    raise
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
        elif backend != "mujoco":
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = self.kin.name
        self.arms = {}
        for side in SIDES:
            names = _arm_names(side)
            ids = [self.model.joint(n).id for n in names]
            self.arms[side] = {"names": names,
                               "lo": self.model.jnt_range[ids, 0] + margin,
                               "hi": self.model.jnt_range[ids, 1] - margin}

    def joint_names(self, side):
        return list(self.arms[side]["names"])

    def _kin_for(self, side, pose):
        """The backend for this pose, with the pose applied."""
        kin = self.kin
        if kin is not self.reference and not kin.supports(side, pose):
            kin = self.reference
        kin.set_pose(pose)
        return kin

    def fk(self, side, q, pose=None):
        """Hand-tip position and pointing axis for arm angles q (array or name dict)."""
        arm = self.arms[side]
        if isinstance(q, dict):
            q = [q.get(n, (pose or {}).get(n, 0.0)) for n in arm["names"]]
        pos, axis, _, _ = self._kin_for(side, pose).eval(side, np.asarray(q, float))
        return pos, axis

    @staticmethod
    def _step_position(q, ep, Jp, q_ref, lo, hi, err):
        """Fast path of _step for position-only targets: closed-form 3x3 inverse, few numpy calls.

        Returns None when a joint would cross its limit; the caller then takes the general step.
        """
        lam2 = (1e-3 + 0.02 * min(err, 0.5)) ** 2
        (a, b, c), (_, d, e), (_, _, f) = (Jp @ Jp.T).tolist()
        a += lam2; d += lam2; f += lam2
        # symmetric 3x3 inverse by cofactors
        A, B, C = d * f - e * e, c * e - b * f, b * e - c * d
        det = a * A + b * B + c * C
        D, E, F = a * f - c * c, b * c - a * e, a * d - b * b
        Minv = np.array([[A, B, C], [B, D, E], [C, E, F]]) / det
        pinv = Jp.T @ Minv
        dq = pinv @ ep
        task = float(np.abs(dq).max())
        post = 0.05 * (q_ref - q)
        dq += post - pinv @ (Jp @ post)
        scale = float(np.abs(dq).max())
        if scale > 0.25:
            dq *= 0.25 / scale
        new = q + dq
        if (new < lo).any() or (new > hi).any():
            return None
        return dq, task

    @staticmethod
    def _step(q, ep, ed, Jp, Jd, q_ref, lo, hi, err):
        """One prioritised step: position first, then direction, then posture.

        Joints that would cross a limit are moved onto it and locked, and the
        others re-solve without them (plain clamping would stall the solve).
        """
        n = len(q)
        locked = np.zeros(n, bool)
        target_q = np.zeros(n)
        lam = 1e-3 + 0.02 * min(err, 0.5)   # more damping far away and near singularities
        for _ in range(n):
            free = (~locked).astype(float)
            fixed = np.where(locked, target_q - q, 0.0)
            Jpf = Jp * free
            Jp_pinv = _dpinv(Jpf, lam)
            task = Jp_pinv @ (ep - Jp @ fixed)
            N = np.diag(free) - Jp_pinv @ Jpf
            if ed is not None:
                Jdf = Jd * free
                JdN = Jdf @ N
                JdN_pinv = _dpinv(JdN, 0.05)
                task = task + N @ (JdN_pinv @ (ed - Jd @ fixed - Jdf @ task))
                N = N @ (np.eye(n) - JdN_pinv @ JdN)
            dq = fixed + task + N @ (0.05 * (q_ref - q) * free)
            scale = float(np.max(np.abs(dq)))
            if scale > 0.25:
                dq *= 0.25 / scale
            new = q + dq
            cross = ~locked & ((new < lo) | (new > hi))
            if not cross.any():
                return dq, float(np.max(np.abs(task)))
            locked |= cross
            target_q = np.where(new < lo, lo, np.where(new > hi, hi, target_q))
        return np.clip(q + dq, lo, hi) - q, float(np.max(np.abs(task)))

    def _descend(self, kin, side, arm, q, target, direction, q_ref, tol, max_iter):
        for it in range(1, max_iter + 1):
            pos, axis, Jp, Jd = kin.eval(side, q, direction is not None)
            ep = target - pos
            err = float(np.linalg.norm(ep))
            ed = None if direction is None else direction - axis
            # Posture is only a tie-breaker: stop as soon as the hand is there
            # (and, with a direction, once the direction task has settled).
            if err < tol and it > 1 and (direction is None or task_step < 1e-4):
                break
            fast = None if ed is not None else self._step_position(q, ep, Jp, q_ref, arm["lo"], arm["hi"], err)
            dq, task_step = fast or self._step(q, ep, ed, Jp, Jd, q_ref, arm["lo"], arm["hi"], err)
            q = np.clip(q + dq, arm["lo"], arm["hi"])
        pos, axis, _, _ = kin.eval(side, q)
        return q, pos, axis, it

    def solve(self, side, target, direction=None, seed=None, pose=None, tol=1e-4,
              accept=1e-3, dir_accept_deg=2.0, max_iter=60, restarts=10, max_jump=None, rng=None):
        """Arm angles that put the hand tip at `target` (robot_base, m).

        direction: optional unit vector the forearm/hand should point along (soft).
        seed: starting angles (name dict or array); default is the current `pose`.
        pose: angles for every other joint (other arm, waist, wrist roll), name -> rad.
        The seed is tried first, then random restarts within the limits. The first
        solution within `accept` metres (and, with a direction, within
        dir_accept_deg) wins; otherwise the best one found is returned. With
        max_jump, restarts that land more than that far from the seed in any
        joint are ignored, so paths stay on one branch.
        """
        arm = self.arms[side]
        target = np.asarray(target, float)
        if direction is not None:
            direction = np.asarray(direction, float)
            direction = direction / np.linalg.norm(direction)
        kin = self._kin_for(side, pose)
        if seed is None:
            seed = [(pose or {}).get(nm, 0.0) for nm in arm["names"]]
        elif isinstance(seed, dict):
            seed = [seed[nm] for nm in arm["names"]]
        seed = np.clip(np.asarray(seed, float), arm["lo"], arm["hi"])
        rng = rng or np.random.default_rng(0)
        best, best_key = None, None
        total = 0
        for attempt in range(restarts + 1):
            start = seed if attempt == 0 else rng.uniform(arm["lo"], arm["hi"])
            q, pos, axis, it = self._descend(kin, side, arm, start.copy(), target, direction, seed, tol, max_iter)
            total += it
            if attempt and max_jump is not None and np.max(np.abs(q - seed)) > max_jump:
                continue
            err = float(np.linalg.norm(target - pos))
            derr = None if direction is None else math.degrees(math.acos(float(np.clip(axis @ direction, -1, 1))))
            # Reaching the point beats pointing the right way; ties go to the earlier (seeded) attempt.
            key = (err >= accept, derr or 0.0 if err < accept else err)
            if best_key is None or key < best_key:
                best, best_key = (err, derr, q, pos, axis), key
            if err < accept and (direction is None or derr < dir_accept_deg):
                break
        err, derr, q, pos, axis = best
        if err >= accept and direction is not None:
            # Near limits the soft direction task can hold position off target; position wins.
            sol = self.solve(side, target, None, seed, pose, tol, accept, dir_accept_deg, max_iter, restarts, max_jump, rng)
            if sol.ok:
                sol.direction_error_deg = math.degrees(math.acos(float(np.clip(sol.direction @ direction, -1, 1))))
                sol.iterations += total
                return sol
        return Solution({nm: float(v) for nm, v in zip(arm["names"], q)}, pos, axis, err, derr, total, err < accept)

    def solve_path(self, side, targets, directions=None, pose=None, seed=None, max_jump=0.5, **kw):
        """Solve waypoints in order, each seeded by the previous solution.

        Raises ValueError if a waypoint is unreachable or the arm would have to
        jump more than max_jump rad in one joint between consecutive waypoints
        (a branch flip; add intermediate waypoints or change the targets).
        """
        solutions = []
        prev = seed
        for i, target in enumerate(targets):
            sol = self.solve(side, target, None if directions is None else directions[i], seed=prev, pose=pose,
                             max_jump=max_jump if solutions else None, **kw)
            if not sol.ok:
                raise ValueError(f"waypoint {i} {np.round(target, 3).tolist()} unreachable: "
                                 f"closest {sol.position_error * 1000:.1f} mm away")
            if prev is not None:
                prev_q = prev if isinstance(prev, dict) else dict(zip(self.arms[side]["names"], prev))
                jump = max(abs(sol.q[n] - prev_q[n]) for n in sol.q)
                if solutions and jump > max_jump:
                    raise ValueError(f"waypoint {i}: joints jump {jump:.2f} rad from the previous waypoint")
            solutions.append(sol)
            prev = sol.q
        return solutions


STEP_M = 0.01          # Cartesian spacing of solved points along a straight segment
MAX_GAP_S = 0.1        # keyframe spacing; dense, so arm_lift.py interpolates linearly (its cutoff is 0.25 s)


def _timed_segment(Q, s_param, names, t0, max_vel, min_segment_s):
    """Keyframes every MAX_GAP_S along one cosine-eased segment, joints read off the solved path.

    With the ease s(t) = (1 - cos(pi t / T)) / 2 the arc speed is (pi / T) sqrt(s (1 - s)),
    so fast joint motion near the ends of a segment costs little time. T starts from
    that estimate and is stretched until the keyframes, interpolated linearly as
    arm_lift.py and the previews do, keep every joint at or under max_vel.
    """
    ds = np.diff(s_param)
    slope = np.max(np.abs(np.diff(Q, axis=0)), axis=1) / ds
    mid = s_param[:-1] + ds / 2
    T = max(min_segment_s, float(np.max(math.pi * np.sqrt(mid * (1 - mid)) * slope)) / max_vel)
    for _ in range(20):
        count = max(1, math.ceil(T / MAX_GAP_S))
        rel = np.linspace(0, T, count + 1)
        eased = (1 - np.cos(math.pi * rel / T)) / 2
        qs = np.array([np.interp(eased, s_param, Q[:, k]) for k in range(len(names))]).T
        peak = float(np.max(np.abs(np.diff(qs, axis=0)) / np.diff(rel)[:, None]))
        if peak <= max_vel:
            break
        T *= peak / max_vel * 1.01
    return [{"time_s": round(t0 + float(dt), 4), "joint_targets_rad": {n: round(float(v), 5) for n, v in zip(names, q)}}
            for dt, q in zip(rel[1:], qs[1:])]


def plan_from_waypoints(ik, side, targets, pose=None, directions=None, name="ik plan",
                        max_vel=MAX_VEL, min_segment_s=0.6, hold_s=0.0):
    """A sim-contract plan (schema_version 1) that moves `side`'s hand through targets.

    The hand travels in straight lines: from where `pose` puts it to the first
    target, then target to target. Each line is solved every STEP_M, seeded
    from the previous point, and timed with one cosine ease so every joint
    stays under max_vel. The keyframes come out dense (gaps <= MAX_GAP_S), so
    tools/arm_lift.py and the previews interpolate them linearly. The first
    keyframe is the start pose; arm_lift.py replaces it with the measured pose,
    so start from a dry-run file (pose_from_plan) when planning for the robot.
    """
    pose = dict(pose or {})
    names = ik.joint_names(side)
    q_prev = {n: float(pose.get(n, 0.0)) for n in names}
    p_prev, _ = ik.fk(side, q_prev, pose)
    keyframes = [{"time_s": 0.0, "joint_targets_rad": {n: round(q_prev[n], 5) for n in names}}]
    t, sols = 0.0, []
    for i, target in enumerate(targets):
        target = np.asarray(target, float)
        direction = None if directions is None else directions[i]
        dist = float(np.linalg.norm(target - p_prev))
        steps = max(1, math.ceil(dist / STEP_M))
        fractions = np.linspace(0, 1, steps + 1)[1:]
        line = [p_prev + (target - p_prev) * f for f in fractions]
        path = ik.solve_path(side, line, None if direction is None else [direction] * len(line),
                             pose=pose, seed=q_prev, max_jump=0.25)
        Q = np.array([[q_prev[n] for n in names]] + [[s.q[n] for n in names] for s in path])
        s_param = np.r_[0.0, fractions]
        keyframes.extend(_timed_segment(Q, s_param, names, t, max_vel, min_segment_s))
        times = [keyframes[-1]["time_s"]]
        t = float(times[-1])
        if hold_s:
            t += hold_s
            keyframes.append({"time_s": round(t, 4), "joint_targets_rad": dict(keyframes[-1]["joint_targets_rad"])})
        sols.append(path[-1])
        q_prev, p_prev = path[-1].q, path[-1].position
    held = {k: v for k, v in pose.items() if k not in names}
    plan = {"schema_version": 1, "name": name, "duration_s": keyframes[-1]["time_s"], "keyframes": keyframes,
            "ik": {"side": side, "frame": "robot_base", "targets_m": [np.round(x, 4).tolist() for x in targets],
                   "position_error_mm": [round(s.position_error * 1000, 3) for s in sols]}}
    if held:
        plan["held_joints_rad"] = held
    return plan, sols


def pose_from_plan(path):
    """Start pose from a plan: its held joints plus its first keyframe (a dry-run file gives the measured pose)."""
    plan = json.loads(Path(path).read_text())
    first = min(plan["keyframes"], key=lambda f: f["time_s"])
    return {**plan.get("held_joints_rad", {}), **first["joint_targets_rad"]}


def bench(ik, n, rng):
    """Targets from random in-limit configurations, solved from the zero pose."""
    times, ok, errs = [], 0, []
    for i in range(n):
        side = SIDES[i % 2]
        arm = ik.arms[side]
        target, _ = ik.fk(side, rng.uniform(arm["lo"], arm["hi"]))
        t0 = time.perf_counter()
        sol = ik.solve(side, target, rng=rng)
        times.append(time.perf_counter() - t0)
        ok += sol.ok
        errs.append(sol.position_error)
    times = np.array(times) * 1000
    print(f"{n} random reachable targets: {ok / n:.1%} solved within 1 mm; "
          f"time median {np.median(times):.2f} ms, p95 {np.percentile(times, 95):.2f} ms, max {times.max():.1f} ms; "
          f"median error {np.median(errs) * 1e6:.1f} um")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=SIDES, default="right")
    ap.add_argument("--to", nargs=3, type=float, action="append", metavar=("X", "Y", "Z"),
                    help="hand-tip target in robot_base metres; repeat for a path")
    ap.add_argument("--dir", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
                    help="preferred pointing direction for every target (soft)")
    ap.add_argument("--start", help="plan whose held joints + first keyframe give the start pose "
                                    "(e.g. sim/plans/arm_lift_dryrun.json for the measured pose)")
    ap.add_argument("--out", help="write a plan (sim contract) for sim/preview.py and tools/arm_lift.py")
    ap.add_argument("--name", default=None)
    ap.add_argument("--hold-s", type=float, default=0.0, help="pause at each target")
    ap.add_argument("--bench", type=int, metavar="N", help="time N random solves and exit")
    ap.add_argument("--backend", choices=("auto", "pinocchio", "mujoco"), default="auto",
                    help="kinematics backend (auto: Pinocchio if installed and consistent with MuJoCo)")
    a = ap.parse_args()
    ik = ArmIK(backend=a.backend)
    print(f"kinematics: {ik.backend}" + (f" (Pinocchio unavailable: {ik.fallback_reason})" if ik.fallback_reason else ""),
          file=sys.stderr)
    if a.bench:
        bench(ik, a.bench, np.random.default_rng(1))
        return
    if not a.to:
        ap.error("give --to X Y Z (or --bench N)")
    pose = pose_from_plan(a.start) if a.start else {}
    dirs = [a.dir] * len(a.to) if a.dir else None
    try:
        plan, sols = plan_from_waypoints(ik, a.side, a.to, pose, dirs, name=a.name or f"ik {a.side} reach", hold_s=a.hold_s)
    except ValueError as exc:
        raise SystemExit(f"IK failed: {exc}")
    for target, s in zip(a.to, sols):
        d = f", direction off by {s.direction_error_deg:.1f} deg" if s.direction_error_deg is not None else ""
        print(f"target {target}: error {s.position_error * 1000:.3f} mm{d}; " +
              ", ".join(f"{n.replace('_joint', '')} {v:+.3f}" for n, v in s.q.items()))
    print(f"plan: {len(plan['keyframes'])} keyframes over {plan['duration_s']:.1f} s")
    if a.out:
        Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
        print(f"wrote {a.out}; preview: .venv/bin/python sim/preview.py --plan {a.out} --preview-only")


if __name__ == "__main__":
    main()
