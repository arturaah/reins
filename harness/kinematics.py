"""Forward and inverse kinematics of one R1 A5 arm on the MuJoCo model. No SDK, no physics.

The end effector is the hand-preview site: the wrist roll link plus 0.13 m along its x axis
(sim/models/r1/R1_fixed_base.xml). Wrist roll spins about that axis, so it does not move the
site: position IK runs over the four proximal joints and the roll is passed through.
IK is damped least squares with joint limits (minus a margin) enforced at every iteration and a
weak pull toward the seed pose, so a small Cartesian step gives a small joint step.
"""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = {side: [f"{side}_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
              for side in ("left", "right")}
OTHER_JOINTS = ["waist_roll_joint", "waist_yaw_joint"]      # posed from the robot; they move the shoulders


@dataclass
class IKResult:
    q: np.ndarray            # 5 joint targets (rad), the last one is the requested roll
    err_m: float             # remaining position error
    ok: bool
    iterations: int
    reason: str = ""


class ArmKinematics:
    def __init__(self, model_path, arm, margin_rad=0.05):
        path = Path(model_path)
        if not path.is_absolute():
            path = ROOT / path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.arm = arm
        self.joint_names = ARM_JOINTS[arm]
        self.jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        assert min(self.jid) >= 0, f"missing arm joints for {arm}"
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jid])
        self.dofadr = np.array([self.model.jnt_dofadr[j] for j in self.jid])
        self.limits = np.array([self.model.jnt_range[j] for j in self.jid])        # (5, 2)
        self.margin = margin_rad
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_hand_preview")
        assert self.site >= 0
        self.other_adr = {n: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                          for n in (self.model.joint(i).name for i in range(self.model.njnt))
                          if n and n not in self.joint_names}
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # -- posing --------------------------------------------------------------------------------
    def _pose(self, q5, others=None):
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qadr] = q5
        for n, v in (others or {}).items():
            if n in self.other_adr:
                self.data.qpos[self.other_adr[n]] = v
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)      # mj_jacSite needs subtree COM positions

    def fk(self, q5, others=None):
        """-> (site position (3,), site rotation (3,3)) in the base frame."""
        self._pose(np.asarray(q5, float), others)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def q_from_dict(self, joints):
        return np.array([float(joints[n]) for n in self.joint_names])

    def q_to_dict(self, q5):
        return {n: float(v) for n, v in zip(self.joint_names, q5)}

    def within_limits(self, q5, margin=None):
        m = self.margin if margin is None else margin
        lo, hi = self.limits[:, 0] + m, self.limits[:, 1] - m
        return [n for n, v, a, b in zip(self.joint_names, q5, lo, hi) if not (a <= v <= b)]

    def clamp(self, q5):
        return np.clip(q5, self.limits[:, 0] + self.margin, self.limits[:, 1] - self.margin)

    # -- inverse kinematics ----------------------------------------------------------------------
    def ik(self, p_target, roll, q_seed, others=None, iters=200, tol=0.005, damping=0.05, max_step=0.15):
        p_target = np.asarray(p_target, float)
        q = self.clamp(np.array(q_seed, float))
        q[4] = float(np.clip(roll, self.limits[4, 0] + self.margin, self.limits[4, 1] - self.margin))
        seed = q.copy()
        pos_dofs = self.dofadr[:4]
        err = np.inf
        for it in range(1, iters + 1):
            p, _ = self.fk(q, others)
            e = p_target - p
            err = float(np.linalg.norm(e))
            if err < tol * 0.2:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
            J = self._jacp[:, pos_dofs]                                   # 3 x 4
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, e)
            # weak pull toward the seed in the nullspace: keeps the elbow where it was
            N = np.eye(4) - np.linalg.pinv(J) @ J
            dq += 0.1 * (N @ (seed[:4] - q[:4]))
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q[:4] = self.clamp(np.r_[q[:4] + dq, q[4]])[:4]
        p, _ = self.fk(q, others)
        err = float(np.linalg.norm(p_target - p))
        ok = err <= tol
        return IKResult(q, err, ok, it, "" if ok else f"IK stopped {err * 100:.1f} cm short of the target")

    def joint_violations(self, q5):
        """Names of joints outside [lo+margin, hi-margin]; empty when fine."""
        return self.within_limits(q5)

    def contacts(self, q5, others=None):
        """Number of contacts in the model at this pose (self-collision proxy; needs mj_forward)."""
        self._pose(np.asarray(q5, float), others)
        mujoco.mj_forward(self.model, self.data)
        return int(self.data.ncon)
