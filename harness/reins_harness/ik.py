"""Position IK for one R1 arm, from wherever the robot thinks it is standing.

The same least-squares approach as `sim/plan_pick.py`, on a scratch copy of the
world so solving never moves the robot. Targets and results are in the odom
frame: the scratch copy puts the base at the odometry estimate, so arm plans
line up with what the robot perceived, drift and all. Solves the four joints that place the
hand (shoulder pitch, roll, yaw and elbow); the wrist roll stays where it is.
"""
from __future__ import annotations

import math

import mujoco
from reins_loco.sim import walk_preview
import numpy as np
from scipy.optimize import least_squares

from .world import SimWorld, arm_joint_names

IK_JOINTS = 4  # of ARM_JOINTS: shoulder pitch, roll, yaw, elbow
TOLERANCE = 0.02  # m
MAX_JOINT_VEL = 0.8  # rad/s, what generated trajectories aim for
PATH_SPACING = 0.04  # m between Cartesian IK waypoints


class Unreachable(ValueError):
    pass


class ArmIK:
    def __init__(self, world: SimWorld):
        self.world = world
        self.model = world.model
        self.data = mujoco.MjData(self.model)

    def _sync(self) -> None:
        self.data.qpos[:] = self.world.data.qpos
        base = self.world.loco.poser.base
        odom = self.world.odom_pose()
        self.data.qpos[base:base + 2] = (odom.x, odom.y)
        self.data.qpos[base + 3:base + 7] = walk_preview.yaw_quat(odom.yaw)

    def _setup(self, hand: str):
        names = arm_joint_names(hand)[:IK_JOINTS]
        joints = [self.model.joint(n).id for n in names]
        qadr = [int(self.model.jnt_qposadr[j]) for j in joints]
        limits = np.array([self.model.jnt_range[j] for j in joints])
        return names, qadr, limits, self.model.site(hand).id

    def fk(self, hand: str, q: dict[str, float]) -> np.ndarray:
        """Hand position in the odom frame for arm joint values `q` (others as the world has them)."""
        self._sync()
        for name, value in q.items():
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.model.site(hand).id].copy()

    def solve(self, hand: str, target_map, seed: dict[str, float] | None = None) -> dict[str, float]:
        self._sync()
        names, qadr, limits, site = self._setup(hand)
        seed_q = np.array([(seed or self.world.arm_q)[n] for n in names])
        seed_q = np.clip(seed_q, limits[:, 0] + 1e-4, limits[:, 1] - 1e-4)
        target = np.asarray(target_map, float)

        def pos(q):
            self.data.qpos[qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            return self.data.site_xpos[site]

        result = least_squares(lambda q: np.r_[30 * (pos(q) - target), 0.035 * (q - seed_q)], seed_q,
                               bounds=(limits[:, 0] + 1e-5, limits[:, 1] - 1e-5), max_nfev=400)
        error = float(np.linalg.norm(pos(result.x) - target))
        if error > TOLERANCE:
            raise Unreachable(f"{hand} can't reach {np.round(target, 3).tolist()} (odom): "
                              f"closest is {error:.3f} m away")
        return {n: float(v) for n, v in zip(names, result.x)}

    def cartesian_path(self, hand: str, waypoints_map: list, start_q: dict[str, float] | None = None
                       ) -> tuple[list[float], list[dict[str, float]], list[np.ndarray]]:
        """Move the hand along straight lines through `waypoints_map`.

        Returns (times_s, joint rows, hand positions in the odom frame), timed so
        no joint exceeds MAX_JOINT_VEL. Raises Unreachable if any point fails or
        the arm would have to jump between solutions.
        """
        names = arm_joint_names(hand)[:IK_JOINTS]
        q = {n: (start_q or self.world.arm_q)[n] for n in names}
        start = self.fk(hand, q)
        rows, points, times = [q], [start], [0.0]
        prev = start
        for wp in waypoints_map:
            wp = np.asarray(wp, float)
            n = max(1, math.ceil(np.linalg.norm(wp - prev) / PATH_SPACING))
            for i in range(1, n + 1):
                target = prev + (wp - prev) * i / n
                nxt = self.solve(hand, target, seed=q)
                step = max(abs(nxt[k] - q[k]) for k in names)
                if step > 0.6:
                    raise Unreachable(f"{hand} would have to swing {step:.2f} rad between two points "
                                      f"{PATH_SPACING} m apart near {np.round(target, 3).tolist()}")
                times.append(times[-1] + max(step / MAX_JOINT_VEL, 0.05))
                rows.append(nxt)
                points.append(self.fk(hand, nxt))
                q = nxt
            prev = wp
        return times, rows, points

    def joint_path(self, hand: str, goal_q: dict[str, float], start_q: dict[str, float] | None = None,
                   samples: int = 12) -> tuple[list[float], list[dict[str, float]], list[np.ndarray]]:
        """Interpolate in joint space (for going home), with the hand path for preview."""
        names = list(goal_q)
        q0 = {n: (start_q or self.world.arm_q)[n] for n in names}
        span = max(abs(goal_q[n] - q0[n]) for n in names)
        duration = max(span / MAX_JOINT_VEL, 0.3)
        times, rows, points = [], [], []
        for i in range(samples + 1):
            a = i / samples
            row = {n: q0[n] + (goal_q[n] - q0[n]) * a for n in names}
            times.append(duration * a)
            rows.append(row)
            points.append(self.fk(hand, row))
        return times, rows, points
