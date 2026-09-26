"""Run with .venv/bin/python -m unittest core.test_ik."""
import math
import unittest

import mujoco
import numpy as np

from core.ik import MAX_GAP_S, MAX_VEL, SIDES, ArmIK, plan_from_waypoints
from sim.preview import load_plan
from spectacles.plan_feed import hand_paths


class IKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ik = ArmIK()

    def random_q(self, side, rng):
        arm = self.ik.arms[side]
        return rng.uniform(arm["lo"], arm["hi"])

    def test_default_backend_is_pinocchio_when_installed(self):
        try:
            import pinocchio  # noqa: F401
        except ImportError:
            self.skipTest("pinocchio not installed; ArmIK uses MuJoCo")
        self.assertEqual(self.ik.backend, "pinocchio", self.ik.fallback_reason)

    def test_jacobians_match_finite_differences_on_both_backends(self):
        rng = np.random.default_rng(3)
        for backend in ("mujoco", self.ik.backend):
            kin = ArmIK(backend=backend).kin
            for side in SIDES:
                pose = {"waist_yaw_joint": rng.uniform(-1, 1), f"{side}_wrist_roll_joint": rng.uniform(-1, 1)}
                q = self.random_q(side, rng)
                kin.set_pose(pose)
                _, _, Jp, Jd = kin.eval(side, q)
                h = 1e-6
                for k in range(len(q)):
                    dq = np.zeros(len(q)); dq[k] = h
                    p1, a1, _, _ = kin.eval(side, q + dq)
                    p0, a0, _, _ = kin.eval(side, q - dq)
                    np.testing.assert_allclose(Jp[:, k], (p1 - p0) / (2 * h), atol=1e-6)
                    np.testing.assert_allclose(Jd[:, k], (a1 - a0) / (2 * h), atol=1e-6)
                pos, axis, Jp2, Jd2 = kin.eval(side, q, direction=False)
                self.assertIsNone(axis); self.assertIsNone(Jd2)
                np.testing.assert_allclose(Jp2, Jp)

    def test_backends_agree_including_waist_roll(self):
        rng = np.random.default_rng(4)
        fast, ref = self.ik.kin, self.ik.reference
        for i in range(100):
            side = SIDES[i % 2]
            other = "left" if side == "right" else "right"
            # waist roll exists only in the MuJoCo model; the Pinocchio backend re-anchors for it
            pose = {"waist_yaw_joint": rng.uniform(-1, 1), "waist_roll_joint": rng.uniform(-0.4, 0.4),
                    f"{other}_elbow_joint": rng.uniform(0, 1.5)}
            q = self.random_q(side, rng)
            fast.set_pose(pose); ref.set_pose(pose)
            a, b = fast.eval(side, q), ref.eval(side, q)
            np.testing.assert_allclose(a[0], b[0], atol=1e-5)
            np.testing.assert_allclose(a[1], b[1], atol=1e-5)
            np.testing.assert_allclose(a[2], b[2], atol=1e-5)
            np.testing.assert_allclose(a[3], b[3], atol=1e-5)

    def test_pose_the_urdf_cannot_express_falls_back_to_mujoco(self):
        # MuJoCo's right wrist pitch has no URDF counterpart (the A5 hardware lacks it)
        pose = {"right_wrist_pitch_joint": 0.3}
        target = [0.35, -0.15, 0.85]
        sol = self.ik.solve("right", target, pose=pose)
        self.assertTrue(sol.ok)
        ref = ArmIK(backend="mujoco")
        np.testing.assert_allclose(ref.fk("right", sol.q, pose)[0], target, atol=1e-3)

    def test_solutions_match_across_backends(self):
        ref = ArmIK(backend="mujoco")
        rng = np.random.default_rng(9)
        for i in range(30):
            side = SIDES[i % 2]
            target, _ = ref.fk(side, self.random_q(side, rng))
            a, b = self.ik.solve(side, target), ref.solve(side, target)
            self.assertTrue(a.ok and b.ok)
            # the same algorithm on (numerically) the same kinematics lands in the same place
            for n in a.q:
                self.assertAlmostEqual(a.q[n], b.q[n], places=3)

    def test_reaches_random_reachable_targets_within_limits(self):
        rng = np.random.default_rng(7)
        for i in range(300):
            side = SIDES[i % 2]
            arm = self.ik.arms[side]
            target, _ = self.ik.fk(side, self.random_q(side, rng))
            sol = self.ik.solve(side, target, rng=np.random.default_rng(i))
            self.assertTrue(sol.ok, f"{side} {target}")
            self.assertLess(sol.position_error, 1e-3)
            q = np.array(list(sol.q.values()))
            self.assertTrue(np.all(q >= arm["lo"] - 1e-9) and np.all(q <= arm["hi"] + 1e-9))
            np.testing.assert_allclose(self.ik.fk(side, sol.q)[0], sol.position, atol=1e-9)

    def test_direction_is_soft_but_usually_met(self):
        # Direction is a secondary goal: position is always reached; the direction
        # is matched closely in most cases and traded away near joint limits.
        rng = np.random.default_rng(11)
        errors = []
        for i in range(200):
            side = SIDES[i % 2]
            target, axis = self.ik.fk(side, self.random_q(side, rng))
            sol = self.ik.solve(side, target, direction=axis, rng=np.random.default_rng(i))
            self.assertTrue(sol.ok)
            errors.append(sol.direction_error_deg)
        self.assertLess(np.median(errors), 0.5)
        self.assertGreaterEqual(np.mean(np.array(errors) < 3.0), 0.9)

    def test_unreachable_target_reports_closest(self):
        sol = self.ik.solve("right", [2.0, 0.0, 1.0], restarts=2)
        self.assertFalse(sol.ok)
        self.assertGreater(sol.position_error, 1.0)

    def test_other_joints_come_from_pose(self):
        target = [0.35, -0.15, 0.85]
        rolled = self.ik.solve("right", target, pose={"waist_yaw_joint": 0.3})
        self.assertTrue(rolled.ok)
        np.testing.assert_allclose(self.ik.fk("right", rolled.q, pose={"waist_yaw_joint": 0.3})[0], target, atol=1e-3)
        self.assertNotAlmostEqual(rolled.q["right_shoulder_yaw_joint"], self.ik.solve("right", target).q["right_shoulder_yaw_joint"], places=2)

    def test_plan_moves_in_straight_lines_within_speed_and_matches_previews(self):
        start = {"right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 1.2,
                 "left_elbow_joint": 0.5, "right_wrist_roll_joint": 0.1}
        targets = [np.array([0.28, -0.22, 0.78]), np.array([0.40, -0.10, 0.92])]
        plan, sols = plan_from_waypoints(self.ik, "right", targets, pose=start, name="test reach", hold_s=0.5)
        model = mujoco.MjModel.from_xml_path("sim/models/r1/R1_fixed_base.xml")
        load_plan(model, _tmp_plan(plan))               # the preview's own validation, incl. joint ranges
        self.assertEqual(plan["held_joints_rad"], {"left_elbow_joint": 0.5, "right_wrist_roll_joint": 0.1})
        names = self.ik.joint_names("right")
        times = np.array([f["time_s"] for f in plan["keyframes"]])
        qs = np.array([[f["joint_targets_rad"][n] for n in names] for f in plan["keyframes"]])
        gaps = np.diff(times)
        moving = np.abs(np.diff(qs, axis=0)).max(axis=1) > 1e-9
        # dense enough that arm_lift.py interpolates linearly (median gap < 0.25 s) ...
        self.assertLess(np.median(gaps), 0.25)
        self.assertLessEqual(gaps[moving].max(), MAX_GAP_S + 1e-3)
        # ... and with linear interpolation no joint exceeds MAX_VEL
        self.assertLessEqual((np.abs(np.diff(qs, axis=0)) / gaps[:, None]).max(), MAX_VEL + 1e-3)
        # every keyframe's hand position lies on the straight segments
        p0, _ = self.ik.fk("right", start, start)
        segments = [(p0, targets[0]), (targets[0], targets[1])]
        def off_line(p):
            return min(np.linalg.norm(np.cross(b - a, p - a)) / np.linalg.norm(b - a) for a, b in segments)
        for f in plan["keyframes"]:
            p, _ = self.ik.fk("right", f["joint_targets_rad"], start)
            self.assertLess(off_line(p), 0.003)
        # the Spectacles feed and the preview draw the hand through the targets
        drawn = np.array(hand_paths(model, plan, 400)["right"])
        for target in targets:
            self.assertLess(np.linalg.norm(drawn - target, axis=1).min(), 0.005)
        self.assertTrue(all(math.isfinite(e) and e < 1.0 for e in plan["ik"]["position_error_mm"]))

def _tmp_plan(plan):
    import json, tempfile
    from pathlib import Path
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(plan, f)
    f.close()
    return Path(f.name)


if __name__ == "__main__":
    unittest.main()
