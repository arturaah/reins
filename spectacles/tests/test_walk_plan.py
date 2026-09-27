"""Room-fixed hand paths for a planned walk followed by an arm action."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_walk_plan import combine
from plan_feed import BasePoseFile, Feed, MJCF, RelayState, hand_paths


class WalkPlanTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(MJCF))
        self.arm = {"schema_version": 1, "name": "reach", "duration_s": 2,
                    "keyframes": [
                        {"time_s": 0, "joint_targets_rad": {"right_shoulder_pitch_joint": 0}},
                        {"time_s": 2, "joint_targets_rad": {"right_shoulder_pitch_joint": 0.5}},
                    ]}

    def test_straight_walk_keeps_hand_at_initial_height_then_reaches(self):
        plan = combine(self.arm, 1.0, 3.0)
        self.assertEqual(plan["duration_s"], 5.0)
        self.assertEqual([f["time_s"] for f in plan["keyframes"]], [0, 3, 5])
        paths = hand_paths(self.model, plan, 101)["right"]
        self.assertAlmostEqual(paths[60][0] - paths[0][0], 1.0, places=3)
        self.assertAlmostEqual(paths[60][2], paths[0][2], places=3)
        self.assertNotAlmostEqual(paths[-1][2], paths[60][2], places=2)

    def test_yaw_rotates_hand_path_in_fixed_map(self):
        plan = combine(self.arm, 1.0, 3.0)
        plan["base_keyframes"][1]["yaw_rad"] = 1.5707963267948966
        plan["base_keyframes"][2]["yaw_rad"] = 1.5707963267948966
        robot = hand_paths(self.model, self.arm, 2)["right"][0]
        mapped = hand_paths(self.model, plan, 101)["right"][60]
        self.assertAlmostEqual(mapped[0], 1.0 - robot[1], places=3)
        self.assertAlmostEqual(mapped[1], robot[0], places=3)

    def test_map_path_does_not_advance_without_measured_base_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "walk.json"
            path.write_text(json.dumps(combine(self.arm, 1.0, 3.0)))
            feed = Feed(self.model, path, 101)
            full = json.loads(feed.current())
            self.assertEqual(full["frame"], "map")
            state = RelayState()
            state.update({"type": "r1_state", "version": 1,
                          "q": {"right_shoulder_pitch_joint": 0.5},
                          "cmd": {"weight": 1}})
            self.assertEqual(json.loads(feed.current(state)), full)

    def test_measured_base_pose_shortens_walk_then_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "walk.json"
            path.write_text(json.dumps(combine(self.arm, 1.0, 3.0)))
            feed = Feed(self.model, path, 101)
            state = RelayState()

            def update(x, joint, commanding=False):
                state.update({"type": "r1_state", "version": 1,
                              "q": {"right_shoulder_pitch_joint": joint},
                              "cmd": {"weight": 1 if commanding else 0}})
                return json.loads(feed.current(state, {"x_m": x, "y_m": 0,
                                                        "yaw_rad": 0}))

            before = update(0, 0)
            self.assertEqual(len(before["hands"]["right"]), 101)
            midway = update(0.5, 0)
            self.assertLess(len(midway["hands"]["right"]), 101)
            self.assertGreater(len(midway["hands"]["right"]), 40)
            reach = update(1.0, 0.25, True)
            self.assertLess(len(reach["hands"]["right"]), 45)
            complete = update(1.0, 0.5, True)
            self.assertEqual(len(complete["hands"]["right"]), 1)
            self.assertEqual(len(complete["hands"]["left"]), 1)

    def test_base_pose_file_must_be_fresh_and_in_map_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pose.json"
            source = BasePoseFile(path)
            self.assertIsNone(source.read())
            path.write_text(json.dumps({"frame": "map", "x_m": 0.2,
                                        "y_m": 0, "yaw_rad": 0}))
            self.assertEqual(source.read()["x_m"], 0.2)
            old = time.time() - 5
            import os
            os.utime(path, (old, old))
            self.assertIsNone(source.read())

    def test_base_keyframes_require_initial_origin(self):
        plan = combine(self.arm, 1.0, 3.0)
        plan["base_keyframes"][0]["x_m"] = 0.2
        with self.assertRaisesRegex(ValueError, "map origin"):
            hand_paths(self.model, plan, 101)


if __name__ == "__main__":
    unittest.main()
