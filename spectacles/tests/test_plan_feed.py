"""Offline checks for measured-state trimming; no DDS or robot connection."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_feed import DirectRobotState, Feed, MJCF, RelayState


class PlanFeedTests(unittest.TestCase):
    def test_direct_robot_state_uses_fresh_measured_dds_sample(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import twin
        with patch.object(twin, "listen_dds", return_value=[object()]) as subscribe:
            state = DirectRobotState("en6")
        subscribe.assert_called_once_with(state.robot, "en6", 0)
        self.assertFalse(state.fresh())
        state.robot.set_state({"right_shoulder_pitch_joint": 0.3})
        state.robot.set_cmd(1.0, {"right_shoulder_pitch_joint": 0.4})
        self.assertTrue(state.fresh())
        self.assertTrue(state.commanding)
        self.assertEqual(state.q["right_shoulder_pitch_joint"], 0.3)
        state.robot.t -= 1
        self.assertFalse(state.fresh())

    def test_relay_state_rejects_repeated_stale_robot_sample(self):
        state = RelayState()
        sample = {"type": "r1_state", "version": 1, "t": 123.0,
                  "q": {"right_shoulder_pitch_joint": 0.2}, "cmd": {"weight": 1}}
        state.update(sample)
        self.assertTrue(state.fresh())
        state.source_change_at = time.monotonic() - 1
        state.update(sample)  # relay repeats an old lowstate sample
        self.assertFalse(state.fresh())
        state.update({**sample, "t": 124.0})
        self.assertTrue(state.fresh())

    def test_measured_joint_state_shortens_and_replacement_resets(self):
        model = mujoco.MjModel.from_xml_path(str(MJCF))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.json"
            plan = {"schema_version": 1, "name": "test", "duration_s": 2,
                    "keyframes": [
                        {"time_s": 0, "joint_targets_rad": {"right_shoulder_pitch_joint": 0}},
                        {"time_s": 2, "joint_targets_rad": {"right_shoulder_pitch_joint": 0.8}},
                    ]}
            path.write_text(json.dumps(plan))
            feed = Feed(model, path, 101)
            full = json.loads(feed.current())
            self.assertEqual(len(full["hands"]["right"]), 101)
            state = RelayState()
            state.update({"type": "r1_state", "version": 1,
                          "q": {"right_shoulder_pitch_joint": 0}, "cmd": None})
            idle = json.loads(feed.current(state))
            self.assertEqual(len(idle["hands"]["left"]), 1)
            self.assertEqual(len(idle["hands"]["right"]), 101)
            state.update({"type": "r1_state", "version": 1,
                          "q": {"right_shoulder_pitch_joint": 0.4},
                          "cmd": {"weight": 1, "q": {"right_shoulder_pitch_joint": 0.4}}})
            middle = json.loads(feed.current(state))
            self.assertEqual(middle["progress_source"], "measured_joints")
            self.assertEqual(len(middle["hands"]["left"]), 1)
            self.assertGreater(len(middle["hands"]["right"]), 2)
            self.assertLess(len(middle["hands"]["right"]), 101)
            self.assertEqual(middle["hands"]["right"][0],
                             [round(float(v), 4) for v in feed.live_data.site_xpos[feed.site["right"]]])

            state.update({"type": "r1_state", "version": 1,
                          "q": {"right_shoulder_pitch_joint": 0.8},
                          "cmd": {"weight": 1, "q": {"right_shoulder_pitch_joint": 0.8}}})
            done = json.loads(feed.current(state))
            self.assertEqual(len(done["hands"]["right"]), 1)
            state.received_at = 0  # relay loss must not restore the full path
            self.assertEqual(json.loads(feed.current(state)), done)

            plan["keyframes"][1]["joint_targets_rad"]["right_shoulder_pitch_joint"] = 0.4
            path.write_text(json.dumps(plan))
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
            new_plan = json.loads(feed.current(state))
            self.assertEqual(len(new_plan["hands"]["right"]), 101)
            self.assertNotIn("progress_source", new_plan)

    def test_return_leg_does_not_erase_outbound_path(self):
        model = mujoco.MjModel.from_xml_path(str(MJCF))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "return.json"
            path.write_text(json.dumps({
                "schema_version": 1, "name": "out and back", "duration_s": 4,
                "keyframes": [
                    {"time_s": 0, "joint_targets_rad": {"right_shoulder_pitch_joint": 0}},
                    {"time_s": 2, "joint_targets_rad": {"right_shoulder_pitch_joint": 0.8}},
                    {"time_s": 4, "joint_targets_rad": {"right_shoulder_pitch_joint": 0}},
                ],
            }))
            feed = Feed(model, path, 101)
            state = RelayState()

            def measured(angle):
                state.update({"type": "r1_state", "version": 1,
                              "q": {"right_shoulder_pitch_joint": angle},
                              "cmd": {"weight": 1}})
                return json.loads(feed.current(state))

            outbound = measured(0.397)
            self.assertLess(feed.progress, 50)
            self.assertGreater(len(outbound["hands"]["right"]), 50)
            measured(0.8)
            returning = measured(0.4)
            self.assertGreater(feed.progress, 50)
            self.assertGreater(len(returning["hands"]["right"]), 2)


if __name__ == "__main__":
    unittest.main()
