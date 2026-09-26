"""Offline checks for measured-state trimming; no DDS or robot connection."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_feed import Feed, MJCF, RelayState


class PlanFeedTests(unittest.TestCase):
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
                          "q": {"right_shoulder_pitch_joint": 0.4},
                          "cmd": {"weight": 1, "q": {"right_shoulder_pitch_joint": 0.4}}})
            middle = json.loads(feed.current(state))
            self.assertEqual(middle["progress_source"], "measured_joints")
            self.assertEqual(middle["hands"]["left"], [])
            self.assertGreater(len(middle["hands"]["right"]), 2)
            self.assertLess(len(middle["hands"]["right"]), 101)
            self.assertEqual(middle["hands"]["right"][0],
                             [round(float(v), 4) for v in feed.live_data.site_xpos[feed.site["right"]]])

            state.update({"type": "r1_state", "version": 1,
                          "q": {"right_shoulder_pitch_joint": 0.8},
                          "cmd": {"weight": 1, "q": {"right_shoulder_pitch_joint": 0.8}}})
            done = json.loads(feed.current(state))
            self.assertEqual(done["hands"]["right"], [])
            state.received_at = 0  # relay loss must not restore the full path
            self.assertEqual(json.loads(feed.current(state)), done)

            plan["keyframes"][1]["joint_targets_rad"]["right_shoulder_pitch_joint"] = 0.4
            path.write_text(json.dumps(plan))
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
            new_plan = json.loads(feed.current(state))
            self.assertEqual(len(new_plan["hands"]["right"]), 101)
            self.assertNotIn("progress_source", new_plan)


if __name__ == "__main__":
    unittest.main()
