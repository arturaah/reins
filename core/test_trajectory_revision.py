"""Trajectory replanning uses real IK/collision checks and fake model replies."""
import copy
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.codex_chat import CodexResponder, ToolLink
from core.claude_chat import ClaudeResponder
from core.dashboard_chat import DashboardChat, TrajectoryReviser
from core.ik import ArmIK
from core.perception import PerceptionError
from core.prompt_planner import PromptPlanner
from core.test_generated_motion import DRAFT, ANSWER

UNREACHABLE = {**DRAFT, 'waypoints': [{'position_m': [1.5, -.2, 1.0], 'hold_s': 0}]}
COLLIDING = {**DRAFT, 'waypoints': [DRAFT['waypoints'][0],
             {'position_m': [.10, -.12, 1.28], 'hold_s': 0}, DRAFT['waypoints'][-1]]}


class RevisionTests(unittest.TestCase):
    def wait(self, planner):
        deadline = time.monotonic() + 15
        while planner.status()['state'] == 'planning' and time.monotonic() < deadline:
            time.sleep(.01)
        status = planner.status()
        self.assertNotEqual(status['state'], 'planning', status)
        return status

    def test_ik_then_collision_failures_recalculate_and_validate(self):
        replies = iter([COLLIDING, DRAFT])
        calls = []
        def provider(messages, context):
            calls.append((messages, context))
            return {**ANSWER, 'trajectory': next(replies)}
        with tempfile.TemporaryDirectory() as tmp:
            planner = PromptPlanner(output_dir=tmp)
            planner.submit(ANSWER['robot_request'], trajectory=UNREACHABLE,
                           reviser=TrajectoryReviser(lambda: provider))
            status = self.wait(planner)
            self.assertEqual(status['state'], 'proposed', status['message'])
            self.assertEqual(status['attempt'], 3)
            self.assertEqual(status['revision'], 3)
            self.assertEqual(len(calls), 2)
            first = calls[0][1]['trajectory_revision']['failures'][0]
            self.assertEqual(first['stage'], 'plan')
            self.assertRegex(first['error'], 'unreachable|joints jump')
            self.assertEqual(first['details']['waypoint_number'], 1)
            self.assertEqual(first['details']['target_position_m'], [1.5, -.2, 1.0])
            failures = calls[1][1]['trajectory_revision']['failures']
            self.assertEqual(len(failures), 2)
            self.assertEqual(failures[-1]['error'], 'Self-collision: head envelope')
            self.assertEqual(failures[-1]['stage'], 'validate')
            self.assertIn('hand_position_m', failures[-1]['details'])
            self.assertEqual(calls[1][0][1]['trajectory'], COLLIDING)
            self.assertEqual(calls[1][1]['motion_authoring'], calls[0][1]['motion_authoring'])
            self.assertEqual(calls[1][1]['trajectory_revision']['original_request'], ANSWER['robot_request'])
            self.assertEqual(planner.plan['generated_trajectory'], DRAFT)
            self.assertEqual(planner.plan['prompt_proposal']['revision'], 3)
            self.assertGreater(status['validation']['samples'], 100)
            self.assertLessEqual(status['validation']['max_velocity_rad_s'], .4)
            self.assertLessEqual(status['validation']['max_sampled_acceleration_rad_s2'], 1.5)
            self.assertFalse(status['execution_allowed'])
            self.assertTrue(any(e['stage'] == 'revise' for e in status['events']))
            shown = Mock()
            planner.show_once(status['id'], shown)
            shown.assert_called_once()
            with self.assertRaises(ValueError):
                planner.show_once(status['id'], shown)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_repeated_failure_stops_after_three_attempts(self):
        repair = Mock(return_value=UNREACHABLE)
        planner = PromptPlanner()
        planner.submit('Blow a kiss', trajectory=UNREACHABLE, reviser=repair)
        status = self.wait(planner)
        self.assertEqual(status['state'], 'blocked')
        self.assertEqual(status['attempt'], 3)
        self.assertEqual(len(status['failures']), 3)
        self.assertEqual(repair.call_count, 2)
        self.assertIn('after 3 attempt', status['message'])
        self.assertRegex(status['message'], 'unreachable|joints jump')
        self.assertIsNone(planner.plan)

    def test_first_success_never_calls_revision_provider(self):
        factory = Mock()
        planner = PromptPlanner()
        planner.submit('Blow a kiss', trajectory=DRAFT, reviser=TrajectoryReviser(factory))
        self.assertEqual(self.wait(planner)['state'], 'proposed')
        factory.assert_not_called()

    def test_provider_failure_keeps_path_error_and_stops(self):
        repair = Mock(side_effect=ValueError('Provider usage limit reached'))
        planner = PromptPlanner()
        planner.submit('Blow a kiss', trajectory=UNREACHABLE, reviser=repair)
        status = self.wait(planner)
        self.assertEqual(status['state'], 'blocked')
        self.assertIn('usage limit', status['message'])
        self.assertRegex(status['message'], 'unreachable|joints jump')
        repair.assert_called_once()
        self.assertIsNone(planner.plan)

    def test_invalid_revision_or_changed_arm_cannot_become_a_preview(self):
        for candidate in ({**DRAFT, 'arm': 'left'}, {**DRAFT, 'return_to_start': False},
                          {**DRAFT, 'waypoints': []}):
            planner = PromptPlanner()
            repair = Mock(return_value=candidate)
            planner.submit('Blow a kiss', trajectory=UNREACHABLE, reviser=repair)
            status = self.wait(planner)
            self.assertEqual(status['state'], 'blocked')
            self.assertIsNone(planner.plan)
            repair.assert_called_once()

    def test_missing_context_and_invalid_pose_do_not_trigger_revisions(self):
        for planner, source in [(PromptPlanner(), 'camera'),
                                (PromptPlanner(preview_pose=lambda: {'right_elbow_joint': float('nan')}), 'auto')]:
            repair = Mock(return_value=DRAFT)
            planner.submit('Blow a kiss', source, UNREACHABLE, reviser=repair)
            self.assertEqual(self.wait(planner)['state'], 'blocked')
            repair.assert_not_called()
        repair = Mock(return_value=DRAFT)
        with patch('core.prompt_planner.Observation.load', side_effect=PerceptionError('Stale depth')):
            planner = PromptPlanner(observation_path='unused.npz')
            planner.submit('Blow a kiss', trajectory=UNREACHABLE, reviser=repair)
            self.assertIn('Stale depth', self.wait(planner)['message'])
        repair.assert_not_called()

    def test_observation_expiring_during_revision_cannot_pass(self):
        ik = ArmIK(backend='mujoco')
        observation = SimpleNamespace(pose={ik.model.joint(i).name: 0. for i in range(ik.model.njnt)},
                                      captured_at=100., require_free=Mock())
        now = [100.]
        def repair(*args):
            now[0] = 110.
            return DRAFT
        with patch('core.prompt_planner.Observation.load', return_value=observation), \
             patch('core.prompt_planner.time.time', side_effect=lambda: now[0]):
            planner = PromptPlanner(observation_path='unused.npz')
            planner.submit('Blow a kiss', trajectory=UNREACHABLE, reviser=repair)
            status = self.wait(planner)
        self.assertEqual(status['state'], 'blocked')
        self.assertIn('Observation expired', status['message'])
        self.assertIsNone(planner.plan)
        observation.require_free.assert_not_called()

    def test_cancel_during_revision_discards_late_provider_result(self):
        entered, release = threading.Event(), threading.Event()
        def provider(messages, context):
            entered.set()
            release.wait(3)
            return ANSWER
        reviser = TrajectoryReviser(lambda: provider)
        planner = PromptPlanner()
        planner.submit('Blow a kiss', trajectory=UNREACHABLE, reviser=reviser)
        self.assertTrue(entered.wait(3))
        self.assertEqual(planner.status()['stage'], 'revise')
        planner.cancel()
        release.set()
        self.assertEqual(self.wait(planner)['state'], 'cancelled')
        self.assertIsNone(planner.plan)
        with self.assertRaises(ValueError):
            planner.show_once(planner.status()['id'], Mock())
        # The next request gets a new cancellation lifecycle.
        planner.submit('Blow a kiss', trajectory=DRAFT)
        self.assertEqual(self.wait(planner)['state'], 'proposed')

    def test_cancel_before_lazy_provider_initialization(self):
        factory = Mock()
        reviser = TrajectoryReviser(factory)
        reviser.cancel()
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            reviser('Blow a kiss', UNREACHABLE, [], {})
        factory.assert_not_called()

    def test_cli_revision_captures_backend_and_has_no_tools_or_chat_process(self):
        for backend, cls in [('codex', CodexResponder), ('claude', ClaudeResponder)]:
            with self.subTest(backend=backend), \
                 patch.dict(os.environ, {'REINS_CODEX_BIN': '/no-cli', 'REINS_CLAUDE_BIN': '/no-cli'}), \
                 patch.object(cls, '__call__', return_value=ANSWER):
                chat = DashboardChat(backend=backend, tools=ToolLink('http://unused', 'unused'))
                self.addCleanup(chat.close)
                original = chat.responder
                reviser = chat.motion_reviser()
                self.assertIsNone(reviser.responder)  # no second sign-in check until needed
                chat.set_backend('openai')
                self.assertEqual(reviser('Blow a kiss', UNREACHABLE, [], {}), DRAFT)
                self.assertIsInstance(reviser.responder, cls)
                self.assertIsNot(reviser.responder, original)
                self.assertIsNone(reviser.responder.tools)
                reviser.cancel()
                self.assertTrue(reviser.responder.cancelled.is_set())
                self.assertFalse(original.cancelled.is_set())
                reviser.close()


if __name__ == '__main__':
    unittest.main()
