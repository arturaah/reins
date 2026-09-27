"""Generated motion tests use real local IK/validation, never a model or hardware."""
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from core.dashboard_chat import DashboardChat, SCHEMA, validate_reply
from core.generated_motion import compile_trajectory, motion_context, validate_trajectory
from core.ik import ArmIK
from core.motion_validation import MotionValidator
from core.prompt_planner import PromptPlanner

# A newly authored non-contact gesture, intentionally absent from route_intent.
DRAFT = {
    'name': 'Blow a kiss approximation',
    'arm': 'right',
    'frame': 'robot_base',
    'waypoints': [
        {'position_m': [.22, -.22, 1.0], 'hold_s': 0},
        {'position_m': [.22, -.14, 1.17], 'hold_s': .4},
        {'position_m': [.38, -.18, 1.1], 'hold_s': .2},
    ],
    'return_to_start': True,
}
ANSWER = {'reply': 'I drafted a non-contact arm gesture. Prepare its preview to validate it.',
          'robot_request': 'Blow a kiss with your right hand', 'trajectory': DRAFT}


class DraftTests(unittest.TestCase):
    def test_novel_motion_is_not_filtered_through_builtin_router(self):
        with patch('core.action_context.route_intent', side_effect=AssertionError('builtin router called')):
            result = validate_reply(ANSWER)
        self.assertEqual(result['trajectory'], DRAFT)
        result['trajectory']['waypoints'][0]['position_m'][0] = 99
        self.assertNotEqual(result['trajectory'], DRAFT)
        self.assertIn('trajectory', SCHEMA['required'])

    def test_malformed_or_unbounded_drafts_are_rejected(self):
        bad = []
        for key, value in [('arm', 'both'), ('frame', 'camera'), ('return_to_start', 1),
                           ('name', ''), ('waypoints', []), ('waypoints', DRAFT['waypoints']*6),
                           ('execute', True)]:
            draft = copy.deepcopy(DRAFT); draft[key] = value; bad.append(draft)
        for value in [float('nan'), float('inf'), True, '0.3', 3., -3., 10**400]:
            draft = copy.deepcopy(DRAFT); draft['waypoints'][0]['position_m'][0] = value; bad.append(draft)
        for value in [-1, 6, True, float('nan')]:
            draft = copy.deepcopy(DRAFT); draft['waypoints'][0]['hold_s'] = value; bad.append(draft)
        draft = copy.deepcopy(DRAFT); draft['waypoints'][0]['position_m'][2] = -.1; bad.append(draft)
        for draft in bad:
            with self.subTest(draft=draft), self.assertRaises(ValueError):
                validate_trajectory(draft)
        self.assertEqual(validate_reply({**ANSWER, 'robot_request': None})['robot_request'], DRAFT['name'])

    def test_history_carries_draft_and_rejects_stale_or_cleared_suggestions(self):
        provider = Mock(side_effect=[ANSWER, {'reply': 'Hello', 'robot_request': None}])
        chat = DashboardChat(responder=provider)
        chat.configuration = lambda: {'configured': True, 'provider_label': 'Test'}
        with patch('core.prompt_planner.PromptPlanner.submit') as submit:
            chat.send('Blow a kiss')
            wait_chat(chat)
            submit.assert_not_called()
        message = chat.status()['messages'][-1]
        request = chat.motion_request(message['id'])
        self.assertEqual(request['trajectory'], DRAFT)
        request['trajectory']['name'] = 'mutated'
        self.assertEqual(chat.motion_request(message['id'])['trajectory']['name'], DRAFT['name'])
        chat.send('Hello'); wait_chat(chat)
        self.assertEqual(provider.call_args.args[0][-2]['trajectory'], DRAFT)
        with self.assertRaisesRegex(ValueError, 'no longer current'):
            chat.motion_request(message['id'])
        chat.clear()
        with self.assertRaises(ValueError): chat.motion_request(message['id'])

    def test_history_budget_counts_waypoints(self):
        chat = DashboardChat(responder=Mock(return_value=ANSWER))
        chat.configuration = lambda: {'configured': True, 'provider_label': 'Test'}
        chat.MAX_CHARACTERS = 1800
        for _ in range(4):
            chat.send('Blow a kiss'); wait_chat(chat)
        status = chat.status()
        self.assertTrue(status['trimmed'])
        self.assertLessEqual(sum(len(json.dumps(m)) for m in status['messages']), chat.MAX_CHARACTERS)


def wait_chat(chat):
    deadline = time.monotonic()+3
    while chat.status()['busy'] and time.monotonic()<deadline: time.sleep(.01)
    if chat.status()['busy']: raise AssertionError('Chat did not finish')


class GeneratedPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ik = ArmIK(backend='mujoco')

    def wait(self, planner):
        deadline = time.monotonic()+20
        while planner.status()['state']=='planning' and time.monotonic()<deadline: time.sleep(.01)
        self.assertNotEqual(planner.status()['state'], 'planning')
        return planner.status()

    def test_chat_draft_compiles_validates_and_shows_once_without_saving(self):
        from sim.preview import load_plan
        from tools.dashboard import Simulation
        with tempfile.TemporaryDirectory() as tmp:
            planner = PromptPlanner(output_dir=tmp)
            with patch('core.prompt_planner.route_intent', side_effect=AssertionError('builtin router called')), \
                 patch('core.prompt_planner.gesture_plan', side_effect=AssertionError('builtin gesture called')), \
                 patch('core.prompt_planner.ground_openai') as vision:
                planner.submit(ANSWER['robot_request'], trajectory=validate_reply(ANSWER)['trajectory'])
                status = self.wait(planner)
                vision.assert_not_called()
            self.assertEqual(status['state'], 'proposed', status['message'])
            self.assertEqual(status['context']['skill'], 'generated_trajectory')
            self.assertFalse(status['execution_allowed'])
            self.assertFalse(status['contact_enabled'])
            self.assertGreater(status['validation']['samples'], 100)
            self.assertLessEqual(status['validation']['max_velocity_rad_s'], .4)
            self.assertLessEqual(status['validation']['max_sampled_acceleration_rad_s2'], 1.5)
            simulation = Simulation()
            planner.show_once(status['id'],simulation.show_proposal)
            self.assertTrue(simulation.playing)
            self.assertEqual(simulation.position,0)
            self.assertEqual(simulation.plan['generated_trajectory'],DRAFT)
            self.assertTrue(simulation.plan['preview_only'])
            self.assertFalse(list(Path(tmp).iterdir()))
            with self.assertRaises(ValueError):planner.show_once(status['id'],simulation.show_proposal)

    def test_compile_preserves_nonzero_start_held_joints_pauses_and_hand_return(self):
        pose = {'right_shoulder_pitch_joint': -.1, 'right_elbow_joint': .15,
                'left_elbow_joint': .2, 'waist_yaw_joint': .05}
        start, _ = self.ik.fk('right', pose, pose)
        draft = {**copy.deepcopy(DRAFT), 'waypoints': [{'position_m': (start+[.02, -.01, .02]).tolist(), 'hold_s': .8}]}
        plan = compile_trajectory(self.ik, draft, pose)
        self.assertEqual(plan['keyframes'][0]['joint_targets_rad']['right_elbow_joint'], .15)
        self.assertEqual(plan['held_joints_rad'], {'left_elbow_joint': .2, 'waist_yaw_joint': .05})
        end, _ = self.ik.fk('right', plan['keyframes'][-1]['joint_targets_rad'], pose)
        np.testing.assert_allclose(start, end, atol=.001)
        self.assertTrue(any(a['joint_targets_rad']==b['joint_targets_rad'] and b['time_s']-a['time_s']>=.79
                            for a,b in zip(plan['keyframes'],plan['keyframes'][1:])))
        MotionValidator(self.ik.model).check(plan, 'right', [])

    def test_geometry_is_model_based(self):
        pose = {'right_shoulder_pitch_joint': -.4}
        context = motion_context(self.ik, pose)
        actual, _ = self.ik.fk('right', pose, pose)
        np.testing.assert_allclose(context['arms']['right']['hand_position_m'], actual, atol=.0001)
        self.assertIn('simulation', context['pose_source'])

    def test_unreachable_and_head_collision_block_without_saving(self):
        for point in [[1.5, -.2, 1.0], [.10, -.10, 1.13]]:
            draft = {**DRAFT, 'waypoints': [{'position_m':point,'hold_s':0}], 'return_to_start':False}
            with tempfile.TemporaryDirectory() as tmp:
                planner = PromptPlanner(output_dir=tmp)
                planner.submit('New gesture', trajectory=draft)
                status = self.wait(planner)
                self.assertEqual(status['state'], 'blocked', status)
                self.assertIsNone(planner.plan)
                self.assertFalse(list(Path(tmp).iterdir()))
                with self.assertRaises(ValueError): planner.preview(status['id'])

    def test_explicit_camera_never_falls_back_to_simulation(self):
        planner = PromptPlanner()
        planner.submit('New gesture', 'camera', DRAFT)
        status = self.wait(planner)
        self.assertEqual(status['state'], 'blocked')
        self.assertIn('No calibrated depth', status['message'])

    def test_cancel_discards_inflight_compilation(self):
        entered, release = threading.Event(), threading.Event()
        def paused(*args, **kwargs):
            entered.set(); release.wait(3)
            return compile_trajectory(*args, **kwargs)
        planner = PromptPlanner()
        with patch('core.prompt_planner.compile_trajectory', side_effect=paused):
            planner.submit('New gesture', trajectory=DRAFT)
            self.assertTrue(entered.wait(2))
            planner.cancel(); release.set()
            status = self.wait(planner)
        self.assertEqual(status['state'], 'cancelled')
        self.assertIsNone(planner.plan)


if __name__=='__main__': unittest.main()
