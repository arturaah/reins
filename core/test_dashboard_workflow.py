"""Automatic chat → validated preview → one Accept; fake models, no hardware."""
import copy
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from core.dashboard_chat import DashboardChat
from core.dashboard_workflow import bind_motion_workflow
from core.prompt_planner import PromptPlanner
from core.reins_tools import ReinsTools, ToolError
from core.robot_pipeline import RobotPipeline
from core.test_generated_motion import ANSWER, DRAFT
from core.test_trajectory_revision import UNREACHABLE
from tools.dashboard import Simulation


class AutomaticMotionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sim = Simulation()
        self.planner = PromptPlanner(output_dir=self.tmp.name)
        self.pipeline = RobotPipeline(self.planner, self.sim, {}, run_dir=self.tmp.name,
                                      simulation_only=True)
        self.registry = ReinsTools(None, {}, self.planner, self.sim.show_proposal,
                                   lambda: {'playing': self.sim.playing})
        self.chat = None
        self.sim.show_proposal = Mock(wraps=self.sim.show_proposal)

    def tearDown(self):
        self.pipeline.close()
        if self.planner.worker:
            self.planner.worker.join(3)
        if self.chat:
            self.chat.close()
        self.tmp.cleanup()

    def bind(self, responder):
        self.chat = DashboardChat(responder=responder)
        self.chat.configuration = lambda: {'configured': True, 'provider_label': 'Test model'}
        bind_motion_workflow(self.chat, self.registry, self.planner, self.pipeline)
        return self.chat

    def wait_chat(self):
        deadline = time.monotonic() + 15
        while self.chat.status()['busy'] and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(self.chat.status()['busy'])
        self.assertIsNone(self.chat.status()['error'])

    def wait_review(self):
        deadline = time.monotonic() + 15
        while self.pipeline.status()['state'] not in ('review', 'blocked') and time.monotonic() < deadline:
            time.sleep(.01)
        status = self.pipeline.status()
        self.assertEqual(status['state'], 'review', status)
        return status['proposal']

    @staticmethod
    def tool_args(draft):
        return {key: copy.deepcopy(value) for key, value in draft.items() if key != 'frame'}

    def test_final_trajectory_automatically_previews_and_never_executes_before_accept(self):
        before = self.pipeline.backend.joints()
        self.bind(Mock(return_value=ANSWER))
        with patch.object(self.pipeline, '_execute') as execute:
            self.chat.send('Blow a kiss')
            self.wait_chat()
            proposal = self.wait_review()
            self.assertTrue(self.sim.playing)
            self.assertEqual(self.pipeline.glasses_message()['review']['id'], proposal['id'])
            self.assertEqual(self.pipeline.backend.joints(), before)
            self.assertIsNone(self.pipeline.last_result)
            execute.assert_not_called()
        # One explicit decision is the only operation needed after the prompt.
        self.pipeline.decide(proposal['id'], proposal['digest'], 'approve')
        deadline = time.monotonic() + 3
        while self.pipeline.status()['state'] != 'completed' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.pipeline.status()['state'], 'completed')
        self.assertEqual(self.sim.show_proposal.call_count, 2)
        preview, accepted = self.sim.show_proposal.call_args_list
        self.assertEqual(accepted.args[1], preview.args[1] + ':accepted')
        preview_plan, accepted_plan = copy.deepcopy(preview.args[0]), copy.deepcopy(accepted.args[0])
        preview_plan['prompt_proposal'].pop('id')
        accepted_plan['prompt_proposal'].pop('id')
        self.assertEqual(preview_plan, accepted_plan)
        with self.assertRaises(ValueError):
            self.pipeline.decide(proposal['id'], proposal['digest'], 'approve')

    def test_trajectory_without_robot_request_also_automatically_plans(self):
        self.bind(Mock(return_value={**ANSWER, 'robot_request': None}))
        self.chat.send('Make this motion')
        self.wait_chat()
        self.assertEqual(self.wait_review()['name'], DRAFT['name'])
        assistant = next(m for m in self.chat.messages if m['role'] == 'assistant')
        self.assertEqual(assistant['robot_request'], DRAFT['name'])

    def test_novel_text_only_request_authors_waypoints_without_presets_or_clicks(self):
        contexts = []
        def respond(messages, context):
            contexts.append(copy.deepcopy(context))
            if 'trajectory_revision' in context:
                return ANSWER
            return {'reply': 'I will prepare that gesture.', 'robot_request': 'Blow a kiss', 'trajectory': None}
        self.bind(respond)
        self.chat.send('Blow a kiss')
        self.wait_chat()
        self.wait_review()
        self.assertEqual(len(contexts), 2)
        self.assertIn('motion_authoring', contexts[1])
        self.assertEqual(contexts[1]['trajectory_revision']['failures'][0]['stage'], 'author')
        self.assertEqual(self.planner.plan['generated_trajectory'], DRAFT)

    def test_model_plans_with_tool_and_final_draft_is_automatically_proposed(self):
        seen = []
        def respond(messages, context):
            seen.append(self.registry.call('plan_hand_path', self.tool_args(DRAFT)))
            self.assertIsNone(self.pipeline.proposal)
            return {'reply': 'The complete path is ready.', 'robot_request': None, 'trajectory': None}
        self.bind(respond)
        with patch.object(self.planner, 'submit', wraps=self.planner.submit) as submit:
            self.chat.send('Blow a kiss')
            self.wait_chat()
            proposal = self.wait_review()
            submit.assert_not_called()
        self.assertEqual(proposal['plan_id'], seen[0]['plan_id'])
        self.assertEqual(self.registry.plans, 1)
        self.assertTrue(self.sim.playing)

    def test_tool_submission_plus_final_trajectory_never_replans_or_replaces_review(self):
        proposed = []
        def respond(messages, context):
            draft = self.registry.call('plan_hand_path', self.tool_args(DRAFT))
            proposed.append(self.registry.call('propose_motion',
                {'plan_id': draft['plan_id'], 'request_id': 'model-final'}))
            return ANSWER
        self.bind(respond)
        with patch.object(self.planner, 'submit', wraps=self.planner.submit) as submit:
            self.chat.send('Blow a kiss')
            self.wait_chat()
            proposal = self.wait_review()
            submit.assert_not_called()
        self.assertEqual(proposal['id'], proposed[0]['proposal_id'])
        self.assertEqual(len(self.pipeline.requests), 1)
        self.assertEqual(self.registry.plans, 1)
        self.assertIsNone(self.pipeline.last_result)

    def test_blocked_tool_path_revises_in_same_turn_and_reports_reason_in_chat(self):
        calls = []
        def respond(messages, context):
            calls.append((self.registry.generation, copy.deepcopy(messages)))
            draft = UNREACHABLE if len(calls) == 1 else DRAFT
            self.registry.call('plan_hand_path', self.tool_args(draft))
            return {'reply': 'Checking the path.', 'robot_request': None, 'trajectory': None}
        self.bind(respond)
        self.chat.send('Blow a kiss')
        self.wait_chat()
        self.wait_review()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertEqual(self.registry.plans, 2)
        self.assertTrue(any('outside configured workspace' in m['text']
                            for m in self.chat.messages if m.get('planning_event')))
        self.assertTrue(any('Revising automatically' in m['text'] for m in calls[1][1]))
        self.assertEqual(sum(m['role'] == 'user' for m in self.chat.messages), 1)

    def test_fallback_planner_failure_resumes_same_tool_capable_model_turn(self):
        calls = []
        contexts = []
        def respond(messages, context):
            calls.append((self.registry.generation, copy.deepcopy(messages), copy.deepcopy(context)))
            if len(calls) == 1:
                return {**ANSWER, 'trajectory': UNREACHABLE}
            if 'trajectory_revision' in context:
                return {'reply': 'I need fresh robot context before revising this path.',
                        'robot_request': None, 'trajectory': None}
            contexts.append(self.registry.call('get_robot_context', {}))
            self.registry.call('plan_hand_path', self.tool_args(DRAFT))
            return {'reply': 'The complete revised motion is ready.',
                    'robot_request': None, 'trajectory': None}
        self.bind(respond)
        self.chat.tools = object()  # Injected provider emulates a transport with tools.
        self.chat.before_turn = Mock(wraps=self.chat.before_turn)
        self.registry.detector = Mock(available=False, open_vocabulary=False)
        self.registry.detector.name = 'disabled test detector'
        self.registry.MAX_PLANS = 3
        self.chat.MAX_AUTOMATIC_REVISIONS = 1
        before = self.pipeline.backend.joints()
        with patch.object(self.pipeline, '_execute') as execute:
            self.chat.send('Blow a kiss')
            self.wait_chat()
            self.wait_review()
            execute.assert_not_called()
        self.chat.before_turn.assert_called_once_with('Blow a kiss')
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({call[0] for call in calls}), 1)
        self.assertIn('trajectory_revision', calls[1][2])
        self.assertNotIn('trajectory_revision', calls[2][2])
        self.assertEqual(self.registry.plans, 2)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]['planning_budget']['remaining_plans'], 2)
        self.assertTrue(any('fresh robot context' in m['text'] and m.get('planning_event')
                            for m in self.chat.messages))
        self.assertTrue(any('Revising automatically' in m['text'] for m in calls[2][1]))
        self.assertEqual(self.pipeline.backend.joints(), before)
        self.assertIsNone(self.pipeline.decision)
        self.assertTrue(self.sim.playing)

    def test_repeated_tool_failure_stops_at_shared_plan_budget(self):
        generations = []
        def respond(messages, context):
            generations.append(self.registry.generation)
            self.registry.call('plan_hand_path', self.tool_args(UNREACHABLE))
            return {'reply': 'This candidate failed validation.', 'robot_request': None, 'trajectory': None}
        self.bind(respond)
        self.registry.MAX_PLANS = 2
        self.chat.send('Blow a kiss')
        self.wait_chat()
        self.assertEqual(len(generations), 2)
        self.assertEqual(len(set(generations)), 1)
        self.assertEqual(self.registry.plans, 2)
        self.assertIsNone(self.pipeline.proposal)
        self.assertFalse(self.sim.playing)
        self.assertTrue(any(m.get('planning_event', {}).get('retryable') is False
                            for m in self.chat.messages))

    def test_final_reply_fallback_spends_only_remaining_path_attempts(self):
        calls = []
        def respond(messages, context):
            calls.append(context)
            self.registry.call('plan_hand_path', self.tool_args(UNREACHABLE))
            return {**ANSWER, 'trajectory': UNREACHABLE}
        self.bind(respond)
        self.registry.MAX_PLANS = 2
        self.chat.send('Blow a kiss')
        self.wait_chat()
        deadline = time.monotonic() + 10
        while self.pipeline.status()['state'] != 'blocked' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.pipeline.status()['state'], 'blocked')
        self.assertEqual(self.planner.status()['attempt'], 1)
        self.assertEqual(self.planner.status()['max_attempts'], 1)
        self.assertEqual(len(calls), 1)  # No second provider call can exceed the shared budget.
        self.assertIsNone(self.pipeline.proposal)
        self.assertFalse(self.sim.playing)

    def test_ordinary_chat_preserves_a_pending_review(self):
        self.bind(Mock(side_effect=[ANSWER,
            {'reply': 'The path returns your hand to its starting position.',
             'robot_request': None, 'trajectory': None}]))
        self.chat.send('Blow a kiss')
        self.wait_chat()
        original = self.wait_review()
        self.planner.worker.join(3)
        before = self.pipeline.backend.joints()
        self.chat.send('What does return to start mean?')
        self.wait_chat()
        self.assertEqual(self.pipeline.status()['proposal'], original)
        self.assertEqual(self.pipeline.status()['state'], 'review')
        self.assertEqual(self.sim.show_proposal.call_count, 1)
        self.assertEqual(self.pipeline.backend.joints(), before)
        self.assertIsNone(self.pipeline.decision)

    def test_new_explicit_reply_motion_replaces_pending_review_without_stop(self):
        revised = copy.deepcopy(DRAFT)
        revised['name'] = 'Kiss with shorter pause'
        revised['waypoints'][1]['hold_s'] = .2
        self.bind(Mock(side_effect=[ANSWER, {**ANSWER, 'trajectory': revised}]))
        self.chat.send('Blow a kiss')
        self.wait_chat()
        original = self.wait_review()
        self.planner.worker.join(3)
        before = self.pipeline.backend.joints()
        self.chat.send('Make the pause shorter')
        self.wait_chat()
        replacement = self.wait_review()
        self.assertNotEqual(replacement['id'], original['id'])
        self.assertNotEqual(replacement['digest'], original['digest'])
        self.assertEqual(replacement['name'], revised['name'])
        self.assertGreater(replacement['revision'], original['revision'])
        with self.assertRaises(ValueError):
            self.pipeline.decide(original['id'], original['digest'], 'approve')
        self.assertEqual(self.pipeline.glasses_message()['review']['id'], replacement['id'])
        self.assertEqual(self.sim.show_proposal.call_count, 2)
        self.assertEqual(self.pipeline.backend.joints(), before)
        self.assertIsNone(self.pipeline.decision)

    def test_new_tool_plan_replaces_pending_review_without_stop(self):
        calls = []
        def respond(messages, context):
            draft = copy.deepcopy(DRAFT)
            if calls:
                draft['name'] = 'Kiss with shorter pause'
                draft['waypoints'][1]['hold_s'] = .2
            result = self.registry.call('plan_hand_path', self.tool_args(draft))
            calls.append(result)
            return {'reply': 'The complete motion is ready.', 'robot_request': None, 'trajectory': None}
        self.bind(respond)
        self.chat.send('Blow a kiss')
        self.wait_chat()
        original = self.wait_review()
        before = self.pipeline.backend.joints()
        self.chat.send('Make the pause shorter')
        self.wait_chat()
        replacement = self.wait_review()
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(replacement['id'], original['id'])
        self.assertEqual(replacement['plan_id'], calls[1]['plan_id'])
        with self.assertRaises(ValueError):
            self.pipeline.decide(original['id'], original['digest'], 'approve')
        self.assertEqual(self.sim.show_proposal.call_count, 2)
        self.assertEqual(self.pipeline.backend.joints(), before)
        self.assertIsNone(self.pipeline.decision)

    def assert_accepted_motion_cannot_be_replaced(self, use_tool, execution_state):
        reply_entered, reply_release = threading.Event(), threading.Event()
        execute_entered, execute_release = threading.Event(), threading.Event()
        calls = []
        def respond(messages, context):
            calls.append(True)
            if len(calls) == 1:
                return ANSWER
            reply_entered.set()
            reply_release.wait(5)
            if use_tool:
                try:
                    self.registry.call('plan_hand_path', self.tool_args(DRAFT))
                except ToolError as exc:
                    # Real tool transports deliver ToolError as model-visible feedback.
                    return {'reply': str(exc), 'robot_request': None, 'trajectory': None}
                return {'reply': 'Checking the requested revision.', 'robot_request': None, 'trajectory': None}
            return ANSWER
        def execute():
            if execution_state == 'executing':
                self.pipeline.event('executing', 'Test execution is in progress.')
            execute_entered.set()
            execute_release.wait(5)
        self.bind(respond)
        self.registry.MAX_PLANS = 1
        self.chat.send('Blow a kiss')
        self.wait_chat()
        original = self.wait_review()
        self.planner.worker.join(3)
        with patch.object(self.pipeline, '_execute', side_effect=execute):
            try:
                # Begin revising while review is still pending, then let Accept win
                # before the model submits its replacement candidate.
                self.chat.send('Make the pause shorter')
                self.assertTrue(reply_entered.wait(2))
                self.pipeline.decide(original['id'], original['digest'], 'approve')
                self.assertTrue(execute_entered.wait(2))
                reply_release.set()
                self.wait_chat()
                self.assertEqual(self.pipeline.status()['state'], execution_state)
                self.assertEqual(self.pipeline.status()['proposal']['id'], original['id'])
                self.assertEqual(self.pipeline.decision['decision'], 'approve')
                self.assertEqual(self.sim.show_proposal.call_count, 1)
            finally:
                reply_release.set()
                execute_release.set()
                self.pipeline.worker.join(3)

    def test_accept_winning_race_blocks_late_explicit_reply_replacement(self):
        self.assert_accepted_motion_cannot_be_replaced(use_tool=False, execution_state='approved')

    def test_executing_motion_cannot_be_replaced_by_late_tool_plan(self):
        self.assert_accepted_motion_cannot_be_replaced(use_tool=True, execution_state='executing')

    def test_clear_during_reply_cannot_later_generate_a_motion(self):
        entered, release = threading.Event(), threading.Event()
        def respond(messages, context):
            entered.set()
            release.wait(3)
            return ANSWER
        self.bind(respond)
        with patch.object(self.planner, 'submit', wraps=self.planner.submit) as submit:
            self.chat.send('Blow a kiss')
            self.assertTrue(entered.wait(2))
            self.chat.clear()
            release.set()
            self.wait_chat()
            submit.assert_not_called()
        self.assertIsNone(self.pipeline.proposal)
        self.assertFalse(self.sim.playing)
        self.assertEqual(self.chat.messages, [])


if __name__ == '__main__':
    unittest.main()
