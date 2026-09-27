"""Reins chat tools, the MCP server and the CLI wiring. No robot, cameras, model calls or network
beyond loopback. Run with .venv/bin/python -m unittest core.test_reins_tools."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import numpy as np

from core.claude_chat import ClaudeResponder
from core.codex_chat import CodexResponder, ToolLink
from core.dashboard_chat import INSTRUCTIONS, SCHEMA
from core.prompt_planner import PromptPlanner
from core.reins_tools import ReinsTools, ToolError
from core.tool_specs import TOOL_NAMES, TOOL_SPECS
from core.robot_pipeline import RobotPipeline
from core.test_generated_motion import DRAFT
from tools.dashboard import Simulation

ROOT = Path(__file__).resolve().parents[1]
CUP = {'label': 'cup', 'confidence': .81, 'bbox': [.4, .5, .5, .7]}


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.detector = Mock(available=True, open_vocabulary=True)
        self.detector.name = 'OmDet-Turbo · test'
        self.detector.detect.return_value = [CUP, {**CUP, 'label': 'plate', 'confidence': .6}]
        self.frame = np.zeros((48, 64, 3), np.uint8)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.planner = PromptPlanner(output_dir=self.tmp.name)             # no calibrated observation
        self.shown = []
        self.tools = ReinsTools(self.detector, {'head': lambda: (self.frame, time.monotonic() - .2, 'f1', None)},
                                self.planner, lambda plan, pid: self.shown.append(pid),
                                lambda: {'plan': None, 'name': None, 'playing': False},
                                camera_status=lambda: {'head': True, 'left': False})
        self.pipeline = RobotPipeline(self.planner, Simulation(), {}, run_dir=self.tmp.name, simulation_only=True)
        self.pipeline.cfg['perception']['pose_view'] = False
        self.tools.pipeline = self.pipeline
        self.addCleanup(self.pipeline.close)

    def test_specs_match_implementations(self):
        self.assertEqual(len(TOOL_NAMES), len(set(TOOL_NAMES)))
        for spec in TOOL_SPECS:
            self.assertTrue(callable(getattr(self.tools, spec['name'])))
            self.assertEqual(spec['inputSchema']['type'], 'object')
        self.assertNotIn('execute', ' '.join(TOOL_NAMES)); self.assertNotIn('gesture', ' '.join(TOOL_NAMES))

    def test_malformed_final_reply_invalidates_earlier_tool_draft_and_spends_budget(self):
        turn = self.tools.begin_turn('Plan a kiss')
        draft = self.tools.call('plan_hand_path', {k: v for k, v in DRAFT.items() if k != 'frame'})
        self.assertEqual(draft['state'], 'draft')
        rejected = self.tools.reject_final_motion('Waypoint below floor', turn)
        self.assertTrue(rejected['retry'])
        self.assertEqual(self.tools.plans, 2)
        self.assertEqual(self.tools.finish_turn(turn)['state'], 'blocked')
        self.assertIsNone(self.pipeline.proposal)

    def test_context_and_guidance_say_there_is_no_depth(self):
        from core.tool_specs import INSTRUCTIONS
        context = self.tools.call('get_robot_context', {})
        self.assertIn('no depth estimation', context['object_positions'])
        self.assertEqual(context['cameras'], {'head': True, 'left': False})
        self.assertIn('cannot approve or execute', context['physical_execution'])
        self.assertNotIn('locate_object', TOOL_NAMES); self.assertNotIn('plan_object_action', TOOL_NAMES)
        self.assertNotIn('request_visual_guidance', TOOL_NAMES); self.assertIn('Never invent', INSTRUCTIONS)
        self.assertIn('propose_motion', INSTRUCTIONS)

    def test_detect_objects_returns_2d_boxes_and_validates_input(self):
        result = self.tools.call('detect_objects', {'camera': 'head', 'labels': ['The cup', 'screwdriver']})
        self.detector.detect.assert_called_with(self.frame, labels=['cup', 'screwdriver'])
        self.assertEqual([o['label'] for o in result['objects']], ['cup', 'plate'])
        self.assertEqual(result['image_size'], [64, 48]); self.assertGreaterEqual(result['frame_age_s'], .2)
        self.assertIn('not a position in metres', result['note'])
        for bad in ({'camera': 'left'}, {'camera': 'observation'}, {'camera': 'nose'},
                    {'camera': 'head', 'labels': ['cup; rm -rf /']}, {'camera': 'head', 'labels': ['cup and plate']},
                    {'camera': 'head', 'extra': 1}):
            with self.assertRaises(ToolError):
                self.tools.call('detect_objects', bad)
        with self.assertRaises(ToolError):
            self.tools.call('open_gripper', {})
        log = self.tools.recent()
        self.assertTrue(log[0]['ok']); self.assertFalse(log[-1]['ok'])

    def test_hand_path_drafts_repeated_previews_and_one_proposal(self):
        result = self.tools.call('plan_hand_path', {'name': 'Right arm hello', 'arm': 'right', 'return_to_start': True,
            'waypoints': [{'position_m': [.25, -.25, 1.05], 'hold_s': .3}, {'position_m': [.25, -.32, 1.1], 'hold_s': 0}]})
        self.assertEqual(result['state'], 'draft', result)
        self.assertFalse(result['execution_allowed'])
        pid = result['plan_id']
        for _ in range(2):
            self.assertEqual(self.tools.call('preview_plan', {'plan_id': pid})['state'], 'previewed')
        self.assertIsNone(self.pipeline.proposal)
        proposed = self.tools.call('propose_motion', {'plan_id': pid, 'request_id': 'test-request'})
        self.assertEqual(proposed['state'], 'review')
        self.assertEqual(self.tools.call('propose_motion', {'plan_id': pid, 'request_id': 'test-request'})['proposal_id'], proposed['proposal_id'])
        self.assertIsNone(self.tools.call('get_motion_result', {'proposal_id': proposed['proposal_id']})['outcome'])

    def test_unreachable_path_reports_the_reason(self):
        result = self.tools.call('plan_hand_path', {'name': 'Too far', 'arm': 'right', 'return_to_start': False,
                                                    'waypoints': [{'position_m': [1.5, -.2, 1.0], 'hold_s': 0}]})
        self.assertEqual(result['state'], 'blocked'); self.assertTrue(result['message'])
        self.assertTrue(result['retryable'])
        self.assertIn('Waypoint 1', result['failures'][0]['error'])
        self.assertIn('plan_hand_path', result['next_step'])

    def test_completed_tool_turn_automatically_previews_and_waits_for_accept(self):
        token = self.tools.begin_turn('Blow a kiss')
        draft = self.tools.call('plan_hand_path', {k:v for k,v in DRAFT.items() if k != 'frame'})
        self.assertEqual(draft['state'], 'draft')
        self.assertIsNone(self.pipeline.proposal)
        result = self.tools.finish_turn(token)
        self.assertEqual(result['state'], 'review')
        self.assertTrue(self.pipeline.sim.playing)
        self.assertIsNotNone(self.pipeline.glasses_message()['review'])
        self.assertIsNone(self.pipeline.decision)
        self.assertEqual(self.tools.finish_turn(token), result)
        self.assertEqual(self.pipeline.revision, 1)

    def test_failed_revision_never_promotes_previous_valid_draft(self):
        notices = []
        self.tools.on_planning_event = notices.append
        token = self.tools.begin_turn('Blow a kiss, then revise it')
        self.tools.call('plan_hand_path', {k:v for k,v in DRAFT.items() if k != 'frame'})
        failure = self.tools.call('plan_hand_path', {'name': 'Too far', 'arm': 'right', 'return_to_start': False,
                                                  'waypoints': [{'position_m': [1.5, -.2, 1.0], 'hold_s': 0}]})
        result = self.tools.finish_turn(token)
        self.assertEqual(result['state'], 'blocked')
        self.assertEqual(result['message'], failure['message'])
        self.assertIsNone(self.pipeline.proposal)
        self.assertEqual(notices[0]['generation'], token)
        self.assertEqual(notices[0]['attempt'], 2)
        # An automatic continuation shares the budget and can finish a repaired path.
        self.tools.call('plan_hand_path', {k:v for k,v in DRAFT.items() if k != 'frame'})
        self.assertEqual(self.tools.finish_turn(token)['state'], 'review')
        self.assertEqual(self.tools.plans, 3)

    def test_invalid_or_budget_rejected_revision_clears_final_candidate(self):
        args = {k:v for k,v in DRAFT.items() if k != 'frame'}
        token = self.tools.begin_turn('Blow a kiss')
        self.tools.call('plan_hand_path', args)
        with self.assertRaises(ToolError):
            self.tools.call('plan_hand_path', None)
        self.assertEqual(self.tools.finish_turn(token)['state'], 'blocked')
        self.assertIsNone(self.pipeline.proposal)
        token = self.tools.begin_turn('Try again')
        self.tools.MAX_PLANS = 1
        self.tools.call('plan_hand_path', args)
        with self.assertRaisesRegex(ToolError, 'budget'):
            self.tools.call('plan_hand_path', args)
        result = self.tools.finish_turn(token)
        self.assertEqual(result['state'], 'blocked')
        self.assertFalse(result['retryable'])
        self.assertIsNone(self.pipeline.proposal)

    def test_final_reply_motion_supersedes_tool_draft_without_resetting_budget(self):
        token = self.tools.begin_turn('Blow a kiss')
        self.tools.call('plan_hand_path', {k:v for k,v in DRAFT.items() if k != 'frame'})
        self.assertEqual(self.tools.finish_turn(token, has_motion=True)['state'], 'none')
        self.assertIsNone(self.pipeline.proposal)
        self.assertEqual(self.tools.plans, 1)
        token = self.tools.begin_turn('Only one attempt')
        self.tools.MAX_PLANS = 1
        self.tools.call('plan_hand_path', {k:v for k,v in DRAFT.items() if k != 'frame'})
        self.assertEqual(self.tools.finish_turn(token, has_motion=True)['state'], 'blocked')
        self.assertIsNone(self.pipeline.proposal)

    def test_completed_explicit_proposal_is_not_submitted_again(self):
        token = self.tools.begin_turn('Blow a kiss')
        draft = self.tools.call('plan_hand_path', {k:v for k,v in DRAFT.items() if k != 'frame'})
        proposal = self.tools.call('propose_motion', {'plan_id': draft['plan_id'], 'request_id': 'one-review'})
        p = proposal['proposal']
        self.pipeline.decide(p['id'], p['digest'], 'decline')
        result = self.tools.finish_turn(token, has_motion=True)
        self.assertEqual(result['outcome'], 'declined')
        self.assertIsNone(self.pipeline.proposal)
        self.assertEqual(self.pipeline.revision, 1)

    def test_stale_completed_turn_cannot_submit_a_new_turn_draft(self):
        old_token = self.tools.begin_turn('First task')
        token = self.tools.begin_turn('Replacement')
        self.tools.call('plan_hand_path', {k:v for k,v in DRAFT.items() if k != 'frame'})
        self.assertEqual(self.tools.finish_turn(old_token)['state'], 'cancelled')
        self.assertIsNone(self.pipeline.proposal)
        self.tools.cancel()
        self.assertEqual(self.tools.finish_turn(token)['state'], 'cancelled')
        self.assertIsNone(self.pipeline.proposal)

    def test_observe_returns_images_and_detection_uses_the_exact_frame(self):
        import base64
        import io
        from PIL import Image
        result = self.tools.call('observe', {'cameras': ['head']})
        observation = result['observation']
        block = next(b for b in result['content_blocks'] if b['type'] == 'image')
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(block['data']))).size, (64, 48))
        self.frame[:] = 255
        self.tools.call('detect_objects', {'camera': 'head', 'observation_id': observation['id']})
        self.assertTrue(np.all(self.detector.detect.call_args.args[0] == 0))
        self.assertIn(observation['id'], self.pipeline.observations)
        self.assertNotIn('data', json.dumps(self.tools.recent()))

    def test_stale_images_and_bounded_planning_retries(self):
        self.tools.sources['head'] = lambda: (self.frame, time.monotonic()-5, 'stale', None)
        with self.assertRaisesRegex(ToolError, 'stale'): self.tools.call('observe', {})
        self.tools.MAX_PLANS = 1
        args = {'name':'Far', 'arm':'right', 'return_to_start':False, 'waypoints':[{'position_m':[1.5,-.2,1.], 'hold_s':0}]}
        self.assertFalse(self.tools.call('plan_hand_path', args)['retryable'])
        with self.assertRaisesRegex(ToolError, 'budget'): self.tools.call('plan_hand_path', args)
        self.tools.begin_turn('A new request')
        self.assertEqual(self.tools.plans, 0)
        self.tools.cancel()
        with self.assertRaisesRegex(ToolError, 'budget'): self.tools.call('get_robot_context', {})

    def run_tool(self, name, arguments, results):
        try: results.append(self.tools.call(name, arguments))
        except Exception as exc: results.append(exc)

    def test_cancel_and_new_turn_cannot_publish_an_inflight_compilation(self):
        from core.generated_motion import compile_trajectory
        entered, resume, results = threading.Event(), threading.Event(), []
        def paused(*args):
            entered.set()
            if not resume.wait(3): raise RuntimeError('test compilation not resumed')
            return compile_trajectory(*args)
        args = {k:v for k,v in DRAFT.items() if k != 'frame'}
        with patch('core.robot_pipeline.compile_trajectory', side_effect=paused):
            worker = threading.Thread(target=self.run_tool,args=('plan_hand_path',args,results))
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                self.tools.cancel()
                self.tools.begin_turn('A different motion')
            finally:
                resume.set(); worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(results[0],ToolError)
        self.assertIsNone(self.pipeline.draft)
        self.assertIsNone(self.pipeline.proposal)
        self.assertEqual(self.pipeline.drafts,{})
        self.assertEqual(self.tools.call('plan_hand_path',args)['state'],'draft')

    def test_queued_old_turn_does_not_start_planning_after_new_turn(self):
        results=[]
        self.tools.planning_lock.acquire()
        with patch.object(self.pipeline,'prepare_hand') as prepare:
            worker=threading.Thread(target=self.run_tool,args=('plan_hand_action',{'arm':'right','closed':True},results))
            worker.start()
            try:
                end=time.monotonic()+3
                while self.tools.calls != 1 and time.monotonic()<end: time.sleep(.005)
                self.assertEqual(self.tools.calls,1)
                self.tools.begin_turn('Replace the queued request')
            finally:
                self.tools.planning_lock.release(); worker.join(3)
            prepare.assert_not_called()
        self.assertIsInstance(results[0],ToolError)

    def test_cancelled_observation_does_not_enter_the_new_turn(self):
        entered,resume,results=threading.Event(),threading.Event(),[]
        def source():
            entered.set()
            if not resume.wait(3): raise RuntimeError('test image not resumed')
            return self.frame,time.monotonic(),'old-frame',None
        self.tools.sources['head']=source
        worker=threading.Thread(target=self.run_tool,args=('observe',{},results));worker.start()
        try:
            self.assertTrue(entered.wait(3))
            self.tools.cancel();self.tools.begin_turn('Fresh scene')
        finally:
            resume.set();worker.join(3)
        self.assertIsInstance(results[0],ToolError)
        self.assertEqual(self.tools.observations,{})
        self.assertEqual(self.pipeline.observations,{})

    def test_new_turn_invalidates_unsent_draft_but_preserves_review_and_feedback(self):
        d=self.pipeline.compile_hand_path(DRAFT)
        self.tools.begin_turn('Change that motion')
        with self.assertRaisesRegex(ValueError,'expired'): self.pipeline.propose_motion(d['id'])
        d=self.pipeline.compile_hand_path(DRAFT)
        p=self.pipeline.propose_motion(d['id'])['proposal']
        self.tools.cancel();self.tools.begin_turn('Explain this pending motion')
        self.assertEqual(self.pipeline.proposal['id'],p['id'])
        self.pipeline.decide(p['id'],p['digest'],'decline','Keep the hand lower')
        self.tools.begin_turn('Use my feedback')
        result=self.tools.call('get_motion_result',{'proposal_id':p['id']})
        self.assertEqual(result['outcome'],'declined')
        self.assertEqual(result['decision']['note'],'Keep the hand lower')


class FakeDashboard(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeDashboard.calls.append((self.path, self.headers.get('X-Reins-Tool-Token'), body))
        ok = self.path.endswith(('/detect_objects', '/observe'))
        answer = {'objects': [CUP]}
        if self.path.endswith('/observe'):
            answer = {'observation': {'id': 'frame123'}, 'content_blocks': [
                {'type': 'text', 'text': 'Head camera'}, {'type': 'image', 'data': 'aW1hZ2U=', 'mimeType': 'image/jpeg'}]}
        payload = json.dumps(answer if ok else {'error': 'Proposal is no longer available'}).encode()
        self.send_response(200 if ok else 400); self.send_header('Content-Length', str(len(payload))); self.end_headers()
        self.wfile.write(payload)


class McpServerTests(unittest.TestCase):
    def test_stdio_protocol_and_forwarding(self):
        FakeDashboard.calls = []
        server = ThreadingHTTPServer(('127.0.0.1', 0), FakeDashboard)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp:
            token = Path(tmp) / 'tool.token'; token.write_text('secret-token')
            env = {**os.environ, 'REINS_TOOL_URL': f'http://127.0.0.1:{server.server_port}/api/tools',
                   'REINS_TOOL_TOKEN_FILE': str(token)}
            messages = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 't', 'version': '1'}}},
                        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'detect_objects', 'arguments': {'camera': 'head'}}},
                        {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'preview_plan', 'arguments': {'proposal_id': 'stale-proposal'}}},
                        {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {'name': 'execute_plan', 'arguments': {}}},
                        {'jsonrpc': '2.0', 'id': 6, 'method': 'resources/list'},
                        {'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call', 'params': {'name': 'observe', 'arguments': {}}}]
            out = subprocess.run([sys.executable, str(ROOT / 'tools/reins_mcp.py')], input='\n'.join(map(json.dumps, messages)) + '\n',
                                 capture_output=True, text=True, env=env, timeout=30).stdout
        replies = {r['id']: r for r in map(json.loads, out.splitlines())}
        self.assertEqual(sorted(replies), [1, 2, 3, 4, 5, 6, 7])                   # the notification got no reply
        self.assertEqual(replies[1]['result']['protocolVersion'], '2025-06-18')
        self.assertEqual([t['name'] for t in replies[2]['result']['tools']], TOOL_NAMES)
        self.assertFalse(replies[3]['result']['isError'])
        self.assertEqual(json.loads(replies[3]['result']['content'][0]['text'])['objects'][0]['label'], 'cup')
        self.assertTrue(replies[4]['result']['isError'])
        self.assertIn('no longer available', replies[4]['result']['content'][0]['text'])
        self.assertEqual(replies[5]['error']['code'], -32602)                   # unknown tool never forwarded
        self.assertEqual(replies[6]['error']['code'], -32601)
        observed = replies[7]['result']['content']
        self.assertEqual(observed[-1], {'type':'image','data':'aW1hZ2U=','mimeType':'image/jpeg'})
        self.assertNotIn('content_blocks',json.loads(observed[0]['text']))
        self.assertEqual([(p, t) for p, t, _ in FakeDashboard.calls],
                         [('/api/tools/detect_objects', 'secret-token'), ('/api/tools/preview_plan', 'secret-token'), ('/api/tools/observe', 'secret-token')])


class CliWiringTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'REINS_CLAUDE_BIN': '/no-such-claude', 'REINS_CODEX_BIN': '/no-such-codex'}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.link = ToolLink('http://127.0.0.1:9/api/tools', 'very-secret-token')

    def test_claude_gets_only_the_reins_tools(self):
        bridge = ClaudeResponder(INSTRUCTIONS, SCHEMA, tools=self.link); bridge.binary = 'claude'
        with tempfile.TemporaryDirectory() as tmp:
            args = bridge.command(Path(tmp) / 'reply.schema.json')
            config = json.loads(Path(args[args.index('--mcp-config') + 1]).read_text())
            token_file = Path(config['mcpServers']['reins']['env']['REINS_TOOL_TOKEN_FILE'])
            self.assertEqual(token_file.read_text(), 'very-secret-token')
            self.assertEqual(oct(token_file.stat().st_mode & 0o777), '0o600')
        self.assertNotIn('very-secret-token', ' '.join(args))
        self.assertEqual(args[args.index('--tools') + 1], '')                   # built-ins stay off
        self.assertIn('--strict-mcp-config', args)
        self.assertEqual(args[args.index('--allowedTools') + 1].split(','), [f'mcp__reins__{n}' for n in TOOL_NAMES])
        self.assertIn('plan_hand_path', args[args.index('--system-prompt') + 1])
        self.assertGreaterEqual(bridge.timeout, 300)

    def test_codex_gets_the_reins_server(self):
        bridge = CodexResponder(INSTRUCTIONS, SCHEMA, tools=self.link); bridge.binary = 'codex'
        with tempfile.TemporaryDirectory() as tmp:
            args = bridge.command(Path(tmp) / 'reply.schema.json')
        joined = ' '.join(args)
        self.assertNotIn('very-secret-token', joined)
        self.assertIn('mcp_servers.reins.args=', joined); self.assertIn('reins_mcp.py', joined)
        self.assertIn('--sandbox read-only', joined); self.assertEqual(args[-1], '-')
        self.assertIn('plan_hand_path', bridge.instructions)

    def test_without_tools_nothing_changes(self):
        bridge = ClaudeResponder(INSTRUCTIONS, SCHEMA); bridge.binary = 'claude'
        with tempfile.TemporaryDirectory() as tmp:
            args = bridge.command(Path(tmp) / 'reply.schema.json')
        self.assertNotIn('--mcp-config', args)
        self.assertNotIn('You are the planning agent in Reins.', bridge.instructions)


if __name__ == '__main__':
    unittest.main()
