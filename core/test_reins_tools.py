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

    def test_specs_match_implementations(self):
        self.assertEqual(len(TOOL_NAMES), len(set(TOOL_NAMES)))
        for spec in TOOL_SPECS:
            self.assertTrue(callable(getattr(self.tools, spec['name'])))
            self.assertEqual(spec['inputSchema']['type'], 'object')
        self.assertNotIn('execute', ' '.join(TOOL_NAMES)); self.assertNotIn('gesture', ' '.join(TOOL_NAMES))

    def test_context_and_guidance_say_there_is_no_depth(self):
        from core.tool_specs import INSTRUCTIONS
        context = self.tools.call('get_robot_context', {})
        self.assertIn('no depth estimation', context['object_positions'])
        self.assertEqual(context['cameras'], {'head': True, 'left': False})
        self.assertIn('cannot approve or execute', context['physical_execution'])
        self.assertNotIn('locate_object', TOOL_NAMES); self.assertNotIn('plan_object_action', TOOL_NAMES)
        self.assertIn('request_visual_guidance', INSTRUCTIONS); self.assertIn('Never invent', INSTRUCTIONS)

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

    def test_hand_path_plans_then_previews_once(self):
        result = self.tools.call('plan_hand_path', {'name': 'Right arm hello', 'arm': 'right', 'return_to_start': True,
            'waypoints': [{'position_m': [.25, -.25, 1.05], 'hold_s': .3}, {'position_m': [.25, -.32, 1.1], 'hold_s': 0}]})
        self.assertEqual(result['state'], 'proposed', result['message'])
        self.assertFalse(result['execution_allowed'])
        pid = result['proposal_id']
        self.assertEqual(self.tools.call('preview_plan', {'proposal_id': pid})['state'], 'previewed')
        self.assertEqual(self.shown, [pid])
        with self.assertRaises(ToolError):
            self.tools.call('preview_plan', {'proposal_id': pid})                # shown once only

    def test_unreachable_path_reports_the_reason(self):
        result = self.tools.call('plan_hand_path', {'name': 'Too far', 'arm': 'right', 'return_to_start': False,
                                                    'waypoints': [{'position_m': [1.5, -.2, 1.0], 'hold_s': 0}]})
        self.assertEqual(result['state'], 'blocked'); self.assertTrue(result['message'])
        self.assertTrue(result['retryable'])
        self.assertEqual(result['failures'][0]['details']['waypoint_number'], 1)
        self.assertIn('plan_hand_path', result['next_step'])


class FakeDashboard(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeDashboard.calls.append((self.path, self.headers.get('X-Reins-Tool-Token'), body))
        ok = self.path.endswith('/detect_objects')
        payload = json.dumps({'objects': [CUP]} if ok else {'error': 'Proposal is no longer available'}).encode()
        self.send_response(200 if ok else 400); self.send_header('Content-Length', str(len(payload))); self.end_headers()
        self.wfile.write(payload)


class McpServerTests(unittest.TestCase):
    def test_stdio_protocol_and_forwarding(self):
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
                        {'jsonrpc': '2.0', 'id': 6, 'method': 'resources/list'}]
            out = subprocess.run([sys.executable, str(ROOT / 'tools/reins_mcp.py')], input='\n'.join(map(json.dumps, messages)) + '\n',
                                 capture_output=True, text=True, env=env, timeout=30).stdout
        replies = {r['id']: r for r in map(json.loads, out.splitlines())}
        self.assertEqual(sorted(replies), [1, 2, 3, 4, 5, 6])                   # the notification got no reply
        self.assertEqual(replies[1]['result']['protocolVersion'], '2025-06-18')
        self.assertEqual([t['name'] for t in replies[2]['result']['tools']], TOOL_NAMES)
        self.assertFalse(replies[3]['result']['isError'])
        self.assertEqual(json.loads(replies[3]['result']['content'][0]['text'])['objects'][0]['label'], 'cup')
        self.assertTrue(replies[4]['result']['isError'])
        self.assertIn('no longer available', replies[4]['result']['content'][0]['text'])
        self.assertEqual(replies[5]['error']['code'], -32602)                   # unknown tool never forwarded
        self.assertEqual(replies[6]['error']['code'], -32601)
        self.assertEqual([(p, t) for p, t, _ in FakeDashboard.calls],
                         [('/api/tools/detect_objects', 'secret-token'), ('/api/tools/preview_plan', 'secret-token')])


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
        self.assertNotIn('--mcp-config', args); self.assertNotIn('plan_hand_path', bridge.instructions)


if __name__ == '__main__':
    unittest.main()
