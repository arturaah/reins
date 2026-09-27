"""Claude CLI bridge and assistant switching. A tiny fake `claude` executable stands in for the
real CLI: no model calls, no real sign-in. Run with .venv/bin/python -m unittest core.test_claude_chat."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from core.claude_chat import ClaudeResponder
from core.dashboard_chat import DashboardChat, INSTRUCTIONS, SCHEMA

FAKE_CLAUDE = '''#!{python}
import json, sys, time
args = sys.argv[1:]
if args[:2] == ['auth', 'status']:
    print(json.dumps({{'loggedIn': {logged_in}, 'authMethod': 'claude.ai'}})); sys.exit(0)
text = sys.stdin.read()
if 'HANG_FOR_TEST' in text: time.sleep(10)
if 'FAIL_FOR_TEST' in text:
    print('private-token must not appear in browser', file=sys.stderr); sys.exit(1)
if 'AUTH_FOR_TEST' in text:
    print(json.dumps({{'type': 'result', 'subtype': 'success', 'is_error': True, 'result': 'Not logged in. Please run /login'}})); sys.exit(1)
if 'VERIFY_ARGS' in text:
    assert args[0] == '-p' and '--bare' not in args
    assert args[args.index('--output-format') + 1] == 'json'
    assert args[args.index('--tools') + 1] == ''
    assert '--strict-mcp-config' in args and '--disable-slash-commands' in args and '--no-session-persistence' in args
    assert args[args.index('--setting-sources') + 1] == 'local'
    assert args[args.index('--permission-mode') + 1] == 'dontAsk'
    assert json.loads(args[args.index('--json-schema') + 1])['required'] == ['reply', 'robot_request', 'trajectory']
    assert 'You are Reins' in args[args.index('--system-prompt') + 1]
    assert 'VERIFY_ARGS' not in ' '.join(args)          # user text only ever on stdin
    assert 'You are Reins' not in text                  # instructions are not repeated in the data
answer = {{'reply': 'Claude says hello.', 'robot_request': None, 'trajectory': None}}
print(json.dumps({{'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'private chatter',
                  'structured_output': answer, 'total_cost_usd': 0.001}}))
'''


class ClaudeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.binary = Path(self.tmp.name) / 'fake-claude'
        self.write(logged_in=True)
        self.env = patch.dict(os.environ, {'REINS_CLAUDE_BIN': str(self.binary), 'REINS_CODEX_BIN': '/no-such-codex'}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)

    def write(self, logged_in):
        self.binary.write_text(FAKE_CLAUDE.format(python=sys.executable, logged_in='True' if logged_in else 'False'))
        self.binary.chmod(0o700)

    def bridge(self, timeout=3):
        bridge = ClaudeResponder(INSTRUCTIONS, SCHEMA, timeout=timeout)
        self.addCleanup(bridge.close)
        return bridge

    def wait(self, chat):
        limit = time.monotonic() + 5
        while chat.status()['busy'] and time.monotonic() < limit:
            time.sleep(.01)
        self.assertFalse(chat.status()['busy'])
        return chat.status()

    def test_locked_down_cli_uses_saved_sign_in_and_stdin(self):
        bridge = self.bridge()
        config = bridge.configuration()
        self.assertTrue(config['configured']); self.assertEqual(config['provider'], 'claude')
        result = bridge([{'role': 'user', 'text': 'VERIFY_ARGS; $(echo cannot-execute); `false`'}], {})
        self.assertEqual(result['reply'], 'Claude says hello.')
        self.assertNotIn('private', json.dumps(result)); self.assertIsNone(bridge.proc)

    def test_signed_out_or_missing_cli_needs_setup(self):
        self.write(logged_in=False)
        bridge = self.bridge()
        self.assertFalse(bridge.configuration()['configured'])
        with self.assertRaisesRegex(ValueError, 'claude auth login'):
            bridge([], {})
        with patch.dict(os.environ, {'REINS_CLAUDE_BIN': '/no-such-claude'}):
            self.assertFalse(self.bridge().configuration()['configured'])

    def test_failures_are_sanitized_and_mapped(self):
        bridge = self.bridge()
        for text, match in [('FAIL_FOR_TEST', 'could not finish'), ('AUTH_FOR_TEST', 'sign-in')]:
            with self.assertRaisesRegex(ValueError, match) as exc:
                bridge([{'role': 'user', 'text': text}], {})
            self.assertNotIn('private-token', str(exc.exception)); self.assertIsNone(bridge.proc)
        for output, diagnostic, match in [('', 'error: unknown option --json-schema', 'version'),
                                          ('', '429 rate limit', 'usage limit'),
                                          ('{"type":"result","subtype":"success","is_error":false,"result":"no json"}', '', 'no usable')]:
            with self.assertRaisesRegex(ValueError, match):
                ClaudeResponder._parse(output, diagnostic, 0 if output else 1)

    def test_timeout_kills_the_child(self):
        bridge = self.bridge(timeout=.2)
        with self.assertRaisesRegex(ValueError, 'Claude CLI timed out'):
            bridge([{'role': 'user', 'text': 'HANG_FOR_TEST'}], {})
        self.assertIsNone(bridge.proc)

    def test_switching_assistants_keeps_the_conversation(self):
        replies = []
        def openai(messages, context):
            replies.append(len(messages))
            return {'reply': 'API says hi.', 'robot_request': None}
        chat = DashboardChat(backend='claude'); self.addCleanup(chat.close)
        chat.backends['openai'] = (openai, lambda: {'provider': 'openai', 'provider_label': 'OpenAI API',
                                                    'configured': True, 'model': 'test', 'setup': ''})
        status = chat.status()
        self.assertEqual(status['backend'], 'claude'); self.assertEqual(status['provider_label'], 'Claude CLI')
        self.assertEqual({o['backend']: o['configured'] for o in status['backends']}, {'claude': True, 'codex': False, 'openai': True})
        chat.send('hello'); status = self.wait(chat)
        self.assertEqual(status['messages'][-1]['provider'], 'Claude CLI')
        chat.set_backend('openai'); chat.send('and you?'); status = self.wait(chat)
        self.assertEqual([m.get('provider') for m in status['messages']], [None, 'Claude CLI', None, 'OpenAI API'])
        self.assertEqual(replies, [3])                         # the API saw the whole conversation
        with self.assertRaisesRegex(ValueError, 'Choose'):
            chat.set_backend('gemini')

    def test_no_switching_while_a_reply_is_running(self):
        chat = DashboardChat(backend='claude'); self.addCleanup(chat.close)
        chat.send('HANG_FOR_TEST')
        with self.assertRaisesRegex(ValueError, 'Wait for the current reply'):
            chat.set_backend('codex')
        chat.cancel(); status = self.wait(chat)
        self.assertEqual(status['backend'], 'claude')
        self.assertEqual(chat.set_backend('codex')['backend'], 'codex')
        with self.assertRaisesRegex(ValueError, 'codex'):       # Codex is not installed in this test
            chat.send('hi')


if __name__ == '__main__':
    unittest.main()
