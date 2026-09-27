"""CLI bridge tests use a tiny local executable; no model calls or real CLI auth."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from core.codex_chat import CodexResponder
from core.dashboard_chat import DashboardChat, INSTRUCTIONS, SCHEMA


class CodexTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.binary=Path(self.tmp.name)/'fake-codex'
        self.binary.write_text('#!'+sys.executable+'''\nimport json,sys,time
if sys.argv[1:]==['login','status']:
    print('Logged in using ChatGPT');sys.exit(0)
text=sys.stdin.read()
if 'HANG_FOR_TEST' in text:time.sleep(10)
if 'FAIL_FOR_TEST' in text:
    print('private-token must not appear in browser',file=sys.stderr);sys.exit(1)
if 'LARGE_FOR_TEST' in text:
    print('x'*2200000);sys.exit(0)
answer={'reply':'CLI says hello.','robot_request':None}
if 'VERIFY_ARGS' in text:
    assert '--ignore-user-config' in sys.argv and '--ignore-rules' in sys.argv
    assert '--ephemeral' in sys.argv and '--json' in sys.argv
    assert sys.argv[sys.argv.index('--sandbox')+1]=='read-only'
    assert 'approval_policy="never"' in sys.argv
    assert sys.argv[-1]=='-' and 'VERIFY_ARGS' not in ' '.join(sys.argv)
    assert 'shell_tool' in sys.argv and 'hooks' in sys.argv and 'plugins' in sys.argv
print(json.dumps({'type':'item.completed','item':{'type':'reasoning','text':'private reasoning'}}))
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(answer)}}))
print(json.dumps({'type':'turn.completed','usage':{}}))
''')
        self.binary.chmod(0o700)
        self.env=patch.dict(os.environ,{'REINS_CODEX_BIN':str(self.binary)},clear=True)
        self.env.start();self.addCleanup(self.env.stop)

    def bridge(self,timeout=2):
        bridge=CodexResponder(INSTRUCTIONS,SCHEMA,timeout=timeout)
        self.addCleanup(bridge.close)
        return bridge

    def wait(self,chat):
        limit=time.monotonic()+4
        while chat.status()['busy'] and time.monotonic()<limit:time.sleep(.01)
        self.assertFalse(chat.status()['busy']);return chat.status()

    def test_signed_in_cli_needs_no_api_key_and_uses_stdin(self):
        bridge=self.bridge()
        self.assertTrue(bridge.configuration()['configured'])
        self.assertEqual(bridge.configuration()['provider'],'codex')
        result=bridge([{'role':'user','text':'VERIFY_ARGS; $(echo cannot-execute); `false`'}],{})
        self.assertEqual(result['reply'],'CLI says hello.')
        self.assertNotIn('private',json.dumps(result));self.assertIsNone(bridge.proc)

    def test_missing_binary_and_login_failure(self):
        with patch.dict(os.environ,{'REINS_CODEX_BIN':'/no-such-codex'}):
            self.assertFalse(self.bridge().configuration()['configured'])
        self.binary.write_text('#!'+sys.executable+'\nimport sys\nsys.exit(1)\n')
        bridge=self.bridge();self.assertFalse(bridge.configuration()['configured'])
        with self.assertRaisesRegex(ValueError,'codex login'):bridge([], {})

    def test_cli_is_dashboard_default_with_optional_api_backend(self):
        chat=DashboardChat();self.addCleanup(chat.close)
        self.assertEqual(chat.status()['provider'],'codex')
        chat.send('hello');status=self.wait(chat)
        self.assertEqual(status['messages'][-1]['text'],'CLI says hello.')
        self.assertFalse(DashboardChat(backend='openai').status()['configured'])
        with self.assertRaises(ValueError):DashboardChat(backend='unknown')

    def test_timeout_kills_and_reaps_the_child(self):
        bridge=self.bridge(timeout=.1)
        with self.assertRaisesRegex(ValueError,'timed out'):
            bridge([{'role':'user','text':'HANG_FOR_TEST'}],{})
        self.assertIsNone(bridge.proc)

    def test_cancel_keeps_message_for_retry_and_stops_process(self):
        chat=DashboardChat();self.addCleanup(chat.close)
        chat.send('HANG_FOR_TEST')
        deadline=time.monotonic()+2
        while chat.responder.proc is None and time.monotonic()<deadline:time.sleep(.01)
        proc=chat.responder.proc;self.assertIsNotNone(proc)
        chat.cancel();status=self.wait(chat)
        self.assertEqual(len(status['messages']),1)
        self.assertIn('stopped',status['error'])
        self.assertIsNotNone(proc.poll())
        chat.clear();chat.send('Hello again');status=self.wait(chat)
        self.assertEqual(status['messages'][-1]['role'],'assistant')

    def test_clear_discards_inflight_reply_and_close_prevents_new_process(self):
        chat=DashboardChat();self.addCleanup(chat.close)
        chat.send('HANG_FOR_TEST');chat.clear();status=self.wait(chat)
        self.assertFalse(status['messages']);self.assertIsNone(status['error'])
        bridge=chat.responder;bridge.close()
        with self.assertRaisesRegex(ValueError,'shutting down'):bridge([], {})

    def test_failure_and_output_limit_are_bounded_and_sanitized(self):
        bridge=self.bridge()
        for text,match in [('FAIL_FOR_TEST','could not finish'),('LARGE_FOR_TEST','too much output')]:
            with self.assertRaisesRegex(ValueError,match) as exc:
                bridge([{'role':'user','text':text}],{})
            self.assertNotIn('private-token',str(exc.exception))
            self.assertIsNone(bridge.proc)

    def test_incomplete_or_invalid_results_are_rejected(self):
        for output in ['{}','{"type":"turn.completed"}']:
            with self.assertRaises(ValueError):CodexResponder._parse(output,'',0)
        for diagnostic,match in [('401 unauthorized','sign-in'),('429 rate limit','usage limit'),('unexpected argument --ephemeral','version')]:
            with self.assertRaisesRegex(ValueError,match):CodexResponder._parse('',diagnostic,1)


if __name__=='__main__':unittest.main()
