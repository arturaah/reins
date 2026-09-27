"""Conversation and transport tests; provider mocked, no credentials or hardware needed."""
import io
import json
import os
import threading
import time
import unittest
import urllib.error
from unittest.mock import Mock, patch

from core.dashboard_chat import DashboardChat, configuration, respond_openai, validate_reply

SETTINGS={'OPENAI_API_KEY':'test-only-secret','REINS_CHAT_MODEL':'test-model'}
REPLY={'reply':'Hello. What would you like to work on?', 'robot_request':None}


class TransportTests(unittest.TestCase):
    def provider(self, answer=REPLY, status='completed'):
        response=Mock()
        response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        response.read.return_value=json.dumps({'status':status,'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(answer)}]}]}).encode()
        return response

    def test_api_uses_text_history_without_images_tools_or_stored_state(self):
        messages=[{'role':'user','text':'My name is Chris.'},{'role':'assistant','text':'Hello Chris.','robot_request':None},{'role':'user','text':'What is my name?'}]
        with patch.dict(os.environ,SETTINGS,clear=True),patch('urllib.request.urlopen',return_value=self.provider()) as call:
            self.assertEqual(respond_openai(messages,{'cameras':{'head':{'online':True}}}),REPLY)
        request=call.call_args.args[0];payload=json.loads(request.data)
        self.assertFalse(payload['store']);self.assertNotIn('tools',payload)
        self.assertNotIn('input_image',json.dumps(payload))
        self.assertEqual(payload['model'],'test-model')
        self.assertEqual([m['role'] for m in payload['input']],['developer','user','assistant','user'])
        self.assertEqual(payload['input'][-1]['content'],'What is my name?')
        self.assertTrue(payload['text']['format']['strict'])

    def test_missing_configuration_fails_before_network(self):
        with patch.dict(os.environ,{},clear=True),patch('urllib.request.urlopen') as call:
            with self.assertRaisesRegex(ValueError,'OPENAI_API_KEY'):respond_openai([], {})
            call.assert_not_called()

    def test_vision_model_fallback_and_chat_override(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'test','REINS_VISION_MODEL':'vision'},clear=True):
            self.assertEqual(configuration()['model'],'vision')
            with patch.dict(os.environ,{'REINS_CHAT_MODEL':'chat'}):
                self.assertEqual(configuration()['model'],'chat')
        self.assertNotIn('key',configuration())

    def test_provider_error_is_sanitized(self):
        error=urllib.error.HTTPError('https://api.openai.com/v1/responses',401,'bad',{},io.BytesIO(b'private provider response test-only-secret'))
        with patch.dict(os.environ,SETTINGS,clear=True),patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(ValueError,'HTTP 401') as exc:respond_openai([], {})
            self.assertNotIn('test-only-secret',str(exc.exception));self.assertNotIn('private',str(exc.exception))

    def test_incomplete_refused_and_invalid_responses_do_not_offer_actions(self):
        with patch.dict(os.environ,SETTINGS,clear=True):
            with patch('urllib.request.urlopen',return_value=self.provider(status='incomplete')):
                with self.assertRaisesRegex(ValueError,'did not finish'):respond_openai([], {})
            response=self.provider();response.read.return_value=json.dumps({'status':'completed','output':[{'type':'message','content':[{'type':'refusal','refusal':'reason'}]}]}).encode()
            with patch('urllib.request.urlopen',return_value=response):
                self.assertIsNone(respond_openai([], {})['robot_request'])
            with patch('urllib.request.urlopen',return_value=self.provider({'reply':'ok','robot_request':{'execute':True}})):
                with self.assertRaises(ValueError):respond_openai([], {})

    def test_unsupported_suggestion_is_removed_but_answer_is_kept(self):
        reply=validate_reply({'reply':'We can discuss walking.', 'robot_request':'walk forwards'})
        self.assertIn('We can discuss walking.',reply['reply']);self.assertIsNone(reply['robot_request'])
        self.assertEqual(validate_reply({'reply':'Here is a suggestion.', 'robot_request':'wave your left hand'})['robot_request'],'wave your left hand')


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.settings=patch.dict(os.environ,SETTINGS,clear=True);self.settings.start();self.addCleanup(self.settings.stop)

    def wait(self,chat):
        limit=time.monotonic()+3
        while chat.status()['busy'] and time.monotonic()<limit:time.sleep(.01)
        self.assertFalse(chat.status()['busy']);return chat.status()

    def test_multiturn_context_and_read_only_suggestion(self):
        provider=Mock(side_effect=[REPLY,{'reply':'You can prepare a wave.', 'robot_request':'wave your right hand'}])
        context=Mock(return_value={'robot_run_active':False})
        chat=DashboardChat(context,provider)
        with patch('core.prompt_planner.PromptPlanner.submit') as planner:
            chat.send('Hello');self.wait(chat)
            chat.send('Please prepare a wave');status=self.wait(chat)
            planner.assert_not_called()
        self.assertEqual(len(status['messages']),4)
        self.assertEqual(status['messages'][-1]['robot_request'],'wave your right hand')
        self.assertEqual(len(provider.call_args.args[0]),3)
        self.assertEqual(provider.call_args.args[1],{'robot_run_active':False})
        status['messages'].clear();self.assertEqual(len(chat.status()['messages']),4)

    def test_only_one_request_at_a_time_clear_discards_late_reply(self):
        entered=threading.Event();release=threading.Event()
        def response(*args):entered.set();release.wait(2);return REPLY
        chat=DashboardChat(responder=response)
        self.addCleanup(release.set)
        chat.send('Hello');self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(ValueError,'still replying'):chat.send('Second')
        chat.clear()
        self.assertTrue(chat.status()['busy']);self.assertFalse(chat.status()['messages'])
        with self.assertRaisesRegex(ValueError,'still replying'):chat.send('No extra concurrent call')
        release.set();status=self.wait(chat)
        self.assertFalse(status['messages']);self.assertIsNone(status['error'])

    def test_failed_message_is_kept_and_retry_does_not_duplicate_it(self):
        provider=Mock(side_effect=[ValueError('Provider unavailable'),REPLY])
        chat=DashboardChat(responder=provider);chat.send('Hello');status=self.wait(chat)
        self.assertEqual(status['error'],'Provider unavailable');self.assertEqual(len(status['messages']),1)
        chat.retry();status=self.wait(chat)
        self.assertIsNone(status['error']);self.assertEqual(len(status['messages']),2)
        with self.assertRaises(ValueError):chat.retry()

    def test_history_is_bounded_and_new_chat_resets_it(self):
        chat=DashboardChat(responder=Mock(return_value=REPLY));chat.MAX_MESSAGES=4
        for message in ['first','second','third']:
            chat.send(message);self.wait(chat)
        status=chat.status();self.assertEqual(len(status['messages']),4)
        self.assertEqual(status['messages'][0]['text'],'second');self.assertTrue(status['trimmed'])
        chat.clear();self.assertFalse(chat.status()['messages']);self.assertFalse(chat.status()['trimmed'])

    def test_invalid_messages_and_configuration_preserve_conversation(self):
        chat=DashboardChat(responder=Mock(return_value=REPLY))
        for message in ['',None,{},'x'*4001]:
            with self.assertRaises(ValueError):chat.send(message)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'OPENAI_API_KEY'):chat.send('Hello')
        self.assertFalse(chat.status()['messages']);chat.responder.assert_not_called()


if __name__=='__main__':unittest.main()
