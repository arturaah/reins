"""Exercise the real local dashboard HTTP handlers with a fake firmware transport."""
import json
from functools import partial
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from core.dashboard_chat import DashboardChat
from core.prompt_planner import PromptPlanner
from core.r1_gestures import GestureController, parse_presets
from core.test_generated_motion import ANSWER
from core.test_trajectory_revision import UNREACHABLE
from tools import dashboard


class DashboardHTTPTests(unittest.TestCase):
    def test_gestures_and_one_time_preview_with_no_legacy_endpoints(self):
        firmware=[]
        actions=parse_presets([[{'id':27,'name':'shake_hand'},{'id':99,'name':'release_arm'}],[]])
        def request(iface,action_id):
            firmware.append(action_id)
            return {'actions':actions}
        controller=GestureController('fake-interface',request=request)
        revision_calls=[]
        def provider(messages,context):
            if 'trajectory_revision' in context:
                revision_calls.append(context)
                return ANSWER
            return {**ANSWER,'trajectory':UNREACHABLE}
        conversation=DashboardChat(responder=provider)
        conversation.configuration=lambda:{'configured':True,'provider':'test','provider_label':'Test','model':'test'}
        serve=dashboard.ThreadingHTTPServer.serve_forever
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as logs:
            planner=PromptPlanner(output_dir=tmp)
            def exercise(server):
                thread=threading.Thread(target=serve,args=(server,),daemon=True)
                thread.start()
                base='http://127.0.0.1:'+str(server.server_port)
                token=None
                def call(path,body=None,authenticated=True):
                    headers={'Content-Type':'application/json'}
                    if token and authenticated:headers['X-Reins-Token']=token
                    req=urllib.request.Request(base+path,json.dumps(body).encode() if body is not None else None,headers=headers)
                    try:
                        with urllib.request.urlopen(req,timeout=5) as response:return response.status,json.load(response)
                    except urllib.error.HTTPError as exc:return exc.code,json.load(exc)
                def until(path,done):
                    deadline=time.monotonic()+15
                    while time.monotonic()<deadline:
                        result=call(path)[1]
                        if done(result):return result
                        time.sleep(.01)
                    self.fail('Timed out waiting for '+path)
                try:
                    token=call('/api/session')[1]['token']
                    until('/api/gestures',lambda x:not x['busy'])
                    self.assertEqual(firmware,[None]) # startup only discovers; no movement
                    for path in ['/api/plans','/api/run','/api/abort-beacon']:
                        self.assertEqual(call(path)[0],404)
                    self.assertEqual(call('/api/run',{'action':'execute','confirm':True})[0],404)
                    self.assertEqual(call('/api/gestures',{'action':'gesture','id':27},False)[0],403)
                    self.assertEqual(call('/api/gestures',{'action':'gesture','id':100})[0],400)
                    self.assertEqual(call('/api/gestures',{'action':'gesture','id':27})[0],200)
                    until('/api/gestures',lambda x:not x['busy'])
                    self.assertEqual(firmware,[None,27])

                    self.assertEqual(call('/api/chat',{'message':'Blow a kiss'})[0],200)
                    reply=until('/api/chat',lambda x:not x['busy'])['messages'][-1]
                    self.assertEqual(call('/api/prompt',{'action':'submit','chat_message_id':reply['id']})[0],200)
                    result=until('/api/prompt',lambda x:x['state']!='planning')
                    self.assertEqual(result['state'],'proposed',result['message'])
                    self.assertEqual(result['attempt'],2)
                    self.assertEqual(len(revision_calls),1)
                    self.assertRegex(revision_calls[0]['trajectory_revision']['failures'][0]['error'],'unreachable|joints jump')
                    self.assertEqual(call('/api/prompt',{'action':'preview','id':result['id']})[0],200)
                    status=call('/api/status')[1]
                    self.assertTrue(status['simulation']['playing'])
                    self.assertNotIn('run',status)
                    self.assertEqual(call('/api/prompt',{'action':'preview','id':result['id']})[0],400)
                    self.assertEqual(call('/api/control',{'action':'seek','time':0})[0],400)
                    self.assertEqual(call('/api/control',{'action':'play'})[0],400)
                    self.assertEqual(call('/api/control',{'action':'stop'})[0],200)
                    self.assertFalse(call('/api/status')[1]['simulation']['playing'])
                    self.assertFalse(list(Path(tmp).iterdir()))
                    self.assertEqual(firmware,[None,27]) # simulation never reaches firmware
                finally:
                    server.shutdown();thread.join(2)
            with patch.object(dashboard.ThreadingHTTPServer,'serve_forever',exercise), \
                 patch.object(dashboard,'GestureController',return_value=controller), \
                 patch.object(dashboard,'DashboardChat',return_value=conversation), \
                 patch.object(dashboard,'PromptPlanner',return_value=planner), \
                 patch.object(dashboard,'RobotPipeline',partial(dashboard.RobotPipeline,run_dir=logs)), \
                 patch.object(dashboard.Simulation,'run'), \
                 patch.object(dashboard,'make_detector',return_value=SimpleNamespace(name='test',available=False,confidence=.4)), \
                 patch('sys.argv',['dashboard','--port','0','--glasses-host','127.0.0.1','--glasses-port','0','--head','','--left-wrist','','--right-wrist','','--twin','']):
                dashboard.main()


if __name__=='__main__':unittest.main()
