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
    def test_automatic_preview_single_accept_and_paired_voice_pinch(self):
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
                def receive_until(socket,done):
                    deadline=time.monotonic()+10
                    while time.monotonic()<deadline:
                        result=json.loads(socket.recv(timeout=max(.01,deadline-time.monotonic())))
                        if done(result):return result
                    self.fail('Timed out waiting for glasses message')
                try:
                    token=call('/api/session')[1]['token']
                    until('/api/gestures',lambda x:not x['busy'])
                    self.assertEqual(firmware,[]) # --sim never even initializes firmware discovery
                    status=call('/api/status')[1]
                    self.assertEqual(status['mode'],'sim')
                    self.assertEqual(status['voice_url'],'http://127.0.0.1:8770/')
                    self.assertEqual(call('/api/robot',{'action':'connect','table_z_m':.6})[0],400)
                    self.assertEqual(call('/api/gestures',{'action':'refresh'})[0],400)
                    for path in ['/api/plans','/api/run','/api/abort-beacon']:
                        self.assertEqual(call(path)[0],404)
                    self.assertEqual(call('/api/run',{'action':'execute','confirm':True})[0],404)
                    self.assertEqual(call('/api/gestures',{'action':'gesture','id':27},False)[0],403)
                    self.assertEqual(call('/api/gestures',{'action':'gesture','id':100})[0],400)
                    self.assertEqual(call('/api/gestures',{'action':'gesture','id':27})[0],400)
                    until('/api/gestures',lambda x:not x['busy'])
                    self.assertEqual(firmware,[])

                    self.assertEqual(call('/api/chat',{'message':'Blow a kiss'})[0],200)
                    until('/api/chat',lambda x:not x['busy'])
                    result=until('/api/prompt',lambda x:x['state']!='planning')
                    self.assertEqual(result['state'],'proposed',result['message'])
                    self.assertEqual(result['attempt'],2)
                    self.assertEqual(len(revision_calls),1)
                    self.assertRegex(revision_calls[0]['trajectory_revision']['failures'][0]['error'],'unreachable|joints jump')
                    runtime=until('/api/robot',lambda x:x['state'] in ('review','blocked'))
                    self.assertEqual(runtime['state'],'review',runtime['message'])
                    self.assertIsNone(runtime['last_result']) # preview is not execution
                    status=until('/api/status',lambda x:x['simulation']['playing'] or x['pipeline']['state']=='blocked')
                    self.assertTrue(status['simulation']['playing'],status['pipeline']['message'])
                    self.assertNotIn('run',status)
                    proposal=runtime['proposal']
                    self.assertEqual(proposal['mode'],'sim')
                    approval={'action':'decision','id':proposal['id'],'digest':proposal['digest'],'decision':'approve'}
                    self.assertEqual(call('/api/robot',approval,False)[0],403)
                    self.assertEqual(call('/api/robot',{'action':'decision','id':proposal['id'],'digest':'changed',
                                                        'decision':'approve'})[0],400)
                    self.assertEqual(call('/api/robot',approval)[0],200)
                    completed=until('/api/robot',lambda x:x['state']=='completed')
                    self.assertEqual(completed['last_result']['outcome'],'executed')
                    self.assertEqual(call('/api/experience')[1]['count'],1)
                    self.assertEqual(call('/api/experience',{'action':'forget'},False)[0],403)
                    self.assertEqual(call('/api/experience')[1]['count'],1)
                    self.assertEqual(call('/api/experience',{'action':'forget'})[1]['count'],0)
                    self.assertEqual(call('/api/robot',approval)[0],400)
                    self.assertEqual(call('/api/glasses',{'action':'create_device','label':'Lab glasses'},False)[0],403)
                    code,paired=call('/api/glasses',{'action':'create_device','label':'Lab glasses'})
                    self.assertEqual(code,200)
                    device=paired['device']
                    self.assertTrue(device['token'])
                    metadata=call('/api/glasses')[1]
                    self.assertNotIn(device['token'],json.dumps(metadata))
                    self.assertEqual(metadata['paired_devices'][0]['device_id'],device['device_id'])
                    from websockets.sync.client import connect
                    with connect('ws://127.0.0.1:'+str(metadata['port'])) as socket:
                        socket.send(json.dumps({'type':'authenticate','device_id':device['device_id'],'token':device['token']}))
                        authenticated=json.loads(socket.recv(timeout=3))
                        self.assertTrue(authenticated['accepted'])
                        voice={'type':'voice_command','version':1,'id':'voice-once','text':'Wave at me',
                               'session':authenticated['session']}
                        for _ in range(2):
                            socket.send(json.dumps(voice))
                            answer=receive_until(socket,lambda x:x['type']=='voice_ack')
                            self.assertTrue(answer['accepted'],answer)
                        conversation_state=until('/api/chat',lambda x:not x['busy'])
                        self.assertEqual([m['text'] for m in conversation_state['messages'] if m['role']=='user'],
                                         ['Blow a kiss','Wave at me'])
                        voiced=until('/api/robot',lambda x:x['state'] in ('review','blocked'))
                        self.assertEqual(voiced['state'],'review',voiced['message'])
                        self.assertEqual(voiced['last_result']['proposal_id'],proposal['id']) # voice cannot approve
                        reviewed=voiced['proposal']
                        displayed=receive_until(socket,lambda x:(x.get('review') or {}).get('id')==reviewed['id'])
                        self.assertEqual(displayed['phase'],'review')
                        self.assertEqual(displayed['review']['digest'],reviewed['digest'])
                        self.assertTrue(displayed['hands']['right'])
                        pinch={'type':'review_decision','version':1,'id':reviewed['id'],'digest':reviewed['digest'],
                               'revision':reviewed['revision'],'decision':'approve','session':authenticated['session'],
                               'tracking':{'registered':True,'age_s':0}}
                        socket.send(json.dumps(pinch))
                        answer=receive_until(socket,lambda x:x['type']=='review_ack')
                        self.assertTrue(answer['accepted'],answer)
                        completed=until('/api/robot',lambda x:x['state']=='completed')
                        self.assertEqual(completed['last_result']['proposal_id'],reviewed['id'])
                        self.assertEqual(completed['last_result']['outcome'],'executed')
                        socket.send(json.dumps(pinch))
                        answer=receive_until(socket,lambda x:x['type']=='review_ack')
                        self.assertFalse(answer['accepted'])
                    self.assertEqual(call('/api/glasses',{'action':'revoke_device','device_id':device['device_id']})[0],200)
                    self.assertEqual(call('/api/control',{'action':'seek','time':0})[0],400)
                    self.assertEqual(call('/api/control',{'action':'play'})[0],400)
                    self.assertEqual(call('/api/control',{'action':'stop'})[0],200)
                    self.assertFalse(call('/api/status')[1]['simulation']['playing'])
                    self.assertFalse(list(Path(tmp).iterdir()))
                    self.assertEqual(firmware,[]) # simulation never reaches firmware
                finally:
                    server.shutdown();thread.join(2)
            with patch.object(dashboard.ThreadingHTTPServer,'serve_forever',exercise), \
                 patch.object(dashboard,'GestureController',return_value=controller), \
                 patch.object(dashboard,'DashboardChat',return_value=conversation), \
                 patch.object(dashboard,'PromptPlanner',return_value=planner), \
                 patch.object(dashboard,'RobotPipeline',partial(dashboard.RobotPipeline,run_dir=logs)), \
                 patch.object(dashboard.Simulation,'run'), \
                 patch.object(dashboard,'make_detector',return_value=SimpleNamespace(name='test',available=False,confidence=.4)), \
                 patch.dict('os.environ',{'REINS_STATE_DIR':logs}), \
                 patch('sys.argv',['dashboard','--sim','--voice-url','http://127.0.0.1:8770/','--port','0','--glasses-host','127.0.0.1','--glasses-port','0','--head','','--left-wrist','','--right-wrist','','--twin','']):
                dashboard.main()


if __name__=='__main__':unittest.main()
