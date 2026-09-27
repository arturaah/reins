"""R1 gesture protocol tests. All DDS calls are fake; no hardware is contacted."""
import json
import threading
import time
import unittest
from unittest.mock import Mock, patch

from core.r1_gestures import (EXECUTE_PRESET, GET_ACTION_LIST, GestureController,
                              check_result, parse_presets, request_firmware, run_helper)

RAW = [[{'id':27,'name':'shake_hand'},{'id':99,'name':'release_arm'}],
       [{'name':'recorded_custom_motion','time':12.3}]]


def client(raw=RAW):
    result=Mock()
    result._Call.side_effect=[(0,json.dumps(raw)),(0,'')]
    return result


class FirmwareTests(unittest.TestCase):
    def test_discovery_only_reads_presets_and_excludes_recordings(self):
        rpc=client()
        result=request_firmware(rpc)
        rpc._Call.assert_called_once_with(GET_ACTION_LIST,'{}')
        self.assertEqual(result['actions'],[
            {'id':27,'name':'shake_hand','label':'Shake hand'},
            {'id':99,'name':'release_arm','label':'Release arms'}])
        self.assertNotIn('recorded',json.dumps(result))

    def test_execute_uses_r1_preset_payload_after_fresh_discovery(self):
        rpc=client()
        result=request_firmware(rpc,27)
        self.assertEqual(result['completed_id'],27)
        self.assertEqual(rpc._Call.call_args_list[1].args,(7106,'{"action_id": 27}'))
        rpc.SetTimeout.assert_called_once_with(60.)

    def test_missing_stale_or_custom_action_cannot_execute(self):
        for action in [100,-1,999,'shake_hand',True,{}]:
            rpc=client()
            with self.assertRaises(ValueError):request_firmware(rpc,action)
            self.assertEqual(rpc._Call.call_count,1)
        rpc=client([[{'id':99,'name':'release_arm'}],[]])
        with self.assertRaises(ValueError):request_firmware(rpc,27)
        self.assertEqual(rpc._Call.call_count,1)

    def test_firmware_refusals_are_preserved(self):
        for code,word in [(7400,'busy'),(7401,'Release arms'),(7404,'mode'),(7406,'battery'),(7407,'motor')]:
            rpc=client();rpc._Call.side_effect=[(0,json.dumps(RAW)),(code,'private')]
            with self.assertRaisesRegex(ValueError,word):request_firmware(rpc,27)
        rpc=client();rpc._Call.side_effect=[(3102,'')]
        with self.assertRaisesRegex(ValueError,'connection'):request_firmware(rpc)

    def test_invalid_catalogs_rejected_and_reserved_ids_filtered(self):
        for raw in [None,'not json',{},[],[[],[],[]],[[{'id':True,'name':'fake'}],[]],
                    [[{'id':27,'name':'one'},{'id':27,'name':'two'}],[]],
                    [[{'id':27,'name':''}],[]]]:
            with self.assertRaises(ValueError):parse_presets(raw)
        self.assertEqual(parse_presets([[{'id':100,'name':'custom'},{'id':-1,'name':'teaching'}],[]]),[])

    def test_helper_uses_fixed_argv_no_shell_or_recording_flags(self):
        with patch('subprocess.run',return_value=Mock(returncode=0,stdout='SDK startup\n'+json.dumps({'actions':parse_presets(RAW)}))) as run:
            result=run_helper('eth-test',27)
        args=run.call_args.args[0]
        self.assertEqual(args[-3:],['eth-test','--action','27'])
        self.assertNotIn('shell',run.call_args.kwargs)
        self.assertEqual(len(result['actions']),2)


class ControllerTests(unittest.TestCase):
    def wait(self,controller):
        deadline=time.monotonic()+3
        while controller.status()['busy'] and time.monotonic()<deadline:time.sleep(.01)
        self.assertFalse(controller.status()['busy'])
        return controller.status()

    def test_only_explicit_buttons_dispatch_motion_and_busy_is_serialized(self):
        entered,release=threading.Event(),threading.Event()
        calls=[]
        def request(iface,action):
            calls.append((iface,action))
            if action is not None: entered.set(); release.wait(2)
            return {'actions':parse_presets(RAW)}
        c=GestureController('eth-test',request=request)
        self.assertFalse(calls)
        with self.assertRaises(ValueError):c.command({'action':'gesture','id':27})
        c.command({'action':'refresh'});self.assertTrue(self.wait(c)['connected'])
        self.assertEqual(calls,[('eth-test',None)])
        c.command({'action':'gesture','id':27});self.assertTrue(entered.wait(1))
        self.addCleanup(release.set)
        with self.assertRaisesRegex(ValueError,'finish'):c.command({'action':'gesture','id':99})
        release.set();result=self.wait(c)
        self.assertIn('finished',result['message'])
        c.command({'action':'gesture','id':99});self.assertIn('released',self.wait(c)['message'])
        c.close()
        with self.assertRaisesRegex(ValueError,'shutting down'):c.command({'action':'refresh'})

    def test_failed_connection_clears_buttons_and_does_not_fallback(self):
        request=Mock(side_effect=[{'actions':parse_presets(RAW)},ValueError('robot offline')])
        c=GestureController('eth-test',request=request)
        c.command({'action':'refresh'});self.wait(c)
        c.command({'action':'gesture','id':27});result=self.wait(c)
        self.assertEqual(result['error'],'robot offline')
        self.assertFalse(result['connected']);self.assertFalse(result['actions'])
        with self.assertRaises(ValueError):c.command({'action':'gesture','id':27})
        self.assertEqual(request.call_count,2)

    def test_arbitrary_execution_and_recording_requests_are_rejected(self):
        request=Mock()
        c=GestureController('eth-test',request=request)
        for command in [{'action':'execute','plan':'x'}, {'action':'record'}, {'action':'custom','name':'x'},None]:
            with self.assertRaises(ValueError):c.command(command)
        request.assert_not_called()


if __name__=='__main__':unittest.main()
