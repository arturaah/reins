"""Human preset buttons share Accept; tests use only a fake firmware RPC."""
import tempfile
import threading
import time
import unittest

from core.prompt_planner import PromptPlanner
from core.r1_gestures import GestureController, parse_presets
from core.robot_pipeline import RobotPipeline
from tools.dashboard import Simulation


class FirmwareApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sim = Simulation()
        self.pipe = RobotPipeline(PromptPlanner(), self.sim, {}, run_dir=self.tmp.name)
        self.sent = []
        self.actions = parse_presets([[{'id': 27, 'name': 'shake_hand'}, {'id': 99, 'name': 'release_arm'}], []])
        def request(iface, action_id):
            self.sent.append(action_id)
            return {'actions': self.actions}
        self.gestures = GestureController('fake', request=request)
        self.gestures.actions = self.actions
        self.gestures.connected = True

    def tearDown(self):
        self.pipe.close()
        self.gestures.close()
        self.tmp.cleanup()

    def choose(self):
        self.pipe.firmware({'action': 'gesture', 'id': 27}, self.gestures)
        return self.pipe.status()['proposal']

    def wait(self, state):
        until = time.monotonic()+3
        while time.monotonic() < until:
            if self.pipe.status()['state'] == state and not self.pipe.status()['busy']:
                return self.pipe.status()
            time.sleep(.01)
        self.fail(str(self.pipe.status()))

    def test_selection_and_glasses_review_do_not_dispatch_until_single_accept(self):
        p = self.choose()
        self.assertEqual(self.sent, [])
        self.assertEqual(p['mode'], 'live')
        self.assertIsNone(p['duration_s'])
        self.assertFalse(self.sim.playing)
        ar = self.pipe.glasses_message()
        self.assertEqual(ar['review']['id'], p['id'])
        self.assertEqual(ar['hands'], {'left': [], 'right': []})
        self.pipe.decide(p['id'], p['digest'], 'approve')
        result = self.wait('completed')['last_result']
        self.assertEqual(self.sent, [27])
        self.assertEqual(result['feedback']['source'], 'firmware_rpc')
        with self.assertRaises(ValueError):
            self.pipe.decide(p['id'], p['digest'], 'approve')
        self.assertEqual(self.sent, [27])

    def test_cancel_expiry_and_wrong_digest_cannot_dispatch(self):
        p = self.choose()
        with self.assertRaises(ValueError):
            self.pipe.decide(p['id'], 'changed', 'approve')
        self.pipe.stop()
        with self.assertRaises(ValueError):
            self.pipe.decide(p['id'], p['digest'], 'approve')
        p = self.choose()
        self.pipe.proposal['expires_at'] = time.time()-1
        with self.assertRaises(ValueError):
            self.pipe.decide(p['id'], p['digest'], 'approve')
        self.assertEqual(self.sent, [])

    def test_changed_payload_or_firmware_identity_cannot_dispatch(self):
        p = self.choose()
        self.pipe.plan['action_id'] = 99
        self.pipe.decide(p['id'], p['digest'], 'approve')
        self.wait('failed')
        p = self.choose()
        self.gestures.actions = [{**a, 'name': 'different'} for a in self.actions]
        self.pipe.decide(p['id'], p['digest'], 'approve')
        self.wait('failed')
        self.assertEqual(self.sent, [])

    def test_stop_during_onboard_rpc_does_not_claim_physical_stop(self):
        started, release = threading.Event(), threading.Event()
        def request(iface, action_id):
            self.sent.append(action_id); started.set(); release.wait(3)
            return {'actions': self.actions}
        self.gestures.request = request
        self.addCleanup(release.set)
        p = self.choose()
        self.pipe.decide(p['id'], p['digest'], 'approve')
        self.assertTrue(started.wait(1))
        self.pipe.stop()
        self.assertIn('Unitree controller', self.pipe.status()['message'])
        self.assertEqual(self.sent, [27])
        with self.assertRaises(ValueError):
            self.choose()
        release.set()
        self.wait('stopped')

    def test_stop_during_last_preset_check_prevents_dispatch(self):
        p = self.choose()
        checking, release = threading.Event(), threading.Event()
        status = self.gestures.status
        def held_status():
            if threading.current_thread() is self.pipe.worker:
                checking.set(); release.wait(3)
            return status()
        self.gestures.status = held_status
        self.addCleanup(release.set)
        self.pipe.decide(p['id'], p['digest'], 'approve')
        self.assertTrue(checking.wait(1))
        stopping = threading.Thread(target=self.pipe.stop)
        stopping.start()
        self.assertTrue(self.pipe.cancelled.wait(1))
        release.set(); stopping.join(3)
        self.assertFalse(stopping.is_alive())
        self.wait('stopped')
        self.assertEqual(self.sent, [])


if __name__ == '__main__':
    unittest.main()
