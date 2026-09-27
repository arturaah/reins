"""Run with .venv/bin/python -m unittest tools.test_dashboard."""
import math
import io
import json
import os
import tempfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image
import time
import unittest
from pathlib import Path
from tools.dashboard import ROOT, CameraFeed, Runner, Simulation

# Stands in for tools/arm_lift.py: same CLI and output shape, never imports the robot SDK.
FAKE_TOOL = textwrap.dedent("""
    import json, os, sys, time
    with open(os.environ['FAKE_ARGS_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')
    print('take sample error', flush=True)
    print('fsm id: 811 = Start (balance control)   fsm mode: 0   (rpc codes 0, 0)', flush=True)
    if os.environ.get('FAKE_FAIL'): sys.exit('ABORT: speed cap')
    if '--execute' in sys.argv:
        try:
            print('EXECUTE: ramping weight up', flush=True); time.sleep(float(os.environ.get('FAKE_EXEC_S', '0')))
        except KeyboardInterrupt:
            print('interrupted: releasing', flush=True); sys.exit(130)
    print('DRY RUN, nothing published.', flush=True)
""")

# Tests must not depend on an operator's saved plan, which can be deleted.
TEST_PLAN = {'schema_version': 1, 'name': 'test_right_reach', 'duration_s': 2,
             'keyframes': [
                 {'time_s': 0, 'joint_targets_rad': {'right_shoulder_pitch_joint': 0}},
                 {'time_s': 2, 'joint_targets_rad': {'right_shoulder_pitch_joint': .2}},
             ]}


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation()
        cls.tmp = tempfile.TemporaryDirectory()
        plan = Path(cls.tmp.name) / 'right_reach.json'
        plan.write_text(json.dumps(TEST_PLAN))
        cls.sim.files['test/right_reach.json'] = plan

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.sim.select('test/right_reach.json')

    def test_plan_library_only_contains_validated_files(self):
        self.assertTrue(self.sim.plans)
        self.assertTrue(all(p['id'] in self.sim.files and p['duration'] > 0 for p in self.sim.plans))
        with self.assertRaises(ValueError):
            self.sim.control({'action': 'plan', 'id': '../../etc/passwd'})

    def test_seek_clamps_and_rejects_nonfinite(self):
        self.sim.control({'action': 'seek', 'time': 1e6})
        self.assertEqual(self.sim.position, self.sim.plan['duration_s'])
        self.sim.control({'action': 'seek', 'time': -10})
        self.assertEqual(self.sim.position, 0)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                self.sim.control({'action': 'seek', 'time': value})
        self.assertEqual(self.sim.position, 0)

    def test_play_at_end_restarts_and_pause_holds(self):
        self.sim.position = self.sim.plan['duration_s']
        self.sim.control({'action': 'play'})
        self.assertTrue(self.sim.playing)
        self.assertEqual(self.sim.position, 0)
        self.sim.control({'action': 'pause'})
        self.assertFalse(self.sim.playing)

    def test_speed_allowlist_and_no_execution_action(self):
        self.sim.control({'action': 'speed', 'value': .5})
        self.assertEqual(self.sim.speed, .5)
        for command in ({'action':'speed', 'value':-1}, {'action':'execute'}):
            with self.assertRaises(ValueError):
                self.sim.control(command)

    def test_hand_paths_are_finite_and_match_plan(self):
        self.assertEqual(set(self.sim.paths), {'left', 'right'})
        for points in self.sim.paths.values():
            self.assertEqual(len(points), 90)
            self.assertTrue(all(len(p) == 3 and all(math.isfinite(v) for v in p) for p in points))
        self.assertNotEqual(self.sim.paths['right'][0], self.sim.paths['right'][45])

    def test_mjpeg_reader_accepts_fragmented_jpeg(self):
        output = io.BytesIO()
        Image.new('RGB', (32, 24), (40, 90, 50)).save(output, 'JPEG')
        jpg = output.getvalue()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=test')
                self.end_headers()
                data = b'--test\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                for offset in range(0, len(data), 31):
                    self.wfile.write(data[offset:offset + 31])
                    self.wfile.flush()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            feed = CameraFeed(f'http://127.0.0.1:{server.server_port}/')
            deadline = time.monotonic() + 3
            while not feed.status()['online'] and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(feed.status()['online'])
            self.assertEqual(feed.jpg, jpg)
        finally:
            server.shutdown()
            server.server_close()

    def test_stale_feed_is_never_reported_online(self):
        feed = CameraFeed('')
        self.assertFalse(feed.status()['online'])
        feed.updated = time.monotonic()
        self.assertTrue(feed.status()['online'])
        feed.updated -= 4
        self.assertFalse(feed.status()['online'])


class RunnerTests(unittest.TestCase):
    """arm_lift.py is replaced by FAKE_TOOL; nothing here can reach the robot."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.PLAN = Path(self.tmp.name) / 'right_reach.json'
        self.PLAN.write_text(json.dumps(TEST_PLAN))
        tool = Path(self.tmp.name) / 'fake_arm_lift.py'
        tool.write_text(FAKE_TOOL)
        self.args_log = Path(self.tmp.name) / 'args.log'
        os.environ['FAKE_ARGS_LOG'] = str(self.args_log)
        for key in ('FAKE_FAIL', 'FAKE_EXEC_S'):
            os.environ.pop(key, None)
        self.exits = []
        self.runner = Runner('eth9', tool=tool, on_exit=lambda job, code: self.exits.append((job['kind'], code)))

    def tearDown(self):
        self.runner.shutdown(1)
        self.tmp.cleanup()

    def wait(self, count=1):
        deadline = time.monotonic() + 10
        while len(self.exits) < count and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(len(self.exits), count)

    def calls(self):
        return [json.loads(line) for line in self.args_log.read_text().splitlines()]

    def dry(self, speed=0.5, kp=1.2):
        self.runner.start('dry', 'tools/plans/cup_grab_right.json', self.PLAN, speed, kp)
        self.wait(len(self.exits) + 1)

    def test_dry_run_passes_settings_and_never_executes(self):
        self.dry()
        self.assertEqual(self.calls(), [['eth9', '--plan', str(self.PLAN), '--speed', '0.5', '--kp-scale', '1.2']])
        status = self.runner.status()
        self.assertEqual(status['exit'], 0)
        self.assertEqual(status['fsm'], {'id': 811, 'name': 'Start (balance control)', 'ok': True})
        self.assertEqual(status['cleared']['plan'], 'tools/plans/cup_grab_right.json')
        self.assertFalse(any('take sample error' in line for line in status['lines']))

    def test_execute_requires_fresh_dry_run_and_confirmation(self):
        with self.assertRaisesRegex(ValueError, 'dry run'):
            self.runner.start('execute', None, None, confirm=True)
        self.dry()
        with self.assertRaisesRegex(ValueError, 'confirmation'):
            self.runner.start('execute', None, None)
        self.runner.cleared['at'] -= Runner.DRY_RUN_VALID_S + 1
        with self.assertRaisesRegex(ValueError, 'dry run'):
            self.runner.start('execute', None, None, confirm=True)
        self.assertEqual(len(self.calls()), 1)

    def test_execute_uses_dry_run_settings_once(self):
        self.dry(speed=0.35, kp=1.5)
        self.runner.start('execute', 'ignored', ROOT / 'sim/plans/left_reach.json', speed=2, kp_scale=2, confirm=True)
        self.wait(2)
        self.assertEqual(self.calls()[1], ['eth9', '--plan', str(self.PLAN), '--speed', '0.35', '--kp-scale', '1.5', '--execute'])
        self.assertIsNone(self.runner.status()['cleared'])
        with self.assertRaisesRegex(ValueError, 'dry run'):
            self.runner.start('execute', None, None, confirm=True)

    def test_failed_dry_run_does_not_clear(self):
        os.environ['FAKE_FAIL'] = '1'
        self.dry()
        self.assertEqual(self.runner.status()['exit'], 1)
        self.assertIsNone(self.runner.status()['cleared'])

    def test_abort_interrupts_execute(self):
        self.dry()
        os.environ['FAKE_EXEC_S'] = '30'
        self.runner.start('execute', None, None, confirm=True)
        deadline = time.monotonic() + 5
        while not any('EXECUTE' in line for line in self.runner.status()['lines']) and time.monotonic() < deadline:
            time.sleep(.02)
        with self.assertRaisesRegex(ValueError, 'still active'):
            self.runner.start('dry', 'tools/plans/cup_grab_right.json', self.PLAN)
        self.assertTrue(self.runner.abort())
        self.wait(2)
        lines = self.runner.status()['lines']
        self.assertEqual(self.exits[-1], ('execute', 130))
        self.assertIn('interrupted: releasing', lines)
        self.assertFalse(self.runner.abort())

    def test_rejects_out_of_range_settings(self):
        for speed, kp in ((0, 1), (3, 1), (math.nan, 1), (1, 0.1), (1, 5), (1, math.inf)):
            with self.assertRaises(ValueError):
                self.runner.start('dry', 'tools/plans/cup_grab_right.json', self.PLAN, speed, kp)
        with self.assertRaises(ValueError):
            self.runner.start('teach', 'x', self.PLAN)
        self.assertFalse(self.args_log.exists())


if __name__ == '__main__':
    unittest.main()
