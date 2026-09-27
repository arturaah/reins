"""Run with .venv/bin/python -m unittest tools.test_dashboard."""
import math
import errno
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
from unittest.mock import patch
from tools.dashboard import ROOT, CameraFeed, Simulation, bind_dashboard_server

class DashboardTests(unittest.TestCase):
    def setUp(self):
        from core.action_context import gesture_plan
        from core.ik import ArmIK
        self.sim = Simulation()
        self.plan = gesture_plan(ArmIK(model=self.sim.model,backend='mujoco'),'raise_arm','right',{})
        self.plan.update(preview_only=True,prompt_proposal={'id':'new-motion'},scene_boxes=[])

    def test_starts_neutral_without_loading_a_library(self):
        self.assertIsNone(self.sim.status()['plan'])
        self.assertFalse(self.sim.playing)
        self.assertFalse(hasattr(self.sim,'files'))
        self.assertFalse(hasattr(self.sim,'scan'))

    def test_new_proposal_shows_once_and_cannot_be_replayed(self):
        self.sim.show_proposal(self.plan,'new-motion')
        self.assertTrue(self.sim.playing)
        self.sim.advance(self.plan['duration_s']+1)
        self.assertFalse(self.sim.playing)
        self.assertEqual(self.sim.position,self.plan['duration_s'])
        with self.assertRaisesRegex(ValueError,'already been shown'):
            self.sim.show_proposal(self.plan,'new-motion')

    def test_only_validated_proposals_are_loaded(self):
        for plan,pid in [({**self.plan,'preview_only':False},'new-motion'),(self.plan,'wrong-id')]:
            with self.assertRaises(ValueError):self.sim.show_proposal(plan,pid)
        bad=json.loads(json.dumps(self.plan));bad['keyframes'][1]['time_s']=float('nan')
        with self.assertRaises(ValueError):self.sim.show_proposal(bad,'new-motion')

    def test_stop_freezes_preview_and_old_replay_commands_are_removed(self):
        self.sim.show_proposal(self.plan,'new-motion')
        self.sim.advance(.5)
        self.sim.control({'action':'stop'})
        self.sim.advance(10)
        self.assertEqual(self.sim.position,.5)
        for action in ['play','pause','seek','speed','plan','execute','record']:
            with self.assertRaises(ValueError):self.sim.control({'action':action})

    def test_hand_paths_are_finite_and_match_generated_plan(self):
        self.sim.show_proposal(self.plan,'new-motion')
        for points in self.sim.paths.values():
            self.assertEqual(len(points),90)
            self.assertTrue(all(len(p)==3 and all(math.isfinite(v) for v in p) for p in points))
        self.assertNotEqual(self.sim.paths['right'][0],self.sim.paths['right'][-1])

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


class DashboardPortTests(unittest.TestCase):
    def test_default_skips_occupied_port(self):
        with patch('tools.dashboard.ThreadingHTTPServer') as factory:
            factory.side_effect = [OSError(errno.EADDRINUSE, 'busy'), factory.return_value]
            server = bind_dashboard_server()
            self.assertEqual([call.args[0] for call in factory.call_args_list],
                             [('127.0.0.1',8090),('127.0.0.1',8091)])
            self.assertIs(server, factory.return_value)

    def test_explicit_port_does_not_change_silently(self):
        with patch('tools.dashboard.ThreadingHTTPServer', side_effect=OSError(errno.EADDRINUSE, 'busy')) as factory:
            with self.assertRaisesRegex(OSError, 'Port 8123.*--port 0'):
                bind_dashboard_server(8123)
            factory.assert_called_once()

    def test_other_bind_errors_are_not_retried(self):
        with patch('tools.dashboard.ThreadingHTTPServer', side_effect=OSError(errno.EACCES, 'denied')) as factory:
            with self.assertRaises(OSError) as exc:
                bind_dashboard_server()
            self.assertEqual(exc.exception.errno, errno.EACCES)
            factory.assert_called_once()

    def test_busy_range_reports_a_free_port_option(self):
        with patch('tools.dashboard.ThreadingHTTPServer', side_effect=OSError(errno.EADDRINUSE, 'busy')) as factory:
            with self.assertRaisesRegex(OSError, '8090–8099.*--port 0'):
                bind_dashboard_server()
            self.assertEqual(factory.call_count,10)

    def test_os_selected_port_can_be_reserved(self):
        server = bind_dashboard_server(0)
        try:
            self.assertGreater(server.server_address[1], 0)
        finally:
            server.server_close()
