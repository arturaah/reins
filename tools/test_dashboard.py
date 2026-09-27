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
import mujoco
import numpy as np
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch
from tools.dashboard import ROOT, CameraFeed, Simulation, MeasuredRobotView, bind_dashboard_server
from tools import dashboard

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

    def test_walk_preview_moves_robot_geometry_and_leaves_scene_fixed(self):
        plan=json.loads(json.dumps(self.plan))
        plan.update(motion_kind='walk',description='Planned base displacement; no gait simulation.',
                    base_keyframes=[{'time_s':0,'x_m':0,'y_m':0,'yaw_rad':0},
                                    {'time_s':plan['duration_s'],'x_m':.1,'y_m':.05,'yaw_rad':.1}])
        self.sim.show_proposal(plan,'new-motion')
        self.sim.advance(plan['duration_s'])
        scene=mujoco.MjvScene(self.sim.model,maxgeom=3000)
        mujoco.mj_forward(self.sim.model,self.sim.data)
        mujoco.mjv_updateScene(self.sim.model,self.sim.data,mujoco.MjvOption(),None,
                              self.sim.camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        before=[geom.pos.copy() for geom in scene.geoms[:scene.ngeom]]
        self.sim.pose_scene_base(scene)
        pelvis=self.sim.model.body('pelvis').id
        moved=0
        for previous,geom in zip(before,scene.geoms[:scene.ngeom]):
            if geom.objtype!=mujoco.mjtObj.mjOBJ_GEOM or geom.objid<0:continue
            body=int(self.sim.model.geom_bodyid[geom.objid])
            if body==pelvis:
                self.assertFalse(np.allclose(previous,geom.pos));moved+=1
            elif body==0:
                np.testing.assert_array_equal(previous,geom.pos)
        self.assertGreater(moved,0)
        np.testing.assert_allclose(self.sim.status()['base_path'][-1],[.1,.05,.1])
        self.assertEqual(self.sim.status()['motion_kind'],'walk')

    def test_walk_preview_rejects_nonfinite_base_path(self):
        self.plan['base_keyframes']=[{'time_s':0,'x_m':0,'y_m':0,'yaw_rad':0},
            {'time_s':self.plan['duration_s'],'x_m':float('nan'),'y_m':0,'yaw_rad':0}]
        with self.assertRaisesRegex(ValueError,'finite'):
            self.sim.show_proposal(self.plan,'new-motion')

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


class MeasuredViewTests(unittest.TestCase):
    def test_local_twin_renders_only_fresh_coordinator_telemetry(self):
        snapshot = {'connected': True, 'state': {'lowstate_age_s': 0., 'joints': {'right_elbow_joint': .2}}}
        view = MeasuredRobotView(pose_source=lambda: snapshot)
        thread = threading.Thread(target=view.run, daemon=True)
        try:
            # The view must never create a second robot connection.
            with patch('socket.create_connection', side_effect=AssertionError('Unexpected robot connection')):
                thread.start()
                deadline = time.monotonic()+6
                while not view.status()['online'] and not view.error and time.monotonic()<deadline: time.sleep(.02)
                self.assertTrue(view.status()['online'],view.error)
                with view.lock:
                    self.assertAlmostEqual(view.data.qpos[view.model.joint('right_elbow_joint').qposadr[0]],.2)
                snapshot['state']['lowstate_age_s'] = 1.
                deadline = time.monotonic()+2
                while view.status()['online'] and time.monotonic()<deadline: time.sleep(.02)
                self.assertFalse(view.status()['online'])
                self.assertEqual(view.jpg,b'')
        finally:
            view.close(); thread.join(3)
        self.assertFalse(thread.is_alive())


class DashboardPortTests(unittest.TestCase):
    def test_voice_service_rejects_external_or_credentialed_urls_before_startup(self):
        for url in ('https://localhost:8770/', 'http://example.com:8770/',
                    'http://user:password@localhost:8770/', 'http://localhost:8770/other'):
            with self.subTest(url=url), patch('sys.argv',['dashboard','--sim','--voice-url',url]), \
                 patch.object(dashboard,'bind_dashboard_server') as bind, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    dashboard.main()
                self.assertEqual(exc.exception.code,2)
                bind.assert_not_called()

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
