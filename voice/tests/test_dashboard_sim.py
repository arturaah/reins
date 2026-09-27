"""Simulation lockout at the canonical coordinator and production HTTP boundary."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
import pytest
from unittest.mock import Mock, patch

from core.prompt_planner import PromptPlanner
from core.robot_pipeline import RobotPipeline
from tools.dashboard import Simulation


def test_sim_pipeline_never_connects_or_launches_hardware(tmp_path):
    factory=Mock()
    pipeline=RobotPipeline(PromptPlanner(output_dir=tmp_path/'plans'),Simulation(),{},
                           simulation_only=True,backend_factory=factory,run_dir=tmp_path/'runs')
    gestures=Mock()
    try:
        with patch('core.robot_pipeline.subprocess.Popen') as spawn:
            with pytest.raises(ValueError,match='simulation'):
                pipeline.connect(.6)
            for command in ({'action':'refresh'},{'action':'gesture','id':27}):
                with pytest.raises(ValueError,match='simulation'):
                    pipeline.firmware(command,gestures)
            factory.assert_not_called()
            spawn.assert_not_called()
            gestures.command.assert_not_called()
    finally:
        pipeline.close()


def test_sim_http_disables_feeds_observations_and_hardware(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    root=Path(__file__).parents[2]
    process=subprocess.Popen([sys.executable,'tools/dashboard.py','--sim','--port',str(port),
                              '--glasses-host','127.0.0.1','--glasses-port','0','--detector','nanodet',
                              '--head','http://127.0.0.1:1/should-not-read','--observation','/nonexistent'],
                             cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                             env={**os.environ,'REINS_STATE_DIR':str(tmp_path),
                                  'MUJOCO_GL':'cgl' if sys.platform=='darwin' else 'egl'})
    base=f'http://127.0.0.1:{port}'
    try:
        deadline=time.monotonic()+30
        while True:
            try:
                with urllib.request.urlopen(base+'/api/session',timeout=1) as response:
                    token=json.load(response)['token']
                break
            except (OSError,ValueError):
                assert process.poll() is None,process.stderr.read().decode()
                if time.monotonic()>deadline:pytest.fail('Dashboard did not start')
                time.sleep(.1)
        with urllib.request.urlopen(base+'/api/status') as response:state=json.load(response)
        assert state['mode']=='sim'
        assert all(not feed['configured'] for feed in state['feeds'].values())
        assert state['prompt']['configured']['observation'] is False
        assert state['pipeline']['capabilities']['hardware_allowed'] is False
        assert not state['gestures']['connected']
        for path,command in [('/api/robot',{'action':'connect','table_z_m':.6}),
                             ('/api/gestures',{'action':'refresh'}),
                             ('/api/gestures',{'action':'gesture','id':27}),
                             ('/api/prompt',{'prompt':'touch the bottle','source':'camera'})]:
            request=urllib.request.Request(base+path,json.dumps(command).encode(),headers={'X-Reins-Token':token})
            with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
            assert error.value.code==400
            assert 'simulation mode' in error.value.read().decode()
        # Removed replay routes cannot become a hardware escape hatch.
        for path in ('/api/run','/api/plans'):
            with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+path)
            assert error.value.code==404
        request=urllib.request.Request(base+'/api/control',b'{"action":"play"}',headers={'X-Reins-Token':token})
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
        assert error.value.code==400
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        process.stderr.close()
