"""Production HTTP routes in --sim, including forged hardware requests."""
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
from unittest.mock import patch
from tools.dashboard import Runner


def test_sim_runner_never_launches_process_even_with_clearance():
    runner = Runner('irrelevant', simulation_only=True)
    runner.cleared = {'at':time.time(), 'plan':'test', 'path':'test', 'speed':1, 'kp_scale':1, 'fsm':(811,'Start')}
    with patch('tools.dashboard.subprocess.Popen') as spawn:
        for kind in ('dry','execute'):
            with pytest.raises(ValueError, match='simulation mode'):
                runner.start(kind,'test','test',confirm=True)
        spawn.assert_not_called()


def test_sim_http_disables_feeds_observations_and_runs():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    root=Path(__file__).parents[2]
    process=subprocess.Popen([sys.executable,'tools/dashboard.py','--sim','--port',str(port),
                              '--head','http://127.0.0.1:1/should-not-read','--observation','/nonexistent'],
                             cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             env={**os.environ,'MUJOCO_GL':'cgl' if sys.platform=='darwin' else 'egl'})
    base=f'http://127.0.0.1:{port}'
    try:
        deadline=time.monotonic()+20
        while True:
            try:
                with urllib.request.urlopen(base+'/api/plans',timeout=1) as r: plans=json.load(r)
                token=plans['token']
                break
            except (OSError,ValueError):
                assert process.poll() is None, process.stderr.read().decode()
                if time.monotonic()>deadline: pytest.fail('Dashboard did not start')
                time.sleep(.1)
        with urllib.request.urlopen(base+'/api/status') as r: state=json.load(r)
        assert state['mode']=='sim'
        assert all(not f['configured'] for f in state['feeds'].values())
        assert state['prompt']['configured']['observation'] is False
        for path,command in [('/api/run',{'action':'dry','plan':plans['plans'][0]['id']}),
                             ('/api/run',{'action':'execute','confirm':True}),
                             ('/api/prompt',{'prompt':'touch the bottle','source':'camera'})]:
            req=urllib.request.Request(base+path,json.dumps(command).encode(),headers={'X-Reins-Token':token})
            with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(req)
            assert error.value.code==400
            assert 'simulation mode' in error.value.read().decode()
        req=urllib.request.Request(base+'/api/control',b'{"action":"play"}',headers={'X-Reins-Token':token})
        with urllib.request.urlopen(req) as r: assert json.load(r)['playing'] is True
    finally:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        process.stderr.close()
