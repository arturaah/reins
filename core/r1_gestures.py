"""R1 firmware preset gestures only. No trajectories, recording or custom-action RPCs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
EXECUTE_PRESET = 7106
GET_ACTION_LIST = 7107
RELEASE_ARMS = 99
ERRORS = {
    7399: 'The robot controller reported an internal error.',
    7400: 'The arms are busy with another action or controller.',
    7401: 'The robot is holding its last pose. Use Release arms first.',
    7402: 'This gesture is unavailable. Refresh the gesture list.',
    7403: 'The robot could not load the gesture.',
    7404: 'The current robot mode does not allow gestures. Use the Unitree controller to select a supported mode.',
    7405: 'The controller reported an action-name conflict.',
    7406: 'The robot battery is too low for this gesture.',
    7407: 'The robot reported a motor error.',
}


def check_result(code):
    if code != 0:
        raise ValueError(ERRORS.get(code, f'R1 request failed (code {code}). Check the robot connection and interface.'))


def parse_presets(raw):
    """Firmware returns [presets, recordings]; never expose recordings as buttons."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        raise ValueError('The R1 returned an unreadable gesture list.') from None
    if not isinstance(data, list) or len(data) != 2 or not isinstance(data[0], list) or len(data[0]) > 100:
        raise ValueError('The R1 returned an invalid gesture list.')
    actions, seen = [], set()
    for item in data[0]:
        if (not isinstance(item, dict) or type(item.get('id')) is not int
                or not isinstance(item.get('name'), str) or not 1 <= len(item['name'].strip()) <= 100):
            raise ValueError('The R1 returned an invalid preset gesture.')
        action_id = item['id']
        if action_id in (-1, 100):   # R1 teaching and custom playback state IDs
            continue
        if not 0 <= action_id < 100 or action_id in seen:
            raise ValueError('The R1 returned an invalid preset gesture ID.')
        seen.add(action_id)
        name = item['name'].strip()
        actions.append({'id': action_id, 'name': name,
                        'label': 'Release arms' if action_id == RELEASE_ARMS else name.replace('_', ' ').replace('-', ' ').capitalize()})
    return actions


def create_client(iface):
    """Import/initialize DDS only in the explicitly launched helper process."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.rpc.client import Client

    class R1ArmClient(Client):
        def __init__(self):
            super().__init__('arm', False)
            self._SetApiVerson('1.0.0.0')
            self._RegistApi(EXECUTE_PRESET, 0)
            self._RegistApi(GET_ACTION_LIST, 0)

    ChannelFactoryInitialize(0, iface)
    client = R1ArmClient()
    client.SetTimeout(3.)
    return client


def request_firmware(client, action_id=None):
    code, data = client._Call(GET_ACTION_LIST, '{}')
    check_result(code)
    actions = parse_presets(data)
    if action_id is None:
        return {'actions': actions}
    if type(action_id) is not int or action_id not in {a['id'] for a in actions}:
        raise ValueError('Choose a preset gesture supported by this R1.')
    # R1 preset RPCs block until the gesture finishes, unlike custom recordings.
    client.SetTimeout(60.)
    code, _ = client._Call(EXECUTE_PRESET, json.dumps({'action_id': action_id}))
    check_result(code)
    return {'actions': actions, 'completed_id': action_id}


def run_helper(iface, action_id=None):
    args = [sys.executable, str(ROOT/'tools/r1_gestures.py'), iface]
    args += ['--list'] if action_id is None else ['--action', str(action_id)]
    try:
        result = subprocess.run(args, cwd=ROOT, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=70 if action_id is not None else 12)
    except subprocess.TimeoutExpired:
        raise ValueError('R1 request timed out. Check the robot with its controller before sending another gesture.') from None
    except OSError:
        raise ValueError('Could not start the R1 gesture connection.') from None
    # SDK diagnostics go to stdout/stderr; only the helper's final structured line is exposed.
    try:
        answer = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise ValueError('Could not connect to R1 gestures. Check the network interface and Unitree SDK installation.') from None
    if result.returncode or not isinstance(answer, dict) or answer.get('error'):
        error = answer.get('error') if isinstance(answer, dict) else None
        raise ValueError(str(error or 'The R1 gesture request failed.')[:500])
    return answer


class GestureController:
    """One user-triggered RPC at a time; dashboard HTTP requests never block on DDS."""
    def __init__(self, iface, request=None):
        self.iface = iface
        self.request = request or run_helper
        self.lock = threading.RLock()
        self.actions = []
        self.connected = False
        self.busy = False
        self.active = None
        self.message = 'Connect to the R1 to load its built-in gestures.'
        self.error = None
        self.closed = False

    def status(self):
        with self.lock:
            return copy.deepcopy({'iface':self.iface, 'connected':self.connected, 'busy':self.busy,
                                  'active':self.active, 'actions':self.actions, 'message':self.message, 'error':self.error})

    def command(self, command):
        if not isinstance(command, dict):
            raise ValueError('Expected a gesture command.')
        action = command.get('action')
        if action not in ('refresh', 'gesture'):
            raise ValueError('Unknown gesture command.')
        with self.lock:
            if self.closed:
                raise ValueError('Dashboard is shutting down.')
            if self.busy:
                raise ValueError('Wait for the current R1 request to finish.')
            action_id = None
            if action == 'gesture':
                action_id = command.get('id')
                if not self.connected or type(action_id) is not int or action_id not in {a['id'] for a in self.actions}:
                    raise ValueError('Connect and choose an available R1 gesture.')
                self.active = next(a for a in self.actions if a['id'] == action_id)
                self.message = 'Running '+self.active['label'].lower()+'…'
            else:
                self.active = None
                self.message = 'Loading gestures from the R1…'
            self.error = None
            self.busy = True
            threading.Thread(target=self._work, args=(action_id,), daemon=True).start()
        return self.status()

    def _work(self, action_id):
        try:
            result = self.request(self.iface, action_id)
            # Validate even the local transport boundary; discard any custom-action fields.
            actions = parse_presets([result['actions'], []])
            with self.lock:
                self.actions = actions
                self.connected = True
                self.message = ('Gesture finished. If the robot holds the pose, use Release arms.'
                                if action_id is not None and action_id != RELEASE_ARMS
                                else 'Arms released.' if action_id == RELEASE_ARMS else 'Choose a gesture to perform on the R1.')
        except Exception as exc:
            with self.lock:
                self.error = str(exc)[:500] if isinstance(exc, ValueError) else 'The R1 gesture request failed. Reconnect and try again.'
                self.message = self.error
                self.connected = False
                self.actions = []
        finally:
            with self.lock:
                self.busy = False
                self.active = None

    def close(self):
        with self.lock:
            self.closed = True
