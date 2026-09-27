"""Codex CLI transport for dashboard chat, using the local CLI's saved sign-in.

Each request is an ephemeral, non-interactive text conversation. No shell command
is constructed from user text. Existing CLI threads and config are not modified.
"""
from collections import namedtuple
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from core.tool_specs import INSTRUCTIONS as TOOL_INSTRUCTIONS, TOOL_NAMES

# Link to the dashboard's tool endpoint; the MCP server (tools/reins_mcp.py) forwards calls there.
ToolLink = namedtuple('ToolLink', 'url token')
MCP_SERVER = Path(__file__).resolve().parents[1] / 'tools' / 'reins_mcp.py'

DISABLED_FEATURES = ('shell_tool', 'unified_exec', 'shell_snapshot', 'apps', 'plugins',
                     'hooks', 'multi_agent', 'browser_use', 'computer_use',
                     'image_generation', 'memories')


class CodexResponder:
    MAX_OUTPUT = 2 * 1024 * 1024
    name = 'Codex'                 # used in user-facing messages; subclasses (claude_chat) override
    label = 'Codex CLI'

    def __init__(self, instructions, schema, timeout=180, tools=None):
        """tools: optional ToolLink; the model then gets the Reins MCP tools and a longer timeout."""
        self.tools = tools
        self.instructions = instructions + (TOOL_INSTRUCTIONS if tools else '')
        self.schema, self.timeout = schema, max(timeout, 480) if tools else timeout
        self.binary = shutil.which(os.path.expanduser(os.environ.get('REINS_CODEX_BIN', 'codex')))
        self.model = os.environ.get('REINS_CODEX_MODEL', '')
        self.lock = threading.Lock()
        self.proc = None
        self.cancelled = threading.Event()
        self.closed = False
        self.config = self._configuration()

    def _configuration(self):
        config = {'provider': 'codex', 'provider_label': 'Codex CLI', 'configured': False,
                  'model': self.model or 'CLI default', 'setup': 'Install Codex CLI, run codex login, then restart the dashboard.'}
        if not self.binary:
            return config
        try:
            result = subprocess.run([self.binary, 'login', 'status'], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            config['setup'] = 'Could not check Codex sign-in. Run codex login status, then restart the dashboard.'
            return config
        config['configured'] = result.returncode == 0
        config['setup'] = '' if config['configured'] else 'Run codex login in a terminal, then restart the dashboard.'
        return config

    def configuration(self):
        return dict(self.config)

    def command(self, schema_path):
        args = [self.binary, 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral',
                '--skip-git-repo-check', '--sandbox', 'read-only', '--json', '--color', 'never',
                '--output-schema', str(schema_path), '-c', 'approval_policy="never"',
                '-c', 'web_search="disabled"']
        for feature in DISABLED_FEATURES:
            args.extend(['--disable', feature])
        if self.model:
            args.extend(['--model', self.model])
        if self.tools:
            env = self._tool_env(schema_path.parent)
            args += ['-c', f'mcp_servers.reins.command={json.dumps(sys.executable)}',
                     '-c', f'mcp_servers.reins.args=[{json.dumps(str(MCP_SERVER))}]',
                     '-c', 'mcp_servers.reins.env={' + ', '.join(f'{k}={json.dumps(v)}' for k, v in env.items()) + '}',
                     '-c', 'mcp_servers.reins.tool_timeout_sec=450',
                     '-c', 'mcp_servers.reins.startup_timeout_sec=20',
                     # exec cannot prompt; these tools cannot move the robot (see core/tool_specs.py)
                     '-c', 'mcp_servers.reins.default_tools_approval_mode="approve"']
        return args + ['-']  # All conversation content goes over stdin, never argv or a shell.

    def _tool_env(self, workdir):
        """Environment for the MCP server. The token goes in a private file, never in argv."""
        token = Path(workdir) / 'tool.token'
        if not token.exists():
            fd = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as f:
                f.write(self.tools.token)
        return {'REINS_TOOL_URL': self.tools.url, 'REINS_TOOL_TOKEN_FILE': str(token)}

    @staticmethod
    def _stop(proc):
        if proc.poll() is not None:
            return
        try:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass

    def prepare(self):
        with self.lock:
            self.cancelled.clear()

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            if self.proc:
                self._stop(self.proc)

    def close(self):
        with self.lock:
            self.closed = True
        self.cancel()

    @staticmethod
    def _parse(output, diagnostic, returncode):
        # Never return raw stderr, tool output or internal reasoning to the browser.
        text, completed, failed = None, False, False
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            item = event.get('item', {})
            if event.get('type') == 'item.completed' and isinstance(item, dict) and item.get('type') == 'agent_message':
                text = item.get('text')
            if event.get('type') == 'turn.completed':
                completed = True
            if event.get('type') == 'turn.failed':
                failed = True
        if returncode or failed or not completed:
            diagnostic = (diagnostic + output).lower()
            if 'unexpected argument' in diagnostic or 'unrecognized' in diagnostic:
                raise ValueError('This Codex CLI version lacks required integration options. Update Codex CLI, then restart the dashboard.')
            if any(word in diagnostic for word in ('unauthorized', '401', 'not logged in', 'authentication', 'token expired')):
                raise ValueError('Codex sign-in needs attention. Run codex login in a terminal, then restart the dashboard.')
            if any(word in diagnostic for word in ('usage limit', 'quota', 'rate limit', '429')):
                raise ValueError('Codex usage limit reached. Try later or check your Codex account limits.')
            raise ValueError('Codex CLI could not finish the reply. Check codex login status, connectivity and model access, then retry.')
        try:
            answer = json.loads(text)
        except (ValueError, TypeError):
            raise ValueError('Codex CLI returned no usable chat reply. Please retry.') from None
        return answer

    def _prompt(self, context, conversation):
        tool_rule = ('Use the Reins tools when they help, then return' if self.tools
                     else 'Return only') + ' the JSON object described by the output schema.'
        tool_rule += '' if self.tools else ' Do not use tools.'
        return (self.instructions + '\n\nReply to the last user message in this conversation. '
                + tool_rule + '\n'
                + json.dumps({'dashboard_status': context, 'conversation': conversation}, ensure_ascii=True, allow_nan=False))

    def __call__(self, messages, context):
        if not self.config['configured']:
            raise ValueError(self.config['setup'])
        conversation = [{'role': m['role'], 'text': m['text'], 'robot_request': m.get('robot_request'),
                         'trajectory': m.get('trajectory')} for m in messages]
        prompt = self._prompt(context, conversation)
        with tempfile.TemporaryDirectory(prefix=f'reins-{self.name.lower()}-chat-') as tmp, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            schema_path = Path(tmp) / 'reply.schema.json'
            schema_path.write_text(json.dumps(self.schema))
            with self.lock:
                if self.closed:
                    raise ValueError('Dashboard chat is shutting down.')
                if self.cancelled.is_set():
                    raise ValueError(f'{self.name} reply stopped.')
                try:
                    proc = subprocess.Popen(self.command(schema_path), cwd=tmp, stdin=subprocess.PIPE,
                              stdout=out, stderr=err, start_new_session=os.name == 'posix')
                except OSError:
                    raise ValueError(f'Could not start {self.label}. Check the installation and restart the dashboard.') from None
                self.proc = proc
            def write_input():
                try:
                    proc.stdin.write(prompt.encode('utf-8'))
                except (OSError, ValueError):
                    pass
                finally:
                    try:
                        proc.stdin.close()
                    except (OSError, ValueError):
                        pass
            writer = threading.Thread(target=write_input, daemon=True)
            writer.start()
            started = time.monotonic()
            try:
                while proc.poll() is None:
                    if self.cancelled.is_set():
                        raise ValueError(f'{self.name} reply stopped.')
                    if time.monotonic() - started > self.timeout:
                        raise ValueError(f'{self.label} timed out. Your message is kept; please retry.')
                    if os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size > self.MAX_OUTPUT:
                        raise ValueError(f'{self.label} produced too much output; the reply was stopped.')
                    self.cancelled.wait(.05)
                if self.cancelled.is_set():
                    raise ValueError(f'{self.name} reply stopped.')
                if os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size > self.MAX_OUTPUT:
                    raise ValueError(f'{self.label} produced too much output; the reply was stopped.')
                out.seek(0); err.seek(0)
                return self._parse(out.read(self.MAX_OUTPUT).decode('utf-8', errors='replace'),
                                   err.read(self.MAX_OUTPUT).decode('utf-8', errors='replace'), proc.returncode)
            finally:
                self._stop(proc)
                proc.wait(timeout=5)
                writer.join(timeout=1)
                if proc.stdin and not proc.stdin.closed:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
                with self.lock:
                    self.proc = None
