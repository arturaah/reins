#!/usr/bin/env python3
"""Reins MCP server (stdio): gives the chat model the dashboard's tools.

Started by the Claude or Codex CLI for each chat turn (see core/claude_chat.py and
core/codex_chat.py); it forwards every tool call to the running dashboard at
REINS_TOOL_URL (loopback) with the tool token read from REINS_TOOL_TOKEN_FILE, so the
token never appears in a command line. Standard library only; no model imports.
None of the tools moves the physical robot (see core/tool_specs.py).

Protocol: MCP over stdio, newline-delimited JSON-RPC 2.0 (initialize, tools/list, tools/call, ping).
"""
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.tool_specs import TOOL_NAMES, TOOL_SPECS  # noqa: E402

PROTOCOL_VERSIONS = ('2025-06-18', '2025-03-26', '2024-11-05')
TIMEOUT_S = 150


def call_dashboard(name, arguments):
    url = os.environ.get('REINS_TOOL_URL', '').rstrip('/')
    token_file = os.environ.get('REINS_TOOL_TOKEN_FILE', '')
    if not url.startswith(('http://127.0.0.1:', 'http://localhost:')) or not token_file:
        return True, 'Reins tools are not connected to a dashboard (REINS_TOOL_URL / REINS_TOOL_TOKEN_FILE).'
    try:
        token = Path(token_file).read_text().strip()
    except OSError:
        return True, 'Reins tool token is unavailable.'
    request = urllib.request.Request(f'{url}/{name}', json.dumps(arguments).encode(),
                                     headers={'Content-Type': 'application/json', 'X-Reins-Tool-Token': token})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return False, response.read(1024 * 1024).decode()
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read(64 * 1024)).get('error') or f'HTTP {exc.code}'
        except ValueError:
            message = f'HTTP {exc.code}'
        return True, message
    except (urllib.error.URLError, TimeoutError, OSError):
        return True, 'The Reins dashboard could not be reached.'


def handle(message):
    method, params, ident = message.get('method'), message.get('params') or {}, message.get('id')
    if method == 'initialize':
        asked = params.get('protocolVersion')
        return {'protocolVersion': asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                'capabilities': {'tools': {'listChanged': False}},
                'serverInfo': {'name': 'reins', 'version': '0.1'},
                'instructions': 'Reins robot tools: observe, plan and preview in simulation. None moves the robot.'}
    if method == 'ping':
        return {}
    if method == 'tools/list':
        return {'tools': TOOL_SPECS}
    if method == 'tools/call':
        name, arguments = params.get('name'), params.get('arguments') or {}
        if name not in TOOL_NAMES:
            raise LookupError(f'Unknown tool {name!r}')
        is_error, text = call_dashboard(name, arguments)
        return {'content': [{'type': 'text', 'text': text}], 'isError': is_error}
    if ident is None:
        return None            # notifications (initialized, cancelled, ...) need no reply
    raise NotImplementedError(method)


def main():
    debug = os.environ.get('REINS_MCP_DEBUG_LOG')   # optional: log the raw protocol for troubleshooting
    log = open(debug, 'a') if debug else None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if log:
            log.write('< ' + line[:2000] + '\n'); log.flush()
        try:
            message = json.loads(line)
        except ValueError:
            reply = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Parse error'}}
        else:
            ident = message.get('id') if isinstance(message, dict) else None
            try:
                result = handle(message if isinstance(message, dict) else {})
                if result is None and ident is None:
                    continue
                reply = {'jsonrpc': '2.0', 'id': ident, 'result': result}
            except NotImplementedError as exc:
                reply = {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32601, 'message': f'Method not found: {exc}'}}
            except LookupError as exc:
                reply = {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32602, 'message': str(exc)}}
        if log:
            log.write('> ' + json.dumps(reply)[:2000] + '\n'); log.flush()
        sys.stdout.write(json.dumps(reply) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
