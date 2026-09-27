"""Operator CLI transport and retirement tests; never opens hardware or models."""
import builtins
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import runpy
import threading
import unittest
from unittest.mock import patch

from tools.reins import ClientError, DashboardClient, dashboard_url

ROOT = Path(__file__).resolve().parents[1]


class DashboardClientTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def respond(self, value, code=200):
                data = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_GET(self):
                calls.append(("GET", self.path, self.headers.get("X-Reins-Token")))
                if self.path == "/api/session": return self.respond({"token":"test-browser-token"})
                if self.path == "/api/status": return self.respond({"pipeline":{"state":"idle"}})
                return self.respond({"error":"unknown endpoint"}, 404)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(("POST", self.path, self.headers.get("X-Reins-Token"), body))
                if self.headers.get("X-Reins-Token") != "test-browser-token":
                    return self.respond({"error":"missing token"}, 403)
                return self.respond({"ok":True})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = DashboardClient("http://127.0.0.1:"+str(self.server.server_port))

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(2)

    def test_prompt_status_and_stop_use_dashboard_browser_session(self):
        self.client.call("prompt", "  wave right ")
        self.client.call("status")
        self.client.call("stop")
        self.assertEqual(self.calls, [
            ("GET","/api/session",None),
            ("POST","/api/chat","test-browser-token",{"action":"send","message":"wave right"}),
            ("GET","/api/session",None), ("GET","/api/status","test-browser-token"),
            ("GET","/api/session",None),
            ("POST","/api/robot","test-browser-token",{"action":"stop"})])

    def test_no_approve_execute_or_remote_endpoint(self):
        for action in ("approve", "execute", "connect", "gesture"):
            with self.subTest(action=action), self.assertRaises(ClientError): self.client.call(action)
        for url in ("http://example.com", "http://127.0.0.1.evil.invalid:8090", "https://localhost:8090",
                    "http://user:secret@localhost:8090", "http://localhost:8090/api/robot", "http://localhost:8090#x"):
            with self.subTest(url=url), self.assertRaises(ClientError): DashboardClient(url)
        self.assertEqual(self.calls, [])
        self.assertEqual(dashboard_url("http://localhost:8091/"), "http://127.0.0.1:8091")

    def test_redirects_are_refused_before_token_can_leave_loopback(self):
        from tools.reins import NoRedirects
        with self.assertRaises(ClientError):
            NoRedirects().redirect_request(None, None, 307, "", {}, "http://example.com/")


class RetiredControlTests(unittest.TestCase):
    def run_without_sdk(self, path, args):
        original = builtins.__import__
        def deny_sdk(name, *args, **kwargs):
            if name.startswith("unitree_sdk2py"):
                self.fail("Retired command imported the robot SDK: "+name)
            return original(name, *args, **kwargs)
        output = io.StringIO()
        with patch("sys.argv", [str(path), *args]), patch("builtins.__import__", side_effect=deny_sdk), redirect_stderr(output), redirect_stdout(output):
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(path), run_name="__main__")
        self.assertEqual(stopped.exception.code, 2)
        return output.getvalue()

    def test_arm_execute_refuses_before_sdk_import_or_dds(self):
        output = self.run_without_sdk(ROOT/"tools/arm_lift.py", ["unused", "--execute"])
        self.assertIn("--execute is retired", output)

    def test_teach_is_retired_without_sdk_or_service_start(self):
        output = self.run_without_sdk(ROOT/"tools/teach.py", ["unused", "recording"])
        self.assertIn("teaching is retired", output)

    def test_legacy_live_builder_refuses_before_backend_or_model(self):
        from harness.__main__ import build
        with self.assertRaisesRegex(ValueError, "live harness is retired"):
            build({}, "live", "unused")

    def test_legacy_live_cli_refuses_before_config_and_no_confirm_is_gone(self):
        from harness import __main__ as cli
        for args in (["live","unused","wave"], ["live","unused","wave","--no-confirm"]):
            with self.subTest(args=args), patch("sys.argv", ["harness",*args]), patch.object(cli.hcfg, "load", side_effect=AssertionError("No config/backend may start")), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    cli.main()
                self.assertEqual(stopped.exception.code, 2)

    def test_legacy_launchers_have_no_remote_services_or_publisher(self):
        for path in ("tools/arm_lift.py", "tools/teach.py", "tools/reins_ui.py"):
            text = (ROOT/path).read_text()
            self.assertNotIn("ChannelPublisher", text, path)
        for path in ("tools/start_all.sh", "tools/harness_live.sh", "tools/start_dashboard.sh"):
            text = (ROOT/path).read_text()
            for command in ("ssh ", "pkill ", "teach.py", "arm_stream", "revo2 serve"):
                self.assertNotIn(command, text, path)
