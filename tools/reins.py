"""HTTP-only operator client: prompt, status and stop use the running dashboard.

No hardware SDK, model credentials, approval or execution API is exposed here.
Start tools/dashboard.py first. Approve complete motions in its browser or glasses.
"""
import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class ClientError(ValueError):
    pass


def dashboard_url(value):
    try:
        parsed = urlsplit(value)
        port = parsed.port or 8090
    except (ValueError, TypeError) as exc:
        raise ClientError("Use a loopback dashboard URL such as http://127.0.0.1:8090") from exc
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or
            parsed.username is not None or parsed.password is not None or parsed.path not in ("", "/") or
            parsed.query or parsed.fragment or not 1 <= port <= 65535):
        raise ClientError("Use a loopback dashboard URL such as http://127.0.0.1:8090")
    # Avoid DNS-dependent localhost resolution and IPv6 Host incompatibility.
    return f"http://127.0.0.1:{port}"


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ClientError("Dashboard redirects are refused")


class DashboardClient:
    def __init__(self, url="http://127.0.0.1:8090", timeout=5):
        self.url = dashboard_url(url)
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirects())

    def _request(self, path, body=None, token=None):
        headers = {"Accept":"application/json"}
        if token is not None:
            headers["X-Reins-Token"] = token
        data = None
        if body is not None:
            data = json.dumps(body, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.url+path, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(2*1024*1024+1)
            if len(raw) > 2*1024*1024:
                raise ClientError("Dashboard response exceeded the size limit")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ClientError("Invalid dashboard response")
            return value
        except HTTPError as exc:
            try:
                error = json.loads(exc.read(4096)).get("error", str(exc.code))
            except (ValueError, AttributeError):
                error = str(exc.code)
            raise ClientError(f"Dashboard refused the request: {error}") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise ClientError(f"Cannot reach dashboard at {self.url}; start tools/dashboard.py first") from exc
        except json.JSONDecodeError as exc:
            raise ClientError("Dashboard returned invalid JSON") from exc

    def call(self, action, text=None):
        if action not in ("prompt", "status", "stop"):
            raise ClientError("Choose prompt, status or stop; approve in the dashboard/glasses")
        if action == "prompt" and (not isinstance(text, str) or not 1 <= len(text.strip()) <= 4000):
            raise ClientError("Prompt must contain 1–4000 characters")
        token = self._request("/api/session").get("token")
        if not isinstance(token, str) or not token:
            raise ClientError("Dashboard did not provide a local session")
        if action == "status":
            return self._request("/api/status", token=token)
        if action == "prompt":
            return self._request("/api/chat", {"action":"send", "message":text.strip()}, token=token)
        return self._request("/api/robot", {"action":"stop"}, token=token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("REINS_DASHBOARD_URL", "http://127.0.0.1:8090"))
    commands = parser.add_subparsers(dest="action", required=True)
    prompt = commands.add_parser("prompt", help="send a task to the dashboard agent")
    prompt.add_argument("text")
    commands.add_parser("status", help="read dashboard, planning and robot status")
    commands.add_parser("stop", help="cancel dashboard work and release robot control")
    args = parser.parse_args(argv)
    try:
        result = DashboardClient(args.url).call(args.action, getattr(args, "text", None))
    except ClientError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
