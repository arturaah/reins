"""Live page for the R1 head camera, served from the Mac over the control link.

The head camera lives on the controller (192.168.123.161), not the Jetson, and is
read through the SDK's video service (the same GetImageSample RPC Go2 uses). This
polls it at --fps and serves MJPEG at http://localhost:<port>/. Read-only.

    .venv/bin/python tools/headcam.py en6 [--port 8081] [--fps 10]
"""
import argparse, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("iface"); ap.add_argument("--port", type=int, default=8081); ap.add_argument("--fps", type=float, default=10.0)
a = ap.parse_args()

ChannelFactoryInitialize(0, a.iface)
client = VideoClient(); client.SetTimeout(2.0); client.Init()
latest = {"jpg": b"", "n": 0, "fail": 0}

def poll():
    while True:
        t = time.time()
        code, data = client.GetImageSample()
        if code == 0 and data:
            latest["jpg"] = bytes(data); latest["n"] += 1
        else:
            latest["fail"] += 1
        time.sleep(max(0.0, 1.0 / a.fps - (time.time() - t)))
threading.Thread(target=poll, daemon=True).start()

class H(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        if self.path == "/":
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(b"<title>R1 head camera</title><body style='margin:0;background:#111'><img src='/cam' style='max-width:100vw'></body>"); return
        if self.path == "/cam":
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            try:
                while True:
                    if (jpg := latest["jpg"]):
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")
                    time.sleep(1.0 / a.fps)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(404); self.end_headers()

class Server(ThreadingMixIn, HTTPServer): daemon_threads = True
print(f"head camera on http://localhost:{a.port}/  (Ctrl-C to stop)", flush=True)
Server(("127.0.0.1", a.port), H).serve_forever()
