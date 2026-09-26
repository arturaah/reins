"""MJPEG livestream of the R1 head cameras. Runs ON the Jetson, viewed in a browser.

    python3 tools/camstream.py [--devices auto|0,2] [--port 8080] [--width 640] [--fps 15]
    then open http://<jetson-ip>:8080/ on the Mac

The head's stereo cameras are USB UVC devices on the Jetson (not on the controller),
so this is the only place they can be read. Nothing here talks to the controller.
"""
import argparse, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cv2

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--devices", default="auto", help="comma list of /dev/videoN indices, or auto")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--fps", type=int, default=15)
a = ap.parse_args()


def open_cam(i):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(a.width * 9 / 16))
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    ok, _ = cap.read()
    if not ok:
        cap.release(); return None
    return cap


if a.devices == "auto":
    cams = {i: c for i in range(8) if (c := open_cam(i))}
else:
    cams = {int(i): c for i in a.devices.split(",") if (c := open_cam(int(i)))}
if not cams:
    raise SystemExit("no camera delivered a frame; check ls /dev/video* and permissions")
print("cameras:", ", ".join(f"/dev/video{i} {int(c.get(3))}x{int(c.get(4))}" for i, c in cams.items()))

latest = {}
def grab(i, cap):
    while True:
        ok, frame = cap.read()
        if ok:
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok: latest[i] = jpg.tobytes()
        else:
            time.sleep(0.05)
for i, c in cams.items():
    threading.Thread(target=grab, args=(i, c), daemon=True).start()

PAGE = ("<title>R1 cameras</title><body style='margin:0;background:#111;display:flex;flex-wrap:wrap;gap:4px'>"
        + "".join(f"<img src='/cam/{i}' style='max-width:49vw'>" for i in cams) + "</body>")

class H(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        if self.path == "/":
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(PAGE.encode()); return
        if self.path.startswith("/cam/") and int(self.path[5:]) in cams:
            i = int(self.path[5:])
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            try:
                while True:
                    if (jpg := latest.get(i)):
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")
                    time.sleep(1.0 / a.fps)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(404); self.end_headers()

class Server(ThreadingMixIn, HTTPServer): daemon_threads = True
print(f"serving on http://0.0.0.0:{a.port}/  (Ctrl-C to stop)")
Server(("0.0.0.0", a.port), H).serve_forever()
