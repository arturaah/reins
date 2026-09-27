"""tools/framelog.py: the contact sheet a recording gets from the camera streams, and its metadata."""
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

from tools.framelog import FrameLogger, contact_sheet, save_sheet


def jpeg(w, h, color):
    b = io.BytesIO(); Image.new("RGB", (w, h), color).save(b, "JPEG"); return b.getvalue()


def test_contact_sheet_layout_and_meta():
    times = [i * 0.1 for i in range(50)]
    q = [[0.0, 0.2 * min(i, 25) / 25] for i in range(50)]                         # moves for 2.5 s, then still
    frames = {"context": [(t, jpeg(64, 36, (200, 0, 0))) for t in times[::3]],
              "left wrist": [(t, jpeg(32, 24, (0, 0, 200))) for t in times[:20:3]]}   # stops halfway: later columns empty
    sheet, meta = contact_sheet(times, q, frames, n=6, tile_w=64)
    assert meta["moments"][0] == 0 and meta["moments"][-1] == 49 and len(meta["moments"]) <= 6
    assert meta["cameras"] == ["context", "left wrist"] and meta["times_s"][-1] == 4.9
    cols = len(meta["moments"])
    assert sheet.width == cols * 66 - 2 and sheet.height == (18 + 36 + 2) + (18 + 48 + 2) - 2
    px = sheet.getpixel((32, 18 + 30))                                            # inside the first context tile
    assert px[0] > 150 and px[1] < 60                                             # the red frame landed there
    last = sheet.getpixel((sheet.width - 3, 18 + 36 + 2 + 18 + 48 - 3))         # last wrist tile's corner: no frame near t=4.9
    assert last == (40, 40, 48)


def test_save_sheet_writes_file_and_meta(tmp_path):
    rec = [(i * 0.05, {"right_elbow_joint": 0.02 * i, "right_shoulder_pitch_joint": 0.0}) for i in range(40)]
    p = tmp_path / "take.json"
    p.write_text(json.dumps({"schema_version": 1, "name": "take", "keyframes": [{"time_s": t, "joint_targets_rad": q} for t, q in rec]}))
    msg = save_sheet(p, rec, {"context": [(t, jpeg(48, 27, (0, 120, 0))) for t, _ in rec[::4]]})
    assert "contact sheet take.sheet.jpg" in msg and (tmp_path / "take.sheet.jpg").exists()
    d = json.loads(p.read_text())
    assert d["sheet"]["file"] == "take.sheet.jpg" and d["sheet"]["cameras"] == ["context"] and d["keyframes"][0]["time_s"] == 0.0
    assert "no camera frames" in save_sheet(p, rec, {})


class MJPEG(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
        jpg = jpeg(16, 9, (1, 2, 3))
        try:
            for _ in range(3):
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n"); time.sleep(0.05)
        except BrokenPipeError:
            pass


def test_frame_logger_polls_live_and_skips_dead_cameras():
    srv = HTTPServer(("127.0.0.1", 0), MJPEG); threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/cam"
    fl = FrameLogger({"context": url, "left wrist": "http://127.0.0.1:1/cam"}, hz=10).start()
    time.sleep(0.7)
    frames = fl.finish(); srv.shutdown()
    assert set(frames) == {"context"} and len(frames["context"]) >= 3
    assert all(0 <= t < 1.0 for t, _ in frames["context"]) and frames["context"][0][1].startswith(b"\xff\xd8")
    assert fl.summary().startswith("context ")
