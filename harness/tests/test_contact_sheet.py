"""tools/framelog.py: the contact sheet a recording gets from the camera streams, and its metadata."""
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image, ImageDraw

from tools.framelog import FrameLogger, contact_sheet, save_sheet


def jpeg(w, h, color):
    b = io.BytesIO(); Image.new("RGB", (w, h), color).save(b, "JPEG"); return b.getvalue()


def frames_with_motion(n=50, w=160, h=90):
    """A grey scene with a white square walking left to right: motion_box must find the strip it crosses."""
    out = []
    for i in range(n):
        im = Image.new("RGB", (w, h), (80, 80, 80)); d = ImageDraw.Draw(im)
        x = 10 + int(100 * i / (n - 1)); d.rectangle([x, 30, x + 20, 50], fill=(255, 255, 255))
        b = io.BytesIO(); im.save(b, "JPEG", quality=90); out.append(b.getvalue())
    return out


def test_contact_sheet_moments_crop_and_budget():
    times = [i * 0.1 for i in range(50)]
    q = [[0.4 * min(i, 25) / 25, 0.4 * max(0, i - 25) / 24] for i in range(50)]        # an L-shaped path: the turn is a moment
    ctx = frames_with_motion()
    frames = {"context": list(zip(times, ctx)),
              "left wrist": [(t, jpeg(32, 24, (0, 0, 200))) for t in times[:20]]}     # stops at 2 s: the last column has no frame
    sheet, meta = contact_sheet(times, q, frames, n_max=8, tol=0.08, tile_w=320, max_w=1568)
    assert meta["moments"] == [0, 25, 49] and meta["times_s"] == [0.0, 2.5, 4.9]
    assert "context" in meta["crop"] and "left wrist" not in meta["crop"]           # only the fixed camera is cropped
    x0, y0, x1, y1 = meta["crop"]["context"]
    assert x0 < 0.1 and x1 > 0.75 and 0.15 < y0 < 0.35 and 0.5 < y1 < 0.85          # the strip the square crossed, padded
    cols = len(meta["moments"]) + 1                                                  # plus the full-view tile
    assert sheet.width == cols * 322 - 2 == 1286 and meta["size_px"] == [sheet.width, sheet.height]
    assert meta["tokens_est"] == sheet.width * sheet.height // 750
    h_ctx = meta["tile_px"][1]
    px = sheet.getpixel((322 + 160, 18 + h_ctx // 2))                                # first context moment, tile centre: scene grey or the square
    assert px[0] > 60
    y_wrist = 18 + h_ctx + 2 + 18
    assert sheet.getpixel((3 * 322 + 317, y_wrist + 240 - 3)) == (40, 40, 48)         # wrist row, last column: no frame
    assert sheet.getpixel((317, y_wrist + 30)) == (30, 30, 36)                        # wrist row has no full-view tile (blank)


def test_contact_sheet_without_motion_has_no_crop_and_narrower_tiles():
    times = [i * 0.1 for i in range(50)]
    q = np.zeros((50, 2)); q[:, 0] = np.linspace(0, 1, 50)
    still = jpeg(160, 90, (80, 80, 80))
    sheet, meta = contact_sheet(times, q, {"context": [(t, still) for t in times]}, n_max=8, max_w=640)
    assert meta["crop"] == {} and len(meta["moments"]) == 3
    assert meta["tile_px"][0] == (640 + 2) // 3 - 2 and sheet.width <= 640            # the width budget shrinks the tiles


def test_save_sheet_writes_file_and_meta(tmp_path):
    rec = [(i * 0.05, {"right_elbow_joint": 0.02 * i, "right_shoulder_pitch_joint": 0.0}) for i in range(40)]
    p = tmp_path / "take.json"
    p.write_text(json.dumps({"schema_version": 1, "name": "take", "keyframes": [{"time_s": t, "joint_targets_rad": q} for t, q in rec]}))
    msg = save_sheet(p, rec, {"context": [(t, jpeg(48, 27, (0, 120, 0))) for t, _ in rec[::4]]})
    assert "contact sheet take.sheet.jpg" in msg and "tokens" in msg and (tmp_path / "take.sheet.jpg").exists()
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
