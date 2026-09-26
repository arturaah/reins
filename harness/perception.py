"""One frame per camera, resized and JPEG-encoded, optional grid and hand-tip marker, proprio numbers.

Sources: HttpCameras reads one frame from each MJPEG stream (tools/headcam.py for the head camera,
the Jetson's camstream.py forwarded to port 8080 for the wrists). MockCameras renders the sim.
No SDK here: the DDS side lives in the stream servers.
"""
import io
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Packet:
    images: list                          # [(label, jpeg_bytes)] in prompt order: context first, then wrist(s)
    pil: dict = field(default_factory=dict)      # label -> PIL image (annotated)
    missing: list = field(default_factory=list)  # labels that could not be captured
    t: float = 0.0


def grab_mjpeg_frame(url, timeout=3.0):
    """First JPEG of an MJPEG stream, or None."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            while True:
                line = r.readline()
                if not line:
                    return None
                if line.lower().startswith(b"content-length:"):
                    n = int(line.split(b":")[1]); r.readline()
                    return r.read(n)
    except Exception:
        return None


class HttpCameras:
    def __init__(self, cfg, arm):
        p = cfg["perception"]
        self.urls = {"CONTEXT VIEW": p["context_url"], f"{arm.upper()} WRIST VIEW": p["wrist_urls"][arm]}

    def frames(self):
        out = {}
        for label, url in self.urls.items():
            jpg = grab_mjpeg_frame(url)
            out[label] = Image.open(io.BytesIO(jpg)).convert("RGB") if jpg else None
        return out


class MockCameras:
    def __init__(self, backend, arm, width=640, height=360):
        self.backend, self.arm, self.w, self.h = backend, arm, width, height

    def frames(self):
        out = {}
        for label, view in (("CONTEXT VIEW", "context"), (f"{self.arm.upper()} WRIST VIEW", "wrist")):
            arr = self.backend.render(view, self.arm, self.w, self.h)
            out[label] = Image.fromarray(arr) if arr is not None else placeholder(self.w, self.h, f"{label} (no renderer)")
        return out


def placeholder(w, h, text):
    im = Image.new("RGB", (w, h), (40, 40, 48))
    ImageDraw.Draw(im).text((10, 10), text, fill=(220, 220, 220))
    return im


def resize(im, width):
    if im.width == width:
        return im
    return im.resize((width, max(1, round(im.height * width / im.width))))


def draw_grid(im, cols=8, rows=6):
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    for c in range(1, cols):
        x = w * c / cols; d.line([(x, 0), (x, h)], fill=(255, 255, 255, 90), width=1)
    for r in range(1, rows):
        y = h * r / rows; d.line([(0, y), (w, y)], fill=(255, 255, 255, 90), width=1)
    for c in range(cols):
        d.text((w * (c + 0.5) / cols - 4, 2), chr(ord("A") + c), fill=(255, 255, 0, 230))
    for r in range(rows):
        d.text((3, h * (r + 0.5) / rows - 5), str(r + 1), fill=(255, 255, 0, 230))
    return im


def project(cam, p):
    """Pinhole projection of a robot-frame point. cam: {pos, forward, up, fx, fy, cx, cy}; returns (u, v) or None."""
    pos, fwd, up = (np.asarray(cam[k], float) for k in ("pos", "forward", "up"))
    z = fwd / np.linalg.norm(fwd)
    x = np.cross(z, up); x /= np.linalg.norm(x)          # image right
    y = np.cross(z, x)                                   # image down
    d = np.asarray(p, float) - pos
    Z = float(d @ z)
    if Z <= 1e-6:
        return None
    return float(cam["fx"] * (d @ x) / Z + cam["cx"]), float(cam["fy"] * (d @ y) / Z + cam["cy"])


def draw_marker(im, uv, label="tip", color=(0, 255, 255)):
    if uv is None:
        return im
    u, v = uv
    d = ImageDraw.Draw(im)
    d.ellipse([u - 7, v - 7, u + 7, v + 7], outline=color, width=3)
    d.line([(u - 12, v), (u + 12, v)], fill=color, width=2); d.line([(u, v - 12), (u, v + 12)], fill=color, width=2)
    d.text((u + 10, v + 8), label, fill=color)
    return im


class Perception:
    def __init__(self, cfg, arm, cameras, context_camera=None):
        self.cfg, self.arm, self.cameras = cfg, arm, cameras
        p = cfg["perception"]
        self.width, self.quality = int(p["width_px"]), int(p["jpeg_quality"])
        self.grid, self.cols, self.rows = bool(p["grid"]), int(p["grid_cols"]), int(p["grid_rows"])
        self.marker = bool(p["hand_marker"]) and context_camera is not None
        self.context_camera = context_camera

    def capture(self, hand_tip=None):
        frames = self.cameras.frames()
        images, pil, missing = [], {}, []
        for label, im in frames.items():
            if im is None:
                missing.append(label)
                im = placeholder(self.width, self.width * 9 // 16, f"{label}: NO IMAGE")
            src_w = im.width
            im = resize(im, self.width)
            if label == "CONTEXT VIEW":
                if self.marker and hand_tip is not None:
                    uv = project(self.context_camera, hand_tip)
                    if uv is not None:
                        k = self.width / src_w
                        draw_marker(im, (uv[0] * k, uv[1] * k), f"{self.arm} hand tip")
                if self.grid:
                    draw_grid(im, self.cols, self.rows)
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=self.quality)
            images.append((label, buf.getvalue())); pil[label] = im
        return Packet(images, pil, missing, time.time())


def height_above_table_cm(p, table_z):
    return (float(p[2]) - float(table_z)) * 100.0
