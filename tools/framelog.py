"""Camera frames alongside a recording, and a contact sheet of its key moments. No SDK, no DDS.

FrameLogger polls the stream servers the desktop window already uses (tools/headcam.py on port
8081 for the head camera, the Jetson's camstream.py forwarded to 8080 for the wrists) a few times
a second while tools/teach.py or tools/record.py sample the joints, keeping every frame with its
time on the recording's clock. contact_sheet() picks the key moments of the motion (equal
joint-space arc-length fractions, first and last sample always: harness.demos.key_moments) and
tiles the nearest frame of every camera under each moment, one row per camera, one column per
moment, labelled with the time. save_sheet() writes recordings/<name>.sheet.jpg next to the
recording and adds a "sheet" entry to its JSON; harness/demos.py shows it to the VLM when the
recording is selected as a demonstration in the window's AI pane.
"""
import io, json, os, sys, threading, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from harness.demos import key_moments

CAMERAS = {"context": "http://localhost:8081/cam", "left wrist": "http://localhost:8080/cam/0", "right wrist": "http://localhost:8080/cam/2"}


def grab_frame(url, timeout=2.0):
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


class FrameLogger:
    """Polls every camera in its own thread from start() until finish(); frames[camera] = [(t, jpeg)]."""
    def __init__(self, cameras=None, hz=3.0):
        self.cameras = dict(CAMERAS if cameras is None else cameras); self.hz = hz
        self.frames = {n: [] for n in self.cameras}; self.t0 = None
        self.stop = threading.Event(); self.threads = []

    def start(self, t0=None):
        """t0: the recording's zero on time.time()'s clock, so frame times match the samples."""
        self.t0 = time.time() if t0 is None else t0
        for name, url in self.cameras.items():
            th = threading.Thread(target=self._poll, args=(name, url), daemon=True); th.start(); self.threads.append(th)
        return self

    def _poll(self, name, url):
        misses = 0
        while not self.stop.is_set():
            t = time.time(); jpg = grab_frame(url)
            if jpg:
                self.frames[name].append((t - self.t0, jpg)); misses = 0
            else:
                misses += 1
                if misses >= 3: self.stop.wait(2.0)                 # camera down: back off instead of hammering the port
            self.stop.wait(max(0.0, 1.0 / self.hz - (time.time() - t)))

    def finish(self):
        """Stop polling; -> {camera: [(t, jpeg)]} for the cameras that delivered anything."""
        self.stop.set()
        for th in self.threads: th.join(timeout=3.0)
        return {n: f for n, f in self.frames.items() if f}

    def summary(self):
        return ", ".join(f"{n} {len(f)}" for n, f in self.frames.items() if f) or "none"


def contact_sheet(times, q, frames, n=6, tile_w=320, max_gap_s=1.0):
    """-> (PIL image, meta). times, q: the recording's samples (q rows = joints); frames: {camera: [(t, jpeg)]}."""
    idx = key_moments(times, q, n)
    keys = [float(times[i]) for i in idx]
    font = ImageFont.load_default(size=14)
    gap, band = 2, 18
    rows = []
    for cam, fr in frames.items():
        ft = np.array([t for t, _ in fr])
        tiles = []
        for t in keys:
            j = int(np.argmin(np.abs(ft - t))); im = None
            if abs(ft[j] - t) <= max_gap_s:
                try: im = Image.open(io.BytesIO(fr[j][1])).convert("RGB")
                except Exception: im = None
            tiles.append(im)
        first = next((im for im in tiles if im is not None), None)
        tile_h = round(tile_w * first.height / first.width) if first else tile_w * 9 // 16
        rows.append((cam, tiles, tile_h))
    W = len(keys) * (tile_w + gap) - gap
    H = sum(band + h + gap for _, _, h in rows) - gap
    sheet = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(sheet)
    y = 0
    for cam, tiles, h in rows:
        d.text((4, y + 2), cam.upper(), fill=(255, 220, 80), font=font); y += band
        for c, (t, im) in enumerate(zip(keys, tiles)):
            x = c * (tile_w + gap)
            if im is None:
                d.rectangle([x, y, x + tile_w, y + h], fill=(40, 40, 48)); d.text((x + 8, y + h // 2 - 7), "no frame", fill=(160, 160, 160), font=font)
            else:
                sheet.paste(im.resize((tile_w, h)), (x, y))
            label = f"t={t:.1f}s"
            d.rectangle([x, y, x + 8 + 7 * len(label), y + 17], fill=(0, 0, 0)); d.text((x + 4, y + 1), label, fill=(255, 255, 255), font=font)
        y += h + gap
    meta = {"moments": [int(i) for i in idx], "times_s": [round(t, 3) for t in keys], "cameras": list(frames), "tile_px": [tile_w, rows[0][2] if rows else 0]}
    return sheet, meta


def save_sheet(json_path, rec, frames, n=6):
    """rec: [(t, {joint: q})] as teach/record hold it. Writes <name>.sheet.jpg and adds "sheet" to the JSON. -> text for the log."""
    json_path = json_path if hasattr(json_path, "with_name") else __import__("pathlib").Path(json_path)
    if not frames:
        return "no camera frames were logged, so no contact sheet (is tools/headcam.py serving port 8081?)"
    names = sorted(rec[0][1]); times = [t for t, _ in rec]; q = [[qq[k] for k in names] for _, qq in rec]
    sheet, meta = contact_sheet(times, q, frames, n)
    sp = json_path.with_name(json_path.stem + ".sheet.jpg")
    sheet.save(sp, "JPEG", quality=85)
    d = json.loads(json_path.read_text()); d["sheet"] = {"file": sp.name, **meta}
    json_path.write_text(json.dumps(d, indent=1) + "\n")
    return (f"contact sheet {sp.name}: {len(meta['times_s'])} moments x {len(meta['cameras'])} camera(s) ({', '.join(meta['cameras'])}), "
            f"{sheet.width}x{sheet.height}; it is what the AI pane shows the model when this recording is selected as context")
