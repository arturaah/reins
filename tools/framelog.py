"""Camera frames alongside a recording, and a contact sheet of its key moments. No SDK, no DDS.

FrameLogger polls the stream servers the desktop window already uses (tools/headcam.py on port
8081 for the head camera, the Jetson's camstream.py forwarded to 8080 for the wrists) a few times
a second while tools/teach.py or tools/record.py sample the joints, keeping every frame with its
time on the recording's clock. contact_sheet() picks the key moments of the motion
(harness.demos.smart_moments: Douglas-Peucker on the joint-space path, 3 to demos.max_moments
frames depending on how much the path turns) and tiles the nearest frame of every camera under
each moment, one row per camera, one column per moment, labelled with the time. The context row
is cropped to the region that changed during the recording (motion_box: where the arm and the
objects moved) with a full-view tile showing the crop, and the sheet is kept under the model's
long-edge limit, so it costs a few hundred tokens (meta["tokens_est"]). save_sheet() writes
recordings/<name>.sheet.jpg next to the recording and adds a "sheet" entry to its JSON;
harness/demos.py shows it to the VLM when the recording is selected as a demonstration.
"""
import io, json, os, sys, threading, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from harness.demos import smart_moments
try:
    from harness.config import load as _load_cfg
    D = dict(_load_cfg()["demos"])
except Exception:
    D = {"max_moments": 8, "tolerance_rad": 0.08, "max_width_px": 1568}

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


def motion_box(jpegs, thr=0.12, min_frac=0.45, margin=0.12, width=80):
    """Normalised (x0, y0, x1, y1) of the image region that changed across the frames, or None when too little (or nearly
    everything: lighting, a moved camera) did. Frames are compared as 80 px wide greyscale; the box around the changed
    pixels is padded by margin and made at least min_frac of each side, so the crop keeps some surroundings."""
    small = []
    for b in jpegs:
        try:
            im = Image.open(io.BytesIO(b)).convert("L")
            small.append(np.asarray(im.resize((width, max(1, round(width * im.height / im.width)))), float) / 255.0)
        except Exception:
            pass
    if len(small) < 2:
        return None
    h, w = small[0].shape
    small = [a for a in small if a.shape == (h, w)]
    acc = np.zeros((h, w))
    for a, b in zip(small[:-1], small[1:]):
        acc = np.maximum(acc, np.abs(a - b))
    mask = acc > thr
    if mask.mean() < 0.005 or mask.mean() > 0.6:
        return None
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min() / w, (xs.max() + 1) / w, ys.min() / h, (ys.max() + 1) / h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max(min_frac, (x1 - x0) * (1 + 2 * margin)), max(min_frac, (y1 - y0) * (1 + 2 * margin))
    return (max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(1.0, cx + bw / 2), min(1.0, cy + bh / 2))


def _tile(im, box, tile_w, tile_h):
    """Crop im to the normalised box (when given), scale to tile_w wide, letterbox into tile_w x tile_h."""
    if box:
        x0, y0, x1, y1 = box
        im = im.crop((round(x0 * im.width), round(y0 * im.height), max(round(x0 * im.width) + 2, round(x1 * im.width)),
                      max(round(y0 * im.height) + 2, round(y1 * im.height))))
    h = max(1, round(tile_w * im.height / im.width)); im = im.resize((tile_w, h))
    if h == tile_h:
        return im
    out = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
    out.paste(im.crop((0, 0, tile_w, min(h, tile_h))), (0, max(0, (tile_h - h) // 2)))
    return out


def _label(d, x, y, text, font):
    d.rectangle([x, y, x + 8 + 7 * len(text), y + 17], fill=(0, 0, 0)); d.text((x + 4, y + 1), text, fill=(255, 255, 255), font=font)


def contact_sheet(times, q, frames, n_max=None, tol=None, tile_w=320, max_w=None, max_gap_s=1.0, crop=True):
    """-> (PIL image, meta). times, q: the recording's samples (q rows = joints); frames: {camera: [(t, jpeg)]}.
    Columns = the motion's key moments (smart_moments); the context row is cropped to where the image changed
    (motion_box) and gets a leftmost full-view tile with that box drawn; wrist rows (moving camera) are never cropped.
    The sheet is at most max_w wide, so it costs about size_px / 750 tokens (meta["tokens_est"])."""
    n_max = int(n_max or D["max_moments"]); tol = float(tol or D["tolerance_rad"]); max_w = int(max_w or D["max_width_px"])
    idx = smart_moments(times, q, tol, n_max)
    keys = [float(times[i]) for i in idx]
    boxes = {}
    if crop:
        for cam, fr in frames.items():
            if "wrist" not in cam and fr:
                step = max(1, len(fr) // 60); boxes[cam] = motion_box([b for _, b in fr[::step]])
    full_col = any(boxes.values())
    cols = len(keys) + (1 if full_col else 0)
    gap, band = 2, 18
    tile_w = min(tile_w, max(120, (max_w + gap) // cols - gap))
    font = ImageFont.load_default(size=14)
    rows = []
    for cam, fr in frames.items():
        ft = np.array([t for t, _ in fr]); box = boxes.get(cam)
        tiles = []
        for t in keys:
            j = int(np.argmin(np.abs(ft - t))); im = None
            if abs(ft[j] - t) <= max_gap_s:
                try: im = Image.open(io.BytesIO(fr[j][1])).convert("RGB")
                except Exception: im = None
            tiles.append(im)
        first = next((im for im in tiles if im is not None), None)
        if first is None: tile_h = tile_w * 9 // 16
        elif box: tile_h = max(1, round(tile_w * ((box[3] - box[1]) * first.height) / ((box[2] - box[0]) * first.width)))
        else: tile_h = max(1, round(tile_w * first.height / first.width))
        full = None
        if full_col and box:
            try: full = Image.open(io.BytesIO(fr[0][1])).convert("RGB")
            except Exception: full = None
        rows.append((cam, tiles, tile_h, box, full))
    W = cols * (tile_w + gap) - gap
    H = sum(band + h + gap for _, _, h, _, _ in rows) - gap
    sheet = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(sheet); y = 0
    for cam, tiles, h, box, full in rows:
        d.text((4, y + 2), cam.upper() + ("   cropped to the motion; leftmost tile = full view, red box = the crop" if box else ""),
               fill=(255, 220, 80), font=font); y += band
        c0 = 0
        if full_col:
            if full is not None:
                fv = _tile(full, None, tile_w, h)
                k = tile_w / full.width; hh = round(full.height * k); oy = max(0, (h - hh) // 2)
                ImageDraw.Draw(fv).rectangle([box[0] * tile_w, oy + box[1] * hh, box[2] * tile_w - 1, oy + box[3] * hh - 1], outline=(255, 70, 70), width=2)
                sheet.paste(fv, (0, y)); _label(d, 0, y, "full view", font)
            else:
                d.rectangle([0, y, tile_w, y + h], fill=(30, 30, 36)); _label(d, 0, y, "(not cropped)", font)
            c0 = 1
        for c, (t, im) in enumerate(zip(keys, tiles)):
            x = (c + c0) * (tile_w + gap)
            if im is None:
                d.rectangle([x, y, x + tile_w, y + h], fill=(40, 40, 48)); d.text((x + 8, y + h // 2 - 7), "no frame", fill=(160, 160, 160), font=font)
            else:
                sheet.paste(_tile(im, box, tile_w, h), (x, y))
            _label(d, x, y, f"t={t:.1f}s", font)
        y += h + gap
    meta = {"moments": [int(i) for i in idx], "times_s": [round(t, 3) for t in keys], "cameras": list(frames),
            "tile_px": [tile_w, rows[0][2] if rows else 0], "size_px": [W, H],
            "crop": {cam: [round(v, 3) for v in box] for cam, box in boxes.items() if box}, "tokens_est": int(W * H / 750)}
    return sheet, meta


def save_sheet(json_path, rec, frames, n_max=None):
    """rec: [(t, {joint: q})] as teach/record hold it. Writes <name>.sheet.jpg and adds "sheet" to the JSON. -> text for the log."""
    json_path = json_path if hasattr(json_path, "with_name") else __import__("pathlib").Path(json_path)
    if not frames:
        return "no camera frames were logged, so no contact sheet (is tools/headcam.py serving port 8081?)"
    names = sorted(rec[0][1]); times = [t for t, _ in rec]; q = [[qq[k] for k in names] for _, qq in rec]
    sheet, meta = contact_sheet(times, q, frames, n_max)
    sp = json_path.with_name(json_path.stem + ".sheet.jpg")
    sheet.save(sp, "JPEG", quality=85)
    d = json.loads(json_path.read_text()); d["sheet"] = {"file": sp.name, **meta}
    json_path.write_text(json.dumps(d, indent=1) + "\n")
    return (f"contact sheet {sp.name}: {len(meta['times_s'])} moments x {len(meta['cameras'])} camera(s) ({', '.join(meta['cameras'])}), "
            f"{sheet.width}x{sheet.height}, about {meta['tokens_est']} tokens; it is what the AI pane shows the model when this recording is selected as context")
