"""Recorded motions as in-context demonstrations for the VLM. Pure data, no SDK.

A demonstration is one recordings/<name>.json (tools/teach.py, tools/record.py or
tools/arm_lift.py --execute) plus, when the recording was made with the camera frame logger
(tools/framelog.py), its contact sheet recordings/<name>.sheet.jpg: one column per key moment of
the motion, one row per camera. The text that goes with it comes from forward kinematics of the
recorded joints at those moments, so the model gets both what the scene looked like and how far
the hand actually travelled, in the robot frame and in the centimetres its own steps use.

Key moments (smart_moments) are Douglas-Peucker samples of the joint-space path: the fewest
samples whose straight-line interpolation stays within demos.tolerance_rad of every sample, so a
straight reach gives 3 frames and a motion with turns up to demos.max_moments. The frame logger
uses the same function, so a sheet's columns and the text agree. All selected sheets are stacked
into ONE image (demo_images), because with `claude -p` every image is a separate Read turn.
"""
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .kinematics import ARM_JOINTS, ROOT

DEMOS_INTRO = (
    "DEMONSTRATIONS: {n} motion(s) recorded on this robot earlier, chosen by the operator as references for this task. "
    "{sheets}Use them to recognise the objects and the workspace, to see how the hand approached and moved, and to judge "
    "the scale of the motion in centimetres. They are references, not scripts: objects may sit elsewhere now, so decide "
    "from the CURRENT images.")
SHEETS_INTRO = ("Their contact sheets are stacked in ONE image labelled DEMOS: each block is titled DEMO_k and has one row per "
                "camera (named at its left edge) and one column per key moment, left to right in time; where a context row says "
                "\"cropped\", its leftmost tile is the full camera view with a red box showing the region the other tiles are "
                "cropped to. ")
DEFAULTS = {"max_moments": 8, "tolerance_rad": 0.08, "max_width_px": 1568}


@dataclass
class Demo:
    name: str
    text: str
    image: Optional[tuple] = None      # (label, jpeg bytes): the contact sheet, when the recording has one
    path: str = ""
    label: str = ""                    # DEMO_k


def key_moments(times, q, n=6):
    """Indices of n moments at equal joint-space arc-length fractions; first and last always included."""
    times = np.asarray(times, float)
    q = np.asarray(q, float).reshape(len(times), -1)
    if len(times) <= n:
        return list(range(len(times)))
    if float((q.max(axis=0) - q.min(axis=0)).max()) < 0.02:          # nothing moved (sensor noise only): spread in time
        return sorted(set(int(round(v)) for v in np.linspace(0, len(times) - 1, n)))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    idx = [min(int(np.searchsorted(s, v)), len(times) - 1) for v in np.linspace(0.0, s[-1], n)]
    idx[-1] = len(times) - 1
    return sorted(set(idx))


def smart_moments(times, q, tol=0.08, n_max=8, n_min=3, min_gap_s=0.25):
    """Douglas-Peucker on the joint-space path (see the module docstring). Ends always in; moments closer than
    min_gap_s are thinned; more than n_max falls back to equal arc length; fewer than n_min adds arc-length moments."""
    times = np.asarray(times, float)
    q = np.asarray(q, float).reshape(len(times), -1)
    n = len(times)
    if n <= n_min:
        return list(range(n))
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d = q[j] - q[i]; L2 = float(d @ d)
        seg = q[i + 1:j]
        t = np.clip(((seg - q[i]) * d).sum(axis=1) / L2, 0.0, 1.0) if L2 > 1e-12 else np.zeros(len(seg))   # no BLAS: Accelerate warns spuriously
        dist = np.linalg.norm(seg - (q[i] + t[:, None] * d), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep.add(m); stack += [(i, m), (m, j)]
    idx = sorted(keep)
    out = [idx[0]]
    for i in idx[1:]:
        if times[i] - times[out[-1]] >= min_gap_s or i == idx[-1]:
            out.append(i)
    if len(out) > n_max:
        out = key_moments(times, q, n_max)
    if len(out) < n_min:
        out = sorted(set(out) | set(key_moments(times, q, n_min)))
    return out


def _delta_words(d_cm):
    words = []
    for v, pos, neg in ((d_cm[0], "forward", "back"), (d_cm[1], "left", "right"), (d_cm[2], "up", "down")):
        if abs(v) >= 1.0:
            words.append(f"{pos if v > 0 else neg} {abs(v):.0f}")
    return ", ".join(words) or "still"


def describe(src, kins, tol=0.08, n_max=8):
    """Text summary of one recording dict. kins: {"_model": path, "left"/"right": ArmKinematics} (created on demand)."""
    kfs = sorted(src["keyframes"], key=lambda f: f["time_s"])
    times = [float(f["time_s"]) for f in kfs]
    names = sorted(kfs[0]["joint_targets_rad"])
    q = np.array([[float(f["joint_targets_rad"][k]) for k in names] for f in kfs])
    sheet = src.get("sheet") or {}
    moments = [i for i in sheet.get("moments", []) if 0 <= i < len(kfs)] or smart_moments(times, q, tol, n_max)
    span = dict(zip(names, q.max(axis=0) - q.min(axis=0)))
    lines, arms = [], []
    for side in ("left", "right"):
        have = [k for k in ARM_JOINTS[side] if k in names]
        if not have:
            continue
        moved = any(span[k] > 0.02 for k in have)
        arms.append(f"{side} arm {'moved' if moved else 'still'}" + ("" if len(have) == 5 else f" (only {len(have)} of 5 joints recorded)"))
        if not moved or len(have) < 5:
            continue
        kin = kins.get(side)
        if kin is None:
            from .kinematics import ArmKinematics
            kin = kins[side] = ArmKinematics(kins["_model"], side)
        prev = None
        lines.append(f"{side.capitalize()} hand tip at the key moments, robot frame in cm (x forward, y left, z up), "
                     "then its change since the previous moment:")
        for i in moments:
            q5 = [float(kfs[i]["joint_targets_rad"][k]) for k in ARM_JOINTS[side]]
            p, _ = kin.fk(q5)
            p_cm = p * 100.0
            delta = f"   ({_delta_words(p_cm - prev)})" if prev is not None else ""
            lines.append(f"  t={times[i]:.1f}s  x={p_cm[0]:.0f} y={p_cm[1]:.0f} z={p_cm[2]:.0f}{delta}")
            prev = p_cm
    head = f"\"{src.get('name', '?')}\" ({src.get('source', 'recording')}; {times[-1]:.1f} s; " + ", ".join(arms) + ")."
    if sheet:
        cams = ", ".join(sheet.get("cameras", [])) or "cameras"
        head += " Contact sheet columns: " + ", ".join(f"t={times[i]:.1f}s" for i in moments) + f"; rows: {cams}."
    else:
        head += " No camera frames were logged for this recording: text only."
    return "\n".join([head] + lines)


def load_demos(paths, cfg):
    """-> [Demo] in the given order; the k-th is DEMO_k."""
    d = {**DEFAULTS, **(cfg.get("demos") or {})}
    kins = {"_model": cfg["robot"]["model"]}
    demos = []
    for k, path in enumerate(paths, 1):
        p = Path(path)
        p = p if p.is_absolute() else ROOT / p
        src = json.loads(p.read_text())
        label = f"DEMO_{k}"
        text = f"{label}: " + describe(src, kins, float(d["tolerance_rad"]), int(d["max_moments"]))
        image = None
        sheet_file = p.with_name((src.get("sheet") or {}).get("file") or (p.stem + ".sheet.jpg"))
        if sheet_file.exists():
            image = (f"{label} contact sheet of '{src.get('name', p.stem)}'", sheet_file.read_bytes())
        demos.append(Demo(src.get("name", p.stem), text, image, str(p), label))
    return demos


def demos_block(demos):
    """Prompt text for a list of demos; empty when there are none."""
    if not demos:
        return ""
    sheets = SHEETS_INTRO if any(d.image for d in demos) else ""
    return DEMOS_INTRO.format(n=len(demos), sheets=sheets) + "\n\n" + "\n\n".join(d.text for d in demos)


def demo_images(demos, max_w=1568):
    """One image stacking every demonstration's contact sheet, each under a DEMO_k title band; [] when none has one."""
    with_sheet = [d for d in demos if d.image]
    if not with_sheet:
        return []
    from PIL import Image, ImageDraw, ImageFont
    ims = []
    for d in with_sheet:
        try:
            ims.append((d, Image.open(io.BytesIO(d.image[1])).convert("RGB")))
        except Exception:
            pass
    if not ims:
        return []
    W = min(max_w, max(im.width for _, im in ims))
    band, gap = 20, 4
    font = ImageFont.load_default(size=14)
    rows = []
    for d, im in ims:
        if im.width != W:
            im = im.resize((W, max(1, round(im.height * W / im.width))))
        rows.append((d, im))
    H = sum(band + im.height + gap for _, im in rows) - gap
    out = Image.new("RGB", (W, H), (0, 0, 0)); draw = ImageDraw.Draw(out); y = 0
    for d, im in rows:
        draw.text((4, y + 2), f"{d.label}: {d.name}", fill=(255, 220, 80), font=font); y += band
        out.paste(im, (0, y)); y += im.height + gap
    buf = io.BytesIO(); out.save(buf, "JPEG", quality=80)
    return [("DEMOS contact sheets, one block per demonstration, titled DEMO_k", buf.getvalue())]
