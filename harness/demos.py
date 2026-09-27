"""Recorded motions as in-context demonstrations for the VLM. Pure data, no SDK.

A demonstration is one recordings/<name>.json (tools/teach.py, tools/record.py or
tools/arm_lift.py --execute) plus, when the recording was made with the camera frame logger
(tools/framelog.py), its contact sheet recordings/<name>.sheet.jpg: one column per key moment of
the motion, one row per camera. The text that goes with it comes from forward kinematics of the
recorded joints at those moments, so the model gets both what the scene looked like and how far
the hand actually travelled, in the robot frame and in the centimetres its own steps use.
key_moments() is shared with the frame logger so the sheet's columns and the text agree.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .kinematics import ARM_JOINTS, ROOT

DEMOS_INTRO = (
    "DEMONSTRATIONS: {n} motion(s) recorded on this robot earlier, chosen by the operator as references for this task. "
    "Each contact sheet image (label DEMO_k) has one column per key moment, left to right in time, and one row per camera "
    "(named at its left edge). Use them to recognise the objects and the workspace, to see how the hand approached and "
    "moved, and to judge the scale of the motion in centimetres. They are references, not scripts: objects may sit "
    "elsewhere now, so decide from the CURRENT images.")


@dataclass
class Demo:
    name: str
    text: str
    image: Optional[tuple] = None      # (label, jpeg bytes) for the prompt, when a contact sheet exists
    path: str = ""


def key_moments(times, q, n=6):
    """Indices of n moments of a motion at equal joint-space arc-length fractions; first and last always included."""
    times = np.asarray(times, float)
    q = np.asarray(q, float).reshape(len(times), -1)
    if len(times) <= n:
        return list(range(len(times)))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    if s[-1] < 0.05:                                                  # nothing moved (sensor noise only): spread in time
        return sorted(set(int(round(v)) for v in np.linspace(0, len(times) - 1, n)))
    idx = [min(int(np.searchsorted(s, v)), len(times) - 1) for v in np.linspace(0.0, s[-1], n)]
    idx[-1] = len(times) - 1
    return sorted(set(idx))


def _delta_words(d_cm):
    words = []
    for v, pos, neg in ((d_cm[0], "forward", "back"), (d_cm[1], "left", "right"), (d_cm[2], "up", "down")):
        if abs(v) >= 1.0:
            words.append(f"{pos if v > 0 else neg} {abs(v):.0f}")
    return ", ".join(words) or "still"


def describe(src, kins, n_moments=6):
    """Text summary of one recording dict. kins: {"left": ArmKinematics, "right": ArmKinematics} (created on demand)."""
    kfs = sorted(src["keyframes"], key=lambda f: f["time_s"])
    times = [float(f["time_s"]) for f in kfs]
    names = sorted(kfs[0]["joint_targets_rad"])
    q = np.array([[float(f["joint_targets_rad"][k]) for k in names] for f in kfs])
    sheet = src.get("sheet") or {}
    moments = [i for i in sheet.get("moments", []) if 0 <= i < len(kfs)] or key_moments(times, q, n_moments)
    span = dict(zip(names, q.max(axis=0) - q.min(axis=0)))
    lines = []
    arms = []
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
    head = (f"\"{src.get('name', '?')}\" ({src.get('source', 'recording')}; {times[-1]:.1f} s; " + ", ".join(arms) + ").")
    if sheet:
        cams = ", ".join(sheet.get("cameras", [])) or "cameras"
        head += f" Contact sheet columns: " + ", ".join(f"t={times[i]:.1f}s" for i in moments) + f"; rows: {cams}."
    else:
        head += " No camera frames were logged for this recording: text only."
    return "\n".join([head] + lines)


def load_demos(paths, cfg, n_moments=6):
    """-> [Demo] in the given order; the k-th gets the image label DEMO_k."""
    kins = {"_model": cfg["robot"]["model"]}
    demos = []
    for k, path in enumerate(paths, 1):
        p = Path(path)
        p = p if p.is_absolute() else ROOT / p
        src = json.loads(p.read_text())
        text = f"DEMO_{k}: " + describe(src, kins, n_moments)
        image = None
        sheet_file = p.with_name((src.get("sheet") or {}).get("file") or (p.stem + ".sheet.jpg"))
        if sheet_file.exists():
            image = (f"DEMO_{k} contact sheet of '{src.get('name', p.stem)}'", sheet_file.read_bytes())
        demos.append(Demo(src.get("name", p.stem), text, image, str(p)))
    return demos


def demos_block(demos):
    """Prompt text for a list of demos; empty when there are none."""
    if not demos:
        return ""
    return DEMOS_INTRO.format(n=len(demos)) + "\n\n" + "\n\n".join(d.text for d in demos)


def demo_images(demos):
    return [d.image for d in demos if d.image]
