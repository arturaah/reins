"""Write every step of an episode to runs/<mode>_<timestamp>_<slug>/ so prompts can be replayed offline, and export the
episode's accepted moves as a recording (recordings/ai_*.json plus a contact sheet from the step images) that the
trajectory tools replay and the AI pane offers as context."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .kinematics import ARM_JOINTS, ROOT


def _json(o):
    if isinstance(o, np.ndarray):
        return [round(float(v), 5) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


class Recorder:
    def __init__(self, cfg, mode, task, root=None):
        slug = "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:32] or "task"
        base = Path(root or cfg["recorder"]["root"])
        base = base if base.is_absolute() else ROOT / base
        self.dir = base / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"task": task, "mode": mode, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": cfg}
        self._write("meta.json", self.meta)
        self.steps_file = self.dir / "steps.jsonl"

    def _write(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=1, default=_json) + "\n")

    def plan(self, prompt, response, stages, images):
        (self.dir / "plan_prompt.txt").write_text(prompt)
        (self.dir / "plan_response.txt").write_text(response.text)
        self._write("plan.json", {"stages": stages, "model": response.model, "latency_s": response.latency_s,
                                  "input_tokens": response.input_tokens, "output_tokens": response.output_tokens, "error": response.error})
        for label, jpg in images:
            (self.dir / f"plan_{label.split()[0].lower()}.jpg").write_bytes(jpg)

    def step(self, i, record, packet=None, prompt=None, response=None):
        d = self.dir / f"step_{i:03d}"
        d.mkdir(exist_ok=True)
        if packet is not None:
            for label, jpg in packet.images:
                (d / f"{label.split()[0].lower()}.jpg").write_bytes(jpg)
        if prompt is not None:
            (d / "prompt.txt").write_text(prompt)
        if response is not None:
            (d / "response.txt").write_text(response.text or response.error)
            record = {**record, "model": response.model, "latency_s": round(response.latency_s, 3),
                      "input_tokens": response.input_tokens, "output_tokens": response.output_tokens, "vlm_error": response.error}
        record = {"step": i, "t": time.time(), **record}
        (d / "step.json").write_text(json.dumps(record, indent=1, default=_json) + "\n")
        with self.steps_file.open("a") as f:
            f.write(json.dumps(record, default=_json) + "\n")

    def finish(self, summary):
        self.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S"); self.meta["summary"] = summary
        self._write("meta.json", self.meta)
        return self.dir

    def export_recording(self, arm, out_dir=None, pause_s=0.5, name=None):
        """The episode's executed moves as a replayable recording in the sim contract, with a contact sheet built from the
        images the model saw before each move. Time is compressed: each move takes the duration the gate gave it (from the
        joint change and the speed cap), then pause_s; the thinking time between moves is dropped. Samples are dense
        (20 Hz, cosine-eased like the streamed frames) so tools/arm_lift.py treats the file as a recording: it keeps the
        first sample and approaches it from the measured pose with a lead-in. -> (path, message)."""
        steps = [json.loads(l) for l in self.steps_file.read_text().splitlines() if l.strip()] if self.steps_file.exists() else []
        moves = []                                                   # (step record, move record, first move of that step)
        for s in steps:
            for k, m in enumerate(s.get("moves") or [s]):            # a trajectory step carries one record per move
                if m.get("ok") and m.get("q_target") and m.get("joints_before"):
                    moves.append((s, m, k == 0))
        if not moves:
            return None, "no executed moves to export"
        lim = self.meta["config"]["limits"]
        names = ARM_JOINTS[arm]
        keep = ARM_JOINTS["left"] + ARM_JOINTS["right"] + ["waist_yaw_joint"]      # what the arm topic carries (no waist roll)
        hz, kfs, frames, t = 20.0, [], {}, 0.0
        for s, m, first in moves:
            before = {k: round(float(v), 5) for k, v in m["joints_before"].items() if k in keep}
            after = dict(before); after.update({n: round(float(v), 5) for n, v in zip(names, m["q_target"])})
            dq = max(abs(after[n] - before.get(n, after[n])) for n in names)
            dur = max(float(lim["min_move_s"]), dq / float(lim["max_joint_vel_rad_s"]) * math.pi / 2)
            d = self.dir / f"step_{int(s['step']):03d}"
            for fname, cam in (("context.jpg", "context"), ("left.jpg", "left wrist"), ("right.jpg", "right wrist")):
                if first and (d / fname).exists():
                    frames.setdefault(cam, []).append((t, (d / fname).read_bytes()))
            n = max(1, int(round(dur * hz)))
            for i in range(n + 1):                                   # the move, cosine-eased, one sample per 1/hz
                r = 0.5 - 0.5 * math.cos(math.pi * i / n)
                kfs.append({"time_s": round(t + dur * i / n, 3),
                            "joint_targets_rad": {k: round(before[k] + (after[k] - before[k]) * r, 5) for k in before}})
            t += dur
            for i in range(1, int(round(pause_s * hz)) + 1):         # then hold
                kfs.append({"time_s": round(t + i / hz, 3), "joint_targets_rad": dict(after)})
            t += pause_s
        mode, task = self.meta["mode"], self.meta["task"]
        slug = "".join(c if c.isalnum() else "_" for c in task.lower()).strip("_")[:32] or "task"
        name = name or f"ai_{mode}_{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
        out = Path(out_dir) if out_dir else ROOT / "recordings"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.json"
        path.write_text(json.dumps({"schema_version": 1, "name": name, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "source": f"harness episode ({mode}): commanded targets of the {len(moves)} accepted moves, thinking time removed",
                                    "task": task, "arm": arm, "run_dir": str(self.dir), "duration_s": kfs[-1]["time_s"], "keyframes": kfs}, indent=1) + "\n")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools.framelog import save_sheet                      # no SDK in there; it lives with the recording tools
        msg = save_sheet(path, [(k["time_s"], k["joint_targets_rad"]) for k in kfs], frames)
        return path, f"{len(moves)} accepted move(s), {kfs[-1]['time_s']:.1f} s; {msg}"


def load_step(run_dir, i):
    """(step record, prompt text, image list) of a recorded step, for the replay tool."""
    d = Path(run_dir) / f"step_{i:03d}"
    rec = json.loads((d / "step.json").read_text())
    prompt = (d / "prompt.txt").read_text() if (d / "prompt.txt").exists() else None
    images = []
    for name, label in (("context.jpg", "CONTEXT VIEW"), ("left.jpg", "LEFT WRIST VIEW"), ("right.jpg", "RIGHT WRIST VIEW"), ("robot.jpg", "ROBOT POSE VIEW")):
        if (d / name).exists():
            images.append((label, (d / name).read_bytes()))
    return rec, prompt, images
