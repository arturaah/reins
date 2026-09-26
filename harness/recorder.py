"""Write every step of an episode to runs/<mode>_<timestamp>_<slug>/ so prompts can be replayed offline."""
import json
import time
from pathlib import Path

import numpy as np

from .kinematics import ROOT


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


def load_step(run_dir, i):
    """(step record, prompt text, image list) of a recorded step, for the replay tool."""
    d = Path(run_dir) / f"step_{i:03d}"
    rec = json.loads((d / "step.json").read_text())
    prompt = (d / "prompt.txt").read_text() if (d / "prompt.txt").exists() else None
    images = []
    for name, label in (("context.jpg", "CONTEXT VIEW"), ("left.jpg", "LEFT WRIST VIEW"), ("right.jpg", "RIGHT WRIST VIEW")):
        if (d / name).exists():
            images.append((label, (d / name).read_bytes()))
    return rec, prompt, images
