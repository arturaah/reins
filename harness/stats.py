"""Per-call inference log and the "inference time vs context" plot. No SDK.

Every VLM call appends one JSON line to stats.path: latency, the tokens the provider reported
(for `claude -p` nearly all of the input is a cache read, so input_tokens alone says little) and an
estimated context size that is comparable across providers: text characters / 4 plus image pixels
/ 750 (Claude's image rule of thumb). After each call the scatter plot stats.plot is redrawn
(matplotlib, Agg backend): x = estimated context tokens, y = latency in seconds, this session's
calls in colour (plan = triangle, act = dot), earlier sessions grey. The desktop window shows the
PNG under the twin and reloads it whenever the file changes.
"""
import io
import json
import time
from pathlib import Path

from .kinematics import ROOT


def _abs(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def image_pixels(images):
    from PIL import Image
    total = 0
    for _, jpg in images:
        try:
            with Image.open(io.BytesIO(jpg)) as im:
                total += im.width * im.height
        except Exception:
            pass
    return total


def context_estimate(prompt_chars, pixels):
    """Comparable across providers: text at 4 characters per token, images at 750 pixels per token."""
    return int(prompt_chars / 4 + pixels / 750)


class InferenceLog:
    def __init__(self, path, plot_path=None, session=None, mode="", task=""):
        self.path = _abs(path)
        self.plot_path = _abs(plot_path) if plot_path else None
        self.session = session or time.strftime("%Y%m%d_%H%M%S")
        self.mode, self.task = mode, task
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.last = None

    def record(self, kind, resp, prompt, images):
        px = image_pixels(images)
        raw = resp.raw or {}
        entry = {"t": round(time.time(), 3), "session": self.session, "mode": self.mode, "task": self.task[:80], "kind": kind,
                 "model": resp.model, "latency_s": round(float(resp.latency_s), 3),
                 "prompt_chars": len(prompt), "n_images": len(images), "image_px": px,
                 "context_est_tokens": context_estimate(len(prompt), px),
                 "input_tokens": int(resp.input_tokens or 0), "output_tokens": int(resp.output_tokens or 0),
                 "cache_read_tokens": int(raw.get("cache_read_input_tokens") or 0),
                 "cache_creation_tokens": int(raw.get("cache_creation_input_tokens") or 0),
                 "cost_usd": raw.get("total_cost_usd"), "error": resp.error or ""}
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self.count += 1
        self.last = entry
        if self.plot_path:
            try:
                self.plot()
            except Exception as e:                       # a plotting problem must never stop the loop
                entry["plot_error"] = str(e)
        return entry

    @staticmethod
    def load(path):
        p = _abs(path)
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def plot(self, path=None):
        """Scatter of every logged call, this session highlighted. Returns the PNG path."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path = _abs(path) if path else self.plot_path
        rows = [r for r in self.load(self.path) if not r.get("error")]
        bg, fg, panel = "#0f1419", "#c9d1d9", "#161c23"
        fig, ax = plt.subplots(figsize=(6.4, 2.0), dpi=100)
        fig.patch.set_facecolor(bg); ax.set_facecolor(panel)
        old = [r for r in rows if r["session"] != self.session]
        cur = [r for r in rows if r["session"] == self.session]
        if old:
            ax.scatter([r["context_est_tokens"] for r in old], [r["latency_s"] for r in old], s=12, c="#6e7681", alpha=0.6,
                       label=f"earlier sessions ({len(old)})")
        for kind, marker, color in (("plan", "^", "#f5a623"), ("act", "o", "#4fc3f7")):
            pts = [r for r in cur if r["kind"] == kind]
            if pts:
                ax.scatter([r["context_est_tokens"] for r in pts], [r["latency_s"] for r in pts], s=26, marker=marker, c=color,
                           label=f"this session: {kind} ({len(pts)})")
        ax.set_xlabel("context, estimated tokens (text/4 + image pixels/750)", fontsize=8, color=fg)
        ax.set_ylabel("inference time, s", fontsize=8, color=fg)
        ax.tick_params(labelsize=7, colors=fg)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.grid(alpha=0.2, color=fg)
        if rows:
            last = rows[-1]
            ax.set_title(f"{len(rows)} calls · last: {last['latency_s']:.1f} s at ~{last['context_est_tokens']} tokens, "
                         f"{last['n_images']} image(s), {last['model']}", fontsize=8, color=fg)
            ax.legend(fontsize=7, loc="upper left", frameon=False, labelcolor=fg)
            ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        else:
            ax.set_title("no inference calls logged yet", fontsize=8, color=fg)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
