"""What the operator accepted and rejected, kept across sessions and shown to the model. No SDK.

Every Accept / Reject at a PROPOSAL (with the note typed in the window, or after `y` / `n` on the
terminal) is appended to feedback.path as one JSON line: task, stage, action, accepted, note, hand
tip and height at the time. FeedbackStore.block() turns the file into a short prompt block for
later sessions: entries for the same task first, newest first; rejections and accepts that carry a
note are listed individually up to feedback.max_in_prompt, accepts without a note are only counted.
The running session's own decisions are already in the model's history and recovery notes, so the
block skips them.
"""
import json
import time
from pathlib import Path

from .kinematics import ROOT

HEAD = ("OPERATOR FEEDBACK from earlier sessions on this robot (✓ = the operator accepted the move, ✗ = rejected, NOT "
        "executed; quoted notes are the operator's guidance, follow them when the situation is similar):")


class FeedbackStore:
    def __init__(self, path, session=None, max_in_prompt=10):
        self.path = Path(path) if Path(path).is_absolute() else ROOT / path
        self.session = session or time.strftime("%Y%m%d_%H%M%S")
        self.max_in_prompt = int(max_in_prompt)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, task, stage, action, accepted, note="", hand_tip=None, height_cm=None, mode=""):
        entry = {"t": round(time.time(), 3), "session": self.session, "mode": mode, "task": task, "stage": stage,
                 "action": action, "accepted": bool(accepted), "note": (note or "").strip(),
                 "hand_tip_m": [round(float(v), 3) for v in hand_tip] if hand_tip is not None else None,
                 "height_cm": round(float(height_cm), 1) if height_cm is not None else None}
        with self.path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self):
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def block(self, task):
        rows = [r for r in self.entries() if r.get("session") != self.session]
        if not rows:
            return ""
        same = [r for r in rows if r.get("task") == task]
        other = [r for r in rows if r.get("task") != task]
        ordered = same[::-1] + other[::-1]
        detailed = [r for r in ordered if not r["accepted"] or r.get("note")]
        n_bare = sum(1 for r in ordered if r["accepted"] and not r.get("note"))
        lines = []
        for r in detailed[:self.max_in_prompt]:
            where = "this task" if r.get("task") == task else f"task \"{(r.get('task') or '')[:48]}\""
            h = f", hand {r['height_cm']:.0f} cm above the floor" if r.get("height_cm") is not None else ""
            lines.append(f"- {where}, stage {r.get('stage', '?')}{h}: {'✓' if r['accepted'] else '✗'} {r.get('action', '?')}"
                         + (f" — \"{r['note']}\"" if r.get("note") else ""))
        if n_bare:
            lines.append(f"- plus {n_bare} accepted move(s) without comment")
        return "\n".join([HEAD] + lines)
