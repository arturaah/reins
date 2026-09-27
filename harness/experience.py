"""Past proposals as pictures for the model: what the operator accepted and rejected, kept across sessions. No SDK.

Every answered PROPOSAL becomes one card in experience.dir: the ROBOT POSE VIEW at that moment with the proposed hand
path drawn (cyan line to the trajectory's end, magenta sphere at the end) next to the CONTEXT VIEW the model saw,
captioned with the verdict (✓ accepted and executed / ✗ rejected, NOT executed), the task, the stage, the actions,
the operator's note and, for an executed proposal, what happened. index.jsonl lists them. ExperienceStore.images(task)
stacks the most relevant cards (same task first, newest first, the running session's included, up to
experience.max_in_prompt) into one image that goes with every VLM call, and block(task) is the matching text, so every
later prediction, in this session and in all the next ones, has seen what went down and what went through.
"""
import io
import json
import time
from pathlib import Path

from .kinematics import ROOT

HEAD = ("EXPERIENCE from earlier proposals on this robot: the image titled EXPERIENCE shows them, one card per proposal (EXP_k), "
        "left the ROBOT POSE VIEW at that moment with the proposed hand path drawn as a cyan line ending in a magenta sphere, "
        "right the CONTEXT VIEW the model saw then. ✓ = the operator accepted it and it was executed, ✗ = the operator rejected "
        "it and it was NOT executed; a quoted note is the operator's guidance. Propose trajectories like the accepted ones and "
        "unlike the rejected ones whenever the situation is similar:")
LABEL = ("EXPERIENCE: earlier proposals with the operator's verdict, one card per proposal titled EXP_k "
         "(left: ROBOT POSE VIEW with the proposed hand path in cyan; right: the CONTEXT VIEW at that moment)")
CARD_H = 240


def _fit_height(im, h):
    return im.resize((max(1, round(im.width * h / im.height)), h))


def make_card(title, lines, pose_im=None, context_im=None, width=800, accepted=True):
    """One card: a caption band (title in green / red, then the lines) over the pose view and the context view side by side.
    Missing images are grey boxes that say so. -> PIL image, width px wide."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=13)
    panels = []
    for im, what in ((pose_im, "no ROBOT POSE VIEW (no renderer)"), (context_im, "no CONTEXT VIEW")):
        if im is None:
            box = Image.new("RGB", (round(CARD_H * 16 / 9), CARD_H), (40, 40, 48))
            ImageDraw.Draw(box).text((8, CARD_H // 2 - 8), what, fill=(170, 170, 170), font=font)
            panels.append(box)
        else:
            panels.append(_fit_height(im.convert("RGB"), CARD_H))
    band = 18 * (1 + len(lines)) + 4
    w = sum(p.width for p in panels) + 4
    out = Image.new("RGB", (w, band + CARD_H), (0, 0, 0))
    d = ImageDraw.Draw(out)
    d.text((4, 2), title, fill=(80, 220, 120) if accepted else (255, 110, 110), font=font)
    for i, line in enumerate(lines):
        d.text((4, 2 + 18 * (i + 1)), line, fill=(230, 230, 230), font=font)
    x = 0
    for pnl in panels:
        out.paste(pnl, (x, band)); x += pnl.width + 4
    if out.width != width:
        out = out.resize((width, max(1, round(out.height * width / out.width))))
    return out


class ExperienceStore:
    def __init__(self, dir, session=None, max_in_prompt=6, card_width=800, mode=""):
        self.dir = Path(dir) if Path(dir).is_absolute() else ROOT / dir
        self.index = self.dir / "index.jsonl"
        self.session = session or time.strftime("%Y%m%d_%H%M%S")
        self.max_in_prompt, self.card_width, self.mode = int(max_in_prompt), int(card_width), mode
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n_session = 0

    def add(self, task, stage, actions, accepted, note="", outcome="", pose_im=None, context_im=None, mode=None):
        """One answered proposal -> a card file and an index line. actions: the proposal's tokens as text."""
        self.n_session += 1
        name = f"{self.session}_{self.n_session:03d}_{'ok' if accepted else 'no'}.jpg"
        note, outcome = (note or "").strip(), (outcome or "").strip()
        title = f"{'✓ ACCEPTED, executed' if accepted else '✗ REJECTED, not executed'}   task: {task[:60]}   stage: {stage}"
        lines = [f"proposal: {actions[:110]}"]
        if note:
            lines.append(f"operator: \"{note[:110]}\"")
        if accepted and outcome:
            lines.append(f"outcome: {outcome[:120]}")
        make_card(title, lines, pose_im, context_im, self.card_width, accepted).save(self.dir / name, "JPEG", quality=80)
        entry = {"t": round(time.time(), 3), "session": self.session, "mode": mode if mode is not None else self.mode, "task": task,
                 "stage": stage, "actions": actions, "accepted": bool(accepted), "note": note, "outcome": outcome, "image": name}
        with self.index.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self):
        if not self.index.exists():
            return []
        out = []
        for line in self.index.read_text().splitlines():
            if line.strip():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (self.dir / e.get("image", "")).exists():
                    out.append(e)
        return out

    def select(self, task):
        """The cards for a call: this task's newest first, then the others newest first, up to max_in_prompt."""
        rows = self.entries()
        same = [r for r in rows if r.get("task") == task][::-1]
        other = [r for r in rows if r.get("task") != task][::-1]
        return (same + other)[:self.max_in_prompt]

    def block(self, task):
        sel = self.select(task)
        if not sel:
            return ""
        lines = []
        for k, r in enumerate(sel, 1):
            where = "this task" if r.get("task") == task else f"task \"{(r.get('task') or '')[:48]}\""
            lines.append(f"- EXP_{k} {'✓' if r['accepted'] else '✗'} {where}, stage {r.get('stage', '?')}: {r.get('actions', '?')}"
                         + (f" — \"{r['note']}\"" if r.get("note") else "") + (f" (outcome: {r['outcome']})" if r["accepted"] and r.get("outcome") else ""))
        return "\n".join([HEAD] + lines)

    def images(self, task):
        """[(label, jpeg)]: one image stacking the selected cards, each under an EXP_k band; [] when there is none."""
        sel = self.select(task)
        if not sel:
            return []
        from PIL import Image, ImageDraw, ImageFont
        ims = []
        for k, r in enumerate(sel, 1):
            try:
                ims.append((k, r, Image.open(self.dir / r["image"]).convert("RGB")))
            except Exception:
                pass
        if not ims:
            return []
        W = self.card_width
        band, gap = 18, 4
        font = ImageFont.load_default(size=13)
        rows = [(k, r, im if im.width == W else im.resize((W, max(1, round(im.height * W / im.width))))) for k, r, im in ims]
        H = sum(band + im.height + gap for _, _, im in rows) - gap
        out = Image.new("RGB", (W, H), (0, 0, 0)); draw = ImageDraw.Draw(out); y = 0
        for k, r, im in rows:
            draw.text((4, y + 2), f"EXP_{k}  {'✓ accepted' if r['accepted'] else '✗ rejected'}", fill=(255, 220, 80), font=font); y += band
            out.paste(im, (0, y)); y += im.height + gap
        buf = io.BytesIO(); out.save(buf, "JPEG", quality=80)
        return [(LABEL, buf.getvalue())]
