"""One self-contained page for a harness run: every step's cameras, who decided, and why.

    .venv/bin/python tools/run_view.py runs/<run_dir> [--open]

Writes <run_dir>/view.html (images embedded, no server). Per step: the context and wrist frames, the stage, the action
and who chose it (Jev / the scripted decider / geometry / Claude), Claude's looks (why, goal offset, reasoning, stage
complete), the text Jev read about the gap, Jev's answers with their probabilities, any escalation, and the move result.
Runs from the plain loop (Claude every step) show Claude's reasoning instead.
"""
import argparse
import base64
import html
import json
import subprocess
import sys
from pathlib import Path


def img(path):
    if not path.exists():
        return '<div class="noimg">no image</div>'
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{path.stem}">'


def esc(x):
    return html.escape(str(x))


def pct(p):
    return f"{100 * float(p):.0f}%"


def answers_html(ans):
    if not ans:
        return ""
    out = []
    a = ans.get("action")
    if a:
        probs = sorted((a.get("probabilities") or {}).items(), key=lambda kv: -kv[1])[:4]
        bars = "".join(f'<div class="bar"><span class="lab">{esc(k)}</span><span class="track"><span class="fill" '
                       f'style="width:{100 * v:.0f}%"></span></span><span class="num">{pct(v)}</span></div>' for k, v in probs)
        out.append(f'<div class="kv"><b>action</b> {esc(a.get("choice"))} · confidence {float(a.get("confidence") or 0):.2f}</div>{bars}')
    for k in ("stage_done", "needs_look"):
        if k in ans and ans[k].get("noul") is not None:
            out.append(f'<div class="kv"><b>{k}</b> {pct(ans[k]["noul"])} yes</div>')
    return "".join(out)


def step_card(d, rec):
    parts = []
    by = rec.get("decided_by") or ("stage done" if rec.get("stage_done") else ("failed" if rec.get("failed") else "claude"))
    cls = {"claude": "claude", "stage done": "done", "failed": "bad"}.get(by, "fast")
    act = rec.get("action") or "—"
    parts.append(f'<div class="head"><span class="n">step {rec["step"]}</span><span class="stage">{esc(rec.get("stage", ""))}</span>'
                 f'<span class="act">{esc(act)}</span><span class="pill {cls}">{esc(by)}</span></div>')
    parts.append(f'<div class="cams">{img(d / "context.jpg")}{img(next(iter(sorted(d.glob("[lr]*.jpg"))), d / "wrist.jpg"))}</div>')
    for lk in rec.get("looks", []):
        if "error" in lk:
            parts.append(f'<div class="look bad"><b>Claude look failed</b> ({esc(lk["why"])}): {esc(lk["error"])}</div>'); continue
        o = lk.get("goal_offset_cm", {})
        parts.append(f'<div class="look"><div><b>Claude looked</b> <span class="why">{esc(lk["why"])}</span></div>'
                     f'<div class="kv">goal: forward {o.get("forward", 0):+.0f} · left {o.get("left", 0):+.0f} · up {o.get("up", 0):+.0f} cm'
                     f' · {esc(lk.get("confidence", ""))}{" · <b>stage complete</b>" if lk.get("stage_complete") else ""}</div>'
                     f'<div class="reason">{esc(lk.get("reasoning", ""))}</div></div>')
    prompt = d / "prompt.txt"
    dec = rec.get("decider_after_denied_done") or rec.get("decider_after_look") or rec.get("decider")
    if dec is not None and prompt.exists():
        try:
            st = json.loads(prompt.read_text())["state"]
            g = st["goal_relative_to_hand_tip"]
            words = "".join(f"<li>{esc(v)}</li>" for k, v in g.items() if k != "largest_gap")
            parts.append(f'<div class="jev"><div><b>what the fast model read</b></div><ul>{words}</ul>'
                         f'<div class="kv">largest gap: {esc(g.get("largest_gap"))} · last move: {esc(st.get("last_move_result"))}</div>'
                         f'{answers_html(dec.get("answers"))}</div>')
        except (json.JSONDecodeError, KeyError):
            pass                                      # a Claude-controller prompt: shown below from the response
    elif prompt.exists() and (d / "response.txt").exists():
        try:
            r = json.loads((d / "response.txt").read_text())
            parts.append(f'<div class="look"><b>Claude decided</b><div class="reason">{esc(r.get("reasoning", ""))}</div></div>')
        except json.JSONDecodeError:
            pass
    if rec.get("escalation"):
        parts.append(f'<div class="esc">escalated: {esc(rec["escalation"])}</div>')
    if rec.get("feedback"):
        parts.append(f'<div class="res">→ {esc(rec["feedback"])}</div>')
    if rec.get("why"):
        parts.append(f'<div class="res">→ {esc(rec["why"])}</div>')
    return f'<section class="step">{"".join(parts)}</section>'


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d232b;--mut:#5b6573;--line:#dfe3e8;--fast:#1a7f4b;--claude:#7a4fd0;--done:#0f6fae;--bad:#b42318;--track:#e8ebef}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1419;--card:#161c23;--fg:#d6dde5;--mut:#8b96a3;--line:#29313b;--fast:#5fd49a;--claude:#b69cff;--done:#63b3ed;--bad:#ff8a80;--track:#232b35}}
:root[data-theme=dark]{--bg:#0f1419;--card:#161c23;--fg:#d6dde5;--mut:#8b96a3;--line:#29313b;--fast:#5fd49a;--claude:#b69cff;--done:#63b3ed;--bad:#ff8a80;--track:#232b35}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 60px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:14px}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 18px}.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.stat b{display:block;font-size:18px}.stat span{color:var(--mut);font-size:12px}
.plan{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin-bottom:18px}.plan li{margin:3px 0}
.step{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}
.head{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:8px}.n{color:var(--mut)}.stage{font-weight:600}.act{font-family:Menlo,monospace}
.pill{margin-left:auto;border-radius:99px;padding:2px 10px;font-size:12px;color:#fff}.pill.fast{background:var(--fast)}.pill.claude{background:var(--claude)}.pill.done{background:var(--done)}.pill.bad{background:var(--bad)}
.cams{display:grid;grid-template-columns:1fr 1fr;gap:8px}.cams img{width:100%;border-radius:6px;display:block}.noimg{color:var(--mut);padding:30px;text-align:center;border:1px dashed var(--line);border-radius:6px}
.look,.jev,.esc,.res{margin-top:8px;padding:8px 10px;border-radius:6px;border-left:3px solid var(--line)}
.look{border-left-color:var(--claude)}.jev{border-left-color:var(--fast)}.esc{border-left-color:var(--bad);color:var(--bad)}.look.bad{border-left-color:var(--bad)}
.why{color:var(--mut)}.reason{color:var(--mut);font-style:italic;margin-top:3px}.kv{margin-top:3px}ul{margin:4px 0;padding-left:20px}.res{color:var(--mut)}
.bar{display:grid;grid-template-columns:80px 1fr 40px;gap:8px;align-items:center;font-size:12px;margin-top:3px}.lab{font-family:Menlo,monospace}
.track{background:var(--track);border-radius:3px;height:8px;overflow:hidden}.fill{display:block;height:100%;background:var(--fast)}.num{text-align:right;color:var(--mut)}
@media (max-width:640px){.cams{grid-template-columns:1fr}}
"""


def build(run):
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    plan = json.loads((run / "plan.json").read_text()) if (run / "plan.json").exists() else {}
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()] if (run / "steps.jsonl").exists() else []
    summ = meta.get("summary", {})
    stats = [(("success" if summ.get("success") else "failed"), summ.get("reason", "")), (summ.get("steps", len(recs)), "steps")]
    for k, lab in (("decider_steps", "fast-model steps"), ("claude_looks", "Claude looks"), ("claude_steps", "Claude-decided steps")):
        if k in summ:
            stats.append((summ[k], lab))
    stages = "".join(f'<li><b>{esc(s["id"])}</b> — {esc(s.get("completion", ""))}</li>' for s in plan.get("stages") or [])
    body = (f'<h1>{esc(meta.get("task", run.name))}</h1><div class="sub">{esc(meta.get("mode", ""))} · {esc(meta.get("started", ""))} · '
            f'plan by {esc(plan.get("model", "?"))} in {float(plan.get("latency_s") or 0):.0f} s</div>'
            f'<div class="stats">{"".join(f"<div class=stat><b>{esc(v)}</b><span>{esc(l)}</span></div>" for v, l in stats)}</div>'
            f'<div class="plan"><b>Plan</b><ol>{stages}</ol></div>'
            + "".join(step_card(run / f"step_{r['step']:03d}", r) for r in recs))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Run replay</title><style>{CSS}</style></head><body><main>{body}</main></body></html>')
    out = run / "view.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir"); ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    out = build(a.run_dir)
    print(out)
    if a.open and sys.platform == "darwin":
        subprocess.run(["open", str(out)])
