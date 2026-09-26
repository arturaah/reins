"""Reins desktop window: cameras | live MuJoCo twin | trajectories and skill recording. Tk, no browser.

Left pane: head camera and both wrist cameras. Middle: the twin. Right: every
recording and plan with Dry run, Execute (behind a confirmation) and Abort, and
a Record block that teaches a new skill by hand (tools/teach.py, behind a
confirmation: both arms go soft and follow your hands) or logs one passively
(tools/record.py, subscribe-only). Finish & save ends a recording early; the
tool writes recordings/NAME.json, the list refreshes with the new file selected,
and Dry run / Execute replay it.
The window only views the streams the servers already serve and runs the tools
as subprocesses, so the same gates apply: dry run first, execute on a go,
Abort sends the tool its interrupt, which ramps the arm weight down.

    tools/start_all.sh            # starts the stream servers, then this window
    .venv/bin/python tools/reins_ui.py [--iface en6]
"""
import argparse, io, os, queue, re, signal, subprocess, sys, threading, time, urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv/bin/python")
STREAMS = {"head": "http://localhost:8081/cam", "twin": "http://localhost:8082/twin",
           "wrist_l": "http://localhost:8080/cam/0", "wrist_r": "http://localhost:8080/cam/2"}
SIZES = {}
def compute_sizes(w, h):
    cam_w = max(240, int(w * 0.27) - 20); twin_w = max(320, int(w * 0.45) - 20); wrist_w = cam_w // 2 - 6
    SIZES.update({"head": (cam_w, cam_w * 9 // 16), "wrist_l": (wrist_w, wrist_w * 9 // 16), "wrist_r": (wrist_w, wrist_w * 9 // 16),
                  "twin": (twin_w, min(twin_w * 3 // 4, max(240, h - 140)))})
def fit_to(im, box):
    k = min(box[0] / im.width, box[1] / im.height)
    return im.resize((max(1, int(im.width * k)), max(1, int(im.height * k)))) if abs(k - 1) > 0.02 else im

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--iface", default="en6")
ap.add_argument("--selftest", metavar="NAME", help="run a 3 s passive log called NAME through the button code path, print the log, exit (subscribe-only)")
a = ap.parse_args()
latest = {k: None for k in STREAMS}
events = queue.Queue()          # ("log", text) ("status", text) ("rec_start", "") ("done", code); threads never touch Tk


def reader(name, url):
    """Follow an MJPEG stream forever; keep only the newest frame."""
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                while True:
                    line = r.readline()
                    if not line: break
                    if line.lower().startswith(b"content-length:"):
                        n = int(line.split(b":")[1]); r.readline()
                        latest[name] = r.read(n)
        except Exception:
            latest[name] = None; time.sleep(1.0)

for k, u in STREAMS.items():
    threading.Thread(target=reader, args=(k, u), daemon=True).start()

root = tk.Tk(); root.title("Reins · R1"); root.configure(bg="#0f1419")
SW, SH = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{SW}x{SH - 80}+0+0"); compute_sizes(SW, SH - 80)
style = ttk.Style(); style.theme_use("clam")
style.configure(".", background="#0f1419", foreground="#c9d1d9", fieldbackground="#161c23")
style.configure("TButton", padding=6); style.configure("Danger.TButton", foreground="#ff6b6b"); style.configure("Go.TButton", foreground="#7ee787")
panes = ttk.PanedWindow(root, orient="horizontal"); panes.pack(fill="both", expand=True)

photos = {}
def blank(name):
    w, h = SIZES[name]; photos[name] = ImageTk.PhotoImage(Image.new("RGB", (w, h), (0, 0, 0))); return photos[name]
def tile(parent, name, title):
    f = ttk.Frame(parent); ttk.Label(f, text=title).pack(anchor="w", padx=6, pady=(6, 0))
    lbl = tk.Label(f, bg="#000", image=blank(name), compound="center", text="waiting for stream…", fg="#666"); lbl.pack(padx=6, pady=4)
    return f, lbl

cams = ttk.Frame(panes); panes.add(cams, weight=0)
_, head_lbl = tile(cams, "head", "Head camera (controller)"); _.pack()
row = ttk.Frame(cams); row.pack()
_, wl_lbl = tile(row, "wrist_l", "Left wrist"); _.pack(side="left")
_, wr_lbl = tile(row, "wrist_r", "Right wrist"); _.pack(side="left")
twin = ttk.Frame(panes); panes.add(twin, weight=1)
_, twin_lbl = tile(twin, "twin", "Live twin (MuJoCo from rt/lowstate)"); _.pack(fill="both", expand=True)
srow = ttk.Frame(twin); srow.pack(fill="x", padx=6, pady=(0, 6))
status = ttk.Label(srow, text="…"); status.pack(side="left")
def cockpit(path):
    """Ask the twin server something (preview start/stop); the answer goes to the log."""
    def go():
        try: events.put(("log", urllib.request.urlopen("http://localhost:8082" + path, timeout=3).read().decode() + "\n"))
        except Exception as e: events.put(("log", f"twin server: {e}\n"))
    threading.Thread(target=go, daemon=True).start()
ttk.Button(srow, text="Stop preview", width=12, command=lambda: cockpit("/preview/stop")).pack(side="right")

# ---- control pane: trajectories ---------------------------------------------------------------
ctrl = ttk.Frame(panes, width=380); panes.add(ctrl, weight=0)
hdr = ttk.Frame(ctrl); hdr.pack(fill="x", padx=6, pady=(6, 0))
ttk.Label(hdr, text="Trajectories").pack(side="left")
files = []
lb = tk.Listbox(ctrl, height=8, bg="#161c23", fg="#c9d1d9", selectbackground="#0f766e", exportselection=False)
def listdir(d):
    p = os.path.join(ROOT, d); return sorted(os.listdir(p)) if os.path.isdir(p) else []
def reload_files(select=None):
    """Rescan tools/plans and recordings; keep or set the selection."""
    if select is None and lb.curselection(): select = files[lb.curselection()[0]]
    files[:] = [os.path.join("tools/plans", f) for f in listdir("tools/plans") if f.endswith(".json")] + \
               [os.path.join("recordings", f) for f in listdir("recordings") if f.endswith(".json")]
    lb.delete(0, "end")
    for f in files: lb.insert("end", f)
    if select in files:
        i = files.index(select); lb.selection_set(i); lb.see(i); lb.activate(i)
ttk.Button(hdr, text="Refresh", width=7, command=reload_files).pack(side="right")
def delete_selected():
    """Delete the selected trajectory file after a confirmation. Plans and recordings are plain files; git has the committed ones."""
    if busy(): return
    sel = lb.curselection()
    if not sel: log("\nselect a trajectory first\n"); return
    f = files[sel[0]]
    if not messagebox.askokcancel("Delete trajectory", f"Delete {f}?\n\nThe file is removed from disk. Committed files can be restored from git."):
        return
    try:
        os.remove(os.path.join(ROOT, f)); log(f"deleted {f}\n")
    except OSError as e:
        log(f"could not delete {f}: {e}\n")
    reload_files(select=files[min(sel[0], len(files) - 2)] if len(files) > 1 else None)
ttk.Button(hdr, text="Delete…", width=7, style="Danger.TButton", command=delete_selected).pack(side="right", padx=4)
reload_files(); lb.pack(fill="x", padx=6, pady=4)
opts = ttk.Frame(ctrl); opts.pack(fill="x", padx=6)
ttk.Label(opts, text="speed").pack(side="left"); speed = tk.StringVar(value="1.0"); ttk.Entry(opts, textvariable=speed, width=6).pack(side="left", padx=4)
ttk.Label(opts, text="kp scale").pack(side="left"); kps = tk.StringVar(value="1.0"); ttk.Entry(opts, textvariable=kps, width=6).pack(side="left", padx=4)
btns = ttk.Frame(ctrl); btns.pack(fill="x", padx=6, pady=4)

# ---- control pane: record a new skill ---------------------------------------------------------
ttk.Label(ctrl, text="Record a new skill").pack(anchor="w", padx=6, pady=(10, 0))
rf = ttk.Frame(ctrl); rf.pack(fill="x", padx=6)
ttk.Label(rf, text="name").grid(row=0, column=0, sticky="w")
rname = tk.StringVar(); ttk.Entry(rf, textvariable=rname, width=28).grid(row=0, column=1, columnspan=3, sticky="w", padx=4, pady=2)
ttk.Label(rf, text="seconds").grid(row=1, column=0, sticky="w")
rsec = tk.StringVar(value="30"); ttk.Entry(rf, textvariable=rsec, width=6).grid(row=1, column=1, sticky="w", padx=4, pady=2)
ttk.Label(rf, text="teach kp").grid(row=1, column=2, sticky="w")
rkp = tk.StringVar(value="20"); ttk.Entry(rf, textvariable=rkp, width=6).grid(row=1, column=3, sticky="w", padx=4, pady=2)
rbtns = ttk.Frame(ctrl); rbtns.pack(fill="x", padx=6, pady=4)
rec_lbl = tk.Label(ctrl, text="", bg="#0f1419", fg="#ff6b6b", font=("Menlo", 11, "bold"), anchor="w"); rec_lbl.pack(fill="x", padx=6)

# ---- control pane: say (onboard text-to-speech via tools/say.py; independent of the trajectory slot) -----------
ttk.Label(ctrl, text="Say").pack(anchor="w", padx=6, pady=(10, 0))
sf = ttk.Frame(ctrl); sf.pack(fill="x", padx=6)
say_text = tk.StringVar(); say_entry = ttk.Entry(sf, textvariable=say_text); say_entry.pack(side="left", fill="x", expand=True)
voice = tk.StringVar(value="English"); ttk.Combobox(sf, textvariable=voice, values=("English", "Chinese"), state="readonly", width=8).pack(side="left", padx=4)
ttk.Label(sf, text="vol").pack(side="left"); vol = tk.StringVar(); ttk.Entry(sf, textvariable=vol, width=4).pack(side="left", padx=2)
def say():
    text = say_text.get().strip()
    if not text: log("\ntype something for the robot to say\n"); return
    cmd = [PY, "tools/say.py", a.iface, text, "--speaker", "0" if voice.get() == "Chinese" else "1"]
    if vol.get().strip(): cmd += ["--volume", vol.get().strip()]
    log(f"\n🔊 {text}\n")
    # Start the child from the main thread: on macOS a fork from a worker thread inside a Tk process can take the
    # window down. Only the waiting and reading happen in a thread (same pattern as launch()).
    try:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except Exception as ex:
        log(f"say failed to start: {ex}\n"); return
    def go():
        try:
            out_, _ = p.communicate(timeout=30)
            lines = [l for l in out_.splitlines() if l.strip() and "take sample error" not in l]
            events.put(("log", "".join(l + "\n" for l in lines) + ("" if p.returncode == 0 else f"■ say failed (code {p.returncode})\n")))
        except Exception as ex:
            p.kill(); events.put(("log", f"say failed: {ex}\n"))
    threading.Thread(target=go, daemon=True).start()
ttk.Button(sf, text="Say", command=say).pack(side="left", padx=4)
say_entry.bind("<Return>", lambda _e: say())

lhdr = ttk.Frame(ctrl); lhdr.pack(fill="x", padx=6, pady=(6, 0))
ttk.Label(lhdr, text="Log").pack(side="left")
out = tk.Text(ctrl, height=14, width=48, bg="#0b0f14", fg="#c9d1d9", font=("Menlo", 10))
ttk.Button(lhdr, text="Clear", width=6, command=lambda: out.delete("1.0", "end")).pack(side="right")
out.pack(fill="both", expand=True, padx=6, pady=4)
proc = {"p": None, "kind": None, "result": None, "t0": None, "secs": 0.0}
SKIP = re.compile(r"^(replay: |\s+\d+ s$|fsm \d)")      # CLI hints and progress the window already shows
EXIT = {0: "", 130: "stopped on request", -2: "stopped on request", 2: "aborted by the tool's own safety check", 3: "refused: the robot is not in FSM 4 or 811",
        4: "plan rejected by the checks, nothing was sent"}

def log(s):
    out.insert("end", s); out.see("end")

def busy():
    if proc["p"] and proc["p"].poll() is None:
        log(f"\n(a {proc['kind']} is still running; stop it first)\n"); return True
    return False

def launch(cmd, kind, title, result=None, secs=0.0):
    """Run one tool as a subprocess; its lines arrive through the event queue. The log keeps earlier runs until Clear."""
    if out.get("1.0", "end").strip(): log("\n")
    log(f"▶ {title}   {time.strftime('%H:%M:%S')}\n")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    proc.update(p=p, kind=kind, result=result, t0=None, secs=secs, finishing=False)
    def pump():
        for line in p.stdout:
            if "take sample error" in line or SKIP.match(line): continue
            if line.startswith(("TEACH:", "recording ")): events.put(("rec_start", ""))   # the tool's own "now recording" line
            if line.startswith("EXECUTE:"): events.put(("exec_start", ""))               # arm_lift has just written the resolved plan
            if (m := re.search(r"Lower the speed to ([0-9.]+)", line)): events.put(("speed", m.group(1)))
            if line.startswith(("releasing", "ramping weight down", "interrupted")): proc["finishing"] = True
            events.put(("log", line))
        events.put(("done", p.wait()))
    threading.Thread(target=pump, daemon=True).start()

def run(execute):
    if busy(): return
    sel = lb.curselection()
    if not sel: log("\nselect a trajectory first\n"); return
    plan = files[sel[0]]
    cmd = [PY, "tools/arm_lift.py", a.iface, "--plan", plan, "--speed", speed.get(), "--kp-scale", kps.get(), "--brief"]
    if execute:
        if not messagebox.askokcancel("Execute on the robot", f"Move the robot through\n{plan}\nat speed {speed.get()}, kp x{kps.get()}?\n\nRobot standing, arms clear, remote in hand."):
            return
        cmd.append("--execute")
    launch(cmd, "execute" if execute else "dry run", f"{'EXECUTE' if execute else 'Dry run'} {plan}  (speed {speed.get()}, kp ×{kps.get()})")

def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def start_recording(mode):
    """mode 'teach': tools/teach.py (arms go soft, publishes rt/arm_sdk). mode 'record': tools/record.py (listens only)."""
    if busy(): return
    name = slug(rname.get())
    if not name: log("\nname the skill first\n"); return
    try:
        secs, kp = float(rsec.get()), float(rkp.get())
    except ValueError:
        log("\nseconds and teach kp must be numbers\n"); return
    path = f"recordings/{name}.json"
    if os.path.exists(os.path.join(ROOT, path)) and not messagebox.askokcancel("Overwrite?", f"{path} exists.\nRecord over it?"):
        return
    if mode == "teach":
        if not messagebox.askokcancel("Teach by hand",
                f"Both arms go soft (kp {kp:g}) and follow your hands for up to {secs:g} s, recording to {path}.\n\n"
                "Robot standing in FSM 4 or 811, someone holding the arms, remote in hand.\n"
                "Finish & save ends early. When the log says 'releasing', let go: the controller takes the arms back."):
            return
        cmd = [PY, "tools/teach.py", a.iface, name, "--seconds", f"{secs:g}", "--kp", f"{kp:g}"]; cockpit("/preview/stop")
    else:
        cmd = [PY, "tools/record.py", a.iface, name, "--seconds", f"{secs:g}"]
    launch(cmd, mode, f"{'Teach by hand' if mode == 'teach' else 'Passive log'} → {path}  (up to {secs:g} s)", result=path, secs=secs)

def interrupt():
    """Ctrl-C to the running tool: arm_lift releases the arm; teach/record save the file (teach releases too)."""
    p = proc["p"]
    if p and p.poll() is None:
        if proc.get("finishing"):
            log("(already finishing: wait for the weight to ramp down)\n"); return
        p.send_signal(signal.SIGINT)
        if proc["kind"] in ("teach", "record"):
            log("\n[finish sent: the tool saves the recording" + (" and releases the arms" if proc["kind"] == "teach" else "") + "]\n")
        else:
            log("\n[abort sent: the tool releases the arm]\n")

def finished(code):
    proc["t0"] = None; rec_lbl.configure(text="")
    msg = EXIT.get(code, f"tool exited with code {code}")
    if msg: log(f"■ {msg}\n")
    res = proc["result"]; ok = code == 0 and res and os.path.exists(os.path.join(ROOT, res))
    reload_files(select=res if ok else None)
    if proc["kind"] == "dry run" and code == 0:
        cockpit("/preview?file=sim/plans/arm_lift_dryrun.json")
        log("the twin pane plays this exact trajectory 3 times (cyan ghost arms, hand paths); the robot itself does not move\n")
    if ok:
        log(f"saved {res}: selected above. Dry run it, then Execute.\n")
    elif proc["kind"] in ("teach", "record"):
        log("no recording saved\n")
    proc["result"] = None

ttk.Button(btns, text="Dry run + preview", command=lambda: run(False)).pack(side="left")
ttk.Button(btns, text="Execute…", command=lambda: run(True)).pack(side="left", padx=6)
ttk.Button(btns, text="Abort", style="Danger.TButton", command=interrupt).pack(side="left")
ttk.Button(rbtns, text="Teach by hand…", command=lambda: start_recording("teach")).pack(side="left")
ttk.Button(rbtns, text="Passive log", command=lambda: start_recording("record")).pack(side="left", padx=6)
ttk.Button(rbtns, text="Finish & save", style="Go.TButton", command=interrupt).pack(side="left")
hint = ttk.Label(ctrl, text="Dry run checks the trajectory and plays it in the twin pane; while anything streams to the arms the twin shows the "
                     "sent pose in yellow. Execute and Teach ask first. Abort / Finish & save = "
                     "Ctrl-C to the tool: a replay ramps the weight down, teach saves the file and releases the arms, passive log saves the file.",
          wraplength=360); hint.pack(anchor="w", padx=6, pady=(0, 6))

def refresh():
    for name, lbl in (("head", head_lbl), ("twin", twin_lbl), ("wrist_l", wl_lbl), ("wrist_r", wr_lbl)):
        jpg = latest.get(name)
        if jpg:
            try:
                im = fit_to(Image.open(io.BytesIO(jpg)), SIZES[name]); photos[name] = ImageTk.PhotoImage(im)
                lbl.configure(image=photos[name], text="")
            except Exception as e:
                lbl.configure(text=f"bad frame: {e}"[:60])
        elif lbl.cget("text") == "":
            lbl.configure(image=blank(name), text="stream lost")
    root.after(66, refresh)

def status_thread():
    while True:
        try:
            txt = urllib.request.urlopen("http://localhost:8082/status", timeout=2).read().decode()
        except Exception:
            txt = "twin server (tools/cockpit.py) not reachable"
        events.put(("status", txt)); time.sleep(1.0)
threading.Thread(target=status_thread, daemon=True).start()
def drain():
    try:
        while True:
            kind, val = events.get_nowait()
            if kind == "log": log(val)
            elif kind == "status": status.configure(text=val)
            elif kind == "rec_start": proc["t0"] = time.time()
            elif kind == "speed":
                speed.set(val); log(f"speed field set to {val}: press Dry run or Execute again\n")
            elif kind == "exec_start":
                cockpit("/preview?file=sim/plans/arm_lift_dryrun.json")
                log("the twin pane shows the planned hand paths and, in yellow, the pose being sent right now\n")
            elif kind == "done": finished(val)
    except queue.Empty:
        pass
    if proc["t0"]:
        rec_lbl.configure(text=f"● {proc['kind'].upper()}  {proc['result']}   {time.time() - proc['t0']:3.0f} / {proc['secs']:g} s")
    root.after(100, drain)

_resize = {"job": None}
def on_resize(e):
    if e.widget is root:
        if _resize["job"]: root.after_cancel(_resize["job"])
        _resize["job"] = root.after(150, lambda: (compute_sizes(root.winfo_width(), root.winfo_height()), place_sashes()))
def place_sashes():
    try:
        w = root.winfo_width(); panes.sashpos(0, int(w * 0.27)); panes.sashpos(1, int(w * 0.27) + int(w * 0.45))
    except Exception: pass
def on_close(deadline=None):
    """Interrupt a running tool and keep reading its output until it has released and saved (up to 6 s), then quit."""
    p = proc["p"]
    if p and p.poll() is None:
        if deadline is None:
            interrupt(); deadline = time.time() + 6.0
        if time.time() < deadline:
            root.after(200, lambda: on_close(deadline)); return
    root.destroy()
root.bind("<Configure>", on_resize)
root.after(300, place_sashes)
root.after(200, refresh); root.after(300, drain)
root.protocol("WM_DELETE_WINDOW", on_close)
def report_callback_exception(exc, val, tb):                       # a Tk callback error must not kill the window silently
    import traceback; msg = "".join(traceback.format_exception(exc, val, tb))
    sys.stderr.write(msg); open("/tmp/reins_ui_crash.log", "a").write(time.strftime("%F %T ") + msg)
    try: log(f"\n■ window error: {val}\n")
    except Exception: pass
root.report_callback_exception = report_callback_exception
if a.selftest and a.selftest.startswith("say:"):                 # e.g. --iface lo0 --selftest "say:hello": exercises the Say path, no robot
    say_text.set(a.selftest[4:]); root.after(1500, say)
    def geometry_report():
        print(f"screen {SW}x{SH}, window {root.winfo_width()}x{root.winfo_height()}, control pane height {ctrl.winfo_height()}")
        for name, w in (("trajectory list", lb), ("record block", rf), ("say row", sf), ("log", out), ("hint", hint)):
            print(f"  {name:16s} mapped={bool(w.winfo_ismapped())} y={w.winfo_y()} h={w.winfo_height()}")
    root.after(4000, geometry_report)
    root.after(9000, lambda: (print(out.get("1.0", "end")), print("window alive after say"), root.destroy()))
elif a.selftest:
    rname.set(a.selftest); rsec.set("3")
    root.after(1500, lambda: start_recording("record"))
    def watch():
        if proc["p"] and proc["p"].poll() is not None and proc["result"] is None:      # finished() has run
            sel = lb.curselection()
            print(out.get("1.0", "end")); print("selected:", files[sel[0]] if sel else None); root.destroy()
        else:
            root.after(200, watch)
    root.after(2000, watch)
def note(msg):                                                     # why did the window end? (/tmp/reins_ui_crash.log)
    try: open("/tmp/reins_ui_crash.log", "a").write(time.strftime("%F %T ") + msg + "\n")
    except Exception: pass
for _sig in (signal.SIGHUP, signal.SIGTERM):
    signal.signal(_sig, lambda n, f: (note(f"signal {signal.Signals(n).name} received, window closing"), sys.exit(128 + n)))
try:
    root.mainloop()
    note("mainloop ended normally (window closed)")
except BaseException as e:
    import traceback; msg = traceback.format_exc()
    sys.stderr.write(msg); note(f"mainloop ended by {type(e).__name__}: {e}\n{msg}"); raise
