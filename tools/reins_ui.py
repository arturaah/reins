"""Reins desktop window: cameras | live MuJoCo twin | trajectory control. Tk, no browser.

Left pane: head camera and both wrist cameras. Middle: the twin. Right: every
recording and plan, with Dry run, Execute (behind a confirmation) and Abort.
It only views the streams the servers already serve and runs tools/arm_lift.py
as a subprocess, so the same gates apply: dry run first, execute on a go,
Abort sends the tool its interrupt, which ramps the arm weight down.

    tools/start_all.sh            # starts the stream servers, then this window
    .venv/bin/python tools/reins_ui.py [--iface en6]
"""
import argparse, io, os, queue, signal, subprocess, sys, threading, time, urllib.request
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
a = ap.parse_args()
latest = {k: None for k in STREAMS}
events = queue.Queue()          # ("log", text) or ("status", text); Tk is not thread-safe, so threads never call it


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
style.configure("TButton", padding=6); style.configure("Danger.TButton", foreground="#ff6b6b")
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
status = ttk.Label(twin, text="…"); status.pack(anchor="w", padx=6, pady=(0, 6))

ctrl = ttk.Frame(panes, width=380); panes.add(ctrl, weight=0)
ttk.Label(ctrl, text="Trajectories").pack(anchor="w", padx=6, pady=(6, 0))
files = sorted(os.path.join("tools/plans", f) for f in os.listdir(os.path.join(ROOT, "tools/plans"))) + \
        sorted(os.path.join("recordings", f) for f in os.listdir(os.path.join(ROOT, "recordings")) if f.endswith(".json"))
lb = tk.Listbox(ctrl, height=14, bg="#161c23", fg="#c9d1d9", selectbackground="#0f766e", exportselection=False)
for f in files: lb.insert("end", f)
lb.pack(fill="x", padx=6, pady=4)
opts = ttk.Frame(ctrl); opts.pack(fill="x", padx=6)
ttk.Label(opts, text="speed").pack(side="left"); speed = tk.StringVar(value="1.0"); ttk.Entry(opts, textvariable=speed, width=6).pack(side="left", padx=4)
ttk.Label(opts, text="kp scale").pack(side="left"); kps = tk.StringVar(value="1.0"); ttk.Entry(opts, textvariable=kps, width=6).pack(side="left", padx=4)
btns = ttk.Frame(ctrl); btns.pack(fill="x", padx=6, pady=4)
out = tk.Text(ctrl, height=22, width=48, bg="#0b0f14", fg="#c9d1d9", font=("Menlo", 10)); out.pack(fill="both", expand=True, padx=6, pady=4)
proc = {"p": None}

def log(s):
    out.insert("end", s); out.see("end")

def run(execute):
    if proc["p"] and proc["p"].poll() is None:
        log("\n(a run is still active; Abort it first)\n"); return
    sel = lb.curselection()
    if not sel: log("\nselect a trajectory first\n"); return
    plan = files[sel[0]]
    cmd = [PY, "tools/arm_lift.py", a.iface, "--plan", plan, "--speed", speed.get(), "--kp-scale", kps.get()]
    if execute:
        if not messagebox.askokcancel("Execute on the robot", f"Move the robot through\n{plan}\nat speed {speed.get()}, kp x{kps.get()}?\n\nRobot standing, arms clear, remote in hand."):
            return
        cmd.append("--execute")
    out.delete("1.0", "end"); log("$ " + " ".join(cmd[1:]) + "\n")
    proc["p"] = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    p = proc["p"]
    def pump():
        for line in p.stdout:
            if "take sample error" in line: continue
            events.put(("log", line))
        events.put(("log", f"\n[exit {p.wait()}]\n"))
    threading.Thread(target=pump, daemon=True).start()

def abort():
    p = proc["p"]
    if p and p.poll() is None:
        p.send_signal(signal.SIGINT); log("\n[abort sent: the tool releases the arm]\n")

ttk.Button(btns, text="Dry run", command=lambda: run(False)).pack(side="left")
ttk.Button(btns, text="Execute…", command=lambda: run(True)).pack(side="left", padx=6)
ttk.Button(btns, text="Abort", style="Danger.TButton", command=abort).pack(side="left")
ttk.Label(ctrl, text="Execute asks for confirmation. Abort = Ctrl-C to the tool (weight ramps down).", wraplength=360).pack(anchor="w", padx=6, pady=(0, 6))

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
            kind, txt = events.get_nowait()
            if kind == "log": log(txt)
            else: status.configure(text=txt)
    except queue.Empty:
        pass
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
root.bind("<Configure>", on_resize)
root.after(300, place_sashes)
root.after(200, refresh); root.after(300, drain)
root.protocol("WM_DELETE_WINDOW", lambda: (abort(), root.destroy()))
root.mainloop()
