"""Command line for the VLM end-effector harness.

    python -m harness sim "move your hand above the block" [--vlm scripted|anthropic] [--realtime]
    python -m harness dry-run en6 "..."      real joint state and cameras, VLM calls and IK, nothing published
    python -m harness live en6 "..."         needs `python -m harness.robot.arm_stream en6` running; asks before every move
        --demos recordings/a.json ...        selected recordings (text summary + contact sheet) as demonstrations in every call
        --confirm --preview runs/ui_preview.json   ask before every move in any mode and write each proposal as a plan file
                                             first (the desktop window's AI pane drives this: PROPOSAL lines, Accept/Reject)
    python -m harness packet [--sim | --iface en6]   dump one perception packet (images + proprio text) and exit
    python -m harness replay RUN_DIR --step N [--vlm anthropic]   rebuild a recorded step's prompt and re-query
    python -m harness measure-table en6      print the hand tip height from rt/lowstate (subscribe-only)
Options: --config FILE, --set key.path=value (repeatable), --arm left|right, --profile precision|coarse_fine.
Keyboard while running: type x + Enter for the e-stop (arms freeze, episode ends); Ctrl-C releases the arms.
At a PROPOSAL: Enter sends it, `y <note>` sends it and passes the note to the model, `n` rejects it, `n <note>` rejects it
with the note, x is the e-stop. Every answer is appended to feedback.path and shown to the model in later sessions.
"""
import argparse
import json
import queue
import sys
import threading
import time

from . import config as hcfg
from .demos import load_demos
from .executor import ArmExecutor
from .feedback import FeedbackStore
from .kinematics import ROOT, ArmKinematics
from .perception import Perception, height_above_table_cm
from .recorder import Recorder, load_step
from .safety import SafetyGate
from .vlm import base as vlm_base

stdin_lines = queue.Queue()


def stdin_reader():
    for line in sys.stdin:
        stdin_lines.put(line.strip())


def make_confirm(gate, backend, preview_file=None, review_file=None, mode="live"):
    """Ask on the terminal before a move; 'x' at any time is the e-stop. With preview_file the proposal is written there
    as a plan first (the twin's ghost). PROPOSAL lines and the answers are the desktop window's protocol."""
    from .preview import write_plan
    from spectacles.review import ReviewMailbox

    mailbox = ReviewMailbox(ROOT / review_file if review_file else None) if review_file else None
    if mailbox and not preview_file:
        raise ValueError("Spectacles review needs --preview so the exact proposal can be shown")

    def confirm(text, preview=None):
        proposal = None
        if preview_file and preview is not None:
            plan_path = write_plan(preview_file, preview["arm"], preview["q_now"], preview["frames"], preview["dt"], preview["joints"], text)
            if mailbox:
                proposal = mailbox.propose(plan_path, text, mode)
        print(f"\nPROPOSAL: {text}")
        print("  [Enter] send   y [note] Enter send with a note   n [note] Enter reject   x Enter e-stop > ", end="", flush=True)
        try:
            while True:
                try:
                    line = stdin_lines.get(timeout=0.1)
                    break
                except queue.Empty:
                    if proposal:
                        decision = mailbox.take(proposal["id"], plan_path)
                        if decision:
                            print("\nSpectacles: " + decision, flush=True)
                            return decision == "approve"
        finally:
            if proposal:
                mailbox.clear(proposal["id"])
        print()
        if line == "x":
            gate.estop.set(); backend.freeze(); print("E-STOP set"); return False
        if line == "":
            return True
        if line[:1] == "y":
            return True, line[1:].strip()
        note = line[1:].strip() if line[:1] == "n" else line.strip()
        return False, note
    return confirm


def estop_watch(gate, backend):
    """Background: an 'x' line at any moment sets the e-stop (used when --no-confirm)."""
    while True:
        if stdin_lines.get() == "x":
            gate.estop.set(); backend.freeze(); print("E-STOP set: arms frozen, the episode ends after the current step")


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def build(cfg, mode, iface=None, log=print):
    """-> (backend, executor, perception, table_z)"""
    arm = cfg["robot"]["arm"]
    kin = ArmKinematics(cfg["robot"]["model"], arm, float(cfg["limits"]["joint_margin_rad"]))
    if mode == "sim":
        from .sim.mock_robot import MockBackend
        from .perception import MockCameras
        backend = MockBackend(cfg, realtime=cfg.get("_realtime", False))
        gate = SafetyGate(cfg, kin, None, live=False)
        cams = MockCameras(backend, arm, int(cfg["perception"]["width_px"]), int(cfg["perception"]["width_px"]) * 9 // 16)
        per = Perception(cfg, arm, cams, backend.context_camera(int(cfg["perception"]["width_px"]), int(cfg["perception"]["width_px"]) * 9 // 16))
    else:
        from .perception import HttpCameras
        table_z = cfg["workspace"]["table_z_m"]
        if mode == "dry-run":
            from .robot.dry_run import DryRunBackend
            backend = DryRunBackend(iface, log)
            if table_z is None:
                log("WARNING: workspace.table_z_m is not measured; the dry run uses the sim table height")
            gate = SafetyGate(cfg, kin, table_z, live=False)
        else:
            from .robot.arm_client import ArmClientBackend
            backend = ArmClientBackend(cfg, log)
            gate = SafetyGate(cfg, kin, table_z, live=True)           # refuses without a measured table
        per = Perception(cfg, arm, HttpCameras(cfg, arm), cfg["perception"].get("context_camera"))
    ex = ArmExecutor(cfg, kin, gate, backend, arm)
    return backend, ex, per


def run_episode(a, cfg, mode):
    from .loop import Episode
    log = print
    backend, ex, per = build(cfg, mode, a.iface, log)
    vlm = vlm_base.make(cfg, a.vlm)
    demos = load_demos(a.demos, cfg) if getattr(a, "demos", None) else []
    for d in demos:
        print(f"demo: {d.name} ({'with' if d.image else 'no'} contact sheet, {len(d.text)} chars)")
    threading.Thread(target=stdin_reader, daemon=True).start()
    ask = (mode == "live" and not a.no_confirm) or getattr(a, "confirm", False)
    if ask:
        ex.confirm = make_confirm(ex.gate, backend, getattr(a, "preview", None),
                                  getattr(a, "spectacles_review", None), mode)
    elif mode != "dry-run":
        threading.Thread(target=estop_watch, args=(ex.gate, backend), daemon=True).start()
    if mode == "live":
        if backend.fsm not in (4, 811):
            sys.exit(f"refusing: FSM {backend.fsm} = {backend.fsm_name}")
        print("ENGAGE: the streamer takes the arms (weight ramps to 1) and holds them until the episode ends.")
        print("  Enter to continue, anything else to quit > ", end="", flush=True)
        if stdin_lines.get() != "":
            return
        backend.engage()
    rec = Recorder(cfg, mode, a.task)
    print(f"recording to {rec.dir}")
    from .stats import InferenceLog
    session = rec.dir.name
    stats = InferenceLog(cfg["stats"]["path"], cfg["stats"]["plot"], session, mode, a.task)
    feedback = FeedbackStore(cfg["feedback"]["path"], session, cfg["feedback"]["max_in_prompt"])
    try:
        if mode != "sim" or a.start_pose:
            r = ex.go_to_joints(cfg["robot"]["start_pose_rad"][ex.arm], "start pose")
            print(f"start pose: {r.feedback}")
            if not r.ok and mode == "live":
                return
        ep = Episode(cfg, vlm, ex, per, rec, log, demos=demos, feedback=feedback, stats=stats)
        summary = ep.run(a.task)
        print(json.dumps(summary, indent=1, default=str))
        if stats.count:
            print(f"inference: {stats.count} call(s) logged to {stats.path.relative_to(stats.path.parents[1])}, plot {cfg['stats']['plot']}")
        path, msg = rec.export_recording(ex.arm, cfg["recorder"].get("export_dir"))
        print(f"EXPORTED {path.relative_to(ROOT) if path and path.is_relative_to(ROOT) else path}: {msg}" if path else f"no recording exported: {msg}")
    finally:
        if mode == "live":
            backend.release()


def cmd_packet(a, cfg):
    mode = "sim" if a.sim else "dry-run"
    backend, ex, per = build(cfg, mode, a.iface)
    from .prompts import proprio_text
    state = ex.sync()
    pk = per.capture(state.p)
    rec = Recorder(cfg, "packet", "packet")
    for label, jpg in pk.images:
        (rec.dir / f"{label.split()[0].lower()}.jpg").write_bytes(jpg)
    h = height_above_table_cm(state.p, ex.gate.table_z)
    pro = proprio_text(h, 1.0, "no hand")
    (rec.dir / "proprio.txt").write_text(pro["text"] + "\n")
    print(f"hand tip {state.p.round(3).tolist()} m, wrist roll {state.roll:.2f} rad, table z {ex.gate.table_z}")
    print(pro["text"])
    print(f"images: {[(l, len(j) // 1024) for l, j in pk.images]} KB; missing: {pk.missing}")
    print(f"written to {rec.dir}")


def cmd_replay(a, cfg):
    from .actions import OUTPUT_SCHEMA, parse_decision
    rec, prompt, images = load_step(a.run_dir, a.step)
    if prompt is None:
        sys.exit("that step has no prompt (it was a chunked or DONE step)")
    print(f"recorded: {rec.get('action')}  feedback: {rec.get('feedback')}")
    if a.edit:
        prompt = open(a.edit).read()
    vlm = vlm_base.make(cfg, a.vlm)
    resp = vlm.act(prompt, images, OUTPUT_SCHEMA)
    print(f"{resp.model} {resp.latency_s:.1f} s, {resp.input_tokens}+{resp.output_tokens} tokens, error={resp.error!r}")
    print(resp.text)
    try:
        d = parse_decision(resp.text, (cfg["robot"]["arm"],))
        print(f"parsed: {d.action(cfg['robot']['arm']).raw}  wrist={d.wrist_visible}  plan={[p.raw for p in d.plan]}")
    except Exception as e:
        print(f"parse error: {e}")


def cmd_measure(a, cfg):
    from .robot.lowstate import LowStateReader
    arm = cfg["robot"]["arm"]
    kin = ArmKinematics(cfg["robot"]["model"], arm)
    rd = LowStateReader(a.iface)
    if not rd.wait():
        sys.exit("no rt/lowstate")
    print("hand tip in the robot frame (floor under the pelvis, pelvis at 0.74 m). Ctrl-C to stop.")
    try:
        while True:
            j = rd.joints(); q = kin.q_from_dict(j)
            p, _ = kin.fk(q, {n: j[n] for n in ("waist_roll_joint", "waist_yaw_joint")})
            print(f"  {arm} hand tip x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:.3f} m   (set workspace.table_z_m to z when the tip rests on the table)")
            if not a.watch:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    rd.close()


def main():
    sys.stdout.reconfigure(line_buffering=True)          # progress lines show up live in logs and pipes
    ap = argparse.ArgumentParser(prog="harness", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config"); ap.add_argument("--set", action="append", metavar="KEY=VALUE")
    ap.add_argument("--arm", choices=["left", "right"]); ap.add_argument("--profile", choices=["precision", "coarse_fine"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sim", "dry-run", "live"):
        p = sub.add_parser(name)
        if name != "sim":
            p.add_argument("iface")
        p.add_argument("task")
        p.add_argument("--vlm", help="anthropic | openai | scripted (default: config)")
        if name == "sim":
            p.add_argument("--realtime", action="store_true"); p.add_argument("--start-pose", action="store_true")
        if name == "live":
            p.add_argument("--no-confirm", action="store_true", help="do not ask before each move (e-stop: x + Enter)")
        p.add_argument("--confirm", action="store_true", help="ask before every move (live does by default)")
        p.add_argument("--preview", metavar="FILE", help="write each proposal as a plan file before asking (the twin previews it)")
        p.add_argument("--spectacles-review", metavar="FILE", help="publish this proposal to a Spectacles review mailbox; needs --preview")
        p.add_argument("--demos", nargs="+", metavar="RECORDING", help="recordings/*.json shown to the model as demonstrations")
    p = sub.add_parser("packet"); p.add_argument("--sim", action="store_true"); p.add_argument("--iface")
    p = sub.add_parser("replay"); p.add_argument("run_dir"); p.add_argument("--step", type=int, required=True)
    p.add_argument("--vlm"); p.add_argument("--edit", help="use this file as the prompt instead of the recorded one")
    p = sub.add_parser("measure-table"); p.add_argument("iface"); p.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    over = parse_overrides(a.set)
    if a.arm: over["robot.arm"] = a.arm
    if a.profile: over["steps.profile"] = a.profile
    cfg = hcfg.load(a.config, over)
    if a.cmd == "sim":
        cfg["_realtime"] = a.realtime
        a.iface = None; a.no_confirm = not a.confirm
        run_episode(a, cfg, "sim")
    elif a.cmd in ("dry-run", "live"):
        if a.cmd == "dry-run":
            a.no_confirm = not a.confirm; a.start_pose = True
        else:
            a.start_pose = True
        run_episode(a, cfg, a.cmd)
    elif a.cmd == "packet":
        if not a.sim and not a.iface:
            sys.exit("packet needs --sim or --iface")
        cmd_packet(a, cfg)
    elif a.cmd == "replay":
        cmd_replay(a, cfg)
    elif a.cmd == "measure-table":
        cmd_measure(a, cfg)


if __name__ == "__main__":
    main()
