"""Offline review probes. No model calls, camera requests, DDS initialization or robot commands."""
import json, math, socket, tempfile, threading, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from harness.config import load
from harness.actions import parse_action
from harness.executor import ArmExecutor, interpolate
from harness.interpreter import Interpreter
from harness.kinematics import ArmKinematics, ARM_JOINTS
from harness.loop import Episode
from harness.recorder import Recorder
from harness.safety import SafetyGate
from harness.sim.mock_robot import MockBackend
from contract.reins_contract import check_session, read_jsonl
from sim.preview import prepare_plan

out={}
cfg=load()
kin=ArmKinematics(cfg["robot"]["model"],"right")
gate=SafetyGate(cfg,kin,None,live=False)
backend=MockBackend(cfg,render=False)
ex=ArmExecutor(cfg,kin,gate,backend,"right")
out["oscillation"]={"forward_then_back_detected":Episode.same_token("MV_FWD",parse_action("MV_BACK").opposite_of)}
out["nonfinite_action"]={"nan_accepted":math.isnan(parse_action("MOVE forward nan").amount),
                         "infinity_becomes_metres":parse_action("MOVE forward inf").amount}

# A queued e-stop during the approval callback is not re-checked by the executor.
it=Interpreter(cfg["frames"]["view_forward"],cfg["frames"]["view_left"])
s=ex.sync()
def confirm(*args):
    gate.estop.set()
    return True
ex.confirm=confirm
r=ex.execute(it.propose(s,parse_action("MV_UP"),.02,.1),s)
out["estop_during_review"]={"estop_set":gate.estop.is_set(),"result_ok":r.ok,"frames_sent":backend.frames_sent}

# Export two accepted moves; inspect the exact artifact with the real simulation loader.
with tempfile.TemporaryDirectory() as tmp:
    rec=Recorder(cfg,"sim","review probe",root=tmp)
    before=backend.joints()
    for i in range(2):
        target=np.array([before[n] for n in ARM_JOINTS["right"]]); target[0]-=.02
        rec.step(i,{"ok":True,"q_target":target.tolist(),"joints_before":before})
        before={**before,**dict(zip(ARM_JOINTS["right"],target.tolist()))}
    with patch("tools.framelog.save_sheet",return_value="omitted for review"):
        path,_=rec.export_recording("right",Path(tmp)/"exports")
    plan=json.loads(path.read_text())
    ts=[f["time_s"] for f in plan["keyframes"]]
    dups=[a for a,b in zip(ts,ts[1:]) if b<=a]
    try:
        prepare_plan(kin.model,plan); result="accepted"
    except ValueError as e:
        result=str(e)
    out["export"]={"nonincreasing_times":dups,"simulation_loader":result}

# Protocol validator accepts an execution even when the current approval was removed.
session=read_jsonl("contract/examples/session_reach.jsonl")
session=[m for m in session if m["id"]!="g-5"]
check_session(session)
out["contract"]={"session_without_final_approval_accepted":True}

# Kinematics only poses waist + selected arm; the other measured arm is ignored.
kin.contacts(np.zeros(5),{"left_shoulder_pitch_joint":-1.0})
jid=kin.model.joint("left_shoulder_pitch_joint")
out["opposite_arm_pose"]={"requested":-1.0,"collision_model_value":float(kin.data.qpos[jid.qposadr[0]])}

# Search a deterministic move with baseline contacts at endpoints and extra contacts in between.
rng=np.random.default_rng(273)
pool=[]
for i in range(1500):
    q=rng.uniform(kin.limits[:,0]+kin.margin,kin.limits[:,1]-kin.margin)
    p,_=kin.fk(q)
    if np.all(p>=gate.box_min) and np.all(p<=gate.box_max) and p[2]>=gate.floor_z :
        pool.append((q,kin.contacts(q)))
baseline=min(n for q,n in pool)
candidates=[q for q,n in pool if n==baseline]
found=None
for trial in range(3000):
    a,b=(candidates[int(rng.integers(len(candidates)))] for _ in range(2))
    for frac in (.25,.5,.75):
        mid=a+(b-a)*frac
        n=kin.contacts(mid)
        if n>baseline:
            found=(a,b,frac,n); break
    if found: break
if found:
    a,b,frac,n=found
    cfg2=load(); be=MockBackend(cfg2,render=False)
    be.q.update(dict(zip(ARM_JOINTS["right"],a)))
    ga=SafetyGate(cfg2,kin,None,live=False); ga.set_baseline(a)
    ex2=ArmExecutor(cfg2,kin,ga,be,"right")
    r=ex2.go_to_joints(b,"reviewed joint move")
    duration=max(cfg2["limits"]["min_move_s"],float(np.abs(b-a).max())/.8*math.pi/2)
    frames=interpolate(a,b,duration,50)
    out["swept_collision"]={"q_start":a.tolist(),"q_end":b.tolist(),"start_contacts":kin.contacts(a),
      "end_contacts":kin.contacts(b),"intermediate_fraction":frac,"intermediate_contacts":n,
      "trajectory_check":ga.check_trajectory(frames,.02),"executor_ok":r.ok,"frames_sent":be.frames_sent}
else:
    out["swept_collision"]={"found":False,"candidates":len(candidates)}

# Socket disconnect while serving a long command: all DDS dependencies are faked.
from harness.robot import arm_stream as am
from harness.tests.test_streamer_watchdog import FakeReader, FakePub
with patch.object(am,"LowStateReader",FakeReader),patch.object(am,"ChannelPublisher",FakePub),\
     patch.object(am,"CRC",lambda:SimpleNamespace(Crc=lambda cmd:0)),patch.object(am,"query_fsm",lambda:(811,"fake")):
    cfg3=load(); cfg3["streamer"]["watchdog_s"]=.2; cfg3["robot"]["weight_ramp_s"]=.02
    st=am.Streamer(cfg3,"unused",log=lambda *a:None)
    st.targets=st.measured(); st.weight=1.; st.engaged=True
    sent=[]
    def fake_send():
        with st.lock:
            targets=dict(st.targets)
        for slot,value in targets.items():
            st.reader.msg.motor_state[slot].q=value
        sent.append(time.monotonic())
    st.send=fake_send
    server,client=socket.socketpair()
    def serve_one():
        try: st.handle(server)
        except (BrokenPipeError,ConnectionResetError): pass
        finally:
            server.close()
            if st.engaged: st.release("client disconnected",.02)
    hold=threading.Thread(target=st.hold_loop,daemon=True)
    worker=threading.Thread(target=serve_one,daemon=True)
    hold.start(); worker.start()
    try:
        client.sendall((json.dumps({"cmd":"frames","arm":"right","frames":[[i*.001]*5 for i in range(100)],"dt":.02})+"\n").encode())
        deadline=time.monotonic()+1
        while not st.streaming and time.monotonic()<deadline: time.sleep(.005)
        sent_before=len(sent); client.close()
        time.sleep(.65)
        out["disconnect"]={"watchdog_s":st.watchdog_s,"elapsed_after_disconnect_s":.65,
                           "still_engaged":st.engaged,"still_streaming":st.streaming,
                           "additional_publishes":len(sent)-sent_before}
        worker.join(3)
    finally:
        st.stop.set(); st.engaged=False; hold.join(1); server.close(); client.close()
print(json.dumps(out,indent=2))
Path("/tmp/reins-review-probe-results.json").write_text(json.dumps(out,indent=2)+"\n")
