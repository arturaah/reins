"""Complete motion integration: real IK/validation, fake actuators and loopback AR."""
import asyncio
import copy
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from websockets.asyncio.client import connect
from contract.runtime import digest, validate_approval
from core.glasses_pairing import PairingStore
from core.glasses_bridge import GlassesBridge
from core.prompt_planner import PromptPlanner
from core.robot_pipeline import RobotPipeline, PreviewBackend
from core.robot_lease import RobotLease
from core.test_generated_motion import DRAFT
from core.trajectory import validate
from harness.actions import ActionError, parse_action
from harness.loop import Episode
from tools.dashboard import Simulation


class FakeRobot(PreviewBackend):
    name = "arm_sdk"
    dry_run = False
    authenticated = True

    def __init__(self, pose):
        super().__init__(pose)
        self.sent, self.engaged, self.frozen = [], False, False

    def snapshot(self):
        return {"joints":dict(self.q),"targets":dict(self.q),"engaged":self.engaged,"lowstate_age_s":0}

    def release(self): self.engaged = False
    def freeze(self): self.frozen = True

    def execute_motion(self,payload,approval):
        validate_approval(approval,digest(payload))
        self.sent.append(copy.deepcopy(payload))
        if payload["kind"] == "arm":
            self.engaged = True
            self.q.update(payload["plan"]["keyframes"][-1]["joint_targets_rad"])
        elif payload["kind"] == "hand": self.hands[payload["arm"]] = payload["closed"]
        return {"joints":self.joints(),"ok":True}


class Feed:
    def __init__(self):
        self.lock = threading.Lock(); self.online = True
        buf = io.BytesIO(); Image.new("RGB",(64,64)).save(buf,"JPEG"); self.jpg = buf.getvalue()
    def status(self): return {"online":self.online}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sim, self.planner = Simulation(), PromptPlanner()
        self.pipe = RobotPipeline(self.planner,self.sim,{},run_dir=self.tmp.name)

    def tearDown(self):
        self.pipe.close(); self.tmp.cleanup()

    def wait(self,*states):
        end = time.monotonic()+12
        while time.monotonic()<end:
            value = self.pipe.status()
            if value["state"] in states and not (value["state"] in ("idle","completed") and value["busy"]): return value
            time.sleep(.01)
        self.fail(str(self.pipe.status()))

    def draft(self):
        return self.pipe.compile_hand_path(DRAFT)

    def ready(self):
        d = self.draft()
        return self.pipe.propose_motion(d["id"])["proposal"]

    def approve(self,p):
        self.pipe.decide(p["id"],p["digest"],"approve")
        return self.wait("completed","failed","blocked")

    def live(self):
        robot = FakeRobot(self.pipe.planning_pose())
        self.pipe.backend_factory = lambda:robot
        self.pipe.command({"action":"connect","table_z_m":.6})
        self.wait("idle")
        return robot

    def test_draft_and_repeated_preview_never_open_review_or_actuate(self):
        before = self.pipe.backend.joints()
        draft = self.draft()
        self.assertIsNone(self.pipe.proposal)
        for _ in range(2): self.pipe.preview_plan(draft["id"])
        self.assertIsNone(self.pipe.glasses_message()["review"])
        self.assertEqual(self.pipe.backend.joints(),before)
        self.assertEqual(self.pipe.status()["state"],"draft")

    def test_connection_is_read_only_then_exact_complete_approved_payload_executes(self):
        robot = self.live()
        self.assertFalse(robot.engaged)
        p = self.ready()
        self.assertEqual(robot.sent,[])
        expected = copy.deepcopy(self.pipe.plan)
        self.assertEqual(self.approve(p)["last_result"]["outcome"],"executed")
        self.assertEqual(robot.sent,[expected])
        self.assertEqual(digest(robot.sent[0]),p["digest"])
        self.assertEqual(self.pipe.last_result["tracking_error"],0.)

    def test_planning_pose_reports_measured_joints_not_held_targets(self):
        state=self.pipe.backend.snapshot()
        joint='right_elbow_joint'
        state.update(engaged=True,targets={joint:state['joints'][joint]+.2})
        with patch.object(self.pipe.backend,'snapshot',return_value=state):
            pose=self.pipe.planning_pose()
        self.assertEqual(pose[joint],state['joints'][joint])
        self.assertEqual(self.pipe.robot_state['targets'][joint],state['targets'][joint])

    def test_private_hand_bridge_startup_and_shutdown_uses_session_token(self):
        self.pipe.cfg['hand']['revo2']['iface'] = 'test-hand-interface'
        absent = RuntimeError('no hand listener'); absent.__cause__ = ConnectionRefusedError()
        hand, process, backend = Mock(authenticated=True), Mock(), Mock()
        process.poll.return_value = None
        with patch('harness.robot.hand_client.Revo2Client', side_effect=[absent,hand]), patch('core.robot_pipeline.subprocess.Popen',return_value=process) as spawn:
            self.pipe._connect_hands(backend)
        self.assertIs(backend.hands,hand)
        args = spawn.call_args.args[0]
        self.assertIn('test-hand-interface',args)
        token = Path(args[args.index('--control-token-file')+1])
        self.assertEqual(token.stat().st_mode & 0o777,0o600)
        self.assertEqual(str(token),self.pipe.cfg['streamer']['control_token_file'])
        hand.execute_motion.assert_not_called()
        self.pipe.close()
        process.terminate.assert_called_once()

    def test_existing_foreign_hand_bridge_is_never_replaced(self):
        with patch('harness.robot.hand_client.Revo2Client',side_effect=RuntimeError('Authentication refused')), patch('core.robot_pipeline.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError,'Authentication'): self.pipe._connect_hands(Mock())
        spawn.assert_not_called()

    def test_streamer_walking_budget_survives_reconnection(self):
        self.pipe.cfg['locomotion']['enabled'] = True
        self.pipe._update_robot_state({**self.pipe.backend.snapshot(),'walked_m':self.pipe.cfg['locomotion']['max_total_m']})
        with self.assertRaisesRegex(ValueError,'budget'): self.pipe.prepare_walk(.1,0,0)

    def test_primary_ui_path_yields_draft_not_per_step_review(self):
        self.planner.submit("Blow a kiss",trajectory=DRAFT)
        self.wait("draft","blocked")
        self.assertIsNotNone(self.pipe.draft)
        self.assertIsNone(self.pipe.proposal)
        # Compilation publishes the draft before its preview callback finishes.
        end = time.monotonic()+3
        while not self.sim.playing and time.monotonic()<end: time.sleep(.01)
        self.assertTrue(self.sim.playing)

    def test_idempotent_submission_and_terminal_result_do_not_repeat_motion(self):
        robot = self.live(); d = self.draft()
        first = self.pipe.propose_motion(d["id"],"request1")
        self.assertEqual(self.pipe.propose_motion(d["id"],"request1")["proposal_id"],first["proposal_id"])
        p = first["proposal"]; self.approve(p)
        for key in ("request1","request2"):
            self.assertEqual(self.pipe.propose_motion(d["id"],key)["outcome"],"executed")
        self.assertEqual(len(robot.sent),1)
        with self.assertRaises(ValueError): self.pipe.decide(p["id"],p["digest"],"approve")

    def test_wrong_id_digest_and_changed_pose_refused(self):
        p = self.ready()
        for ident,h in (("old",p["digest"]),(p["id"],"wrong")):
            with self.assertRaises(ValueError): self.pipe.decide(ident,h,"approve")
        self.pipe.backend.q["right_elbow_joint"] += .1
        self.assertEqual(self.approve(p)["last_result"]["outcome"],"failed")

    def test_mutated_approved_payload_refused(self):
        p = self.ready()
        self.pipe.plan["plan"]["name"] = "mutation"
        result = self.approve(p)
        self.assertEqual(result["last_result"]["outcome"],"failed")
        self.assertIn("changed",result["message"])

    def test_expiry_and_concurrent_decisions(self):
        p = self.ready(); self.pipe.proposal["expires_at"] = time.time()-1
        with self.assertRaises(ValueError): self.pipe.decide(p["id"],p["digest"],"approve")
        self.wait("expired")
        self.assertEqual(self.pipe.motion_result(p["id"])["outcome"],"expired")

    def test_only_one_of_two_operator_decisions_wins(self):
        p = self.ready(); results=[]
        def choose():
            try: self.pipe.decide(p["id"],p["digest"],"decline"); results.append(True)
            except ValueError: results.append(False)
        threads=[threading.Thread(target=choose) for _ in range(2)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(sorted(results),[False,True])

    def test_cancel_invalidates_draft_and_proposal(self):
        p = self.ready(); d = p["plan_id"]; self.pipe.stop()
        with self.assertRaises(ValueError): self.pipe.decide(p["id"],p["digest"],"approve")
        with self.assertRaises(ValueError): self.pipe.preview_plan(d)
        self.assertEqual(self.pipe.motion_result(p["id"])["outcome"],"cancelled")

    def test_stop_and_restart_during_compilation_does_not_adopt_old_generation(self):
        from core.generated_motion import compile_trajectory
        entered,resume,results=threading.Event(),threading.Event(),[]
        def paused(*args):
            entered.set()
            if not resume.wait(3): raise RuntimeError('test compilation not resumed')
            return compile_trajectory(*args)
        def compile_old():
            try: results.append(self.draft())
            except Exception as exc: results.append(exc)
        with patch('core.robot_pipeline.compile_trajectory',side_effect=paused):
            worker=threading.Thread(target=compile_old);worker.start()
            try:
                self.assertTrue(entered.wait(3))
                self.pipe.stop();self.pipe.before_submit()
            finally:
                resume.set();worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(results[0],ValueError)
        self.assertIn('superseded',str(results[0]))
        self.assertIsNone(self.pipe.draft)
        self.assertEqual(self.pipe.drafts,{})

    def test_rejection_keeps_feedback_and_allows_new_motion(self):
        p = self.ready(); self.pipe.decide(p["id"],p["digest"],"decline","Keep it lower")
        self.assertEqual(self.pipe.last_result["decision"]["note"],"Keep it lower")
        self.assertIsNone(self.pipe.proposal)
        self.pipe.command({"action":"jog","direction":"up","arm":"right"})
        self.assertEqual(self.wait("review","blocked")["state"],"review")

    def test_manual_wrist_roll_and_nudge_each_make_one_complete_proposal(self):
        before = self.pipe.backend.joints()
        self.pipe.command({"action":"roll","arm":"right","sign":1})
        p=self.wait("review","blocked")["proposal"]
        self.assertEqual(self.pipe.backend.joints(),before)
        self.approve(p)
        self.assertAlmostEqual(self.pipe.backend.q["right_wrist_roll_joint"]-before["right_wrist_roll_joint"],.0872664626)
        self.pipe.command({"action":"jog","direction":"up","arm":"right"})
        self.assertEqual(self.wait("review","blocked")["state"],"review")

    def test_walk_and_revo2_drafts_use_same_review_and_preserve_base_preview(self):
        self.pipe.cfg["locomotion"]["enabled"] = True
        self.pipe.cfg["hand"]["type"] = "virtual"
        d=self.pipe.prepare_walk(.1,0,.1); self.pipe.preview_plan(d["id"])
        self.assertEqual(self.pipe.glasses_message()["frame"],"map")
        self.assertIsNotNone(self.sim.base_path)
        p=self.pipe.propose_motion(d["id"])["proposal"]; self.approve(p)
        d=self.pipe.prepare_hand("left",True); p=self.pipe.propose_motion(d["id"])["proposal"]
        self.assertFalse(self.pipe.backend.hand_state("left")); self.approve(p)
        self.assertTrue(self.pipe.backend.hand_state("left"))

    def test_simulation_only_cannot_connect_or_run_firmware(self):
        self.pipe.simulation_only=True
        with self.assertRaises(ValueError): self.pipe.connect(.6)
        with self.assertRaises(ValueError): self.pipe.firmware({"action":"refresh"},None)

    def test_foreign_bridge_does_not_become_actuator_authority(self):
        robot=FakeRobot(self.pipe.planning_pose());robot.authenticated=False
        self.pipe.backend_factory=lambda:robot
        with self.assertRaisesRegex(ValueError,"another session"):self.pipe.connect(.6)
        self.assertFalse(self.pipe.connected)

    def test_stale_original_observation_or_camera_blocks_submission(self):
        feed=Feed();self.pipe.cameras={"head":feed}; self.live()
        self.pipe.register_observation({"id":"obs","observed_at":time.time(),"cameras":{"head":{}},"pose":self.pipe.planning_pose()})
        d=self.pipe.compile_hand_path(DRAFT,"obs")
        feed.online=False
        with self.assertRaisesRegex(ValueError,"stale"):self.pipe.propose_motion(d["id"])
        feed.online=True;self.pipe.observations["obs"]["observed_at"]-=121
        with self.assertRaisesRegex(ValueError,"expired"):self.pipe.propose_motion(d["id"])

    def test_stop_during_final_validation_prevents_streaming(self):
        robot=self.live();p=self.ready()
        def stopped(*a,**kw):
            result=validate(*a,**kw);self.pipe.stop();return result
        with patch("core.robot_pipeline.trajectory.validate",side_effect=stopped):
            self.pipe.decide(p["id"],p["digest"],"approve");self.wait("stopped")
        self.assertEqual(robot.sent,[])

    def test_operator_disconnect_releases_fake_robot(self):
        robot=self.live();self.pipe.last_operator-=11;self.wait("stopped")
        self.assertTrue(robot.frozen);self.assertFalse(self.pipe.connected)

    def test_glasses_approve_exact_revision_after_pairing_and_tracking(self):
        p=self.ready();store=PairingStore(Path(self.tmp.name)/"pairing.json");device=store.create_device("Test Lens")
        bridge=GlassesBridge(self.pipe,"127.0.0.1",0,pairing=store);self.assertTrue(bridge.ready.wait(3))
        async def run():
            async with connect("ws://127.0.0.1:"+str(self.pipe.glasses["port"])) as ws:
                await ws.send(json.dumps({"type":"authenticate","device_id":device["device_id"],"token":device["token"]}))
                session=json.loads(await ws.recv())["session"]
                await ws.recv()
                await ws.send(json.dumps({"type":"review_decision","version":1,"session":session,"id":p["id"],"digest":p["digest"],"revision":p["revision"],"decision":"approve","tracking":{"registered":True,"age_s":.1}}))
                while True:
                    ack=json.loads(await ws.recv())
                    if ack["type"]=="review_ack":self.assertTrue(ack["accepted"]);break
        try:asyncio.run(run())
        finally:bridge.close()
        self.assertEqual(self.wait("completed","failed")["last_result"]["outcome"],"executed")


class BoundaryTests(unittest.TestCase):
    def test_nonfinite_actions_and_oscillation(self):
        for value in ("nan","inf","-inf"):
            with self.assertRaises(ActionError):parse_action("MOVE forward "+value)
        self.assertTrue(Episode.same_token("MV_FWD",parse_action("MV_BACK").opposite_of))

    def test_exclusive_robot_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            one,two=RobotLease("one",Path(tmp)/"robot.lock"),RobotLease("two",Path(tmp)/"robot.lock")
            with one:
                with self.assertRaises(ValueError):two.acquire()
            with two:pass

    def test_contract_requires_approval(self):
        from contract.reins_contract import ContractError,check_session,read_jsonl
        with self.assertRaisesRegex(ContractError,"approval"):
            check_session([m for m in read_jsonl("contract/examples/session_reach.jsonl") if m["id"]!="g-5"])
