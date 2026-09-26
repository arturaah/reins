"""Read-only tests: no model API calls, physical camera access or DDS."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from core.perception import Observation, PerceptionError, stereo_depth
from core.prompt_planner import PromptPlanner, ground_openai
from core.ik import ArmIK, plan_from_waypoints
from core.motion_validation import MotionValidator, MotionRejected, slow_acceleration


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'observation.npz'
        now=time.time()
        self.fields=dict(rgb=np.zeros((64,96,3),np.uint8),depth_m=np.ones((64,96)),
                         K=np.array([[100.,0,48],[0,100,32],[0,0,1]]),T_robot_camera=np.eye(4),
                         captured_at=now,pose_at=now,pose_json=json.dumps({'waist_yaw_joint':0}),calibration_id='test')
    def tearDown(self): self.tmp.cleanup()
    def load(self):
        np.savez(self.path,**self.fields); return Observation.load(self.path)
    def test_metric_surface_and_transform(self):
        self.fields['T_robot_camera'][:3,3]=[.2,.3,.4]
        obs=self.load(); point,normal,quality=obs.locate([.25,.25,.75,.75])
        np.testing.assert_allclose(point,[.195,.295,1.4],atol=.015)
        self.assertGreater(quality['points'],20); self.assertLess(normal[2],-.99)
    def test_missing_depth_is_rejected(self):
        self.fields['depth_m'][:]=np.nan
        with self.assertRaisesRegex(PerceptionError,'Insufficient'):self.load().locate([.2,.2,.8,.8])
    def test_stale_and_unsynchronized_observations(self):
        self.fields['captured_at']-=10
        with self.assertRaisesRegex(PerceptionError,'stale'):self.load()
        self.fields['captured_at']=time.time(); self.fields['pose_at']-=1
        with self.assertRaisesRegex(PerceptionError,'50 ms'):self.load()
    def test_invalid_transform_and_box(self):
        self.fields['T_robot_camera'][0,0]=2
        with self.assertRaisesRegex(PerceptionError,'rigid'):self.load()
        self.fields['T_robot_camera']=np.eye(4)
        with self.assertRaises(PerceptionError):self.load().locate([-.1,0,1,1])
    def test_unknown_or_occluded_volume_blocks(self):
        obs=self.load(); obs.require_free([[0,0,.5]],.02)
        with self.assertRaisesRegex(PerceptionError,'obstacle'):obs.require_free([[0,0,.99]],.02)
        obs.depth[:]=np.nan
        with self.assertRaisesRegex(PerceptionError,'Missing'):obs.require_free([[0,0,.5]],.02)
        with self.assertRaises(PerceptionError):obs.require_free([[20,0,.5]],.02)
    def test_synthetic_calibrated_stereo_depth(self):
        rng=np.random.default_rng(8)
        left=rng.integers(0,255,(96,320,3),dtype=np.uint8)
        right=np.zeros_like(left); right[:,:-10]=left[:,10:]
        K=[[200.,0,160],[0,200.,48],[0,0,1]]
        calibration={'image_size':[320,96],'K_left':K,'K_right':K,'dist_left':[0]*5,'dist_right':[0]*5,
                     'R_right_left':np.eye(3).tolist(),'t_right_left_m':[-.05,0,0]}
        _,depth,_,_=stereo_depth(left,right,calibration,100,100)
        center=depth[15:-15,130:190]
        self.assertGreater(np.isfinite(center).mean(),.7)
        self.assertAlmostEqual(float(np.nanmedian(center)),1.,places=1)
        with self.assertRaisesRegex(PerceptionError,'15 ms'):stereo_depth(left,right,calibration,100,101)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ik=ArmIK(backend='mujoco')
    def wait(self,p):
        limit=time.monotonic()+15
        while p.status()['state']=='planning' and time.monotonic()<limit:time.sleep(.02)
        self.assertNotEqual(p.status()['state'],'planning'); return p.status()
    def test_demo_generates_gated_preview_and_exact_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=PromptPlanner(output_dir=tmp); p.submit('touch the bottle','demo'); result=self.wait(p)
            self.assertEqual(result['state'],'proposed',result['message'])
            self.assertFalse(result['execution_allowed']); self.assertFalse(result['contact_enabled'])
            with self.assertRaises(ValueError):p.preview('wrong-id')
            path=p.preview(result['id']); plan=json.loads(path.read_text())
            self.assertTrue(plan['preview_only']); self.assertGreater(plan['duration_s'],0)
            self.assertEqual(plan['prompt_proposal']['id'],result['id'])
            p.cancel()
            with self.assertRaises(ValueError):p.preview(result['id'])
    def test_generated_plan_cannot_start_robot_runner(self):
        from tools.dashboard import Runner
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'preview.json'
            path.write_text(json.dumps({'preview_only':True}))
            runner=Runner('unused-test-interface')
            with patch('subprocess.Popen') as popen:
                with self.assertRaisesRegex(ValueError,'preview-only'):
                    runner.start('dry','generated',path)
                runner.cleared={'plan':'generated','path':path,'speed':1,'kp_scale':1,'at':time.time()}
                with self.assertRaisesRegex(ValueError,'preview-only'):
                    runner.start('execute',None,None,confirm=True)
                popen.assert_not_called()

    def test_real_camera_never_falls_back_to_fixture(self):
        p=PromptPlanner();p.submit('touch the bottle','camera'); result=self.wait(p)
        self.assertEqual(result['state'],'blocked');self.assertIn('No calibrated depth',result['message'])
        self.assertIsNone(p.plan)
    def test_missing_model_credentials_explicit(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'OPENAI_API_KEY'):ground_openai('touch the bottle',np.zeros((10,10,3),np.uint8))
    def test_unsupported_demo_prompt_blocks(self):
        p=PromptPlanner();p.submit('walk across the room','demo');self.assertEqual(self.wait(p)['state'],'blocked')
    def test_unreachable_target(self):
        with self.assertRaisesRegex(ValueError,'unreachable'):
            plan_from_waypoints(self.ik,'right',[[2,-.2,1]])
    def test_obstacle_and_velocity_rejection(self):
        plan,_=plan_from_waypoints(self.ik,'right',[[.25,-.2,.86]],max_vel=.3);slow_acceleration(plan)
        validator=MotionValidator(self.ik.model)
        with self.assertRaisesRegex(MotionRejected,'Collision'):
            validator.check(plan,'right',[{'name':'blocking wall','min':[-2,-2,0],'max':[2,2,2]}])
        for frame in plan['keyframes']:frame['time_s']/=10
        with self.assertRaisesRegex(MotionRejected,'velocity'):validator.check(plan,'right',[])
    def test_provider_ambiguity_and_schema_failure(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def read(self,*a):return json.dumps({'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'action':'touch','status':'ambiguous','explanation':'Two bottles'})}]}]}).encode()
        with patch.dict('os.environ',{'OPENAI_API_KEY':'test','REINS_VISION_MODEL':'test'}),patch('urllib.request.urlopen',return_value=Response()):
            with self.assertRaisesRegex(ValueError,'ambiguous'):ground_openai('touch the bottle',np.zeros((10,10,3),np.uint8))


if __name__=='__main__':unittest.main()

class AutoContextTests(unittest.TestCase):
    def wait(self,p):
        deadline=time.monotonic()+20
        while p.status()['state']=='planning' and time.monotonic()<deadline:time.sleep(.02)
        self.assertNotEqual(p.status()['state'],'planning')
        return p.status()

    def test_wave_and_raise_never_request_images_or_model(self):
        for prompt in ('wave','wave your left hand','raise your right arm'):
            with patch('core.prompt_planner.ground_openai') as vision,patch('core.prompt_planner.Observation.load') as load:
                p=PromptPlanner();p.submit(prompt)
                result=self.wait(p)
                self.assertEqual(result['state'],'proposed',result['message'])
                self.assertEqual(result['context']['vision'],'not_required')
                self.assertGreater(result['validation']['samples'],0)
                self.assertIn('unknown',result['validation']['coverage'])
                vision.assert_not_called();load.assert_not_called()
                self.assertTrue(p.plan['preview_only'])

    def test_object_action_requests_context_and_does_not_become_gesture(self):
        from core.action_context import route_intent
        self.assertEqual(route_intent('point at the bottle')['skill'],'point')
        for text in ('wave at the person','wave and touch the bottle','wave with the left hand and then move'):
            with self.assertRaises(ValueError):route_intent(text)
        p=PromptPlanner();p.submit('point at the bottle')
        result=self.wait(p);self.assertEqual(result['state'],'blocked')
        self.assertIn('calibrated depth',result['message'])

    def test_cached_target_reused_only_for_identical_fresh_observation(self):
        now=time.time()
        obs=Observation(np.zeros((24,24,3),np.uint8),np.ones((24,24)),np.eye(3),np.eye(4),now,{'waist_yaw_joint':0},'test')
        target={'action':'touch','arm':'right','label':'bottle','bbox':[.2,.2,.8,.8]}
        p=PromptPlanner();p.job['context']={};p.job['events']=[]
        with patch('core.prompt_planner.ground_openai',return_value=target) as vision:
            p._ground({'skill':'touch','arm':'auto','selector':'bottle'},obs,'auto')
            p._ground({'skill':'point','arm':'auto','selector':'it'},obs,'auto')
            self.assertEqual(vision.call_count,1)
            self.assertEqual(p.job['context']['vision'],'reused')
            obs.rgb[0,0,0]=1
            p._ground({'skill':'touch','arm':'auto','selector':'bottle'},obs,'auto')
            self.assertEqual(vision.call_count,2)
            obs.captured_at-=10
            p._ground({'skill':'touch','arm':'auto','selector':'bottle'},obs,'auto')
            self.assertEqual(vision.call_count,3)

    def test_pronoun_without_history_requires_target(self):
        p=PromptPlanner();p.job['context']={}
        obs=Observation(np.zeros((24,24,3),np.uint8),np.ones((24,24)),np.eye(3),np.eye(4),time.time(),{},'test')
        with patch('core.prompt_planner.ground_openai') as vision:
            with self.assertRaisesRegex(ValueError,'No previous object'):p._ground({'skill':'point','arm':'auto','selector':'it'},obs,'auto')
            vision.assert_not_called()

    def test_pointing_does_not_require_reaching_distant_object(self):
        from core.action_context import pointing_plan
        ik=ArmIK(backend='mujoco');surface=np.array([2.,-.4,1.])
        plan,goal,error=next(pointing_plan(ik,'right',surface,{}))
        self.assertLess(error,10.01)
        self.assertGreater(np.linalg.norm(surface-goal),1)
        self.assertTrue(plan['keyframes'])
