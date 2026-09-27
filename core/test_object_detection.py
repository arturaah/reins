"""Local inference contracts and planning gates; no downloads, camera or robot access."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from core.object_detection import ObjectDetector, canonical_label, select_target
from core.detection_stream import DetectionStream
from core.perception import Observation
from core.prompt_planner import PromptPlanner

BOX = {'label': 'bottle', 'confidence': .85, 'bbox': [.2, .2, .8, .8]}


class DetectorTests(unittest.TestCase):
    def test_model_setup_is_explicit(self):
        detector = ObjectDetector(Path(tempfile.gettempdir())/'reins-no-such-model.onnx')
        with patch('urllib.request.urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'model missing'):
                detector.detect(np.zeros((100, 100, 3), np.uint8))
            network.assert_not_called()

    def test_checksum_failure_leaves_existing_file_untouched(self):
        from core.object_detection import download_model
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.onnx'; path.write_bytes(b'previous')
            with patch('urllib.request.urlopen') as response:
                response.return_value.__enter__.return_value.read.return_value = b'not the verified model'
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    download_model(path)
            self.assertEqual(path.read_bytes(), b'previous')

    def test_letterbox_is_bgr_and_preserves_non_square_geometry(self):
        rgb = np.zeros((200, 400, 3), np.uint8); rgb[:, :, 0] = 255
        blob, mapping = ObjectDetector._prepare(rgb)
        self.assertEqual(mapping, (0, 104, 416, 208))
        np.testing.assert_allclose(blob[0, :, 208, 208],
                                   (np.array([0, 0, 255])-[103.53,116.28,123.675])/[57.375,57.12,58.395], rtol=1e-6)

    def test_decode_maps_boxes_and_suppresses_only_same_class_duplicates(self):
        outputs = []
        for stride in (8,16,32):
            n = (416//stride)**2
            cls = np.zeros((1, n, 80), np.float32)
            dist = np.full((1, n, 4, 8), -20., np.float32)
            dist[:, :, :, 4] = 20.
            outputs.extend((cls, dist.reshape(1, n, 32)))
        # Neighboring anchors overlap heavily. Keep one bottle plus overlapping person.
        anchor = 26*52+26
        outputs[0][0, anchor, 39] = .9
        outputs[0][0, anchor+1, 39] = .8
        outputs[0][0, anchor-1, 0] = .85
        results = ObjectDetector()._decode(outputs, (0,104,416,208))
        self.assertEqual([r['label'] for r in results], ['bottle','person'])
        np.testing.assert_allclose(results[0]['bbox'], [179.5/416,75.5/208,243.5/416,139.5/208])

    def test_qualified_or_ambiguous_targets_are_not_guessed(self):
        self.assertEqual(canonical_label('a mug'), 'cup')
        self.assertIsNone(canonical_label('red bottle'))
        self.assertIsNone(select_target([BOX], 'leftmost bottle'))
        self.assertEqual(select_target([BOX], 'bottle'), BOX)
        with self.assertRaisesRegex(ValueError, 'Multiple bottle'):
            select_target([BOX, BOX], 'bottle')


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.detector = Mock(available=True, confidence=.4)
        self.detector.detect.return_value = [BOX]
        self.rgb = np.zeros((64,96,3), np.uint8)
        self.received = time.monotonic()
        self.source = Mock(side_effect=lambda: (self.rgb, self.received, 'frame1', None))
        self.stream = DetectionStream(self.detector, {'head':self.source,'left':self.source}, autostart=False)

    def test_opt_in_and_exact_frame_image(self):
        self.stream.step(); self.detector.detect.assert_not_called()
        self.stream.configure(True,'head'); self.stream.step()
        result = self.stream.status()['result']
        self.assertEqual(result['objects'][0]['surface_m'],None)
        self.assertTrue(self.stream.image(result['id']).startswith(b'\xff\xd8'))
        self.assertFalse(self.stream.image('old-id'))
        self.stream.step(); self.assertEqual(self.detector.detect.call_count,1)
        self.stream.configure(False,'head')
        self.assertFalse(self.stream.status()['ready']); self.assertFalse(self.stream.image(result['id']))

    def test_stale_frames_are_not_inferred_or_served(self):
        self.received -= 10
        self.stream.configure(True,'head'); self.stream.step()
        self.detector.detect.assert_not_called();self.assertFalse(self.stream.status()['ready'])
        self.received = time.monotonic(); self.stream.step()
        result=self.stream.status()['result']
        with patch('core.detection_stream.time.monotonic',return_value=self.received+4):
            self.assertIsNone(self.stream.status()['result'])
            self.assertFalse(self.stream.image(result['id']))

    def test_source_switch_during_inference_discards_result(self):
        def detect(_):
            self.stream.configure(True,'left')
            return [BOX]
        self.detector.detect.side_effect=detect
        self.stream.configure(True,'head');self.stream.step()
        self.assertEqual(self.stream.status()['source'],'left')
        self.assertIsNone(self.stream.status()['result'])

    def test_expiry_during_inference_discards_result(self):
        self.stream.configure(True,'head')
        with patch('core.detection_stream.time.monotonic',side_effect=[self.received+.1,self.received+.1,self.received+4]):
            self.stream.step()
        self.assertIsNone(self.stream.status()['result'])
        self.assertIn('expired', self.stream.status()['message'])

    def test_calibrated_depth_enriches_objects_and_missing_depth_stays_unknown(self):
        obs=Observation(self.rgb,np.ones((64,96)),np.array([[100,0,48],[0,100,32],[0,0,1.]]),np.eye(4),time.time(),{},'test')
        self.source.side_effect=lambda:(self.rgb,self.received,str(obs.captured_at),obs)
        self.stream.configure(True,'head');self.stream.step()
        obj=self.stream.status()['result']['objects'][0]
        self.assertAlmostEqual(obj['surface_m'][2],1.)
        obs.depth[:]=np.nan;obs.captured_at+=.001
        self.stream.step()
        obj=self.stream.status()['result']['objects'][0]
        self.assertIsNone(obj['surface_m']);self.assertIn('Insufficient',obj['depth_detail'])

    def test_bad_config_does_not_change_source(self):
        for enabled,source in [('yes','head'),(True,'twin'),(True,[])]:
            with self.assertRaises(ValueError):self.stream.configure(enabled,source)
        self.assertFalse(self.stream.status()['enabled'])


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.detector=Mock(available=True,confidence=.4)
        self.detector.detect.return_value=[BOX]
        self.planner=PromptPlanner(detector=self.detector)
        self.planner.job['context']={}
        self.obs=Observation(np.zeros((64,96,3),np.uint8),np.ones((64,96)),np.eye(3),np.eye(4),time.time(),{},'test')
        self.intent={'skill':'touch','arm':'right','selector':'bottle'}

    def test_unique_local_target_bypasses_api_and_reuses_same_observation(self):
        with patch('core.prompt_planner.ground_openai') as model:
            target=self.planner._ground(self.intent,self.obs,'auto')
            self.assertEqual(target['bbox'],BOX['bbox'])
            self.assertEqual(target['arm'],'right')
            self.assertEqual(self.planner.job['context']['vision'],'local_detector')
            self.planner._ground(self.intent,self.obs,'auto')
            self.assertEqual(self.detector.detect.call_count,1)
            model.assert_not_called()
        np.testing.assert_array_equal(self.detector.detect.call_args.args[0],self.obs.rgb)

    def test_ambiguous_local_target_blocks_instead_of_silently_picking(self):
        self.detector.detect.return_value=[BOX,BOX]
        with patch('core.prompt_planner.ground_openai') as model:
            with self.assertRaisesRegex(ValueError,'Multiple bottle'):
                self.planner._ground(self.intent,self.obs,'auto')
            model.assert_not_called()

    def test_rich_description_or_missed_object_uses_configured_model(self):
        for selector,objects in [('red bottle',[BOX]),('bottle',[])]:
            self.detector.detect.return_value=objects
            with patch('core.prompt_planner.ground_openai',return_value={**BOX,'arm':'auto'}) as model:
                self.planner._ground({**self.intent,'selector':selector},self.obs,'camera')
                model.assert_called_once()

    def test_gesture_never_runs_detector(self):
        with patch('core.prompt_planner.ground_openai') as model:
            self.planner.submit('wave','demo')
            deadline=time.monotonic()+20
            while self.planner.status()['state']=='planning' and time.monotonic()<deadline:time.sleep(.02)
            self.assertEqual(self.planner.status()['state'],'proposed',self.planner.status()['message'])
            self.detector.detect.assert_not_called();model.assert_not_called()
            self.assertTrue(self.planner.plan['preview_only'])

    def test_slow_grounding_cannot_localize_expired_observation(self):
        def detect(_):
            self.obs.captured_at-=10
            return [BOX]
        self.detector.detect.side_effect=detect
        self.planner.observation_path=Path('unused')
        with patch('core.prompt_planner.Observation.load',return_value=self.obs),patch.object(self.obs,'locate') as locate:
            self.planner.submit('touch the bottle')
            deadline=time.monotonic()+5
            while self.planner.status()['state']=='planning' and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(self.planner.status()['state'],'blocked')
            self.assertIn('expired',self.planner.status()['message']);locate.assert_not_called()


class OpenVocabularyTests(unittest.TestCase):
    """OmDet-Turbo wrapper and detector selection; no weights, downloads or robot."""

    def test_auto_falls_back_to_nanodet_without_omdet_weights(self):
        from core.object_detection import make_detector
        missing = Path(tempfile.gettempdir())/'reins-no-omdet'
        self.assertIsInstance(make_detector('auto', omdet_dir=missing), ObjectDetector)
        with self.assertRaises(ValueError):
            make_detector('yolo')

    def test_missing_omdet_model_never_downloads(self):
        from core.object_detection import make_detector
        detector = make_detector('omdet', omdet_dir=Path(tempfile.gettempdir())/'reins-no-omdet')
        self.assertFalse(detector.available)
        import sys
        hub = Mock()
        with patch.dict(sys.modules, {'huggingface_hub': hub}), patch('urllib.request.urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'model missing'):
                detector.detect(np.zeros((32, 32, 3), np.uint8))
            hub.snapshot_download.assert_not_called(); network.assert_not_called()

    def test_omdet_download_rejects_wrong_weights_and_keeps_existing(self):
        import sys, types
        from core.object_detection import download_omdet
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'omdet'; target.mkdir(); (target/'model.safetensors').write_bytes(b'previous')
            def fake(repo, revision, local_dir):
                Path(local_dir).mkdir(parents=True, exist_ok=True); (Path(local_dir)/'model.safetensors').write_bytes(b'tampered')
            hub = types.SimpleNamespace(snapshot_download=fake)       # stands in for huggingface_hub; no network
            with patch.dict(sys.modules, {'huggingface_hub': hub}):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    download_omdet(target)
            self.assertEqual((target/'model.safetensors').read_bytes(), b'previous')

    def test_results_become_normalized_sorted_boxes(self):
        from core.object_detection import OmDetDetector
        result = {'text_labels': ['plate', 'screwdriver', 'plate'],       # tensors in real use; lists work the same
                  'scores': np.array([.5, .9, .7]),
                  'boxes': np.array([[10., 20., 50., 60.], [0., 0., 200., 100.], [30., 30., 30., 40.]])}
        out = OmDetDetector._format(result, ['plate', 'screwdriver'], 200, 100)
        self.assertEqual([d['label'] for d in out], ['screwdriver', 'plate'])    # zero-width box dropped
        np.testing.assert_allclose(out[1]['bbox'], [.05, .2, .25, .6])
        np.testing.assert_allclose(out[0]['bbox'], [0, 0, 1, 1])

    def test_prompt_validation(self):
        from core.object_detection import OmDetDetector
        detector = OmDetDetector(Path(tempfile.gettempdir())/'reins-no-omdet')
        for labels in (['x' * 61], ['a'] * 401, [' ']):
            with self.assertRaisesRegex(ValueError, 'object names'):
                detector.detect(np.zeros((8, 8, 3), np.uint8), labels=labels)
        with self.assertRaisesRegex(ValueError, 'RGB uint8'):
            detector.detect(np.zeros((8, 8), np.uint8))

    def test_household_names_are_selectable_but_qualifiers_still_go_to_the_vlm(self):
        self.assertEqual(canonical_label('the plate'), 'plate')
        self.assertEqual(canonical_label('screwdriver'), 'screwdriver')
        self.assertIsNone(canonical_label('red plate'))
        plate = {'label': 'plate', 'confidence': .8, 'bbox': [.1, .1, .3, .3]}
        self.assertEqual(select_target([plate, BOX], 'plate'), plate)

    def test_nanodet_label_filter(self):
        detector = ObjectDetector()
        detector.net = Mock(); detector.net.forward.return_value = []       # loaded net stand-in; no weights needed
        with patch.object(detector, '_decode', return_value=[BOX, {**BOX, 'label': 'person'}]), \
             patch.object(ObjectDetector, '_prepare', return_value=(None, None)):
            found = detector.detect(np.zeros((8, 8, 3), np.uint8), labels=['person'])
        self.assertEqual([d['label'] for d in found], ['person'])


if __name__=='__main__':
    unittest.main()
