"""Local RGB object detection. Boxes are normalized xyxy, never metric positions.

Default: OmDet-Turbo swin-tiny (Apache-2.0), an open-vocabulary detector. It finds
whatever it is asked for by name, from VOCABULARY (COCO's 80 classes plus common
household objects) or from labels passed to detect(). Runs on CUDA, Apple MPS or CPU.
Fallback: NanoDet-M-Plus 1.5x/416 from OpenCV Zoo (Apache-2.0, 80 COCO classes,
OpenCV CPU), used when PyTorch/transformers or the OmDet weights are not installed.
See models/README.md. Model downloads are explicit (tools/detect_objects.py
--download) and checksum-verified; start-up and inference never touch the network.
"""
import hashlib
import io
import os
from pathlib import Path
import re
import threading
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw

MODEL_REV = '47534e27c9851bb1128ccc0102f1145e27f23f98'
MODEL_NAME = 'object_detection_nanodet_2022nov.onnx'
MODEL_SHA256 = '4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186'
MODEL_URL = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_REV}/models/object_detection_nanodet/{MODEL_NAME}'
DEFAULT_MODEL = Path(__file__).resolve().parent / 'models' / MODEL_NAME
LABELS = ('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,'
          'stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,'
          'backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,'
          'baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,'
          'banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,'
          'bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,'
          'refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush').split(',')
# Extra names the open-vocabulary detector looks for by default: tabletop things COCO lacks.
# Kept short on purpose: every name costs time on each frame (80 names ~70 ms, 144 ~143 ms on an
# RTX 4060). Any other name can be passed to detect(labels=...).
HOUSEHOLD = ('plate,drinking glass,mug,jar,box,lid,tray,basket,pan,pot,can,bag,towel,sponge,handle,'
             'drawer,pen,marker,tape,screwdriver,tool,toy,cable,glasses').split(',')
VOCABULARY = tuple(LABELS) + tuple(h for h in HOUSEHOLD if h not in LABELS)

OMDET_REPO = 'omlab/omdet-turbo-swin-tiny-hf'
OMDET_REV = '7fe93cecfb770c4d76cf71163956221249cab566'
OMDET_SHA256 = '439d1575d7e237ad565ed6969ea2a2dfcedf2086155e9ca3ac96cd6180a48cfd'   # model.safetensors at OMDET_REV
OMDET_DIR = Path(__file__).resolve().parent / 'models' / 'omdet-turbo-swin-tiny'


def download_model(path=DEFAULT_MODEL):
    """Explicit setup only; checksum-pin weights and atomically publish the file."""
    path = Path(path)
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        data = response.read(8 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError('NanoDet download checksum mismatch; model was not installed')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.download')
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def download_omdet(path=OMDET_DIR):
    """Explicit setup only: fetch the pinned OmDet-Turbo revision, verify the weights, publish atomically."""
    from huggingface_hub import snapshot_download
    path = Path(path)
    tmp = path.with_name(path.name + '.download')
    snapshot_download(OMDET_REPO, revision=OMDET_REV, local_dir=tmp)
    weights = tmp / 'model.safetensors'
    if _sha256(weights) != OMDET_SHA256:
        raise ValueError('OmDet-Turbo download checksum mismatch; model was not installed')
    if path.exists():
        import shutil
        shutil.rmtree(path)
    tmp.replace(path)
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_label(selector):
    """Only unqualified class names; never discard colour/spatial qualifiers."""
    label = re.sub(r'^(?:the|a|an)\s+', '', selector.strip().lower())
    label = {'mug': 'cup', 'sofa': 'couch', 'phone': 'cell phone', 'television': 'tv'}.get(label, label)
    return label if label in VOCABULARY else None


def select_target(detections, selector):
    label = canonical_label(selector)
    if label is None:
        return None  # rich descriptions belong to the configured VLM
    matches = [d for d in detections if d['label'] == label]
    if len(matches) > 1:
        raise ValueError(f'Multiple {label} objects detected. Describe which one, for example its colour or position.')
    return matches[0] if matches else None


class ObjectDetector:
    """NanoDet fallback: 80 COCO classes on OpenCV's CPU backend."""
    name = 'NanoDet · local CPU'
    open_vocabulary = False

    def __init__(self, model_path=DEFAULT_MODEL, confidence=.4):
        if not np.isfinite(confidence) or not .1 <= confidence <= .95:
            raise ValueError('Detection confidence must be between 0.1 and 0.95')
        self.path, self.confidence = Path(model_path), float(confidence)
        self.lock = threading.Lock()
        self.net = None

    @property
    def available(self):
        return self.path.is_file()

    @staticmethod
    def _prepare(rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('Detector expects an RGB uint8 image')
        h, w = rgb.shape[:2]
        scale = 416 / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        left, top = (416 - nw) // 2, (416 - nh) // 2
        canvas = np.zeros((416, 416, 3), np.uint8)
        canvas[top:top+nh, left:left+nw] = cv2.resize(rgb[:, :, ::-1], (nw, nh), interpolation=cv2.INTER_AREA)
        normalized = (canvas.astype(np.float32) - [103.53, 116.28, 123.675]) / [57.375, 57.12, 58.395]
        return cv2.dnn.blobFromImage(normalized.astype(np.float32)), (left, top, nw, nh)

    def detect(self, rgb, labels=None):
        """labels: optional names to keep; NanoDet can only ever return COCO classes."""
        if labels is not None:
            wanted = set(labels)
            return [d for d in self.detect(rgb) if d['label'] in wanted]
        blob, mapping = self._prepare(rgb)
        with self.lock:  # OpenCV nets are mutable; dashboard and planner share one instance.
            if self.net is None:
                if not self.available:
                    raise ValueError('Detector model missing. Run .venv/bin/python tools/detect_objects.py --download')
                if hashlib.sha256(self.path.read_bytes()).hexdigest() != MODEL_SHA256:
                    raise ValueError('Detector requires the verified NanoDet model; run tools/detect_objects.py --download')
                self.net = cv2.dnn.readNetFromONNX(str(self.path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.net.setInput(blob)
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self._decode(outputs, mapping)

    def _decode(self, outputs, mapping):
        boxes, scores, classes = [], [], []
        if len(outputs) != 6:
            raise ValueError('Unexpected NanoDet output layout')
        for level, stride in enumerate((8, 16, 32)):
            cls = outputs[level*2].reshape(-1, 80)
            logits = outputs[level*2+1].reshape(-1, 4, 8)
            side = 416 // stride
            if cls.shape[0] != side * side or logits.shape[0] != side * side:
                raise ValueError('Unexpected NanoDet output size')
            ids = cls.argmax(axis=1)
            conf = cls.max(axis=1)
            keep = np.flatnonzero(np.isfinite(conf) & (conf >= self.confidence))
            if not len(keep):
                continue
            logits = logits[keep]
            probs = np.exp(logits - logits.max(axis=2, keepdims=True))
            distances = (probs / probs.sum(axis=2, keepdims=True)) @ np.arange(8) * stride
            centers = np.column_stack((keep % side, keep // side)) * stride + (stride-1)*.5
            box = np.column_stack((centers - distances[:, :2], centers + distances[:, 2:]))
            boxes.extend(box); scores.extend(conf[keep]); classes.extend(ids[keep])
        if not boxes:
            return []
        boxes = np.asarray(boxes).clip(0, 416)
        left, top, nw, nh = mapping
        boxes = ((boxes - [left, top, left, top]) / [nw, nh, nw, nh]).clip(0, 1)
        scores, classes = np.asarray(scores), np.asarray(classes)
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
        # Suppress duplicate boxes within each class, not overlapping different objects.
        selected = []
        for cls in np.unique(classes):
            indices = np.flatnonzero(classes == cls)
            xywh = boxes[indices].copy(); xywh[:, 2:] -= xywh[:, :2]
            keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores[indices].tolist(), self.confidence, .5)
            selected.extend(indices[np.asarray(keep, dtype=int).reshape(-1)])
        selected.sort(key=lambda i: float(scores[i]), reverse=True)
        return [{'label': LABELS[classes[i]], 'confidence': round(float(scores[i]), 4),
                 'bbox': boxes[i].tolist()} for i in selected[:100]]


class OmDetDetector:
    """OmDet-Turbo swin-tiny: open-vocabulary detection of any named object.

    Same interface as ObjectDetector (available, confidence, detect -> label/confidence/
    normalized xyxy bbox). Loads lazily from a local, checksum-verified copy; never downloads.
    """
    open_vocabulary = True

    def __init__(self, model_dir=OMDET_DIR, confidence=.4, device='auto', vocabulary=VOCABULARY):
        if not np.isfinite(confidence) or not .1 <= confidence <= .95:
            raise ValueError('Detection confidence must be between 0.1 and 0.95')
        self.path, self.confidence, self.requested_device = Path(model_dir), float(confidence), device
        self.vocabulary = list(vocabulary)
        self.lock = threading.Lock()
        self.model = self.processor = self.device = None

    @staticmethod
    def dependencies_installed():
        import importlib.util
        return all(importlib.util.find_spec(m) for m in ('torch', 'transformers'))

    @property
    def available(self):
        return (self.path / 'model.safetensors').is_file() and (self.path / 'config.json').is_file() \
            and self.dependencies_installed()

    @property
    def name(self):
        return f"OmDet-Turbo · {self.device or self._pick_device()}"

    def _pick_device(self):
        if self.requested_device != 'auto':
            return self.requested_device
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _load(self):
        if not self.available:
            raise ValueError('Detector model missing. Run .venv/bin/python tools/detect_objects.py --download')
        if _sha256(self.path / 'model.safetensors') != OMDET_SHA256:
            raise ValueError('Detector requires the verified OmDet-Turbo weights; run tools/detect_objects.py --download')
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')   # run any op MPS lacks on the CPU instead of failing
        import torch
        from transformers import AutoProcessor, OmDetTurboForObjectDetection
        self.torch = torch
        self.device = self._pick_device()
        self.processor = AutoProcessor.from_pretrained(self.path, local_files_only=True)
        self.model = OmDetTurboForObjectDetection.from_pretrained(self.path, local_files_only=True).to(self.device).eval()

    def detect(self, rgb, labels=None):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('Detector expects an RGB uint8 image')
        classes = [str(l).strip().lower() for l in (labels or self.vocabulary) if str(l).strip()]
        if not classes or len(classes) > 400 or any(len(c) > 60 for c in classes):
            raise ValueError('Give between 1 and 400 object names of at most 60 characters')
        h, w = rgb.shape[:2]
        with self.lock:   # one model instance is shared by the dashboard stream and the planner
            if self.model is None:
                self._load()
            inputs = self.processor(Image.fromarray(rgb), text=classes, return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs, text_labels=classes, target_sizes=[(h, w)], threshold=self.confidence,
                nms_threshold=.5, max_num_det=100)[0]
        return self._format(result, classes, w, h)

    @staticmethod
    def _format(result, classes, w, h):
        names = result.get('text_labels') or [classes[int(k)] for k in _as_list(result['labels'])]
        out = []
        for name, score, box in zip(names, _as_list(result['scores']), _as_list(result['boxes'])):
            x0, y0, x1, y1 = np.clip(np.array(box, float) / [w, h, w, h], 0, 1)
            if np.isfinite([x0, y0, x1, y1]).all() and x1 > x0 and y1 > y0:
                out.append({'label': name, 'confidence': round(float(score), 4), 'bbox': [float(x0), float(y0), float(x1), float(y1)]})
        out.sort(key=lambda d: -d['confidence'])
        return out[:100]


def _as_list(x):
    return x.tolist() if hasattr(x, 'tolist') else list(x)


def make_detector(kind='auto', confidence=None, nanodet_model=DEFAULT_MODEL, omdet_dir=OMDET_DIR, device='auto'):
    """'auto': OmDet-Turbo when its weights and PyTorch are installed, else NanoDet."""
    if kind not in ('auto', 'omdet', 'nanodet'):
        raise ValueError('Detector must be auto, omdet or nanodet')
    if kind == 'omdet' or (kind == 'auto' and OmDetDetector(omdet_dir, device=device).available):
        return OmDetDetector(omdet_dir, .4 if confidence is None else confidence, device)
    return ObjectDetector(nanodet_model, .4 if confidence is None else confidence)


def annotate(rgb, detections):
    """Render labels on exactly the pixels inferred, without temporal overlays."""
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for detection in detections:
        x0, y0, x1, y1 = np.array(detection['bbox']) * [w, h, w, h]
        text = f"{detection['label']} {detection['confidence']:.0%}"
        color = (111, 245, 210)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=max(2, w//400))
        tw = draw.textbbox((0, 0), text)[2] + 10
        tx, ty = max(0, min(x0, w-tw)), max(0, y0-19)
        draw.rectangle((tx, ty, tx+tw, ty+18), fill=(12, 28, 28))
        draw.text((tx+5, ty+2), text, fill=color)
    out = io.BytesIO(); image.save(out, 'JPEG', quality=88)
    return out.getvalue()
