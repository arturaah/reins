"""Bounded, opt-in detection worker; never queues frames or commands hardware."""
import copy
import threading
import time
import uuid

from core.object_detection import annotate
from core.perception import PerceptionError


class DetectionStream:
    # RGB feed age is receive age, not a guarantee about sensor capture latency.
    MAX_AGE = 3.0

    def __init__(self, detector, sources, autostart=True):
        self.detector, self.sources = detector, sources
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.enabled, self.source, self.generation = False, next(iter(sources)), 0
        self.result, self.jpg, self.message = None, b'', 'Enable detection to inspect a camera.'
        if autostart:
            threading.Thread(target=self._run, daemon=True).start()

    def configure(self, enabled, source):
        if type(enabled) is not bool or not isinstance(source, str) or source not in self.sources:
            raise ValueError('Choose a detection source and a boolean enabled value')
        with self.lock:
            if (enabled, source) != (self.enabled, self.source):
                self.enabled, self.source = enabled, source
                self.generation += 1
                self.result, self.jpg = None, b''
                self.message = 'Waiting for a current frame.' if enabled else 'Detection paused.'
        return self.status()

    def _current(self):
        return bool(self.enabled and self.result and 0 <= time.monotonic()-self.result['received_at'] < self.MAX_AGE)

    def status(self):
        with self.lock:
            current = self._current()
            result = copy.deepcopy(self.result) if current else None
            if result:
                result['age_s'] = round(time.monotonic()-result.pop('received_at'), 2)
            message = self.message
            if self.enabled and self.result and not current:
                message = 'Detection frame expired; waiting for a fresh source.'
            return {'enabled': self.enabled, 'source': self.source, 'sources': list(self.sources),
                    'available': self.detector.available, 'confidence': self.detector.confidence,
                    'model': getattr(self.detector, 'name', 'local detector'), 'ready': current, 'result': result, 'message': message}

    def image(self, frame_id):
        with self.lock:
            return self.jpg if self._current() and self.result['id'] == frame_id else b''

    def step(self):
        """One synchronous iteration, also usable in deterministic concurrency tests."""
        with self.lock:
            if not self.enabled:
                return
            generation, source = self.generation, self.source
            previous = self.result['source_id'] if self.result else None
        try:
            if not self.detector.available:
                raise ValueError('Install the detector: .venv/bin/python tools/detect_objects.py --download')
            # A source returns immutable RGB, receive time, frame identity, optional calibrated observation.
            rgb, received_at, source_id, observation = self.sources[source]()
            if not 0 <= time.monotonic()-received_at < self.MAX_AGE:
                raise ValueError('Camera frame is stale; waiting for a fresh source.')
            if source_id == previous:
                return
            start = time.monotonic()
            objects = copy.deepcopy(self.detector.detect(rgb))
            for obj in objects:
                obj['surface_m'] = None
                obj['depth_detail'] = 'RGB only · position unknown'
                if observation is not None:
                    try:
                        surface, _, quality = observation.locate(obj['bbox'])
                        obj['surface_m'] = surface.tolist()
                        obj['depth_detail'] = f"Measured depth {quality['depth_m']:.2f} m"
                    except PerceptionError as exc:
                        obj['depth_detail'] = str(exc)
            jpg = annotate(rgb, objects)
            if time.monotonic()-received_at >= self.MAX_AGE:
                raise ValueError('Frame expired during inference; detections discarded.')
            result = {'id': uuid.uuid4().hex, 'source_id': source_id, 'received_at': received_at,
                      'captured_at': observation.captured_at if observation is not None else None,
                      'calibration_id': observation.calibration_id if observation is not None else None,
                      'objects': objects, 'inference_ms': round((time.monotonic()-start)*1000),
                      'width': rgb.shape[1], 'height': rgb.shape[0]}
            with self.lock:
                if generation == self.generation:
                    self.result, self.jpg = result, jpg
                    self.message = f'{len(objects)} objects detected.' if objects else 'No objects above the confidence threshold.'
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.result, self.jpg = None, b''
                    self.message = str(exc) or type(exc).__name__

    def _run(self):
        while not self.stopped.is_set():
            self.step()
            self.stopped.wait(.3)  # latest frame only, at most ~3 Hz; shared net serialized

    def close(self):
        self.stopped.set()
