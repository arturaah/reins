#!/usr/bin/env python3
"""Download the verified local detectors, or detect objects in an RGB image."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from core.object_detection import DEFAULT_MODEL, OmDetDetector, annotate, download_model, download_omdet, make_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Install checksum-verified weights: OmDet-Turbo (462 MB, needs torch + transformers) and the NanoDet fallback (3.8 MB)')
    parser.add_argument('--detector', choices=('auto', 'omdet', 'nanodet'), default='auto')
    parser.add_argument('--labels', help='Comma-separated object names to look for (OmDet-Turbo only), e.g. "screwdriver,tape,red cup"')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--input', type=Path, help='Local image; never commands the robot')
    parser.add_argument('--output', type=Path, help='Annotated JPEG output')
    parser.add_argument('--confidence', type=float, default=None)
    args = parser.parse_args()
    if not args.download and not args.input:
        parser.error('Use --download and/or --input IMAGE')
    if args.output and not args.input:
        parser.error('--output requires --input')
    if args.download:
        print(f'Installed {download_model(args.model)}')
        if OmDetDetector.dependencies_installed():
            print(f'Installed {download_omdet()}')
        else:
            print('Skipped OmDet-Turbo: install torch and transformers first (see core/requirements.txt)')
    if args.input:
        with Image.open(args.input) as image:
            rgb = np.array(image.convert('RGB'))
        detector = make_detector(args.detector, args.confidence, nanodet_model=args.model)
        labels = [l.strip() for l in args.labels.split(',')] if args.labels else None
        detections = detector.detect(rgb, labels=labels)
        print(json.dumps({'detector': detector.name, 'objects': detections, 'coordinates': 'normalized xyxy; no depth'}, indent=2))
        if args.output:
            args.output.write_bytes(annotate(rgb, detections))


if __name__ == '__main__':
    main()
