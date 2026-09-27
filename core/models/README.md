# Local object detector assets

**Default: OmDet-Turbo swin-tiny** (open vocabulary, Apache-2.0), from
https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf at revision
`7fe93cecfb770c4d76cf71163956221249cab566`. `tools/detect_objects.py --download`
installs it into `omdet-turbo-swin-tiny/` (git-ignored, 462 MB) and verifies
`model.safetensors` against SHA-256
`439d1575d7e237ad565ed6969ea2a2dfcedf2086155e9ca3ac96cd6180a48cfd`; a mismatch leaves
any existing copy untouched. Loaded with `local_files_only`, so nothing is fetched at
run time. Needs `torch` and `transformers`.

**Fallback: NanoDet**, below, used when PyTorch or the OmDet weights are absent.

Reins uses OpenCV Zoo's **NanoDet-M-Plus 1.5x, 416 × 416**, trained for 80 COCO
classes. The model and reference preprocessing/decoder are Apache-2.0 licensed;
the upstream license is included in [NANODET_LICENSE](NANODET_LICENSE).

- Source: https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_nanodet
- File: `object_detection_nanodet_2022nov.onnx` (3,800,954 bytes).
- SHA-256: `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186`.
- Install: `.venv/bin/python tools/detect_objects.py --download`.

`core/object_detection.py` adapts the Zoo preprocessing and distribution decoder
for RGB input, original-image normalized boxes, stable softmax, class-wise NMS,
thread-safe inference, lazy loading and checksum validation. The network outputs
three stride levels (8, 16 and 32), each with class probabilities and four
8-bin box-distance distributions. Model download is explicit and atomic;
startup and inference do not download anything. ONNX files are ignored by Git.

This is a lightweight baseline; small objects and clutter are challenging.
Labels and confidence are observations for review, not a safety or free-space
certification. It provides boxes, not segmentation, object pose or tracking.
