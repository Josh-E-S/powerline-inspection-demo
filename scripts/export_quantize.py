"""Export models to ONNX and quantize the detector to INT8.

Detector: Ultralytics FP32 ONNX export, then ONNX Runtime static
quantization calibrated on 500-1000 shuffled validation images.
Classifier: FP32 ONNX export only (quantization scope is detector-only).

Includes a sanity check comparing FP32 vs INT8 detections on the same
batch before any metrics are trusted.

Status: stub. Requirements in docs/powerline-inspection-demo-spec.md.
"""
