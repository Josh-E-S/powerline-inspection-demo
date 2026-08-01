"""Generate the FP32 vs INT8 comparison table.

Measures on CPU: detector latency (p50/p95 over >=100 single-image runs
after warmup), full-pipeline latency (detector + classifier over crops),
mAP50 and mAP50-95 on the held-out test set at identical confidence/NMS
settings, per-class AP, classifier balanced accuracy, and model file sizes.
Emits a markdown table ready to paste into the README.

Status: stub. Requirements in docs/powerline-inspection-demo-spec.md.
"""
