Benchmarked on: Apple M5 (CPU), imgsz=640, 200 timed runs after warmup.

| Model | Size (MB) | mAP50 | mAP50-95 | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| FP32 ONNX | 38.0 | 0.8926 | 0.7343 | 34.7 | 36.3 |
| INT8 ONNX | 10.1 | 0.889 | 0.7103 | 30.1 | 32.1 |
