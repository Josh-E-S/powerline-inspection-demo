# Measured results

All numbers measured, none estimated. Detector metrics are on the held-out
test set, which is InsPLAD's official val split (2,626 images, 6,324
instances), so they are directly comparable to the published baselines.
The test set was touched exactly once, by scripts/benchmark.py.

## Training

- Detector: YOLO11-s, imgsz 640, 120 epochs (~2.8 h), Colab NVIDIA L4,
  seed 42. Best checkpoint by fitness around epoch 99.
- Classifier: EfficientNetV2-S, class-balanced sampling, early-stopped at
  epoch 20 (best epoch 10), ~11 min on the same L4.

![Detector training curves](detector_training_curves.png)

The visible step at epoch 111 is `close_mosaic`: mosaic augmentation
switches off for the final 10 epochs and train losses drop sharply. Val
losses keep improving to the end; no overfitting signature.

## Detector: FP32 vs INT8 (the deployment trade)

Benchmarked on: Apple M5 (CPU), imgsz=640, 200 timed single-image runs
after warmup. INT8 is static quantization (QDQ, per-channel, Conv/MatMul
only) calibrated on 800 validation images.

| Model | Size (MB) | mAP50 | mAP50-95 | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| FP32 ONNX | 38.0 | 0.893 | 0.734 | 34.7 | 36.3 |
| INT8 ONNX | 10.1 | 0.889 | 0.710 | 30.1 | 32.1 |

Quantization cost: 0.4 points of mAP50 (2.4 of mAP50-95) for a 3.8x
smaller model that is ~13% faster on this CPU. Note Apple Silicon gains
less from INT8 than x86 with VNNI; the deployed Space (x86) should see a
larger speedup. Published context: the InsPLAD paper's best detector
(DetectoRS) reports 0.721 Box AP on this split.

![INT8 precision-recall curve](int8_pr_curve.png)

## Per-class error analysis (INT8, AP50)

Strong (0.89-0.995): everything with training support, including all
polymer insulator hardware, dampers, yokes, vari-grip, tower id plates.

Weak, and why:

| Class | AP50 | Train instances | Hypothesis |
|---|---|---|---|
| glass insulator big shackle | 0.57 | 259 | rare + small + visually similar to small/tower shackle variants |
| glass insulator small shackle | 0.55 | 263 | same confusion cluster |
| glass insulator tower shackle | 0.61 | 195 | same cluster, fewest examples |
| lightning rod shackle | 0.79 | 195 | rare, small object |

The three glass-insulator shackle variants form one confusion cluster:
rare classes (195-263 instances vs 6,953 for the most common class),
physically small, and mutually similar. Higher training resolution
(1280) and rare-class oversampling are the planned levers; both are
wired into train_detector.py as flags.

![INT8 confusion matrix](int8_confusion_matrix.png)

Sample test-set predictions (INT8):

![Sample predictions](int8_sample_predictions.jpg)

## Condition classifier

Balanced accuracy on the fault test split (official val, 6,417 crops):
**0.960** vs the paper's 0.954 EfficientNet baseline. Per-crop CPU
latency: 11.3 ms (p50).

Weakest class: glass-insulator missing-cap at 0.70 recall on a 30-crop
test set (9 misses). Every other class is at or above 0.95. The test
split is heavily skewed (yoke-suspension: 5,742 good vs 20 rust), which
is exactly why balanced accuracy is the reported metric.

## Quantization pitfall worth documenting

The first INT8 export scored zero mAP despite a 0.9996 whole-tensor
output correlation with FP32. Cause: YOLO's final output tensor mixes
pixel coordinates (0-640) and class probabilities (0-1); quantizing that
tensor forces one shared scale, rounding every probability to zero, and
the coordinate channels dominate any whole-tensor similarity statistic.
Fix: quantize Conv/MatMul ops only, and validate coordinates and
probabilities separately (see scripts/export_quantize.py). The lesson:
sanity checks must inspect the signal you actually care about.

## Raw data

- [benchmark_table.md](benchmark_table.md), [benchmark_results.json](benchmark_results.json)
- Full per-epoch training logs live in the Colab run dirs (results.csv,
  metrics.jsonl) on Drive; reproduce with notebooks/colab_training.ipynb.
