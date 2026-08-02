# Milestone 1: Training and Quantization Complete

Date: 2026-08-01. Repo state: commit `3e4551e` on main.
Status: both models trained, quantized, and benchmarked on the held-out
test set. App and deployment not yet built. This document records what
was done, the measured results, published work to compare against, and a
detailed plan for a second training run intended to meet or beat the
published numbers. It is written to be handed to a fresh Claude session
with no other context.

## 1. Project context

Portfolio work sample for a Solutions Engineer (AI/ML) role: a demo-scale
replica of an AI powerline inspection pipeline. UAV imagery goes in,
detected components with condition labels come out, gated by a confidence
threshold into auto-accepted findings and a human review queue, exported
as GeoJSON for utility GIS ingestion. Full spec: `docs/powerline-inspection-demo-spec.md`.
Plain-English overview: `docs/PROJECT_OVERVIEW.md`. Owner: Josh
(presents and defends every claim in interviews, so every number must be
measured and reproducible; no overclaiming anywhere).

Architecture is two-stage because the dataset forces it: InsPLAD-det has
boxes with no condition labels, InsPLAD-fault has condition crops with no
boxes. Stage 1: YOLO11-s detects 17 asset classes. Stage 2:
EfficientNetV2-S classifies crops of the five defect-prone asset types
into 11 combined asset__condition classes.

## 2. What was accomplished

- Dataset research and acquisition. The official Google Drive link is
  dead (404). Working source is the authors' Mendeley deposit:
  https://data.mendeley.com/datasets/5n3fjgvfyz/1 (single 6.4 GB zip,
  CC BY-NC 3.0). Direct download URL is in `data/README.md`.
- `scripts/prep_insplad.py`: COCO to YOLO conversion, dedupe of 46
  duplicate JSON image entries, dropped the `sphere` class (26 instances;
  excluding it reproduces the paper's 28,933 instance count exactly),
  normalized inconsistent Portuguese/English fault labels, seeded splits.
  Split strategy: the official val split (2,626 images) is the held-out
  TEST set so results are comparable to the paper; our own val (794
  images, 10% of train, seed 42) handles early stopping and INT8
  calibration. Test is touched only by `scripts/benchmark.py`.
- Training on Colab Pro (NVIDIA L4) via `notebooks/colab_training.ipynb`.
  Checkpoints on Google Drive under `MyDrive/powerline-runs/`. Smoke-test
  flags (`--smoke`) on both training scripts caught a dataset-path bug
  before it could waste a full run.
- Detector: 120 epochs at 640, ~2.8 h. Classifier: early-stopped at
  epoch 20 (best epoch 10), ~11 min.
- ONNX export and INT8 static quantization (`scripts/export_quantize.py`),
  benchmark on the test set (`scripts/benchmark.py`). Full plots and
  analysis: `docs/results/RESULTS.md`.

## 3. Measured results

Detector, held-out test set (= official InsPLAD val split), Apple M5 CPU,
200 timed single-image runs after warmup:

| Model | Size (MB) | mAP50 | mAP50-95 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| FP32 ONNX | 38.0 | 0.893 | 0.734 | 34.7 | 36.3 |
| INT8 ONNX | 10.1 | 0.889 | 0.710 | 30.1 | 32.1 |

Classifier, fault test split (6,417 crops): balanced accuracy **0.960**,
11.3 ms/crop p50.

Weak detector classes (INT8 AP50), the target of the re-run:

| Class | AP50 | Train instances |
|---|---|---|
| glass insulator small shackle | 0.55 | 263 |
| glass insulator big shackle | 0.57 | 259 |
| glass insulator tower shackle | 0.61 | 195 |
| lightning rod shackle | 0.79 | 195 |

Every other class scores 0.89 to 0.995. The three glass-insulator
shackle variants are one confusion cluster: rare, physically small, and
visually similar to each other.

Quantization pitfall found and fixed (documented in
`docs/results/RESULTS.md`): quantizing YOLO's final output tensor, which
mixes 0-640 pixel coordinates with 0-1 class probabilities, crushes all
probabilities to zero under the shared scale. Fix: quantize Conv/MatMul
ops only. The sanity check in `export_quantize.py` now validates
coordinate and probability channels separately.

## 4. Published context and reading list

Baselines from the InsPLAD paper (measured on the same split we use as
test): DetectoRS 0.721 Box AP for detection; EfficientNet 0.954 balanced
accuracy for fault classification. Our classifier already beats its
baseline (0.960 vs 0.954). Our FP32 detector's 0.734 mAP50-95 exceeds
0.721, but VERIFY before claiming: the paper's "Box AP" must be confirmed
as COCO-style mAP50-95 (read the paper's evaluation section). Report both
mAP50 and mAP50-95 in all cases so no one can accuse metric-shopping.

- InsPLAD paper: https://arxiv.org/abs/2311.01619 (primary reference)
- YOLOv8-ECCa, a modified YOLOv8 built for InsPLAD, reports ~90% mAP:
  https://www.mdpi.com/1999-4893/19/1/66 (closest competitor; confirm
  which mAP variant their 90% is when comparing)
- DetectoRS: https://arxiv.org/abs/2006.02334
- EfficientNetV2: https://arxiv.org/abs/2104.00298
- Integer quantization principles (NVIDIA): https://arxiv.org/abs/2004.09602
- SAHI, slicing inference for small objects in aerial imagery:
  https://arxiv.org/abs/2202.06934 (relevant to the shackle problem;
  currently out of scope for the deployed app)

## 5. Re-run plan: meet or beat the published numbers at small size

Goal: exceed ~0.90 mAP50 (YOLOv8-ECCa territory) and clearly exceed
0.721 mAP50-95 (DetectoRS) on the same test split, without architecture
surgery, keeping the deployable-model story intact.

Why it is plausible: the deficit is concentrated in three small rare
classes. If resolution and oversampling lift the three shackle classes
from ~0.58 to a modest 0.80 (well below what well-supported classes
achieve), overall mAP50 gains roughly 4 points to ~0.93. Half that
improvement still clears 0.90.

### Runs, in order

All commands run in `notebooks/colab_training.ipynb` on Colab. Prefer an
A100 (Runtime > Change runtime type); L4 works but is ~3x slower. Always
check `nvidia-smi` output after any session restart: a recycled session
can silently come back CPU-only (this happened once; the tell is a
"pin_memory ... no accelerator" warning and glacial epochs).

1. Smoke first, after any fresh session (5 min):
   `!python3 scripts/train_detector.py --smoke --runs-dir $RUNS`
2. Main attempt, s @ 1280 with rare-class oversampling (~1.5 h A100):
   `!python3 scripts/train_detector.py --imgsz 1280 --oversample --runs-dir $RUNS`
3. Ceiling run, m @ 1280 (~2.5-3 h A100), only if (2) shows the expected
   lift:
   `!python3 scripts/train_detector.py --model yolo11m.pt --imgsz 1280 --oversample --runs-dir $RUNS`
4. Do NOT retrain the classifier. 0.960 already beats the baseline.

Notes on the flags: `--oversample` repeats images containing rare classes
(capped 4x, sqrt scaling on median/rarest frequency,
see `build_oversampled_yaml` in `scripts/train_detector.py`).
Hyperparameters are otherwise unchanged (120 epochs, patience 25, cosine
LR, close_mosaic 10, seed 42); resist tuning more things at once or the
ablation story muddies.

### Evaluation protocol (unchanged, non-negotiable)

- Download best.pt from Drive, then locally:
  `python scripts/export_quantize.py --weights <best.pt> --imgsz 1280`
  then `python scripts/benchmark.py --imgsz 1280`.
- The test set is only ever touched by benchmark.py. Same conf/NMS for
  FP32 and INT8. Check the sanity output: prob corr must be >= 0.95.
- Compare per-class AP50 against the table in section 3. The claim is
  only "beaten" if overall mAP50-95 exceeds 0.721 and mAP50 exceeds
  0.90 on this split, with the metric definitions verified.

### Expected outcomes and decision tree

- s@1280+os reaches ~0.92+ mAP50: headline achieved with the small
  model. Run the m@1280 ceiling anyway if budget allows, for the
  three-row latency-vs-accuracy table.
- s@1280+os lands 0.90-0.92: still a win; check whether the shackle
  cluster moved. If it did not, the bottleneck is label quality or
  intrinsic ambiguity, not resolution; stop and write that up honestly.
- No meaningful lift: unlikely, but then the honest README story is
  "we identified the ceiling for this model class" with the per-class
  evidence. That is still a good story.
- Deployment decision either way: the Space keeps serving 640 INT8
  (latency at 1280 is ~4x worse on CPU, ~120 ms/image). The 1280 model
  becomes the "best model trained" row of the README table. If 1280
  deploys anywhere, note that a p50 of ~120 ms is still usable for
  non-realtime review workflows.

### Budget

Roughly 25-45 Colab compute units total (both runs on A100), inside the
Pro monthly allotment. Wall clock: one evening.

## 6. Remaining project work after the re-run (or in parallel)

1. `scripts/make_pr_table.py`: precision/recall vs threshold JSON from
   the deployed INT8 model on the test set, for the app's slider readout.
2. `app/app.py`: Gradio app per `diagrams/app-architecture.md`. Models
   already exported in `app/models/` (detector_int8.onnx,
   classifier.onnx + classifier.onnx.data + classifier_classes.json).
   Inference via ONNX Runtime only. Threshold gates on detection
   confidence; review queue below threshold; GeoJSON export with
   simulated coordinates along a real transmission corridor east of
   Tucson, AZ, labeled simulated; 4-6 preloaded test-set examples.
3. Deploy to Hugging Face Spaces free CPU tier; re-measure latency there
   (x86 with VNNI should favor INT8 more than the Apple M5 did).
4. README per the spec's structure, embedding docs/results/ material.
   Owner's prose rules: no em dashes, natural voice, no marketing tone.

## 7. Handoff notes (for the next Claude session)

- Local layout: repo at
  `~/Desktop/Resume Projects/powerline-inspection-demo`, venv at `.venv`
  (python 3.14; ultralytics 8.4.115, onnxruntime 1.28.0, torch 2.13.0,
  onnxscript). Trained weights: `model/weights/detect_best.pt` (19 MB),
  `model/weights/classify_best.pt` (82 MB). Raw data: `data/raw/` (4.9
  GB, gitignored). Staged data: regenerate anytime with
  `python3 scripts/prep_insplad.py` (idempotent, seeded).
- Drive layout: `MyDrive/powerline-runs/detect/det_yolo11s_640/` and
  `classify/cls_effv2s/` hold the full run artifacts (results.csv,
  metrics.jsonl, weights, plots).
- GitHub: https://github.com/Josh-E-S/powerline-inspection-demo (public).
- Gotchas already survived, do not rediscover them: the dataset yaml
  needs an absolute `path:` (Ultralytics resolves relative paths against
  its global datasets dir); Finder once duplicated staged symlinks
  ("name 2.jpg"), fixed by wiping data/yolo/images+labels and re-running
  prep, so verify counts (7141/794/2626) if benchmark numbers look
  halved; the quantizer must keep `op_types_to_quantize=["Conv","MatMul"]`;
  metrics.jsonl on the Colab Drive mount syncs lazily, the .pt file
  timestamps are the reliable heartbeat; Colab sessions can restart
  CPU-only, check nvidia-smi.
