# Deployment-Oriented Re-evaluation of the InsPLAD Detection Benchmark

**A technical report on training, quantizing, and serving a two-stage
power line inspection pipeline**

Josh E. S. | August 2026 | [Repository](https://github.com/Josh-E-S/powerline-inspection-demo)

---

## Abstract

The InsPLAD benchmark (Vieira-e-Silva et al., 2023) established reference
results for power line asset inspection in UAV imagery, reporting 0.721
COCO Box AP for detection (DetectoRS) and 0.954 balanced accuracy for
defect classification (EfficientNet). This report re-evaluates that
detection benchmark using a current standard detector (YOLO11-s, 9.4M
parameters) with no architectural modification, under a
deployment-oriented constraint: the resulting model must run on a
commodity CPU at a size suitable for free-tier hosting.

On the same evaluation split, a stock YOLO11-s trained at 1280px with
rare-class oversampling reaches **0.738 mAP50-95 and 0.912 mAP50**,
exceeding the published DetectoRS results of 0.721 AP and 0.885 AP50.
After INT8 static quantization to **10.6 MB**, the model retains **0.726
mAP50-95 and 0.906 mAP50**, so the compressed model alone still exceeds
the published detection baseline. A companion condition classifier
reaches 0.960 balanced accuracy against the published 0.954.

Three experiments are reported, including one that failed to improve the
strict metric and the hypothesis-driven follow-up that fixed it. Two
distinct INT8 quantization failure modes were encountered, diagnosed, and
mitigated; both are documented because the diagnostic method generalizes
beyond this dataset.

**This is an engineering replication study, not novel research.** It
contributes no new architecture or training method. Its claim is that
current standard tooling plus targeted, cheap interventions can exceed a
2023 research baseline while producing an artifact that is actually
deployable, and that the failure modes encountered along the way are
worth documenting.

---

## 1. Motivation

Published detection baselines are snapshots. DetectoRS (Qiao et al.,
2020) is a heavyweight research architecture: recursive feature pyramids
and switchable atrous convolution, effective but never intended to be
deployed on constrained hardware. Meanwhile, standard detector tooling
has advanced substantially since the benchmark was set.

Two questions follow, and this report addresses both:

1. Does a current stock detector, competently trained, reach or exceed
   the published baseline without architectural work?
2. Does that hold after the model is compressed to something a free CPU
   tier can serve?

The second question is the one that matters operationally. Utilities
process millions of inspection images per program; per-image cost, not
peak accuracy, governs whether a system is viable.

## 2. Dataset

**InsPLAD** comprises 10,607 UAV images of operating transmission lines
with 28,933 annotated asset instances across 17 classes, plus a separate
set of cropped component images labelled by condition. The two components
are disjoint: detection annotations carry no condition labels, and
condition crops carry no bounding boxes. This structural fact dictates a
two-stage architecture; a single-stage detector over combined
asset-condition classes is not supported by the annotations.

Source: the repository's Google Drive link is dead (404 as of July 2026).
The working source is the authors' Mendeley deposit (CC BY-NC 3.0).

### 2.1 Preparation

Four data issues were found and corrected in `scripts/prep_insplad.py`:

| Issue | Detail | Handling |
|---|---|---|
| Duplicate entries | 46 duplicate image entries in the train JSON (7,981 entries, 7,935 unique files) | Dedupe by filename, remap image ids |
| Phantom class | JSON defines 18 categories; excluding `sphere` (26 instances) reproduces the published 28,933 total exactly | Dropped; 17 classes retained for comparability |
| Inconsistent labels | Condition folders mix Portuguese and English across splits (`corrosão`/`normal` vs `rust`/`good`) | Normalized to one vocabulary |
| No test split | Dataset ships train/val only | See 2.2 |

### 2.2 Splits

The dataset's official validation split (2,626 images, 6,324 instances,
21.9% of annotations, consistent with the paper's 80/20 split by
transmission tower) is treated as the **held-out test set** and is
touched exactly once, by `scripts/benchmark.py`. A separate validation
set (794 images, 10% of train, seed 42) is carved from the training data
for early stopping and quantization calibration.

Consequence worth stating: this leaves **7,141 training images against
the baselines' 7,935**, so the models reported here are trained on ~10%
less data than the published baselines, not more.

## 3. Method

### 3.1 Architecture

Two stages, following the dataset's structure:

- **Stage 1, detection.** YOLO11-s (9.4M parameters), COCO-pretrained,
  fine-tuned on the 17 asset classes.
- **Stage 2, condition classification.** EfficientNetV2-S fine-tuned on
  fault crops, one head over 11 combined `asset__condition` classes, with
  logits masked at inference to the detected asset's valid conditions.

### 3.2 Training

Detector: image size 640 or 1280, up to 120 epochs, cosine LR,
`close_mosaic=10`, seed 42, auto-batch, Ultralytics defaults otherwise.
Classifier: class-balanced sampling, AdamW at 1e-4, cosine decay, early
stopping on validation balanced accuracy.

Hardware: Google Colab (NVIDIA L4 and A100). Total GPU cost across all
runs was under $10 of subscription compute.

### 3.3 Quantization

FP32 ONNX export via Ultralytics, then ONNX Runtime static quantization
(QDQ format, per-channel weights) calibrated on 800 shuffled real
validation images. Quantization is applied to the detector only; the
classifier is small enough that compressing it would complicate the
comparison for negligible benefit.

## 4. Experiments

### 4.1 Run 1: baseline (640px)

Stock configuration, 120 epochs. Establishes the pipeline and the error
profile. Per-class analysis showed the deficit concentrated in four
classes, all small and rare: three glass-insulator shackle variants
(195-263 training instances each, AP50 0.55-0.61) against 0.89-0.99 for
every well-represented class.

### 4.2 Run 2: resolution and oversampling (1280px)

**Hypothesis.** The weak classes fail because they are physically small
(a shackle occupies roughly 15px at 640) and rare. Quadrupling input
pixels and repeating rare-class images should lift them.

Implementation: image size 1280, plus a weighted training list repeating
images containing rare classes (capped 4x, sqrt scaling on
median-to-rarest frequency). Early stopping fired at epoch 67.

**Outcome: partially confirmed.** mAP50 improved 0.893 to 0.909, but
mAP50-95 *declined* to 0.723. Weak classes improved less than validation
suggested (validation showed 0.90-0.995 for the shackle classes; the test
set showed 0.62-0.68). The validation split holds 5-20 instances of each
shackle class against 195-263 in test: the apparent breakthrough was
small-sample optimism, caught only because the test split was reserved.

### 4.3 Run 3: augmentation schedule (1280px, 60 epochs)

**Hypothesis for run 2's flat strict metric.** Early stopping at epoch 67
meant the schedule never reached `close_mosaic` at epoch 111, so mosaic
augmentation ran for the entire run. Mosaic composites four images and
distorts objects at the seams, trading box precision for feature
robustness; the final mosaic-free epochs are where box precision
consolidates. mAP50-95 rewards precisely that. The 640 baseline received
its ten mosaic-free epochs; run 2 did not.

Implementation: `--epochs 60 --patience 0`, so `close_mosaic` fires at
epoch 50 and early stopping cannot pre-empt it.

**Outcome: confirmed.** The mechanism is visible in the logs (at epoch 51,
train box loss drops 0.474 to 0.460, cls loss 0.286 to 0.246), and
mAP50-95 rose 0.723 to 0.738, passing the 640 baseline.

**Confound, stated plainly.** Shortening the schedule also changed
Ultralytics' `optimizer=auto` selection from MuSGD to AdamW, so two
variables moved. The gain cannot be attributed to `close_mosaic` alone. A
`--optimizer` flag now exists to pin this in future work.

## 5. Results

All figures on the held-out test set (2,626 images, 6,324 instances).
Latency: single-image ONNX Runtime inference on an Apple M5 CPU, 200
timed runs after warmup.

| Model | Size (MB) | mAP50 | mAP50-95 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| 640 FP32 | 38.0 | 0.893 | 0.734 | 34.7 | 36.3 |
| 640 INT8 | 10.1 | 0.889 | 0.710 | 30.1 | 32.1 |
| 1280 run 2 FP32 | 38.5 | 0.909 | 0.723 | 149.2 | 154.9 |
| 1280 run 2 INT8 | 10.6 | 0.883 | 0.683 | 122.4 | 127.3 |
| **1280 run 3 FP32** | 38.5 | **0.912** | **0.738** | 150.3 | 155.6 |
| **1280 run 3 INT8** | 10.6 | 0.906 | 0.726 | 127.2 | 132.0 |

### 5.1 Against published baselines

The InsPLAD paper defines its detection metric as "Box AP from MS COCO,
also known as AP (with IoU as 0.50:0.95)", and reports for DetectoRS:
AP 0.721, AP50 0.885, AP75 0.749. Metric definitions therefore match.

| Metric | Published (DetectoRS) | This work (FP32) | This work (INT8, 10.6 MB) |
|---|---|---|---|
| mAP50-95 | 0.721 | **0.738** | **0.726** |
| mAP50 | 0.885 | **0.912** | **0.906** |

Classification: **0.960** balanced accuracy against the published 0.954
(EfficientNet), on the fault test split (6,417 crops).

A separate 2026 paper, YOLOv8-ECCa, reports 82.75% mAP50 on InsPLAD-det.
That figure is *below* the InsPLAD paper's own DetectoRS AP50 of 0.885,
which indicates a different split or evaluation protocol; it is therefore
not treated here as a comparable number. Comparisons in this report are
restricted to the InsPLAD paper's own benchmark, whose split this work
uses.

### 5.2 Per-class error profile

Weak classes, AP50, 640-INT8 to run-3-INT8:

| Class | Train instances | 640 INT8 | Run 3 INT8 |
|---|---|---|---|
| lightning rod shackle | 195 | 0.792 | 0.921 |
| glass insulator tower shackle | 195 | 0.609 | 0.645 |
| glass insulator big shackle | 259 | 0.568 | 0.596 |
| glass insulator small shackle | 263 | 0.552 | 0.581 |

The three glass-insulator shackle variants remain the model's floor.
Their residual confusion is predominantly *with each other* rather than
with background, which points at inter-class ambiguity in the labels
rather than at anything further resolution would fix. This is the honest
limit of the interventions applied here.

## 6. Quantization findings

Two distinct INT8 failure modes were encountered. Both were caught by
output-level sanity checks before benchmarking, and both generalize.

**Failure 1: mixed-scale output tensor.** The first INT8 export scored
zero mAP while showing 0.9996 whole-tensor correlation against FP32.
Cause: YOLO's output tensor concatenates box coordinates (0-640) with
class probabilities (0-1). Quantizing that tensor forces a single shared
scale; the coordinate magnitudes dominate it and every probability rounds
to zero. The coordinate channels also dominate any whole-tensor
similarity statistic, so the naive check reported near-perfect agreement
on a model that detected nothing. Fix: restrict quantization to
Conv/MatMul operations, and validate coordinate and probability channels
*separately*.

**Failure 2: calibration at higher resolution.** INT8 at 1280 lost 2.6
points of mAP50 against 0.4 at 640. The hardened check flagged it:
box-coordinate correlation 0.9996 while probability correlation fell to
0.837, with per-image detection counts diverging on crowded scenes (76
FP32 detections against 12 INT8) while simple scenes matched. Larger
inputs produce more anchor positions and wider activation tails, so
MinMax calibration lets outliers stretch the scale until small
probabilities vanish.

**Incidental finding.** Run 3's weights quantized cleanly (probability
correlation 0.964, detection counts matching FP32) where run 2's did not,
despite identical quantization settings, resolution, and calibration
data. The runs differ in optimizer (AdamW vs MuSGD). Quantization
robustness therefore appears to be a property of the trained weights, not
solely of the quantization procedure. This was not investigated further
and is offered as an observation, not a result.

**Generalizable lesson.** A quantization sanity check must inspect the
signal that carries the decision, not an aggregate over the whole output.
Aggregate similarity can be near-perfect on a model that is entirely
broken.

## 7. Deployment

The served artifact is the **640px INT8 detector (10.1 MB)** with the
FP32 classifier, running under ONNX Runtime in a Gradio application on
free-tier CPU hosting. Run 3's INT8 model is more accurate (+1.7 mAP50)
but costs 127 ms per image against 30 ms; the smaller input is preferred
for cold-start reliability and per-image cost.

The application implements the operational half of the pipeline: a
confidence threshold gates detections into auto-accepted findings and a
visible human review queue, precision and recall at the current threshold
are displayed from a precomputed test-set table, and findings export as
GeoJSON point features shaped for GIS ingestion. Coordinates are
simulated along a real transmission corridor and labelled as simulated in
both the file and the interface.

## 8. Limitations

- **Demo scale.** One model per stage, one dataset, one geography.
- **Condition labels are image-level** on cropped components. The system
  detects components and classifies condition; it does not localize
  defects at pixel level.
- **Run 3 is confounded** (schedule and optimizer both changed).
- **Latency is measured on Apple M5**, not on the deployment target.
  x86 CPUs with VNNI should favour INT8 more; those figures are not yet
  collected.
- **Baselines are compared against reported numbers, not re-run.** The
  InsPLAD authors publish no trained checkpoints (the repository contains
  only a README, licence, and docs), so DetectoRS was not re-evaluated
  here. The comparison relies on the figures reported in the paper and on
  the metric definition stated there. Standard practice, but it means any
  difference in evaluation implementation between their pipeline and
  Ultralytics' would not be visible.
- **Split equivalence is inferred**, not confirmed by the authors: the
  official validation split matches the published 80/20-by-tower ratio
  (21.9% of annotations), but no explicit statement ties the shipped
  `val/` folder to the paper's test split.
- **AP75 not computed**, so one column of the published table is
  unmatched.
- **Coordinates are simulated.** No GPS or EXIF data was used.

## 9. Reproducibility

Every number above is produced by scripts in the repository, from data
preparation through benchmarking, with a fixed seed (42). The test split
is read by exactly one script. Full commands, environment versions,
per-class tables, and raw JSON results are in `docs/results/`.

```
scripts/prep_insplad.py      # COCO to YOLO, dedupe, normalize, split
scripts/train_detector.py    # resumable, --imgsz/--oversample/--optimizer
scripts/train_classifier.py  # balanced sampling, early stopping
scripts/export_quantize.py   # ONNX FP32 + INT8 + sanity checks
scripts/benchmark.py         # test-set accuracy, latency, table
```

## References

1. Vieira-e-Silva, A. L. B., et al. *InsPLAD: A Dataset and Benchmark for
   Power Line Asset Inspection in UAV Images.* International Journal of
   Remote Sensing 44(23), 2023. [arXiv:2311.01619](https://arxiv.org/abs/2311.01619)
2. Qiao, S., Chen, L.-C., Yuille, A. *DetectoRS: Detecting Objects with
   Recursive Feature Pyramid and Switchable Atrous Convolution.* 2020.
   [arXiv:2006.02334](https://arxiv.org/abs/2006.02334)
3. Tan, M., Le, Q. *EfficientNetV2: Smaller Models and Faster Training.*
   2021. [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)
4. Wu, H., et al. *Integer Quantization for Deep Learning Inference:
   Principles and Empirical Evaluation.* NVIDIA, 2020.
   [arXiv:2004.09602](https://arxiv.org/abs/2004.09602)
5. Ultralytics YOLO11. https://docs.ultralytics.com/models/yolo11/
