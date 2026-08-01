# Project Spec: Powerline Inspection Demo (Weekend Build)

## Purpose

Build a demo-scale replica of an AI-powered powerline inspection workflow: UAV imagery → component detection → condition classification → confidence-threshold gating with a human review queue → GeoJSON export shaped for utility GIS ingestion.

This is a portfolio work sample for a Solutions Engineer (AI/ML) role at a computer vision company serving electric utilities. It must demonstrate end-to-end model deployment thinking, not just training. The owner (Josh) will present and defend every claim in interviews, so **every number must be measured, every claim must be reproducible, and framing must be honest** (this is a demo-scale workflow replica, not a production system).

## Deliverables

1. **Trained detector**: YOLO11-small fine-tuned on InsPLAD-det, detecting the 17 power line asset classes.
2. **Condition classifier**: small ImageNet-pretrained model fine-tuned on InsPLAD-fault crops; labels detected components of the five defect-prone asset types (normal / corroded / broken / bird's nest as applicable).
3. **Quantized export**: detector exported to ONNX FP32 and ONNX INT8 (static quantization with in-domain calibration). Classifier ships as FP32 ONNX; it is small enough that quantizing it buys little and would muddy the comparison story.
4. **Comparison table** (the centerpiece artifact): detector FP32 vs INT8 — model file size, CPU inference latency (p50/p95), and detection accuracy on a held-out set. Full-pipeline latency (detector + classifier over the image's crops) reported alongside, separately.
5. **Gradio demo app**:
   - Image upload → rendered detections
   - Confidence threshold slider that live-updates displayed precision/recall (precomputed from eval set)
   - Human review queue panel: detections below threshold routed to a visible review list instead of discarded
   - "Export findings (GeoJSON)" button producing detections with lat/long properties formatted for ArcGIS feature-service ingestion
6. **Deployment**: Hugging Face Spaces, free CPU tier (INT8 detector + FP32 classifier serve inference).
7. **README**: architecture diagram, metrics table, per-class error analysis, honest scope framing, reproduction steps.

## Dataset

**InsPLAD** — Inspection of Power Line Assets Dataset.
- Repo: https://github.com/andreluizbvs/InsPLAD
- Paper: arXiv 2311.01619
- 10,607 high-res UAV images, 17 asset classes. Five asset types carry six defect labels: four corrosion variants, one broken component, one bird's nest.
- Published baselines for context: 0.721 Box AP (DetectoRS, detection); 0.954 balanced accuracy (EfficientNet, defect classification). Cite these in the README so results have a reference point.

**Resolved (verified against the InsPLAD repo, 2026-07-31):**
- The Google Drive folder linked from the repo README is dead (404, verified 2026-07-31). Working source: the authors' official Mendeley Data deposit (https://data.mendeley.com/datasets/5n3fjgvfyz/1), a single 6.4 GB zip with a direct, scriptable download URL (see data/README.md). License: CC BY-NC 3.0, so the license research task is resolved: attribution + non-commercial, fine for this demo; cite the paper.
- InsPLAD ships as two disconnected components: **InsPLAD-det** (full images, bounding boxes for 17 asset classes, no condition labels on boxes) and **InsPLAD-fault** (pre-cropped component images labeled by condition, no boxes, no link back to source images). There is no per-box condition annotation anywhere.
- **DECIDED: two-stage architecture.** Stage 1: YOLO11-small detects the 17 asset classes (trained on InsPLAD-det). Stage 2: a small classifier labels crops of the five defect-prone asset types (trained on InsPLAD-fault). Single-stage combined asset-condition classes would require manually joining the two datasets and is out of scope for a weekend. Two-stage also mirrors how the published baselines are structured (detection AP on -det, balanced accuracy on -fault), so results stay directly comparable. Framing in README: "detect components and classify condition," NOT "localize defects" (condition labels are image-level on crops, not pixel-level).

**Resolved from the downloaded data (verified on disk, 2026-07-31):**
- Detection annotations are **COCO JSON** (`annotations/instances_train.json` / `instances_val.json`), bbox = [x, y, w, h] absolute pixels. prep_insplad.py converts to YOLO txt.
- The train JSON contains **46 duplicate image entries** (7,981 entries, 7,935 unique filenames; all files present on disk). prep must dedupe by file_name and remap image ids before conversion.
- Splits on disk: train 7,981 images / 22,635 instances; val 2,626 / 6,324 (total 10,607, matches the paper). **No test split ships with the dataset**; prep must carve the held-out test set itself (from train, fixed seed) since val doubles as calibration pool.
- The JSON defines **18 categories, not 17**. Excluding `sphere` (26 instances) the instance total is exactly 28,933, the published number, so the paper's 17-class set excludes `sphere`. DECIDED: train on the 17 paper classes (drop `sphere`) so results stay comparable to the published baselines; note this in the README.
- Heavy class imbalance: stockbridge damper 6,953 instances down to polymer insulator tower shackle 57. Feeds the per-class error analysis.
- InsPLAD-fault (`defect_supervised/`) is folder-per-class inside per-asset train/val dirs: glass-insulator {good, missing-cap}, lightning-rod-suspension {good, rust}, polymer-insulator-upper-shackle {normal, corrosão in train but good, rust in val}, vari-grip {good, rust, bird-nest}, yoke-suspension {good, rust}.
- Label names are **inconsistent across splits** (corrosão vs rust, normal vs good); prep must normalize them.
- Fault val splits are skewed (e.g. yoke-suspension val: 5,742 good vs 20 rust), which is why balanced accuracy is the right metric. Classifier head decision: single backbone with combined asset-condition classes (~11), masking logits to the detected asset's valid conditions at inference; one ONNX file keeps the Space simple.

## Training

- Detector base: `yolo11s.pt` (COCO pretrained), Ultralytics package. Image size 640, ~50 epochs, early stop on plateau. Default augmentation is fine; do not over-engineer.
- Classifier: small ImageNet-pretrained backbone (EfficientNet-B0 or MobileNetV3-small) fine-tuned on InsPLAD-fault crops, ~20 epochs. Decide single multi-class head vs one head per defect-prone asset type during prep, based on how the fault data is organized on disk; document the choice in the README.
- Hardware: Vast.ai RTX 3090 (owner's standard lane, $0.20–0.35/hr). Write the training script so it runs headless: single entry point, resumable, checkpoints + metrics saved to a mounted/synced dir. Assume the instance can die.
- Split strategy (revised for baseline comparability): the dataset's **official val split (2,626 images) becomes the held-out test set**, untouched until benchmark.py, so mAP is measured on the same images the paper benchmarked. Our own val (~10% of train, seed 42) is carved from train and used for early stopping and as the INT8 calibration pool. Verify which metric the paper's 0.721 AP is (mAP50 vs mAP50-95) and report both.

## Model quality plan (accuracy push)

The deployment story stays unchanged (YOLO11-s INT8 serves the Space). On top of it, a deliberate accuracy push, in ROI order:

1. **Resolution**: the biggest lever. Small rare classes (shackles, plates) are tens of pixels at 640. Train/eval at 1280 after the 640 baseline.
2. **Model scale as a second claim**: also train YOLO11-m (or -l) at 1280 as the accuracy-ceiling run. README table then shows "best model trained" vs "model deployed and why," a deliberate latency-vs-accuracy trade.
3. **Longer training**: 100–150 epochs with early stopping, cosine LR, `close_mosaic` for the final ~10 epochs, modest mixup.
4. **Imbalance**: oversample training images containing rare classes (epoch image list weighted by rarest class present). Show per-class AP before/after in the error analysis.
5. **Classifier**: balanced sampling per batch, full fine-tune at low LR, EfficientNetV2-class small backbone. Target: match or beat 0.954 balanced accuracy.
6. **Skip** (poor ROI or undermines the deployable story): ensembles, WBF, TTA headline numbers, architecture surgery, pseudo-labeling the unsupervised zip.

Run order: s@640 baseline first (end-to-end pipeline proof), then s@1280, then m@1280. Budget rises to roughly $4–6 GPU spend; training runs unattended overnight.

## Quantization (critical path — do this correctly)

- **Scope: detector only.** The classifier ships as FP32 ONNX; it is tiny, and quantizing it would complicate the comparison story for negligible gain. State this in the README.
- Export FP32 ONNX via Ultralytics export.
- INT8 via ONNX Runtime **static** quantization.
- Calibration: 500–1000 images sampled (shuffled) from the validation set — in-domain calibration is what preserves accuracy. Do not use random/synthetic calibration data.
- Verify INT8 output sanity: run both models on the same batch, compare detections qualitatively before trusting metrics.
- Research task: check current ONNX Runtime quantization API (quantize_static, CalibrationDataReader) for the installed version; APIs have shifted across versions.

## Benchmarks

Measure on CPU (matches Spaces free tier deployment):
- Detector latency: p50 and p95 over ≥100 single-image inferences, after warmup, fixed input size. Record CPU model in README.
- Pipeline latency: detector + classifier over the image's crops, reported separately so the FP32-vs-INT8 table stays a clean detector-only comparison.
- Detector accuracy: mAP50 and mAP50-95 on the held-out test set for both FP32 and INT8 at the same confidence/NMS settings, plus per-class AP.
- Classifier accuracy: balanced accuracy on the InsPLAD-fault test split (directly comparable to the published 0.954 EfficientNet baseline).
- Model size on disk (both models; FP32 vs INT8 for the detector).

Emit the table as markdown ready to paste into README.

## Gradio App

- INT8 ONNX Runtime session for inference (no Ultralytics runtime dependency in the app if avoidable; keep the Space lightweight).
- Threshold slider (0.05–0.95) gates on **detection confidence** (stage 1). Classifier output is displayed on each detection but does not drive the gate; one threshold, one clear story. Slider re-filters the current image's detections client-side (no re-inference) and displays precision/recall at that threshold from a precomputed PR table (generated during eval, shipped as JSON with the app). App copy must state that the P/R numbers are dataset-level (held-out test set), not per-image.
- Review queue: detections with conf below slider value listed in a side panel with crop thumbnails and scores, labeled "Routed to human review." This is the human-in-the-loop story; make it visually obvious.
- GeoJSON export: FeatureCollection of Point features; properties = {asset_class, condition, confidence, image_id, timestamp}. Mock coordinates: scatter along a real transmission corridor east of Tucson, AZ (research task: pick plausible lat/longs along an actual line visible on OpenStreetMap; findings pinned to real-looking corridor). Note clearly in README/app that coordinates are simulated for demo purposes.
- Include 4–6 preloaded example images (from test set) so reviewers can try it without hunting for UAV imagery.

## Deployment

- Hugging Face Spaces, Gradio SDK, free CPU tier.
- Research task: current Spaces file size limits and whether the INT8 model needs Git LFS.
- App must cold-start reliably; lazy-load nothing critical.

## README Structure

1. One-paragraph pitch + link to live demo
2. Architecture diagram (owner has a d2-diagrams skill; generate a D2 diagram of the pipeline: UAV imagery → detector → condition classes → threshold gate → review queue / auto-accept → GeoJSON → GIS)
3. Results table (FP32 vs INT8) with hardware noted
4. Per-class error analysis: which condition classes fail at which thresholds, hypotheses why (e.g., corrosion variants confuse each other; small distant components)
5. The threshold tradeoff explained in business terms (miss a corroded insulator vs. flood reviewers with false alarms)
6. Honest scope section: demo-scale replica; image-level condition labels, not pixel-level defect localization; simulated coordinates; published-baseline context
7. Reproduction: exact commands, seed, environment

## Working Agreements

- Owner's standing preferences: no overclaiming anywhere; every claim survives hostile questioning; no em dashes in written prose; natural voice in README, not AI-sounding marketing language.
- Plan before building: Claude Code should start by (1) resolving the research tasks above, (2) proposing the class-set decision and repo layout, (3) confirming the split strategy — then build.
- Suggested repo layout:

```
powerline-inspection-demo/
├── data/                    # gitignored; see data/README.md for download steps
├── scripts/
│   ├── prep_insplad.py      # verify download, convert to YOLO format, splits
│   ├── train_detector.py    # headless, resumable (runs on Vast.ai)
│   ├── train_classifier.py  # InsPLAD-fault crops
│   ├── export_quantize.py   # ONNX FP32 + INT8 static quant w/ calibration
│   ├── benchmark.py         # latency + accuracy table generator
│   └── make_pr_table.py     # precision/recall vs threshold JSON for the app
├── app/
│   ├── app.py               # Gradio
│   ├── requirements.txt     # lean deps for the Space
│   ├── pr_table.json        # generated
│   ├── examples/            # 4–6 preloaded test-set images
│   └── models/              # detector INT8 + classifier ONNX (generated)
├── diagrams/
├── docs/                    # this spec, overview, learning resources
├── requirements.txt         # full dev/training environment
└── README.md
```

## Timeline (weekend)

- Sat AM: dataset research + prep script + launch training on Vast.ai
- Sat PM: eval, export, quantize, benchmark table
- Sun AM: Gradio app (slider, PR readout, review queue, GeoJSON)
- Sun PM: Spaces deploy, README, diagram, error analysis
- Stretch: per-class analysis depth, corridor map screenshot in README

## Success Criteria

- Live public Spaces URL with working slider, review queue, and GeoJSON export
- Comparison table with measured (not estimated) numbers
- README an interviewer can read in 3 minutes and a skeptic can reproduce
- Total GPU spend under $5
