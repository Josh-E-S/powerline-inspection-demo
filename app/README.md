---
title: Powerline Inspection Demo
emoji: ⚡
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
short_description: UAV powerline inspection with human review and GeoJSON
---

# Powerline Inspection Demo

A working replica of an AI powerline inspection pipeline: drone imagery in,
human-reviewed findings out as GIS-ready data.

**Drone photo → component detection → condition classification → confidence
threshold gate → auto-accepted findings + human review queue → GeoJSON export.**

## Try it

1. Pick an example image (or upload your own UAV shot) and press **Analyze**
2. Drag the **confidence threshold** slider. Detections move between
   auto-accepted (green) and the human review queue (orange), and the
   precision/recall readout updates
3. Press **Export findings (GeoJSON)** and drop the file on
   [geojson.io](https://geojson.io) to see it on a map

The slider is the point of the demo. Set it high and you miss real defects;
set it low and you bury reviewers in false alarms. That trade-off is a
business decision, and this makes it something you can touch.

## What is running

Two ONNX Runtime models on free-tier CPU, no GPU:

| Stage | Model | Size |
|---|---|---|
| Detection (17 asset classes) | YOLO11-s, INT8 quantized | 9.6 MB |
| Condition classification | EfficientNetV2-S, FP32 | 78 MB |

Measured on the held-out test set (2,626 images, the InsPLAD benchmark
split): **0.889 mAP50 / 0.710 mAP50-95** for this deployed INT8 detector,
and **0.960 balanced accuracy** for the classifier.

The best model trained in this project scores **0.912 mAP50 / 0.738
mAP50-95**, exceeding the published InsPLAD baselines (DetectoRS: 0.885
AP50 / 0.721 AP). It runs at 1280px and costs about 4x the CPU latency, so
the smaller 640px model is served here instead. That trade is the whole
point: full results, per-class error analysis, and the two INT8
quantization failures found along the way are in the
[technical report](https://github.com/Josh-E-S/powerline-inspection-demo/blob/main/docs/TECHNICAL-REPORT.md).

## Honest scope

- Demo-scale replica, not a production system.
- Condition labels are image-level on cropped components. This detects
  components and classifies their condition; it does not localize defects
  at pixel level.
- **Map coordinates are simulated** along a real transmission corridor east
  of Tucson, AZ. No GPS or EXIF data is used. Every exported feature is
  labelled as simulated.
- Precision/recall shown by the slider is dataset-level (measured across the
  whole test set), not a per-image figure.

## Credits

Dataset: [InsPLAD](https://github.com/andreluizbvs/InsPLAD) by Vieira-e-Silva
et al., CC BY-NC 3.0
([paper](https://arxiv.org/abs/2311.01619)). Models here are derived from it
and are likewise for non-commercial use, with attribution to the dataset
authors.

Code and full write-up:
[github.com/Josh-E-S/powerline-inspection-demo](https://github.com/Josh-E-S/powerline-inspection-demo)
