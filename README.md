# Powerline Inspection Demo

**Status: in progress. Nothing below is measured yet; placeholders are marked.**

A demo-scale replica of an AI powerline inspection pipeline: UAV imagery in,
human-reviewed findings out the other side as GIS-ready GeoJSON. Two-stage
model (YOLO11-small detector + condition classifier), quantized to run on a
free CPU tier, with every number measured and every trade-off made visible.

Live demo: (Spaces link goes here)

## Architecture

(D2 diagram goes here: UAV imagery -> detector -> condition classifier ->
threshold gate -> review queue / auto-accept -> GeoJSON -> GIS)

## Results

(Measured FP32 vs INT8 table goes here, with CPU model noted.)

## Per-class error analysis

(Goes here after benchmarking.)

## The threshold trade-off

(Business framing: a missed corroded insulator can mean an outage or a fire;
a too-low threshold floods reviewers with false alarms until they tune it
out. The slider in the demo makes that trade-off tangible.)

## Scope and honesty

- This is a demo-scale workflow replica, not a production system.
- Condition labels are image-level on cropped components, not pixel-level
  defect outlines. The pipeline detects components and classifies condition;
  it does not localize defects.
- Map coordinates in the GeoJSON export are simulated along a real
  transmission corridor and labeled as such.
- Published baselines for context: 0.721 box AP (DetectoRS, detection) and
  0.954 balanced accuracy (EfficientNet, defect classification) on InsPLAD
  ([paper](https://arxiv.org/abs/2311.01619)).

## Reproduction

(Exact commands, seed, pinned environment go here.)

## Project docs

- [Spec](docs/powerline-inspection-demo-spec.md)
- [Plain-English overview](docs/PROJECT_OVERVIEW.md)
- [Learning resources](docs/RESOURCES.md)
- [Data setup](data/README.md)
