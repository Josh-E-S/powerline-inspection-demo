"""Gradio demo: powerline inspection pipeline.

Image upload -> INT8 detector -> condition classifier on crops ->
confidence-threshold gate -> auto-accepted findings + human review queue ->
GeoJSON export (simulated coordinates, labeled as such).

Serves inference with ONNX Runtime only (models in app/models/); reads
dataset-level P/R from pr_table.json. Deployed to Hugging Face Spaces,
free CPU tier.

Status: stub. Requirements in docs/powerline-inspection-demo-spec.md.
"""
