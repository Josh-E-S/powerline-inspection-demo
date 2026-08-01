"""Precompute precision/recall vs detection-confidence threshold.

Evaluates the INT8 detector on the held-out test set across thresholds
(0.05-0.95) and writes app/pr_table.json, which the Gradio app reads to
display dataset-level P/R as the slider moves (no re-inference).

Status: stub. Requirements in docs/powerline-inspection-demo-spec.md.
"""
