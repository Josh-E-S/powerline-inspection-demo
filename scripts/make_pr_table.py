"""Precompute precision/recall vs detection-confidence threshold.

Runs the deployed INT8 detector over the held-out test set once (via
Ultralytics val) and extracts the P and R curves vs confidence, sampled
at the app slider's thresholds. Ships as app/pr_table.json so the app
can show dataset-level P/R for any slider position without re-inference.

Run after export_quantize.py: python3 scripts/make_pr_table.py
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "app" / "models" / "detector_int8.onnx"
OUT = REPO / "app" / "pr_table.json"
IMGSZ = 640


def main():
    from ultralytics import YOLO

    m = YOLO(str(MODEL), task="detect")
    r = m.val(
        data=str(REPO / "data" / "yolo" / "insplad.yaml"),
        split="test",
        imgsz=IMGSZ,
        device="cpu",
        verbose=False,
        project=str(REPO / "bench"),
        name="pr_table",
        exist_ok=True,
    )

    # curves_results: [[x, y, xlabel, ylabel], ...] for F1, P, R vs conf;
    # y has shape (n_classes, len(x)). Macro-average across classes.
    curves = {
        c[3]: (np.asarray(c[0]), np.asarray(c[1]))
        for c in r.box.curves_results
    }
    (px, p_y), (_, r_y) = curves["Precision"], curves["Recall"]
    p_macro, r_macro = p_y.mean(0), r_y.mean(0)

    thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    table = {
        "meta": {
            "model": MODEL.name,
            "split": "held-out test (official InsPLAD val, 2626 images)",
            "imgsz": IMGSZ,
            "note": "macro-averaged over 17 classes; dataset-level, "
            "not per-image",
        },
        "thresholds": thresholds,
        "precision": [
            round(float(np.interp(t, px, p_macro)), 4) for t in thresholds
        ],
        "recall": [
            round(float(np.interp(t, px, r_macro)), 4) for t in thresholds
        ],
    }
    OUT.write_text(json.dumps(table, indent=2) + "\n")
    print(f"Wrote {OUT}")
    for t, p, rr in zip(
        table["thresholds"], table["precision"], table["recall"], strict=True
    ):
        print(f"  conf {t:.2f}: P {p:.3f}  R {rr:.3f}")


if __name__ == "__main__":
    main()
