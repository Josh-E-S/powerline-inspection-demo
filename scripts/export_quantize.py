"""Export models to ONNX and quantize the detector to INT8.

Detector: Ultralytics FP32 ONNX export, then ONNX Runtime static
quantization (QDQ, per-channel weights) calibrated on real validation
images. Includes a sanity check comparing FP32 vs INT8 raw outputs on
the same batch before any metrics are trusted.

Classifier: FP32 ONNX export only (quantization scope is detector-only,
per spec), plus a class-name JSON the app reads.

Outputs land in app/models/:
  detector_fp32.onnx, detector_int8.onnx,
  classifier.onnx, classifier_classes.json

Example:
  python3 scripts/export_quantize.py \
      --weights runs/detect/det_yolo11s_640/weights/best.pt \
      --classifier runs/classify/cls_effv2s/best.pt --imgsz 640
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "app" / "models"
CALIB_DIR = REPO / "data" / "yolo" / "images" / "val"
SEED = 42


def letterbox(path: Path, size: int) -> np.ndarray:
    """Ultralytics-style preprocexss: fit into size x size, pad with 114."""
    img = Image.open(path).convert("RGB")
    r = min(size / img.width, size / img.height)
    nw, nh = round(img.width * r), round(img.height * r)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR),
                 ((size - nw) // 2, (size - nh) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]  # 1x3xHxW


def export_detector(weights: str, imgsz: int) -> Path:
    from ultralytics import YOLO

    onnx_path = Path(YOLO(weights).export(
        format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
    dst = OUT / "detector_fp32.onnx"
    shutil.copy(onnx_path, dst)
    return dst


def quantize_detector(fp32_path: Path, imgsz: int, n_calib: int) -> Path:
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    files = sorted(CALIB_DIR.glob("*.*"))
    random.Random(SEED).shuffle(files)
    files = files[:n_calib]
    print(f"Calibrating on {len(files)} validation images")

    input_name = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(files)

        def get_next(self):
            f = next(self.it, None)
            return None if f is None else {input_name: letterbox(f, imgsz)}

    pre = fp32_path.with_name("detector_fp32_preproc.onnx")
    quant_pre_process(str(fp32_path), str(pre))

    int8_path = OUT / "detector_int8.onnx"
    quantize_static(
        model_input=str(pre),
        model_output=str(int8_path),
        calibration_data_reader=Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        # Conv/MatMul only: the detect head's output mixes pixel coords
        # (0-640) and class probs (0-1) in one tensor; quantizing that
        # tensor forces one shared scale and rounds every prob to zero.
        op_types_to_quantize=["Conv", "MatMul"],
    )
    pre.unlink()
    return int8_path


def sanity_check(fp32_path: Path, int8_path: Path, imgsz: int, n: int = 8):
    """FP32 vs INT8 raw outputs on the same images; abort-worthy if wild."""
    import onnxruntime as ort

    files = sorted(CALIB_DIR.glob("*.*"))[:n]
    s32 = ort.InferenceSession(str(fp32_path),
                               providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8_path),
                              providers=["CPUExecutionProvider"])
    name = s32.get_inputs()[0].name
    box_cors, prob_cors, det_counts = [], [], []
    for f in files:
        x = letterbox(f, imgsz)
        a = s32.run(None, {name: x})[0][0]  # (4+nc, anchors)
        b = s8.run(None, {name: x})[0][0]
        # coords and probs judged separately: coords (0-640) dominate any
        # whole-tensor statistic and can mask fully-crushed probabilities
        box_cors.append(float(np.corrcoef(a[:4].ravel(), b[:4].ravel())[0, 1]))
        prob_cors.append(
            float(np.corrcoef(a[4:].ravel(), b[4:].ravel())[0, 1]))
        det_counts.append((int((a[4:].max(0) > 0.25).sum()),
                           int((b[4:].max(0) > 0.25).sum())))
    print(f"Sanity ({len(files)} images): box corr={np.mean(box_cors):.4f}, "
          f"prob corr={np.mean(prob_cors):.4f}")
    print(f"  detections above 0.25 per image (fp32 vs int8): {det_counts}")
    if np.mean(prob_cors) < 0.95:
        print("WARNING: INT8 class probabilities diverge from FP32; do not "
              "trust metrics (check op exclusions / calibration).")


def export_classifier(ckpt_path: str):
    import torch
    from torchvision import models

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["arch"] == "efficientnet_v2_s", ckpt["arch"]
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dst = OUT / "classifier.onnx"
    torch.onnx.export(
        model, torch.zeros(1, 3, 224, 224), str(dst),
        input_names=["image"], output_names=["logits"], opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    (OUT / "classifier_classes.json").write_text(
        json.dumps(ckpt["classes"], indent=2) + "\n")
    print(f"Classifier -> {dst} ({len(ckpt['classes'])} classes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="detector best.pt")
    ap.add_argument("--classifier", help="classifier best.pt (optional)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="must match the detector's training imgsz")
    ap.add_argument("--calib-n", type=int, default=800)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fp32 = export_detector(args.weights, args.imgsz)
    int8 = quantize_detector(fp32, args.imgsz, args.calib_n)
    for p in (fp32, int8):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")
    sanity_check(fp32, int8, args.imgsz)
    if args.classifier:
        export_classifier(args.classifier)


if __name__ == "__main__":
    main()
