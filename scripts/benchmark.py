"""Compare the FP32 and INT8 detectors on the held-out test set.

Measures accuracy, single-image latency, and file size for both variants,
plus balanced accuracy for the condition classifier. Everything runs on
CPU because that is what the deployment target uses.

This is the only script that reads the test split. Training and
quantization calibration both use a separate validation set.

Results are written to bench/results.json, with a markdown summary in
bench/table.md.

    python scripts/benchmark.py --imgsz 640
"""

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "app" / "models"
TEST_IMAGES = REPO / "data" / "yolo" / "images" / "test"
FAULT_TEST = REPO / "data" / "fault" / "test"
BENCH = REPO / "bench"
SEED = 42


def cpu_name() -> str:
    """Identify the CPU so reported latencies mean something."""
    if platform.system() == "Darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def letterbox(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    r = min(size / img.width, size / img.height)
    nw, nh = round(img.width * r), round(img.height * r)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(
        img.resize((nw, nh), Image.BILINEAR),
        ((size - nw) // 2, (size - nh) // 2),
    )
    return np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0


def detector_accuracy(onnx_path: Path, imgsz: int) -> dict:
    """Run Ultralytics validation over the test split for one ONNX model."""
    from ultralytics import YOLO

    m = YOLO(str(onnx_path), task="detect")
    r = m.val(
        data=str(REPO / "data" / "yolo" / "insplad.yaml"),
        split="test",
        imgsz=imgsz,
        device="cpu",
        verbose=False,
        project=str(BENCH),
        name=f"val_{onnx_path.stem}",
        exist_ok=True,
    )
    idx = r.box.ap_class_index
    per_class_ap50 = {
        r.names[i]: round(float(ap), 4)
        for i, ap in zip(idx, r.box.ap50, strict=True)
    }
    # Both per-class metrics are kept because they are easy to confuse.
    # The published per-class table uses Box AP at IoU 0.50:0.95, and
    # comparing our lenient AP50 against it would inflate the result.
    per_class_ap = {
        r.names[i]: round(float(v), 4)
        for i, v in zip(idx, r.box.maps[idx], strict=True)
    }
    return {
        "mAP50": round(float(r.box.map50), 4),
        "mAP50_95": round(float(r.box.map), 4),
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap,
    }


def latency(onnx_path: Path, files, size: int, runs: int, warmup: int = 10):
    """Time single-image inference, reporting the median and 95th percentile."""
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    name = sess.get_inputs()[0].name
    batches = [letterbox(f, size) for f in files[:24]]
    for i in range(warmup):
        sess.run(None, {name: batches[i % len(batches)]})
    times = []
    for i in range(runs):
        x = batches[i % len(batches)]
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
    }


def classifier_metrics(runs: int) -> dict:
    """Score the classifier on the fault test split.

    Balanced accuracy is used because the split is severely skewed.
    """
    import onnxruntime as ort

    classes = json.loads((MODELS / "classifier_classes.json").read_text())
    sess = ort.InferenceSession(
        str(MODELS / "classifier.onnx"), providers=["CPUExecutionProvider"]
    )
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

    def prep(path):
        img = Image.open(path).convert("RGB")
        r = 256 / min(img.size)
        img = img.resize(
            (round(img.width * r), round(img.height * r)), Image.BILINEAR
        )
        left, top = (img.width - 224) // 2, (img.height - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        arr = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
        return ((arr - mean) / std)[None]

    correct = np.zeros(len(classes))
    total = np.zeros(len(classes))
    times = []
    for ci, cls in enumerate(classes):
        for f in sorted((FAULT_TEST / cls).iterdir()):
            x = prep(f)
            t0 = time.perf_counter()
            logits = sess.run(None, {"image": x})[0]
            times.append((time.perf_counter() - t0) * 1000)
            total[ci] += 1
            correct[ci] += int(logits.argmax()) == ci
    recalls = correct[total > 0] / total[total > 0]
    return {
        "balanced_accuracy": round(float(recalls.mean()), 4),
        "per_class_recall": {
            c: round(float(correct[i] / total[i]), 4)
            for i, c in enumerate(classes)
            if total[i]
        },
        "n_test": int(total.sum()),
        "latency_p50_ms": round(float(np.percentile(times, 50)), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="must match detector training/export imgsz",
    )
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--skip-classifier", action="store_true")
    args = ap.parse_args()

    BENCH.mkdir(exist_ok=True)
    fp32, int8 = MODELS / "detector_fp32.onnx", MODELS / "detector_int8.onnx"
    files = sorted(TEST_IMAGES.glob("*.*"))
    random.Random(SEED).shuffle(files)

    results = {"cpu": cpu_name(), "imgsz": args.imgsz, "runs": args.runs}
    for tag, path in (("fp32", fp32), ("int8", int8)):
        print(f"Evaluating {tag} ...")
        results[tag] = {
            "size_mb": round(path.stat().st_size / 1e6, 1),
            **detector_accuracy(path, args.imgsz),
            **latency(path, files, args.imgsz, args.runs),
        }
    if not args.skip_classifier:
        print("Evaluating classifier ...")
        results["classifier"] = classifier_metrics(args.runs)

    f32, i8 = results["fp32"], results["int8"]
    table = [
        f"Benchmarked on: {results['cpu']} (CPU), imgsz={args.imgsz}, "
        f"{args.runs} timed runs after warmup.",
        "",
        "| Model | Size (MB) | mAP50 | mAP50-95 | p50 latency (ms) | p95 latency (ms) |",
        "|---|---|---|---|---|---|",
        f"| FP32 ONNX | {f32['size_mb']} | {f32['mAP50']} | {f32['mAP50_95']} "
        f"| {f32['p50_ms']} | {f32['p95_ms']} |",
        f"| INT8 ONNX | {i8['size_mb']} | {i8['mAP50']} | {i8['mAP50_95']} "
        f"| {i8['p50_ms']} | {i8['p95_ms']} |",
    ]
    md = "\n".join(table)
    (BENCH / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (BENCH / "table.md").write_text(md + "\n")
    print("\n" + md)
    if "classifier" in results:
        c = results["classifier"]
        print(
            f"\nClassifier: balanced accuracy {c['balanced_accuracy']} "
            f"on {c['n_test']} test crops, p50 {c['latency_p50_ms']} ms/crop"
        )
    print("\nFull results: bench/results.json, table: bench/table.md")


if __name__ == "__main__":
    main()
