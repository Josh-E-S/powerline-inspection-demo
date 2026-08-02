"""Fine-tune YOLO11 on InsPLAD-det (17 asset classes).

Headless and resumable: if the run directory already holds a last.pt, the
run continues from it automatically (safe on a Vast.ai instance that can
die mid-run). Checkpoints and metrics land in runs/detect/<name>/.

Run prep first: python3 scripts/prep_insplad.py

Examples:
  python3 scripts/train_detector.py                          # s @ 640 baseline
  python3 scripts/train_detector.py --imgsz 1280             # resolution push
  python3 scripts/train_detector.py --model yolo11m.pt --imgsz 1280
  python3 scripts/train_detector.py --imgsz 1280 --oversample

The test split (data/yolo test key) is never touched here; benchmark.py
owns it.
"""

import argparse
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "yolo"
SEED = 42


def build_oversampled_yaml(max_repeat: int = 4) -> Path:
    """Write a train list where images holding rare classes repeat.

    Repeat factor: sqrt(median_class_freq / rarest_class_in_image), capped.
    Backgrounds and common-class images stay at 1x.
    """
    label_dir = DATA / "labels" / "train"
    img_dir = DATA / "images" / "train"
    image_classes = {}
    freq = Counter()
    for txt in sorted(label_dir.glob("*.txt")):
        classes = {
            int(line.split()[0])
            for line in txt.read_text().splitlines()
            if line.strip()
        }
        image_classes[txt.stem] = classes
        freq.update(classes)
    median = sorted(freq.values())[len(freq) // 2]

    lines = []
    for stem, classes in sorted(image_classes.items()):
        matches = sorted(img_dir.glob(f"{stem}.*"))
        if not matches:
            continue
        repeat = 1
        if classes:
            rarest = min(freq[c] for c in classes)
            repeat = min(max_repeat, max(1, round((median / rarest) ** 0.5)))
        # keep the images/train symlink path (do NOT resolve): Ultralytics
        # finds labels by substituting /images/ -> /labels/ in each path
        lines.extend([str(matches[0])] * repeat)

    (DATA / "train_oversampled.txt").write_text("\n".join(lines) + "\n")

    base = (DATA / "insplad.yaml").read_text().replace(
        "train: images/train", "train: train_oversampled.txt"
    )
    out = DATA / "insplad_oversampled.yaml"
    out.write_text(base)
    print(f"Oversampled train list: {len(lines)} entries "
          f"(from {len(image_classes)} images)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=float, default=-1,
                    help="-1 = auto-fit to GPU memory")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--optimizer", default="auto",
                    help="auto picks by estimated iteration count, so "
                         "changing --epochs can silently swap optimizers; "
                         "pin it (e.g. AdamW, SGD) to keep runs comparable")
    ap.add_argument("--oversample", action="store_true",
                    help="repeat rare-class images in the train list")
    ap.add_argument("--name", default=None,
                    help="run name; default derived from model/imgsz/flags")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing last.pt and start over")
    ap.add_argument("--runs-dir", default=str(REPO / "runs"),
                    help="checkpoint/metrics root; point at a persistent "
                         "mount (e.g. Google Drive) on ephemeral instances")
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline check: 3 epochs on 5%% of train")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 3

    if not (DATA / "insplad.yaml").exists():
        raise SystemExit("data/yolo/insplad.yaml missing; "
                         "run scripts/prep_insplad.py first.")

    data_yaml = build_oversampled_yaml() if args.oversample \
        else DATA / "insplad.yaml"

    name = args.name or "det_{}_{}{}{}".format(
        Path(args.model).stem, args.imgsz,
        "_os" if args.oversample else "", "_smoke" if args.smoke else ""
    )
    runs_dir = Path(args.runs_dir)
    last = runs_dir / "detect" / name / "weights" / "last.pt"
    resume = last.exists() and not args.fresh

    from ultralytics import YOLO

    model = YOLO(str(last) if resume else args.model)
    if resume:
        print(f"Resuming from {last}")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        optimizer=args.optimizer,
        fraction=0.05 if args.smoke else 1.0,
        seed=SEED,
        cos_lr=True,
        close_mosaic=10,
        project=str(runs_dir / "detect"),
        name=name,
        exist_ok=True,
        resume=resume,
    )
    print(f"Done. Best weights: {runs_dir}/detect/{name}/weights/best.pt")


if __name__ == "__main__":
    main()
