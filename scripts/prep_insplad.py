"""Prepare InsPLAD for training.

Detection (data/raw -> data/yolo):
- Converts COCO JSON (annotations/instances_{train,val}.json) to YOLO txt.
- Dedupes the 46 duplicate image entries in the train JSON (by file_name),
  merging and deduping their annotations.
- Drops the `sphere` class (26 instances; the paper's 17-class set and its
  28,933-instance total exclude it) and remaps ids to 0..16.
- Splits: official val (2,626 images) becomes the held-out TEST set so
  results are comparable to the published baselines. Our own val (10% of
  train, seed 42) is carved from train for early stopping and INT8
  calibration.
- Images are symlinked, not copied. Emits data/yolo/insplad.yaml.

Fault classification (data/raw/defect_supervised -> data/fault):
- Normalizes inconsistent condition labels (corrosão -> rust, normal -> good).
- Combined asset__condition class folders (11 classes).
- Official val becomes TEST; our val is 10% of train per class, seed 42.
- Emits data/fault/classes.json.

Run from the repo root: python3 scripts/prep_insplad.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
OUT_DET = REPO / "data" / "yolo"
OUT_FAULT = REPO / "data" / "fault"

SEED = 42
VAL_FRAC = 0.10
DROP_CLASSES = {"sphere"}
COND_MAP = {"corrosão": "rust", "normal": "good"}


def link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def load_coco(path: Path):
    with open(path) as f:
        d = json.load(f)
    return d


def dedupe_images(coco):
    """Collapse duplicate image entries (same file_name) to one canonical id."""
    canon = {}          # file_name -> canonical image dict
    id_map = {}         # original image id -> canonical image id
    for img in coco["images"]:
        name = img["file_name"]
        if name in canon:
            id_map[img["id"]] = canon[name]["id"]
        else:
            canon[name] = img
            id_map[img["id"]] = img["id"]
    return list(canon.values()), id_map


def collect_boxes(coco, id_map, cat_to_yolo):
    """Group deduped, class-filtered boxes by canonical image id."""
    boxes = defaultdict(set)
    for ann in coco["annotations"]:
        if ann["category_id"] not in cat_to_yolo:
            continue
        img_id = id_map[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        boxes[img_id].add((cat_to_yolo[ann["category_id"]], x, y, w, h))
    return boxes


def write_split(images, boxes, img_dir, split):
    n_boxes = 0
    for img in images:
        W, H = img["width"], img["height"]
        lines = []
        for cls, x, y, w, h in sorted(boxes.get(img["id"], ())):
            xc = min(max((x + w / 2) / W, 0.0), 1.0)
            yc = min(max((y + h / 2) / H, 0.0), 1.0)
            lines.append(
                f"{cls} {xc:.6f} {yc:.6f} {min(w / W, 1.0):.6f} {min(h / H, 1.0):.6f}"
            )
        n_boxes += len(lines)
        stem = Path(img["file_name"]).stem
        label_path = OUT_DET / "labels" / split / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        link(OUT_DET / "images" / split / img["file_name"], img_dir / img["file_name"])
    return n_boxes


def prep_detection():
    train_coco = load_coco(RAW / "annotations" / "instances_train.json")
    test_coco = load_coco(RAW / "annotations" / "instances_val.json")

    cats = sorted(train_coco["categories"], key=lambda c: c["id"])
    kept = [c for c in cats if c["name"] not in DROP_CLASSES]
    cat_to_yolo = {c["id"]: i for i, c in enumerate(kept)}
    names = [c["name"] for c in kept]

    images, id_map = dedupe_images(train_coco)
    boxes = collect_boxes(train_coco, id_map, cat_to_yolo)

    rng = random.Random(SEED)
    images = sorted(images, key=lambda i: i["file_name"])
    rng.shuffle(images)
    n_val = round(len(images) * VAL_FRAC)
    splits = {"val": images[:n_val], "train": images[n_val:]}

    counts = {}
    for split, imgs in splits.items():
        counts[split] = (len(imgs), write_split(imgs, boxes, RAW / "train", split))

    test_images, test_id_map = dedupe_images(test_coco)
    test_boxes = collect_boxes(test_coco, test_id_map, cat_to_yolo)
    counts["test"] = (
        len(test_images),
        write_split(test_images, test_boxes, RAW / "val", "test"),
    )

    yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (OUT_DET / "insplad.yaml").write_text("\n".join(yaml_lines) + "\n")

    print("Detection (data/yolo):")
    for split in ("train", "val", "test"):
        n_img, n_box = counts[split]
        print(f"  {split:5s}: {n_img:5d} images, {n_box:6d} boxes")
    print(f"  classes: {len(names)}")
    return counts


def prep_fault():
    src = RAW / "defect_supervised"
    rng = random.Random(SEED)
    class_counts = defaultdict(lambda: defaultdict(int))

    for asset_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        asset = asset_dir.name
        for raw_split, out_of in (("train", ("train", "val")), ("val", ("test",))):
            for cond_dir in sorted(
                p for p in (asset_dir / raw_split).iterdir() if p.is_dir()
            ):
                cond = COND_MAP.get(cond_dir.name, cond_dir.name)
                cls = f"{asset}__{cond}"
                files = sorted(cond_dir.iterdir())
                if out_of == ("test",):
                    assigned = {"test": files}
                else:
                    rng.shuffle(files)
                    n_val = round(len(files) * VAL_FRAC)
                    assigned = {"val": files[:n_val], "train": files[n_val:]}
                for split, split_files in assigned.items():
                    for f in split_files:
                        link(OUT_FAULT / split / cls / f.name, f)
                    class_counts[cls][split] += len(split_files)

    classes = sorted(class_counts)
    (OUT_FAULT / "classes.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "condition_normalization": COND_MAP,
                "counts": {c: dict(class_counts[c]) for c in classes},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print("\nFault classification (data/fault):")
    for c in classes:
        s = class_counts[c]
        print(
            f"  {c:45s} train {s.get('train', 0):5d}  val {s.get('val', 0):4d}"
            f"  test {s.get('test', 0):5d}"
        )
    print(f"  classes: {len(classes)}")


def main():
    for req in (RAW / "annotations", RAW / "train", RAW / "val", RAW / "defect_supervised"):
        if not req.exists():
            sys.exit(f"Missing {req}. See data/README.md for download steps.")
    prep_detection()
    prep_fault()
    print("\nDone. Detector config: data/yolo/insplad.yaml")


if __name__ == "__main__":
    main()
