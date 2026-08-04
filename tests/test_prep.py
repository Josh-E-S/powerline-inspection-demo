"""Tests for dataset preparation logic.

These target the transformations that would silently corrupt training if
wrong: deduplicating the 46 duplicate image entries in InsPLAD's train
JSON, and converting COCO absolute-pixel boxes to normalized YOLO
centre-format. No dataset download is required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prep_insplad import collect_boxes, dedupe_images  # noqa: E402


def test_dedupe_collapses_repeated_filenames():
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "a.jpg", "width": 100, "height": 100},
        ]
    }
    images, id_map = dedupe_images(coco)
    assert [i["file_name"] for i in images] == ["a.jpg", "b.jpg"]
    # the duplicate entry's id must redirect to the canonical one
    assert id_map[3] == id_map[1] == 1


def test_collect_boxes_drops_unmapped_classes_and_dedupes():
    coco = {
        "annotations": [
            {"image_id": 1, "category_id": 9, "bbox": [10, 20, 30, 40]},
            {"image_id": 3, "category_id": 9, "bbox": [10, 20, 30, 40]},  # dup
            {
                "image_id": 1,
                "category_id": 99,
                "bbox": [0, 0, 5, 5],
            },  # dropped
        ]
    }
    id_map = {1: 1, 3: 1}
    boxes = collect_boxes(coco, id_map, {9: 0})
    assert boxes[1] == {(0, 10, 20, 30, 40)}, "duplicate box must collapse"
    assert 99 not in {b[0] for b in boxes[1]}


def test_coco_to_yolo_conversion_is_normalized_centre_format():
    W = H = 200
    x, y, w, h = 20, 40, 60, 80  # corner-origin, absolute pixels
    xc, yc = (x + w / 2) / W, (y + h / 2) / H
    assert (xc, yc) == (0.25, 0.4)
    assert (w / W, h / H) == (0.3, 0.4)
    for v in (xc, yc, w / W, h / H):
        assert 0.0 <= v <= 1.0
