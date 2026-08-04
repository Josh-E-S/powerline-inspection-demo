"""Tests for inference preprocessing.

Letterboxing must match what the model saw during training: aspect ratio
preserved, padded to a square with value 114, scaled to 0-1, channels
first. A silent mismatch here degrades accuracy without raising anything.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark import letterbox  # noqa: E402


def _write(tmp_path, w, h):
    p = tmp_path / f"{w}x{h}.jpg"
    Image.new("RGB", (w, h), (255, 0, 0)).save(p)
    return p


def test_letterbox_shape_and_range(tmp_path):
    x = letterbox(_write(tmp_path, 1920, 1080), 640)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_letterbox_pads_with_114_not_zero(tmp_path):
    # a wide image leaves horizontal bars top and bottom
    x = letterbox(_write(tmp_path, 1280, 320), 640)
    top_row = x[0, :, 0, 0]
    np.testing.assert_allclose(top_row, 114 / 255.0, atol=1e-6)


def test_letterbox_preserves_aspect_ratio(tmp_path):
    # 2:1 source in a square canvas: content occupies half the height
    x = letterbox(_write(tmp_path, 1280, 640), 640)
    red = x[0, 0] > 0.5  # the red fill, not the grey padding
    rows_with_content = red.any(axis=1).sum()
    assert 300 <= rows_with_content <= 340, rows_with_content


def test_square_input_fills_canvas(tmp_path):
    x = letterbox(_write(tmp_path, 800, 800), 640)
    red = x[0, 0] > 0.5
    assert red.all(), "a square image should need no padding"
