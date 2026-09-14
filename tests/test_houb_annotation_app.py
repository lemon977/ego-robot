from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tools.houb_annotation_app import LABEL_COLORS, labels_from_editor


def _one_pixel_layer(rgb: np.ndarray) -> Image.Image:
    rgba = np.zeros((1, 1, 4), dtype=np.uint8)
    rgba[0, 0, :3] = rgb.astype(np.uint8)
    rgba[0, 0, 3] = 255
    return Image.fromarray(rgba)


def _one_premultiplied_pixel_layer(rgb: np.ndarray, alpha: int = 191) -> Image.Image:
    rgba = np.zeros((1, 1, 4), dtype=np.uint8)
    rgba[0, 0, :3] = np.rint(rgb.astype(np.float32) * alpha / 255.0).astype(np.uint8)
    rgba[0, 0, 3] = alpha
    return Image.fromarray(rgba)


@pytest.mark.parametrize("expected", [0, 1, 2, 3])
def test_each_fixed_brush_color_preserves_semantic_index(expected: int) -> None:
    labels = labels_from_editor(
        {"layers": [_one_pixel_layer(LABEL_COLORS[expected])]},
        (1, 1),
    )
    assert labels.tolist() == [[expected]]


@pytest.mark.parametrize("expected", [0, 1, 2, 3])
def test_each_premultiplied_gradio_brush_preserves_semantic_index(expected: int) -> None:
    labels = labels_from_editor(
        {"layers": [_one_premultiplied_pixel_layer(LABEL_COLORS[expected])]},
        (1, 1),
    )
    assert labels.tolist() == [[expected]]


@pytest.mark.parametrize(
    ("rgba", "expected"),
    [
        ([191, 239, 48, 239], 3),
        ([239, 203, 0, 239], 3),
        ([191, 251, 60, 251], 3),
    ],
)
def test_gradio_repaint_blends_map_to_last_dominant_brush(
    rgba: list[int], expected: int
) -> None:
    layer = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(1, 1, 4))
    labels = labels_from_editor({"layers": [layer]}, (1, 1))
    assert labels.tolist() == [[expected]]


def test_worst_opposite_class_repaint_still_maps_to_dominant_last_brush() -> None:
    # 75%-opacity H painted over an opaque O pixel.
    layer = Image.fromarray(np.asarray([64, 207, 191, 255], dtype=np.uint8).reshape(1, 1, 4))
    labels = labels_from_editor({"layers": [layer]}, (1, 1))
    assert labels.tolist() == [[1]]


def test_transparent_pixels_are_background() -> None:
    layer = Image.fromarray(np.zeros((2, 3, 4), dtype=np.uint8))
    labels = labels_from_editor({"layers": [layer]}, (3, 2))
    assert labels.tolist() == [[0, 0, 0], [0, 0, 0]]


def test_later_layer_wins_without_changing_other_pixels() -> None:
    first = np.zeros((2, 2, 4), dtype=np.uint8)
    first[..., :3] = LABEL_COLORS[1].astype(np.uint8)
    first[..., 3] = 255
    second = np.zeros((2, 2, 4), dtype=np.uint8)
    second[0, 1, :3] = LABEL_COLORS[2].astype(np.uint8)
    second[0, 1, 3] = 255
    labels = labels_from_editor(
        {"layers": [Image.fromarray(first), Image.fromarray(second)]},
        (2, 2),
    )
    assert labels.tolist() == [[1, 2], [1, 1]]


@pytest.mark.parametrize(
    ("rgba", "expected"),
    [
        ([112, 255, 141, 255], 1),
        ([141, 255, 112, 255], 3),
    ],
)
def test_reloaded_layer_high_distance_blends_do_not_block_save(
    rgba: list[int], expected: int
) -> None:
    layer = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(1, 1, 4))
    labels = labels_from_editor({"layers": [layer]}, (1, 1))
    assert labels.tolist() == [[expected]]
