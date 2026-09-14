from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.houb_label_freezer import build_snapshot, compute_snapshot


def _palette_image(labels: np.ndarray) -> Image.Image:
    image = Image.new("P", (labels.shape[1], labels.shape[0]))
    image.putdata(labels.astype(np.uint8).ravel().tolist())
    image.putpalette([0] * 768)
    return image


def _package(tmp_path: Path, values: list[np.ndarray]) -> Path:
    root = tmp_path / "package"
    (root / "schemas").mkdir(parents=True)
    schema = {
        "classes": [
            {"symbol": "B", "value": 0},
            {"symbol": "H", "value": 1},
            {"symbol": "O", "value": 2},
            {"symbol": "U", "value": 3},
        ]
    }
    schema_bytes = (json.dumps(schema, sort_keys=True) + "\n").encode()
    (root / "schemas/labels.json").write_bytes(schema_bytes)
    frames = []
    for index, labels in enumerate(values):
        relative = f"human_labels/palette_png/s/{index:05d}.png"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = _palette_image(labels)
        image.save(destination)
        frames.append(
            {
                "benchmark_index": index,
                "session_id": "s",
                "frame_index": index,
                "width": labels.shape[1],
                "height": labels.shape[0],
                "expected_human_label_path": relative,
            }
        )
    manifest = {
        "annotation_gate": {"required_human_labels": len(frames)},
        "frames": frames,
        "label_schema_ref": {
            "bytes": len(schema_bytes),
            "path": "schemas/labels.json",
            "sha256": hashlib.sha256(schema_bytes).hexdigest(),
        },
    }
    (root / "BENCHMARK_MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_binary_derivative_is_exhaustive_mutually_exclusive_and_repeatable(
    tmp_path: Path,
) -> None:
    labels = np.asarray([[0, 1, 2], [3, 1, 0]], dtype=np.uint8)
    package = _package(tmp_path, [labels])
    manifest, target, manifest_sha = build_snapshot(package, tmp_path / "derived")
    second_manifest, second_target, second_sha = build_snapshot(
        package, tmp_path / "derived"
    )

    assert manifest == second_manifest
    assert target == second_target
    assert manifest_sha == second_sha
    assert manifest["freeze_ref_issued"] is False
    assert manifest["status"] == "DERIVED_T0_NOT_FROZEN"
    masks = []
    for symbol in ("H", "O", "U", "B"):
        path = target / manifest["frames"][0]["masks"][symbol]["path"]
        mask = np.asarray(Image.open(path), dtype=np.uint8)
        assert set(np.unique(mask)).issubset({0, 1})
        masks.append(mask)
    assert np.all(sum(masks) == 1)
    assert np.array_equal(masks[0], labels == 1)
    assert not (target / "FREEZE_REF.json").exists()


def test_source_change_creates_a_new_content_addressed_snapshot(tmp_path: Path) -> None:
    package = _package(tmp_path, [np.asarray([[0, 1], [2, 3]], dtype=np.uint8)])
    _, first_target, _ = build_snapshot(package, tmp_path / "derived")
    label_path = package / "human_labels/palette_png/s/00000.png"
    changed = _palette_image(np.asarray([[1, 1], [2, 3]], dtype=np.uint8))
    changed.save(label_path)
    _, second_target, _ = build_snapshot(package, tmp_path / "derived")
    assert first_target != second_target
    assert first_target.exists() and second_target.exists()


def test_invalid_palette_value_fails_without_writing(tmp_path: Path) -> None:
    package = _package(tmp_path, [np.asarray([[0, 1], [2, 3]], dtype=np.uint8)])
    label_path = package / "human_labels/palette_png/s/00000.png"
    invalid = _palette_image(np.asarray([[0, 1], [2, 255]], dtype=np.uint8))
    invalid.save(label_path)
    with pytest.raises(ValueError, match="invalid palette values"):
        compute_snapshot(package)
    assert not (tmp_path / "derived").exists()
