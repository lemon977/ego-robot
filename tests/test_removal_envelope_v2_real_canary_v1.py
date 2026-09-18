from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from chaoyang.ops.run_0915_removal_envelope_v2_real_canary_v1 import (
    BITORDER,
    FRAME_COUNT,
    HEIGHT,
    WIDTH,
    aggregate_real_canary_quality,
    unpack_packed_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def _rows(**overrides: float | int | None) -> list[dict]:
    rows = []
    for frame in range(FRAME_COUNT):
        row = {
            "area_inflation": 1.10,
            "background_spill_ratio": 0.01,
            "repair_contribution_ratio": 0.10,
            "temporal_area_derivative": None if frame == 0 else 0.05,
            "semantic_temporal_area_derivative": None if frame == 0 else 0.04,
            "protected_object_core_damage_pixels": 0,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def test_packed_reader_materializes_exactly_one_frame() -> None:
    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[10:30, 40:80] = True
    packed = np.zeros((2, (HEIGHT * WIDTH + 7) // 8), np.uint8)
    packed[1] = np.packbits(mask.reshape(-1), bitorder=BITORDER)
    restored = unpack_packed_frame(packed, 1)
    assert restored.shape == (HEIGHT, WIDTH)
    assert np.array_equal(restored, mask)


def test_aggregate_passes_bounded_repairs_and_no_worse_flicker() -> None:
    result = aggregate_real_canary_quality(
        _rows(), stable_background_fraction=0.20,
    )
    assert result["status"] == "PASS"
    assert all(result["gates"].values())


def test_aggregate_rejects_repair_majority_even_if_other_metrics_pass() -> None:
    result = aggregate_real_canary_quality(
        _rows(repair_contribution_ratio=0.60), stable_background_fraction=0.20,
    )
    assert result["status"] == "REJECTED_QUALITY"
    assert result["metrics"]["repair_majority_frames"] == FRAME_COUNT
    assert result["gates"]["repair_never_majority"] is False


def test_aggregate_rejects_flicker_worse_than_semantic_baseline() -> None:
    result = aggregate_real_canary_quality(
        _rows(temporal_area_derivative=0.20, semantic_temporal_area_derivative=0.05),
        stable_background_fraction=0.20,
    )
    assert result["status"] == "REJECTED_QUALITY"
    assert result["gates"]["temporal_flicker_no_worse_than_semantic_plus_0_05"] is False


def test_runner_does_not_call_sam_or_inpaint_or_gpu() -> None:
    path = ROOT / "src/chaoyang/ops/run_0915_removal_envelope_v2_real_canary_v1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("sam3" in name for name in imports)
    assert not any("inpaint" in name for name in imports)
    assert "cuda" not in source.lower()
    assert "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP" in source

