from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from chaoyang.pipeline.stereo_encoded_domain_preflight_v1 import (
    EncodedStereoGateV1,
    aggregate_metrics,
    foundation_disparity_adapter,
    frame_metrics,
    split_source_index_eyes,
)


def test_negative_disparity_uses_reflection_without_swapping_physical_eyes():
    aggregate = {
        "decision": "PASS_DIRECT_FOUNDATION_INPUT",
        "metrics": {"dominant_disparity_sign": "NEGATIVE_LEFT_MINUS_RIGHT"},
    }
    adapter = foundation_disparity_adapter(aggregate)
    assert adapter["authorized"] is True
    assert adapter["horizontal_reflection_for_disparity_sign"] is True
    assert adapter["swap_physical_eyes"] is False
    assert adapter["output_spatial_unflip_required"] is True


ROOT = Path(__file__).resolve().parents[1]


def test_source_index_split_returns_physical_left_then_right() -> None:
    source_zero = np.full((3, 4, 3), 10, np.uint8)
    source_one = np.full((3, 4, 3), 20, np.uint8)
    sbs = np.hstack((source_zero, source_one))
    left, right = split_source_index_eyes(
        sbs, eye_width=4, eye_height=3,
        physical_left_source_index=1, physical_right_source_index=0,
    )
    assert np.all(left == 20)
    assert np.all(right == 10)


def _rows(
    *, vertical_px: float, disparity_px: float, frames: int = 150,
) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    records: list[dict] = []
    vertical: list[np.ndarray] = []
    disparity: list[np.ndarray] = []
    for frame in range(frames):
        x = np.linspace(50.0, 1200.0, 20)
        y = np.linspace(30.0, 900.0, 20)
        left = np.stack((x, y), axis=1)
        right = np.stack((x - disparity_px, y + vertical_px), axis=1)
        records.append({
            "frame_index": frame,
            **frame_metrics(left, right, width=1280, height=960),
        })
        vertical.append(np.abs(left[:, 1] - right[:, 1]))
        disparity.append(left[:, 0] - right[:, 0])
    return records, vertical, disparity


def test_direct_encoded_input_passes_when_all_gates_close() -> None:
    rows, vertical, disparity = _rows(vertical_px=1.5, disparity_px=16.0)
    result = aggregate_metrics(rows, vertical, disparity)
    assert result["decision"] == "PASS_DIRECT_FOUNDATION_INPUT"
    assert result["gpu_successor_authorized"] is True
    assert result["metrics"]["dominant_disparity_sign"] == "POSITIVE_LEFT_MINUS_RIGHT"


def test_vertical_failure_requires_encoded_epipolar_alignment() -> None:
    rows, vertical, disparity = _rows(vertical_px=7.0, disparity_px=16.0)
    result = aggregate_metrics(rows, vertical, disparity)
    assert result["decision"] == "NEEDS_ENCODED_EPIPOLAR_ALIGNMENT"
    assert result["gpu_successor_authorized"] is False


def test_insufficient_matches_rejects_instead_of_authorizing_gpu() -> None:
    gate = EncodedStereoGateV1(minimum_total_robust_matches=10)
    rows = [{
        "frame_index": frame,
        "robust_matches": 0,
        "spatial_grid_coverage": 0.0,
    } for frame in range(150)]
    empty = [np.empty(0, np.float64) for _ in rows]
    result = aggregate_metrics(rows, empty, empty, gate=gate)
    assert result["decision"] == "REJECTED_QUALITY_INSUFFICIENT_CORRESPONDENCE_EVIDENCE"


def test_runner_has_mandatory_frames_and_no_geometric_pixel_transform() -> None:
    path = ROOT / "src/chaoyang/ops/run_0915_stereo_encoded_domain_preflight_v1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "cv2.remap" not in called
    assert not any("undistort" in name.lower() for name in called)
    assert "NOTABLE_FRAMES = (81, 94)" in source
    assert "split_resize_physical_eyes" in source
