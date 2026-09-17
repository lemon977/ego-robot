from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.run_occlusion_visible_surface_watcher_v71 import build_silver_report, ref, select_focus_frames


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_focus_selection_uses_maximum_sparse_robot_object_overlap(tmp_path: Path) -> None:
    frame_ids = np.asarray([0, 5, 9], dtype=np.int64)
    labels = np.full((3, 4, 4), -1, dtype=np.int32)
    labels[0, 0, 0] = 1
    labels[1, :2, :2] = 1
    labels[2, 0, :2] = 1
    archive = tmp_path / "zbuffer.npz"
    np.savez_compressed(archive, frame_ids=frame_ids, render_label=labels)
    z_result = tmp_path / "zresult.json"
    _write_json(z_result, {"outputs": [ref(archive)]})

    rows = []
    for frame in frame_ids:
        mask_path = tmp_path / f"mask_{frame}.png"
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :2] = 255
        assert cv2.imwrite(str(mask_path), mask)
        rows.append({
            "source_frame": int(frame),
            "physical_instances": {
                "0": {"valid": True, "mask": ref(mask_path)},
                "1": {"valid": False},
                "2": {"valid": False},
            },
        })
    manifest = tmp_path / "objects.json"
    _write_json(manifest, {"frames": rows})
    arm = tmp_path / "arm.npz"
    np.savez_compressed(arm, valid_side_frame=np.ones((2, 10), dtype=np.bool_))

    focus, selection = select_focus_frames(z_result, manifest, arm, 3)
    assert selection["best_sparse_frame"] == 5
    assert selection["best_sparse_overlap_pixels"] == 4
    assert focus == [4, 5, 6]


def test_terminal_visible_surface_index_does_not_claim_full_silver():
    report = build_silver_report({
        "s": {
            "session": "s",
            "status": "PASSED_DEVELOPMENT_VISIBLE_SURFACE_CANARY",
            "gates": {
                "known_coverage_70": True,
                "unknown_ratio_30": True,
                "conditional_retention_99": True,
            },
        }
    }, "2026-09-15T00:00:00+08:00")
    assert report["status"] == "FAILED_QUALITY_C"
    assert report["silver_authority"] is False
    assert report["accuracy_reported"] is False
    assert "full_session_temporal_consistency" in report["required_but_unclosed_gates"]
