import json
from pathlib import Path

import pytest

from chaoyang.ops import run_visual_aux_candidate_bundle_watcher_v71 as watcher


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_validate_bundle_accepts_only_causal_any_endpoint(tmp_path: Path) -> None:
    bundle = tmp_path / "session"
    write_json(
        bundle / "RESULT.json",
        {
            "status": "PASS_DEVELOPMENT_VISUAL_AUX_BUNDLE_NO_OCCLUSION_AUTHORITY",
            "eligible_h50_windows": 23,
            "rgb_valid_pixel_fraction": 0.9,
        },
    )
    write_json(
        bundle / "VISUAL_AUX_SESSION_MANIFEST.json",
        {
            "input_mode": "CAUSAL_TRAINING_INPUT",
            "causal_proof": {"enabled": True},
            "h50_eligibility_mode": "ANY_ENDPOINT_40_OF_50",
        },
    )
    result = watcher.validate_bundle(bundle)
    assert result is not None
    assert result["status"] == "READY"
    assert result["eligible_h50_windows"] == 23


def test_validate_bundle_rejects_offline_bidirectional_input(tmp_path: Path) -> None:
    bundle = tmp_path / "session"
    write_json(
        bundle / "RESULT.json",
        {
            "status": "PASS_DEVELOPMENT_VISUAL_AUX_BUNDLE_NO_OCCLUSION_AUTHORITY",
            "eligible_h50_windows": 1,
            "rgb_valid_pixel_fraction": 1.0,
        },
    )
    write_json(
        bundle / "VISUAL_AUX_SESSION_MANIFEST.json",
        {
            "input_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION",
            "causal_proof": {"enabled": True},
            "h50_eligibility_mode": "ANY_ENDPOINT_40_OF_50",
        },
    )
    with pytest.raises(RuntimeError, match="non-causal"):
        watcher.validate_bundle(bundle)


def test_build_one_closes_missing_clean_as_blocked(tmp_path: Path) -> None:
    candidate = tmp_path / "candidates" / "s1" / "RESULT.json"
    write_json(candidate, {"session": "s1"})
    row = {"session_id": "s1", "task": "chips", "split": "train", "clean_state": "FAILED_QUALITY_C"}
    result = watcher.build_one(candidate, row, tmp_path / "bundles", tmp_path / "logs", 1)
    assert result["status"] == "BLOCKED_PREREQ_CLEAN"


def test_matrix_rows_rejects_duplicate_session(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    write_json(matrix, {"rows": [{"session_id": "s1"}, {"session_id": "s1"}]})
    with pytest.raises(RuntimeError, match="duplicate"):
        watcher.matrix_rows(matrix)
