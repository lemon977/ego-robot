from __future__ import annotations

import json
from pathlib import Path

from chaoyang.ops import preflight_ai2_current_asset_paths_v31 as op
from chaoyang.ops import run_ai2_real_assets_cohort_audit_v31 as cohort


def _materialize_expected(root: Path) -> None:
    for spec in cohort.SESSION_SPECS:
        for path in cohort._paths(root, spec).values():
            if path is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"fixture": str(path)}), encoding="utf-8")


def test_preflight_binds_all_eight_current_assets_and_preserves_logical_task(
    tmp_path: Path,
) -> None:
    (tmp_path / "_run/current").mkdir(parents=True)
    _materialize_expected(tmp_path)

    result = op.build_preflight(root=tmp_path)

    assert result["status"] == "PASS_8_CURRENT_ASSET_BINDINGS"
    assert result["counts"] == {
        "total": 8,
        "path_complete": 8,
        "missing_or_rejected_files": 0,
    }
    assert result["task_to_asset_directory"] == {
        "poker": "playing_cards",
        "potato_chips": "potato_chips",
    }
    poker = [row for row in result["sessions"] if row["task"] == "poker"]
    assert len(poker) == 4
    assert all(row["asset_directory"] == "playing_cards" for row in poker)
    assert all(row["logical_task_preserved"] is True for row in poker)
    assert result["read_only"] is True
    assert result["model_calls"] == result["gpu_calls"] == 0


def test_preflight_fails_closed_for_one_missing_current_file(tmp_path: Path) -> None:
    (tmp_path / "_run/current").mkdir(parents=True)
    _materialize_expected(tmp_path)
    spec = next(row for row in cohort.SESSION_SPECS if row["task"] == "poker")
    missing = cohort._paths(tmp_path, spec)["hawor_npz"]
    assert missing is not None
    missing.unlink()

    result = op.build_preflight(root=tmp_path)

    assert result["status"] == "BLOCKED_CURRENT_ASSET_BINDINGS"
    assert result["counts"]["path_complete"] == 7
    assert result["missing"] == [
        {
            "session_id": spec["session_id"],
            "role": "hawor_npz",
            "reason": "MISSING_FILE",
        }
    ]
