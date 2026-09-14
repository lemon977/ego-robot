from pathlib import Path

import pytest

from tools.resolve_legacy_path import resolve_legacy_path


def _registry():
    return {
        "schema_version": 1,
        "redirects": [
            {
                "old_path": "_run/example_v1",
                "new_path": "archive/legacy_runs/shared/example_v1",
                "tree_content_sha256": "a" * 64,
                "inode": 42,
                "apparent_bytes_lstat_sum": 17,
            }
        ],
    }


def test_resolves_descendant_and_reports_existence(tmp_path: Path):
    target = tmp_path / "archive/legacy_runs/shared/example_v1/result.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    result = resolve_legacy_path("_run/example_v1/result.json", _registry(), tmp_path)
    assert result.new_path == "archive/legacy_runs/shared/example_v1/result.json"
    assert result.exists is True
    assert result.recorded_tree_sha256 == "a" * 64


def test_accepts_workspace_absolute_old_path(tmp_path: Path):
    result = resolve_legacy_path(str(tmp_path / "_run/example_v1"), _registry(), tmp_path)
    assert result.matched_old_root == "_run/example_v1"
    assert result.new_path == "archive/legacy_runs/shared/example_v1"


def test_rejects_unknown_or_outside_path(tmp_path: Path):
    with pytest.raises(KeyError):
        resolve_legacy_path("_run/unknown", _registry(), tmp_path)
    with pytest.raises(ValueError):
        resolve_legacy_path("/outside/_run/example_v1", _registry(), tmp_path)
