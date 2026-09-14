import json
from pathlib import Path

import numpy as np
import pytest

from tools import verify_hawor_environment_migration as migration


def test_float_comparison_records_bit_and_frozen_numeric_results() -> None:
    baseline = np.asarray([0.0, 1.0, np.nan], dtype=np.float32)
    candidate = baseline.copy()
    candidate[1] += np.float32(5e-7)
    result = migration._array_comparison(baseline, candidate)
    assert result["status"] == "PASS"
    assert result["bitwise_equal"] is False
    assert result["numeric_equal"] is True
    assert result["nan_positions_exact"] is True
    assert 0.0 < result["max_abs_error"] <= migration.FLOAT_ATOL


def test_discrete_comparison_requires_bitwise_exactness() -> None:
    baseline = np.asarray([True, False])
    candidate = np.asarray([True, True])
    result = migration._array_comparison(baseline, candidate)
    assert result["status"] == "FAIL"
    assert result["bitwise_equal"] is False
    assert result["numeric_equal"] is False


def test_npz_comparison_checks_keys_shapes_dtypes_and_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(migration, "PROJECT", tmp_path)
    baseline = tmp_path / "baseline.npz"
    equal = tmp_path / "equal.npz"
    changed = tmp_path / "changed.npz"
    np.savez_compressed(
        baseline,
        floating=np.asarray([1.0, np.nan], dtype=np.float32),
        discrete=np.asarray([1, 2], dtype=np.int32),
    )
    np.savez_compressed(
        equal,
        floating=np.asarray([1.0, np.nan], dtype=np.float32),
        discrete=np.asarray([1, 2], dtype=np.int32),
    )
    np.savez_compressed(
        changed,
        floating=np.asarray([1.0, np.nan], dtype=np.float32),
        discrete=np.asarray([1, 3], dtype=np.int32),
    )

    equal_result = migration.compare_npz(baseline, equal)
    changed_result = migration.compare_npz(baseline, changed)
    assert equal_result["status"] == "PASS"
    assert all(row["bitwise_equal"] for row in equal_result["arrays"].values())
    assert changed_result["status"] == "FAIL"
    assert "ARRAY_MISMATCH:discrete" in changed_result["failures"]


def test_semantic_result_ignores_paths_and_timing_but_not_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(migration, "PROJECT", tmp_path)
    semantic = {key: None for key in migration.SEMANTIC_RESULT_KEYS}
    semantic.update(
        schema_version="v1",
        status="PASS",
        task_id="chips",
        session_id="demo",
        target_interval=[1, 37],
        context_interval=[0, 37],
        gates={"numeric": "PASS"},
    )
    baseline = {**semantic, "outputs": {"npz": "/historical/path"}, "elapsed": 1.0}
    candidate = {**semantic, "outputs": {"npz": "/new/path"}, "elapsed": 9.0}
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    assert migration.compare_result(baseline_path, candidate_path)["status"] == "PASS"

    candidate["gates"] = {"numeric": "FAIL"}
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    result = migration.compare_result(baseline_path, candidate_path)
    assert result["status"] == "FAIL"
    assert any("$.gates.numeric" in failure for failure in result["failures"])


def test_evidence_writer_is_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(migration, "PROJECT", tmp_path)
    output = tmp_path / "evidence.json"
    migration.write_json_no_clobber(output, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        migration.write_json_no_clobber(output, {"status": "REPLACED"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "PASS"}


def test_checked_in_authority_is_promoted_and_has_three_frozen_roles() -> None:
    authority = json.loads(migration.AUTHORITY.read_text(encoding="utf-8"))
    assert authority["canonical_runtime_authorized"] is True
    assert authority["status"] == "PASS_LOCAL_ENVIRONMENT_PROMOTED"
    lock = json.loads(
        (migration.PROJECT / authority["local_environment_lock"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert lock["status"] == "PASS_LOCAL_ENVIRONMENT_PROMOTED"
    assert lock["authority"]["sha256"] == migration.sha256(migration.AUTHORITY)
    assert (
        lock["promotion_transaction"]["pre_promotion_authority_sha256"]
        == authority["promotion_evidence"]["pre_promotion_authority_sha256"]
    )
    assert tuple(
        row["role"] for row in authority["replay_authority"]["rows"]
    ) == migration.REQUIRED_ROLES
    for row in authority["replay_authority"]["rows"]:
        for artifact in ("npz", "overlay", "result"):
            path = migration.PROJECT / row[f"baseline_{artifact}"]
            assert migration.sha256(path) == row[f"baseline_{artifact}_sha256"]
    assert authority["historical_records"]["runtime_pin"]["immutable"] is True
