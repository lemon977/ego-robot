from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chaoyang.ops.run_0915_robot15h_virtual_arm_wave0_v1 import (
    COLLISION_COVERAGE,
    EXPECTED_SESSIONS,
    PINNED_SHA256,
    VirtualR2Error,
    load_r0_arrays,
    load_virtual_installation_contract,
    published_ref,
    relative_hand_root_targets,
    sha256,
    timestamp_derivatives,
    validate_r0_batch,
    verify_static_assets,
)


def _proper(tx: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = tx
    return value


def _write_r0(path: Path, *, filled_invalid: bool = False) -> np.ndarray:
    q22 = np.full((4, 2, 22), np.nan, dtype=np.float64)
    wrist = np.full((4, 2, 4, 4), np.nan, dtype=np.float64)
    valid = np.zeros((4, 2), dtype=bool)
    valid[[0, 1, 3], 0] = True
    q22[valid] = np.arange(int(valid.sum()) * 22, dtype=np.float64).reshape(-1, 22) / 100.0
    wrist[0, 0] = _proper(0.0)
    wrist[1, 0] = _proper(0.01)
    wrist[3, 0] = _proper(0.03)
    if filled_invalid:
        q22[2, 0] = 0.0
        wrist[2, 0] = np.eye(4)
    np.savez_compressed(
        path,
        q22_init=q22,
        relative_wrist_T=wrist,
        valid_side_frame=valid,
        timestamps_s=np.asarray([0.0, 0.1, 0.2, 0.4]),
        frame_id=np.arange(4),
        ignored_upstream_field=np.asarray([123]),
    )
    return q22


def test_load_r0_reads_only_authorized_fields_and_preserves_q22_bits(tmp_path: Path) -> None:
    source = tmp_path / "r0.npz"
    q22 = _write_r0(source)
    arrays = load_r0_arrays(source)
    assert set(arrays) == {
        "q22_init", "relative_wrist_T", "valid_side_frame", "timestamps_s", "frame_id"
    }
    assert np.array_equal(arrays["q22_init"], q22, equal_nan=True)
    assert np.isnan(arrays["q22_init"][~arrays["valid_side_frame"]]).all()


def test_load_r0_rejects_filled_invalid_rows(tmp_path: Path) -> None:
    source = tmp_path / "r0-filled.npz"
    _write_r0(source, filled_invalid=True)
    with pytest.raises(VirtualR2Error, match="filled instead of NaN"):
        load_r0_arrays(source)


def test_load_r0_rejects_dtype_coercion(tmp_path: Path) -> None:
    source = tmp_path / "r0-wrong-dtype.npz"
    _write_r0(source)
    with np.load(source, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["frame_id"] = values["frame_id"].astype(np.int32)
    np.savez_compressed(source, **values)
    with pytest.raises(VirtualR2Error, match="dtype mismatch for frame_id"):
        load_r0_arrays(source)


def test_relative_targets_apply_exact_unit_gain_and_keep_invalid_nan() -> None:
    relative = np.full((3, 2, 4, 4), np.nan)
    valid = np.zeros((3, 2), dtype=bool)
    relative[0, 0] = _proper(0.0)
    relative[1, 0] = _proper(0.125)
    valid[:2, 0] = True
    neutral = np.stack((_proper(0.4), _proper(-0.4)))
    target = relative_hand_root_targets(relative, valid, neutral)
    assert target[0, 0, 0, 3] == pytest.approx(0.4)
    assert target[1, 0, 0, 3] == pytest.approx(0.525)
    assert np.isnan(target[2, 0]).all()
    assert np.isnan(target[:, 1]).all()


def test_timestamp_derivatives_use_real_dt_and_do_not_bridge_gap() -> None:
    q = np.full((4, 2, 1), np.nan)
    valid = np.zeros((4, 2), dtype=bool)
    q[0, 0, 0], q[1, 0, 0], q[3, 0, 0] = 0.0, 0.2, 9.0
    valid[[0, 1, 3], 0] = True
    result = timestamp_derivatives(q, np.asarray([0.0, 0.1, 0.2, 0.4]), valid)
    assert result["velocity_rad_s_max"] == pytest.approx(2.0)
    assert result["velocity_sample_count"] == 1
    assert result["acceleration_sample_count"] == 0


def test_static_virtual_assets_are_exactly_pinned_and_proxy_is_not_calibration() -> None:
    refs, mounts = verify_static_assets()
    assert refs["mount_visual_proxy_contract"]["sha256"] == PINNED_SHA256["mount_visual_proxy_contract"]
    assert mounts.shape == (2, 4, 4)
    assert np.allclose(np.linalg.det(mounts[:, :3, :3]), 1.0)


def test_virtual_installation_contract_authorizes_only_bounded_virtual_use() -> None:
    contract, _ = load_virtual_installation_contract()
    assert contract["calibration"]["measured_installation_transform"] == "ABSENT"
    assert contract["motion_authorization"]["r0_relative_virtual_ik_allowed"] is True
    assert contract["motion_authorization"]["hardware_motion_allowed"] is False
    assert contract["collision_authorization"]["coverage"] == COLLISION_COVERAGE
    assert contract["collision_authorization"]["object_collision"] == "UNVERIFIED"
    assert contract["output_authority"]["physical_deployment_authorized"] is False


def test_r0_batch_requires_results_key_and_exact_unique_sessions() -> None:
    rows = [
        {
            "session_id": session_id,
            "task": task,
            "status": "COMPLETED_DEVELOPMENT_BASELINE",
            "r0_exported": True,
            "r0_quality_admitted": False,
            "result": {"path": "x", "bytes": 1, "sha256": "0" * 64},
        }
        for session_id, task in EXPECTED_SESSIONS.items()
    ]
    batch = {
        "schema_version": "0915-robot15h-kai22-r0-wave0-batch-v1",
        "status": "COMPLETED_ALL_TERMINAL",
        "window_run_id": "robot15h-0915-20260919T000959+0800",
        "session_count": 4,
        "r0_exported": 4,
        "failed_runtime": 0,
        "results": rows,
    }
    assert validate_r0_batch(batch) == rows
    invalid = {**batch, "sessions": rows}
    invalid.pop("results")
    with pytest.raises(VirtualR2Error, match="exact results row key"):
        validate_r0_batch(invalid)
    duplicate = {**batch, "results": [rows[0], rows[0], rows[2], rows[3]]}
    with pytest.raises(VirtualR2Error, match="identity/uniqueness"):
        validate_r0_batch(duplicate)


def test_runner_has_no_dynamic_hawor_depth_object_or_archive_binding() -> None:
    module_path = Path(__file__).parents[1] / "src/chaoyang/ops/run_0915_robot15h_virtual_arm_wave0_v1.py"
    text = module_path.read_text(encoding="utf-8")
    # Mentions in the policy/docstring are fine; actual path constants/imports are not.
    forbidden = (
        "HAWOR_ROOT =",
        "FOUNDATION_ROOT =",
        "SAM_ROOT =",
        "OBJECT6D_ROOT =",
        "archive/baseline-",
        "/mnt/data/egodata",
    )
    assert not any(value in text for value in forbidden)
    assert sha256(module_path)


def test_published_ref_binds_staging_bytes_to_final_path(tmp_path: Path) -> None:
    staging = tmp_path / ".session.staging" / "state.npz"
    staging.parent.mkdir()
    staging.write_bytes(b"r2-state")
    published = tmp_path / "session" / "state.npz"
    value = published_ref(staging, published)
    assert value["path"] == str(published.resolve())
    assert ".staging" not in value["path"]
    assert value["bytes"] == len(b"r2-state")
    assert value["sha256"] == sha256(staging)
