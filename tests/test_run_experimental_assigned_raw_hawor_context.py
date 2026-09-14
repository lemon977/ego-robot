from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

from tools import run_experimental_assigned_raw_hawor_context as candidate


def _raw(frame_count: int) -> dict[str, np.ndarray]:
    joints_2d = np.empty((2, frame_count, 21, 2), dtype=np.float32)
    joints_3d = np.empty((2, frame_count, 21, 3), dtype=np.float32)
    for slot in range(2):
        joints_2d[slot].fill(10.0 + 10.0 * slot)
        joints_3d[slot].fill(1.0 + slot)
    return {
        "joints_2d": joints_2d,
        "joints_3d_camera": joints_3d,
        "observed": np.ones((2, frame_count), dtype=np.bool_),
        "final_valid": np.ones((2, frame_count), dtype=np.bool_),
        "focal_length_px": np.asarray(640.0, dtype=np.float32),
        "principal_point": np.asarray([640.0, 480.0], dtype=np.float32),
        "image_size": np.asarray([1280, 960], dtype=np.int32),
    }


def _mapping(frame_count: int) -> dict[str, np.ndarray]:
    return {
        "frame_index": np.arange(frame_count, dtype=np.int32),
        "frame_assignment_status_code": np.ones(frame_count, dtype=np.uint8),
        "state_code": np.zeros(frame_count, dtype=np.int8),
        "physical_to_raw_slots": np.repeat(
            np.asarray([[0, 1]], dtype=np.int8), frame_count, axis=0
        ),
        "raw_axis_observed": np.ones((frame_count, 2), dtype=np.bool_),
        "raw_axis_final_valid": np.ones((frame_count, 2), dtype=np.bool_),
        "raw_axis_input_kind_code": np.ones((frame_count, 2), dtype=np.uint8),
        "assignment_source_code": np.ones(frame_count, dtype=np.uint8),
    }


def test_invalid_mapping_stays_missing_without_fallback() -> None:
    raw = _raw(2)
    raw["observed"][:, 1] = False
    raw["final_valid"][:, 1] = False
    mapping = _mapping(2)
    mapping["frame_assignment_status_code"][1] = 0
    mapping["physical_to_raw_slots"][1] = (-1, -1)
    mapping["raw_axis_observed"][1] = False
    mapping["raw_axis_final_valid"][1] = False
    mapping["raw_axis_input_kind_code"][1] = 0
    mapping["assignment_source_code"][1] = 0
    result = candidate.assign_physical_axes(raw, mapping, frame_count=2)
    assert not result["valid"][:, 1].any()
    assert np.isnan(result["joints_2d"][:, 1]).all()
    assert np.isnan(result["joints_3d_camera"][:, 1]).all()


def test_filled_raw_is_consumed_and_disclosed() -> None:
    raw = _raw(1)
    raw["observed"][1, 0] = False
    mapping = _mapping(1)
    mapping["raw_axis_observed"][0, 1] = False
    mapping["raw_axis_input_kind_code"][0, 1] = 2
    mapping["assignment_source_code"][0] = 2
    result = candidate.assign_physical_axes(raw, mapping, frame_count=1)
    assert result["valid"][:, 0].all()
    assert result["assignment_source_code"].tolist() == [2]
    assert result["raw_axis_input_kind_code"][:, 0].tolist() == [1, 2]
    assert result["raw_observed"][:, 0].tolist() == [True, False]


def test_swap_mapping_changes_physical_axis_without_aliasing() -> None:
    raw = _raw(1)
    mapping = _mapping(1)
    mapping["physical_to_raw_slots"][0] = (1, 0)
    result = candidate.assign_physical_axes(raw, mapping, frame_count=1)
    assert np.all(result["joints_2d"][0, 0] == 20.0)
    assert np.all(result["joints_2d"][1, 0] == 10.0)
    assert result["physical_to_raw_slots"].tolist() == [[1, 0]]


def test_assigned_slot_out_of_bounds_fails_closed() -> None:
    mapping = _mapping(1)
    mapping["physical_to_raw_slots"][0] = (0, 2)
    with pytest.raises(candidate.AssignedRawContextError, match="out of bounds"):
        candidate.assign_physical_axes(_raw(1), mapping, frame_count=1)


def test_cross_session_mapping_raw_binding_is_rejected(tmp_path: Path) -> None:
    raw_path = tmp_path / "wrong.npz"
    raw_path.write_bytes(b"wrong-session")
    payload = raw_path.read_bytes()
    arrays = {
        "session_ids": np.asarray(["grap_a_cap_005"]),
        "session_frame_counts": np.asarray([1], dtype=np.int32),
        "session_row_offsets": np.asarray([0, 1], dtype=np.int64),
        "literal_raw_path": np.asarray([str(tmp_path / "expected.npz")]),
        "literal_raw_bytes": np.asarray([12], dtype=np.int64),
        "literal_raw_sha256": np.asarray(["a" * 64]),
        **_mapping(1),
    }
    with pytest.raises(
        candidate.AssignedRawContextError, match="cross-session raw/mapping"
    ):
        candidate.session_mapping_slice(
            arrays,
            session_id="grap_a_cap_005",
            frame_count=1,
            raw_ref={
                "path": str(raw_path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )


def test_forbidden_and_noneligible_sessions_are_rejected() -> None:
    for session_id in ("grap_a_cap_025", "grap_a_cap_012", "../grap_a_cap_005"):
        with pytest.raises(candidate.base.ContextError, match="literal eligible68"):
            candidate.require_context_session(session_id)


def test_authority_payload_round_trip_keeps_experimental_gate() -> None:
    arrays = candidate.assign_physical_axes(_raw(1), _mapping(1), frame_count=1)
    payload = candidate._authority_payload("grap_a_cap_005", arrays)
    summary = candidate.validate_assigned_authority(
        payload, session_id="grap_a_cap_005", frame_count=1
    )
    assert summary == {
        "frames": 1,
        "assigned": 1,
        "unassigned": 0,
        "both_observed": 1,
        "uses_filled_raw": 0,
    }
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert str(archive["authority_mode"]) == candidate.AUTHORITY_MODE
        assert bool(archive["experimental_only"])
        assert not bool(archive["optimizer_rerun"])
        assert not bool(archive["selector_recomputed"])
        np.testing.assert_array_equal(archive["thresholds"], [0.20, 0.05, 0.12])


def test_real_mapping_authority_fully_decodes_and_validates() -> None:
    payload = candidate.MAPPING_PATH.read_bytes()
    assert len(payload) == candidate.MAPPING_BYTES
    assert hashlib.sha256(payload).hexdigest() == candidate.MAPPING_SHA256
    summary = candidate.validate_mapping_authority(
        candidate._npz_from_payload(payload, label="test-v9")
    )
    assert summary == {
        "sessions": 68,
        "frames": 28_265,
        "assigned": 28_135,
        "unassigned": 130,
        "both_observed": 22_422,
        "uses_filled_raw": 5_713,
    }
