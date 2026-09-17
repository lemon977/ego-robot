from __future__ import annotations

import numpy as np

from chaoyang.ops.audit_rc1_pico_c2w_causality import inspect_frame, nominal_grid_delta_ms, pose_tracking_rows


def test_past_sample_and_closed_metadata_are_distinct_checks() -> None:
    pose = np.eye(4)
    row = inspect_frame(
        metadata={"idx": 2, "ts": 100_000_000, "tracking_index": 5,
                  "tracking_sync_error_ms": -7.0, "c2w": pose.tolist()},
        tracking={"timeStampNs": 95_000_000, "TrackerState": "notAccurate", "Head": {"status": 3}}, c2w=pose,
    )
    assert row["indexed_tracking_is_past"] is True
    assert row["indexed_tracking_minus_frame_ms"] == -5.0
    assert row["metadata_timestamp_equals_indexed_tracking"] is False
    assert row["stored_sync_error_minus_indexed_delta_ms"] == -2.0
    assert row["tracker_state"] == "notAccurate"
    assert row["head_status"] == 3
    assert row["metadata_vs_archive_c2w_max_abs"] == 0.0


def test_stored_sync_error_does_not_hide_future_sample() -> None:
    row = inspect_frame(
        metadata={"idx": 3, "ts": 100_000_000, "tracking_index": 6,
                  "tracking_sync_error_ms": -7.0, "c2w": np.eye(4).tolist()},
        tracking={"timeStampNs": 105_000_000}, c2w=np.eye(4),
    )
    assert row["indexed_tracking_is_past"] is False
    assert row["indexed_tracking_minus_frame_ms"] == 5.0


def test_tracking_index_excludes_merge_header() -> None:
    physical = [
        {"timeStampNs": 100, "_merge": {"sourceSessions": ["A", "B"]}},
        {"timeStampNs": 101, "Head": {"pose": "0,0,0,0,0,0,1"}},
        {"timeStampNs": 102, "Head": {"pose": "0,0,0,0,0,0,1"}},
    ]
    rows, skipped = pose_tracking_rows(physical)
    assert skipped == 1
    assert rows[0]["timeStampNs"] == 101
    assert rows[1]["timeStampNs"] == 102


def test_sync_error_can_be_nominal_grid_drift_not_sensor_skew() -> None:
    start = 1_000_000_000
    actual_tracker_ns = start + 2 * 1e9 / 30 + 3_101_000
    assert abs(nominal_grid_delta_ms(int(actual_tracker_ns), start, 2, 30) - 3.101) < 1e-5
