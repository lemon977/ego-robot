from __future__ import annotations

import numpy as np

from chaoyang.ops.audit_hawor_temporal_prefix_causality import _apply, _prefix_track, compare_prefix


def _track(frames: int = 40) -> dict[str, np.ndarray]:
    eye = np.eye(3, dtype=np.float64)
    return {
        "root_translation_camera": np.zeros((2, frames, 3), dtype=np.float64),
        "root_orient_camera": np.broadcast_to(eye, (2, frames, 3, 3)).copy(),
        "hand_pose_rotmat": np.broadcast_to(eye, (2, frames, 15, 3, 3)).copy(),
        "betas": np.zeros((2, frames, 10), dtype=np.float64),
        "observed": np.ones((2, frames), dtype=bool),
        "detector_confidence": np.ones((2, frames), dtype=np.float64),
    }


def test_full_sequence_smoother_changes_prefix_when_suffix_changes() -> None:
    track = _track()
    track["root_translation_camera"][:, 20:, 2] = 0.05
    full = _apply(track, 1.0)
    prefix_track = _prefix_track(track, 20)
    prefix = _apply(prefix_track, 1.0)
    metrics = compare_prefix(full, prefix, track["observed"])
    assert metrics["prefix_invariant"] is False
    assert metrics["root_translation_terminal_mm"] > 0.0


def test_identical_prefix_only_rebuild_is_invariant() -> None:
    track = _track(20)
    first = _apply(track, 1.0)
    second = _apply(track, 1.0)
    metrics = compare_prefix(first, second, track["observed"])
    assert metrics["prefix_invariant"] is True


def test_missing_rows_with_nan_rotations_are_ignored() -> None:
    track = _track(20)
    track["observed"][:, 10] = False
    track["root_orient_camera"][:, 10] = np.nan
    track["hand_pose_rotmat"][:, 10] = np.nan
    first = _apply(track, 1.0)
    second = _apply(track, 1.0)
    metrics = compare_prefix(first, second, track["observed"])
    assert metrics["prefix_invariant"] is True
