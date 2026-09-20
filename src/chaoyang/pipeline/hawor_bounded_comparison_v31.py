"""Fair raw-versus-bounded HaWoR metrics on one frozen observable set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SCHEMA_VERSION = "HAWOR_BOUNDED_COMPARISON_V31"
MANO_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


class BoundedComparisonError(ValueError):
    """Raised when a before/after comparison would not be fair."""


@dataclass(frozen=True)
class AdoptionThresholds:
    minimum_relative_temporal_improvement: float = 0.05
    reprojection_p95_regression_px_max: float = 0.0
    bone_length_cv_regression_max: float = 0.0
    absolute_lag_frames_max: int = 1


def _p95(values: list[np.ndarray]) -> float | None:
    if not values:
        return None
    chosen = np.concatenate([np.asarray(value, np.float64).reshape(-1) for value in values])
    chosen = chosen[np.isfinite(chosen)]
    return float(np.percentile(chosen, 95)) if chosen.size else None


def _segments(mask: np.ndarray, frame_ids: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    splits = np.flatnonzero(np.diff(frame_ids[indices]) != 1) + 1
    return [part for part in np.split(indices, splits) if part.size]


def _motion_metrics(
    joints: np.ndarray, valid: np.ndarray, timestamps_s: np.ndarray, frame_ids: np.ndarray
) -> dict[str, float | None]:
    steps: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    jerks: list[np.ndarray] = []
    for ids in _segments(valid, frame_ids):
        values = joints[ids]
        times = timestamps_s[ids]
        if ids.size >= 2:
            dt = np.diff(times)
            if np.any(dt <= 0):
                raise BoundedComparisonError("timestamps must increase inside every segment")
            delta = np.diff(values, axis=0)
            steps.append(np.linalg.norm(delta, axis=-1) * 1000.0)
            velocity = delta / dt[:, None, None]
            speeds.append(np.linalg.norm(velocity, axis=-1))
        else:
            continue
        if ids.size >= 3:
            velocity_times = (times[:-1] + times[1:]) / 2.0
            acceleration = np.diff(velocity, axis=0) / np.diff(velocity_times)[:, None, None]
            accelerations.append(np.linalg.norm(acceleration, axis=-1))
        else:
            continue
        if ids.size >= 4:
            acceleration_times = (velocity_times[:-1] + velocity_times[1:]) / 2.0
            jerk = np.diff(acceleration, axis=0) / np.diff(acceleration_times)[:, None, None]
            jerks.append(np.linalg.norm(jerk, axis=-1))
    return {
        "wrist_step_p95_mm": _p95([value[:, 0] for value in steps]),
        "all_joint_speed_p95_m_per_s": _p95(speeds),
        "all_joint_acceleration_p95_m_per_s2": _p95(accelerations),
        "all_joint_jerk_p95_m_per_s3": _p95(jerks),
    }


def _bone_cv(joints: np.ndarray, selected: np.ndarray) -> float | None:
    if not selected.any():
        return None
    lengths = np.stack(
        [np.linalg.norm(joints[:, child] - joints[:, parent], axis=-1) for parent, child in MANO_EDGES],
        axis=1,
    )[selected]
    means = lengths.mean(axis=0)
    return float(np.max(lengths.std(axis=0) / np.maximum(means, 1e-12)))


def _reprojection_p95(values: np.ndarray, selected: np.ndarray) -> float | None:
    array = np.asarray(values, np.float64)
    if array.shape[0] != selected.shape[0]:
        raise BoundedComparisonError("reprojection time axis mismatch")
    chosen = array[selected]
    chosen = chosen[np.isfinite(chosen)]
    return float(np.percentile(chosen, 95)) if chosen.size else None


def _lag_frames(raw: np.ndarray, candidate: np.ndarray, selected: np.ndarray, maximum: int = 3) -> int | None:
    ids = np.flatnonzero(selected)
    if ids.size < 8:
        return None
    raw_centered = raw[ids, 0] - np.mean(raw[ids, 0], axis=0)
    candidate_centered = candidate[ids, 0] - np.mean(candidate[ids, 0], axis=0)
    rows: list[tuple[float, int]] = []
    for lag in range(-maximum, maximum + 1):
        if lag < 0:
            left, right = raw_centered[-lag:], candidate_centered[:lag]
        elif lag > 0:
            left, right = raw_centered[:-lag], candidate_centered[lag:]
        else:
            left, right = raw_centered, candidate_centered
        if len(left) < 4:
            continue
        rows.append((float(np.mean(np.square(left - right))), lag))
    return min(rows)[1] if rows else None


def compare_bounded_hawor(
    *,
    raw_joints: np.ndarray,
    candidate_joints: np.ndarray,
    raw_valid: np.ndarray,
    candidate_valid: np.ndarray,
    frozen_observable_mask: np.ndarray,
    timestamps_s: np.ndarray,
    frame_ids: np.ndarray,
    raw_reprojection_error_px: np.ndarray,
    candidate_reprojection_error_px: np.ndarray,
    raw_observed: np.ndarray,
    candidate_observed: np.ndarray,
    thresholds: AdoptionThresholds = AdoptionThresholds(),
) -> dict[str, Any]:
    """Compare tracks without allowing candidate-side frame deletion or invention."""

    raw = np.asarray(raw_joints, np.float64)
    candidate = np.asarray(candidate_joints, np.float64)
    if raw.shape != candidate.shape or raw.ndim != 3 or raw.shape[1:] != (21, 3):
        raise BoundedComparisonError("raw and candidate joints must share shape [T,21,3]")
    frame_count = raw.shape[0]
    masks = [
        np.asarray(value, bool)
        for value in (raw_valid, candidate_valid, frozen_observable_mask, raw_observed, candidate_observed)
    ]
    if any(value.shape != (frame_count,) for value in masks):
        raise BoundedComparisonError("all validity/observability arrays must have shape [T]")
    raw_valid_mask, candidate_valid_mask, frozen, raw_obs, candidate_obs = masks
    times = np.asarray(timestamps_s, np.float64)
    ids = np.asarray(frame_ids, np.int64)
    if times.shape != (frame_count,) or ids.shape != (frame_count,):
        raise BoundedComparisonError("timestamp/frame axes must have shape [T]")
    if np.any(np.diff(times) <= 0) or np.any(np.diff(ids) <= 0):
        raise BoundedComparisonError("timestamps and frame_ids must be strictly increasing")
    if not np.isfinite(raw[raw_valid_mask]).all() or not np.isfinite(candidate[candidate_valid_mask]).all():
        raise BoundedComparisonError("valid joints must be finite")

    frozen_raw = frozen & raw_valid_mask
    frozen_candidate = frozen_raw & candidate_valid_mask
    dropped = frozen_raw & ~candidate_valid_mask
    added_observed = candidate_obs & ~raw_obs
    dropped_observed = raw_obs & ~candidate_obs
    same_metric_set = bool(np.array_equal(frozen_raw, frozen_candidate))
    raw_metrics = _motion_metrics(raw, frozen_raw, times, ids)
    candidate_metrics = _motion_metrics(candidate, frozen_raw & candidate_valid_mask, times, ids)
    raw_metrics["bone_length_cv_max"] = _bone_cv(raw, frozen_raw)
    candidate_metrics["bone_length_cv_max"] = _bone_cv(candidate, frozen_raw & candidate_valid_mask)
    raw_metrics["reprojection_p95_px"] = _reprojection_p95(raw_reprojection_error_px, frozen_raw)
    candidate_metrics["reprojection_p95_px"] = _reprojection_p95(
        candidate_reprojection_error_px, frozen_raw & candidate_valid_mask
    )
    corrections = np.linalg.norm(candidate - raw, axis=-1) * 1000.0
    selected_corrections = corrections[frozen_raw & candidate_valid_mask]
    correction_p95 = float(np.percentile(selected_corrections, 95)) if selected_corrections.size else None
    correction_max = float(np.max(selected_corrections)) if selected_corrections.size else None
    lag = _lag_frames(raw, candidate, frozen_raw & candidate_valid_mask)

    failures: list[str] = []
    if dropped.any() or not same_metric_set:
        failures.append("CANDIDATE_DROPPED_FROZEN_EVALUATION_FRAMES")
    if added_observed.any():
        failures.append("CANDIDATE_INVENTED_OBSERVED_FRAMES")
    if dropped_observed.any():
        failures.append("CANDIDATE_REMOVED_OBSERVED_FRAMES")

    def require_improvement(name: str) -> None:
        before = raw_metrics[name]
        after = candidate_metrics[name]
        if before is None or after is None:
            failures.append(f"{name}:INSUFFICIENT_CONTIGUOUS_EVIDENCE")
        elif before == 0.0:
            if after > 0.0:
                failures.append(f"{name}:REGRESSED_FROM_ZERO")
        elif after > before * (1.0 - thresholds.minimum_relative_temporal_improvement):
            failures.append(f"{name}:INSUFFICIENT_IMPROVEMENT")

    require_improvement("wrist_step_p95_mm")
    require_improvement("all_joint_acceleration_p95_m_per_s2")
    for name, allowance in (
        ("reprojection_p95_px", thresholds.reprojection_p95_regression_px_max),
        ("bone_length_cv_max", thresholds.bone_length_cv_regression_max),
    ):
        before = raw_metrics[name]
        after = candidate_metrics[name]
        if before is None or after is None:
            failures.append(f"{name}:MISSING")
        elif after > before + allowance:
            failures.append(f"{name}:REGRESSION")
    if lag is not None and abs(lag) > thresholds.absolute_lag_frames_max:
        failures.append("TEMPORAL_PHASE_LAG_EXCEEDS_BOUND")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ADOPT" if not failures else "REJECT",
        "frozen_evaluation_frame_ids": ids[frozen_raw].astype(int).tolist(),
        "frozen_evaluation_count": int(frozen_raw.sum()),
        "candidate_evaluation_count": int(frozen_candidate.sum()),
        "same_frozen_metric_set": same_metric_set,
        "dropped_frame_ids": ids[dropped].astype(int).tolist(),
        "invented_observed_frame_ids": ids[added_observed].astype(int).tolist(),
        "removed_observed_frame_ids": ids[dropped_observed].astype(int).tolist(),
        "raw": raw_metrics,
        "candidate": candidate_metrics,
        "correction_p95_mm": correction_p95,
        "correction_max_mm": correction_max,
        "estimated_phase_lag_frames": lag,
        "thresholds": {
            "minimum_relative_temporal_improvement": thresholds.minimum_relative_temporal_improvement,
            "reprojection_p95_regression_px_max": thresholds.reprojection_p95_regression_px_max,
            "bone_length_cv_regression_max": thresholds.bone_length_cv_regression_max,
            "absolute_lag_frames_max": thresholds.absolute_lag_frames_max,
        },
        "reason_codes": sorted(set(failures)),
        "observed_authority_promoted": False,
    }
