"""Frozen, HaWoR-independent frame selection for the AI2 V3.2 first loop.

The selector is intentionally a pure function over the physical-left RGB
timeline and an independently produced region-evaluability ledger.  It has no
parameter through which HaWoR, MANO, q22, or a mask derived from any of those
artifacts can be supplied.  Raw and bounded candidates are bound as opaque
SHA references only and are never opened by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

from chaoyang.pipeline.hawor_bounded_comparison_v31 import compare_bounded_hawor


SCHEMA_VERSION = "AI2_INDEPENDENT_OBSERVABILITY_V32"
FIXED_SESSIONS = (
    "play_cards_0915_031",
    "get_potato_chips_0915_007",
)
IMAGE_DOMAIN = {
    "camera": "VST",
    "physical_eye": "left",
    "source_index": 1,
    "pixel_transform": "RESIZE_ONLY",
    "lens_remap_applied": False,
}
REVIEW_BIN_COUNT = 16
MAX_WINDOW_COUNT = 2
MAX_WINDOW_FRAMES = 30
REGION_STATES = frozenset({"REGION_EVALUABLE", "NOT_REGION_EVALUABLE", "UNKNOWN"})
ALLOWED_EVIDENCE_FAMILIES = frozenset(
    {"PHYSICAL_LEFT_RGB_REVIEW", "INDEPENDENT_HAND_REGION_PROXY"}
)
FORBIDDEN_DEPENDENCY_TOKENS = ("hawor", "mano", "q22")


class IndependentObservabilityError(ValueError):
    """Raised when selection could depend on the trajectory being evaluated."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sha_ref(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    required = {"path", "bytes", "sha256"}
    if set(value) != required:
        raise IndependentObservabilityError(
            f"{name} must be an opaque exact path/bytes/sha256 reference"
        )
    path = value["path"]
    byte_count = value["bytes"]
    sha = value["sha256"]
    if not isinstance(path, str) or not path:
        raise IndependentObservabilityError(f"{name}.path must be non-empty")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise IndependentObservabilityError(f"{name}.bytes must be non-negative")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
    ):
        raise IndependentObservabilityError(f"{name}.sha256 must be lowercase SHA-256")
    # Deliberately do not resolve or open path here.  Selection must happen
    # before the evaluated candidates can be read.
    return {"path": path, "bytes": byte_count, "sha256": sha}


def _validate_evidence_sources(
    sources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not sources:
        raise IndependentObservabilityError("at least one independent evidence source is required")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, source in enumerate(sources):
        source_id = source.get("source_id")
        family = source.get("family")
        dependencies = source.get("dependencies")
        if not isinstance(source_id, str) or not source_id or source_id in ids:
            raise IndependentObservabilityError(f"evidence_sources[{index}] source_id is invalid")
        ids.add(source_id)
        if family not in ALLOWED_EVIDENCE_FAMILIES:
            raise IndependentObservabilityError(
                f"evidence source {source_id} has forbidden family {family!r}"
            )
        if source.get("independent_from_evaluated_trajectory") is not True:
            raise IndependentObservabilityError(
                f"evidence source {source_id} is not independently frozen"
            )
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) or not item for item in dependencies
        ):
            raise IndependentObservabilityError(
                f"evidence source {source_id} dependencies must be strings"
            )
        lowered = " ".join([source_id, str(family), *dependencies]).casefold()
        forbidden = [token for token in FORBIDDEN_DEPENDENCY_TOKENS if token in lowered]
        if forbidden:
            raise IndependentObservabilityError(
                f"evidence source {source_id} depends on evaluated data: {forbidden}"
            )
        exact_ref = _validate_sha_ref(source.get("artifact_ref", {}), name=source_id)
        result.append(
            {
                "source_id": source_id,
                "family": family,
                "dependencies": list(dependencies),
                "independent_from_evaluated_trajectory": True,
                "artifact_ref": exact_ref,
            }
        )
    return result


def _select_equal_time_frames(frame_ids: np.ndarray, timestamps_s: np.ndarray) -> list[dict[str, Any]]:
    if frame_ids.size < REVIEW_BIN_COUNT:
        raise IndependentObservabilityError(
            f"at least {REVIEW_BIN_COUNT} frames are required for equal-time review"
        )
    start = float(timestamps_s[0])
    stop = float(timestamps_s[-1])
    if stop <= start:
        raise IndependentObservabilityError("timeline duration must be positive")
    edges = np.linspace(start, stop, REVIEW_BIN_COUNT + 1, dtype=np.float64)
    selected: list[dict[str, Any]] = []
    for bin_index in range(REVIEW_BIN_COUNT):
        if bin_index + 1 == REVIEW_BIN_COUNT:
            members = np.flatnonzero(
                (timestamps_s >= edges[bin_index]) & (timestamps_s <= edges[bin_index + 1])
            )
        else:
            members = np.flatnonzero(
                (timestamps_s >= edges[bin_index]) & (timestamps_s < edges[bin_index + 1])
            )
        if members.size == 0:
            raise IndependentObservabilityError(
                f"equal-time review bin {bin_index} contains no source frame"
            )
        midpoint = 0.5 * (edges[bin_index] + edges[bin_index + 1])
        offsets = np.abs(timestamps_s[members] - midpoint)
        chosen = int(members[int(np.argmin(offsets))])
        selected.append(
            {
                "bin_index": bin_index,
                "bin_start_s": float(edges[bin_index]),
                "bin_end_s": float(edges[bin_index + 1]),
                "source_frame": int(frame_ids[chosen]),
                "timestamp_s": float(timestamps_s[chosen]),
            }
        )
    if len({row["source_frame"] for row in selected}) != REVIEW_BIN_COUNT:
        raise IndependentObservabilityError("equal-time bins did not yield 16 unique frames")
    return selected


def _maximal_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    chunks = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size]


def _select_chronological_windows(
    frame_ids: np.ndarray,
    timestamps_s: np.ndarray,
    region_states: np.ndarray,
) -> list[dict[str, Any]]:
    # Chronological-first is the frozen policy.  No trajectory quality score
    # participates, so a later HaWoR-success interval cannot win selection.
    windows: list[dict[str, Any]] = []
    mask = region_states == "REGION_EVALUABLE"
    for run_start, run_end in _maximal_true_runs(mask):
        if len(windows) == MAX_WINDOW_COUNT:
            break
        selected_end = min(run_end, run_start + MAX_WINDOW_FRAMES - 1)
        windows.append(
            {
                "window_index": len(windows),
                "source_frame_start": int(frame_ids[run_start]),
                "source_frame_end": int(frame_ids[selected_end]),
                "source_index_start": run_start,
                "source_index_end": selected_end,
                "frame_count": selected_end - run_start + 1,
                "timestamp_start_s": float(timestamps_s[run_start]),
                "timestamp_end_s": float(timestamps_s[selected_end]),
                "selection_policy": "CHRONOLOGICAL_FIRST_INDEPENDENT_REGION_RUN",
            }
        )
    return windows


def freeze_ai2_independent_observability(
    *,
    session_id: str,
    frame_ids: np.ndarray,
    timestamps_s: np.ndarray,
    region_states: Sequence[str] | np.ndarray,
    region_evidence_source_ids: Sequence[Sequence[str]],
    evidence_sources: Sequence[Mapping[str, Any]],
    rgb_input_ref: Mapping[str, Any],
    raw_candidate_ref: Mapping[str, Any],
    bounded_candidate_ref: Mapping[str, Any],
    producer_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Freeze review frames/windows without accepting evaluated pose data."""

    if session_id not in FIXED_SESSIONS:
        raise IndependentObservabilityError(f"session is outside the fixed first cohort: {session_id}")
    ids = np.asarray(frame_ids, dtype=np.int64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    states = np.asarray(region_states)
    if ids.ndim != 1 or ids.size == 0 or times.shape != ids.shape or states.shape != ids.shape:
        raise IndependentObservabilityError("frame_ids, timestamps_s, and region_states must be [T]")
    if np.any(np.diff(ids) <= 0) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise IndependentObservabilityError("source frame and timestamp axes must be strictly increasing")
    if any(str(state) not in REGION_STATES for state in states.tolist()):
        raise IndependentObservabilityError("region_states contains an unknown state")
    states = states.astype("U32")
    sources = _validate_evidence_sources(evidence_sources)
    source_ids = {row["source_id"] for row in sources}
    if len(region_evidence_source_ids) != ids.size:
        raise IndependentObservabilityError("region evidence axis must have one row per source frame")
    evidence_by_frame: list[list[str]] = []
    for index, row in enumerate(region_evidence_source_ids):
        if isinstance(row, (str, bytes)):
            raise IndependentObservabilityError(
                f"region_evidence_source_ids[{index}] must be a sequence of IDs"
            )
        values = sorted(set(row))
        if any(not isinstance(value, str) or value not in source_ids for value in values):
            raise IndependentObservabilityError(f"frame {index} cites an unknown evidence source")
        if states[index] != "UNKNOWN" and not values:
            raise IndependentObservabilityError(
                f"frame {index} has a decided region state without independent evidence"
            )
        evidence_by_frame.append(values)

    raw_ref = _validate_sha_ref(raw_candidate_ref, name="raw_candidate_ref")
    bounded_ref = _validate_sha_ref(bounded_candidate_ref, name="bounded_candidate_ref")
    rgb_ref = _validate_sha_ref(rgb_input_ref, name="rgb_input_ref")
    for name, sha in (("producer_sha256", producer_sha256), ("config_sha256", config_sha256)):
        _validate_sha_ref({"path": name, "bytes": 0, "sha256": sha}, name=name)

    review_frames = _select_equal_time_frames(ids, times)
    windows = _select_chronological_windows(ids, times, states)
    review_ids = {row["source_frame"] for row in review_frames}
    selected_window_indices: set[int] = set()
    for window in windows:
        selected_window_indices.update(
            range(int(window["source_index_start"]), int(window["source_index_end"]) + 1)
        )
    ledger = [
        {
            "source_frame": int(frame_id),
            "timestamp_s": float(times[index]),
            "region_state": str(states[index]),
            "evidence_source_ids": evidence_by_frame[index],
            "selected_for_equal_time_review": int(frame_id) in review_ids,
            "selected_for_continuous_window": index in selected_window_indices,
        }
        for index, frame_id in enumerate(ids)
    ]
    selection_payload = {
        "session_id": session_id,
        "frame_ids": ids.tolist(),
        "timestamps_s": times.tolist(),
        "region_states": states.tolist(),
        "region_evidence_source_ids": evidence_by_frame,
        "review_frames": review_frames,
        "continuous_windows": windows,
        "rgb_input_ref": rgb_ref,
        "evidence_sources": sources,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_EVALUATED_TRAJECTORY_READ",
        "session_id": session_id,
        "image_domain": dict(IMAGE_DOMAIN),
        "review_policy": {
            "equal_time_bin_count": REVIEW_BIN_COUNT,
            "selected_review_frame_count": len(review_frames),
            "maximum_continuous_windows": MAX_WINDOW_COUNT,
            "maximum_frames_per_window": MAX_WINDOW_FRAMES,
            "window_order": "CHRONOLOGICAL_FIRST",
        },
        "review_frames": review_frames,
        "continuous_windows": windows,
        "frame_ledger": ledger,
        "state_counts": {
            state: int(np.sum(states == state)) for state in sorted(REGION_STATES)
        },
        "evidence_sources": sources,
        "evaluated_candidates": {
            "raw": {"artifact_ref": raw_ref, "disposition": "FROZEN_EXISTING_SHA"},
            "bounded": {
                "artifact_ref": bounded_ref,
                "disposition": "HOLD_NUMERIC_GATES",
            },
            "candidate_content_read_during_selection": False,
            "new_smoothed_candidate_created": False,
        },
        "provenance": {
            "source_frame_axis": "frame_ledger.source_frame",
            "timestamp_axis": "frame_ledger.timestamp_s",
            "units": {"timestamp": "seconds"},
            "side": "physical_left",
            "validity": "region_state_and_evidence_source_ids",
            "observed_or_inferred": "OBSERVED_OR_UNKNOWN_NO_FORWARD_FILL",
            "temporal_authority": "CURRENT_RGB_ONLY_NO_FUTURE_TRAJECTORY_DATA",
            "producer_sha256": producer_sha256,
            "config_sha256": config_sha256,
            "input_sha256": rgb_ref["sha256"],
            "selection_sha256": _canonical_sha(selection_payload),
        },
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }


def frozen_evaluation_mask(
    frozen_selection: Mapping[str, Any], frame_ids: np.ndarray
) -> np.ndarray:
    """Recover the immutable union of review frames and continuous windows."""

    if (
        frozen_selection.get("schema_version") != SCHEMA_VERSION
        or frozen_selection.get("status")
        != "FROZEN_BEFORE_EVALUATED_TRAJECTORY_READ"
    ):
        raise IndependentObservabilityError("a frozen V3.2 selection is required")
    ids = np.asarray(frame_ids, dtype=np.int64)
    ledger = frozen_selection.get("frame_ledger")
    if not isinstance(ledger, list) or ids.ndim != 1 or len(ledger) != ids.size:
        raise IndependentObservabilityError("frozen selection and evaluated frame axes differ")
    frozen_ids = np.asarray([row.get("source_frame") for row in ledger], dtype=np.int64)
    if not np.array_equal(ids, frozen_ids):
        raise IndependentObservabilityError("evaluated frame IDs differ from frozen selection")
    return np.asarray(
        [
            row.get("selected_for_equal_time_review") is True
            or row.get("selected_for_continuous_window") is True
            for row in ledger
        ],
        dtype=bool,
    )


def compare_frozen_raw_bounded_v32(
    *,
    frozen_selection: Mapping[str, Any],
    frame_ids: np.ndarray,
    timestamps_s: np.ndarray,
    raw_candidate_ref: Mapping[str, Any],
    bounded_candidate_ref: Mapping[str, Any],
    raw_joints: np.ndarray,
    bounded_joints: np.ndarray,
    raw_valid: np.ndarray,
    bounded_valid: np.ndarray,
    raw_reprojection_error_px: np.ndarray,
    bounded_reprojection_error_px: np.ndarray,
    raw_observed: np.ndarray,
    bounded_observed: np.ndarray,
) -> dict[str, Any]:
    """Compare existing candidates on the already frozen independent frame set."""

    candidates = frozen_selection.get("evaluated_candidates", {})
    expected_raw = candidates.get("raw", {}).get("artifact_ref")
    expected_bounded = candidates.get("bounded", {}).get("artifact_ref")
    supplied_raw = _validate_sha_ref(raw_candidate_ref, name="raw_candidate_ref")
    supplied_bounded = _validate_sha_ref(
        bounded_candidate_ref, name="bounded_candidate_ref"
    )
    if supplied_raw != expected_raw or supplied_bounded != expected_bounded:
        raise IndependentObservabilityError("raw/bounded candidate SHA lineage drift")
    if candidates.get("bounded", {}).get("disposition") != "HOLD_NUMERIC_GATES":
        raise IndependentObservabilityError("bounded candidate disposition was renamed")
    ids = np.asarray(frame_ids, dtype=np.int64)
    mask = frozen_evaluation_mask(frozen_selection, ids)
    if not mask.any():
        raise IndependentObservabilityError("frozen independent evaluation set is empty")
    comparison = compare_bounded_hawor(
        raw_joints=raw_joints,
        candidate_joints=bounded_joints,
        raw_valid=raw_valid,
        candidate_valid=bounded_valid,
        frozen_observable_mask=mask,
        timestamps_s=timestamps_s,
        frame_ids=ids,
        raw_reprojection_error_px=raw_reprojection_error_px,
        candidate_reprojection_error_px=bounded_reprojection_error_px,
        raw_observed=raw_observed,
        candidate_observed=bounded_observed,
    )
    return {
        "schema_version": "AI2_FROZEN_RAW_BOUNDED_COMPARISON_V32",
        "session_id": frozen_selection["session_id"],
        "selection_sha256": frozen_selection["provenance"]["selection_sha256"],
        "raw_candidate_ref": supplied_raw,
        "bounded_candidate_ref": supplied_bounded,
        "bounded_disposition_before_comparison": "HOLD_NUMERIC_GATES",
        "same_frozen_independent_frame_set": comparison["same_frozen_metric_set"],
        "new_smoothed_candidate_created": False,
        "comparison": comparison,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
