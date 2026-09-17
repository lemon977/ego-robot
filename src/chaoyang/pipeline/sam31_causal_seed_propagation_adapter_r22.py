"""Fail-closed accepted-seed-ID to causal SAM3.1 propagation adapter.

This module never selects an object.  A caller must provide the already
accepted object ID and a live SAM3.1 session.  The adapter forces forward-only
propagation and emits exactly one compact row for every expected frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import cv2
import numpy as np


class CausalSeedPropagationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CausalSeedContract:
    session_id: str
    accepted_object_id: int
    start_frame_index: int
    frame_count: int
    source_frame_offset: int
    height: int
    width: int
    output_prob_thresh: float = 0.5
    ignored_object_ids: tuple[int, ...] = ()

    def validate(self) -> None:
        if not self.session_id:
            raise CausalSeedPropagationError("session_id is required")
        if self.accepted_object_id < 0:
            raise CausalSeedPropagationError("accepted_object_id must be nonnegative")
        if self.start_frame_index < 0 or self.frame_count <= 0:
            raise CausalSeedPropagationError("invalid frame interval")
        if self.source_frame_offset < 0 or self.height <= 0 or self.width <= 0:
            raise CausalSeedPropagationError("invalid image/source domain")
        if not 0 <= self.output_prob_thresh <= 1:
            raise CausalSeedPropagationError("output_prob_thresh outside [0,1]")


def _row(frame: int, source_frame: int, object_id: int, mask: np.ndarray, unexpected: list[int], reason: str) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area:
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        centroid = [float(xs.mean()), float(ys.mean())]
        components = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
    else:
        bbox, centroid, components = None, None, 0
    return {
        "frame_index": frame,
        "source_frame": source_frame,
        "accepted_object_id": object_id,
        "present": bool(area),
        "area_pixels": area,
        "bbox_xyxy": bbox,
        "centroid_xy": centroid,
        "connected_components": components,
        "unexpected_object_ids": unexpected,
        "reason": reason,
    }


def collect_causal_forward(
    adapter: Any,
    contract: CausalSeedContract,
    normalize_outputs: Callable[[dict[str, Any], int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Collect monotonic forward outputs without any prompt re-selection."""
    contract.validate()
    request = {
        "type": "propagate_in_video",
        "session_id": contract.session_id,
        "propagation_direction": "forward",
        "start_frame_index": contract.start_frame_index,
        "max_frame_num_to_track": contract.frame_count,
        "output_prob_thresh": contract.output_prob_thresh,
    }
    observed: dict[int, tuple[np.ndarray, list[int], str]] = {}
    previous = contract.start_frame_index - 1
    end = contract.start_frame_index + contract.frame_count
    for item in adapter.handle_stream_request(request):
        frame = int(item["frame_index"])
        if not contract.start_frame_index <= frame < end:
            raise CausalSeedPropagationError(f"stream frame outside frozen interval: {frame}")
        if frame <= previous:
            raise CausalSeedPropagationError("non-monotonic or duplicate stream frame")
        previous = frame
        masks, _, ids = normalize_outputs(item["outputs"], contract.height, contract.width)
        ids_int = [int(value) for value in ids]
        matches = [index for index, value in enumerate(ids_int) if value == contract.accepted_object_id]
        if len(matches) > 1:
            raise CausalSeedPropagationError("accepted object ID emitted more than once")
        unexpected = sorted(
            set(ids_int)
            - {contract.accepted_object_id}
            - set(contract.ignored_object_ids)
        )
        if matches:
            mask = np.asarray(masks[matches[0]], dtype=bool)
            reason = "ACCEPTED_SEED_ID_PRESENT" if not unexpected else "ACCEPTED_ID_WITH_UNEXPECTED_IDS"
        else:
            mask = np.zeros((contract.height, contract.width), dtype=bool)
            reason = "ACCEPTED_SEED_ID_ABSENT" if not unexpected else "ACCEPTED_ID_ABSENT_WITH_UNEXPECTED_IDS"
        observed[frame] = (mask, unexpected, reason)
    rows: list[dict[str, Any]] = []
    masks_by_source: dict[int, np.ndarray] = {}
    for frame in range(contract.start_frame_index, end):
        source = contract.source_frame_offset + (frame - contract.start_frame_index)
        if frame in observed:
            mask, unexpected, reason = observed[frame]
        else:
            mask = np.zeros((contract.height, contract.width), dtype=bool)
            unexpected, reason = [], "NO_STREAM_OUTPUT"
        rows.append(_row(frame, source, contract.accepted_object_id, mask, unexpected, reason))
        masks_by_source[source] = mask
    return rows, masks_by_source


def gate_offscreen_empty(rows: list[dict[str, Any]], expected_visible: dict[int, bool], minimum_empty_fraction: float) -> dict[str, Any]:
    offscreen = [row for row in rows if expected_visible.get(row["source_frame"]) is False]
    empty = [row for row in offscreen if not row["present"]]
    fraction = len(empty) / len(offscreen) if offscreen else 1.0
    return {"pass": fraction >= minimum_empty_fraction, "offscreen_frames": len(offscreen), "empty_frames": len(empty), "empty_fraction": fraction, "failure_frames": [row["source_frame"] for row in offscreen if row["present"]]}


def gate_reentry_latency(rows: list[dict[str, Any]], expected_visible: dict[int, bool], maximum_latency: int) -> dict[str, Any]:
    by_frame = {row["source_frame"]: row for row in rows}
    frames = sorted(by_frame)
    reentries = []
    for index, frame in enumerate(frames):
        if not expected_visible.get(frame, False) or (index and expected_visible.get(frames[index - 1], False)):
            continue
        latency = None
        for candidate in frames[index:]:
            if not expected_visible.get(candidate, False):
                break
            if by_frame[candidate]["present"]:
                latency = candidate - frame
                break
        reentries.append({"reentry_frame": frame, "latency_frames": latency, "pass": latency is not None and latency <= maximum_latency})
    return {"pass": all(item["pass"] for item in reentries), "maximum_latency_frames": maximum_latency, "reentries": reentries}


def gate_anatomical_side_nearest(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], left_wrist: dict[int, list[float]], right_wrist: dict[int, list[float]], minimum_fraction: float) -> dict[str, Any]:
    left = {row["source_frame"]: row for row in left_rows}
    right = {row["source_frame"]: row for row in right_rows}
    checks, passed, failures = 0, 0, []
    for frame in sorted(set(left) & set(right) & set(left_wrist) & set(right_wrist)):
        for side, row, own, other in (("left", left[frame], left_wrist[frame], right_wrist[frame]), ("right", right[frame], right_wrist[frame], left_wrist[frame])):
            if not row["present"] or row["centroid_xy"] is None:
                continue
            centroid = np.asarray(row["centroid_xy"], dtype=float)
            own_distance = float(np.linalg.norm(centroid - np.asarray(own, dtype=float)))
            other_distance = float(np.linalg.norm(centroid - np.asarray(other, dtype=float)))
            checks += 1
            if own_distance < other_distance:
                passed += 1
            else:
                failures.append({"source_frame": frame, "side": side, "own_distance": own_distance, "other_distance": other_distance})
    fraction = passed / checks if checks else 0.0
    return {"pass": bool(checks) and fraction >= minimum_fraction, "checks": checks, "passed": passed, "own_side_nearest_fraction": fraction, "failures": failures}


def gate_single_instance(rows: list[dict[str, Any]], seed_area: int, maximum_components: int, maximum_area_multiple: float) -> dict[str, Any]:
    failures = []
    for row in rows:
        reasons = []
        if row["unexpected_object_ids"]:
            reasons.append("UNEXPECTED_OBJECT_ID")
        if row["connected_components"] > maximum_components:
            reasons.append("TOO_MANY_COMPONENTS")
        if row["area_pixels"] > seed_area * maximum_area_multiple:
            reasons.append("AREA_MULTIPLE_EXCEEDED")
        if reasons:
            failures.append({"source_frame": row["source_frame"], "reasons": reasons})
    return {"pass": not failures, "seed_area": seed_area, "maximum_components": maximum_components, "maximum_area_multiple": maximum_area_multiple, "failure_frames": failures}
