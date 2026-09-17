#!/usr/bin/env python3
"""Pure audits for contact-assisted, per-instance 2D object authorities."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np


class ObjectAuthorityError(RuntimeError):
    pass


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


def point_distance(mask: np.ndarray, point_xy: Sequence[float]) -> float | None:
    y, x = np.where(mask)
    if not len(x):
        return None
    point = np.asarray(point_xy, dtype=np.float64)
    return float(np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2).min())


def frame_mask_metrics(
    mask: np.ndarray,
    *,
    raw_id: int,
    area_fraction_min: float,
    area_fraction_max: float,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ObjectAuthorityError("object mask must be a 2D binary array")
    area = int(value.sum())
    area_fraction = float(area / value.size)
    labels, _, stats, _ = cv2.connectedComponentsWithStats(value.astype(np.uint8), 8)
    component_areas = [int(row[cv2.CC_STAT_AREA]) for row in stats[1:] if row[cv2.CC_STAT_AREA] >= 16]
    largest_fraction = float(max(component_areas) / area) if area and component_areas else 0.0
    edge_pixels = int(
        value[0].sum() + value[-1].sum() + value[:, 0].sum() + value[:, -1].sum()
    )
    present = area > 0
    spatial_valid = bool(
        present
        and area_fraction <= float(area_fraction_max)
        and len(component_areas) <= int(gate["connected_components_over_16px_max"])
        and largest_fraction >= float(gate["largest_component_area_fraction_min"])
        and edge_pixels <= int(gate["image_edge_contact_pixels_max"])
    )
    return {
        "raw_id": int(raw_id),
        "present": present,
        "area_pixels": area,
        "area_fraction": area_fraction,
        "area_above_role_min": area_fraction >= float(area_fraction_min),
        "area_below_role_max": area_fraction <= float(area_fraction_max),
        "components_over_16px": len(component_areas),
        "largest_component_area_fraction": largest_fraction,
        "image_edge_contact_pixels": edge_pixels,
        "spatial_valid": spatial_valid,
        "connected_components_raw": int(max(labels - 1, 0)),
    }


def missing_runs(present: Sequence[bool]) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate([*present, True]):
        if not value and start is None:
            start = index
        elif value and start is not None:
            rows.append((start, index - 1))
            start = None
    return rows


def evaluate_instance_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_count: int,
    raw_id: int,
    contact_frame: int | None,
    contact_reseed_distance_px: float | None,
    frame_gate: Mapping[str, Any],
    identity_gate: Mapping[str, Any],
) -> dict[str, Any]:
    if [int(row.get("frame_index", -1)) for row in rows] != list(range(frame_count)):
        raise ObjectAuthorityError("object instance rows do not cover the full ordered source timeline")
    ids = {int(row.get("raw_id", -1)) for row in rows}
    present = [bool(row.get("present")) for row in rows]
    gaps = missing_runs(present)
    internal_gaps = [(start, end) for start, end in gaps if start > 0 and end < frame_count - 1]
    maximum_gap = max((end - start + 1 for start, end in gaps), default=0)
    present_fraction = float(sum(present) / frame_count)
    spatial_fraction = float(sum(bool(row.get("spatial_valid")) for row in rows) / frame_count)
    area_min_fraction = float(sum(bool(row.get("area_above_role_min")) for row in rows) / frame_count)
    area_ratios = []
    for previous, current in zip(rows, rows[1:]):
        previous_area = int(previous.get("area_pixels", 0))
        current_area = int(current.get("area_pixels", 0))
        if previous_area and current_area:
            area_ratios.append(current_area / previous_area)
    contact_indices = [] if contact_frame is None else [
        max(contact_frame - 1, 0), contact_frame, min(contact_frame + 1, frame_count - 1)
    ]
    contact_presence = all(present[index] for index in contact_indices)
    contact_distance_pass = bool(
        contact_frame is None
        or (
            contact_reseed_distance_px is not None
            and contact_reseed_distance_px
            <= float(identity_gate["contact_reseed_point_to_mask_px_max"])
        )
    )
    gates = {
        "full_source_frame_coverage": len(rows) == frame_count,
        "fixed_raw_id": ids == {int(raw_id)},
        "frame0_present": present[0],
        "last_frame_present": present[-1],
        "present_fraction": present_fraction >= float(frame_gate["present_fraction_min"]),
        "longest_missing_run": maximum_gap <= int(frame_gate["longest_missing_run_frames_max"]),
        "spatial_valid_fraction": spatial_fraction >= float(frame_gate["spatial_valid_fraction_min"]),
        "area_min_fraction": area_min_fraction >= float(frame_gate["spatial_valid_fraction_min"]),
        "adjacent_area_ratio": bool(
            area_ratios
            and min(area_ratios) >= float(frame_gate["adjacent_area_ratio_min"])
            and max(area_ratios) <= float(frame_gate["adjacent_area_ratio_max"])
        ),
        "contact_before_contact_after_presence": contact_presence,
        "contact_reseed_point_to_mask": contact_distance_pass,
        "post_gap_same_raw_id_reappearance": all(end < frame_count - 1 for _, end in internal_gaps),
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "raw_ids_observed": sorted(ids),
        "present_fraction": present_fraction,
        "spatial_valid_fraction": spatial_fraction,
        "area_above_role_min_fraction": area_min_fraction,
        "missing_runs": [{"start": start, "end": end} for start, end in gaps],
        "longest_missing_run_frames": maximum_gap,
        "adjacent_area_ratio_min": min(area_ratios) if area_ratios else None,
        "adjacent_area_ratio_max": max(area_ratios) if area_ratios else None,
        "contact_frames_checked": contact_indices,
        "contact_reseed_distance_px": contact_reseed_distance_px,
    }


def evaluate_cross_instance_identity(
    masks_by_role: Mapping[str, Sequence[np.ndarray]],
    *,
    duplicate_iou_max: float,
) -> dict[str, Any]:
    roles = sorted(masks_by_role)
    lengths = {len(masks_by_role[role]) for role in roles}
    if len(lengths) != 1:
        raise ObjectAuthorityError("object instance timelines have different lengths")
    frame_count = next(iter(lengths), 0)
    collisions: list[dict[str, Any]] = []
    maximum = 0.0
    for frame in range(frame_count):
        for left_index, left_role in enumerate(roles):
            for right_role in roles[left_index + 1:]:
                left = masks_by_role[left_role][frame]
                right = masks_by_role[right_role][frame]
                if not left.any() or not right.any():
                    continue
                value = binary_iou(left, right)
                maximum = max(maximum, value)
                if value > duplicate_iou_max:
                    collisions.append({
                        "frame_index": frame,
                        "left_role": left_role,
                        "right_role": right_role,
                        "iou": value,
                    })
    return {
        "pass": not collisions,
        "identity_collapse_frame_count": len({row["frame_index"] for row in collisions}),
        "maximum_cross_instance_iou": maximum,
        "duplicate_iou_max": float(duplicate_iou_max),
        "collisions": collisions,
        "masks_modified_or_mutually_excluded": False,
    }
