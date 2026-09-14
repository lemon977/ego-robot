#!/usr/bin/env python3
"""Pure comparison gates for isolated-vs-joint object-state benchmarks."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline.contact_assisted_object_authority import binary_iou


class StateLayoutABError(RuntimeError):
    pass


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    y, x = np.where(np.asarray(mask, dtype=bool))
    if not len(x):
        return None
    return float(x.mean()), float(y.mean())


def compare_binary_masks(
    isolated: np.ndarray,
    joint: np.ndarray,
    *,
    frame_index: int,
    role: str,
) -> dict[str, Any]:
    left = np.asarray(isolated, dtype=bool)
    right = np.asarray(joint, dtype=bool)
    if left.shape != right.shape or left.ndim != 2:
        raise StateLayoutABError("A/B masks must be same-shape 2D arrays")
    area_a = int(left.sum())
    area_b = int(right.sum())
    if area_a == area_b == 0:
        ratio = 1.0
    elif area_a == 0:
        ratio = math.inf
    else:
        ratio = float(area_b / area_a)
    center_a = centroid(left)
    center_b = centroid(right)
    if center_a is None and center_b is None:
        center_delta = 0.0
    elif center_a is None or center_b is None:
        center_delta = math.inf
    else:
        center_delta = float(math.dist(center_a, center_b))
    return {
        "frame_index": int(frame_index),
        "role": str(role),
        "exact": bool(np.array_equal(left, right)),
        "iou": binary_iou(left, right),
        "area_a": area_a,
        "area_b": area_b,
        "area_ratio_b_over_a": ratio,
        "centroid_delta_px": center_delta,
    }


def summarize_equivalence(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_role_frame_count: int,
    gate: Mapping[str, Any],
    per_role_gate_vectors_exact: bool,
    cross_instance_gate_vectors_exact: bool,
    final_status_exact: bool,
    cross_id_bleed_frame_count: int,
    internal_nonoverlap_or_suppression_applied: bool,
    wall_seconds_a: float,
    wall_seconds_b: float,
) -> dict[str, Any]:
    if len(rows) != expected_role_frame_count:
        raise StateLayoutABError("A/B comparison does not cover every role/frame")
    exact_fraction = float(sum(bool(row["exact"]) for row in rows) / len(rows))
    ious = [float(row["iou"]) for row in rows]
    ratios = [float(row["area_ratio_b_over_a"]) for row in rows]
    deltas = [float(row["centroid_delta_px"]) for row in rows]
    speedup = float(wall_seconds_a / wall_seconds_b) if wall_seconds_b > 0 else math.inf
    gates = {
        "role_frame_inventory": len(rows) == expected_role_frame_count,
        "binary_mask_exact": exact_fraction == 1.0,
        "iou": min(ious) >= float(gate["per_role_per_frame_binary_mask_iou_min"]),
        "area_ratio": (
            min(ratios) >= float(gate["per_role_per_frame_area_ratio_min"])
            and max(ratios) <= float(gate["per_role_per_frame_area_ratio_max"])
        ),
        "centroid_delta": max(deltas) <= float(gate["per_role_per_frame_centroid_delta_px_max"]),
        "per_role_gate_vector": bool(per_role_gate_vectors_exact),
        "cross_instance_gate_vector": bool(cross_instance_gate_vectors_exact),
        "final_status": bool(final_status_exact),
        "cross_id_bleed": cross_id_bleed_frame_count <= int(gate["cross_id_bleed_frame_count_max"]),
        "no_internal_nonoverlap_or_suppression": (
            internal_nonoverlap_or_suppression_applied
            is bool(gate["internal_nonoverlap_or_suppression_applied"])
        ),
        "speedup": speedup >= float(gate["speedup_ratio_min"]),
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "role_frame_count": len(rows),
        "exact_mask_fraction": exact_fraction,
        "iou_min": min(ious),
        "iou_median": float(np.median(ious)),
        "area_ratio_min": min(ratios),
        "area_ratio_max": max(ratios),
        "centroid_delta_px_max": max(deltas),
        "cross_id_bleed_frame_count": int(cross_id_bleed_frame_count),
        "internal_nonoverlap_or_suppression_applied": bool(
            internal_nonoverlap_or_suppression_applied
        ),
        "wall_seconds_a": float(wall_seconds_a),
        "wall_seconds_b": float(wall_seconds_b),
        "speedup_ratio_a_over_b": speedup,
    }
