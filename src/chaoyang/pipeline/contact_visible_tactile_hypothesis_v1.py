"""Development contact hypotheses from processed tactile and visible geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
TACTILE_SCHEMA = "tactile-acquisition-aligned-frame-v2"
AUTHORITY = "TACTILE_SUPPORTED_HYPOTHESIS"


class ContactHypothesisError(ValueError):
    pass


def _tactile_finger_activity(side: Mapping[str, Any]) -> np.ndarray:
    if side.get("schema_version") != TACTILE_SCHEMA:
        raise ContactHypothesisError("processed tactile-v2 schema required")
    if side.get("offline_source_valid") is not True:
        return np.zeros(5, np.int64)
    offset = side.get("offline_source_offset_ms")
    if not isinstance(offset, (int, float)) or abs(float(offset)) > 40.0:
        raise ContactHypothesisError("offline tactile alignment exceeds 40 ms")
    grid = np.asarray(side.get("finger_grid_5x4x8"))
    valid = np.asarray(side.get("valid_mask_5x4x8"), bool)
    active = np.asarray(side.get("active_mask_5x4x8"), bool)
    if grid.shape != (5, 4, 8) or valid.shape != grid.shape or active.shape != grid.shape:
        raise ContactHypothesisError("tactile finger grid shape mismatch")
    return np.count_nonzero((grid != 0) & valid & active, axis=(1, 2))


def tactile_visible_surface_hypotheses(
    *, tactile: Mapping[str, Any], fingertip_xyz_m: np.ndarray,
    fingertip_observed: np.ndarray, visible_surface_centroid_m: np.ndarray,
    visible_surface_normal: np.ndarray, maximum_plane_distance_m: float = 0.015,
) -> dict[str, Any]:
    tips = np.asarray(fingertip_xyz_m, np.float64)
    observed = np.asarray(fingertip_observed, bool)
    centroid = np.asarray(visible_surface_centroid_m, np.float64)
    normal = np.asarray(visible_surface_normal, np.float64)
    if tips.shape != (2, 5, 3) or observed.shape != (2, 5):
        raise ContactHypothesisError("fingertips must be side x five fingers")
    if centroid.shape != (3,) or normal.shape != (3,) or not np.isfinite(normal).all():
        raise ContactHypothesisError("visible surface plane is invalid")
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        raise ContactHypothesisError("visible surface normal is degenerate")
    normal = normal / norm
    if set(tactile) != {"left", "right"}:
        raise ContactHypothesisError("tactile must contain exact left/right sides")

    rows = []
    for side_index, side_name in enumerate(("left", "right")):
        activity = _tactile_finger_activity(tactile[side_name])
        distances = np.abs((tips[side_index] - centroid) @ normal)
        for finger_index, finger_name in enumerate(FINGER_NAMES):
            geometry_valid = bool(
                observed[side_index, finger_index]
                and np.isfinite(tips[side_index, finger_index]).all()
            )
            tactile_supported = bool(activity[finger_index] > 0)
            geometry_supported = bool(
                geometry_valid
                and distances[finger_index] <= maximum_plane_distance_m
            )
            supported = tactile_supported and geometry_supported
            rows.append({
                "side": side_name,
                "finger": finger_name,
                "status": AUTHORITY if supported else "NOT_SUPPORTED",
                "tactile_nonzero_active_taxels": int(activity[finger_index]),
                "visible_surface_plane_distance_m": (
                    float(distances[finger_index]) if geometry_valid else None
                ),
                "direct_visible_geometry": geometry_valid,
                "offline_tactile_valid": bool(
                    tactile[side_name].get("offline_source_valid") is True
                ),
            })
    return {
        "schema_version": "contact-visible-tactile-hypothesis-v1",
        "authority": AUTHORITY,
        "hypotheses": rows,
        "supported_count": sum(row["status"] == AUTHORITY for row in rows),
        "inputs": {
            "tactile": "PROCESSED_ENTITIES_TACTILE_OFFLINE_ALIGNED",
            "hand": "HAWOR_VISIBLE_FINGERTIPS",
            "object": "DIRECT_VISIBLE_OBJECT_SURFACE",
            "clean": "NOT_CONSUMED",
            "pico26_hand": "NOT_CONSUMED",
        },
        "force_claim": False,
        "contact_ground_truth": False,
        "causal_training_input": False,
        "claim_limit": (
            "Uncalibrated tactile activity plus directly visible geometric "
            "proximity; not force, contact ground truth, or causal policy input."
        ),
    }
