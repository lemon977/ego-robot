"""Auditable dual wrist representation for Exact78/AI1 development.

An anatomical wrist frame and a visible surface observation are deliberately
kept separate.  The surface observation can validate registration but is not
silently converted into a joint centre.  Fusion is bounded relative to the
static-calibrated tracker prior on every frame, so corrections cannot
accumulate into an implicit installation calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np


TEMPORAL_AUTHORITIES = {
    "CAUSAL_CURRENT",
    "OFFLINE_NONCAUSAL",
    "INFERRED",
    "UNKNOWN_TEMPORAL_AUTHORITY",
}


def validate_rigid_transforms(values: np.ndarray, *, label: str) -> np.ndarray:
    transforms = np.asarray(values, dtype=np.float64)
    if transforms.shape[-2:] != (4, 4):
        raise ValueError(f"{label} must end in [4,4]")
    if not np.isfinite(transforms).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(transforms[..., 3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{label} homogeneous row is invalid")
    rotation = transforms[..., :3, :3]
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(
        np.swapaxes(rotation, -1, -2) @ rotation,
        identity,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"{label} rotation determinant is not +1")
    return transforms


def compose_camera_wrist(
    T_camera_controller: np.ndarray,
    T_controller_wrist: np.ndarray,
) -> np.ndarray:
    """Apply the fixed direction: T_camera_wrist = T_camera_controller @ T_controller_wrist."""

    controller = validate_rigid_transforms(
        T_camera_controller, label="T_camera_controller"
    )
    wrist = validate_rigid_transforms(
        T_controller_wrist, label="T_controller_wrist"
    )
    if controller.ndim != 4 or controller.shape[1:] != (2, 4, 4):
        raise ValueError("T_camera_controller must have shape [T,2,4,4]")
    if wrist.shape == (2, 4, 4):
        wrist = np.broadcast_to(wrist[None], controller.shape)
    elif wrist.shape != controller.shape:
        raise ValueError("T_controller_wrist must be [2,4,4] or [T,2,4,4]")
    return controller @ wrist


def surface_observation_admission(
    *,
    source_pixel_uv: np.ndarray,
    surface_point_xyz: np.ndarray,
    independent_rgb_wrist_region: bool,
    source_pixel_traceable: bool,
    local_mask_pure: bool,
    depth_rgb_registration_passed: bool,
    local_depth_continuous: bool,
    free_of_object_cable_or_unknown_attachment: bool,
    temporal_authority: str,
) -> dict[str, Any]:
    uv = np.asarray(source_pixel_uv, dtype=np.float64)
    xyz = np.asarray(surface_point_xyz, dtype=np.float64)
    if temporal_authority not in TEMPORAL_AUTHORITIES:
        raise ValueError(f"unknown temporal authority: {temporal_authority}")
    checks = {
        "independent_rgb_wrist_region": bool(independent_rgb_wrist_region),
        "source_pixel_traceable": bool(source_pixel_traceable),
        "local_mask_pure": bool(local_mask_pure),
        "depth_rgb_registration_passed": bool(depth_rgb_registration_passed),
        "local_depth_continuous": bool(local_depth_continuous),
        "free_of_object_cable_or_unknown_attachment": bool(
            free_of_object_cable_or_unknown_attachment
        ),
        "finite_source_pixel_uv": uv.shape == (2,) and np.isfinite(uv).all(),
        "finite_positive_surface_xyz": (
            xyz.shape == (3,) and np.isfinite(xyz).all() and xyz[2] > 0.0
        ),
    }
    valid = all(checks.values())
    region_valid = checks["independent_rgb_wrist_region"]
    return {
        "surface_point_valid": valid,
        "region_registration_only": region_valid and not valid,
        "source_pixel_uv": uv.tolist() if checks["finite_source_pixel_uv"] else None,
        "surface_point_xyz": xyz.tolist() if valid else None,
        "surface_role": "VISIBLE_WRIST_SURFACE" if valid else "UNKNOWN",
        "temporal_authority": temporal_authority,
        "checks": checks,
        "claim_limit": (
            "Visible surface observation; never an anatomical wrist joint centre"
        ),
    }


def _bounded_translation(
    prior: np.ndarray,
    candidate: np.ndarray,
    bound_m: float,
) -> tuple[np.ndarray, float, bool]:
    delta = candidate - prior
    norm = float(np.linalg.norm(delta))
    touched = norm > bound_m
    if touched and norm > 0.0:
        delta = delta * (bound_m / norm)
    return prior + delta, float(np.linalg.norm(delta)), touched


def build_dual_wrist_representation(
    *,
    frame_ids: np.ndarray,
    timestamps_s: np.ndarray,
    T_camera_controller_raw: np.ndarray,
    T_controller_wrist: np.ndarray,
    controller_valid: np.ndarray,
    hawor_T_camera_wrist: np.ndarray,
    hawor_valid: np.ndarray,
    visible_surface_xyz_camera: np.ndarray,
    visible_surface_uv: np.ndarray,
    visible_surface_valid: np.ndarray,
    visible_region_valid: np.ndarray,
    source_temporal_authority: dict[str, str],
    correction_bound_m: float = 0.030,
    tracker_weight: float = 0.65,
    hawor_weight: float = 0.35,
) -> dict[str, np.ndarray | dict[str, Any]]:
    frames = np.asarray(frame_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    raw = validate_rigid_transforms(
        T_camera_controller_raw, label="T_camera_controller_raw"
    )
    hawor = validate_rigid_transforms(
        hawor_T_camera_wrist, label="hawor_T_camera_wrist"
    )
    if raw.shape != hawor.shape or raw.ndim != 4 or raw.shape[1:] != (2, 4, 4):
        raise ValueError("controller and HaWoR transforms must be [T,2,4,4]")
    count = raw.shape[0]
    expected_side = (count, 2)
    if frames.shape != (count,) or timestamps.shape != (count,):
        raise ValueError("frame_ids/timestamps must have one value per frame")
    if not np.all(np.diff(frames) > 0) or not np.all(np.diff(timestamps) > 0):
        raise ValueError("frame IDs and timestamps must be strictly increasing")
    controller_ok = np.asarray(controller_valid, dtype=bool)
    hawor_ok = np.asarray(hawor_valid, dtype=bool)
    surface_ok = np.asarray(visible_surface_valid, dtype=bool)
    region_ok = np.asarray(visible_region_valid, dtype=bool)
    surface_xyz = np.asarray(visible_surface_xyz_camera, dtype=np.float64)
    surface_uv = np.asarray(visible_surface_uv, dtype=np.float64)
    if any(value.shape != expected_side for value in (controller_ok, hawor_ok, surface_ok, region_ok)):
        raise ValueError("all validity arrays must be [T,2]")
    if surface_xyz.shape != (count, 2, 3) or surface_uv.shape != (count, 2, 2):
        raise ValueError("visible surface arrays must be [T,2,3] and [T,2,2]")
    if correction_bound_m <= 0.0:
        raise ValueError("correction bound must be positive")
    if tracker_weight < 0.0 or hawor_weight < 0.0 or tracker_weight + hawor_weight <= 0.0:
        raise ValueError("fusion weights must be non-negative with positive sum")
    for source in ("controller", "hawor", "visible_surface"):
        if source_temporal_authority.get(source) not in TEMPORAL_AUTHORITIES:
            raise ValueError(f"missing/invalid temporal authority for {source}")

    tracker = compose_camera_wrist(raw, T_controller_wrist)
    fused = tracker.copy()
    fused_valid = controller_ok.copy()
    correction = np.zeros((count, 2, 3), dtype=np.float64)
    correction_norm = np.zeros(expected_side, dtype=np.float64)
    touched = np.zeros(expected_side, dtype=bool)
    fusion_source = np.full(expected_side, "INVALID", dtype="U32")
    for frame in range(count):
        for side in range(2):
            if controller_ok[frame, side] and hawor_ok[frame, side]:
                prior = tracker[frame, side, :3, 3]
                candidate = (
                    tracker_weight * prior
                    + hawor_weight * hawor[frame, side, :3, 3]
                ) / (tracker_weight + hawor_weight)
                value, norm, clipped = _bounded_translation(
                    prior, candidate, correction_bound_m
                )
                fused[frame, side, :3, 3] = value
                correction[frame, side] = value - prior
                correction_norm[frame, side] = norm
                touched[frame, side] = clipped
                fusion_source[frame, side] = "TRACKER_PLUS_HAWOR"
            elif controller_ok[frame, side]:
                fusion_source[frame, side] = "STATIC_CALIBRATED_TRACKER"
            elif hawor_ok[frame, side]:
                fused[frame, side] = hawor[frame, side]
                fused_valid[frame, side] = True
                correction[frame, side] = np.nan
                correction_norm[frame, side] = np.nan
                fusion_source[frame, side] = "HAWOR_ONLY_NO_TRACKER_PRIOR"
            else:
                fused[frame, side] = np.nan

    # Invalid surface rows are zeroed to NaN so a consumer cannot accidentally
    # use a region-only observation as three-dimensional evidence.
    surface_xyz = surface_xyz.copy()
    surface_uv = surface_uv.copy()
    surface_xyz[~surface_ok] = np.nan
    surface_uv[~(surface_ok | region_ok)] = np.nan
    return {
        "frame_ids": frames,
        "timestamps_s": timestamps,
        "T_camera_controller_raw": raw,
        "T_controller_wrist": np.asarray(T_controller_wrist, dtype=np.float64),
        "T_camera_wrist": tracker,
        "anatomical_wrist_center_camera": tracker[..., :3, 3].copy(),
        "hawor_T_camera_wrist": hawor,
        "hawor_valid": hawor_ok,
        "visible_wrist_surface_xyz_camera": surface_xyz,
        "visible_wrist_surface_uv": surface_uv,
        "visible_wrist_surface_valid": surface_ok,
        "visible_wrist_region_valid": region_ok,
        "fused_T_camera_wrist": fused,
        "fused_valid": fused_valid,
        "fused_correction_camera": correction,
        "fused_correction_norm_m": correction_norm,
        "fused_correction_bound_touched": touched,
        "fusion_source": fusion_source,
        "metadata": {
            "schema_version": "WRIST_DUAL_REPRESENTATION_V1",
            "transform_convention": (
                "T_A_B maps points in B coordinates into A coordinates; "
                "T_camera_wrist=T_camera_controller@T_controller_wrist"
            ),
            "development_numeric_correction_bound_m": correction_bound_m,
            "correction_accumulation": False,
            "surface_is_anatomical_center": False,
            "uncertainty_status": "UNKNOWN_UNCALIBRATED",
            "heuristic_weights": {
                "tracker": tracker_weight,
                "hawor": hawor_weight,
                "calibrated_covariance": False,
            },
            "temporal_authority": dict(source_temporal_authority),
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
            "external_metric_authority": False,
        },
    }


def transform_points(T_target_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    transforms = validate_rigid_transforms(T_target_source, label="T_target_source")
    points = np.asarray(points_source, dtype=np.float64)
    if transforms.shape[:-2] != points.shape[:-1]:
        raise ValueError("transform and point leading shapes differ")
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    return np.einsum("...ij,...j->...i", transforms, homogeneous)[..., :3]
