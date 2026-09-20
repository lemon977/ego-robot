from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chaoyang.ops import build_exact78_wrist_comparison_v31 as subject


def _dual_result(npz: Path) -> dict:
    reference = subject.artifact(npz)
    placeholder = {"path": "/input", "bytes": 1, "sha256": "a" * 64}
    return {
        "schema_version": "chaoyang-wrist-dual-representation-v1",
        "status": "DEVELOPMENT_RESULT_GENERATED",
        "transform_convention": {
            "notation": "T_A_B_MAPS_B_TO_A",
            "composition": "T_camera_wrist=T_camera_controller@T_controller_wrist",
            "saved_transform_shape": "[T,2,4,4]",
        },
        "representations": {
            "anatomical_wrist_center": "MODEL_DEFINED_KINEMATIC_WRIST_NOT_EXTERNAL_ANATOMICAL_TRUTH",
            "visible_wrist_surface": "VIEW_DEPENDENT_OBSERVATION_NEVER_JOINT_SUBSTITUTE",
        },
        "calibration": {
            "models": ["M0_LEGACY", "M1_CONTROLLER_LOCAL_TRANSLATION", "M2_STATIC_SE3"],
            "selected_model": {"left": "M1_CONTROLLER_LOCAL_TRANSLATION", "right": "M0_LEGACY"},
            "fit_roles": ["development"],
            "holdout_consumed": False,
        },
        "fusion": {
            "nonaccumulating": True,
            "static_prior_preserved": True,
            "development_numeric_correction_bound_mm": 30.0,
            "uncertainty_status": "UNKNOWN",
        },
        "temporal_authority": {"observed_wrist": "OFFLINE_NONCAUSAL"},
        "inputs": {"bundle": placeholder, "config": placeholder, "raw_video": None},
        "outputs": {"npz": reference, "metrics": placeholder, "video": None},
        "authority": {
            "control_ground_truth": False,
            "external_wrist_truth": False,
            "external_metric_authority": False,
            "physical_deployment_authorized": False,
        },
    }


def test_exact_adapter_consumes_the_shared_dual_wrist_authority(tmp_path: Path) -> None:
    dual = tmp_path / "WRIST_DUAL_REPRESENTATION_V1.npz"
    np.savez_compressed(dual, marker=np.asarray([1]))
    result = tmp_path / "RESULT.json"
    result.write_text(json.dumps(_dual_result(dual)), encoding="utf-8")
    validated = subject.validate_dual_wrist_result(result, dual.resolve())
    assert validated["authority"]["external_wrist_truth"] is False


def test_mixed_surface_center_distances_are_never_named_wrist_error() -> None:
    transforms = np.broadcast_to(np.eye(4), (3, 2, 4, 4)).copy()
    observed = transforms.copy()
    observed[..., 0, 3] = 0.010
    fused = transforms.copy()
    fused[..., 0, 3] = 0.005
    surface = np.full((3, 2, 3), (0.0, 0.020, 0.5), np.float64)
    product = {
        "static_T_camera_wrist": transforms,
        "observed_T_camera_wrist": observed,
        "observed_valid": np.ones((3, 2), bool),
        "fused_T_camera_wrist": fused,
        "fusion_valid": np.ones((3, 2), bool),
        "visible_wrist_surface_point_camera": surface,
        "visible_wrist_surface_valid": np.ones((3, 2), bool),
        "visible_wrist_region_registration_only": np.zeros((3, 2), bool),
        "correction_clipped": np.zeros((3, 2), bool),
    }
    metrics = subject.comparison_metrics(product)
    left = metrics["sides"]["left"]
    assert left["tracker_vs_hawor_anatomical_center_distance_mm"]["mean"] == 10.0
    assert "surface_to_tracker_mixed_semantic_distance_mm" in left
    assert not any("surface" in key and "error" in key for key in left)


def test_world_transform_keeps_nan_surface_unknown_and_moves_finite_points() -> None:
    transforms = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    transforms[:, 0, 3] = 1.0
    points = np.zeros((2, 2, 3), np.float64)
    points[1, 1] = np.nan
    world = subject.transform_camera_points_to_world(transforms, points)
    assert np.allclose(world[0, :, 0], 1.0)
    assert np.isnan(world[1, 1]).all()
