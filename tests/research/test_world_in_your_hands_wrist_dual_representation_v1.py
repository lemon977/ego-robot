import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

from chaoyang.ops.build_wiyh_wrist_dual_representation_v1 import preflight, produce
from chaoyang.research.world_in_your_hands.wrist_dual_representation_v1 import (
    WristDualRepresentationError,
    admit_visible_wrist_surface,
    audit_suffix_invariance,
    bounded_nonaccumulating_fusion,
    compose_camera_wrist,
    fit_m1_translation,
    fit_m2_se3,
    invert_transform,
    lever_arm_residuals,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _pose(translation=(0.0, 0.0, 0.0), angle: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = _rotation_z(angle)
    value[:3, 3] = translation
    return value


def _controller_series(count: int) -> np.ndarray:
    result = np.stack(
        [_pose((0.01 * index, -0.002 * index, 0.55), angle=0.04 * index) for index in range(count)]
    )
    return result


def test_transform_direction_and_inverse_are_explicit() -> None:
    camera_controller = _pose((1.0, 2.0, 3.0), angle=np.pi / 2)
    controller_wrist = _pose((0.10, 0.0, 0.0), angle=-np.pi / 4)
    camera_wrist = compose_camera_wrist(camera_controller[None], controller_wrist)[0]
    np.testing.assert_allclose(camera_wrist[:3, 3], (1.0, 2.1, 3.0), atol=1e-10)
    np.testing.assert_allclose(
        invert_transform(camera_controller[None])[0] @ camera_wrist,
        controller_wrist,
        atol=1e-10,
    )


def test_pose_dependent_camera_residual_is_constant_controller_lever_arm() -> None:
    controller = _controller_series(20)
    nominal = _pose((0.02, -0.01, 0.04))
    actual = nominal.copy()
    actual[:3, 3] += (0.03, -0.02, 0.01)
    observed = compose_camera_wrist(controller, actual)[:, :3, 3]
    residual = lever_arm_residuals(controller, nominal, observed)
    assert np.std(residual["residual_camera"], axis=0).max() > 1e-3
    np.testing.assert_allclose(
        residual["residual_controller"],
        np.broadcast_to((0.03, -0.02, 0.01), (20, 3)),
        atol=1e-10,
    )
    fitted, evidence = fit_m1_translation(controller, observed, nominal)
    np.testing.assert_allclose(fitted, actual, atol=1e-10)
    assert evidence["fit_frames"] == 20


def test_m2_requires_orientation_evidence_and_recovers_static_se3() -> None:
    controller = _controller_series(20)
    actual = _pose((0.03, -0.02, 0.06), angle=0.18)
    observed = compose_camera_wrist(controller, actual)
    fitted, evidence = fit_m2_se3(
        controller,
        observed,
        minimum_orientation_samples=8,
        minimum_controller_rotation_span_deg=10.0,
    )
    np.testing.assert_allclose(fitted, actual, atol=1e-10)
    assert evidence["controller_rotation_span_deg"] > 10.0
    with pytest.raises(WristDualRepresentationError, match="rotation span"):
        fit_m2_se3(
            np.repeat(controller[:1], 20, axis=0),
            observed,
            minimum_orientation_samples=8,
            minimum_controller_rotation_span_deg=10.0,
        )


def test_surface_observation_fails_closed_to_region_only() -> None:
    admitted = admit_visible_wrist_surface(
        source_pixel_uv=(25.0, 30.0),
        surface_patch_xyz_camera=np.asarray(((0.0, 0.0, 0.5), (0.0, 0.01, 0.51))),
        rgb_wrist_region_valid=True,
        source_pixel_traceable=True,
        mask_purity_valid=True,
        depth_rgb_registration_valid=True,
        local_depth_continuity_valid=True,
        contaminant_free=True,
        surface_role="skin",
    )
    assert admitted.surface_point_valid
    np.testing.assert_allclose(admitted.point_camera, (0.0, 0.005, 0.505))
    blocked = admit_visible_wrist_surface(
        source_pixel_uv=(25.0, 30.0),
        surface_patch_xyz_camera=np.asarray(((0.0, 0.0, 0.5),)),
        rgb_wrist_region_valid=True,
        source_pixel_traceable=True,
        mask_purity_valid=False,
        depth_rgb_registration_valid=True,
        local_depth_continuity_valid=True,
        contaminant_free=True,
        surface_role="unknown",
    )
    assert not blocked.surface_point_valid
    assert blocked.region_registration_only
    assert blocked.blocker == "MASK_PURITY_INVALID"
    assert np.isnan(blocked.point_camera).all()


def test_bounded_fusion_is_nonaccumulating_and_uncertainty_is_unknown() -> None:
    prior = np.repeat(np.eye(4)[None], 3, axis=0)
    prior[:, 2, 3] = 0.5
    measured = np.asarray(((0.10, 0.0, 0.5), (0.10, 0.0, 0.5), (0.10, 0.0, 0.5)))
    result = bounded_nonaccumulating_fusion(
        prior, measured, np.asarray((True, True, True)), heuristic_weight=1.0, correction_bound_mm=30.0
    )
    np.testing.assert_allclose(result["fused_T_camera_wrist"][:, 0, 3], 0.03)
    assert result["correction_clipped"].all()
    assert set(result["uncertainty_status"]) == {"UNKNOWN"}
    # Identical measurements do not accumulate 30 mm per frame.
    np.testing.assert_allclose(np.diff(result["fused_T_camera_wrist"][:, :3, 3], axis=0), 0.0)


def test_suffix_invariance_marks_only_mismatching_field_noncausal() -> None:
    full = {
        "rgb": np.arange(5 * 3, dtype=np.uint8).reshape(5, 3),
        "state": np.linspace(0.0, 1.0, 10).reshape(5, 2),
    }
    prefix = {name: value[:3].copy() for name, value in full.items()}
    passed = audit_suffix_invariance(full, prefix, target_frame=2)
    assert passed["status"] == "PASS_SUFFIX_INVARIANCE"
    prefix["state"][-1, 0] += 1e-3
    failed = audit_suffix_invariance(full, prefix, target_frame=2)
    assert failed["fields"]["rgb"]["temporal_authority"] == "CAUSAL_CURRENT"
    assert failed["fields"]["state"]["temporal_authority"] == "OFFLINE_NONCAUSAL"


def _write_input(path: Path, frames: int = 12) -> None:
    controllers = np.stack([_controller_series(frames), _controller_series(frames)], axis=1)
    nominal = np.stack((_pose((0.01, 0.0, 0.05)), _pose((-0.01, 0.0, 0.05))))
    actual = nominal.copy()
    actual[0, :3, 3] += (0.015, -0.005, 0.002)
    actual[1, :3, 3] += (-0.012, 0.004, 0.003)
    observed = compose_camera_wrist(controllers, actual[None])
    patches = np.zeros((frames, 2, 3, 3), dtype=np.float64)
    patches[..., 2] = 0.5
    surface_uv = np.full((frames, 2, 2), (30.0, 24.0), dtype=np.float64)
    flags = np.ones((frames, 2), dtype=bool)
    np.savez_compressed(
        path,
        frame_id=np.arange(frames),
        timestamp_s=np.arange(frames) / 30.0,
        recording_id=np.asarray(
            ["session097" if frame < frames // 2 else "session098" for frame in range(frames)],
            dtype="U16",
        ),
        T_camera_controller_raw=controllers,
        T_controller_wrist_M0=nominal,
        observed_T_camera_wrist=observed,
        observed_valid=flags,
        orientation_valid=flags,
        surface_source_pixel_uv=surface_uv,
        surface_patch_xyz_camera=patches,
        rgb_wrist_region_valid=flags,
        source_pixel_traceable=flags,
        mask_purity_valid=flags,
        depth_rgb_registration_valid=flags,
        local_depth_continuity_valid=flags,
        contaminant_free=flags,
        surface_role=np.full((frames, 2), "skin", dtype="U16"),
        static_wrist_uv=surface_uv,
        observed_anatomical_wrist_uv=surface_uv,
        fused_wrist_uv=surface_uv,
    )


def test_producer_writes_schema_valid_numeric_and_video_outputs(tmp_path: Path) -> None:
    source = tmp_path / "input.npz"
    _write_input(source)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "chaoyang-wrist-dual-producer-config-v1",
                "fit_recording_ids": ["session097", "session098"],
                "forbidden_fit_recording_ids": ["session101", "session102", "session103"],
                "selected_model": {
                    "left": "M1_CONTROLLER_LOCAL_TRANSLATION",
                    "right": "M1_CONTROLLER_LOCAL_TRANSLATION",
                },
                "minimum_m2_orientation_samples": 8,
                "minimum_m2_controller_rotation_span_deg": 10.0,
                "fusion_heuristic_weight": 0.5,
                "development_numeric_correction_bound_mm": 30.0,
                "fps": 30.0,
                "review_image_domain": {
                    "name": "SYNTHETIC_SELECTED_RGB",
                    "width": 64,
                    "height": 48,
                    "projection": "PRECOMPUTED_SOURCE_UV",
                },
                "temporal_authority": {
                    "T_camera_controller_raw": "CAUSAL_CURRENT",
                    "observed_T_camera_wrist": "OFFLINE_NONCAUSAL",
                    "visible_wrist_surface_observation": "OFFLINE_NONCAUSAL",
                    "fused_T_camera_wrist": "OFFLINE_NONCAUSAL"
                },
            }
        ),
        encoding="utf-8",
    )
    config_schema = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/wrist_dual_producer_config_v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(json.loads(config.read_text(encoding="utf-8")), config_schema)
    preflight_result = preflight(input_npz=source, config_path=config)
    assert preflight_result["status"] == "PASS_CPU_PREFLIGHT"
    assert preflight_result["writes_performed"] is False
    raw = tmp_path / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    assert writer.isOpened()
    for frame in range(12):
        writer.write(np.full((48, 64, 3), frame * 10, dtype=np.uint8))
    writer.release()
    output = tmp_path / "output"
    result = produce(input_npz=source, config_path=config, output_root=output, raw_video=raw)
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/wrist_dual_representation_v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(result, schema)
    assert result["calibration"]["holdout_consumed"] is False
    assert result["fusion"]["uncertainty_status"] == "UNKNOWN"
    metrics = json.loads((output / "METRICS.json").read_text(encoding="utf-8"))
    assert set(metrics["calibration"]["sides"]["left"]["leave_one_recording_out"]) == {
        "session097",
        "session098",
    }
    with np.load(output / "WRIST_DUAL_REPRESENTATION_V1.npz", allow_pickle=False) as arrays:
        assert arrays["static_T_camera_wrist"].shape == (12, 2, 4, 4)
        assert arrays["fused_T_camera_wrist"].shape == (12, 2, 4, 4)
        assert set(arrays["uncertainty_status"].reshape(-1)) == {"UNKNOWN"}
    capture = cv2.VideoCapture(str(output / "WRIST_DUAL_REPRESENTATION_REVIEW.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    capture.release()
