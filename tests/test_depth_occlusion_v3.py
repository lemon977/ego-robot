import io
from pathlib import Path

import numpy as np
import pytest

from pipeline.depth_occlusion_v3 import (
    OCCLUSION_EPSILON_M,
    SOURCE_OBJECT,
    SOURCE_ROBOT,
    DepthV3Error,
    Object6DCylinderFrame,
    compose_s8_frame_bundle,
    compose_depth_v3,
    decode_s8_frame_bundle,
    load_object6d_cylinder_frame,
    project_amodal_cylinder_depth,
    supersampled_intrinsics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECT6D_004 = PROJECT_ROOT / (
    "data/benchmarks/hand/production_runs/final_v3_grap_a_cap_0812/grap_a_cap_004/"
    "10_object6d_v2/v2_cylinder_center_rotation_gated/object_6dof_v2.npz"
)


def _range_for_z(z: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = z.shape
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    norm = np.sqrt(
        1.0
        + ((u - intrinsics[0, 2]) / intrinsics[0, 0]) ** 2
        + ((v - intrinsics[1, 2]) / intrinsics[1, 1]) ** 2
    )
    return z * norm


def _compose_inputs(height: int, width: int) -> dict:
    high = (height * 2, width * 2)
    intrinsics = np.array(
        [[20.0, 0.0, high[1] / 2 - 0.5], [0.0, 20.0, high[0] / 2 - 0.5], [0, 0, 1]],
        dtype=np.float64,
    )
    return {
        "clean_rgb": np.zeros((height, width, 3), dtype=np.uint8),
        "object_rgb_2x": np.full((*high, 3), (200, 0, 0), dtype=np.uint8),
        "object_texture_source": "raw_observation",
        "object_depth_near_m_2x": np.full(high, np.nan),
        "object_depth_far_m_2x": np.full(high, np.nan),
        "robot_rgb_2x": np.full((*high, 3), (0, 0, 200), dtype=np.uint8),
        "robot_range_m_2x": _range_for_z(np.ones(high), intrinsics),
        "robot_alpha_2x": np.zeros(high, dtype=np.float64),
        "robot_intrinsics_2x": intrinsics,
    }


def _frame_bundle_bytes(session_id: str = "grap_a_cap_004") -> bytes:
    height, width = 2, 3
    high = (height * 2, width * 2)
    camera = np.array([[10.0, 0, 1.0], [0, 10.0, 0.5], [0, 0, 1.0]])
    buffer = io.BytesIO()
    np.savez(
        buffer,
        schema_version=np.asarray("s8-depth-occlusion-v3-frame-bundle-v1"),
        session_id=np.asarray(session_id),
        frame_name=np.asarray("00000"),
        frame_index=np.asarray(0, dtype=np.int64),
        object_texture_source=np.asarray("raw_observation"),
        clean_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        object_rgb_2x=np.full((*high, 3), 200, dtype=np.uint8),
        robot_rgb_2x=np.full((*high, 3), 100, dtype=np.uint8),
        robot_range_m_2x=_range_for_z(np.full(high, 2.0), supersampled_intrinsics(camera)),
        robot_alpha_2x=np.ones(high, dtype=np.float32),
        camera_intrinsics=camera,
        object_index_2x=np.ones(high, dtype=np.int16),
    )
    return buffer.getvalue()


def test_analytic_closed_y_cylinder_produces_amodal_near_and_far() -> None:
    transform = np.eye(4)
    transform[2, 3] = 1.0
    intrinsics = np.array([[1e9, 0, 0], [0, 1e9, 0], [0, 0, 1]], dtype=float)
    depth = project_amodal_cylinder_depth(transform, 0.2, 0.6, intrinsics, 1, 1)
    assert depth.amodal_mask.shape == (2, 2)
    assert depth.amodal_mask.all()
    assert np.allclose(depth.near_m, 0.8, atol=1e-8)
    assert np.allclose(depth.far_m, 1.2, atol=1e-8)


def test_frozen_004_frame_240_projects_complete_cylinder() -> None:
    frame = load_object6d_cylinder_frame(OBJECT6D_004, 240)
    intrinsics = np.array(
        [[160.0, 0, 159.5], [0, 160.0, 119.5], [0, 0, 1]], dtype=float
    )
    depth = project_amodal_cylinder_depth(
        frame.transform_object_to_camera,
        frame.radius_m,
        frame.height_m,
        intrinsics,
        240,
        320,
    )
    assert depth.amodal_mask.any()
    assert np.all(depth.near_m[depth.amodal_mask] > 0.0)
    assert np.all(depth.far_m[depth.amodal_mask] >= depth.near_m[depth.amodal_mask])


def test_frozen_three_mm_rule_selects_front_source_on_both_sides() -> None:
    assert OCCLUSION_EPSILON_M == 0.003
    inputs = _compose_inputs(1, 2)
    inputs["object_depth_near_m_2x"][:] = 1.0
    inputs["object_depth_far_m_2x"][:] = 1.1
    inputs["robot_alpha_2x"][:] = 1.0
    robot_z = np.array([[0.996, 0.996, 1.004, 1.004]] * 2)
    inputs["robot_range_m_2x"] = _range_for_z(
        robot_z, inputs["robot_intrinsics_2x"]
    )
    result = compose_depth_v3(**inputs)
    assert np.all(result.source_map_2x[:, :2] == SOURCE_ROBOT)
    assert np.all(result.source_map_2x[:, 2:] == SOURCE_OBJECT)
    assert result.rgb.tolist() == [[[0, 0, 200], [200, 0, 0]]]


def test_two_by_two_edge_fraction_controls_final_alpha_and_rgb() -> None:
    inputs = _compose_inputs(1, 1)
    inputs["object_depth_near_m_2x"][0, 0] = 0.8
    inputs["object_depth_far_m_2x"][0, 0] = 1.2
    result = compose_depth_v3(**inputs)
    assert result.object_alpha[0, 0] == 0.25
    assert result.clean_bg_alpha[0, 0] == 0.75
    assert result.robot_alpha[0, 0] == 0.0
    assert result.rgb[0, 0].tolist() == [50, 0, 0]
    assert np.allclose(result.source_fractions.sum(axis=-1), 1.0)


def test_robot_fully_behind_analytic_cylinder_is_invisible() -> None:
    height, width = 12, 18
    transform = np.eye(4)
    transform[2, 3] = 1.0
    intrinsics = np.array([[18.0, 0, 8.5], [0, 18.0, 5.5], [0, 0, 1]], dtype=float)
    depth = project_amodal_cylinder_depth(
        transform, 0.25, 0.6, intrinsics, height, width
    )
    inputs = _compose_inputs(height, width)
    inputs["object_depth_near_m_2x"] = depth.near_m
    inputs["object_depth_far_m_2x"] = depth.far_m
    inputs["robot_alpha_2x"] = depth.amodal_mask.astype(float)
    robot_z = np.full(depth.near_m.shape, 1.5)
    inputs["robot_range_m_2x"] = _range_for_z(
        robot_z, inputs["robot_intrinsics_2x"]
    )
    result = compose_depth_v3(**inputs)
    assert np.count_nonzero(inputs["robot_alpha_2x"]) > 0
    assert np.count_nonzero(result.source_map_2x == SOURCE_ROBOT) == 0


def test_object_index_is_qa_only_and_cannot_change_decision() -> None:
    inputs = _compose_inputs(1, 1)
    inputs["object_depth_near_m_2x"][:] = 0.8
    inputs["object_depth_far_m_2x"][:] = 1.2
    inputs["robot_alpha_2x"][:] = 1.0
    without_roles = compose_depth_v3(**inputs)
    with_roles = compose_depth_v3(
        **inputs, object_index_2x=np.array([[0, 7], [99, 0]], dtype=np.int32)
    )
    assert np.array_equal(without_roles.rgb, with_roles.rgb)
    assert np.array_equal(without_roles.source_map_2x, with_roles.source_map_2x)
    assert with_roles.object_index_qa == {"nonzero_subpixels": 2, "unique_roles": 3}


def test_cad_material_texture_and_invalid_depth_fail_closed() -> None:
    inputs = _compose_inputs(1, 1)
    inputs["object_texture_source"] = "cad_material"
    with pytest.raises(DepthV3Error, match="RAW observation or an object donor"):
        compose_depth_v3(**inputs)
    inputs = _compose_inputs(1, 1)
    inputs["robot_alpha_2x"][:] = 1.0
    inputs["robot_range_m_2x"][:] = np.nan
    with pytest.raises(DepthV3Error, match="robot Range is invalid"):
        compose_depth_v3(**inputs)


def test_frame_bundle_decodes_and_composes_with_exact_frozen_rule() -> None:
    bundle = decode_s8_frame_bundle(_frame_bundle_bytes())
    transform = np.eye(4)
    transform[2, 3] = 1.0
    object6d = Object6DCylinderFrame(0, transform, 0.2, 0.6, 1.0)
    result = compose_s8_frame_bundle(bundle, object6d)
    assert result.rgb.shape == (2, 3, 3)
    assert np.any(result.source_map_2x == SOURCE_OBJECT)
    assert result.object_index_qa is not None


@pytest.mark.parametrize(
    "session_id",
    ["grap_a_cap_025", "blind_partition", "processed_004", "labels_development"],
)
def test_frame_bundle_forbidden_population_fails_closed(session_id: str) -> None:
    with pytest.raises(DepthV3Error, match="forbidden"):
        decode_s8_frame_bundle(_frame_bundle_bytes(session_id))


def test_frame_bundle_rejects_hidden_extra_arrays_before_use() -> None:
    source = np.load(io.BytesIO(_frame_bundle_bytes()), allow_pickle=False)
    try:
        values = {name: source[name] for name in source.files}
    finally:
        source.close()
    values["labels"] = np.ones((2, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    with pytest.raises(DepthV3Error, match="key closure differs"):
        decode_s8_frame_bundle(buffer.getvalue())


def test_integer_rgb_outside_byte_range_fails_closed() -> None:
    inputs = _compose_inputs(1, 1)
    inputs["robot_rgb_2x"] = np.full((2, 2, 3), 256, dtype=np.int16)
    with pytest.raises(DepthV3Error, match=r"values must be in \[0,255\]"):
        compose_depth_v3(**inputs)


def test_supersampled_intrinsics_preserve_pixel_centres() -> None:
    base = np.array([[100.0, 0, 9.5], [0, 120.0, 7.5], [0, 0, 1.0]])
    high = supersampled_intrinsics(base)
    assert high.tolist() == [[200.0, 0.0, 19.5], [0.0, 240.0, 15.5], [0.0, 0.0, 1.0]]
