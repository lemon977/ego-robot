import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pipeline.robot_wrist_kai_adapter import (
    FINAL_V3_MANO_FINGER_CHAINS,
    FINAL_V3_MANO_MCP_INDICES,
    FINAL_V3_MANO_TIP_INDICES,
    FINAL_V3_MANO_21_JOINT_NAMES,
    HUMANEGO_21_JOINT_NAMES,
    anatomical_basis_from_landmarks,
    anatomical_wrist_frame,
    aggregate_world_roots,
    WristKaiAdapterError,
    build_kai_root_transform,
    chordal_rotation_mean,
    fit_fixed_wrist_to_kai_rotation,
    fit_world_rig_from_bilateral_roots,
    final_v3_mano_palm_basis,
    final_v3_mano_unit_bones_local,
    named_joint,
    named_joint_index,
    resample_polyline_unit_directions,
    unit_bone_directions,
)


def test_final_v3_authoritative_mano_21_joint_order():
    assert len(FINAL_V3_MANO_21_JOINT_NAMES) == 21
    assert FINAL_V3_MANO_21_JOINT_NAMES[0] == "wrist"
    assert FINAL_V3_MANO_21_JOINT_NAMES[5] == "index_proximal"
    assert FINAL_V3_MANO_21_JOINT_NAMES[9] == "middle_proximal"
    assert FINAL_V3_MANO_21_JOINT_NAMES[13] == "ring_proximal"
    assert FINAL_V3_MANO_21_JOINT_NAMES[17] == "pinky_proximal"
    assert FINAL_V3_MANO_21_JOINT_NAMES[20] == "pinky_fingertip"
    assert FINAL_V3_MANO_MCP_INDICES == (2, 5, 9, 13, 17)
    assert FINAL_V3_MANO_TIP_INDICES == (4, 8, 12, 16, 20)
    assert FINAL_V3_MANO_FINGER_CHAINS["thumb"] == (0, 1, 2, 3, 4)
    assert FINAL_V3_MANO_FINGER_CHAINS["index"] == (0, 5, 6, 7, 8)


def test_humanego_authoritative_21_joint_order():
    assert len(HUMANEGO_21_JOINT_NAMES) == 21
    assert HUMANEGO_21_JOINT_NAMES[0] == "thumb_fingertip"
    assert HUMANEGO_21_JOINT_NAMES[5] == "wrist"
    assert HUMANEGO_21_JOINT_NAMES[8] == "index_proximal"
    assert HUMANEGO_21_JOINT_NAMES[11] == "middle_proximal"
    assert HUMANEGO_21_JOINT_NAMES[17] == "pinky_proximal"
    assert HUMANEGO_21_JOINT_NAMES[20] == "palm_center"


def test_named_wrist_is_not_assumed_to_be_joint_zero():
    points = np.asarray([[91.0, 0.0, 0.0], [7.0, 8.0, 9.0], [4.0, 5.0, 6.0]])
    names = ("thumb_fingertip", "wrist", "index_fingertip")
    assert named_joint_index(names, "wrist") == 1
    assert np.array_equal(named_joint(points, names, "wrist"), [7.0, 8.0, 9.0])


@pytest.mark.parametrize("names", [("thumb", "index"), ("wrist", "wrist")])
def test_named_wrist_rejects_missing_or_duplicate_identity(names):
    with pytest.raises(WristKaiAdapterError):
        named_joint_index(names, "wrist")


def test_named_joint_checks_joint_axis_length():
    with pytest.raises(WristKaiAdapterError):
        named_joint(np.zeros((4, 3)), ("wrist",), "wrist")


def _synthetic_mano() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    chains = FINAL_V3_MANO_FINGER_CHAINS
    bases = {
        "thumb": [0.8, 0.7, 0.0],
        "index": [0.6, 1.5, 0.0],
        "middle": [0.0, 1.7, 0.0],
        "ring": [-0.5, 1.5, 0.0],
        "pinky": [-0.9, 1.3, 0.0],
    }
    for finger, indices in chains.items():
        start = np.asarray(bases[finger], dtype=np.float64)
        for offset, index in enumerate(indices[1:]):
            points[index] = start + [0.03 * offset, 0.5 * offset, 0.1 * offset]
    return points


def test_final_v3_mano_basis_is_explicitly_wrist0_mcp5_9_17():
    points = _synthetic_mano()
    expected = anatomical_basis_from_landmarks(
        points[0], points[5], points[9], points[17], handedness="left"
    )
    assert np.allclose(final_v3_mano_palm_basis(points, handedness="left"), expected)
    with pytest.raises(WristKaiAdapterError):
        final_v3_mano_palm_basis(points[:20], handedness="left")


def test_unit_bone_retarget_features_ignore_translation_and_scale():
    points = _synthetic_mano()
    first = final_v3_mano_unit_bones_local(points, handedness="left")
    second = final_v3_mano_unit_bones_local(points * 7.5 + [9.0, -3.0, 2.0], handedness="left")
    assert first.keys() == second.keys()
    for finger in first:
        assert first[finger].shape == (4, 3)
        assert np.allclose(first[finger], second[finger], atol=1e-12, rtol=0)


def test_resampled_polyline_unit_directions_are_scale_free():
    points = np.asarray([[0, 0, 0], [0, 1, 0], [1, 2, 0], [1, 3, 1]], dtype=np.float64)
    first = resample_polyline_unit_directions(points, bone_count=4)
    second = resample_polyline_unit_directions(points * 0.03 + 20, bone_count=4)
    assert first.shape == (4, 3)
    assert np.allclose(first, second, atol=1e-12, rtol=0)
    assert np.allclose(np.linalg.norm(unit_bone_directions(points), axis=1), 1.0)


def test_anatomical_basis_uses_named_mcp_landmarks_and_is_right_handed():
    points = np.zeros((21, 3), dtype=np.float64)
    points[5] = [0.0, 0.0, 0.0]
    points[8] = [0.5, 2.0, 0.0]
    points[11] = [-0.5, 2.0, 0.0]
    points[17] = [-1.5, 2.0, 0.0]
    basis = anatomical_wrist_frame(
        points, HUMANEGO_21_JOINT_NAMES, handedness="left"
    )
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12, rtol=0)
    assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(basis[:, 0], [1.0, 0.0, 0.0], atol=1e-12, rtol=0)
    assert np.allclose(basis[:, 1], [0.0, 1.0, 0.0], atol=1e-12, rtol=0)
    assert np.allclose(basis[:, 2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0)


def test_anatomical_basis_batched_and_handed_lateral_standardizes_mirror():
    wrist = np.zeros((2, 3), dtype=np.float64)
    index = np.asarray([[0.5, 2.0, 0.0], [-0.5, 2.0, 0.0]])
    middle = np.asarray([[-0.5, 2.0, 0.0], [0.5, 2.0, 0.0]])
    pinky = np.asarray([[-1.5, 2.0, 0.0], [1.5, 2.0, 0.0]])
    left = anatomical_basis_from_landmarks(
        wrist[:1], index[:1], middle[:1], pinky[:1], handedness="left"
    )
    right = anatomical_basis_from_landmarks(
        wrist[1:], index[1:], middle[1:], pinky[1:], handedness="right"
    )
    assert left.shape == right.shape == (1, 3, 3)
    assert np.dot(left[0, :, 0], right[0, :, 0]) == pytest.approx(1.0, abs=1e-12)
    assert np.dot(left[0, :, 2], right[0, :, 2]) == pytest.approx(1.0, abs=1e-12)


def test_anatomical_basis_rejects_bad_handedness_and_degenerate_landmarks():
    with pytest.raises(WristKaiAdapterError):
        anatomical_basis_from_landmarks(
            np.zeros(3), np.ones(3), np.ones(3), -np.ones(3), handedness="unknown"
        )
    with pytest.raises(WristKaiAdapterError):
        anatomical_basis_from_landmarks(
            np.zeros(3), [1, 0, 0], [2, 0, 0], [-1, 0, 0], handedness="left"
        )


def test_fixed_rotation_fit_maps_kai_rays_to_human_rays():
    expected = Rotation.from_euler("xyz", [31.0, -17.0, 83.0], degrees=True).as_matrix()
    kai = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 2.0, 3.0]]
    )
    human = kai @ expected.T
    fitted, rms = fit_fixed_wrist_to_kai_rotation(human, kai)
    assert np.allclose(fitted, expected, atol=1e-9, rtol=0)
    assert rms < 1e-6


def test_root_composition_uses_exact_wrist_and_postmultiplies_adapter():
    human = Rotation.from_euler("z", 25.0, degrees=True).as_matrix()
    adapter = Rotation.from_euler("x", -40.0, degrees=True).as_matrix()
    wrist = np.asarray([0.1, -0.2, 0.9])
    result = build_kai_root_transform(wrist, human, adapter)
    assert np.array_equal(result[:3, 3], wrist)
    assert np.allclose(result[:3, :3], human @ adapter, atol=1e-12, rtol=0)
    assert np.array_equal(result[3], [0.0, 0.0, 0.0, 1.0])


def test_coordinate_median_and_chordal_mean_aggregate_bilateral_roots():
    roots = np.repeat(np.repeat(np.eye(4)[None, None], 3, axis=0), 2, axis=1)
    roots[:, 0, :3, 3] = [[1, 8, 3], [9, 2, 4], [3, 4, 5]]
    roots[:, 1, :3, 3] = [[-1, 1, 9], [-3, 2, 2], [-2, 7, 4]]
    rotations = Rotation.from_euler("z", [[10.0], [20.0], [30.0]], degrees=True).as_matrix()
    roots[:, 0, :3, :3] = rotations
    roots[:, 1, :3, :3] = rotations
    result = aggregate_world_roots(roots)
    assert np.array_equal(result[0, :3, 3], [3, 4, 4])
    assert np.array_equal(result[1, :3, 3], [-2, 2, 4])
    assert np.allclose(result[0, :3, :3], Rotation.from_euler("z", 20, degrees=True).as_matrix(), atol=1e-12)
    assert np.allclose(result[1, :3, :3], result[0, :3, :3], atol=1e-12)


def test_bilateral_rigid_fit_recovers_exact_session_transform():
    source = np.repeat(np.eye(4)[None], 2, axis=0)
    source[0, :3, 3] = [0.3, 0.1, -0.2]
    source[1, :3, 3] = [-0.3, 0.1, -0.2]
    source[0, :3, :3] = Rotation.from_euler("x", 12, degrees=True).as_matrix()
    source[1, :3, :3] = Rotation.from_euler("x", -12, degrees=True).as_matrix()
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("xyz", [21, -13, 44], degrees=True).as_matrix()
    expected[:3, 3] = [1.0, -2.0, 0.7]
    target = expected[None] @ source
    fitted = fit_world_rig_from_bilateral_roots(source, target)
    assert np.allclose(fitted, expected, atol=1e-12, rtol=0)


def test_chordal_mean_rejects_empty_input():
    with pytest.raises(WristKaiAdapterError):
        chordal_rotation_mean(np.empty((0, 3, 3)))
