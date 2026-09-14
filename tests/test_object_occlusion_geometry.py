import numpy as np
import pytest

from pipeline.depth_occlusion_v3 import DepthV3Error
from pipeline.object_occlusion_geometry import ObjectGeometry, project_object_depth


K = np.asarray([[100.0, 0.0, 15.5], [0.0, 100.0, 11.5], [0.0, 0.0, 1.0]])


def test_thin_card_box_projects_positive_finite_depth():
    pose = np.eye(4)
    pose[2, 3] = 1.0
    result = project_object_depth(
        ObjectGeometry("box_xyz", transform_object_to_camera=pose,
                       box_size_xyz_m=np.asarray([0.09, 0.064, 0.001])),
        K, 24, 32,
    )
    assert result.amodal_mask.any()
    assert np.all(result.near_m[result.amodal_mask] > 0.0)
    assert np.all(result.far_m[result.amodal_mask] >= result.near_m[result.amodal_mask])


def test_unknown_shape_fails_closed():
    with pytest.raises(DepthV3Error):
        project_object_depth(ObjectGeometry("capsule"), K, 24, 32)  # type: ignore[arg-type]


def test_mask_depth_requires_matching_validity():
    near = np.full((48, 64), np.nan)
    far = near.copy()
    near[10, 10] = 0.7
    with pytest.raises(DepthV3Error):
        project_object_depth(ObjectGeometry("mask_depth", near_depth_m_2x=near,
                                            far_depth_m_2x=far), K, 24, 32)
