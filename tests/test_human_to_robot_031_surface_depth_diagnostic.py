import numpy as np

from chaoyang.ops.run_human_to_robot_031_surface_depth_diagnostic import (
    FRAMES, admissible_pixels, rasterize_optical_z,
)


def test_perspective_nearest_surface_not_back_face():
    k = np.array([[10., 0., 0.], [0., 10., 0.], [0., 0., 1.]])
    near = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    far = near * 2
    vertices = np.vstack((near, far))
    faces = np.array([[3, 4, 5], [0, 1, 2]])
    z = rasterize_optical_z(vertices, faces, k, (12, 12))
    assert np.isclose(z[2, 2], 1.0)
    assert np.isinf(z[11, 11])


def test_missing_depth_and_mask_boundary_not_admitted():
    model = np.full((480, 640), 0.4, np.float32)
    depth = np.full((480, 640), 0.35, np.float32)
    mask = np.zeros((960, 1280), bool)
    mask[100:150, 100:150] = True
    stereo = {"depth_m": depth, "valid": np.ones_like(depth, bool),
              "lr_consistent": np.ones_like(depth, bool)}
    valid = admissible_pixels(model, stereo, mask)
    assert valid[62, 62]
    assert not valid[50, 50]
    depth[62, 62] = np.nan
    assert not admissible_pixels(model, stereo, mask)[62, 62]


def test_frame_selection_precedes_result_and_preserves_denominator():
    assert len(FRAMES) == 16
    assert FRAMES[0] == 47 and FRAMES[-1] == 148
    assert len(set(FRAMES)) == len(FRAMES)
