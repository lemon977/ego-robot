from __future__ import annotations

import numpy as np

from tools.build_robot_corrected_contact_successor import triangle_box_sat


def test_triangle_box_sat_detects_inside_and_far_triangle() -> None:
    triangles = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0]],
            [[2.0, 2.0, 2.0], [2.2, 2.0, 2.0], [2.0, 2.2, 2.0]],
        ],
        dtype=np.float64,
    )
    assert triangle_box_sat(triangles, np.ones(3)).tolist() == [True, False]


def test_triangle_box_sat_detects_thin_crossing_without_inside_vertices() -> None:
    triangle = np.asarray(
        [[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        dtype=np.float64,
    )
    assert triangle_box_sat(triangle, np.asarray([1.0, 1.0, 0.001])).item()


def test_triangle_box_sat_rejects_edge_axis_separation() -> None:
    triangle = np.asarray(
        [[[1.1, 1.1, 0.9], [1.1, 0.9, 1.1], [0.9, 1.1, 1.1]]],
        dtype=np.float64,
    )
    assert not triangle_box_sat(triangle, np.ones(3)).item()
