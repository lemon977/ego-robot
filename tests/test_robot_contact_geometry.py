from __future__ import annotations

import numpy as np
import pytest

from pipeline.robot_contact_geometry import (
    deterministic_triangle_samples,
    finite_y_cylinder_regions,
    finite_y_cylinder_sdf,
    legacy_finite_z_cylinder_sdf_canary_only,
    select_kaihand_pad_patch,
    triangle_triangle_intersects_sat,
)


RADIUS = 0.025
HEIGHT = 0.080


def test_finite_y_axis_is_distinguished_from_x_and_z() -> None:
    points = np.asarray(
        [
            [0.035, 0.000, 0.000],  # radial X: 10 mm outside
            [0.000, 0.045, 0.000],  # canonical Y cap: 5 mm outside
            [0.000, 0.000, 0.035],  # radial Z: 10 mm outside
            [0.000, 0.035, 0.000],  # inside, 5 mm below Y cap
        ]
    )
    actual = finite_y_cylinder_sdf(points, RADIUS, HEIGHT)
    np.testing.assert_allclose(actual, [0.010, 0.005, 0.010, -0.005], atol=1e-12)


def test_side_cap_corner_inside_and_surface_regions() -> None:
    points = np.asarray(
        [
            [0.035, 0.000, 0.000],
            [0.000, 0.045, 0.000],
            [0.035, 0.045, 0.000],
            [0.020, 0.000, 0.000],
            [0.000, 0.038, 0.000],
            [0.025, 0.000, 0.000],
            [0.000, 0.040, 0.000],
        ]
    )
    np.testing.assert_array_equal(
        finite_y_cylinder_regions(points, RADIUS, HEIGHT),
        [
            "outside_side",
            "outside_cap",
            "outside_corner",
            "inside_side_nearest",
            "inside_cap_nearest",
            "surface_side",
            "surface_cap",
        ],
    )
    np.testing.assert_allclose(
        finite_y_cylinder_sdf(points, RADIUS, HEIGHT),
        [0.010, 0.005, np.hypot(0.010, 0.005), -0.005, -0.002, 0.0, 0.0],
        atol=1e-12,
    )


def test_old_wrong_axis_equation_is_a_failing_canary() -> None:
    # The old helper interpreted these as a Z-axis cylinder.  The pair forces
    # opposite inside/outside decisions and would fail if production regressed.
    points = np.asarray([[0.000, 0.035, 0.000], [0.000, 0.000, 0.035]])
    correct = finite_y_cylinder_sdf(points, RADIUS, HEIGHT)
    legacy = legacy_finite_z_cylinder_sdf_canary_only(points, RADIUS, HEIGHT)
    np.testing.assert_allclose(correct, [-0.005, 0.010], atol=1e-12)
    np.testing.assert_allclose(legacy, [0.010, -0.005], atol=1e-12)
    assert np.sign(correct).tolist() == [-1.0, 1.0]
    assert np.sign(legacy).tolist() == [1.0, -1.0]


@pytest.mark.parametrize(
    ("points", "radius", "height"),
    [
        (np.zeros((2, 2)), RADIUS, HEIGHT),
        (np.asarray([[np.nan, 0.0, 0.0]]), RADIUS, HEIGHT),
        (np.zeros((1, 3)), 0.0, HEIGHT),
        (np.zeros((1, 3)), RADIUS, -1.0),
    ],
)
def test_sdf_rejects_invalid_inputs(points: np.ndarray, radius: float, height: float) -> None:
    with pytest.raises(ValueError):
        finite_y_cylinder_sdf(points, radius, height)


def test_triangle_lattice_respects_max_cell_edge_and_is_deterministic() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.0, 0.005, 0.0]])
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    first = deterministic_triangle_samples(vertices, faces, 0.002)
    second = deterministic_triangle_samples(vertices, faces, 0.002)
    np.testing.assert_array_equal(first, second)
    # ceil(5*sqrt(2)/2)=4 subdivisions => 15 barycentric lattice points.
    assert first.shape == (15, 3)
    assert any(np.allclose(point, vertices[0]) for point in first)
    assert any(np.allclose(point, vertices[1]) for point in first)
    assert any(np.allclose(point, vertices[2]) for point in first)


def _box_with_oriented_bottom_faces() -> tuple[np.ndarray, np.ndarray]:
    # Synthetic distal finger shell: two triangles have normal -Y and centroids
    # in the distal -Z portion selected for a left index fingertip.
    vertices = np.asarray(
        [
            [-1.0, 0.0, -2.0],
            [1.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [3, 4, 5]], dtype=np.int64)
    return vertices, faces


def test_pad_authority_is_link_local_and_not_object_selected() -> None:
    vertices, faces = _box_with_oriented_bottom_faces()
    patch = select_kaihand_pad_patch("hand_l_index_link4", "left", vertices, faces)
    np.testing.assert_array_equal(patch.face_indices, [0, 1])
    assert np.all(patch.face_normals[:, 1] < -0.99)
    # Object/world coordinates are not accepted by the API; translating the
    # entire local mesh preserves the same anatomical face identities.
    moved = select_kaihand_pad_patch(
        "hand_l_index_link4", "left", vertices + [13.0, -4.0, 7.0], faces
    )
    np.testing.assert_array_equal(moved.face_indices, patch.face_indices)


def test_pad_authority_rejects_wrong_chirality_or_non_pad_link() -> None:
    vertices, faces = _box_with_oriented_bottom_faces()
    with pytest.raises(ValueError, match="chirality disagree"):
        select_kaihand_pad_patch("hand_l_index_link4", "right", vertices, faces)
    with pytest.raises(ValueError, match="not an authoritative"):
        select_kaihand_pad_patch("hand_l_index_link3", "left", vertices, faces)


def test_triangle_sat_crossing_and_separated_3d() -> None:
    horizontal = np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    crossing = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.0]])
    separated = horizontal + [0.0, 0.0, 2.0]
    assert triangle_triangle_intersects_sat(horizontal, crossing)
    assert not triangle_triangle_intersects_sat(horizontal, separated)


def test_triangle_sat_handles_coplanar_overlap_and_separation() -> None:
    first = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    overlap = np.asarray([[0.25, 0.25, 0.0], [1.0, 0.25, 0.0], [0.25, 1.0, 0.0]])
    separated = overlap + [3.0, 0.0, 0.0]
    assert triangle_triangle_intersects_sat(first, overlap)
    assert not triangle_triangle_intersects_sat(first, separated)


def test_triangle_sat_rejects_degenerate_input() -> None:
    point = np.zeros((3, 3))
    triangle = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="degenerate"):
        triangle_triangle_intersects_sat(point, triangle)
