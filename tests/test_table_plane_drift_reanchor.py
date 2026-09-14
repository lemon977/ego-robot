import numpy as np
import pytest

from pipeline.table_plane_drift_reanchor import (
    Plane,
    ReprojectionFrame,
    TablePlaneError,
    fit_axis_constrained_plane,
    finite_cylinder_sdf,
    leave_one_out_reprojection,
    reanchor_normal_offset,
    signed_distances,
    tangential_coordinates,
    wrist_table_physical_gate,
)


def test_axis_constrained_fit_and_reanchor_are_normal_only():
    normal = np.array([0.0, 1.0, 0.0])
    points = np.array([[0.0, 2.0, 0.0], [1.0, 2.01, 1.0], [-1.0, 1.99, 2.0]])
    plane = fit_axis_constrained_plane(points, np.tile(normal, (3, 1)))
    assert np.allclose(plane.normal, normal)
    assert plane.offset_m == pytest.approx(-2.0)

    anchor = np.array([4.0, 2.25, -3.0])
    before_uv = tangential_coordinates(plane.normal, anchor)
    shifted = reanchor_normal_offset(plane, anchor, contact_verified=True)
    assert np.array_equal(shifted.normal, plane.normal)
    assert signed_distances(shifted, anchor) == pytest.approx(0.0)
    assert np.array_equal(tangential_coordinates(shifted.normal, anchor), before_uv)


def test_reanchor_fails_closed_without_independent_contact():
    with pytest.raises(TablePlaneError, match="contact"):
        reanchor_normal_offset(
            Plane(np.array([0.0, 1.0, 0.0]), -1.0),
            np.array([0.0, 1.2, 0.0]),
            contact_verified=False,
        )


def test_leave_one_out_identity_camera_and_plane_have_zero_error():
    height, width = 24, 32
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    rgb = np.stack(
        (np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width)), np.full((height, width), 80, np.uint8)),
        axis=2,
    )
    intrinsic = np.array([[20.0, 0.0, 15.5], [0.0, 20.0, 11.5], [0.0, 0.0, 1.0]])
    frames = [
        ReprojectionFrame(rgb.copy(), np.ones((height, width), bool), intrinsic, np.eye(4))
        for _ in range(3)
    ]
    plane = Plane(np.array([0.0, 0.0, 1.0]), -2.0)
    metrics = leave_one_out_reprojection(
        frames, [plane] * 3, grid_columns=16, grid_rows=12, min_donors=1
    )
    assert all(row["coverage_ratio"] == pytest.approx(1.0) for row in metrics)
    assert all(row["mae_rgb_0_255"] == pytest.approx(0.0) for row in metrics)


def test_plane_validates_unit_normal():
    with pytest.raises(TablePlaneError):
        Plane(np.array([0.0, 2.0, 0.0]), 0.0)


def test_physical_gates_do_not_accept_confidence_as_geometry():
    plane = Plane(np.array([0.0, 1.0, 0.0]), 0.0)
    result = wrist_table_physical_gate(
        plane,
        np.array([[0.0, -0.1, 0.0], [0.0, 0.2, 0.0]]),
        np.array([0.855, 0.99]),
    )
    assert result["wrist_below_table"].tolist() == [True, False]
    assert result["high_confidence_below_table"].tolist() == [True, False]

    identity = np.eye(4)
    sdf = finite_cylinder_sdf(
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.6, 0.0]]),
        identity,
        radius_m=0.1,
        height_m=1.0,
    )
    assert sdf[0] == pytest.approx(-0.1)
    assert sdf[1] == pytest.approx(0.1)
    assert sdf[2] == pytest.approx(0.1)
