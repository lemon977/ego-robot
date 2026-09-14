import numpy as np
import pytest

from pipeline.table_static_contact_support import (
    TagObservation,
    contact_observability,
    eroded_tag_support,
    intersect_pixel_with_tag_plane,
    tag_plane_camera,
    visible_object_bottom_pixel,
)


def observation(camera_x=0.0):
    transform = np.eye(4)
    transform[:3, 3] = [-camera_x, 0.0, 2.0]
    return TagObservation(
        0,
        np.array([[10, 10], [20, 10], [20, 20], [10, 20]], float),
        transform,
        0.1,
    )


def test_tag_plane_and_pixel_intersection():
    item = observation()
    plane = tag_plane_camera(item)
    assert np.array_equal(plane.normal, np.array([0.0, 0.0, 1.0]))
    assert plane.offset_m == pytest.approx(-2.0)
    xy, depth = intersect_pixel_with_tag_plane(np.array([0.0, 0.0]), np.eye(3), item)
    assert np.allclose(xy, [0.0, 0.0])
    assert depth == pytest.approx(2.0)


def test_support_and_visible_bottom_are_deterministic():
    support = eroded_tag_support((32, 32), observation().corners_uv, erosion_px=1)
    assert support[15, 15]
    assert not support[10, 10]
    mask = np.zeros((20, 20), bool)
    mask[3:11, 4:10] = True
    assert np.array_equal(visible_object_bottom_pixel(mask), [6.5, 10.0])


def test_contact_observability_fails_on_short_baseline_or_scatter():
    observations = [observation(0.0), observation(0.005), observation(0.01)]
    stable = np.zeros((3, 2))
    short = contact_observability(observations, stable)
    assert not short["baseline_sufficient"]
    assert not short["contact_evidence_valid"]

    frozen_centers = np.array([[0, 0, 0], [0.005, 0, 0], [0.01, 0, 0]], float)
    frozen = contact_observability(
        [observation(0.0), observation(0.03), observation(0.06)],
        stable,
        camera_centers_metric=frozen_centers,
    )
    assert frozen["metric_baseline_source"] == "FROZEN_C2W_CAMERA_CENTER"
    assert not frozen["baseline_sufficient"]

    observations = [observation(0.0), observation(0.02), observation(0.04)]
    scattered = contact_observability(observations, np.array([[0, 0], [0.02, 0], [0.04, 0]]))
    assert scattered["baseline_sufficient"]
    assert not scattered["plane_contact_hypothesis_consistent"]
    assert scattered["contact_state"] == "UNRESOLVED"
