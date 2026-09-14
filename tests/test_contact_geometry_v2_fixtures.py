from __future__ import annotations

import numpy as np

from pipeline.contact_geometry_v2 import SurfaceSide, card_front_signed_clearance, measure_contact, oriented_box_sdf
from pipeline.robot_contact_geometry import finite_y_cylinder_sdf


CARD_DIMS = np.asarray([0.088, 0.063, 0.00030])
IDENTITY = np.eye(4)
PAD = {"hand_l_index_pad"}


def card_faces() -> np.ndarray:
    x, y, z = CARD_DIMS / 2
    front = np.asarray([[[-x, -y, z], [x, -y, z], [x, y, z]], [[-x, -y, z], [x, y, z], [-x, y, z]]])
    back = np.asarray([[[-x, -y, -z], [x, y, -z], [x, -y, -z]], [[-x, -y, -z], [-x, y, -z], [x, y, -z]]])
    return np.concatenate([front, back])


def horizontal(z: float) -> np.ndarray:
    return np.asarray([[[-0.003, -0.003, z], [0.003, -0.003, z], [0.0, 0.003, z]]])


def vertical(z0: float, z1: float, x: float = 0.0) -> np.ndarray:
    return np.asarray([[[x, -0.003, z0], [x, 0.003, z0], [x, 0.0, z1]]])


def measure(part: str, triangles: np.ndarray):
    return measure_contact(
        part_id=part, object_id="card_0", authorized_pad_parts=PAD,
        part_triangles_world=triangles, object_triangles_world=card_faces(),
        object_to_world=IDENTITY, dimensions_m=CARD_DIMS,
    )


def test_pad_two_mm_above_front_surface() -> None:
    value = measure("hand_l_index_pad", horizontal(CARD_DIMS[2] / 2 + 0.002))
    assert np.isclose(value.signed_distance_m, 0.002)
    assert value.nearest_surface == SurfaceSide.FRONT
    assert value.triangle_intersection_count == 0
    assert value.authorized_pad


def test_pad_exactly_touches_front_surface() -> None:
    value = measure("hand_l_index_pad", horizontal(CARD_DIMS[2] / 2))
    assert abs(value.signed_distance_m) < 1e-12
    assert value.nearest_surface == SurfaceSide.FRONT
    assert value.triangle_intersection_count >= 1


def test_pad_penetrates_one_mm_and_crosses_both_surfaces() -> None:
    point = np.asarray([[0.0, 0.0, CARD_DIMS[2] / 2 - 0.001]])
    one_mm = card_front_signed_clearance(point, IDENTITY, CARD_DIMS[2])[0]
    # A 1 mm directed front-face penetration passes through a real thin card;
    # closed SDF therefore becomes positive again while directed clearance and
    # triangle topology retain the error.
    assert np.isclose(one_mm, -0.001)
    assert oriented_box_sdf(point, IDENTITY, CARD_DIMS)[0] > 0
    value = measure("hand_l_index_pad", vertical(0.002, -0.002))
    assert value.triangle_intersection_count >= 2


def test_knuckle_hits_edge_but_is_not_authorized_pad() -> None:
    x = CARD_DIMS[0] / 2
    value = measure("hand_l_index_link2", vertical(0.002, -0.002, x=x))
    assert not value.authorized_pad
    assert value.triangle_intersection_count >= 1


def test_pad_on_back_is_directionally_distinct() -> None:
    value = measure("hand_l_index_pad", horizontal(-CARD_DIMS[2] / 2 - 0.002))
    assert np.isclose(value.signed_distance_m, 0.002)
    assert value.nearest_surface == SurfaceSide.BACK


def test_legal_pad_does_not_hide_nonpad_penetration() -> None:
    pad = measure("hand_l_index_pad", horizontal(CARD_DIMS[2] / 2 + 0.002))
    nonpad = measure("hand_l_index_link3", vertical(0.002, -0.002))
    assert pad.authorized_pad and pad.triangle_intersection_count == 0
    assert not nonpad.authorized_pad and nonpad.triangle_intersection_count >= 2


def test_three_chips_instances_are_measured_independently() -> None:
    points = np.asarray([[0.026, 0.0, 0.0], [0.050, 0.0, 0.0], [0.020, 0.0, 0.0]])
    distances = []
    centers = np.asarray([[0.0, 0.0, 0.0], [0.030, 0.0, 0.0], [0.080, 0.0, 0.0]])
    for center in centers:
        distances.append(float(finite_y_cylinder_sdf(points[:1] - center, 0.025, 0.08)[0]))
    assert np.allclose(distances, [0.001, -0.021, 0.029])
    assert len(set(np.round(distances, 6))) == 3
