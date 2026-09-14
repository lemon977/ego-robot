from __future__ import annotations

"""Deterministic metric contact geometry used by exact78 V5.2 fixtures.

All values are metres.  Card geometry is a closed thin box, so front and back
surfaces are distinct.  Part and object IDs are inputs to every audit row and
are never inferred from proximity.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from pipeline.robot_contact_geometry import triangle_triangle_intersects_sat


class SurfaceSide(StrEnum):
    FRONT = "FRONT"
    BACK = "BACK"
    EDGE = "EDGE"
    INSIDE = "INSIDE"


@dataclass(frozen=True)
class ContactMeasurement:
    part_id: str
    object_id: str
    authorized_pad: bool
    signed_distance_m: float
    nearest_surface: SurfaceSide
    triangle_intersection_count: int


def _points(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("points must be finite [N,3]")
    return result


def oriented_box_sdf(points_world: np.ndarray, object_to_world: np.ndarray, dimensions_m: np.ndarray) -> np.ndarray:
    """Exact signed distance to a closed oriented box."""
    points = _points(points_world)
    transform = np.asarray(object_to_world, dtype=np.float64)
    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("object_to_world must be finite [4,4]")
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("dimensions_m must be positive [3]")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("object_to_world must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or np.linalg.det(rotation) <= 0:
        raise ValueError("object rotation must be right-handed orthonormal")
    local = (points - transform[:3, 3]) @ rotation
    delta = np.abs(local) - dimensions[None] * 0.5
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.minimum(np.max(delta, axis=1), 0.0)
    return outside + inside


def nearest_box_surface(points_world: np.ndarray, object_to_world: np.ndarray, dimensions_m: np.ndarray) -> np.ndarray:
    points = _points(points_world)
    transform = np.asarray(object_to_world, dtype=np.float64)
    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    normalized = np.abs(local) / (dimensions[None] * 0.5)
    axis = np.argmax(normalized, axis=1)
    labels = np.full(len(points), SurfaceSide.EDGE, dtype=object)
    labels[(axis == 2) & (local[:, 2] >= 0)] = SurfaceSide.FRONT
    labels[(axis == 2) & (local[:, 2] < 0)] = SurfaceSide.BACK
    inside = np.all(np.abs(local) < dimensions[None] * 0.5, axis=1)
    labels[inside] = SurfaceSide.INSIDE
    return labels


def card_front_signed_clearance(points_world: np.ndarray, object_to_world: np.ndarray, thickness_m: float) -> np.ndarray:
    """Directed clearance to the card front plane.

    Positive is above the front face, zero touches it, and negative has moved
    inward through the front face.  This remains negative after crossing the
    thin back face; the closed-box SDF and triangle intersections separately
    distinguish inside-card from through-card topology.
    """
    points = _points(points_world)
    transform = np.asarray(object_to_world, dtype=np.float64)
    thickness = float(thickness_m)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("object_to_world must be finite [4,4]")
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("thickness_m must be positive")
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    return local[:, 2] - thickness * 0.5


def transform_triangles(triangles_local: np.ndarray, local_to_world: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles_local, dtype=np.float64)
    transform = np.asarray(local_to_world, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError("triangles must be finite [N,3,3]")
    if transform.shape != (4, 4):
        raise ValueError("local_to_world must be [4,4]")
    return triangles @ transform[:3, :3].T + transform[:3, 3]


def count_triangle_intersections(first: np.ndarray, second: np.ndarray) -> int:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    return sum(triangle_triangle_intersects_sat(x, y) for x in a for y in b)


def measure_contact(
    *,
    part_id: str,
    object_id: str,
    authorized_pad_parts: set[str],
    part_triangles_world: np.ndarray,
    object_triangles_world: np.ndarray,
    object_to_world: np.ndarray,
    dimensions_m: np.ndarray,
) -> ContactMeasurement:
    triangles = np.asarray(part_triangles_world, dtype=np.float64)
    samples = triangles.reshape(-1, 3)
    distances = oriented_box_sdf(samples, object_to_world, dimensions_m)
    nearest_index = int(np.argmin(np.abs(distances)))
    sides = nearest_box_surface(samples, object_to_world, dimensions_m)
    return ContactMeasurement(
        part_id=part_id,
        object_id=object_id,
        authorized_pad=part_id in authorized_pad_parts,
        signed_distance_m=float(distances[nearest_index]),
        nearest_surface=SurfaceSide(sides[nearest_index]),
        triangle_intersection_count=count_triangle_intersections(triangles, object_triangles_world),
    )
