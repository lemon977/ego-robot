"""CPU geometry gates for Robot/KaiHand object contact.

The Object6D cylinder used by the ``grap_a_cap`` sessions is expressed in its
own canonical frame with its axis along **Y**.  Keeping this convention in a
small, tested module prevents the old X/Y-radial (canonical-Z) canary from
silently returning physically wrong contact labels.

This module deliberately has no renderer or CUDA dependency.  Its contact-pad
selector consumes only a pinned KaiHand mesh in link-local coordinates; object
position is not an input, so the selected anatomical surface cannot drift to
whichever CAD vertex happens to be closest to an object in a particular run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


CylinderRegion = Literal[
    "inside_side_nearest",
    "inside_cap_nearest",
    "surface_side",
    "surface_cap",
    "outside_side",
    "outside_cap",
    "outside_corner",
]


def _points_nx3(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(result).all():
        raise ValueError("points must be finite")
    return result


def _positive_dimension(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def finite_y_cylinder_components(
    points: np.ndarray, radius: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed radial and axial excess for a finite Y-axis cylinder.

    ``radial`` is ``sqrt(x**2 + z**2) - radius`` and ``axial`` is
    ``abs(y) - height/2``.  Negative values are within the corresponding
    infinite primitive.
    """

    p = _points_nx3(points)
    r = _positive_dimension(radius, "radius")
    h = _positive_dimension(height, "height")
    radial = np.hypot(p[:, 0], p[:, 2]) - r
    axial = np.abs(p[:, 1]) - 0.5 * h
    return radial, axial


def finite_y_cylinder_sdf(
    points: np.ndarray, radius: float, height: float
) -> np.ndarray:
    """Exact signed distance to a capped finite cylinder with canonical Y axis."""

    radial, axial = finite_y_cylinder_components(points, radius, height)
    components = np.stack((radial, axial), axis=1)
    outside = np.linalg.norm(np.maximum(components, 0.0), axis=1)
    inside = np.minimum(np.maximum(radial, axial), 0.0)
    return outside + inside


def finite_y_cylinder_regions(
    points: np.ndarray, radius: float, height: float, *, atol: float = 1e-12
) -> np.ndarray:
    """Classify which finite-cylinder feature determines each point's SDF."""

    if atol < 0.0 or not np.isfinite(atol):
        raise ValueError("atol must be finite and non-negative")
    radial, axial = finite_y_cylinder_components(points, radius, height)
    labels = np.empty(radial.shape, dtype=object)
    radial_out = radial > atol
    axial_out = axial > atol
    radial_surface = np.abs(radial) <= atol
    axial_surface = np.abs(axial) <= atol
    labels[radial_out & axial_out] = "outside_corner"
    labels[radial_out & ~axial_out] = "outside_side"
    labels[~radial_out & axial_out] = "outside_cap"
    labels[radial_surface & ~axial_out] = "surface_side"
    labels[~radial_out & axial_surface] = "surface_cap"
    inside = ~(radial_out | axial_out | radial_surface | axial_surface)
    labels[inside & (radial >= axial)] = "inside_side_nearest"
    labels[inside & (radial < axial)] = "inside_cap_nearest"
    # A rim point lies on both surfaces.  Give it a deterministic side label.
    labels[radial_surface & axial_surface] = "surface_side"
    return labels.astype(str)


def legacy_finite_z_cylinder_sdf_canary_only(
    points: np.ndarray, radius: float, height: float
) -> np.ndarray:
    """The historical wrong-axis equation, retained only for regression tests.

    Production collision/contact code must never call this function.
    """

    p = _points_nx3(points)
    r = _positive_dimension(radius, "radius")
    h = _positive_dimension(height, "height")
    radial = np.hypot(p[:, 0], p[:, 1]) - r
    axial = np.abs(p[:, 2]) - 0.5 * h
    components = np.stack((radial, axial), axis=1)
    outside = np.linalg.norm(np.maximum(components, 0.0), axis=1)
    inside = np.minimum(np.maximum(radial, axial), 0.0)
    return outside + inside


def deterministic_triangle_samples(
    vertices: np.ndarray, faces: np.ndarray, max_cell_edge: float
) -> np.ndarray:
    """Sample every mesh triangle on a deterministic barycentric lattice.

    A source triangle whose longest edge is ``L`` is subdivided with
    ``ceil(L / max_cell_edge)`` intervals.  Consequently every lattice cell
    edge is no longer than ``max_cell_edge`` (up to floating-point roundoff).
    """

    v = _points_nx3(vertices)
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("faces must have integer shape (M, 3)")
    if f.size and (int(f.min()) < 0 or int(f.max()) >= len(v)):
        raise ValueError("faces contain an out-of-range vertex index")
    spacing = _positive_dimension(max_cell_edge, "max_cell_edge")
    if not len(f):
        return np.empty((0, 3), dtype=np.float64)
    triangles = v[f.astype(np.int64, copy=False)]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
        ),
        axis=1,
    )
    subdivisions = np.maximum(
        1, np.ceil(np.max(edge_lengths, axis=1) / spacing).astype(np.int32)
    )
    blocks: list[np.ndarray] = []
    for count in np.unique(subdivisions):
        selected = triangles[subdivisions == count]
        weights = np.asarray(
            [
                (i / count, j / count, (count - i - j) / count)
                for i in range(count + 1)
                for j in range(count + 1 - i)
            ],
            dtype=np.float64,
        )
        blocks.append(
            np.einsum("wf,tfc->twc", weights, selected, optimize=True).reshape(-1, 3)
        )
    return np.concatenate(blocks, axis=0)


@dataclass(frozen=True)
class KaiHandPadPatch:
    """An object-independent, link-local anatomical contact patch."""

    link_name: str
    chirality: Literal["left", "right"]
    face_indices: np.ndarray
    face_centroids: np.ndarray
    face_normals: np.ndarray


_TERMINAL_FINGERS = ("index", "middle", "ring", "pinky")


def select_kaihand_pad_patch(
    link_name: str,
    chirality: Literal["left", "right"],
    vertices: np.ndarray,
    faces: np.ndarray,
) -> KaiHandPadPatch:
    """Select a fixed finger-pad/palm patch from a KaiHand link mesh.

    Authority rules are expressed only in the pinned CAD link-local frame:

    * non-thumb distal links (``*_link4``): distal 72% along local ``-Z`` and
      the palmar-facing half of the shell (``-Y`` for left, ``+Y`` for right);
    * thumb distal link (``thumb_link6``): distal 72% along local ``+X`` and
      its flexion-facing shell (local ``-Z`` for both chiralities);
    * base link palm: central 70% in X, middle/distal 72% in ``-Z``, and the
      same chirality-specific palmar-facing shell as the fingers.

    The normal threshold excludes rim/side faces.  Object transforms and SDF
    values are intentionally absent from the selector API.
    """

    if chirality not in ("left", "right"):
        raise ValueError("chirality must be 'left' or 'right'")
    prefix = f"hand_{chirality[0]}_"
    if not link_name.startswith(prefix):
        raise ValueError("link name and chirality disagree")
    v = _points_nx3(vertices)
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("faces must have integer shape (M, 3)")
    if not len(f) or int(f.min()) < 0 or int(f.max()) >= len(v):
        raise ValueError("faces must be non-empty and index vertices")
    triangles = v[f.astype(np.int64, copy=False)]
    centroids = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0

    span = np.maximum(v.max(axis=0) - v.min(axis=0), 1e-12)
    palmar_sign = -1.0 if chirality == "left" else 1.0
    if link_name.endswith("_thumb_link6"):
        distal = centroids[:, 0] >= v[:, 0].min() + 0.28 * span[0]
        facing = normals[:, 2] <= -0.35
    elif any(link_name.endswith(f"_{finger}_link4") for finger in _TERMINAL_FINGERS):
        distal = centroids[:, 2] <= v[:, 2].max() - 0.28 * span[2]
        facing = palmar_sign * normals[:, 1] >= 0.35
    elif link_name.endswith("_base_link"):
        central_x = np.abs(centroids[:, 0] - 0.5 * (v[:, 0].min() + v[:, 0].max())) <= 0.35 * span[0]
        palm_z = (
            (centroids[:, 2] <= v[:, 2].max() - 0.20 * span[2])
            & (centroids[:, 2] >= v[:, 2].min() + 0.08 * span[2])
        )
        distal = central_x & palm_z
        facing = palmar_sign * normals[:, 1] >= 0.35
    else:
        raise ValueError("link is not an authoritative KaiHand palm/finger-pad link")
    selected = np.flatnonzero(valid & distal & facing)
    if not len(selected):
        raise ValueError("anatomical pad rule selected no nondegenerate mesh faces")
    return KaiHandPadPatch(
        link_name=link_name,
        chirality=chirality,
        face_indices=selected,
        face_centroids=centroids[selected],
        face_normals=normals[selected],
    )


def triangle_triangle_intersects_sat(
    first: np.ndarray, second: np.ndarray, *, tolerance: float = 1e-10
) -> bool:
    """Return whether two closed triangles intersect using separating axes.

    Besides the ordinary triangle normals and 3x3 edge cross-products, the
    implementation includes in-plane edge-normal axes.  Those additional axes
    are required for the coplanar case; omitting them makes separated coplanar
    triangles a false collision.  Touching within ``tolerance`` counts as an
    intersection, which is conservative for the self-collision terminal.
    """

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("each triangle must be a finite (3, 3) array")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    edges_a = np.asarray((a[1] - a[0], a[2] - a[1], a[0] - a[2]))
    edges_b = np.asarray((b[1] - b[0], b[2] - b[1], b[0] - b[2]))
    normal_a = np.cross(edges_a[0], edges_a[1])
    normal_b = np.cross(edges_b[0], edges_b[1])
    if np.linalg.norm(normal_a) <= tolerance or np.linalg.norm(normal_b) <= tolerance:
        raise ValueError("degenerate triangles are not valid collision primitives")
    axes = [normal_a, normal_b]
    axes.extend(np.cross(edge_a, edge_b) for edge_a in edges_a for edge_b in edges_b)
    # These axes make the SAT complete when the two triangles are coplanar.
    axes.extend(np.cross(normal_a, edge) for edge in edges_a)
    axes.extend(np.cross(normal_a, edge) for edge in edges_b)
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length <= tolerance:
            continue
        unit = axis / length
        first_projection = a @ unit
        second_projection = b @ unit
        if float(first_projection.max()) < float(second_projection.min()) - tolerance:
            return False
        if float(second_projection.max()) < float(first_projection.min()) - tolerance:
            return False
    return True
