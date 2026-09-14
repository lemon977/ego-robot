#!/usr/bin/env python3
"""CPU-only parametric geometry adapters for the poker/chips new tasks.

The module deliberately sits behind :mod:`pipeline.newtask_object_firewall`.
It turns an already schema-validated registry into explicit meshes, analytic
box depth, curved-mesh depth, and clean/mask protection layers.  It contains no
geometry guessing and no legacy cylinder fallback.

Coordinates are object-local metres.  ``CARD_THIN_BOX`` and ``RACK_BOX`` use
``(+x length, +y width, +z thickness/height)``.  ``CHIP_SADDLE`` uses an
elliptical footprint in x/y and thickness about its quadratic mid-surface.
``BOWL_REVOLVE`` uses +z from the outer base to the rim.  Poses are rigid
``T_object_to_camera`` transforms and all raster depth is optical camera Z on
the project's fixed 2x subpixel grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pipeline.depth_occlusion_v3 import CylinderDepth, DepthV3Error, SUPERSAMPLE
from pipeline.newtask_object_firewall import (
    canonical_task,
    authorise_geometry,
    reject_legacy_cylinder,
)
from pipeline.object_occlusion_geometry import (
    ObjectGeometry,
    project_object_depth,
)


GEOMETRY_ADAPTER_VERSION = "newtask-parametric-geometry-v2.0"
NON_RENDERED_GEOMETRY = "SUPPORT_PLANE"
VISIBLE_STATES = frozenset(
    {"ON_RACK", "IN_HAND", "AIRBORNE", "ON_MAT", "IN_BOWL", "ON_BOWL_RIM"}
)
NON_VISIBLE_STATES = frozenset({"OUT_OF_VIEW", "UNKNOWN_TRANSIENT"})


@dataclass(frozen=True)
class ParametricMesh:
    """A deterministic indexed triangle mesh in object-local metres."""

    vertices_m: np.ndarray
    faces: np.ndarray
    surface: str


@dataclass(frozen=True)
class GeometryAsset:
    """Geometry selected by one exact registry entry."""

    task: str
    object_id: str
    role: str
    geometry_type: str
    depth_adapter: str
    clean_policy: str
    visual_mesh: ParametricMesh | None
    collision_mesh: ParametricMesh | None
    box_size_xyz_m: np.ndarray | None
    measurement_id: str | None


@dataclass(frozen=True)
class FrameGeometry:
    """Per-frame depth and protection layers at the fixed 2x grid."""

    depth_by_object: Mapping[str, CylinderDepth]
    protect_mask_by_object_2x: Mapping[str, np.ndarray]
    protected_union_2x: np.ndarray
    union_near_m_2x: np.ndarray
    union_far_m_2x: np.ndarray
    object_index_2x: np.ndarray
    object_index_labels: Mapping[int, str]
    skipped_objects: Mapping[str, str]


def _as_positive_finite(value: Any, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise DepthV3Error(f"{name} must be positive finite metres")
    return scalar


def _validate_mesh(mesh: ParametricMesh) -> ParametricMesh:
    vertices = np.asarray(mesh.vertices_m, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise DepthV3Error("mesh vertices must be Nx3")
    if not np.isfinite(vertices).all():
        raise DepthV3Error("mesh vertices must be finite")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
        raise DepthV3Error("mesh faces must be Mx3")
    if not np.issubdtype(faces.dtype, np.integer):
        raise DepthV3Error("mesh faces must use integer indices")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise DepthV3Error("mesh face index is outside the vertex array")
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        raise DepthV3Error("mesh contains a repeated-index triangle")
    if not mesh.surface:
        raise DepthV3Error("mesh surface label must be non-empty")
    return ParametricMesh(vertices, faces.astype(np.int32, copy=False), mesh.surface)


def make_box_mesh(size_xyz_m: np.ndarray, *, surface: str) -> ParametricMesh:
    """Construct a closed 8-vertex/12-triangle box centred at the origin."""

    size = np.asarray(size_xyz_m, dtype=np.float64)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise DepthV3Error("box size must be three positive finite metres")
    x, y, z = size / 2.0
    vertices = np.asarray(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return _validate_mesh(ParametricMesh(vertices, faces, surface))


def _chip_mid_surface(
    x: np.ndarray,
    y: np.ndarray,
    length_m: float,
    width_m: float,
    sag_long_m: float,
    sag_short_m: float,
) -> np.ndarray:
    xn = x / (length_m / 2.0)
    yn = y / (width_m / 2.0)
    return sag_long_m * (1.0 - xn * xn) + sag_short_m * (1.0 - yn * yn)


def make_chip_saddle_mesh(
    *,
    length_m: float,
    width_m: float,
    thickness_m: float,
    center_sag_long_m: float,
    center_sag_short_m: float,
    radial_segments: int = 7,
    angular_segments: int = 48,
    thickness_scale: float = 1.0,
    footprint_scale: float = 1.0,
    surface: str = "chip_visual",
) -> ParametricMesh:
    """Build a closed elliptical double-surface saddle mesh.

    The signed sag pair controls the two quadratic principal curvatures.  A
    positive/negative pair therefore produces a true saddle.  This template is
    parametric approximation geometry; it is not inferred from phone pixels.
    """

    length = _as_positive_finite(length_m, "chip length_m") * float(footprint_scale)
    width = _as_positive_finite(width_m, "chip width_m") * float(footprint_scale)
    thickness = _as_positive_finite(thickness_m, "chip thickness_m") * float(thickness_scale)
    sag_long = float(center_sag_long_m)
    sag_short = float(center_sag_short_m)
    if not np.isfinite((sag_long, sag_short)).all():
        raise DepthV3Error("chip saddle sag must be finite")
    if sag_long == 0.0 or sag_short == 0.0 or np.sign(sag_long) == np.sign(sag_short):
        raise DepthV3Error("CHIP_SADDLE requires non-zero opposite-signed principal sag")
    if radial_segments < 2 or angular_segments < 8:
        raise DepthV3Error("chip saddle tessellation is too small")
    if not np.isfinite((thickness_scale, footprint_scale)).all() or min(
        thickness_scale, footprint_scale
    ) <= 0.0:
        raise DepthV3Error("chip collision scaling must be positive finite")

    xy: list[tuple[float, float]] = [(0.0, 0.0)]
    for ring in range(1, radial_segments + 1):
        radius = ring / radial_segments
        for sector in range(angular_segments):
            angle = 2.0 * np.pi * sector / angular_segments
            xy.append(
                (
                    length * 0.5 * radius * float(np.cos(angle)),
                    width * 0.5 * radius * float(np.sin(angle)),
                )
            )
    xy_array = np.asarray(xy, dtype=np.float64)
    mid = _chip_mid_surface(
        xy_array[:, 0], xy_array[:, 1], length, width, sag_long, sag_short
    )
    upper = np.column_stack((xy_array, mid + thickness / 2.0))
    lower = np.column_stack((xy_array, mid - thickness / 2.0))
    vertices = np.concatenate((upper, lower), axis=0)
    layer_size = len(xy_array)

    top_faces: list[list[int]] = []
    # Centre fan.
    first_ring = 1
    for sector in range(angular_segments):
        nxt = (sector + 1) % angular_segments
        top_faces.append([0, first_ring + sector, first_ring + nxt])
    # Ring strips.
    for ring in range(1, radial_segments):
        inner_start = 1 + (ring - 1) * angular_segments
        outer_start = 1 + ring * angular_segments
        for sector in range(angular_segments):
            nxt = (sector + 1) % angular_segments
            a = inner_start + sector
            b = inner_start + nxt
            c = outer_start + sector
            d = outer_start + nxt
            top_faces.extend(([a, c, d], [a, d, b]))
    bottom_faces = [
        [a + layer_size, c + layer_size, b + layer_size]
        for a, b, c in top_faces
    ]
    side_faces: list[list[int]] = []
    outer_start = 1 + (radial_segments - 1) * angular_segments
    for sector in range(angular_segments):
        nxt = (sector + 1) % angular_segments
        a = outer_start + sector
        b = outer_start + nxt
        side_faces.extend(
            ([a, a + layer_size, b + layer_size], [a, b + layer_size, b])
        )
    faces = np.asarray(top_faces + bottom_faces + side_faces, dtype=np.int32)
    return _validate_mesh(ParametricMesh(vertices, faces, surface))


def make_chip_saddle_meshes(measurement: Mapping[str, Any]) -> tuple[ParametricMesh, ParametricMesh]:
    """Return distinct visual and conservatively thickened collision meshes."""

    reject_legacy_cylinder(measurement)
    shared = dict(
        length_m=measurement["length_m"],
        width_m=measurement["width_m"],
        thickness_m=measurement["thickness_m"],
        center_sag_long_m=measurement["center_sag_long_m"],
        center_sag_short_m=measurement["center_sag_short_m"],
    )
    visual = make_chip_saddle_mesh(**shared, surface="chip_visual")
    collision = make_chip_saddle_mesh(
        **shared,
        thickness_scale=1.5,
        footprint_scale=1.01,
        surface="chip_collision_thickened",
    )
    return visual, collision


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def make_bowl_revolve_mesh(
    *,
    outer_rim_diameter_m: float,
    inner_rim_diameter_m: float,
    outer_bottom_diameter_m: float,
    inner_bottom_diameter_m: float,
    outer_height_m: float,
    inner_depth_m: float,
    wall_thickness_m: float,
    radial_segments: int = 12,
    angular_segments: int = 64,
    collision_inflate_m: float = 0.0,
    surface: str = "bowl_visual",
) -> ParametricMesh:
    """Build a closed bowl shell with separate outer and cavity surfaces."""

    outer_rim = _as_positive_finite(outer_rim_diameter_m, "outer rim diameter") / 2.0
    inner_rim = _as_positive_finite(inner_rim_diameter_m, "inner rim diameter") / 2.0
    outer_bottom = _as_positive_finite(outer_bottom_diameter_m, "outer bottom diameter") / 2.0
    inner_bottom = _as_positive_finite(inner_bottom_diameter_m, "inner bottom diameter") / 2.0
    height = _as_positive_finite(outer_height_m, "outer height")
    inner_depth = _as_positive_finite(inner_depth_m, "inner depth")
    wall = _as_positive_finite(wall_thickness_m, "wall thickness")
    inflate = float(collision_inflate_m)
    if not np.isfinite(inflate) or inflate < 0.0:
        raise DepthV3Error("bowl collision inflation must be finite and non-negative")
    if not (inner_rim < outer_rim and inner_bottom < outer_bottom and inner_depth < height):
        raise DepthV3Error("BOWL_REVOLVE dimensions are physically inconsistent")
    if wall * 2.0 >= outer_rim_diameter_m:
        raise DepthV3Error("BOWL_REVOLVE wall thickness is physically inconsistent")
    if radial_segments < 2 or angular_segments < 8:
        raise DepthV3Error("bowl tessellation is too small")

    outer_rim += inflate
    outer_bottom += inflate
    inner_rim -= inflate
    inner_bottom -= inflate
    if min(inner_rim, inner_bottom) <= 0.0:
        raise DepthV3Error("bowl collision inflation closes the cavity")
    cavity_floor_z = height - inner_depth + inflate
    if cavity_floor_z >= height:
        raise DepthV3Error("bowl collision inflation removes the cavity depth")

    vertices: list[list[float]] = []
    outer_ring_indices: list[list[int]] = []
    inner_ring_indices: list[list[int]] = []
    for ring in range(radial_segments + 1):
        t = ring / radial_segments
        radius = outer_bottom + (outer_rim - outer_bottom) * float(_smoothstep(np.asarray(t)))
        z = height * t
        indices: list[int] = []
        for sector in range(angular_segments):
            angle = 2.0 * np.pi * sector / angular_segments
            indices.append(len(vertices))
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), z])
        outer_ring_indices.append(indices)
    for ring in range(radial_segments + 1):
        t = ring / radial_segments
        radius = inner_bottom + (inner_rim - inner_bottom) * float(_smoothstep(np.asarray(t)))
        z = cavity_floor_z + (height - cavity_floor_z) * t
        indices = []
        for sector in range(angular_segments):
            angle = 2.0 * np.pi * sector / angular_segments
            indices.append(len(vertices))
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), z])
        inner_ring_indices.append(indices)

    faces: list[list[int]] = []
    for ring in range(radial_segments):
        for sector in range(angular_segments):
            nxt = (sector + 1) % angular_segments
            a, b = outer_ring_indices[ring][sector], outer_ring_indices[ring][nxt]
            c, d = outer_ring_indices[ring + 1][sector], outer_ring_indices[ring + 1][nxt]
            faces.extend(([a, c, d], [a, d, b]))
            ia, ib = inner_ring_indices[ring][sector], inner_ring_indices[ring][nxt]
            ic, id_ = inner_ring_indices[ring + 1][sector], inner_ring_indices[ring + 1][nxt]
            faces.extend(([ia, id_, ic], [ia, ib, id_]))

    # Rim annulus closes outer and inner top rings.
    outer_top = outer_ring_indices[-1]
    inner_top = inner_ring_indices[-1]
    for sector in range(angular_segments):
        nxt = (sector + 1) % angular_segments
        faces.extend(
            (
                [outer_top[sector], inner_top[sector], inner_top[nxt]],
                [outer_top[sector], inner_top[nxt], outer_top[nxt]],
            )
        )

    # Independent underside and cavity-floor caps close the shell.
    outer_center = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    inner_center = len(vertices)
    vertices.append([0.0, 0.0, cavity_floor_z])
    for sector in range(angular_segments):
        nxt = (sector + 1) % angular_segments
        faces.append([outer_center, outer_ring_indices[0][nxt], outer_ring_indices[0][sector]])
        faces.append([inner_center, inner_ring_indices[0][sector], inner_ring_indices[0][nxt]])

    return _validate_mesh(
        ParametricMesh(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int32),
            surface,
        )
    )


def make_bowl_revolve_meshes(measurement: Mapping[str, Any]) -> tuple[ParametricMesh, ParametricMesh]:
    """Return visual and separately thickened collision bowl meshes."""

    reject_legacy_cylinder(measurement)
    names = (
        "outer_rim_diameter_m",
        "inner_rim_diameter_m",
        "outer_bottom_diameter_m",
        "inner_bottom_diameter_m",
        "outer_height_m",
        "inner_depth_m",
        "wall_thickness_m",
    )
    shared = {name: measurement[name] for name in names}
    visual = make_bowl_revolve_mesh(**shared, surface="bowl_visual_inner_outer")
    collision = make_bowl_revolve_mesh(
        **shared,
        collision_inflate_m=float(measurement["wall_thickness_m"]) * 0.25,
        surface="bowl_collision_thickened_inner_outer",
    )
    return visual, collision


def split_bowl_inner_outer_surfaces(
    bowl_mesh: ParametricMesh,
) -> tuple[ParametricMesh, ParametricMesh]:
    """Split a generated revolve mesh into outer(+rim) and inner surfaces.

    The split preserves the deterministic vertex ordering produced by
    :func:`make_bowl_revolve_mesh`: outer rings, inner rings, underside centre,
    cavity-floor centre.  The returned meshes are intentionally open surface
    layers for diagnostic inner/outer depth; the combined visual mesh remains
    the closed rendering authority.
    """

    mesh = _validate_mesh(bowl_mesh)
    if not mesh.surface.startswith("bowl_") or (len(mesh.vertices_m) - 2) % 2:
        raise DepthV3Error("inner/outer split requires a generated BOWL_REVOLVE mesh")
    ring_vertex_count = (len(mesh.vertices_m) - 2) // 2
    outer_center = ring_vertex_count * 2
    inner_center = outer_center + 1
    faces = mesh.faces
    outer_vertex = (faces < ring_vertex_count) | (faces == outer_center)
    inner_vertex = (
        ((faces >= ring_vertex_count) & (faces < ring_vertex_count * 2))
        | (faces == inner_center)
    )
    outer_only = np.all(outer_vertex, axis=1)
    inner_only = np.all(inner_vertex, axis=1)
    rim = ~(outer_only | inner_only)
    if not outer_only.any() or not inner_only.any() or not rim.any():
        raise DepthV3Error("generated BOWL_REVOLVE surface groups are incomplete")
    outer = _validate_mesh(
        ParametricMesh(
            mesh.vertices_m,
            faces[outer_only | rim],
            "bowl_outer_plus_rim_depth_surface",
        )
    )
    inner = _validate_mesh(
        ParametricMesh(
            mesh.vertices_m,
            faces[inner_only],
            "bowl_inner_cavity_depth_surface",
        )
    )
    return inner, outer


def build_geometry_assets(
    *,
    task: str,
    measurements: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, GeometryAsset]:
    """Build all registry assets after exact task/object/type authorisation."""

    canonical = canonical_task(task)
    reject_legacy_cylinder(measurements)
    reject_legacy_cylinder(registry)
    if measurements.get("task") != canonical or registry.get("task") != canonical:
        raise DepthV3Error("geometry inputs disagree with the canonical task")
    measurements_by_id = {
        str(record["measurement_id"]): record for record in measurements["measurements"]
    }
    assets: dict[str, GeometryAsset] = {}
    shared_meshes: dict[str, tuple[ParametricMesh, ParametricMesh]] = {}
    for record in registry["objects"]:
        object_id = str(record["object_id"])
        geometry_type = str(record["geometry_type"])
        adapter = authorise_geometry(canonical, object_id, geometry_type)
        if adapter != record["depth_adapter"]:
            raise DepthV3Error(f"{object_id} registry depth adapter disagrees with firewall")
        measurement_id = record["measurement_id"]
        measurement = measurements_by_id.get(measurement_id) if measurement_id else None
        if geometry_type == NON_RENDERED_GEOMETRY:
            visual = collision = None
            box_size = None
        elif measurement is None or measurement["geometry_type"] != geometry_type:
            raise DepthV3Error(f"{object_id} lacks exact {geometry_type} measurement authority")
        elif geometry_type == "CARD_THIN_BOX":
            box_size = np.asarray(
                [measurement["length_m"], measurement["width_m"], measurement["thickness_m"]],
                dtype=np.float64,
            )
            visual = make_box_mesh(box_size, surface="card_visual_thin_box")
            collision = make_box_mesh(
                np.asarray([box_size[0], box_size[1], box_size[2] * 2.0]),
                surface="card_collision_thickened",
            )
        elif geometry_type == "RACK_BOX":
            box_size = np.asarray(
                [measurement["length_m"], measurement["width_m"], measurement["height_m"]],
                dtype=np.float64,
            )
            visual = collision = make_box_mesh(box_size, surface="rack_static_box")
        elif geometry_type == "CHIP_SADDLE":
            box_size = None
            key = str(measurement_id)
            if key not in shared_meshes:
                shared_meshes[key] = make_chip_saddle_meshes(measurement)
            visual, collision = shared_meshes[key]
        elif geometry_type == "BOWL_REVOLVE":
            box_size = None
            visual, collision = make_bowl_revolve_meshes(measurement)
        else:  # pragma: no cover - the firewall fails first
            raise DepthV3Error(f"no new-task adapter for {geometry_type}")
        assets[object_id] = GeometryAsset(
            task=canonical,
            object_id=object_id,
            role=str(record["role"]),
            geometry_type=geometry_type,
            depth_adapter=adapter,
            clean_policy=str(record["clean_policy"]),
            visual_mesh=visual,
            collision_mesh=collision,
            box_size_xyz_m=box_size,
            measurement_id=measurement_id,
        )
    return assets


def _validate_pose(transform_object_to_camera: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform_object_to_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise DepthV3Error("object pose must be finite 4x4")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise DepthV3Error("object pose has invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise DepthV3Error("object pose is not right-handed rigid SE(3)")
    return transform


def rasterize_mesh_depth(
    mesh: ParametricMesh,
    transform_object_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
) -> CylinderDepth:
    """Rasterize closed triangle geometry into deterministic 2x near/far Z."""

    mesh = _validate_mesh(mesh)
    transform = _validate_pose(transform_object_to_camera)
    camera = np.asarray(intrinsics, dtype=np.float64)
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise DepthV3Error("camera intrinsics must be finite 3x3")
    if camera[0, 0] <= 0.0 or camera[1, 1] <= 0.0 or not np.allclose(
        camera[2], (0.0, 0.0, 1.0), atol=1e-9
    ):
        raise DepthV3Error("camera intrinsics are not a positive pinhole model")
    if image_height < 1 or image_width < 1:
        raise DepthV3Error("image dimensions must be positive")

    vertices_camera = mesh.vertices_m @ transform[:3, :3].T + transform[:3, 3]
    if np.any(vertices_camera[:, 2] <= 1e-6):
        raise DepthV3Error("mesh intersects or crosses the camera plane")
    out_h = image_height * SUPERSAMPLE
    out_w = image_width * SUPERSAMPLE
    fx = camera[0, 0] * SUPERSAMPLE
    fy = camera[1, 1] * SUPERSAMPLE
    cx = (camera[0, 2] + 0.5) * SUPERSAMPLE - 0.5
    cy = (camera[1, 2] + 0.5) * SUPERSAMPLE - 0.5
    projected = np.column_stack(
        (
            fx * vertices_camera[:, 0] / vertices_camera[:, 2] + cx,
            fy * vertices_camera[:, 1] / vertices_camera[:, 2] + cy,
        )
    )

    near = np.full((out_h, out_w), np.inf, dtype=np.float64)
    far = np.full((out_h, out_w), -np.inf, dtype=np.float64)
    for face in mesh.faces:
        points = projected[face]
        depths = vertices_camera[face, 2]
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) <= 1e-12:
            continue
        left = max(0, int(np.ceil(np.min(points[:, 0]))))
        right = min(out_w - 1, int(np.floor(np.max(points[:, 0]))))
        top = max(0, int(np.ceil(np.min(points[:, 1]))))
        bottom = min(out_h - 1, int(np.floor(np.max(points[:, 1]))))
        if left > right or top > bottom:
            continue
        xx, yy = np.meshgrid(
            np.arange(left, right + 1, dtype=np.float64),
            np.arange(top, bottom + 1, dtype=np.float64),
        )
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        inv_depth = w0 / depths[0] + w1 / depths[1] + w2 / depths[2]
        valid = inside & (inv_depth > 0.0) & np.isfinite(inv_depth)
        depth = np.divide(1.0, inv_depth, out=np.zeros_like(inv_depth), where=valid)
        near_view = near[top : bottom + 1, left : right + 1]
        far_view = far[top : bottom + 1, left : right + 1]
        np.minimum(near_view, np.where(valid, depth, np.inf), out=near_view)
        np.maximum(far_view, np.where(valid, depth, -np.inf), out=far_view)

    mask = np.isfinite(near) & np.isfinite(far) & (far >= near) & (near > 0.0)
    near = np.where(mask, near, np.nan)
    far = np.where(mask, far, np.nan)
    return CylinderDepth(near_m=near, far_m=far, amodal_mask=mask)


def project_asset_depth(
    asset: GeometryAsset,
    transform_object_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
) -> CylinderDepth:
    """Project one firewall-authorised visual asset; support planes are rejected."""

    authorise_geometry(asset.task, asset.object_id, asset.geometry_type)
    if asset.depth_adapter == "box_xyz":
        if asset.box_size_xyz_m is None:
            raise DepthV3Error(f"{asset.object_id} lacks box dimensions")
        return project_object_depth(
            ObjectGeometry(
                "box_xyz",
                transform_object_to_camera=transform_object_to_camera,
                box_size_xyz_m=asset.box_size_xyz_m,
            ),
            intrinsics,
            image_height,
            image_width,
        )
    if asset.depth_adapter == "mask_depth":
        if asset.visual_mesh is None:
            raise DepthV3Error(f"{asset.object_id} lacks visual mesh")
        return rasterize_mesh_depth(
            asset.visual_mesh,
            transform_object_to_camera,
            intrinsics,
            image_height,
            image_width,
        )
    raise DepthV3Error(f"{asset.object_id} is not a rendered operated/static object")


def project_registry_frame(
    *,
    task: str,
    assets: Mapping[str, GeometryAsset],
    registry: Mapping[str, Any],
    timeline_frame: Mapping[str, Any],
    transforms_object_to_camera: Mapping[str, np.ndarray],
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
) -> FrameGeometry:
    """Apply registry roles and explicit state to depth/protection generation."""

    canonical = canonical_task(task)
    reject_legacy_cylinder(registry)
    reject_legacy_cylinder(timeline_frame)
    state_by_id = {state["object_id"]: state for state in timeline_frame["states"]}
    expected_tracked = {
        object_id for object_id, asset in assets.items() if asset.role == "OPERATED_OBJECT"
    }
    if set(state_by_id) != expected_tracked:
        raise DepthV3Error("timeline frame must explicitly state every operated object")

    shape = (image_height * SUPERSAMPLE, image_width * SUPERSAMPLE)
    union_near = np.full(shape, np.nan, dtype=np.float64)
    union_far = np.full(shape, np.nan, dtype=np.float64)
    union_mask = np.zeros(shape, dtype=bool)
    object_index = np.zeros(shape, dtype=np.int16)
    depth_by_object: dict[str, CylinderDepth] = {}
    masks: dict[str, np.ndarray] = {}
    labels: dict[int, str] = {}
    skipped: dict[str, str] = {}

    ordered_registry = list(registry["objects"])
    for qa_index, record in enumerate(ordered_registry, start=1):
        object_id = str(record["object_id"])
        if object_id not in assets:
            raise DepthV3Error(f"registry asset missing: {object_id}")
        asset = assets[object_id]
        authorise_geometry(canonical, object_id, asset.geometry_type)
        labels[qa_index] = object_id
        if asset.role == "SUPPORT_SURFACE":
            skipped[object_id] = "SUPPORT_AUTHORITY_NOT_RENDERED_AS_OPERATED_OBJECT"
            continue
        if asset.role == "OPERATED_OBJECT":
            state_record = state_by_id[object_id]
            state = str(state_record["state"])
            if state not in VISIBLE_STATES | NON_VISIBLE_STATES:
                raise DepthV3Error(f"unsupported explicit state for geometry: {state}")
            if not bool(state_record["pose_valid"]) or state in NON_VISIBLE_STATES:
                skipped[object_id] = f"STATE_{state}_POSE_NOT_RENDERED"
                continue
        if object_id not in transforms_object_to_camera:
            raise DepthV3Error(f"visible/static object lacks pose: {object_id}")
        depth = project_asset_depth(
            asset,
            transforms_object_to_camera[object_id],
            intrinsics,
            image_height,
            image_width,
        )
        depth_by_object[object_id] = depth
        protect = depth.amodal_mask.copy()
        if asset.clean_policy not in {"PRESERVE", "STATE_AWARE_LAYER"}:
            raise DepthV3Error(f"rendered object has unsupported clean policy: {object_id}")
        masks[object_id] = protect
        closer = protect & (~union_mask | (depth.near_m < union_near))
        union_near[closer] = depth.near_m[closer]
        union_far[closer] = depth.far_m[closer]
        object_index[closer] = np.int16(qa_index)
        union_mask |= protect

    return FrameGeometry(
        depth_by_object=depth_by_object,
        protect_mask_by_object_2x=masks,
        protected_union_2x=union_mask,
        union_near_m_2x=union_near,
        union_far_m_2x=union_far,
        object_index_2x=object_index,
        object_index_labels=labels,
        skipped_objects=skipped,
    )


def downsample_protected_any(mask_2x: np.ndarray) -> np.ndarray:
    """Conservatively reduce the fixed 2x protect mask to delivery resolution."""

    mask = np.asarray(mask_2x)
    if mask.ndim != 2 or mask.shape[0] % SUPERSAMPLE or mask.shape[1] % SUPERSAMPLE:
        raise DepthV3Error("2x protect mask must have even height and width")
    if mask.dtype != np.bool_:
        raise DepthV3Error("protect mask must be boolean")
    height, width = mask.shape
    return mask.reshape(
        height // SUPERSAMPLE,
        SUPERSAMPLE,
        width // SUPERSAMPLE,
        SUPERSAMPLE,
    ).any(axis=(1, 3))
