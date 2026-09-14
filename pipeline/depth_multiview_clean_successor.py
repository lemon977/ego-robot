"""Depth-aware, evidence-only temporal donors for the Clean successor.

This module contains the geometry and provenance core only.  It deliberately
has no inpainting, canvas extrapolation, texture synthesis, or generative
fallback.  Every published replacement pixel is an exact RGB sample from a
same-session donor whose metric depth and camera pose project into the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SOURCE_TARGET = np.uint8(0)
SOURCE_DONOR = np.uint8(1)
SOURCE_PROTECTED_OBJECT = np.uint8(2)
SOURCE_UNSUPPORTED = np.uint8(3)


class CleanGeometryError(RuntimeError):
    """Raised when an input cannot satisfy the geometric contract."""


@dataclass(frozen=True)
class Camera:
    intrinsic: np.ndarray
    camera_to_world: np.ndarray

    def checked(self, width: int, height: int) -> "Camera":
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        camera_to_world = np.asarray(self.camera_to_world, dtype=np.float64)
        if intrinsic.shape != (3, 3) or camera_to_world.shape != (4, 4):
            raise CleanGeometryError("camera matrices must be 3x3 K and 4x4 c2w")
        if not np.isfinite(intrinsic).all() or not np.isfinite(camera_to_world).all():
            raise CleanGeometryError("camera matrices must be finite")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise CleanGeometryError("camera focal lengths must be positive")
        if abs(float(intrinsic[2, 2]) - 1.0) > 1e-9:
            raise CleanGeometryError("intrinsic homogeneous scale must be one")
        rotation = camera_to_world[:3, :3]
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-3:
            raise CleanGeometryError("c2w rotation is not orthonormal")
        if np.max(np.abs(camera_to_world[3] - np.asarray([0, 0, 0, 1]))) > 1e-9:
            raise CleanGeometryError("c2w final row is malformed")
        if not (-width <= intrinsic[0, 2] <= 2 * width):
            raise CleanGeometryError("principal point x is implausible")
        if not (-height <= intrinsic[1, 2] <= 2 * height):
            raise CleanGeometryError("principal point y is implausible")
        return Camera(intrinsic, camera_to_world)


@dataclass(frozen=True)
class WarpObservation:
    valid: np.ndarray
    rgb: np.ndarray
    target_z_m: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    stable_depth_comparisons: int
    stable_depth_consistent: int

    @property
    def stable_depth_consistency(self) -> float:
        return self.stable_depth_consistent / max(1, self.stable_depth_comparisons)


@dataclass(frozen=True)
class FusedDonors:
    valid: np.ndarray
    rgb: np.ndarray
    source_frame: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    source_depth_m: np.ndarray
    support_count: np.ndarray


def _check_raster(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise CleanGeometryError(f"{name} shape {array.shape} != {shape}")
    return array


def forward_warp_real_pixels(
    donor_rgb: np.ndarray,
    donor_depth_m: np.ndarray,
    donor_camera: Camera,
    target_depth_m: np.ndarray,
    target_camera: Camera,
    donor_exclusion: np.ndarray,
    target_removal: np.ndarray,
    target_object_protect: np.ndarray,
    *,
    depth_near_m: float = 0.10,
    depth_far_m: float = 3.0,
    consistency_abs_m: float = 0.035,
    consistency_rel: float = 0.035,
) -> WarpObservation:
    """Forward-project exact donor samples and retain the nearest z per target.

    The target depth is used only to score pose/depth consistency on target
    pixels that are not being removed.  A foreground depth inside a removal
    mask therefore cannot incorrectly reject the background we are trying to
    recover.  Protected object pixels can never become donor support.
    """

    rgb = np.asarray(donor_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise CleanGeometryError("donor RGB must be uint8 HxWx3")
    height, width = rgb.shape[:2]
    shape = (height, width)
    donor_depth = _check_raster("donor_depth_m", donor_depth_m, shape).astype(
        np.float64, copy=False
    )
    target_depth = _check_raster("target_depth_m", target_depth_m, shape).astype(
        np.float64, copy=False
    )
    excluded = _check_raster("donor_exclusion", donor_exclusion, shape).astype(bool)
    removal = _check_raster("target_removal", target_removal, shape).astype(bool)
    protected = _check_raster(
        "target_object_protect", target_object_protect, shape
    ).astype(bool)
    donor_camera = donor_camera.checked(width, height)
    target_camera = target_camera.checked(width, height)
    if np.any(removal & protected):
        raise CleanGeometryError("target removal and object-protect masks overlap")
    valid_depth = (
        np.isfinite(donor_depth)
        & (donor_depth >= depth_near_m)
        & (donor_depth <= depth_far_m)
        & ~excluded
    )
    output_valid = np.zeros(shape, dtype=bool)
    output_rgb = np.zeros_like(rgb)
    output_z = np.full(shape, np.nan, dtype=np.float32)
    output_x = np.full(shape, -1, dtype=np.int32)
    output_y = np.full(shape, -1, dtype=np.int32)
    if not valid_depth.any():
        return WarpObservation(output_valid, output_rgb, output_z, output_x, output_y, 0, 0)

    source_y, source_x = np.nonzero(valid_depth)
    z = donor_depth[source_y, source_x]
    pixels = np.stack((source_x * z, source_y * z, z), axis=0)
    donor_xyz = np.linalg.inv(donor_camera.intrinsic) @ pixels
    world_xyz = (
        donor_camera.camera_to_world[:3, :3] @ donor_xyz
        + donor_camera.camera_to_world[:3, 3, None]
    )
    world_to_target = np.linalg.inv(target_camera.camera_to_world)
    target_xyz = world_to_target[:3, :3] @ world_xyz + world_to_target[:3, 3, None]
    projected_z = target_xyz[2]
    front = projected_z > depth_near_m
    projected = target_camera.intrinsic @ target_xyz
    projected_x = np.rint(projected[0] / projected_z).astype(np.int64)
    projected_y = np.rint(projected[1] / projected_z).astype(np.int64)
    inside = (
        front
        & (projected_x >= 0)
        & (projected_x < width)
        & (projected_y >= 0)
        & (projected_y < height)
        & (projected_z <= depth_far_m)
    )
    source_x = source_x[inside]
    source_y = source_y[inside]
    projected_x = projected_x[inside]
    projected_y = projected_y[inside]
    projected_z = projected_z[inside]
    if not projected_z.size:
        return WarpObservation(output_valid, output_rgb, output_z, output_x, output_y, 0, 0)

    flat = projected_y * width + projected_x
    order = np.lexsort((projected_z, flat))
    ordered_flat = flat[order]
    nearest = np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]
    selected = order[nearest]
    tx = projected_x[selected]
    ty = projected_y[selected]
    sx = source_x[selected]
    sy = source_y[selected]
    tz = projected_z[selected]

    stable = ~removal[ty, tx] & ~protected[ty, tx]
    target_z = target_depth[ty, tx]
    comparable = stable & np.isfinite(target_z) & (target_z >= depth_near_m) & (target_z <= depth_far_m)
    tolerance = consistency_abs_m + consistency_rel * np.abs(target_z)
    consistent = comparable & (np.abs(tz - target_z) <= tolerance)

    fill = removal[ty, tx] & ~protected[ty, tx]
    tx_fill, ty_fill = tx[fill], ty[fill]
    sx_fill, sy_fill = sx[fill], sy[fill]
    output_valid[ty_fill, tx_fill] = True
    output_rgb[ty_fill, tx_fill] = rgb[sy_fill, sx_fill]
    output_z[ty_fill, tx_fill] = tz[fill].astype(np.float32)
    output_x[ty_fill, tx_fill] = sx_fill.astype(np.int32)
    output_y[ty_fill, tx_fill] = sy_fill.astype(np.int32)
    return WarpObservation(
        output_valid,
        output_rgb,
        output_z,
        output_x,
        output_y,
        int(comparable.sum()),
        int(consistent.sum()),
    )


def forward_warp_real_pixels_conservative_splat(
    donor_rgb: np.ndarray,
    donor_depth_m: np.ndarray,
    donor_camera: Camera,
    target_depth_m: np.ndarray,
    target_camera: Camera,
    donor_exclusion: np.ndarray,
    target_removal: np.ndarray,
    target_object_protect: np.ndarray,
    *,
    depth_near_m: float = 0.10,
    depth_far_m: float = 3.0,
    local_depth_abs_m: float = 0.018,
    local_depth_rel: float = 0.018,
    local_color_l1: int = 65,
    splat_radius: int = 1,
    consistency_abs_m: float = 0.035,
    consistency_rel: float = 0.035,
) -> WarpObservation:
    """Rasterize continuous real-pixel donor surfaces with a nearest z-buffer.

    A point-only forward warp leaves sampling holes after camera motion.  This
    bounded successor permits a conservative 3x3 splat, but only for source
    pixels supported by at least three four-neighbours with continuous metric
    depth and colour.  Depth or colour edges are never expanded.  Each output
    sample still records one exact source pixel; no RGB interpolation occurs.
    """

    rgb = np.asarray(donor_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise CleanGeometryError("donor RGB must be uint8 HxWx3")
    if splat_radius != 1:
        raise CleanGeometryError("the reviewed successor permits only a 3x3 splat")
    height, width = rgb.shape[:2]
    shape = (height, width)
    donor_depth = _check_raster("donor_depth_m", donor_depth_m, shape).astype(
        np.float64, copy=False
    )
    target_depth = _check_raster("target_depth_m", target_depth_m, shape).astype(
        np.float64, copy=False
    )
    excluded = _check_raster("donor_exclusion", donor_exclusion, shape).astype(bool)
    removal = _check_raster("target_removal", target_removal, shape).astype(bool)
    protected = _check_raster(
        "target_object_protect", target_object_protect, shape
    ).astype(bool)
    donor_camera = donor_camera.checked(width, height)
    target_camera = target_camera.checked(width, height)
    if np.any(removal & protected):
        raise CleanGeometryError("target removal and object-protect masks overlap")

    base = (
        np.isfinite(donor_depth)
        & (donor_depth >= depth_near_m)
        & (donor_depth <= depth_far_m)
        & ~excluded
    )
    depth_neighbours = np.zeros(shape, dtype=np.uint8)
    colour_neighbours = np.zeros(shape, dtype=np.uint8)
    rgb_i16 = rgb.astype(np.int16)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        ny0, ny1 = y0 + dy, y1 + dy
        nx0, nx1 = x0 + dx, x1 + dx
        here = donor_depth[y0:y1, x0:x1]
        neighbour = donor_depth[ny0:ny1, nx0:nx1]
        tolerance = local_depth_abs_m + local_depth_rel * np.abs(here)
        depth_ok = (
            np.isfinite(here)
            & np.isfinite(neighbour)
            & (np.abs(here - neighbour) <= tolerance)
        )
        colour_distance = np.abs(
            rgb_i16[y0:y1, x0:x1] - rgb_i16[ny0:ny1, nx0:nx1]
        ).sum(axis=2)
        depth_neighbours[y0:y1, x0:x1] += depth_ok
        colour_neighbours[y0:y1, x0:x1] += colour_distance <= local_color_l1
    valid_depth = base & (depth_neighbours >= 3) & (colour_neighbours >= 3)

    output_valid = np.zeros(shape, dtype=bool)
    output_rgb = np.zeros_like(rgb)
    output_z = np.full(shape, np.nan, dtype=np.float32)
    output_x = np.full(shape, -1, dtype=np.int32)
    output_y = np.full(shape, -1, dtype=np.int32)
    if not valid_depth.any():
        return WarpObservation(output_valid, output_rgb, output_z, output_x, output_y, 0, 0)

    source_y, source_x = np.nonzero(valid_depth)
    z = donor_depth[source_y, source_x]
    pixels = np.stack((source_x * z, source_y * z, z), axis=0)
    donor_xyz = np.linalg.inv(donor_camera.intrinsic) @ pixels
    world_xyz = (
        donor_camera.camera_to_world[:3, :3] @ donor_xyz
        + donor_camera.camera_to_world[:3, 3, None]
    )
    world_to_target = np.linalg.inv(target_camera.camera_to_world)
    target_xyz = world_to_target[:3, :3] @ world_xyz + world_to_target[:3, 3, None]
    projected_z = target_xyz[2]
    projected = target_camera.intrinsic @ target_xyz
    center_x = np.rint(projected[0] / projected_z).astype(np.int64)
    center_y = np.rint(projected[1] / projected_z).astype(np.int64)
    front = (
        np.isfinite(projected_z)
        & (projected_z >= depth_near_m)
        & (projected_z <= depth_far_m)
    )

    # Two passes avoid materializing nine copies of the million-pixel raster:
    # first nearest Z, then a deterministic minimum source index at that Z.
    nearest_z = np.full(height * width, np.inf, dtype=np.float64)
    offsets = tuple(
        (dy, dx)
        for dy in range(-splat_radius, splat_radius + 1)
        for dx in range(-splat_radius, splat_radius + 1)
    )
    for dy, dx in offsets:
        tx, ty = center_x + dx, center_y + dy
        inside = front & (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
        flat = ty[inside] * width + tx[inside]
        np.minimum.at(nearest_z, flat, projected_z[inside])
    source_linear = source_y * width + source_x
    chosen_source = np.full(height * width, np.iinfo(np.int64).max, dtype=np.int64)
    for dy, dx in offsets:
        tx, ty = center_x + dx, center_y + dy
        inside = front & (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
        flat = ty[inside] * width + tx[inside]
        candidate_z = projected_z[inside]
        nearest = np.abs(candidate_z - nearest_z[flat]) <= 1e-9
        np.minimum.at(chosen_source, flat[nearest], source_linear[inside][nearest])

    chosen = chosen_source != np.iinfo(np.int64).max
    ty, tx = np.divmod(np.flatnonzero(chosen), width)
    sy, sx = np.divmod(chosen_source[chosen], width)
    tz = nearest_z[chosen]
    stable = ~removal[ty, tx] & ~protected[ty, tx]
    target_z = target_depth[ty, tx]
    comparable = (
        stable
        & np.isfinite(target_z)
        & (target_z >= depth_near_m)
        & (target_z <= depth_far_m)
    )
    tolerance = consistency_abs_m + consistency_rel * np.abs(target_z)
    consistent = comparable & (np.abs(tz - target_z) <= tolerance)

    fill = removal[ty, tx] & ~protected[ty, tx]
    tx_fill, ty_fill = tx[fill], ty[fill]
    sx_fill, sy_fill = sx[fill], sy[fill]
    output_valid[ty_fill, tx_fill] = True
    output_rgb[ty_fill, tx_fill] = rgb[sy_fill, sx_fill]
    output_z[ty_fill, tx_fill] = tz[fill].astype(np.float32)
    output_x[ty_fill, tx_fill] = sx_fill.astype(np.int32)
    output_y[ty_fill, tx_fill] = sy_fill.astype(np.int32)
    return WarpObservation(
        output_valid,
        output_rgb,
        output_z,
        output_x,
        output_y,
        int(comparable.sum()),
        int(consistent.sum()),
    )


def fuse_visible_donors(
    observations: Sequence[tuple[int, WarpObservation]],
    *,
    minimum_observations: int = 2,
    surface_abs_m: float = 0.025,
    surface_rel: float = 0.02,
    color_l1_max: float = 90.0,
) -> FusedDonors:
    """Fuse the nearest mutually consistent surface using an exact RGB medoid."""

    if len(observations) < minimum_observations:
        raise CleanGeometryError("not enough donor observations")
    shape = observations[0][1].valid.shape
    if any(row.valid.shape != shape for _, row in observations):
        raise CleanGeometryError("donor observation shapes differ")
    valid = np.stack([row.valid for _, row in observations])
    depths = np.stack([row.target_z_m for _, row in observations]).astype(np.float64)
    rgbs = np.stack([row.rgb for _, row in observations])
    nearest = np.min(np.where(valid, depths, np.inf), axis=0)
    nearest[~np.isfinite(nearest)] = np.nan
    surface = valid & (
        depths <= nearest[None] + surface_abs_m + surface_rel * nearest[None]
    )
    support_count = surface.sum(axis=0).astype(np.uint8)

    # Pick the real donor color minimizing total L1 distance to other visible
    # surface observations.  No averaging or photometric synthesis is used.
    colors = rgbs.astype(np.int16)
    scores = np.zeros(surface.shape, dtype=np.float64)
    for first in range(len(observations)):
        score = np.zeros(shape, dtype=np.float64)
        for second in range(len(observations)):
            pair = surface[first] & surface[second]
            distance = np.abs(colors[first] - colors[second]).sum(axis=2)
            score += np.where(pair, distance, 0)
        scores[first] = np.where(surface[first], score, np.inf)
    chosen = np.argmin(scores, axis=0)
    row, column = np.indices(shape)
    chosen_rgb = rgbs[chosen, row, column]
    chosen_depth = depths[chosen, row, column].astype(np.float32)
    chosen_x = np.stack([item.source_x for _, item in observations])[chosen, row, column]
    chosen_y = np.stack([item.source_y for _, item in observations])[chosen, row, column]
    frame_ids = np.asarray([frame for frame, _ in observations], dtype=np.int32)
    chosen_frame = frame_ids[chosen]
    fused_valid = support_count >= minimum_observations

    # Reject multi-view colors with no coherent real-pixel consensus.
    selected_score = scores[chosen, row, column]
    mean_l1 = selected_score / np.maximum(support_count - 1, 1)
    fused_valid &= mean_l1 <= color_l1_max
    chosen_frame = np.where(fused_valid, chosen_frame, -1).astype(np.int32)
    chosen_x = np.where(fused_valid, chosen_x, -1).astype(np.int32)
    chosen_y = np.where(fused_valid, chosen_y, -1).astype(np.int32)
    chosen_depth = np.where(fused_valid, chosen_depth, np.nan).astype(np.float32)
    chosen_rgb = np.where(fused_valid[..., None], chosen_rgb, 0).astype(np.uint8)
    return FusedDonors(
        fused_valid,
        chosen_rgb,
        chosen_frame,
        chosen_x,
        chosen_y,
        chosen_depth,
        support_count,
    )


def composite_evidence_only(
    raw: np.ndarray,
    removal: np.ndarray,
    tracker: np.ndarray,
    object_protect: np.ndarray,
    fused: FusedDonors,
) -> dict[str, np.ndarray | float | int]:
    """Composite supported donors; unsupported holes remain byte-exact raw."""

    image = np.asarray(raw)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise CleanGeometryError("raw must be uint8 HxWx3")
    shape = image.shape[:2]
    remove = _check_raster("removal", removal, shape).astype(bool)
    tracker_mask = _check_raster("tracker", tracker, shape).astype(bool)
    protected = _check_raster("object_protect", object_protect, shape).astype(bool)
    if np.any(remove & protected):
        raise CleanGeometryError("removal and object protect overlap")
    if fused.valid.shape != shape:
        raise CleanGeometryError("fused donor geometry mismatch")
    publish = remove & ~protected & fused.valid
    unresolved = remove & ~protected & ~fused.valid
    clean = image.copy()
    clean[publish] = fused.rgb[publish]
    source_kind = np.full(shape, SOURCE_TARGET, dtype=np.uint8)
    source_kind[protected] = SOURCE_PROTECTED_OBJECT
    source_kind[unresolved] = SOURCE_UNSUPPORTED
    source_kind[publish] = SOURCE_DONOR
    source_frame = np.where(publish, fused.source_frame, -1).astype(np.int32)
    source_x = np.where(publish, fused.source_x, -1).astype(np.int32)
    source_y = np.where(publish, fused.source_y, -1).astype(np.int32)
    source_depth = np.where(publish, fused.source_depth_m, np.nan).astype(np.float32)
    changed = np.any(clean != image, axis=2)
    delta = np.max(np.abs(clean.astype(np.int16) - image.astype(np.int16)), axis=2)
    tracker_core = tracker_mask & ~protected
    return {
        "clean": clean,
        "source_kind": source_kind,
        "source_frame": source_frame,
        "source_x": source_x,
        "source_y": source_y,
        "source_depth_m": source_depth,
        "support_count": fused.support_count,
        "changed_outside_removal_pixels": int((changed & ~remove).sum()),
        "changed_protected_object_pixels": int((changed & protected).sum()),
        "unsupported_pixels": int(unresolved.sum()),
        "supported_fraction": float(publish.sum() / max(1, (remove & ~protected).sum())),
        "tracker_residual_ratio": float(((delta <= 8) & tracker_core).sum() / max(1, int(tracker_core.sum()))),
    }


def audit_exact_provenance(
    raw: np.ndarray,
    clean: np.ndarray,
    source_kind: np.ndarray,
    source_frame: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    donors: Mapping[int, np.ndarray],
) -> dict[str, int | bool]:
    """Prove every changed pixel is byte-identical to its declared donor."""

    shape = raw.shape[:2]
    arrays = (clean, source_kind, source_frame, source_x, source_y)
    if clean.shape != raw.shape or any(value.shape != shape for value in arrays[1:]):
        raise CleanGeometryError("provenance audit geometry mismatch")
    mismatch = np.zeros(shape, dtype=bool)
    unchanged_classes = np.isin(source_kind, (SOURCE_TARGET, SOURCE_PROTECTED_OBJECT, SOURCE_UNSUPPORTED))
    mismatch |= unchanged_classes & np.any(clean != raw, axis=2)
    donor_pixels = source_kind == SOURCE_DONOR
    mismatch |= (~donor_pixels) & ((source_frame >= 0) | (source_x >= 0) | (source_y >= 0))
    known = np.isin(
        source_kind,
        (SOURCE_TARGET, SOURCE_DONOR, SOURCE_PROTECTED_OBJECT, SOURCE_UNSUPPORTED),
    )
    mismatch |= ~known
    for frame in np.unique(source_frame[donor_pixels]):
        frame_id = int(frame)
        donor = donors.get(frame_id)
        selected = donor_pixels & (source_frame == frame_id)
        if donor is None or donor.shape != raw.shape:
            mismatch[selected] = True
            continue
        yy, xx = np.nonzero(selected)
        sx, sy = source_x[yy, xx], source_y[yy, xx]
        inside = (sx >= 0) & (sx < shape[1]) & (sy >= 0) & (sy < shape[0])
        mismatch[yy[~inside], xx[~inside]] = True
        yy, xx, sx, sy = yy[inside], xx[inside], sx[inside], sy[inside]
        mismatch[yy, xx] |= np.any(clean[yy, xx] != donor[sy, sx], axis=1)
    return {
        "known_source_values": bool(known.all()),
        "mismatch_pixels": int(mismatch.sum()),
        "verified": bool(not mismatch.any()),
    }


def deterministic_gate_frames(frame_count: int) -> tuple[list[int], list[list[int]]]:
    """Return exactly 12 spatial frames and three consecutive 12-frame windows."""

    if frame_count < 36:
        raise CleanGeometryError("formal Clean requires at least 36 frames")
    spatial = np.rint(np.linspace(0, frame_count - 1, 12)).astype(int).tolist()
    starts = [0, (frame_count - 12) // 2, frame_count - 12]
    windows = [list(range(start, start + 12)) for start in starts]
    if len(set(spatial)) != 12 or len({tuple(window) for window in windows}) != 3:
        raise CleanGeometryError("cannot construct distinct formal gate frames")
    return spatial, windows
