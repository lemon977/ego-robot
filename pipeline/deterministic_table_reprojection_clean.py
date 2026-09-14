"""Deterministic, evidence-preserving table reprojection for CLEAN pixels.

This module deliberately contains no image completion algorithm.  A target
pixel is copied only from the nearest (by frame distance, then frame index)
same-session donor whose projection of the same static-plane point is inside
the donor image and outside the donor human mask.  Pixels without such an
observation remain unresolved.

There is no morphology, connected-component operation, interpolation,
inpainting, generative model, mask propagation, crop, or hidden fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from typing import Iterable

import numpy as np


MAGENTA_RGB = np.asarray([255, 0, 255], dtype=np.uint8)
PLANE_ID = "session_static_table_v1"
PIXEL_PROVENANCE_NPZ_SCHEMA = "deterministic-table-clean-pixel-provenance-npz-v1"

# This is one global, a-priori donor admission rule.  It is deliberately not a
# function argument or session setting.  The uncertainty terms were frozen
# before looking at any CLEAN result: one pixel of arm-mask boundary tolerance,
# the formal one-pixel reprojection gate, and the worst pixel-centre error from
# nearest-pixel quantization.  Their 2.707... px sum is conservatively ceiled.
DONOR_ARM_MARGIN_MASK_BOUNDARY_TOLERANCE_PX = 1.0
DONOR_ARM_MARGIN_REPROJECTION_MAX_PX = 1.0
DONOR_ARM_MARGIN_NEAREST_PIXEL_QUANTIZATION_MAX_PX = math.sqrt(0.5)
DONOR_ARM_MARGIN_PRIOR_SUM_PX = (
    DONOR_ARM_MARGIN_MASK_BOUNDARY_TOLERANCE_PX
    + DONOR_ARM_MARGIN_REPROJECTION_MAX_PX
    + DONOR_ARM_MARGIN_NEAREST_PIXEL_QUANTIZATION_MAX_PX
)
DONOR_ARM_MARGIN_MIN_DISTANCE_PX = math.ceil(DONOR_ARM_MARGIN_PRIOR_SUM_PX)
DONOR_ARM_MARGIN_METRIC = "PIXEL_CENTER_EUCLIDEAN_L2"
DONOR_ARM_MARGIN_RULE = (
    "ACCEPT_IFF_DISTANCE_TO_DONOR_FRAME_ARM_MASK_SET_GE_GLOBAL_MINIMUM"
)

if DONOR_ARM_MARGIN_MIN_DISTANCE_PX != 3:  # pragma: no cover - import invariant
    raise RuntimeError("frozen donor arm margin prior no longer ceils to 3 px")


class TableReprojectionError(RuntimeError):
    """An input or deterministic reprojection invariant failed."""


def _array_or_table_error(value: object, message: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TableReprojectionError(message) from error


@dataclass(frozen=True)
class Camera:
    """Rectified pinhole camera with camera-to-world pose."""

    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        camera_to_world = np.asarray(self.camera_to_world, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise TableReprojectionError("camera intrinsic must be 3x3")
        if camera_to_world.shape != (4, 4):
            raise TableReprojectionError("camera_to_world must be 4x4")
        if not np.all(np.isfinite(intrinsic)) or not np.all(
            np.isfinite(camera_to_world)
        ):
            raise TableReprojectionError("camera matrices must be finite")
        if self.width <= 0 or self.height <= 0:
            raise TableReprojectionError("camera resolution must be positive")
        if not np.allclose(camera_to_world[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
            raise TableReprojectionError("camera_to_world homogeneous row is invalid")
        if abs(float(np.linalg.det(intrinsic))) <= 1e-12:
            raise TableReprojectionError("camera intrinsic is singular")
        if abs(float(np.linalg.det(camera_to_world[:3, :3]))) <= 1e-12:
            raise TableReprojectionError("camera rotation is singular")


@dataclass(frozen=True)
class StaticPlane:
    """A single session-static world-space plane."""

    normal_world: np.ndarray
    offset_m: float
    plane_id: str = PLANE_ID

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal_world, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise TableReprojectionError(
                "plane normal must be a finite length-3 vector"
            )
        magnitude = float(np.linalg.norm(normal))
        if abs(magnitude - 1.0) > 1e-6:
            raise TableReprojectionError("plane normal must already be unit length")
        if not np.isfinite(self.offset_m):
            raise TableReprojectionError("plane offset must be finite")
        if self.plane_id != PLANE_ID:
            raise TableReprojectionError(f"plane_id must be {PLANE_ID}")


@dataclass(frozen=True)
class ReprojectionBatch:
    """Pixels that one donor can resolve for the current target."""

    target_xy: np.ndarray
    donor_xy: np.ndarray
    donor_rgb: np.ndarray
    homography: np.ndarray
    homography_sha256: str
    margin_rejected_candidate_pixel_count: int


@dataclass(frozen=True)
class PixelProvenanceArrays:
    """Lossless per-pixel source coordinates decoded from a compressed NPZ."""

    target_xy: np.ndarray
    donor_frame: np.ndarray
    donor_xy: np.ndarray


def donor_arm_margin_contract() -> dict[str, object]:
    """Return the exact, non-overridable donor arm-margin contract."""

    return {
        "rule": DONOR_ARM_MARGIN_RULE,
        "metric": DONOR_ARM_MARGIN_METRIC,
        "source_set": "TRUE_PIXEL_CENTERS_OF_DONOR_FRAME_ARM_MASK",
        "comparison": "DISTANCE_GREATER_THAN_OR_EQUAL_TO_MINIMUM",
        "minimum_distance_px": DONOR_ARM_MARGIN_MIN_DISTANCE_PX,
        "basis": {
            "selection": "A_PRIORI_GEOMETRY_UNCERTAINTY_NOT_CLEAN_RESULT_SCAN",
            "mask_boundary_tolerance_px": (DONOR_ARM_MARGIN_MASK_BOUNDARY_TOLERANCE_PX),
            "formal_reprojection_max_px": DONOR_ARM_MARGIN_REPROJECTION_MAX_PX,
            "nearest_pixel_quantization_max_px": (
                DONOR_ARM_MARGIN_NEAREST_PIXEL_QUANTIZATION_MAX_PX
            ),
            "prior_sum_px": DONOR_ARM_MARGIN_PRIOR_SUM_PX,
            "rounding": "CEIL_TO_INTEGER_PIXEL_DISTANCE",
        },
        "rejected_count_semantics": (
            "GEOMETRY_AND_DONOR_IDENTITY_VALID_CANDIDATE_PIXELS_WITH_DISTANCE_"
            "BELOW_MINIMUM_COUNTED_PER_DONOR_ATTEMPT"
        ),
    }


def _validated_provenance_arrays(
    target_xy: np.ndarray,
    donor_frame: np.ndarray,
    donor_xy: np.ndarray,
    *,
    require_int32: bool,
) -> PixelProvenanceArrays:
    target = np.asarray(target_xy)
    frames = np.asarray(donor_frame)
    donor = np.asarray(donor_xy)
    if target.ndim != 2 or target.shape[1:] != (2,):
        raise TableReprojectionError("provenance target_xy must have shape Nx2")
    if frames.ndim != 1 or frames.shape != (len(target),):
        raise TableReprojectionError("provenance donor_frame must have shape N")
    if donor.ndim != 2 or donor.shape != target.shape:
        raise TableReprojectionError("provenance donor_xy must have shape Nx2")
    for name, value in (
        ("target_xy", target),
        ("donor_frame", frames),
        ("donor_xy", donor),
    ):
        if value.dtype.kind not in "iu" or value.dtype == np.dtype(np.bool_):
            raise TableReprojectionError(f"provenance {name} must be an integer array")
        if require_int32 and value.dtype != np.dtype("<i4"):
            raise TableReprojectionError(
                f"provenance {name} must use little-endian int32"
            )
        if np.any(value < 0) or np.any(value > np.iinfo(np.int32).max):
            raise TableReprojectionError(f"provenance {name} is outside int32 range")
    target_i32 = np.asarray(target, dtype="<i4")
    frames_i32 = np.asarray(frames, dtype="<i4")
    donor_i32 = np.asarray(donor, dtype="<i4")
    if len(target_i32):
        packed = target_i32[:, 1].astype(np.int64) << 32
        packed |= target_i32[:, 0].astype(np.uint32).astype(np.int64)
        if len(np.unique(packed)) != len(packed):
            raise TableReprojectionError(
                "provenance target_xy contains duplicate pixels"
            )
    for value in (target_i32, frames_i32, donor_i32):
        value.setflags(write=False)
    return PixelProvenanceArrays(target_i32, frames_i32, donor_i32)


def encode_pixel_provenance_npz(
    target_xy: np.ndarray,
    donor_frame: np.ndarray,
    donor_xy: np.ndarray,
) -> bytes:
    """Encode lossless target/donor coordinates without JSON record inflation."""

    arrays = _validated_provenance_arrays(
        target_xy, donor_frame, donor_xy, require_int32=False
    )
    output = BytesIO()
    np.savez_compressed(
        output,
        target_xy=arrays.target_xy,
        donor_frame=arrays.donor_frame,
        donor_xy=arrays.donor_xy,
    )
    return output.getvalue()


def decode_pixel_provenance_npz(payload: bytes) -> PixelProvenanceArrays:
    """Decode and strictly validate the compressed, non-pickle provenance."""

    if not isinstance(payload, bytes) or not payload:
        raise TableReprojectionError(
            "pixel provenance NPZ payload must be non-empty bytes"
        )
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"target_xy", "donor_frame", "donor_xy"}:
                raise TableReprojectionError("pixel provenance NPZ keys mismatch")
            target_xy = archive["target_xy"].copy()
            donor_frame = archive["donor_frame"].copy()
            donor_xy = archive["donor_xy"].copy()
    except TableReprojectionError:
        raise
    except Exception as error:
        raise TableReprojectionError(
            "pixel provenance NPZ cannot be decoded"
        ) from error
    return _validated_provenance_arrays(
        target_xy, donor_frame, donor_xy, require_int32=True
    )


def canonical_homography_bytes(matrix: np.ndarray) -> bytes:
    """Serialize a normalized homography deterministically for SHA binding."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise TableReprojectionError("homography must be finite 3x3")
    scale = float(value[2, 2])
    if abs(scale) <= 1e-12:
        scale = float(np.linalg.norm(value))
    if abs(scale) <= 1e-12:
        raise TableReprojectionError("homography scale is zero")
    normalized = value / scale
    payload = {
        "dtype": "float64",
        "matrix": [[float(item) for item in row] for row in normalized],
        "normalization": "H_DIV_H22_ELSE_FROBENIUS",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def plane_homography(
    target_camera: Camera,
    donor_camera: Camera,
    plane: StaticPlane,
) -> np.ndarray:
    """Return the target-image to donor-image plane-induced homography."""

    target_rotation = np.asarray(target_camera.camera_to_world[:3, :3], np.float64)
    target_center = np.asarray(target_camera.camera_to_world[:3, 3], np.float64)
    donor_rotation = np.asarray(donor_camera.camera_to_world[:3, :3], np.float64)
    donor_center = np.asarray(donor_camera.camera_to_world[:3, 3], np.float64)

    relative_rotation = donor_rotation.T @ target_rotation
    relative_translation = donor_rotation.T @ (target_center - donor_center)
    normal_target = target_rotation.T @ np.asarray(plane.normal_world, np.float64)
    offset_target = float(
        np.asarray(plane.normal_world) @ target_center + plane.offset_m
    )
    if abs(offset_target) <= 1e-9:
        raise TableReprojectionError("target camera center lies on the table plane")
    camera_homography = (
        relative_rotation
        - np.outer(relative_translation, normal_target) / offset_target
    )
    matrix = (
        np.asarray(donor_camera.intrinsic, np.float64)
        @ camera_homography
        @ np.linalg.inv(np.asarray(target_camera.intrinsic, np.float64))
    )
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) <= 1e-15:
        raise TableReprojectionError("plane-induced homography is invalid")
    return matrix


def deterministic_donor_order(
    target_frame: int, donor_frames: Iterable[int]
) -> tuple[int, ...]:
    """Sort donors without looking at pixel results."""

    values = tuple(int(value) for value in donor_frames if int(value) != target_frame)
    if len(values) != len(set(values)):
        raise TableReprojectionError("donor frame inventory contains duplicates")
    return tuple(sorted(values, key=lambda value: (abs(value - target_frame), value)))


def _target_plane_points(
    target_xy: np.ndarray,
    camera: Camera,
    plane: StaticPlane,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(target_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise TableReprojectionError("target_xy must have shape Nx2")
    homogeneous = np.column_stack((xy, np.ones(len(xy), dtype=np.float64)))
    rays_camera = homogeneous @ np.linalg.inv(camera.intrinsic).T
    rotation = camera.camera_to_world[:3, :3]
    center = camera.camera_to_world[:3, 3]
    rays_world = rays_camera @ rotation.T
    denominator = rays_world @ plane.normal_world
    numerator = -float(plane.normal_world @ center + plane.offset_m)
    ray_scale = np.full(len(xy), np.nan, dtype=np.float64)
    non_parallel = np.abs(denominator) > 1e-12
    ray_scale[non_parallel] = numerator / denominator[non_parallel]
    points_world = center[None, :] + ray_scale[:, None] * rays_world
    valid = non_parallel & np.isfinite(ray_scale) & (ray_scale > 0.0)
    valid &= np.all(np.isfinite(points_world), axis=1)
    return points_world, valid


def _validated_target_xy(target_xy: np.ndarray, camera: Camera) -> np.ndarray:
    xy = _array_or_table_error(
        target_xy, "target_xy must be a signed integer Nx2 array"
    )
    if xy.ndim != 2 or xy.shape[1:] != (2,) or xy.dtype.kind != "i":
        raise TableReprojectionError("target_xy must be a signed integer Nx2 array")
    xy = xy.astype(np.int64, copy=False)
    if len(xy) and (
        np.any(xy[:, 0] < 0)
        or np.any(xy[:, 0] >= camera.width)
        or np.any(xy[:, 1] < 0)
        or np.any(xy[:, 1] >= camera.height)
    ):
        raise TableReprojectionError("target_xy must be inside the target camera frame")
    return xy


def _donor_pixels_meet_global_arm_margin(
    donor_xy: np.ndarray,
    donor_h_mask: np.ndarray,
) -> np.ndarray:
    """Test exact pixel-centre L2 distance without constructing a grown mask.

    Pixel centres have integer coordinates.  Therefore a source pixel violates
    the fixed 3 px rule exactly when an arm-mask centre exists at an integer
    offset whose squared norm is strictly less than 9.  This evaluates that
    set-distance predicate only for candidate pixels; it performs no dilation,
    morphology, connected-component work, or mask repair.
    """

    xy = _array_or_table_error(donor_xy, "donor_xy must be a signed integer Nx2 array")
    mask = _array_or_table_error(
        donor_h_mask, "donor H mask must be a two-dimensional bool plane"
    )
    if xy.ndim != 2 or xy.shape[1:] != (2,) or xy.dtype.kind != "i":
        raise TableReprojectionError("donor_xy must be a signed integer Nx2 array")
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise TableReprojectionError(
            "donor H mask must be a two-dimensional bool plane"
        )
    xy = xy.astype(np.int64, copy=False)
    if len(xy) and (
        np.any(xy[:, 0] < 0)
        or np.any(xy[:, 0] >= mask.shape[1])
        or np.any(xy[:, 1] < 0)
        or np.any(xy[:, 1] >= mask.shape[0])
    ):
        raise TableReprojectionError("donor_xy must be inside the donor H mask plane")

    eligible = np.ones(len(xy), dtype=np.bool_)
    if not len(xy) or not np.any(mask):
        return eligible

    radius = DONOR_ARM_MARGIN_MIN_DISTANCE_PX
    radius_squared = radius * radius
    for offset_y in range(-(radius - 1), radius):
        for offset_x in range(-(radius - 1), radius):
            if offset_x * offset_x + offset_y * offset_y >= radius_squared:
                continue
            active = np.flatnonzero(eligible)
            if not len(active):
                return eligible
            neighbor_x = xy[active, 0] + offset_x
            neighbor_y = xy[active, 1] + offset_y
            inside = (
                (neighbor_x >= 0)
                & (neighbor_x < mask.shape[1])
                & (neighbor_y >= 0)
                & (neighbor_y < mask.shape[0])
            )
            inside_active = active[inside]
            if len(inside_active):
                eligible[inside_active] &= ~mask[neighbor_y[inside], neighbor_x[inside]]
    return eligible


def _reproject_unresolved_from_donor(
    *,
    target_xy: np.ndarray,
    target_camera: Camera,
    donor_camera: Camera,
    donor_rgb: np.ndarray,
    donor_h_mask: np.ndarray,
    donor_table_identity_mask: np.ndarray | None,
    plane: StaticPlane,
) -> ReprojectionBatch:
    """Resolve candidates satisfying identity and the global arm margin.

    Sampling is nearest-neighbour by definition.  It is an exact observation
    lookup, not interpolation.  The caller applies donors in the frozen order
    and removes resolved pixels before the next call.
    """

    rgb = _array_or_table_error(donor_rgb, "donor RGB shape/dtype mismatch")
    mask = _array_or_table_error(donor_h_mask, "donor H mask shape/dtype mismatch")
    if (
        rgb.shape != (donor_camera.height, donor_camera.width, 3)
        or rgb.dtype != np.uint8
    ):
        raise TableReprojectionError("donor RGB shape/dtype mismatch")
    if (
        mask.shape != (donor_camera.height, donor_camera.width)
        or mask.dtype != np.bool_
    ):
        raise TableReprojectionError("donor H mask shape/dtype mismatch")
    identity = None
    if donor_table_identity_mask is not None:
        identity = _array_or_table_error(
            donor_table_identity_mask,
            "verified donor table identity must be bool with donor resolution",
        )
        if identity.shape != mask.shape or identity.dtype != np.bool_:
            raise TableReprojectionError(
                "verified donor table identity must be bool with donor resolution"
            )

    target_values = _validated_target_xy(target_xy, target_camera)
    points_world, valid = _target_plane_points(target_values, target_camera, plane)
    world_to_donor_rotation = donor_camera.camera_to_world[:3, :3].T
    donor_center = donor_camera.camera_to_world[:3, 3]
    points_donor = (points_world - donor_center[None, :]) @ world_to_donor_rotation.T
    depth = points_donor[:, 2]
    valid &= np.isfinite(depth) & (depth > 0.0)

    projected = points_donor @ donor_camera.intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        donor_float_xy = projected[:, :2] / projected[:, 2:3]
    valid &= np.all(np.isfinite(donor_float_xy), axis=1)
    rounded = np.zeros_like(donor_float_xy, dtype=np.int64)
    finite_rows = np.all(np.isfinite(donor_float_xy), axis=1)
    rounded[finite_rows] = np.rint(donor_float_xy[finite_rows]).astype(np.int64)
    valid &= rounded[:, 0] >= 0
    valid &= rounded[:, 0] < donor_camera.width
    valid &= rounded[:, 1] >= 0
    valid &= rounded[:, 1] < donor_camera.height

    candidate_indices = np.flatnonzero(valid)
    if len(candidate_indices) and identity is not None:
        candidate_xy = rounded[candidate_indices]
        donor_has_identity = identity[candidate_xy[:, 1], candidate_xy[:, 0]]
        candidate_indices = candidate_indices[donor_has_identity]
    margin_rejected = 0
    if len(candidate_indices):
        candidate_xy = rounded[candidate_indices]
        meets_margin = _donor_pixels_meet_global_arm_margin(candidate_xy, mask)
        margin_rejected = int(np.count_nonzero(~meets_margin))
        candidate_indices = candidate_indices[meets_margin]
    selected_target = target_values[candidate_indices]
    selected_donor = rounded[candidate_indices]
    selected_rgb = rgb[selected_donor[:, 1], selected_donor[:, 0]].copy()

    homography = plane_homography(target_camera, donor_camera, plane)
    digest = hashlib.sha256(canonical_homography_bytes(homography)).hexdigest()
    return ReprojectionBatch(
        target_xy=selected_target,
        donor_xy=selected_donor,
        donor_rgb=selected_rgb,
        homography=homography,
        homography_sha256=digest,
        margin_rejected_candidate_pixel_count=margin_rejected,
    )


def reproject_unresolved_from_donor(
    *,
    target_xy: np.ndarray,
    target_camera: Camera,
    donor_camera: Camera,
    donor_rgb: np.ndarray,
    donor_h_mask: np.ndarray,
    plane: StaticPlane,
) -> ReprojectionBatch:
    """Resolve pixels only from donor observations at the global arm margin."""

    return _reproject_unresolved_from_donor(
        target_xy=target_xy,
        target_camera=target_camera,
        donor_camera=donor_camera,
        donor_rgb=donor_rgb,
        donor_h_mask=donor_h_mask,
        donor_table_identity_mask=None,
        plane=plane,
    )


def target_xy_with_verified_table_support(
    unresolved: np.ndarray,
    table_support: np.ndarray,
) -> np.ndarray:
    """Return unresolved target pixels covered by authority-supplied table support.

    This function does not infer or expand support.  In particular, it does
    not derive a polygon from RGB, H, connectivity, or the plane itself.  A
    caller must provide the independently reviewed per-frame support plane.
    Pixels outside that plane remain unresolved.
    """

    unresolved_plane = np.asarray(unresolved)
    support_plane = np.asarray(table_support)
    if unresolved_plane.ndim != 2 or unresolved_plane.dtype != np.bool_:
        raise TableReprojectionError("unresolved must be a two-dimensional bool plane")
    if support_plane.shape != unresolved_plane.shape or support_plane.dtype != np.bool_:
        raise TableReprojectionError(
            "verified table support must be bool and match the unresolved plane"
        )
    target_y, target_x = np.nonzero(unresolved_plane & support_plane)
    return np.column_stack((target_x, target_y)).astype(np.int64, copy=False)


def reproject_unresolved_from_verified_table_donor(
    *,
    target_xy: np.ndarray,
    target_camera: Camera,
    donor_camera: Camera,
    donor_rgb: np.ndarray,
    donor_h_mask: np.ndarray,
    donor_table_identity_mask: np.ndarray,
    plane: StaticPlane,
) -> ReprojectionBatch:
    """Resolve only pixels whose donor observation has verified table identity.

    ``donor_table_identity_mask`` is an authority input, not an image-derived
    heuristic.  The ordinary donor path remains available only for explicitly
    DEVELOPMENT_ONLY plumbing runs.  Formal callers must use this function so
    that a projected sample is accepted only when it is both outside donor H
    and inside the supplied table-identity evidence.
    """

    human = _array_or_table_error(donor_h_mask, "donor H mask shape/dtype mismatch")
    identity = _array_or_table_error(
        donor_table_identity_mask,
        "verified donor table identity must be bool with donor resolution",
    )
    expected = (donor_camera.height, donor_camera.width)
    if human.shape != expected or human.dtype != np.bool_:
        raise TableReprojectionError("donor H mask shape/dtype mismatch")
    if identity.shape != expected or identity.dtype != np.bool_:
        raise TableReprojectionError(
            "verified donor table identity must be bool with donor resolution"
        )
    return _reproject_unresolved_from_donor(
        target_xy=target_xy,
        target_camera=target_camera,
        donor_camera=donor_camera,
        donor_rgb=donor_rgb,
        donor_h_mask=human,
        donor_table_identity_mask=identity,
        plane=plane,
    )


def initialize_clean(
    raw_rgb: np.ndarray, target_h_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Copy RAW outside H and place explicit magenta/unresolved inside H."""

    raw = np.asarray(raw_rgb)
    mask = np.asarray(target_h_mask)
    if raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
        raise TableReprojectionError("RAW must be uint8 RGB")
    if mask.shape != raw.shape[:2] or mask.dtype != np.bool_:
        raise TableReprojectionError("target H mask must be bool with RAW resolution")
    clean = raw.copy()
    clean[mask] = MAGENTA_RGB
    unresolved = mask.copy()
    return clean, unresolved


def apply_batch(
    clean_rgb: np.ndarray,
    unresolved: np.ndarray,
    batch: ReprojectionBatch,
) -> None:
    """Apply a donor batch only to pixels that remain unresolved."""

    if len(batch.target_xy) != len(batch.donor_xy) or len(batch.target_xy) != len(
        batch.donor_rgb
    ):
        raise TableReprojectionError("reprojection batch cardinality mismatch")
    if not len(batch.target_xy):
        return
    x = batch.target_xy[:, 0]
    y = batch.target_xy[:, 1]
    if not np.all(unresolved[y, x]):
        raise TableReprojectionError(
            "batch attempts to overwrite an already resolved pixel"
        )
    clean_rgb[y, x] = batch.donor_rgb
    unresolved[y, x] = False
