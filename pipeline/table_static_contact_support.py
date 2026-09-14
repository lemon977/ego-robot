"""AprilTag-anchored static table support and fail-closed contact evidence."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.table_plane_drift_reanchor import Plane, TablePlaneError


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    corners_uv: np.ndarray
    tag_to_camera: np.ndarray
    reprojection_error_px: float

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_uv, dtype=np.float64)
        transform = np.asarray(self.tag_to_camera, dtype=np.float64)
        if corners.shape != (4, 2) or transform.shape != (4, 4):
            raise TablePlaneError("tag observation shape drift")
        if not np.all(np.isfinite(corners)) or not np.all(np.isfinite(transform)):
            raise TablePlaneError("tag observation contains non-finite values")
        if self.tag_id < 0 or self.reprojection_error_px < 0:
            raise TablePlaneError("tag observation metadata invalid")
        object.__setattr__(self, "corners_uv", corners)
        object.__setattr__(self, "tag_to_camera", transform)


def detect_apriltag_36h11(
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    *,
    tag_id: int,
    tag_size_m: float,
) -> TagObservation:
    image = np.asarray(rgb)
    k = np.asarray(intrinsic, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise TablePlaneError("tag detector requires RGB uint8")
    if k.shape != (3, 3) or tag_size_m <= 0:
        raise TablePlaneError("tag detector camera/size invalid")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if ids is None or tag_id not in ids.reshape(-1):
        raise TablePlaneError(f"required AprilTag {tag_id} not detected")
    index = int(np.flatnonzero(ids.reshape(-1) == tag_id)[0])
    observed = np.asarray(corners[index][0], dtype=np.float64)
    half = float(tag_size_m) / 2.0
    object_points = np.asarray(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        observed,
        k,
        np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise TablePlaneError("AprilTag IPPE pose failed")
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    projected = cv2.projectPoints(object_points, rvec, tvec, k, np.zeros(5))[0].reshape(4, 2)
    error = float(np.mean(np.linalg.norm(projected - observed, axis=1)))
    return TagObservation(tag_id, observed, transform, error)


def tag_plane_camera(observation: TagObservation) -> Plane:
    normal = observation.tag_to_camera[:3, 2].copy()
    normal /= np.linalg.norm(normal)
    point = observation.tag_to_camera[:3, 3]
    return Plane(normal, -float(normal @ point))


def tag_plane_world(observation: TagObservation, camera_to_world: np.ndarray) -> Plane:
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
        raise TablePlaneError("camera_to_world invalid")
    camera_plane = tag_plane_camera(observation)
    normal = c2w[:3, :3] @ camera_plane.normal
    point_camera = observation.tag_to_camera[:3, 3]
    point_world = c2w[:3, :3] @ point_camera + c2w[:3, 3]
    normal /= np.linalg.norm(normal)
    if normal[1] > 0:
        normal *= -1
    return Plane(normal, -float(normal @ point_world))


def eroded_tag_support(shape: tuple[int, int], corners_uv: np.ndarray, erosion_px: int = 3) -> np.ndarray:
    if len(shape) != 2 or min(shape) <= 0 or erosion_px < 0:
        raise TablePlaneError("tag support arguments invalid")
    support = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(support, np.rint(corners_uv).astype(np.int32), 1)
    if erosion_px:
        width = 2 * erosion_px + 1
        support = cv2.erode(support, np.ones((width, width), np.uint8))
    return support.astype(bool)


def visible_object_bottom_pixel(object_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(object_mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not np.any(mask):
        raise TablePlaneError("visible object mask missing/invalid")
    y, x = np.nonzero(mask)
    threshold = np.percentile(y, 99.5)
    tail = y >= threshold
    return np.asarray([float(np.median(x[tail])), float(np.max(y))], dtype=np.float64)


def intersect_pixel_with_tag_plane(
    pixel_uv: np.ndarray,
    intrinsic: np.ndarray,
    observation: TagObservation,
) -> tuple[np.ndarray, float]:
    pixel = np.asarray(pixel_uv, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    if pixel.shape != (2,) or k.shape != (3, 3):
        raise TablePlaneError("pixel/tag intersection input invalid")
    ray = np.linalg.inv(k) @ np.asarray([pixel[0], pixel[1], 1.0])
    rotation = observation.tag_to_camera[:3, :3]
    translation = observation.tag_to_camera[:3, 3]
    system = np.column_stack((ray, -rotation[:, 0], -rotation[:, 1]))
    if np.linalg.cond(system) > 1e8:
        raise TablePlaneError("pixel ray nearly parallel to tag plane")
    depth, x, y = np.linalg.solve(system, translation)
    if depth <= 0:
        raise TablePlaneError("tag plane intersection is behind camera")
    return np.asarray([x, y], dtype=np.float64), float(depth)


def camera_center_in_tag(observation: TagObservation) -> np.ndarray:
    return np.linalg.inv(observation.tag_to_camera)[:3, 3]


def contact_observability(
    observations: list[TagObservation],
    inferred_bottom_xy: np.ndarray,
    *,
    camera_centers_metric: np.ndarray | None = None,
    minimum_metric_baseline_m: float = 0.02,
    maximum_plane_xy_scatter_m: float = 0.01,
) -> dict[str, float | bool | str]:
    xy = np.asarray(inferred_bottom_xy, dtype=np.float64)
    if xy.shape != (len(observations), 2) or len(observations) < 2:
        raise TablePlaneError("contact observability inputs invalid")
    pnp_centers = np.stack([camera_center_in_tag(item) for item in observations])
    if camera_centers_metric is None:
        centers = pnp_centers
        baseline_source = "APRILTAG_PNP_CAMERA_CENTER"
    else:
        centers = np.asarray(camera_centers_metric, dtype=np.float64)
        if centers.shape != (len(observations), 3) or not np.all(np.isfinite(centers)):
            raise TablePlaneError("metric camera centers invalid")
        baseline_source = "FROZEN_C2W_CAMERA_CENTER"
    pairwise = centers[:, None, :] - centers[None, :, :]
    baseline = float(np.max(np.linalg.norm(pairwise, axis=2)))
    pnp_pairwise = pnp_centers[:, None, :] - pnp_centers[None, :, :]
    pnp_baseline = float(np.max(np.linalg.norm(pnp_pairwise, axis=2)))
    median = np.median(xy, axis=0)
    scatter = float(np.max(np.linalg.norm(xy - median, axis=1)))
    baseline_ok = baseline >= minimum_metric_baseline_m
    scatter_ok = scatter <= maximum_plane_xy_scatter_m
    valid = baseline_ok and scatter_ok
    return {
        "max_metric_baseline_m": baseline,
        "metric_baseline_source": baseline_source,
        "apriltag_pnp_apparent_baseline_m": pnp_baseline,
        "max_inferred_plane_xy_scatter_m": scatter,
        "minimum_metric_baseline_m": minimum_metric_baseline_m,
        "maximum_plane_xy_scatter_m": maximum_plane_xy_scatter_m,
        "baseline_sufficient": baseline_ok,
        "plane_contact_hypothesis_consistent": scatter_ok,
        "contact_evidence_valid": valid,
        "contact_state": "CONTACT_SUPPORTED" if valid else "UNRESOLVED",
    }
