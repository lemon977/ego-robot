"""Depth-checked rigid Robot pixel correspondence within one frozen camera."""
from __future__ import annotations

import numpy as np


def correspond_robot_pixel(
    xy: tuple[int, int], z_current: np.ndarray, id_current: np.ndarray,
    z_next: np.ndarray, id_next: np.ndarray, k: np.ndarray,
    current_frames: dict[int, np.ndarray], next_frames: dict[int, np.ndarray],
    *, depth_tolerance_m: float = .01,
) -> dict:
    x, y = map(int, xy)
    height, width = z_current.shape
    if z_next.shape != (height, width) or id_current.shape != (height, width) or id_next.shape != (height, width):
        return {"valid": False, "reason": "DOMAIN_MISMATCH"}
    if not (0 <= x < width and 0 <= y < height):
        return {"valid": False, "reason": "SOURCE_OUT_OF_FRAME"}
    key = int(id_current[y, x])
    z = float(z_current[y, x])
    if key < 0 or not np.isfinite(z) or z <= 0:
        return {"valid": False, "reason": "INVALID_SOURCE_SURFACE"}
    if key not in current_frames or key not in next_frames:
        return {"valid": False, "reason": "LINK_FRAME_MISSING", "link_key": key}
    fx, fy, cx, cy = map(float, (k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        return {"valid": False, "reason": "INVALID_INTRINSICS"}
    point_camera = np.asarray([(x - cx) * z / fx, (y - cy) * z / fy, z, 1.], np.float64)
    point_link = np.linalg.solve(np.asarray(current_frames[key], np.float64), point_camera)
    projected = np.asarray(next_frames[key], np.float64) @ point_link
    if not np.isfinite(projected).all() or projected[2] <= 0:
        return {"valid": False, "reason": "NONPOSITIVE_TARGET_Z", "link_key": key}
    u = float(projected[0] * fx / projected[2] + cx)
    v = float(projected[1] * fy / projected[2] + cy)
    xx, yy = int(np.rint(u)), int(np.rint(v))
    base = {"link_key": key, "source_xy": [x, y], "target_xy_float": [u, v],
            "point_link_m": point_link[:3].tolist(), "projected_depth_m": float(projected[2])}
    if not (0 <= xx < width and 0 <= yy < height):
        return {"valid": False, "reason": "TARGET_OUT_OF_FRAME", **base}
    target_key = int(id_next[yy, xx])
    target_z = float(z_next[yy, xx])
    base["target_xy"] = [xx, yy]
    base["target_link_key"] = target_key
    if target_key != key:
        return {"valid": False, "reason": "OTHER_LINK_OR_DISOCCLUDED", **base}
    if not np.isfinite(target_z) or target_z <= 0:
        return {"valid": False, "reason": "INVALID_TARGET_DEPTH", **base}
    difference = target_z - float(projected[2])
    base["target_depth_delta_m"] = difference
    if abs(difference) > depth_tolerance_m:
        return {"valid": False, "reason": "DEPTH_INCONSISTENT_OR_OCCLUDED", **base}
    return {"valid": True, "reason": "SAME_LINK_RIGID_SURFACE", **base}
