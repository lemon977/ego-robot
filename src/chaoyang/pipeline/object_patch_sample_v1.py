"""Sample an observed object point only at a valid, same-instance centre."""
from __future__ import annotations

import numpy as np


def sample_object_patch(mask: np.ndarray, depth_m: np.ndarray, valid: np.ndarray,
                        intrinsics: np.ndarray, source_xy: tuple[int, int], *,
                        scale: int = 2, radius: int = 2, max_spread_m: float = .025,
                        min_support: int = 5) -> dict:
    if mask.ndim != 2 or depth_m.shape != valid.shape or mask.shape != (depth_m.shape[0] * scale, depth_m.shape[1] * scale):
        return {"valid": False, "reason": "DOMAIN_MISMATCH"}
    x_source, y_source = map(int, source_xy)
    if not (0 <= x_source < mask.shape[1] and 0 <= y_source < mask.shape[0]):
        return {"valid": False, "reason": "SOURCE_OUT_OF_FRAME"}
    object_id = int(mask[y_source, x_source])
    if object_id <= 0:
        return {"valid": False, "reason": "CENTER_NOT_OBJECT"}
    x = int(np.clip(np.rint((x_source - .5) / scale), 0, depth_m.shape[1] - 1))
    y = int(np.clip(np.rint((y_source - .5) / scale), 0, depth_m.shape[0] - 1))
    center_z = float(depth_m[y, x])
    if not bool(valid[y, x]) or not np.isfinite(center_z) or center_z <= 0:
        return {"valid": False, "reason": "INVALID_CENTER_DEPTH", "object_id": object_id,
                "depth_xy": [x, y]}
    y0, y1 = max(0, y - radius), min(depth_m.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth_m.shape[1], x + radius + 1)
    samples = []
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            sx = min(mask.shape[1] - 1, xx * scale + scale // 2)
            sy = min(mask.shape[0] - 1, yy * scale + scale // 2)
            z = float(depth_m[yy, xx])
            if int(mask[sy, sx]) == object_id and bool(valid[yy, xx]) and np.isfinite(z) and z > 0:
                samples.append(z)
    if len(samples) < min_support:
        return {"valid": False, "reason": "INSUFFICIENT_SAME_INSTANCE_SUPPORT", "object_id": object_id,
                "depth_xy": [x, y], "support": len(samples)}
    spread = float(np.percentile(samples, 90) - np.percentile(samples, 10))
    if spread > max_spread_m:
        return {"valid": False, "reason": "DISCONTINUOUS_PATCH", "object_id": object_id,
                "depth_xy": [x, y], "support": len(samples), "spread_m": spread}
    fx, fy, cx, cy = float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        return {"valid": False, "reason": "INVALID_INTRINSICS"}
    point = [(x - cx) * center_z / fx, (y - cy) * center_z / fy, center_z]
    return {"valid": True, "reason": "OBSERVED_CENTER_VALID_SAME_INSTANCE_PATCH",
            "object_id": object_id, "source_xy": [x_source, y_source], "depth_xy": [x, y],
            "center_depth_m": center_z, "point_camera_m": point, "support": len(samples),
            "spread_m": spread, "neighborhood_used_for_depth": False}
