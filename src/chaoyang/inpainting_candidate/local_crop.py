"""Pure NumPy ROI planning and native mapping; no file or model I/O."""
from copy import deepcopy
import hashlib
import json
import numpy as np

ROLES = frozenset(("human", "glove", "object", "attachment"))
PACKING = "NUMPY_PACKBITS_C_ORDER_AXIS1_BIG_BITORDER"


def _ints(values):
    values = list(values)
    if any(isinstance(v, (bool, np.bool_)) or
           not isinstance(v, (int, np.integer)) for v in values):
        raise ValueError("integer coordinates/indices required")
    return tuple(int(v) for v in values)


def _flags(packed, proposal, name, count):
    key = proposal.get(name + "_array_key")
    flags = np.zeros(count, dtype=bool) if key is None else np.asarray(packed[key])
    if flags.shape != (count,) or flags.dtype != np.bool_:
        raise ValueError("invalid native unsupported-frame flags")
    return flags


def _rect(crop, height, width):
    rect = _ints(crop["rect_xyxy"])
    if len(rect) != 4:
        raise ValueError("four rectangle coordinates required")
    x0, y0, x1, y1 = rect
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("rectangle outside native RGB")
    return rect


def build_local_crop_manifest(rgb_shape, text_meta, packed, anchors):
    shape = _ints(rgb_shape)
    if len(shape) != 4 or shape[3] != 3:
        raise ValueError("RGB shape must be (N,H,W,3)")
    n, height, width, _ = shape
    if n < 3 or height < 1 or width < 1:
        raise ValueError("at least three real native frames required")
    if (text_meta.get("schema") != "inpainting-sam-native-proposal-diagnostic-v1" or
        tuple(text_meta["native_shape"]) != shape[:3] or
        text_meta.get("packing") != PACKING or
        text_meta.get("coordinate_domain") != "NATIVE_RAW_RGB" or
        text_meta.get("stage") != "TEXT"):
        raise ValueError("invalid TEXT/native coordinate contract")
    requested = list(_ints(anchors))
    windows = {}
    for anchor in requested:
        if not 0 <= anchor < n:
            raise ValueError("anchor outside native frames")
        start = max(0, min(anchor - 1, n - 3))
        win = windows.setdefault(start, {
            "native_frames": list(range(start, start + 3)), "anchors": []})
        if anchor not in win["anchors"]:
            win["anchors"].append(anchor)
    proposals = []
    for index, p in enumerate(text_meta["proposals"]):
        if p["role"] not in ROLES:
            continue
        if (p.get("source") != "SAM31_RGB_TEXT" or
            tuple(p["native_shape"]) != shape[:3]):
            raise ValueError("invalid RGB TEXT proposal")
        data = np.asarray(packed[p["array_key"]])
        if data.dtype != np.uint8 or data.shape != (n, (height * width + 7) // 8):
            raise ValueError("invalid packed native mask")
        flags = [_flags(packed, p, name, n) for name in
                 ("unsupported_frames", "prior_unsupported_frames")]
        proposals.append((index, p, data, *flags))
    for start, win in windows.items():
        unique = {}
        for index, p, data, unsupported, prior in proposals:
            masks = np.unpackbits(data[start:start + 3], axis=1,
                count=height * width, bitorder="big").reshape(3, height, width)
            raw_counts = masks.sum(axis=(1, 2), dtype=np.int64).tolist()
            masks[unsupported[start:start + 3]] = 0
            counts = masks.sum(axis=(1, 2), dtype=np.int64).tolist()
            ys, xs = np.nonzero(masks.any(axis=0))
            if not xs.size:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            px, py = (x1 - x0 + 3) // 4, (y1 - y0 + 3) // 4
            rect = (max(0, x0 - px), max(0, y0 - py),
                    min(width, x1 + px), min(height, y1 + py))
            crop = unique.setdefault(rect, {"rect_xyxy": list(rect), "provenance": []})
            crop["provenance"].append({
                "proposal_index": index, "proposal": deepcopy(p),
                "native_union_bbox_xyxy": [x0, y0, x1, y1],
                "raw_mask_pixels_per_frame": raw_counts,
                "valid_mask_pixels_per_frame": counts,
                "unsupported_native_frames": [
                    start + int(i) for i in np.flatnonzero(unsupported[start:start + 3])],
                "prior_unsupported_native_frames": [
                    start + int(i) for i in np.flatnonzero(prior[start:start + 3])]})
        win["crops"] = list(unique.values())
    return {"schema": "inpainting-local-crop-plan-v1",
        "native_rgb_shape": list(shape), "requested_anchors": requested,
        "bbox_scale": 1.5, "threshold_changes": False,
        "coordinate_convention": "XYXY_HALF_OPEN",
        "text_metadata": deepcopy(text_meta), "windows": list(windows.values()),
        "crop_count": sum(len(w["crops"]) for w in windows.values())}


def native_rgb_crop(rgb, plan, window, crop):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or tuple(rgb.shape) != tuple(plan["native_rgb_shape"]):
        raise ValueError("RGB must match the native uint8 plan")
    n, height, width, _ = rgb.shape
    frames = _ints(window["native_frames"])
    if len(frames) != 3 or frames != tuple(range(frames[0], frames[0] + 3)):
        raise ValueError("three real consecutive native frames required")
    if not 0 <= frames[0] <= n - 3:
        raise ValueError("window outside native RGB")
    x0, y0, x1, y1 = _rect(crop, height, width)
    return rgb[frames[0]:frames[0] + 3, y0:y1, x0:x1, :]


def remap_native_crop_masks(local_masks, plan, crop):
    local_masks = np.asarray(local_masks)
    _, height, width, _ = plan["native_rgb_shape"]
    x0, y0, x1, y1 = _rect(crop, height, width)
    if local_masks.shape != (3, y1 - y0, x1 - x0):
        raise ValueError("mask must already match native crop dimensions")
    native = np.zeros((3, height, width), dtype=local_masks.dtype)
    native[:, y0:y1, x0:x1] = local_masks
    return native


def flatten_roi_manifest(sessions):
    """sessions maps each session key to (real RGB ndarray, corresponding plan)."""
    rows = []
    for session_key, (rgb, plan) in sessions.items():
        if not isinstance(session_key, str) or not session_key:
            raise ValueError("nonempty session key required")
        for window in plan["windows"]:
            for crop in window["crops"]:
                pixels = native_rgb_crop(rgb, plan, window, crop)
                row = {"session_key": session_key,
                    "native_frames": list(window["native_frames"]),
                    "rect_xyxy": list(crop["rect_xyxy"]),
                    "crop_shape": list(pixels.shape),
                    "frame_pixel_sha256": [
                        hashlib.sha256(frame.tobytes(order="C")).hexdigest()
                        for frame in pixels],
                    "provenance": deepcopy(crop["provenance"])}
                identity = {k: v for k, v in row.items() if k != "provenance"}
                row["crop_id"] = "crop_" + hashlib.sha256(json.dumps(
                    identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                rows.append(row)
    return {"schema": "inpainting-controller-roi-manifest-v1",
        "threshold_changes": False, "crops": rows}
