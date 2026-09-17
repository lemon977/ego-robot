"""Evidence-preserving visual Clean composition for 0915.

Clean never invents hidden pixels.  Human skin/forearm, finger attachments and
cables become transparent/invalid, except directly visible task-object pixels
which are protected.  Geometry stages are forbidden consumers.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from chaoyang.pipeline.visual_role_mask_v1 import CLEAN_REMOVAL_ROLES


class CleanVisualError(ValueError):
    pass


def compose_clean_visual(
    rgb: np.ndarray, masks_by_role: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray | dict[str, object]]:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise CleanVisualError("RGB must be uint8 HxWx3")
    shape = image.shape[:2]
    unknown = set(masks_by_role) - (set(CLEAN_REMOVAL_ROLES) | {"task_object"})
    if unknown:
        raise CleanVisualError(f"unknown roles: {sorted(unknown)}")
    normalized: dict[str, np.ndarray] = {}
    for role, mask in masks_by_role.items():
        value = np.asarray(mask, bool)
        if value.shape != shape:
            raise CleanVisualError(f"{role} mask shape differs from RGB")
        normalized[role] = value
    removal = np.zeros(shape, bool)
    per_role_pixels: dict[str, int] = {}
    for role in sorted(CLEAN_REMOVAL_ROLES):
        value = normalized.get(role, np.zeros(shape, bool))
        removal |= value
        per_role_pixels[role] = int(value.sum())
    protected = normalized.get("task_object", np.zeros(shape, bool))
    effective = removal & ~protected
    valid = ~effective
    cleaned = image.copy()
    cleaned[~valid] = 0
    rgba = np.concatenate((cleaned, (valid.astype(np.uint8) * 255)[..., None]), axis=2)
    source_map = np.where(valid, 1, 0).astype(np.uint8)
    return {
        "clean_rgb": cleaned,
        "clean_rgba": rgba,
        "valid_mask": valid,
        "invalid_mask": ~valid,
        "task_object_protection_mask": protected,
        "source_map": source_map,
        "evidence": {
            "per_role_mask_pixels": per_role_pixels,
            "raw_removal_union_pixels": int(removal.sum()),
            "protected_task_object_pixels": int((removal & protected).sum()),
            "invalid_hidden_pixels": int((~valid).sum()),
            "hidden_pixels_synthesized": 0,
            "source_map_values": {"0": "INVALID_UNKNOWN", "1": "ORIGINAL_RGB"},
            "geometry_consumers_forbidden": ["Depth", "Object6D", "Contact"],
        },
    }
