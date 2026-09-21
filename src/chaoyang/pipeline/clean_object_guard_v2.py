"""CPU-only, fail-closed reuse of a frozen Clean visual reference.

This is an engineering safety guard, not new inpainting or object recovery.
Unknown object identity abstains for the entire frame. Known frames may reuse
legacy Clean only inside independently supplied 2D role masks minus protected
objects, intersected with the legacy write region. No 3D evidence is consumed.
"""
from __future__ import annotations

import numpy as np

SOURCE_KIND = {
    0: "TARGET_RAW_UNCHANGED",
    1: "PROTECTED_2D_OBJECT_TARGET_RAW",
    2: "INHERITED_LEGACY_DONOR_NOT_REVALIDATED",
    3: "INHERITED_LEGACY_SYNTHETIC_NOT_BACKGROUND_TRUTH",
    4: "ABSTAIN_OBJECT_IDENTITY_UNKNOWN_RAW",
}


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"{name}: expected boolean mask {shape}")
    return value


def guard_frame(raw: np.ndarray, legacy: np.ndarray, legacy_kind: np.ndarray,
                semantic_roles: np.ndarray, protected: np.ndarray,
                identities: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError("raw: expected HxWx3 uint8 RGB")
    if legacy.dtype != np.uint8 or legacy.shape != raw.shape:
        raise ValueError("legacy: incompatible RGB")
    shape = raw.shape[:2]
    if legacy_kind.dtype != np.uint8 or legacy_kind.shape != shape:
        raise ValueError("legacy_kind: incompatible source map")
    if not set(np.unique(legacy_kind)).issubset({0, 1, 2, 3}):
        raise ValueError("legacy_kind: unknown schema/source code")
    roles = _mask(semantic_roles, shape, "semantic_roles")
    protect = _mask(protected, shape, "protected")
    if not isinstance(identities, list) or not identities:
        raise ValueError("identities: explicit nonempty expected object set required")
    # Strict bool prevents strings such as 'false' from granting write permission.
    known = all(isinstance(x, dict) and x.get("observed") is True
                and x.get("valid") is True
                and type(x.get("physical_instance_id")) is int
                and x["physical_instance_id"] >= 0
                and type(x.get("expected_instance_id")) is int
                and x["physical_instance_id"] == x["expected_instance_id"]
                and type(x.get("mask_area_px")) is int and x["mask_area_px"] > 0
                for x in identities)
    old_write = (legacy_kind == 1) | (legacy_kind == 3)
    write = old_write & roles & ~protect if known else np.zeros(shape, bool)
    out = raw.copy()
    out[write] = legacy[write]
    source = np.zeros(shape, np.uint8)
    source[protect] = 1
    source[write & (legacy_kind == 1)] = 2
    source[write & (legacy_kind == 3)] = 3
    if not known:
        source[~protect] = 4
    outside_changed = int(np.any(out != raw, axis=-1)[~write].sum())
    protected_changed = int(np.any(out != raw, axis=-1)[protect].sum())
    if outside_changed or protected_changed:
        raise AssertionError("guard invariant broken")
    stats = {
        "status": "RESTRICTED_LEGACY_VISUAL_REUSE" if known else "ABSTAIN_OBJECT_IDENTITY_UNKNOWN",
        "identity_all_known": known,
        "legacy_write_pixels": int(old_write.sum()),
        "candidate_write_pixels": int(write.sum()),
        "withheld_legacy_write_pixels": int((old_write & ~write).sum()),
        "protected_pixels": int(protect.sum()),
        "outside_write_changed_pixels": outside_changed,
        "protected_changed_pixels": protected_changed,
        "candidate_changed_pixels": int(np.any(out != raw, axis=-1).sum()),
        "raw_foreground_may_return": bool(np.any(old_write & ~write)),
        "legacy_donor_revalidated": False,
        "object_recovered": False,
    }
    return out, source, write, stats
