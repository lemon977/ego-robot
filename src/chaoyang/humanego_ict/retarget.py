"""Pure, separately named DERIVED_RETARGET; never measured Revo2 supervision.

The v2 MANUS source has incomplete feature/index/gain definitions. This module
does not guess them or claim complete legacy parity. It implements an explicit
new angle-feature/affine contract and permits a source-bound feature provider.
Reference rules: A010/A011/A025-A030, source SHA below. No SDK or hardware I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np

CHANNEL_NAMES = ("thumb_bend", "thumb_rotation", "index", "middle", "ring", "pinky")
FEATURE_NAMES = ("thumb_bend", "thumb_rotation", "thumb_full", "index", "middle", "ring", "pinky")
MOTOR_MAX = np.array([590, 1000, 1000, 1000, 1000, 1000], dtype=np.int64)
REFERENCE_RETARGET_SHA256 = "769bc18d4f35a0c6155b51e47a9dc9c2910fb4950ffe2a8c9de68fca24daaa25"


class RetargetError(ValueError):
    pass


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RetargetError("BLOCKED_RETARGET_CONTRACT: missing source SHA")


def _motor_array(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 6 or not np.isfinite(array).all():
        raise RetargetError("six finite canonical hand channels required")
    return array


def normalize_motor(raw, *, clip=False):
    """SDK/application positions -> [0,1]; 590 is not a physical end stop."""
    array = _motor_array(raw)
    if not clip and (np.any(array < 0) or np.any(array > MOTOR_MAX)):
        raise RetargetError("hand motor outside application envelope")
    return np.clip(array / MOTOR_MAX, 0, 1)


def denormalize_motor(normalized, *, clip=False):
    """[0,1] -> raw integer positions, with the reference's round-to-even rule."""
    array = _motor_array(normalized)
    if not clip and (np.any(array < 0) or np.any(array > 1)):
        raise RetargetError("normalized hand outside [0,1]")
    return np.rint(np.clip(array, 0, 1) * MOTOR_MAX).astype(np.int64)


@dataclass(frozen=True)
class RetargetResult:
    raw_motor: np.ndarray
    normalized: np.ndarray
    valid: np.ndarray
    features: Mapping[str, float]
    missing_items: tuple[str, ...]
    status: str
    diagnostics: Mapping[str, Any]
    supervision_kind: str = "DERIVED_RETARGET"
    control_ground_truth: bool = False
    physical_deployment_authorized: bool = False

    def to_dict(self):
        return {"status": self.status, "raw_motor": [int(x) if ok else None for x, ok in zip(self.raw_motor, self.valid)],
                "normalized": [float(x) if ok else None for x, ok in zip(self.normalized, self.valid)],
                "valid": self.valid.tolist(), "features": dict(self.features),
                "missing_items": list(self.missing_items), "diagnostics": dict(self.diagnostics),
                "channel_order": list(CHANNEL_NAMES), "hand_command_unit": "NORMALIZED_0_1",
                "supervision_kind": self.supervision_kind, "control_ground_truth": False,
                "physical_deployment_authorized": False}


def _blocked(message):
    return RetargetResult(np.full(6, np.nan), np.full(6, np.nan), np.zeros(6, dtype=bool),
                          {}, (str(message),), str(message).split(":", 1)[0], {})


def _contract(contract, allow_synthetic_fixture):
    if not isinstance(contract, Mapping):
        raise RetargetError("BLOCKED_RETARGET_CONTRACT: explicit hand contract required")
    if contract.get("synthetic_fixture_only") and not allow_synthetic_fixture:
        raise RetargetError("BLOCKED_SYNTHETIC_FIXTURE")
    if (contract.get("schema") != "derived-revo2-hand-v1"
            or contract.get("supervision_kind") != "DERIVED_RETARGET"
            or contract.get("hand_command_unit") != "NORMALIZED_0_1"
            or tuple(contract.get("channel_order", ())) != CHANNEL_NAMES
            or contract.get("reference_retarget_sha256") != REFERENCE_RETARGET_SHA256):
        raise RetargetError("BLOCKED_RETARGET_CONTRACT: incompatible schema/order/unit/reference")
    provider, layout, mapping = (contract.get(key) for key in ("feature_provider", "layout", "mapping"))
    if not isinstance(provider, Mapping) or not isinstance(layout, Mapping):
        raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: source-bound provider/layout required")
    _sha(provider.get("source_sha256"))
    _sha(layout.get("source_sha256"))
    if not isinstance(mapping, Mapping) or mapping.get("kind") != "DERIVED_AFFINE_THUMB_GATED_V1":
        raise RetargetError("BLOCKED_RETARGET_CONTRACT: explicit new mapping required, not guessed legacy gain")
    _sha(mapping.get("source_sha256"))
    return provider, layout, mapping


def _unit(vector):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-10:
        raise RetargetError("BLOCKED_HAND_GEOMETRY: zero/degenerate bone or palm")
    return vector / norm


def _index(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: invalid explicit joint index")
    return value


def extract_hand_features(joints, *, side, hand_contract, allow_synthetic_fixture=False):
    """New explicit triplet-angle features, not the missing legacy calc_finger_bend.

    Palm/thumb equations and side signs follow the referenced A027/A028 form;
    every index, sign and length unit is supplied by the source-bound contract.
    """
    provider, layout, _ = _contract(hand_contract, allow_synthetic_fixture)
    if side not in {"left", "right"} or provider.get("kind") != "explicit_joint_angles_v1":
        raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: unsupported side/feature provider")
    points = np.asarray(joints, dtype=np.float64)
    count = layout.get("joint_count")
    if type(count) is not int or points.shape != (count, 3) or not np.isfinite(points).all():
        raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: joint array disagrees with explicit layout")
    if layout.get("coordinate_frame") != "wrist_local" or layout.get("position_unit") not in {"meters", "millimeters"}:
        raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: frame/length unit required")
    points = points.copy() / (1000 if layout["position_unit"] == "millimeters" else 1)
    features = {}
    triplets = layout.get("finger_triplets", {})
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        rows = triplets.get(finger)
        if not isinstance(rows, list) or not rows:
            raise RetargetError(f"BLOCKED_SOURCE_HAND_LAYOUT: {finger} triplets absent")
        total = 0.0
        for row in rows:
            if not isinstance(row, list) or len(row) != 3 or len(set(row)) != 3:
                raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: explicit nonrepeated bone triplet required")
            a, b, c = (_index(x, count) for x in row)
            total += float(np.arccos(np.clip(np.dot(_unit(points[b] - points[a]),
                                                      _unit(points[c] - points[b])), -1, 1)))
        features["thumb_bend" if finger == "thumb" else finger] = total
    indices = layout.get("thumb_palm_indices", {})
    names = ("palm_root", "index_mcp", "middle_mcp", "pinky_mcp", "thumb_mcp", "thumb_ip", "thumb_tip")
    if any(name not in indices for name in names):
        raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: palm/thumb indices absent")
    root, index, middle, pinky, thumb, ip, tip = (points[_index(indices[name], count)] for name in names)
    forward = _unit((index + middle) / 2 - root)
    normal = _unit(np.cross(index - root, middle - root))
    thumb_vector = _unit(ip - thumb)
    projected = _unit(thumb_vector - np.dot(thumb_vector, normal) * normal)
    rotation_sign = layout.get("thumb_rotation_side_sign", {}).get(side)
    depth_sign = layout.get("palm_depth_side_sign", {}).get(side)
    if rotation_sign not in {-1, 1} or depth_sign not in {-1, 1}:
        raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: explicit left/right signs required")
    features["thumb_rotation"] = float(rotation_sign * np.arctan2(
        np.dot(normal, np.cross(forward, projected)), np.clip(np.dot(forward, projected), -1, 1)))
    depth_normal = depth_sign * _unit(np.cross(index - root, pinky - root))
    features["thumb_full"] = -float(np.dot(_unit(tip - thumb), depth_normal))
    return features


def retarget_features(features, *, side, hand_contract, allow_synthetic_fixture=False):
    """Actual human feature -> six-channel derived mapping with explicit endpoints/gates."""
    try:
        _, _, mapping = _contract(hand_contract, allow_synthetic_fixture)
        if side not in {"left", "right"}:
            raise RetargetError("BLOCKED_RETARGET_CONTRACT: side required")
        calibration = mapping.get("calibration", {}).get(side, {})
        gains, scaled, clipped = mapping.get("gains", {}), {}, []
        for name in FEATURE_NAMES:
            value = float(features[name])
            opened, closed = float(calibration["open"][name]), float(calibration["closed"][name])
            gain = float(gains[name])
            if not np.isfinite([value, opened, closed, gain]).all() or gain <= 0 or abs(closed - opened) <= 1e-10:
                raise RetargetError("BLOCKED_RETARGET_CONTRACT: finite, nondegenerate sourced endpoints/gain required")
            ratio = (value - opened) / (closed - opened) * gain
            if not 0 <= ratio <= 1:
                clipped.append(name)
            scaled[name] = float(np.clip(ratio, 0, 1))
        thresholds = [mapping[key] for key in ("thumb_opposition_strong", "thumb_full_gate", "thumb_opposition_deadband")]
        if not np.isfinite(thresholds).all() or any(not 0 <= x <= 1000 for x in thresholds):
            raise RetargetError("BLOCKED_RETARGET_CONTRACT: explicit finite thumb gates required")
        if type(mapping.get("thumb_rotation_enabled")) is not bool:
            raise RetargetError("BLOCKED_RETARGET_CONTRACT: explicit thumb mode required")
        raw = np.array([round(scaled[name] * high) for name, high in zip(CHANNEL_NAMES, MOTOR_MAX)], dtype=np.int64)
        opposition = raw[1] if mapping["thumb_rotation_enabled"] else 0
        if opposition < thresholds[0] and round(scaled["thumb_full"] * 1000) < thresholds[1]:
            opposition = 0
        raw[1] = 0 if opposition < thresholds[2] else opposition
        margin = np.asarray(mapping["soft_margin"], dtype=np.float64)
        if margin.shape != (6, 2) or not np.isfinite(margin).all() or np.any(margin < 0) or np.any(margin.sum(axis=1) > MOTOR_MAX):
            raise RetargetError("BLOCKED_RETARGET_CONTRACT: explicit software margins required")
        before_soft = raw.copy()
        raw = np.rint(np.clip(raw, margin[:, 0], MOTOR_MAX - margin[:, 1])).astype(np.int64)
        return RetargetResult(raw, normalize_motor(raw), np.ones(6, dtype=bool),
            {name: float(features[name]) for name in FEATURE_NAMES}, (), "DERIVED_RETARGET_READY",
            {"mapping_kind": mapping["kind"], "clipped_features": clipped,
             "soft_clipped": (before_soft != raw).tolist(), "legacy_complete_parity": False})
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _blocked(str(exc) if isinstance(exc, RetargetError) else f"BLOCKED_RETARGET_CONTRACT: {exc}")


def retarget_hand(joints, *, side, hand_contract, feature_provider=None, allow_synthetic_fixture=False):
    """Source joints -> extracted human features -> derived motors; missing evidence stays masked.

    An injected offline provider exposes source_sha256 and extract(joints,side,layout).
    Callers must verify source/contract artifact hashes before using this pure API.
    """
    try:
        provider, layout, _ = _contract(hand_contract, allow_synthetic_fixture)
        points = np.asarray(joints, dtype=np.float64)
        count = layout.get("joint_count")
        if (type(count) is not int or points.shape != (count, 3) or not np.isfinite(points).all()
                or layout.get("coordinate_frame") != "wrist_local"
                or layout.get("position_unit") not in {"meters", "millimeters"}):
            raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: joint array/frame/unit disagrees with source layout")
        if feature_provider is None:
            features = extract_hand_features(joints, side=side, hand_contract=hand_contract,
                                             allow_synthetic_fixture=allow_synthetic_fixture)
        else:
            if (getattr(feature_provider, "offline_only", False) is not True
                    or getattr(feature_provider, "source_sha256", None) != provider["source_sha256"]):
                raise RetargetError("BLOCKED_SOURCE_HAND_LAYOUT: offline feature provider signature mismatch")
            features = feature_provider.extract(points.copy(), side, layout)
        return retarget_features(features, side=side, hand_contract=hand_contract,
                                 allow_synthetic_fixture=allow_synthetic_fixture)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _blocked(str(exc) if isinstance(exc, RetargetError) else f"BLOCKED_SOURCE_HAND_LAYOUT: {exc}")
