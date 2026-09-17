"""Manifest-bound SAM 2.1 evidence producer for the G2 MASK/CLEAN review gate.

This module intentionally does not inpaint, refine Object6D, render a robot, or
decide whether an unsupported pixel may be hidden.  It writes only MASK evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from jsonschema import Draft202012Validator


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular_nofollow(path: Path, expected_sha256: str | None = None) -> bytes:
    path.lstat()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"formal input is not a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        data = b""
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if expected_sha256 is not None:
        observed = _sha256_bytes(data)
        if observed != expected_sha256:
            raise RuntimeError(f"SHA mismatch for {path}: {observed} != {expected_sha256}")
    return data


def _verify_regular_sha256_nofollow(path: Path, expected_sha256: str) -> None:
    """Stream a large immutable input without retaining it in memory."""

    path.lstat()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"formal input is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f"SHA mismatch for {path}: {observed} != {expected_sha256}")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _artifact_ref(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "producer": "mask_producer",
    }


def _write_png(path: Path, image: np.ndarray) -> dict[str, Any]:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError(f"failed to encode PNG: {path}")
    data = encoded.tobytes()
    _atomic_write_bytes(path, data)
    return _artifact_ref(path, data)


def _write_json_ref(path: Path, value: Any) -> dict[str, Any]:
    data = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    _atomic_write_bytes(path, data)
    return _artifact_ref(path, data)


def _load_json_nofollow(path: Path, expected_sha256: str | None = None) -> Any:
    return json.loads(_read_regular_nofollow(path, expected_sha256).decode("utf-8"))


def _load_rgb(frame: dict[str, Any]) -> np.ndarray:
    ref = frame["image"]
    data = _read_regular_nofollow(Path(ref["path"]), ref["sha256"])
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"image decode failed: {ref['path']}")
    if bgr.shape[:2] != (int(frame["height"]), int(frame["width"])):
        raise RuntimeError(f"image dimensions disagree with manifest: {ref['path']}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_binary_mask_ref(ref: dict[str, Any]) -> np.ndarray:
    data = _read_regular_nofollow(Path(ref["path"]), ref["sha256"])
    mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"mask decode failed: {ref['path']}")
    return mask > 0


def _clamp_px(value: float, scale: float, low_ratio: float, high_ratio: float) -> int:
    return max(1, int(round(np.clip(value, low_ratio * scale, high_ratio * scale))))


def _contact_band_radius_px(
    projected_object_area_px: int,
    *,
    object_scale_ratio: float,
    image_min_dimension: int,
    low_resolution_ratio: float,
    high_resolution_ratio: float,
) -> int:
    """Normalize contact width by sqrt(projected object area), then resolution-clamp."""

    if projected_object_area_px <= 0:
        raise RuntimeError("contact band requires a non-empty projected object")
    object_scale_px = math.sqrt(float(projected_object_area_px))
    return _clamp_px(
        object_scale_ratio * object_scale_px,
        float(image_min_dimension),
        low_resolution_ratio,
        high_resolution_ratio,
    )


def _assign_contact_ownership(
    human_candidate: np.ndarray,
    object_candidate: np.ndarray,
    analytic_object: np.ndarray,
    contact_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Assign ambiguous hand/object contact pixels to U, never silently to O.

    The returned H/O/U planes are mutually exclusive.  The raw overlap is
    measured before ownership assignment so object protection cannot hide a
    human miss by simple subtraction.
    """

    human = np.asarray(human_candidate, dtype=bool)
    visible_object = np.asarray(object_candidate, dtype=bool)
    analytic = np.asarray(analytic_object, dtype=bool)
    if human.shape != visible_object.shape or human.shape != analytic.shape:
        raise RuntimeError("contact ownership planes have different dimensions")
    pre_overlap = human & visible_object
    contact_band = (
        cv2.dilate(human.astype(np.uint8), contact_kernel).astype(bool) & analytic
    )
    uncertain = contact_band | pre_overlap
    human_core = human & ~uncertain
    object_core = visible_object & ~uncertain
    if np.any(human_core & object_core) or np.any(human_core & uncertain) or np.any(
        object_core & uncertain
    ):
        raise RuntimeError("H/O/U contact ownership is not mutually exclusive")
    return human_core, object_core, uncertain, {
        "pre_contact_object_overlap_pixels": float(np.count_nonzero(pre_overlap)),
        "u_contact_pixels": float(np.count_nonzero(uncertain)),
        "post_contact_object_overlap_pixels": float(
            np.count_nonzero(human_core & object_core)
        ),
        "contact_assignment_subtraction_without_u_pixels": 0.0,
    }


def _valid_hand_points(hand: dict[str, Any], width: int, height: int) -> tuple[np.ndarray, list[str]]:
    points = np.asarray(hand["keypoints_2d"], dtype=np.float32)
    valid = np.asarray(hand["joint_valid"], dtype=bool) & np.asarray(hand["joint_in_image"], dtype=bool)
    valid &= np.isfinite(points).all(axis=1)
    valid &= (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    names = [str(name) for name, keep in zip(hand["joint_names"], valid) if keep]
    return points[valid], names


def _named_point(hand: dict[str, Any], name: str) -> np.ndarray:
    names = list(hand["joint_names"])
    if name not in names:
        raise RuntimeError(f"required hand keypoint missing: {name}")
    point = np.asarray(hand["keypoints_2d"][names.index(name)], dtype=np.float32)
    if not np.isfinite(point).all():
        raise RuntimeError(f"required hand keypoint is not finite: {name}")
    return point


def _hand_scale(points: np.ndarray) -> float:
    if len(points) < 3:
        raise RuntimeError("fewer than three valid hand points")
    extent = points.max(axis=0) - points.min(axis=0)
    return float(max(np.linalg.norm(extent), 1.0))


def _ray_to_border(origin: np.ndarray, direction: np.ndarray, width: int, height: int) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        raise RuntimeError("forearm direction is degenerate; refusing default downward ray")
    direction = direction / norm
    candidates: list[float] = []
    if direction[0] > 1e-6:
        candidates.append((width - 1 - origin[0]) / direction[0])
    elif direction[0] < -1e-6:
        candidates.append((0 - origin[0]) / direction[0])
    if direction[1] > 1e-6:
        candidates.append((height - 1 - origin[1]) / direction[1])
    elif direction[1] < -1e-6:
        candidates.append((0 - origin[1]) / direction[1])
    positive = [value for value in candidates if value >= 0]
    if not positive:
        raise RuntimeError("forearm ray does not intersect image")
    end = origin + min(positive) * direction
    return np.clip(end, [0, 0], [width - 1, height - 1]).astype(np.float32)


def _box_from_points(points: np.ndarray, padding: float, width: int, height: int) -> np.ndarray:
    lo = points.min(axis=0) - padding
    hi = points.max(axis=0) + padding
    lo = np.maximum(lo, [0, 0])
    hi = np.minimum(hi, [width - 1, height - 1])
    if np.any(hi <= lo):
        raise RuntimeError("degenerate SAM box")
    return np.asarray([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)


def _point_recall(mask: np.ndarray, points: np.ndarray) -> float:
    if not len(points):
        return 0.0
    xy = np.rint(points).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
    return float(mask[xy[:, 1], xy[:, 0]].mean())


def _keep_prompt_bearing_components(
    mask: np.ndarray, positive_points: np.ndarray
) -> tuple[np.ndarray, int]:
    """Remove disconnected islands without discarding valid prompted components."""

    component_count, labels = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8
    )
    xy = np.rint(positive_points).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
    keep_labels = np.unique(labels[xy[:, 1], xy[:, 0]])
    keep_labels = keep_labels[keep_labels != 0]
    if not len(keep_labels):
        return np.zeros_like(mask, dtype=bool), max(component_count - 1, 0)
    return np.isin(labels, keep_labels), max(component_count - 1, 0)


def _anatomy_support_masks(
    points: np.ndarray,
    wrist: np.ndarray,
    palm: np.ndarray,
    wrist_width: float,
    *,
    forearm_width_ratio: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build session-agnostic hand envelope and wrist-to-border corridor.

    Both spatial widths are expressed in measured wrist-width units.  The hand
    support is the 21-point convex hull plus the same normalized envelope.  The
    forearm support is analytic and reaches the image boundary along the
    palm-to-wrist ray; it is not inferred from semantic image appearance.
    """

    if len(points) < 3:
        raise RuntimeError("anatomy support requires at least three valid points")
    if not math.isfinite(wrist_width) or wrist_width <= 0:
        raise RuntimeError("anatomy support requires positive measured wrist width")
    if not math.isfinite(forearm_width_ratio) or forearm_width_ratio <= 0:
        raise RuntimeError("forearm width ratio must be finite and positive")
    half_width = 0.5 * forearm_width_ratio * wrist_width
    envelope_radius = max(1, int(round(half_width)))

    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    hand_hull = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(hand_hull, hull, 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * envelope_radius + 1, 2 * envelope_radius + 1)
    )
    hand_envelope = cv2.dilate(hand_hull, kernel).astype(bool)

    end = _ray_to_border(wrist, wrist - palm, width, height)
    length = float(np.linalg.norm(end - wrist))
    direction = (end - wrist) / max(length, 1e-6)
    normal = np.asarray([-direction[1], direction[0]], np.float32)
    polygon_float = np.asarray(
        [
            wrist + normal * half_width,
            wrist - normal * half_width,
            end - normal * half_width,
            end + normal * half_width,
        ],
        np.float32,
    )
    polygon = np.rint(polygon_float).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    corridor = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(corridor, polygon, 1)
    return hand_envelope, corridor.astype(bool), {
        "normalization": "measured_wrist_width_times_k_forearm",
        "forearm_width_ratio": float(forearm_width_ratio),
        "wrist_width_px": float(wrist_width),
        "envelope_radius_px": float(envelope_radius),
        "forearm_half_width_px": float(half_width),
        "forearm_end_xy": end.tolist(),
        "forearm_corridor_polygon_xy": polygon.tolist(),
        "hand_envelope_area_px": float(np.count_nonzero(hand_envelope)),
        "forearm_corridor_area_px": float(np.count_nonzero(corridor)),
    }


def _select_wrist_connected_forearm(
    forearm_mask: np.ndarray,
    wrist_mask: np.ndarray,
    wrist_region: np.ndarray,
    forearm_corridor: np.ndarray,
    wrist: np.ndarray,
    palm: np.ndarray,
    wrist_width: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one observed wrist-to-boundary forearm component.

    The analytic corridor is a prior/score, never a producer.  A forearm is
    accepted only when observed SAM/wrist evidence itself forms one component
    from the wrist anchor to a crop boundary.  PCA is recorded as edge evidence
    for a future, separately gated minimal bridge; this implementation does not
    add such a bridge.
    """

    shapes = {value.shape for value in (forearm_mask, wrist_mask, wrist_region, forearm_corridor)}
    if len(shapes) != 1:
        raise RuntimeError("forearm evidence masks must have identical shapes")
    observed_candidate = forearm_mask.astype(bool)
    wrist_anchor = wrist_mask & wrist_region
    observed_forearm = observed_candidate | wrist_anchor

    component_count, labels = cv2.connectedComponents(
        observed_forearm.astype(np.uint8), connectivity=8
    )
    anchor_labels, anchor_counts = np.unique(labels[wrist_anchor], return_counts=True)
    valid_anchor = anchor_labels != 0
    anchor_labels = anchor_labels[valid_anchor]
    anchor_counts = anchor_counts[valid_anchor]
    selected = np.zeros_like(observed_forearm, dtype=bool)
    selected_label = 0
    if len(anchor_labels):
        component_areas = np.bincount(labels.ravel(), minlength=component_count)
        ranking = sorted(
            zip(anchor_labels.tolist(), anchor_counts.tolist(), strict=True),
            key=lambda item: (item[1], int(component_areas[item[0]])),
            reverse=True,
        )
        selected_label = int(ranking[0][0])
        selected = labels == selected_label

    touched_edges: list[str] = []
    if np.any(selected[0, :]):
        touched_edges.append("top")
    if np.any(selected[-1, :]):
        touched_edges.append("bottom")
    if np.any(selected[:, 0]):
        touched_edges.append("left")
    if np.any(selected[:, -1]):
        touched_edges.append("right")

    axis_prior_cosine = 0.0
    target_edge = "UNAVAILABLE"
    if np.count_nonzero(selected) >= 3:
        y, x = np.nonzero(selected)
        xy = np.column_stack([x, y]).astype(np.float64)
        covariance = np.cov(xy - xy.mean(axis=0), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        prior = np.asarray(wrist - palm, np.float64)
        prior /= max(float(np.linalg.norm(prior)), 1e-6)
        if float(np.dot(principal, prior)) < 0:
            principal = -principal
        axis_prior_cosine = float(np.clip(np.dot(principal, prior), -1.0, 1.0))
        joint_direction = principal + prior
        if float(np.linalg.norm(joint_direction)) <= 1e-6:
            joint_direction = prior
        endpoint = _ray_to_border(
            np.asarray(wrist, np.float32),
            np.asarray(joint_direction, np.float32),
            selected.shape[1],
            selected.shape[0],
        )
        distances = {
            "left": abs(float(endpoint[0])),
            "right": abs(float(endpoint[0]) - (selected.shape[1] - 1)),
            "top": abs(float(endpoint[1])),
            "bottom": abs(float(endpoint[1]) - (selected.shape[0] - 1)),
        }
        target_edge = min(distances, key=distances.get)

    boundary_connected = float(bool(touched_edges))
    held = float(selected_label == 0 or boundary_connected == 0.0)
    return selected, {
        "raw_forearm_pixels": float(np.count_nonzero(forearm_mask)),
        "bounded_forearm_pixels": float(np.count_nonzero(observed_candidate)),
        "prior_corridor_overlap_pixels": float(
            np.count_nonzero(observed_candidate & forearm_corridor)
        ),
        "prior_corridor_precision": float(
            np.count_nonzero(observed_candidate & forearm_corridor)
            / max(np.count_nonzero(observed_candidate), 1)
        ),
        "wrist_anchor_pixels": float(np.count_nonzero(wrist_anchor)),
        "wrist_connected_forearm_pixels": float(np.count_nonzero(selected)),
        "rejected_disconnected_forearm_pixels": float(
            np.count_nonzero(observed_forearm & ~selected)
        ),
        "analytic_corridor_completion_pixels": 0.0,
        "bridge_pixels": 0.0,
        "bridge_policy": "DISABLED_HOLD_IF_NOT_BOUNDARY_CONNECTED",
        "selected_component_label": float(selected_label),
        "selected_component_boundary_connected": boundary_connected,
        "selected_component_touched_edges": touched_edges,
        "pca_axis_prior_cosine": axis_prior_cosine,
        "pca_prior_target_edge": target_edge,
        "pca_target_edge_observed": float(target_edge in touched_edges),
        "wrist_width_px": float(wrist_width),
        "held": held,
    }


def _apply_anatomy_bounds(
    hand_mask: np.ndarray,
    wrist_mask: np.ndarray,
    wrist_region: np.ndarray,
    forearm_mask: np.ndarray,
    hand_envelope: np.ndarray,
    forearm_corridor: np.ndarray,
    wrist: np.ndarray,
    palm: np.ndarray,
    wrist_width: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject non-anatomic pixels and keep only observed forearm evidence."""

    shapes = {
        value.shape
        for value in (
            hand_mask,
            wrist_mask,
            wrist_region,
            forearm_mask,
            hand_envelope,
            forearm_corridor,
        )
    }
    if len(shapes) != 1:
        raise RuntimeError("anatomy-bound masks must have identical shapes")
    raw_hand = hand_mask | (wrist_mask & wrist_region)
    bounded_hand = raw_hand & hand_envelope
    selected_forearm, forearm_metrics = _select_wrist_connected_forearm(
        forearm_mask,
        wrist_mask,
        wrist_region,
        forearm_corridor,
        wrist,
        palm,
        wrist_width,
    )
    combined = bounded_hand | selected_forearm
    return combined, {
        "raw_hand_pixels": float(np.count_nonzero(raw_hand)),
        "bounded_hand_pixels": float(np.count_nonzero(bounded_hand)),
        "rejected_non_anatomic_hand_pixels": float(
            np.count_nonzero(raw_hand & ~hand_envelope)
        ),
        **forearm_metrics,
        "combined_pixels": float(np.count_nonzero(combined)),
    }


def _final_anatomy_evidence(
    metadata: dict[str, Any],
    human_core: np.ndarray,
    object_core: np.ndarray,
    uncertain_contact: np.ndarray,
    side_metrics: dict[str, Any],
    *,
    hand_recall_gate: float,
    forearm_recall_gate: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Evaluate final H_core under object/contact protection and crop topology."""

    _, labels = cv2.connectedComponents(human_core.astype(np.uint8), connectivity=8)
    border_labels = set(
        int(value)
        for value in np.unique(
            np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        )
        if int(value) != 0
    )
    protected = object_core | uncertain_contact
    results: dict[str, dict[str, float]] = {}
    for side in ("left", "right"):
        hand = metadata["entities"]["hands"][side]
        points, _ = _valid_hand_points(hand, human_core.shape[1], human_core.shape[0])
        xy = np.rint(points).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, human_core.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, human_core.shape[0] - 1)
        evaluable = ~protected[xy[:, 1], xy[:, 0]]
        recall = (
            float(human_core[xy[evaluable, 1], xy[evaluable, 0]].mean())
            if np.any(evaluable)
            else float("nan")
        )
        wrist = np.rint(_named_point(hand, "wrist")).astype(int)
        wrist[0] = int(np.clip(wrist[0], 0, human_core.shape[1] - 1))
        wrist[1] = int(np.clip(wrist[1], 0, human_core.shape[0] - 1))
        wrist_label = int(labels[wrist[1], wrist[0]])
        crop_connected = float(wrist_label != 0 and wrist_label in border_labels)
        forearm_prompt_recall = float(
            side_metrics[side]["forearm_candidate"]["prompt_recall"]
        )
        observed_boundary_connected = float(
            side_metrics[side]
            .get("anatomy_bounds", {})
            .get("selected_component_boundary_connected", 0.0)
        )
        forearm_supported = float(
            forearm_prompt_recall >= forearm_recall_gate
            and observed_boundary_connected == 1.0
            and crop_connected == 1.0
        )
        side_supported = float(
            np.isfinite(recall)
            and recall >= hand_recall_gate
            and forearm_supported == 1.0
            and float(side_metrics[side].get("independent_hold", 1.0)) == 0.0
        )
        results[side] = {
            "contact_aware_keypoint_recall": recall,
            "evaluable_keypoint_count": float(np.count_nonzero(evaluable)),
            "wrist_component_crop_connected": crop_connected,
            "forearm_prompt_recall": forearm_prompt_recall,
            "observed_forearm_boundary_connected": observed_boundary_connected,
            "forearm_supported": forearm_supported,
            "independent_hold": float(
                side_metrics[side].get("independent_hold", 1.0)
            ),
            "side_supported": side_supported,
        }
    return float(all(value["side_supported"] == 1.0 for value in results.values())), results


def _combine_human_support(final_anatomy_supported: float, temporal_supported: float) -> float:
    """Both independent evidence families must support a candidate."""

    return float(final_anatomy_supported == 1.0 and temporal_supported == 1.0)


def _degradation_flags(
    final_anatomy: dict[str, dict[str, float]],
    side_metrics: dict[str, Any],
    *,
    human_supported: float,
    object_supported: float,
    temporal_supported: float | None,
) -> dict[str, float]:
    """Manifest-safe, explicit reasons for every fail-closed degradation."""

    flags: dict[str, float] = {
        "degraded_reason_human_unsupported": float(human_supported != 1.0),
        "degraded_reason_object_unsupported": float(object_supported != 1.0),
    }
    if temporal_supported is not None:
        flags["degraded_reason_temporal_unsupported"] = float(
            temporal_supported != 1.0
        )
    for side in ("left", "right"):
        flags[f"degraded_reason_{side}_no_evaluable_keypoints"] = float(
            final_anatomy[side]["evaluable_keypoint_count"] == 0.0
        )
        flags[f"degraded_reason_{side}_candidate_hold"] = float(
            side_metrics[side].get("independent_hold", 1.0) == 1.0
        )
    reason_count = float(sum(flags.values()))
    return {"degraded": float(reason_count > 0.0), "degradation_reason_count": reason_count, **flags}


def _json_safe_anatomy_evidence(
    evidence: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    """Serialize undefined recall as JSON null while retaining its HOLD flag."""

    return {
        side: {
            key: (float(value) if np.isfinite(value) else None)
            for key, value in metrics.items()
        }
        for side, metrics in evidence.items()
    }


def _bidirectional_mask_consistency(
    grays: list[np.ndarray], masks: list[np.ndarray]
) -> list[float]:
    if len(grays) != len(masks) or not grays:
        raise RuntimeError("temporal evidence requires equal non-empty image/mask sequences")
    height, width = grays[0].shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    pair_scores: list[float] = []
    for current_gray, next_gray, current_mask, next_mask in zip(
        grays[:-1], grays[1:], masks[:-1], masks[1:], strict=True
    ):
        if not (
            current_gray.shape == next_gray.shape == current_mask.shape == next_mask.shape
            == (height, width)
        ):
            raise RuntimeError("temporal evidence sequence has inconsistent dimensions")
        forward = cv2.calcOpticalFlowFarneback(
            current_gray, next_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        backward = cv2.calcOpticalFlowFarneback(
            next_gray, current_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        current_in_next = cv2.remap(
            current_mask.astype(np.uint8),
            grid_x + backward[..., 0],
            grid_y + backward[..., 1],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
        next_in_current = cv2.remap(
            next_mask.astype(np.uint8),
            grid_x + forward[..., 0],
            grid_y + forward[..., 1],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
        next_iou = float(
            np.count_nonzero(current_in_next & next_mask)
            / max(np.count_nonzero(current_in_next | next_mask), 1)
        )
        current_iou = float(
            np.count_nonzero(next_in_current & current_mask)
            / max(np.count_nonzero(next_in_current | current_mask), 1)
        )
        pair_scores.append(0.5 * (current_iou + next_iou))
    frame_scores: list[float] = []
    for index in range(len(masks)):
        neighbors = []
        if index > 0:
            neighbors.append(pair_scores[index - 1])
        if index < len(masks) - 1:
            neighbors.append(pair_scores[index])
        frame_scores.append(float(np.mean(neighbors)) if neighbors else 1.0)
    return frame_scores


def _select_hand_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    positive_points: np.ndarray,
    box: np.ndarray,
    min_recall: float,
    max_box_area_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1.0)
    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    for mask, sam_score in zip(masks.astype(bool), scores):
        raw_recall = _point_recall(mask, positive_points)
        regularized, component_count = _keep_prompt_bearing_components(
            mask, positive_points
        )
        recall = _point_recall(regularized, positive_points)
        area_ratio = float(regularized.sum() / box_area)
        overflow = max(0.0, area_ratio - max_box_area_ratio)
        rank = 3.0 * recall + float(sam_score) - 2.0 * overflow
        candidates.append(
            (
                rank,
                regularized,
                {
                    "sam_score": float(sam_score),
                    "raw_point_recall": raw_recall,
                    "point_recall": recall,
                    "box_area_ratio": area_ratio,
                    "component_count_before_regularization": float(component_count),
                },
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    eligible = [
        item
        for item in candidates
        if item[2]["point_recall"] >= min_recall
        and item[2]["box_area_ratio"] <= max_box_area_ratio
    ]
    _, best, metrics = eligible[0] if eligible else candidates[0]
    held = not eligible
    held_candidate_pixels = float(np.count_nonzero(best)) if held else 0.0
    if held:
        best = np.zeros_like(best, dtype=bool)
    return best, {
        **metrics,
        "held_candidate_pixels": held_candidate_pixels,
        "held": float(held),
    }


def _select_forearm_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    sleeve_points: np.ndarray,
    wrist: np.ndarray,
    corridor: np.ndarray,
    min_recall: float = 0.60,
    debug_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    corridor_area = max(float(corridor.sum()), 1.0)
    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    all_points = np.concatenate([sleeve_points, wrist[None]], axis=0)
    border = np.zeros_like(corridor, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    for mask, sam_score in zip(masks.astype(bool), scores):
        recall = _point_recall(mask, all_points)
        corridor_precision = float((mask & corridor).sum() / max(mask.sum(), 1))
        boundary = float(bool((mask & border).any()))
        boundary_in_prior = float(bool((mask & border & corridor).any()))
        area_ratio = float(mask.sum() / corridor_area)
        rank = 2.5 * recall + float(sam_score) + corridor_precision + boundary - max(0.0, area_ratio - 1.8)
        candidates.append((rank, mask, {
            "sam_score": float(sam_score),
            "prompt_recall": recall,
            "corridor_precision": corridor_precision,
            "boundary_connection": boundary,
            "boundary_in_prior_corridor": boundary_in_prior,
            "corridor_area_ratio": area_ratio,
        }))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best, raw_metrics = candidates[0]
    regularized, component_count = _keep_prompt_bearing_components(
        best, all_points
    )
    retained_component_count = max(
        cv2.connectedComponents(regularized.astype(np.uint8), connectivity=8)[0] - 1,
        0,
    )
    regularized_recall = _point_recall(regularized, all_points)
    regularized_boundary = float(bool((regularized & border).any()))
    regularized_boundary_in_prior = float(bool((regularized & border & corridor).any()))
    held = regularized_recall < min_recall
    metrics = {
        **raw_metrics,
        "raw_prompt_recall": raw_metrics["prompt_recall"],
        "raw_boundary_connection": raw_metrics["boundary_connection"],
        "prompt_recall": regularized_recall,
        "boundary_connection": regularized_boundary,
        "boundary_in_prior_corridor": regularized_boundary_in_prior,
        "component_count_before_regularization": float(component_count),
        "raw_component_count_before_regularization": float(component_count),
        "retained_prompt_component_count": float(retained_component_count),
        "rejected_disconnected_component_count": float(
            max(component_count - retained_component_count, 0)
        ),
        "held": float(held),
    }
    held_candidate_pixels = float(np.count_nonzero(regularized)) if held else 0.0
    if held:
        regularized = np.zeros_like(regularized, dtype=bool)
        metrics["retained_prompt_component_count"] = 0.0
    metrics["held_candidate_pixels"] = held_candidate_pixels
    if debug_masks is not None:
        debug_masks["raw_selected_forearm_sam"] = best.copy()
        debug_masks["regularized_forearm_sam"] = regularized.copy()
    return regularized, metrics


def _project_cylinder_mask(
    transform: np.ndarray,
    intrinsics: np.ndarray,
    radius: float,
    height_m: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    ys = np.asarray([-height_m / 2.0, height_m / 2.0])
    points = []
    for y in ys:
        for angle in theta:
            points.append([radius * math.cos(angle), y, radius * math.sin(angle), 1.0])
    points.extend([[0.0, -height_m / 2.0, 0.0, 1.0], [0.0, height_m / 2.0, 0.0, 1.0]])
    camera = (transform @ np.asarray(points, dtype=np.float64).T).T[:, :3]
    camera = camera[camera[:, 2] > 1e-5]
    if len(camera) < 6:
        raise RuntimeError("analytic object projects behind camera")
    uvw = (intrinsics @ camera.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    hull = cv2.convexHull(np.rint(uv).astype(np.int32))
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    center_camera = (transform @ np.asarray([0.0, 0.0, 0.0, 1.0]))[:3]
    center_uvw = intrinsics @ center_camera
    center = center_uvw[:2] / center_uvw[2]
    return mask.astype(bool), center.astype(np.float32)


def _select_object_visible(
    predictor: Any,
    rgb: np.ndarray,
    analytic_mask: np.ndarray,
    center: np.ndarray,
    hand_points: np.ndarray,
    min_saturation: float,
    min_analytic_iou: float,
) -> tuple[np.ndarray, dict[str, float]]:
    ys, xs = np.nonzero(analytic_mask)
    if len(xs) == 0:
        raise RuntimeError("empty analytic object silhouette")
    box = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    orange = analytic_mask & (saturation >= min_saturation) & (hsv[..., 0] >= 2) & (hsv[..., 0] <= 35)
    oy, ox = np.nonzero(orange)
    positives: list[np.ndarray] = []
    if len(ox):
        order = np.linspace(0, len(ox) - 1, min(6, len(ox)), dtype=int)
        positives.extend(np.stack([ox[order], oy[order]], axis=1).astype(np.float32))
    if analytic_mask[int(np.clip(round(center[1]), 0, rgb.shape[0] - 1)), int(np.clip(round(center[0]), 0, rgb.shape[1] - 1))]:
        positives.append(center)
    if not positives:
        positives.append(np.asarray([float(np.median(xs)), float(np.median(ys))], np.float32))
    pos = np.asarray(positives, dtype=np.float32).reshape(-1, 2)
    negatives = hand_points[:: max(1, len(hand_points) // 8)] if len(hand_points) else np.empty((0, 2), np.float32)
    prompt_points = np.concatenate([pos, negatives], axis=0)
    labels = np.concatenate([np.ones(len(pos), np.int32), np.zeros(len(negatives), np.int32)])
    masks, scores, _ = predictor.predict(
        point_coords=prompt_points,
        point_labels=labels,
        box=box,
        multimask_output=True,
    )
    best: tuple[float, np.ndarray, dict[str, float]] | None = None
    for mask, score in zip(masks.astype(bool), scores):
        clipped = mask & analytic_mask
        union = float((mask | analytic_mask).sum())
        analytic_iou = float(clipped.sum() / max(union, 1.0))
        orange_recall = float((clipped & orange).sum() / max(orange.sum(), 1))
        hand_leak = _point_recall(clipped, negatives)
        rank = 2.0 * orange_recall + analytic_iou + float(score) - 2.0 * hand_leak
        metrics = {"sam_score": float(score), "analytic_iou": analytic_iou, "orange_recall": orange_recall, "hand_keypoint_leak": hand_leak}
        if best is None or rank > best[0]:
            best = (rank, clipped, metrics)
    assert best is not None
    _, visible, metrics = best
    if metrics["analytic_iou"] < min_analytic_iou and metrics["orange_recall"] < 0.80:
        return np.zeros_like(analytic_mask), {**metrics, "held": 1.0}
    return visible, {**metrics, "held": 0.0}


@dataclass(frozen=True)
class MaskConfig:
    hand_padding: float
    forearm_half_width: float
    forearm_width_ratio: float
    forearm_step: float
    wrist_radius: float
    dilation_ratio: float
    dilation_min: float
    dilation_max: float
    contact_ratio: float
    contact_min: float
    contact_max: float
    min_recall: float
    forearm_min_recall: float
    temporal_iou_gate: float
    temporal_analysis_ratio: float
    max_box_area_ratio: float
    object_core_erosion: float
    object_min_saturation: float
    object_min_iou: float

    @classmethod
    def from_overlay(cls, overlay: dict[str, Any]) -> "MaskConfig":
        return cls(
            hand_padding=float(overlay["hand_box_padding_hand_scale"]),
            forearm_half_width=float(overlay["forearm_half_width_hand_scale"]),
            forearm_width_ratio=float(overlay["k_forearm"]),
            forearm_step=float(overlay["forearm_prompt_step_hand_scale"]),
            wrist_radius=float(overlay["wrist_device_radius_hand_scale"]),
            dilation_ratio=float(overlay["k_dilate"]),
            dilation_min=float(overlay["dilation_min_image_ratio"]),
            dilation_max=float(overlay["dilation_max_image_ratio"]),
            contact_ratio=float(overlay["k_contact"]),
            contact_min=float(overlay["contact_min_image_ratio"]),
            contact_max=float(overlay["contact_max_image_ratio"]),
            min_recall=float(overlay["mask_keypoint_recall_ratio"]),
            forearm_min_recall=float(overlay["forearm_prompt_recall_ratio"]),
            temporal_iou_gate=float(overlay["mask_forward_backward_iou_ratio"]),
            temporal_analysis_ratio=float(
                overlay["temporal_validation_analysis_image_ratio"]
            ),
            max_box_area_ratio=float(overlay["candidate_max_box_area_ratio"]),
            object_core_erosion=float(overlay["object_core_erosion_object_scale"]),
            object_min_saturation=float(overlay["object_visible_saturation_ratio"]),
            object_min_iou=float(overlay["object_visible_analytic_iou_ratio"]),
        )


class _InferencePredictor:
    """Run every SAM call under H20-native BF16 inference without changing API."""

    def __init__(self, predictor: Any, torch_module: Any) -> None:
        self._predictor = predictor
        self._torch = torch_module

    def set_image(self, image: np.ndarray) -> Any:
        with self._torch.inference_mode(), self._torch.autocast(
            device_type="cuda", dtype=self._torch.bfloat16
        ):
            return self._predictor.set_image(image)

    def predict(self, **kwargs: Any) -> Any:
        with self._torch.inference_mode(), self._torch.autocast(
            device_type="cuda", dtype=self._torch.bfloat16
        ):
            return self._predictor.predict(**kwargs)


def _segment_hand_side(
    predictor: Any,
    hand: dict[str, Any],
    width: int,
    height: int,
    config: MaskConfig,
    prompt_exclusion_mask: np.ndarray,
    debug_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    points, names = _valid_hand_points(hand, width, height)
    if len(points) < 8:
        raise RuntimeError("insufficient valid hand keypoints")
    prompt_xy = np.rint(points).astype(int)
    prompt_xy[:, 0] = np.clip(prompt_xy[:, 0], 0, width - 1)
    prompt_xy[:, 1] = np.clip(prompt_xy[:, 1], 0, height - 1)
    prompt_points = points[
        ~prompt_exclusion_mask[prompt_xy[:, 1], prompt_xy[:, 0]]
    ]
    if len(prompt_points) < 8:
        raise RuntimeError(
            "fewer than eight non-contact hand prompts remain after object protection"
        )
    scale = _hand_scale(points)
    padding = config.hand_padding * scale
    hand_box = _box_from_points(points, padding, width, height)
    hand_masks, hand_scores, _ = predictor.predict(
        point_coords=prompt_points,
        point_labels=np.ones(len(prompt_points), np.int32),
        box=hand_box,
        multimask_output=True,
    )
    hand_mask, hand_metrics = _select_hand_candidate(
        hand_masks,
        hand_scores,
        prompt_points,
        hand_box,
        config.min_recall,
        config.max_box_area_ratio,
    )

    wrist = _named_point(hand, "wrist")
    palm = _named_point(hand, "palm_center")
    index_proximal = _named_point(hand, "index_proximal")
    pinky_proximal = _named_point(hand, "pinky_proximal")
    wrist_width = float(np.linalg.norm(index_proximal - pinky_proximal))
    hand_envelope, corridor, anatomy_metrics = _anatomy_support_masks(
        points,
        wrist,
        palm,
        wrist_width,
        forearm_width_ratio=config.forearm_width_ratio,
        width=width,
        height=height,
    )
    end = np.asarray(anatomy_metrics["forearm_end_xy"], np.float32)
    length = float(np.linalg.norm(end - wrist))
    direction = (end - wrist) / max(length, 1e-6)
    steps = np.arange(config.forearm_step * scale, length, config.forearm_step * scale)
    if len(steps) == 0:
        steps = np.asarray([length * 0.5])
    sleeve_points = wrist[None] + steps[:, None] * direction[None]
    sleeve_points = np.concatenate([wrist[None] + 0.15 * scale * direction[None], sleeve_points], axis=0)
    corridor_y, corridor_x = np.nonzero(corridor)
    corridor_corners = np.asarray(
        [
            [corridor_x.min(), corridor_y.min()],
            [corridor_x.max(), corridor_y.max()],
        ],
        np.float32,
    )
    arm_box = _box_from_points(corridor_corners, 0.05 * scale, width, height)
    arm_masks, arm_scores, _ = predictor.predict(
        point_coords=sleeve_points,
        point_labels=np.ones(len(sleeve_points), np.int32),
        box=arm_box,
        multimask_output=True,
    )
    arm_mask, arm_metrics = _select_forearm_candidate(
        arm_masks,
        arm_scores,
        sleeve_points,
        wrist,
        corridor,
        config.forearm_min_recall,
        debug_masks,
    )

    wrist_radius = config.wrist_radius * scale
    wrist_box = np.asarray([
        max(0.0, wrist[0] - wrist_radius),
        max(0.0, wrist[1] - wrist_radius),
        min(width - 1.0, wrist[0] + wrist_radius),
        min(height - 1.0, wrist[1] + wrist_radius),
    ], np.float32)
    wrist_masks, wrist_scores, _ = predictor.predict(
        point_coords=np.asarray([wrist], np.float32),
        point_labels=np.ones(1, np.int32),
        box=wrist_box,
        multimask_output=True,
    )
    wrist_mask, wrist_metrics = _select_hand_candidate(
        wrist_masks, wrist_scores, np.asarray([wrist]), wrist_box, 1.0, 1.5
    )
    wrist_region = np.zeros((height, width), np.uint8)
    cv2.circle(wrist_region, tuple(np.rint(wrist).astype(int)), int(round(wrist_radius)), 1, -1)
    combined, anatomy_bound_metrics = _apply_anatomy_bounds(
        hand_mask,
        wrist_mask,
        wrist_region.astype(bool),
        arm_mask,
        hand_envelope,
        corridor,
        wrist,
        palm,
        wrist_width,
    )
    bounded_hand_recall = _point_recall(combined, prompt_points)
    independent_hold = float(
        hand_metrics["held"] == 1.0
        or arm_metrics["held"] == 1.0
        or wrist_metrics["held"] == 1.0
        or anatomy_bound_metrics["held"] == 1.0
        or bounded_hand_recall < config.min_recall
    )
    anatomy_support = combined.copy()
    if debug_masks is not None:
        debug_masks.update(
            {
                "hand_candidate": hand_mask.copy(),
                "forearm_candidate": arm_mask.copy(),
                "wrist_candidate": wrist_mask.copy(),
                "hand_envelope": hand_envelope.copy(),
                "prior_corridor": corridor.copy(),
                "wrist_region": wrist_region.astype(bool),
                "selected_combined": combined.copy(),
            }
        )
    return combined, anatomy_support, scale, {
        "valid_keypoints": len(points),
        "excluded_contact_prompt_count": len(points) - len(prompt_points),
        "positive_prompt_count": len(prompt_points),
        "joint_names": names,
        "hand_scale_px": scale,
        "hand_box_xyxy": hand_box.tolist(),
        "forearm_end_xy": end.tolist(),
        "forearm_prompt_points_xy": sleeve_points.tolist(),
        "forearm_half_width_px": anatomy_metrics["forearm_half_width_px"],
        "wrist_width_px": wrist_width,
        "anatomy_support": anatomy_metrics,
        "anatomy_bounds": anatomy_bound_metrics,
        "anatomy_bounded_point_recall": bounded_hand_recall,
        "independent_hold": independent_hold,
        "hand_candidate": hand_metrics,
        "forearm_candidate": arm_metrics,
        "wrist_candidate": wrist_metrics,
    }


def _reevaluate_existing_mask_evidence(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    context_path: Path,
    output_root: Path,
    frames: list[dict[str, Any]],
    context: dict[str, Any],
    task_card_path: Path,
    task_card_sha: str,
    config: MaskConfig,
) -> None:
    if not args.reevaluate_input_manifest_sha256:
        raise RuntimeError("MASK reevaluation requires an expected input-manifest SHA")
    previous = _load_json_nofollow(
        Path(args.reevaluate_input_manifest), args.reevaluate_input_manifest_sha256
    )
    if not (
        previous.get("schema_version") == "mask-evidence-v1"
        and previous.get("artifact_state") == "G2_CALIBRATION_CANDIDATE"
        and previous.get("producer") == "mask_producer"
        and previous.get("session_id") == args.session_id
        and previous.get("frame_count") == len(frames)
        and previous.get("source_manifest_ref", {}).get("sha256")
        == args.source_manifest_sha256
        and len(previous.get("frames", [])) == len(frames)
    ):
        raise RuntimeError("input MASK evidence is not an exact compatible G2 candidate")
    if not 0 < config.temporal_analysis_ratio <= 1:
        raise RuntimeError("temporal analysis image ratio must be in (0, 1]")
    records: list[dict[str, Any]] = []
    anatomy_evidence_by_frame: list[dict[str, dict[str, float]]] = []
    temporal_grays: list[np.ndarray] = []
    temporal_masks: list[np.ndarray] = []
    for expected_index, (source_frame, old_record) in enumerate(
        zip(frames, previous["frames"], strict=True)
    ):
        if (
            int(old_record["frame_index"]) != expected_index
            or old_record["source_frame_sha256"] != source_frame["image"]["sha256"]
        ):
            raise RuntimeError("MASK reevaluation frame identity mismatch")
        human_core = _load_binary_mask_ref(old_record["h_core"])
        object_core = _load_binary_mask_ref(old_record["o_visible_core"])
        uncertain_contact = _load_binary_mask_ref(old_record["u_contact"])
        if not (
            human_core.shape == object_core.shape == uncertain_contact.shape
            == (int(source_frame["height"]), int(source_frame["width"]))
        ):
            raise RuntimeError("MASK reevaluation dimensions disagree with source manifest")
        metadata = _load_json_nofollow(
            Path(source_frame["metadata"]["path"]),
            source_frame["metadata"]["sha256"],
        )
        prompt = _load_json_nofollow(
            Path(old_record["prompt_provenance"]["path"]),
            old_record["prompt_provenance"]["sha256"],
        )
        supported, final_anatomy = _final_anatomy_evidence(
            metadata,
            human_core,
            object_core,
            uncertain_contact,
            prompt["sides"],
            hand_recall_gate=config.min_recall,
            forearm_recall_gate=config.forearm_min_recall,
        )
        metrics = dict(old_record["calibration_metrics"])
        metrics["human_mask_supported"] = supported
        metrics["final_anatomy_supported"] = supported
        for side in ("left", "right"):
            for metric, value in final_anatomy[side].items():
                metrics[f"{side}_{metric}_defined"] = float(np.isfinite(value))
                if np.isfinite(value):
                    metrics[f"{side}_{metric}"] = float(value)
        record = dict(old_record)
        record["calibration_metrics"] = metrics
        records.append(record)
        anatomy_evidence_by_frame.append(final_anatomy)
        rgb = _load_rgb(source_frame)
        target_size = (
            max(1, int(round(rgb.shape[1] * config.temporal_analysis_ratio))),
            max(1, int(round(rgb.shape[0] * config.temporal_analysis_ratio))),
        )
        temporal_grays.append(
            cv2.resize(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                target_size,
                interpolation=cv2.INTER_AREA,
            )
        )
        temporal_masks.append(
            cv2.resize(
                human_core.astype(np.uint8),
                target_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        )

    temporal_scores = _bidirectional_mask_consistency(
        temporal_grays, temporal_masks
    )
    for record, temporal_score, final_anatomy in zip(
        records, temporal_scores, anatomy_evidence_by_frame, strict=True
    ):
        metrics = record["calibration_metrics"]
        metrics["forward_backward_consistency"] = temporal_score
        metrics["temporal_evidence_supported"] = float(
            temporal_score >= config.temporal_iou_gate
        )
        metrics["human_mask_supported"] = _combine_human_support(
            metrics["final_anatomy_supported"],
            metrics["temporal_evidence_supported"],
        )
        side_metrics = {
            side: {
                "independent_hold": metrics.get(f"{side}_independent_hold", 1.0)
            }
            for side in ("left", "right")
        }
        metrics.update(
            _degradation_flags(
                final_anatomy,
                side_metrics,
                human_supported=metrics["human_mask_supported"],
                object_supported=metrics.get("object_visible_supported", 0.0),
                temporal_supported=metrics["temporal_evidence_supported"],
            )
        )
        record["forward_backward_consistency"] = temporal_score

    context_payload = _read_regular_nofollow(context_path)
    reevaluated = {
        "schema_version": "mask-evidence-v1",
        "artifact_state": "G2_CALIBRATION_CANDIDATE",
        "producer": "mask_producer",
        "session_id": args.session_id,
        "product_line": "004_CONTACT_GOLD",
        "source_manifest_ref": {
            "path": str(manifest_path),
            "sha256": args.source_manifest_sha256,
            "producer": "source_resolver",
        },
        "session_context_ref": {
            "path": str(context_path),
            "sha256": _sha256_bytes(context_payload),
            "producer": "session_profiler",
        },
        "task_card_ref": {
            "path": str(task_card_path),
            "sha256": task_card_sha,
            "producer": "explicit_task_card",
        },
        "calibration_overlay_ref": {
            "path": str(task_card_path),
            "sha256": task_card_sha,
            "producer": "g2_task_card_calibration_overlay",
        },
        "model": dict(previous["model"]),
        "sam3_used": False,
        "frame_count": len(records),
        "frames": records,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    schema = _load_json_nofollow(Path(args.schema))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(reevaluated)
    output_root.mkdir(parents=True, exist_ok=False)
    _atomic_write_bytes(
        output_root / "mask_evidence_manifest.json",
        (json.dumps(reevaluated, indent=2, ensure_ascii=False) + "\n").encode(),
    )


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.source_manifest).resolve()
    context_path = Path(args.session_context).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty MASK output: {output_root}")
    source_manifest = _load_json_nofollow(manifest_path, args.source_manifest_sha256)
    context = _load_json_nofollow(context_path)
    task_card_path = Path(args.profile_overlay).resolve()
    task_card_payload = _read_regular_nofollow(task_card_path)
    task_card_sha = _sha256_bytes(task_card_payload)
    overlay = yaml.safe_load(task_card_payload.decode("utf-8"))
    if not isinstance(overlay, dict) or not isinstance(
        overlay.get("dimensionless_calibration_overlay"), dict
    ):
        raise RuntimeError("profile overlay must be the explicit task-card section")
    config = MaskConfig.from_overlay(overlay["dimensionless_calibration_overlay"])
    if not (
        source_manifest.get("document_status") == "VERIFIED_G0_CALIBRATION_INPUT"
        and source_manifest.get("promotion_scope") == "G2_004_MASK_CLEAN_CALIBRATION_ONLY"
        and source_manifest.get("formal_production_allowed") is False
        and source_manifest.get("no_fallback") is True
        and args.session_id in source_manifest.get("authorized_calibration_sessions", [])
    ):
        raise RuntimeError("source manifest is not authorized for G2 004 calibration")
    if not (
        context.get("execution_allowed") is True
        and context.get("execution_blockers") == []
        and context.get("execution_mode") == "G2_CALIBRATION"
        and context.get("formal_pass_claim_allowed") is False
        and context.get("product_line") == "004_CONTACT_GOLD"
        and context.get("source", {}).get("source_manifest_sha256")
        == args.source_manifest_sha256
        and context.get("task_card_ref", {}).get("sha256") == task_card_sha
        and context.get("calibration_overlay_ref", {}).get("sha256") == task_card_sha
    ):
        raise RuntimeError("session context does not authorize this exact G2 producer input")
    if not (
        overlay.get("status") == "AUTHORIZED_G2_CALIBRATION"
        and overlay.get("session_id") == args.session_id
        and overlay.get("product_line") == "004_CONTACT_GOLD"
    ):
        raise RuntimeError("task card does not authorize requested calibration")
    sessions = [item for item in source_manifest["sessions"] if item["session_id"] == args.session_id]
    if len(sessions) != 1:
        raise RuntimeError(f"source manifest must contain exactly one requested session: {args.session_id}")
    session = sessions[0]
    if context["session_id"] != args.session_id:
        raise RuntimeError("session context identity mismatch")
    frames = session["frames"]
    if len(frames) != int(session["frame_count"]):
        raise RuntimeError("source manifest frame count mismatch")

    if args.reevaluate_input_manifest:
        _reevaluate_existing_mask_evidence(
            args,
            manifest_path=manifest_path,
            context_path=context_path,
            output_root=output_root,
            frames=frames,
            context=context,
            task_card_path=task_card_path,
            task_card_sha=task_card_sha,
            config=config,
        )
        return

    _verify_regular_sha256_nofollow(Path(args.sam2_checkpoint), args.sam2_checkpoint_sha256)
    _verify_regular_sha256_nofollow(Path(args.sam2_config_file), args.sam2_config_sha256)
    _verify_regular_sha256_nofollow(Path(args.object_pose_npz), args.object_pose_sha256)
    os.environ.setdefault("PYTHONPATH", args.sam2_root)
    import sys
    if args.sam2_root not in sys.path:
        sys.path.insert(0, args.sam2_root)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("G2 calibration task card pins CUDA execution; CUDA is unavailable")
    model = build_sam2(args.sam2_config, args.sam2_checkpoint, device="cuda")
    predictor = _InferencePredictor(SAM2ImagePredictor(model), torch)
    with np.load(args.object_pose_npz, allow_pickle=False) as object_npz:
        transforms = np.asarray(object_npz["T_object_to_camera"], dtype=np.float64)
        object_valid = np.asarray(object_npz["valid"], dtype=bool)
        object_confidence = np.asarray(object_npz["confidence"], dtype=float)
        radius = float(object_npz["cylinder_radius_m"])
        cylinder_height = float(object_npz["cylinder_height_m"])
    if transforms.shape != (len(frames), 4, 4):
        raise RuntimeError("Object6D frame count/shape mismatch")

    records: list[dict[str, Any]] = []
    per_frame_dir = output_root / "frames"
    per_frame_dir.mkdir(parents=True, exist_ok=False)
    for expected_index, frame in enumerate(frames):
        if int(frame["frame_index"]) != expected_index:
            raise RuntimeError("non-contiguous source frame order")
        rgb = _load_rgb(frame)
        height, width = rgb.shape[:2]
        metadata = _load_json_nofollow(Path(frame["metadata"]["path"]), frame["metadata"]["sha256"])
        if int(metadata["metadata"]["idx"]) != expected_index:
            raise RuntimeError("metadata frame identity mismatch")
        intrinsics = np.asarray(metadata["metadata"]["k"], dtype=np.float64)
        predictor.set_image(rgb)

        analytic_object, object_center = _project_cylinder_mask(
            transforms[expected_index],
            intrinsics,
            radius,
            cylinder_height,
            width,
            height,
        )
        object_scale = math.sqrt(float(analytic_object.sum()))
        contact = _contact_band_radius_px(
            int(np.count_nonzero(analytic_object)),
            object_scale_ratio=config.contact_ratio,
            image_min_dimension=min(width, height),
            low_resolution_ratio=config.contact_min,
            high_resolution_ratio=config.contact_max,
        )
        contact_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * contact + 1, 2 * contact + 1)
        )
        prompt_exclusion_mask = cv2.dilate(
            analytic_object.astype(np.uint8), contact_kernel
        ).astype(bool)

        side_masks: dict[str, np.ndarray] = {}
        side_anatomy_supports: dict[str, np.ndarray] = {}
        side_metrics: dict[str, Any] = {}
        all_hand_points = []
        scales = []
        for side in ("left", "right"):
            hand = metadata["entities"]["hands"][side]
            mask, anatomy_support, scale, metrics = _segment_hand_side(
                predictor,
                hand,
                width,
                height,
                config,
                prompt_exclusion_mask,
            )
            side_masks[side] = mask
            side_anatomy_supports[side] = anatomy_support
            side_metrics[side] = metrics
            scales.append(scale)
            valid_points, _ = _valid_hand_points(hand, width, height)
            all_hand_points.append(valid_points)

        visible_object, object_metrics = _select_object_visible(
            predictor,
            rgb,
            analytic_object,
            object_center,
            np.concatenate(all_hand_points, axis=0),
            config.object_min_saturation,
            config.object_min_iou,
        )
        erosion = max(1, int(round(config.object_core_erosion * object_scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
        object_candidate = cv2.erode(visible_object.astype(np.uint8), kernel).astype(bool)

        human_raw = side_masks["left"] | side_masks["right"]
        median_hand_scale = float(np.median(scales))
        dilation = _clamp_px(
            config.dilation_ratio * median_hand_scale,
            min(width, height),
            config.dilation_min,
            config.dilation_max,
        )
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        human_core = cv2.morphologyEx(human_raw.astype(np.uint8), cv2.MORPH_CLOSE, dilate_kernel)
        human_core = cv2.dilate(human_core, dilate_kernel).astype(bool)
        anatomy_support = cv2.dilate(
            human_raw.astype(np.uint8), dilate_kernel
        ).astype(bool)
        human_core &= anatomy_support
        morphology_added_pixels = float(np.count_nonzero(human_core & ~human_raw))
        human_core, object_core, uncertain_contact, contact_metrics = _assign_contact_ownership(
            human_core,
            object_candidate,
            analytic_object,
            contact_kernel,
        )
        instance = np.zeros((height, width), np.uint8)
        instance[side_masks["left"] & human_core] = 1
        instance[side_masks["right"] & human_core] = 2
        instance[object_core] = 3

        human_confidence = float(
            np.mean(
                [
                    side_metrics[side][part]["sam_score"]
                    for side in ("left", "right")
                    for part in ("hand_candidate", "forearm_candidate", "wrist_candidate")
                ]
            )
        )
        object_confidence_sam = float(object_metrics["sam_score"])
        human_mask_supported, final_anatomy = _final_anatomy_evidence(
            metadata,
            human_core,
            object_core,
            uncertain_contact,
            side_metrics,
            hand_recall_gate=config.min_recall,
            forearm_recall_gate=config.forearm_min_recall,
        )
        object_visible_supported = float(object_metrics.get("held", 0.0) == 0.0)
        degradation = _degradation_flags(
            final_anatomy,
            side_metrics,
            human_supported=human_mask_supported,
            object_supported=object_visible_supported,
            temporal_supported=None,
        )
        confidence = np.zeros((height, width), np.uint8)
        confidence[human_core] = np.uint8(np.clip(round(255 * human_confidence), 0, 255))
        confidence[object_core] = np.uint8(np.clip(round(255 * object_confidence_sam), 0, 255))

        frame_dir = per_frame_dir / f"{expected_index:05d}"
        outputs = {
            "h_core": _write_png(frame_dir / "H_core.png", human_core.astype(np.uint8) * 255),
            "instance_id": _write_png(frame_dir / "instance_id.png", instance),
            "o_visible_core": _write_png(frame_dir / "O_visible_core.png", object_core.astype(np.uint8) * 255),
            "u_contact": _write_png(frame_dir / "U_contact.png", uncertain_contact.astype(np.uint8) * 255),
            "object_analytic": _write_png(frame_dir / "object_analytic.png", analytic_object.astype(np.uint8) * 255),
            "confidence": _write_png(frame_dir / "confidence.png", confidence),
        }
        prompt_provenance = _write_json_ref(
            frame_dir / "prompt_provenance.json",
            {
                "source_metadata": frame["metadata"],
                "task_card_sha256": task_card_sha,
                "sides": side_metrics,
                "final_anatomy_evidence": _json_safe_anatomy_evidence(final_anatomy),
                "object": object_metrics,
                "degraded": bool(degradation["degraded"]),
                "degradation_reasons": [
                    key
                    for key, value in degradation.items()
                    if key.startswith("degraded_reason_") and value == 1.0
                ],
                "dilation_px": dilation,
                "contact_band_px": contact,
                "contact_ownership": contact_metrics,
                "temporal_consistency_status": "PENDING_SEPARATE_MACHINE_QA",
            },
        )
        evidence_record = {
            "frame_index": expected_index,
            "timestamp_ns": int(frame["timestamp_ns"]),
            "video_time_s": float(frame["video_time_s"]),
            "source_image_sha256": frame["image"]["sha256"],
            "object6d_valid": bool(object_valid[expected_index]),
            "object6d_confidence": float(object_confidence[expected_index]),
            "hand_scale_px": median_hand_scale,
            "object_scale_px": object_scale,
            "dilation_px": dilation,
            "contact_band_px": contact,
            "contact_ownership": contact_metrics,
            "human_area_ratio": float(human_core.mean()),
            "morphology_added_pixels": morphology_added_pixels,
            "post_morphology_local_support_radius_px": float(dilation),
            "object_visible_ratio": float(object_core.sum() / max(analytic_object.sum(), 1)),
            "sides": side_metrics,
            "object": object_metrics,
            "outputs": outputs,
            "prompt_provenance": prompt_provenance,
        }
        _write_json_ref(frame_dir / "evidence.json", evidence_record)
        records.append(
            {
                "frame_index": expected_index,
                "source_frame_sha256": frame["image"]["sha256"],
                "h_core": outputs["h_core"],
                "o_visible_core": outputs["o_visible_core"],
                "u_contact": outputs["u_contact"],
                "instance_id": outputs["instance_id"],
                "confidence": outputs["confidence"],
                "prompt_provenance": prompt_provenance,
                "forward_backward_consistency": 0.0,
                "calibration_metrics": {
                    "object6d_valid": float(bool(object_valid[expected_index])),
                    "object6d_confidence": float(object_confidence[expected_index]),
                    "hand_scale_px": median_hand_scale,
                    "object_scale_px": object_scale,
                    "dilation_px": float(dilation),
                    "contact_band_px": float(contact),
                    **contact_metrics,
                    "human_area_ratio": float(human_core.mean()),
                    "morphology_added_pixels": morphology_added_pixels,
                    "post_morphology_local_support_radius_px": float(dilation),
                    "object_visible_ratio": float(object_core.sum() / max(analytic_object.sum(), 1)),
                    "sam_human_confidence": human_confidence,
                    "sam_object_confidence": object_confidence_sam,
                    "left_independent_hold": float(
                        side_metrics["left"]["independent_hold"]
                    ),
                    "right_independent_hold": float(
                        side_metrics["right"]["independent_hold"]
                    ),
                    "human_mask_supported": human_mask_supported,
                    "object_visible_supported": object_visible_supported,
                    "temporal_consistency_pending": 1.0,
                    **degradation,
                },
            }
        )

    context_payload = _read_regular_nofollow(context_path)
    manifest = {
        "schema_version": "mask-evidence-v1",
        "artifact_state": "G2_CALIBRATION_CANDIDATE",
        "producer": "mask_producer",
        "session_id": args.session_id,
        "product_line": "004_CONTACT_GOLD",
        "frame_count": len(records),
        "source_manifest_ref": {
            "path": str(manifest_path), "sha256": args.source_manifest_sha256,
            "producer": "source_resolver",
        },
        "session_context_ref": {
            "path": str(context_path), "sha256": _sha256_bytes(context_payload),
            "producer": "session_profiler",
        },
        "task_card_ref": {
            "path": str(task_card_path), "sha256": task_card_sha,
            "producer": "explicit_task_card",
        },
        "calibration_overlay_ref": {
            "path": str(task_card_path), "sha256": task_card_sha,
            "producer": "g2_task_card_calibration_overlay",
        },
        "model": {
            "identity": "SAM2_1_HIERA_LARGE",
            "repository": "https://github.com/IDEA-Research/Grounded-SAM-2.git",
            "commit": "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28",
            "config_path": "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "config_sha256": args.sam2_config_sha256,
            "weight_path": str(Path(args.sam2_checkpoint)),
            "weight_sha256": args.sam2_checkpoint_sha256,
        },
        "sam3_used": False,
        "frames": records,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    schema = _load_json_nofollow(Path(args.schema))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(output_root / "mask_evidence_manifest.json", encoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--session-context", required=True)
    parser.add_argument("--profile-overlay", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sam2-root", required=True)
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-config-file", required=True)
    parser.add_argument("--sam2-config-sha256", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--sam2-checkpoint-sha256", required=True)
    parser.add_argument("--object-pose-npz", required=True)
    parser.add_argument("--object-pose-sha256", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--reevaluate-input-manifest")
    parser.add_argument("--reevaluate-input-manifest-sha256")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
