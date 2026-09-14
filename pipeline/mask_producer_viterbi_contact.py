#!/usr/bin/env python3
"""T1 candidate-only reconstruction of the historical v11/v12 MASK architecture.

This is a new ``mask_producer`` implementation.  It does not inherit the v9
heuristic line and never reads human labels.  Optical flow may score current-
frame SAM candidates and align contact evidence; it is never used to create or
warp an output MASK.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml

# The registered SAM implementation inventory is immutable input.  Enforce this
# in code rather than relying on the caller's PYTHONDONTWRITEBYTECODE setting.
sys.dont_write_bytecode = True


PRODUCER_ID = "producer_p1"
COMPONENT_ID = "mask_producer"
NOT_A_V9_HEURISTIC_PATCH = True
LINEAGE = (
    "No inheritance from the v9 heuristic line; architecture reconstructed only "
    "from the historical deleted-v11 specification."
)
METHOD_STAGE1 = "independent_per_frame_part_aware_hawor_prompted_sam2"
METHOD_STAGE2 = "global_viterbi_mutual_flow_current_frame_candidate_selection"
METHOD_STAGE3 = "pose_prior_occlusion_holes_plus_symmetric_distance_median"
METHOD_STAGE4 = "v12_narrow_contact_band_ownership_noncontact_bit_exact"
PIXEL_SEMANTICS = {
    "arm_candidate": "union of current-frame SAM2 hand/wrist/device/sleeve candidates selected by full-sequence Viterbi",
    "object_observation": "current-frame SAM2 visible-object candidate",
    "pose_prior": "full analytic cylinder projection from Object6D; evidence/protection only",
    "final_arm": "stage3 mutually-exclusive ownership, optionally refined only inside the stage4 contact band",
}
PIXEL_SEMANTICS_SHA = hashlib.sha256(
    json.dumps(PIXEL_SEMANTICS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_RUNS_ROOT = PROJECT_ROOT / "_run"

HAND_SEGMENTS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class ProducerConfig:
    flow_analysis_long_side: int = 640
    viterbi_unary_weight: float = 1.0
    viterbi_transition_weight: float = 2.0
    local_support_padding_hand_scale: float = 0.65
    local_forearm_extent_hand_scale: float = 2.25
    digit_tube_hand_scale: float = 0.075
    contact_band_object_scale_ratio: float = 0.08
    contact_depth_margin_object_radius: float = 0.24
    symmetric_sdf_radius_frames: int = 2


@dataclass
class SamCandidate:
    mask: np.ndarray
    unary: float
    sam_score: float
    positive_recall: float
    negative_pollution: float
    outside_support_ratio: float
    boundary_ratio: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_inventory(path: Path) -> tuple[int, int, str]:
    """Exact inventory algorithm registered by A0c and the model ASSET_PIN."""
    root = path.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"implementation is missing, symlinked, or not a directory: {root}")
    rows: list[str] = []
    regular_bytes = 0
    for value in sorted(root.rglob("*")):
        relative = value.relative_to(root).as_posix()
        if value.is_symlink():
            rows.append(f"{relative}\0symlink\0{os.readlink(value)}\n")
        elif value.is_file():
            size = value.stat().st_size
            regular_bytes += size
            rows.append(f"{relative}\0file\0{size}\0{sha256_file(value)}\n")
    digest = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    return len(rows), regular_bytes, digest


def validate_candidate_root(path_value: str | Path, *, require_new: bool) -> Path:
    root = Path(path_value).resolve()
    allowed = CANDIDATE_RUNS_ROOT.resolve()
    if root.parent != allowed or not root.name.startswith("a2_"):
        raise ValueError("output root must be a new direct child named a2_* under project/_run")
    if require_new:
        allowed.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(root)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite existing candidate root: {root}") from error
    elif not root.is_dir() or root.is_symlink():
        raise ValueError("candidate run root is missing, symlinked, or not a directory")
    return root


def exclusive_bytes(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"output parent is missing, symlinked, or not a directory: {path.parent}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def validate_registered_model_asset(
    refs: dict[str, dict[str, Any]],
    implementation_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Require actual bytes, ASSET_PIN, and both governing contracts to agree."""
    asset_pin = json.loads(Path(refs["model_asset_pin"]["path"]).read_text(encoding="utf-8"))
    pipeline_contract = yaml.safe_load(
        Path(refs["pipeline_contract"]["path"]).read_text(encoding="utf-8")
    )
    project_profile = yaml.safe_load(
        Path(refs["project_profile"]["path"]).read_text(encoding="utf-8")
    )
    if asset_pin.get("model_identifier") != "SAM2_1_HIERA_LARGE":
        raise ValueError("ASSET_PIN model identifier mismatch")
    local = asset_pin.get("local", {})
    pipeline_pin = pipeline_contract.get("model_pins", {}).get("SAM2_1_HIERA_LARGE", {})
    profile_pin = project_profile.get("dependencies", {}).get("sam2_1_mask_evidence", {})
    inventory = implementation_inventory(implementation_root)
    expected_implementation = (PROJECT_ROOT / local.get("implementation_ref", "")).resolve()
    expected_weight = (PROJECT_ROOT / local.get("weight_path", "")).resolve()
    expected_config = (implementation_root / local.get("config_path", "")).resolve()
    asset_pin_path = Path(refs["model_asset_pin"]["path"]).resolve()
    actual_weight = Path(refs["model_weight"]["path"]).resolve()
    comparisons = {
        "implementation_path": implementation_root == expected_implementation,
        "weight_path": actual_weight == expected_weight,
        "config_path": config_path == expected_config,
        "inventory_entries": inventory[0] == local.get("implementation_inventory_entries"),
        "inventory_bytes": inventory[1] == local.get("implementation_regular_bytes"),
        "inventory_sha_asset": inventory[2] == local.get("implementation_inventory_sha256"),
        "inventory_sha_pipeline": inventory[2] == pipeline_pin.get("implementation_inventory_sha256"),
        "inventory_sha_profile": inventory[2] == profile_pin.get("implementation_inventory_sha256"),
        "asset_ref_pipeline": asset_pin_path == (PROJECT_ROOT / pipeline_pin.get("asset_pin_ref", "")).resolve(),
        "asset_ref_profile": asset_pin_path == (PROJECT_ROOT / profile_pin.get("asset_pin_ref", "")).resolve(),
        "asset_sha_pipeline": refs["model_asset_pin"]["sha256"] == pipeline_pin.get("asset_pin_sha256"),
        "asset_sha_profile": refs["model_asset_pin"]["sha256"] == profile_pin.get("asset_pin_sha256"),
        "weight_bytes_asset": refs["model_weight"]["bytes"] == local.get("weight_bytes"),
        "weight_bytes_pipeline": refs["model_weight"]["bytes"] == pipeline_pin.get("weight_bytes"),
        "weight_bytes_profile": refs["model_weight"]["bytes"] == profile_pin.get("weight_bytes"),
        "weight_sha_asset": refs["model_weight"]["sha256"] == local.get("weight_sha256"),
        "weight_sha_pipeline": refs["model_weight"]["sha256"] == pipeline_pin.get("weight_sha256"),
        "weight_sha_profile": refs["model_weight"]["sha256"] == profile_pin.get("weight_sha256"),
        "config_sha_asset": sha256_file(config_path) == local.get("config_sha256"),
        "config_sha_pipeline": sha256_file(config_path) == pipeline_pin.get("config_sha256"),
        "config_sha_profile": sha256_file(config_path) == profile_pin.get("config_sha256"),
        "config_path_pipeline": local.get("config_path") == pipeline_pin.get("config"),
        "config_path_profile": local.get("config_path") == profile_pin.get("config_path"),
        "implementation_ref_pipeline": local.get("implementation_ref") == pipeline_pin.get("implementation_ref"),
        "implementation_ref_profile": local.get("implementation_ref") == profile_pin.get("implementation_ref"),
        "weight_ref_pipeline": local.get("weight_path") == pipeline_pin.get("weight_path"),
        "weight_ref_profile": local.get("weight_path") == profile_pin.get("weight_path"),
    }
    failed = sorted(key for key, value in comparisons.items() if not value)
    if failed:
        raise ValueError(f"registered SAM2.1 pin mismatch: {','.join(failed)}")
    return {
        "model_identifier": "SAM2_1_HIERA_LARGE",
        "asset_pin_sha256": refs["model_asset_pin"]["sha256"],
        "pipeline_contract_sha256": refs["pipeline_contract"]["sha256"],
        "project_profile_sha256": refs["project_profile"]["sha256"],
        "implementation_inventory_entries": inventory[0],
        "implementation_regular_bytes": inventory[1],
        "implementation_inventory_sha256": inventory[2],
        "config_path": local["config_path"],
        "config_sha256": local["config_sha256"],
        "weight_path": local["weight_path"],
        "weight_bytes": local["weight_bytes"],
        "weight_sha256": local["weight_sha256"],
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    exclusive_bytes(path, encoded)


def write_mask(path: Path, mask: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    exclusive_bytes(path, encoded.tobytes())


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    exclusive_bytes(path, buffer.getvalue())


def safe_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    finite = np.isfinite(values).all(axis=1)
    inside = (
        (values[:, 0] >= 0)
        & (values[:, 0] < width)
        & (values[:, 1] >= 0)
        & (values[:, 1] < height)
    )
    return values[finite & inside]


def mask_point_fraction(mask: np.ndarray, points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    xy = np.rint(points).astype(np.int32)
    xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
    return float(mask[xy[:, 1], xy[:, 0]].mean())


def hand_scale(joints: np.ndarray) -> float:
    joints = np.asarray(joints, dtype=np.float32)
    distances = np.linalg.norm(joints[None, :, :] - joints[:, None, :], axis=-1)
    finite = distances[np.isfinite(distances)]
    return float(np.percentile(finite, 90)) if finite.size else 0.0


def local_forearm_direction(joints: np.ndarray) -> np.ndarray | None:
    wrist = joints[0]
    palm = np.nanmean(joints[[5, 9, 13, 17]], axis=0)
    direction = wrist - palm
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    return direction / norm


def part_prompt_points(joints: np.ndarray, part: str) -> np.ndarray:
    """Finite local prompts only; no ray, corridor, polygon, or border target."""
    scale = hand_scale(joints)
    direction = local_forearm_direction(joints)
    if scale <= 0 or direction is None:
        return np.empty((0, 2), dtype=np.float32)
    wrist = joints[0]
    if part == "hand":
        return joints[[0, 4, 8, 12, 16, 20]].astype(np.float32)
    if part == "wrist_band":
        return np.vstack([wrist, np.nanmean(joints[[5, 9, 13, 17]], axis=0)]).astype(np.float32)
    if part == "wrist_device":
        return np.vstack([wrist + direction * scale * offset for offset in (0.30, 0.55)]).astype(np.float32)
    if part == "connected_sleeve":
        return np.vstack([wrist + direction * scale * offset for offset in (0.75, 1.25, 1.90)]).astype(np.float32)
    raise ValueError(f"unknown part {part}")


def local_support_box(
    joints: np.ndarray, width: int, height: int, config: ProducerConfig
) -> tuple[int, int, int, int]:
    scale = hand_scale(joints)
    direction = local_forearm_direction(joints)
    if scale <= 0 or direction is None:
        raise ValueError("degenerate hand geometry")
    points = np.vstack(
        [joints, joints[0] + direction * scale * config.local_forearm_extent_hand_scale]
    )
    padding = config.local_support_padding_hand_scale * scale
    x0 = max(0, int(math.floor(float(np.nanmin(points[:, 0]) - padding))))
    y0 = max(0, int(math.floor(float(np.nanmin(points[:, 1]) - padding))))
    x1 = min(width, int(math.ceil(float(np.nanmax(points[:, 0]) + padding))))
    y1 = min(height, int(math.ceil(float(np.nanmax(points[:, 1]) + padding))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty local support box")
    return x0, y0, x1, y1


def boundary_ratio(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area == 0:
        return 1.0
    border = np.zeros_like(mask, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return float(np.logical_and(mask, border).sum() / area)


def score_sam_candidates(
    masks: np.ndarray,
    scores: np.ndarray,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
    support_box: tuple[int, int, int, int],
) -> list[SamCandidate]:
    height, width = masks.shape[-2:]
    x0, y0, x1, y1 = support_box
    support = np.zeros((height, width), dtype=bool)
    support[y0:y1, x0:x1] = True
    candidates: list[SamCandidate] = []
    for mask_value, score_value in zip(masks, scores, strict=True):
        mask = np.asarray(mask_value, dtype=bool)
        area = int(mask.sum())
        if area == 0:
            continue
        positive_recall = mask_point_fraction(mask, positive_points)
        negative_pollution = mask_point_fraction(mask, negative_points) if len(negative_points) else 0.0
        outside = float(np.logical_and(mask, ~support).sum() / area)
        touches = boundary_ratio(mask)
        unary = (
            float(score_value)
            + positive_recall
            - negative_pollution
            - outside
            - touches
        )
        candidates.append(
            SamCandidate(
                mask=mask,
                unary=unary,
                sam_score=float(score_value),
                positive_recall=positive_recall,
                negative_pollution=negative_pollution,
                outside_support_ratio=outside,
                boundary_ratio=touches,
            )
        )
    return sorted(candidates, key=lambda item: item.unary, reverse=True)


def predict_part_candidates(
    predictor: Any,
    joints: np.ndarray,
    other_points: np.ndarray,
    object_negative_points: np.ndarray,
    part: str,
    image_shape: tuple[int, int],
    config: ProducerConfig,
) -> list[SamCandidate]:
    height, width = image_shape
    positive = safe_points(part_prompt_points(joints, part), width, height)
    negative = safe_points(
        np.vstack([value for value in (other_points, object_negative_points) if len(value)]),
        width,
        height,
    ) if len(other_points) or len(object_negative_points) else np.empty((0, 2), np.float32)
    if len(positive) == 0:
        return []
    point_coords = np.vstack([positive, negative]).astype(np.float32)
    point_labels = np.concatenate(
        [np.ones(len(positive), np.int32), np.zeros(len(negative), np.int32)]
    )
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return score_sam_candidates(
        masks,
        scores,
        positive,
        negative,
        local_support_box(joints, width, height, config),
    )


def assemble_current_frame_arm_candidates(
    part_candidates: dict[str, list[SamCandidate]],
) -> list[SamCandidate]:
    """Union only SAM pixels observed in this frame.

    The hand is the required anchor. Wrist/device/sleeve are optional because
    their evidence points can legitimately lie outside the image when an arm
    enters through a border. An unobservable optional part contributes no
    pixels; it is never replaced by geometry, a previous mask, or an empty
    pseudo-SAM candidate.
    """
    if not part_candidates.get("hand"):
        raise ValueError("hand anchor requires a current-frame SAM candidate")
    fixed_components = [part_candidates["hand"][0]]
    for part in ("wrist_band", "wrist_device"):
        if part_candidates.get(part):
            fixed_components.append(part_candidates[part][0])
    fixed = np.logical_or.reduce([value.mask for value in fixed_components])
    sleeves: list[SamCandidate | None] = list(part_candidates.get("connected_sleeve", []))
    if not sleeves:
        sleeves = [None]
    output: list[SamCandidate] = []
    for sleeve in sleeves:
        components = fixed_components + ([] if sleeve is None else [sleeve])
        union = fixed.copy() if sleeve is None else (fixed | sleeve.mask)
        output.append(
            SamCandidate(
                mask=union,
                unary=float(np.mean([value.unary for value in components])),
                sam_score=float(np.mean([value.sam_score for value in components])),
                positive_recall=float(np.mean([value.positive_recall for value in components])),
                negative_pollution=float(np.mean([value.negative_pollution for value in components])),
                outside_support_ratio=float(np.mean([value.outside_support_ratio for value in components])),
                boundary_ratio=float(np.mean([value.boundary_ratio for value in components])),
            )
        )
    return sorted(output, key=lambda item: item.unary, reverse=True)


def choose_object_observation(
    predictor: Any,
    pose_prior: np.ndarray,
    hand_points: np.ndarray,
) -> SamCandidate | None:
    ys, xs = np.nonzero(pose_prior)
    if len(xs) == 0:
        return None
    center = np.array([[float(np.median(xs)), float(np.median(ys))]], np.float32)
    negative = safe_points(hand_points, pose_prior.shape[1], pose_prior.shape[0])
    coords = np.vstack([center, negative])
    labels = np.concatenate([np.ones(1, np.int32), np.zeros(len(negative), np.int32)])
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
    )
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    candidates = score_sam_candidates(masks, scores, center, negative, (x0, y0, x1, y1))
    if not candidates:
        return None
    # CAD is evidence for ranking only; the returned pixels remain an unmodified SAM mask.
    def object_unary(candidate: SamCandidate) -> float:
        union = np.logical_or(candidate.mask, pose_prior).sum()
        iou = float(np.logical_and(candidate.mask, pose_prior).sum() / union) if union else 0.0
        return candidate.unary + iou

    return max(candidates, key=object_unary)


def cylinder_depth_map(
    object_to_camera: np.ndarray,
    radius_m: float,
    height_m: float,
    focal_length_px: float,
    principal_point: Sequence[float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Analytic ray/capped-cylinder intersection; finite values form full CAD projection."""
    height, width = image_shape
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = map(float, principal_point)
    rays_camera = np.stack(
        [(xx - cx) / focal_length_px, (yy - cy) / focal_length_px, np.ones_like(xx)],
        axis=-1,
    ).reshape(-1, 3)
    transform = np.asarray(object_to_camera, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    origin_object = -rotation.T @ translation
    directions_object = rays_camera @ rotation
    ox, oy, oz = origin_object
    dx, dy, dz = directions_object.T
    best = np.full(len(dx), np.inf, dtype=np.float64)

    aa = dx * dx + dz * dz
    bb = 2.0 * (ox * dx + oz * dz)
    cc = ox * ox + oz * oz - radius_m * radius_m
    discriminant = bb * bb - 4.0 * aa * cc
    valid_quadratic = (aa > 1e-12) & (discriminant >= 0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    for candidate in ((-bb - root) / (2.0 * aa + 1e-30), (-bb + root) / (2.0 * aa + 1e-30)):
        candidate_y = oy + candidate * dy
        valid = valid_quadratic & (candidate > 0) & (np.abs(candidate_y) <= height_m * 0.5)
        best = np.where(valid & (candidate < best), candidate, best)

    for cap_y in (-height_m * 0.5, height_m * 0.5):
        candidate = (cap_y - oy) / (dy + 1e-30)
        candidate_x = ox + candidate * dx
        candidate_z = oz + candidate * dz
        valid = (
            (np.abs(dy) > 1e-12)
            & (candidate > 0)
            & (candidate_x * candidate_x + candidate_z * candidate_z <= radius_m * radius_m)
        )
        best = np.where(valid & (candidate < best), candidate, best)
    return best.reshape(height, width).astype(np.float32)


def digit_tubes_and_depth(
    joints_2d: np.ndarray,
    joints_3d_camera: np.ndarray,
    image_shape: tuple[int, int],
    width_hand_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    tubes = np.zeros((height, width), dtype=np.uint8)
    depths = np.full((height, width), np.inf, dtype=np.float32)
    scale = hand_scale(joints_2d)
    line_width = max(1, int(round(scale * width_hand_scale)))
    for start, end in HAND_SEGMENTS:
        points = np.rint(joints_2d[[start, end]]).astype(np.int32)
        if not np.isfinite(joints_2d[[start, end]]).all():
            continue
        cv2.line(tubes, tuple(points[0]), tuple(points[1]), 1, line_width, cv2.LINE_AA)
        segment_depth = float(np.nanmean(joints_3d_camera[[start, end], 2]))
        if np.isfinite(segment_depth) and segment_depth > 0:
            temporary = np.zeros_like(tubes)
            cv2.line(temporary, tuple(points[0]), tuple(points[1]), 1, line_width, cv2.LINE_AA)
            depths[temporary.astype(bool)] = np.minimum(depths[temporary.astype(bool)], segment_depth)
    return tubes.astype(bool), depths


def flow_pair(gray_a: np.ndarray, gray_b: np.ndarray, long_side: int) -> tuple[np.ndarray, np.ndarray]:
    scale = min(1.0, long_side / max(gray_a.shape))
    size = (max(1, int(round(gray_a.shape[1] * scale))), max(1, int(round(gray_a.shape[0] * scale))))
    a = cv2.resize(gray_a, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray_b, size, interpolation=cv2.INTER_AREA)
    forward = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    backward = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return forward.astype(np.float32), backward.astype(np.float32)


def remap_for_comparison(value: np.ndarray, target_to_source_flow: np.ndarray) -> np.ndarray:
    if value.shape[:2] != target_to_source_flow.shape[:2]:
        value = cv2.resize(value.astype(np.float32), target_to_source_flow.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[0:value.shape[0], 0:value.shape[1]].astype(np.float32)
    map_x = xx + target_to_source_flow[..., 0]
    map_y = yy + target_to_source_flow[..., 1]
    return cv2.remap(
        value.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool, b_bool = np.asarray(a, bool), np.asarray(b, bool)
    union = int(np.logical_or(a_bool, b_bool).sum())
    return float(np.logical_and(a_bool, b_bool).sum() / union) if union else 1.0


def mutual_flow_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    flow_a_to_b: np.ndarray,
    flow_b_to_a: np.ndarray,
) -> float:
    size = flow_a_to_b.shape[1::-1]
    a = cv2.resize(mask_a.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    b = cv2.resize(mask_b.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    a_at_b = remap_for_comparison(a, flow_b_to_a) > 0.5
    b_at_a = remap_for_comparison(b, flow_a_to_b) > 0.5
    return min(binary_iou(a_at_b, b), binary_iou(b_at_a, a))


def viterbi_select(
    candidates_by_frame: Sequence[Sequence[SamCandidate]],
    forward_flows: Sequence[np.ndarray],
    backward_flows: Sequence[np.ndarray],
    config: ProducerConfig,
) -> list[int]:
    if not candidates_by_frame or any(not values for values in candidates_by_frame):
        raise ValueError("each frame must contain at least one current-frame candidate")
    if len(forward_flows) != len(candidates_by_frame) - 1 or len(backward_flows) != len(forward_flows):
        raise ValueError("flow/candidate sequence length mismatch")
    scores = [np.full(len(values), -np.inf, np.float64) for values in candidates_by_frame]
    back = [np.full(len(values), -1, np.int32) for values in candidates_by_frame]
    scores[0] = np.array([config.viterbi_unary_weight * item.unary for item in candidates_by_frame[0]])
    for frame_index in range(1, len(candidates_by_frame)):
        for current_index, current in enumerate(candidates_by_frame[frame_index]):
            best_score = -np.inf
            best_previous = -1
            for previous_index, previous in enumerate(candidates_by_frame[frame_index - 1]):
                transition = mutual_flow_iou(
                    previous.mask,
                    current.mask,
                    forward_flows[frame_index - 1],
                    backward_flows[frame_index - 1],
                )
                score = (
                    scores[frame_index - 1][previous_index]
                    + config.viterbi_transition_weight * transition
                    + config.viterbi_unary_weight * current.unary
                )
                if score > best_score:
                    best_score, best_previous = score, previous_index
            scores[frame_index][current_index] = best_score
            back[frame_index][current_index] = best_previous
    selected = [0] * len(candidates_by_frame)
    selected[-1] = int(np.argmax(scores[-1]))
    for frame_index in range(len(selected) - 1, 0, -1):
        selected[frame_index - 1] = int(back[frame_index][selected[frame_index]])
    return selected


def signed_distance(mask: np.ndarray, object_scale: float) -> np.ndarray:
    value = np.asarray(mask, np.uint8)
    inside = cv2.distanceTransform(value, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - value, cv2.DIST_L2, 3)
    return (inside - outside) / max(float(object_scale), 1.0)


def align_scalar_neighbor_to_current(
    value: np.ndarray,
    neighbor_index: int,
    current_index: int,
    forward_flows: Sequence[np.ndarray],
    backward_flows: Sequence[np.ndarray],
) -> np.ndarray:
    aligned = value.astype(np.float32)
    if neighbor_index < current_index:
        for step in range(neighbor_index, current_index):
            aligned = remap_for_comparison(aligned, backward_flows[step])
    elif neighbor_index > current_index:
        for step in range(neighbor_index - 1, current_index - 1, -1):
            aligned = remap_for_comparison(aligned, forward_flows[step])
    return aligned


def stable_holes_symmetric_sdf(
    raw_holes: Sequence[np.ndarray],
    pose_priors: Sequence[np.ndarray],
    digit_tubes: Sequence[np.ndarray],
    arm_candidates: Sequence[np.ndarray],
    forward_flows: Sequence[np.ndarray],
    backward_flows: Sequence[np.ndarray],
    radius_frames: int,
) -> list[np.ndarray]:
    flow_shape = forward_flows[0].shape[:2] if forward_flows else raw_holes[0].shape
    low_raw = [cv2.resize(value.astype(np.uint8), flow_shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool) for value in raw_holes]
    low_pose = [cv2.resize(value.astype(np.uint8), flow_shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool) for value in pose_priors]
    sdfs = [signed_distance(value, math.sqrt(max(int(pose.sum()), 1))) for value, pose in zip(low_raw, low_pose, strict=True)]
    output: list[np.ndarray] = []
    for current in range(len(raw_holes)):
        aligned = []
        for neighbor in range(max(0, current - radius_frames), min(len(raw_holes), current + radius_frames + 1)):
            aligned.append(
                align_scalar_neighbor_to_current(
                    sdfs[neighbor], neighbor, current, forward_flows, backward_flows
                )
            )
        median = np.median(np.stack(aligned), axis=0) > 0
        full = cv2.resize(median.astype(np.uint8), raw_holes[current].shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        output.append(full & digit_tubes[current] & arm_candidates[current] & pose_priors[current])
    return output


def stage3_ownership(
    arm_candidate: np.ndarray,
    object_observation: np.ndarray,
    pose_prior: np.ndarray,
    stable_hole: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_wins = stable_hole & pose_prior
    effective_object_protect = (object_observation | pose_prior) & ~hand_wins
    final_arm = arm_candidate & ~effective_object_protect
    if np.logical_and(final_arm, effective_object_protect).any():
        raise AssertionError("stage3 ownership must be mutually exclusive")
    return final_arm, effective_object_protect, hand_wins


def noncontact_changed_pixels(before: np.ndarray, after: np.ndarray, contact_band: np.ndarray) -> int:
    return int(np.logical_and(np.logical_xor(before, after), ~contact_band).sum())


def refine_contact_band(
    stage3_arm: np.ndarray,
    arm_candidate: np.ndarray,
    object_observation: np.ndarray,
    pose_prior: np.ndarray,
    digit_tube: np.ndarray,
    stable_hole: np.ndarray,
    hand_depth: np.ndarray,
    cad_depth: np.ndarray,
    object_radius_m: float,
    config: ProducerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    object_scale = math.sqrt(max(int(pose_prior.sum()), 1))
    band_width = max(1, int(round(config.contact_band_object_scale_ratio * object_scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1))
    anatomy_near_object = cv2.dilate(digit_tube.astype(np.uint8), kernel).astype(bool)
    contact_band = pose_prior & anatomy_near_object
    refined = stage3_arm.copy()
    depth_valid = np.isfinite(hand_depth) & np.isfinite(cad_depth)
    normalized_front_margin = np.zeros(stage3_arm.shape, np.float32)
    normalized_front_margin[depth_valid] = (
        cad_depth[depth_valid] - hand_depth[depth_valid]
    ) / max(float(object_radius_m), 1e-9)
    # [-margin,+margin] is the deadband: retain stage3 ownership exactly.
    promote = (
        contact_band
        & arm_candidate
        & digit_tube
        & depth_valid
        & (normalized_front_margin > config.contact_depth_margin_object_radius)
        & (~object_observation | stable_hole)
    )
    demote = (
        contact_band
        & object_observation
        & depth_valid
        & (normalized_front_margin < -config.contact_depth_margin_object_radius)
    )
    refined[promote] = True
    refined[demote] = False
    if noncontact_changed_pixels(stage3_arm, refined, contact_band) != 0:
        raise AssertionError("stage4 changed a non-contact pixel")
    return refined, contact_band


def adjacent_unregistered_iou(masks: Sequence[np.ndarray]) -> list[float]:
    return [binary_iou(left, right) for left, right in zip(masks[:-1], masks[1:], strict=True)]


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def freeze_inputs(args: argparse.Namespace) -> dict[str, Any]:
    run_root = validate_candidate_root(args.output_root, require_new=True)
    source_manifest_path = Path(args.source_manifest).resolve()
    if not source_manifest_path.is_file() or source_manifest_path.is_symlink():
        raise ValueError("source manifest is missing, symlinked, or not an ordinary file")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    session = next(
        (value for value in source_manifest["sessions"] if value["session_id"] == args.session_id),
        None,
    )
    if session is None:
        raise ValueError("session absent from verified source manifest")
    stop = args.frame_start + args.frame_count
    if args.frame_start < 0 or stop > session["frame_count"]:
        raise ValueError("requested window outside source manifest")
    frames = session["frames"][args.frame_start:stop]
    if [value["frame_index"] for value in frames] != list(range(args.frame_start, stop)):
        raise ValueError("source frames are not contiguous")
    for value in frames:
        raw_path = Path(value["image"]["path"])
        if not raw_path.is_file() or raw_path.is_symlink():
            raise ValueError(f"RAW is missing or non-ordinary: {raw_path}")
        if sha256_file(raw_path) != value["image"]["sha256"]:
            raise ValueError(f"RAW digest mismatch: {raw_path}")
    refs = {}
    for name in (
        "hawor_projection", "object6d", "model_asset_pin", "model_weight",
        "fallback_evidence",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"input is missing or non-ordinary: {path}")
        refs[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for name, path in (
        ("pipeline_contract", PROJECT_ROOT / "contracts" / "pipeline_contract_v1.yaml"),
        ("project_profile", PROJECT_ROOT / "contracts" / "project_profile_v1.yaml"),
    ):
        path = path.resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"registered model contract is missing or non-ordinary: {path}")
        refs[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    implementation_root = Path(args.model_implementation).resolve()
    config_path = (implementation_root / "sam2" / args.model_config).resolve()
    if implementation_root not in config_path.parents:
        raise ValueError("model config must be contained in the pinned implementation")
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError("model config is missing, symlinked, or not an ordinary file")
    registered_model = validate_registered_model_asset(refs, implementation_root, config_path)
    payload = {
        "schema_version": "a2-input-freeze-v1",
        "status": "frozen_inputs_for_t1_candidate_only",
        "auth_tier": "T1",
        "producer_id": PRODUCER_ID,
        "component_id": COMPONENT_ID,
        "not_a_v9_heuristic_patch": NOT_A_V9_HEURISTIC_PATCH,
        "lineage": LINEAGE,
        "producer_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "labels_read": False,
        "held_out_session_149_read": False,
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
            "session_digest": source_manifest["session_digests"][args.session_id],
        },
        "session_id": args.session_id,
        "frame_start": args.frame_start,
        "frame_count": args.frame_count,
        "frame_indices": [value["frame_index"] for value in frames],
        "raw_frames": [
            {"frame_index": value["frame_index"], "path": value["image"]["path"], "sha256": value["image"]["sha256"]}
            for value in frames
        ],
        "inputs": refs,
        "model": {
            "identity": "SAM2_1_HIERA_LARGE",
            "implementation_ref": str(implementation_root),
            "implementation_inventory_entries": registered_model["implementation_inventory_entries"],
            "implementation_regular_bytes": registered_model["implementation_regular_bytes"],
            "implementation_inventory_sha256": registered_model["implementation_inventory_sha256"],
            "config_path": args.model_config,
            "config_ref": {
                "path": str(config_path),
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            "registered_pin": registered_model,
        },
        "producer_config": asdict(ProducerConfig()),
        "method_ids": [METHOD_STAGE1, METHOD_STAGE2, METHOD_STAGE3, METHOD_STAGE4],
        "pixel_semantics": PIXEL_SEMANTICS,
        "pixel_semantics_sha256": PIXEL_SEMANTICS_SHA,
        "candidate_policy": {
            "status": "candidate_requires_human_review",
            "next_bucket_blocked": True,
            "advancement_authorized": False,
            "processed_write_allowed": False,
        },
    }
    output = run_root / "INPUT_FREEZE.json"
    atomic_json(output, payload)
    return {"path": str(output), "sha256": sha256_file(output), "frame_count": len(frames)}


def load_sam_predictor(freeze: dict[str, Any], device: str) -> tuple[Any, Any]:
    sys.dont_write_bytecode = True
    implementation = freeze["model"]["implementation_ref"]
    if implementation not in sys.path:
        sys.path.insert(0, implementation)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = freeze["inputs"]["model_weight"]["path"]
    model = build_sam2(
        freeze["model"]["config_path"], checkpoint, device=device, mode="eval"
    )
    return SAM2ImagePredictor(model), torch


def verify_registered_model_inventory(freeze: dict[str, Any]) -> None:
    implementation = Path(freeze["model"]["implementation_ref"])
    inventory = implementation_inventory(implementation)
    if inventory != (
        freeze["model"]["implementation_inventory_entries"],
        freeze["model"]["implementation_regular_bytes"],
        freeze["model"]["implementation_inventory_sha256"],
    ):
        raise ValueError("frozen SAM implementation inventory drift")
    config_path = Path(freeze["model"]["config_ref"]["path"])
    current_registration = validate_registered_model_asset(
        freeze["inputs"], implementation, config_path
    )
    if current_registration != freeze["model"]["registered_pin"]:
        raise ValueError("frozen SAM registered pin snapshot drift")


def verify_frozen_inputs(freeze: dict[str, Any]) -> None:
    """Re-verify every frozen byte identity immediately before model loading."""
    if freeze.get("candidate_policy") != {
        "status": "candidate_requires_human_review",
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "processed_write_allowed": False,
    }:
        raise ValueError("T1 candidate policy drift")
    source_ref = freeze["source_manifest"]
    source_path = Path(source_ref["path"])
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError("frozen source manifest is missing or non-ordinary")
    if sha256_file(source_path) != source_ref["sha256"]:
        raise ValueError("frozen source manifest digest drift")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["session_digests"].get(freeze["session_id"]) != source_ref["session_digest"]:
        raise ValueError("frozen source session digest drift")
    source_session = next(
        (value for value in source["sessions"] if value["session_id"] == freeze["session_id"]),
        None,
    )
    if source_session is None:
        raise ValueError("frozen session absent from source manifest")
    source_frames = {value["frame_index"]: value["image"] for value in source_session["frames"]}
    for raw_ref in freeze["raw_frames"]:
        source_image = source_frames.get(raw_ref["frame_index"])
        if source_image is None or any(
            source_image.get(key) != raw_ref[key] for key in ("path", "sha256")
        ):
            raise ValueError("frozen RAW reference differs from source manifest")
        raw_path = Path(raw_ref["path"])
        if not raw_path.is_file() or raw_path.is_symlink():
            raise ValueError("frozen RAW is missing or non-ordinary")
        if sha256_file(raw_path) != raw_ref["sha256"]:
            raise ValueError("frozen RAW digest drift")
    for name, ref in freeze["inputs"].items():
        path = Path(ref["path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"frozen {name} is missing or non-ordinary")
        if path.stat().st_size != ref["bytes"] or sha256_file(path) != ref["sha256"]:
            raise ValueError(f"frozen {name} byte identity drift")
    config_ref = freeze["model"]["config_ref"]
    config_path = Path(config_ref["path"])
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError("frozen SAM config is missing or non-ordinary")
    if config_path.stat().st_size != config_ref["bytes"] or sha256_file(config_path) != config_ref["sha256"]:
        raise ValueError("frozen SAM config byte identity drift")
    verify_registered_model_inventory(freeze)
    producer_ref = freeze["producer_implementation"]
    producer_path = Path(producer_ref["path"])
    if not producer_path.is_file() or producer_path.is_symlink():
        raise ValueError("frozen producer implementation is missing or non-ordinary")
    if sha256_file(producer_path) != producer_ref["sha256"]:
        raise ValueError("frozen producer implementation digest drift")


def run_candidate(args: argparse.Namespace) -> dict[str, Any]:
    run_root = validate_candidate_root(args.output_root, require_new=False)
    children = {value.name for value in run_root.iterdir()}
    if children != {"INPUT_FREEZE.json"}:
        raise FileExistsError("candidate root is partial or already used; refusing any overwrite")
    freeze_path = run_root / "INPUT_FREEZE.json"
    if not freeze_path.is_file() or freeze_path.is_symlink():
        raise ValueError("INPUT_FREEZE.json missing or non-ordinary")
    freeze_sha = sha256_file(freeze_path)
    if freeze_sha != args.expected_input_freeze_sha:
        raise ValueError("input freeze digest mismatch")
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite candidate run")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["producer_id"] != PRODUCER_ID or freeze["labels_read"] is not False:
        raise ValueError("input freeze producer/label policy mismatch")
    verify_frozen_inputs(freeze)
    config = ProducerConfig(**freeze["producer_config"])
    hawor = np.load(freeze["inputs"]["hawor_projection"]["path"], allow_pickle=False)
    object6d = np.load(freeze["inputs"]["object6d"]["path"], allow_pickle=False)
    frame_indices = freeze["frame_indices"]
    if hawor["joints_2d"].shape[1] <= max(frame_indices) or len(object6d["valid"]) <= max(frame_indices):
        raise ValueError("sidecar does not cover frozen frame window")
    if not np.all(hawor["final_valid"][:, frame_indices]) or not np.all(object6d["valid"][frame_indices]):
        raise ValueError("invalid HaWoR/Object6D frame; fail closed before model execution")

    predictor, torch = load_sam_predictor(freeze, args.device)
    verify_registered_model_inventory(freeze)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frame_states: list[dict[str, Any]] = []
    grays: list[np.ndarray] = []
    arm_candidates_by_side: list[list[list[SamCandidate]]] = [[], []]
    top_unary_arms: list[np.ndarray] = []
    pose_priors: list[np.ndarray] = []
    cad_depths: list[np.ndarray] = []
    object_observations: list[np.ndarray] = []
    digit_tubes_all: list[np.ndarray] = []
    digit_depths_all: list[np.ndarray] = []
    degraded: list[dict[str, Any]] = []

    for local_index, (frame_index, raw_ref) in enumerate(zip(frame_indices, freeze["raw_frames"], strict=True)):
        raw_path = Path(raw_ref["path"])
        image_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"failed to read {raw_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        grays.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
        height, width = image_bgr.shape[:2]
        cad_depth = cylinder_depth_map(
            object6d["T_object_to_camera"][frame_index],
            float(object6d["cylinder_radius_m"]),
            float(object6d["cylinder_height_m"]),
            float(hawor["focal_length_px"]),
            hawor["principal_point"],
            (height, width),
        )
        pose_prior = np.isfinite(cad_depth)
        pose_priors.append(pose_prior)
        cad_depths.append(cad_depth)
        predictor.set_image(image_rgb)
        joints_frame = hawor["joints_2d"][:, frame_index]
        all_hand_points = safe_points(joints_frame.reshape(-1, 2), width, height)
        object_observation_candidate = choose_object_observation(predictor, pose_prior, all_hand_points)
        if object_observation_candidate is None:
            raise RuntimeError(f"frame {frame_index}: no current-frame object SAM candidate")
        object_observation = object_observation_candidate.mask
        object_observations.append(object_observation)
        ys, xs = np.nonzero(pose_prior)
        object_negatives = np.array([[float(np.median(xs)), float(np.median(ys))]], np.float32)

        combined_digit = np.zeros((height, width), bool)
        combined_digit_depth = np.full((height, width), np.inf, np.float32)
        top_side_masks: list[np.ndarray] = []
        per_side_metadata: list[dict[str, Any]] = []
        for side in range(2):
            joints = joints_frame[side]
            other = safe_points(joints_frame[1 - side][[0, 4, 8, 12, 16, 20]], width, height)
            part_candidates: dict[str, list[SamCandidate]] = {}
            part_observable: dict[str, bool] = {}
            for part in ("hand", "wrist_band", "wrist_device", "connected_sleeve"):
                part_observable[part] = bool(
                    len(safe_points(part_prompt_points(joints, part), width, height))
                )
                part_candidates[part] = predict_part_candidates(
                    predictor,
                    joints,
                    other,
                    object_negatives,
                    part,
                    (height, width),
                    config,
                )
                if part_observable[part] and not part_candidates[part]:
                    raise RuntimeError(f"frame {frame_index} side {side}: no {part} current-frame candidate")
            sequence_candidates = assemble_current_frame_arm_candidates(part_candidates)
            arm_candidates_by_side[side].append(sequence_candidates)
            top_side_masks.append(sequence_candidates[0].mask)
            tubes, tube_depth = digit_tubes_and_depth(
                joints,
                hawor["joints_3d_camera"][side, frame_index],
                (height, width),
                config.digit_tube_hand_scale,
            )
            combined_digit |= tubes
            combined_digit_depth = np.minimum(combined_digit_depth, tube_depth)
            per_side_metadata.append(
                {
                    "side_index": side,
                    "candidate_count": len(sequence_candidates),
                    "part_observable": part_observable,
                    "part_candidate_counts": {
                        key: len(value) for key, value in part_candidates.items()
                    },
                    "top_unary": sequence_candidates[0].unary,
                    "hand_scale_px": hand_scale(joints),
                }
            )
        top_unary = top_side_masks[0] | top_side_masks[1]
        top_unary_arms.append(top_unary)
        digit_tubes_all.append(combined_digit)
        digit_depths_all.append(combined_digit_depth)
        frame_states.append(
            {
                "frame_index": frame_index,
                "source_path": str(raw_path),
                "source_sha256": raw_ref["sha256"],
                "side_candidates": per_side_metadata,
                "object_sam_score": object_observation_candidate.sam_score,
                "pose_prior_pixels": int(pose_prior.sum()),
            }
        )

    forward_flows: list[np.ndarray] = []
    backward_flows: list[np.ndarray] = []
    for left, right in zip(grays[:-1], grays[1:], strict=True):
        forward, backward = flow_pair(left, right, config.flow_analysis_long_side)
        forward_flows.append(forward)
        backward_flows.append(backward)
    selected_by_side = [
        viterbi_select(values, forward_flows, backward_flows, config)
        for values in arm_candidates_by_side
    ]
    arm_candidates: list[np.ndarray] = []
    for local_index in range(len(frame_indices)):
        selected = [
            arm_candidates_by_side[side][local_index][selected_by_side[side][local_index]].mask
            for side in range(2)
        ]
        # Union only current-frame SAM candidates; no flow/geometry-generated pixels.
        arm_candidates.append(selected[0] | selected[1])
        frame_states[local_index]["viterbi_selected_indices"] = [
            selected_by_side[0][local_index], selected_by_side[1][local_index]
        ]

    raw_holes = [
        pose & ~observation & digits & arm
        for pose, observation, digits, arm in zip(
            pose_priors, object_observations, digit_tubes_all, arm_candidates, strict=True
        )
    ]
    stable_holes = stable_holes_symmetric_sdf(
        raw_holes,
        pose_priors,
        digit_tubes_all,
        arm_candidates,
        forward_flows,
        backward_flows,
        config.symmetric_sdf_radius_frames,
    )
    stage3_masks: list[np.ndarray] = []
    object_protects: list[np.ndarray] = []
    final_masks: list[np.ndarray] = []
    contact_bands: list[np.ndarray] = []
    hand_wins_values: list[np.ndarray] = []
    for index in range(len(frame_indices)):
        stage3, protect, hand_wins = stage3_ownership(
            arm_candidates[index], object_observations[index], pose_priors[index], stable_holes[index]
        )
        refined, contact_band = refine_contact_band(
            stage3,
            arm_candidates[index],
            object_observations[index],
            pose_priors[index],
            digit_tubes_all[index],
            stable_holes[index],
            digit_depths_all[index],
            cad_depths[index],
            float(object6d["cylinder_radius_m"]),
            config,
        )
        stage3_masks.append(stage3)
        object_protects.append(protect & ~refined)
        hand_wins_values.append(hand_wins)
        final_masks.append(refined)
        contact_bands.append(contact_band)

    # Imports and all candidate computation must leave the registered model tree
    # bit-identical; fail before writing any candidate artifact if it drifted.
    verify_registered_model_inventory(freeze)
    frames_root = run_root / "frames"
    os.mkdir(frames_root)
    frame_records: list[dict[str, Any]] = []
    for local_index, frame_index in enumerate(frame_indices):
        frame_root = run_root / "frames" / f"{frame_index:05d}"
        os.mkdir(frame_root)
        values = {
            "stage1_top_unary_arm": top_unary_arms[local_index],
            "stage2_viterbi_arm": arm_candidates[local_index],
            "object_observation": object_observations[local_index],
            "pose_prior": pose_priors[local_index],
            "raw_hole": raw_holes[local_index],
            "stable_hole": stable_holes[local_index],
            "stage3_arm": stage3_masks[local_index],
            "effective_object_protect": object_protects[local_index],
            "contact_band": contact_bands[local_index],
            "final_arm": final_masks[local_index],
        }
        refs = {}
        for name, value in values.items():
            path = frame_root / f"{name}.png"
            write_mask(path, value)
            refs[name] = artifact_ref(path, run_root)
        candidates_path = frame_root / "current_frame_sam_candidates.npz"
        atomic_npz(
            candidates_path,
            side_0=np.stack([item.mask for item in arm_candidates_by_side[0][local_index]]),
            side_1=np.stack([item.mask for item in arm_candidates_by_side[1][local_index]]),
        )
        refs["current_frame_sam_candidates"] = artifact_ref(candidates_path, run_root)
        record = copy.deepcopy(frame_states[local_index])
        record["artifacts"] = refs
        record["metrics"] = {
            "stage1_pixels": int(top_unary_arms[local_index].sum()),
            "stage2_pixels": int(arm_candidates[local_index].sum()),
            "raw_hole_pixels": int(raw_holes[local_index].sum()),
            "stable_hole_pixels": int(stable_holes[local_index].sum()),
            "contact_band_pixels": int(contact_bands[local_index].sum()),
            "stage4_noncontact_changed_pixels": noncontact_changed_pixels(
                stage3_masks[local_index], final_masks[local_index], contact_bands[local_index]
            ),
            "protected_object_overlap_pixels": int(
                np.logical_and(final_masks[local_index], object_protects[local_index]).sum()
            ),
            "final_pixels": int(final_masks[local_index].sum()),
        }
        frame_records.append(record)

    adjacent = adjacent_unregistered_iou(final_masks)
    areas = np.array([mask.mean() for mask in final_masks], np.float64)
    nonempty_count = sum(bool(mask.any()) for mask in final_masks)
    elapsed = time.perf_counter() - started
    forensic_metrics = {
        "interpretation": "candidate self-check only; never a PASS or advancement decision",
        "frame_count": len(frame_indices),
        "nonempty_frame_count": nonempty_count,
        "adjacent_unregistered_iou_mean": float(np.mean(adjacent)) if adjacent else None,
        "adjacent_unregistered_iou_min": float(np.min(adjacent)) if adjacent else None,
        "mask_area_ratio_mean": float(np.mean(areas)),
        "mask_area_ratio_min": float(np.min(areas)),
        "mask_area_ratio_p95": float(np.percentile(areas, 95)),
        "mask_area_ratio_max": float(np.max(areas)),
        "protected_object_overlap_pixels_max": max(
            value["metrics"]["protected_object_overlap_pixels"] for value in frame_records
        ),
        "stage4_noncontact_changed_pixels_total": sum(
            value["metrics"]["stage4_noncontact_changed_pixels"] for value in frame_records
        ),
        "raw_hole_nonempty_frames": sum(bool(value.any()) for value in raw_holes),
        "stable_hole_nonempty_frames": sum(bool(value.any()) for value in stable_holes),
        "elapsed_seconds": elapsed,
        "seconds_per_frame": elapsed / len(frame_indices),
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
        ),
    }
    manifest = {
        "schema_version": "mask-evidence-v1",
        "status": "candidate_requires_human_review",
        "artifact_state": "G2_CALIBRATION_CANDIDATE",
        "auth_tier": "T1",
        "component_id": COMPONENT_ID,
        "producer_id": PRODUCER_ID,
        "not_a_v9_heuristic_patch": NOT_A_V9_HEURISTIC_PATCH,
        "lineage": LINEAGE,
        "new_producer_identity": {
            "producer_version": PRODUCER_ID,
            "pixel_semantics_sha256": PIXEL_SEMANTICS_SHA,
            "implementation_path": str(Path(__file__).resolve()),
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
        },
        "fallback": {
            "used": True,
            "reason": "SAM3.1 official gated asset unavailable: A0b recorded HTTP 401 and no load smoke",
            "evidence_ref": {
                "path": freeze["inputs"]["fallback_evidence"]["path"],
                "sha256": freeze["inputs"]["fallback_evidence"]["sha256"],
            },
            "segments_1_2": "SAM2.1 current-frame candidates plus full-sequence Viterbi",
        },
        "input_freeze_ref": {"path": "INPUT_FREEZE.json", "sha256": freeze_sha},
        "session_id": freeze["session_id"],
        "frame_start": freeze["frame_start"],
        "frame_count": freeze["frame_count"],
        "frame_indices": frame_indices,
        "labels_read": False,
        "same_or_cross_blind_labels_read": False,
        "held_out_session_149_read": False,
        "method_ids": freeze["method_ids"],
        "pixel_semantics": PIXEL_SEMANTICS,
        "producer_config": freeze["producer_config"],
        "flow_policy": {
            "role": "transition_and_symmetric_contact_evidence_alignment_only",
            "output_pixel_warp_allowed": False,
            "viterbi_output_is_current_frame_sam_candidate": True,
        },
        "forbidden_mechanisms": {
            "ray_to_border": False,
            "pca_hard_crop": False,
            "lab_or_color_classifier": False,
            "prior_generated_output_pixels": False,
            "session_or_frame_branch": False,
        },
        "forensic_metrics": forensic_metrics,
        "frames": frame_records,
        "degraded": bool(degraded),
        "degraded_reasons": degraded,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumable": False,
        "processed_write_allowed": False,
        "candidate_decision": "AWAITING_HUMAN_AND_T2_GOVERNANCE_REVIEW_NO_AUTOMATIC_DECISION",
    }
    atomic_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "forensic_metrics": forensic_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze-inputs", "run"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest", default="contracts/manifests/verified_source_manifest_v1.json")
    parser.add_argument("--session-id")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--hawor-projection")
    parser.add_argument("--object6d")
    parser.add_argument("--model-asset-pin", default="assets/models/sam2_1_hiera_large/ASSET_PIN.json")
    parser.add_argument("--model-weight", default="assets/models/sam2_1_hiera_large/sam2.1_hiera_large.pt")
    parser.add_argument("--model-implementation", default="assets/models/sam2_1_hiera_large/implementation")
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--fallback-evidence", default="archive/audits/TASK_A0b_REPORT.md")
    parser.add_argument("--expected-input-freeze-sha")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "freeze-inputs":
        required = (args.session_id, args.frame_start, args.frame_count, args.hawor_projection, args.object6d)
        if any(value is None for value in required):
            raise SystemExit("freeze-inputs requires session/window/HaWoR/Object6D arguments")
        result = freeze_inputs(args)
    else:
        if not args.expected_input_freeze_sha:
            raise SystemExit("run requires --expected-input-freeze-sha")
        result = run_candidate(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
