"""Generic session profiling and draft session-context generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from .raw_source import (
    RawSourceError,
    StrictRawResolver,
    read_local_regular_readonly,
    sha256_bytes,
)
from .stress_frames import StressSelection, select_stress_frames


@dataclass(frozen=True)
class ProfilerConfig:
    """Task-card parameters; ratios are dimensionless and times are seconds."""

    analysis_scale: float
    forearm_width_ratio: float
    contact_radius_ratio: float
    contact_projection_ratio: float
    count_min: int
    count_max: int
    target_count: int
    min_gap_seconds: float
    stable_contact_seconds: float
    pre_contact_seconds: float
    contact_on_threshold: float

    def validate(self) -> None:
        for name, value in {
            "analysis_scale": self.analysis_scale,
            "forearm_width_ratio": self.forearm_width_ratio,
            "contact_radius_ratio": self.contact_radius_ratio,
            "contact_projection_ratio": self.contact_projection_ratio,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.analysis_scale > 1:
            raise ValueError("analysis_scale is a source-resolution ratio and must be <= 1")


@dataclass(frozen=True)
class ContextBuildResult:
    context: Mapping[str, Any]
    selection: StressSelection
    frame_features: Mapping[str, tuple[float, ...]]


def _digest_ref(path: str, digest: str, *, producer: str | None = None) -> dict[str, str]:
    result = {"path": path, "sha256": digest}
    if producer is not None:
        result["producer"] = producer
    return result


def _measured(
    value: float,
    unit: str,
    method: str,
    sample_count: int,
    evidence_refs: list[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "measurement_status": "MEASURED",
        "value": float(value),
        "unit": unit,
        "method": method,
        "sample_count": int(sample_count),
        "evidence_refs": evidence_refs,
    }


def _missing(unit: str | None, method: str) -> dict[str, Any]:
    return {
        "measurement_status": "MISSING",
        "value": None,
        "unit": unit,
        "method": method,
        "sample_count": None,
        "evidence_refs": [],
    }


def _robust_median(values: list[float]) -> tuple[float, int] | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return None
    return float(np.median(finite)), int(finite.size)


def _load_json_evidence(path: str, expected_sha256: str) -> tuple[Mapping[str, Any], bytes]:
    payload = read_local_regular_readonly(path)
    observed = sha256_bytes(payload)
    if observed != expected_sha256:
        raise RawSourceError(
            f"evidence digest mismatch for {path}: expected {expected_sha256}, observed {observed}"
        )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RawSourceError(f"evidence JSON root must be an object: {path}")
    return value, payload


def _load_npz_evidence(path: str, expected_sha256: str) -> Mapping[str, np.ndarray]:
    payload = read_local_regular_readonly(path)
    observed = sha256_bytes(payload)
    if observed != expected_sha256:
        raise RawSourceError(
            f"evidence digest mismatch for {path}: expected {expected_sha256}, observed {observed}"
        )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _hand_points(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, np.ndarray]]:
    entities = metadata.get("entities")
    if not isinstance(entities, dict) or not isinstance(entities.get("hands"), dict):
        raise RawSourceError("frame metadata has no entities.hands object")
    result: dict[str, Mapping[str, np.ndarray]] = {}
    for side, hand in entities["hands"].items():
        if not isinstance(side, str) or not isinstance(hand, dict):
            continue
        names = hand.get("joint_names")
        points_2d = np.asarray(hand.get("keypoints_2d"), dtype=np.float64)
        points_3d = np.asarray(hand.get("keypoints_3d_camera"), dtype=np.float64)
        if (
            not isinstance(names, list)
            or points_2d.shape != (len(names), 2)
            or points_3d.shape != (len(names), 3)
        ):
            continue
        result[side] = {
            "names": np.asarray(names, dtype=str),
            "points_2d": points_2d,
            "points_3d": points_3d,
        }
    return result


def _point_by_name(hand: Mapping[str, np.ndarray], name: str) -> np.ndarray | None:
    matches = np.flatnonzero(hand["names"] == name)
    if len(matches) != 1:
        return None
    point = hand["points_2d"][int(matches[0])]
    return point if np.isfinite(point).all() else None


def _palm_width(hand: Mapping[str, np.ndarray]) -> float | None:
    index = _point_by_name(hand, "index_proximal")
    pinky = _point_by_name(hand, "pinky_proximal")
    if index is None or pinky is None:
        return None
    value = float(np.linalg.norm(index - pinky))
    return value if value > 0 else None


def _ray_to_boundary(start: np.ndarray, direction: np.ndarray, width: int, height: int) -> np.ndarray | None:
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    direction = direction / norm
    candidates: list[float] = []
    for coordinate, delta, upper in zip(start, direction, (width - 1, height - 1)):
        if delta > 1e-9:
            candidates.append((upper - coordinate) / delta)
        elif delta < -1e-9:
            candidates.append((0 - coordinate) / delta)
    positive = [value for value in candidates if value >= 0]
    if not positive:
        return None
    return start + min(positive) * direction


def _arm_proxy_mask(
    hands: Mapping[str, Mapping[str, np.ndarray]],
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    forearm_width_ratio: float,
) -> tuple[np.ndarray, list[float], list[float], dict[str, bool]]:
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    scale = np.array([target_width / source_width, target_height / source_height], dtype=np.float64)
    hand_scales: list[float] = []
    wrist_widths: list[float] = []
    visibility: dict[str, bool] = {"left": False, "right": False}
    for side, hand in hands.items():
        points = hand["points_2d"]
        finite = np.isfinite(points).all(axis=1)
        in_bounds = (
            finite
            & (points[:, 0] >= 0)
            & (points[:, 0] < source_width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < source_height)
        )
        if side in visibility:
            visibility[side] = bool(np.mean(in_bounds) >= 0.8)
        valid = points[in_bounds]
        if len(valid) < 3:
            continue
        x_extent = float(np.ptp(valid[:, 0]))
        y_extent = float(np.ptp(valid[:, 1]))
        hand_scales.append(math.sqrt(max(x_extent * y_extent, 0.0)))
        hull = cv2.convexHull(np.round(valid * scale).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)

        palm_width = _palm_width(hand)
        wrist = _point_by_name(hand, "wrist")
        palm = _point_by_name(hand, "palm_center")
        if palm_width is None or wrist is None or palm is None:
            continue
        wrist_widths.append(palm_width)
        end = _ray_to_boundary(wrist, wrist - palm, source_width, source_height)
        if end is None:
            continue
        direction = end - wrist
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            continue
        perpendicular = np.array([-direction[1], direction[0]]) / direction_norm
        half_width = 0.5 * forearm_width_ratio * palm_width
        polygon = np.stack(
            [
                wrist + perpendicular * half_width,
                end + perpendicular * half_width,
                end - perpendicular * half_width,
                wrist - perpendicular * half_width,
            ]
        )
        polygon = np.round(polygon * scale).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask, hand_scales, wrist_widths, visibility


def _cylinder_local_points(radius: float, half_height: float, samples: int = 96) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    rings = []
    for y in (-half_height, 0.0, half_height):
        rings.append(
            np.stack(
                [radius * np.cos(angles), np.full_like(angles, y), radius * np.sin(angles)],
                axis=1,
            )
        )
    return np.concatenate(rings, axis=0)


def _object_projection_mask(
    transform_object_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    local_points: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, float | None]:
    points_camera = (
        transform_object_to_camera[:3, :3] @ local_points.T
    ).T + transform_object_to_camera[:3, 3]
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-6)
    if np.count_nonzero(valid) < 3:
        return np.zeros((target_height, target_width), dtype=np.uint8), None
    projected = (intrinsics @ points_camera[valid].T).T
    projected = projected[:, :2] / projected[:, 2:3]
    finite = np.isfinite(projected).all(axis=1)
    projected = projected[finite]
    if len(projected) < 3:
        return np.zeros((target_height, target_width), dtype=np.uint8), None
    x_extent = float(np.ptp(projected[:, 0]))
    y_extent = float(np.ptp(projected[:, 1]))
    object_scale = math.sqrt(max(x_extent * y_extent, 0.0))
    scale = np.array([target_width / source_width, target_height / source_height])
    hull = cv2.convexHull(np.round(projected * scale).astype(np.int32))
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask, object_scale


def _capped_cylinder_sdf(points_local: np.ndarray, radius: float, half_height: float) -> np.ndarray:
    radial = np.linalg.norm(points_local[:, (0, 2)], axis=1) - radius
    axial = np.abs(points_local[:, 1]) - half_height
    outside = np.linalg.norm(np.maximum(np.stack([radial, axial], axis=1), 0), axis=1)
    inside = np.minimum(np.maximum(radial, axial), 0)
    return outside + inside


def _contact_score(
    hands: Mapping[str, Mapping[str, np.ndarray]],
    transform_object_to_camera: np.ndarray,
    radius: float,
    half_height: float,
    contact_radius_ratio: float,
) -> float | None:
    points = [hand["points_3d"] for hand in hands.values()]
    if not points:
        return None
    points_camera = np.concatenate(points, axis=0)
    points_camera = points_camera[np.isfinite(points_camera).all(axis=1)]
    if not len(points_camera):
        return None
    inverse = np.linalg.inv(transform_object_to_camera)
    local = (inverse[:3, :3] @ points_camera.T).T + inverse[:3, 3]
    sdf = _capped_cylinder_sdf(local, radius, half_height)
    surface_distance = max(float(np.min(sdf)), 0.0)
    band = contact_radius_ratio * min(radius, half_height)
    return float(np.clip(1.0 - surface_distance / band, 0.0, 1.0))


def _projected_contact_score(
    hands: Mapping[str, Mapping[str, np.ndarray]],
    object_mask: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    object_scale: float | None,
    contact_projection_ratio: float,
) -> float | None:
    """Score 2D hand/object proximity in projected-object-scale units.

    This is intentionally only a second evidence channel: the caller fuses it
    with metric 3D SDF proximity.  It makes the phase selector robust to hand
    depth noise without introducing pixel thresholds or session-specific rules.
    """

    if object_scale is None or not math.isfinite(object_scale) or object_scale <= 0:
        return None
    target_height, target_width = object_mask.shape
    scale = np.array(
        [target_width / source_width, target_height / source_height],
        dtype=np.float64,
    )
    points = [hand["points_2d"] for hand in hands.values()]
    if not points:
        return None
    points_source = np.concatenate(points, axis=0)
    finite = np.isfinite(points_source).all(axis=1)
    in_bounds = (
        finite
        & (points_source[:, 0] >= 0)
        & (points_source[:, 0] < source_width)
        & (points_source[:, 1] >= 0)
        & (points_source[:, 1] < source_height)
    )
    if not np.any(in_bounds) or not np.any(object_mask):
        return None
    points_target = np.round(points_source[in_bounds] * scale).astype(np.int32)
    points_target[:, 0] = np.clip(points_target[:, 0], 0, target_width - 1)
    points_target[:, 1] = np.clip(points_target[:, 1], 0, target_height - 1)
    distance_to_object = cv2.distanceTransform(
        (object_mask == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    minimum_distance = float(
        np.min(distance_to_object[points_target[:, 1], points_target[:, 0]])
    )
    analysis_linear_scale = math.sqrt(float(scale[0] * scale[1]))
    band = contact_projection_ratio * object_scale * analysis_linear_scale
    return float(np.clip(1.0 - minimum_distance / band, 0.0, 1.0))


def build_session_context(
    resolver: StrictRawResolver,
    *,
    execution_mode: str,
    product_line: str,
    diagnostic_track: str,
    profile_path: str,
    expected_profile_sha256: str,
    task_card_path: str,
    expected_task_card_sha256: str,
    calibration_overlay_path: str | None,
    expected_calibration_overlay_sha256: str | None,
    config: ProfilerConfig,
    object6d_npz_path: str | None = None,
    object6d_npz_sha256: str | None = None,
    object_geometry_path: str | None = None,
    object_geometry_sha256: str | None = None,
) -> ContextBuildResult:
    """Profile a manifest-selected session without starting a visual producer."""

    config.validate()
    profile_payload = read_local_regular_readonly(profile_path)
    observed_profile_sha = sha256_bytes(profile_payload)
    if observed_profile_sha != expected_profile_sha256:
        raise RawSourceError(
            f"profile digest mismatch: expected {expected_profile_sha256}, observed {observed_profile_sha}"
        )
    profile = yaml.safe_load(profile_payload)
    if not isinstance(profile, dict):
        raise RawSourceError("project profile must be a YAML object")

    task_card_payload = read_local_regular_readonly(task_card_path)
    observed_task_card_sha = sha256_bytes(task_card_payload)
    if observed_task_card_sha != expected_task_card_sha256:
        raise RawSourceError(
            f"task-card digest mismatch: expected {expected_task_card_sha256}, "
            f"observed {observed_task_card_sha}"
        )
    task_card = yaml.safe_load(task_card_payload)
    if not isinstance(task_card, dict):
        raise RawSourceError("task card must be a YAML object")

    overlay_ref: Mapping[str, str] | None = None
    overlay: Mapping[str, Any] | None = None
    if execution_mode == "G2_CALIBRATION":
        if calibration_overlay_path is None or expected_calibration_overlay_sha256 is None:
            raise RawSourceError("G2 calibration requires a SHA-bound calibration overlay")
        overlay_payload = read_local_regular_readonly(calibration_overlay_path)
        observed_overlay_sha = sha256_bytes(overlay_payload)
        if observed_overlay_sha != expected_calibration_overlay_sha256:
            raise RawSourceError(
                f"calibration-overlay digest mismatch: expected {expected_calibration_overlay_sha256}, "
                f"observed {observed_overlay_sha}"
            )
        if Path(calibration_overlay_path).resolve() != Path(task_card_path).resolve():
            raise RawSourceError("G2 calibration overlay must be the explicit task-card section")
        overlay_document = yaml.safe_load(overlay_payload)
        overlay = (
            overlay_document.get("dimensionless_calibration_overlay")
            if isinstance(overlay_document, dict)
            else None
        )
        if not isinstance(overlay, dict) or not overlay:
            raise RawSourceError("task card has no dimensionless_calibration_overlay object")
        overlay_ref = _digest_ref(
            calibration_overlay_path,
            observed_overlay_sha,
            producer="g2_task_card_calibration_overlay",
        )
    elif execution_mode != "FORMAL_PRODUCTION":
        raise RawSourceError(f"unsupported execution_mode: {execution_mode}")
    elif calibration_overlay_path is not None or expected_calibration_overlay_sha256 is not None:
        raise RawSourceError("formal production must not consume a calibration overlay")

    session = resolver.session
    frame_count = int(session["frame_count"])
    source_width = int(session["width"])
    source_height = int(session["height"])
    fps = float(session["fps"])
    target_width = max(1, int(round(source_width * config.analysis_scale)))
    target_height = max(1, int(round(source_height * config.analysis_scale)))

    source_ref = _digest_ref(
        resolver.manifest_path,
        resolver.manifest_sha256,
        producer="source_resolver",
    )
    profile_ref = _digest_ref(profile_path, observed_profile_sha, producer="project_profile")
    task_card_ref = _digest_ref(
        task_card_path, observed_task_card_sha, producer="explicit_task_card"
    )
    evidence_refs: list[Mapping[str, str]] = [source_ref, profile_ref, task_card_ref]
    if overlay_ref is not None:
        evidence_refs.append(overlay_ref)

    object_data: Mapping[str, np.ndarray] | None = None
    radius: float | None = None
    half_height: float | None = None
    if any(
        value is not None
        for value in (
            object6d_npz_path,
            object6d_npz_sha256,
            object_geometry_path,
            object_geometry_sha256,
        )
    ):
        if not all(
            value is not None
            for value in (
                object6d_npz_path,
                object6d_npz_sha256,
                object_geometry_path,
                object_geometry_sha256,
            )
        ):
            raise RawSourceError("object evidence requires both paths and both expected digests")
        assert object6d_npz_path is not None and object6d_npz_sha256 is not None
        assert object_geometry_path is not None and object_geometry_sha256 is not None
        object_data = _load_npz_evidence(object6d_npz_path, object6d_npz_sha256)
        geometry, _ = _load_json_evidence(object_geometry_path, object_geometry_sha256)
        required_arrays = {"T_object_to_camera", "confidence", "valid"}
        if not required_arrays.issubset(object_data):
            raise RawSourceError("Object6D evidence is missing required arrays")
        if object_data["T_object_to_camera"].shape != (frame_count, 4, 4):
            raise RawSourceError("Object6D frame count or transform shape mismatch")
        if object_data["confidence"].shape != (frame_count,) or object_data["valid"].shape != (frame_count,):
            raise RawSourceError("Object6D confidence/valid shape mismatch")
        radius = float(geometry["radius_m"])
        half_height = 0.5 * float(geometry["height_m"])
        if radius <= 0 or half_height <= 0:
            raise RawSourceError("Object6D analytic geometry must have positive metric dimensions")
        if "cylinder_radius_m" in object_data and not np.isclose(float(object_data["cylinder_radius_m"]), radius):
            raise RawSourceError("Object6D NPZ and geometry radius disagree")
        if "cylinder_height_m" in object_data and not np.isclose(
            float(object_data["cylinder_height_m"]), 2 * half_height
        ):
            raise RawSourceError("Object6D NPZ and geometry height disagree")
        evidence_refs.extend(
            [
                _digest_ref(object6d_npz_path, object6d_npz_sha256, producer="object_geometry_producer"),
                _digest_ref(object_geometry_path, object_geometry_sha256, producer="object_geometry_producer"),
            ]
        )

    features = {
        "arm_area_ratio_proxy": np.full(frame_count, np.nan, dtype=np.float64),
        "contact_score": np.full(frame_count, np.nan, dtype=np.float64),
        "object_occlusion_proxy": np.full(frame_count, np.nan, dtype=np.float64),
        "motion": np.full(frame_count, np.nan, dtype=np.float64),
        "sharpness": np.full(frame_count, np.nan, dtype=np.float64),
        "object_confidence": np.full(frame_count, np.nan, dtype=np.float64),
        "object_valid": np.full(frame_count, np.nan, dtype=np.float64),
    }
    all_hand_scales: list[float] = []
    all_wrist_widths: list[float] = []
    all_forearm_widths: list[float] = []
    all_object_scales: list[float] = []
    visibility_counts = {"left": 0, "right": 0}
    previous_gray: np.ndarray | None = None
    local_object_points = (
        _cylinder_local_points(radius, half_height)
        if radius is not None and half_height is not None
        else None
    )

    for frame in resolver.frames:
        metadata = resolver.read_metadata_json(frame.frame_index)
        metadata_header = metadata.get("metadata")
        if not isinstance(metadata_header, dict):
            raise RawSourceError(f"missing metadata header at frame {frame.frame_index}")
        if (
            metadata_header.get("idx") != frame.frame_index
            or metadata_header.get("w") != source_width
            or metadata_header.get("h") != source_height
            or not np.isclose(float(metadata_header.get("fps", -1)), fps)
        ):
            raise RawSourceError(f"metadata/source-manifest mismatch at frame {frame.frame_index}")
        hands = _hand_points(metadata)
        arm_mask, hand_scales, wrist_widths, visible = _arm_proxy_mask(
            hands,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            forearm_width_ratio=config.forearm_width_ratio,
        )
        all_hand_scales.extend(hand_scales)
        all_wrist_widths.extend(wrist_widths)
        all_forearm_widths.extend(config.forearm_width_ratio * value for value in wrist_widths)
        for side in visibility_counts:
            visibility_counts[side] += int(visible.get(side, False))
        features["arm_area_ratio_proxy"][frame.frame_index] = float(np.mean(arm_mask))

        image_payload = resolver.read_frame_artifact(frame.frame_index, "image")
        gray = cv2.imdecode(np.frombuffer(image_payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (source_height, source_width):
            raise RawSourceError(f"image decode/dimension mismatch at frame {frame.frame_index}")
        gray = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if previous_gray is None:
            features["motion"][frame.frame_index] = 0.0
        else:
            features["motion"][frame.frame_index] = float(np.mean(np.abs(gray - previous_gray)))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        features["sharpness"][frame.frame_index] = float(
            np.mean(gx * gx + gy * gy) / (float(np.var(gray)) + 1e-6)
        )
        previous_gray = gray

        if object_data is not None and local_object_points is not None and radius is not None and half_height is not None:
            transform = np.asarray(object_data["T_object_to_camera"][frame.frame_index], dtype=np.float64)
            intrinsics = np.asarray(metadata_header.get("k"), dtype=np.float64)
            if intrinsics.shape != (3, 3):
                raise RawSourceError(f"invalid camera intrinsics at frame {frame.frame_index}")
            object_mask, object_scale = _object_projection_mask(
                transform,
                intrinsics,
                local_object_points,
                source_width=source_width,
                source_height=source_height,
                target_width=target_width,
                target_height=target_height,
            )
            if object_scale is not None:
                all_object_scales.append(object_scale)
            object_pixels = int(np.count_nonzero(object_mask))
            features["object_occlusion_proxy"][frame.frame_index] = (
                float(np.count_nonzero((arm_mask > 0) & (object_mask > 0)) / object_pixels)
                if object_pixels
                else np.nan
            )
            score = _contact_score(
                hands,
                transform,
                radius,
                half_height,
                config.contact_radius_ratio,
            )
            projected_score = _projected_contact_score(
                hands,
                object_mask,
                source_width=source_width,
                source_height=source_height,
                object_scale=object_scale,
                contact_projection_ratio=config.contact_projection_ratio,
            )
            evidence_scores = [
                value for value in (score, projected_score) if value is not None
            ]
            features["contact_score"][frame.frame_index] = (
                max(evidence_scores) if evidence_scores else np.nan
            )
            features["object_confidence"][frame.frame_index] = float(
                object_data["confidence"][frame.frame_index]
            )
            features["object_valid"][frame.frame_index] = float(
                bool(object_data["valid"][frame.frame_index])
            )

    selection = select_stress_frames(
        features,
        fps=fps,
        count_min=config.count_min,
        count_max=config.count_max,
        target_count=config.target_count,
        min_gap_seconds=config.min_gap_seconds,
        stable_contact_seconds=config.stable_contact_seconds,
        pre_contact_seconds=config.pre_contact_seconds,
        contact_on_threshold=config.contact_on_threshold,
    )

    hand_scale = _robust_median(all_hand_scales)
    wrist_width = _robust_median(all_wrist_widths)
    forearm_width = _robust_median(all_forearm_widths)
    object_scale = _robust_median(all_object_scales)
    arm_values = features["arm_area_ratio_proxy"]
    motion_values = features["motion"]
    sharpness_values = features["sharpness"]
    contact_values = features["contact_score"]
    valid_values = features["object_valid"]

    blockers: list[str] = []
    lifecycle_state = str(profile.get("lifecycle_state", "DRAFT"))
    if execution_mode == "G2_CALIBRATION":
        if resolver.manifest_status != "VERIFIED_G0_CALIBRATION_INPUT":
            blockers.append("SOURCE_MANIFEST_NOT_VERIFIED_G0_CALIBRATION_INPUT")
        if not (
            profile.get("document_status") == "DRAFT"
            and lifecycle_state == "DRAFT"
            and profile.get("calibration_execution_ready") is True
            and profile.get("calibration_execution_scope")
            == "G2_004_MASK_CLEAN_REVIEW_ONLY"
            and profile.get("formal_production_allowed") is False
        ):
            blockers.append("PROJECT_PROFILE_NOT_READY_FOR_G2_CALIBRATION")
        if not (
            task_card.get("status") == "AUTHORIZED_G2_CALIBRATION"
            and isinstance(task_card.get("task_id"), str)
            and task_card["task_id"].startswith("g2_004_mask_clean_candidate_v")
            and task_card.get("stage") == "G2_MASK_CLEAN_CALIBRATION"
            and task_card.get("session_id") == resolver.session_id
            and task_card.get("product_line") == product_line
        ):
            blockers.append("TASK_CARD_NOT_AUTHORIZED_FOR_REQUESTED_G2_SCOPE")
        frozen_source = task_card.get("frozen_inputs", {}).get("source_manifest", {})
        if not (
            isinstance(frozen_source.get("path"), str)
            and Path(frozen_source["path"]).resolve() == Path(resolver.manifest_path).resolve()
            and frozen_source.get("sha256") == resolver.manifest_sha256
            and frozen_source.get("session_digest") == resolver.session_identity_sha256
        ):
            blockers.append("TASK_CARD_SOURCE_IDENTITY_MISMATCH")
        allowed_overlay_keys = set(
            profile.get("calibration_overlay_policy", {}).get("allowed_keys", [])
        )
        if overlay is None or set(overlay) != allowed_overlay_keys:
            blockers.append("CALIBRATION_OVERLAY_KEYS_NOT_EXACTLY_ALLOWED")
        elif any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in overlay.values()
        ):
            blockers.append("CALIBRATION_OVERLAY_NOT_FINITE_NUMERIC_DIMENSIONLESS")
    else:
        if resolver.manifest_status != "FORMALLY_PROMOTED":
            blockers.append("SOURCE_MANIFEST_NOT_FORMALLY_PROMOTED")
        if not (
            profile.get("document_status") == "APPROVED"
            and lifecycle_state == "FROZEN"
            and profile.get("execution_ready") is True
            and profile.get("formal_production_allowed") is True
        ):
            blockers.append("PROJECT_PROFILE_NOT_READY_FOR_FORMAL_PRODUCTION")
    if object_data is None:
        blockers.append("OBJECT_EVIDENCE_NOT_PINNED")

    algorithm_path = str(Path(__file__).with_name("stress_frames.py"))
    algorithm_digest = sha256_bytes(read_local_regular_readonly(algorithm_path))
    measurements = {
        "hand_scale_px": (
            _measured(hand_scale[0], "px", "median_sqrt_2d_keypoint_bbox_area_per_visible_hand", hand_scale[1], [source_ref])
            if hand_scale
            else _missing("px", "no_valid_hand_keypoints")
        ),
        "wrist_width_px": (
            _measured(wrist_width[0], "px", "median_index_to_pinky_proximal_span_proxy", wrist_width[1], [source_ref])
            if wrist_width
            else _missing("px", "no_valid_proximal_keypoint_span")
        ),
        "forearm_width_px": (
            _measured(
                forearm_width[0],
                "px",
                "derived_forearm_prompt_width=forearm_width_ratio*wrist_width_proxy",
                forearm_width[1],
                [source_ref],
            )
            if forearm_width
            else _missing("px", "no_valid_wrist_width_proxy")
        ),
        "object_scale_px": (
            _measured(object_scale[0], "px", "median_sqrt_projected_analytic_object_bbox_area", object_scale[1], evidence_refs[-2:])
            if object_scale
            else _missing("px", "object_geometry_or_pose_unavailable")
        ),
        "arm_area_ratio": _measured(
            float(np.nanmedian(arm_values)),
            "ratio",
            "median_keypoint_hull_plus_boundary_connected_forearm_prompt_proxy",
            int(np.count_nonzero(np.isfinite(arm_values))),
            [source_ref],
        ),
        "contact_ratio": (
            _measured(
                float(np.mean(contact_values[np.isfinite(contact_values)] >= config.contact_on_threshold)),
                "ratio",
                "fraction_above_fused_dimensionless_3d_sdf_and_2d_projected_proximity_contact_score",
                int(np.count_nonzero(np.isfinite(contact_values))),
                evidence_refs,
            )
            if np.any(np.isfinite(contact_values))
            else _missing("ratio", "object_pose_or_hand_3d_unavailable")
        ),
        "motion": _measured(
            float(np.nanquantile(motion_values, 0.90)),
            "normalized_luma_difference",
            "p90_mean_absolute_luma_difference_at_source_relative_analysis_scale",
            int(np.count_nonzero(np.isfinite(motion_values))),
            [source_ref],
        ),
        "blur": _measured(
            float(np.nanquantile(1.0 / (1.0 + sharpness_values), 0.90)),
            "dimensionless_blur_score",
            "p90_inverse_variance_normalized_sobel_energy",
            int(np.count_nonzero(np.isfinite(sharpness_values))),
            [source_ref],
        ),
        "object6d_valid_ratio": (
            _measured(
                float(np.mean(valid_values[np.isfinite(valid_values)] >= 0.5)),
                "ratio",
                "mean_manifest_pinned_object6d_valid",
                int(np.count_nonzero(np.isfinite(valid_values))),
                evidence_refs[-2:],
            )
            if np.any(np.isfinite(valid_values))
            else _missing("ratio", "object6d_evidence_unavailable")
        ),
        "donor_coverage": _missing("ratio", "not_available_before_clean_producer"),
        "left_hand_visibility": _measured(
            visibility_counts["left"] / frame_count,
            "ratio",
            "fraction_frames_with_at_least_80_percent_finite_in_bounds_keypoints",
            frame_count,
            [source_ref],
        ),
        "right_hand_visibility": _measured(
            visibility_counts["right"] / frame_count,
            "ratio",
            "fraction_frames_with_at_least_80_percent_finite_in_bounds_keypoints",
            frame_count,
            [source_ref],
        ),
    }

    source_measurements = {
        "frame_count": _measured(frame_count, "frames", "source_manifest_declared_and_strictly_read", frame_count, [source_ref]),
        "width_px": _measured(source_width, "px", "source_manifest_and_decoded_frame_shape", frame_count, [source_ref]),
        "height_px": _measured(source_height, "px", "source_manifest_and_decoded_frame_shape", frame_count, [source_ref]),
        "fps": _measured(fps, "frames_per_second", "source_manifest_and_frame_metadata", frame_count, [source_ref]),
    }
    context: Mapping[str, Any] = {
        "schema_version": "session-context-v1",
        "contract_document_status": "DRAFT",
        "session_id": resolver.session_id,
        "product_line": product_line,
        "diagnostic_track": diagnostic_track,
        "execution_mode": execution_mode,
        "profile_lifecycle_state": lifecycle_state,
        "profile_ref": profile_ref,
        "task_card_ref": task_card_ref,
        "calibration_overlay_ref": overlay_ref,
        "formal_pass_claim_allowed": execution_mode == "FORMAL_PRODUCTION",
        "execution_allowed": not blockers,
        "execution_blockers": blockers,
        "source": {
            "authority_root": resolver.raw_root,
            "access": "READ_ONLY",
            "authority_readiness": "VERIFIED",
            "source_manifest_ref": resolver.manifest_path,
            "source_manifest_sha256": resolver.manifest_sha256,
            **source_measurements,
            "raw_identity": {
                "identity_method": "sha256_canonical_source_manifest_session_entry",
                "identity_digest": resolver.session_identity_sha256,
            },
        },
        "frame_policy": {
            "frame_order": "SOURCE_MANIFEST_ORDER",
            "timestamp_policy": str(session["timestamp_policy"]),
            "timestamp_unit": "nanoseconds_tracking_and_seconds_video_clock",
        },
        "measurements": measurements,
        "canary_selection": {
            "algorithm_ref": algorithm_path,
            "algorithm_sha256": algorithm_digest,
            "frames": list(selection.frames),
            "covered_strata": list(selection.covered_strata),
            "manual_review_required": product_line == "004_CONTACT_GOLD",
        },
        "evidence_refs": evidence_refs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    serializable_features = {
        name: tuple(float(value) for value in values)
        for name, values in features.items()
    }
    return ContextBuildResult(context=context, selection=selection, frame_features=serializable_features)
