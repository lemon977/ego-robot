"""Conservative, auditable repair envelopes for visual-only cleanup.

V2 is intentionally asymmetric: admitted SAM evidence defines the foreground
base, while geometry and appearance may only add small, validated repairs.
No weak prior in this module is allowed to create an unconstrained erase area.
The resulting removal mask and feather alpha remain visual-only artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np


class RemovalEnvelopeV2Error(ValueError):
    """Raised when inputs or consumers violate the V2 contract."""


FINGER_CHAINS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
SOURCE_BITS: dict[str, int] = {
    "admitted_sam_hand": 1,
    "selected_sam_forearm": 2,
    "admitted_sam_equipment": 4,
    "validated_mano_gap_repair": 8,
    "validated_attached_sleeve_repair": 16,
    "validated_cable_instance": 32,
}
ALLOWED_CONSUMERS = ("CLEAN_MASK_QA", "VISUAL_INPAINT")
FORBIDDEN_CONSUMERS = (
    "DEPTH",
    "OBJECT6D",
    "CONTACT",
    "ROBOT_GEOMETRY",
    "CONTROL_GROUND_TRUTH",
)


def enforce_consumer_firewall(artifact_class: str, consumer: str) -> None:
    """Reject use of inferred erase support as physical evidence."""

    if artifact_class not in {"removal_envelope_v2", "feather_alpha_v2"}:
        raise RemovalEnvelopeV2Error(f"unsupported artifact class: {artifact_class}")
    if consumer in FORBIDDEN_CONSUMERS:
        raise RemovalEnvelopeV2Error(
            f"{artifact_class} is visual-only and forbidden for {consumer}"
        )
    if consumer not in ALLOWED_CONSUMERS:
        raise RemovalEnvelopeV2Error(f"consumer is not allow-listed: {consumer}")


@dataclass(frozen=True)
class CableInstanceProfileV2:
    """Replaceable appearance profile used to rank cable instances."""

    profile_id: str
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    minimum_component_pixels: int = 12
    maximum_component_pixels: int = 6_000
    maximum_instances: int = 1
    minimum_total_score: float = 0.54
    minimum_anchor_score: float = 0.10
    anchor_distance_px: int = 55
    temporal_association_radius_px: int = 22
    output_dilation_px: int = 2
    candidate_close_radius_px: int = 1
    maximum_association_gap_frames: int = 1
    require_reverse_support_after_seed: bool = True
    anchor_weight: float = 0.28
    thinness_weight: float = 0.20
    path_weight: float = 0.16
    endpoint_weight: float = 0.16
    forward_backward_temporal_weight: float = 0.20


@dataclass(frozen=True)
class RemovalEnvelopeV2Config:
    cable_profile: CableInstanceProfileV2
    mano_bridge_radius_px: int = 3
    mano_near_sam_radius_px: int = 16
    mano_gap_close_radius_px: int = 10
    mano_endpoint_support_radius_px: int = 7
    mano_maximum_bridge_length_px: float = 34.0
    sleeve_hand_boundary_radius_px: int = 12
    sleeve_mano_radius_px: int = 13
    sleeve_minimum_component_pixels: int = 8
    sleeve_maximum_component_pixels: int = 2_000
    sleeve_temporal_radius_px: int = 12
    sleeve_minimum_comotion_score: float = 0.30
    sleeve_maximum_instances: int = 2
    sleeve_maximum_association_gap_frames: int = 1
    forearm_wrist_anchor_radius_px: int = 20
    forearm_minimum_semantic_overlap: float = 0.20
    forearm_minimum_component_pixels: int = 40
    forearm_maximum_image_fraction: float = 0.15
    forearm_maximum_components: int = 2
    forearm_minimum_elongation: float = 1.15
    forearm_temporal_radius_px: int = 20
    forearm_minimum_temporal_score: float = 0.15
    forearm_maximum_association_gap_frames: int = 1
    feather_radius_px: int = 5
    maximum_area_inflation: float = 1.50
    maximum_background_spill_ratio: float = 0.03
    maximum_temporal_area_derivative: float = 0.35
    maximum_repair_contribution_ratio: float = 0.25
    object_core_erosion_radius_px: int = 4
    object_hand_interaction_radius_px: int = 10
    maximum_protected_object_core_damage_ratio: float = 0.0


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(value, bool)
    if result.shape != shape:
        raise RemovalEnvelopeV2Error(
            f"{name} must have shape {shape}, got {result.shape}"
        )
    return result


def _kernel(radius: int) -> np.ndarray:
    size = max(1, int(radius) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, bool).copy()
    return cv2.dilate(np.asarray(mask, np.uint8), _kernel(radius)).astype(bool)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, bool).copy()
    return cv2.erode(np.asarray(mask, np.uint8), _kernel(radius)).astype(bool)


def _finite_hand(joints: np.ndarray, shape: tuple[int, int]) -> bool:
    if joints.shape != (21, 2) or not np.isfinite(joints).all():
        return False
    height, width = shape
    inside = (
        (joints[:, 0] >= 0.0)
        & (joints[:, 0] < width)
        & (joints[:, 1] >= 0.0)
        & (joints[:, 1] < height)
    )
    return int(inside.sum()) >= 16


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    y, x = np.nonzero(mask)
    if not x.size:
        return None
    return np.asarray([float(x.mean()), float(y.mean())], np.float64)


def _translate_mask(mask: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    matrix = np.asarray(
        [[1.0, 0.0, float(delta_xy[0])], [0.0, 1.0, float(delta_xy[1])]],
        np.float32,
    )
    height, width = mask.shape
    return cv2.warpAffine(
        mask.astype(np.uint8), matrix, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)


def _mano_centerline(
    joints_2d: np.ndarray,
    observed: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    line = np.zeros(shape, np.uint8)
    for side in range(2):
        joints = joints_2d[side]
        if not observed[side] or not _finite_hand(joints, shape):
            continue
        points = np.rint(joints).astype(np.int32)
        for chain in FINGER_CHAINS:
            for first, second in zip(chain[:-1], chain[1:]):
                cv2.line(
                    line, tuple(points[first]), tuple(points[second]), 255, 1,
                    lineType=cv2.LINE_8,
                )
    return line.astype(bool)


def _mano_gap_repair(
    sam_hand: np.ndarray,
    task_object: np.ndarray,
    joints_2d: np.ndarray,
    observed: np.ndarray,
    config: RemovalEnvelopeV2Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bridge only short SAM gaps whose two ends have SAM support."""

    shape = sam_hand.shape
    repair = np.zeros(shape, bool)
    closed = cv2.morphologyEx(
        sam_hand.astype(np.uint8), cv2.MORPH_CLOSE,
        _kernel(config.mano_gap_close_radius_px),
    ).astype(bool)
    local_gap = closed & ~sam_hand
    near_sam = _dilate(sam_hand, config.mano_near_sam_radius_px)
    supported_segments = 0
    rejected_segments = 0
    for side in range(2):
        joints = joints_2d[side]
        if not observed[side] or not _finite_hand(joints, shape):
            continue
        points = np.rint(joints).astype(np.int32)
        for chain in FINGER_CHAINS:
            for first, second in zip(chain[:-1], chain[1:]):
                first_xy = points[first]
                second_xy = points[second]
                length = float(np.linalg.norm(first_xy - second_xy))
                if length > config.mano_maximum_bridge_length_px:
                    rejected_segments += 1
                    continue
                endpoint_support: list[bool] = []
                for xy in (first_xy, second_xy):
                    endpoint = np.zeros(shape, np.uint8)
                    cv2.circle(
                        endpoint, tuple(xy), config.mano_endpoint_support_radius_px,
                        1, -1,
                    )
                    endpoint_support.append(bool(np.any(endpoint.astype(bool) & sam_hand)))
                if not all(endpoint_support):
                    rejected_segments += 1
                    continue
                bridge = np.zeros(shape, np.uint8)
                cv2.line(
                    bridge, tuple(first_xy), tuple(second_xy), 1,
                    max(1, config.mano_bridge_radius_px * 2), lineType=cv2.LINE_8,
                )
                local = bridge.astype(bool) & local_gap & near_sam & ~task_object
                if local.any():
                    repair |= local
                    supported_segments += 1
    return repair, {
        "state": "VALIDATED_LOCAL_REPAIR" if repair.any() else "UNKNOWN_NO_VALID_GAP",
        "supported_segments": supported_segments,
        "rejected_segments": rejected_segments,
        "local_gap_pixels": int(local_gap.sum()),
    }


def _select_wrist_connected_forearm(
    semantic_forearm: np.ndarray,
    foreground_proposals: np.ndarray,
    joints_2d: np.ndarray,
    observed: np.ndarray,
    config: RemovalEnvelopeV2Config,
    predicted_previous: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select real proposal components; never synthesize wrist-to-edge pixels."""

    shape = semantic_forearm.shape
    wrist_anchor = np.zeros(shape, np.uint8)
    for side in range(2):
        if observed[side] and _finite_hand(joints_2d[side], shape):
            wrist = tuple(np.rint(joints_2d[side, 0]).astype(np.int32))
            cv2.circle(
                wrist_anchor, wrist, config.forearm_wrist_anchor_radius_px, 1, -1,
            )
    if not wrist_anchor.any():
        return np.zeros(shape, bool), {
            "state": "UNKNOWN_NO_OBSERVED_WRIST", "candidate_components": 0,
            "accepted_components": 0,
        }
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_proposals.astype(np.uint8), connectivity=8,
    )
    candidates: list[tuple[float, int, np.ndarray, float, float]] = []
    maximum_pixels = int(round(np.prod(shape) * config.forearm_maximum_image_fraction))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (config.forearm_minimum_component_pixels <= area <= maximum_pixels):
            continue
        component = labels == label
        if not np.any(component & wrist_anchor.astype(bool)):
            continue
        semantic_overlap = int((component & semantic_forearm).sum())
        overlap_ratio = semantic_overlap / max(1, int(semantic_forearm.sum()))
        if overlap_ratio < config.forearm_minimum_semantic_overlap:
            continue
        y, x = np.nonzero(component)
        coordinates = np.stack((x, y), axis=1).astype(np.float64)
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
        elongation = float(np.sqrt(eigenvalues[1] / max(eigenvalues[0], 1e-6)))
        if elongation < config.forearm_minimum_elongation:
            continue
        temporal_score = (
            1.0 if predicted_previous is None else _symmetric_proximity(
                component, predicted_previous, config.forearm_temporal_radius_px,
            )
        )
        if (
            predicted_previous is not None
            and temporal_score < config.forearm_minimum_temporal_score
        ):
            continue
        candidates.append((overlap_ratio + temporal_score, label, component,
                           elongation, temporal_score))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    accepted = np.zeros(shape, bool)
    for _, _, component, _, _ in candidates[: config.forearm_maximum_components]:
        accepted |= component
    return accepted, {
        "state": "SELECTED_WRIST_CONNECTED_PROPOSAL" if accepted.any()
        else "UNKNOWN_NO_VALID_FOREGROUND_PROPOSAL",
        "candidate_components": len(candidates),
        "accepted_components": min(len(candidates), config.forearm_maximum_components),
        "selected_pixels": int(accepted.sum()),
        "selected_shape_temporal_scores": [
            {"elongation": row[3], "temporal_score": row[4]}
            for row in candidates[: config.forearm_maximum_components]
        ],
    }


def _symmetric_proximity(first: np.ndarray, second: np.ndarray, radius: int) -> float:
    if not first.any() or not second.any():
        return 0.0
    forward = float((first & _dilate(second, radius)).sum()) / float(first.sum())
    backward = float((second & _dilate(first, radius)).sum()) / float(second.sum())
    return (forward + backward) / 2.0


def _distance_score(component: np.ndarray, anchor: np.ndarray, maximum: int) -> float:
    if not component.any() or not anchor.any() or maximum <= 0:
        return 0.0
    distance = cv2.distanceTransform((~anchor).astype(np.uint8), cv2.DIST_L2, 3)
    minimum = float(distance[component].min())
    return float(np.clip(1.0 - minimum / maximum, 0.0, 1.0))


def _skeleton(mask: np.ndarray) -> np.ndarray:
    work = mask.astype(np.uint8).copy()
    result = np.zeros_like(work)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while work.any():
        eroded = cv2.erode(work, cross)
        opened = cv2.dilate(eroded, cross)
        result |= work & ~opened
        work = eroded
    return result.astype(bool)


def _cable_shape_scores(
    component: np.ndarray,
    hand_anchor: np.ndarray,
    maximum_anchor_distance: int,
) -> tuple[float, float, float, float, int]:
    y, x = np.nonzero(component)
    coordinates = np.stack((x, y), axis=1).astype(np.float64)
    if len(coordinates) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    thinness = 1.0 - float(np.sqrt(eigenvalues[0] / max(eigenvalues[1], 1e-6)))
    skeleton = _skeleton(component)
    skeleton_pixels = int(skeleton.sum())
    path = float(np.clip(skeleton_pixels / max(1.0, 2.0 * np.sqrt(component.sum())), 0.0, 1.0))
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    endpoints = skeleton & (neighbours == 2)  # count includes the center pixel
    endpoint_count = int(endpoints.sum())
    endpoint_plausibility = float(np.exp(-abs(endpoint_count - 2)))
    endpoint_anchor = _distance_score(endpoints, hand_anchor, maximum_anchor_distance)
    endpoint_score = endpoint_plausibility * endpoint_anchor
    anchor_score = _distance_score(component, hand_anchor, maximum_anchor_distance)
    return anchor_score, thinness, path, endpoint_score, endpoint_count


class RemovalEnvelopeV2Builder:
    """Sequential base-plus-validated-repair builder for one image domain."""

    def __init__(self, shape: tuple[int, int], config: RemovalEnvelopeV2Config) -> None:
        if len(shape) != 2 or min(shape) <= 0:
            raise RemovalEnvelopeV2Error("positive HxW shape required")
        if not 1 <= config.cable_profile.maximum_instances <= 2:
            raise RemovalEnvelopeV2Error("cable maximum_instances must be 1 or 2")
        weights = (
            config.cable_profile.anchor_weight,
            config.cable_profile.thinness_weight,
            config.cable_profile.path_weight,
            config.cable_profile.endpoint_weight,
            config.cable_profile.forward_backward_temporal_weight,
        )
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise RemovalEnvelopeV2Error("cable score weights must be non-negative")
        self.shape = tuple(int(value) for value in shape)
        self.config = config
        self._last_sleeve: np.ndarray | None = None
        self._last_hand_centroid: np.ndarray | None = None
        self._sleeve_gap_age = 0
        self._last_forearm: np.ndarray | None = None
        self._last_wrist_centroid: np.ndarray | None = None
        self._forearm_gap_age = 0
        self._last_cable: np.ndarray | None = None
        self._cable_gap_age = 0
        self._previous_removal_pixels: int | None = None

    def config_record(self) -> dict[str, Any]:
        value = asdict(self.config)
        value["design"] = "SAM_BASE_PLUS_VALIDATED_LOCAL_REPAIRS"
        value["evidence_direction"] = {
            "semantic_role_mask": "DEVELOPMENT_EVIDENCE_WITH_OWN_GATES",
            "removal_envelope_v2": "VISUAL_CLEAN_ONLY",
            "feather_alpha_v2": "VISUAL_CLEAN_ONLY",
            "allowed_consumers": list(ALLOWED_CONSUMERS),
            "forbidden_consumers": list(FORBIDDEN_CONSUMERS),
        }
        return value

    def _attached_sleeve_repair(
        self,
        candidate: np.ndarray,
        sam_hand: np.ndarray,
        mano_line: np.ndarray,
        task_object: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        config = self.config
        boundary = _dilate(sam_hand, 1) ^ _erode(sam_hand, 1)
        eligible = (
            candidate
            & _dilate(boundary, config.sleeve_hand_boundary_radius_px)
            & _dilate(mano_line, config.sleeve_mano_radius_px)
            & ~task_object
        )
        current_centroid = _centroid(sam_hand)
        predicted_previous: np.ndarray | None = None
        if (
            self._last_sleeve is not None
            and self._last_hand_centroid is not None
            and current_centroid is not None
            and self._sleeve_gap_age <= config.sleeve_maximum_association_gap_frames
        ):
            predicted_previous = _translate_mask(
                self._last_sleeve, current_centroid - self._last_hand_centroid,
            )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            eligible.astype(np.uint8), connectivity=8,
        )
        rows: list[tuple[float, int, np.ndarray]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not (config.sleeve_minimum_component_pixels <= area
                    <= config.sleeve_maximum_component_pixels):
                continue
            component = labels == label
            score = 1.0 if predicted_previous is None else _symmetric_proximity(
                component, predicted_previous, config.sleeve_temporal_radius_px,
            )
            if predicted_previous is not None and score < config.sleeve_minimum_comotion_score:
                continue
            rows.append((score, label, component))
        rows.sort(key=lambda row: (-row[0], row[1]))
        selected = np.zeros(self.shape, bool)
        for _, _, component in rows[: config.sleeve_maximum_instances]:
            selected |= component
        if selected.any():
            state = "SEEDED_ATTACHED" if predicted_previous is None else "TRACKED_COMOTION"
            self._last_sleeve = selected.copy()
            self._last_hand_centroid = current_centroid
            self._sleeve_gap_age = 0
        else:
            state = "UNKNOWN_NO_VALID_ATTACHED_INSTANCE"
            self._sleeve_gap_age += 1
            if self._sleeve_gap_age > config.sleeve_maximum_association_gap_frames:
                self._last_sleeve = None
                self._last_hand_centroid = None
        return selected, {
            "state": state,
            "eligible_pixels": int(eligible.sum()),
            "accepted_components": min(len(rows), config.sleeve_maximum_instances),
            "selected_pixels": int(selected.sum()),
        }

    def _track_cable_instance(
        self,
        frame_bgr: np.ndarray,
        explicit_candidate: np.ndarray,
        reverse_support: np.ndarray | None,
        hand_anchor: np.ndarray,
        task_object: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        profile = self.config.cable_profile
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        appearance = cv2.inRange(
            hsv, np.asarray(profile.hsv_lower, np.uint8),
            np.asarray(profile.hsv_upper, np.uint8),
        ).astype(bool)
        candidate = appearance | explicit_candidate
        if profile.candidate_close_radius_px > 0:
            candidate = cv2.morphologyEx(
                candidate.astype(np.uint8), cv2.MORPH_CLOSE,
                _kernel(profile.candidate_close_radius_px),
            ).astype(bool)
        candidate &= ~task_object
        previous = (
            self._last_cable
            if self._last_cable is not None
            and self._cable_gap_age <= profile.maximum_association_gap_frames
            else None
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8,
        )
        scored: list[tuple[float, int, np.ndarray, dict[str, Any]]] = []
        weights = {
            "anchor": profile.anchor_weight,
            "thinness": profile.thinness_weight,
            "path": profile.path_weight,
            "endpoint": profile.endpoint_weight,
            "forward_backward_temporal": profile.forward_backward_temporal_weight,
        }
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not (profile.minimum_component_pixels <= area
                    <= profile.maximum_component_pixels):
                continue
            component = labels == label
            anchor, thinness, path, endpoint, endpoint_count = _cable_shape_scores(
                component, hand_anchor, profile.anchor_distance_px,
            )
            if anchor < profile.minimum_anchor_score:
                continue
            temporal: float | None = None
            active_weights = dict(weights)
            if previous is not None:
                forward_score = _symmetric_proximity(
                    component, previous, profile.temporal_association_radius_px,
                )
                if reverse_support is not None:
                    backward_score = _symmetric_proximity(
                        component, reverse_support,
                        profile.temporal_association_radius_px,
                    )
                    temporal = (forward_score + backward_score) / 2.0
                elif profile.require_reverse_support_after_seed:
                    continue
                else:
                    temporal = forward_score
            elif reverse_support is not None:
                temporal = _symmetric_proximity(
                    component, reverse_support,
                    profile.temporal_association_radius_px,
                )
            else:
                active_weights.pop("forward_backward_temporal")
            scores: dict[str, float | None] = {
                "anchor": anchor,
                "thinness": thinness,
                "path": path,
                "endpoint": endpoint,
                "forward_backward_temporal": temporal,
            }
            denominator = sum(active_weights.values())
            total = sum(
                active_weights[name] * float(scores[name])
                for name in active_weights
            ) / denominator
            if total < profile.minimum_total_score:
                continue
            scored.append((total, label, component, {
                "component_label": label,
                "pixels": area,
                "total_score": float(total),
                "scores": scores,
                "endpoint_count": endpoint_count,
            }))
        scored.sort(key=lambda row: (-row[0], row[1]))
        accepted_rows = scored[: profile.maximum_instances]
        selected = np.zeros(self.shape, bool)
        for _, _, component, _ in accepted_rows:
            selected |= component
        if selected.any():
            selected = _dilate(selected, profile.output_dilation_px) & ~task_object
            state = "SEEDED_TOP_K" if previous is None else "TRACKED_TOP_K_IDENTITY"
            self._last_cable = selected.copy()
            self._cable_gap_age = 0
        else:
            state = "UNKNOWN_NO_VALID_INSTANCE"
            self._cable_gap_age += 1
            if self._cable_gap_age > profile.maximum_association_gap_frames:
                self._last_cable = None
        return selected, {
            "state": state,
            "appearance_profile_id": profile.profile_id,
            "raw_candidate_components": count - 1,
            "valid_scored_components": len(scored),
            "accepted_components": len(accepted_rows),
            "maximum_instances": profile.maximum_instances,
            "accepted": [row[3] for row in accepted_rows],
            "selected_pixels": int(selected.sum()),
            "unbounded_hold_used": False,
            "reverse_support_supplied": reverse_support is not None,
            "reverse_support_required_after_seed": (
                profile.require_reverse_support_after_seed
            ),
        }

    def _quality(
        self,
        *,
        semantic_base: np.ndarray,
        removal: np.ndarray,
        repair_union: np.ndarray,
        stable_background: np.ndarray | None,
        protected_object_core: np.ndarray,
    ) -> dict[str, Any]:
        base_pixels = int(semantic_base.sum())
        removal_pixels = int(removal.sum())
        added_pixels = int((repair_union & ~semantic_base).sum())
        area_inflation = removal_pixels / base_pixels if base_pixels else None
        repair_ratio = added_pixels / removal_pixels if removal_pixels else None
        temporal_derivative = (
            abs(removal_pixels - self._previous_removal_pixels)
            / max(1, self._previous_removal_pixels)
            if self._previous_removal_pixels is not None else None
        )
        background_spill = (
            int((removal & stable_background).sum()) / max(1, removal_pixels)
            if stable_background is not None and removal_pixels else None
        )
        protected_object_pixels = int(protected_object_core.sum())
        protected_object_damage = int((removal & protected_object_core).sum())
        protected_object_damage_ratio = (
            protected_object_damage / protected_object_pixels
            if protected_object_pixels else 0.0
        )

        def gate(value: float | None, maximum: float, warmup: bool = False) -> dict[str, Any]:
            if value is None:
                return {"status": "WARMUP" if warmup else "UNKNOWN", "value": None,
                        "maximum": maximum}
            return {"status": "PASS" if value <= maximum else "FAIL",
                    "value": float(value), "maximum": maximum}

        gates = {
            "semantic_base_nonempty": {
                "status": "PASS" if base_pixels else "UNKNOWN",
                "value": base_pixels,
                "minimum": 1,
            },
            "area_inflation": gate(area_inflation, self.config.maximum_area_inflation),
            "background_spill": gate(
                background_spill, self.config.maximum_background_spill_ratio,
            ),
            "temporal_area_derivative": gate(
                temporal_derivative, self.config.maximum_temporal_area_derivative,
                warmup=self._previous_removal_pixels is None,
            ),
            "repair_contribution": gate(
                repair_ratio, self.config.maximum_repair_contribution_ratio,
            ),
            "protected_object_core_damage": gate(
                protected_object_damage_ratio,
                self.config.maximum_protected_object_core_damage_ratio,
            ),
        }
        statuses = {record["status"] for record in gates.values()}
        if "FAIL" in statuses:
            status = "REJECTED_QUALITY"
        elif "UNKNOWN" in statuses:
            status = "UNKNOWN_FAIL_CLOSED"
        else:
            status = "PASS"
        self._previous_removal_pixels = removal_pixels
        return {
            "status": status,
            "gates": gates,
            "metrics": {
                "semantic_base_pixels": base_pixels,
                "removal_pixels": removal_pixels,
                "repair_added_pixels": added_pixels,
                "area_inflation": area_inflation,
                "background_spill_ratio": background_spill,
                "temporal_area_derivative": temporal_derivative,
                "repair_contribution_ratio": repair_ratio,
                "protected_object_core_pixels": protected_object_pixels,
                "protected_object_core_damage_pixels": protected_object_damage,
                "protected_object_core_damage_ratio": protected_object_damage_ratio,
            },
        }

    def step(
        self,
        *,
        frame_bgr: np.ndarray,
        semantic_masks: Mapping[str, np.ndarray],
        foreground_proposals: np.ndarray,
        sleeve_candidate_mask: np.ndarray,
        cable_candidate_mask: np.ndarray,
        reverse_cable_support_mask: np.ndarray | None,
        task_object_mask: np.ndarray,
        joints_2d: np.ndarray,
        observed: np.ndarray,
        stable_background_mask: np.ndarray | None,
    ) -> dict[str, Any]:
        image = np.asarray(frame_bgr)
        if image.shape != (*self.shape, 3) or image.dtype != np.uint8:
            raise RemovalEnvelopeV2Error("frame_bgr must be uint8 HxWx3")
        if set(semantic_masks) != {"hand", "forearm", "equipment"}:
            raise RemovalEnvelopeV2Error(
                "semantic masks must be exactly hand, forearm, equipment"
            )
        semantic = {
            name: _mask(mask, self.shape, f"semantic.{name}")
            for name, mask in semantic_masks.items()
        }
        proposals = _mask(foreground_proposals, self.shape, "foreground_proposals")
        sleeve_candidate = _mask(
            sleeve_candidate_mask, self.shape, "sleeve_candidate_mask",
        )
        cable_candidate = _mask(
            cable_candidate_mask, self.shape, "cable_candidate_mask",
        )
        reverse_cable_support = (
            None if reverse_cable_support_mask is None
            else _mask(
                reverse_cable_support_mask,
                self.shape,
                "reverse_cable_support_mask",
            )
        )
        task_object = _mask(task_object_mask, self.shape, "task_object_mask")
        background = (
            None if stable_background_mask is None
            else _mask(stable_background_mask, self.shape, "stable_background_mask")
        )
        joints = np.asarray(joints_2d, np.float64)
        direct = np.asarray(observed, bool)
        if joints.shape != (2, 21, 2) or direct.shape != (2,):
            raise RemovalEnvelopeV2Error("expected two MANO21 hands and observation flags")

        observed_wrists = [
            joints[side, 0] for side in range(2)
            if direct[side] and _finite_hand(joints[side], self.shape)
        ]
        wrist_centroid = (
            np.mean(np.stack(observed_wrists), axis=0) if observed_wrists else None
        )
        predicted_forearm: np.ndarray | None = None
        if (
            self._last_forearm is not None
            and self._last_wrist_centroid is not None
            and wrist_centroid is not None
            and self._forearm_gap_age
            <= self.config.forearm_maximum_association_gap_frames
        ):
            predicted_forearm = _translate_mask(
                self._last_forearm, wrist_centroid - self._last_wrist_centroid,
            )
        selected_forearm, forearm_record = _select_wrist_connected_forearm(
            semantic["forearm"], proposals, joints, direct, self.config,
            predicted_forearm,
        )
        if selected_forearm.any():
            forearm_record["state"] = (
                "SEEDED_WRIST_CONNECTED_PROPOSAL" if predicted_forearm is None
                else "TRACKED_WRIST_CONNECTED_PROPOSAL"
            )
            self._last_forearm = selected_forearm.copy()
            self._last_wrist_centroid = wrist_centroid
            self._forearm_gap_age = 0
        else:
            self._forearm_gap_age += 1
            if (
                self._forearm_gap_age
                > self.config.forearm_maximum_association_gap_frames
            ):
                self._last_forearm = None
                self._last_wrist_centroid = None
        protected_object_core = (
            _erode(task_object, self.config.object_core_erosion_radius_px)
            & ~_dilate(
                semantic["hand"], self.config.object_hand_interaction_radius_px,
            )
        )
        selected_forearm &= ~protected_object_core
        admitted_equipment = semantic["equipment"] & ~protected_object_core
        semantic_base = semantic["hand"] | selected_forearm | admitted_equipment
        mano_line = _mano_centerline(joints, direct, self.shape)
        mano_repair, mano_record = _mano_gap_repair(
            semantic["hand"], task_object, joints, direct, self.config,
        )
        sleeve_repair, sleeve_record = self._attached_sleeve_repair(
            sleeve_candidate, semantic["hand"], mano_line, task_object,
        )
        cable, cable_record = self._track_cable_instance(
            image, cable_candidate, reverse_cable_support,
            semantic["hand"] | semantic["equipment"],
            task_object,
        )
        repair_sources = {
            "validated_mano_gap_repair": mano_repair,
            "validated_attached_sleeve_repair": sleeve_repair,
            "validated_cable_instance": cable,
        }
        repair_union = np.zeros(self.shape, bool)
        exclusive_contributions: dict[str, int] = {}
        for name, repair in repair_sources.items():
            local = repair & ~task_object
            exclusive_contributions[name] = int((local & ~semantic_base & ~repair_union).sum())
            repair_union |= local
        removal = semantic_base | repair_union

        source_masks = {
            "admitted_sam_hand": semantic["hand"],
            "selected_sam_forearm": selected_forearm,
            "admitted_sam_equipment": admitted_equipment,
            **repair_sources,
        }
        source_bits = np.zeros(self.shape, np.uint16)
        for name, source in source_masks.items():
            source_bits[source & removal] |= np.uint16(SOURCE_BITS[name])
        if not np.array_equal(source_bits != 0, removal):
            raise RemovalEnvelopeV2Error("pixel provenance does not close V2 removal")

        expanded = _dilate(removal, self.config.feather_radius_px)
        size = self.config.feather_radius_px * 2 + 1
        feather = cv2.GaussianBlur(expanded.astype(np.float32), (size, size), 0)
        feather[removal] = 1.0
        feather = np.clip(feather, 0.0, 1.0)
        quality = self._quality(
            semantic_base=semantic_base,
            removal=removal,
            repair_union=repair_union,
            stable_background=background,
            protected_object_core=protected_object_core,
        )
        return {
            "semantic_base": semantic_base,
            "selected_forearm": selected_forearm,
            "mano_gap_repair": mano_repair,
            "attached_sleeve_repair": sleeve_repair,
            "cable_instance": cable,
            "protected_object_core": protected_object_core,
            "removal_envelope": removal,
            "source_bits": source_bits,
            "feather_alpha": feather,
            "quality": quality,
            "provenance": {
                "forearm": forearm_record,
                "mano": mano_record,
                "sleeve": sleeve_record,
                "cable": cable_record,
                "source_pixels": {
                    name: int(mask.sum()) for name, mask in source_masks.items()
                },
                "exclusive_repair_contribution_pixels": exclusive_contributions,
                "source_bits": dict(SOURCE_BITS),
                "policy": {
                    "design": "SAM_BASE_PLUS_VALIDATED_LOCAL_REPAIRS",
                    "unknown_behavior": "EMPTY_REPAIR_FAIL_CLOSED",
                    "wrist_to_image_boundary_generation": False,
                    "full_mano_capsule_generation": False,
                    "all_appearance_components_accepted": False,
                    "removal_geometry_authority": False,
                },
            },
        }
