"""Auditable visual-only removal envelopes for human/equipment cleanup.

This module deliberately separates semantic evidence from pixels that a visual
cleaner is allowed to erase.  The latter may contain geometric expansion and
short, explicitly labelled temporal holds, so it is never valid input to
Depth, Object6D, Contact, or control-ground-truth consumers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np


class RemovalEnvelopeError(ValueError):
    """Raised when an input violates the removal-envelope contract."""


FINGER_CHAINS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
MCP_INDICES = np.asarray([1, 5, 9, 13, 17], np.int64)
SOURCE_BITS: dict[str, int] = {
    "admitted_sam_hand_forearm": 1,
    "optional_sam_sleeve": 2,
    "admitted_sam_cable": 4,
    "observed_mano_finger_capsule": 8,
    "bounded_hold_mano_finger_capsule": 16,
    "observed_palm_wrist_forearm_corridor": 32,
    "bounded_hold_palm_wrist_forearm_corridor": 64,
    "appearance_tracked_cable": 128,
}
ALLOWED_CONSUMERS = ("CLEAN_MASK_QA", "VISUAL_INPAINT")
FORBIDDEN_CONSUMERS = (
    "DEPTH", "OBJECT6D", "CONTACT", "ROBOT_GEOMETRY", "CONTROL_GROUND_TRUTH",
)


def enforce_consumer_firewall(artifact_class: str, consumer: str) -> None:
    """Fail closed when inferred erase support is presented as geometry evidence."""

    if artifact_class not in {"removal_envelope", "feather_alpha"}:
        raise RemovalEnvelopeError(f"unsupported artifact class: {artifact_class}")
    if consumer in FORBIDDEN_CONSUMERS:
        raise RemovalEnvelopeError(
            f"{artifact_class} is visual-only and forbidden for {consumer}"
        )
    if consumer not in ALLOWED_CONSUMERS:
        raise RemovalEnvelopeError(f"consumer is not allow-listed: {consumer}")


@dataclass(frozen=True)
class CableAppearanceProfile:
    """One replaceable device-instance profile, not a system-wide cable class."""

    profile_id: str
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    min_component_pixels: int = 6
    max_component_pixels: int = 18_000
    hand_anchor_radius_px: int = 78
    previous_track_radius_px: int = 70
    output_radius_px: int = 4
    maximum_hold_frames: int = 2


@dataclass(frozen=True)
class RemovalEnvelopeConfig:
    cable_profile: CableAppearanceProfile
    accessory_scale: float = 1.8
    high_confidence: float = 0.45
    medium_confidence: float = 0.25
    low_confidence_scale: float = 0.82
    medium_confidence_scale: float = 0.92
    maximum_radius_growth: float = 1.20
    minimum_finger_radius_px: float = 5.0
    maximum_finger_radius_px: float = 38.0
    maximum_geometry_hold_frames: int = 2
    held_radius_decay: float = 0.95
    hold_endpoint_confidence_minimum: float = 0.40
    hold_maximum_joint_step_px: float = 40.0
    palm_dilation_scale: float = 0.16
    forearm_radius_scale: float = 0.22
    minimum_forearm_radius_px: float = 12.0
    maximum_forearm_radius_px: float = 55.0
    object_interior_erosion_px: int = 5
    interaction_band_px: int = 5
    feather_radius_px: int = 6


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(value, bool)
    if result.shape != shape:
        raise RemovalEnvelopeError(f"{name} must have shape {shape}, got {result.shape}")
    return result


def _kernel(radius: int) -> np.ndarray:
    size = max(1, int(radius) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, bool).copy()
    return cv2.dilate(mask.astype(np.uint8), _kernel(radius)).astype(bool)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, bool).copy()
    return cv2.erode(mask.astype(np.uint8), _kernel(radius)).astype(bool)


def _finite_hand(joints: np.ndarray, shape: tuple[int, int]) -> bool:
    if joints.shape != (21, 2) or not np.isfinite(joints).all():
        return False
    height, width = shape
    inside = (
        (joints[:, 0] >= 0.0) & (joints[:, 0] < width)
        & (joints[:, 1] >= 0.0) & (joints[:, 1] < height)
    )
    return int(inside.sum()) >= 16


def projected_palm_width(joints: np.ndarray) -> float:
    """Estimate image-space palm width from stable MANO MCP landmarks."""

    points = np.asarray(joints, np.float64)
    spans = (
        np.linalg.norm(points[5] - points[17]),
        np.linalg.norm(points[5] - points[13]),
        np.linalg.norm(points[9] - points[17]),
    )
    finite = np.asarray(spans)[np.isfinite(spans)]
    if not finite.size:
        raise RemovalEnvelopeError("cannot estimate projected palm width")
    return float(np.median(finite))


def bounded_internal_geometry(
    joints_2d: np.ndarray,
    observed: np.ndarray,
    detector_confidence: np.ndarray,
    *,
    maximum_gap_frames: int = 2,
    endpoint_confidence_minimum: float = 0.40,
    maximum_joint_step_px: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only short, two-sided, endpoint-verified internal gaps.

    Returned states never alter the source ``observed`` array.  Leading,
    trailing, over-length, low-confidence, or high-motion gaps remain UNKNOWN.
    """

    joints = np.asarray(joints_2d, np.float64)
    direct = np.asarray(observed, bool)
    confidence = np.asarray(detector_confidence, np.float64)
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 2):
        raise RemovalEnvelopeError("joints_2d must be 2xFx21x2")
    if direct.shape != joints.shape[:2] or confidence.shape != joints.shape[:2]:
        raise RemovalEnvelopeError("observation/confidence axes differ from joints")
    output = joints.copy()
    states = np.full(direct.shape, "UNKNOWN", dtype="<U24")
    states[direct] = "OBSERVED"
    frame_count = joints.shape[1]
    for side in range(2):
        frame = 0
        while frame < frame_count:
            if direct[side, frame]:
                frame += 1
                continue
            start = frame
            while frame < frame_count and not direct[side, frame]:
                frame += 1
            end = frame
            length = end - start
            if start == 0 or end == frame_count or length > maximum_gap_frames:
                continue
            left = start - 1
            right = end
            if (
                confidence[side, left] < endpoint_confidence_minimum
                or confidence[side, right] < endpoint_confidence_minimum
                or not np.isfinite(joints[side, [left, right]]).all()
            ):
                continue
            steps = length + 1
            maximum_step = float(np.max(np.linalg.norm(
                joints[side, right] - joints[side, left], axis=-1,
            )) / steps)
            if maximum_step > maximum_joint_step_px:
                continue
            for offset, target in enumerate(range(start, end), start=1):
                alpha = offset / steps
                output[side, target] = (
                    (1.0 - alpha) * joints[side, left]
                    + alpha * joints[side, right]
                )
                states[side, target] = "BOUNDED_INTERNAL_HOLD"
    return output, states


def _ray_to_boundary(
    origin_xy: np.ndarray, direction_xy: np.ndarray, shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    origin = np.asarray(origin_xy, np.float64)
    direction = np.asarray(direction_xy, np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-6:
        direction = np.asarray([0.0, 1.0])
    else:
        direction /= norm
    candidates: list[float] = []
    for coordinate, delta, maximum in zip(origin, direction, (width - 1, height - 1)):
        if delta > 1e-9:
            candidates.append((maximum - coordinate) / delta)
        elif delta < -1e-9:
            candidates.append((0.0 - coordinate) / delta)
    positive = [value for value in candidates if value >= 0.0 and np.isfinite(value)]
    if not positive:
        return origin.copy()
    endpoint = origin + min(positive) * direction
    endpoint[0] = np.clip(endpoint[0], 0.0, width - 1.0)
    endpoint[1] = np.clip(endpoint[1], 0.0, height - 1.0)
    return endpoint


def _draw_geometry(
    joints: np.ndarray,
    *,
    finger_radius: float,
    config: RemovalEnvelopeConfig,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, float | list[float]]]:
    finger = np.zeros(shape, np.uint8)
    radius = max(1, int(round(finger_radius)))
    points = np.rint(joints).astype(np.int32)
    for chain in FINGER_CHAINS:
        for first, second in zip(chain[:-1], chain[1:]):
            cv2.line(finger, tuple(points[first]), tuple(points[second]), 255,
                     thickness=radius * 2, lineType=cv2.LINE_AA)
        for index in chain:
            cv2.circle(finger, tuple(points[index]), radius, 255, -1, cv2.LINE_AA)

    palm_width = projected_palm_width(joints)
    palm = np.zeros(shape, np.uint8)
    palm_points = points[np.asarray([0, 1, 5, 9, 13, 17], np.int64)]
    hull = cv2.convexHull(palm_points)
    cv2.fillConvexPoly(palm, hull, 255, lineType=cv2.LINE_AA)
    palm_radius = max(2, int(round(palm_width * config.palm_dilation_scale)))
    palm = cv2.dilate(palm, _kernel(palm_radius))

    palm_center = np.mean(joints[MCP_INDICES], axis=0)
    wrist = joints[0]
    endpoint = _ray_to_boundary(wrist, wrist - palm_center, shape)
    forearm_radius = float(np.clip(
        palm_width * config.forearm_radius_scale,
        config.minimum_forearm_radius_px,
        config.maximum_forearm_radius_px,
    ))
    corridor = np.zeros(shape, np.uint8)
    cv2.line(
        corridor, tuple(np.rint(wrist).astype(np.int32)),
        tuple(np.rint(endpoint).astype(np.int32)), 255,
        thickness=max(2, int(round(forearm_radius * 2))), lineType=cv2.LINE_AA,
    )
    cv2.circle(corridor, tuple(np.rint(wrist).astype(np.int32)),
               max(1, int(round(forearm_radius))), 255, -1, cv2.LINE_AA)
    palm_corridor = (palm > 0) | (corridor > 0)
    return finger > 0, palm_corridor, {
        "projected_palm_width_px": palm_width,
        "finger_radius_px": float(finger_radius),
        "forearm_radius_px": forearm_radius,
        "forearm_boundary_endpoint_xy": [float(endpoint[0]), float(endpoint[1])],
    }


class RemovalEnvelopeBuilder:
    """Sequential, bounded-state builder for one fixed image domain."""

    def __init__(
        self, shape: tuple[int, int], config: RemovalEnvelopeConfig,
    ) -> None:
        if len(shape) != 2 or min(shape) <= 0:
            raise RemovalEnvelopeError("positive HxW shape required")
        self.shape = tuple(int(value) for value in shape)
        self.config = config
        self._last_geometry: list[dict[str, Any] | None] = [None, None]
        self._last_cable: np.ndarray | None = None
        self._cable_hold_age = 0

    def config_record(self) -> dict[str, Any]:
        value = asdict(self.config)
        value["evidence_direction"] = {
            "semantic_role_mask": "DEVELOPMENT_EVIDENCE_WITH_OWN_GATES",
            "removal_envelope": "VISUAL_CLEAN_ONLY",
            "feather_alpha": "VISUAL_CLEAN_ONLY",
            "allowed_consumers": list(ALLOWED_CONSUMERS),
            "forbidden_consumers": list(FORBIDDEN_CONSUMERS),
        }
        return value

    def _geometry_for_side(
        self,
        side: int,
        joints: np.ndarray,
        observed: bool,
        confidence: float,
        geometry_state: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        config = self.config
        valid = (
            geometry_state in {"OBSERVED", "BOUNDED_INTERNAL_HOLD"}
            and _finite_hand(joints, self.shape)
        )
        last = self._last_geometry[side]
        if valid and geometry_state == "OBSERVED" and bool(observed):
            palm_width = projected_palm_width(joints)
            projected_finger_width = palm_width * (0.18 if side in (0, 1) else 0.18)
            confidence_scale = (
                1.0 if confidence >= config.high_confidence
                else config.medium_confidence_scale
                if confidence >= config.medium_confidence
                else config.low_confidence_scale
            )
            candidate = float(np.clip(
                projected_finger_width * config.accessory_scale * confidence_scale,
                config.minimum_finger_radius_px,
                config.maximum_finger_radius_px,
            ))
            if last is not None:
                previous = float(last["radius"])
                if confidence < config.high_confidence:
                    candidate = min(candidate, previous)
                else:
                    candidate = min(candidate, previous * config.maximum_radius_growth)
            finger, corridor, metrics = _draw_geometry(
                joints, finger_radius=candidate, config=config, shape=self.shape,
            )
            self._last_geometry[side] = {
                "joints": np.asarray(joints, np.float64).copy(),
                "radius": candidate,
                "age": 0,
            }
            return finger, corridor, {
                "state": "DIRECT_OBSERVED",
                "detector_confidence": float(confidence),
                **metrics,
            }

        if valid and geometry_state == "BOUNDED_INTERNAL_HOLD" and last is not None:
            age = int(last["age"]) + 1
            inferred_width = projected_palm_width(joints) * 0.18 * config.accessory_scale
            radius = min(
                float(last["radius"]) * config.held_radius_decay ** age,
                inferred_width,
            )
            radius = float(np.clip(
                radius, config.minimum_finger_radius_px,
                config.maximum_finger_radius_px,
            ))
            finger, corridor, metrics = _draw_geometry(
                joints, finger_radius=radius,
                config=config, shape=self.shape,
            )
            last["age"] = age
            return finger, corridor, {
                "state": "BOUNDED_INTERNAL_HOLD",
                "hold_age_frames": age,
                "detector_confidence": float(confidence),
                **metrics,
            }

        if last is not None:
            last["age"] = int(last["age"]) + 1
        empty = np.zeros(self.shape, bool)
        return empty, empty.copy(), {
            "state": "UNKNOWN_NO_GEOMETRY",
            "detector_confidence": float(confidence),
        }

    def _track_cable(
        self,
        frame_bgr: np.ndarray,
        geometry_anchor: np.ndarray,
        semantic_cable: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        profile = self.config.cable_profile
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        candidate = cv2.inRange(
            hsv, np.asarray(profile.hsv_lower, np.uint8),
            np.asarray(profile.hsv_upper, np.uint8),
        )
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, _kernel(1))
        candidate_bool = candidate.astype(bool)
        anchor = _dilate(geometry_anchor | semantic_cable, profile.hand_anchor_radius_px)
        if self._last_cable is not None:
            anchor |= _dilate(self._last_cable, profile.previous_track_radius_px)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8,
        )
        selected = np.zeros(self.shape, bool)
        accepted_components = 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not (profile.min_component_pixels <= area <= profile.max_component_pixels):
                continue
            component = labels == label
            if not np.any(component & anchor):
                continue
            selected |= component
            accepted_components += 1
        selected |= semantic_cable
        if selected.any():
            selected = _dilate(selected, profile.output_radius_px)
            self._last_cable = selected.copy()
            self._cable_hold_age = 0
            state = "APPEARANCE_TRACKED"
        elif self._last_cable is not None and self._cable_hold_age < profile.maximum_hold_frames:
            self._cable_hold_age += 1
            selected = self._last_cable.copy()
            state = "BOUNDED_TEMPORAL_HOLD"
        else:
            self._cable_hold_age += 1
            selected = np.zeros(self.shape, bool)
            state = "UNKNOWN_NO_CABLE_EVIDENCE"
        return selected, {
            "state": state,
            "appearance_profile_id": profile.profile_id,
            "appearance_candidate_pixels": int(candidate_bool.sum()),
            "accepted_components": accepted_components,
            "semantic_cable_pixels": int(semantic_cable.sum()),
            "tracked_cable_pixels": int(selected.sum()),
            "hold_age_frames": self._cable_hold_age if "HOLD" in state else 0,
        }

    def step(
        self,
        *,
        frame_bgr: np.ndarray,
        joints_2d: np.ndarray,
        observed: np.ndarray,
        detector_confidence: np.ndarray,
        semantic_masks: Mapping[str, np.ndarray],
        task_object_mask: np.ndarray,
        geometry_states: np.ndarray | None = None,
    ) -> dict[str, Any]:
        image = np.asarray(frame_bgr)
        if image.shape != (*self.shape, 3) or image.dtype != np.uint8:
            raise RemovalEnvelopeError("frame_bgr must be uint8 HxWx3 in the fixed domain")
        joints = np.asarray(joints_2d, np.float64)
        direct = np.asarray(observed, bool)
        confidence = np.asarray(detector_confidence, np.float64)
        if joints.shape != (2, 21, 2) or direct.shape != (2,) or confidence.shape != (2,):
            raise RemovalEnvelopeError("expected two MANO21 hands and two observation states")
        if geometry_states is None:
            state_values = np.where(direct, "OBSERVED", "UNKNOWN")
        else:
            state_values = np.asarray(geometry_states).astype(str)
            if state_values.shape != (2,):
                raise RemovalEnvelopeError("geometry_states must contain two side states")
            unknown_states = set(state_values.tolist()) - {
                "OBSERVED", "BOUNDED_INTERNAL_HOLD", "UNKNOWN",
            }
            if unknown_states:
                raise RemovalEnvelopeError(
                    f"unsupported geometry states: {sorted(unknown_states)}"
                )
        allowed = {"hand", "forearm", "sleeve", "cable"}
        if set(semantic_masks) != allowed:
            raise RemovalEnvelopeError(f"semantic masks must be exactly {sorted(allowed)}")
        semantic = {
            name: _mask(value, self.shape, f"semantic.{name}")
            for name, value in semantic_masks.items()
        }
        task_object = _mask(task_object_mask, self.shape, "task_object")

        fingers = np.zeros(self.shape, bool)
        palm_forearm = np.zeros(self.shape, bool)
        observed_fingers = np.zeros(self.shape, bool)
        held_fingers = np.zeros(self.shape, bool)
        observed_corridors = np.zeros(self.shape, bool)
        held_corridors = np.zeros(self.shape, bool)
        hand_records: list[dict[str, Any]] = []
        for side in range(2):
            finger, corridor, record = self._geometry_for_side(
                side, joints[side], bool(direct[side]), float(confidence[side]),
                str(state_values[side]),
            )
            fingers |= finger
            palm_forearm |= corridor
            if record["state"] == "DIRECT_OBSERVED":
                observed_fingers |= finger
                observed_corridors |= corridor
            elif record["state"] == "BOUNDED_INTERNAL_HOLD":
                held_fingers |= finger
                held_corridors |= corridor
            hand_records.append({"side": "left" if side == 0 else "right", **record})

        geometry = fingers | palm_forearm
        cable, cable_record = self._track_cable(image, geometry, semantic["cable"])
        semantic_human = semantic["hand"] | semantic["forearm"]
        semantic_optional = semantic["sleeve"] | semantic["cable"]
        erase_candidate = semantic_human | semantic_optional | geometry | cable

        visible_interior = _erode(task_object, self.config.object_interior_erosion_px)
        interaction_band = _dilate(geometry, self.config.interaction_band_px)
        protected_visible_object = visible_interior & ~interaction_band
        removal = erase_candidate & ~protected_visible_object

        source_bits = np.zeros(self.shape, np.uint16)
        source_masks = {
            "admitted_sam_hand_forearm": semantic_human,
            "optional_sam_sleeve": semantic["sleeve"],
            "admitted_sam_cable": semantic["cable"],
            "observed_mano_finger_capsule": observed_fingers,
            "bounded_hold_mano_finger_capsule": held_fingers,
            "observed_palm_wrist_forearm_corridor": observed_corridors,
            "bounded_hold_palm_wrist_forearm_corridor": held_corridors,
            "appearance_tracked_cable": cable & ~semantic["cable"],
        }
        for name, mask in source_masks.items():
            source_bits[mask] |= np.uint16(SOURCE_BITS[name])
        source_bits[~removal] = 0
        if not np.array_equal(source_bits != 0, removal):
            raise RemovalEnvelopeError("pixel provenance does not close removal envelope")

        expanded = _dilate(removal, self.config.feather_radius_px)
        size = self.config.feather_radius_px * 2 + 1
        alpha = cv2.GaussianBlur(
            expanded.astype(np.float32), (size, size), 0,
        )
        alpha[removal] = 1.0
        alpha = np.clip(alpha, 0.0, 1.0)

        direct_joints = 0
        covered_joints = 0
        for side in range(2):
            if not (direct[side] and _finite_hand(joints[side], self.shape)):
                continue
            rounded = np.rint(joints[side]).astype(np.int64)
            height, width = self.shape
            inside = (
                (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
                & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
            )
            direct_joints += int(inside.sum())
            xy = rounded[inside]
            covered_joints += int(removal[xy[:, 1], xy[:, 0]].sum())

        return {
            "mano_finger_capsules": fingers,
            "palm_wrist_forearm_corridors": palm_forearm,
            "cable_tracked_region": cable,
            "protected_visible_object": protected_visible_object,
            "removal_envelope": removal,
            "source_bits": source_bits,
            "feather_alpha": alpha,
            "provenance": {
                "hands": hand_records,
                "cable": cable_record,
                "semantic_pixels": {name: int(mask.sum()) for name, mask in semantic.items()},
                "component_pixels": {
                    "mano_finger_capsules": int(fingers.sum()),
                    "palm_wrist_forearm_corridors": int(palm_forearm.sum()),
                    "cable_tracked_region": int(cable.sum()),
                    "protected_visible_object": int(protected_visible_object.sum()),
                    "removal_envelope": int(removal.sum()),
                },
                "source_bits": dict(SOURCE_BITS),
                "direct_mano_joint_coverage": {
                    "covered": covered_joints,
                    "eligible": direct_joints,
                    "ratio": float(covered_joints / direct_joints) if direct_joints else None,
                },
                "protected_object_overlap_after_finalization": int(
                    (removal & protected_visible_object).sum()
                ),
                "erase_candidate_task_object_overlap": int(
                    (erase_candidate & task_object).sum()
                ),
                "authority": {
                    "removal_envelope": "VISUAL_CLEAN_ONLY_INFERRED_ERASE_SUPPORT",
                    "feather_alpha": "VISUAL_CLEAN_ONLY_SYNTHETIC_BOUNDARY",
                    "geometry_consumers_forbidden": [
                        "Depth", "Object6D", "Contact", "RobotControlTruth",
                    ],
                },
            },
        }
