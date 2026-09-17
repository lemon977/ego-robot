#!/usr/bin/env python3
"""Colour-free PICO/task-geometry anchors for raw-point Mask inference."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


class AnchorError(RuntimeError):
    pass


def clipped_unique(points: Sequence[Sequence[float]], width: int, height: int, margin: int) -> list[list[int]]:
    seen: set[tuple[int, int]] = set()
    result: list[list[int]] = []
    for point in points:
        x, y = map(int, np.rint(point))
        if not (margin <= x < width - margin and margin <= y < height - margin):
            continue
        if (x, y) not in seen:
            seen.add((x, y))
            result.append([x, y])
    return result


def tracker_candidate_points(
    wrist_xy: Sequence[float],
    palm_xy: Sequence[float],
    width: int,
    height: int,
    radii_px: Sequence[float],
    angles_per_radius: int,
    margin_px: int = 2,
) -> list[list[int]]:
    """Sample a wrist-centred polar lattice aligned to the wrist/palm axis.

    Full-circle coverage avoids assuming which surface of a rotated wrist
    carries the device.  The image is never inspected.
    """
    if angles_per_radius < 4:
        raise AnchorError("angles_per_radius must be at least 4")
    wrist = np.asarray(wrist_xy, dtype=np.float64)
    palm = np.asarray(palm_xy, dtype=np.float64)
    if wrist.shape != (2,) or palm.shape != (2,) or not np.isfinite([*wrist, *palm]).all():
        raise AnchorError("wrist and palm must be finite 2D points")
    direction = wrist - palm
    base = math.atan2(direction[1], direction[0]) if np.linalg.norm(direction) > 1e-6 else 0.0
    candidates = []
    for radius in radii_px:
        if radius <= 0:
            raise AnchorError("tracker radii must be positive")
        for index in range(angles_per_radius):
            angle = base + 2.0 * math.pi * index / angles_per_radius
            candidates.append(wrist + float(radius) * np.asarray([math.cos(angle), math.sin(angle)]))
    return clipped_unique(candidates, width, height, margin_px)


def apply_homography(point_xy: Sequence[float], matrix: Sequence[Sequence[float]] | None) -> np.ndarray:
    point = np.asarray([float(point_xy[0]), float(point_xy[1]), 1.0])
    if matrix is None:
        return point[:2]
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise AnchorError("homography must be a finite 3x3 matrix")
    projected = transform @ point
    if abs(projected[2]) < 1e-9:
        raise AnchorError("homography projected point to infinity")
    return projected[:2] / projected[2]


def project_world_point(
    point_world: Sequence[float],
    camera_to_world: Sequence[Sequence[float]],
    intrinsics: Sequence[Sequence[float]],
) -> tuple[np.ndarray, float]:
    """Project one metric world point using the authoritative PICO camera pose."""
    point = np.asarray(point_world, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if point.shape != (3,) or c2w.shape != (4, 4) or k.shape != (3, 3):
        raise AnchorError("world point, c2w, or intrinsics geometry is invalid")
    if not np.isfinite(point).all() or not np.isfinite(c2w).all() or not np.isfinite(k).all():
        raise AnchorError("world projection authority contains nonfinite values")
    try:
        camera = np.linalg.inv(c2w) @ np.concatenate((point, [1.0]))
    except np.linalg.LinAlgError as error:
        raise AnchorError("camera-to-world matrix is singular") from error
    depth = float(camera[2])
    if depth <= 1e-4:
        raise AnchorError("world anchor is behind the current camera")
    pixel = k @ camera[:3]
    return pixel[:2] / pixel[2], depth


def derive_setup_world_anchors(
    task_config: Mapping[str, Any],
    reference_metadata: Mapping[str, Any],
    pinch_world_samples: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    *,
    open_pinch_penalty_px_per_m: float,
    closed_pinch_reference_m: float,
    reference_match_distance_px_max: float,
    contact_binding_config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Lift exact setup pixels to world using nearby PICO pinch depth evidence.

    PICO samples choose depth only.  The setup pixel itself defines the ray, so
    no hand coordinate is silently substituted for a task anchor.  The result
    can then be reprojected by each frame's c2w/K without inspecting RGB.
    """
    if width <= 0 or height <= 0 or not pinch_world_samples:
        raise AnchorError("setup-world derivation requires image geometry and PICO pinch samples")
    c2w = np.asarray(reference_metadata.get("c2w"), dtype=np.float64)
    k = np.asarray(reference_metadata.get("k"), dtype=np.float64)
    if c2w.shape != (4, 4) or k.shape != (3, 3) or not np.isfinite(c2w).all() or not np.isfinite(k).all():
        raise AnchorError("reference PICO c2w/K authority is malformed")
    try:
        inverse_k = np.linalg.inv(k)
    except np.linalg.LinAlgError as error:
        raise AnchorError("reference intrinsics matrix is singular") from error

    projected_samples: list[dict[str, Any]] = []
    for sample in pinch_world_samples:
        world = np.asarray(sample.get("world_xyz"), dtype=np.float64)
        pinch_distance = float(sample.get("thumb_index_distance_m", float("nan")))
        if world.shape != (3,) or not np.isfinite(world).all() or not np.isfinite(pinch_distance):
            continue
        try:
            pixel, depth = project_world_point(world, c2w, k)
        except AnchorError:
            continue
        if not 0.1 <= depth <= 3.0:
            continue
        projected_samples.append({
            **sample,
            "world_xyz": world,
            "reference_xy": pixel,
            "reference_depth_m": depth,
            "thumb_index_distance_m": pinch_distance,
        })
    if not projected_samples:
        raise AnchorError("no valid metric PICO pinch samples project into the reference camera")

    task_object_roles = set(
        task_config.get("role_groups", {}).get("task_object", {}).get("setup_roles", [])
    )
    contact_config = dict(contact_binding_config or {})
    contact_distance_m_max = float(contact_config.get("thumb_index_distance_m_max", 0.085))
    contact_reference_px_max = float(contact_config.get("reference_distance_px_max", 60.0))
    release_distance_m_max = float(contact_config.get("active_pinch_distance_m_max", 0.10))
    active_reference_px_max = float(
        contact_config.get("active_reference_distance_px_max", contact_reference_px_max)
    )
    maximum_gap_frames = int(contact_config.get("active_interval_max_gap_frames", 2))
    if not 0 < contact_distance_m_max <= release_distance_m_max:
        raise AnchorError("contact/release pinch thresholds are invalid")
    if contact_reference_px_max <= 0 or active_reference_px_max <= 0 or maximum_gap_frames < 0:
        raise AnchorError("contact binding geometry thresholds are invalid")

    anchors: dict[str, list[float]] = {}
    rows: dict[str, Any] = {}
    contact_bindings: dict[str, Any] = {}
    for role, normalized in task_config["setup_geometry_normalized_xy"].items():
        setup_xy = np.asarray([float(normalized[0]) * width, float(normalized[1]) * height])

        def score(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
            distance = float(np.linalg.norm(np.asarray(row["reference_xy"]) - setup_xy))
            open_penalty = open_pinch_penalty_px_per_m * max(
                float(row["thumb_index_distance_m"]) - closed_pinch_reference_m,
                0.0,
            )
            return (
                distance + open_penalty,
                distance,
                int(row.get("source_frame", 10**9)),
                str(row.get("side", "")),
            )

        contact_candidates = [
            row for row in projected_samples
            if float(row["thumb_index_distance_m"]) <= contact_distance_m_max
            and float(np.linalg.norm(np.asarray(row["reference_xy"]) - setup_xy)) <= contact_reference_px_max
        ]
        if role in task_object_roles and not contact_candidates:
            raise AnchorError(f"no closed-pinch contact binding near task-object setup role {role}")
        # A movable setup role is lifted with the same closed-pinch sample that
        # binds its initial identity.  Static fixtures retain the broader depth
        # proxy rule because they never inherit a hand/object identity claim.
        chosen = min(contact_candidates if role in task_object_roles else projected_samples, key=score)
        _, reference_distance, _, _ = score(chosen)
        if reference_distance > reference_match_distance_px_max:
            raise AnchorError(
                f"no PICO metric depth evidence near setup role {role}: "
                f"{reference_distance:.2f}px > {reference_match_distance_px_max:.2f}px"
            )
        ray = inverse_k @ np.asarray([setup_xy[0], setup_xy[1], 1.0])
        point_reference = ray * (float(chosen["reference_depth_m"]) / float(ray[2]))
        point_world = c2w @ np.concatenate((point_reference, [1.0]))
        anchors[str(role)] = point_world[:3].astype(float).tolist()
        rows[str(role)] = {
            "setup_reference_xy": setup_xy.astype(float).tolist(),
            "depth_proxy_source_frame": int(chosen.get("source_frame", -1)),
            "depth_proxy_side": str(chosen.get("side", "unknown")),
            "depth_proxy_thumb_index_distance_m": float(chosen["thumb_index_distance_m"]),
            "depth_proxy_projected_reference_xy": np.asarray(chosen["reference_xy"]).astype(float).tolist(),
            "depth_proxy_reference_distance_px": reference_distance,
            "reference_depth_m": float(chosen["reference_depth_m"]),
            "derived_world_xyz": anchors[str(role)],
        }
        if role in task_object_roles:
            chosen_frame = int(chosen.get("source_frame", -1))
            chosen_side = str(chosen.get("side", ""))
            side_samples = sorted(
                (
                    row for row in projected_samples
                    if str(row.get("side", "")) == chosen_side
                    and float(row["thumb_index_distance_m"]) <= release_distance_m_max
                    and float(np.linalg.norm(np.asarray(row["reference_xy"]) - setup_xy))
                    <= active_reference_px_max
                ),
                key=lambda row: int(row.get("source_frame", -1)),
            )
            qualifying_frames = [int(row.get("source_frame", -1)) for row in side_samples]
            if chosen_frame not in qualifying_frames:
                raise AnchorError(f"contact frame for {role} is outside its active pinch interval")
            chosen_position = qualifying_frames.index(chosen_frame)
            start_position = chosen_position
            while (
                start_position > 0
                and qualifying_frames[start_position] - qualifying_frames[start_position - 1]
                <= maximum_gap_frames + 1
            ):
                start_position -= 1
            end_position = chosen_position
            while (
                end_position + 1 < len(qualifying_frames)
                and qualifying_frames[end_position + 1] - qualifying_frames[end_position]
                <= maximum_gap_frames + 1
            ):
                end_position += 1
            contact_bindings[str(role)] = {
                "role": str(role),
                "provider_id": "PICO21",
                "side": chosen_side,
                "contact_frame": chosen_frame,
                "active_start_frame": qualifying_frames[start_position],
                "active_end_frame": qualifying_frames[end_position],
                "contact_thumb_index_distance_m": float(chosen["thumb_index_distance_m"]),
                "contact_projected_reference_xy": np.asarray(chosen["reference_xy"]).astype(float).tolist(),
                "contact_reference_distance_px": reference_distance,
                "static_setup_valid_through_frame": chosen_frame,
                "static_setup_forbidden_from_frame": chosen_frame + 1,
                "moving_seed_authority": "bound PICO thumb-index midpoint only during active pinch interval",
                "post_release_authority": "UNRESOLVED_REQUIRES_INSTANCE_PROPAGATION_OR_OBJECT6D",
            }
    return anchors, {
        "authority": "PICO keypoints_3d_world plus per-frame c2w/K; RGB not inspected",
        "sample_count": len(projected_samples),
        "reference_frame": int(reference_metadata.get("idx", 0)),
        "open_pinch_penalty_px_per_m": float(open_pinch_penalty_px_per_m),
        "closed_pinch_reference_m": float(closed_pinch_reference_m),
        "reference_match_distance_px_max": float(reference_match_distance_px_max),
        "roles": rows,
        "contact_bindings": contact_bindings,
        "setup_anchor_scope": {
            "fixture": "SESSION_STATIC_REPROJECTABLE_ALL_FRAMES",
            "task_object": "INITIAL_SLOT_ONLY_THROUGH_BOUND_CONTACT_FRAME",
            "post_contact": "STATIC_SETUP_FORBIDDEN; independent same-instance propagation or Object6D required",
        },
        "contact_binding_thresholds": {
            "thumb_index_distance_m_max": contact_distance_m_max,
            "reference_distance_px_max": contact_reference_px_max,
            "active_pinch_distance_m_max": release_distance_m_max,
            "active_reference_distance_px_max": active_reference_px_max,
            "active_interval_max_gap_frames": maximum_gap_frames,
        },
    }


def object_identity_requirements(
    task_config: Mapping[str, Any],
    contact_bindings: Mapping[str, Mapping[str, Any]],
    source_frame: int,
) -> dict[str, Any]:
    """Return the only legal evidence mode for each movable setup instance.

    Once contact has happened, a setup-world coordinate is historical scene
    geometry, not evidence for the moving object.  A frame outside the bound
    pinch interval therefore remains unresolved until an independent instance
    propagation/flow result or Object6D pose is supplied.
    """
    task_roles = list(task_config["role_groups"]["task_object"]["setup_roles"])
    if set(contact_bindings) != set(task_roles):
        raise AnchorError("contact binding inventory differs from task-object setup roles")
    required: dict[str, str] = {}
    unresolved: list[str] = []
    for role in task_roles:
        binding = contact_bindings[role]
        contact = int(binding["contact_frame"])
        active_start = int(binding["active_start_frame"])
        active_end = int(binding["active_end_frame"])
        if source_frame <= contact:
            required[role] = "INITIAL_SLOT_STATIC_PRECONTACT"
        elif active_start <= source_frame <= active_end:
            required[role] = "CONTACT_BOUND_MOVING_INSTANCE"
        else:
            unresolved.append(role)
    active_by_side: dict[str, list[str]] = {"left": [], "right": []}
    for role, mode in required.items():
        if mode == "CONTACT_BOUND_MOVING_INSTANCE":
            active_by_side[str(contact_bindings[role]["side"])].append(role)
    ambiguous = {
        side: sorted(roles) for side, roles in active_by_side.items() if len(roles) > 1
    }
    return {
        "source_frame": int(source_frame),
        "required_mode_by_role": required,
        "unresolved_moved_roles": unresolved,
        "ambiguous_active_bindings_by_side": ambiguous,
        "post_contact_static_anchor_forbidden": True,
        "pass_without_external_object_authority": not unresolved and not ambiguous,
    }


def project_setup_world_anchors(
    anchors_world: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Project frozen per-session setup world anchors into one PICO frame."""
    points: dict[str, list[float]] = {}
    depths: dict[str, float] = {}
    for role, world in anchors_world.items():
        pixel, depth = project_world_point(world, metadata.get("c2w"), metadata.get("k"))
        points[str(role)] = pixel.astype(float).tolist()
        depths[str(role)] = depth
    return points, {
        "authority": "PICO frame c2w/K metric reprojection",
        "source_frame": int(metadata.get("idx", -1)),
        "depth_m_by_role": depths,
        "rgb_used": False,
    }


def task_object_candidate_points(
    task_config: Mapping[str, Any],
    pico_hands: Mapping[str, Mapping[str, Any]],
    width: int,
    height: int,
    search_offsets_px: Sequence[Sequence[float]],
    reference_to_frame: Sequence[Sequence[float]] | None = None,
    setup_points_xy: Mapping[str, Sequence[float]] | None = None,
    margin_px: int = 2,
) -> list[dict[str, Any]]:
    """Create static-setup and dynamic pinch candidates without image cues."""
    rows: list[dict[str, Any]] = []
    expected_roles = set(task_config["setup_geometry_normalized_xy"])
    if setup_points_xy is not None and set(setup_points_xy) != expected_roles:
        raise AnchorError("projected setup role inventory differs from task configuration")
    for role, normalized in task_config["setup_geometry_normalized_xy"].items():
        reference = np.asarray([float(normalized[0]) * width, float(normalized[1]) * height])
        center = (
            np.asarray(setup_points_xy[role], dtype=np.float64)
            if setup_points_xy is not None
            else apply_homography(reference, reference_to_frame)
        )
        for offset in search_offsets_px:
            rows.append({"source": f"setup:{role}", "xy": center + np.asarray(offset, dtype=np.float64)})
    for side in ("left", "right"):
        hand = pico_hands[side]
        points = np.asarray(hand["keypoints_2d"], dtype=np.float64)
        valid = np.asarray(hand["joint_valid"], dtype=bool) & np.asarray(hand["joint_in_image"], dtype=bool)
        if len(points) != 21 or len(valid) != 21:
            raise AnchorError("PICO21 geometry required")
        names = list(hand.get("joint_names", []))
        if names:
            if len(names) != 21 or "thumb_fingertip" not in names or "index_fingertip" not in names:
                raise AnchorError("PICO21 joint names missing fingertip authority")
            thumb_index = names.index("thumb_fingertip")
            index_index = names.index("index_fingertip")
        else:
            thumb_index, index_index = 0, 1
        if valid[thumb_index] and valid[index_index]:
            rows.append({
                "source": f"pico:{side}:thumb_index_pinch_midpoint",
                "xy": (points[thumb_index] + points[index_index]) / 2.0,
            })
    unique = clipped_unique([row["xy"] for row in rows], width, height, margin_px)
    source_by_xy: dict[tuple[int, int], str] = {}
    for row in rows:
        rounded = tuple(map(int, np.rint(row["xy"])))
        source_by_xy.setdefault(rounded, str(row["source"]))
    return [{"xy": point, "source": source_by_xy[tuple(point)]} for point in unique]


def authorize_object_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    task_config: Mapping[str, Any],
    contact_bindings: Mapping[str, Mapping[str, Any]],
    source_frame: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the static-vs-moving authority boundary before model prompting."""
    requirements = object_identity_requirements(task_config, contact_bindings, source_frame)
    required_modes = requirements["required_mode_by_role"]
    task_roles = set(task_config["role_groups"]["task_object"]["setup_roles"])
    fixture_roles = set(task_config["role_groups"]["fixture"]["setup_roles"])
    authorized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        source = str(row["source"])
        if source.startswith("setup:"):
            role = source.split(":", 1)[1]
            if role in fixture_roles:
                authorized.append({
                    **row,
                    "identity_mode": "SESSION_STATIC_FIXTURE",
                    "identity_roles": [role],
                    "authority_scope": "all_frames",
                })
            elif role in task_roles and required_modes.get(role) == "INITIAL_SLOT_STATIC_PRECONTACT":
                authorized.append({
                    **row,
                    "identity_mode": "INITIAL_SLOT_STATIC_PRECONTACT",
                    "identity_roles": [role],
                    "authority_scope": f"through_frame_{int(contact_bindings[role]['contact_frame'])}",
                })
            else:
                rejected.append({
                    "source": source,
                    "reason": "STATIC_TASK_ANCHOR_FORBIDDEN_AFTER_BOUND_CONTACT",
                })
            continue
        if source.startswith("pico:"):
            parts = source.split(":")
            if len(parts) < 2 or parts[1] not in {"left", "right"}:
                raise AnchorError(f"malformed PICO candidate source: {source}")
            side = parts[1]
            bound_roles = [
                role for role, mode in required_modes.items()
                if mode == "CONTACT_BOUND_MOVING_INSTANCE"
                and str(contact_bindings[role]["side"]) == side
            ]
            if len(bound_roles) == 1:
                authorized.append({
                    **row,
                    "source": f"{source}:bindings={','.join(sorted(bound_roles))}",
                    "identity_mode": "CONTACT_BOUND_MOVING_INSTANCE",
                    "identity_roles": sorted(bound_roles),
                    "authority_scope": "bound_active_pinch_interval_only",
                })
            else:
                rejected.append({
                    "source": source,
                    "reason": (
                        "AMBIGUOUS_MULTIPLE_ACTIVE_BINDINGS_FOR_SIDE"
                        if len(bound_roles) > 1 else "NO_ACTIVE_CONTACT_BINDING_FOR_SIDE"
                    ),
                })
            continue
        raise AnchorError(f"unknown task candidate source: {source}")
    return authorized, {**requirements, "rejected_candidates": rejected}


def pico_negative_points(
    own_hand: Mapping[str, Any],
    opposing_hand: Mapping[str, Any],
    width: int,
    height: int,
    task_points: Sequence[Sequence[float]],
    margin_px: int = 2,
) -> list[list[int]]:
    own_points = np.asarray(own_hand["keypoints_2d"], dtype=np.float64)
    own_valid = np.asarray(own_hand["joint_valid"], bool) & np.asarray(own_hand["joint_in_image"], bool)
    opposing_points = np.asarray(opposing_hand["keypoints_2d"], dtype=np.float64)
    opposing_valid = np.asarray(opposing_hand["joint_valid"], bool) & np.asarray(opposing_hand["joint_in_image"], bool)
    points = [point for index, point in enumerate(own_points) if own_valid[index]]
    points.extend(
        opposing_points[index] for index in (5, 20) if opposing_valid[index]
    )
    points.extend(np.asarray(point, dtype=np.float64) for point in task_points)
    return clipped_unique(points, width, height, margin_px)


def raw_candidate_rank(record: Mapping[str, Any], hard_gate_names: Sequence[str]) -> tuple[Any, ...]:
    """Rank only hard-gate-clean candidates; lower tuples are better."""
    gates = record.get("gates", {})
    if any(gates.get(name) is not True for name in hard_gate_names):
        return (1, int(record.get("candidate_id", 10**9)))
    return (
        0,
        int(record.get("pico_negative_points_inside", 10**9)),
        float(record.get("own_human_min_distance_px", 1e12)),
        float(record.get("centroid_to_own_wrist_px", 1e12)),
        -float(record.get("largest_component_area_fraction", -1)),
        int(record.get("protected_object_overlap_pixels", 10**9)),
        int(record.get("candidate_id", 10**9)),
    )


def choose_raw_candidate(
    records: Sequence[Mapping[str, Any]], hard_gate_names: Sequence[str]
) -> Mapping[str, Any] | None:
    eligible = [record for record in records if raw_candidate_rank(record, hard_gate_names)[0] == 0]
    return None if not eligible else min(eligible, key=lambda row: raw_candidate_rank(row, hard_gate_names))


def _point_inside(mask: np.ndarray, point_xy: Sequence[float]) -> bool:
    x, y = map(int, np.rint(point_xy))
    return bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    y, x = np.where(mask)
    return None if not len(x) else np.asarray([x.mean(), y.mean()], dtype=np.float64)


def _point_distance(mask: np.ndarray, point_xy: Sequence[float]) -> float:
    y, x = np.where(mask)
    if not len(x):
        return float("inf")
    point = np.asarray(point_xy, dtype=np.float64)
    return float(np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2).min())


def _mask_distance(left: np.ndarray, right: np.ndarray) -> float:
    if not left.any() or not right.any():
        return float("inf")
    distance = cv2.distanceTransform((~right).astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[left].min())


def _components(mask: np.ndarray) -> tuple[int, float]:
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    significant = areas[areas >= 64]
    largest_fraction = float(areas.max() / max(int(areas.sum()), 1)) if len(areas) else 0.0
    return int(len(significant)), largest_fraction


def tracker_candidate_record(
    mask: np.ndarray,
    candidate_id: int,
    candidate_xy: Sequence[float],
    negative_points: Sequence[Sequence[float]],
    own_human: np.ndarray,
    opposing_human: np.ndarray,
    own_wrist_xy: Sequence[float],
    opposing_wrist_xy: Sequence[float],
    protected_objects: np.ndarray,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure one untouched raw tracker proposal against frozen hard gates."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if any(value.shape != mask.shape for value in (own_human, opposing_human, protected_objects)):
        raise AnchorError("candidate and authority masks must have identical geometry")
    center = _centroid(mask)
    own_centroid = float("inf") if center is None else float(np.linalg.norm(center - own_wrist_xy))
    opposing_centroid = float("inf") if center is None else float(np.linalg.norm(center - opposing_wrist_xy))
    component_count, largest_fraction = _components(mask)
    edge = np.zeros_like(mask)
    edge[:3] = edge[-3:] = True
    edge[:, :3] = edge[:, -3:] = True
    own_near_edge = cv2.dilate(own_human.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    negative_inside = sum(_point_inside(mask, point) for point in negative_points)
    area_fraction = float(mask.mean())
    own_distance = _mask_distance(mask, own_human)
    object_overlap = int(np.logical_and(mask, protected_objects).sum())
    edge_leak = int(np.logical_and(mask, edge & ~own_near_edge).sum())
    side_margin = opposing_centroid - own_centroid
    metrics = {
        "candidate_id": int(candidate_id),
        "candidate_xy": list(map(int, np.rint(candidate_xy))),
        "present": bool(mask.any()),
        "positive_candidate_point_inside": _point_inside(mask, candidate_xy),
        "pico_negative_points_inside": int(negative_inside),
        "own_human_min_distance_px": own_distance,
        "centroid_to_own_wrist_px": own_centroid,
        "opposing_minus_own_wrist_centroid_px": side_margin,
        "frame_area_fraction": area_fraction,
        "connected_components_over_64px": component_count,
        "largest_component_area_fraction": largest_fraction,
        "protected_object_overlap_pixels": object_overlap,
        "non_wrist_authorized_image_edge_contact_pixels": edge_leak,
    }
    metrics["gates"] = {
        "present": metrics["present"],
        "positive_candidate_point_inside": metrics["positive_candidate_point_inside"]
        == bool(gate["positive_candidate_point_inside"]),
        "valid_pico_finger_or_palm_points_inside": negative_inside
        <= int(gate["valid_pico_finger_or_palm_points_inside_max"]),
        "own_human_min_distance": own_distance <= float(gate["own_human_min_distance_px_max"]),
        "centroid_to_own_wrist": own_centroid <= float(gate["centroid_to_own_pico_wrist_px_max"]),
        "side_identity_margin": side_margin >= float(gate["opposing_minus_own_wrist_centroid_px_min"]),
        "area": float(gate["frame_area_fraction_min"]) <= area_fraction <= float(gate["frame_area_fraction_max"]),
        "components": component_count <= int(gate["connected_components_over_64px_max"]),
        "largest_component": largest_fraction >= float(gate["largest_component_area_fraction_min"]),
        "object_overlap": object_overlap <= int(gate["protected_object_overlap_pixels_max"]),
        "background_edge_leak": edge_leak <= int(gate["non_wrist_authorized_image_edge_contact_pixels_max"]),
    }
    return metrics


def object_candidate_record(
    mask: np.ndarray,
    candidate_id: int,
    candidate_xy: Sequence[float],
    pico_joint_points: Sequence[Sequence[float]],
    role_limits: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure one untouched task-object/fixture proposal without RGB cues."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    component_count, largest_fraction = _components(mask)
    edge = np.zeros_like(mask)
    edge[:3] = edge[-3:] = True
    edge[:, :3] = edge[:, -3:] = True
    pico_inside = sum(_point_inside(mask, point) for point in pico_joint_points)
    area_fraction = float(mask.mean())
    edge_pixels = int(np.logical_and(mask, edge).sum())
    record = {
        "candidate_id": int(candidate_id),
        "candidate_xy": list(map(int, np.rint(candidate_xy))),
        "present": bool(mask.any()),
        "positive_candidate_point_inside": _point_inside(mask, candidate_xy),
        "pico_joint_points_inside": int(pico_inside),
        "frame_area_fraction": area_fraction,
        "connected_components_over_64px": component_count,
        "largest_component_area_fraction": largest_fraction,
        "image_edge_contact_pixels": edge_pixels,
    }
    record["gates"] = {
        "present": record["present"],
        "positive_candidate_point_inside": record["positive_candidate_point_inside"]
        == bool(gate["positive_candidate_point_inside"]),
        "pico_joint_points": pico_inside <= int(gate["valid_pico_joint_points_inside_max"]),
        "area": float(role_limits["frame_area_fraction_min"])
        <= area_fraction
        <= float(role_limits["frame_area_fraction_max"]),
        "components": component_count <= int(gate["connected_components_over_64px_max"]),
        "largest_component": largest_fraction >= float(gate["largest_component_area_fraction_min"]),
        "image_edge": edge_pixels <= int(gate["image_edge_contact_pixels_max"]),
    }
    return record


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return 1.0 if not union else float(np.logical_and(left, right).sum() / union)


def select_distinct_object_candidates(
    records_and_masks: Sequence[tuple[Mapping[str, Any], np.ndarray]],
    maximum_instances: int,
    duplicate_iou_max: float,
) -> list[tuple[Mapping[str, Any], np.ndarray]]:
    """Select only hard-gate-clean, nonduplicate raw object masks."""
    selected: list[tuple[Mapping[str, Any], np.ndarray]] = []
    for record, mask in sorted(records_and_masks, key=lambda item: int(item[0]["candidate_id"])):
        if not all(record.get("gates", {}).values()):
            continue
        if any(binary_iou(mask, other) > duplicate_iou_max for _, other in selected):
            continue
        selected.append((record, mask))
        if len(selected) == maximum_instances:
            break
    return selected
