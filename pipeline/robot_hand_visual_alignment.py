"""Chirality-safe Human MANO21 to KaiHand visual alignment measurements.

The KaiHand base and distal CAD axes are not interchangeable with an
unsigned landmark cross product.  In particular, the palmar-facing base mesh
normal is local ``-Y`` for the physical left hand and local ``+Y`` for the
physical right hand.  These helpers keep that signed CAD fact explicit and
measure the thumb--index web (the visible ``虎口``) in addition to endpoint
directions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline import robot_scene_state_cpu as official
from pipeline.robot_wrist_kai_adapter import final_v3_mano_palm_basis


MANO_TIPS: tuple[int, ...] = (4, 8, 12, 16, 20)
KAI_TIPS: tuple[int, ...] = (5, 6, 7, 8, 9)
FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


class HandVisualAlignmentError(ValueError):
    """Raised when a hand visual metric cannot be computed unambiguously."""


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or length <= 1e-9:
        raise HandVisualAlignmentError(f"{name} must be one finite nonzero 3-vector")
    return value / length


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned angle between two finite nonzero vectors."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise HandVisualAlignmentError("angle vectors must share one finite 1-D shape")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-9 or b_norm <= 1e-9:
        raise HandVisualAlignmentError("zero vector has no direction")
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / (a_norm * b_norm), -1.0, 1.0))))


def kai_palmar_axis_local(physical_side: str) -> np.ndarray:
    """Return the signed Kai base-link palmar direction in pinned CAD axes.

    The sign is the same one used by :mod:`pipeline.robot_contact_geometry`
    when selecting the physical palm/finger-pad faces.
    """

    if physical_side == "left":
        return np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
    if physical_side == "right":
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    raise HandVisualAlignmentError("physical_side must be 'left' or 'right'")


def human_mano_palmar_normal(points_camera: np.ndarray, *, human_side: str) -> np.ndarray:
    """Return the signed MANO palmar normal from identity-bearing MANO21.

    ``final_v3_mano_palm_basis`` has a chirality-normalized dorsal third axis;
    palmar is its negative.  The sign is therefore comparable across hands.
    """

    points = np.asarray(points_camera, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise HandVisualAlignmentError("one finite MANO21 camera array is required")
    return -final_v3_mano_palm_basis(points, handedness=human_side)[:, 2]


def kai_visual_landmarks(
    model: Any,
    joint_names: tuple[str, ...],
    q: np.ndarray,
    *,
    physical_side: str,
) -> np.ndarray:
    """Return wrist, four palm roots and five true CAD terminal points.

    Output order is ``wrist, thumb_root, index_root, middle_root, pinky_root,
    thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip``.  Distal offsets
    exactly match the pinned ``retargeting_human.KaiHandRetargeter`` tracking
    definition; using the link-frame origins as fingertips is forbidden.
    """

    if physical_side not in {"left", "right"}:
        raise HandVisualAlignmentError("physical_side must be 'left' or 'right'")
    values = np.asarray(q, dtype=np.float64)
    if values.shape != (len(joint_names),) or not np.isfinite(values).all():
        raise HandVisualAlignmentError("q and joint_names must describe one finite hand state")
    prefix = "hand_l" if physical_side == "left" else "hand_r"
    fk = official.forward_kinematics(model, dict(zip(joint_names, values, strict=True)))

    def origin(link: str) -> np.ndarray:
        if link not in fk:
            raise HandVisualAlignmentError(f"Kai FK is missing {link}")
        return np.asarray(fk[link][:3, 3], dtype=np.float64)

    output = [
        np.zeros(3, dtype=np.float64),
        origin(f"{prefix}_thumb_link1"),
        origin(f"{prefix}_index_link1"),
        origin(f"{prefix}_middle_link1"),
        origin(f"{prefix}_pinky_link1"),
    ]
    thumb = fk[f"{prefix}_thumb_link6"]
    output.append(thumb[:3, 3] + thumb[:3, :3] @ np.asarray((0.02731, 0.0, -0.0028)))
    lateral = 0.00045 if physical_side == "left" else -0.00045
    for finger, distal_x in zip(
        ("index", "middle", "ring", "pinky"),
        (-0.0001, -0.0003, -0.0005, -0.0007),
        strict=True,
    ):
        transform = fk[f"{prefix}_{finger}_link4"]
        offset = np.asarray((distal_x, lateral, -0.02465), dtype=np.float64)
        output.append(transform[:3, 3] + transform[:3, :3] @ offset)
    result = np.asarray(output, dtype=np.float64)
    if result.shape != (10, 3) or not np.isfinite(result).all():
        raise HandVisualAlignmentError("Kai visual landmark construction failed")
    return result


def project_camera(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project positive-depth camera points through one exact 3x3 K."""

    points = np.asarray(points_camera, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or k.shape != (3, 3):
        raise HandVisualAlignmentError("projection requires points (N,3) and K (3,3)")
    if not np.isfinite(points).all() or not np.isfinite(k).all() or np.any(points[:, 2] <= 1e-5):
        raise HandVisualAlignmentError("projection inputs must be finite and in front of camera")
    return np.stack(
        (
            k[0, 0] * points[:, 0] / points[:, 2] + k[0, 2],
            k[1, 1] * points[:, 1] / points[:, 2] + k[1, 2],
        ),
        axis=1,
    )


def measure_alignment(
    kai_points_camera: np.ndarray,
    human_mano_camera: np.ndarray,
    human_mano_uv: np.ndarray,
    intrinsics: np.ndarray,
    *,
    physical_side: str,
    human_side: str,
    other_wrist_camera: np.ndarray,
    kai_root_rotation_camera: np.ndarray,
) -> dict[str, object]:
    """Measure signed palm, thumb, five fingers and thumb--index web.

    This is a metric, not an acceptance policy.  Callers must state thresholds
    and may add an explicit task/user direction constraint separately.
    """

    kai = np.asarray(kai_points_camera, dtype=np.float64)
    human = np.asarray(human_mano_camera, dtype=np.float64)
    human_uv = np.asarray(human_mano_uv, dtype=np.float64)
    rotation = np.asarray(kai_root_rotation_camera, dtype=np.float64)
    other = np.asarray(other_wrist_camera, dtype=np.float64)
    if kai.shape != (10, 3) or human.shape != (21, 3) or human_uv.shape != (21, 2):
        raise HandVisualAlignmentError("Kai10, MANO21 camera and MANO21 image arrays are required")
    if rotation.shape != (3, 3) or other.shape != (3,):
        raise HandVisualAlignmentError("root rotation and other wrist shapes are invalid")
    if not all(np.isfinite(value).all() for value in (kai, human, human_uv, rotation, other)):
        raise HandVisualAlignmentError("alignment input contains NaN/Inf")
    kai_uv = project_camera(kai, intrinsics)
    palmar = _unit(rotation @ kai_palmar_axis_local(physical_side), name="Kai palmar normal")
    human_palmar = _unit(human_mano_palmar_normal(human, human_side=human_side), name="MANO palmar normal")
    toward_other = _unit(other - kai[0], name="other-hand direction")

    finger_errors = [
        angle_deg(kai_uv[kai_index] - kai_uv[0], human_uv[mano_index] - human_uv[0])
        for kai_index, mano_index in zip(KAI_TIPS, MANO_TIPS, strict=True)
    ]
    length_ratios = [
        float(np.linalg.norm(kai_uv[kai_index] - kai_uv[0]) / max(np.linalg.norm(human_uv[mano_index] - human_uv[0]), 1e-9))
        for kai_index, mano_index in zip(KAI_TIPS, MANO_TIPS, strict=True)
    ]
    kai_web_angle = angle_deg(kai_uv[5] - kai_uv[0], kai_uv[6] - kai_uv[0])
    human_web_angle = angle_deg(human_uv[4] - human_uv[0], human_uv[8] - human_uv[0])
    kai_gap = float(np.linalg.norm(kai_uv[5] - kai_uv[6]) / max(np.linalg.norm(kai_uv[7] - kai_uv[0]), 1e-9))
    human_gap = float(np.linalg.norm(human_uv[4] - human_uv[8]) / max(np.linalg.norm(human_uv[12] - human_uv[0]), 1e-9))
    kai_mcp_web = angle_deg(kai_uv[1] - kai_uv[0], kai_uv[2] - kai_uv[0])
    human_mcp_web = angle_deg(human_uv[2] - human_uv[0], human_uv[5] - human_uv[0])

    # Isotropic normalized overlay error: translation and morphology scale do
    # not decide the gate, but relative projected layout does.
    kai_vectors = kai_uv[np.asarray(KAI_TIPS)] - kai_uv[0]
    human_vectors = human_uv[np.asarray(MANO_TIPS)] - human_uv[0]
    scale = float(np.sum(kai_vectors * human_vectors) / max(np.sum(kai_vectors * kai_vectors), 1e-9))
    human_reference = max(float(np.mean(np.linalg.norm(human_vectors, axis=1))), 1e-9)
    overlay_nrmse = float(np.sqrt(np.mean(np.sum((scale * kai_vectors - human_vectors) ** 2, axis=1))) / human_reference)

    return {
        "physical_side": physical_side,
        "human_side": human_side,
        "kai_palmar_normal_camera": palmar.tolist(),
        "human_palmar_normal_camera": human_palmar.tolist(),
        "palmar_normal_angle_deg": angle_deg(palmar, human_palmar),
        "palmar_toward_other_dot": float(palmar @ toward_other),
        "thumb_tip_minus_wrist_image_v_px": float(kai_uv[5, 1] - kai_uv[0, 1]),
        "thumb_direction_angle_deg": float(finger_errors[0]),
        "finger_direction_angle_deg": [float(value) for value in finger_errors],
        "finger_direction_angle_deg_mean": float(np.mean(finger_errors)),
        "projected_length_ratio": length_ratios,
        "projected_length_ratio_mean": float(np.mean(length_ratios)),
        "thumb_index_web_angle_deg": {"kai": kai_web_angle, "human": human_web_angle, "abs_error": abs(kai_web_angle - human_web_angle)},
        "thumb_index_mcp_web_angle_deg": {"kai": kai_mcp_web, "human": human_mcp_web, "abs_error": abs(kai_mcp_web - human_mcp_web)},
        "thumb_index_gap_over_middle_length": {"kai": kai_gap, "human": human_gap, "abs_error": abs(kai_gap - human_gap)},
        "five_tip_isotropic_overlay_nrmse": overlay_nrmse,
    }
