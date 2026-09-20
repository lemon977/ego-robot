"""Reconstruct the historical Kai22 pre/post-clip trajectory exactly.

The historical W0 producer retained the source MANO21 trajectory and the
post-clip q22 but not the pre-clip desired q.  Its run signature pins the
deterministic mapping implementation.  This module materialises that omitted
intermediate without changing the mapping or its URDF limits and requires the
reconstructed post-clip q to match the frozen q22 byte-for-byte.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from chaoyang.pipeline.robot_visual_relative_v1 import (
    HAND_GROUPS,
    HandLimits,
    mano_flexion,
)


SCHEMA_VERSION = "KAI22_CLIP_RECONSTRUCTION_V1"
SIDES = ("left", "right")


class Kai22ClipReconstructionError(ValueError):
    """Raised when the historical clip intermediate cannot be reconstructed."""


def _desired_q22(points: np.ndarray, limits: HandLimits) -> np.ndarray:
    flexion = mano_flexion(points)
    desired_q = limits.neutral.copy()
    for finger, group in enumerate(HAND_GROUPS):
        indices = np.asarray(group, dtype=np.int64)
        if finger == 0:
            desired = np.asarray(
                [
                    desired_q[indices[0]],
                    flexion[finger, 0],
                    flexion[finger, 1],
                    0.5 * flexion[finger, 2],
                    flexion[finger, 2],
                    flexion[finger, 2],
                ],
                dtype=np.float64,
            )
        else:
            desired = np.asarray(
                [
                    desired_q[indices[0]],
                    flexion[finger, 0],
                    flexion[finger, 1],
                    flexion[finger, 2],
                ],
                dtype=np.float64,
            )
        desired_q[indices] = desired
    return desired_q


def reconstruct_kai22_clip_delta(
    *,
    joints_3d_camera: np.ndarray,
    source_observed_anatomical: np.ndarray,
    q22_postclip: np.ndarray,
    q22_valid_physical: np.ndarray,
    human_to_physical: np.ndarray,
    limits: Sequence[HandLimits],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Rebuild desired q, post-clip q, and signed ``post - desired`` delta."""

    joints = np.asarray(joints_3d_camera, dtype=np.float64)
    observed = np.asarray(source_observed_anatomical, dtype=bool)
    frozen_q = np.asarray(q22_postclip, dtype=np.float64)
    valid = np.asarray(q22_valid_physical, dtype=bool)
    mapping = np.asarray(human_to_physical, dtype=np.int64)
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise Kai22ClipReconstructionError("joints_3d_camera must be [2,T,21,3]")
    frames = joints.shape[1]
    if observed.shape != (frames, 2):
        raise Kai22ClipReconstructionError("source observed axis must be [T,2] anatomical")
    if frozen_q.shape != (frames, 2, 22) or valid.shape != (frames, 2):
        raise Kai22ClipReconstructionError("frozen q22/valid axes are not [T,2,22]/[T,2]")
    if not np.array_equal(np.sort(mapping), np.asarray([0, 1])):
        raise Kai22ClipReconstructionError("human_to_physical must be an exact side bijection")
    if len(limits) != 2:
        raise Kai22ClipReconstructionError("two physical-side limit sets are required")

    expected_valid = np.zeros_like(valid)
    for anatomical_side, physical_side in enumerate(mapping.tolist()):
        expected_valid[:, physical_side] = (
            observed[:, anatomical_side]
            & np.isfinite(joints[anatomical_side]).all(axis=(1, 2))
        )
    if not np.array_equal(valid, expected_valid):
        raise Kai22ClipReconstructionError(
            "frozen q22 validity differs from direct finite anatomical observations"
        )
    if np.any(valid & ~np.isfinite(frozen_q).all(axis=2)):
        raise Kai22ClipReconstructionError("valid frozen q22 contains non-finite values")

    desired_q = np.full_like(frozen_q, np.nan)
    reconstructed_postclip = np.full_like(frozen_q, np.nan)
    for anatomical_side, physical_side in enumerate(mapping.tolist()):
        side_limits = limits[physical_side]
        if (
            side_limits.lower.shape != (22,)
            or side_limits.upper.shape != (22,)
            or side_limits.neutral.shape != (22,)
        ):
            raise Kai22ClipReconstructionError("KaiHand limits must be exact 22-vectors")
        for frame in np.flatnonzero(valid[:, physical_side]):
            desired = _desired_q22(joints[anatomical_side, frame], side_limits)
            desired_q[frame, physical_side] = desired
            reconstructed_postclip[frame, physical_side] = np.clip(
                desired, side_limits.lower, side_limits.upper
            )

    expanded_valid = np.broadcast_to(valid[..., None], frozen_q.shape)
    if not np.array_equal(
        reconstructed_postclip[expanded_valid], frozen_q[expanded_valid]
    ):
        difference = np.abs(reconstructed_postclip - frozen_q)
        finite = difference[np.isfinite(difference)]
        raise Kai22ClipReconstructionError(
            "reconstructed post-clip q differs from frozen q22; "
            f"max_abs={float(finite.max()) if finite.size else None}"
        )
    invalid = ~expanded_valid
    if not np.isnan(frozen_q[invalid]).all():
        raise Kai22ClipReconstructionError("invalid frozen q22 entries are not explicit NaN")

    clip_delta = reconstructed_postclip - desired_q
    clipped_joint = expanded_valid & (np.abs(clip_delta) > 1e-12)
    clipped_side_frame = np.any(clipped_joint, axis=2)
    side_rows = []
    for physical_side, physical_name in enumerate(SIDES):
        anatomical_side = int(np.flatnonzero(mapping == physical_side)[0])
        selected = valid[:, physical_side]
        side_rows.append(
            {
                "physical_robot_side": physical_name,
                "anatomical_side": SIDES[anatomical_side],
                "valid_frames": int(selected.sum()),
                "clipped_frames": int(clipped_side_frame[:, physical_side].sum()),
                "clipped_joint_values": int(clipped_joint[:, physical_side].sum()),
                "max_abs_clip_delta_rad": (
                    float(np.max(np.abs(clip_delta[:, physical_side][selected])))
                    if selected.any()
                    else None
                ),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_BYTE_EXACT_POSTCLIP_RECONSTRUCTION",
        "clip_delta_semantics": "POSTCLIP_Q22_MINUS_PRECLIP_DESIRED_Q22",
        "mapping_semantics": (
            "PINNED_HISTORICAL_MANO_FLEXION_DESIRED_Q_THEN_NUMPY_CLIP_URDF_LIMITS"
        ),
        "postclip_matches_frozen_q22_byte_exact": True,
        "validity_matches_direct_finite_source_observations": True,
        "side_mapping": {
            "anatomical_to_physical_index": mapping.tolist(),
            "source_axis": ["anatomical_left", "anatomical_right"],
            "q22_axis": ["kaihand_left", "kaihand_right"],
        },
        "sides": side_rows,
        "valid_side_frames": int(valid.sum()),
        "clipped_side_frames": int(clipped_side_frame.sum()),
        "clipped_joint_values": int(clipped_joint.sum()),
        "zero_clip_delta_all_valid_frames": not bool(clipped_joint.any()),
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    arrays = {
        "q22_preclip_desired": desired_q,
        "q22_reconstructed_postclip": reconstructed_postclip,
        "q22_frozen_postclip": frozen_q,
        "clip_delta": clip_delta,
        "clipped_joint": clipped_joint,
        "clipped_side_frame": clipped_side_frame,
        "q22_valid_physical": valid,
        "source_observed_anatomical": observed,
        "human_to_physical": mapping,
    }
    return summary, arrays
