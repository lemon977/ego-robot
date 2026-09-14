"""Contract and algorithm skeleton for a visual-only constant robot mount proxy.

No fitting or pixels are produced by this module.  It defines the cross-frame
parameterization and the manifest fields that a later authorized CPU-8 fitter
must use.  Physical, collision, reachability, training, and real-robot claims
are deliberately outside this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pipeline.robot_tool_definition_contract import load_pinned_tool_definition


MOUNT_PROVENANCE = "PROVISIONAL_MOUNT_VISUAL_ONLY"
CONTACT_INFEASIBLE = "UNMEASURED"
SIDES = ("left", "right")


class VisualMountProxyError(RuntimeError):
    pass


def validate_se3(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise VisualMountProxyError(f"{name} must be one finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise VisualMountProxyError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise VisualMountProxyError(f"{name} rotation is not orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        raise VisualMountProxyError(f"{name} rotation determinant is not +1")
    return matrix


def validate_session_constant_mounts(mounts: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(mounts) != set(SIDES):
        raise VisualMountProxyError("exactly one left and one right mount are required")
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        value = np.asarray(mounts[side])
        if value.ndim != 2:
            raise VisualMountProxyError(f"{side} mount must be session-constant, not per-frame")
        result[side] = validate_se3(value, name=f"{side}_tool_to_hand_root")
    return result


@dataclass(frozen=True)
class VisualFitInputContract:
    session_id: str
    frame_indices: tuple[int, ...]
    hawor_target_record: Mapping[str, Any]
    camera_intrinsics_record: Mapping[str, Any]
    r2_hand_state_record: Mapping[str, Any] | None = None
    side_axis_record: Mapping[str, Any] | None = None

    def validate(self) -> None:
        lowered = self.session_id.lower()
        if "025" in lowered or "blind" in lowered:
            raise VisualMountProxyError("blind/025 input is forbidden")
        if not self.frame_indices or len(set(self.frame_indices)) != len(self.frame_indices):
            raise VisualMountProxyError("a non-empty unique cross-frame fit set is required")
        if tuple(sorted(self.frame_indices)) != self.frame_indices or self.frame_indices[0] < 0:
            raise VisualMountProxyError("fit frame indices must be sorted non-negative integers")
        for name, record in (
            ("hawor_target", self.hawor_target_record),
            ("camera_intrinsics", self.camera_intrinsics_record),
            ("r2_hand_state", self.r2_hand_state_record),
            ("side_axis", self.side_axis_record),
        ):
            if record is None:
                continue
            if not isinstance(record.get("sha256"), str) or len(str(record["sha256"])) != 64:
                raise VisualMountProxyError(f"{name} is not SHA-bound")
            if not isinstance(record.get("bytes"), int) or int(record["bytes"]) <= 0:
                raise VisualMountProxyError(f"{name} is not bytes-bound")


def build_fit_manifest_skeleton(
    project_root: Path,
    inputs: VisualFitInputContract,
) -> dict[str, Any]:
    """Return a preregistered manifest skeleton; this is not a fit result."""

    inputs.validate()
    tool = load_pinned_tool_definition(project_root)
    return {
        "schema_version": "robot-mount-visual-proxy-fit-v1",
        "status": "ALGORITHM_AND_INPUT_CONTRACT_ONLY_NOT_EXECUTED",
        "mount_provenance": MOUNT_PROVENANCE,
        "visual_only": True,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "contact_infeasible": CONTACT_INFEASIBLE,
        "tool_definition": tool.as_manifest(),
        "inputs": {
            "session_id": inputs.session_id,
            "frame_indices": list(inputs.frame_indices),
            "hawor_2d_targets": dict(inputs.hawor_target_record),
            "camera_intrinsics": dict(inputs.camera_intrinsics_record),
            "r2_hand_states": dict(inputs.r2_hand_state_record or {}),
            "side_axis_binding": dict(inputs.side_axis_record or {}),
        },
        "parameterization": {
            "left": "ONE_SESSION_CONSTANT_T_left_tool_to_hand_root_SE3",
            "right": "ONE_SESSION_CONSTANT_T_right_tool_to_hand_root_SE3",
            "per_frame_mount_forbidden": True,
        },
        "objective_contract": {
            "domain": "IMAGE_PIXELS_ONLY",
            "target": "HAWOR_IN_IMAGE_HAND_JOINT_DISTRIBUTION",
            "residual": "ROBUST_2D_REPROJECTION_BY_SIDE_AND_FRAME",
            "report": [
                "per_side_per_frame_residual_px",
                "per_side_residual_distribution_px",
                "fit_session_and_exact_frame_set",
            ],
            "ik_residual_must_not_select_mount": True,
        },
        "prohibited_claims": [
            "PHYSICAL_FEASIBILITY",
            "COLLISION_OR_REACHABILITY",
            "TRAINING_GROUND_TRUTH",
            "REAL_ROBOT_EXECUTION",
            "CONTACT_INFEASIBILITY_VALUE",
        ],
        "outputs_required_after_execution": {
            "mount_left_shape": [4, 4],
            "mount_right_shape": [4, 4],
            "residual_distribution_required": True,
            "renderer_ik_fit_tool_definition_equality_required": True,
        },
    }
