#!/usr/bin/env python3
"""Correct the P1 mount frame and keep temporal conflicts joint-local.

The frozen mount proxy is flange->hand-root, while IK and rendering consume
tool->hand-root.  This successor derives the fixed flange->tool transform from
the pinned Tianji URDF before solving.  It also resolves an empty temporal
intersection per joint instead of freezing all seven joints.  Existing
0.12 rad/frame and 0.06 rad/frame^2 limits are unchanged and still reported.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
from chaoyang.ops import run_exact78_session_base_arm_v1 as base
from chaoyang.ops import run_exact78_session_base_arm_v4 as previous
from chaoyang.pipeline.robot_hand_visual_alignment import kai_palmar_axis_local
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES


def corrected_mounts() -> tuple[np.ndarray, dict]:
    flange_hand, source = base.mounts()
    assets = base.arm.old.load_pinned_robot_assets(REPO_ROOT)
    values = {name: 0.0 for row in ARM_JOINT_NAMES for name in row}
    fk = forward_kinematics(assets.tianji, values)
    result = []
    flange_tool = []
    for physical, side in enumerate(("left", "right")):
        flange = fk[f"flange_{'L' if physical == 0 else 'R'}"]
        tool = fk[f"{side}_tool"]
        fixed = np.linalg.inv(flange) @ tool
        converted = np.linalg.inv(fixed) @ flange_hand[physical]
        if not np.allclose(fixed @ converted, flange_hand[physical], atol=1e-10):
            raise RuntimeError(f"MOUNT_FRAME_CONVERSION_FAILED:{side}")
        flange_tool.append(fixed)
        result.append(converted)
    reference = dict(source)
    reference.update({
        "source_matrix_direction": "p_flange = T_flange_hand_root_proxy @ p_hand_root",
        "consumer_matrix_direction": "p_tool = T_tool_hand_root_proxy @ p_hand_root",
        "conversion": "inv(T_flange_tool_from_pinned_urdf) @ T_flange_hand_root_proxy",
        "T_flange_tool_from_pinned_urdf": np.asarray(flange_tool).tolist(),
        "cards042_fitted_values_read": False,
    })
    return np.asarray(result), reference


def componentwise_bounded_limits(lower: np.ndarray, upper: np.ndarray,
                                 previous_q: np.ndarray,
                                 previous_previous: np.ndarray | None,
                                 step_limit: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    previous_q = np.clip(np.asarray(previous_q, dtype=np.float64), lower, upper)
    lo = np.maximum(lower, previous_q - float(step_limit))
    hi = np.minimum(upper, previous_q + float(step_limit))
    if previous_previous is None:
        return lo, hi
    older = np.clip(np.asarray(previous_previous, dtype=np.float64), lower, upper)
    predicted = 2.0 * previous_q - older
    accel_lo = predicted - float(base.arm.temporal.ACCELERATION_LIMIT)
    accel_hi = predicted + float(base.arm.temporal.ACCELERATION_LIMIT)
    candidate_lo = np.maximum(lo, accel_lo)
    candidate_hi = np.minimum(hi, accel_hi)
    feasible = candidate_lo <= candidate_hi + 1e-10
    lo[feasible] = candidate_lo[feasible]
    hi[feasible] = candidate_hi[feasible]
    # A hard-limit/acceleration contradiction in one coordinate must not hold
    # the other six.  Use the closest hard+velocity-feasible coordinate and
    # leave the resulting acceleration miss visible in the output metrics.
    conflict = ~feasible
    closest = np.clip(predicted, lo, hi)
    lo[conflict] = closest[conflict]
    hi[conflict] = closest[conflict]
    midpoint = 0.5 * (lo + hi)
    collapsed = hi - lo <= 1e-12
    lo[collapsed] = midpoint[collapsed]
    hi[collapsed] = midpoint[collapsed]
    return lo, hi


def _argument(name: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1]).resolve(strict=True)
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"ARGUMENT_REQUIRED:{name}") from error


def enrich_result() -> None:
    output = _argument("--output-root")
    hawor_path = _argument("--hawor")
    result_path = output / "RESULT.json"
    state_path = output / "ARM_SESSION_BASE_STATES.npz"
    with np.load(state_path, allow_pickle=False) as archive:
        actual = np.asarray(archive["T_actual_hand_root_world"], dtype=np.float64)
    with np.load(hawor_path, allow_pickle=False) as archive:
        c2w = np.asarray(archive["c2w"], dtype=np.float64)
    values: dict[str, list[float]] = {"left": [], "right": []}
    for frame in range(len(c2w)):
        camera_world = np.linalg.inv(c2w[frame])
        for side, name in enumerate(("left", "right")):
            root_camera = camera_world @ actual[frame, side]
            normal = root_camera[:3, :3] @ kai_palmar_axis_local(name)
            values[name].append(float(normal[1]))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema_version"] = "exact78-session-base-arm-v5"
    result["mount_frame_correction"] = {
        "source": artifact_ref(base.MOUNT_CONTRACT),
        "source_frame": "flange",
        "consumer_frame": "tool",
        "derived_from_pinned_urdf": True,
        "cards042_fitted_values_read": False,
    }
    result["metrics"]["palmar_normal_camera_y"] = {
        side: {
            "minimum": float(np.min(rows)),
            "mean": float(np.mean(rows)),
            "positive_fraction": float(np.mean(np.asarray(rows) > 0.0)),
        }
        for side, rows in values.items()
    }
    result["runtime_correction"] = {
        "changed": ["flange_to_tool_mount_frame_conversion", "joint_local_temporal_conflict"],
        "numeric_thresholds_changed": False,
        "solver_changed": False,
        "model_changed": False,
        "frame_failure_semantics": "per-frame status; process exit remains zero",
    }
    atomic_json(result_path, result)


def main() -> int:
    old_mounts = base.mounts
    old_bounds = base.arm.temporal.bounded_limits
    base.mounts = corrected_mounts
    base.arm.temporal.bounded_limits = componentwise_bounded_limits
    try:
        code = previous.main()
    finally:
        base.mounts = old_mounts
        base.arm.temporal.bounded_limits = old_bounds
    if code == 0:
        enrich_result()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
