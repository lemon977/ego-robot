#!/usr/bin/env python3
"""Evaluate one fixed-base, position-first 031 arm candidate.

The candidate keeps the camera/base, 68.4 mm mount, hand q22, validity and
wrist targets frozen. Only valid physical arm q values are re-solved. Rotation
remains an explicit hard-gate diagnostic, so translation improvement cannot
silently become an adopted R0 result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_v2 import motion_derivatives, time_edges
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/position_first_031/attempt_0002"
SOURCE = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
MOUNT = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def stats(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def fk(assets, q_both: np.ndarray) -> dict[str, np.ndarray]:
    values = {name: float(q_both[side, joint])
              for side, names in enumerate(ARM_JOINT_NAMES)
              for joint, name in enumerate(names)}
    return forward_kinematics(assets.tianji, values)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    OUT.mkdir(parents=True)
    with np.load(SOURCE, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    mount = json.loads(MOUNT.read_text(encoding="utf-8"))
    contract_mount = np.asarray([
        mount["transforms"]["left_flange_to_hand_root"],
        mount["transforms"]["right_flange_to_hand_root"],
    ], dtype=np.float64)
    if not np.allclose(contract_mount, source["T_flange_hand"], atol=1e-10, rtol=0):
        raise ValueError("MOUNT_CONTRACT_MISMATCH")
    if not np.array_equal(source["human_to_physical"], [0, 1]):
        raise ValueError("UNEXPECTED_SIDE_MAPPING")

    assets = load_pinned_robot_assets(REPO)
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    neutral_fk = fk(assets, neutral)
    tool_mount = np.stack([
        np.linalg.inv(neutral_fk["left_tool"]) @ neutral_fk["flange_L"] @ contract_mount[0],
        np.linalg.inv(neutral_fk["right_tool"]) @ neutral_fk["flange_R"] @ contract_mount[1],
    ])
    valid = np.asarray(source["wrist_valid"], dtype=bool)
    q_candidate = np.full_like(source["q_arm"], np.nan, dtype=np.float64)
    actual = np.full_like(source["T_actual_root_cam"], np.nan, dtype=np.float64)
    position = np.full_like(source["position_residual_mm"], np.nan, dtype=np.float64)
    rotation = np.full_like(source["rotation_residual_deg"], np.nan, dtype=np.float64)
    nfev = np.zeros_like(valid, dtype=np.int32)
    success = np.zeros_like(valid)
    t_base_cam = np.linalg.inv(source["T_cam_base"])

    for frame, side_raw in zip(*np.nonzero(valid)):
        side = int(side_raw)
        target_base = t_base_cam @ source["T_target_root_cam"][frame, side]
        seed = np.asarray(source["q_arm"][frame, side], dtype=np.float64)

        def residual(candidate: np.ndarray) -> np.ndarray:
            root = arm._tool_fk(assets, side, candidate) @ tool_mount[side]
            return (root[:3, 3] - target_base[:3, 3]) / arm.POSITION_SCALE_M

        solved = least_squares(
            residual, np.clip(seed, lower[side], upper[side]),
            bounds=(lower[side], upper[side]), method="trf",
            ftol=arm.IK_TRF_FTOL, xtol=arm.IK_TRF_XTOL, gtol=arm.IK_TRF_GTOL,
            max_nfev=arm.FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS,
        )
        q_candidate[frame, side] = solved.x
        nfev[frame, side] = solved.nfev
        success[frame, side] = bool(solved.success)
        root_cam = source["T_cam_base"] @ arm._tool_fk(assets, side, solved.x) @ tool_mount[side]
        actual[frame, side] = root_cam
        delta = np.linalg.inv(source["T_target_root_cam"][frame, side]) @ root_cam
        position[frame, side] = np.linalg.norm(delta[:3, 3]) * 1000.0
        rotation[frame, side] = np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))

    within_position = valid & success & (position <= 20.0)
    within_pose = within_position & (rotation <= 15.0)
    at_limit = np.zeros_like(valid)
    for side in range(2):
        rows = valid[:, side]
        at_limit[rows, side] = np.any(
            np.isclose(q_candidate[rows, side], lower[side], atol=1e-5)
            | np.isclose(q_candidate[rows, side], upper[side], atol=1e-5), axis=1)
    edges, times = time_edges(source["timestamp_ns"], source["frame_id"])
    derivatives = motion_derivatives(q_candidate, valid, times, edges)

    candidate = dict(source)
    candidate.update(
        q_arm=q_candidate, T_actual_root_cam=actual,
        position_residual_mm=position, rotation_residual_deg=rotation,
        solver_success=success, solver_nfev=nfev, tolerance_pass=within_pose,
        time_edge_valid=edges, arm_velocity=derivatives["velocity"],
        arm_acceleration=derivatives["acceleration"], arm_jerk=derivatives["jerk"],
        arm_edge_valid=derivatives["edge_valid"],
        candidate_method=np.asarray("FIXED_BASE_POSITION_ONLY_FROM_EXISTING_Q"),
    )
    output_npz = OUT / "ROBOT_R0_POSITION_FIRST_CANDIDATE_V1.npz"
    np.savez_compressed(output_npz, **candidate)
    source_position = np.asarray(source["position_residual_mm"], dtype=np.float64)
    source_rotation = np.asarray(source["rotation_residual_deg"], dtype=np.float64)
    valid_count = int(valid.sum())
    pose_pass_count = int(within_pose.sum())
    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_POSITION_FIRST_CANDIDATE_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "PASS" if pose_pass_count == valid_count else "REJECTED_QUALITY",
        "adoption": "NOT_ADOPTED", "method": "FIXED_BASE_POSITION_ONLY_FROM_EXISTING_Q",
        "frozen": ["T_cam_base", "T_target_root_cam", "T_flange_hand", "q_hand22",
                   "validity", "camera", "mount", "robot_assets"],
        "valid_rows": valid_count,
        "source_full_pose_position_mm": stats(source_position[valid]),
        "source_full_pose_rotation_deg": stats(source_rotation[valid]),
        "candidate_position_mm": stats(position[valid]),
        "candidate_rotation_deg": stats(rotation[valid]),
        "position_within_20mm_count": int(within_position.sum()),
        "position_within_20mm_fraction": float(within_position.sum() / valid_count),
        "full_pose_within_20mm_15deg_count": pose_pass_count,
        "full_pose_within_20mm_15deg_fraction": float(pose_pass_count / valid_count),
        "arm_limit_hit_count": int(at_limit.sum()),
        "arm_limit_hit_fraction": float(at_limit.sum() / valid_count),
        "collision_scope": "NOT_EVALUATED_IN_POSITION_FIRST_DIAGNOSTIC",
        "inputs": {"source_robot_r0": ref(SOURCE), "mount_contract": ref(MOUNT)},
        "candidate": ref(output_npz),
        "claim_limit": "Development fixed-base diagnostic only. Translation does not override rotation, collision, motion, provenance, control, or deployment gates.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "quality": result["quality"],
                      "position_within_20mm_fraction": result["position_within_20mm_fraction"],
                      "full_pose_within_20mm_15deg_fraction": result["full_pose_within_20mm_15deg_fraction"],
                      "output": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
