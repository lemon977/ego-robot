#!/usr/bin/env python3
"""Test whether 031's large residual is caused by skipping position homotopy."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_v2 import motion_derivatives, time_edges
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.ops.run_human_to_robot_s2_031_position_first_candidate import fk


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/two_stage_031/attempt_0001"
SOURCE = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
MOUNT = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"


def ref(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    OUT.mkdir(parents=True)
    with np.load(SOURCE, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    mount = json.loads(MOUNT.read_text(encoding="utf-8"))
    mounts = np.asarray([mount["transforms"]["left_flange_to_hand_root"],
                         mount["transforms"]["right_flange_to_hand_root"]], dtype=np.float64)
    if not np.allclose(mounts, source["T_flange_hand"], atol=1e-10, rtol=0):
        raise ValueError("MOUNT_CONTRACT_MISMATCH")
    if not np.array_equal(source["human_to_physical"], [0, 1]):
        raise ValueError("SIDE_MAPPING_MISMATCH")

    assets = load_pinned_robot_assets(REPO)
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    neutral_fk = fk(assets, neutral)
    tool_mount = np.stack([
        np.linalg.inv(neutral_fk["left_tool"]) @ neutral_fk["flange_L"] @ mounts[0],
        np.linalg.inv(neutral_fk["right_tool"]) @ neutral_fk["flange_R"] @ mounts[1],
    ])
    valid = np.asarray(source["wrist_valid"], dtype=bool)
    q = np.full_like(source["q_arm"], np.nan, dtype=np.float64)
    actual = np.full_like(source["T_actual_root_cam"], np.nan, dtype=np.float64)
    position = np.full_like(source["position_residual_mm"], np.nan, dtype=np.float64)
    rotation = np.full_like(source["rotation_residual_deg"], np.nan, dtype=np.float64)
    position_nfev = np.zeros_like(valid, dtype=np.int32)
    pose_nfev = np.zeros_like(valid, dtype=np.int32)
    success = np.zeros_like(valid)
    t_base_cam = np.linalg.inv(source["T_cam_base"])

    for frame, side_raw in zip(*np.nonzero(valid)):
        side = int(side_raw)
        target_base = t_base_cam @ source["T_target_root_cam"][frame, side]
        target_tool = target_base @ np.linalg.inv(tool_mount[side])
        seed = np.clip(np.asarray(source["q_arm"][frame, side], dtype=np.float64),
                       lower[side], upper[side])

        def position_residual(candidate: np.ndarray) -> np.ndarray:
            return arm._position_residual(arm._tool_fk(assets, side, candidate), target_tool)

        stage1 = least_squares(
            position_residual, seed, bounds=(lower[side], upper[side]), method="trf",
            ftol=arm.IK_TRF_FTOL, xtol=arm.IK_TRF_XTOL, gtol=arm.IK_TRF_GTOL,
            max_nfev=arm.FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS)

        def pose_residual(candidate: np.ndarray) -> np.ndarray:
            return arm._pose_residual(arm._tool_fk(assets, side, candidate), target_tool)

        stage2 = least_squares(
            pose_residual, stage1.x, bounds=(lower[side], upper[side]), method="trf",
            ftol=arm.IK_TRF_FTOL, xtol=arm.IK_TRF_XTOL, gtol=arm.IK_TRF_GTOL,
            max_nfev=arm.FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS)
        q[frame, side] = stage2.x
        position_nfev[frame, side] = stage1.nfev
        pose_nfev[frame, side] = stage2.nfev
        success[frame, side] = bool(stage1.success and stage2.success)
        root_cam = source["T_cam_base"] @ arm._tool_fk(assets, side, stage2.x) @ tool_mount[side]
        actual[frame, side] = root_cam
        delta = np.linalg.inv(source["T_target_root_cam"][frame, side]) @ root_cam
        position[frame, side] = np.linalg.norm(delta[:3, 3]) * 1000.0
        rotation[frame, side] = np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))

    pose_gate = valid & success & (position <= 20.0) & (rotation <= 15.0)
    at_limit = np.zeros_like(valid)
    for side in range(2):
        rows = valid[:, side]
        at_limit[rows, side] = np.any(
            np.isclose(q[rows, side], lower[side], atol=1e-5)
            | np.isclose(q[rows, side], upper[side], atol=1e-5), axis=1)
    edges, times = time_edges(source["timestamp_ns"], source["frame_id"])
    derivatives = motion_derivatives(q, valid, times, edges)
    candidate = dict(source)
    candidate.update(
        q_arm=q, T_actual_root_cam=actual, position_residual_mm=position,
        rotation_residual_deg=rotation, solver_success=success,
        solver_position_nfev=position_nfev, solver_full_pose_nfev=pose_nfev,
        tolerance_pass=pose_gate, arm_velocity=derivatives["velocity"],
        arm_acceleration=derivatives["acceleration"], arm_jerk=derivatives["jerk"],
        time_edge_valid=derivatives["edge_valid"],
    )
    candidate_path = OUT / "ROBOT_R0_TWO_STAGE_CANDIDATE_V1.npz"
    np.savez_compressed(candidate_path, **candidate)
    source_pos = np.asarray(source["position_residual_mm"], dtype=np.float64)[valid]
    source_rot = np.asarray(source["rotation_residual_deg"], dtype=np.float64)[valid]
    passed = int(np.count_nonzero(pose_gate))
    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_TWO_STAGE_CANDIDATE_V1", "task_id": TASK,
        "session_id": "play_cards_0915_031", "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "PASS_NUMERIC_POSE_ONLY" if passed == int(valid.sum()) else "REJECTED_QUALITY",
        "adoption": "CANDIDATE_ONLY" if passed == int(valid.sum()) else "NOT_ADOPTED",
        "method": "POSITION_HOMOTOPY_THEN_FULL_POSE_FROM_SAME_PAIR",
        "valid_rows": int(valid.sum()), "pose_gate_pass_count": passed,
        "pose_gate_pass_fraction": passed / int(valid.sum()),
        "source_position_mm": stats(source_pos), "source_rotation_deg": stats(source_rot),
        "candidate_position_mm": stats(position[valid]), "candidate_rotation_deg": stats(rotation[valid]),
        "position_stage_nfev": stats(position_nfev[valid]),
        "full_pose_stage_nfev": stats(pose_nfev[valid]),
        "arm_limit_hit_count": int(at_limit[valid].sum()),
        "collision_scope": "NOT_EVALUATED_IN_TWO_STAGE_DIAGNOSTIC",
        "inputs": {"source_robot_r0": ref(SOURCE), "mount_contract": ref(MOUNT)},
        "candidate": ref(candidate_path),
        "claim_limit": "Fixed-base numeric IK diagnostic only; no collision, Scene, Contact, control or deployment authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["quality"], "pose_gate": f"{passed}/{int(valid.sum())}",
                      "position_p95_mm": result["candidate_position_mm"]["p95"],
                      "rotation_p95_deg": result["candidate_rotation_deg"]["p95"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
