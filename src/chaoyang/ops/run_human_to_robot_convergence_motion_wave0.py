#!/usr/bin/env python3
"""Run the frozen 031 direction-authority and hard-position IK window.

This is deliberately a finite development experiment.  It never changes the
149/102/0 denominator, never fills the missing left hand, and only expands to
the 102 input rows if every applicable fixed-window gate passes.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_s2_031_position_first_candidate import fk
from chaoyang.ops.run_kai22_r0_quality_successor_v1 import PinnedRobotBackend
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.huro_hand_frame_v2 import palm_basis
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_baseline_v1_convergence_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/partial_direction_031/wave0"
INDEX = REPO / "tasks/current/INDEX.json"
SOURCE = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
SEED = REPO / "_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/position_first_031/attempt_0002/ROBOT_R0_POSITION_FIRST_CANDIDATE_V1.npz"
HAWOR = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
MOUNT = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
WINDOW = np.arange(66, 82, dtype=np.int64)
POSITION_LIMIT_M = 0.020
DIRECTION_LIMIT_DEG = 15.0
FULL_ROTATION_LIMIT_DEG = 15.0
HAND_COLLISION_TOLERANCE_M = 0.0005


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _stats(values: np.ndarray) -> dict[str, float | int] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    return {"count": int(finite.size), "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)), "max": float(np.max(finite))}


def _assert_route() -> None:
    index = load_json(INDEX)
    rows = index.get("task_packets", [])
    if len(rows) != 1 or rows[0].get("task_id") != TASK or rows[0].get("execution_allowed") is not True:
        raise RuntimeError("TASK_NOT_CURRENTLY_ROUTABLE")


def _qualify_directions(hawor: dict[str, np.ndarray], source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Freeze model-conditioned palm directions before reading candidate IK."""
    count = int(source["frame_id"].shape[0])
    qualified = np.zeros((count, 2, 2), dtype=bool)
    directions_camera = np.full((count, 2, 2, 3), np.nan, dtype=np.float64)
    directions_root = np.full_like(directions_camera, np.nan)
    reason = np.full((count, 2, 2), "NO_SOURCE_INPUT", dtype="U96")
    authority = np.full((count, 2, 2), "NOT_EVALUATED", dtype="U64")
    independent_rgb_support = np.full((count, 2), "UNKNOWN_NOT_INDEPENDENTLY_JUDGED", dtype="U64")
    joints = np.asarray(hawor["joints_3d_camera"], dtype=np.float64)
    predicted = np.asarray(hawor["predicted_valid"], dtype=bool)
    inferred = np.asarray(hawor["inferred"], dtype=bool)
    observed = np.asarray(hawor["observed"], dtype=bool)
    source_valid = np.asarray(source["wrist_valid"], dtype=bool)
    mapping = np.asarray(source["human_to_physical"], dtype=np.int64)
    if joints.shape != (2, count, 21, 3) or not np.array_equal(mapping, [0, 1]):
        raise ValueError("DIRECTION_SOURCE_SHAPE_OR_SIDE_MAPPING_MISMATCH")
    for anatomical in range(2):
        physical = int(mapping[anatomical])
        for frame in range(count):
            if not (source_valid[frame, physical] and predicted[anatomical, frame]):
                continue
            try:
                basis = palm_basis(joints[anatomical, frame])
            except Exception as exc:
                reason[frame, physical, :] = f"DEGENERATE_MODEL_PALM:{type(exc).__name__}"
                continue
            # Frozen semantic directions: longitudinal middle-ray and palm normal.
            axes = np.stack([basis[:, 1], basis[:, 2]])
            target_rotation = np.asarray(source["T_target_root_cam"][frame, physical, :3, :3], dtype=np.float64)
            root_axes = (target_rotation.T @ axes.T).T
            directions_camera[frame, physical] = axes
            directions_root[frame, physical] = root_axes
            qualified[frame, physical] = True
            provenance = "MODEL_OBSERVED" if observed[anatomical, frame] else (
                "MODEL_INFERRED_OFFLINE_NONCAUSAL" if inferred[anatomical, frame] else "MODEL_PREDICTED")
            authority[frame, physical, :] = provenance
            reason[frame, physical, :] = "QUALIFIED_MODEL_DERIVED_NOT_MEASURED_WRIST_ORIENTATION"
    return {
        "direction_qualified": qualified,
        "direction_camera": directions_camera,
        "direction_root": directions_root,
        "direction_reason": reason,
        "direction_authority": authority,
        "independent_rgb_support": independent_rgb_support,
    }


def _solve_rows(source: dict[str, np.ndarray], seed_data: dict[str, np.ndarray], qualification: dict[str, np.ndarray],
                rows: np.ndarray, assets: Any, lower: np.ndarray, upper: np.ndarray,
                tool_mount: np.ndarray, hand_backend: PinnedRobotBackend) -> dict[str, np.ndarray]:
    count = int(source["frame_id"].shape[0])
    q = np.full_like(source["q_arm"], np.nan, dtype=np.float64)
    actual = np.full_like(source["T_actual_root_cam"], np.nan, dtype=np.float64)
    pos_mm = np.full((count, 2), np.nan, dtype=np.float64)
    direction_deg = np.full((count, 2, 2), np.nan, dtype=np.float64)
    rotation_deg = np.full((count, 2), np.nan, dtype=np.float64)
    attempted = np.zeros((count, 2), dtype=bool)
    solver_success = np.zeros_like(attempted)
    position_pass = np.zeros_like(attempted)
    direction_pass = np.zeros_like(attempted)
    full_rotation_pass = np.zeros_like(attempted)
    limit_pass = np.zeros_like(attempted)
    hand_collision_known = np.zeros_like(attempted)
    hand_collision_pass = np.zeros_like(attempted)
    hand_collision_count = np.full((count, 2), -1, dtype=np.int32)
    nfev = np.zeros((count, 2), dtype=np.int32)
    status = np.full((count, 2), "NOT_ATTEMPTED", dtype="U96")
    t_base_cam = np.linalg.inv(source["T_cam_base"])
    source_valid = np.asarray(source["wrist_valid"], dtype=bool)
    for frame in rows.tolist():
        side = 1
        if not source_valid[frame, side]:
            status[frame, side] = "SOURCE_INVALID"
            continue
        allowed = qualification["direction_qualified"][frame, side]
        seed = np.asarray(seed_data["q_arm"][frame, side], dtype=np.float64)
        if not np.isfinite(seed).all():
            seed = np.asarray(source["q_arm"][frame, side], dtype=np.float64)
        seed = np.clip(seed, lower[side], upper[side])
        target = np.asarray(source["T_target_root_cam"][frame, side], dtype=np.float64)
        target_base = t_base_cam @ target
        target_axes = np.asarray(qualification["direction_camera"][frame, side], dtype=np.float64)
        attempted[frame, side] = True

        def root_camera(candidate: np.ndarray) -> np.ndarray:
            return source["T_cam_base"] @ arm._tool_fk(assets, side, candidate) @ tool_mount[side]

        def objective(candidate: np.ndarray) -> float:
            root = root_camera(candidate)
            total = 1e-8 * float(np.sum((candidate - seed) ** 2))
            for axis_index in range(2):
                if allowed[axis_index]:
                    predicted_axis = root[:3, :3] @ qualification["direction_root"][frame, side, axis_index]
                    total += float(np.sum((predicted_axis - target_axes[axis_index]) ** 2))
            return total

        def position_constraint(candidate: np.ndarray) -> float:
            return POSITION_LIMIT_M - float(np.linalg.norm(root_camera(candidate)[:3, 3] - target[:3, 3]))

        result = minimize(objective, seed, method="SLSQP", bounds=list(zip(lower[side], upper[side], strict=True)),
                          constraints=({"type": "ineq", "fun": position_constraint},),
                          options={"maxiter": 240, "ftol": 1e-10, "disp": False})
        candidate = np.asarray(result.x, dtype=np.float64)
        root = root_camera(candidate)
        q[frame, side] = candidate
        actual[frame, side] = root
        nfev[frame, side] = int(result.nfev)
        solver_success[frame, side] = bool(result.success and np.isfinite(candidate).all())
        pos_mm[frame, side] = 1000.0 * np.linalg.norm(root[:3, 3] - target[:3, 3])
        delta = np.linalg.inv(target) @ root
        rotation_deg[frame, side] = np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))
        for axis_index in range(2):
            if allowed[axis_index]:
                predicted_axis = root[:3, :3] @ qualification["direction_root"][frame, side, axis_index]
                direction_deg[frame, side, axis_index] = _angle_deg(predicted_axis, target_axes[axis_index])
        position_pass[frame, side] = solver_success[frame, side] and pos_mm[frame, side] <= 20.0 + 1e-6
        direction_pass[frame, side] = bool(allowed.any() and np.all(direction_deg[frame, side, allowed] <= DIRECTION_LIMIT_DEG + 1e-6))
        full_rotation_pass[frame, side] = bool(rotation_deg[frame, side] <= FULL_ROTATION_LIMIT_DEG + 1e-6)
        limit_pass[frame, side] = bool(np.all(candidate >= lower[side] - 1e-9) and np.all(candidate <= upper[side] + 1e-9))
        collision_ok, collision_count, _, _ = hand_backend.collision(
            np.asarray(source["q_hand22"][frame, side], dtype=np.float64), side, HAND_COLLISION_TOLERANCE_M)
        hand_collision_known[frame, side] = True
        hand_collision_pass[frame, side] = collision_ok
        hand_collision_count[frame, side] = collision_count
        if not allowed.any():
            status[frame, side] = "POSITION_DIAGNOSTIC_DIRECTION_NOT_EVALUATED"
        elif not solver_success[frame, side]:
            status[frame, side] = "SOLVER_FAILED"
        elif position_pass[frame, side] and direction_pass[frame, side] and limit_pass[frame, side] and collision_ok:
            status[frame, side] = "PARTIAL_DIRECTION_LIMITED_SCOPE_PASS"
        else:
            status[frame, side] = "QUALITY_REJECTED"
    return {
        "q_eval": q, "T_actual_root_cam_eval": actual, "position_residual_mm": pos_mm,
        "direction_angle_deg": direction_deg, "full_rotation_angle_deg": rotation_deg,
        "attempted": attempted, "solver_success": solver_success,
        "position_pass": position_pass, "direction_pass": direction_pass,
        "full_rotation_pass": full_rotation_pass, "limit_pass": limit_pass,
        "hand_collision_known": hand_collision_known, "hand_collision_pass": hand_collision_pass,
        "hand_collision_count": hand_collision_count, "solver_nfev": nfev, "row_status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-full-expansion", action="store_true")
    args = parser.parse_args()
    _assert_route()
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    OUT.mkdir(parents=True)
    source, seed_data, hawor = _load_npz(SOURCE), _load_npz(SEED), _load_npz(HAWOR)
    if source["frame_id"].shape != (149,) or int(np.asarray(source["wrist_valid"], bool)[:, 1].sum()) != 102:
        raise ValueError("FROZEN_149_102_DENOMINATOR_MISMATCH")
    if int(np.asarray(source["wrist_valid"], bool)[:, 0].sum()) != 0:
        raise ValueError("FROZEN_LEFT_ZERO_DENOMINATOR_MISMATCH")
    qualification = _qualify_directions(hawor, source)

    assets = load_pinned_robot_assets(REPO)
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    neutral_fk = fk(assets, neutral)
    mount = json.loads(MOUNT.read_text(encoding="utf-8"))
    contract_mount = np.asarray([mount["transforms"]["left_flange_to_hand_root"],
                                 mount["transforms"]["right_flange_to_hand_root"]], dtype=np.float64)
    if not np.allclose(contract_mount, source["T_flange_hand"], atol=1e-10, rtol=0):
        raise ValueError("MOUNT_CONTRACT_MISMATCH")
    tool_mount = np.stack([
        np.linalg.inv(neutral_fk["left_tool"]) @ neutral_fk["flange_L"] @ contract_mount[0],
        np.linalg.inv(neutral_fk["right_tool"]) @ neutral_fk["flange_R"] @ contract_mount[1],
    ])
    backend = PinnedRobotBackend(REPO)
    try:
        window = _solve_rows(source, seed_data, qualification, WINDOW, assets, lower, upper, tool_mount, backend)
        side = 1
        window_gate = (
            qualification["direction_qualified"][WINDOW, side].any(axis=1)
            & window["solver_success"][WINDOW, side]
            & window["position_pass"][WINDOW, side]
            & window["direction_pass"][WINDOW, side]
            & window["limit_pass"][WINDOW, side]
            & window["hand_collision_known"][WINDOW, side]
            & window["hand_collision_pass"][WINDOW, side]
        )
        window_pass = bool(window_gate.all())
        full = None
        if args.allow_full_expansion and window_pass:
            full_rows = np.flatnonzero(np.asarray(source["wrist_valid"], bool)[:, side])
            full = _solve_rows(source, seed_data, qualification, full_rows, assets, lower, upper, tool_mount, backend)
    finally:
        backend.close()

    direction_path = OUT / "DIRECTION_QUALIFICATION_V1.npz"
    _atomic_npz(direction_path, frame_id=source["frame_id"], timestamp_ns=source["timestamp_ns"],
                human_to_physical=source["human_to_physical"], **qualification)
    window_path = OUT / "IK_WINDOW_PARTIAL_DIRECTION_V1.npz"
    _atomic_npz(window_path, frame_id=source["frame_id"], timestamp_ns=source["timestamp_ns"],
                q_source=source["q_arm"], q_inferred=seed_data["q_arm"],
                q_hand22_frozen=source["q_hand22"], source_valid=source["wrist_valid"],
                direction_qualified=qualification["direction_qualified"], **window)
    full_path = None
    if full is not None:
        full_path = OUT / "IK_FULL_PARTIAL_DIRECTION_V1.npz"
        _atomic_npz(full_path, frame_id=source["frame_id"], timestamp_ns=source["timestamp_ns"],
                    q_source=source["q_arm"], q_inferred=seed_data["q_arm"],
                    q_hand22_frozen=source["q_hand22"], source_valid=source["wrist_valid"],
                    direction_qualified=qualification["direction_qualified"], **full)
    right_qualified = qualification["direction_qualified"][:, 1].any(axis=1) & np.asarray(source["wrist_valid"], bool)[:, 1]
    result = {
        "schema_version": "HUMAN_TO_ROBOT_PARTIAL_DIRECTION_IK_WAVE0_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "PASS_LIMITED_WINDOW" if window_pass else "REJECTED_QUALITY",
        "improvement": "EVALUATED_AGAINST_FROZEN_OLD_R0",
        "adoption": "CANDIDATE_ONLY" if window_pass else "NOT_ADOPTED",
        "frozen_denominator": {"timeline": 149, "right_input": 102, "left_input": 0},
        "direction_contract": {
            "source": "MODEL_DERIVED_PALM_BASIS_NOT_MEASURED_WRIST_ORIENTATION",
            "directions": ["palm_longitudinal", "palm_normal"],
            "threshold_deg": DIRECTION_LIMIT_DEG,
            "qualified_right_input_frames": int(right_qualified.sum()),
            "independent_rgb_support": "UNKNOWN_NOT_INDEPENDENTLY_JUDGED",
        },
        "window": {
            "frame_ids": WINDOW.tolist(), "attempted": int(window["attempted"][WINDOW, 1].sum()),
            "gate_pass_count": int(window_gate.sum()), "all_applicable_gates_pass": window_pass,
            "position_mm": _stats(window["position_residual_mm"][WINDOW, 1]),
            "palm_longitudinal_deg": _stats(window["direction_angle_deg"][WINDOW, 1, 0]),
            "palm_normal_deg": _stats(window["direction_angle_deg"][WINDOW, 1, 1]),
            "full_rotation_deg": _stats(window["full_rotation_angle_deg"][WINDOW, 1]),
            "full_rotation_pass_count_separate": int(window["full_rotation_pass"][WINDOW, 1].sum()),
        },
        "full_expansion": {
            "requested": bool(args.allow_full_expansion),
            "executed": full is not None,
            "reason": "WINDOW_ALL_APPLICABLE_GATES_PASS" if full is not None else (
                "WINDOW_REJECTED_NO_EXPANSION" if not window_pass else "NOT_REQUESTED"),
            "artifact": artifact_ref(full_path) if full_path else None,
        },
        "collision_scope": {
            "evaluated": "FROZEN_KAIHAND_INTRA_HAND_ONLY",
            "not_evaluated": ["adapter", "arm_hand", "cross_arm", "continuous_swept_collision"],
        },
        "inputs": {"source_r0": artifact_ref(SOURCE), "position_seed": artifact_ref(SEED),
                   "hawor": artifact_ref(HAWOR), "mount": artifact_ref(MOUNT)},
        "artifacts": {"direction_qualification": artifact_ref(direction_path),
                      "window_candidate": artifact_ref(window_path)},
        "claim_limit": "Development-only partial-direction candidate. A pass does not grant full-pose, complete collision, contact, control, metric or deployment authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    _write_json(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["quality"], "window_gate": f"{int(window_gate.sum())}/16",
                      "qualified_right": int(right_qualified.sum()), "full_executed": full is not None,
                      "output": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
