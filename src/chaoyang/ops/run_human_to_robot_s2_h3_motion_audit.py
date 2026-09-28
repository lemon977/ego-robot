#!/usr/bin/env python3
"""Classify the 031 wrist residual and freeze its source provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import (
    ARM_JOINT_NAMES,
    load_pinned_robot_assets,
)
from chaoyang.pipeline import robot_scene_state_cpu as arm


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
OUT = REPO / (
    f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/"
    "h3_motion_audit_031/attempt_0002"
)
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
V3 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"


def ref(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def percentile(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def fk(assets, q_both: np.ndarray) -> dict[str, np.ndarray]:
    values = {
        name: float(q_both[side, joint])
        for side, names in enumerate(ARM_JOINT_NAMES)
        for joint, name in enumerate(names)
    }
    return forward_kinematics(assets.tianji, values)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    OUT.mkdir(parents=True)
    robot_path = R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
    hand_path = R2 / "lanes/lane2_motion/recovered_031_wave0/HAND_MOTION_V1.npz"
    upstream_path = V3 / "shared/robot/camera031_epoch7_c3/CAMERA_ROBOT_MOTION_V3.npz"
    hawor_result_path = V3 / "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/RESULT.json"
    roi_spec_path = V3 / "lanes/ai2/FULL031_COMPONENT_EPOCH7_SPEC.json"
    role_path = V3 / "lanes/exact78/masks031_right_epoch7_v1/play_cards_0915_031/ROLE_MANIFEST_human.json"
    mount_path = V3 / "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
    data = load_npz(robot_path)
    hand = load_npz(hand_path)
    upstream = load_npz(upstream_path)
    hawor_result = json.loads(hawor_result_path.read_text(encoding="utf-8"))
    roi_spec = json.loads(roi_spec_path.read_text(encoding="utf-8"))
    role = json.loads(role_path.read_text(encoding="utf-8"))
    mount = json.loads(mount_path.read_text(encoding="utf-8"))

    mapping = np.asarray(data["human_to_physical"], dtype=np.int64)
    if not np.array_equal(mapping, [0, 1]):
        raise ValueError("UNEXPECTED_031_SIDE_MAPPING")
    contract_hand = np.asarray([
        mount["transforms"]["left_flange_to_hand_root"],
        mount["transforms"]["right_flange_to_hand_root"],
    ], dtype=np.float64)
    mount_matches = bool(np.allclose(contract_hand, data["T_flange_hand"], atol=1e-10, rtol=0.0))
    if not mount_matches:
        raise ValueError("MOUNT_CONTRACT_MISMATCH")

    assets = load_pinned_robot_assets(REPO)
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    neutral_fk = fk(assets, neutral)
    tool_mount = np.stack([
        np.linalg.inv(neutral_fk["left_tool"]) @ neutral_fk["flange_L"] @ contract_hand[0],
        np.linalg.inv(neutral_fk["right_tool"]) @ neutral_fk["flange_R"] @ contract_hand[1],
    ])
    valid = np.asarray(data["wrist_valid"], dtype=bool)
    renderer_saved_mm: list[float] = []
    producer_saved_mm: list[float] = []
    independent_residual_mm: list[float] = []
    position_only_mm: list[float] = []
    at_limit_rows = 0
    valid_rows = 0
    per_frame = []
    for frame in range(len(valid)):
        q_both = np.asarray(data["q_arm"][frame], dtype=np.float64).copy()
        # The missing anatomical/physical side is intentionally NaN.  Fill only
        # that unobserved side with the pinned neutral state so the shared dual-
        # arm FK can evaluate the valid side; the missing side remains invalid
        # and is never reported or consumed.
        q_both[~np.isfinite(q_both)] = neutral[~np.isfinite(q_both)]
        if not valid[frame].any():
            continue
        frames = fk(assets, q_both)
        for side in np.flatnonzero(valid[frame]):
            side = int(side)
            valid_rows += 1
            flange_name = ("flange_L", "flange_R")[side]
            tool_name = ("left_tool", "right_tool")[side]
            renderer_root = data["T_cam_base"] @ frames[flange_name] @ contract_hand[side]
            producer_root = data["T_cam_base"] @ frames[tool_name] @ tool_mount[side]
            saved_root = data["T_actual_root_cam"][frame, side]
            target_root = data["T_target_root_cam"][frame, side]
            renderer_delta = float(np.linalg.norm(renderer_root[:3, 3] - saved_root[:3, 3]) * 1000.0)
            producer_delta = float(np.linalg.norm(producer_root[:3, 3] - saved_root[:3, 3]) * 1000.0)
            residual = float(np.linalg.norm(renderer_root[:3, 3] - target_root[:3, 3]) * 1000.0)
            renderer_saved_mm.append(renderer_delta)
            producer_saved_mm.append(producer_delta)
            independent_residual_mm.append(residual)
            q = q_both[side]
            at_limit = bool(np.any(np.isclose(q, lower[side], atol=1e-5)) or
                            np.any(np.isclose(q, upper[side], atol=1e-5)))
            at_limit_rows += int(at_limit)
            target_base = np.linalg.inv(data["T_cam_base"]) @ target_root

            def position_error(candidate: np.ndarray) -> np.ndarray:
                root = arm._tool_fk(assets, side, candidate) @ tool_mount[side]
                return (root[:3, 3] - target_base[:3, 3]) * 1000.0

            solved = least_squares(
                position_error, q, bounds=(lower[side], upper[side]), max_nfev=150,
                ftol=1e-9, xtol=1e-9, gtol=1e-9,
            )
            best = float(np.linalg.norm(position_error(solved.x)))
            position_only_mm.append(best)
            per_frame.append((frame, side, residual, best, renderer_delta, at_limit))

    display = percentile(renderer_saved_mm)
    producer = percentile(producer_saved_mm)
    current = percentile(independent_residual_mm)
    position_only = percentile(position_only_mm)
    feasible_count = int(np.sum(np.asarray(position_only_mm) <= 20.0))
    feasible_fraction = float(feasible_count / valid_rows) if valid_rows else 0.0
    display_consistent = bool(display is not None and display["max"] <= 1e-5)
    producer_consistent = bool(producer is not None and producer["max"] <= 1e-5)
    if not display_consistent or not producer_consistent:
        classification = "FK_OR_DISPLAY_CONSUMPTION_MISMATCH"
    elif current is not None and current["p95"] > 20.0 and feasible_fraction >= 0.8:
        classification = "FULL_POSE_OBJECTIVE_OR_LOCAL_OPTIMIZER_NOT_POSITION_REACHABILITY"
    elif current is not None and current["p95"] > 20.0 and feasible_fraction < 0.8:
        classification = "FIXED_PLACEMENT_OR_REACHABILITY_LIMITED"
    else:
        classification = "NO_LARGE_POSITION_RESIDUAL_REPRODUCED"

    direct = int(np.asarray(hand["observed_physical"], dtype=bool).sum())
    inferred = int(np.asarray(hand["inferred_physical"], dtype=bool).sum())
    role_unknown_rows = sum(
        1 for rows in role["rows"] for row in rows
        if row.get("anatomical_side") == "UNKNOWN"
    )
    np.savez_compressed(
        OUT / "WRIST_AUDIT_V1.npz",
        frame_id=np.asarray([row[0] for row in per_frame], dtype=np.int32),
        side=np.asarray([row[1] for row in per_frame], dtype=np.int8),
        full_pose_position_residual_mm=np.asarray([row[2] for row in per_frame]),
        position_only_best_residual_mm=np.asarray([row[3] for row in per_frame]),
        renderer_vs_saved_root_mm=np.asarray([row[4] for row in per_frame]),
        arm_at_limit=np.asarray([row[5] for row in per_frame], dtype=bool),
    )
    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_H3_MOTION_AUDIT_V1",
        "task_id": TASK,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": "play_cards_0915_031",
        "execution": "EXECUTED",
        "structure": "PASS" if display_consistent and producer_consistent else "FAIL",
        "quality": "REJECTED_QUALITY",
        "adoption": "NOT_ADOPTED",
        "root_cause_classification": classification,
        "valid_wrist_rows": valid_rows,
        "mount_contract_matches_saved_motion": mount_matches,
        "renderer_fk_vs_saved_actual_mm": display,
        "producer_fk_vs_saved_actual_mm": producer,
        "full_pose_target_residual_mm": current,
        "position_only_best_residual_mm": position_only,
        "position_only_within_existing_20mm_count": feasible_count,
        "position_only_within_existing_20mm_fraction": feasible_fraction,
        "arm_limit_hit_rows": at_limit_rows,
        "arm_limit_hit_fraction": float(at_limit_rows / valid_rows) if valid_rows else 0.0,
        "source_provenance": {
            "hawor_predicted_left_right": hawor_result["predicted_by_side"],
            "direct_detector_roi_left_right": hawor_result["direct_roi_by_side"],
            "non_direct_detector_roi_left_right": hawor_result["inferred_roi_by_side"],
            "exported_observed_count": direct,
            "exported_inferred_count": inferred,
            "inferred_semantics": "HAWOR_MODEL_RUN_ON_SAM_TEMPORAL_ROI_NOT_DIRECT_DETECTOR;NOT_PROOF_OF_MOTION_INFILLER",
            "roi_identity_evidence": roi_spec["sessions"][0]["identity_evidence"],
            "sam_role_rows_with_unknown_anatomical_side": role_unknown_rows,
            "source_authority": "AI_VISUAL_SIDE_ANCHOR_PLUS_SAM_TEMPORAL_TRACK_NOT_INDEPENDENT_GOLD",
            "temporal_authority": "OFFLINE_NONCAUSAL",
            "left_status": "NO_ROI_NO_MODEL_OUTPUT_NOT_FILLED",
        },
        "inputs": {
            "robot_r0": ref(robot_path), "hand_motion": ref(hand_path),
            "upstream_robot": ref(upstream_path), "hawor_result": ref(hawor_result_path),
            "roi_spec": ref(roi_spec_path), "sam_role_manifest": ref(role_path),
            "mount_contract": ref(mount_path),
        },
        "arrays": ref(OUT / "WRIST_AUDIT_V1.npz"),
        "claim_limit": "Development root-cause classification in the pinned virtual camera/base and URDF only; not external wrist accuracy, reachability certification, control or deployment authority.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    (OUT / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "root_cause_classification": classification,
        "position_only_feasible_fraction": feasible_fraction,
        "output": str(OUT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
