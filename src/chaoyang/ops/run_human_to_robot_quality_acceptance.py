"""Bounded, registered quality-acceptance consumers; old results remain read-only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import subprocess

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.full_robot_review_v2 import (
    motion_derivatives, seed_usable_solution, solve_full_chain, time_edges,
)
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline.huro_hand_only_retarget_v1 import (
    FINGER_CHAINS, load_hand_model, keypoints_from_q,
)

TASK = "human_to_robot_quality_acceptance_20260924"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
CHIPS_R0 = REPO_ROOT / ("_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001/"
    "sessions/potato_chips/get_potato_chips_0915_042/KAI22_R0_BASELINE_V1.npz")
OLD = REPO_ROOT / ("_run/current/human_to_robot_representative_baseline_20260924/attempts/"
    "attempt_0001/get_potato_chips_0915_042_FULL_ARM_DEVELOPMENT.npz")
MOUNT = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
    "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json")
SENSOR_ROOT = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/"
    "attempts/attempt_0001/lanes/lane3_sensor")
SENSOR_HAND_ROOT = REPO_ROOT / ("_run/current/four_stream_visual_delivery_v5/"
    "attempts/attempt_0001/lanes/sensor/run_0001")
POKER_SCENE = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/"
    "attempts/attempt_0001/lanes/lane1_scene/clean_candidate_042_wave4")
POKER_ACCEPTED = REPO_ROOT / ("archive/baseline-20260917-0aa69e9/content/history/tasks/"
    "control/runs/20260908_two_task_e2e_baseline_v1/clean_synthetic_propainter_v1")
POKER_R0 = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
    "lanes/ai2/camera_exact_mount_v2/play_cards_0902_042/CAMERA_ROBOT_MOTION_V3.npz")
POKER_SOURCE = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
    "lanes/ai2/exact_bounded_camera_source_v3/play_cards_0902_042/BOUNDED_CAMERA_SOURCE_V3.npz")
POKER_PRODUCER = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
    "epoch6_upload/motion/camera_robot_motion_v3.py")
POKER_OLD_PRODUCT = REPO_ROOT / ("_run/current/human_to_robot_evidence_unlock_s2_20260923/"
    "attempts/attempt_0001/lanes/motion_product/formal_product_play_cards_0902_042/"
    "attempt_0002/PRODUCT_RESULT.json")


def _archived_poker_path(original: str) -> Path:
    marker = "/tasks/control/runs/20260908_two_task_e2e_baseline_v1/"
    if marker not in original:
        raise RuntimeError("ARCHIVE_PATH_NOT_IN_FROZEN_RUN")
    relative = original.split(marker, 1)[1]
    return (POKER_ACCEPTED.parent / relative).resolve(strict=True)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _check_task() -> None:
    packet = load_json(REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if packet["task_id"] != TASK or not any(
        row.get("task_id") == TASK and row.get("execution_allowed") is True
        for row in index.get("task_packets", [])
    ):
        raise RuntimeError("TASK_NOT_ROUTABLE")


def _max_step(q: np.ndarray, valid: np.ndarray, frames: range) -> tuple[float | None, int | None]:
    values = []
    for frame in frames:
        if frame > 0 and valid[frame] and valid[frame-1]:
            values.append((float(np.max(np.abs(q[frame]-q[frame-1]))), frame))
    return max(values, default=(None, None))


def chips_seed_counterfactual() -> dict:
    """One-change 0:160 prefix; 0:15 controls and 156:160 diagnose the seed."""
    _check_task()
    output = ATTEMPT / "lanes/motion_product/CHIPS_SEED_COUNTERFACTUAL_0_160.npz"
    receipt = output.with_suffix(".json")
    if output.exists() or receipt.exists():
        raise FileExistsError("COUNTERFACTUAL_ALREADY_EXISTS")
    source, old = _load(CHIPS_R0), _load(OLD)
    mount = load_json(MOUNT)
    flange = np.asarray([mount["transforms"]["left_flange_to_hand_root"],
                         mount["transforms"]["right_flange_to_hand_root"]], dtype=float)
    if not np.array_equal(source["human_to_physical"], [1, 0]):
        raise RuntimeError("CHIPS_SIDE_MAPPING_DRIFT")
    arrays = {"q22": source["q22_init"][:161],
              "finger_valid": source["valid_side_frame"][:161],
              "wrist_valid": source["valid_side_frame"][:161],
              "relative_wrist_T": source["relative_wrist_T"][:161],
              "timestamp_ns": source["timestamp_ns"][:161],
              "frame_id": source["frame_id"][:161],
              "human_to_physical": source["human_to_physical"]}
    solved = solve_full_chain(arrays, load_pinned_robot_assets(REPO_ROOT), flange,
                              max_nfev=80, continuity_weight=0.0,
                              experimental_seed_recovery_frames={158})
    if not np.array_equal(solved["frame_id"], old["frame_id"][:161]):
        raise RuntimeError("FRAME_MAP_DRIFT")
    unchanged_prefix = np.allclose(solved["q_arm"][:159], old["q_arm"][:159],
                                   atol=1e-8, equal_nan=True)
    if not unchanged_prefix:
        raise RuntimeError("SINGLE_VARIABLE_PREFIX_DRIFT")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **solved)
    with np.load(output, allow_pickle=False) as reloaded:
        if not np.array_equal(reloaded["frame_id"], solved["frame_id"]):
            raise RuntimeError("NPZ_RELOAD_FRAME_DRIFT")
    side = 0  # physical-left anomaly in the frozen original result
    rows = []
    for frame in (*range(16), *range(156, 161)):
        rows.append({"frame_id": int(solved["frame_id"][frame]), "index": frame,
                     "old_success": bool(old["solver_success"][frame, side]),
                     "new_success": bool(solved["solver_success"][frame, side]),
                     "new_status": int(solved["solver_status"][frame, side]),
                     "new_nfev": int(solved["solver_nfev"][frame, side]),
                     "seed_usable": bool(solved["seed_usable"][frame, side]),
                     "seed_source_frame": int(solved["seed_source_frame"][frame, side]),
                     "reset_reason": str(solved["reset_reason"][frame, side]),
                     "old_position_mm": float(old["position_residual_mm"][frame, side]),
                     "new_position_mm": float(solved["position_residual_mm"][frame, side]),
                     "old_rotation_deg": float(old["rotation_residual_deg"][frame, side]),
                     "new_rotation_deg": float(solved["rotation_residual_deg"][frame, side]),
                     "old_q": old["q_arm"][frame, side].tolist(),
                     "seed_in": solved["seed_in"][frame, side].tolist(),
                     "new_q": solved["q_arm"][frame, side].tolist()})
    valid = solved["wrist_valid"][:, side]
    before_step = _max_step(old["q_arm"][:, side], valid, range(156, 161))
    after_step = _max_step(solved["q_arm"][:, side], valid, range(156, 161))
    result = {"schema_version": "CHIPS_SEED_COUNTERFACTUAL_V1", "task_id": TASK,
              "input": artifact_ref(CHIPS_R0), "old": artifact_ref(OLD),
              "mount": artifact_ref(MOUNT), "output": artifact_ref(output),
              "changed_variable": "status_0_valid_pose_seeding_only",
              "fixed_control_prefix_identical": bool(unchanged_prefix),
              "old_window_max_joint_step_rad": before_step,
              "new_window_max_joint_step_rad": after_step,
              "window_rows": rows,
              "quality_authority": "DIAGNOSTIC_ONLY_CURRENT_FRAME_CONVERGENCE_UNCHANGED",
              "next_action": "ASSESS_FIXED_WINDOW_AND_RECOMPUTE_DEPENDENT_SUFFIX_IF_IMPROVED"}
    atomic_json(receipt, result)
    return {"receipt": str(receipt), "old_max_step": before_step,
            "new_max_step": after_step, "fixed_prefix": unchanged_prefix}


def sensor_local_audit() -> dict:
    """Check saved Sensor q against independent FK and source local geometry."""
    _check_task()
    output = ATTEMPT / "lanes/sensor/SENSOR_LOCAL_KAI22_QUALITY_AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    assets = load_pinned_robot_assets(REPO_ROOT)
    hands = (load_hand_model(assets.left_hand.path, "left"),
             load_hand_model(assets.right_hand.path, "right"))
    mcp = np.asarray([1, 5, 9, 13, 17])
    tips = np.asarray([4, 8, 12, 16, 20])
    rows = []
    for short, wave in (("097", 0), ("098", 1), ("101", 1)):
        session = f"play_cards_0916_{short}"
        stage = SENSOR_ROOT / f"run_{short}_wave{wave}"
        hand_path = SENSOR_HAND_ROOT / session / "HAND_MOTION_V1.npz"
        backend_path = stage / "KAI22_COMMON_BACKEND_V1.npz"
        collision_path = stage / ("COLLISION_AUDIT.json" if short == "097" else "RESULT.json")
        hand, backend, collision = _load(hand_path), _load(backend_path), load_json(collision_path)
        if not np.array_equal(hand["frame_id"], backend["source_frame_id"]):
            raise RuntimeError("SENSOR_FRAME_DRIFT")
        if not np.array_equal(hand["timestamp_ns"], backend["timestamp_ns"]):
            raise RuntimeError("SENSOR_TIME_DRIFT")
        if not np.array_equal(backend["human_to_physical"], [1, 0]):
            raise RuntimeError("SENSOR_SIDE_DRIFT")
        valid = backend["valid"].astype(bool)
        if int(valid.sum()) != int(collision["collision"]["evaluated_side_frames"]):
            raise RuntimeError("COLLISION_DENOMINATOR_DRIFT")
        if not collision["collision"]["pass"]:
            raise RuntimeError("COLLISION_AUDIT_NOT_PASS")
        independent_fk_error = []
        source_mcp = []
        source_tip = []
        robot_mcp = []
        robot_tip = []
        scaled_target_tip = []
        for frame, physical in np.argwhere(valid):
            frame, physical = int(frame), int(physical)
            anatomical = int(np.flatnonzero(backend["human_to_physical"] == physical)[0])
            q = backend["q22"][frame, physical]
            if not np.array_equal(backend["joint_names"][physical], hands[physical].joint_names):
                raise RuntimeError("SENSOR_JOINT_ORDER_DRIFT")
            fk = keypoints_from_q(hands[physical], q)
            fk -= fk[0]
            independent_fk_error.append(float(np.max(np.abs(
                fk - backend["fk21_root_relative"][frame, physical]))))
            source = hand["manus_local_21_m"][frame, anatomical]
            target = backend["target21_robot_frame"][frame, physical]
            source_mcp.append(float(np.median(np.linalg.norm(source[mcp] - source[0], axis=1))))
            source_tip.append(float(np.median(np.linalg.norm(source[tips] - source[0], axis=1))))
            robot_mcp.append(float(np.median(np.linalg.norm(fk[mcp] - fk[0], axis=1))))
            robot_tip.append(float(np.median(np.linalg.norm(fk[tips] - fk[0], axis=1))))
            scaled_target_tip.append(float(np.median(np.linalg.norm(target[tips] - target[0], axis=1))))
        max_fk = max(independent_fk_error, default=None)
        if max_fk is None or max_fk > 1e-8:
            raise RuntimeError(f"SENSOR_STORED_FK_MISMATCH:{session}:{max_fk}")
        rows.append({"session_id": session, "frames": int(len(hand["frame_id"])),
                     "valid_side_frames": int(valid.sum()),
                     "collision_scope": collision["collision"],
                     "independent_fk_max_abs_m": max_fk,
                     "source_mcp_median_m": float(np.median(source_mcp)),
                     "source_tip_median_m": float(np.median(source_tip)),
                     "robot_mcp_median_m": float(np.median(robot_mcp)),
                     "robot_tip_median_m": float(np.median(robot_tip)),
                     "saved_scaled_target_tip_median_m": float(np.median(scaled_target_tip)),
                     "local_shape_quality": "NOT_ESTABLISHED_SCALE_SEMANTICS_MISMATCH",
                     "kai22_training_eligible_h50": 0,
                     "inputs": {"manus": artifact_ref(hand_path), "kai22": artifact_ref(backend_path),
                                "collision": artifact_ref(collision_path)}})
    result = {"schema_version": "SENSOR_LOCAL_KAI22_QUALITY_AUDIT_V1", "task_id": TASK,
              "sessions": rows,
              "finding": "Saved targets have much longer fingertip radius than independent Kai22 FK despite similar root-to-MCP radius; current root-to-MCP scale is not a validated local-shape authority.",
              "consumer_qualification": "KINEMATIC_ONLY_NO_LOCAL_SHAPE_TRAINING_LABEL",
              "next_action": "TRACE_MANUS_ROOT_SEMANTICS_AND_TARGET_SCALING_WITH_FIXED_097_WINDOW_BEFORE_ANY_REBRANDING",
              "training_eligible": False, "control_ground_truth": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"receipt": str(output), "sessions": len(rows),
            "valid_side_frames": [row["valid_side_frames"] for row in rows]}


def poker_scene_mask_audit() -> dict:
    """Read-only evidence: compare current rejected support with accepted historical support.

    The archived user acceptance applies only to the archived Clean and does
    not promote this task's exact-domain Scene or product quality.
    """
    _check_task()
    output = ATTEMPT / "lanes/scene/POKER_ACCEPTED_MASK_VS_CURRENT_AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    current_path = POKER_SCENE / "SCENE_PREP_MANIFEST.json"
    historical_path = POKER_ACCEPTED / "poker_expanded_role_handoff_v3/FRAME_MANIFEST.json"
    acceptance_path = POKER_ACCEPTED.parent / "CLEAN_EXPANDED_V3_USER_ACCEPTANCE_20260909.json"
    current = load_json(current_path)
    historical = load_json(historical_path)
    acceptance = load_json(acceptance_path)
    if (current["session_id"] != "play_cards_0902_042"
            or historical["session"] != current["session_id"]
            or len(current["rows"]) != 171 or len(historical["frames"]) != 171):
        raise RuntimeError("POKER_SCENE_FRAME_OR_SESSION_DRIFT")
    rows = []
    for index, (new, old) in enumerate(zip(current["rows"], historical["frames"], strict=True)):
        if int(new["frame_id"]) != index or int(old["source_frame"]) != index:
            raise RuntimeError(f"POKER_FRAME_DRIFT:{index}")
        old_rgb = cv2.imread(old["source_rgb"]["path"], cv2.IMREAD_COLOR)
        new_rgb = cv2.imread(new["raw"], cv2.IMREAD_COLOR)
        old_mask_path = _archived_poker_path(old["clean_removal_object_protected"]["path"])
        old_mask = cv2.imread(str(old_mask_path), cv2.IMREAD_GRAYSCALE)
        new_mask = cv2.imread(new["write"], cv2.IMREAD_GRAYSCALE)
        if any(value is None for value in (old_rgb, new_rgb, old_mask, new_mask)):
            raise RuntimeError(f"POKER_SOURCE_OR_MASK_MISSING:{index}")
        if old_rgb.shape != new_rgb.shape or old_mask.shape != new_mask.shape != old_rgb.shape[:2]:
            raise RuntimeError(f"POKER_SOURCE_DOMAIN_SIZE_MISMATCH:{index}")
        difference = cv2.absdiff(old_rgb, new_rgb)
        older, newer = old_mask > 0, new_mask > 0
        rows.append({"frame_id": index,
                     "source_exact": bool(np.array_equal(old_rgb, new_rgb)),
                     "source_max_abs_channel": int(difference.max()),
                     "source_mean_abs_channel": float(difference.mean()),
                     "historical_write_px": int(older.sum()),
                     "current_write_px": int(newer.sum()),
                     "historical_only_px": int((older & ~newer).sum()),
                     "current_only_px": int((newer & ~older).sum()),
                     "historical_support": artifact_ref(old_mask_path),
                     "current_support": artifact_ref(Path(new["write"]))})
    accepted = acceptance["accepted"]["poker"]
    result = {
        "schema_version": "POKER_HISTORICAL_MASK_COMPARISON_V1", "task_id": TASK,
        "session_id": current["session_id"], "frame_count": len(rows),
        "historical_user_acceptance": artifact_ref(acceptance_path),
        "historical_scope": accepted["scope"], "historical_video_sha256": accepted["video_sha256"],
        "historical_mask_manifest": artifact_ref(historical_path),
        "current_scene_prep": artifact_ref(current_path),
        "source_exact_frames": sum(row["source_exact"] for row in rows),
        "source_max_abs_channel": max(row["source_max_abs_channel"] for row in rows),
        "historical_only_px_total": sum(row["historical_only_px"] for row in rows),
        "current_only_px_total": sum(row["current_only_px"] for row in rows),
        "complaint_window": rows[76:92],
        "rows": rows,
        "quality_authority": "DIAGNOSTIC_ONLY_HISTORICAL_ACCEPTANCE_NOT_CURRENT_PRODUCT_PASS",
        "next_action": "TRACE_HISTORICAL_ROLE_IDENTITY_AND_CURRENT_REFERENCE_CONSUMPTION_BEFORE_SINGLE_CORRECTED_CLEAN_CANDIDATE",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"receipt": str(output), "exact_frames": result["source_exact_frames"],
            "historical_only_px": result["historical_only_px_total"]}


def poker_motion_audit() -> dict:
    """Reproduce the frozen old solver around Poker jumps, without changing q."""
    _check_task()
    from scipy.optimize import least_squares
    from chaoyang.pipeline import robot_scene_state_cpu as arm
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES
    output = ATTEMPT / "lanes/motion_product/POKER_096_111_SOLVER_CAUSAL_AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    saved, source = _load(POKER_R0), _load(POKER_SOURCE)
    assets = load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm._arm_limits(assets)
    neutral = (lower + upper) / 2
    fk = forward_kinematics(assets.tianji, {
        name: float(neutral[side, j]) for side in range(2)
        for j, name in enumerate(ARM_JOINT_NAMES[side])})
    neutral_roots = np.stack([fk["flange_L"], fk["flange_R"]]) @ saved["T_flange_hand"]
    tool_mounts = np.stack([np.linalg.inv(fk["left_tool"]) @ neutral_roots[0],
                            np.linalg.inv(fk["right_tool"]) @ neutral_roots[1]])
    spec = importlib.util.spec_from_file_location("frozen_poker_camera_solver", POKER_PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError("FROZEN_POKER_PRODUCER_NOT_LOADABLE")
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    side = 1  # frozen physical-right anomaly
    hands = (load_hand_model(assets.left_hand.path, "left"),
             load_hand_model(assets.right_hand.path, "right"))
    rows = []
    for frame in (98, 99, 100, 104, 105):
        target = saved["T_target_root_base"][frame, side] @ np.linalg.inv(tool_mounts[side])
        dt = float((saved["timestamp_ns"][frame] - saved["timestamp_ns"][frame-1]) / 1e9)
        if dt <= 0:
            raise RuntimeError("POKER_TIME_NONPOSITIVE")
        use_previous = bool(saved["arm_solver_success"][frame-1, side])
        old_seed = saved["q_arm"][frame-1, side] if use_previous else neutral[side]
        temporal = .001 * (1 / 30) / dt if use_previous else 0.0
        def residual(q: np.ndarray) -> np.ndarray:
            return np.r_[arm._pose_residual(arm._tool_fk(assets, side, q), target),
                         temporal * (q - old_seed)]
        old_fit = least_squares(residual, old_seed, bounds=(lower[side], upper[side]),
                                max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        old_distance = float(np.max(np.abs(old_fit.x - saved["q_arm"][frame, side])))
        row = {"frame_id": frame, "target_valid": bool(saved["target_valid"][frame, side]),
               "saved_arm_success": bool(saved["arm_solver_success"][frame, side]),
               "saved_arm_nfev": int(saved["solver_evaluations"][frame, side, 0]),
               "reproduced_arm_success": bool(old_fit.success),
               "reproduced_arm_status": int(old_fit.status), "reproduced_arm_nfev": int(old_fit.nfev),
               "reproduced_arm_q_max_abs_difference": old_distance,
               "saved_position_mm": float(saved["position_residual_mm"][frame, side]),
               "saved_rotation_deg": float(saved["rotation_residual_deg"][frame, side])}
        if frame == 100:
            previous_seed = saved["q_arm"][99, side]
            previous_temporal = .001 * (1 / 30) / dt
            def recovered_residual(q: np.ndarray) -> np.ndarray:
                return np.r_[arm._pose_residual(arm._tool_fk(assets, side, q), target),
                             previous_temporal * (q - previous_seed)]
            recovered = least_squares(recovered_residual, previous_seed,
                                       bounds=(lower[side], upper[side]), max_nfev=150,
                                       ftol=1e-9, xtol=1e-9, gtol=1e-9)
            row["status0_seed_counterfactual"] = {
                "status": int(recovered.status), "success": bool(recovered.success),
                "nfev": int(recovered.nfev),
                "joint_step_from_99_rad": float(np.max(np.abs(recovered.x - previous_seed))),
                "old_joint_step_from_99_rad": float(np.max(np.abs(saved["q_arm"][100, side] - previous_seed))),
                "q": recovered.x.tolist()}
        if frame == 105:
            target_hand = saved["target21_root"][frame, side]
            qhand, hand_ok, nfev, points = producer.fit_fingers(
                hands[side], target_hand, saved["q22"][104, side], dt)
            old_q = saved["q22"][frame, side]
            old_fk = keypoints_from_q(hands[side], old_q)
            target_step = np.linalg.norm(saved["target21_root"][105, side]
                                         - saved["target21_root"][104, side], axis=1)
            row["finger_104_to_105"] = {
                "saved_joint_max_step_rad": float(np.max(np.abs(old_q - saved["q22"][104, side]))),
                "saved_joint_max_step_index": int(np.argmax(np.abs(old_q - saved["q22"][104, side]))),
                "source21_max_step_mm": float(np.nanmax(np.linalg.norm(
                    source["joints_3d_camera"][:, 105] - source["joints_3d_camera"][:, 104], axis=2)) * 1000),
                "target21_max_step_mm": float(np.max(target_step) * 1000),
                "independent_fingertip_fk_max_step_mm": float(np.max(np.linalg.norm(
                    old_fk[[4, 8, 12, 16, 20]] - keypoints_from_q(
                        hands[side], saved["q22"][104, side])[[4, 8, 12, 16, 20]], axis=1)) * 1000),
                "reproduced_solver_success": bool(hand_ok), "reproduced_nfev": int(nfev),
                "reproduced_q_max_abs_difference": float(np.max(np.abs(qhand - old_q))),
                "reproduced_fk_max_abs_difference_m": float(np.max(np.abs(points - old_fk))),
                "joint_name": str(hands[side].joint_names[int(np.argmax(np.abs(old_q - saved["q22"][104, side])))]),
            }
        rows.append(row)
    result = {"schema_version": "POKER_SOLVER_CAUSAL_AUDIT_V1", "task_id": TASK,
              "session_id": "play_cards_0902_042", "frozen_producer": artifact_ref(POKER_PRODUCER),
              "source": artifact_ref(POKER_SOURCE), "saved_robot": artifact_ref(POKER_R0),
              "rows": rows, "quality_authority": "DIAGNOSTIC_ONLY_NO_Q_REPLACEMENT",
              "next_action": "ASSESS_REPRODUCIBILITY_AND_FIX_FIRST_CONFIRMED_LAYER_ONLY"}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"receipt": str(output), "rows": len(rows)}


def poker_seed_suffix() -> dict:
    """Recompute only the right-arm suffix after the reproduced status-0 reset.

    This is a new numeric candidate, not a quality-pass or an edit to V3.
    Finger q and the independent left side remain byte-identical to V3.
    """
    _check_task()
    from scipy.optimize import least_squares
    from chaoyang.pipeline import robot_scene_state_cpu as arm
    from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES
    output = ATTEMPT / "lanes/motion_product/POKER_SEED_RECOVERY_SUFFIX_V1.npz"
    receipt = output.with_suffix(".json")
    if output.exists() or receipt.exists():
        raise FileExistsError("POKER_SEED_SUFFIX_ALREADY_EXISTS")
    audit = load_json(ATTEMPT / "lanes/motion_product/POKER_096_111_SOLVER_CAUSAL_AUDIT.json")
    frame99 = next(row for row in audit["rows"] if row["frame_id"] == 99)
    if (frame99["reproduced_arm_status"] != 0
            or frame99["reproduced_arm_q_max_abs_difference"] > 1e-8):
        raise RuntimeError("STATUS0_CAUSE_NOT_REPRODUCED")
    old = _load(POKER_R0)
    assets = load_pinned_robot_assets(REPO_ROOT)
    lo, hi = arm._arm_limits(assets)
    neutral = (lo + hi) / 2
    base_fk = forward_kinematics(assets.tianji, {
        name: float(neutral[physical, j]) for physical in range(2)
        for j, name in enumerate(ARM_JOINT_NAMES[physical])})
    neutral_roots = np.stack([base_fk["flange_L"], base_fk["flange_R"]]) @ old["T_flange_hand"]
    tool_mounts = np.stack([np.linalg.inv(base_fk["left_tool"]) @ neutral_roots[0],
                            np.linalg.inv(base_fk["right_tool"]) @ neutral_roots[1]])
    if not np.allclose(neutral_roots, old["neutral_roots"], atol=1e-8):
        raise RuntimeError("POKER_MOUNT_RECONSTRUCTION_DRIFT")
    new = {key: value.copy() for key, value in old.items()}
    n = len(old["frame_id"])
    side = 1
    edge, times = time_edges(old["timestamp_ns"], old["frame_id"])
    if not bool(old["target_valid"][99, side]) or not edge[100]:
        raise RuntimeError("POKER_STATUS0_NOT_SAME_VALID_SEGMENT")
    status = np.full((n, 2), -999, dtype=np.int32)
    status[99, side] = 0
    seed_source = np.full((n, 2), -1, dtype=np.int64)
    seed_usable = np.zeros((n, 2), dtype=bool)
    hand = load_hand_model(assets.right_hand.path, "right")
    previous_q = old["q_arm"][99, side].copy()
    previous_good = seed_usable_solution(
        previous_q, lo[side], hi[side], solver_success=False, solver_status=0,
        position_mm=float(old["position_residual_mm"][99, side]),
        rotation_deg=float(old["rotation_residual_deg"][99, side]),
        independent_fk=old["T_target_root_base"][99, side])
    # The helper requires finite FK; actual independent parity is checked below.
    old_joints = {name: float(neutral[p, j]) for p in range(2)
                  for j, name in enumerate(ARM_JOINT_NAMES[p])}
    old_joints.update({name: float(q) for name, q in zip(ARM_JOINT_NAMES[side], previous_q, strict=True)})
    independent99 = forward_kinematics(assets.tianji, old_joints)["flange_R"] @ old["T_flange_hand"][side]
    expected99 = arm._tool_fk(assets, side, previous_q) @ tool_mounts[side]
    previous_good &= bool(np.allclose(independent99, expected99, atol=3e-6))
    if not previous_good:
        raise RuntimeError("POKER_99_SEED_NOT_LEGAL")
    seed_usable[99, side] = True
    rows = []
    end = 99
    for frame in range(100, n):
        if not old["target_valid"][frame, side] or not edge[frame]:
            break  # the next segment is independent and the old suffix is reusable
        dt = (old["timestamp_ns"][frame] - old["timestamp_ns"][frame-1]) / 1e9
        seed = previous_q if previous_good else neutral[side]
        temporal = .001 * (1 / 30) / dt if previous_good else 0.0
        seed_source[frame, side] = frame - 1 if previous_good else -1
        target = old["T_target_root_base"][frame, side] @ np.linalg.inv(tool_mounts[side])
        def residual(q: np.ndarray) -> np.ndarray:
            return np.r_[arm._pose_residual(arm._tool_fk(assets, side, q), target),
                         temporal * (q - seed)]
        fit = least_squares(residual, seed, bounds=(lo[side], hi[side]), max_nfev=150,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        root_base = arm._tool_fk(assets, side, fit.x) @ tool_mounts[side]
        joint_map = {name: float(neutral[p, j]) for p in range(2)
                     for j, name in enumerate(ARM_JOINT_NAMES[p])}
        joint_map.update({name: float(q) for name, q in zip(ARM_JOINT_NAMES[side], fit.x, strict=True)})
        independent = forward_kinematics(assets.tianji, joint_map)["flange_R"] @ old["T_flange_hand"][side]
        if not np.allclose(independent, root_base, atol=3e-6):
            raise RuntimeError(f"POKER_INDEPENDENT_FK_MISMATCH:{frame}")
        root_camera = old["T_cam_base"] @ root_base
        delta = np.linalg.inv(old["T_target_root_cam"][frame, side]) @ root_camera
        pos = float(np.linalg.norm(delta[:3, 3]) * 1000)
        rot = float(np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3]))))
        legal = seed_usable_solution(fit.x, lo[side], hi[side],
                                     solver_success=bool(fit.success), solver_status=int(fit.status),
                                     position_mm=pos, rotation_deg=rot, independent_fk=independent)
        new["q_arm"][frame, side] = fit.x
        new["T_actual_root_cam"][frame, side] = root_camera
        new["position_residual_mm"][frame, side] = pos
        new["rotation_residual_deg"][frame, side] = rot
        new["arm_solver_success"][frame, side] = fit.success
        new["solver_evaluations"][frame, side, 0] = fit.nfev
        new["actual21_camera"][frame, side] = (
            keypoints_from_q(hand, old["q22"][frame, side]) @ root_camera[:3, :3].T
            + root_camera[:3, 3])
        new["tolerance_pass"][frame, side] = bool(fit.success and pos <= 20 and rot <= 15)
        status[frame, side] = fit.status
        seed_usable[frame, side] = legal
        rows.append({"frame_id": frame, "seed_source_frame": int(seed_source[frame, side]),
                     "status": int(fit.status), "solver_success": bool(fit.success),
                     "nfev": int(fit.nfev), "position_mm": pos, "rotation_deg": rot,
                     "seed_usable_for_next": legal,
                     "joint_step_rad": float(np.max(np.abs(fit.x - new["q_arm"][frame-1, side]))),
                     "tolerance_pass": bool(new["tolerance_pass"][frame, side])})
        previous_q, previous_good = fit.x.copy(), legal
        end = frame
    derivatives = motion_derivatives(new["q_arm"], old["wrist_valid"], times, edge)
    for key, value in derivatives.items():
        new[f"arm_{key}"] = value
    new["arm_solver_status"] = status
    new["seed_source_frame"] = seed_source
    new["seed_usable"] = seed_usable
    if not np.array_equal(new["q_arm"][:, 0], old["q_arm"][:, 0], equal_nan=True):
        raise RuntimeError("INDEPENDENT_LEFT_SIDE_CHANGED")
    if not np.array_equal(new["q22"], old["q22"], equal_nan=True):
        raise RuntimeError("FINGER_Q_CHANGED")
    if not np.array_equal(new["q_arm"][:100], old["q_arm"][:100], equal_nan=True):
        raise RuntimeError("PRE_CAUSAL_PREFIX_CHANGED")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **new)
    with np.load(output, allow_pickle=False) as reloaded:
        if not np.array_equal(reloaded["frame_id"], old["frame_id"]):
            raise RuntimeError("RELOADED_FRAME_MAP_CHANGED")
    old_step = float(np.max(np.abs(old["q_arm"][100, side] - old["q_arm"][99, side])))
    new_step = float(np.max(np.abs(new["q_arm"][100, side] - new["q_arm"][99, side])))
    result = {"schema_version": "POKER_STATUS0_SEED_SUFFIX_V1", "task_id": TASK,
              "source": artifact_ref(POKER_R0), "audit": artifact_ref(
                  ATTEMPT / "lanes/motion_product/POKER_096_111_SOLVER_CAUSAL_AUDIT.json"),
              "output": artifact_ref(output), "changed_side": "physical_right",
              "first_recomputed_frame": 100, "last_recomputed_frame": end,
              "old_99_to_100_max_joint_step_rad": old_step,
              "new_99_to_100_max_joint_step_rad": new_step,
              "old_right_tolerance_pass": int(old["tolerance_pass"][:, side].sum()),
              "new_right_tolerance_pass": int(new["tolerance_pass"][:, side].sum()),
              "frame_99_original_convergence_failure_preserved": bool(
                  not new["arm_solver_success"][99, side]),
              "finger_104_105_jump_unmodified": True,
              "rows": rows, "quality_authority": "NUMERIC_CANDIDATE_ONLY_NOT_PRODUCT_PASS",
              "next_action": "EVALUATE_FULL_TIMELINE_CONTINUITY_COLLISION_AND_FINGER_JUMP_BEFORE_PRODUCT_BINDING"}
    atomic_json(receipt, result)
    return {"receipt": str(receipt), "frames_recomputed": len(rows),
            "old_step": old_step, "new_step": new_step}


def poker_historical_clean_rebase() -> dict:
    """Rebase already-accepted synthetic pixels onto current exact-domain Raw.

    This does not rerun ProPainter and does not inherit historical acceptance.
    All output outside the historical removal support is current Raw byte-for-byte.
    """
    _check_task()
    root = ATTEMPT / "lanes/scene/POKER_HISTORICAL_CLEAN_EXACT_REBASE"
    if root.exists():
        raise FileExistsError(root)
    audit_path = ATTEMPT / "lanes/scene/POKER_ACCEPTED_MASK_VS_CURRENT_AUDIT.json"
    audit = load_json(audit_path)
    if (audit["frame_count"] != 171 or audit["source_max_abs_channel"] > 4
            or max(row["source_mean_abs_channel"] for row in audit["rows"]) > .05):
        raise RuntimeError("HISTORICAL_SOURCE_NOT_PIXEL_ALIGNED")
    current = load_json(POKER_SCENE / "SCENE_PREP_MANIFEST.json")
    archived_root = POKER_ACCEPTED / "play_cards_0902_042_expanded_role_v3"
    historical = load_json(POKER_ACCEPTED / "poker_expanded_role_handoff_v3/FRAME_MANIFEST.json")
    (root / "clean").mkdir(parents=True)
    rows = []
    for frame, (source, old, comparison) in enumerate(zip(
            current["rows"], historical["frames"], audit["rows"], strict=True)):
        raw = cv2.imread(source["raw"], cv2.IMREAD_COLOR)
        historical_clean_path = archived_root / "clean_frames" / f"{frame:06d}.png"
        old_clean = cv2.imread(str(historical_clean_path), cv2.IMREAD_COLOR)
        support_path = _archived_poker_path(old["clean_removal_object_protected"]["path"])
        protect_path = _archived_poker_path(old["physical_object_0"]["path"])
        support = cv2.imread(str(support_path), cv2.IMREAD_GRAYSCALE)
        protect = cv2.imread(str(protect_path), cv2.IMREAD_GRAYSCALE)
        if any(x is None for x in (raw, old_clean, support, protect)):
            raise RuntimeError(f"HISTORICAL_CLEAN_FRAME_MISSING:{frame}")
        if raw.shape != old_clean.shape or raw.shape[:2] != support.shape != protect.shape:
            raise RuntimeError(f"HISTORICAL_CLEAN_SIZE_DRIFT:{frame}")
        write, guarded = support > 0, protect > 0
        if np.any(write & guarded) or int(old["published_removal_object_overlap_pixels"]) != 0:
            raise RuntimeError(f"HISTORICAL_OBJECT_PROTECTION_CONFLICT:{frame}")
        clean = raw.copy()
        clean[write] = old_clean[write]
        path = root / "clean" / f"{frame:06d}.png"
        if not cv2.imwrite(str(path), clean):
            raise RuntimeError(f"REBASED_CLEAN_WRITE_FAILED:{frame}")
        if not np.array_equal(clean[~write], raw[~write]):
            raise RuntimeError(f"CURRENT_RAW_OUTSIDE_WRITE_CHANGED:{frame}")
        rows.append({"frame_id": frame, "clean": artifact_ref(path),
                     "current_raw": artifact_ref(Path(source["raw"])),
                     "historical_clean": artifact_ref(historical_clean_path),
                     "write": artifact_ref(support_path),
                     "historical_protect": artifact_ref(protect_path),
                     "changed_px": int(np.any(clean != raw, axis=2).sum()),
                     "outside_write_changed_px": 0, "protected_changed_px": 0,
                     "source_max_abs_channel": comparison["source_max_abs_channel"]})
    video = root / "CLEAN_EXACT_REBASE_CANDIDATE.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-y",
               "-framerate", "30", "-i", str(root / "clean/%06d.png"),
               "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "19", "-pix_fmt", "yuv420p", str(video)]
    complete = subprocess.run(command, capture_output=True, text=True)
    if complete.returncode:
        raise RuntimeError("CLEAN_REVIEW_ENCODE:" + complete.stderr[-1000:])
    cap = cv2.VideoCapture(str(video)); decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != 171:
        raise RuntimeError(f"CLEAN_REVIEW_DECODE:{decoded}")
    result = {"schema_version": "POKER_HISTORICAL_CLEAN_EXACT_REBASE_V1",
              "task_id": TASK, "session_id": "play_cards_0902_042", "frame_count": 171,
              "execution": "EXECUTED_REBASE_NO_NEW_PROPAINTER_RUN",
              "structure": "PASS", "quality": "INCONCLUSIVE_PENDING_INDEPENDENT_CURRENT_REVIEW",
              "adoption": "NOT_ADOPTED_CURRENT_PRODUCT", "source_kind":
                  "HISTORICAL_USER_ACCEPTED_SYNTHETIC_CLEAN_REBASED_TO_CURRENT_RAW",
              "historical_acceptance_scope": audit["historical_scope"],
              "historical_acceptance": audit["historical_user_acceptance"],
              "domain_alignment_audit": artifact_ref(audit_path),
              "outside_write_changed_px": 0, "protected_changed_px": 0,
              "video": artifact_ref(video), "decoded_frames": decoded,
              "rows": rows,
              "claim_limit": "Archived user acceptance applies to archived pixels only; current exact-domain rebase is a new offline candidate with no automatic quality or adoption authority.",
              "next_action": "INDEPENDENTLY_REVIEW_FIXED_76_91_AND_CONTROLS_THEN_BIND_ONLY_AS_EXPLICIT_PRODUCT_CANDIDATE"}
    atomic_json(root / "RESULT.json", result)
    return {"result": str(root / "RESULT.json"), "video": str(video), "decoded": decoded}


def assess_product_gates(gates: dict[str, bool], *, user_confirmed_sha: str | None = None,
                         product_sha: str | None = None) -> dict:
    """Quality and adoption are separate; missing required evidence is failure."""
    required = ("source_time_domain", "motion_tracking", "finger_motion",
                "scene_clean", "occlusion", "full_decode_and_sha", "visual_review")
    quality = all(gates.get(key) is True for key in required)
    adopted = bool(quality and product_sha and user_confirmed_sha == product_sha)
    return {"quality": "PASS" if quality else "REJECTED_QUALITY",
            "adoption": "ADOPTED" if adopted else "NOT_ADOPTED",
            "failed_or_missing_gates": [key for key in required if gates.get(key) is not True]}


def poker_technical_quality() -> dict:
    """Independently score the currently published Poker product, never a new unbound q."""
    _check_task()
    output = ATTEMPT / "lanes/motion_product/POKER_171_TECHNICAL_QUALITY.json"
    if output.exists():
        raise FileExistsError(output)
    product = load_json(POKER_OLD_PRODUCT)
    scene = load_json(POKER_SCENE / "RESULT.json")
    original = _load(POKER_R0)
    if (product["session_id"] != "play_cards_0902_042"
            or product["expected_frames"] != 171 or product["decoded_frames"] != 171):
        raise RuntimeError("POKER_PRODUCT_SESSION_OR_FRAME_DRIFT")
    video = Path(product["product_video"]["path"])
    if artifact_ref(video) != product["product_video"]:
        raise RuntimeError("POKER_PRODUCT_VIDEO_SHA_DRIFT")
    product_motion_path = Path(product["robot_r0"]["path"])
    if artifact_ref(product_motion_path) != product["robot_r0"]:
        raise RuntimeError("POKER_PRODUCT_ACTUAL_Q_SHA_DRIFT")
    product_motion = _load(product_motion_path)
    if (not np.array_equal(product_motion["q_arm"], original["q_arm"], equal_nan=True)
            or not np.array_equal(product_motion["q_hand22"], original["q22"], equal_nan=True)):
        raise RuntimeError("POKER_PRODUCT_ACTUAL_Q_BINDING_DRIFT")
    valid = original["target_valid"].astype(bool)
    arm_pass = (original["arm_solver_success"] & (original["position_residual_mm"] <= 20)
                & (original["rotation_residual_deg"] <= 15))
    finger_pass = original["finger_solver_success"] & original["finger_valid"]
    motion_fail = np.argwhere(valid & ~arm_pass)
    finger_fail = np.argwhere(valid & ~finger_pass)
    unknown = int(product["ownership_counts"]["UNKNOWN"])
    gates = {
        "source_time_domain": bool(len(original["frame_id"]) == 171
                                   and np.array_equal(original["frame_id"], np.arange(171))
                                   and np.all(np.diff(original["timestamp_ns"]) > 0)),
        "motion_tracking": bool(np.all(arm_pass[valid]) and valid.all()),
        "finger_motion": bool(np.all(finger_pass[valid]) and valid.all()),
        "scene_clean": bool(scene.get("quality") == "PASS" and
                            product["scene_clean_manifest"] == artifact_ref(POKER_SCENE / "RESULT.json")),
        "occlusion": bool(unknown == 0 and product.get("geometry_mode") == "VISIBLE_SURFACE_DEPTH"),
        "full_decode_and_sha": True,
        "visual_review": False,  # fixed 81/90 still show card-edge smearing
    }
    decision = assess_product_gates(gates, product_sha=product["product_video"]["sha256"])
    suffix = load_json(ATTEMPT / "lanes/motion_product/POKER_SEED_RECOVERY_SUFFIX_V1.json")
    result = {
        "schema_version": "POKER_PRODUCT_INDEPENDENT_TECHNICAL_QUALITY_V1",
        "task_id": TASK, "session_id": "play_cards_0902_042", "original_denominator": 171,
        "published_product": artifact_ref(POKER_OLD_PRODUCT),
        "published_product_video": product["product_video"],
        "actually_bound_motion": product["robot_r0"],
        "actually_bound_scene": product["scene_clean_manifest"],
        "gates": gates, "decision": decision,
        "motion_valid_side_frames": int(valid.sum()),
        "motion_arm_pass_side_frames": int(arm_pass[valid].sum()),
        "motion_fail_first_20": motion_fail[:20].tolist(),
        "finger_fail_first_20": finger_fail[:20].tolist(),
        "occlusion_unknown_overlap_pixels": unknown,
        "new_scene_candidate": artifact_ref(ATTEMPT / "lanes/scene/POKER_HISTORICAL_CLEAN_EXACT_REBASE/RESULT.json"),
        "new_scene_candidate_not_bound_to_published_product": True,
        "new_motion_candidate": artifact_ref(ATTEMPT / "lanes/motion_product/POKER_SEED_RECOVERY_SUFFIX_V1.json"),
        "new_motion_candidate_rejected_tracking_pass_regression": bool(
            suffix["new_right_tolerance_pass"] < suffix["old_right_tolerance_pass"]),
        "visual_review_scope": {"frames_inspected": [81, 90],
                                "finding": "VISIBLE_CARD_EDGE_SMEAR_REMAINS_IN_NEW_CLEAN_REBASE",
                                "quality": "REJECTED_ON_FIXED_COMPLAINT_SAMPLES"},
        "result": "QUALITY_0_OF_4_UNCHANGED_POKER_NOT_TECHNICALLY_PASSABLE_YET",
        "next_action": "RUN_ONE_EVIDENCE_DRIVEN_CURRENT_FRAME_CARD_PROTECTION_REPAIR_AND_REEVALUATE_SCENE;_KEEP_MOTION_AND_DEPTH_GATES_SEPARATE",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"receipt": str(output), "quality": decision["quality"],
            "failed": decision["failed_or_missing_gates"]}


def publish_progress() -> dict:
    """Project immutable run receipts into the four existing lane state slots."""
    _check_task()
    quality = ATTEMPT / "lanes/motion_product/POKER_171_TECHNICAL_QUALITY.json"
    receipt = load_json(quality)
    if receipt["decision"]["quality"] != "REJECTED_QUALITY":
        raise RuntimeError("QUALITY_PROGRESS_NOT_REVIEWED")
    now = datetime.now(timezone.utc).isoformat()
    evidence = {
        "scene": [ATTEMPT / "lanes/scene/POKER_ACCEPTED_MASK_VS_CURRENT_AUDIT.json",
                  ATTEMPT / "lanes/scene/POKER_HISTORICAL_CLEAN_EXACT_REBASE/RESULT.json",
                  ATTEMPT / "lanes/scene/POKER_076_091_CURRENT_CARD_SUPPORT/RESULT.json",
                  ATTEMPT / "lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN/VISUAL_REVIEW.json",
                  ATTEMPT / "lanes/scene/POKER_0902_STEREO_G0.json"],
        "motion_product": [ATTEMPT / "lanes/motion_product/CHIPS_SEED_COUNTERFACTUAL_0_160.json",
                           ATTEMPT / "lanes/motion_product/POKER_096_111_SOLVER_CAUSAL_AUDIT.json",
                           ATTEMPT / "lanes/motion_product/POKER_104_105_FINGER_FIRST_LAYER.json",
                           ATTEMPT / "lanes/motion_product/POKER_SEED_RECOVERY_SUFFIX_V1.json", quality],
        "sensor": [ATTEMPT / "lanes/sensor/SENSOR_LOCAL_KAI22_QUALITY_AUDIT.json",
                   ATTEMPT / "lanes/sensor/SENSOR_097_ROOT_FREE_BONE_QUALITY.json",
                   ATTEMPT / "lanes/sensor/SENSOR_097_REAL_Q_FIELD_MASK_CONSUMER.json",
                   ATTEMPT / "lanes/sensor/SENSOR_097_QONLY_H50_DIAGNOSTIC.json",
                   ATTEMPT / "lanes/sensor/SENSOR_097_QONLY_REAL_RGB_FORWARD.json",
                   ATTEMPT / "lanes/sensor/SENSOR_097_NATIVE_BONE_SEMANTICS.json"],
        "compare": [REPO_ROOT / "_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/"
                    "lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json"],
    }
    next_actions = {
        "scene": "CARD_PROTECTED_16_FRAME_CLEAN_REJECTED; NO_SAME_RECIPE_171_EXPANSION; FULL_STEREO_NOT_USEFUL_FOR_PRODUCT_PASS_WHILE_MOTION_AND_CLEAN_FAIL",
        "motion_product": "OLD_Q_RETAINED; PINKY_FIRST_ERROR_IS_SOLVER_Q_AT_LIMITS; DO_NOT_BIND_REJECTED_SUFFIX",
        "sensor": "REAL_Q_H50_FORWARD_DONE_WITH_ZERO_ADMITTED_LABELS; NATIVE_BONE_SEMANTICS_REJECT_SINGLE_ROOT_SCALE; REQUIRE_NEW_VALIDATED_MAPPING",
        "compare": "NO_NEW_HURO_SOLVE; PUBLISH_REUSED_COMPARISON_SCOPE_AND_EXIT",
    }
    qualities = {"scene": "REJECTED_QUALITY_NEW_CARD_PROTECTED_16_FRAME_CLEAN",
                 "motion_product": "REJECTED_QUALITY_TRACKING_GATE_AND_FINGER_JUMP",
                 "sensor": "REAL_Q_H50_DIRECT_FORWARD_AND_RELOAD_PASS_BUT_LOCAL_SHAPE_LABEL_NOT_ADMITTED",
                 "compare": "LIMITED_REUSED_COMPARISON_NO_WINNER"}
    for lane, paths in evidence.items():
        if any(not path.is_file() for path in paths):
            raise RuntimeError(f"LANE_EVIDENCE_MISSING:{lane}")
        state = {"schema_version": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_LANE_STATE_V1",
                 "task_id": TASK, "lane": lane, "status": "ACTIVE" if lane != "compare" else "TERMINAL_LIMITED",
                 "execution": "EXECUTED_PARTIAL" if lane != "compare" else "REUSED_NO_NEW_SOLVE",
                 "structure": "PASS_FOR_REFERENCED_ARTIFACTS", "quality": qualities[lane],
                 "adoption": "NOT_ADOPTED", "evidence": [artifact_ref(path) for path in paths],
                 "next_action": next_actions[lane], "updated_at": now,
                 "training_eligible": False, "control_ground_truth": False,
                 "physical_deployable": False, "external_metric_authority": False}
        atomic_json(ATTEMPT / "lanes" / lane / "STATE.json", state)
    progress = {"schema_version": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_PROGRESS_V1",
                "task_id": TASK, "updated_at": now, "execution": "RUNNING",
                "product_counts": {"products_structure": "4/4", "products_quality": "0/4",
                                   "products_adopted": "0/4"},
                "poker_quality": artifact_ref(quality),
                "lane_states": {lane: artifact_ref(ATTEMPT / "lanes" / lane / "STATE.json")
                                for lane in evidence},
                "next_action": "NO_SAME_RECIPE_RETRY_OR_FULL_DEPTH_BATCH_CAN_CHANGE_CURRENT_REQUIRED_GATE_REJECTIONS; COMPLETE_FINAL_VALIDATION_AND_TERMINAL_REPORT",
                "claim_limit": "Partial execution; quality and adoption remain 0/4 and no rejected candidate is promoted."}
    atomic_json(ATTEMPT / "PROGRESS.json", progress)
    return {"progress": str(ATTEMPT / "PROGRESS.json"),
            "counts": progress["product_counts"]}


def sensor_root_free_shape_audit() -> dict:
    """Evaluate real Sensor local bone directions without borrowing wrist authority."""
    _check_task()
    from chaoyang.pipeline.huro_hand_frame_v2 import palm_basis
    output = ATTEMPT / "lanes/sensor/SENSOR_097_ROOT_FREE_BONE_QUALITY.json"
    if output.exists():
        raise FileExistsError(output)
    hand_path = SENSOR_HAND_ROOT / "play_cards_0916_097/HAND_MOTION_V1.npz"
    backend_path = SENSOR_ROOT / "run_097_wave0/KAI22_COMMON_BACKEND_V1.npz"
    hand, backend = _load(hand_path), _load(backend_path)
    assets = load_pinned_robot_assets(REPO_ROOT)
    models = (load_hand_model(assets.left_hand.path, "left"),
              load_hand_model(assets.right_hand.path, "right"))
    if (not np.array_equal(hand["frame_id"], backend["source_frame_id"])
            or not np.array_equal(hand["timestamp_ns"], backend["timestamp_ns"])):
        raise RuntimeError("SENSOR_097_FRAME_TIME_DRIFT")
    n = len(hand["frame_id"])
    if n != 165:
        raise RuntimeError("SENSOR_097_FRAME_COUNT_DRIFT")
    pairs = []
    for chain in FINGER_CHAINS:
        pairs.extend((int(chain[j-1]), int(chain[j])) for j in range(1, len(chain)))
    mcp_pairs = [(1, 5), (5, 9), (9, 13), (13, 17)]
    by_side = []
    for physical, model in enumerate(models):
        anatomical = int(np.flatnonzero(backend["human_to_physical"] == physical)[0])
        neutral = keypoints_from_q(model, .5 * (model.lower + model.upper))
        robot_basis = palm_basis(neutral)
        rows = []
        valid = backend["valid"][:, physical].astype(bool)
        for frame in np.flatnonzero(valid):
            source = hand["manus_local_21_m"][frame, anatomical]
            fk = keypoints_from_q(model, backend["q22"][frame, physical])
            if not np.isfinite(source).all():
                raise RuntimeError(f"VALID_SENSOR_SOURCE_NONFINITE:{frame}:{physical}")
            source_basis = palm_basis(source)
            source_aligned = (source - source[0]) @ source_basis @ robot_basis.T
            angles = []
            length_ratios = []
            for parent, child in pairs:
                a = source_aligned[child] - source_aligned[parent]
                b = fk[child] - fk[parent]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if min(na, nb) < 1e-6:
                    continue
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na*nb), -1, 1)))))
                length_ratios.append(float(nb / na))
            src_width = np.median([np.linalg.norm(source_aligned[a]-source_aligned[b])
                                   for a, b in mcp_pairs])
            robot_width = np.median([np.linalg.norm(fk[a]-fk[b]) for a, b in mcp_pairs])
            rows.append({"frame_id": int(frame), "bone_direction_median_deg": float(np.median(angles)),
                         "bone_direction_p95_deg": float(np.percentile(angles, 95)),
                         "root_free_segment_length_ratio_median": float(np.median(length_ratios)),
                         "source_mcp_span_m": float(src_width),
                         "robot_mcp_span_m": float(robot_width),
                         "palm_width_scale_ratio": float(robot_width/src_width)})
        if len(rows) != int(valid.sum()):
            raise RuntimeError("SENSOR_VALID_DENOMINATOR_CHANGED")
        def p50(key: str) -> float:
            return float(np.median([row[key] for row in rows]))
        # A structural H50 anchor requires all 51 real consecutive source frames.
        time = hand["timestamp_ns"].astype(np.int64)
        delta = np.diff(time)
        edges = ((np.diff(hand["frame_id"]) == 1)
                 & (delta > 0) & (delta <= 2.5 * np.median(delta)))
        anchors = [start for start in range(n-50)
                   if valid[start:start+51].all() and edges[start:start+50].all()]
        by_side.append({"physical_side": model.side, "anatomical_side":
                        str(hand["anatomical_side_names"][anatomical]),
                        "valid_frames": len(rows), "structural_h50_windows": len(anchors),
                        "first_structural_h50_anchor": anchors[0] if anchors else None,
                        "bone_direction_median_deg": p50("bone_direction_median_deg"),
                        "bone_direction_p95_across_frames_deg": float(np.percentile(
                            [row["bone_direction_p95_deg"] for row in rows], 95)),
                        "root_free_segment_length_ratio_median": p50(
                            "root_free_segment_length_ratio_median"),
                        "source_mcp_span_median_m": p50("source_mcp_span_m"),
                        "robot_mcp_span_median_m": p50("robot_mcp_span_m"),
                        "palm_width_scale_ratio_median": p50("palm_width_scale_ratio"),
                        "rows": rows})
    result = {"schema_version": "SENSOR_097_ROOT_FREE_SHAPE_AUDIT_V1", "task_id": TASK,
              "source": artifact_ref(hand_path), "saved_kai22": artifact_ref(backend_path),
              "robot_assets": [artifact_ref(model.urdf_path) for model in models],
              "method": "SOURCE_PALM_BASIS_TO_ROBOT_PALM_BASIS_ROOT_FREE_INTRA_FINGER_BONES",
              "sides": by_side, "consumer_check": "INDEPENDENT_FK_AND_CONTIGUOUS_H50_ENUMERATION_ONLY",
              "local_kai22_training_eligible": False,
              "reason": "No preauthorized local-shape quality threshold or independent RGB-to-finger correspondence; root-free diagnostic cannot by itself admit Kai22 supervision.",
              "next_action": "CHECK_FIXED_097_RGB_PROXY_AND_EXISTING_FLOW_CONSUMER_FIELD_MASK_SEMANTICS; DO_NOT_REUSE_BAD_WRIST_SCALE"}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"receipt": str(output), "valid_frames": [row["valid_frames"] for row in by_side],
            "structural_h50": [row["structural_h50_windows"] for row in by_side]}


def sensor_real_q_loss_consumer() -> dict:
    """Exercise the existing FM sanitization/loss on real Sensor q, diagnostic only."""
    _check_task()
    import torch
    from chaoyang.human_ego.training.FlowMatchingTrainer import sanitize_flow_action_targets
    from chaoyang.human_ego.training.FlowMatchingModel import FlowMatchingModel
    output = ATTEMPT / "lanes/sensor/SENSOR_097_REAL_Q_FIELD_MASK_CONSUMER.json"
    if output.exists():
        raise FileExistsError(output)
    shape_audit = load_json(ATTEMPT / "lanes/sensor/SENSOR_097_ROOT_FREE_BONE_QUALITY.json")
    if any(side["first_structural_h50_anchor"] != 0 for side in shape_audit["sides"]):
        raise RuntimeError("SENSOR_097_H50_ANCHOR_NOT_FROZEN")
    backend_path = SENSOR_ROOT / "run_097_wave0/KAI22_COMMON_BACKEND_V1.npz"
    backend = _load(backend_path)
    if (not np.array_equal(backend["human_to_physical"], [1, 0])
            or not backend["valid"][:51].all()):
        raise RuntimeError("SENSOR_097_Q_WINDOW_NOT_STRUCTURALLY_VALID")
    q_anatomical = backend["q22"][1:51][:, [1, 0]].astype(np.float32)
    if not np.isfinite(q_anatomical).all():
        raise RuntimeError("SENSOR_Q_NONFINITE")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    diagnostic_mask = torch.zeros((1, 50, 62), dtype=torch.bool)
    diagnostic_mask[..., 18:62] = True
    sample_weight = torch.ones(1)
    actual = torch.full((1, 50, 62), float("nan"))
    actual[0, :, 18:40] = torch.from_numpy(q_anatomical[:, 0])
    actual[0, :, 40:62] = torch.from_numpy(q_anatomical[:, 1])
    different_invalid = actual.clone()
    different_invalid[..., :18] = 123.0
    safe_nan = sanitize_flow_action_targets(actual, diagnostic_mask, sample_weight)
    safe_fill = sanitize_flow_action_targets(different_invalid, diagnostic_mask, sample_weight)
    if not torch.equal(safe_nan, safe_fill) or not bool(torch.isfinite(safe_nan).all()):
        raise RuntimeError("INVALID_WRIST_PAYLOAD_AFFECTED_SAFE_FLOW_INPUT")
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(safe_nan.shape, generator=generator)
    t = torch.full((1, 1, 1), .5)
    xt_nan = (1-t)*noise + t*safe_nan
    xt_fill = (1-t)*noise + t*safe_fill
    if not torch.equal(xt_nan, xt_fill):
        raise RuntimeError("INVALID_WRIST_PAYLOAD_AFFECTED_FLOW_STATE")
    model = FlowMatchingModel(single_hand=False, pred_horizon=50, img_size=(240, 320),
                              vision_embed_dim=128, num_decoder_layers=2, num_heads=4,
                              dropout=0.0, use_pcd_features=False,
                              use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
                              use_aux_temporal_contrastive=False, use_region_attn=False,
                              hand_action_representation="kaihand_joint_state").eval()
    def loss(target: torch.Tensor) -> float:
        safe = sanitize_flow_action_targets(target, diagnostic_mask, sample_weight)
        value = model.compute_loss({"v_pred": torch.zeros_like(safe)},
                                   {"v_target": safe-noise,
                                    "action_valid_mask": diagnostic_mask,
                                    "sample_weight": sample_weight})["loss_flow"]
        return float(value)
    loss_nan, loss_fill = loss(actual), loss(different_invalid)
    if loss_nan != loss_fill or not np.isfinite(loss_nan):
        raise RuntimeError("INVALID_WRIST_PAYLOAD_AFFECTED_VALID_LOSS")
    changed_valid = actual.clone()
    changed_valid[0, 0, 18] += .1
    sensitive_loss = loss(changed_valid)
    if abs(sensitive_loss-loss_nan) < 1e-9:
        raise RuntimeError("VALID_Q_TARGET_NOT_SENSITIVE")
    zero_supervision_rejected = False
    try:
        sanitize_flow_action_targets(actual, torch.zeros_like(diagnostic_mask), sample_weight)
    except RuntimeError as error:
        zero_supervision_rejected = "No effective action supervision" in str(error)
    if not zero_supervision_rejected:
        raise RuntimeError("ZERO_SUPERVISION_NOT_REJECTED")
    assets = load_pinned_robot_assets(REPO_ROOT)
    models = (load_hand_model(assets.left_hand.path, "left"),
              load_hand_model(assets.right_hand.path, "right"))
    fk_error = []
    for index in range(50):
        for physical in range(2):
            fk = keypoints_from_q(models[physical], backend["q22"][index+1, physical])
            fk -= fk[0]
            fk_error.append(float(np.max(np.abs(
                fk - backend["fk21_root_relative"][index+1, physical]))))
    if max(fk_error) > 1e-8:
        raise RuntimeError("SENSOR_Q_LOADER_FK_ROUNDTRIP_DRIFT")
    result = {"schema_version": "SENSOR_097_REAL_Q_FIELD_MASK_CONSUMER_V1",
              "task_id": TASK, "source": artifact_ref(backend_path),
              "shape_quality_audit": artifact_ref(
                  ATTEMPT / "lanes/sensor/SENSOR_097_ROOT_FREE_BONE_QUALITY.json"),
              "anchor_source_frame": 0, "future_source_frames": [1, 50],
              "action_layout": "Lpos3,Rpos3,Lrot6,Rrot6,Lq22,Rq22",
              "diagnostic_q_supervised_elements": int(diagnostic_mask.sum()),
              "training_admitted_supervised_elements": 0,
              "invalid_payload_and_flow_input_invariant": True,
              "invalid_payload_loss_invariant": True,
              "valid_q_target_sensitive": True,
              "zero_supervision_rejected": zero_supervision_rejected,
              "loss_diagnostic_random_initialized_no_forward": loss_nan,
              "changed_valid_loss": sensitive_loss,
              "q_to_independent_fk_max_abs_m": max(fk_error),
              "model_forward_executed": False, "optimizer_steps": 0,
              "consumer_qualification": "PREPROCESS_AND_LOSS_KERNEL_ONLY_NO_LABEL_QUALITY_AUTHORITY",
              "training_eligible": False,
              "reason": "Current saved Kai22 q has unresolved MANUS virtual-root scale and root-free bone-direction quality; no current-anchor wrist authority for the complete 62D consumer.",
              "next_action": "REPAIR_LOCAL_REPRESENTATION_WITH_FROZEN_097_VISUAL_EVIDENCE_BEFORE_FULL_LOADER_OR_TRAINING_ADMISSION"}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"receipt": str(output), "diagnostic_elements": result["diagnostic_q_supervised_elements"],
            "training_elements": 0}


def poker_finger_first_layer() -> dict:
    """Pin the first changed layer of the fixed 104→105 pinky complaint."""
    _check_task()
    from chaoyang.pipeline.huro_hand_only_retarget_v1 import _local_directions
    output = ATTEMPT / "lanes/motion_product/POKER_104_105_FINGER_FIRST_LAYER.json"
    if output.exists():
        raise FileExistsError(output)
    saved = _load(POKER_R0)
    source = _load(POKER_SOURCE)
    assets = load_pinned_robot_assets(REPO_ROOT)
    hand = load_hand_model(assets.right_hand.path, "right")
    side = 1
    before, after = 104, 105
    if not saved["target_valid"][before:after+1, side].all():
        raise RuntimeError("POKER_FINGER_TARGET_INVALID")
    target = saved["target21_root"][:, side]
    q = saved["q22"][:, side]
    dt = float((saved["timestamp_ns"][after] - saved["timestamp_ns"][before]) / 1e9)
    if dt <= 0:
        raise RuntimeError("POKER_FINGER_TIME_INVALID")
    bone_rows = []
    for chain in FINGER_CHAINS:
        parent = 0
        for child in chain:
            first = target[before, child] - target[before, parent]
            second = target[after, child] - target[after, parent]
            na, nb = float(np.linalg.norm(first)), float(np.linalg.norm(second))
            if min(na, nb) <= 1e-8:
                raise RuntimeError("POKER_FINGER_DEGENERATE_TARGET_BONE")
            angle = float(np.degrees(np.arccos(np.clip(np.dot(first, second) / (na * nb), -1, 1))))
            bone_rows.append({"parent": int(parent), "child": int(child),
                              "length_104_mm": na * 1000, "length_105_mm": nb * 1000,
                              "target_direction_step_deg": angle})
            parent = child
    wanted = target[after] - target[after, 0]
    directions = _local_directions(wanted)
    neutral = .5 * (hand.lower + hand.upper)
    temporal_weight = np.sqrt(10.) * (1 / 30) / dt
    def terms(value: np.ndarray) -> dict:
        points = keypoints_from_q(hand, value)
        points -= points[0]
        residuals = {
            "direction": 20 * (_local_directions(points) - directions),
            "fingertips": 8 * (points[[4, 8, 12, 16, 20]] - wanted[[4, 8, 12, 16, 20]]),
            "temporal": temporal_weight * (value - q[before]),
            "neutral": np.sqrt(.15) * (value - neutral),
        }
        return {"cost": .5 * sum(float(np.square(part).sum()) for part in residuals.values()),
                "term_norms": {key: float(np.linalg.norm(part)) for key, part in residuals.items()},
                "pinky_direction_norm": float(np.linalg.norm(residuals["direction"][-4:])),
                "pinky_tip_norm": float(np.linalg.norm(residuals["fingertips"][-1]))}
    changed = np.flatnonzero(np.abs(q[after] - q[before]) > .05)
    rows = [{"index": int(index), "name": str(hand.joint_names[index]),
             "q_104_rad": float(q[before, index]), "q_105_rad": float(q[after, index]),
             "delta_rad": float(q[after, index] - q[before, index]),
             "lower_rad": float(hand.lower[index]), "upper_rad": float(hand.upper[index])}
            for index in changed]
    if len(rows) == 0 or max(abs(row["delta_rad"]) for row in rows) < .5:
        raise RuntimeError("POKER_FINGER_COMPLAINT_NOT_REPRODUCED")
    result = {"schema_version": "POKER_104_105_FINGER_FIRST_LAYER_V1", "task_id": TASK,
              "source": artifact_ref(POKER_SOURCE), "saved_robot": artifact_ref(POKER_R0),
              "frame_ids": [before, after], "physical_side": "right", "dt_s": dt,
              "source21_max_joint_displacement_mm": float(np.nanmax(np.linalg.norm(
                  source["joints_3d_camera"][:, after] - source["joints_3d_camera"][:, before], axis=2)) * 1000),
              "target21_max_joint_displacement_mm": float(np.max(np.linalg.norm(
                  target[after] - target[before], axis=1)) * 1000),
              "target_bones": bone_rows, "changed_q_over_0_05_rad": rows,
              "objective_at_previous_q_for_frame_105": terms(q[before]),
              "objective_at_saved_q_105": terms(q[after]),
              "first_changed_layer": "FINGER_SOLVER_Q_NOT_SOURCE_OR_TARGET_JUMP",
              "quality": "REJECTED_QUALITY", "q_replacement": False,
              "claim_limit": "The frozen objective rewards a lower-cost saturated pinky solution; this does not establish an allowable retarget fix or quality pass.",
              "next_action": "KEEP_OLD_Q_AND_MARK_FINGER_JUMP; ANY_NEW_SOLVER_OBJECTIVE_REQUIRES_PREDECLARED_SINGLE_CANDIDATE_AND_FULL_TIMELINE_GATE"}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"receipt": str(output), "first_changed_layer": result["first_changed_layer"],
            "max_q_step_rad": max(abs(row["delta_rad"]) for row in rows)}


def poker_card_support_prepare() -> dict:
    """Freeze exactly the complaint-window current Raw for one existing SAM3 producer."""
    _check_task()
    root = ATTEMPT / "lanes/scene/POKER_076_091_CURRENT_CARD_SUPPORT"
    if root.exists():
        raise FileExistsError(root)
    source = load_json(POKER_SCENE / "SCENE_PREP_MANIFEST.json")
    if len(source["rows"]) != 171:
        raise RuntimeError("POKER_SCENE_TIMELINE_DRIFT")
    (root / "frames").mkdir(parents=True)
    rows = []
    for local, frame in enumerate(range(76, 92)):
        raw_path = Path(source["rows"][frame]["raw"])
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None or raw.shape != (960, 1280, 3):
            raise RuntimeError(f"POKER_CARD_RAW_DOMAIN:{frame}")
        destination = root / "frames" / f"{local:06d}.png"
        if not cv2.imwrite(str(destination), raw):
            raise RuntimeError(f"POKER_CARD_FRAME_WRITE:{frame}")
        if not np.array_equal(cv2.imread(str(destination), cv2.IMREAD_COLOR), raw):
            raise RuntimeError(f"POKER_CARD_FRAME_COPY_DRIFT:{frame}")
        rows.append({"source_frame": frame, "local_index": local,
                     "source": artifact_ref(raw_path), "model_frame": artifact_ref(destination)})
    spec = {"schema_version": "POKER_CURRENT_CARD_SUPPORT_INPUT_V1", "task_id": TASK,
            "session_id": "play_cards_0902_042", "image_domain": source["image_domain"],
            "frame_count": 16, "source_frame_range_inclusive": [76, 91],
            "seed_source_frame": 80, "seed_local_index": 4,
            "prompt": "a purple-backed playing card",
            "seed_boxes_xyxy": [[663, 436, 762, 528], [775, 435, 881, 539], [905, 507, 1009, 630]],
            "model": "PINNED_EXISTING_SAM3_1_TEXT_PLUS_BOX_PROPAGATION",
            "purpose": "CURRENT_FRAME_VISIBLE_CARD_SUPPORT_DIAGNOSTIC_NOT_GROUND_TRUTH",
            "rows": rows}
    atomic_json(root / "INPUT.json", spec)
    return {"input": str(root / "INPUT.json"), "frames": 16}


def poker_card_support_infer() -> dict:
    """Run one fixed current-frame object-support producer under the GPU lease."""
    _check_task()
    from chaoyang.ops.run_v5_scene import _sam_outputs
    from chaoyang.pipeline import sam31_compat_adapter_v1 as sam_adapter
    import torch
    root = ATTEMPT / "lanes/scene/POKER_076_091_CURRENT_CARD_SUPPORT"
    spec = load_json(root / "INPUT.json")
    output = root / "RESULT.json"
    if output.exists():
        raise FileExistsError(output)
    lease_path = REPO_ROOT / "_run/current/GPU_LEASE.json"
    lease_state = load_json(lease_path)
    if (lease_state.get("status") != "ACQUIRED"
            or lease_state.get("task_id") != f"{TASK}:scene"
            or lease_state.get("gpu_process_pid") != os.getpid()
            or str(lease_state.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")
            or not lease_state.get("fencing_token")):
        raise RuntimeError("QUALITY_SCENE_GPU_LEASE_REQUIRED")
    lease = {"lease": artifact_ref(lease_path), "token": lease_state["fencing_token"]}
    weight = REPO_ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
    if not weight.is_file():
        raise RuntimeError("PINNED_SAM_WEIGHT_MISSING")
    sys.path.insert(0, str(REPO_ROOT / "vendor/SAM3"))
    sam_adapter.PROJECT_ROOT = REPO_ROOT
    adapter, evidence = sam_adapter.build_pinned_adapter(
        official_code_root=REPO_ROOT / "vendor/SAM3", checkpoint_path=weight)
    model = adapter.model
    state = model.init_state(resource_path=str(root / "frames"),
                             offload_video_to_cpu=True, async_loading_frames=False)
    boxes = []
    for x0, y0, x1, y1 in spec["seed_boxes_xyxy"]:
        boxes.append([x0 / 1280, y0 / 960, (x1-x0) / 1280, (y1-y0) / 960])
    seen: dict[int, dict] = {}
    (root / "visible_card").mkdir(exist_ok=False)
    def collect(index: int, value: dict, direction: str) -> None:
        masks, ids, scores = _sam_outputs(value, (960, 1280))
        union = np.any(masks, axis=0) if len(masks) else np.zeros((960, 1280), bool)
        if index not in seen or (direction == "forward" and index == 4):
            seen[index] = {"mask": union, "ids": [int(x) for x in ids],
                           "scores": [float(x) for x in scores], "direction": direction}
    try:
        _, seed = model.add_prompt(inference_state=state, frame_idx=4,
                                   text_str=spec["prompt"], boxes_xywh=boxes,
                                   box_labels=[1, 1, 1], clear_old_boxes=True,
                                   output_prob_thresh=.5)
        collect(4, seed, "seed")
        for index, value in model.propagate_in_video(
                inference_state=state, start_frame_idx=4,
                max_frame_num_to_track=12, reverse=False, output_prob_thresh=.5):
            collect(int(index), value, "forward")
        for index, value in model.propagate_in_video(
                inference_state=state, start_frame_idx=4,
                max_frame_num_to_track=4, reverse=True, output_prob_thresh=.5):
            collect(int(index), value, "reverse")
    finally:
        state.clear()
        torch.cuda.empty_cache()
    rows = []
    for local, source_frame in enumerate(range(76, 92)):
        value = seen.get(local)
        mask = value["mask"] if value else np.zeros((960, 1280), bool)
        path = root / "visible_card" / f"{source_frame:06d}.png"
        if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"POKER_CARD_MASK_WRITE:{source_frame}")
        rows.append({"source_frame": source_frame, "local_index": local,
                     "mask": artifact_ref(path), "visible_pixels": int(mask.sum()),
                     "observed_instance_ids": value["ids"] if value else [],
                     "model_scores": value["scores"] if value else [],
                     "direction": value["direction"] if value else "NO_OUTPUT_UNKNOWN"})
    nonempty = sum(row["visible_pixels"] > 0 for row in rows)
    result = {"schema_version": "POKER_CURRENT_CARD_SUPPORT_RESULT_V1", "task_id": TASK,
              "input": artifact_ref(root / "INPUT.json"), "checkpoint": artifact_ref(weight),
              "model_evidence": evidence, "gpu_lease": lease,
              "rows": rows, "nonempty_frames": nonempty,
              "quality": "DIAGNOSTIC_ONLY_NOT_VISIBLE_OBJECT_GROUND_TRUTH",
              "adoption": "NOT_ADOPTED", "next_action":
              "VERIFY_EACH_CURRENT_FRAME_CARD_PIXEL_AND_MASK_CONFLICT_BEFORE_ONE_PROPAINTER_CANDIDATE"}
    atomic_json(output, result)
    return {"receipt": str(output), "nonempty_frames": nonempty}


def poker_clean_card_window_prepare() -> dict:
    """Freeze one 16-frame ProPainter input with current visible-card protection."""
    _check_task()
    card_root = ATTEMPT / "lanes/scene/POKER_076_091_CURRENT_CARD_SUPPORT"
    support = load_json(card_root / "RESULT.json")
    if support["nonempty_frames"] != 16:
        raise RuntimeError("CARD_SUPPORT_G0_INCOMPLETE")
    root = ATTEMPT / "lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN"
    if root.exists():
        raise FileExistsError(root)
    for folder in ("frames", "model_masks", "write", "protect"):
        (root / folder).mkdir(parents=True, exist_ok=False)
    old = load_json(POKER_SCENE / "SCENE_PREP_MANIFEST.json")
    rows = []
    for local, frame in enumerate(range(76, 92)):
        original = old["rows"][frame]
        raw = cv2.imread(original["raw"], cv2.IMREAD_COLOR)
        resized = cv2.imread(str(POKER_SCENE / "prep/frames" / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
        old_model = cv2.imread(str(POKER_SCENE / "prep/model_masks" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        old_write = cv2.imread(original["write"], cv2.IMREAD_GRAYSCALE)
        visible = cv2.imread(str(card_root / "visible_card" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if any(x is None for x in (raw, resized, old_model, old_write, visible)):
            raise RuntimeError(f"CARD_CLEAN_INPUT_MISSING:{frame}")
        if raw.shape != (960, 1280, 3) or resized.shape != (720, 960, 3):
            raise RuntimeError(f"CARD_CLEAN_IMAGE_DOMAIN:{frame}")
        # Interior is protected only where the current frame actually shows a card.
        protect = cv2.erode((visible > 0).astype(np.uint8),
                            np.ones((5, 5), np.uint8), iterations=1) > 0
        write = (old_write > 0) & ~protect
        model_protect = cv2.resize(protect.astype(np.uint8), (960, 720),
                                   interpolation=cv2.INTER_NEAREST) > 0
        model = (old_model > 0) & ~model_protect
        if np.any(write & protect) or not np.any(write):
            raise RuntimeError(f"CARD_CLEAN_PERMISSION_CONFLICT:{frame}")
        destinations = {key: root / key / f"{local:06d}.png"
                        for key in ("frames", "model_masks", "write", "protect")}
        for key, value in (("frames", resized), ("model_masks", model.astype(np.uint8) * 255),
                           ("write", write.astype(np.uint8) * 255),
                           ("protect", protect.astype(np.uint8) * 255)):
            if not cv2.imwrite(str(destinations[key]), value):
                raise RuntimeError(f"CARD_CLEAN_PREP_WRITE:{frame}:{key}")
        rows.append({"source_frame": frame, "local_index": local,
                     "raw": artifact_ref(Path(original["raw"])),
                     "model_frame": artifact_ref(destinations["frames"]),
                     "model_mask": artifact_ref(destinations["model_masks"]),
                     "write": artifact_ref(destinations["write"]),
                     "protect": artifact_ref(destinations["protect"]),
                     "old_write_px": int((old_write > 0).sum()), "new_write_px": int(write.sum()),
                     "current_visible_card_px": int((visible > 0).sum()),
                     "protected_card_in_old_write_px": int(((old_write > 0) & protect).sum()),
                     "model_mask_removed_for_current_card_px": int(((old_model > 0) & model_protect).sum())})
    recipe = {"schema_version": "POKER_CARD_PROTECTED_CLEAN_INPUT_V1", "task_id": TASK,
              "session_id": "play_cards_0902_042", "source_frame_range_inclusive": [76, 91],
              "frame_count": 16, "image_domain": old["image_domain"],
              "existing_scene_prep": artifact_ref(POKER_SCENE / "SCENE_PREP_MANIFEST.json"),
              "current_frame_card_support": artifact_ref(card_root / "RESULT.json"),
              "changed_input": "CURRENT_FRAME_VISIBLE_CARD_INTERIOR_PROTECTION_ONLY",
              "context_limit": "16_FRAME_WINDOW_NOT_171_FRAME_REFERENCE_CONTEXT",
              "model_parameters": {"width": 960, "height": 720, "mask_dilation": 0,
                                   "ref_stride": 10, "neighbor_length": 10,
                                   "subvideo_length": 80, "raft_iter": 20, "fp16": True},
              "rows": rows, "quality_authority": "INPUT_PREPARATION_ONLY"}
    atomic_json(root / "INPUT.json", recipe)
    return {"input": str(root / "INPUT.json"), "frames": 16,
            "protected_card_in_old_write_px": sum(row["protected_card_in_old_write_px"] for row in rows)}


def poker_clean_card_window_infer() -> dict:
    """One pinned ProPainter candidate; preserve current Raw outside M_write."""
    _check_task()
    from chaoyang.pipeline.v5_scene import composite_clean
    root = ATTEMPT / "lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN"
    recipe = load_json(root / "INPUT.json")
    result_path = root / "RESULT.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    lease_path = REPO_ROOT / "_run/current/GPU_LEASE.json"
    lease = load_json(lease_path)
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != f"{TASK}:scene"
            or lease.get("gpu_process_pid") != os.getpid()
            or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("QUALITY_SCENE_GPU_LEASE_REQUIRED")
    vendor = REPO_ROOT / "vendor/ProPainter"
    weights = [vendor / "weights" / name for name in
               ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    if any(not path.is_file() for path in weights):
        raise RuntimeError("PROPAINTER_WEIGHT_MISSING")
    command = [sys.executable, "-B", str(vendor / "inference_propainter.py"),
               "--video", str(root / "frames"), "--mask", str(root / "model_masks"),
               "--output", str(root / "upstream"), "--width", "960", "--height", "720",
               "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
               "--save_frames", "--fp16"]
    invocation = {"task_id": TASK, "command": command, "weights": [artifact_ref(p) for p in weights],
                  "producer": artifact_ref(vendor / "inference_propainter.py"),
                  "input": artifact_ref(root / "INPUT.json"), "lease": artifact_ref(lease_path),
                  "started_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(root / "INVOCATION.json", invocation)
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    log_path = root / "PROPAINTER.log"
    with log_path.open("xb") as log:
        completed = subprocess.run(command, cwd=vendor, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"POKER_PROPAINTER_EXIT:{completed.returncode}:{log_path}")
    generated = sorted((root / "upstream/frames/frames").glob("*.png"))
    if len(generated) != 16:
        raise RuntimeError(f"POKER_PROPAINTER_OUTPUT_COUNT:{len(generated)}")
    (root / "clean").mkdir()
    rows = []
    for local, (input_row, generated_path) in enumerate(zip(recipe["rows"], generated, strict=True)):
        raw = cv2.imread(input_row["raw"]["path"], cv2.IMREAD_COLOR)
        predicted = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        write = cv2.imread(input_row["write"]["path"], cv2.IMREAD_GRAYSCALE) > 0
        protect = cv2.imread(input_row["protect"]["path"], cv2.IMREAD_GRAYSCALE) > 0
        if raw is None or predicted is None:
            raise RuntimeError(f"POKER_PROPAINTER_DECODE:{local}")
        predicted = cv2.resize(predicted, (1280, 960), interpolation=cv2.INTER_LINEAR)
        clean = composite_clean(raw, predicted, write, protect)
        changed = np.any(clean != raw, axis=2)
        if changed[~write].any() or changed[protect].any():
            raise RuntimeError(f"POKER_CLEAN_WRITE_BOUNDARY:{local}")
        destination = root / "clean" / f"{input_row['source_frame']:06d}.png"
        if not cv2.imwrite(str(destination), clean):
            raise RuntimeError(f"POKER_CLEAN_WRITE:{local}")
        rows.append({"source_frame": input_row["source_frame"],
                     "clean": artifact_ref(destination), "outside_write_changed_px": 0,
                     "protected_changed_px": 0, "changed_px": int(changed.sum())})
    # Record the exact neighbor/reference indices used by this pinned vendor selection loop.
    from chaoyang.ops.run_r2_scene_candidate import IMAGE_SIZE
    if IMAGE_SIZE != (960, 720):
        raise RuntimeError("PROPAINTER_IMAGE_SIZE_DRIFT")
    refs = []
    for midpoint in range(0, 16, 5):
        neighbors = list(range(max(0, midpoint-5), min(16, midpoint+6)))
        references = [i for i in range(0, 16, 10) if i not in neighbors]
        refs.append({"midpoint_local": midpoint, "neighbor_local": neighbors,
                     "reference_local": references,
                     "selected_mask_refs": [recipe["rows"][i]["model_mask"] for i in neighbors+references]})
    result = {"schema_version": "POKER_CARD_PROTECTED_CLEAN_WINDOW_V1", "task_id": TASK,
              "execution": "REAL_PROPAINTER_SINGLE_CANDIDATE", "structure": "PASS",
              "quality": "PENDING_INDEPENDENT_FIXED_WINDOW_REVIEW", "adoption": "NOT_ADOPTED",
              "frame_count": 16, "source_frame_range_inclusive": [76, 91],
              "input": artifact_ref(root / "INPUT.json"),
              "invocation": artifact_ref(root / "INVOCATION.json"), "log": artifact_ref(log_path),
              "references": refs, "rows": rows,
              "claim_limit": "Current-card protection candidate only; 16-frame context differs from archived 171-frame Clean, so no one-factor quality attribution or full-product authority.",
              "next_action": "INDEPENDENT_REVIEW_RAW_OLD_NEW_CARD_EDGE_GHOSTING_AND_FLICKER; EXPAND_ONLY_IF_FROZEN_QUALITY_GATE_PASSES"}
    atomic_json(result_path, result)
    return {"receipt": str(result_path), "clean_frames": len(rows)}


def poker_clean_card_window_review() -> dict:
    """Deliver all 16 fixed complaint frames without claiming a quality pass."""
    _check_task()
    root = ATTEMPT / "lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN"
    candidate = load_json(root / "RESULT.json")
    output = root / "VISUAL_REVIEW.json"
    if output.exists():
        raise FileExistsError(output)
    old_root = ATTEMPT / "lanes/scene/POKER_HISTORICAL_CLEAN_EXACT_REBASE/clean"
    card_root = ATTEMPT / "lanes/scene/POKER_076_091_CURRENT_CARD_SUPPORT/visible_card"
    frozen_input = load_json(root / "INPUT.json")
    review_frames = root / "review_frames"
    review_frames.mkdir()
    def panel(image: np.ndarray, label: str) -> np.ndarray:
        item = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(item, (0, 0), (640, 35), (0, 0, 0), -1)
        cv2.putText(item, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (255, 255, 255), 2, cv2.LINE_AA)
        return item
    for local, frame in enumerate(range(76, 92)):
        # The source path is separately frozen in INPUT, never inferred from the video title.
        raw = cv2.imread(frozen_input["rows"][local]["raw"]["path"], cv2.IMREAD_COLOR)
        old = cv2.imread(str(old_root / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
        new = cv2.imread(str(root / "clean" / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
        card = cv2.imread(str(card_root / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if any(value is None for value in (raw, old, new, card)):
            raise RuntimeError(f"POKER_REVIEW_INPUT_MISSING:{frame}")
        overlay = raw.copy()
        overlay[card > 0] = .35 * overlay[card > 0] + .65 * np.array([0, 255, 255])
        frame_image = np.vstack((np.hstack((panel(raw, f"RAW source {frame}"),
                                             panel(old, "HISTORICAL REBASE REJECTED"))),
                                 np.hstack((panel(overlay, "CURRENT CARD SUPPORT"),
                                            panel(new, "NEW PROPAINTER REJECTED")))))
        path = review_frames / f"{local:06d}.png"
        if not cv2.imwrite(str(path), frame_image):
            raise RuntimeError(f"POKER_REVIEW_FRAME_WRITE:{frame}")
    video = root / "POKER_076_091_RAW_OLD_SUPPORT_NEW_REVIEW.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-y",
               "-framerate", "30", "-i", str(review_frames / "%06d.png"), "-an", "-c:v",
               "libx264", "-threads", "2", "-preset", "fast", "-crf", "19",
               "-pix_fmt", "yuv420p", str(video)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError("POKER_REVIEW_VIDEO_ENCODE:" + done.stderr[-1000:])
    cap = cv2.VideoCapture(str(video)); decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != 16:
        raise RuntimeError(f"POKER_REVIEW_VIDEO_DECODE:{decoded}")
    result = {"schema_version": "POKER_CARD_CLEAN_VISUAL_REVIEW_V1", "task_id": TASK,
              "candidate": artifact_ref(root / "RESULT.json"), "video": artifact_ref(video),
              "frame_count": 16, "decoded_frames": decoded,
              "fixed_raw_complaint_frames_reviewed": [80, 91],
              "observations": ["Current-frame card interior is visibly retained relative to historical rebase.",
                               "Held-card boundary remains smeared in frames 80 and 91.",
                               "Both wristbands and adjacent green edges remain visible/ghosted in the new Clean."],
              "structure": "PASS", "quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED",
              "full_session_extension": "NOT_RUN_FIXED_WINDOW_FAILED",
              "review_scope": "TWO_FIXED_FRAME_SPOT_CHECKS_PLUS_16_FRAME_SYNCHRONIZED_VIDEO_DELIVERED_NOT_FULL_SESSION_VISUAL_PASS",
              "next_action": "NO_SAME_RECIPE_FULL_171_RUN; KEEP_PRODUCT_SCENE_GATE_FAILED_AND_CONTINUE_INDEPENDENT_SENSOR_AND_GEOMETRY_WORK"}
    atomic_json(output, result)
    return {"review": str(output), "video": str(video), "quality": result["quality"]}


def sensor_qonly_h50_export() -> dict:
    """Export one real, explicitly non-admitted q-only H50 diagnostic packet."""
    _check_task()
    root = ATTEMPT / "lanes/sensor"
    path = root / "SENSOR_097_QONLY_H50_DIAGNOSTIC.npz"
    receipt = root / "SENSOR_097_QONLY_H50_DIAGNOSTIC.json"
    if path.exists() or receipt.exists():
        raise FileExistsError("SENSOR_QONLY_H50_ALREADY_EXISTS")
    kernel = load_json(root / "SENSOR_097_REAL_Q_FIELD_MASK_CONSUMER.json")
    if not kernel["invalid_payload_and_flow_input_invariant"] or not kernel["zero_supervision_rejected"]:
        raise RuntimeError("SENSOR_QONLY_KERNEL_NOT_CHECKED")
    backend_path = SENSOR_ROOT / "run_097_wave0/KAI22_COMMON_BACKEND_V1.npz"
    backend = _load(backend_path)
    if not np.array_equal(backend["human_to_physical"], [1, 0]):
        raise RuntimeError("SENSOR_QONLY_SIDE_MAPPING_DRIFT")
    frames = np.asarray(backend["source_frame_id"][:51], np.int64)
    timestamps = np.asarray(backend["timestamp_ns"][:51], np.int64)
    if not (np.array_equal(frames, np.arange(51)) and np.all(np.diff(timestamps) > 0)
            and backend["valid"][:51].all()):
        raise RuntimeError("SENSOR_QONLY_WINDOW_NOT_CONTIGUOUS")
    q_current = backend["q22"][0, [1, 0]].astype(np.float32)
    q_future = backend["q22"][1:51][:, [1, 0]].astype(np.float32)
    if not np.isfinite(q_current).all() or not np.isfinite(q_future).all():
        raise RuntimeError("SENSOR_QONLY_NONFINITE")
    action = np.full((50, 62), np.nan, np.float32)
    action[:, 18:40] = q_future[:, 0]
    action[:, 40:62] = q_future[:, 1]
    structural = np.zeros((50, 62), bool)
    structural[:, 18:62] = True
    admitted = np.zeros((50, 62), bool)
    assets = load_pinned_robot_assets(REPO_ROOT)
    names = np.stack([np.asarray(load_hand_model(assets.left_hand.path, "left").joint_names),
                      np.asarray(load_hand_model(assets.right_hand.path, "right").joint_names)])
    np.savez_compressed(path, source_frame_id=frames, timestamp_ns=timestamps,
                        q_current_anatomical=q_current, action_future=action,
                        structural_valid=structural, training_admitted_valid=admitted,
                        joint_names=names, human_to_physical=np.array([1, 0]),
                        source_session_id=np.array("play_cards_0916_097"))
    reloaded = _load(path)
    if (not np.array_equal(reloaded["action_future"], action, equal_nan=True)
            or not np.array_equal(reloaded["structural_valid"], structural)
            or reloaded["training_admitted_valid"].any()):
        raise RuntimeError("SENSOR_QONLY_RELOAD_DRIFT")
    result = {"schema_version": "SENSOR_097_QONLY_H50_DIAGNOSTIC_V1", "task_id": TASK,
              "source": artifact_ref(backend_path), "kernel_check": artifact_ref(
                  root / "SENSOR_097_REAL_Q_FIELD_MASK_CONSUMER.json"),
              "data": artifact_ref(path), "anchor_source_frame": 0,
              "future_source_frame_range_inclusive": [1, 50],
              "action_layout": "Lpos3,Rpos3,Lrot6,Rrot6,Lq22,Rq22",
              "joint_order": "left_then_right_explicit_joint_names_in_npz",
              "units": {"position": "m", "rotation_6d": "representation_unpopulated",
                        "q22": "rad", "timestamp": "ns"},
              "structurally_valid_q_elements": int(structural.sum()),
              "training_admitted_elements": 0,
              "reason": "Local MANUS-to-Kai22 shape scale/directions and absolute wrist authority are not qualified; this is a real consumer diagnostic, not a training label.",
              "execution": "EXPORTED_AND_RELOADED", "structure": "PASS",
              "quality": "NOT_ADMITTED", "adoption": "NOT_ADOPTED",
              "training_eligible": False}
    atomic_json(receipt, result)
    return {"receipt": str(receipt), "data": str(path), "training_admitted_elements": 0}


def sensor_qonly_real_rgb_forward() -> dict:
    """Run one actual existing FM forward with real RGB and diagnostic q labels.

    Unqualified current wrist/ICT conditions are masked tokens, not invented poses.
    This is deliberately not the complete loader or training admission.
    """
    _check_task()
    import torch
    from chaoyang.human_ego.training.FlowMatchingModel import FlowMatchingModel
    from chaoyang.human_ego.training.FlowMatchingTrainer import sanitize_flow_action_targets
    output = ATTEMPT / "lanes/sensor/SENSOR_097_QONLY_REAL_RGB_FORWARD.json"
    if output.exists():
        raise FileExistsError(output)
    packet_path = ATTEMPT / "lanes/sensor/SENSOR_097_QONLY_H50_DIAGNOSTIC.npz"
    packet = _load(packet_path)
    source_receipt = load_json(SENSOR_HAND_ROOT / "play_cards_0916_097/RESULT.json")
    video_path = Path(source_receipt["source_video"]["path"])
    cap = cv2.VideoCapture(str(video_path))
    ok, stereo = cap.read()
    cap.release()
    if not ok or stereo.shape != (1536, 4096, 3):
        raise RuntimeError("SENSOR_097_SBS_FRAME0_DECODE_OR_DOMAIN")
    # Frozen Sensor producer contract: sourceIndex=1 is physical-left.
    bgr = stereo[:, 2048:]
    rgb = cv2.cvtColor(cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)
    x_rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None] / 255
    raw_action = torch.from_numpy(packet["action_future"].copy())[None]
    mask = torch.from_numpy(packet["structural_valid"].copy())[None]
    weights = torch.ones(1)
    if tuple(raw_action.shape) != (1, 50, 62) or int(mask.sum()) != 2200:
        raise RuntimeError("SENSOR_QONLY_H50_SHAPE_DRIFT")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = FlowMatchingModel(single_hand=False, pred_horizon=50, img_size=(240, 320),
                              vision_embed_dim=128, num_decoder_layers=2, num_heads=4,
                              dropout=0.0, use_pcd_features=False,
                              use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
                              use_aux_temporal_contrastive=False, use_region_attn=False,
                              hand_action_representation="kaihand_joint_state").eval()
    noise = torch.randn(raw_action.shape, generator=torch.Generator().manual_seed(7))
    t = torch.full((1, 1), .5)
    def evaluate(action: torch.Tensor, current_payload: float) -> tuple[torch.Tensor, torch.Tensor]:
        safe = sanitize_flow_action_targets(action, mask, weights)
        xt = (1-t[:, :, None]) * noise + t[:, :, None] * safe
        with torch.inference_mode():
            pred = model(x_rgb=x_rgb, x_ict=torch.zeros((1, 1, 29)),
                         ict_mask=torch.zeros((1, 1), dtype=torch.bool),
                         x_t=xt, t=t,
                         x_robot_state=torch.full((1, 2, 33), current_payload),
                         robot_state_mask=torch.zeros((1, 2), dtype=torch.bool))
            loss = model.compute_loss(pred, {"v_target": safe-noise,
                                             "action_valid_mask": mask,
                                             "sample_weight": weights})["loss_flow"]
        return pred["v_pred"], loss
    first_pred, first_loss = evaluate(raw_action, float("nan"))
    altered = raw_action.clone()
    altered[..., :18] = 123.0
    second_pred, second_loss = evaluate(altered, -777.0)
    if (not torch.equal(first_pred, second_pred)
            or not torch.equal(first_loss, second_loss)
            or not torch.isfinite(first_pred).all()
            or not bool(torch.isfinite(first_loss))):
        raise RuntimeError("SENSOR_QONLY_INVALID_PAYLOAD_CHANGED_REAL_FORWARD")
    result = {"schema_version": "SENSOR_097_QONLY_REAL_RGB_FORWARD_V1", "task_id": TASK,
              "packet": artifact_ref(packet_path), "source_sbs_video": artifact_ref(video_path),
              "source_frame": 0, "source_index": 1,
              "rgb_domain": "SENSOR_0916_PHYSICAL_LEFT_SBS_RESIZE_ONLY",
              "real_rgb_shape": list(x_rgb.shape), "real_q_action_shape": list(raw_action.shape),
              "structurally_valid_q_elements": int(mask.sum()), "training_admitted_elements": 0,
              "masked_current_robot_state": True, "masked_ict_condition": True,
              "invalid_wrist_and_current_payload_forward_invariant": True,
              "actual_forward_shape": list(first_pred.shape), "loss_random_initialization": float(first_loss),
              "optimizer_steps": 0, "checkpoint_loaded": False,
              "interface_scope": "DIRECT_MODEL_FORWARD_WITH_REAL_RGB_AND_Q_NOT_FULL_DATALOADER",
              "quality": "LABEL_NOT_ADMITTED", "training_eligible": False,
              "claim_limit": "Finite model forward verifies the existing partial-field interface only; masked current conditions and unresolved local-shape quality prohibit training or learning-benefit claims."}
    atomic_json(output, result)
    return {"receipt": str(output), "loss": result["loss_random_initialization"],
            "training_admitted_elements": 0}


def sensor_native_bone_semantics() -> dict:
    """Compare named MANUS25→21 segment geometry with the actual Kai22 FK."""
    _check_task()
    output = ATTEMPT / "lanes/sensor/SENSOR_097_NATIVE_BONE_SEMANTICS.json"
    if output.exists():
        raise FileExistsError(output)
    source_path = SENSOR_HAND_ROOT / "play_cards_0916_097/HAND_MOTION_V1.npz"
    backend_path = SENSOR_ROOT / "run_097_wave0/KAI22_COMMON_BACKEND_V1.npz"
    hand, backend = _load(source_path), _load(backend_path)
    assets = load_pinned_robot_assets(REPO_ROOT)
    models = (load_hand_model(assets.left_hand.path, "left"),
              load_hand_model(assets.right_hand.path, "right"))
    if str(hand["joint_names"][0]) != "virtual_wrist_root":
        raise RuntimeError("SENSOR_MANUS_ROOT_NAME_DRIFT")
    if not np.array_equal(hand["manus25_to_21"],
                          [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24]):
        raise RuntimeError("SENSOR_25_TO_21_MAPPING_DRIFT")
    rows = []
    for physical, model in enumerate(models):
        anatomical = int(np.flatnonzero(backend["human_to_physical"] == physical)[0])
        valid = backend["valid"][:, physical]
        original = hand["manus_local_21_m"][valid, anatomical]
        if len(original) != 165 or not np.isfinite(original).all():
            raise RuntimeError("SENSOR_NATIVE_BONE_INVALID_INPUT")
        neutral = keypoints_from_q(model, .5 * (model.lower + model.upper))
        segments = []
        for chain in FINGER_CHAINS:
            parent = 0
            for child in chain:
                measured = np.linalg.norm(original[:, child] - original[:, parent], axis=1)
                robot = float(np.linalg.norm(neutral[child] - neutral[parent]))
                segments.append({"source_parent_name": str(hand["joint_names"][parent]),
                                 "source_child_name": str(hand["joint_names"][child]),
                                 "source_length_p50_m": float(np.median(measured)),
                                 "source_length_p5_m": float(np.percentile(measured, 5)),
                                 "source_length_p95_m": float(np.percentile(measured, 95)),
                                 "robot_neutral_length_m": robot,
                                 "source_to_robot_ratio": float(np.median(measured) / robot)})
                parent = child
        rows.append({"anatomical_side": str(hand["anatomical_side_names"][anatomical]),
                     "physical_robot_side": model.side, "valid_source_frames": int(valid.sum()),
                     "segments": segments})
    result = {"schema_version": "SENSOR_097_NATIVE_BONE_SEMANTICS_V1", "task_id": TASK,
              "source": artifact_ref(source_path), "backend": artifact_ref(backend_path),
              "node_mapping": hand["manus25_to_21"].tolist(),
              "root_name": "virtual_wrist_root", "sides": rows,
              "quality": "INTERNAL_GEOMETRY_MISMATCH_LOCAL_LABEL_NOT_ADMITTED",
              "training_eligible": False,
              "claim_limit": "Named local-node lengths are inconsistent with Kai neutral geometry; these are vendor-derived positions, not external anatomical truth. A single root-radius scale cannot establish Kai22 label fidelity.",
              "next_action": "DO_NOT_REUSE_WRIST_ROOT_SCALE_FOR_FINGER_LABELS; REQUIRE_INDEPENDENT_LOCAL_SHAPE_CORRESPONDENCE_BEFORE_RESTORE"}
    atomic_json(output, result)
    return {"receipt": str(output), "sides": len(rows),
            "valid_side_frames": sum(row["valid_source_frames"] for row in rows)}


def poker_stereo_g0() -> dict:
    """Separate same-domain Stereo feasibility from product occlusion authority."""
    _check_task()
    output = ATTEMPT / "lanes/scene/POKER_0902_STEREO_G0.json"
    if output.exists():
        raise FileExistsError(output)
    binding_path = REPO_ROOT / ("_run/current/human_to_robot_evidence_unlock_s2_20260923/"
        "attempts/attempt_0001/bindings/play_cards_0902_042/attempt_0001/BINDING.json")
    depth_path = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
        "lanes/exact78/depth_exact_canary_v1/play_cards_0902_042/DEPTH_MANIFEST.json")
    stereo_path = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
        "lanes/exact78/stereo_canary_v1/play_cards_0902_042/RESULT.json")
    binding, depth, stereo = (load_json(path) for path in (binding_path, depth_path, stereo_path))
    same_domain = (binding["session_id"] == depth["session_id"] == stereo["session_id"]
                   and binding["image_domain"] == depth["image_domain"])
    if not same_domain:
        raise RuntimeError("POKER_STEREO_SOURCE_DOMAIN_MISMATCH")
    sample_frames = [int(row["frame_id"]) for row in depth["frames"]]
    if sample_frames != [0, 85, 170] or depth["full_source_frame_count"] != 171:
        raise RuntimeError("POKER_STEREO_CANARY_FRAME_DRIFT")
    if any(not Path(row["path"]).is_file() for row in depth["frames"]):
        raise RuntimeError("POKER_STEREO_CANARY_PAYLOAD_MISSING")
    product_quality = load_json(ATTEMPT / "lanes/motion_product/POKER_171_TECHNICAL_QUALITY.json")
    scene_quality = load_json(ATTEMPT / "lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN/VISUAL_REVIEW.json")
    if (product_quality["decision"]["quality"] != "REJECTED_QUALITY"
            or scene_quality["quality"] != "REJECTED_QUALITY"):
        raise RuntimeError("POKER_STEREO_QUEUE_DEPENDENCY_DRIFT")
    result = {"schema_version": "POKER_0902_STEREO_G0_V1", "task_id": TASK,
              "product_binding": artifact_ref(binding_path),
              "same_domain_depth_canary": artifact_ref(depth_path),
              "stereo_calibration": artifact_ref(stereo_path),
              "session_and_image_domain_match": True,
              "source_index": 0, "image_domain": binding["image_domain"],
              "sample_depth_frames": sample_frames, "depth_frames_needed_for_full_product": 171,
              "depth_frames_available": 3,
              "capture_baseline_authority": depth["metric_scale_authority"],
              "external_metric_accuracy": depth["external_metric_accuracy"],
              "scene_depth_product_qualified": False,
              "occlusion_unknown_overlap_pixels_current_product": 843458,
              "producer_queue": "CONDITIONALLY_READY_SAME_DOMAIN_EXISTING_FOUNDATIONSTEREO",
              "consumer_queue": "WAIT_VALID_FULL_DEPTH_AND_OBJECT_SURFACE_REGISTRATION",
              "execution_decision": "NO_FULL_171_GPU_BATCH_WHILE_SAME_SESSION_SCENE_AND_MOTION_REQUIRED_GATES_ARE_REJECTED",
              "reason": "Three canary frames establish domain feasibility only; current full product cannot gain quality PASS from a costly full Stereo run while independent Clean and Motion gates fail.",
              "not_affected": ["Sensor q-only consumer", "existing full product review", "Local/HuRo historical comparison"],
              "quality": "NOT_QUALIFIED_FOR_PRODUCT_OCCLUSION", "adoption": "NOT_ADOPTED"}
    atomic_json(output, result)
    return {"receipt": str(output), "same_domain": True,
            "available_depth_frames": len(sample_frames), "needed": 171}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("chips-seed-counterfactual", "sensor-local-audit",
                                          "poker-scene-mask-audit", "poker-motion-audit",
                                          "poker-seed-suffix", "poker-historical-clean-rebase",
                                          "poker-technical-quality", "publish-progress",
                                          "sensor-root-free-shape-audit",
                                          "sensor-real-q-loss-consumer", "poker-finger-first-layer",
                                          "poker-card-support-prepare", "poker-card-support-infer",
                                          "poker-clean-card-window-prepare", "poker-clean-card-window-infer",
                                          "poker-clean-card-window-review", "sensor-qonly-h50-export",
                                          "sensor-qonly-real-rgb-forward", "sensor-native-bone-semantics",
                                          "poker-stereo-g0"))
    args = parser.parse_args()
    result = ({"chips-seed-counterfactual": chips_seed_counterfactual,
              "sensor-local-audit": sensor_local_audit,
              "poker-scene-mask-audit": poker_scene_mask_audit,
              "poker-motion-audit": poker_motion_audit,
              "poker-seed-suffix": poker_seed_suffix,
              "poker-historical-clean-rebase": poker_historical_clean_rebase,
              "poker-technical-quality": poker_technical_quality,
              "publish-progress": publish_progress,
              "sensor-root-free-shape-audit": sensor_root_free_shape_audit,
              "sensor-real-q-loss-consumer": sensor_real_q_loss_consumer,
              "poker-finger-first-layer": poker_finger_first_layer,
              "poker-card-support-prepare": poker_card_support_prepare,
              "poker-card-support-infer": poker_card_support_infer,
              "poker-clean-card-window-prepare": poker_clean_card_window_prepare,
              "poker-clean-card-window-infer": poker_clean_card_window_infer,
              "poker-clean-card-window-review": poker_clean_card_window_review,
              "sensor-qonly-h50-export": sensor_qonly_h50_export,
              "sensor-qonly-real-rgb-forward": sensor_qonly_real_rgb_forward,
              "sensor-native-bone-semantics": sensor_native_bone_semantics,
              "poker-stereo-g0": poker_stereo_g0}[args.stage]())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
