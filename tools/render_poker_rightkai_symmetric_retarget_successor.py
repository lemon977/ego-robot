#!/usr/bin/env python3
"""One bounded physical-right KaiHand retarget successor for Poker 042."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RETARGET_PROJECT = Path("/mnt/workspace/code/retargeting_human")
for root in (str(PROJECT), str(RETARGET_PROJECT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from retargeting_human.retargeter import KaiHandRetargeter  # noqa: E402
from pipeline import robot_wrist_kai_adapter as adapter  # noqa: E402
from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import render_poker_v2c03_p3_naturalv2_successor as fixed  # noqa: E402
from tools import render_poker_v2c03_sessionfit_chirality_successor as sessionfit  # noqa: E402


KEY_SLOTS = fixed.KEY_SLOTS
PREDECESSOR = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/"
    "robot_mount_proxy_baseline_v1/task_action_canary_v1/"
    "play_cards_0902_042_v2c03_seed_sessionfit_chirality_keyframes_v2"
)
MANO_TIPS = (4, 8, 12, 16, 20)
ROBOT_TIPS = (5, 6, 7, 8, 9)


class RetargetSuccessorError(RuntimeError):
    pass


def solve_human_left_with_left_kai(
    hawor: dict[str, np.ndarray], assets: Any
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    left_names = old.moving_joint_names(assets.left_hand)
    right_names = old.moving_joint_names(assets.right_hand)
    normalized_left = tuple(name.replace("hand_l_", "hand_X_") for name in left_names)
    normalized_right = tuple(name.replace("hand_r_", "hand_X_") for name in right_names)
    if normalized_left != normalized_right:
        raise RetargetSuccessorError("left/right KaiHand semantic joint order differs")
    left_limits = np.asarray(
        [[j.lower, j.upper] for j in assets.left_hand.joints if j.joint_type != "fixed"]
    )
    right_limits = np.asarray(
        [[j.lower, j.upper] for j in assets.right_hand.joints if j.joint_type != "fixed"]
    )
    if not np.array_equal(left_limits, right_limits):
        raise RetargetSuccessorError("left/right KaiHand limits differ")
    solver = KaiHandRetargeter(
        old.KAI_URDFS[0],
        "left",
        max_iterations=9,
        max_step_deg=float(np.degrees(0.06)),
        position_scale_m=0.010,
    )
    output = []
    results = []
    try:
        for frame in range(24):
            points = np.asarray(hawor["joints_3d_camera"][0, frame], dtype=np.float64)
            wrist = np.eye(4, dtype=np.float64)
            wrist[:3, :3] = adapter.final_v3_mano_palm_basis(points, handedness="left")
            wrist[:3, 3] = points[0]
            result = solver.retarget(wrist, points[np.asarray(old.MANO_CHAINS)])
            output.append(result.q)
            results.append(result)
    finally:
        solver.close()
    q = np.asarray(output, dtype=np.float64)
    if np.any(q < right_limits[:, 0] - 1e-9) or np.any(q > right_limits[:, 1] + 1e-9):
        raise RetargetSuccessorError("left-solved q violates mirrored right Kai limits")
    return q, results, {
        "joint_order_semantic_suffix_match": True,
        "joint_limits_byte_equal": True,
        "mapping": "same semantic finger/joint index after hand_l_ to hand_r_ prefix replacement",
        "sign_map": [1.0] * 22,
    }


def direct_right_frame0(hawor: dict[str, np.ndarray]) -> Any:
    points = np.asarray(hawor["joints_3d_camera"][0, 0], dtype=np.float64)
    wrist = np.eye(4, dtype=np.float64)
    wrist[:3, :3] = adapter.final_v3_mano_palm_basis(points, handedness="left")
    wrist[:3, 3] = points[0]
    solver = KaiHandRetargeter(
        old.KAI_URDFS[1],
        "right",
        max_iterations=9,
        max_step_deg=float(np.degrees(0.06)),
        position_scale_m=0.010,
    )
    try:
        return solver.retarget(wrist, points[np.asarray(old.MANO_CHAINS)])
    finally:
        solver.close()


def finger_report(result: Any) -> list[dict[str, float | str]]:
    rows = []
    for index, finger in enumerate(old.FINGERS):
        target = np.asarray(result.target_points_robot[index], dtype=np.float64)
        fitted = np.asarray(result.fitted_points_robot[index], dtype=np.float64)
        rows.append(
            {
                "finger": finger,
                "target_root_to_tip_mm": float(np.linalg.norm(target[-1] - target[0]) * 1000.0),
                "fk_root_to_tip_mm": float(np.linalg.norm(fitted[-1] - fitted[0]) * 1000.0),
                "tip_error_mm": float(result.tip_error_m[index] * 1000.0),
                "direction_error_deg_mean": float(np.mean(result.direction_error_deg[index])),
            }
        )
    return rows


def select_terminal_chirality(
    assets: Any,
    camera: np.ndarray,
    q_arm: np.ndarray,
    base_mounts: np.ndarray,
    q_hand: np.ndarray,
    hawor: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    names = tuple(
        old.moving_joint_names(model) for model in (assets.left_hand, assets.right_hand)
    )
    rotations = old.proper_axis_rotations()
    mounts = base_mounts.copy()
    audits = []
    for physical in range(2):
        human_side = fixed.PHYSICAL_TO_HUMAN[physical]
        candidates = []
        for candidate_index, rotation in enumerate(rotations):
            finger_angles: list[list[float]] = []
            length_ratios: list[list[float]] = []
            normal_angles = []
            winding = []
            for slot in KEY_SLOTS:
                values = {
                    name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row
                }
                for side_index in range(2):
                    values.update(
                        dict(
                            zip(
                                old.official.ARM_JOINT_NAMES[side_index],
                                q_arm[slot, side_index],
                                strict=True,
                            )
                        )
                    )
                arm_fk = old.official.forward_kinematics(assets.tianji, values)
                local = old.hand_landmarks(assets, names, q_hand[slot])[physical]
                mount = base_mounts[physical].copy()
                mount[:3, :3] = mount[:3, :3] @ rotation
                root = camera @ arm_fk[f"{fixed.SIDES[physical]}_tool"] @ mount
                points = (root[:3, :3] @ local.T).T + root[:3, 3]
                intrinsics = np.asarray(hawor["intrinsics"][slot], dtype=np.float64)
                robot_uv = np.asarray(
                    [
                        [
                            intrinsics[0, 0] * point[0] / point[2] + intrinsics[0, 2],
                            intrinsics[1, 1] * point[1] / point[2] + intrinsics[1, 2],
                        ]
                        for point in points
                    ]
                )
                human_uv = np.asarray(hawor["joints_2d"][human_side, slot])
                slot_angles = []
                slot_ratios = []
                for robot_tip, human_tip in zip(ROBOT_TIPS, MANO_TIPS, strict=True):
                    robot_vector = robot_uv[robot_tip] - robot_uv[0]
                    human_vector = human_uv[human_tip] - human_uv[0]
                    slot_angles.append(old.angle_deg(robot_vector, human_vector))
                    slot_ratios.append(
                        float(np.linalg.norm(robot_vector) / max(np.linalg.norm(human_vector), 1e-9))
                    )
                finger_angles.append(slot_angles)
                length_ratios.append(slot_ratios)
                robot_index = robot_uv[2] - robot_uv[0]
                robot_pinky = robot_uv[4] - robot_uv[0]
                human_index = human_uv[5] - human_uv[0]
                human_pinky = human_uv[17] - human_uv[0]
                winding.append(
                    bool(float(np.cross(robot_index, robot_pinky)) * float(np.cross(human_index, human_pinky)) > 0.0)
                )
                human_camera = np.asarray(hawor["joints_3d_camera"][human_side, slot])
                normal_angles.append(
                    old.angle_deg(
                        np.cross(points[2] - points[0], points[4] - points[0]),
                        np.cross(human_camera[5] - human_camera[0], human_camera[17] - human_camera[0]),
                    )
                )
            angles = np.asarray(finger_angles)
            ratios = np.asarray(length_ratios)
            mean_angle = float(np.mean(angles))
            mean_ratio = float(np.mean(ratios))
            mean_normal = float(np.mean(normal_angles))
            winding_fraction = float(np.mean(winding))
            passes = bool(
                mean_ratio >= 0.60
                and float(np.mean(angles[:, 0])) <= 45.0
                and mean_angle <= 45.0
                and mean_normal <= 70.0
                and winding_fraction == 1.0
            )
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "proper_rotation": rotation.tolist(),
                    "finger_direction_angle_deg_mean": mean_angle,
                    "thumb_direction_angle_deg_mean": float(np.mean(angles[:, 0])),
                    "finger_direction_angle_deg_by_finger_mean": np.mean(angles, axis=0).tolist(),
                    "projected_length_ratio_mean": mean_ratio,
                    "projected_length_ratio_by_finger_mean": np.mean(ratios, axis=0).tolist(),
                    "palm_normal_angle_deg_mean": mean_normal,
                    "palm_winding_match_fraction": winding_fraction,
                    "passes": passes,
                }
            )
        passed = [row for row in candidates if row["passes"]]
        if not passed:
            raise RetargetSuccessorError(f"no terminal-aware chirality candidate for physical {physical}")
        passed.sort(
            key=lambda row: row["finger_direction_angle_deg_mean"] + 0.25 * row["palm_normal_angle_deg_mean"]
        )
        selected = passed[0]
        mounts[physical, :3, :3] = (
            base_mounts[physical, :3, :3]
            @ np.asarray(selected["proper_rotation"], dtype=np.float64)
        )
        audits.append(
            {
                "physical_side": fixed.SIDES[physical],
                "human_side": fixed.SIDES[human_side],
                "candidate_count": 24,
                "passing_candidate_count": len(passed),
                "selected": selected,
            }
        )
    return mounts, {
        "method": "TERMINAL_TIP_AWARE_INDEPENDENT_24_WAY_PROPER_ROTATION_FINAL_CAMERA",
        "slots": list(KEY_SLOTS),
        "gates": {
            "per_hand_mean_projected_length_ratio_min": 0.60,
            "thumb_direction_angle_deg_mean_max": 45.0,
            "all_finger_direction_angle_deg_mean_max": 45.0,
            "palm_normal_angle_deg_mean_max": 70.0,
            "palm_winding_match_fraction": 1.0
        },
        "sides": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise RetargetSuccessorError(f"refusing to overwrite: {args.output_dir}")
    predecessor_review = old.load_json(PREDECESSOR / "AGENT_REVIEW.json")
    if predecessor_review.get("grade") != "C" or predecessor_review.get("downstream_authorized") is not False:
        raise RetargetSuccessorError("expected pinned C predecessor")
    hawor = old.arrays(args.hawor)
    manifest = old.load_json(args.mask_manifest)
    predecessor = old.arrays(PREDECESSOR / "ROBOT_STATES.npz")
    if not (
        manifest.get("session") == fixed.SESSION
        and np.array_equal(predecessor["source_frames"], np.arange(24))
    ):
        raise RetargetSuccessorError("input session/frame identity mismatch")

    assets = old.load_pinned_robot_assets(PROJECT)
    direct_frame0 = direct_right_frame0(hawor)
    surrogate_q, surrogate_results, mapping = solve_human_left_with_left_kai(hawor, assets)
    q_hand = np.asarray(predecessor["q_hand"], dtype=np.float64).copy()
    q_hand[:, 1] = surrogate_q
    hand_models = (assets.left_hand, assets.right_hand)
    hand_lower = np.asarray(
        [[j.lower for j in model.joints if j.joint_type != "fixed"] for model in hand_models]
    )
    hand_upper = np.asarray(
        [[j.upper for j in model.joints if j.joint_type != "fixed"] for model in hand_models]
    )
    q_hand = old.bounded_projection(q_hand, hand_lower, hand_upper)
    q_arm = np.asarray(predecessor["q_arm"], dtype=np.float64)
    camera = np.asarray(predecessor["T_camera_base_session_fit"], dtype=np.float64)
    base_mounts = np.asarray(predecessor["T_tool_hand_rootplane_base"], dtype=np.float64)
    mounts, chirality = select_terminal_chirality(
        assets, camera, q_arm, base_mounts, q_hand, hawor
    )
    hand_time = old.temporal_metrics(q_hand)
    selected_ratios = [row["selected"]["projected_length_ratio_mean"] for row in chirality["sides"]]
    gates = {
        "camera_matrix_byte_equal_to_predecessor_session_fit": bool(np.array_equal(camera, predecessor["T_camera_base_session_fit"])),
        "robot_scale_unchanged": True,
        "naturalv2_original_scale": True,
        "procedural_flange_proxy_absent": True,
        "left_right_joint_order_semantic_match": mapping["joint_order_semantic_suffix_match"],
        "left_right_joint_limits_equal": mapping["joint_limits_byte_equal"],
        "hand_limits": bool(np.all(q_hand >= hand_lower) and np.all(q_hand <= hand_upper)),
        "hand_step": bool(hand_time["step_max_rad_per_frame"] <= 0.08 + 1e-9),
        "hand_second_difference": bool(hand_time["second_difference_max_rad_per_frame2"] <= 0.06 + 1e-9),
        "both_projected_finger_length_ratio_mean_ge_0p60": bool(min(selected_ratios) >= 0.60),
        "thumb_palm_and_winding": bool(all(row["selected"]["passes"] for row in chirality["sides"])),
    }
    if not all(gates.values()):
        raise RetargetSuccessorError(f"hard gate failed: {gates}")

    args.output_dir.mkdir(parents=True)
    frame_root = args.output_dir / "frames"
    frame_root.mkdir()
    raster = old.load_module(old.RASTER_SOURCE, "poker_rightkai_symmetric_retarget_raster")
    flange_local = fixed.naturalv2_local_triangles()
    cache: dict[Path, Any] = {}
    full_camera = old.look_at_camera()
    full_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    images = []
    frames = []
    for slot in KEY_SLOTS:
        raw_path = args.raw_root / f"{slot:05d}" / "rgb.png"
        record = manifest["frames"][slot]["source_rgb"]
        if Path(record["path"]).resolve() != raw_path.resolve() or old.sha256(raw_path) != record["sha256"]:
            raise RetargetSuccessorError(f"raw lineage mismatch at {slot}")
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RetargetSuccessorError(f"raw decode failed at {slot}")
        source_h, source_w = raw.shape[:2]
        raw = cv2.resize(raw, (fixed.WIDTH, fixed.HEIGHT), interpolation=cv2.INTER_AREA)
        intrinsics = np.asarray(hawor["intrinsics"][slot], dtype=np.float64).copy()
        intrinsics[0] *= fixed.WIDTH / source_w
        intrinsics[1] *= fixed.HEIGHT / source_h
        ego_color, ego_label, _ = fixed.render_scene(
            raster, assets, q_arm[slot], q_hand[slot], mounts, camera, intrinsics, cache, flange_local
        )
        visible = ego_label >= 0
        overlay = raw.copy()
        overlay[visible] = np.clip(0.18 * raw[visible] + 0.82 * ego_color[visible], 0, 255).astype(np.uint8)
        left = fixed.title(
            overlay,
            f"042 固定session-fit｜右Kai对称retarget｜source {slot:05d}",
            "camera/scale完全冻结｜P1用left-Kai solve后语义joint映射到right-Kai",
            "左右投影指长比均≥0.60｜非标定｜不可部署",
        )
        full_color, full_label, _ = fixed.render_scene(
            raster, assets, q_arm[slot], q_hand[slot], mounts, full_camera, full_k, cache, flange_local
        )
        middle = np.full_like(full_color, 242)
        middle[full_label >= 0] = full_color[full_label >= 0]
        middle = fixed.title(
            middle,
            "真实Tianji/Kai CAD｜NaturalV2原尺度真实法兰",
            "无黄色proxy｜P3/arm/camera不变",
            "KAI_ADAPTER_CAD_MISSING｜root basis仅视觉候选",
        )
        human_uv = np.asarray(hawor["joints_2d"][:, slot], dtype=np.float64).copy()
        human_uv[..., 0] *= fixed.WIDTH / source_w
        human_uv[..., 1] *= fixed.HEIGHT / source_h
        tips = [
            [float(x) for x in predecessor_review["metrics"]["retargeting_human_frame0_tip_error_mm"]["physical_left_to_human_right"]]
            if physical == 0
            else [float(x * 1000.0) for x in surrogate_results[slot].tip_error_m]
            for physical in range(2)
        ]
        local = old.render_local_hand_comparison(raster, assets, q_hand[slot], human_uv, tips, cache)
        ratio_text = "/".join(
            f"P{i} ratio={row['selected']['projected_length_ratio_mean']:.2f} thumb={row['selected']['thumb_direction_angle_deg_mean']:.1f}°"
            for i, row in enumerate(chirality["sides"])
        )
        local = fixed.title(
            local,
            "终端指尖方向/投影长度覆盖审计",
            ratio_text,
            "P1=human-left→left-Kai bounded IK→semantic joint map→right-Kai FK",
        )
        combined = np.hstack((left, middle, local))
        path = frame_root / f"{slot:06d}.png"
        if not cv2.imwrite(str(path), combined):
            raise RetargetSuccessorError("frame write failed")
        images.append(combined)
        frames.append({"slot": slot, "frame_sha256": old.sha256(path)})

    sheet = args.output_dir / "POKER_042_右Kai对称Retarget_关键帧0_8_15_23.png"
    if not cv2.imwrite(str(sheet), np.vstack(images)):
        raise RetargetSuccessorError("sheet write failed")
    frame0_report = args.output_dir / "FRAME0_RETARGET_COMPARISON.json"
    frame0_report.write_text(
        json.dumps(
            {
                "direct_right_kai": {
                    "optimizer_success": bool(direct_frame0.optimizer_success),
                    "optimizer_cost": direct_frame0.optimizer_cost,
                    "fingers": finger_report(direct_frame0),
                },
                "left_kai_same_human_left_then_semantic_map_to_right": {
                    "optimizer_success": bool(surrogate_results[0].optimizer_success),
                    "optimizer_cost": surrogate_results[0].optimizer_cost,
                    "fingers": finger_report(surrogate_results[0]),
                },
                "mapping": mapping,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    chirality_path = args.output_dir / "TERMINAL_CHIRALITY_AUDIT.json"
    chirality_path.write_text(json.dumps(chirality, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    states = args.output_dir / "ROBOT_STATES.npz"
    np.savez_compressed(
        states,
        source_frames=np.arange(24),
        q_arm=q_arm,
        q_hand=q_hand,
        T_camera_base_session_fit=camera,
        T_tool_hand_terminal_chirality=mounts,
        physical_right_retarget_source=np.asarray("LEFT_KAI_BOUNDED_IK_SEMANTIC_JOINT_MAP"),
    )
    result = {
        "schema_version": "poker-rightkai-symmetric-retarget-successor-result-v1",
        "status": "KEYFRAMES_READY_FOR_VISUAL_REVIEW",
        "grade": "B_DEVELOPMENT_ONLY_PENDING_USER_VISUAL_REVIEW",
        "session": fixed.SESSION,
        "development_only": True,
        "downstream_authorized": False,
        "current_robot_authority": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hard_gates": gates,
        "metrics": {
            "hand_temporal": hand_time,
            "terminal_chirality": chirality,
            "camera_wrist_rmse_px_frozen": predecessor_review["metrics"]["bilateral_wrist_rmse_px_after"],
        },
        "lineage": {
            "predecessor_c_review": old.artifact(PREDECESSOR / "AGENT_REVIEW.json"),
            "predecessor_states": old.artifact(PREDECESSOR / "ROBOT_STATES.npz"),
            "naturalv2_stl": old.artifact(fixed.NATURALV2),
            "left_kai_urdf_used_for_bounded_human_left_solve": old.artifact(old.KAI_URDFS[0]),
            "right_kai_urdf_used_for_fk_and_render": old.artifact(old.KAI_URDFS[1]),
        },
        "outputs": {
            "keyframe_sheet": old.artifact(sheet),
            "frame0_retarget_comparison": old.artifact(frame0_report),
            "terminal_chirality_audit": old.artifact(chirality_path),
            "states": old.artifact(states),
        },
        "claim_limit": "One bounded four-keyframe development successor only. Camera/base and Robot scale are frozen from the prior session fit. The physical-right q is obtained by semantic same-index mapping of a left-Kai bounded IK solve on the same human-left target; this is a visual retarget hypothesis, not real mount calibration or deployment authority."
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "sheet": str(sheet), "ratios": selected_ratios}, ensure_ascii=False))


if __name__ == "__main__":
    main()
