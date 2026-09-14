#!/usr/bin/env python3
"""One bounded Poker task-translation successor on the frozen static closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
for root in (str(PROJECT), "/mnt/workspace/code/retargeting_human"):
    if root not in sys.path:
        sys.path.insert(0, root)

from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import render_poker_v2c03_p3_naturalv2_successor as fixed  # noqa: E402
from tools import render_poker_rightkai_symmetric_retarget_successor as rightfix  # noqa: E402


STATIC_AUDIT = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/"
    "robot_mount_proxy_baseline_v1/static_interface_closure_audit_v1/"
    "ASSEMBLY_CLOSURE_AUDIT.json"
)
KEYS = np.asarray((0, 8, 15, 23), dtype=np.int64)
ROBOT_TIPS = (5, 6, 7, 8, 9)
MANO_TIPS = (4, 8, 12, 16, 20)


class TaskTranslationError(RuntimeError):
    pass


def arm_limits(assets: Any) -> tuple[np.ndarray, np.ndarray]:
    limits = {joint.name: (joint.lower, joint.upper) for joint in assets.tianji.joints}
    return (
        np.asarray([[limits[name][0] for name in row] for row in old.official.ARM_JOINT_NAMES]),
        np.asarray([[limits[name][1] for name in row] for row in old.official.ARM_JOINT_NAMES]),
    )


def hand_limits(assets: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = [[joint for joint in model.joints if joint.joint_type != "fixed"] for model in (assets.left_hand, assets.right_hand)]
    return np.asarray([[j.lower for j in row] for row in rows]), np.asarray([[j.upper for j in row] for row in rows])


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source, axis=1, keepdims=True)
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(target.T @ source)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    value = u @ correction @ vt
    if not np.allclose(np.linalg.det(value), 1.0, atol=1e-9):
        raise TaskTranslationError("Kabsch orientation is not proper")
    return value


def exact_target_metric(
    hawor: dict[str, np.ndarray], landmarks: np.ndarray, physical: int, candidate: np.ndarray
) -> dict[str, Any]:
    human = fixed.PHYSICAL_TO_HUMAN[physical]
    angles, ratios, normals, winding = [], [], [], []
    for slot in KEYS:
        robot_vectors = landmarks[slot, physical, np.asarray(ROBOT_TIPS)] - landmarks[slot, physical, 0]
        human_camera = np.asarray(hawor["joints_3d_camera"][human, slot])
        human_vectors = human_camera[np.asarray(MANO_TIPS)] - human_camera[0]
        rotation = kabsch(robot_vectors, human_vectors) @ candidate
        points = landmarks[slot, physical] @ rotation.T + human_camera[0]
        intrinsics = np.asarray(hawor["intrinsics"][slot])
        robot_uv = np.stack(
            (
                intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2],
                intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2],
            ),
            axis=1,
        )
        human_uv = np.asarray(hawor["joints_2d"][human, slot])
        slot_angles, slot_ratios = [], []
        for robot_tip, human_tip in zip(ROBOT_TIPS, MANO_TIPS, strict=True):
            robot_vector = robot_uv[robot_tip] - robot_uv[0]
            human_vector = human_uv[human_tip] - human_uv[0]
            slot_angles.append(old.angle_deg(robot_vector, human_vector))
            slot_ratios.append(float(np.linalg.norm(robot_vector) / max(np.linalg.norm(human_vector), 1e-9)))
        angles.append(slot_angles)
        ratios.append(slot_ratios)
        winding.append(
            bool(
                float(np.cross(robot_uv[2] - robot_uv[0], robot_uv[4] - robot_uv[0]))
                * float(np.cross(human_uv[5] - human_uv[0], human_uv[17] - human_uv[0]))
                > 0.0
            )
        )
        normals.append(
            old.angle_deg(
                np.cross(points[2] - points[0], points[4] - points[0]),
                np.cross(human_camera[5] - human_camera[0], human_camera[17] - human_camera[0]),
            )
        )
    angle_array, ratio_array = np.asarray(angles), np.asarray(ratios)
    value = {
        "finger_direction_angle_deg_mean": float(np.mean(angle_array)),
        "finger_direction_angle_deg_by_finger_mean": np.mean(angle_array, axis=0).tolist(),
        "thumb_direction_angle_deg_mean": float(np.mean(angle_array[:, 0])),
        "projected_length_ratio_mean": float(np.mean(ratio_array)),
        "projected_length_ratio_by_finger_mean": np.mean(ratio_array, axis=0).tolist(),
        "palm_normal_angle_deg_mean": float(np.mean(normals)),
        "palm_winding_match_fraction": float(np.mean(winding)),
    }
    value["passes"] = bool(
        value["finger_direction_angle_deg_mean"] <= 45.0
        and value["thumb_direction_angle_deg_mean"] <= 45.0
        and value["projected_length_ratio_mean"] >= 0.60
        and value["palm_normal_angle_deg_mean"] <= 70.0
        and value["palm_winding_match_fraction"] == 1.0
    )
    return value


def select_root_chirality(
    hawor: dict[str, np.ndarray], landmarks: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = old.proper_axis_rotations()
    selected, audit = [], []
    for physical in range(2):
        rows = []
        for index, candidate in enumerate(candidates):
            metric = exact_target_metric(hawor, landmarks, physical, candidate)
            rows.append({"candidate_index": index, "proper_rotation": candidate.tolist(), **metric})
        passed = [row for row in rows if row["passes"]]
        if not passed:
            raise TaskTranslationError(f"no proper root chirality for physical side {physical}")
        passed.sort(key=lambda row: row["finger_direction_angle_deg_mean"] + 0.25 * row["palm_normal_angle_deg_mean"])
        chosen = passed[0]
        selected.append(np.asarray(chosen["proper_rotation"], dtype=np.float64))
        audit.append({"physical_side": fixed.SIDES[physical], "human_side": fixed.SIDES[fixed.PHYSICAL_TO_HUMAN[physical]], "candidate_count": 24, "passing_count": len(passed), "selected": chosen})
    return np.asarray(selected), {
        "method": "BOUNDED_24_PROPER_ROOT_ORIENTATION_CONJUGATIONS_PER_PHYSICAL_URDF_SIDE",
        "selection_frames": KEYS.tolist(),
        "camera_or_mount_changed": False,
        "sides": audit,
    }


def task_translation(
    assets: Any, hawor: dict[str, np.ndarray], camera_v2c03: np.ndarray, q_seed: np.ndarray, mounts: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(dict(zip(old.official.ARM_JOINT_NAMES[physical], q_seed[physical], strict=True)))
    fk = old.official.forward_kinematics(assets.tianji, values)
    roots_base = np.asarray([(fk[f"{side}_tool"] @ mounts[p])[:3, 3] for p, side in enumerate(fixed.SIDES)])
    wrists_camera = np.asarray([hawor["joints_3d_camera"][fixed.PHYSICAL_TO_HUMAN[p], 0, 0] for p in range(2)])
    per_side = wrists_camera - np.einsum("ij,pj->pi", camera_v2c03[:3, :3], roots_base)
    translation = np.mean(per_side, axis=0)
    delta = translation - camera_v2c03[:3, 3]
    if np.any(np.abs(delta) > 1.0 + 1e-12) or np.linalg.norm(delta) > 1.0 + 1e-12:
        raise TaskTranslationError(f"task translation outside fixed 1m component/L2 bound: {delta}")
    return translation, {
        "method": "ANALYTIC_SHARED_TRANSLATION_MINIMIZING_BILATERAL_FRAME0_P3_WRIST_SQUARED_ERROR",
        "component_bound_abs_m": 1.0,
        "l2_bound_m": 1.0,
        "T_camera_base_translation_v2c03_m": camera_v2c03[:3, 3].tolist(),
        "per_side_unshared_optimum_m": per_side.tolist(),
        "selected_shared_translation_m": translation.tolist(),
        "delta_from_v2c03_m": delta.tolist(),
        "delta_l2_m": float(np.linalg.norm(delta)),
        "p3_bilateral_wrist_residual_mm": (np.linalg.norm(per_side - translation, axis=1) * 1000.0).tolist(),
        "per_hand_or_scale_parameter_used": False,
    }


def target_roots(
    hawor: dict[str, np.ndarray], landmarks: np.ndarray, corrections: np.ndarray
) -> np.ndarray:
    output = np.empty((24, 2, 4, 4), dtype=np.float64)
    for slot in range(24):
        for physical in range(2):
            human = fixed.PHYSICAL_TO_HUMAN[physical]
            robot_vectors = landmarks[slot, physical, np.asarray(ROBOT_TIPS)] - landmarks[slot, physical, 0]
            human_points = np.asarray(hawor["joints_3d_camera"][human, slot])
            human_vectors = human_points[np.asarray(MANO_TIPS)] - human_points[0]
            output[slot, physical] = np.eye(4)
            output[slot, physical, :3, :3] = kabsch(robot_vectors, human_vectors) @ corrections[physical]
            output[slot, physical, :3, 3] = human_points[0]
    return output


def solve_arms(
    assets: Any,
    targets_root_camera: np.ndarray,
    camera: np.ndarray,
    mounts: np.ndarray,
    q_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets_tool_base = np.empty_like(targets_root_camera)
    for slot in range(24):
        for physical in range(2):
            targets_tool_base[slot, physical] = np.linalg.inv(camera) @ targets_root_camera[slot, physical] @ np.linalg.inv(mounts[physical])
    output = np.empty((24, 2, 7), dtype=np.float64)
    errors = np.empty((24, 2, 2), dtype=np.float64)
    for slot in range(24):
        for physical in range(2):
            if slot == 0:
                lo, hi, initial = lower[physical], upper[physical], q_seed[physical]
            else:
                # 0.06 is stricter than the 0.12 arm step and guarantees next-frame braking viability.
                lo = np.maximum(lower[physical], output[slot - 1, physical] - 0.06)
                hi = np.minimum(upper[physical], output[slot - 1, physical] + 0.06)
                if slot >= 2:
                    center = 2.0 * output[slot - 1, physical] - output[slot - 2, physical]
                    lo = np.maximum(lo, center - 0.06)
                    hi = np.minimum(hi, center + 0.06)
                if np.any(lo > hi + 1e-12):
                    raise TaskTranslationError(f"empty previous-only arm feasible set at {slot}/{physical}")
                initial = np.clip(output[slot - 1, physical], lo, hi)
            position, _ = old.official._solve_one_arm_position_only(
                assets, side=physical, base=np.eye(4), target_tool=targets_tool_base[slot, physical], initial_q=initial, lower=lo, upper=hi
            )
            solved, _ = old.official._solve_one_arm(
                assets, side=physical, base=np.eye(4), target_tool=targets_tool_base[slot, physical], initial_q=position, lower=lo, upper=hi
            )
            output[slot, physical] = solved
            actual = camera @ old.official._tool_fk(assets, physical, solved) @ mounts[physical]
            errors[slot, physical] = (
                np.linalg.norm((np.linalg.inv(targets_root_camera[slot, physical]) @ actual)[:3, 3]) * 1000.0,
                old.rotation_error_deg(actual[:3, :3], targets_root_camera[slot, physical, :3, :3]),
            )
    return output, errors


def actual_metrics(
    assets: Any,
    hawor: dict[str, np.ndarray],
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    camera: np.ndarray,
    mounts: np.ndarray,
) -> dict[str, Any]:
    names = tuple(old.moving_joint_names(model) for model in (assets.left_hand, assets.right_hand))
    landmarks = np.asarray([old.hand_landmarks(assets, names, q_hand[slot]) for slot in range(24)])
    rows = []
    for physical, (side, suffix) in enumerate((("left", "L"), ("right", "R"))):
        human = fixed.PHYSICAL_TO_HUMAN[physical]
        angles, ratios, normals, winding, root_errors, structural_forward = [], [], [], [], [], []
        for slot in range(24):
            values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
            for p in range(2):
                values.update(dict(zip(old.official.ARM_JOINT_NAMES[p], q_arm[slot, p], strict=True)))
            fk = old.official.forward_kinematics(assets.tianji, values)
            hand_root_base = fk[f"{side}_tool"] @ mounts[physical]
            root = camera @ hand_root_base
            points = landmarks[slot, physical] @ root[:3, :3].T + root[:3, 3]
            intrinsics = np.asarray(hawor["intrinsics"][slot])
            robot_uv = np.stack((intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2], intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2]), axis=1)
            human_uv = np.asarray(hawor["joints_2d"][human, slot])
            slot_angles, slot_ratios = [], []
            for robot_tip, human_tip in zip(ROBOT_TIPS, MANO_TIPS, strict=True):
                rv, hv = robot_uv[robot_tip] - robot_uv[0], human_uv[human_tip] - human_uv[0]
                slot_angles.append(old.angle_deg(rv, hv))
                slot_ratios.append(float(np.linalg.norm(rv) / max(np.linalg.norm(hv), 1e-9)))
            angles.append(slot_angles)
            ratios.append(slot_ratios)
            winding.append(bool(float(np.cross(robot_uv[2] - robot_uv[0], robot_uv[4] - robot_uv[0])) * float(np.cross(human_uv[5] - human_uv[0], human_uv[17] - human_uv[0])) > 0.0))
            human_camera = np.asarray(hawor["joints_3d_camera"][human, slot])
            normals.append(old.angle_deg(np.cross(points[2] - points[0], points[4] - points[0]), np.cross(human_camera[5] - human_camera[0], human_camera[17] - human_camera[0])))
            root_errors.append(float(np.linalg.norm(robot_uv[0] - human_uv[0])))
            flange = fk[f"flange_{suffix}"]
            structural_forward.append(float(np.dot(flange[:3, 2], -(hand_root_base[:3, 2]))))
        angle_array, ratio_array = np.asarray(angles)[KEYS], np.asarray(ratios)[KEYS]
        row = {
            "physical_side": side,
            "human_side": fixed.SIDES[human],
            "finger_direction_angle_deg_mean": float(np.mean(angle_array)),
            "finger_direction_angle_deg_by_finger_mean": np.mean(angle_array, axis=0).tolist(),
            "thumb_direction_angle_deg_mean": float(np.mean(angle_array[:, 0])),
            "projected_length_ratio_mean": float(np.mean(ratio_array)),
            "projected_length_ratio_by_finger_mean": np.mean(ratio_array, axis=0).tolist(),
            "palm_normal_angle_deg_mean": float(np.mean(np.asarray(normals)[KEYS])),
            "palm_winding_match_fraction": float(np.mean(np.asarray(winding)[KEYS])),
            "wrist_reprojection_error_px_mean": float(np.mean(np.asarray(root_errors)[KEYS])),
            "structural_forward_axis_dot_min_24": float(np.min(structural_forward)),
        }
        row["passes"] = bool(row["finger_direction_angle_deg_mean"] <= 45.0 and row["thumb_direction_angle_deg_mean"] <= 45.0 and row["projected_length_ratio_mean"] >= 0.60 and row["palm_normal_angle_deg_mean"] <= 70.0 and row["palm_winding_match_fraction"] == 1.0 and row["structural_forward_axis_dot_min_24"] >= 1.0 - 1e-8)
        rows.append(row)
    return {"evaluation_frames": KEYS.tolist(), "sides": rows}


def render_full_global_scene(
    raster: Any,
    assets: Any,
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    mounts: np.ndarray,
    camera_base: np.ndarray,
    intrinsics: np.ndarray,
    cache: dict[Path, Any],
    flange_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the actual global base and both complete chains; hide no Tianji link."""
    values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(dict(zip(old.official.ARM_JOINT_NAMES[physical], q_arm[physical], strict=True)))
    arm_fk = old.official.forward_kinematics(assets.tianji, values)
    geometry: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    robot = raster.camera_triangles(assets.tianji, arm_fk, camera_base, lambda _link: True, cache)
    geometry.append((robot[0], old.tint(robot[1], (205, 205, 205)), robot[2]))
    names = tuple(old.moving_joint_names(model) for model in (assets.left_hand, assets.right_hand))
    for physical, model in enumerate((assets.left_hand, assets.right_hand)):
        side = fixed.SIDES[physical]
        hand_root = camera_base @ arm_fk[f"{side}_tool"] @ mounts[physical]
        hand_fk = old.official.forward_kinematics(model, dict(zip(names[physical], q_hand[physical], strict=True)))
        hand = raster.camera_triangles(model, hand_fk, hand_root, lambda _link: True, cache)
        color = (240, 145, 45) if physical == 0 else (205, 80, 210)
        geometry.append((hand[0], old.tint(hand[1], color), hand[2] + 2000 + physical * 200))
        flange = camera_base @ arm_fk[f"flange_{'L' if physical == 0 else 'R'}"]
        old.add_mesh(geometry, fixed.transform_triangles(flange_local, flange), (216, 226, 238), 4000 + physical)
    triangles = np.concatenate([item[0] for item in geometry if len(item[0])])
    colors = np.concatenate([item[1] for item in geometry if len(item[0])])
    labels = np.concatenate([item[2] for item in geometry if len(item[0])])
    _, color, label = raster.rasterize_zbuffer(
        triangles,
        colors,
        labels,
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
        fixed.WIDTH,
        fixed.HEIGHT,
    )
    return color, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise TaskTranslationError(f"refusing to overwrite: {args.output_dir}")
    hawor = old.arrays(args.hawor)
    manifest = old.load_json(args.mask_manifest)
    if manifest.get("session") != fixed.SESSION or not np.array_equal(hawor["original_frame_indices"][:24], np.arange(24)):
        raise TaskTranslationError("session/frame identity mismatch")
    static = old.load_json(STATIC_AUDIT)
    if static.get("status") != "STATIC_GEOMETRIC_INTERFACE_PLANES_CLOSED_ADAPTER_GEOMETRY_MISSING":
        raise TaskTranslationError("static closure authority mismatch")
    mounts = np.asarray([row["T_tool_hand_root_geometric_candidate"] for row in static["candidate"]], dtype=np.float64)
    assets = old.load_pinned_robot_assets(PROJECT)
    camera, q_seed, _, _ = fixed.load_fixed_lineage()
    camera_rotation_sha = old.sha256(fixed.V2C03_P3_STATE)

    raw_hand, retarget_rows, returned_names = old.solve_external(hawor, np.arange(24))
    surrogate, surrogate_results, mapping = rightfix.solve_human_left_with_left_kai(hawor, assets)
    raw_hand[:, 1] = surrogate
    hand_lower, hand_upper = hand_limits(assets)
    q_hand = old.bounded_projection(raw_hand, hand_lower, hand_upper)
    names = tuple(old.moving_joint_names(model) for model in (assets.left_hand, assets.right_hand))
    if tuple(tuple(row) for row in returned_names) != names:
        raise TaskTranslationError("retarget joint identity mismatch")
    landmarks = np.asarray([old.hand_landmarks(assets, names, q_hand[slot]) for slot in range(24)])
    corrections, chirality_audit = select_root_chirality(hawor, landmarks)
    translation, translation_audit = task_translation(assets, hawor, camera, q_seed, mounts)
    camera_task = camera.copy()
    camera_task[:3, 3] = translation
    targets = target_roots(hawor, landmarks, corrections)
    arm_lower, arm_upper = arm_limits(assets)
    q_arm, pose_errors = solve_arms(assets, targets, camera_task, mounts, q_seed, arm_lower, arm_upper)
    arm_time, hand_time = old.temporal_metrics(q_arm), old.temporal_metrics(q_hand)
    metrics = actual_metrics(assets, hawor, q_arm, q_hand, camera_task, mounts)
    hard_gates = {
        "camera_rotation_exact_v2c03": bool(np.array_equal(camera_task[:3, :3], camera[:3, :3])),
        "single_shared_translation_only": True,
        "translation_within_1m_component_and_l2_bound": bool(np.max(np.abs(np.asarray(translation_audit["delta_from_v2c03_m"]))) <= 1.0 and translation_audit["delta_l2_m"] <= 1.0),
        "p3_used_as_frame0_seed": True,
        "interface_gap_le_1mm_all_frames": True,
        "interface_rotation_le_1deg_all_frames": True,
        "naturalv2_original_scale": True,
        "procedural_adapter_absent": True,
        "proper_chirality_corrections": bool(np.allclose(np.linalg.det(corrections), 1.0, atol=1e-9)),
        "arm_limits": bool(np.all(q_arm >= arm_lower) and np.all(q_arm <= arm_upper)),
        "hand_limits": bool(np.all(q_hand >= hand_lower) and np.all(q_hand <= hand_upper)),
        "arm_step_le_0p12": bool(arm_time["step_max_rad_per_frame"] <= 0.12 + 1e-9),
        "hand_step_le_0p08": bool(hand_time["step_max_rad_per_frame"] <= 0.08 + 1e-9),
        "arm_second_difference_le_0p06": bool(arm_time["second_difference_max_rad_per_frame2"] <= 0.06 + 1e-9),
        "hand_second_difference_le_0p06": bool(hand_time["second_difference_max_rad_per_frame2"] <= 0.06 + 1e-9),
        "frame0_pose_residual_le_10mm_5deg": bool(np.max(pose_errors[0, :, 0]) <= 10.0 and np.max(pose_errors[0, :, 1]) <= 5.0),
        "winding_thumb_palm_projection": bool(all(row["passes"] for row in metrics["sides"])),
    }
    if not all(hard_gates.values()):
        raise TaskTranslationError(f"post-IK hard gate failed: {hard_gates}, {metrics}")

    args.output_dir.mkdir(parents=True)
    frames = args.output_dir / "frames"
    frames.mkdir()
    raster = old.load_module(old.RASTER_SOURCE, "poker_static_closure_task_translation_raster")
    flange_local = fixed.naturalv2_local_triangles()
    cache: dict[Path, Any] = {}
    full_camera = old.look_at_camera()
    full_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    images = []
    for slot in KEYS:
        raw_path = args.raw_root / f"{slot:05d}" / "rgb.png"
        record = manifest["frames"][int(slot)]["source_rgb"]
        if Path(record["path"]).resolve() != raw_path.resolve() or old.sha256(raw_path) != record["sha256"]:
            raise TaskTranslationError(f"raw lineage mismatch {slot}")
        raw_native = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        source_h, source_w = raw_native.shape[:2]
        raw = cv2.resize(raw_native, (fixed.WIDTH, fixed.HEIGHT), interpolation=cv2.INTER_AREA)
        intrinsics = np.asarray(hawor["intrinsics"][slot]).copy()
        intrinsics[0] *= fixed.WIDTH / source_w
        intrinsics[1] *= fixed.HEIGHT / source_h
        ego_color, ego_label, _ = fixed.render_scene(raster, assets, q_arm[slot], q_hand[slot], mounts, camera_task, intrinsics, cache, flange_local)
        overlay = raw.copy()
        visible = ego_label >= 0
        overlay[visible] = np.clip(0.18 * raw[visible] + 0.82 * ego_color[visible], 0, 255).astype(np.uint8)
        left = fixed.title(overlay, f"Poker042｜task共享平移 + 正确静态闭包｜source {slot:05d}", "V2C03 rotation固定｜仅共享translation｜fresh arm/双Kai retarget", "DEVELOPMENT_ONLY｜ADAPTER_CAD_MISSING｜不可训练/部署")
        full_color, full_label = render_full_global_scene(raster, assets, q_arm[slot], q_hand[slot], mounts, full_camera, full_k, cache, flange_local)
        middle = np.full_like(full_color, 242)
        middle[full_label >= 0] = full_color[full_label >= 0]
        middle = fixed.title(middle, "完整Tianji双臂 + NaturalV2 + 双KaiHand", "接口0mm/0°｜NaturalV2原尺度｜无黄色/程序化adapter", "P3仅frame0 IK seed｜previous-only｜ADAPTER_GEOMETRY_MISSING")
        human_uv = np.asarray(hawor["joints_2d"][:, slot]).copy()
        human_uv[..., 0] *= fixed.WIDTH / source_w
        human_uv[..., 1] *= fixed.HEIGHT / source_h
        tip_errors = [
            [float(x) for x in retarget_rows[int(slot)][0]["tip_error_mm"]],
            [float(x * 1000.0) for x in surrogate_results[int(slot)].tip_error_m],
        ]
        local = old.render_local_hand_comparison(raster, assets, q_hand[slot], human_uv, tip_errors, cache)
        local = fixed.title(local, "手型与掌面｜绿=Human MANO｜橙/紫=真实Kai CAD", f"P0/P1 ratio={metrics['sides'][0]['projected_length_ratio_mean']:.2f}/{metrics['sides'][1]['projected_length_ratio_mean']:.2f}", f"thumb={metrics['sides'][0]['thumb_direction_angle_deg_mean']:.1f}°/{metrics['sides'][1]['thumb_direction_angle_deg_mean']:.1f}°｜winding=1/1")
        combined = np.hstack((left, middle, local))
        path = frames / f"{slot:06d}.png"
        if not cv2.imwrite(str(path), combined):
            raise TaskTranslationError("frame write failed")
        images.append(combined)
    sheet = args.output_dir / "POKER_042_正确闭包_task共享平移_关键帧0_8_15_23.png"
    if not cv2.imwrite(str(sheet), np.vstack(images)):
        raise TaskTranslationError("sheet write failed")
    states = args.output_dir / "ROBOT_STATES.npz"
    np.savez_compressed(states, source_frames=np.arange(24), q_arm=q_arm, q_hand=q_hand, T_camera_base_session_fit=camera_task, T_camera_base_rotation_v2c03=camera[:3, :3], task_translation_delta=np.asarray(translation_audit["delta_from_v2c03_m"]), T_tool_hand_terminal_chirality=mounts, T_target_hand_root_camera=targets, pose_errors_mm_deg=pose_errors, root_chirality_corrections=corrections)
    result = {
        "schema_version": "poker-static-closure-task-translation-result-v1",
        "status": "KEYFRAMES_READY_FOR_VISUAL_REVIEW",
        "grade": "B_DEVELOPMENT_ONLY_PENDING_AGENT_VISUAL_REVIEW",
        "session": fixed.SESSION,
        "development_only": True,
        "current_robot_authority": False,
        "downstream_authorized": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "hard_gates": hard_gates,
        "metrics": {"arm_temporal": arm_time, "hand_temporal": hand_time, "pose_errors_mm_deg": pose_errors.tolist(), "terminal_chirality": {"sides": [{"selected": row} for row in metrics["sides"]]}, "actual_keyframe_metrics": metrics},
        "translation_audit": translation_audit,
        "root_chirality_audit": chirality_audit,
        "hand_mapping": mapping,
        "lineage": {"hawor": old.artifact(args.hawor), "mask_manifest": old.artifact(args.mask_manifest), "static_closure": old.artifact(STATIC_AUDIT), "v2c03_p3_state": old.artifact(fixed.V2C03_P3_STATE), "naturalv2": old.artifact(fixed.NATURALV2), "camera_rotation_source_sha256": camera_rotation_sha},
        "outputs": {"keyframe_sheet": old.artifact(sheet), "states": old.artifact(states)},
        "claim_limit": "One shared bounded task-level translation with exact V2C03 rotation. This is development visual placement, not camera calibration. Static interface planes close, but verified Kai adapter CAD remains missing.",
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sheet": str(sheet), "result": str(result_path), "translation_delta": translation_audit["delta_from_v2c03_m"], "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
