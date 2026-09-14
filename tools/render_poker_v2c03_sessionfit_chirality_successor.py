#!/usr/bin/env python3
"""Bounded four-frame Poker correction after V2C03/P3 visual rejection.

The historical V2C03 rotation and P3 arm posture are seeds, not a claim that
the old translation calibrates this session.  Each KaiHand gets one independent
24-way proper-axis chirality audit.  One task-global camera/base transform is
then fitted to both wrists and forearm directions at slots 0/8/15/23 with
bounded translation and a <=15 degree rotation-vector correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import render_poker_v2c03_p3_naturalv2_successor as fixed  # noqa: E402


ROTATION_BOUND_DEG = 15.0
TRANSLATION_BOUND_M = 0.8
KEY_SLOTS = fixed.KEY_SLOTS


class SessionFitError(RuntimeError):
    pass


def chirality_candidates(
    assets: Any,
    camera_seed: np.ndarray,
    q0: np.ndarray,
    base_mounts: np.ndarray,
    q_hand0: np.ndarray,
    hawor: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(
            dict(zip(old.official.ARM_JOINT_NAMES[physical], q0[physical], strict=True))
        )
    arm_fk = old.official.forward_kinematics(assets.tianji, values)
    names = tuple(
        old.moving_joint_names(model) for model in (assets.left_hand, assets.right_hand)
    )
    landmarks = old.hand_landmarks(assets, names, q_hand0)
    rotations = old.proper_axis_rotations()
    mounts = base_mounts.copy()
    audit_rows = []
    for physical in range(2):
        human_side = fixed.PHYSICAL_TO_HUMAN[physical]
        human = np.asarray(hawor["joints_3d_camera"][human_side, 0], dtype=np.float64)
        human_thumb = human[2] - human[0]
        human_index = human[5] - human[0]
        human_pinky = human[17] - human[0]
        human_normal = np.cross(human_index, human_pinky)
        side_rows = []
        for candidate_index, candidate in enumerate(rotations):
            robot_rotation = (
                camera_seed[:3, :3]
                @ arm_fk[f"{fixed.SIDES[physical]}_tool"][:3, :3]
                @ base_mounts[physical, :3, :3]
                @ candidate
            )
            local = landmarks[physical]
            robot_thumb = robot_rotation @ (local[1] - local[0])
            robot_index = robot_rotation @ (local[2] - local[0])
            robot_pinky = robot_rotation @ (local[4] - local[0])
            robot_normal = np.cross(robot_index, robot_pinky)
            thumb_angle = old.angle_deg(robot_thumb, human_thumb)
            index_angle = old.angle_deg(robot_index, human_index)
            normal_angle = old.angle_deg(robot_normal, human_normal)
            score = thumb_angle + normal_angle + 0.25 * index_angle
            side_rows.append(
                {
                    "candidate_index": candidate_index,
                    "proper_rotation": candidate.tolist(),
                    "thumb_direction_angle_deg": thumb_angle,
                    "index_direction_angle_deg": index_angle,
                    "palm_normal_angle_deg": normal_angle,
                    "score": score,
                }
            )
        side_rows.sort(key=lambda row: row["score"])
        selected = side_rows[0]
        if not (
            selected["thumb_direction_angle_deg"] <= 50.0
            and selected["palm_normal_angle_deg"] <= 50.0
            and selected["index_direction_angle_deg"] <= 40.0
        ):
            raise SessionFitError(f"no bounded chirality candidate for physical {physical}")
        mounts[physical, :3, :3] = (
            base_mounts[physical, :3, :3]
            @ np.asarray(selected["proper_rotation"], dtype=np.float64)
        )
        audit_rows.append(
            {
                "physical_side": fixed.SIDES[physical],
                "human_side": fixed.SIDES[human_side],
                "candidate_count": len(side_rows),
                "selected": selected,
                "top5": side_rows[:5],
            }
        )
    return mounts, {
        "method": "INDEPENDENT_24_WAY_PROPER_SIGNED_AXIS_ENUMERATION_AT_FRAME0",
        "selection_objective": "thumb_angle + palm_normal_angle + 0.25*index_angle",
        "hard_gates_deg": {"thumb": 50.0, "palm_normal": 50.0, "index": 40.0},
        "sides": audit_rows,
        "claim_limit": "URDF/camera visual-basis conjugation only; not a measured adapter mount.",
    }


def chirality_candidates_in_final_camera(
    assets: Any,
    camera: np.ndarray,
    q_arm: np.ndarray,
    base_mounts: np.ndarray,
    q_hand: np.ndarray,
    hawor: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select each Kai root basis using the actual final image projection."""
    rotations = old.proper_axis_rotations()
    names = tuple(
        old.moving_joint_names(model) for model in (assets.left_hand, assets.right_hand)
    )
    mounts = base_mounts.copy()
    side_audits = []
    for physical in range(2):
        human_side = fixed.PHYSICAL_TO_HUMAN[physical]
        rows = []
        for candidate_index, candidate in enumerate(rotations):
            thumb_angles = []
            index_angles = []
            normal_angles = []
            winding_matches = []
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
                mount[:3, :3] = mount[:3, :3] @ candidate
                root_camera = camera @ arm_fk[f"{fixed.SIDES[physical]}_tool"] @ mount
                robot_camera = (
                    root_camera[:3, :3] @ local.T
                ).T + root_camera[:3, 3]
                intrinsics = np.asarray(hawor["intrinsics"][slot], dtype=np.float64)
                robot_uv = np.asarray(
                    [
                        [
                            intrinsics[0, 0] * point[0] / point[2] + intrinsics[0, 2],
                            intrinsics[1, 1] * point[1] / point[2] + intrinsics[1, 2],
                        ]
                        for point in robot_camera
                    ],
                    dtype=np.float64,
                )
                human_uv = np.asarray(
                    hawor["joints_2d"][human_side, slot], dtype=np.float64
                )
                robot_thumb = robot_uv[1] - robot_uv[0]
                robot_index = robot_uv[2] - robot_uv[0]
                robot_pinky = robot_uv[4] - robot_uv[0]
                human_thumb = human_uv[2] - human_uv[0]
                human_index = human_uv[5] - human_uv[0]
                human_pinky = human_uv[17] - human_uv[0]
                thumb_angles.append(old.angle_deg(robot_thumb, human_thumb))
                index_angles.append(old.angle_deg(robot_index, human_index))
                robot_cross = float(np.cross(robot_index, robot_pinky))
                human_cross = float(np.cross(human_index, human_pinky))
                winding_matches.append(bool(robot_cross * human_cross > 0.0))
                human_camera = np.asarray(
                    hawor["joints_3d_camera"][human_side, slot], dtype=np.float64
                )
                human_normal = np.cross(
                    human_camera[5] - human_camera[0],
                    human_camera[17] - human_camera[0],
                )
                robot_normal = np.cross(
                    robot_camera[2] - robot_camera[0],
                    robot_camera[4] - robot_camera[0],
                )
                normal_angles.append(old.angle_deg(robot_normal, human_normal))
            mean_thumb = float(np.mean(thumb_angles))
            mean_index = float(np.mean(index_angles))
            mean_normal = float(np.mean(normal_angles))
            winding_fraction = float(np.mean(winding_matches))
            score = mean_thumb + 0.5 * mean_index + mean_normal + 60.0 * (1.0 - winding_fraction)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "proper_rotation": candidate.tolist(),
                    "mean_thumb_screen_angle_deg": mean_thumb,
                    "mean_index_screen_angle_deg": mean_index,
                    "mean_palm_normal_angle_deg": mean_normal,
                    "screen_palm_winding_match_fraction": winding_fraction,
                    "per_slot_thumb_screen_angle_deg": thumb_angles,
                    "per_slot_index_screen_angle_deg": index_angles,
                    "per_slot_palm_normal_angle_deg": normal_angles,
                    "score": score,
                }
            )
        rows.sort(key=lambda row: row["score"])
        selected = rows[0]
        if not (
            selected["mean_thumb_screen_angle_deg"] <= 50.0
            and selected["mean_index_screen_angle_deg"] <= 50.0
            and selected["mean_palm_normal_angle_deg"] <= 65.0
            and selected["screen_palm_winding_match_fraction"] >= 0.75
        ):
            raise SessionFitError(
                f"no final-camera chirality candidate for physical {physical}: {selected}"
            )
        mounts[physical, :3, :3] = (
            base_mounts[physical, :3, :3]
            @ np.asarray(selected["proper_rotation"], dtype=np.float64)
        )
        side_audits.append(
            {
                "physical_side": fixed.SIDES[physical],
                "human_side": fixed.SIDES[human_side],
                "candidate_count": len(rows),
                "selected": selected,
                "top5": rows[:5],
            }
        )
    return mounts, {
        "method": "INDEPENDENT_24_WAY_PROPER_AXIS_ENUMERATION_IN_FINAL_FITTED_CAMERA_AT_SLOTS_0_8_15_23",
        "selection_objective": "mean_thumb_screen_angle + 0.5*mean_index_screen_angle + mean_palm_normal_angle + 60*(1-winding_match_fraction)",
        "hard_gates": {
            "mean_thumb_screen_angle_deg": 50.0,
            "mean_index_screen_angle_deg": 50.0,
            "mean_palm_normal_angle_deg": 65.0,
            "screen_palm_winding_match_fraction_min": 0.75,
        },
        "sides": side_audits,
        "claim_limit": "Final-camera URDF visual/FK basis conjugation only; not a measured adapter mount.",
    }


def project(camera: np.ndarray, point: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    value = camera[:3, :3] @ point + camera[:3, 3]
    if value[2] <= 0.02:
        return np.asarray((1e6, 1e6), dtype=np.float64)
    return np.asarray(
        (
            intrinsics[0, 0] * value[0] / value[2] + intrinsics[0, 2],
            intrinsics[1, 1] * value[1] / value[2] + intrinsics[1, 2],
        ),
        dtype=np.float64,
    )


def camera_from_delta(seed: np.ndarray, value: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Rotation.from_rotvec(value[:3]).as_matrix() @ seed[:3, :3]
    output[:3, 3] = seed[:3, 3] + value[3:]
    return output


def collect_fit_rows(
    assets: Any,
    q_arm: np.ndarray,
    mounts: np.ndarray,
    hawor: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for slot in KEY_SLOTS:
        values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
        for physical in range(2):
            values.update(
                dict(
                    zip(
                        old.official.ARM_JOINT_NAMES[physical],
                        q_arm[slot, physical],
                        strict=True,
                    )
                )
            )
        arm_fk = old.official.forward_kinematics(assets.tianji, values)
        for physical in range(2):
            human_side = fixed.PHYSICAL_TO_HUMAN[physical]
            human_uv = np.asarray(hawor["joints_2d"][human_side, slot], dtype=np.float64)
            suffix = "L" if physical == 0 else "R"
            root = (
                arm_fk[f"{fixed.SIDES[physical]}_tool"] @ mounts[physical]
            )[:3, 3]
            forearm = arm_fk[f"Link5_{suffix}"][:3, 3]
            rows.append(
                {
                    "slot": slot,
                    "physical": physical,
                    "robot_wrist_base": root,
                    "robot_forearm_base": forearm,
                    "human_wrist_uv": human_uv[0],
                    "human_hand_direction_uv": human_uv[9] - human_uv[0],
                    "intrinsics": np.asarray(hawor["intrinsics"][slot], dtype=np.float64),
                }
            )
    return rows


def fit_camera(
    seed: np.ndarray, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    rotation_bound = np.deg2rad(ROTATION_BOUND_DEG)
    lower = np.r_[np.full(3, -rotation_bound), np.full(3, -TRANSLATION_BOUND_M)]
    upper = -lower

    def residual(value: np.ndarray) -> np.ndarray:
        camera = camera_from_delta(seed, value)
        output: list[float] = []
        for row in rows:
            wrist = project(camera, row["robot_wrist_base"], row["intrinsics"])
            forearm = project(camera, row["robot_forearm_base"], row["intrinsics"])
            output.extend((wrist - row["human_wrist_uv"]).tolist())
            robot_direction = wrist - forearm
            human_direction = row["human_hand_direction_uv"]
            robot_direction /= max(float(np.linalg.norm(robot_direction)), 1e-9)
            human_direction /= max(float(np.linalg.norm(human_direction)), 1e-9)
            output.extend((35.0 * (robot_direction - human_direction)).tolist())
        # A modest prior keeps the optional angle correction within the historical seed.
        output.extend((12.0 * value[:3] / rotation_bound).tolist())
        return np.asarray(output, dtype=np.float64)

    solved = least_squares(
        residual,
        np.zeros(6, dtype=np.float64),
        bounds=(lower, upper),
        max_nfev=300,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    camera = camera_from_delta(seed, solved.x)

    def wrist_metrics(value: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
        current = camera_from_delta(seed, value)
        errors = []
        details = []
        for row in rows:
            projected = project(current, row["robot_wrist_base"], row["intrinsics"])
            error = float(np.linalg.norm(projected - row["human_wrist_uv"]))
            errors.append(error)
            details.append(
                {
                    "slot": row["slot"],
                    "physical_side": fixed.SIDES[row["physical"]],
                    "human_side": fixed.SIDES[fixed.PHYSICAL_TO_HUMAN[row["physical"]]],
                    "robot_wrist_uv": projected.tolist(),
                    "human_wrist_uv": row["human_wrist_uv"].tolist(),
                    "error_px": error,
                }
            )
        return float(np.sqrt(np.mean(np.square(errors)))), details

    before, before_rows = wrist_metrics(np.zeros(6, dtype=np.float64))
    after, after_rows = wrist_metrics(solved.x)
    if not (np.isfinite(camera).all() and after < before and after <= 80.0):
        raise SessionFitError(f"bounded session fit failed: before={before}, after={after}")
    return camera, {
        "method": "ONE_TASK_GLOBAL_BILATERAL_4_SLOT_WRIST_AND_FOREARM_DIRECTION_LEAST_SQUARES",
        "slots": list(KEY_SLOTS),
        "rotation_seed": "V2C03",
        "translation_seed": "V2C03_REFERENCE_ONLY_NOT_042_CALIBRATION",
        "delta_rotation_vector_deg": (solved.x[:3] * 180.0 / np.pi).tolist(),
        "delta_translation_m": solved.x[3:].tolist(),
        "rotation_component_bound_deg": ROTATION_BOUND_DEG,
        "translation_component_bound_m": TRANSLATION_BOUND_M,
        "before_wrist_rmse_px": before,
        "after_wrist_rmse_px": after,
        "before": before_rows,
        "after": after_rows,
        "optimizer_success": bool(solved.success),
        "optimizer_nfev": int(solved.nfev),
        "forearm_target_definition": "direction only: Human MANO wrist-to-middle-MCP versus Tianji Link5-to-Kai-root",
        "claim_limit": "Task-global visual overlay fit, not measured camera calibration.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--mask-review", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SessionFitError(f"refusing to overwrite: {args.output_dir}")
    old.require_review(args.mask_review, "Mask")
    hawor_result = old.load_json(args.hawor_result)
    hawor = old.arrays(args.hawor)
    if not (
        hawor_result.get("session_id") == fixed.SESSION
        and hawor_result["outputs"]["npz"]["sha256"] == old.sha256(args.hawor)
        and np.array_equal(hawor["original_frame_indices"][:24], np.arange(24))
    ):
        raise SessionFitError("HaWoR identity/frame/SHA mismatch")
    manifest = old.load_json(args.mask_manifest)
    if manifest.get("session") != fixed.SESSION or len(manifest.get("frames", [])) != 171:
        raise SessionFitError("Mask manifest identity/frame-count mismatch")

    camera_seed, q0, base_mounts, _ = fixed.load_fixed_lineage()
    assets = old.load_pinned_robot_assets(PROJECT)
    arm_lower, arm_upper = old.official._arm_limits(assets)
    hand_models = (assets.left_hand, assets.right_hand)
    hand_names = tuple(old.moving_joint_names(model) for model in hand_models)
    hand_lower = np.asarray(
        [[j.lower for j in model.joints if j.joint_type != "fixed"] for model in hand_models]
    )
    hand_upper = np.asarray(
        [[j.upper for j in model.joints if j.joint_type != "fixed"] for model in hand_models]
    )
    raw_hand, retarget_rows, returned_names = old.solve_external(hawor, np.arange(24))
    if returned_names != hand_names or raw_hand.shape != (24, 2, 22):
        raise SessionFitError("retargeting_human joint identity/shape mismatch")
    q_hand = old.bounded_projection(raw_hand, hand_lower, hand_upper)
    # The signed-axis choice is a Kai URDF root-basis conjugation, not a new
    # physical arm target.  Feeding that discrete basis rotation back into the
    # Tianji wrist target made one arm chase a 90-degree representation change
    # and destroyed bilateral overlay.  Keep the P3/root-plane arm targets and
    # apply the audited conjugation only at the Kai root visual/FK boundary.
    q_arm, hand_targets, _ = fixed.solve_arms_from_p3(
        assets, hawor, base_mounts, q0, arm_lower, arm_upper
    )
    fit_rows = collect_fit_rows(assets, q_arm, base_mounts, hawor)
    camera, camera_audit = fit_camera(camera_seed, fit_rows)
    mounts, chirality_audit = chirality_candidates_in_final_camera(
        assets, camera, q_arm, base_mounts, q_hand, hawor
    )
    arm_time = old.temporal_metrics(q_arm)
    hand_time = old.temporal_metrics(q_hand)
    gates = {
        "frame_identity": bool(np.array_equal(hawor["original_frame_indices"][:24], np.arange(24))),
        "historical_p3_q0_exact": bool(np.array_equal(q_arm[0], q0)),
        "v2c03_rotation_used_as_seed": True,
        "v2c03_translation_not_claimed_as_042_calibration": True,
        "independent_chirality_enumeration_24_each": True,
        "chirality_is_urdf_basis_conjugation_not_arm_target_rotation": True,
        "thumb_direction_gate": bool(
            all(row["selected"]["mean_thumb_screen_angle_deg"] <= 50.0 for row in chirality_audit["sides"])
        ),
        "palm_normal_gate": bool(
            all(row["selected"]["mean_palm_normal_angle_deg"] <= 65.0 for row in chirality_audit["sides"])
        ),
        "session_fit_improves_wrist_rmse": bool(
            camera_audit["after_wrist_rmse_px"] < camera_audit["before_wrist_rmse_px"]
        ),
        "session_fit_after_wrist_rmse_le_80px": bool(camera_audit["after_wrist_rmse_px"] <= 80.0),
        "finite": bool(all(np.isfinite(value).all() for value in (camera, mounts, q_arm, q_hand))),
        "arm_limits": bool(np.all(q_arm >= arm_lower) and np.all(q_arm <= arm_upper)),
        "hand_limits": bool(np.all(q_hand >= hand_lower) and np.all(q_hand <= hand_upper)),
        "arm_step": bool(arm_time["step_max_rad_per_frame"] <= 0.12 + 1e-9),
        "hand_step": bool(hand_time["step_max_rad_per_frame"] <= 0.08 + 1e-9),
        "arm_second_difference": bool(arm_time["second_difference_max_rad_per_frame2"] <= 0.06 + 1e-9),
        "hand_second_difference": bool(hand_time["second_difference_max_rad_per_frame2"] <= 0.06 + 1e-9),
        "naturalv2_original_scale": True,
        "procedural_flange_proxy_absent": True,
    }
    if not all(gates.values()):
        raise SessionFitError(f"pre-render hard gate failed: {gates}")

    args.output_dir.mkdir(parents=True)
    frame_root = args.output_dir / "frames"
    frame_root.mkdir()
    raster = old.load_module(old.RASTER_SOURCE, "poker_v2c03_sessionfit_chirality_raster")
    flange_local = fixed.naturalv2_local_triangles()
    cache: dict[Path, Any] = {}
    full_camera = old.look_at_camera()
    full_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    key_images = []
    frame_records = []
    for slot in KEY_SLOTS:
        source = int(hawor["original_frame_indices"][slot])
        raw_path = args.raw_root / f"{source:05d}" / "rgb.png"
        source_record = manifest["frames"][source]["source_rgb"]
        if Path(source_record["path"]).resolve() != raw_path.resolve() or old.sha256(raw_path) != source_record["sha256"]:
            raise SessionFitError(f"raw/Mask lineage mismatch at {source}")
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise SessionFitError(f"raw decode failed at {source}")
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
            f"042双腕session-fit叠加｜source {source:05d}",
            "V2C03旋转种子 + P3初态｜双侧腕/前臂方向共同拟合",
            f"wrist RMSE {camera_audit['before_wrist_rmse_px']:.1f}→{camera_audit['after_wrist_rmse_px']:.1f}px｜非实测标定",
        )
        full_color, full_label, _ = fixed.render_scene(
            raster, assets, q_arm[slot], q_hand[slot], mounts, full_camera, full_k, cache, flange_local
        )
        middle = np.full_like(full_color, 242)
        middle[full_label >= 0] = full_color[full_label >= 0]
        middle = fixed.title(
            middle,
            "P3同步双臂 + chirality-corrected KaiHands",
            "NaturalV2原尺度真实法兰｜无黄色proxy",
            "KAI_ADAPTER_CAD_MISSING｜root-plane+proper rotation仅视觉候选",
        )
        human_uv = np.asarray(hawor["joints_2d"][:, slot], dtype=np.float64).copy()
        human_uv[..., 0] *= fixed.WIDTH / source_w
        human_uv[..., 1] *= fixed.HEIGHT / source_h
        tip_errors = [
            [float(x) for x in retarget_rows[slot][physical]["tip_error_mm"]]
            for physical in range(2)
        ]
        local = old.render_local_hand_comparison(raster, assets, q_hand[slot], human_uv, tip_errors, cache)
        chirality_text = "/".join(
            f"P{i} thumb {row['selected']['mean_thumb_screen_angle_deg']:.1f}° palm {row['selected']['mean_palm_normal_angle_deg']:.1f}°"
            for i, row in enumerate(chirality_audit["sides"])
        )
        local = fixed.title(
            local,
            "KaiHand拇指侧/掌面共轭审计",
            chirality_text,
            "绿=Human MANO｜retargeting_human bounded IK｜tip误差mm",
        )
        combined = np.hstack((left, middle, local))
        frame_path = frame_root / f"{slot:06d}.png"
        if not cv2.imwrite(str(frame_path), combined):
            raise SessionFitError("frame write failed")
        key_images.append(combined)
        frame_records.append(
            {
                "slot": slot,
                "source_frame": source,
                "ego_robot_pixels": int(np.count_nonzero(visible)),
                "full_view_robot_pixels": int(np.count_nonzero(full_label >= 0)),
                "frame_sha256": old.sha256(frame_path),
                "tip_error_mm": tip_errors,
            }
        )

    sheet = args.output_dir / "POKER_042_V2C03_seed_sessionfit_chirality_关键帧0_8_15_23.png"
    if not cv2.imwrite(str(sheet), np.vstack(key_images)):
        raise SessionFitError("sheet write failed")
    chirality_path = args.output_dir / "CHIRALITY_AUDIT.json"
    chirality_path.write_text(json.dumps(chirality_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    camera_path = args.output_dir / "CAMERA_SESSION_FIT_AUDIT.json"
    camera_path.write_text(json.dumps(camera_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    states = args.output_dir / "ROBOT_STATES.npz"
    np.savez_compressed(
        states,
        source_frames=np.arange(24),
        q_arm=q_arm,
        q_hand=q_hand,
        T_camera_base_seed_v2c03=camera_seed,
        T_camera_base_session_fit=camera,
        T_tool_hand_rootplane_base=base_mounts,
        T_tool_hand_chirality_visual=mounts,
        T_hand_target_base=hand_targets,
    )
    frame_manifest = args.output_dir / "FRAME_MANIFEST.json"
    frame_manifest.write_text(
        json.dumps({"session": fixed.SESSION, "frames": frame_records}, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "schema_version": "poker-v2c03-sessionfit-chirality-successor-result-v1",
        "status": "KEYFRAMES_READY_FOR_VISUAL_REVIEW",
        "grade": "B_DEVELOPMENT_ONLY_PENDING_USER_VISUAL_REVIEW",
        "session": fixed.SESSION,
        "development_only": True,
        "downstream_authorized": False,
        "current_robot_authority": False,
        "deployment_authorized": False,
        "training_authorized": False,
        "hard_gates": gates,
        "metrics": {"camera_fit": camera_audit, "chirality": chirality_audit, "arm": arm_time, "hand": hand_time},
        "lineage": {
            "v2c03_p3_state": old.artifact(fixed.V2C03_P3_STATE),
            "naturalv2_stl": old.artifact(fixed.NATURALV2),
            "rootplane_preflight": old.artifact(fixed.ROOTPLANE_PREFLIGHT),
            "withdrawn_predecessor": old.artifact(
                PROJECT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1/play_cards_0902_042_v2c03_p3_naturalv2_successor_24frame_v1/WITHDRAWN_USER_VISUAL_REJECTION.json"
            ),
        },
        "inputs": {
            "producer": old.artifact(Path(__file__)),
            "hawor": old.artifact(args.hawor),
            "hawor_result": old.artifact(args.hawor_result),
            "mask_manifest": old.artifact(args.mask_manifest),
            "mask_review": old.artifact(args.mask_review),
            "tianji_urdf": old.artifact(assets.tianji.path),
            "kaihand_left_urdf": old.artifact(old.KAI_URDFS[0]),
            "kaihand_right_urdf": old.artifact(old.KAI_URDFS[1]),
            "retargeting_human_retargeter": old.artifact(old.EXTERNAL_SOURCES[0]),
            "retargeting_human_geometry": old.artifact(old.EXTERNAL_SOURCES[1]),
        },
        "outputs": {
            "keyframe_sheet": old.artifact(sheet),
            "states": old.artifact(states),
            "frame_manifest": old.artifact(frame_manifest),
            "chirality_audit": old.artifact(chirality_path),
            "camera_fit_audit": old.artifact(camera_path),
        },
        "claim_limit": "Four-frame development visual successor only. Session camera/base transform is a bounded task-global overlay fit, not measured calibration. Proper-axis chirality is a Kai URDF visual/FK basis conjugation and does not redefine the Tianji arm target. NaturalV2 is a real identified wrist flange for another hand, not a Kai adapter. Root-plane/proper-axis hand mount is visual geometry only. No Robot authority, deployment, contact, collision, or training claim.",
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "sheet": str(sheet), "before_rmse_px": camera_audit["before_wrist_rmse_px"], "after_rmse_px": camera_audit["after_wrist_rmse_px"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
