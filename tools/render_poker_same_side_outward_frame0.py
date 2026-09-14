#!/usr/bin/env python3
"""Build one same-side Poker042 Robot pose still for user confirmation.

The still deliberately fixes two independent historical mistakes:

* physical-left follows human-left and physical-right follows human-right;
* the Robot is viewed from behind, looking outward like the human egocentric
  camera, so same-side wrists do not force the arms to cross through the body;
* all five Kai finger chains are solved independently, with the six-DoF thumb
  chain explicitly separated from the four-DoF regular fingers.

This program handles source frame 0 only and cannot encode a video or promote
Robot authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime
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

from pipeline import robot_wrist_kai_adapter as wrist_adapter  # noqa: E402
from pipeline.robot_hand_visual_alignment import (  # noqa: E402
    KAI_TIPS,
    MANO_TIPS,
    kai_visual_landmarks,
    measure_alignment,
    project_camera,
)
from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import render_poker_static_closure_task_translation_successor as taskfit  # noqa: E402
from tools import render_poker_v2c03_p3_naturalv2_successor as fixed  # noqa: E402
from tools import run_newtask_robot_shared_v4_hand as handfit  # noqa: E402


SIDES = ("left", "right")
SAME_SIDE = (0, 1)
FRAME_ID = 0
EXPECTED_HAWOR_SHA = "1822ed0c954ac2a294cd7860d23e51c125335fcdc4f1b0c07bed1f648ce066a8"
EXPECTED_STATIC_SHA = "83e14e42e0db26cef9645c28c4c0f33fef247664c354d828ca8b8270c6856e39"
# OpenCV/raster colors are BGR. BLUE/RED are identity annotations only. The
# arm/flange palette is sampled from the displayed pixels of the user-provided
# Robot004 visual-override video after Cycles/AgX.  The user subsequently chose
# white KaiHands; blue/red therefore remain identity annotations only.  The
# original Robot004 linear material bases are retained in the audit below,
# while the CPU renderer targets the requested displayed look.
BLUE = (235, 130, 15)              # left annotation, RGB=(15,130,235)
RED = (75, 86, 235)                # right annotation, RGB=(235,86,75)
ROBOT_IVORY = (205, 225, 235)      # Robot004 displayed arm, RGB=(235,225,205)
HAND_WHITE = (235, 242, 245)       # requested warm white hand, RGB=(245,242,235)
FLANGE_IVORY = (195, 218, 232)     # Robot004 displayed flange, RGB=(232,218,195)
REFERENCE_004_MATERIAL_LINEAR = {
    "arm": (0.60, 0.63, 0.68, 1.0),
    "hand": (0.55, 0.57, 0.61, 1.0),
    "connector": (0.34, 0.36, 0.40, 1.0),
    "metallic": 0.72,
    "roughness": 0.27,
}
REFERENCE_VIDEO_LOCAL_PATH = "/home/lemon/音乐/current_clean_full_robot_visual_override_4spp_full460 (2).mp4"
REFERENCE_VIDEO_BYTES = 12682152
REFERENCE_VIDEO_SHA256 = "efc98e79a71b2379793152688346e369c48af1260018e5e3d3ab5a1205ec1888"
MANO_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


class SameSideStillError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    return float(
        np.degrees(
            np.arccos(
                np.clip(float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b))), -1.0, 1.0)
            )
        )
    )


def refine_thumb_webs(
    hawor: dict[str, np.ndarray],
    assets: Any,
    contracts: list[dict[str, Any]],
    q_hand: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit true CAD thumb tips and thumb-index webs after five-chain solve."""
    output = np.asarray(q_hand, dtype=np.float64).copy()
    models = (assets.left_hand, assets.right_hand)
    group = handfit.GROUPS["thumb"]
    audits: list[dict[str, Any]] = []

    def robot_normalized(physical: int, q: np.ndarray) -> np.ndarray:
        contract = contracts[physical]
        points = kai_visual_landmarks(
            models[physical],
            tuple(contract["names"]),
            q,
            physical_side=SIDES[physical],
        )
        local = points @ contract["basis"]
        return local / float(np.linalg.norm(local[7]))

    def human_normalized(physical: int) -> np.ndarray:
        points = np.asarray(hawor["joints_3d_camera"][physical, FRAME_ID], dtype=np.float64)
        basis = wrist_adapter.final_v3_mano_palm_basis(points, handedness=SIDES[physical])
        local = (points - points[0]) @ basis
        return local / float(np.linalg.norm(local[12]))

    def web_metric(robot: np.ndarray, human: np.ndarray) -> dict[str, float]:
        return {
            "kai_web_angle_deg": _angle_deg(robot[5], robot[6]),
            "human_web_angle_deg": _angle_deg(human[4], human[8]),
            "kai_gap_over_middle_length": float(np.linalg.norm(robot[5] - robot[6])),
            "human_gap_over_middle_length": float(np.linalg.norm(human[4] - human[8])),
            "thumb_tip_normalized_l2": float(np.linalg.norm(robot[5] - human[4])),
        }

    for physical, side in enumerate(SIDES):
        contract = contracts[physical]
        human = human_normalized(physical)
        target_thumb = human[4]
        target_index = human[8]
        target_gap = float(np.linalg.norm(target_thumb - target_index))
        target_cosine = float(
            target_thumb @ target_index
            / (np.linalg.norm(target_thumb) * np.linalg.norm(target_index))
        )
        prior = output[physical, group].copy()
        before = web_metric(robot_normalized(physical, output[physical]), human)
        human_thumb_feature = handfit.human_features(
            wrist_adapter,
            np.asarray(hawor["joints_3d_camera"][physical, FRAME_ID], dtype=np.float64),
            side,
        )["thumb"]

        def residual(local: np.ndarray) -> np.ndarray:
            whole = output[physical].copy()
            whole[group] = local
            visual = robot_normalized(physical, whole)
            thumb = visual[5]
            index = visual[6]
            cosine = float(thumb @ index / (np.linalg.norm(thumb) * np.linalg.norm(index)))
            thumb_feature = handfit.robot_finger_features(
                old.official,
                wrist_adapter,
                contract,
                whole,
                "thumb",
                np.eye(3, dtype=np.float64),
            )
            return np.concatenate(
                (
                    thumb - target_thumb,
                    np.asarray((12.0 * (np.linalg.norm(thumb - index) - target_gap),)),
                    np.asarray((8.0 * (cosine - target_cosine),)),
                    0.35 * (thumb_feature["bones"] - human_thumb_feature["bones"]).ravel(),
                    0.02 * (local - prior),
                )
            )

        attempts = []
        for seed_name, seed in (
            ("current_five_chain", prior),
            ("limit_midpoint", contract["neutral"][group]),
            (
                "zero",
                np.clip(
                    np.zeros(len(group), dtype=np.float64),
                    contract["lower"][group],
                    contract["upper"][group],
                ),
            ),
        ):
            solved = least_squares(
                residual,
                seed,
                bounds=(contract["lower"][group], contract["upper"][group]),
                max_nfev=160,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            attempts.append(
                (
                    float(np.linalg.norm(residual(solved.x))),
                    seed_name,
                    np.asarray(solved.x),
                    int(solved.nfev),
                )
            )
        score, seed_name, selected, nfev = min(attempts, key=lambda row: (row[0], row[1]))
        output[physical, group] = selected
        after = web_metric(robot_normalized(physical, output[physical]), human)
        audits.append(
            {
                "physical_side": side,
                "method": "SEPARATE_SIX_DOF_TRUE_CAD_THUMB_TIP_WEB_REFINEMENT",
                "seed": seed_name,
                "attempts": len(attempts),
                "nfev": nfev,
                "residual_norm": score,
                "before": before,
                "after": after,
            }
        )
    return output, audits


def refine_visible_left_thumb(
    hawor: dict[str, np.ndarray],
    assets: Any,
    joint_names: tuple[tuple[str, ...], ...],
    q_pose_reference: np.ndarray,
    actual_roots: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Open the visible left thumb chain after the V3 whole-body pose is locked.

    A wrist-to-tip web scalar can improve while the proximal thumb still looks
    closed.  Fit the true CAD terminal plus link3/link4/link6 projections to
    the MANO thumb chain instead.  The right thumb and every non-thumb joint
    remain byte-identical to the approved V3 pose reference.
    """
    output = np.asarray(q_pose_reference, dtype=np.float64).copy()
    contracts = handfit.model_contract(old.official, wrist_adapter, assets)
    physical = 0
    side = SIDES[physical]
    model = assets.left_hand
    contract = contracts[physical]
    group = handfit.GROUPS["thumb"]
    prior = output[physical, group].copy()
    root = np.asarray(actual_roots[physical], dtype=np.float64)
    intrinsics = np.asarray(hawor["intrinsics"][FRAME_ID], dtype=np.float64)
    human_uv = np.asarray(hawor["joints_2d"][physical, FRAME_ID], dtype=np.float64)
    human_middle_length = float(np.linalg.norm(human_uv[12] - human_uv[0]))
    human = (human_uv - human_uv[0]) / human_middle_length
    target_gap = float(np.linalg.norm(human[4] - human[8]))
    human_thumb_feature = handfit.human_features(
        wrist_adapter,
        np.asarray(hawor["joints_3d_camera"][physical, FRAME_ID], dtype=np.float64),
        side,
    )["thumb"]

    def visual_points(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fk = old.official.forward_kinematics(
            model, dict(zip(joint_names[physical], q, strict=True))
        )
        prefix = "hand_l"
        chain = [np.zeros(3, dtype=np.float64)] + [
            fk[f"{prefix}_thumb_link{index}"][:3, 3] for index in range(1, 7)
        ]
        landmarks = kai_visual_landmarks(
            model, joint_names[physical], q, physical_side=side
        )
        chain.append(landmarks[5])
        chain_camera = np.asarray(chain) @ root[:3, :3].T + root[:3, 3]
        all_camera = landmarks @ root[:3, :3].T + root[:3, 3]
        chain_uv = project_camera(chain_camera, intrinsics)
        all_uv = project_camera(all_camera, intrinsics)
        robot_middle_length = float(np.linalg.norm(all_uv[7] - all_uv[0]))
        return (
            (chain_uv - chain_uv[0]) / robot_middle_length,
            (all_uv - all_uv[0]) / robot_middle_length,
        )

    def visible_metric(q: np.ndarray) -> dict[str, Any]:
        chain, landmarks = visual_points(q)
        return {
            "normalized_chain_wrist_link1_to_link6_true_tip": chain.tolist(),
            "true_tip_direction_error_deg": _angle_deg(chain[7], human[4]),
            "true_tip_normalized_l2": float(np.linalg.norm(chain[7] - human[4])),
            "thumb_index_gap_over_middle_length": float(
                np.linalg.norm(chain[7] - landmarks[6])
            ),
            "human_thumb_index_gap_over_middle_length": target_gap,
        }

    def residual(local: np.ndarray) -> np.ndarray:
        whole = output[physical].copy()
        whole[group] = local
        chain, landmarks = visual_points(whole)
        thumb_feature = handfit.robot_finger_features(
            old.official,
            wrist_adapter,
            contract,
            whole,
            "thumb",
            np.eye(3, dtype=np.float64),
        )
        # Robot link3/link4 approximate MANO joints2/3; link6 lies between
        # MANO joint3 and the terminal.  Tip and visible gap receive stronger
        # weights because those are the directly reviewed image features.
        return np.concatenate(
            (
                chain[3] - human[2],
                chain[4] - human[3],
                0.5 * (chain[6] - (0.35 * human[3] + 0.65 * human[4])),
                4.0 * (chain[7] - human[4]),
                np.asarray(
                    (8.0 * (np.linalg.norm(chain[7] - landmarks[6]) - target_gap),)
                ),
                2.0 * (thumb_feature["tip"] - human_thumb_feature["tip"]),
                0.05 * (thumb_feature["bones"] - human_thumb_feature["bones"]).ravel(),
                0.01 * (local - prior),
            )
        )

    attempts = []
    for seed_name, seed in (
        ("v3_pose_reference", prior),
        ("limit_midpoint", contract["neutral"][group]),
        (
            "zero",
            np.clip(
                np.zeros(len(group), dtype=np.float64),
                contract["lower"][group],
                contract["upper"][group],
            ),
        ),
    ):
        solved = least_squares(
            residual,
            seed,
            bounds=(contract["lower"][group], contract["upper"][group]),
            max_nfev=300,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        attempts.append(
            (
                float(np.linalg.norm(residual(solved.x))),
                seed_name,
                np.asarray(solved.x),
                int(solved.nfev),
            )
        )
    score, seed_name, selected, nfev = min(attempts, key=lambda row: (row[0], row[1]))
    before = visible_metric(output[physical])
    output[physical, group] = selected
    after = visible_metric(output[physical])
    audits = [
        {
            "physical_side": "left",
            "method": "VISIBLE_PROXIMAL_CHAIN_AND_TRUE_CAD_TIP_PROJECTED_REFINEMENT",
            "seed": seed_name,
            "attempts": len(attempts),
            "nfev": nfev,
            "residual_norm": score,
            "before": before,
            "after": after,
        },
        {
            "physical_side": "right",
            "method": "PRESERVE_V3_RIGHT_THUMB_BYTE_EXACT",
            "changed": False,
        },
    ]
    return output, audits


def solve_same_side_hands(
    hawor: dict[str, np.ndarray], assets: Any
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    tuple[tuple[str, ...], ...],
]:
    contracts = handfit.model_contract(old.official, wrist_adapter, assets)
    joint_names = tuple(tuple(contract["names"]) for contract in contracts)
    neutral = np.asarray([contract["neutral"] for contract in contracts])
    targets = [[
        handfit.human_features(
            wrist_adapter,
            np.asarray(hawor["joints_3d_camera"][physical, FRAME_ID], dtype=np.float64),
            SIDES[physical],
        )
        for physical in range(2)
    ]]
    # The legacy frozen thumb-axis adapter was fitted with the old cross-side
    # mapping (left<-right, right<-left).  Same-side anatomical palm bases are
    # already chirality-normalized, so applying that adapter here would double
    # rotate the thumb.  Identity below is an explicit same-side convention;
    # thumb joints 0:6 are nevertheless optimized as their own chain.
    thumb_rotations = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    weights = dict(handfit.FIT_GRID[0])
    q_hand, solve_rows = handfit.solve_frame(
        old.official,
        wrist_adapter,
        contracts,
        targets,
        thumb_rotations,
        neutral,
        neutral,
        FRAME_ID,
        weights,
    )
    # Preserve the exact five-chain solution used by the user-approved V3
    # whole-body pose.  The thumb-web refinement below may change only q_hand;
    # root selection, arm IK, and camera fitting must continue to use this
    # frozen reference so a local finger correction cannot move the robot.
    q_hand_pose_reference = q_hand.copy()
    thumb_web_rows = [
        {"physical_side": side, "method": "DEFER_UNTIL_WHOLE_BODY_POSE_LOCK"}
        for side in SIDES
    ]
    rows: list[dict[str, Any]] = []
    for physical, side in enumerate(SIDES):
        fingers = []
        for finger in handfit.FINGERS:
            actual = handfit.robot_finger_features(
                old.official,
                wrist_adapter,
                contracts[physical],
                q_hand[physical],
                finger,
                thumb_rotations[physical],
            )
            metric = handfit.feature_row(actual, targets[0][physical][finger])
            solve = next(
                row for row in solve_rows
                if row["physical_side"] == side and row["finger"] == finger
            )
            fingers.append({"finger": finger, **metric, **solve})
        rows.append(
            {
                "physical_side": side,
                "human_side": side,
                "method": "INDEPENDENT_FIVE_FINGER_CHAIN_LEAST_SQUARES",
                "thumb_policy": "SIX_DOF_THUMB_SOLVED_SEPARATELY_IN_SAME_SIDE_ANATOMICAL_BASIS",
                "regular_finger_policy": "FOUR_DOF_CHAIN_SOLVED_SEPARATELY_PER_FINGER",
                "weights": weights,
                "thumb_web_refinement": thumb_web_rows[physical],
                "fingers": fingers,
                "tip_direction_error_deg_max": max(
                    float(row["tip_direction_error_deg"]) for row in fingers
                ),
            }
        )
    lower = np.asarray([contract["lower"] for contract in contracts])
    upper = np.asarray([contract["upper"] for contract in contracts])
    if q_hand.shape != (2, 22) or not np.all(np.isfinite(q_hand)):
        raise SameSideStillError("same-side hand retarget produced invalid q")
    if np.any(q_hand < lower - 1e-9) or np.any(q_hand > upper + 1e-9):
        raise SameSideStillError("same-side hand retarget exceeds URDF limits")
    if max(float(row["tip_direction_error_deg_max"]) for row in rows) > 10.0:
        raise SameSideStillError("same-side per-finger retarget exceeds 10 degree tip bound")
    return q_hand, q_hand_pose_reference, rows, joint_names


def outward_camera_rotation(front_facing_rotation: np.ndarray) -> np.ndarray:
    """Turn a front-facing Robot review camera into egocentric outward view."""
    rotation = np.asarray(front_facing_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise SameSideStillError("one finite 3x3 front-facing camera rotation is required")
    return rotation @ Rotation.from_euler("z", 180.0, degrees=True).as_matrix()


def select_same_side_roots(
    hawor: dict[str, np.ndarray],
    assets: Any,
    joint_names: tuple[tuple[str, ...], ...],
    q_hand: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    models = (assets.left_hand, assets.right_hand)
    landmarks = np.asarray(
        [
            kai_visual_landmarks(
                models[physical], joint_names[physical], q_hand[physical],
                physical_side=SIDES[physical],
            )
            for physical in range(2)
        ]
    )
    rotations: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for physical, side in enumerate(SIDES):
        human = np.asarray(hawor["joints_3d_camera"][physical, FRAME_ID], dtype=np.float64)
        human_uv = np.asarray(hawor["joints_2d"][physical, FRAME_ID], dtype=np.float64)
        candidates = []
        for index, correction in enumerate(old.proper_axis_rotations()):
            rotation = (
                taskfit.kabsch(
                    landmarks[physical, np.asarray(KAI_TIPS)] - landmarks[physical, 0],
                    human[np.asarray(MANO_TIPS)] - human[0],
                )
                @ correction
            )
            points = landmarks[physical] @ rotation.T + human[0]
            metric = measure_alignment(
                points,
                human,
                human_uv,
                np.asarray(hawor["intrinsics"][FRAME_ID]),
                physical_side=side,
                human_side=side,
                other_wrist_camera=np.asarray(
                    hawor["joints_3d_camera"][1 - physical, FRAME_ID, 0]
                ),
                kai_root_rotation_camera=rotation,
            )
            direction_pass = (
                float(metric["kai_palmar_normal_camera"][0]) >= 0.45
                if physical == 0
                else float(metric["kai_palmar_normal_camera"][1]) >= 0.45
            )
            passed = bool(
                float(metric["palmar_normal_angle_deg"]) <= 45.0
                and float(metric["finger_direction_angle_deg_mean"]) <= 20.0
                and max(float(value) for value in metric["finger_direction_angle_deg"]) <= 30.0
                and float(metric["thumb_index_web_angle_deg"]["abs_error"]) <= 10.0
                and float(metric["thumb_index_gap_over_middle_length"]["abs_error"]) <= 0.22
                and float(metric["five_tip_isotropic_overlay_nrmse"]) <= 0.23
                and direction_pass
            )
            score = float(
                metric["palmar_normal_angle_deg"]
                + metric["finger_direction_angle_deg_mean"]
                + metric["thumb_index_web_angle_deg"]["abs_error"]
                + 20.0 * metric["five_tip_isotropic_overlay_nrmse"]
            )
            candidates.append(
                {
                    "candidate_index": index,
                    "passes": passed,
                    "direction_pass": direction_pass,
                    "score": score,
                    "rotation": rotation,
                    "metric": metric,
                }
            )
        passing = [row for row in candidates if row["passes"]]
        if not passing:
            raise SameSideStillError(f"no frame-0 same-side root candidate for {side}")
        selected = min(passing, key=lambda row: float(row["score"]))
        rotations.append(np.asarray(selected["rotation"]))
        audits.append(
            {
                "physical_side": side,
                "human_side": side,
                "candidate_count": len(candidates),
                "passing_count": len(passing),
                "selected_index": selected["candidate_index"],
                "score": selected["score"],
                "metric": selected["metric"],
            }
        )
    return np.asarray(rotations), audits


def solve_shared_camera_and_arms(
    hawor: dict[str, np.ndarray],
    assets: Any,
    mounts: np.ndarray,
    target_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    camera, q_seed, _, _ = fixed.load_fixed_lineage()
    front_facing_rotation = camera[:3, :3].copy()
    camera_rotation = outward_camera_rotation(front_facing_rotation)
    lower, upper = taskfit.arm_limits(assets)
    values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(dict(zip(old.official.ARM_JOINT_NAMES[physical], q_seed[physical], strict=True)))
    seed_fk = old.official.forward_kinematics(assets.tianji, values)
    seed_roots = np.asarray(
        [
            (seed_fk[f"{side}_tool"] @ mounts[physical])[:3, 3]
            for physical, side in enumerate(SIDES)
        ]
    )
    wrists = np.asarray(
        [hawor["joints_3d_camera"][physical, FRAME_ID, 0] for physical in range(2)]
    )
    per_side_translation = wrists - np.einsum("ij,pj->pi", camera_rotation, seed_roots)
    translation_seed = np.mean(per_side_translation, axis=0)
    targets = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    targets[:, :3, :3] = target_rotations
    targets[:, :3, 3] = wrists

    camera[:3, :3] = camera_rotation
    camera[:3, 3] = translation_seed
    q_initial = np.empty((2, 7), dtype=np.float64)
    for physical in range(2):
        target_tool = np.linalg.inv(camera) @ targets[physical] @ np.linalg.inv(mounts[physical])
        position, _ = old.official._solve_one_arm_position_only(
            assets, side=physical, base=np.eye(4), target_tool=target_tool,
            initial_q=q_seed[physical], lower=lower[physical], upper=upper[physical],
        )
        q_initial[physical], _ = old.official._solve_one_arm(
            assets, side=physical, base=np.eye(4), target_tool=target_tool,
            initial_q=position, lower=lower[physical], upper=upper[physical],
        )

    def residual(vector: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = camera_rotation
        transform[:3, 3] = vector[:3]
        q_arm = vector[3:].reshape(2, 7)
        terms: list[float] = []
        for physical in range(2):
            actual = (
                transform
                @ old.official._tool_fk(assets, physical, q_arm[physical])
                @ mounts[physical]
            )
            terms.extend(((actual[:3, 3] - targets[physical, :3, 3]) / 0.015).tolist())
            terms.extend(
                (
                    Rotation.from_matrix(
                        targets[physical, :3, :3].T @ actual[:3, :3]
                    ).as_rotvec()
                    / 0.15
                ).tolist()
            )
        # Keep the static P3 posture as a meaningful branch prior.  The former
        # 0.02 coefficient admitted a numerically exact wrist fit whose elbows
        # folded behind and crossed the torso.
        terms.extend(((q_arm - q_seed) * 0.15).ravel().tolist())
        values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
        for physical in range(2):
            values.update(
                dict(zip(old.official.ARM_JOINT_NAMES[physical], q_arm[physical], strict=True))
            )
        fk = old.official.forward_kinematics(assets.tianji, values)
        for physical, suffix in enumerate(("L", "R")):
            side_sign = 1.0 if physical == 0 else -1.0
            link3 = fk[f"Link3_{suffix}"][:3, 3]
            link5 = fk[f"Link5_{suffix}"][:3, 3]
            # Geometry-only branch barriers: upper arm and elbow stay on their
            # physical side, and the forearm stays in front of the torso.
            terms.extend(
                (
                    2.0 * max(0.0, 0.25 - side_sign * float(link3[1])) / 0.10,
                    2.0 * max(0.0, 0.08 - side_sign * float(link5[1])) / 0.10,
                    2.0 * max(0.0, 0.08 - float(link5[0])) / 0.10,
                )
            )
        return np.asarray(terms)

    initial = np.concatenate((translation_seed, q_initial.ravel()))
    result = least_squares(
        residual,
        initial,
        bounds=(
            np.concatenate((translation_seed - 0.5, lower.ravel())),
            np.concatenate((translation_seed + 0.5, upper.ravel())),
        ),
        max_nfev=1200,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    camera[:3, 3] = result.x[:3]
    q_arm = result.x[3:].reshape(2, 7)
    actual_roots = np.empty((2, 4, 4), dtype=np.float64)
    errors: list[dict[str, float]] = []
    for physical in range(2):
        actual_roots[physical] = (
            camera
            @ old.official._tool_fk(assets, physical, q_arm[physical])
            @ mounts[physical]
        )
        errors.append(
            {
                "position_mm": float(
                    np.linalg.norm(actual_roots[physical, :3, 3] - targets[physical, :3, 3])
                    * 1000.0
                ),
                "rotation_deg": float(
                    np.degrees(
                        np.linalg.norm(
                            Rotation.from_matrix(
                                targets[physical, :3, :3].T
                                @ actual_roots[physical, :3, :3]
                            ).as_rotvec()
                        )
                    )
                ),
            }
        )
    if not result.success or max(row["position_mm"] for row in errors) > 3.0:
        raise SameSideStillError(f"joint frame-0 position fit failed: {errors}")
    if max(row["rotation_deg"] for row in errors) > 3.0:
        raise SameSideStillError(f"joint frame-0 orientation fit failed: {errors}")
    return camera, q_arm, actual_roots, errors


def same_direction_global_camera(ego_camera: np.ndarray) -> np.ndarray:
    """Keep the exact ego rotation and recenter only to show the whole robot."""
    ego = np.asarray(ego_camera, dtype=np.float64)
    if ego.shape != (4, 4) or not np.all(np.isfinite(ego)):
        raise SameSideStillError("one finite ego T_camera_base is required")
    rotation = ego[:3, :3].copy()
    target_base = np.asarray((0.0, 0.0, 0.90), dtype=np.float64)
    target_camera = np.asarray((0.0, 0.0, 3.20), dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_camera - rotation @ target_base
    return transform


def render_robot(
    raster: Any,
    assets: Any,
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    mounts: np.ndarray,
    camera: np.ndarray,
    intrinsics: np.ndarray,
    cache: dict[Path, object],
    flange_local: np.ndarray,
    *,
    complete_robot: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def reference_tint(
        raw_colors: np.ndarray,
        labels: np.ndarray,
        visuals: list[Any],
        target: tuple[int, int, int],
    ) -> np.ndarray:
        """Apply one Robot004 material while preserving per-triangle lighting."""
        result = np.zeros_like(raw_colors)
        for label in np.unique(labels):
            mask = labels == label
            link = visuals[int(label)].link
            source_base = np.asarray(raster.base_color(link), dtype=np.float64)
            shade = np.clip(
                np.mean(raw_colors[mask].astype(np.float64), axis=1)
                / max(float(np.mean(source_base)), 1e-9),
                0.42,
                1.0,
            )
            result[mask] = np.clip(
                np.asarray(target, dtype=np.float64)[None] * shade[:, None], 0, 255
            ).astype(np.uint8)
        return result

    def shaded_untextured_mesh(
        triangles: np.ndarray, target: tuple[int, int, int], label: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        triangles = triangles[valid]
        normals = normals[valid] / lengths[valid, None]
        light = np.asarray((-0.35, -0.55, -0.76), dtype=np.float64)
        light /= np.linalg.norm(light)
        shade = 0.42 + 0.58 * np.abs(normals @ light)
        colors = np.clip(
            np.asarray(target, dtype=np.float64)[None] * shade[:, None], 0, 255
        ).astype(np.uint8)
        return triangles, colors, np.full(len(triangles), label, dtype=np.int32)

    values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(dict(zip(old.official.ARM_JOINT_NAMES[physical], q_arm[physical], strict=True)))
    arm_fk = old.official.forward_kinematics(assets.tianji, values)
    selector = (lambda _link: True) if complete_robot else (lambda link: link in fixed.M6_ARM_LINKS)
    robot = raster.camera_triangles(assets.tianji, arm_fk, camera, selector, cache)
    robot_colors = reference_tint(
        robot[1], robot[2], assets.tianji.visuals, ROBOT_IVORY
    )
    geometry = [(robot[0], robot_colors, robot[2])]
    roots = []
    for physical, model in enumerate((assets.left_hand, assets.right_hand)):
        side = SIDES[physical]
        root = camera @ arm_fk[f"{side}_tool"] @ mounts[physical]
        roots.append(root)
        names = old.moving_joint_names(model)
        hand_fk = old.official.forward_kinematics(
            model, dict(zip(names, q_hand[physical], strict=True))
        )
        hand = raster.camera_triangles(model, hand_fk, root, lambda _link: True, cache)
        geometry.append(
            (
                hand[0],
                reference_tint(hand[1], hand[2], model.visuals, HAND_WHITE),
                hand[2] + 2000 + physical * 200,
            )
        )
        flange = camera @ arm_fk[f"flange_{'L' if physical == 0 else 'R'}"]
        geometry.append(
            shaded_untextured_mesh(
                fixed.transform_triangles(flange_local, flange),
                FLANGE_IVORY,
                4000 + physical,
            )
        )
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
    return color, label, np.asarray(roots)


def draw_human_skeletons(panel: np.ndarray, human_uv: np.ndarray) -> None:
    for side, color in enumerate((BLUE, RED)):
        points = human_uv[side]
        for chain in MANO_CHAINS:
            for first, second in zip(chain[:-1], chain[1:], strict=True):
                cv2.line(
                    panel,
                    tuple(np.rint(points[first]).astype(int)),
                    tuple(np.rint(points[second]).astype(int)),
                    color,
                    3,
                    cv2.LINE_AA,
                )
        for point in points[[0, 4, 8, 12, 16, 20]]:
            cv2.circle(panel, tuple(np.rint(point).astype(int)), 5, color, -1, cv2.LINE_AA)


def label_roots(panel: np.ndarray, roots: np.ndarray, intrinsics: np.ndarray) -> None:
    points = old.project_points(roots[:, :3, 3], intrinsics)
    for physical, (text, color) in enumerate((("ROBOT-L BLUE", BLUE), ("ROBOT-R RED", RED))):
        x = int(np.clip(round(float(points[physical, 0])), 5, fixed.WIDTH - 180))
        y = int(np.clip(round(float(points[physical, 1])), 125, fixed.HEIGHT - 10))
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SameSideStillError(f"refusing to overwrite: {args.output_dir}")
    if old.sha256(args.hawor) != EXPECTED_HAWOR_SHA:
        raise SameSideStillError("HaWoR lineage SHA mismatch")
    if old.sha256(taskfit.STATIC_AUDIT) != EXPECTED_STATIC_SHA:
        raise SameSideStillError("static closure lineage SHA mismatch")
    manifest = old.load_json(args.mask_manifest)
    if manifest.get("session") != fixed.SESSION:
        raise SameSideStillError("Mask session mismatch")
    raw_path = args.raw_root / f"{FRAME_ID:05d}" / "rgb.png"
    source_ref = manifest["frames"][FRAME_ID]["source_rgb"]
    if Path(source_ref["path"]).resolve() != raw_path.resolve() or old.sha256(raw_path) != source_ref["sha256"]:
        raise SameSideStillError("frame-0 raw lineage mismatch")

    hawor = old.arrays(args.hawor)
    assets = old.load_pinned_robot_assets(PROJECT)
    q_hand, q_hand_pose_reference, retarget_audit, hand_names = solve_same_side_hands(
        hawor, assets
    )
    target_rotations, root_audit = select_same_side_roots(
        hawor, assets, hand_names, q_hand_pose_reference
    )
    static = old.load_json(taskfit.STATIC_AUDIT)
    mounts = np.asarray(
        [row["T_tool_hand_root_geometric_candidate"] for row in static["candidate"]],
        dtype=np.float64,
    )
    camera, q_arm, actual_roots, arm_errors = solve_shared_camera_and_arms(
        hawor, assets, mounts, target_rotations
    )
    q_hand, visible_thumb_audit = refine_visible_left_thumb(
        hawor, assets, hand_names, q_hand_pose_reference, actual_roots
    )
    contracts = handfit.model_contract(old.official, wrist_adapter, assets)
    thumb_rotations = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    for physical, side in enumerate(SIDES):
        targets = handfit.human_features(
            wrist_adapter,
            np.asarray(hawor["joints_3d_camera"][physical, FRAME_ID], dtype=np.float64),
            side,
        )
        for finger_row in retarget_audit[physical]["fingers"]:
            finger = finger_row["finger"]
            actual = handfit.robot_finger_features(
                old.official,
                wrist_adapter,
                contracts[physical],
                q_hand[physical],
                finger,
                thumb_rotations[physical],
            )
            finger_row.update(handfit.feature_row(actual, targets[finger]))
        retarget_audit[physical]["thumb_web_refinement"] = visible_thumb_audit[physical]
        retarget_audit[physical]["tip_direction_error_deg_max"] = max(
            float(row["tip_direction_error_deg"])
            for row in retarget_audit[physical]["fingers"]
        )

    landmarks = np.asarray(
        [
            kai_visual_landmarks(
                model, hand_names[physical], q_hand[physical], physical_side=SIDES[physical]
            )
            for physical, model in enumerate((assets.left_hand, assets.right_hand))
        ]
    )
    actual_alignment = []
    for physical, side in enumerate(SIDES):
        points = (
            landmarks[physical] @ actual_roots[physical, :3, :3].T
            + actual_roots[physical, :3, 3]
        )
        actual_alignment.append(
            measure_alignment(
                points,
                np.asarray(hawor["joints_3d_camera"][physical, FRAME_ID]),
                np.asarray(hawor["joints_2d"][physical, FRAME_ID]),
                np.asarray(hawor["intrinsics"][FRAME_ID]),
                physical_side=side,
                human_side=side,
                other_wrist_camera=actual_roots[1 - physical, :3, 3],
                kai_root_rotation_camera=actual_roots[physical, :3, :3],
            )
        )

    values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(dict(zip(old.official.ARM_JOINT_NAMES[physical], q_arm[physical], strict=True)))
    arm_fk = old.official.forward_kinematics(assets.tianji, values)
    roots_base = np.asarray(
        [
            (arm_fk[f"{side}_tool"] @ mounts[physical])[:3, 3]
            for physical, side in enumerate(SIDES)
        ]
    )
    arm_branch_points = {
        side: {
            "link3_base_m": arm_fk[f"Link3_{suffix}"][:3, 3].tolist(),
            "link5_base_m": arm_fk[f"Link5_{suffix}"][:3, 3].tolist(),
        }
        for side, suffix in zip(SIDES, ("L", "R"), strict=True)
    }
    arm_same_side = all(
        (1.0 if physical == 0 else -1.0)
        * float(arm_branch_points[side][link][1])
        >= threshold
        for physical, side in enumerate(SIDES)
        for link, threshold in (("link3_base_m", 0.25), ("link5_base_m", 0.08))
    )
    forearms_in_front = all(
        float(arm_branch_points[side]["link5_base_m"][0]) >= 0.08 for side in SIDES
    )
    human_uv = np.asarray(hawor["joints_2d"][:, FRAME_ID], dtype=np.float64)
    wrist_y_gap = float(human_uv[1, 0, 1] - human_uv[0, 0, 1])
    gates = {
        "identity_mapping_physical_left_to_human_left": SAME_SIDE[0] == 0,
        "identity_mapping_physical_right_to_human_right": SAME_SIDE[1] == 1,
        "human_left_palm_points_image_right": float(
            root_audit[0]["metric"]["human_palmar_normal_camera"][0]
        ) > 0.45,
        "robot_left_palm_points_image_right": float(
            actual_alignment[0]["kai_palmar_normal_camera"][0]
        ) > 0.45,
        "human_right_palm_points_image_down": float(
            root_audit[1]["metric"]["human_palmar_normal_camera"][1]
        ) > 0.45,
        "robot_right_palm_points_image_down": float(
            actual_alignment[1]["kai_palmar_normal_camera"][1]
        ) > 0.45,
        "human_left_wrist_above_right_in_image": wrist_y_gap > 100.0,
        "robot_left_root_above_right_in_base": float(roots_base[0, 2] - roots_base[1, 2]) > 0.04,
        "robot_camera_faces_outward_like_human_ego_view": np.allclose(
            camera[:3, :3],
            outward_camera_rotation(fixed.load_fixed_lineage()[0][:3, :3]),
            atol=1e-10,
        ),
        "upper_arms_and_elbows_remain_on_physical_side": arm_same_side,
        "both_forearms_remain_in_front_of_torso": forearms_in_front,
        "all_five_finger_chains_solved_independently": all(
            len(row["fingers"]) == 5 for row in retarget_audit
        ),
        "thumb_six_dof_chain_solved_separately": all(
            row["thumb_policy"].startswith("SIX_DOF_THUMB_SOLVED_SEPARATELY")
            for row in retarget_audit
        ),
        "finger_tip_direction_error_le_10deg": max(
            float(row["tip_direction_error_deg_max"]) for row in retarget_audit
        ) <= 10.0,
        "actual_projected_thumb_direction_error_le_15deg": max(
            float(row["thumb_direction_angle_deg"]) for row in actual_alignment
        ) <= 15.0,
        "left_visible_thumb_true_tip_direction_error_le_5deg": float(
            visible_thumb_audit[0]["after"]["true_tip_direction_error_deg"]
        ) <= 5.0,
        "left_visible_thumb_true_tip_normalized_l2_le_0_08": float(
            visible_thumb_audit[0]["after"]["true_tip_normalized_l2"]
        ) <= 0.08,
        "thumb_index_gap_ratio_error_le_0_05": max(
            float(row["thumb_index_gap_over_middle_length"]["abs_error"])
            for row in actual_alignment
        ) <= 0.05,
        "right_thumb_preserved_from_v3_pose_reference": np.array_equal(
            q_hand[1, :6], q_hand_pose_reference[1, :6]
        ),
        "all_nonthumb_joints_preserved_from_v3_pose_reference": np.array_equal(
            q_hand[:, 6:], q_hand_pose_reference[:, 6:]
        ),
        "arm_frame0_position_error_le_3mm": max(row["position_mm"] for row in arm_errors) <= 3.0,
        "arm_frame0_rotation_error_le_3deg": max(row["rotation_deg"] for row in arm_errors) <= 3.0,
        "no_video_encoded": True,
        "current_robot_authority_remains_false": True,
    }
    if not all(gates.values()):
        raise SameSideStillError(f"frame-0 user pose gate failed: {gates}")

    raw_native = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if raw_native is None:
        raise SameSideStillError("frame-0 RGB decode failed")
    source_h, source_w = raw_native.shape[:2]
    raw = cv2.resize(raw_native, (fixed.WIDTH, fixed.HEIGHT), interpolation=cv2.INTER_AREA)
    intrinsics = np.asarray(hawor["intrinsics"][FRAME_ID], dtype=np.float64).copy()
    intrinsics[0] *= fixed.WIDTH / source_w
    intrinsics[1] *= fixed.HEIGHT / source_h
    human_uv_scaled = human_uv.copy()
    human_uv_scaled[..., 0] *= fixed.WIDTH / source_w
    human_uv_scaled[..., 1] *= fixed.HEIGHT / source_h

    source_panel = raw.copy()
    draw_human_skeletons(source_panel, human_uv_scaled)
    source_panel = fixed.title(
        source_panel,
        "原始第0帧｜同侧人体标注",
        "蓝=human-left（抬起、掌心朝右）｜红=human-right（靠桌、掌心朝下）",
        "这是解剖左右，不按面对观察者后的屏幕左右重新命名",
    )

    raster = old.load_module(old.RASTER_SOURCE, "poker_same_side_frame0_raster")
    flange_local = fixed.naturalv2_local_triangles()
    cache: dict[Path, object] = {}
    ego_color, ego_label, ego_roots = render_robot(
        raster,
        assets,
        q_arm,
        q_hand,
        mounts,
        camera,
        intrinsics,
        cache,
        flange_local,
        complete_robot=False,
    )
    overlay = raw.copy()
    visible = ego_label >= 0
    # The robot layer is deliberately opaque for this pose/color review.  A
    # translucent layer made the raw white wrist tracker look like a flange.
    overlay[visible] = ego_color[visible]
    label_roots(overlay, ego_roots, intrinsics)
    overlay = fixed.title(
        overlay,
        "机器叠加｜同侧映射｜向外视角｜第0帧",
        "蓝=Robot-left←human-left｜红=Robot-right←human-right",
        "用户确认配色：暖白双手｜暖象牙白机械臂/法兰｜蓝/红文字仅标左右",
    )

    rear = same_direction_global_camera(camera)
    rear_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    global_color, global_label, global_roots = render_robot(
        raster,
        assets,
        q_arm,
        q_hand,
        mounts,
        rear,
        rear_k,
        cache,
        flange_local,
        complete_robot=True,
    )
    global_panel = np.full_like(global_color, 242)
    global_panel[global_label >= 0] = global_color[global_label >= 0]
    label_roots(global_panel, global_roots, rear_k)
    global_panel = fixed.title(
        global_panel,
        "整机向外视图｜相机在机器人身后",
        "与人的第一视角同向｜蓝=物理左手｜红=物理右手",
        "左右肘不换边/不过背｜左臂抬起｜右臂靠桌｜仅单帧确认",
    )

    args.output_dir.mkdir(parents=True)
    image_path = args.output_dir / "POKER_042_第0帧_同侧左右手_同向整机姿态确认.png"
    if not cv2.imwrite(str(image_path), np.hstack((source_panel, overlay, global_panel))):
        raise SameSideStillError("frame-0 review image write failed")
    states_path = args.output_dir / "FRAME0_STATES.npz"
    np.savez_compressed(
        states_path,
        source_frame=np.asarray(FRAME_ID),
        physical_to_human=np.asarray(SAME_SIDE),
        q_arm=q_arm,
        q_hand=q_hand,
        T_camera_base=camera,
        T_tool_hand_root=mounts,
        T_actual_hand_root_camera=actual_roots,
        robot_hand_roots_base=roots_base,
    )
    result = {
        "schema_version": "poker-same-side-outward-frame0-pose-result-v3",
        "created_at": now(),
        "status": "STILL_READY_FOR_USER_COLOR_REVIEW",
        "grade": "DEVELOPMENT_ONLY_PENDING_USER_COLOR_REVIEW",
        "session": fixed.SESSION,
        "source_frame": FRAME_ID,
        "physical_to_human": {"left": "left", "right": "right"},
        "pose_lock": "V3_ROOT_ARM_CAMERA_RIGHT_THUMB_AND_ALL_NONTHUMB_JOINTS_LOCKED",
        "global_view": "ROBOT_REAR_EGOCENTRIC_OUTWARD_CAMERA_RECENTERED_FOR_WHOLE_ROBOT",
        "colors_bgr": {
            "left_annotation_blue": list(BLUE),
            "right_annotation_red": list(RED),
            "robot004_arm_warm_ivory": list(ROBOT_IVORY),
            "user_requested_kaihand_warm_white": list(HAND_WHITE),
            "robot004_naturalv2_warm_ivory": list(FLANGE_IVORY),
        },
        "color_reference": {
            "user_local_video": {
                "path": REFERENCE_VIDEO_LOCAL_PATH,
                "bytes": REFERENCE_VIDEO_BYTES,
                "sha256": REFERENCE_VIDEO_SHA256,
            },
            "remote_material_source": old.artifact(
                PROJECT
                / "archive/legacy_runs/robot/"
                "robot_004_full460_exactbase_independent_finger_retarget_cpu_20260901_v1/"
                "render_review.py"
            ),
            "material_base_linear_rgba": REFERENCE_004_MATERIAL_LINEAR,
            "material_policy": "ROBOT004_ARM_FLANGE_PALETTE_PLUS_USER_REQUESTED_WHITE_HAND_WITH_CPU_TRIANGLE_SHADING",
        },
        "gates": gates,
        "retarget_audit": retarget_audit,
        "root_selection_audit": root_audit,
        "actual_alignment": actual_alignment,
        "arm_fit_errors": arm_errors,
        "robot_hand_roots_base_m": roots_base.tolist(),
        "arm_natural_branch_points": arm_branch_points,
        "robot_left_minus_right_height_m": float(roots_base[0, 2] - roots_base[1, 2]),
        "human_right_minus_left_wrist_v_px": wrist_y_gap,
        "development_only": True,
        "current_robot_authority": False,
        "downstream_authorized": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "video_encoded": False,
        "lineage": {
            "hawor": old.artifact(args.hawor),
            "mask_manifest": old.artifact(args.mask_manifest),
            "source_rgb": old.artifact(raw_path),
            "static_closure": old.artifact(taskfit.STATIC_AUDIT),
            "withdrawn_cross_side_candidate": old.artifact(
                PROJECT
                / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/"
                "robot_mount_proxy_baseline_v1/task_action_canary_v1/"
                "play_cards_0902_042_physical_left_hawor_web_24frame_video_v1/"
                "USER_VISUAL_DECISION.json"
            ),
        },
        "outputs": {
            "still": old.artifact(image_path),
            "states": old.artifact(states_path),
        },
        "next_gate": "USER_CONFIRM_OR_REJECT_SINGLE_FRAME_ROBOT_POSE_BEFORE_GENERALIZATION",
        "claim_limit": "One frame for left/right, facing-direction, arm-height and palm-direction confirmation only. No video, contact, collision, task-generalization, Robot authority, training, or deployment claim.",
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(result_path), "still": result["outputs"]["still"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
