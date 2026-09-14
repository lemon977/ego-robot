#!/usr/bin/env python3
"""Render a short same-side, world-first Robot temporal review video.

Frame 0 is locked to a user-accepted still.  Later frames retarget HaWoR in
human world coordinates with one fixed Robot base placement.  This is a
development-only temporal diagnostic and cannot publish Robot authority.
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

from tools import render_poker_same_side_outward_frame0 as shared  # noqa: E402
from tools import render_poker_static_closure_task_translation_successor as taskfit  # noqa: E402
from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import render_poker_v2c03_p3_naturalv2_successor as fixed  # noqa: E402
from tools import run_newtask_robot_shared_v4_hand as handfit  # noqa: E402


EXPECTED = {
    "poker": {
        "session": "play_cards_0902_042",
        "hawor_sha256": "1822ed0c954ac2a294cd7860d23e51c125335fcdc4f1b0c07bed1f648ce066a8",
        "decision": "ACCEPT_POKER_POSE_AND_COLOR_AUTHORIZE_SHORT_WORLD_FIRST_VIDEO",
        "label": "POKER_042",
    },
    "chips": {
        "session": "get_potato_chips_0902_034",
        "hawor_sha256": "b569609aa1fe657c0ed98ac8554ac41e73eee629ed78df2fbea5e914a0560095",
        "decision": "ACCEPT_CHIPS_POSE_COLOR_AND_WORLD_CHAIN_AUTHORIZE_SHORT_VIDEO",
        "label": "CHIPS_034",
    },
}
HAND_STEP_LIMIT = 0.08
HAND_SOLVER_STEP_LIMIT = 0.06
ARM_STEP_LIMIT = 0.12
# Let the IK use the full velocity envelope that the temporal review audits.
# Acceleration remains independently bounded by ACCELERATION_LIMIT below.
ARM_SOLVER_STEP_LIMIT = ARM_STEP_LIMIT
ACCELERATION_LIMIT = 0.06
FRAME0_CAMERA_BASE_ATOL = 1e-12


class TemporalReviewError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def bounded_limits(
    lower: np.ndarray,
    upper: np.ndarray,
    previous: np.ndarray,
    previous_previous: np.ndarray | None,
    step_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    bounded_lower = np.maximum(lower, previous - step_limit)
    bounded_upper = np.minimum(upper, previous + step_limit)
    if previous_previous is not None:
        predicted = 2.0 * previous - previous_previous
        bounded_lower = np.maximum(
            bounded_lower, predicted - ACCELERATION_LIMIT
        )
        bounded_upper = np.minimum(
            bounded_upper, predicted + ACCELERATION_LIMIT
        )
    excess = bounded_lower - bounded_upper
    if np.any(excess > 1e-10):
        raise TemporalReviewError(
            f"empty previous-accepted bounds: max_excess={float(np.max(excess))}"
        )
    # A zero-width intersection is a valid unique state (typically when an
    # accepted joint starts on a URDF limit and the second-difference bound is
    # active).  The caller holds that coordinate fixed instead of widening it.
    midpoint = 0.5 * (bounded_lower + bounded_upper)
    collapsed = excess >= -1e-12
    bounded_lower[collapsed] = midpoint[collapsed]
    bounded_upper[collapsed] = midpoint[collapsed]
    return bounded_lower, bounded_upper


def solve_hand_sequence(
    hawor: dict[str, np.ndarray],
    assets: Any,
    accepted_q0: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]], tuple[tuple[str, ...], ...]]:
    contracts = handfit.model_contract(old.official, shared.wrist_adapter, assets)
    names = tuple(tuple(contract["names"]) for contract in contracts)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    weights = dict(handfit.FIT_GRID[0])
    q_rows = np.empty((frame_count, 2, 22), dtype=np.float64)
    q_rows[0] = accepted_q0
    audits: list[dict[str, Any]] = []

    for frame in range(frame_count):
        for physical, side in enumerate(shared.SIDES):
            target = handfit.human_features(
                shared.wrist_adapter,
                np.asarray(hawor["joints_3d_world"][physical, frame], dtype=np.float64),
                side,
            )
            if frame == 0:
                candidate = q_rows[0, physical].copy()
                policy = "USER_ACCEPTED_FRAME0_LOCK"
            else:
                previous = q_rows[frame - 1, physical]
                previous_previous = (
                    None if frame == 1 else q_rows[frame - 2, physical]
                )
                candidate = previous.copy()
                for finger in handfit.FINGERS:
                    group = handfit.GROUPS[finger]
                    lower, upper = bounded_limits(
                        contracts[physical]["lower"][group],
                        contracts[physical]["upper"][group],
                        previous[group],
                        None
                        if previous_previous is None
                        else previous_previous[group],
                        HAND_SOLVER_STEP_LIMIT,
                    )

                    def residual(local: np.ndarray) -> np.ndarray:
                        whole = candidate.copy()
                        whole[group] = local
                        actual = handfit.robot_finger_features(
                            old.official,
                            shared.wrist_adapter,
                            contracts[physical],
                            whole,
                            finger,
                            rotations[physical],
                        )
                        return np.concatenate(
                            (
                                weights["mcp"] * (actual["mcp"] - target[finger]["mcp"]),
                                weights["bone"]
                                * (actual["bones"] - target[finger]["bones"]).ravel(),
                                weights["tip"] * (actual["tip"] - target[finger]["tip"]),
                                weights["prior"] * (local - previous[group]),
                            )
                        )

                    local = np.clip(previous[group], lower, upper)
                    free = upper - lower > 1e-12
                    if np.any(free):
                        def residual_free(free_values: np.ndarray) -> np.ndarray:
                            whole_local = local.copy()
                            whole_local[free] = free_values
                            return residual(whole_local)

                        solved = least_squares(
                            residual_free,
                            local[free],
                            bounds=(lower[free], upper[free]),
                            max_nfev=120,
                            ftol=1e-9,
                            xtol=1e-9,
                            gtol=1e-9,
                        )
                        local[free] = solved.x
                    local[~free] = lower[~free]
                    candidate[group] = local
                q_rows[frame, physical] = candidate
                policy = "PREVIOUS_ACCEPTED_ONLY_BOUNDED_FIVE_CHAIN_SOLVE"

            finger_rows = []
            for finger in handfit.FINGERS:
                actual = handfit.robot_finger_features(
                    old.official,
                    shared.wrist_adapter,
                    contracts[physical],
                    candidate,
                    finger,
                    rotations[physical],
                )
                finger_rows.append(
                    {"finger": finger, **handfit.feature_row(actual, target[finger])}
                )
            audits.append(
                {
                    "frame": frame,
                    "physical_side": side,
                    "human_side": side,
                    "policy": policy,
                    "fingers": finger_rows,
                    "tip_direction_error_deg_max": max(
                        float(row["tip_direction_error_deg"]) for row in finger_rows
                    ),
                    "bone_error_deg_max": max(
                        float(row["bone_error_deg_max"]) for row in finger_rows
                    ),
                }
            )
    return q_rows, audits, names


def solve_arm_sequence(
    hawor: dict[str, np.ndarray],
    assets: Any,
    mounts: np.ndarray,
    world_base: np.ndarray,
    accepted_q0: np.ndarray,
    frame_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    np.ndarray,
]:
    lower, upper = taskfit.arm_limits(assets)
    q_rows = np.empty((frame_count, 2, 7), dtype=np.float64)
    q_rows[0] = accepted_q0
    target_world = np.tile(np.eye(4), (frame_count, 2, 1, 1))
    actual_world = np.tile(np.eye(4), (frame_count, 2, 1, 1))
    roots_base = np.tile(np.eye(4), (frame_count, 2, 1, 1))
    audits: list[dict[str, Any]] = []
    accepted_roots_world = np.asarray(
        [
            world_base
            @ old.official._tool_fk(assets, physical, accepted_q0[physical])
            @ mounts[physical]
            for physical in range(2)
        ]
    )
    palm_basis_world0 = np.asarray(
        [
            shared.wrist_adapter.final_v3_mano_palm_basis(
                np.asarray(hawor["joints_3d_world"][physical, 0]),
                handedness=side,
            )
            for physical, side in enumerate(shared.SIDES)
        ]
    )

    for frame in range(frame_count):
        for physical, side in enumerate(shared.SIDES):
            palm_basis_world = shared.wrist_adapter.final_v3_mano_palm_basis(
                np.asarray(hawor["joints_3d_world"][physical, frame]),
                handedness=side,
            )
            target_world[frame, physical, :3, :3] = (
                palm_basis_world
                @ palm_basis_world0[physical].T
                @ accepted_roots_world[physical, :3, :3]
            )
            target_world[frame, physical, :3, 3] = np.asarray(
                hawor["joints_3d_world"][physical, frame, 0], dtype=np.float64
            )
            if frame == 0:
                q = q_rows[0, physical].copy()
                evaluations = 0
                policy = "USER_ACCEPTED_FRAME0_LOCK"
            else:
                previous = q_rows[frame - 1, physical]
                previous_previous = (
                    None if frame == 1 else q_rows[frame - 2, physical]
                )
                solve_lower, solve_upper = bounded_limits(
                    lower[physical],
                    upper[physical],
                    previous,
                    previous_previous,
                    ARM_SOLVER_STEP_LIMIT,
                )
                target_tool = (
                    np.linalg.inv(world_base)
                    @ target_world[frame, physical]
                    @ np.linalg.inv(mounts[physical])
                )
                position, n_position = old.official._solve_one_arm_position_only(
                    assets,
                    side=physical,
                    base=np.eye(4),
                    target_tool=target_tool,
                    initial_q=np.clip(previous, solve_lower, solve_upper),
                    lower=solve_lower,
                    upper=solve_upper,
                )
                q, n_pose = old.official._solve_one_arm(
                    assets,
                    side=physical,
                    base=np.eye(4),
                    target_tool=target_tool,
                    initial_q=position,
                    lower=solve_lower,
                    upper=solve_upper,
                )
                q_rows[frame, physical] = q
                evaluations = int(n_position + n_pose)
                policy = "PREVIOUS_ACCEPTED_ONLY_BOUNDED_WORLD_IK"

            roots_base[frame, physical] = (
                old.official._tool_fk(assets, physical, q) @ mounts[physical]
            )
            actual_world[frame, physical] = (
                world_base @ roots_base[frame, physical]
            )
            position_mm = float(
                np.linalg.norm(
                    actual_world[frame, physical, :3, 3]
                    - target_world[frame, physical, :3, 3]
                )
                * 1000.0
            )
            rotation_deg = float(
                np.degrees(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            target_world[frame, physical, :3, :3].T
                            @ actual_world[frame, physical, :3, :3]
                        ).as_rotvec()
                    )
                )
            )
            audits.append(
                {
                    "frame": frame,
                    "physical_side": side,
                    "human_side": side,
                    "policy": policy,
                    "position_mm": position_mm,
                    "rotation_deg": rotation_deg,
                    "solver_evaluations": evaluations,
                }
            )
    return q_rows, target_world, actual_world, audits, roots_base


def motion_metrics(values: np.ndarray) -> dict[str, float]:
    velocity = np.diff(values, axis=0)
    acceleration = np.diff(values, n=2, axis=0)
    return {
        "velocity_max_rad_per_frame": float(np.max(np.abs(velocity)))
        if velocity.size
        else 0.0,
        "acceleration_max_rad_per_frame2": float(np.max(np.abs(acceleration)))
        if acceleration.size
        else 0.0,
    }


def branch_metrics(
    assets: Any, q_arm: np.ndarray
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    passed = True
    for frame in range(len(q_arm)):
        values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
        for physical in range(2):
            values.update(
                dict(
                    zip(
                        old.official.ARM_JOINT_NAMES[physical],
                        q_arm[frame, physical],
                        strict=True,
                    )
                )
            )
        fk = old.official.forward_kinematics(assets.tianji, values)
        for physical, (side, suffix) in enumerate(
            zip(shared.SIDES, ("L", "R"), strict=True)
        ):
            sign = 1.0 if physical == 0 else -1.0
            link3 = fk[f"Link3_{suffix}"][:3, 3]
            link5 = fk[f"Link5_{suffix}"][:3, 3]
            row_pass = bool(
                sign * float(link3[1]) >= 0.20
                and sign * float(link5[1]) >= 0.04
                and float(link5[0]) >= 0.04
            )
            passed = passed and row_pass
            rows.append(
                {
                    "frame": frame,
                    "physical_side": side,
                    "link3_base_m": link3.tolist(),
                    "link5_base_m": link5.tolist(),
                    "pass": row_pass,
                }
            )
    return rows, passed


def label_roots(panel: np.ndarray, roots: np.ndarray, intrinsics: np.ndarray) -> None:
    points = old.project_points(roots[:, :3, 3], intrinsics)
    rows = (("ROBOT-L BLUE", shared.BLUE, -16), ("ROBOT-R RED", shared.RED, 20))
    for physical, (label, color, offset) in enumerate(rows):
        x = int(np.clip(round(float(points[physical, 0])), 5, fixed.WIDTH - 185))
        y = int(
            np.clip(round(float(points[physical, 1])) + offset, 125, fixed.HEIGHT - 10)
        )
        cv2.putText(
            panel,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )


def render_video(
    args: argparse.Namespace,
    config: dict[str, str],
    hawor: dict[str, np.ndarray],
    manifest: dict[str, Any],
    assets: Any,
    mounts: np.ndarray,
    world_base: np.ndarray,
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    actual_world: np.ndarray,
    source_frames: np.ndarray,
) -> Path:
    raster = old.load_module(
        old.RASTER_SOURCE, f"{args.task}_same_side_world_temporal_raster"
    )
    flange_local = fixed.naturalv2_local_triangles()
    cache: dict[Path, object] = {}
    c2w0 = np.asarray(hawor["c2w"][0], dtype=np.float64)
    fixed_global_camera = shared.same_direction_global_camera(
        np.linalg.inv(c2w0) @ world_base
    )
    global_k = np.asarray(
        ((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0))
    )
    video_path = args.output_dir / f"{config['label']}_WORLD_FIRST_SAME_SIDE_48帧白手慢放复核.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.review_fps),
        (fixed.WIDTH * 3, fixed.HEIGHT),
    )
    if not writer.isOpened():
        raise TemporalReviewError("failed to open MP4 writer")
    try:
        for slot, source_frame in enumerate(source_frames):
            raw_path = args.raw_root / f"{int(source_frame):05d}" / "rgb.png"
            source_ref = manifest["frames"][slot]["source_rgb"]
            if (
                Path(source_ref["path"]).resolve() != raw_path.resolve()
                or old.sha256(raw_path) != source_ref["sha256"]
            ):
                raise TemporalReviewError(f"raw lineage mismatch at frame {slot}")
            native = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if native is None:
                raise TemporalReviewError(f"failed to decode {raw_path}")
            source_h, source_w = native.shape[:2]
            raw = cv2.resize(
                native, (fixed.WIDTH, fixed.HEIGHT), interpolation=cv2.INTER_AREA
            )
            intrinsics = np.asarray(hawor["intrinsics"][slot], dtype=np.float64).copy()
            intrinsics[0] *= fixed.WIDTH / source_w
            intrinsics[1] *= fixed.HEIGHT / source_h
            uv = np.asarray(hawor["joints_2d"][:, slot], dtype=np.float64).copy()
            uv[..., 0] *= fixed.WIDTH / source_w
            uv[..., 1] *= fixed.HEIGHT / source_h

            source_panel = raw.copy()
            shared.draw_human_skeletons(source_panel, uv)
            source_panel = fixed.title(
                source_panel,
                f"{config['label']}原始帧{int(source_frame)}｜同侧人手",
                "蓝=human-left｜红=human-right",
                "源帧30FPS｜本视频12FPS慢放｜连续前48帧",
            )

            camera_base = np.linalg.inv(hawor["c2w"][slot]) @ world_base
            ego_color, ego_label, ego_roots = shared.render_robot(
                raster,
                assets,
                q_arm[slot],
                q_hand[slot],
                mounts,
                camera_base,
                intrinsics,
                cache,
                flange_local,
                complete_robot=False,
            )
            overlay = raw.copy()
            overlay[ego_label >= 0] = ego_color[ego_label >= 0]
            label_roots(overlay, ego_roots, intrinsics)
            overlay = fixed.title(
                overlay,
                f"{config['label']} Robot｜world-first｜帧{int(source_frame)}",
                "T_world_base固定｜T_camera_base(t)=inv(c2w(t))@T_world_base",
                "暖白双手｜同侧映射｜蓝/红文字仅标左右",
            )

            global_color, global_label, global_roots = shared.render_robot(
                raster,
                assets,
                q_arm[slot],
                q_hand[slot],
                mounts,
                fixed_global_camera,
                global_k,
                cache,
                flange_local,
                complete_robot=True,
            )
            global_panel = np.full_like(global_color, 242)
            global_panel[global_label >= 0] = global_color[global_label >= 0]
            label_roots(global_panel, global_roots, global_k)
            global_panel = fixed.title(
                global_panel,
                f"固定整机视图｜源帧{int(source_frame)}",
                "观察相机与Robot base都固定｜只有关节运动",
                "检查左右肩肘不换边/不过背｜待用户视频确认",
            )
            writer.write(np.hstack((source_panel, overlay, global_panel)))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video_path))
    try:
        decoded = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if decoded != len(source_frames) or width != fixed.WIDTH * 3 or height != fixed.HEIGHT:
        raise TemporalReviewError(
            f"encoded video closure failed: frames={decoded} size={width}x{height}"
        )
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--accepted-frame0-states", type=Path, required=True)
    parser.add_argument("--accepted-frame0-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=48)
    parser.add_argument("--review-fps", type=float, default=12.0)
    args = parser.parse_args()
    config = EXPECTED[args.task]
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise TemporalReviewError(f"refusing to overwrite: {args.output_dir}")
    if args.frame_count != 48:
        raise TemporalReviewError("this user-authorized canary is fixed to 48 frames")
    if old.sha256(args.hawor) != config["hawor_sha256"]:
        raise TemporalReviewError("HaWoR lineage SHA mismatch")
    decision = old.load_json(args.accepted_frame0_decision)
    if (
        decision.get("status") != "SINGLE_FRAME_ACCEPTED_SHORT_VIDEO_AUTHORIZED"
        or decision.get("decision") != config["decision"]
        or decision["candidate"]["states"]["sha256"]
        != old.sha256(args.accepted_frame0_states)
    ):
        raise TemporalReviewError("accepted frame-0 decision closure failed")
    manifest = old.load_json(args.mask_manifest)
    if manifest.get("session") != config["session"]:
        raise TemporalReviewError("Mask session mismatch")

    hawor = old.arrays(args.hawor)
    frame_count = args.frame_count
    if not bool(np.all(np.asarray(hawor["observed"])[:, :frame_count])):
        raise TemporalReviewError("both HaWoR hands must be observed in all 48 frames")
    source_frames = np.asarray(hawor["original_frame_indices"][:frame_count], dtype=np.int32)
    if not np.array_equal(source_frames, np.arange(frame_count, dtype=np.int32)):
        raise TemporalReviewError("review window must be contiguous source frames 0..47")
    with np.load(args.accepted_frame0_states, allow_pickle=False) as accepted:
        if not np.array_equal(accepted["physical_to_human"], np.asarray((0, 1))):
            raise TemporalReviewError("accepted frame0 is not same-side")
        accepted_q_arm = np.asarray(accepted["q_arm"], dtype=np.float64)
        accepted_q_hand = np.asarray(accepted["q_hand"], dtype=np.float64)
        mounts = np.asarray(accepted["T_tool_hand_root"], dtype=np.float64)
        if "T_world_base" in accepted.files:
            world_base = np.asarray(accepted["T_world_base"], dtype=np.float64)
        else:
            world_base = (
                np.asarray(hawor["c2w"][0], dtype=np.float64)
                @ np.asarray(accepted["T_camera_base"], dtype=np.float64)
            )
        accepted_camera_base = np.asarray(accepted["T_camera_base"], dtype=np.float64)

    camera_points = np.asarray(hawor["joints_3d_camera"][:, :frame_count])
    c2w = np.asarray(hawor["c2w"][:frame_count])
    reconstructed_world = np.einsum(
        "fij,pfkj->pfki", c2w[:, :3, :3], camera_points
    ) + c2w[None, :, None, :3, 3]
    world_points = np.asarray(hawor["joints_3d_world"][:, :frame_count])
    world_closure_max_m = float(
        np.max(np.linalg.norm(reconstructed_world - world_points, axis=3))
    )
    # joints_camera/world are stored as float32 while c2w is float64.  The
    # resulting 20--30 nm roundoff is expected; retain a sub-micron gate.
    if world_closure_max_m > 1e-7:
        raise TemporalReviewError("HaWoR camera/world closure failed")

    assets = old.load_pinned_robot_assets(PROJECT)
    q_hand, hand_audit, hand_names = solve_hand_sequence(
        hawor, assets, accepted_q_hand, frame_count
    )
    q_arm, target_world, actual_world, arm_audit, roots_base = solve_arm_sequence(
        hawor,
        assets,
        mounts,
        world_base,
        accepted_q_arm,
        frame_count,
    )
    arm_motion = motion_metrics(q_arm)
    hand_motion = motion_metrics(q_hand)
    branches, branch_pass = branch_metrics(assets, q_arm)
    camera_base = np.asarray([np.linalg.inv(row) @ world_base for row in c2w])
    frame0_camera_base_max_abs = float(
        np.max(np.abs(camera_base[0] - accepted_camera_base))
    )
    camera_closure_max = float(
        np.max(np.abs(np.einsum("fij,fjk->fik", c2w, camera_base) - world_base))
    )
    camera_translation = np.linalg.norm(c2w[:, :3, 3] - c2w[0, :3, 3], axis=1)
    camera_rotation = np.degrees(
        Rotation.from_matrix(
            np.einsum("ij,fjk->fik", c2w[0, :3, :3].T, c2w[:, :3, :3])
        ).magnitude()
    )
    arm_position_max = max(float(row["position_mm"]) for row in arm_audit)
    arm_rotation_max = max(float(row["rotation_deg"]) for row in arm_audit)
    gates = {
        "user_accepted_frame0_authorized_short_video": True,
        "same_side_identity_physical_left_to_human_left": True,
        "same_side_identity_physical_right_to_human_right": True,
        "frame0_q_arm_bit_exact": np.array_equal(q_arm[0], accepted_q_arm),
        "frame0_q_hand_bit_exact": np.array_equal(q_hand[0], accepted_q_hand),
        "frame0_camera_base_abs_error_le_1e_12": (
            frame0_camera_base_max_abs <= FRAME0_CAMERA_BASE_ATOL
        ),
        "hawor_camera_world_closure_le_1e_7m": world_closure_max_m <= 1e-7,
        "fixed_world_base_camera_closure_le_1e_10": camera_closure_max <= 1e-10,
        "window_contains_nontrivial_camera_motion": bool(
            float(np.max(camera_translation)) >= 0.005
            or float(np.max(camera_rotation)) >= 0.5
        ),
        "arm_world_position_error_le_10mm": arm_position_max <= 10.0,
        "arm_world_rotation_error_le_5deg": arm_rotation_max <= 5.0,
        "arm_velocity_le_0_12rad_per_frame": arm_motion[
            "velocity_max_rad_per_frame"
        ]
        <= ARM_STEP_LIMIT + 1e-9,
        "arm_acceleration_le_0_06rad_per_frame2": arm_motion[
            "acceleration_max_rad_per_frame2"
        ]
        <= ACCELERATION_LIMIT + 1e-9,
        "hand_velocity_le_0_08rad_per_frame": hand_motion[
            "velocity_max_rad_per_frame"
        ]
        <= HAND_STEP_LIMIT + 1e-9,
        "hand_acceleration_le_0_06rad_per_frame2": hand_motion[
            "acceleration_max_rad_per_frame2"
        ]
        <= ACCELERATION_LIMIT + 1e-9,
        "hand_tip_direction_error_le_15deg": max(
            float(row["tip_direction_error_deg_max"]) for row in hand_audit
        )
        <= 15.0,
        "hand_bone_error_le_60deg": max(
            float(row["bone_error_deg_max"]) for row in hand_audit
        )
        <= 60.0,
        "upper_arms_elbows_forearms_keep_accepted_branches": branch_pass,
        "all_states_finite": bool(
            np.isfinite(q_arm).all()
            and np.isfinite(q_hand).all()
            and np.isfinite(target_world).all()
            and np.isfinite(actual_world).all()
        ),
        "current_robot_authority_remains_false": True,
    }

    args.output_dir.mkdir(parents=True)
    trajectory_path = args.output_dir / "WORLD_FIRST_SAME_SIDE_48FRAME_STATES.npz"
    np.savez_compressed(
        trajectory_path,
        local_frames=np.arange(frame_count, dtype=np.int32),
        source_frames=source_frames,
        physical_to_human=np.asarray((0, 1), dtype=np.int32),
        q_arm=q_arm,
        q_hand=q_hand,
        c2w=c2w,
        T_world_base=world_base,
        T_base_world=np.linalg.inv(world_base),
        T_camera_base=camera_base,
        T_tool_hand_root=mounts,
        T_target_hand_root_world=target_world,
        T_actual_hand_root_world=actual_world,
        T_actual_hand_root_base=roots_base,
    )
    video_path = render_video(
        args,
        config,
        hawor,
        manifest,
        assets,
        mounts,
        world_base,
        q_arm,
        q_hand,
        actual_world,
        source_frames,
    )
    status = (
        "VIDEO_READY_FOR_USER_TEMPORAL_REVIEW"
        if all(gates.values())
        else "VIDEO_HOLD_NUMERIC_TEMPORAL_REVIEW"
    )
    result = {
        "schema_version": "same-side-world-first-temporal-review-result-v1",
        "created_at": now(),
        "status": status,
        "grade": "DEVELOPMENT_ONLY_PENDING_USER_VIDEO_REVIEW",
        "task": args.task,
        "session": config["session"],
        "source_frames": source_frames.tolist(),
        "source_fps": float(np.asarray(hawor["fps"]).item()),
        "review_fps": float(args.review_fps),
        "physical_to_human": {"left": "left", "right": "right"},
        "coordinate_chain": {
            "target": "T_world_hand(t)=c2w(t)@T_camera_hand_HaWoR(t)",
            "ik": "T_world_base@FK_base_tool(q(t))@T_tool_hand~=T_world_hand(t)",
            "render": "T_camera_base(t)=inv(c2w(t))@T_world_base",
            "world_base_policy": "ONE_FIXED_TASK_LEVEL_PLACEMENT_FROM_USER_ACCEPTED_FRAME0",
            "matrix_name_convention": "T_A_B maps coordinates from frame B to frame A",
        },
        "metrics": {
            "hawor_camera_world_closure_max_m": world_closure_max_m,
            "camera_world_base_closure_max_abs": camera_closure_max,
            "frame0_camera_base_max_abs_error": frame0_camera_base_max_abs,
            "camera_translation_from_frame0_max_m": float(
                np.max(camera_translation)
            ),
            "camera_rotation_from_frame0_max_deg": float(np.max(camera_rotation)),
            "robot_world_base_motion_max": 0.0,
            "arm_world_position_error_max_mm": arm_position_max,
            "arm_world_rotation_error_max_deg": arm_rotation_max,
            "arm_temporal": arm_motion,
            "hand_temporal": hand_motion,
            "solver_limits": {
                "arm_step_rad_per_frame": ARM_STEP_LIMIT,
                "arm_solver_step_rad_per_frame": ARM_SOLVER_STEP_LIMIT,
                "hand_solver_step_rad_per_frame": HAND_SOLVER_STEP_LIMIT,
                "hand_review_velocity_limit_rad_per_frame": HAND_STEP_LIMIT,
                "second_difference_rad_per_frame2": ACCELERATION_LIMIT,
            },
            "hand_tip_direction_error_deg_max": max(
                float(row["tip_direction_error_deg_max"]) for row in hand_audit
            ),
            "hand_bone_error_deg_max": max(
                float(row["bone_error_deg_max"]) for row in hand_audit
            ),
        },
        "gates": gates,
        "arm_audit": arm_audit,
        "hand_audit": hand_audit,
        "branch_audit": branches,
        "colors_bgr": {
            "left_annotation_blue": list(shared.BLUE),
            "right_annotation_red": list(shared.RED),
            "arm_warm_ivory": list(shared.ROBOT_IVORY),
            "kaihand_warm_white": list(shared.HAND_WHITE),
            "naturalv2_warm_ivory": list(shared.FLANGE_IVORY),
        },
        "development_only": True,
        "current_robot_authority": False,
        "downstream_authorized": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "fullsession_authorized": False,
        "lineage": {
            "hawor": old.artifact(args.hawor),
            "mask_manifest": old.artifact(args.mask_manifest),
            "accepted_frame0_states": old.artifact(args.accepted_frame0_states),
            "accepted_frame0_decision": old.artifact(args.accepted_frame0_decision),
            "renderer": old.artifact(Path(__file__)),
        },
        "outputs": {
            "video": old.artifact(video_path),
            "trajectory": old.artifact(trajectory_path),
        },
        "next_gate": "USER_CONFIRM_OR_REJECT_SHORT_TEMPORAL_VIDEO_BEFORE_FULL_SESSION",
        "claim_limit": "Continuous source frames 0..47 only, slowed from 30 FPS to 12 FPS for visual review. No full-session, contact, collision, action-sidecar, training, deployment, or current Robot authority claim.",
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "status": status,
                "failed_gates": [name for name, passed in gates.items() if not passed],
                "video": result["outputs"]["video"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
