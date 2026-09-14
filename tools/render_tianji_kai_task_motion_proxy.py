#!/usr/bin/env python3
"""Render one bounded Tianji + dual-KaiHand task-motion proxy canary.

This is a development-only, standalone component review.  It consumes no
Clean or Object6D, draws no adapter geometry, and makes no contact, collision,
calibration, or deployment claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pybullet as bullet


PROJECT = Path(__file__).resolve().parents[1]
RETARGET_PROJECT = Path("/mnt/workspace/code/retargeting_human")
for root in (str(PROJECT), str(RETARGET_PROJECT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from pipeline import robot_scene_state_cpu as official  # noqa: E402
from pipeline import robot_wrist_kai_adapter as adapter  # noqa: E402
from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets  # noqa: E402
from tools.compare_kaihand_retargeters import (  # noqa: E402
    EXTERNAL_SOURCES,
    common_direction_errors,
    moving_joint_names,
    solve_external,
)
from tools.render_tianji_kai_mount_proxy_audit import (  # noqa: E402
    HEIGHT,
    WIDTH,
    body_bounds,
    color_bodies,
    evidence,
    font,
    link_frames,
    load_contract,
    place_hand,
    project,
    set_robot_neutral,
    sha256,
)
from tools.run_newtask_robot_kinematic_canary import (  # noqa: E402
    bounded_temporal_limits,
    rigid_initializer,
    robust_transform,
)


SIDES = ("left", "right")
HUMAN_TO_PHYSICAL = (1, 0)
PHYSICAL_TO_HUMAN = (1, 0)


class TaskMotionProxyError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise TaskMotionProxyError(f"ordinary JSON file required: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TaskMotionProxyError(f"JSON object required: {path}")
    return value


def validate_record(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file() or path.is_symlink():
        raise TaskMotionProxyError(f"{label} ordinary file missing")
    if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
        raise TaskMotionProxyError(f"{label} bytes/SHA drift")
    return path


def load_task_contract(path: Path) -> tuple[dict[str, Any], Path]:
    contract = load_json(path)
    exact = {
        "classification": "MOUNT_PROXY_TASK_MOTION_PREVIEW",
        "development_only": True,
        "real_robot_calibration": False,
        "deployment_authorized": False,
        "formal_robot_authorized": False,
        "training_authorized": False,
        "task_motion_preview_authorized": True,
        "ego_overlay_authorized": False,
        "object6d_consumed": False,
        "clean_consumed": False,
        "contact_claim_authorized": False,
        "procedural_connector_geometry_allowed": False,
        "adapter_cad_rendered": False,
    }
    for key, expected in exact.items():
        if contract.get(key) != expected:
            raise TaskMotionProxyError(f"task contract {key} drift")
    rules = contract["task_motion_rules"]
    if rules.get("human_to_physical") != {"left": "right", "right": "left"}:
        raise TaskMotionProxyError("cross-side mapping contract drift")
    if not (
        float(rules["arm_step_max_rad_per_frame"]) == 0.12
        and float(rules["hand_step_max_rad_per_frame"]) == 0.08
        and float(rules["solver_effective_step_max_rad_per_frame"]) == 0.06
        and float(rules["second_difference_max_rad_per_frame2"]) == 0.06
    ):
        raise TaskMotionProxyError("temporal contract drift")
    mount_contract = validate_record(contract["mount_source"], "mount contract")
    validate_record(contract["component_review"], "component review")
    return contract, mount_contract


def bounded_projection(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    effective_step: float = 0.06,
    acceleration: float = 0.06,
) -> np.ndarray:
    if target.ndim != 3 or target.shape[1:] != lower.shape or lower.shape != upper.shape:
        raise TaskMotionProxyError("trajectory/limit shape mismatch")
    if not np.isfinite(target).all():
        raise TaskMotionProxyError("non-finite target trajectory")
    output = np.empty_like(target, dtype=np.float64)
    output[0] = np.clip(target[0], lower, upper)
    for frame in range(1, len(target)):
        lo = np.maximum(lower, output[frame - 1] - effective_step)
        hi = np.minimum(upper, output[frame - 1] + effective_step)
        if frame >= 2:
            stop_center = 2.0 * output[frame - 1] - output[frame - 2]
            lo = np.maximum(lo, stop_center - acceleration)
            hi = np.minimum(hi, stop_center + acceleration)
        if np.any(lo > hi + 1e-12):
            raise TaskMotionProxyError(f"empty temporal feasible set at frame {frame}")
        output[frame] = np.clip(target[frame], lo, hi)
    return output


def temporal_metrics(q: np.ndarray) -> dict[str, float]:
    velocity = np.diff(q, axis=0)
    acceleration = np.diff(q, n=2, axis=0)
    return {
        "step_max_rad_per_frame": float(np.max(np.abs(velocity))) if len(velocity) else 0.0,
        "second_difference_max_rad_per_frame2": (
            float(np.max(np.abs(acceleration))) if len(acceleration) else 0.0
        ),
    }


def solve_arm_trajectory(
    assets: Any,
    targets: np.ndarray,
    world_rig: np.ndarray,
    mounts: tuple[np.ndarray, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Solve soft wrist targets with previous-only viable temporal bounds."""
    neutral = 0.5 * (lower + upper)
    output = np.empty((len(targets), 2, 7), dtype=np.float64)
    previous = neutral.copy()
    previous_previous: np.ndarray | None = None
    for frame in range(len(targets)):
        current = np.empty((2, 7), dtype=np.float64)
        for physical in range(2):
            target_tool = targets[frame, physical] @ np.linalg.inv(mounts[physical])
            if frame == 0:
                solve_lower, solve_upper = lower[physical], upper[physical]
            else:
                solve_lower, solve_upper = bounded_temporal_limits(
                    lower[physical], upper[physical], previous[physical],
                    None if previous_previous is None else previous_previous[physical],
                    0.06, 0.06,
                )
            seed = np.clip(previous[physical], solve_lower, solve_upper)
            q_position, _ = official._solve_one_arm_position_only(
                assets, side=physical, base=world_rig, target_tool=target_tool,
                initial_q=seed, lower=solve_lower, upper=solve_upper,
            )
            q_pose, _ = official._solve_one_arm(
                assets, side=physical, base=world_rig, target_tool=target_tool,
                initial_q=q_position, lower=solve_lower, upper=solve_upper,
            )
            current[physical] = q_pose
        if frame >= 1:
            previous_previous = previous.copy()
        previous = current
        output[frame] = current
    return output


def rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    relative = actual[:3, :3].T @ target[:3, :3]
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(angle))


def reset_named_joints(client: int, body: int, values: dict[str, float]) -> None:
    seen: set[str] = set()
    for index in range(bullet.getNumJoints(body, physicsClientId=client)):
        info = bullet.getJointInfo(body, index, physicsClientId=client)
        name = info[1].decode()
        if name in values:
            bullet.resetJointState(body, index, values[name], physicsClientId=client)
            seen.add(name)
    missing = set(values) - seen
    if missing:
        raise TaskMotionProxyError(f"joint names absent in PyBullet URDF: {sorted(missing)}")


def annotate_motion(
    frame: np.ndarray,
    view: tuple[float, ...],
    projection: tuple[float, ...],
    flange_frames: tuple[np.ndarray, np.ndarray],
    hand_frames: tuple[np.ndarray, np.ndarray],
    source_frame: int,
) -> np.ndarray:
    output = np.ascontiguousarray(frame.copy())
    for side, (flange, hand) in enumerate(zip(flange_frames, hand_frames, strict=True)):
        midpoint = 0.5 * (flange[:3, 3] + hand[:3, 3])
        marker = project(midpoint, view, projection)
        if marker is not None:
            cv2.drawMarker(output, marker, (0, 0, 230), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(
                output,
                f"{'L' if side == 0 else 'R'} MOUNT GAP",
                (marker[0] + 7, marker[1] + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 220),
                1,
                cv2.LINE_AA,
            )
    canvas = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 137), fill=(8, 8, 8))
    draw.text((14, 5), "Poker Robot 动作短预览（完整Tianji + 双KaiHand）", font=font(24), fill=(255, 255, 255))
    draw.text((14, 38), "MOUNT_PROXY | ADAPTER_CAD_MISSING | DEVELOPMENT_ONLY", font=font(21), fill=(255, 177, 30))
    draw.text((14, 69), "NO_OBJECT6D | CONTACT_PENDING | 无Clean依赖", font=font(20), fill=(255, 110, 100))
    draw.text((14, 99), "非实测 / 非标定 / 不可部署 | endpoint与手型仅软目标并报告残差", font=font(18), fill=(255, 125, 125))
    draw.text((14, 122), f"source frame {source_frame:05d} | previous-only | effective step≤0.06 rad | Δ²≤0.06", font=font(14), fill=(205, 225, 255))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def decode_video(path: Path, expected: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    result = {
        "decoded_frames": count,
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if count != expected or (result["width"], result["height"]) != (WIDTH, HEIGHT):
        raise TaskMotionProxyError("video decode contract failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-contract", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise TaskMotionProxyError(f"refusing to overwrite: {args.output_dir}")
    if args.fps <= 0 or args.repeat < 1:
        raise TaskMotionProxyError("fps>0 and repeat>=1 required")

    task_contract, mount_contract_path = load_task_contract(args.task_contract)
    mount_contract, asset_paths, mounts = load_contract(mount_contract_path)
    hawor_result = load_json(args.hawor_result)
    expected_session = task_contract["task_motion_rules"]["source_session"]
    if hawor_result.get("session_id") != expected_session:
        raise TaskMotionProxyError("HaWoR RESULT session mismatch")
    for path in (args.hawor_npz, args.hawor_result, *EXTERNAL_SOURCES):
        if not path.is_file() or path.is_symlink():
            raise TaskMotionProxyError(f"ordinary input file required: {path}")
    with np.load(args.hawor_npz, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    local_frames = np.arange(24, dtype=np.int64)
    source_frames = np.asarray(hawor["original_frame_indices"], dtype=np.int64)[local_frames]
    observed = np.asarray(hawor["observed"], dtype=bool)
    provenance = np.asarray(hawor["provenance"])
    if not (
        observed.shape[0] == 2
        and observed.shape[1] >= 24
        and np.all(observed[:, local_frames])
        and np.all(provenance[:, local_frames] == "BOUNDED_PARAMETER_FIT")
        and np.array_equal(source_frames, np.arange(24))
    ):
        raise TaskMotionProxyError("HaWoR first24 identity/frame/provenance contract failed")
    if hawor_result.get("outputs", {}).get("npz", {}).get("sha256") != sha256(args.hawor_npz):
        raise TaskMotionProxyError("HaWoR RESULT numeric lineage mismatch")

    assets = load_pinned_robot_assets(PROJECT)
    arm_lower, arm_upper = official._arm_limits(assets)
    hand_models = (assets.left_hand, assets.right_hand)
    hand_names = tuple(moving_joint_names(model) for model in hand_models)
    hand_lower = np.asarray(
        [[joint.lower for joint in model.joints if joint.joint_type != "fixed"] for model in hand_models],
        dtype=np.float64,
    )
    hand_upper = np.asarray(
        [[joint.upper for joint in model.joints if joint.joint_type != "fixed"] for model in hand_models],
        dtype=np.float64,
    )
    hand_basis = []
    for physical, model in enumerate(hand_models):
        midpoint = 0.5 * (hand_lower[physical] + hand_upper[physical])
        fk = official.forward_kinematics(
            model, dict(zip(hand_names[physical], midpoint, strict=True))
        )
        prefix = "hand_l" if physical == 0 else "hand_r"
        dummy = np.zeros((21, 3), dtype=np.float64)
        dummy[2] = fk[f"{prefix}_thumb_link1"][:3, 3]
        dummy[5] = fk[f"{prefix}_index_link1"][:3, 3]
        dummy[9] = fk[f"{prefix}_middle_link1"][:3, 3]
        dummy[17] = fk[f"{prefix}_pinky_link1"][:3, 3]
        hand_basis.append(
            adapter.final_v3_mano_palm_basis(dummy, handedness=SIDES[physical])
        )
    world = np.asarray(hawor["joints_3d_world"], dtype=np.float64)
    targets = np.empty((24, 2, 4, 4), dtype=np.float64)
    for slot, local in enumerate(local_frames):
        for physical in range(2):
            human = PHYSICAL_TO_HUMAN[physical]
            points = world[human, local]
            targets[slot, physical] = np.eye(4, dtype=np.float64)
            targets[slot, physical, :3, :3] = (
                adapter.final_v3_mano_palm_basis(points, handedness=SIDES[human])
                @ hand_basis[physical].T
            )
            targets[slot, physical, :3, 3] = points[0]
    arm_neutral = 0.5 * (arm_lower + arm_upper)
    source_roots = [
        official._tool_fk(assets, physical, arm_neutral[physical]) @ mounts[physical]
        for physical in range(2)
    ]
    robust_targets = [
        robust_transform([targets[slot, physical] for slot in range(24)])
        for physical in range(2)
    ]
    world_rig = rigid_initializer(source_roots, robust_targets)
    q_arm = solve_arm_trajectory(
        assets, targets, world_rig, mounts, arm_lower, arm_upper
    )
    raw_hand, external_rows, external_names = solve_external(hawor, local_frames)
    if external_names != hand_names or raw_hand.shape != (24, 2, 22):
        raise TaskMotionProxyError("retargeting_human output joint order/shape mismatch")
    q_hand = bounded_projection(raw_hand, hand_lower, hand_upper)
    arm_temporal = temporal_metrics(q_arm)
    hand_temporal = temporal_metrics(q_hand)
    if not (
        arm_temporal["step_max_rad_per_frame"] <= 0.12 + 1e-9
        and hand_temporal["step_max_rad_per_frame"] <= 0.08 + 1e-9
        and arm_temporal["second_difference_max_rad_per_frame2"] <= 0.06 + 1e-9
        and hand_temporal["second_difference_max_rad_per_frame2"] <= 0.06 + 1e-9
        and np.all(q_arm >= arm_lower - 1e-9)
        and np.all(q_arm <= arm_upper + 1e-9)
        and np.all(q_hand >= hand_lower - 1e-9)
        and np.all(q_hand <= hand_upper + 1e-9)
    ):
        raise TaskMotionProxyError("projected trajectory violates hard gates")

    position_residual = np.empty((24, 2), dtype=np.float64)
    rotation_residual = np.empty((24, 2), dtype=np.float64)
    for slot, local in enumerate(local_frames):
        values = {name: 0.0 for row in official.ARM_JOINT_NAMES for name in row}
        for physical in range(2):
            values.update(dict(zip(official.ARM_JOINT_NAMES[physical], q_arm[slot, physical], strict=True)))
        arm_fk = official.forward_kinematics(assets.tianji, values)
        for physical in range(2):
            tool = "left_tool" if physical == 0 else "right_tool"
            actual = world_rig @ arm_fk[tool] @ mounts[physical]
            position_residual[slot, physical] = 1000.0 * np.linalg.norm(
                actual[:3, 3] - targets[slot, physical, :3, 3]
            )
            rotation_residual[slot, physical] = rotation_error_deg(
                actual, targets[slot, physical]
            )
    hand_common = common_direction_errors(hawor, local_frames, q_hand, hand_models, hand_names)

    args.output_dir.mkdir(parents=True)
    state_path = args.output_dir / "TASK_MOTION_PROXY_STATES.npz"
    np.savez_compressed(
        state_path,
        local_frames=local_frames,
        source_frames=source_frames,
        q_arm=q_arm,
        q_hand=q_hand,
        T_world_rig=world_rig,
        T_flange_hand_root_proxy=np.asarray(mounts),
        human_to_physical=np.asarray(HUMAN_TO_PHYSICAL, dtype=np.int32),
        arm_position_residual_mm=position_residual,
        arm_rotation_residual_deg=rotation_residual,
        hand_common_direction_error_deg=hand_common,
    )

    client = bullet.connect(bullet.DIRECT)
    try:
        robot = bullet.loadURDF(str(asset_paths["tianji_urdf"]), useFixedBase=True, physicsClientId=client)
        set_robot_neutral(client, robot)
        hands = tuple(
            bullet.loadURDF(str(asset_paths[key]), useFixedBase=True, physicsClientId=client)
            for key in ("kaihand_left_urdf", "kaihand_right_urdf")
        )
        color_bodies(client, robot, hands)
        low, high = body_bounds(client, (robot, *hands))
        target = 0.5 * (low + high)
        distance = max(2.25, float(np.linalg.norm(high - low) * 1.30))
        view = bullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target.tolist(), distance=distance, yaw=48.0,
            pitch=-20.0, roll=0.0, upAxisIndex=2,
        )
        projection = bullet.computeProjectionMatrixFOV(48.0, WIDTH / HEIGHT, 0.03, 20.0)
        video = args.output_dir / "POKER_TIANJI_DUAL_KAIHAND_TASK_MOTION_MOUNT_PROXY_CANARY24.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (WIDTH, HEIGHT))
        if not writer.isOpened():
            raise TaskMotionProxyError("failed to open video writer")
        still_slots = set(np.linspace(0, 23, 4, dtype=np.int32).tolist())
        stills: list[np.ndarray] = []
        for slot in range(24):
            arm_values = {
                name: float(value)
                for side in range(2)
                for name, value in zip(official.ARM_JOINT_NAMES[side], q_arm[slot, side], strict=True)
            }
            reset_named_joints(client, robot, arm_values)
            frames = link_frames(client, robot)
            flange_frames = (frames["flange_L"], frames["flange_R"])
            hand_frames = tuple(flange @ mount for flange, mount in zip(flange_frames, mounts, strict=True))
            for physical, body in enumerate(hands):
                place_hand(client, body, hand_frames[physical])
                reset_named_joints(
                    client, body,
                    dict(zip(hand_names[physical], q_hand[slot, physical], strict=True)),
                )
            bullet.performCollisionDetection(physicsClientId=client)
            _, _, rgba, _, _ = bullet.getCameraImage(
                WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=projection,
                renderer=bullet.ER_TINY_RENDERER, shadow=1,
                lightDirection=(-1.0, -1.0, 2.0), physicsClientId=client,
            )
            bgr = cv2.cvtColor(np.asarray(rgba, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
            annotated = annotate_motion(
                bgr, view, projection, flange_frames, hand_frames, int(source_frames[slot])
            )
            for _ in range(args.repeat):
                writer.write(annotated)
            if slot in still_slots:
                stills.append(annotated)
        writer.release()
        sheet = np.vstack((np.hstack(stills[:2]), np.hstack(stills[2:])))
        sheet_path = args.output_dir / "POKER_TIANJI_DUAL_KAIHAND_TASK_MOTION_PROXY_4FRAME.png"
        if len(stills) != 4 or not cv2.imwrite(str(sheet_path), sheet):
            raise TaskMotionProxyError("failed to write review sheet")
    finally:
        bullet.disconnect(client)

    decode = decode_video(video, 24 * args.repeat)
    external_tip = np.asarray(
        [row["tip_error_mm"] for frame in external_rows for row in frame], dtype=np.float64
    )
    result = {
        "schema_version": "tianji-kaihand-task-motion-mount-proxy-result-v1",
        "status": "READY_FOR_USER_TASK_MOTION_PROXY_REVIEW",
        "grade": "B_DEVELOPMENT_ONLY",
        "task": "poker",
        "session": task_contract["task_motion_rules"]["source_session"],
        "frame_count": 24,
        "source_frames": source_frames.tolist(),
        "development_only": True,
        "formal_robot_ready": False,
        "deployment_authorized": False,
        "training_authorized": False,
        "mount_proxy": True,
        "adapter_cad_rendered": False,
        "procedural_connector_geometry_rendered": False,
        "object6d_consumed": False,
        "clean_consumed": False,
        "contact_claim": False,
        "collision_claim": False,
        "human_to_physical": {"left": "right", "right": "left"},
        "hard_gates": {
            "identity_frame_coordinate": "PASS",
            "finite": "PASS",
            "pinned_urdf_limits": "PASS",
            "previous_only": "PASS",
            "arm_step_le_0p12": "PASS",
            "hand_step_le_0p08": "PASS",
            "second_difference_le_0p06": "PASS",
            "video_full_decode": "PASS",
        },
        "soft_quality_metrics_not_hard_gates": {
            "arm_position_residual_mm": {
                "mean": float(np.mean(position_residual)),
                "p95": float(np.percentile(position_residual, 95)),
                "max": float(np.max(position_residual)),
            },
            "arm_rotation_residual_deg": {
                "mean": float(np.mean(rotation_residual)),
                "p95": float(np.percentile(rotation_residual, 95)),
                "max": float(np.max(rotation_residual)),
            },
            "hand_common_direction_error_deg": {
                "mean": float(np.mean(hand_common)),
                "p95": float(np.percentile(hand_common, 95)),
                "max": float(np.max(hand_common)),
            },
            "retargeting_human_tip_error_mm_before_temporal_projection": {
                "mean": float(np.mean(external_tip)),
                "max": float(np.max(external_tip)),
            },
        },
        "temporal": {"arm": arm_temporal, "hand": hand_temporal},
        "inputs": {
            "producer": evidence(Path(__file__)),
            "task_contract": evidence(args.task_contract),
            "mount_contract": evidence(mount_contract_path),
            "hawor": evidence(args.hawor_npz),
            "hawor_result": evidence(args.hawor_result),
            "retargeting_human_retargeter": evidence(EXTERNAL_SOURCES[0]),
            "retargeting_human_geometry": evidence(EXTERNAL_SOURCES[1]),
            "assets": {key: evidence(value) for key, value in asset_paths.items()},
        },
        "outputs": {
            "video": evidence(video),
            "review_sheet": evidence(sheet_path),
            "states": evidence(state_path),
        },
        "video_decode": decode,
        "claim_limit": task_contract["claim_limit"],
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": result["status"], "video": str(video)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
