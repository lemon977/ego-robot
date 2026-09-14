#!/usr/bin/env python3
"""Compare the withdrawn Chaoyang canary hand pose with bounded KaiHand IK.

The comparison is deliberately hand-only.  It does not load Tianji, a mount,
camera-to-robot placement, Object6D, or Clean.  Both columns use the same
pinned KaiHand URDF bytes; only the q trajectory source differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as bullet


PROJECT = Path(__file__).resolve().parents[1]
RETARGET_PROJECT = Path("/mnt/workspace/code/retargeting_human")
for root in (str(PROJECT), str(RETARGET_PROJECT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from pipeline import robot_scene_state_cpu as official  # noqa: E402
from pipeline import robot_wrist_kai_adapter as adapter  # noqa: E402
from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets  # noqa: E402
from retargeting_human.retargeter import KaiHandRetargeter  # noqa: E402


KAI_URDFS = (
    PROJECT
    / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-L-260624(1620)"
    / "urdf/KaiBot-Dexhand shell-URDF-L-260624(1620).urdf",
    PROJECT
    / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-R-260424(1430)"
    / "urdf/KaiBot-Dexhand shell-URDF-R-260424(1430).urdf",
)
EXTERNAL_SOURCES = (
    RETARGET_PROJECT / "retargeting_human/retargeter.py",
    RETARGET_PROJECT / "retargeting_human/geometry.py",
)
SIDE_NAMES = ("left", "right")
HUMAN_TO_PHYSICAL = (1, 0)
PHYSICAL_TO_HUMAN = (1, 0)
MANO_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND_TERMINALS = {"thumb": 6, "index": 4, "middle": 4, "ring": 4, "pinky": 4}
PANEL_W, PANEL_H = 480, 360
RENDER_H = 284


class CompareError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise CompareError("Chinese font unavailable")


def matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    if sy > 1e-8:
        return (
            math.atan2(float(rotation[2, 1]), float(rotation[2, 2])),
            math.atan2(float(-rotation[2, 0]), sy),
            math.atan2(float(rotation[1, 0]), float(rotation[0, 0])),
        )
    return (
        math.atan2(float(-rotation[1, 2]), float(rotation[1, 1])),
        math.atan2(float(-rotation[2, 0]), sy),
        0.0,
    )


def place_root(client: int, body: int, translation: tuple[float, float, float]) -> None:
    dynamics = bullet.getDynamicsInfo(body, -1, physicsClientId=client)
    root_to_com = np.eye(4, dtype=np.float64)
    root_to_com[:3, 3] = np.asarray(dynamics[3], dtype=np.float64)
    root_to_com[:3, :3] = np.asarray(
        bullet.getMatrixFromQuaternion(dynamics[4]), dtype=np.float64
    ).reshape(3, 3)
    quaternion = bullet.getQuaternionFromEuler(matrix_to_euler(root_to_com[:3, :3]))
    position = root_to_com[:3, 3] + np.asarray(translation, dtype=np.float64)
    bullet.resetBasePositionAndOrientation(
        body, position.tolist(), quaternion, physicsClientId=client
    )


def moving_joint_names(model: Any) -> tuple[str, ...]:
    return tuple(j.name for j in model.joints if j.joint_type != "fixed")


def robot_palm_basis(model: Any, names: tuple[str, ...], physical: int) -> np.ndarray:
    moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
    lower = np.asarray([joint.lower for joint in moving], dtype=np.float64)
    upper = np.asarray([joint.upper for joint in moving], dtype=np.float64)
    fk = official.forward_kinematics(
        model, dict(zip(names, 0.5 * (lower + upper), strict=True))
    )
    prefix = "hand_l" if physical == 0 else "hand_r"
    dummy = np.zeros((21, 3), dtype=np.float64)
    dummy[2] = fk[f"{prefix}_thumb_link1"][:3, 3]
    dummy[5] = fk[f"{prefix}_index_link1"][:3, 3]
    dummy[9] = fk[f"{prefix}_middle_link1"][:3, 3]
    dummy[17] = fk[f"{prefix}_pinky_link1"][:3, 3]
    return adapter.final_v3_mano_palm_basis(dummy, handedness=SIDE_NAMES[physical])


def robot_finger_directions(
    model: Any,
    names: tuple[str, ...],
    basis: np.ndarray,
    q: np.ndarray,
    physical: int,
    finger: str,
) -> np.ndarray:
    fk = official.forward_kinematics(model, dict(zip(names, q, strict=True)))
    prefix = "hand_l" if physical == 0 else "hand_r"
    points = [np.zeros(3, dtype=np.float64)]
    for number in range(1, HAND_TERMINALS[finger] + 1):
        points.append(fk[f"{prefix}_{finger}_link{number}"][:3, 3])
    return adapter.resample_polyline_unit_directions(
        np.asarray(points), bone_count=4
    ) @ basis


def common_direction_errors(
    hawor: dict[str, np.ndarray],
    frames: np.ndarray,
    q: np.ndarray,
    models: tuple[Any, Any],
    names: tuple[tuple[str, ...], tuple[str, ...]],
) -> np.ndarray:
    joints = np.asarray(hawor["joints_3d_camera"], dtype=np.float64)
    bases = tuple(robot_palm_basis(models[side], names[side], side) for side in range(2))
    errors = np.empty((len(frames), 2, len(FINGERS), 4), dtype=np.float64)
    for slot, frame_id in enumerate(frames):
        for physical in range(2):
            human = PHYSICAL_TO_HUMAN[physical]
            target = adapter.final_v3_mano_unit_bones_local(
                joints[human, int(frame_id)], handedness=SIDE_NAMES[human]
            )
            for finger_index, finger in enumerate(FINGERS):
                actual = robot_finger_directions(
                    models[physical], names[physical], bases[physical],
                    q[slot, physical], physical, finger,
                )
                errors[slot, physical, finger_index] = np.degrees(
                    np.arccos(np.clip(np.sum(actual * target[finger], axis=1), -1.0, 1.0))
                )
    return errors


def solve_external(
    hawor: dict[str, np.ndarray], frames: np.ndarray
) -> tuple[np.ndarray, list[list[dict[str, Any]]], tuple[tuple[str, ...], ...]]:
    joints = np.asarray(hawor["joints_3d_camera"], dtype=np.float64)
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise CompareError("HaWoR joints_3d_camera must be (2,F,21,3)")
    solvers = tuple(
        KaiHandRetargeter(
            KAI_URDFS[side], SIDE_NAMES[side], max_iterations=9,
            max_step_deg=float(np.degrees(0.06)), position_scale_m=0.010,
        )
        for side in range(2)
    )
    rows: list[list[dict[str, Any]]] = []
    output = np.empty((len(frames), 2, 22), dtype=np.float64)
    try:
        for slot, source_frame in enumerate(frames):
            frame_rows: list[dict[str, Any]] = []
            for physical in range(2):
                human = PHYSICAL_TO_HUMAN[physical]
                points = joints[human, int(source_frame)]
                wrist = np.eye(4, dtype=np.float64)
                wrist[:3, :3] = adapter.final_v3_mano_palm_basis(
                    points, handedness=SIDE_NAMES[human]
                )
                wrist[:3, 3] = points[0]
                points20 = points[np.asarray(MANO_CHAINS, dtype=np.int64)]
                solved = solvers[physical].retarget(wrist, points20)
                output[slot, physical] = solved.q
                frame_rows.append(
                    {
                        "physical_side": SIDE_NAMES[physical],
                        "human_side": SIDE_NAMES[human],
                        "optimizer_success": bool(solved.optimizer_success),
                        "optimizer_evaluations": int(solved.optimizer_evaluations),
                        "optimizer_cost": float(solved.optimizer_cost),
                        "tip_error_mm": (np.asarray(solved.tip_error_m) * 1000.0).tolist(),
                        "direction_error_deg": np.asarray(solved.direction_error_deg).tolist(),
                    }
                )
            rows.append(frame_rows)
    finally:
        for solver in solvers:
            solver.close()
    return output, rows, tuple(tuple(s.joint_names) for s in solvers)


def body_bounds(client: int, body: int) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for link in range(-1, bullet.getNumJoints(body, physicsClientId=client)):
        low, high = bullet.getAABB(body, link, physicsClientId=client)
        lows.append(low)
        highs.append(high)
    return np.min(np.asarray(lows), axis=0), np.max(np.asarray(highs), axis=0)


def render_body(
    client: int,
    body: int,
    all_bodies: tuple[int, int],
    q: np.ndarray,
    *,
    side: int,
    color: tuple[float, float, float, float],
) -> np.ndarray:
    for other in all_bodies:
        place_root(client, other, (0.0, 0.0, 0.0) if other == body else (100.0, 0.0, 0.0))
    for joint, value in enumerate(q):
        bullet.resetJointState(body, joint, float(value), physicsClientId=client)
    bullet.changeVisualShape(body, -1, rgbaColor=color, physicsClientId=client)
    for link in range(bullet.getNumJoints(body, physicsClientId=client)):
        bullet.changeVisualShape(body, link, rgbaColor=color, physicsClientId=client)
    low, high = body_bounds(client, body)
    target = 0.5 * (low + high)
    distance = max(0.34, float(np.linalg.norm(high - low) * 1.45))
    yaw = 150.0 if side == 0 else 35.0
    view = bullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target.tolist(), distance=distance, yaw=yaw,
        pitch=-12.0, roll=0.0, upAxisIndex=2,
    )
    projection = bullet.computeProjectionMatrixFOV(
        43.0, PANEL_W / RENDER_H, 0.005, 3.0
    )
    _, _, rgba, _, _ = bullet.getCameraImage(
        PANEL_W, RENDER_H, viewMatrix=view, projectionMatrix=projection,
        renderer=bullet.ER_TINY_RENDERER, shadow=1,
        lightDirection=(-1.0, -1.0, 2.0), physicsClientId=client,
    )
    return cv2.cvtColor(np.asarray(rgba, dtype=np.uint8), cv2.COLOR_RGBA2BGR)


def labelled_panel(
    rendered: np.ndarray,
    *,
    physical: int,
    method: str,
    frame_id: int,
    q_delta_rms_deg: float,
    common_direction_mean_deg: float,
) -> np.ndarray:
    output = np.full((PANEL_H, PANEL_W, 3), 245, dtype=np.uint8)
    output[76:] = rendered
    canvas = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    is_external = method.startswith("retargeting")
    title_color = (25, 130, 65) if is_external else (35, 85, 190)
    human = PHYSICAL_TO_HUMAN[physical]
    draw.text(
        (10, 5),
        f"物理{'左' if physical == 0 else '右'}手 ← human {'right' if human == 1 else 'left'} | {method}",
        font=font(18), fill=title_color,
    )
    draw.text(
        (10, 35),
        f"source f{frame_id:05d} | q差RMS={q_delta_rms_deg:.1f}° | 同一角度指标={common_direction_mean_deg:.1f}°",
        font=font(15), fill=(30, 30, 30),
    )
    draw.text(
        (10, 57), "HAND_ONLY | NO_TIANJI | NOT_BASELINE", font=font(13), fill=(190, 65, 35)
    )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--canary-scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()
    for path in (args.hawor_npz, args.canary_scene, *KAI_URDFS, *EXTERNAL_SOURCES):
        if not path.is_file() or path.is_symlink():
            raise CompareError(f"ordinary input file required: {path}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise CompareError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    assets = load_pinned_robot_assets(PROJECT)
    with np.load(args.hawor_npz, allow_pickle=False) as archive:
        hawor = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.canary_scene, allow_pickle=False) as archive:
        scene = {name: np.asarray(archive[name]) for name in archive.files}
    frames = np.asarray(scene["source_frames"], dtype=np.int64)
    canary_q = np.asarray(scene["q_hand"], dtype=np.float64)
    if canary_q.shape != (len(frames), 2, 22):
        raise CompareError("canary q_hand shape mismatch")
    if not np.array_equal(scene["human_to_physical"], np.asarray(HUMAN_TO_PHYSICAL)):
        raise CompareError("canary cross-side mapping mismatch")
    external_q, external_rows, external_names = solve_external(hawor, frames)
    expected_names = tuple(moving_joint_names(model) for model in (assets.left_hand, assets.right_hand))
    if external_names != expected_names:
        raise CompareError("retargeting_human joint order differs from pinned URDF order")
    q_difference_deg = np.degrees(external_q - canary_q)
    models = (assets.left_hand, assets.right_hand)
    canary_common = common_direction_errors(
        hawor, frames, canary_q, models, expected_names
    )
    external_common = common_direction_errors(
        hawor, frames, external_q, models, expected_names
    )

    client = bullet.connect(bullet.DIRECT)
    try:
        bodies = tuple(
            bullet.loadURDF(str(KAI_URDFS[side]), useFixedBase=True, physicsClientId=client)
            for side in range(2)
        )
        for body in bodies:
            place_root(client, body, (0.0, 0.0, 0.0))
        video_path = args.output_dir / f"{args.session}_KAIHAND_RETARGET_COMPARE_HAND_ONLY.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
            (PANEL_W * 2, PANEL_H * 2),
        )
        if not writer.isOpened():
            raise CompareError("failed to open comparison video")
        review_frames: list[np.ndarray] = []
        selected = set(np.linspace(0, len(frames) - 1, min(4, len(frames)), dtype=np.int32).tolist())
        for slot, frame_id in enumerate(frames):
            panels: list[np.ndarray] = []
            for physical in range(2):
                rms = float(np.sqrt(np.mean(q_difference_deg[slot, physical] ** 2)))
                canary_direction_mean = float(np.mean(canary_common[slot, physical]))
                external_direction_mean = float(np.mean(external_common[slot, physical]))
                canary = render_body(
                    client, bodies[physical], bodies, canary_q[slot, physical], side=physical,
                    color=(0.16, 0.48, 0.85, 1.0),
                )
                panels.append(
                    labelled_panel(
                        canary, physical=physical, method="chaoyang canary (已撤回路线)",
                        frame_id=int(frame_id), q_delta_rms_deg=rms,
                        common_direction_mean_deg=canary_direction_mean,
                    )
                )
                external = render_body(
                    client, bodies[physical], bodies, external_q[slot, physical], side=physical,
                    color=(0.18, 0.72, 0.34, 1.0),
                )
                panels.append(
                    labelled_panel(
                        external, physical=physical, method="retargeting_human bounded IK",
                        frame_id=int(frame_id), q_delta_rms_deg=rms,
                        common_direction_mean_deg=external_direction_mean,
                    )
                )
            composite = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
            writer.write(composite)
            if slot in selected:
                review_frames.append(composite)
        writer.release()
        sheet_path = args.output_dir / f"{args.session}_KAIHAND_RETARGET_COMPARE_4FRAME.png"
        sheet = np.vstack(review_frames)
        if not cv2.imwrite(str(sheet_path), sheet):
            raise CompareError("failed to write comparison sheet")
    finally:
        bullet.disconnect(client)

    all_direction = np.asarray(
        [row["direction_error_deg"] for frame in external_rows for row in frame],
        dtype=np.float64,
    )
    all_tip = np.asarray(
        [row["tip_error_mm"] for frame in external_rows for row in frame],
        dtype=np.float64,
    )
    states_path = args.output_dir / "HAND_RETARGET_COMPARE_STATES.npz"
    np.savez_compressed(
        states_path,
        source_frames=frames,
        canary_q=canary_q,
        retargeting_human_q=external_q,
        canary_common_direction_error_deg=canary_common,
        retargeting_human_common_direction_error_deg=external_common,
        human_to_physical=np.asarray(HUMAN_TO_PHYSICAL, dtype=np.int32),
    )

    def common_summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_deg": float(np.mean(values)),
            "p95_deg": float(np.percentile(values, 95)),
            "max_deg": float(np.max(values)),
            "distal_mean_deg": float(np.mean(values[..., -1])),
        }

    canary_summary = common_summary(canary_common)
    external_summary = common_summary(external_common)
    recommendation = (
        "CHAOYANG_CANARY_HAND_ONLY"
        if canary_summary["p95_deg"] <= external_summary["p95_deg"]
        else "RETARGETING_HUMAN_BOUNDED_IK_HAND_ONLY"
    )
    result = {
        "schema_version": "kaihand-retargeter-hand-only-comparison-v1",
        "status": "READY_FOR_USER_HAND_SHAPE_REVIEW_NOT_BASELINE",
        "task": args.task,
        "session": args.session,
        "frame_count": int(len(frames)),
        "source_frames": frames.tolist(),
        "human_to_physical": {"left": "right", "right": "left"},
        "same_pinned_kaihand_urdf_for_both_methods": True,
        "tianji_consumed": False,
        "mount_consumed": False,
        "ego_rgb_consumed": False,
        "external_bounded_ik": {
            "max_step_rad": 0.06,
            "joint_limits_from_urdf": True,
            "mean_direction_error_deg": float(np.mean(all_direction)),
            "max_direction_error_deg": float(np.max(all_direction)),
            "mean_tip_error_mm": float(np.mean(all_tip)),
            "max_tip_error_mm": float(np.max(all_tip)),
        },
        "between_method_q_difference_deg": {
            "rms": float(np.sqrt(np.mean(q_difference_deg**2))),
            "max_abs": float(np.max(np.abs(q_difference_deg))),
        },
        "common_mano_unit_bone_direction_metric": {
            "definition": "Both q trajectories evaluated by the same pinned KaiHand FK against the same HaWoR MANO unit-bone directions in the same physical-side palm basis.",
            "chaoyang_canary": canary_summary,
            "retargeting_human_bounded_ik": external_summary,
        },
        "recommended_hand_only_baseline": recommendation,
        "recommendation_rule": "Lower common-metric p95 angular error; this recommendation is limited to hand shape and does not authorize arm/mount/ego Robot.",
        "claim_limit": "Hand-shape diagnostic only. The two methods use different objectives; visual or numeric differences do not establish real-robot correctness, mount, arm IK, contact, or a current baseline.",
        "inputs": {
            "producer": evidence(Path(__file__)),
            "hawor": evidence(args.hawor_npz),
            "withdrawn_canary_scene": evidence(args.canary_scene),
            "kaihand_left_urdf": evidence(KAI_URDFS[0]),
            "kaihand_right_urdf": evidence(KAI_URDFS[1]),
            "retargeting_human_retargeter": evidence(EXTERNAL_SOURCES[0]),
            "retargeting_human_geometry": evidence(EXTERNAL_SOURCES[1]),
        },
        "outputs": {
            "comparison_video": evidence(video_path),
            "comparison_sheet": evidence(sheet_path),
            "comparison_states": evidence(states_path),
        },
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "video": str(video_path), "sheet": str(sheet_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
