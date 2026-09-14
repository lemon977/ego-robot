#!/usr/bin/env python3
"""Render a fixed, explicitly non-calibrated Tianji + dual-KaiHand proxy.

The renderer consumes only pinned URDF assets and an immutable MOUNT_PROXY
contract.  It never consumes an ego session, HaWoR, IK, Object6D, Clean, a
frame-derived fit, or an adapter mesh.  The empty physical adapter span stays
empty in 3D and is called out by a 2D warning label.
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
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pipeline import robot_scene_state_cpu as official  # noqa: E402
from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets  # noqa: E402


WIDTH, HEIGHT = 960, 720
SIDES = ("left", "right")


class ProxyAuditError(RuntimeError):
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
    raise ProxyAuditError("Chinese font unavailable")


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


def quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(map(float, bullet.getQuaternionFromEuler(matrix_to_euler(rotation))))


def transform_from_pose(
    position: tuple[float, ...], orientation_xyzw: tuple[float, ...]
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    result[:3, :3] = np.asarray(
        bullet.getMatrixFromQuaternion(orientation_xyzw), dtype=np.float64
    ).reshape(3, 3)
    return result


def validate_se3(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ProxyAuditError(f"{name} must be one finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ProxyAuditError(f"{name} homogeneous row invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0):
        raise ProxyAuditError(f"{name} rotation not orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-9:
        raise ProxyAuditError(f"{name} rotation determinant not +1")
    return matrix


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, Path], tuple[np.ndarray, ...]]:
    if not path.is_file() or path.is_symlink():
        raise ProxyAuditError(f"ordinary contract file required: {path}")
    contract = json.loads(path.read_text())
    required_flags = {
        "classification": "MOUNT_PROXY",
        "development_only": True,
        "real_robot_calibration": False,
        "deployment_authorized": False,
        "task_motion_authorized": False,
        "camera_or_ego_input_allowed": False,
        "frame0_or_session_fit_allowed": False,
        "procedural_connector_geometry_allowed": False,
        "adapter_cad_rendered": False,
    }
    for key, expected in required_flags.items():
        if contract.get(key) != expected:
            raise ProxyAuditError(f"contract {key} must be {expected!r}")
    assets: dict[str, Path] = {}
    for key in ("tianji_urdf", "kaihand_left_urdf", "kaihand_right_urdf"):
        record = contract["assets"][key]
        asset = Path(record["path"])
        if not asset.is_file() or asset.is_symlink():
            raise ProxyAuditError(f"ordinary asset file required: {asset}")
        if asset.stat().st_size != int(record["bytes"]):
            raise ProxyAuditError(f"{key} byte count drift")
        if sha256(asset) != record["sha256"]:
            raise ProxyAuditError(f"{key} SHA drift")
        assets[key] = asset
    excluded = contract["unverified_cad_excluded"]
    excluded_path = Path(excluded["path"])
    if not excluded_path.is_file() or sha256(excluded_path) != excluded["sha256"]:
        raise ProxyAuditError("excluded unverified CAD evidence drift")
    transforms = contract["proxy_transforms"]
    if transforms["matrix_direction"] != "p_flange = T_flange_hand_root_proxy @ p_hand_root":
        raise ProxyAuditError("unexpected proxy transform direction")
    mounts = (
        validate_se3(transforms["left_flange_L_to_hand_l_base_link"], "left mount"),
        validate_se3(transforms["right_flange_R_to_hand_r_base_link"], "right mount"),
    )
    return contract, assets, mounts


def set_robot_neutral(client: int, robot: int) -> None:
    dynamics = bullet.getDynamicsInfo(robot, -1, physicsClientId=client)
    root_to_com = transform_from_pose(dynamics[3], dynamics[4])
    bullet.resetBasePositionAndOrientation(
        robot,
        root_to_com[:3, 3].tolist(),
        quaternion(root_to_com[:3, :3]),
        physicsClientId=client,
    )
    for joint in range(bullet.getNumJoints(robot, physicsClientId=client)):
        info = bullet.getJointInfo(robot, joint, physicsClientId=client)
        if info[2] == bullet.JOINT_REVOLUTE:
            bullet.resetJointState(robot, joint, 0.0, physicsClientId=client)


def place_hand(client: int, body: int, hand_root_world: np.ndarray) -> None:
    dynamics = bullet.getDynamicsInfo(body, -1, physicsClientId=client)
    root_to_com = transform_from_pose(dynamics[3], dynamics[4])
    com_world = hand_root_world @ root_to_com
    bullet.resetBasePositionAndOrientation(
        body,
        com_world[:3, 3].tolist(),
        quaternion(com_world[:3, :3]),
        physicsClientId=client,
    )
    for joint in range(bullet.getNumJoints(body, physicsClientId=client)):
        bullet.resetJointState(body, joint, 0.0, physicsClientId=client)


def link_frames(client: int, robot: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    base_pose = bullet.getBasePositionAndOrientation(robot, physicsClientId=client)
    base_dynamics = bullet.getDynamicsInfo(robot, -1, physicsClientId=client)
    world_com = transform_from_pose(base_pose[0], base_pose[1])
    root_to_com = transform_from_pose(base_dynamics[3], base_dynamics[4])
    output["base_link"] = world_com @ np.linalg.inv(root_to_com)
    for joint in range(bullet.getNumJoints(robot, physicsClientId=client)):
        name = bullet.getJointInfo(robot, joint, physicsClientId=client)[12].decode()
        state = bullet.getLinkState(
            robot, joint, computeForwardKinematics=True, physicsClientId=client
        )
        output[name] = transform_from_pose(state[4], state[5])
    return output


def color_bodies(client: int, robot: int, hands: tuple[int, int]) -> None:
    bullet.changeVisualShape(
        robot, -1, rgbaColor=(0.34, 0.36, 0.39, 1.0), physicsClientId=client
    )
    for link in range(bullet.getNumJoints(robot, physicsClientId=client)):
        name = bullet.getJointInfo(robot, link, physicsClientId=client)[12].decode()
        if name.endswith("_L") or name == "left_tool":
            color = (0.18, 0.47, 0.72, 1.0)
        elif name.endswith("_R") or name == "right_tool":
            color = (0.80, 0.39, 0.16, 1.0)
        else:
            color = (0.38, 0.40, 0.43, 1.0)
        bullet.changeVisualShape(robot, link, rgbaColor=color, physicsClientId=client)
    for body, color in zip(
        hands, ((0.15, 0.74, 0.88, 1.0), (0.90, 0.28, 0.58, 1.0)), strict=True
    ):
        bullet.changeVisualShape(body, -1, rgbaColor=color, physicsClientId=client)
        for link in range(bullet.getNumJoints(body, physicsClientId=client)):
            bullet.changeVisualShape(body, link, rgbaColor=color, physicsClientId=client)


def body_bounds(client: int, bodies: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    lows: list[tuple[float, ...]] = []
    highs: list[tuple[float, ...]] = []
    for body in bodies:
        for link in range(-1, bullet.getNumJoints(body, physicsClientId=client)):
            low, high = bullet.getAABB(body, link, physicsClientId=client)
            lows.append(low)
            highs.append(high)
    return np.min(lows, axis=0), np.max(highs, axis=0)


def project(
    point: np.ndarray, view: tuple[float, ...], projection: tuple[float, ...]
) -> tuple[int, int] | None:
    view_matrix = np.asarray(view, dtype=np.float64).reshape(4, 4, order="F")
    projection_matrix = np.asarray(projection, dtype=np.float64).reshape(4, 4, order="F")
    clip = projection_matrix @ view_matrix @ np.append(point, 1.0)
    if clip[3] <= 1e-8:
        return None
    ndc = clip[:3] / clip[3]
    if not np.isfinite(ndc).all():
        return None
    return (
        int(round((ndc[0] * 0.5 + 0.5) * WIDTH)),
        int(round((1.0 - (ndc[1] * 0.5 + 0.5)) * HEIGHT)),
    )


def annotate(
    frame: np.ndarray,
    view: tuple[float, ...],
    projection: tuple[float, ...],
    flange_frames: tuple[np.ndarray, np.ndarray],
    hand_frames: tuple[np.ndarray, np.ndarray],
    yaw: float,
) -> np.ndarray:
    output = np.ascontiguousarray(frame.copy())
    axis_colors = ((0, 0, 255), (0, 190, 0), (255, 80, 0))
    for side, (flange, hand) in enumerate(zip(flange_frames, hand_frames, strict=True)):
        side_name = "L" if side == 0 else "R"
        for prefix, pose in (("F", flange), ("H", hand)):
            origin = project(pose[:3, 3], view, projection)
            if origin is None:
                continue
            for axis, color in enumerate(axis_colors):
                endpoint = project(pose[:3, 3] + 0.06 * pose[:3, axis], view, projection)
                if endpoint is not None:
                    cv2.arrowedLine(output, origin, endpoint, color, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(
                output,
                f"{prefix}_{side_name}",
                (origin[0] + 4, origin[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )
        midpoint = 0.5 * (flange[:3, 3] + hand[:3, 3])
        marker = project(midpoint, view, projection)
        if marker is not None:
            cv2.drawMarker(output, marker, (0, 0, 220), cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(
                output,
                "ADAPTER CAD MISSING",
                (marker[0] + 8, marker[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 220),
                1,
                cv2.LINE_AA,
            )
    canvas = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 126), fill=(8, 8, 8))
    draw.text((14, 6), "Tianji 双臂 + 双 KaiHand 固定装配候选", font=font(24), fill=(255, 255, 255))
    draw.text((14, 39), "MOUNT_PROXY | 适配器CAD缺失 | 空隙未用圆柱/CAD伪造", font=font(21), fill=(255, 177, 30))
    draw.text((14, 70), "非实测 / 非标定 / 不可部署 | 无ego、无IK、无接触结论", font=font(20), fill=(255, 100, 100))
    draw.text((14, 99), f"全局base_link(ZJ_Robot_link)+Base_L/R+Link1..7+真实左右KaiHand | yaw {yaw:.1f}°", font=font(16), fill=(210, 225, 255))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def verify_decode(video: Path, expected_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if decoded != expected_frames or (width, height) != (WIDTH, HEIGHT):
        raise ProxyAuditError("output video decode contract failed")
    return {"decoded_frames": decoded, "fps": fps, "width": width, "height": height}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.frames < 4 or args.fps <= 0:
        raise ProxyAuditError("frames>=4 and fps>0 required")
    contract, asset_paths, mounts = load_contract(args.contract)
    pinned_assets = load_pinned_robot_assets(PROJECT)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS_VALIDATE_ONLY_MOUNT_PROXY",
                    "contract": evidence(args.contract),
                    "assets": {key: evidence(value) for key, value in asset_paths.items()},
                    "adapter_geometry_rendered": False,
                    "camera_or_ego_consumed": False,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ProxyAuditError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    q_values = {name: 0.0 for names in official.ARM_JOINT_NAMES for name in names}
    official_frames = official.forward_kinematics(pinned_assets.tianji, q_values)
    flange_frames = (official_frames["flange_L"], official_frames["flange_R"])
    hand_frames = tuple(
        flange @ mount for flange, mount in zip(flange_frames, mounts, strict=True)
    )

    client = bullet.connect(bullet.DIRECT)
    try:
        robot = bullet.loadURDF(
            str(asset_paths["tianji_urdf"]), useFixedBase=True, physicsClientId=client
        )
        set_robot_neutral(client, robot)
        bullet_frames = link_frames(client, robot)
        fk_residuals: dict[str, float] = {}
        for name in ("base_link", "flange_L", "flange_R", "left_tool", "right_tool"):
            residual = float(np.max(np.abs(bullet_frames[name] - official_frames[name])))
            fk_residuals[name] = residual
            if residual > 2e-6:
                raise ProxyAuditError(f"official/PyBullet FK mismatch for {name}: {residual}")

        hands_list: list[int] = []
        for path, root in zip(
            (asset_paths["kaihand_left_urdf"], asset_paths["kaihand_right_urdf"]),
            hand_frames,
            strict=True,
        ):
            body = bullet.loadURDF(str(path), useFixedBase=True, physicsClientId=client)
            place_hand(client, body, root)
            hands_list.append(body)
        hands = (hands_list[0], hands_list[1])
        color_bodies(client, robot, hands)

        hand_root_residuals: dict[str, float] = {}
        for side, body, expected in zip(SIDES, hands, hand_frames, strict=True):
            com_pose = bullet.getBasePositionAndOrientation(body, physicsClientId=client)
            world_com = transform_from_pose(com_pose[0], com_pose[1])
            dynamics = bullet.getDynamicsInfo(body, -1, physicsClientId=client)
            root_to_com = transform_from_pose(dynamics[3], dynamics[4])
            actual_root = world_com @ np.linalg.inv(root_to_com)
            residual = float(np.max(np.abs(actual_root - expected)))
            hand_root_residuals[side] = residual
            if residual > 2e-6:
                raise ProxyAuditError(f"{side} hand root placement mismatch: {residual}")

        low, high = body_bounds(client, (robot, *hands))
        target = 0.5 * (low + high)
        radius = float(np.linalg.norm(high - low) * 0.5)
        distance = max(2.2, radius * 2.55)
        yaws = np.linspace(25.0, 385.0, args.frames, endpoint=False)
        video = args.output_dir / "TIANJI_DUAL_KAIHAND_MOUNT_PROXY_ZERO_ORBIT.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (WIDTH, HEIGHT)
        )
        if not writer.isOpened():
            raise ProxyAuditError("failed to open output video")
        still_slots = set(np.linspace(0, args.frames - 1, 4, dtype=np.int32).tolist())
        stills: list[np.ndarray] = []
        for index, yaw in enumerate(yaws):
            view = bullet.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target.tolist(),
                distance=distance,
                yaw=float(yaw),
                pitch=-18.0,
                roll=0.0,
                upAxisIndex=2,
            )
            projection = bullet.computeProjectionMatrixFOV(
                48.0, WIDTH / HEIGHT, 0.03, 20.0
            )
            _, _, rgba, _, _ = bullet.getCameraImage(
                WIDTH,
                HEIGHT,
                viewMatrix=view,
                projectionMatrix=projection,
                renderer=bullet.ER_TINY_RENDERER,
                shadow=1,
                lightDirection=(-1.0, -1.0, 2.0),
                physicsClientId=client,
            )
            bgr = cv2.cvtColor(np.asarray(rgba, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
            frame = annotate(
                bgr, view, projection, flange_frames, hand_frames, float(yaw)
            )
            writer.write(frame)
            if index in still_slots:
                stills.append(frame)
        writer.release()
        if len(stills) != 4:
            raise ProxyAuditError("four review frames were not collected")
        sheet = np.vstack((np.hstack(stills[:2]), np.hstack(stills[2:])))
        sheet_path = args.output_dir / "TIANJI_DUAL_KAIHAND_MOUNT_PROXY_ZERO_4VIEW.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise ProxyAuditError("failed to write four-view sheet")
    finally:
        bullet.disconnect(client)

    decode = verify_decode(video, args.frames)
    result = {
        "schema_version": "tianji-dual-kaihand-mount-proxy-audit-result-v1",
        "status": "READY_FOR_USER_MOUNT_PROXY_COMPONENT_REVIEW_NOT_REAL_MOUNT",
        "classification": "MOUNT_PROXY",
        "development_only": True,
        "real_robot_calibration": False,
        "deployment_authorized": False,
        "task_motion_authorized": False,
        "consumed_ego_or_camera": False,
        "consumed_hawor_or_ik": False,
        "consumed_object6d_or_clean": False,
        "consumed_frame0_or_session_fit": False,
        "procedural_connector_geometry_rendered": False,
        "adapter_cad_rendered": False,
        "empty_adapter_span_annotated_only_in_2d": True,
        "assembly": {
            "tianji": "complete pinned global base_link/ZJ_Robot_link plus Base_L/R and Link1..7 at zero arm joints",
            "hands": "separate pinned real left and right KaiHand URDFs at zero hand joints",
            "mount": contract["proxy_transforms"],
        },
        "validation": {
            "official_vs_pybullet_fk_max_abs_by_link": fk_residuals,
            "hand_root_placement_max_abs_by_side": hand_root_residuals,
            "video": decode,
        },
        "inputs": {
            "producer": evidence(Path(__file__)),
            "contract": evidence(args.contract),
            "assets": {key: evidence(value) for key, value in asset_paths.items()},
        },
        "outputs": {
            "orbit_video": evidence(video),
            "four_view": evidence(sheet_path),
        },
        "missing_physical_authority": contract["missing_physical_authority"],
        "claim_limit": contract["claim_limit"],
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "video": str(video),
                "four_view": str(sheet_path),
                "result": str(result_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
