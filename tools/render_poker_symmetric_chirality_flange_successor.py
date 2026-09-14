#!/usr/bin/env python3
"""Render the single bounded Poker Robot visual-correction successor.

The artifact is deliberately a development review, not a real mount or camera
calibration.  It starts from the symmetric Tianji URDF neutral configuration,
uses retargeting_human for both real KaiHand URDFs, and draws an explicitly
labelled flange-ring proxy whose length comes from the Tianji URDF flange-to-
tool fixed joint.  The same tool-to-hand transform is used by IK and rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh


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
    FINGERS,
    HAND_TERMINALS,
    KAI_URDFS,
    MANO_CHAINS,
    moving_joint_names,
    solve_external,
)
from tools.render_background_agnostic_robot_view import (  # noqa: E402
    RASTER_SOURCE,
    arrays,
    artifact,
    encode_video,
    load_module,
    tint,
)
from tools.render_tianji_kai_mount_proxy_audit import font, sha256  # noqa: E402
from tools.render_tianji_kai_task_motion_proxy import (  # noqa: E402
    bounded_projection,
    rotation_error_deg,
    temporal_metrics,
)


SESSION = "play_cards_0902_042"
SIDES = ("left", "right")
PHYSICAL_TO_HUMAN = (1, 0)
WIDTH, HEIGHT = 640, 480
OUT_WIDTH = WIDTH * 3
KEY_SLOTS = (0, 8, 15, 23)
FLANGE_RING_LENGTH_M = 0.145
FLANGE_RING_OUTER_RADIUS_M = 0.04100007
FLANGE_RING_INNER_RADIUS_M = 0.020500035
ARM_EFFECTIVE_STEP = 0.06
HAND_EFFECTIVE_STEP = 0.06
SECOND_DIFFERENCE = 0.06
MANO_DRAW_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


class SuccessorError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SuccessorError(f"ordinary JSON required: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SuccessorError(f"JSON object required: {path}")
    return value


def require_review(path: Path, stage: str) -> dict[str, Any]:
    value = load_json(path)
    if not (
        value.get("session") == SESSION
        and value.get("grade") in ("A", "B")
        and value.get("downstream_authorized") is True
    ):
        raise SuccessorError(f"{stage} review identity/grade/downstream mismatch")
    return value


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 180.0
    return float(np.degrees(np.arccos(np.clip((a / na) @ (b / nb), -1.0, 1.0))))


def proper_axis_rotations() -> tuple[np.ndarray, ...]:
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            value = np.zeros((3, 3), dtype=np.float64)
            value[np.arange(3), permutation] = signs
            if np.linalg.det(value) > 0.5:
                result.append(value)
    if len(result) != 24:
        raise SuccessorError("proper signed-axis rotation enumeration drift")
    return tuple(result)


def hand_landmarks(
    assets: Any, names: tuple[tuple[str, ...], ...], q_hand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = []
    for physical, model in enumerate((assets.left_hand, assets.right_hand)):
        prefix = "hand_l" if physical == 0 else "hand_r"
        fk = official.forward_kinematics(
            model, dict(zip(names[physical], q_hand[physical], strict=True))
        )
        links = (
            None,
            f"{prefix}_thumb_link1",
            f"{prefix}_index_link1",
            f"{prefix}_middle_link1",
            f"{prefix}_pinky_link1",
            f"{prefix}_thumb_link4",
            f"{prefix}_index_link4",
            f"{prefix}_middle_link4",
            f"{prefix}_ring_link4",
            f"{prefix}_pinky_link4",
        )
        output.append(
            np.asarray(
                [np.zeros(3) if link is None else fk[link][:3, 3] for link in links],
                dtype=np.float64,
            )
        )
    return output[0], output[1]


def choose_chirality_and_camera(
    assets: Any,
    arm_fk: dict[str, np.ndarray],
    hand_points: tuple[np.ndarray, np.ndarray],
    hawor: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One bounded 24x24 handed-axis audit and one fixed frame-0 PnP."""
    human_indices = np.asarray((0, 2, 5, 9, 17, 4, 8, 12, 16, 20))
    images = tuple(
        np.asarray(hawor["joints_2d"][PHYSICAL_TO_HUMAN[p], 0, human_indices], dtype=np.float64)
        for p in range(2)
    )
    candidates = proper_axis_rotations()
    rows = []
    intrinsics = np.asarray(hawor["intrinsics"][0], dtype=np.float64)
    for left_index, left in enumerate(candidates):
        for right_index, right in enumerate(candidates):
            rotations = (left, right)
            robot_sets = []
            for physical in range(2):
                tool = arm_fk[f"{SIDES[physical]}_tool"]
                root = tool.copy()
                root[:3, :3] = root[:3, :3] @ rotations[physical]
                robot_sets.append(
                    (root[:3, :3] @ hand_points[physical].T).T + root[:3, 3]
                )
            robot = np.vstack(robot_sets).astype(np.float64)
            image = np.vstack(images).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                robot, image, intrinsics, None, flags=cv2.SOLVEPNP_EPNP
            )
            if not ok:
                continue
            ok, rvec, tvec = cv2.solvePnP(
                robot, image, intrinsics, None, rvec, tvec, True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            camera_rotation = cv2.Rodrigues(rvec)[0]
            camera_points = (camera_rotation @ robot.T + tvec).T
            if float(np.min(camera_points[:, 2])) <= 0.01:
                continue
            projected = cv2.projectPoints(robot, rvec, tvec, intrinsics, None)[0][:, 0]
            rmse = float(np.sqrt(np.mean(np.sum((projected - image) ** 2, axis=1))))
            thumb_angles, palm_winding = [], []
            for physical in range(2):
                begin = physical * 10
                robot_image = projected[begin:begin + 10]
                human_image = images[physical]
                thumb_angles.append(
                    angle_deg(robot_image[1] - robot_image[0], human_image[1] - human_image[0])
                )
                robot_cross = float(np.cross(robot_image[2] - robot_image[0], robot_image[4] - robot_image[0]))
                human_cross = float(np.cross(human_image[2] - human_image[0], human_image[4] - human_image[0]))
                palm_winding.append(bool(robot_cross * human_cross > 0.0))
            score = rmse + 0.15 * sum(thumb_angles) + 36.0 * sum(not x for x in palm_winding)
            rows.append({
                "score": float(score),
                "frame0_reprojection_rmse_px": rmse,
                "thumb_root_screen_angle_deg": thumb_angles,
                "palm_screen_winding_match": palm_winding,
                "candidate_indices": [left_index, right_index],
                "rotations": [left.tolist(), right.tolist()],
                "rvec": np.asarray(rvec).reshape(3).tolist(),
                "tvec": np.asarray(tvec).reshape(3).tolist(),
            })
    if not rows:
        raise SuccessorError("no positive-depth chirality/camera candidate")
    rows.sort(key=lambda row: row["score"])
    selected = rows[0]
    mounts = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    mounts[:, :3, :3] = np.asarray(selected["rotations"], dtype=np.float64)
    camera_base = np.eye(4, dtype=np.float64)
    camera_base[:3, :3] = cv2.Rodrigues(np.asarray(selected["rvec"], dtype=np.float64))[0]
    camera_base[:3, 3] = np.asarray(selected["tvec"], dtype=np.float64)
    audit = {
        "method": "BOUNDED_24_X_24_PROPER_SIGNED_AXIS_ENUMERATION_THEN_SINGLE_FRAME0_PNP",
        "candidate_count_total": 576,
        "positive_depth_candidate_count": len(rows),
        "selection_score": "pnp_rmse_px + 0.15*sum(thumb_root_screen_angle_deg) + 36*screen_winding_mismatch_count",
        "selected": selected,
        "top5": rows[:5],
        "matrix_direction": "p_tool = T_tool_hand_root_proxy @ p_hand_root",
        "camera_semantics": "p_camera = T_camera_base_proxy @ p_base",
        "camera_claim": "FRAME0_PNP_VISUAL_PROXY_NOT_MEASUREMENT_NOT_CALIBRATION",
    }
    return mounts, camera_base, audit


def mano_pose(points: np.ndarray, handedness: str) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = adapter.final_v3_mano_palm_basis(points, handedness=handedness)
    value[:3, 3] = points[0]
    return value


def solve_arms(
    assets: Any,
    hawor: dict[str, np.ndarray],
    mounts: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neutral = 0.5 * (lower + upper)
    neutral_values = {name: 0.0 for row in official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        neutral_values.update(dict(zip(official.ARM_JOINT_NAMES[physical], neutral[physical], strict=True)))
    neutral_fk = official.forward_kinematics(assets.tianji, neutral_values)
    world = np.asarray(hawor["joints_3d_world"], dtype=np.float64)
    human = np.empty((24, 2, 4, 4), dtype=np.float64)
    targets = np.empty_like(human)
    for slot in range(24):
        for physical in range(2):
            human_side = PHYSICAL_TO_HUMAN[physical]
            human[slot, physical] = mano_pose(world[human_side, slot], SIDES[human_side])
    for physical in range(2):
        root0 = neutral_fk[f"{SIDES[physical]}_tool"] @ mounts[physical]
        fixed_human_to_robot = root0 @ np.linalg.inv(human[0, physical])
        for slot in range(24):
            targets[slot, physical] = fixed_human_to_robot @ human[slot, physical]
    output = np.empty((24, 2, 7), dtype=np.float64)
    output[0] = neutral
    for slot in range(1, 24):
        for physical in range(2):
            lo = np.maximum(lower[physical], output[slot - 1, physical] - ARM_EFFECTIVE_STEP)
            hi = np.minimum(upper[physical], output[slot - 1, physical] + ARM_EFFECTIVE_STEP)
            if slot >= 2:
                center = 2.0 * output[slot - 1, physical] - output[slot - 2, physical]
                lo = np.maximum(lo, center - SECOND_DIFFERENCE)
                hi = np.minimum(hi, center + SECOND_DIFFERENCE)
            if np.any(lo > hi + 1e-12):
                raise SuccessorError(f"empty arm temporal feasible set at slot {slot}")
            target_tool = targets[slot, physical] @ np.linalg.inv(mounts[physical])
            seed = np.clip(output[slot - 1, physical], lo, hi)
            position, _ = official._solve_one_arm_position_only(
                assets, side=physical, base=np.eye(4), target_tool=target_tool,
                initial_q=seed, lower=lo, upper=hi,
            )
            solved, _ = official._solve_one_arm(
                assets, side=physical, base=np.eye(4), target_tool=target_tool,
                initial_q=position, lower=lo, upper=hi,
            )
            output[slot, physical] = solved
    return output, targets, neutral_fk


def annulus_triangles(transform: np.ndarray) -> np.ndarray:
    mesh = trimesh.creation.annulus(
        r_min=FLANGE_RING_INNER_RADIUS_M,
        r_max=FLANGE_RING_OUTER_RADIUS_M,
        height=FLANGE_RING_LENGTH_M,
        sections=32,
    )
    local = np.asarray(mesh.vertices, dtype=np.float64)
    world = (transform[:3, :3] @ local.T).T + transform[:3, 3]
    return world[np.asarray(mesh.faces, dtype=np.int64)]


def add_mesh(
    geometry: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    triangles: np.ndarray,
    color: tuple[int, int, int],
    label: int,
) -> None:
    if len(triangles):
        geometry.append((
            triangles,
            np.repeat(np.asarray(color, dtype=np.uint8)[None], len(triangles), axis=0),
            np.full(len(triangles), label, dtype=np.int32),
        ))


def render_scene(
    raster: Any,
    assets: Any,
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    mounts: np.ndarray,
    camera_base: np.ndarray,
    intrinsics: np.ndarray,
    cache: dict[Path, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    values = {name: 0.0 for row in official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(dict(zip(official.ARM_JOINT_NAMES[physical], q_arm[physical], strict=True)))
    arm_fk = official.forward_kinematics(assets.tianji, values)
    geometry: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    arm = raster.camera_triangles(assets.tianji, arm_fk, camera_base, lambda _link: True, cache)
    geometry.append((arm[0], tint(arm[1], (205, 205, 205)), arm[2]))
    hand_roots, hand_fks = [], []
    for physical, model in enumerate((assets.left_hand, assets.right_hand)):
        tool = arm_fk[f"{SIDES[physical]}_tool"]
        hand_root_base = tool @ mounts[physical]
        hand_root_camera = camera_base @ hand_root_base
        names = moving_joint_names(model)
        hand_fk = official.forward_kinematics(model, dict(zip(names, q_hand[physical], strict=True)))
        hand = raster.camera_triangles(model, hand_fk, hand_root_camera, lambda _link: True, cache)
        hand_color = (240, 145, 45) if physical == 0 else (205, 80, 210)
        geometry.append((hand[0], tint(hand[1], hand_color), hand[2] + 2000 + physical * 200))
        hand_roots.append(hand_root_camera)
        hand_fks.append(hand_fk)
        flange = arm_fk[f"flange_{'L' if physical == 0 else 'R'}"]
        ring_center = np.eye(4, dtype=np.float64)
        ring_center[2, 3] = FLANGE_RING_LENGTH_M * 0.5
        ring_camera = camera_base @ flange @ ring_center
        add_mesh(geometry, annulus_triangles(ring_camera), (30, 165, 255), 4000 + physical)
    triangles = np.concatenate([item[0] for item in geometry if len(item[0])])
    colors = np.concatenate([item[1] for item in geometry if len(item[0])])
    labels = np.concatenate([item[2] for item in geometry if len(item[0])])
    depth, color, label = raster.rasterize_zbuffer(
        triangles, colors, labels,
        float(intrinsics[0, 0]), float(intrinsics[1, 1]),
        float(intrinsics[0, 2]), float(intrinsics[1, 2]), WIDTH, HEIGHT,
    )
    return color, label, {"arm_fk": arm_fk, "hand_roots": np.asarray(hand_roots), "hand_fks": hand_fks, "depth": depth}


def look_at_camera() -> np.ndarray:
    # Front view keeps the physical y-mirror relation visually inspectable.
    eye = np.asarray((3.15, 0.0, 1.65), dtype=np.float64)
    target = np.asarray((0.05, 0.0, 0.90), dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = -rotation @ eye
    return value


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = points[:, 2] > 1e-4
    result[valid, 0] = intrinsics[0, 0] * points[valid, 0] / points[valid, 2] + intrinsics[0, 2]
    result[valid, 1] = intrinsics[1, 1] * points[valid, 1] / points[valid, 2] + intrinsics[1, 2]
    return result


def draw_human_and_robot_hands(
    panel: np.ndarray,
    human_uv: np.ndarray,
    robot_uv: list[np.ndarray],
    tip_errors: list[list[float]],
) -> None:
    for human_side in range(2):
        points = human_uv[human_side]
        for chain in MANO_DRAW_CHAINS:
            for a, b in zip(chain[:-1], chain[1:], strict=True):
                cv2.line(panel, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), (70, 245, 80), 2, cv2.LINE_AA)
    colors = ((255, 170, 40), (225, 70, 225))
    for physical in range(2):
        points = robot_uv[physical]
        if np.isfinite(points).all():
            cv2.polylines(panel, [np.rint(points).astype(np.int32)], False, colors[physical], 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(panel, tuple(np.rint(point).astype(int)), 5, colors[physical], -1, cv2.LINE_AA)
        values = tip_errors[physical]
        text = "/".join(f"{v:.0f}" for v in values)
        cv2.putText(panel, f"P{physical} tips mm {text}", (12, 350 + physical * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[physical], 1, cv2.LINE_AA)


def normalize_hand_uv(points: np.ndarray, physical: int) -> np.ndarray:
    """Place one human hand in its own review cell without reflection."""
    value = np.asarray(points, dtype=np.float64)
    low, high = np.min(value, axis=0), np.max(value, axis=0)
    extent = max(float(np.max(high - low)), 1.0)
    center = np.asarray((320.0, 198.0 if physical == 0 else 386.0))
    return (value - 0.5 * (low + high)) * (150.0 / extent) + center


def render_local_hand_comparison(
    raster: Any,
    assets: Any,
    q_hand: np.ndarray,
    human_uv: np.ndarray,
    tip_errors: list[list[float]],
    cache: dict[Path, Any],
) -> np.ndarray:
    """Two independent, proper-PnP hand-shape views in one panel."""
    models = (assets.left_hand, assets.right_hand)
    names = tuple(moving_joint_names(model) for model in models)
    landmarks = hand_landmarks(assets, names, q_hand)
    selected_mano = np.asarray((0, 2, 5, 9, 17, 4, 8, 12, 16, 20))
    local_k = np.asarray(((600.0, 0.0, 319.5), (0.0, 600.0, 239.5), (0.0, 0.0, 1.0)))
    geometry = []
    normalized_humans = []
    projected_robots = []
    for physical, model in enumerate(models):
        human_side = PHYSICAL_TO_HUMAN[physical]
        normalized = normalize_hand_uv(human_uv[human_side], physical)
        normalized_humans.append(normalized)
        ok, rvec, tvec = cv2.solvePnP(
            landmarks[physical], normalized[selected_mano], local_k, None,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            raise SuccessorError(f"local hand PnP failed for physical side {physical}")
        ok, rvec, tvec = cv2.solvePnP(
            landmarks[physical], normalized[selected_mano], local_k, None,
            rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise SuccessorError(f"local hand PnP refine failed for physical side {physical}")
        root = np.eye(4, dtype=np.float64)
        root[:3, :3] = cv2.Rodrigues(rvec)[0]
        root[:3, 3] = np.asarray(tvec).reshape(3)
        fk = official.forward_kinematics(model, dict(zip(names[physical], q_hand[physical], strict=True)))
        rendered = raster.camera_triangles(model, fk, root, lambda _link: True, cache)
        color = (240, 145, 45) if physical == 0 else (205, 80, 210)
        geometry.append((rendered[0], tint(rendered[1], color), rendered[2] + 5000 + 200 * physical))
        projected = cv2.projectPoints(landmarks[physical], rvec, tvec, local_k, None)[0][:, 0]
        projected_robots.append(projected)
    triangles = np.concatenate([row[0] for row in geometry if len(row[0])])
    colors = np.concatenate([row[1] for row in geometry if len(row[0])])
    labels = np.concatenate([row[2] for row in geometry if len(row[0])])
    _, rendered_color, rendered_label = raster.rasterize_zbuffer(
        triangles, colors, labels, 600.0, 600.0, 319.5, 239.5, WIDTH, HEIGHT
    )
    panel = np.full((HEIGHT, WIDTH, 3), 22, dtype=np.uint8)
    visible = rendered_label >= 0
    panel[visible] = rendered_color[visible]
    robot_colors = ((255, 170, 40), (225, 70, 225))
    for physical in range(2):
        normalized = normalized_humans[physical]
        for chain in MANO_DRAW_CHAINS:
            for a, b in zip(chain[:-1], chain[1:], strict=True):
                cv2.line(panel, tuple(np.rint(normalized[a]).astype(int)), tuple(np.rint(normalized[b]).astype(int)), (70, 245, 80), 2, cv2.LINE_AA)
        robot_points = projected_robots[physical]
        cv2.line(panel, tuple(np.rint(robot_points[0]).astype(int)), tuple(np.rint(robot_points[1]).astype(int)), robot_colors[physical], 2, cv2.LINE_AA)
        for point in robot_points[5:]:
            cv2.line(panel, tuple(np.rint(robot_points[0]).astype(int)), tuple(np.rint(point).astype(int)), robot_colors[physical], 1, cv2.LINE_AA)
            cv2.circle(panel, tuple(np.rint(point).astype(int)), 4, robot_colors[physical], -1, cv2.LINE_AA)
        errors = "/".join(f"{value:.0f}" for value in tip_errors[physical])
        y = 126 if physical == 0 else 314
        cv2.putText(panel, f"P{physical} tips mm {errors}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, robot_colors[physical], 1, cv2.LINE_AA)
    return panel


def title(panel: np.ndarray, heading: str, line1: str, line2: str) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 105), fill=(5, 5, 5))
    draw.text((10, 5), heading, font=font(19), fill=(255, 255, 255))
    draw.text((10, 34), line1, font=font(15), fill=(255, 180, 45))
    draw.text((10, 59), line2, font=font(15), fill=(255, 105, 105))
    draw.text((10, 84), "MOUNT/CAMERA_PROXY｜非实测｜非标定｜不可部署", font=font(14), fill=(255, 105, 105))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--mask-review", type=Path, required=True)
    parser.add_argument("--object6d", type=Path, required=True)
    parser.add_argument("--object6d-review", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SuccessorError(f"refusing to overwrite: {args.output_dir}")
    require_review(args.mask_review, "Mask")
    require_review(args.object6d_review, "Object6D")
    hawor_result = load_json(args.hawor_result)
    if hawor_result.get("session_id") != SESSION:
        raise SuccessorError("HaWoR RESULT session mismatch")
    hawor = arrays(args.hawor)
    if not (
        np.array_equal(hawor["original_frame_indices"][:24], np.arange(24))
        and np.all(hawor["observed"][:, :24])
        and np.all(hawor["provenance"][:, :24] == "BOUNDED_PARAMETER_FIT")
        and hawor_result["outputs"]["npz"]["sha256"] == sha256(args.hawor)
    ):
        raise SuccessorError("HaWoR first24 identity/provenance/SHA mismatch")
    mask_manifest = load_json(args.mask_manifest)
    if mask_manifest.get("session") != SESSION or len(mask_manifest.get("frames", [])) != 171:
        raise SuccessorError("Mask manifest identity/frame count mismatch")
    object6d = arrays(args.object6d)
    if not np.array_equal(object6d["frame_indices"], np.arange(171)):
        raise SuccessorError("Object6D frame mapping mismatch")

    assets = load_pinned_robot_assets(PROJECT)
    arm_lower, arm_upper = official._arm_limits(assets)
    hand_models = (assets.left_hand, assets.right_hand)
    hand_names = tuple(moving_joint_names(model) for model in hand_models)
    hand_lower = np.asarray([[j.lower for j in m.joints if j.joint_type != "fixed"] for m in hand_models])
    hand_upper = np.asarray([[j.upper for j in m.joints if j.joint_type != "fixed"] for m in hand_models])
    raw_hand, retarget_rows, returned_names = solve_external(hawor, np.arange(24))
    if returned_names != hand_names or raw_hand.shape != (24, 2, 22):
        raise SuccessorError("retargeting_human joint identity/shape mismatch")
    q_hand = bounded_projection(raw_hand, hand_lower, hand_upper)

    neutral = 0.5 * (arm_lower + arm_upper)
    neutral_values = {name: 0.0 for row in official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        neutral_values.update(dict(zip(official.ARM_JOINT_NAMES[physical], neutral[physical], strict=True)))
    neutral_fk = official.forward_kinematics(assets.tianji, neutral_values)
    landmarks = hand_landmarks(assets, hand_names, q_hand[0])
    mounts, camera_base_proxy, chirality_audit = choose_chirality_and_camera(
        assets, neutral_fk, landmarks, hawor
    )
    q_arm, hand_targets, neutral_fk = solve_arms(assets, hawor, mounts, arm_lower, arm_upper)
    arm_time = temporal_metrics(q_arm)
    hand_time = temporal_metrics(q_hand)

    neutral_tool = np.asarray([neutral_fk[f"{side}_tool"][:3, 3] for side in SIDES])
    neutral_flange = np.asarray([neutral_fk[f"flange_{'L' if p == 0 else 'R'}"][:3, 3] for p in range(2)])
    neutral_mirror_error = float(max(
        np.linalg.norm(neutral_tool[0] - neutral_tool[1] * np.asarray((1, -1, 1))),
        np.linalg.norm(neutral_flange[0] - neutral_flange[1] * np.asarray((1, -1, 1))),
    ))
    hard_gates = {
        "frame_identity": True,
        "human_left_to_physical_right_human_right_to_physical_left": True,
        "finite": bool(all(np.isfinite(x).all() for x in (q_arm, q_hand, mounts, camera_base_proxy))),
        "symmetric_neutral_q0": bool(np.allclose(q_arm[0, 0], q_arm[0, 1], atol=1e-12)),
        "symmetric_neutral_fk_q0": bool(neutral_mirror_error <= 2e-6),
        "arm_urdf_limits": bool(np.all(q_arm >= arm_lower - 1e-9) and np.all(q_arm <= arm_upper + 1e-9)),
        "hand_urdf_limits": bool(np.all(q_hand >= hand_lower - 1e-9) and np.all(q_hand <= hand_upper + 1e-9)),
        "arm_step": bool(arm_time["step_max_rad_per_frame"] <= 0.12 + 1e-9),
        "hand_step": bool(hand_time["step_max_rad_per_frame"] <= 0.08 + 1e-9),
        "arm_second_difference": bool(arm_time["second_difference_max_rad_per_frame2"] <= SECOND_DIFFERENCE + 1e-9),
        "hand_second_difference": bool(hand_time["second_difference_max_rad_per_frame2"] <= SECOND_DIFFERENCE + 1e-9),
        "chirality_proper_rotations": bool(np.allclose(np.linalg.det(mounts[:, :3, :3]), 1.0, atol=1e-9)),
        "frame0_palm_screen_winding": bool(all(chirality_audit["selected"]["palm_screen_winding_match"])),
        "frame0_thumb_screen_direction": bool(max(chirality_audit["selected"]["thumb_root_screen_angle_deg"]) <= 45.0),
    }
    if not all(hard_gates.values()):
        raise SuccessorError(f"pre-render hard gate failed: {hard_gates}")

    args.output_dir.mkdir(parents=True)
    frame_root = args.output_dir / "frames"
    frame_root.mkdir()
    raster = load_module(RASTER_SOURCE, "poker_symmetric_chirality_flange_raster")
    cache: dict[Path, Any] = {}
    full_camera = look_at_camera()
    full_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    frame_rows = []
    key_images = []
    for slot in range(24):
        source = int(hawor["original_frame_indices"][slot])
        raw_path = args.raw_root / f"{source:05d}" / "rgb.png"
        source_record = mask_manifest["frames"][source]["source_rgb"]
        if Path(source_record["path"]).resolve() != raw_path.resolve() or sha256(raw_path) != source_record["sha256"]:
            raise SuccessorError(f"raw/Mask frame lineage mismatch at {source}")
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise SuccessorError(f"raw decode failed at {source}")
        source_h, source_w = raw.shape[:2]
        raw = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        k = np.asarray(hawor["intrinsics"][slot], dtype=np.float64).copy()
        k[0] *= WIDTH / source_w
        k[1] *= HEIGHT / source_h
        ego_color, ego_label, ego_aux = render_scene(
            raster, assets, q_arm[slot], q_hand[slot], mounts, camera_base_proxy, k, cache
        )
        visible = ego_label >= 0
        overlay = raw.copy()
        overlay[visible] = np.clip(0.18 * raw[visible] + 0.82 * ego_color[visible], 0, 255).astype(np.uint8)
        left = title(
            overlay, f"原RGB + Robot投影｜source {source:05d}",
            "完整global base/双臂/法兰环/KaiHand", "CAMERA_PROXY_FRAME0_PNP｜CONTACT_PENDING",
        )
        full_color, full_label, _ = render_scene(
            raster, assets, q_arm[slot], q_hand[slot], mounts, full_camera, full_k, cache
        )
        middle = np.full_like(full_color, 242)
        full_visible = full_label >= 0
        middle[full_visible] = full_color[full_visible]
        middle = title(
            middle, "完整Tianji全局结构与同步动作",
            "对称中性构型起步｜previous-only", "FLANGE_RING_PROXY 145mm｜非真实CAD",
        )
        human_uv = np.asarray(hawor["joints_2d"][:, slot], dtype=np.float64).copy()
        human_uv[..., 0] *= WIDTH / source_w
        human_uv[..., 1] *= HEIGHT / source_h
        tip_errors = [[float(x) for x in retarget_rows[slot][p]["tip_error_mm"]] for p in range(2)]
        local = render_local_hand_comparison(
            raster, assets, q_hand[slot], human_uv, tip_errors, cache
        )
        local = title(
            local, "左右手分区局部核对｜绿=Human MANO",
            "橙=P0左Kai←人右｜紫=P1右Kai←人左", "retargeting_human bounded IK｜误差单位mm",
        )
        combined = np.hstack((left, middle, local))
        if not cv2.imwrite(str(frame_root / f"{slot:06d}.png"), combined):
            raise SuccessorError("frame write failed")
        if slot in KEY_SLOTS:
            key_images.append(combined)
        frame_rows.append({
            "slot": slot,
            "source_frame": source,
            "complete_tianji_visual_pixels": int(np.count_nonzero(visible)),
            "full_view_visual_pixels": int(np.count_nonzero(full_visible)),
            "object6d_valid_bound_for_visual_context": bool(object6d["valid"][source]),
            "retarget_tip_error_mm": tip_errors,
        })

    video = args.output_dir / "POKER_对称中性_法兰环代理_手型共轭_24帧三栏中文复核.mp4"
    decoded = encode_video(frame_root, video, args.fps)
    sheet = args.output_dir / "POKER_关键帧0_8_15_23_自动复核.png"
    if not cv2.imwrite(str(sheet), np.vstack(key_images)):
        raise SuccessorError("key-frame sheet write failed")
    hard_gates["video_decode_24_of_24"] = decoded == 24
    hard_gates["all_three_views_nonempty"] = all(
        row["complete_tianji_visual_pixels"] > 0 and row["full_view_visual_pixels"] > 0
        for row in frame_rows
    )
    status = "READY_FOR_USER_24FRAME_DIRECTED_SUCCESSOR_REVIEW" if all(hard_gates.values()) else "TERMINAL_C_AUTOMATIC_REVIEW_FAILED"
    grade = "B_DEVELOPMENT_ONLY" if all(hard_gates.values()) else "C"
    chirality_path = args.output_dir / "CHIRALITY_AUDIT.json"
    chirality_path.write_text(json.dumps(chirality_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    states = args.output_dir / "ROBOT_STATES.npz"
    np.savez_compressed(
        states, source_frames=np.arange(24), q_arm=q_arm, q_hand=q_hand,
        T_tool_hand_root_proxy=mounts, T_camera_base_proxy=camera_base_proxy,
        T_hand_target_base=hand_targets, human_to_physical=np.asarray((1, 0)),
    )
    frame_manifest = args.output_dir / "FRAME_MANIFEST.json"
    frame_manifest.write_text(json.dumps({
        "schema_version": "poker-symmetric-chirality-flange-successor-frame-manifest-v1",
        "session": SESSION, "frame_count": 24, "key_review_slots": list(KEY_SLOTS),
        "frames": frame_rows,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": "poker-symmetric-chirality-flange-successor-result-v1",
        "status": status,
        "grade": grade,
        "session": SESSION,
        "frame_count": 24,
        "downstream_authorized": False,
        "current_robot_authority": False,
        "development_only": True,
        "deployment_authorized": False,
        "training_authorized": False,
        "contact_status": "CONTACT_PENDING",
        "mount_status": "FLANGE_RING_PROXY_DIMENSIONED_FROM_URDF_AND_LINK7_AABB_NOT_REAL_CAD",
        "camera_status": "FRAME0_PNP_VISUAL_PROXY_NOT_CALIBRATION",
        "human_to_physical": {"left": "right", "right": "left"},
        "initial_configuration": "TIANJI_URDF_SYMMETRIC_JOINT_LIMIT_MIDPOINT",
        "task_motion": "FRAME0_NEUTRAL_THEN_RELATIVE_HUMAN_MOTION_PREVIOUS_ACCEPTED_ONLY",
        "hard_gates": hard_gates,
        "metrics": {
            "neutral_mirror_error_m": neutral_mirror_error,
            "arm": arm_time,
            "hand": hand_time,
            "frame0_reprojection_rmse_px_not_ik_accuracy": chirality_audit["selected"]["frame0_reprojection_rmse_px"],
            "frame0_thumb_root_screen_angle_deg": chirality_audit["selected"]["thumb_root_screen_angle_deg"],
            "frame0_palm_screen_winding_match": chirality_audit["selected"]["palm_screen_winding_match"],
        },
        "flange_ring_proxy": {
            "length_m": FLANGE_RING_LENGTH_M,
            "length_source": "Tianji URDF flange_L/R to left/right_tool fixed joint xyz z=0.145m",
            "outer_radius_m": FLANGE_RING_OUTER_RADIUS_M,
            "outer_radius_source": "half of measured Link7 mesh local x AABB 0.08200014m",
            "inner_radius_m": FLANGE_RING_INNER_RADIUS_M,
            "inner_radius_source": "visualization design choice 0.5*outer_radius; not measured hardware",
            "same_geometry_both_sides": True,
            "real_cad": False,
        },
        "inputs": {
            "producer": artifact(Path(__file__)),
            "hawor": artifact(args.hawor),
            "hawor_result": artifact(args.hawor_result),
            "mask_manifest": artifact(args.mask_manifest),
            "mask_review": artifact(args.mask_review),
            "object6d": artifact(args.object6d),
            "object6d_review": artifact(args.object6d_review),
            "tianji_urdf": artifact(assets.tianji.path),
            "kaihand_left_urdf": artifact(KAI_URDFS[0]),
            "kaihand_right_urdf": artifact(KAI_URDFS[1]),
            "retargeting_human_retargeter": artifact(EXTERNAL_SOURCES[0]),
            "retargeting_human_geometry": artifact(EXTERNAL_SOURCES[1]),
        },
        "outputs": {
            "video": artifact(video), "keyframe_sheet": artifact(sheet),
            "states": artifact(states), "frame_manifest": artifact(frame_manifest),
            "chirality_audit": artifact(chirality_path),
        },
        "claim_limit": "One 24-frame visual successor only. FLANGE_RING_PROXY and CAMERA_PROXY are not real CAD, measurement, calibration, collision/contact validation, IK accuracy, deployment, training, or Robot authority.",
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    review = {
        "schema_version": "baseline-agent-stage-review-v1",
        "session": SESSION,
        "stage": "ROBOT_DEVELOPMENT_VISUAL_SUCCESSOR",
        "grade": "B" if grade.startswith("B") else "C",
        "downstream_authorized": False,
        "reviewed_frames": list(KEY_SLOTS),
        "automatic_hard_gates": hard_gates,
        "visual_review_status": "PENDING_AGENT_KEYFRAME_INSPECTION",
        "result_sha256": sha256(result_path),
        "claim_limit": result["claim_limit"],
    }
    review_path = args.output_dir / "AGENT_REVIEW.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "video": str(video), "sheet": str(sheet)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
