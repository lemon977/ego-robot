#!/usr/bin/env python3
"""Render the sole V2C03/P3 Poker correction requested after visual rejection.

This development-only review deliberately reuses the historical user-retained
camera/base transform and arm posture.  It does not estimate a camera pose from
the task video.  The NaturalV2 flange is rendered at its original STL scale and
the KaiHand root uses the bounded historical root-plane geometric transform.
The missing physical Tianji-to-Kai adapter remains an explicit claim limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import render_poker_symmetric_chirality_flange_successor as old


SESSION = old.SESSION
SIDES = old.SIDES
PHYSICAL_TO_HUMAN = old.PHYSICAL_TO_HUMAN
WIDTH, HEIGHT = old.WIDTH, old.HEIGHT
KEY_SLOTS = old.KEY_SLOTS
M6_ARM_LINKS = frozenset(
    [
        *(f"{prefix}_L" for prefix in ("Base", *(f"Link{i}" for i in range(1, 8)))),
        *(f"{prefix}_R" for prefix in ("Base", *(f"Link{i}" for i in range(1, 8)))),
    ]
)
V2C03_P3_STATE = (
    PROJECT
    / "archive/legacy_runs/incomplete/"
    "gpt_robot_static_posture_ab_cycles_scenes_20260831_v1/A/"
    "SCENE_STATE_STATIC_POSTURE.npz"
)
ROOTPLANE_PREFLIGHT = (
    PROJECT
    / "_run/gpt_robot_kaihand_rootplane_mount_cpu_successor_20260831_v1/"
    "CPU_PREFLIGHT.json"
)
NATURALV2 = PROJECT / "assets/robot/V2相机固定法兰PRO版本.STL"
NATURALV2_CENTER_XY_MM = np.asarray((51.62780570983887, 51.30149126052856))
NATURALV2_MATING_Z_MM = 1.612299


class FixedSuccessorError(RuntimeError):
    pass


def load_fixed_lineage() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    state = old.arrays(V2C03_P3_STATE)
    if not (
        str(state["base_name"]) == "V2C03"
        and str(state["pose_id"]) == "P3"
        and np.allclose(state["joint4_rad"], -2.2, atol=0, rtol=0)
    ):
        raise FixedSuccessorError("historical V2C03/P3 state identity drift")
    camera = np.asarray(state["T_camera_base"], dtype=np.float64)
    q0 = np.asarray(state["q_arm"][0], dtype=np.float64)
    preflight = old.load_json(ROOTPLANE_PREFLIGHT)
    sides = preflight.get("sides", {})
    transforms = [
        sides.get("left", {}).get("T_tool_hand_exact"),
        sides.get("right", {}).get("T_tool_hand_exact"),
    ]
    if any(value is None for value in transforms):
        # Fail closed; callers must not silently fall back to the rejected mount.
        raise FixedSuccessorError("root-plane preflight transform field not found")
    mounts = np.asarray(transforms, dtype=np.float64)
    if mounts.shape != (2, 4, 4):
        raise FixedSuccessorError("root-plane mount shape drift")
    if not np.allclose(np.linalg.det(mounts[:, :3, :3]), 1.0, atol=1e-9):
        raise FixedSuccessorError("root-plane mount is not proper SE(3)")
    return camera, q0, mounts, preflight


def solve_arms_from_p3(
    assets: Any,
    hawor: dict[str, np.ndarray],
    mounts: np.ndarray,
    q0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if q0.shape != (2, 7) or np.any(q0 < lower) or np.any(q0 > upper):
        raise FixedSuccessorError("historical P3 outside current Tianji URDF limits")
    initial_values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        initial_values.update(
            dict(zip(old.official.ARM_JOINT_NAMES[physical], q0[physical], strict=True))
        )
    initial_fk = old.official.forward_kinematics(assets.tianji, initial_values)
    world = np.asarray(hawor["joints_3d_world"], dtype=np.float64)
    human = np.empty((24, 2, 4, 4), dtype=np.float64)
    targets = np.empty_like(human)
    for slot in range(24):
        for physical in range(2):
            human_side = PHYSICAL_TO_HUMAN[physical]
            human[slot, physical] = old.mano_pose(
                world[human_side, slot], SIDES[human_side]
            )
    for physical in range(2):
        root0 = initial_fk[f"{SIDES[physical]}_tool"] @ mounts[physical]
        human_to_robot = root0 @ np.linalg.inv(human[0, physical])
        for slot in range(24):
            targets[slot, physical] = human_to_robot @ human[slot, physical]

    output = np.empty((24, 2, 7), dtype=np.float64)
    output[0] = q0
    for slot in range(1, 24):
        for physical in range(2):
            lo = np.maximum(lower[physical], output[slot - 1, physical] - old.ARM_EFFECTIVE_STEP)
            hi = np.minimum(upper[physical], output[slot - 1, physical] + old.ARM_EFFECTIVE_STEP)
            if slot >= 2:
                center = 2.0 * output[slot - 1, physical] - output[slot - 2, physical]
                lo = np.maximum(lo, center - old.SECOND_DIFFERENCE)
                hi = np.minimum(hi, center + old.SECOND_DIFFERENCE)
            if np.any(lo > hi + 1e-12):
                raise FixedSuccessorError(f"empty arm temporal feasible set at slot {slot}")
            target_tool = targets[slot, physical] @ np.linalg.inv(mounts[physical])
            seed = np.clip(output[slot - 1, physical], lo, hi)
            position, _ = old.official._solve_one_arm_position_only(
                assets,
                side=physical,
                base=np.eye(4),
                target_tool=target_tool,
                initial_q=seed,
                lower=lo,
                upper=hi,
            )
            solved, _ = old.official._solve_one_arm(
                assets,
                side=physical,
                base=np.eye(4),
                target_tool=target_tool,
                initial_q=position,
                lower=lo,
                upper=hi,
            )
            output[slot, physical] = solved
    return output, targets, initial_fk


def naturalv2_local_triangles() -> np.ndarray:
    mesh = trimesh.load_mesh(NATURALV2, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices[:, :2] -= NATURALV2_CENTER_XY_MM
    vertices[:, 2] -= NATURALV2_MATING_Z_MM
    vertices *= 0.001
    return vertices[np.asarray(mesh.faces, dtype=np.int64)]


def transform_triangles(triangles: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return triangles @ transform[:3, :3].T + transform[:3, 3]


def render_scene(
    raster: Any,
    assets: Any,
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    mounts: np.ndarray,
    camera_base: np.ndarray,
    intrinsics: np.ndarray,
    cache: dict[Path, Any],
    flange_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = {name: 0.0 for row in old.official.ARM_JOINT_NAMES for name in row}
    for physical in range(2):
        values.update(
            dict(zip(old.official.ARM_JOINT_NAMES[physical], q_arm[physical], strict=True))
        )
    arm_fk = old.official.forward_kinematics(assets.tianji, values)
    geometry: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    arm = raster.camera_triangles(
        assets.tianji,
        arm_fk,
        camera_base,
        lambda link: link in M6_ARM_LINKS,
        cache,
    )
    geometry.append((arm[0], old.tint(arm[1], (205, 205, 205)), arm[2]))
    hand_roots = []
    for physical, model in enumerate((assets.left_hand, assets.right_hand)):
        tool = arm_fk[f"{SIDES[physical]}_tool"]
        hand_root_base = tool @ mounts[physical]
        hand_root_camera = camera_base @ hand_root_base
        names = old.moving_joint_names(model)
        hand_fk = old.official.forward_kinematics(
            model, dict(zip(names, q_hand[physical], strict=True))
        )
        hand = raster.camera_triangles(
            model, hand_fk, hand_root_camera, lambda _link: True, cache
        )
        hand_color = (240, 145, 45) if physical == 0 else (205, 80, 210)
        geometry.append(
            (hand[0], old.tint(hand[1], hand_color), hand[2] + 2000 + physical * 200)
        )
        hand_roots.append(hand_root_camera)
        suffix = "L" if physical == 0 else "R"
        flange_camera = camera_base @ arm_fk[f"flange_{suffix}"]
        flange = transform_triangles(flange_local, flange_camera)
        old.add_mesh(geometry, flange, (216, 226, 238), 4000 + physical)
    triangles = np.concatenate([item[0] for item in geometry if len(item[0])])
    colors = np.concatenate([item[1] for item in geometry if len(item[0])])
    labels = np.concatenate([item[2] for item in geometry if len(item[0])])
    depth, color, label = raster.rasterize_zbuffer(
        triangles,
        colors,
        labels,
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
        WIDTH,
        HEIGHT,
    )
    return color, label, {
        "arm_fk": arm_fk,
        "hand_roots": np.asarray(hand_roots),
        "depth": depth,
    }


def title(panel: np.ndarray, heading: str, line1: str, line2: str) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 112), fill=(5, 5, 5))
    draw.text((10, 5), heading, font=old.font(19), fill=(255, 255, 255))
    draw.text((10, 34), line1, font=old.font(15), fill=(255, 190, 55))
    draw.text((10, 59), line2, font=old.font(15), fill=(255, 150, 150))
    draw.text(
        (10, 84),
        "V2C03历史保留位姿｜NaturalV2真实法兰｜KAI_ADAPTER_CAD_MISSING",
        font=old.font(13),
        fill=(255, 105, 105),
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--mask-review", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keyframes-only", action="store_true")
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FixedSuccessorError(f"refusing to overwrite: {args.output_dir}")
    old.require_review(args.mask_review, "Mask")
    hawor_result = old.load_json(args.hawor_result)
    if hawor_result.get("session_id") != SESSION:
        raise FixedSuccessorError("HaWoR RESULT session mismatch")
    hawor = old.arrays(args.hawor)
    if not (
        np.array_equal(hawor["original_frame_indices"][:24], np.arange(24))
        and np.all(hawor["observed"][:, :24])
        and hawor_result["outputs"]["npz"]["sha256"] == old.sha256(args.hawor)
    ):
        raise FixedSuccessorError("HaWoR first24 identity/provenance/SHA mismatch")
    mask_manifest = old.load_json(args.mask_manifest)
    if mask_manifest.get("session") != SESSION or len(mask_manifest.get("frames", [])) != 171:
        raise FixedSuccessorError("Mask manifest identity/frame-count mismatch")

    camera, q0, mounts, _ = load_fixed_lineage()
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
        raise FixedSuccessorError("retargeting_human joint identity/shape mismatch")
    q_hand = old.bounded_projection(raw_hand, hand_lower, hand_upper)
    q_arm, hand_targets, initial_fk = solve_arms_from_p3(
        assets, hawor, mounts, q0, arm_lower, arm_upper
    )
    arm_time = old.temporal_metrics(q_arm)
    hand_time = old.temporal_metrics(q_hand)
    gates = {
        "historical_v2c03_camera_exact": bool(
            np.array_equal(camera, old.arrays(V2C03_P3_STATE)["T_camera_base"])
        ),
        "historical_p3_q0_exact": bool(np.array_equal(q_arm[0], q0)),
        "frame_identity": True,
        "human_left_to_physical_right": True,
        "finite": bool(all(np.isfinite(x).all() for x in (camera, q_arm, q_hand, mounts))),
        "mount_proper_se3": bool(np.allclose(np.linalg.det(mounts[:, :3, :3]), 1.0, atol=1e-9)),
        "arm_limits": bool(np.all(q_arm >= arm_lower) and np.all(q_arm <= arm_upper)),
        "hand_limits": bool(np.all(q_hand >= hand_lower) and np.all(q_hand <= hand_upper)),
        "arm_step": bool(arm_time["step_max_rad_per_frame"] <= 0.12 + 1e-9),
        "hand_step": bool(hand_time["step_max_rad_per_frame"] <= 0.08 + 1e-9),
        "arm_second_difference": bool(
            arm_time["second_difference_max_rad_per_frame2"] <= old.SECOND_DIFFERENCE + 1e-9
        ),
        "hand_second_difference": bool(
            hand_time["second_difference_max_rad_per_frame2"] <= old.SECOND_DIFFERENCE + 1e-9
        ),
        "naturalv2_original_scale": True,
        "procedural_flange_ring_absent": True,
    }
    if not all(gates.values()):
        raise FixedSuccessorError(f"pre-render gate failed: {gates}")

    args.output_dir.mkdir(parents=True)
    frame_root = args.output_dir / "frames"
    frame_root.mkdir()
    raster = old.load_module(old.RASTER_SOURCE, "poker_v2c03_p3_naturalv2_raster")
    flange_local = naturalv2_local_triangles()
    cache: dict[Path, Any] = {}
    full_camera = old.look_at_camera()
    full_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    selected_slots = KEY_SLOTS if args.keyframes_only else tuple(range(24))
    key_images: list[np.ndarray] = []
    frame_rows = []
    for slot in selected_slots:
        source = int(hawor["original_frame_indices"][slot])
        raw_path = args.raw_root / f"{source:05d}" / "rgb.png"
        source_record = mask_manifest["frames"][source]["source_rgb"]
        if Path(source_record["path"]).resolve() != raw_path.resolve() or old.sha256(raw_path) != source_record["sha256"]:
            raise FixedSuccessorError(f"raw/Mask frame lineage mismatch at {source}")
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise FixedSuccessorError(f"raw decode failed at {source}")
        source_h, source_w = raw.shape[:2]
        raw = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        k = np.asarray(hawor["intrinsics"][slot], dtype=np.float64).copy()
        k[0] *= WIDTH / source_w
        k[1] *= HEIGHT / source_h
        ego_color, ego_label, _ = render_scene(
            raster, assets, q_arm[slot], q_hand[slot], mounts, camera, k, cache, flange_local
        )
        visible = ego_label >= 0
        overlay = raw.copy()
        overlay[visible] = np.clip(
            0.18 * raw[visible] + 0.82 * ego_color[visible], 0, 255
        ).astype(np.uint8)
        left = title(
            overlay,
            f"原RGB + 历史V2C03投影｜source {source:05d}",
            "P3双臂初态｜M6 arms-only（历史同口径）",
            "固定历史相机/基座｜非frame0拟合｜CONTACT_PENDING",
        )
        full_color, full_label, _ = render_scene(
            raster,
            assets,
            q_arm[slot],
            q_hand[slot],
            mounts,
            full_camera,
            full_k,
            cache,
            flange_local,
        )
        middle = np.full_like(full_color, 242)
        full_visible = full_label >= 0
        middle[full_visible] = full_color[full_visible]
        middle = title(
            middle,
            "V2C03/P3 同步整机动作（central ZJ按历史隐藏）",
            "真实Tianji双臂 + 原尺度NaturalV2 + 真实KaiHand",
            "Kai adapter CAD缺失｜root-plane仅几何候选｜不可部署",
        )
        human_uv = np.asarray(hawor["joints_2d"][:, slot], dtype=np.float64).copy()
        human_uv[..., 0] *= WIDTH / source_w
        human_uv[..., 1] *= HEIGHT / source_h
        tip_errors = [
            [float(x) for x in retarget_rows[slot][physical]["tip_error_mm"]]
            for physical in range(2)
        ]
        local = old.render_local_hand_comparison(
            raster, assets, q_hand[slot], human_uv, tip_errors, cache
        )
        local = title(
            local,
            "手型核对｜绿=Human MANO｜橙/紫=KaiHand",
            "左Kai←人右｜右Kai←人左｜无镜像换色作弊",
            "retargeting_human bounded IK｜指尖误差单位mm",
        )
        combined = np.hstack((left, middle, local))
        frame_path = frame_root / f"{slot:06d}.png"
        if not cv2.imwrite(str(frame_path), combined):
            raise FixedSuccessorError("frame write failed")
        key_images.append(combined) if slot in KEY_SLOTS else None
        frame_rows.append(
            {
                "slot": slot,
                "source_frame": source,
                "ego_robot_pixels": int(np.count_nonzero(visible)),
                "full_view_robot_pixels": int(np.count_nonzero(full_visible)),
                "tip_error_mm": tip_errors,
                "frame_sha256": old.sha256(frame_path),
            }
        )

    sheet = args.output_dir / "POKER_V2C03_P3_NaturalV2_关键帧0_8_15_23.png"
    if not cv2.imwrite(str(sheet), np.vstack(key_images)):
        raise FixedSuccessorError("keyframe sheet write failed")
    video = None
    if not args.keyframes_only:
        video = args.output_dir / "POKER_V2C03_P3_NaturalV2_24帧三栏中文复核.mp4"
        if old.encode_video(frame_root, video, args.fps) != 24:
            raise FixedSuccessorError("video decode frame count mismatch")

    states = args.output_dir / "ROBOT_STATES.npz"
    np.savez_compressed(
        states,
        source_frames=np.arange(24),
        q_arm=q_arm,
        q_hand=q_hand,
        T_camera_base=camera,
        T_tool_hand_root_geometry=mounts,
        T_hand_target_base=hand_targets,
        q0_source=np.asarray("V2C03_P3_USER_RETAINED_STATIC_REFERENCE"),
    )
    manifest = args.output_dir / "FRAME_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "poker-v2c03-p3-naturalv2-frame-manifest-v1",
                "session": SESSION,
                "rendered_slots": list(selected_slots),
                "frames": frame_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    result = {
        "schema_version": "poker-v2c03-p3-naturalv2-successor-result-v1",
        "status": "KEYFRAMES_READY_FOR_VISUAL_REVIEW" if args.keyframes_only else "24FRAME_READY_FOR_VISUAL_REVIEW",
        "grade": "B_DEVELOPMENT_ONLY_PENDING_USER_VISUAL_REVIEW",
        "session": SESSION,
        "development_only": True,
        "downstream_authorized": False,
        "current_robot_authority": False,
        "deployment_authorized": False,
        "training_authorized": False,
        "camera_status": "EXACT_HISTORICAL_USER_RETAINED_V2C03_NOT_TASK_FITTED",
        "arm_initial_status": "EXACT_HISTORICAL_USER_RETAINED_P3",
        "flange_status": "NATURALV2_REAL_IDENTIFIED_WRIST_FLANGE_ORIGINAL_STL_SCALE",
        "adapter_status": "KAI_ADAPTER_CAD_MISSING_ROOTPLANE_GEOMETRY_CANDIDATE_ONLY",
        "central_zj_visual": "HIDDEN_TO_MATCH_HISTORICAL_ACCEPTED_ARMS_ONLY_OVERLAY",
        "hard_gates": gates,
        "metrics": {"arm": arm_time, "hand": hand_time},
        "lineage": {
            "v2c03_p3_state": old.artifact(V2C03_P3_STATE),
            "rootplane_preflight": old.artifact(ROOTPLANE_PREFLIGHT),
            "naturalv2_stl": old.artifact(NATURALV2),
            "historical_user_review_image": old.artifact(
                PROJECT / "archive/legacy_runs/robot/gpt_robot_static_posture_ab_raw_preview_20260831_v2/A_V2C03_P3_002_f00030_RAW_OVERLAY.png"
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
            "frame_manifest": old.artifact(manifest),
            **({"video": old.artifact(video)} if video is not None else {}),
        },
        "claim_limit": "Directed 24-frame development review only. V2C03/P3 is a historical visual placement, not a measured current-session camera calibration. NaturalV2 is an identified wrist flange for another hand; no Kai adapter CAD or measured tool-to-hand transform exists. Root-plane mounting is geometric only. No contact, collision, deployment, training, or Robot authority claim.",
    }
    result_path = args.output_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "sheet": str(sheet), "video": str(video) if video else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
