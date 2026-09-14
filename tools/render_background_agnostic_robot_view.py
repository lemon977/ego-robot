#!/usr/bin/env python3
"""Render a Clean-independent Robot view from a kinematic development scene.

Human and Tracker pixels are never presented as recovered background.  The
Mask deletion region is shown as an explicit neutral UNKNOWN pattern, the
protected physical object remains from the immutable source RGB.  Object6D
occlusion is optional so an explicitly labelled ``NO_OBJECT6D`` preview can be
published without consuming a withdrawn Object6D result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pipeline import robot_scene_state_cpu as official  # noqa: E402
from pipeline.object_occlusion_geometry import ObjectGeometry, project_object_depth  # noqa: E402
from pipeline.robot_pbr_palette import load_shared_robot_palette  # noqa: E402
from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets  # noqa: E402


RASTER_SOURCE = PROJECT / "_run/robot_004_mano21_10frame_cpu_zbuffer_review_20260902_v1/run_zbuffer_review.py"
CAD_SOURCE = PROJECT / "tools/render_robot_scene_cpu_cad.py"
CONTRACT = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
    / "BACKGROUND_AGNOSTIC_ROBOT_TRAINING_CONTRACT_V1.json"
)
AB_CONTRACT = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
    / "HUMANEGO_VISUAL_INPUT_AB_COMPARISON_CONTRACT_V1.json"
)
EXPECTED_CONTRACT_SHA256 = "467a05d346ba644ac76e251c7c4e3c72633d48dcec32911c2f5f98c345862b75"
EXPECTED_AB_CONTRACT_SHA256 = "306d82b0c4accd15d3d95fc0a3e651c2b9cedd5f0c1e5d366d6fba8452293189"
FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)
SIDE_NAMES = ("left", "right")
WIDTH, HEIGHT = 640, 480


class RenderError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RenderError(f"ordinary file required: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RenderError(f"refusing to overwrite: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RenderError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RenderError("Chinese font unavailable")


def header(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    value = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(value)
    draw.rectangle((0, 0, value.width, 78), fill=(0, 0, 0))
    active_font = font(18)
    for row, (text, bgr) in enumerate(lines):
        draw.text((10, 4 + row * 24), text, font=active_font, fill=tuple(reversed(bgr)))
    return cv2.cvtColor(np.asarray(value), cv2.COLOR_RGB2BGR)


def tint(colors: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    if not len(colors):
        return colors
    light = np.clip(colors.astype(np.float64).mean(axis=1, keepdims=True) / 190.0, 0.52, 1.0)
    return np.clip(np.asarray(target)[None] * light, 0, 255).astype(np.uint8)


def unknown_pattern(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = raw.copy()
    yy, xx = np.indices(mask.shape)
    checker = ((xx // 14 + yy // 14) % 2).astype(bool)
    neutral = np.zeros_like(result)
    neutral[checker] = (72, 72, 72)
    neutral[~checker] = (112, 112, 112)
    result[mask] = neutral[mask]
    result[mask & (((xx + yy) % 31) < 2)] = (30, 150, 230)
    return result


def encode_video(frame_root: Path, output: Path, fps: float) -> int:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frame_root / "%06d.png"), "-an", "-c:v", "libx264",
            "-crf", "18", "-pix_fmt", "yuv420p", str(output),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1",
            str(output),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    decoded = int(probe.stdout.strip())
    verified = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if verified.returncode != 0:
        raise RenderError("video full decode failed")
    return decoded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--scene-npz", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mask-frame-manifest", type=Path, required=True)
    parser.add_argument("--object6d-npz", type=Path)
    parser.add_argument("--disable-object6d", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()

    if sha256(CONTRACT) != EXPECTED_CONTRACT_SHA256:
        raise RenderError("background-agnostic contract SHA drift")
    if sha256(AB_CONTRACT) != EXPECTED_AB_CONTRACT_SHA256:
        raise RenderError("HumanEgo visual A/B contract SHA drift")
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RenderError(f"fresh output required: {output}")
    output.parent.resolve(strict=True)

    scene_path = args.scene_npz.resolve(strict=True)
    raw_root = args.raw_root.resolve(strict=True)
    mask_manifest_path = args.mask_frame_manifest.resolve(strict=True)
    if args.disable_object6d and args.object6d_npz is not None:
        raise RenderError("--disable-object6d forbids --object6d-npz")
    if not args.disable_object6d and args.object6d_npz is None:
        raise RenderError("--object6d-npz is required unless --disable-object6d is set")
    object_path = (
        None
        if args.disable_object6d
        else args.object6d_npz.resolve(strict=True)
    )
    scene = arrays(scene_path)
    object6d = None if object_path is None else arrays(object_path)
    frames = np.asarray(scene["source_frames"], dtype=np.int64)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
        raise RenderError("scene source_frames must be strictly increasing")
    q_arm = np.asarray(scene["q_arm"], dtype=np.float64)
    q_hand = np.asarray(scene["q_hand"], dtype=np.float64)
    c2w = np.asarray(scene["c2w"], dtype=np.float64)
    intrinsics = np.asarray(scene["intrinsics"], dtype=np.float64)
    base = np.asarray(scene["T_world_rig"], dtype=np.float64)
    mounts = np.asarray(scene["T_tool_hand"], dtype=np.float64)
    if q_arm.shape[:2] != (len(frames), 2) or q_hand.shape != (len(frames), 2, 22):
        raise RenderError("scene trajectory shape mismatch")
    if not all(np.isfinite(value).all() for value in (q_arm, q_hand, c2w, intrinsics, base, mounts)):
        raise RenderError("scene contains non-finite values")

    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    if mask_manifest.get("session") != args.session:
        raise RenderError("Mask manifest session mismatch")
    mask_rows = mask_manifest.get("frames")
    if not isinstance(mask_rows, list):
        raise RenderError("Mask frame rows missing")
    if object6d is None:
        object_frames = object_valid = object_camera = sizes = None
    else:
        object_frames = np.asarray(object6d["frame_indices"], dtype=np.int64)
        object_valid = np.asarray(object6d["valid"], dtype=bool)
        object_camera = np.asarray(object6d["T_object_to_camera"], dtype=np.float64)
        size_key = "object_size_m" if "object_size_m" in object6d else "object_dimensions_m"
        sizes = np.asarray(object6d[size_key], dtype=np.float64)
        if sizes.shape == (3,):
            sizes = np.repeat(sizes[None], len(object_frames), axis=0)

    assets = load_pinned_robot_assets(PROJECT)
    raster = load_module(RASTER_SOURCE, "background_agnostic_robot_raster")
    cad = load_module(CAD_SOURCE, "background_agnostic_robot_cad")
    palette = load_shared_robot_palette(PROJECT)["cpu_preview"]["bgr_uint8"]
    arm_color = tuple(palette["ARM_SHELL_MATTE_WHITE"])
    hand_color = tuple(palette["KAIHAND_SHELL_STEEL_BLUE"])
    connector_color = tuple(palette["CONNECTOR_FLANGE_IVORY"])
    hand_models = (assets.left_hand, assets.right_hand)
    hand_names = [
        tuple(joint.name for joint in model.joints if joint.joint_type != "fixed")
        for model in hand_models
    ]
    visual_cache: dict[Path, Any] = {}

    output.mkdir()
    review_frames = output / "review_frames"
    robot_view_frames = output / "robot_view_rgb_frames"
    rgba_frames = output / "robot_rgba_frames"
    range_frames = output / "robot_range_frames"
    for root in (review_frames, robot_view_frames, rgba_frames, range_frames):
        root.mkdir()
    frame_metrics: list[dict[str, Any]] = []

    for slot, frame_id_value in enumerate(frames):
        frame_id = int(frame_id_value)
        if frame_id < 0 or frame_id >= len(mask_rows):
            raise RenderError("scene frame outside Mask range")
        if object_frames is not None and frame_id >= len(object_frames):
            raise RenderError("scene frame outside Object6D range")
        row = mask_rows[frame_id]
        if row.get("source_frame") != frame_id:
            raise RenderError("Mask frame identity mismatch")
        raw_path = raw_root / "preprocess/all_data" / f"{frame_id:05d}" / "rgb.png"
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RenderError(f"missing raw frame {raw_path}")
        source_h, source_w = raw.shape[:2]
        raw = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        mask_ref = row.get("clean_removal_object_protected")
        if not isinstance(mask_ref, dict):
            raise RenderError("Mask UNKNOWN artifact missing")
        mask_path = Path(mask_ref["path"]).resolve(strict=True)
        if mask_path.stat().st_size != mask_ref.get("bytes") or sha256(mask_path) != mask_ref.get("sha256"):
            raise RenderError("Mask UNKNOWN artifact SHA drift")
        unknown = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if unknown is None:
            raise RenderError("Mask UNKNOWN image cannot be decoded")
        unknown = cv2.resize(unknown, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
        background = unknown_pattern(raw, unknown)

        k = intrinsics[slot].copy()
        k[0] *= WIDTH / source_w
        k[1] *= HEIGHT / source_h
        camera_base = np.linalg.inv(c2w[slot]) @ base
        values = {name: 0.0 for names in official.ARM_JOINT_NAMES for name in names}
        for side in range(2):
            values.update(dict(zip(official.ARM_JOINT_NAMES[side], q_arm[slot, side], strict=True)))
        arm_fk = official.forward_kinematics(assets.tianji, values)
        geometry = []
        for side, suffix in ((0, "L"), (1, "R")):
            tool = "left_tool" if side == 0 else "right_tool"
            arm = raster.camera_triangles(
                assets.tianji,
                arm_fk,
                camera_base,
                lambda link, suffix=suffix, tool=tool: link.endswith(f"_{suffix}") or link == tool,
                visual_cache,
            )
            geometry.append((arm[0], tint(arm[1], arm_color), arm[2]))
            hand_root = camera_base @ arm_fk[tool] @ mounts[side]
            connector = cad.connector_triangles(
                camera_base @ arm_fk[f"flange_{suffix}"], hand_root, 0.012
            )
            geometry.append(
                (
                    connector[0],
                    np.full((len(connector[0]), 3), connector_color, np.uint8),
                    connector[2] + 1000 + side,
                )
            )
            hand_fk = official.forward_kinematics(
                hand_models[side],
                dict(zip(hand_names[side], q_hand[slot, side], strict=True)),
            )
            hand = raster.camera_triangles(
                hand_models[side], hand_fk, hand_root, lambda _link: True, visual_cache
            )
            geometry.append((hand[0], tint(hand[1], hand_color), hand[2] + 2000 + side))
        triangles = np.concatenate([item[0] for item in geometry if len(item[0])])
        colors = np.concatenate([item[1] for item in geometry if len(item[0])])
        labels = np.concatenate([item[2] for item in geometry if len(item[0])])
        robot_z, robot_rgb, link_id = raster.rasterize_zbuffer(
            triangles,
            colors,
            labels,
            float(k[0, 0]),
            float(k[1, 1]),
            float(k[0, 2]),
            float(k[1, 2]),
            WIDTH,
            HEIGHT,
        )
        robot = link_id >= 0
        object_front = np.zeros_like(robot)
        object_outline = np.zeros_like(robot)
        valid_object = bool(object_valid[frame_id]) if object_valid is not None else False
        if valid_object:
            assert object_camera is not None and sizes is not None
            object_depth = project_object_depth(
                ObjectGeometry(
                    kind="box_xyz",
                    transform_object_to_camera=object_camera[frame_id],
                    box_size_xyz_m=sizes[frame_id],
                ),
                k,
                HEIGHT,
                WIDTH,
            )
            object_front = robot & object_depth.amodal_mask & (robot_z >= object_depth.near_m - 1e-4)
            object_outline = cv2.Canny(object_depth.amodal_mask.astype(np.uint8) * 255, 80, 160) > 0
        visible = robot & ~object_front
        robot_view = background.copy()
        robot_view[visible] = robot_rgb[visible]
        robot_view[object_outline] = (0, 220, 255)
        robot_only = unknown_pattern(np.zeros_like(raw) + 32, np.ones_like(unknown))
        robot_only[visible] = robot_rgb[visible]
        robot_only[object_outline] = (0, 220, 255)
        rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        rgba[..., :3] = robot_rgb
        rgba[..., 3] = visible.astype(np.uint8) * 255
        range_value = np.where(visible, robot_z, np.nan).astype(np.float32)

        cv2.imwrite(str(robot_view_frames / f"{slot:06d}.png"), robot_view)
        cv2.imwrite(str(rgba_frames / f"{slot:06d}.png"), rgba)
        np.savez_compressed(
            range_frames / f"{slot:06d}.npz",
            source_frame=np.int32(frame_id),
            range_m=range_value,
            visible=visible,
            object_occluded=object_front,
        )
        raw_panel = header(raw, [(f"原始人类RGB｜源帧 {frame_id}", (255, 255, 255)), ("NO_OBJECT6D｜CONTACT_PENDING", (40, 80, 255)), ("DEVELOPMENT_ONLY", (40, 80, 255))])
        unknown_panel = header(background, [("显式 UNKNOWN 背景", (255, 255, 255)), ("手/Tracker后方不伪装Clean", (30, 150, 230)), ("NO_OBJECT6D｜物体保持原图", (40, 80, 255))])
        object_line = "黄色=Object6D遮挡" if object6d is not None else "NO_OBJECT6D｜不推断遮挡"
        robot_panel = header(robot_only, [("Tianji + 双KaiHand RGBA", (255, 255, 255)), (object_line, (40, 80, 255)), ("DEVELOPMENT_ONLY｜非真机标定", (40, 80, 255))])
        result_panel = header(robot_view, [("ROBOT_VIEW_RGB 运动学预览", (255, 255, 255)), ("CONTACT_PENDING｜不推断接触", (40, 80, 255)), ("DEVELOPMENT_ONLY", (40, 80, 255))])
        canvas = np.concatenate(
            (np.concatenate((raw_panel, unknown_panel), axis=1), np.concatenate((robot_panel, result_panel), axis=1)),
            axis=0,
        )
        if not cv2.imwrite(str(review_frames / f"{slot:06d}.png"), canvas):
            raise RenderError("review frame write failed")
        frame_metrics.append(
            {
                "slot": slot,
                "source_frame": frame_id,
                "unknown_pixels": int(np.count_nonzero(unknown)),
                "robot_pixels": int(np.count_nonzero(robot)),
                "robot_visible_pixels": int(np.count_nonzero(visible)),
                "object_occluded_robot_pixels": int(np.count_nonzero(object_front)),
                "object6d_valid": valid_object,
                "robot_rgba": artifact(rgba_frames / f"{slot:06d}.png"),
                "robot_range": artifact(range_frames / f"{slot:06d}.npz"),
                "robot_view_rgb": artifact(robot_view_frames / f"{slot:06d}.png"),
            }
        )

    review_video = output / f"{args.session}_背景无关Robot视角_中文复核.mp4"
    robot_view_video = output / "ROBOT_VIEW_RGB.mp4"
    review_count = encode_video(review_frames, review_video, args.fps)
    view_count = encode_video(robot_view_frames, robot_view_video, args.fps)
    if review_count != len(frames) or view_count != len(frames):
        raise RenderError("video frame count mismatch")
    frame_manifest = output / "FRAME_MANIFEST.json"
    atomic_json(
        frame_manifest,
        {
            "schema_version": "background-agnostic-robot-view-frame-manifest-v1",
            "task": args.task,
            "session": args.session,
            "frame_count": len(frames),
            "source_frames": frames.tolist(),
            "object6d_consumed": object6d is not None,
            "object6d_status": "ENABLED" if object6d is not None else "NO_OBJECT6D",
            "contact_status": "CONTACT_PENDING",
            "development_only": True,
            "frames": frame_metrics,
        },
    )
    result = {
        "schema_version": "background-agnostic-robot-view-result-v1",
        "status": "PASS_DEVELOPMENT_PREVIEW_NOT_FORMAL_ROBOT_AUTHORITY",
        "task": args.task,
        "session": args.session,
        "frame_count": len(frames),
        "clean_consumed": False,
        "object6d_consumed": object6d is not None,
        "object6d_status": "ENABLED" if object6d is not None else "NO_OBJECT6D",
        "contact_status": "CONTACT_PENDING",
        "development_only": True,
        "human_raw_rgb_branch": "immutable same-session source RGB",
        "robot_view_rgb_branch": "protected source background/object plus explicit UNKNOWN plus kinematic-development Robot RGBA",
        "contact_claim": "NONE_CONTACT_PENDING",
        "formal_robot_authority": False,
        "formal_training_authority": False,
        "inputs": {
            "scene": artifact(scene_path),
            "mask_frame_manifest": artifact(mask_manifest_path),
            "object6d": (
                artifact(object_path)
                if object_path is not None
                else {"status": "NO_OBJECT6D", "consumed": False}
            ),
            "background_contract": artifact(CONTRACT),
            "visual_ab_contract": artifact(AB_CONTRACT),
        },
        "outputs": {
            "frame_manifest": artifact(frame_manifest),
            "review_video": artifact(review_video),
            "robot_view_rgb_video": artifact(robot_view_video),
        },
        "claim_limit": "Immediate Clean-independent kinematic preview. NO_OBJECT6D, CONTACT_PENDING and DEVELOPMENT_ONLY: no object occlusion/contact inference, formal full-session Robot authority, calibration, tactile truth, training authority, or deployment claim.",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
