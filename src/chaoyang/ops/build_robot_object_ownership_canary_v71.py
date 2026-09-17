#!/usr/bin/env python3
"""Build a real-session Robot/Object optical-Z ownership canary.

The canary rasterizes each directly observed Object6D instance into the exact
640x480 camera used by the Robot z-buffer and compares optical-Z.  Object
appearance is legal only where the current Raw identity mask is visible;
amodal pixels without a donor/renderer become UNKNOWN.  This is Silver-style
development evidence, never Gold accuracy, contact truth or Robot authority.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chaoyang.ops import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402


WIDTH, HEIGHT = 640, 480
EPSILON_M = 0.003
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ellipsoid_triangles(size_m: np.ndarray, rings: int = 12, sectors: int = 24) -> np.ndarray:
    size = np.asarray(size_m, dtype=np.float64)
    if size.shape != (3,) or not np.all(np.isfinite(size) & (size > 0)):
        raise ValueError("object size must be three finite positive metric dimensions")
    vertices = []
    for ring in range(rings + 1):
        latitude = -np.pi / 2 + np.pi * ring / rings
        for sector in range(sectors):
            longitude = 2 * np.pi * sector / sectors
            vertices.append(
                (
                    0.5 * size[0] * np.cos(latitude) * np.cos(longitude),
                    0.5 * size[1] * np.cos(latitude) * np.sin(longitude),
                    0.5 * size[2] * np.sin(latitude),
                )
            )
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = []
    for ring in range(rings):
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            a = ring * sectors + sector
            b = ring * sectors + nxt
            c = (ring + 1) * sectors + sector
            d = (ring + 1) * sectors + nxt
            triangles.extend(((vertices[a], vertices[c], vertices[b]), (vertices[b], vertices[c], vertices[d])))
    return np.asarray(triangles, dtype=np.float64)


def ownership_from_depth(
    robot_depth: np.ndarray,
    object_depth: np.ndarray,
    raw_visible_object: np.ndarray,
    epsilon_m: float = EPSILON_M,
) -> np.ndarray:
    robot = np.asarray(robot_depth, dtype=np.float64)
    obj = np.asarray(object_depth, dtype=np.float64)
    visible = np.asarray(raw_visible_object)
    if robot.shape != obj.shape or visible.shape != obj.shape or visible.dtype != np.bool_:
        raise ValueError("depth/mask shapes differ")
    robot_valid = np.isfinite(robot) & (robot > 0)
    object_valid = np.isfinite(obj) & (obj > 0)
    owner = np.zeros(obj.shape, dtype=np.uint8)  # BACKGROUND
    robot_only = robot_valid & ~object_valid
    object_only = object_valid & ~robot_valid
    owner[robot_only] = 3  # ROBOT_FRONT
    owner[object_only & visible] = 2  # OBJECT_FRONT
    owner[object_only & ~visible] = 4  # geometry without legal appearance
    overlap = robot_valid & object_valid
    object_front = overlap & (obj + epsilon_m < robot)
    robot_front = overlap & (robot + epsilon_m < obj)
    owner[robot_front] = 3
    owner[object_front & visible] = 2
    owner[object_front & ~visible] = 4
    owner[overlap & ~(object_front | robot_front)] = 4
    return owner


def rectified_object_to_selected(
    rectified_to_selected: np.ndarray, object_to_rectified: np.ndarray
) -> np.ndarray:
    left = np.asarray(rectified_to_selected, dtype=np.float64)
    right = np.asarray(object_to_rectified, dtype=np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError("camera transforms must be 4x4")
    result = left @ right
    if not np.all(np.isfinite(result)) or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError("invalid selected-camera transform")
    return result


def read_frame(capture: cv2.VideoCapture, frame_id: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, bgr = capture.read()
    if not ok:
        raise RuntimeError(f"cannot decode source frame {frame_id}")
    return cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def draw_header(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = np.zeros((HEIGHT + 72, image_bgr.shape[1], 3), dtype=np.uint8)
    canvas[72:] = image_bgr
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(str(FONT), 20)
    small = ImageFont.truetype(str(FONT), 17)
    draw.text((12, 5), lines[0], font=font, fill=(255, 255, 255))
    draw.text((12, 38), lines[1], font=small, fill=(220, 220, 220))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--robot-zbuffer-result", type=Path, required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--object6d-root", type=Path, required=True)
    parser.add_argument("--object-mask-manifest", type=Path, required=True)
    parser.add_argument("--registration-authority", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.output_dir}")
    inputs = [args.robot_zbuffer_result, args.hawor, args.source_video, args.object_mask_manifest, args.registration_authority]
    inputs = [path.resolve(strict=True) for path in inputs]
    if not all(args.session_id in str(path) for path in inputs):
        raise ValueError("same-session input identity check failed")
    robot_result = json.loads(inputs[0].read_text(encoding="utf-8"))
    if robot_result.get("status") != "PASS_DEVELOPMENT_ONE_SCENE_ZBUFFER_EXPORT":
        raise ValueError("Robot z-buffer development PASS required")
    robot_archive_ref = robot_result["outputs"][0]
    robot_archive = Path(robot_archive_ref["path"]).resolve(strict=True)
    if ref(robot_archive) != robot_archive_ref:
        raise ValueError("Robot z-buffer artifact reference mismatch")
    robot = load_npz(robot_archive)
    frames = np.asarray(robot["frame_ids"], dtype=np.int64)
    if len(frames) == 0 or len(set(frames.tolist())) != len(frames):
        raise ValueError("Robot frame ids must be non-empty and unique")
    hawor = load_npz(inputs[1])
    manifest = json.loads(inputs[3].read_text(encoding="utf-8"))
    if manifest.get("session_id") != args.session_id or manifest.get("physical_instance_count") != 3:
        raise ValueError("Chips three-instance object manifest required")
    manifest_rows = {int(row["frame_id"]): row for row in manifest["frames"]}
    registration = load_npz(inputs[4])
    rectified_to_selected = np.asarray(
        registration["T_stereo_rectified_camera_to_selected_camera"], dtype=np.float64
    )
    selected_k = np.asarray(registration["selected_rgb_intrinsics"], dtype=np.float64)
    selected_width = 2.0 * float(selected_k[0, 2]) + 1.0
    selected_height = 2.0 * float(selected_k[1, 2]) + 1.0
    objects = []
    object_refs = []
    for instance in range(3):
        path = (args.object6d_root / f"physical_object_{instance}/VISUAL_FIXED_INSTANCE_OBJECT6D.npz").resolve(strict=True)
        objects.append(load_npz(path))
        object_refs.append(ref(path))
        if not np.all(objects[-1]["physical_instance_id"][objects[-1]["valid"]] == 0):
            # Each child producer is one-instance local; the directory/manifest
            # supplies the global physical identity and must remain separate.
            raise ValueError("unexpected child-local physical instance id")
    raster = old.load_module(old.RASTER_SOURCE, f"{args.session_id}_object_ownership_v71")
    local_meshes = [ellipsoid_triangles(obj["object_size_m"]) for obj in objects]
    object_depth_rows, object_label_rows, owner_rows = [], [], []
    metrics = []
    args.output_dir.mkdir(parents=True)
    capture = cv2.VideoCapture(str(inputs[2]))
    video_path = args.output_dir / f"{args.session_id}_ROBOT_OBJECT_OWNERSHIP_24FRAME.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (WIDTH * 3, HEIGHT + 72))
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise RuntimeError("source/review video open failed")
        for row_index, frame in enumerate(frames.tolist()):
            triangles, labels, observed_ids = [], [], []
            for instance, (obj, local) in enumerate(zip(objects, local_meshes, strict=True)):
                direct = bool(obj["valid"][frame]) and bool(obj["observed"][frame])
                if not direct:
                    continue
                transform = rectified_object_to_selected(
                    rectified_to_selected, obj["T_object_to_camera"][frame]
                )
                camera_triangles = local @ transform[:3, :3].T + transform[:3, 3]
                triangles.append(camera_triangles)
                labels.append(np.full(len(camera_triangles), instance, dtype=np.int32))
                observed_ids.append(instance)
            k = selected_k.copy()
            k[0] *= WIDTH / selected_width
            k[1] *= HEIGHT / selected_height
            if triangles:
                tri = np.concatenate(triangles)
                lab = np.concatenate(labels)
                colors = np.repeat(
                    np.asarray([[40, 220, 40]], dtype=np.uint8), len(tri), axis=0
                )
                object_depth, _, object_label = raster.rasterize_zbuffer(
                    tri, colors, lab, float(k[0, 0]), float(k[1, 1]),
                    float(k[0, 2]), float(k[1, 2]), WIDTH, HEIGHT,
                )
            else:
                object_depth = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
                object_label = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
            raw = read_frame(capture, frame)
            visible = np.zeros((HEIGHT, WIDTH), dtype=bool)
            for instance in range(3):
                entry = manifest_rows[frame]["physical_instances"][str(instance)]
                if not entry.get("observed"):
                    continue
                mask_path = Path(entry["mask"]["path"]).resolve(strict=True)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"cannot read mask {mask_path}")
                visible |= cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
            robot_depth = np.asarray(robot["depth_m"][row_index], dtype=np.float64)
            owner = ownership_from_depth(robot_depth, object_depth, visible)
            robot_rgb = np.asarray(robot["rgb"][row_index], dtype=np.uint8)
            geometry = raw.copy()
            object_geom = np.isfinite(object_depth)
            geometry[object_geom] = (0.45 * geometry[object_geom] + 0.55 * np.asarray((0, 220, 0))).astype(np.uint8)
            robot_geom = np.isfinite(robot_depth)
            geometry[robot_geom] = (0.35 * geometry[robot_geom] + 0.65 * robot_rgb[robot_geom]).astype(np.uint8)
            ownership = np.zeros_like(raw)
            ownership[owner == 2] = (0, 220, 0)
            ownership[owner == 3] = (255, 120, 20)
            ownership[owner == 4] = (255, 0, 255)
            overlap = np.isfinite(robot_depth) & np.isfinite(object_depth)
            mask_geometry_intersection = int((visible & object_geom).sum())
            mask_geometry_union = int((visible | object_geom).sum())
            metric = {
                "frame_id": frame,
                "time_s": frame / 30.0,
                "direct_object_instances": observed_ids,
                "robot_pixels": int(np.isfinite(robot_depth).sum()),
                "object_geometry_pixels": int(np.isfinite(object_depth).sum()),
                "overlap_pixels": int(overlap.sum()),
                "object_front_pixels": int((owner == 2).sum()),
                "robot_front_pixels": int((owner == 3).sum()),
                "unknown_pixels": int((owner == 4).sum()),
                "unknown_overlap_pixels": int(((owner == 4) & overlap).sum()),
                "raw_visible_object_pixels": int(visible.sum()),
                "mask_geometry_intersection_pixels": mask_geometry_intersection,
                "mask_geometry_union_pixels": mask_geometry_union,
                "mask_geometry_iou": mask_geometry_intersection / max(1, mask_geometry_union),
            }
            metrics.append(metric)
            first = draw_header(raw, [f"原始 RGB｜帧 {frame:04d}｜{frame/30:.2f}s", "当前可见物体像素来自 Raw"])
            second = draw_header(geometry, ["同一相机 optical-Z", f"绿=Object6D 刚体近似｜Robot=真实渲染｜实例 {observed_ids or '无'}"])
            third = draw_header(ownership, ["Robot–Object 前后关系", f"绿=物体前｜蓝=Robot前｜紫=未知｜重叠 {metric['overlap_pixels']} px"])
            writer.write(np.concatenate((first, second, third), axis=1))
            object_depth_rows.append(object_depth)
            object_label_rows.append(object_label)
            owner_rows.append(owner)
    finally:
        capture.release()
        writer.release()
    archive = args.output_dir / "ROBOT_OBJECT_OWNERSHIP_CANARY.npz"
    np.savez_compressed(
        archive, frame_ids=frames, object_depth_m=np.asarray(object_depth_rows, dtype=np.float32),
        object_instance_id=np.asarray(object_label_rows, dtype=np.int32),
        ownership=np.asarray(owner_rows, dtype=np.uint8),
    )
    metrics_path = args.output_dir / "FRAME_METRICS.json"
    atomic_json(metrics_path, {"frames": metrics})
    decoded = 0
    verify = cv2.VideoCapture(str(video_path))
    while True:
        ok, _ = verify.read()
        if not ok:
            break
        decoded += 1
    verify.release()
    status = "PASS_DEVELOPMENT_ROBOT_OBJECT_OWNERSHIP" if decoded == len(frames) else "FAILED_RUNTIME_FINAL"
    direct_ious = [
        row["mask_geometry_iou"] for row in metrics if row["direct_object_instances"]
    ]
    result = {
        "schema_version": "robot-object-ownership-canary-v71-v1",
        "artifact_revision": "R7_1",
        "status": status,
        "session_id": args.session_id,
        "frame_count": len(frames),
        "decoded_review_frames": decoded,
        "inputs": [ref(path) for path in inputs] + object_refs + [ref(robot_archive)],
        "outputs": [ref(archive), ref(metrics_path), ref(video_path)],
        "object_policy": "DIRECT_OBSERVED_ONLY_THREE_INSTANCES_RIGID_OVAL_APPROXIMATION",
        "coordinate_contract": "T_selected_object = T_stereo_rectified_camera_to_selected_camera @ T_rectified_object; selected K scaled 1280x960 to 640x480",
        "appearance_policy": "CURRENT_RAW_VISIBLE_ONLY; missing amodal appearance becomes TIE_UNKNOWN",
        "epsilon_m": EPSILON_M,
        "coordinate_qa": {
            "direct_observed_sample_frames": len(direct_ious),
            "mask_geometry_iou_mean": float(np.mean(direct_ious)) if direct_ious else None,
            "mask_geometry_iou_median": float(np.median(direct_ious)) if direct_ious else None,
            "mask_geometry_iou_max": float(np.max(direct_ious)) if direct_ious else None,
            "robot_object_overlap_frames": sum(row["overlap_pixels"] > 0 for row in metrics),
            "robot_object_overlap_pixels": sum(row["overlap_pixels"] for row in metrics),
            "unknown_overlap_pixels": sum(row["unknown_overlap_pixels"] for row in metrics),
        },
        "authority": False,
        "gold_accuracy_computed": False,
        "contact_truth": False,
        "control_ground_truth": False,
        "claim_limit": "Real-session geometric ownership development evidence only; Chips deformation and missing hidden appearance remain UNKNOWN. No Gold/contact/Robot/physical authority.",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_json(args.output_dir / "RESULT.json", result)
    print(json.dumps({"status": status, "frames": len(frames), "review": str(video_path)}, ensure_ascii=False))
    return 0 if status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
