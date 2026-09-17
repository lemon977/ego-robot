#!/usr/bin/env python3
"""Build a real-session Robot-versus-visible-object optical-Z canary.

The canary deliberately uses only the directly observed task-object mask and
registered Stereo surface Z.  It therefore diagnoses depth ordering where
evidence exists and emits UNKNOWN elsewhere; it does not reconstruct an
occluded object or grant Silver/Gold authority.
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
from chaoyang.pipeline.occlusion_compositor_v1 import (  # noqa: E402
    DepthQualityEvidence,
    ObjectPixelSource,
    Ownership,
    audit_frame,
    choose_object_pixels,
    resolve_ownership,
)


COLORS_BGR = {
    Ownership.BACKGROUND: (80, 80, 80),
    Ownership.HUMAN_FRONT: (50, 200, 50),
    Ownership.OBJECT_FRONT: (230, 120, 30),
    Ownership.ROBOT_FRONT: (20, 150, 240),
    Ownership.TIE_UNKNOWN: (220, 20, 220),
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def exact(reference: dict[str, Any], label: str, base: Path | None = None) -> Path:
    path = Path(str(reference["path"]))
    if not path.is_absolute():
        if base is None:
            raise RuntimeError(f"{label}: relative reference lacks base")
        path = base / path
    path = path.resolve(strict=True)
    if path.stat().st_size != int(reference["bytes"]) or sha256(path) != reference["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA mismatch")
    return path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_frames(path: Path, wanted: set[int]) -> tuple[dict[int, np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"source video has invalid FPS: {path}")
    result: dict[int, np.ndarray] = {}
    index = 0
    try:
        while wanted - result.keys():
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                result[index] = frame
            index += 1
    finally:
        capture.release()
    if result.keys() != wanted:
        raise RuntimeError(f"source video missing frames: {sorted(wanted - result.keys())}")
    return result, fps


def object_union(row: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.bool_)
    for instance_id, instance in sorted(row["physical_instances"].items()):
        if not instance.get("valid"):
            continue
        mask_path = exact(instance["mask"], f"object mask {instance_id}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot decode object mask: {mask_path}")
        union |= cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return union


def quality_evidence(depth: np.ndarray, disparity: np.ndarray, valid: np.ndarray, object_mask: np.ndarray, raw: np.ndarray) -> DepthQualityEvidence:
    finite_disparity = valid & np.isfinite(disparity) & (disparity > 0.0)
    registration = np.ones(valid.shape, dtype=np.bool_)
    eroded = cv2.erode(object_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    textured = cv2.boxFilter(((np.abs(gx) + np.abs(gy)) > 4.0).astype(np.float32), -1, (7, 7)) >= 0.08
    filled = np.where(valid, depth, 0.0).astype(np.float32)
    median = cv2.medianBlur(filled, 5)
    local = valid & (np.abs(depth - median) <= np.maximum(0.02, 0.05 * depth))
    in_range = valid & np.isfinite(depth) & (depth >= 0.20) & (depth <= 3.0)
    return DepthQualityEvidence(
        disparity_finite=finite_disparity,
        registration_pass=registration,
        away_from_occlusion_edge=eroded,
        texture_support=textured,
        local_consistency_pass=local,
        in_valid_depth_range=in_range,
        depth_confidence_present=False,
    )


def protected_loss_reasons(
    *,
    protected: np.ndarray,
    retained: np.ndarray,
    robot_mask: np.ndarray,
    robot_depth_valid: np.ndarray,
    evidence: DepthQualityEvidence,
) -> dict[str, int]:
    """Assign every lost protected pixel to one deterministic first cause.

    The categories are intentionally mutually exclusive.  They diagnose why
    a directly visible Raw object pixel could not be retained by the ordering
    contract; they are not accuracy labels and do not infer hidden geometry.
    """

    remaining = protected & ~retained
    counts: dict[str, int] = {}

    def take(name: str, failed: np.ndarray) -> None:
        nonlocal remaining
        selected = remaining & failed
        counts[name] = int(np.count_nonzero(selected))
        remaining &= ~selected

    take("outside_robot_unexpected", ~robot_mask)
    take("robot_depth_invalid", robot_mask & ~robot_depth_valid)
    take("disparity_invalid", ~evidence.disparity_finite)
    take("registration_failed", ~evidence.registration_pass)
    take("object_occlusion_edge", ~evidence.away_from_occlusion_edge)
    take("low_texture", ~evidence.texture_support)
    take("local_depth_inconsistent", ~evidence.local_consistency_pass)
    take("depth_out_of_range", ~evidence.in_valid_depth_range)
    counts["unclassified_ordering"] = int(np.count_nonzero(remaining))
    counts["total_lost"] = int(np.count_nonzero(protected & ~retained))
    if sum(value for key, value in counts.items() if key != "total_lost") != counts["total_lost"]:
        raise RuntimeError("protected loss reason partition is not closed")
    return counts


def object_front_edge_support(
    *,
    object_mask: np.ndarray,
    robot_mask: np.ndarray,
    object_depth_m: np.ndarray,
    robot_depth_m: np.ndarray,
    evidence: DepthQualityEvidence,
) -> np.ndarray:
    """Recover a one-pixel observed edge only from adjacent interior consensus.

    Stereo values on an occlusion boundary are not trusted on their own.  A
    boundary pixel becomes orderable only when every other quality proxy
    passes, its depth margin says object-front, and a 3x3 neighbour contains a
    quality-approved interior object-front pixel.  This remains visible-surface
    development evidence and never supplies hidden object appearance.
    """

    non_edge_quality = (
        evidence.disparity_finite
        & evidence.registration_pass
        & evidence.texture_support
        & evidence.local_consistency_pass
        & evidence.in_valid_depth_range
    )
    overlap = object_mask & robot_mask
    object_front_margin = (
        np.isfinite(object_depth_m)
        & np.isfinite(robot_depth_m)
        & (object_depth_m + 0.003 < robot_depth_m)
    )
    interior_front = overlap & evidence.away_from_occlusion_edge & non_edge_quality & object_front_margin
    adjacent_interior = cv2.dilate(interior_front.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return (
        overlap
        & ~evidence.away_from_occlusion_edge
        & non_edge_quality
        & object_front_margin
        & adjacent_interior
    )


def title(image: np.ndarray, text: str) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, 34), fill=(0, 0, 0))
    draw.text((8, 4), text, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--robot-zbuffer-result", type=Path, required=True)
    parser.add_argument("--depth-result", type=Path, required=True)
    parser.add_argument("--object-mask-manifest", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)

    z_result_path = args.robot_zbuffer_result.resolve(strict=True)
    depth_result_path = args.depth_result.resolve(strict=True)
    mask_manifest_path = args.object_mask_manifest.resolve(strict=True)
    source_video = args.source_video.resolve(strict=True)
    z_result = load(z_result_path)
    depth_result = load(depth_result_path)
    mask_manifest = load(mask_manifest_path)
    if any(
        value != args.session_id
        for value in (z_result.get("session_id"), depth_result.get("session_id"), mask_manifest.get("session"))
    ):
        raise RuntimeError("same-session identity check failed")
    if depth_result.get("status") != "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION":
        raise RuntimeError("corrected metric depth is not passed")
    if depth_result.get("robot_contact_authorized") is not False:
        raise RuntimeError("unexpected depth authority boundary")

    z_archive = exact(z_result["outputs"][0], "Robot z-buffer")
    depth_manifest_path = exact(depth_result["frame_manifest"], "depth frame manifest")
    depth_manifest = load(depth_manifest_path)
    depth_rows = {int(row["frame_id"]): row for row in depth_manifest["frames"]}
    mask_rows = {int(row["source_frame"]): row for row in mask_manifest["frames"]}
    with np.load(z_archive, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        robot_rgb = np.asarray(archive["rgb"], dtype=np.uint8)
        robot_depth = np.asarray(archive["depth_m"], dtype=np.float32)
        robot_label = np.asarray(archive["render_label"], dtype=np.int32)
        triangle_id = np.asarray(archive["triangle_id"], dtype=np.int64)
    if robot_rgb.shape[0] != len(frame_ids) or robot_depth.shape != robot_label.shape:
        raise RuntimeError("Robot z-buffer array closure failed")
    raw_frames, source_fps = source_frames(source_video, set(frame_ids.tolist()))
    ownership_rows, valid_rows, per_frame, montage_rows, video_rows = [], [], [], [], []
    montage_slots = set(
        np.rint(np.linspace(0, len(frame_ids) - 1, min(8, len(frame_ids)))).astype(np.int64).tolist()
    )

    for slot, frame_id_value in enumerate(frame_ids):
        frame_id = int(frame_id_value)
        raw = cv2.resize(raw_frames[frame_id], (robot_depth.shape[2], robot_depth.shape[1]), interpolation=cv2.INTER_AREA)
        object_mask = object_union(mask_rows[frame_id], robot_depth.shape[1:])
        depth_row = depth_rows[frame_id]
        depth_path = exact(
            {
                "path": depth_row["relative_path"],
                "bytes": depth_row["bytes"],
                "sha256": depth_row["sha256"],
            },
            f"depth frame {frame_id}",
            depth_manifest_path.parent,
        )
        with np.load(depth_path, allow_pickle=False) as depth_file:
            depth = np.asarray(depth_file["depth_m"], dtype=np.float32)
            disparity = np.asarray(depth_file["disparity_px"], dtype=np.float32)
            valid = np.asarray(depth_file["valid"], dtype=np.bool_)
        if depth.shape != robot_depth.shape[1:]:
            raise RuntimeError("Depth and Robot render grids differ")
        pixels, provenance = choose_object_pixels(raw_rgb=raw, raw_visible_mask=object_mask)
        robot_mask = robot_label[slot] >= 0
        decision = (cv2.dilate(object_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & (
            cv2.dilate(robot_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        )
        evidence = quality_evidence(depth, disparity, valid, object_mask, raw)
        supported_edge = object_front_edge_support(
            object_mask=object_mask,
            robot_mask=robot_mask,
            object_depth_m=depth,
            robot_depth_m=robot_depth[slot],
            evidence=evidence,
        )
        evidence = DepthQualityEvidence(
            disparity_finite=evidence.disparity_finite,
            registration_pass=evidence.registration_pass,
            away_from_occlusion_edge=evidence.away_from_occlusion_edge | supported_edge,
            texture_support=evidence.texture_support,
            local_consistency_pass=evidence.local_consistency_pass,
            in_valid_depth_range=evidence.in_valid_depth_range,
            depth_confidence_present=False,
        )
        robot_depth_valid = robot_mask & np.isfinite(robot_depth[slot]) & (robot_depth[slot] > 0.0)
        result = resolve_ownership(
            human_mask=np.zeros_like(object_mask),
            object_amodal_mask=object_mask,
            object_depth_m=depth,
            object_depth_valid=valid & object_mask,
            robot_alpha_mask=robot_mask,
            robot_depth_m=robot_depth[slot],
            robot_depth_valid=robot_depth_valid,
            stereo_depth_valid=valid,
            depth_quality_evidence=evidence,
            object_rgb=pixels,
            object_pixel_source=provenance,
            contact_decision_mask=decision,
        )
        protected = object_mask & valid & ((~robot_mask) | (depth + 0.003 < robot_depth[slot]))
        audit = audit_frame(result, protected_raw_object_mask=protected)
        retained_mask = protected & (result.ownership == int(Ownership.OBJECT_FRONT)) & (
            result.object_pixel_source == int(ObjectPixelSource.RAW_VISIBLE)
        )
        loss_reasons = protected_loss_reasons(
            protected=protected,
            retained=retained_mask,
            robot_mask=robot_mask,
            robot_depth_valid=robot_depth_valid,
            evidence=evidence,
        )
        per_frame.append({
            "frame_id": frame_id,
            "contact_band_pixels": audit.contact_pixels,
            "known_decision_coverage": audit.known_decision_coverage,
            "unknown_pixel_ratio": audit.unknown_pixel_ratio,
            "protected_retention": audit.protected_retention,
            "protected_pixels": audit.protected_pixels,
            "retained_protected_pixels": audit.retained_protected_pixels,
            "object_visible_pixels": int(object_mask.sum()),
            "robot_pixels": int(robot_mask.sum()),
            "direct_overlap_pixels": int((object_mask & robot_mask).sum()),
            "edge_ordering_supported_pixels": int(np.count_nonzero(supported_edge)),
            "protected_loss_reasons": loss_reasons,
            "accuracy_reported": False,
        })
        ownership_rows.append(result.ownership)
        valid_rows.append(result.training_valid_mask)

        overlay = raw.copy()
        for owner, color in COLORS_BGR.items():
            selected = result.ownership == int(owner)
            if owner is Ownership.BACKGROUND:
                continue
            overlay[selected] = (0.35 * overlay[selected] + 0.65 * np.asarray(color)).astype(np.uint8)
        composite = raw.copy()
        robot_front = result.ownership == int(Ownership.ROBOT_FRONT)
        composite[robot_front] = robot_rgb[slot][robot_front]
        unknown = decision & (result.ownership == int(Ownership.TIE_UNKNOWN))
        checker = (np.indices(unknown.shape).sum(axis=0) // 6) % 2 == 0
        composite[unknown & checker] = (255, 0, 255)
        composite[unknown & ~checker] = (30, 30, 30)
        raw_mask = raw.copy()
        raw_mask[object_mask] = (0.45 * raw_mask[object_mask] + 0.55 * np.asarray((255, 180, 30))).astype(np.uint8)
        robot_view = robot_rgb[slot].copy()
        video_rows.append(np.vstack([
            np.hstack([title(raw_mask, f"帧 {frame_id}：Raw＋物体可见Mask"), title(robot_view, "同一z-buffer：全部Robot link")]),
            np.hstack([title(overlay, "前后关系：蓝物体／橙Robot／紫UNKNOWN"), title(composite, "诊断合成：UNKNOWN棋盘，不冒充物体")]),
        ]))
        if slot in montage_slots:
            montage_rows.append(np.hstack([
                title(raw_mask, f"帧 {frame_id}：Raw＋物体可见Mask"),
                title(robot_view, "同一z-buffer：全部Robot link"),
                title(overlay, "前后关系：蓝物体／橙Robot／紫UNKNOWN"),
                title(composite, "诊断合成：UNKNOWN棋盘，不冒充物体"),
            ]))

    arrays_path = output / "VISIBLE_SURFACE_OCCLUSION_CANARY.npz"
    np.savez_compressed(
        arrays_path,
        frame_ids=frame_ids,
        ownership=np.asarray(ownership_rows, dtype=np.uint8),
        training_valid_mask=np.asarray(valid_rows, dtype=np.bool_),
        robot_triangle_id=triangle_id,
    )
    montage_path = output / f"{args.session_id}_可见表面遮挡ZBuffer_{len(frame_ids)}帧_均匀预览.png"
    if not cv2.imwrite(str(montage_path), np.vstack(montage_rows)):
        raise RuntimeError("failed to write montage")
    video_path = output / f"{args.session_id}_可见表面遮挡ZBuffer_{len(frame_ids)}帧.mp4"
    frame_size = (video_rows[0].shape[1], video_rows[0].shape[0])
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError("failed to open diagnostic video writer")
    for frame in video_rows:
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(video_path))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded != len(frame_ids):
        raise RuntimeError(f"diagnostic video decoded {decoded}/{len(frame_ids)} frames")
    total = sum(row["contact_band_pixels"] for row in per_frame)
    known = sum(round(row["contact_band_pixels"] * row["known_decision_coverage"]) for row in per_frame)
    protected = sum(row["protected_pixels"] for row in per_frame)
    retained = sum(row["retained_protected_pixels"] for row in per_frame)
    loss_reason_totals = {
        key: sum(row["protected_loss_reasons"][key] for row in per_frame)
        for key in per_frame[0]["protected_loss_reasons"]
    }
    edge_ordering_supported = sum(row["edge_ordering_supported_pixels"] for row in per_frame)
    result_path = output / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "visible-surface-occlusion-canary-v71-v2",
        "artifact_revision": "R7_2",
        "status": "PASS_DEVELOPMENT_VISIBLE_SURFACE_ZBUFFER_CANARY",
        "session_id": args.session_id,
        "frame_ids": frame_ids.tolist(),
        "metrics": {
            "contact_band_pixels": total,
            "known_decision_coverage": known / total if total else 1.0,
            "unknown_pixel_ratio": 1.0 - known / total if total else 0.0,
            "protected_retention_pixel_weighted": retained / protected if protected else 1.0,
            "protected_pixels": protected,
            "retained_protected_pixels": retained,
            "protected_loss_reasons": loss_reason_totals,
            "edge_ordering_supported_pixels": edge_ordering_supported,
            "accuracy_reported": False,
        },
        "per_frame": per_frame,
        "gates": {
            "same_session_identity": True,
            "robot_optical_z": True,
            "robot_triangle_provenance": True,
            "object_three_instance_union_used_for_identity": False,
            "visible_surface_only": True,
            "hidden_object_completion": False,
            "silver_authority": False,
            "gold_accuracy": False,
        },
        "inputs": [ref(z_result_path), ref(depth_result_path), ref(mask_manifest_path), ref(source_video)],
        "code_closure": [
            ref(Path(__file__)),
            ref(ROOT / "src/chaoyang/pipeline/occlusion_compositor_v1.py"),
        ],
        "outputs": [ref(arrays_path), ref(montage_path), ref(video_path)],
        "video_probe": {
            "decoded_frames": decoded,
            "fps": source_fps,
            "width": frame_size[0],
            "height": frame_size[1],
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "authority": False,
        "control_ground_truth": False,
        "claim_limit": f"{len(frame_ids)}-frame observed-visible-surface Robot/object optical-Z diagnostic only. It does not infer hidden object geometry, solve contact occlusion, measure accuracy, grant Silver/Gold authority, or authorize Robot control/deployment.",
    })
    print(json.dumps(load(result_path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
