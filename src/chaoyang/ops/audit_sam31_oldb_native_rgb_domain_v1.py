#!/usr/bin/env python3
"""Bounded old-B MP4 versus native SAM3.1 PNG image-domain audit.

No inference, authority change, or pixel Gold is implied.  Only frozen
play_cards_0901_001/005 frames 0 and 112 are evaluated.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso, validate_artifact_ref  # noqa: E402

RAW = Path("/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards")
OLD = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_task_object_identity_v1/poker"
NEW = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_sam31_cpu_frames_full_b_v1"
SESSIONS = {"play_cards_0901_001": 645, "play_cards_0901_005": 520}
FRAMES = (0, 112)


def array_sha(frame: np.ndarray) -> str:
    frame = np.ascontiguousarray(frame)
    digest = hashlib.sha256(frame.dtype.str.encode() + b"\0")
    digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
    digest.update(frame.tobytes())
    return digest.hexdigest()


def mask_geometry(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"area_pixels": 0, "bbox_xyxy": None, "centroid_xy": None}
    return {
        "area_pixels": int(len(xs)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [round(float(xs.mean()), 3), round(float(ys.mean()), 3)],
    }


def overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    shown = frame.copy()
    shown[mask] = (shown[mask].astype(np.uint16) * 2 // 5 + np.asarray(color, dtype=np.uint16) * 3 // 5).astype(np.uint8)
    return shown


def card(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    return image


def analyze_session(session: str, count: int) -> tuple[dict, list[np.ndarray]]:
    old_result_path = OLD / session / "RESULT.json"
    old_result = json.loads(old_result_path.read_text(encoding="utf-8"))
    if old_result.get("frame_count") != count or old_result.get("grade") != "B":
        raise RuntimeError(f"old B identity/grade mismatch: {session}")
    old_manifest_ref = old_result["artifacts"]["manifest"]
    if validate_artifact_ref(old_manifest_ref):
        raise RuntimeError(f"old B manifest closure failed: {session}")
    old_manifest = json.loads(Path(old_manifest_ref["path"]).read_text(encoding="utf-8"))
    old_input_path = OLD / session / "INPUT_SNAPSHOT.json"
    old_input = json.loads(old_input_path.read_text(encoding="utf-8"))
    mp4_ref = old_input["selected_rgb"]
    if validate_artifact_ref(mp4_ref):
        raise RuntimeError(f"old selected RGB MP4 closure failed: {session}")
    mp4_path = Path(mp4_ref["path"])
    native_result_path = NEW / "attempts" / session / "attempt_0001/RESULT.json"
    native_result = json.loads(native_result_path.read_text(encoding="utf-8"))
    if native_result.get("session_id") != session or native_result.get("frames") != count:
        raise RuntimeError(f"native identity/count mismatch: {session}")
    png_closure_path = NEW / f"SOURCE_RGB_CLOSURE_{session[-3:]}.json"
    png_closure = json.loads(png_closure_path.read_text(encoding="utf-8"))
    if len(png_closure["frames"]) != count:
        raise RuntimeError(f"new per-frame RGB closure count mismatch: {session}")
    packed_ref = native_result["outputs"]["packed_masks"]
    if validate_artifact_ref(packed_ref):
        raise RuntimeError(f"new packed mask closure failed: {session}")
    with np.load(packed_ref["path"], allow_pickle=False) as payload:
        frame_ids = payload["frame_ids"]
        packed = payload["packed"]
        if frame_ids.shape != (count,) or packed.shape[0] != count:
            raise RuntimeError(f"new packed mask shape mismatch: {session}")
        native_masks = {
            frame: np.unpackbits(packed[frame], count=960 * 1280).reshape(960, 1280).astype(bool)
            for frame in FRAMES
        }

    pngs: dict[int, np.ndarray] = {}
    png_refs: dict[int, dict] = {}
    reduced: dict[int, np.ndarray] = {}
    for frame in FRAMES:
        reference = png_closure["frames"][frame]
        if int(reference["source_frame_id"]) != frame or validate_artifact_ref(reference):
            raise RuntimeError(f"new PNG closure/identity failed: {session} frame {frame}")
        png = card(Path(reference["path"]), "native PNG")
        if png.shape != (960, 1280, 3):
            raise RuntimeError(f"unexpected new PNG shape: {png.shape}")
        pngs[frame] = png
        png_refs[frame] = reference
        reduced[frame] = cv2.resize(png, (160, 120), interpolation=cv2.INTER_AREA).astype(np.int16)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
        raise RuntimeError(f"old MP4 decode/count failed: {session}")
    best = {frame: {"mae_160x120": float("inf"), "mp4_frame_id": None} for frame in FRAMES}
    saved: dict[int, np.ndarray] = {}
    try:
        for mp4_frame_id in range(count):
            ok, image = cap.read()
            if not ok or image.shape != (960, 1280, 3):
                raise RuntimeError(f"old MP4 frame decode/shape failed: {session} {mp4_frame_id}")
            if mp4_frame_id in FRAMES:
                saved[mp4_frame_id] = image.copy()
            resized = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA).astype(np.int16)
            for target in FRAMES:
                mae = float(np.mean(np.abs(resized - reduced[target])))
                if mae < best[target]["mae_160x120"]:
                    best[target] = {"mae_160x120": round(mae, 6), "mp4_frame_id": mp4_frame_id}
    finally:
        cap.release()

    rows: list[dict] = []
    montage_rows: list[np.ndarray] = []
    for frame in FRAMES:
        old_row = old_manifest["frames"][frame]
        old_decoded = saved[frame]
        expected_old_sha = old_row["selected_rgb_decoded_sha256"]
        observed_old_sha = array_sha(old_decoded)
        if observed_old_sha != expected_old_sha:
            raise RuntimeError(f"old MP4 decode does not reproduce frozen old B frame: {session} {frame}")
        mask_ref = old_row["physical_instances"]["0"]["mask"]
        if validate_artifact_ref(mask_ref):
            raise RuntimeError(f"old B mask closure failed: {session} {frame}")
        old_mask = cv2.imread(mask_ref["path"], cv2.IMREAD_GRAYSCALE)
        if old_mask is None or old_mask.shape != (960, 1280):
            raise RuntimeError(f"old B mask decode/shape failed: {session} {frame}")
        old_mask = old_mask > 0
        native_mask = native_masks[frame]
        png = pngs[frame]
        difference = np.abs(old_decoded.astype(np.int16) - png.astype(np.int16))
        overlap = int(np.logical_and(old_mask, native_mask).sum())
        union = int(np.logical_or(old_mask, native_mask).sum())
        row = {
            "session_id": session,
            "frame_id": frame,
            "old_B_mp4_decoded_array_sha256": observed_old_sha,
            "native_png_decoded_array_sha256": array_sha(png),
            "old_B_mp4_shape": list(old_decoded.shape),
            "native_png_shape": list(png.shape),
            "same_index_rgb_mae": round(float(difference.mean()), 6),
            "same_index_rgb_changed_channels": int(np.count_nonzero(difference)),
            "same_index_rgb_channel_count": int(difference.size),
            "best_mp4_match_over_full_clip_downsampled": best[frame],
            "old_B_mask_observed": bool(old_row["physical_instances"]["0"]["observed"]),
            "old_B_mask": {**mask_geometry(old_mask), "file": mask_ref},
            "native_mask": mask_geometry(native_mask),
            "old_B_vs_native_mask_iou_internal": round(overlap / union, 6) if union else None,
            "new_png": png_refs[frame],
        }
        rows.append(row)
        tiles = [old_decoded, png, overlay(old_decoded, old_mask, (255, 255, 0)), overlay(png, native_mask, (255, 0, 255))]
        resized_tiles = [cv2.resize(tile, (480, 360), interpolation=cv2.INTER_AREA) for tile in tiles]
        strip = np.concatenate(resized_tiles, axis=1)
        labels = ("OLD MP4", "NEW PNG", "OLD B MASK", "NATIVE MASK")
        for index, label in enumerate(labels):
            cv2.putText(strip, f"{session} f={frame} {label}", (index * 480 + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        montage_rows.append(strip)

    return {
        "session_id": session,
        "frame_count": count,
        "old_B_result": artifact_ref(old_result_path),
        "old_B_input_snapshot": artifact_ref(old_input_path),
        "old_B_selected_rgb_mp4": mp4_ref,
        "old_B_manifest": old_manifest_ref,
        "native_result": artifact_ref(native_result_path),
        "native_source_rgb_closure": artifact_ref(png_closure_path),
        "native_packed_masks": packed_ref,
        "frames": rows,
    }, montage_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sessions = []
    strips = []
    for session, count in SESSIONS.items():
        record, rows = analyze_session(session, count)
        sessions.append(record)
        strips.extend(rows)
    montage = output / "POKER001_005_OLD_MP4_VS_NATIVE_PNG_FRAMES0_112.png"
    if not cv2.imwrite(str(montage), np.concatenate(strips, axis=0)):
        raise RuntimeError("failed to write comparison montage")
    result = {
        "schema_version": "chaoyang-sam31-oldb-native-rgb-domain-audit-v1",
        "task_id": "research_sam31_oldb_native_rgb_domain_20260916",
        "created_at": now_iso(),
        "status": "PASSED_DEVELOPMENT_DIAGNOSTIC",
        "sessions": sessions,
        "montage": artifact_ref(montage),
        "authority_promoted": False,
        "training_eligible": False,
        "claim_limit": "Exact producer RGB binding and a bounded image-domain comparison; old B is not Gold and no physical-card/face accuracy is established.",
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "result": artifact_ref(output / "RESULT.json"), "montage": result["montage"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
