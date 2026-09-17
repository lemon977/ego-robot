#!/usr/bin/env python3
"""Build full-session Clean contact-protection development candidates.

The candidate is derived without rerunning ProPainter: pixels removed only by
the predecessor's broad dilation are restored byte-for-byte from Raw, while
the smaller contact-aware removal region keeps the predecessor Clean result.
Directly observed task-object pixels are also restored from Raw.  Frames where
the task-object mask is invalid remain explicitly unknown; no amodal texture is
invented and no Clean authority is published.
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
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.ops.audit_clean_contact_protection_successor_v1 import proposed, unions, mask


CASES = {
    "Poker245": {
        "manifest": PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/sessions/play_cards_0903_245/expanded_role_handoff/FRAME_MANIFEST.json",
        "clean_result": PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/propainter_v1/play_cards_0903_245/RESULT.json",
    },
    "Chips039": {
        "manifest": PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/sessions/get_potato_chips_0902_039/expanded_role_handoff/FRAME_MANIFEST.json",
        "clean_result": PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/propainter_v1/get_potato_chips_0902_039/RESULT.json",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256(value)}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def add_text(frame: np.ndarray, rows: list[tuple[int, int, str, tuple[int, int, int], int]]) -> np.ndarray:
    value = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(value)
    fonts: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    for x, y, label, bgr, size in rows:
        fonts.setdefault(size, font(size))
        draw.text((x, y), label, font=fonts[size], fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.asarray(value), cv2.COLOR_RGB2BGR)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    return writer


def run_case(name: str, spec: dict, output_root: Path) -> dict:
    manifest_path = Path(spec["manifest"])
    clean_result_path = Path(spec["clean_result"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_result = json.loads(clean_result_path.read_text(encoding="utf-8"))
    clean_path = Path(clean_result["artifacts"]["clean_synthetic_master"]["path"])
    rows = manifest["frames"]
    first = cv2.imread(rows[0]["source_rgb"]["path"])
    if first is None:
        raise RuntimeError("cannot decode first Raw frame")
    height, width = first.shape[:2]
    fps = 30.0
    capture = cv2.VideoCapture(str(clean_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode Clean master: {clean_path}")
    if int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) != len(rows):
        raise RuntimeError("Clean frame-count mismatch")

    case_root = output_root / name
    case_root.mkdir(parents=True, exist_ok=False)
    master_path = case_root / f"{name}_Clean接触保护候选_MASTER.mp4"
    review_path = case_root / f"{name}_Clean接触保护前后_全片.mp4"
    master_writer = open_writer(master_path, fps, (width, height))
    review_writer = open_writer(review_path, fps, (1280, 720))

    total_old = total_new = total_restored = total_visible_object = 0
    object_observed = 0
    try:
        for frame_id, row in enumerate(rows):
            raw = cv2.imread(row["source_rgb"]["path"])
            ok, clean = capture.read()
            if raw is None or not ok or clean.shape[:2] != (height, width):
                raise RuntimeError(f"frame decode failed at {frame_id}")
            human, tracker, obj = unions(row)
            old_removal = mask(row["clean_removal_object_protected"])
            new_removal = proposed(human, tracker, obj)
            restore = old_removal & ~new_removal
            candidate = clean.copy()
            candidate[restore] = raw[restore]
            candidate[obj] = raw[obj]
            master_writer.write(candidate)

            raw_small = cv2.resize(raw, (640, 360), interpolation=cv2.INTER_AREA)
            clean_small = cv2.resize(clean, (640, 360), interpolation=cv2.INTER_AREA)
            candidate_small = cv2.resize(candidate, (640, 360), interpolation=cv2.INTER_AREA)
            diagnostic = raw.copy()
            diagnostic[old_removal] = (0.55 * diagnostic[old_removal] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            diagnostic[new_removal] = (0.55 * diagnostic[new_removal] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
            diagnostic[obj] = (0.30 * diagnostic[obj] + 0.70 * np.array([0, 255, 0])).astype(np.uint8)
            diagnostic_small = cv2.resize(diagnostic, (640, 360), interpolation=cv2.INTER_AREA)
            canvas = np.vstack((np.hstack((raw_small, clean_small)), np.hstack((candidate_small, diagnostic_small))))
            status = "当前物体Mask有效" if obj.any() else "物体Mask无效：隐藏外观UNKNOWN"
            canvas = add_text(
                canvas,
                [
                    (8, 8, f"Raw｜帧 {frame_id:04d}", (255, 255, 255), 22),
                    (648, 8, "现行 Clean", (255, 255, 255), 22),
                    (8, 368, "接触保护 successor 候选", (255, 255, 255), 22),
                    (648, 368, "红=旧删除 黄=新删除 绿=真实可见物体", (255, 255, 255), 19),
                    (8, 685, status, (0, 255, 0) if obj.any() else (0, 165, 255), 18),
                ],
            )
            review_writer.write(canvas)
            total_old += int(old_removal.sum())
            total_new += int(new_removal.sum())
            total_restored += int(restore.sum())
            total_visible_object += int(obj.sum())
            object_observed += int(obj.any())
        ok, _ = capture.read()
        if ok:
            raise RuntimeError("Clean master has extra frame")
    finally:
        capture.release()
        master_writer.release()
        review_writer.release()

    return {
        "case": name,
        "task": manifest["task"],
        "session": manifest["session"],
        "frame_count": len(rows),
        "fps": fps,
        "object_observed_frame_coverage": object_observed / len(rows),
        "pixels": {
            "predecessor_removal": total_old,
            "successor_removal": total_new,
            "restored_raw_outside_successor_removal": total_restored,
            "direct_visible_object_restored": total_visible_object,
            "removal_reduction_ratio": 1.0 - total_new / max(total_old, 1),
        },
        "inputs": {"manifest": artifact(manifest_path), "clean_result": artifact(clean_result_path), "clean_master": artifact(clean_path)},
        "outputs": {"candidate_master": artifact(master_path), "review_video": artifact(review_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    reports = [run_case(name, spec, output_root) for name, spec in CASES.items()]
    result = {
        "schema_version": "clean-contact-protected-fullsession-candidate-v1",
        "status": "PASS_DEVELOPMENT_FULLSESSION_CANDIDATES_NOT_AUTHORITY",
        "method": {
            "near_object_band_px": 20,
            "human_dilation_near_far_px": [4, 8],
            "tracker_dilation_near_far_px": [8, 20],
            "undo_predecessor_excess_dilation_from_raw": True,
            "restore_direct_visible_object_from_raw": True,
            "invent_amodal_object_texture": False
        },
        "reports": reports,
        "remaining_blocker": "Object-invalid frames need a causal temporal object atlas or are excluded from training; the candidate never hallucinates hidden card/chip texture.",
        "authority": False,
        "claim_limit": "Full-session visual development candidates only. They do not supersede Clean B, prove hidden-object correctness, or authorize Robotized training RGB."
    }
    result_path = output_root / "RESULT.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"result": artifact(result_path), "reports": reports}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
