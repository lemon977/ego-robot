#!/usr/bin/env python3
"""CPU-only, offline visual audit of the immutable Poker015 SAM3.1 224-frame run.

This tool never grants Mask authority.  The selected SAM object id is a tracker
handle, not proof of physical card identity or face-side identity.
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

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


ROOT = Path(__file__).resolve().parents[3]
SESSION = "play_cards_0901_015"
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_sam31_native_reentry_research_v1/attempts/attempt_0001"
REFERENCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_next_cpu_freeze_v1/attempts/attempt_0001/MASK_NEXT_EVALUATION_REFERENCE.json"
RAW_ROOT = Path("/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards") / SESSION / "preprocess/all_data"
FRAME_COUNT = 224
FACE_BACK_REVIEW_FRAMES = frozenset((0, 2, 3, 79, 80, 213, 214, 220, 221))
FACE_FRONT_REVIEW_FRAMES = frozenset((222, 223))
FOCUS_FRAMES = (79, 80, 213, 214, 220, 221, 222, 223)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_ref(ref: dict) -> Path:
    path = Path(ref["path"]).resolve(strict=True)
    if path.stat().st_size != ref["bytes"] or _sha(path) != ref["sha256"]:
        raise ValueError(f"artifact reference mismatch: {path}")
    return path


def _label(image: np.ndarray, label: str, y: int, scale: float = 0.55) -> None:
    cv2.putText(image, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _face_hint(frame: int) -> str:
    if frame in FACE_BACK_REVIEW_FRAMES:
        return "BACK_APPEARING_REVIEW_ONLY"
    if frame in FACE_FRONT_REVIEW_FRAMES:
        return "FRONT_APPEARING_REVIEW_ONLY"
    return "NOT_REVIEWED"


def _panels(raw: np.ndarray, selected: np.ndarray, uncertain: np.ndarray, frame: int, area: int) -> np.ndarray:
    small = (640, 480)
    raw_panel = cv2.resize(raw, small, interpolation=cv2.INTER_AREA)
    overlay = raw.copy()
    overlay[selected] = (0.45 * overlay[selected] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    overlay[uncertain] = (0.2 * overlay[uncertain] + 0.8 * np.array([0, 255, 255])).astype(np.uint8)
    overlay = cv2.resize(overlay, small, interpolation=cv2.INTER_AREA)
    evidence = np.zeros_like(raw_panel)
    reduced_visible = cv2.resize(selected.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST).astype(bool)
    reduced_uncertain = cv2.resize(uncertain.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST).astype(bool)
    evidence[reduced_visible] = (255, 0, 255)
    evidence[reduced_uncertain] = (0, 255, 255)
    _label(raw_panel, f"RAW {SESSION} f={frame:03d}", 30)
    _label(overlay, f"SAM track=0 area={area} px", 30)
    _label(overlay, "yellow=uncertain boundary / transition", 462, 0.48)
    _label(evidence, "PHYSICAL_ID=UNVERIFIED", 30)
    _label(evidence, f"FACE_ID=UNKNOWN; hint={_face_hint(frame)}", 63, 0.45)
    _label(evidence, "VISIBLE_MASK=SAM_CANDIDATE", 95, 0.48)
    _label(evidence, "HIDDEN_SURFACE=UNKNOWN", 128, 0.48)
    _label(evidence, "OFFLINE DIAGNOSTIC - NOT TRAINING", 462, 0.45)
    return np.concatenate((raw_panel, overlay, evidence), axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)

    source_result_path = SOURCE / "RESULT.json"
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_result.get("session_id") != SESSION or source_result.get("frames") != FRAME_COUNT:
        raise ValueError("unexpected source run identity or frame count")
    packed_path = _verify_ref(source_result["outputs"]["packed_masks"])
    _verify_ref(source_result["inputs"]["independent_frame_reference"])
    if _sha(REFERENCE) != source_result["inputs"]["independent_frame_reference"]["sha256"]:
        raise ValueError("frozen frame reference mismatch")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if reference["pixel_gold_available"] or reference["accuracy_authorized"]:
        raise ValueError("unexpected reference authority")
    frozen = set(reference["tasks"]["poker_object"]["diagnostic_frames"])
    if not set(FOCUS_FRAMES).issubset(frozen | {222}):
        raise ValueError("focus frames drifted from frozen reference plus frame222")
    with np.load(packed_path, allow_pickle=False) as packed:
        ids = packed["frame_ids"]
        bits = packed["packed"]
    if not np.array_equal(ids, np.arange(FRAME_COUNT)) or bits.shape != (FRAME_COUNT, 153600):
        raise ValueError("packed masks are not the expected 224 x 960 x 1280 run")

    video_path = output / "POKER015_SAM31_224_实例牌面对照_OFFLINE.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 480))
    if not writer.isOpened():
        raise RuntimeError("video writer failed")
    raw_refs: list[dict] = []
    rows: list[dict] = []
    focus_panels: list[np.ndarray] = []
    uncertainty_bits: list[np.ndarray] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    previous_area: int | None = None
    try:
        for frame in range(FRAME_COUNT):
            raw_path = (RAW_ROOT / f"{frame:05d}" / "rgb.png").resolve(strict=True)
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None or raw.shape[:2] != (960, 1280):
                raise ValueError(f"bad source frame {frame}")
            mask = np.unpackbits(bits[frame], count=960 * 1280).reshape(960, 1280).astype(bool)
            area = int(mask.sum())
            if area != source_result["rows"][frame]["area_pixels"]:
                raise ValueError(f"packed mask / source RESULT area conflict at {frame}")
            edge = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            jump = previous_area is not None and max(area, previous_area) / max(1, min(area, previous_area)) > 2.0
            # A jump may be a real flip or a tracking error. It is not a Gold judgment.
            uncertain = edge | (mask if jump else np.zeros_like(mask))
            uncertainty_bits.append(np.packbits(uncertain.reshape(-1)))
            previous_area = area
            raw_refs.append(artifact_ref(raw_path))
            rows.append({
                "frame_index": frame,
                "raw_source_index": frame,
                "raw_source_path": str(raw_path),
                "raw_source_sha256": raw_refs[-1]["sha256"],
                "selected_mask_source": str(packed_path),
                "selected_mask_source_frame": frame,
                "selected_mask_area_pixels": area,
                "sam_tracker_object_id": 0,
                "physical_instance_id": None,
                "physical_instance_status": "UNVERIFIED",
                "face_id": None,
                "face_identity_status": "UNKNOWN",
                "face_visual_hint": _face_hint(frame),
                "visible_region_source": "SAM31_SELECTED_MASK_DEVELOPMENT_CANDIDATE",
                "hidden_or_occluded_region_status": "UNKNOWN_NOT_ESTIMATED",
                "uncertainty_region_source": "11PX_MORPHOLOGICAL_BOUNDARY_OR_AREA_JUMP_GT_2X",
                "uncertainty_pixels": int(uncertain.sum()),
                "area_jump_gt_2x": bool(jump),
                "may_train": False,
            })
            panel = _panels(raw, mask, uncertain, frame, area)
            writer.write(panel)
            if frame in FOCUS_FRAMES:
                focus_panels.append(panel)
    finally:
        writer.release()

    focus_path = output / "POKER015_79_80_213_223_含222_困难帧.png"
    if not cv2.imwrite(str(focus_path), np.concatenate(focus_panels, axis=0)):
        raise RuntimeError("focus image encode failed")
    uncertain_path = output / "PACKED_UNCERTAINTY_REGIONS.npz"
    np.savez_compressed(uncertain_path, frame_ids=np.arange(FRAME_COUNT, dtype=np.int32), packed=np.stack(uncertainty_bits))
    raw_manifest_path = output / "RAW_FRAME_SOURCE_MANIFEST.json"
    atomic_json(raw_manifest_path, {"session_id": SESSION, "frames": FRAME_COUNT, "frame_sources": raw_refs})
    rows_path = output / "FRAME_PROVENANCE_AND_UNCERTAINTY.json"
    atomic_json(rows_path, {"session_id": SESSION, "input_mode": "OFFLINE_FULL_SEQUENCE_DIAGNOSTIC", "frames": rows})

    result = {
        "schema_version": "sam31-poker015-224-cpu-diagnostic-v1",
        "created_at": now_iso(),
        "session_id": SESSION,
        "status": "PASSED_DEVELOPMENT_DIAGNOSTIC",
        "authority_promoted": False,
        "training_eligible": False,
        "input_mode": "OFFLINE_FULL_SEQUENCE_DIAGNOSTIC",
        "frame_count": FRAME_COUNT,
        "sam_track_id": 0,
        "physical_instance_id": None,
        "face_id": None,
        "pixel_gold_available": False,
        "same_physical_instance_proven": False,
        "area_jump_gt_2x_frames": [row["frame_index"] for row in rows if row["area_jump_gt_2x"]],
        "manual_visual_review_note": "Frame222 raw appears to expose card front; frame221 shows edge/occlusion. This appearance is not a verified physical/face identity label.",
        "claim_limit": "Offline selected-mask visualization and deterministic morphology only; no causal training input, pixel accuracy, or physical/face identity authority.",
        "inputs": {"source_result": artifact_ref(source_result_path), "packed_selected_masks": artifact_ref(packed_path), "frozen_frame_reference": artifact_ref(REFERENCE), "raw_frame_manifest": artifact_ref(raw_manifest_path)},
        "outputs": {"video": artifact_ref(video_path), "focus_image": artifact_ref(focus_path), "frame_provenance": artifact_ref(rows_path), "packed_uncertainty_regions": artifact_ref(uncertain_path)},
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(output / "RESULT.json", result)
    (output / "DECISION.md").write_text(
        "# Poker015 SAM3.1 224-frame CPU diagnostic\n\n"
        "This is an offline, non-authoritative diagnostic. SAM tracker object ID 0 is not a proven physical card ID; face ID remains UNKNOWN.\n\n"
        "Frame 222 has a >2x area jump while the raw image appears to expose the card front. The full selected mask is marked uncertain there, not accepted as corrected segmentation.\n\n"
        "The visible-region mask is the SAM candidate; hidden surface was not estimated. The yellow uncertainty region is a deterministic 11-pixel boundary band plus full mask on >2x area jumps.\n\n"
        "No future frames, atlas, or attachment hypothesis are used to infer training pixels. The source SAM full-sequence propagation itself may use future context, so this bundle is OFFLINE_FULL_SEQUENCE_DIAGNOSTIC and training_eligible=false.\n\n"
        "Next decision requires the two frozen B-session full regressions and independent instance/face review before any successor authority proposal.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "session_id": SESSION, "area_jump_frames": result["area_jump_gt_2x_frames"], "result": str(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
