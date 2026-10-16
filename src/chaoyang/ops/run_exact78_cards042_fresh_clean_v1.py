"""Fresh Cards042 Clean from current role/object masks and pinned ProPainter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline.exact78_task_runtime_v1 import authorize, checked, verify_refs
from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease


FRAME_COUNT = 171
RADII = {"left_human": 24, "right_human": 18, "left_tracker": 60, "right_tracker": 60}


def save(path: Path, image: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"NO_CLOBBER_IMAGE:{path}")
    return artifact_ref(path)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def decode_video(path: Path, expected_frames: int, expected_size: tuple[int, int]) -> bool:
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1::-1] != expected_size:
                return False
            count += 1
    finally:
        capture.release()
    return count == expected_frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    supplied = load_json(args.packet)
    packet, root = authorize(supplied["task_id"])
    if packet != supplied:
        raise RuntimeError("PACKET_MISMATCH")
    verify_refs(packet["inputs"])
    verify_refs(packet["runtime_code"])
    role_result = load_json(checked(packet["inputs"]["role_result"]))
    object_result = load_json(checked(packet["inputs"]["object_result"]))
    if role_result.get("grade") != "B" or not role_result.get("downstream_authorized"):
        raise RuntimeError("ROLE_GRADE_B_REQUIRED")
    if object_result.get("grade") != "B" or not object_result.get("downstream_authorized"):
        raise RuntimeError("OBJECT_GRADE_B_REQUIRED")
    role_manifest = load_json(checked(packet["inputs"]["role_manifest"]))
    object_manifest = load_json(checked(packet["inputs"]["object_manifest"]))
    role_rows = {row["source_frame"]: row for row in role_manifest["frames"]}
    object_rows = {row["source_frame"]: row for row in object_manifest["frames"]}
    raw_frames = {int(frame): checked(ref) for frame, ref in packet["inputs"]["raw_frames"].items()}
    if set(role_rows) != set(range(FRAME_COUNT)) or set(object_rows) != set(range(FRAME_COUNT)) or set(raw_frames) != set(range(FRAME_COUNT)):
        raise RuntimeError("FRAME_SET_DRIFT")
    output = root/"clean"
    output.mkdir()
    started = time.monotonic()

    spatial = {role: [] for role in RADII}
    for frame in range(FRAME_COUNT):
        for role, radius in RADII.items():
            mask = cv2.imread(str(checked(role_rows[frame]["role_masks"][role])), cv2.IMREAD_GRAYSCALE) > 0
            spatial[role].append(dilate(mask, radius))

    prepare_rows = []
    total_removal = total_original = total_added = 0
    for frame in range(FRAME_COUNT):
        original_union = np.zeros((960,1280), dtype=bool)
        expanded_union = np.zeros_like(original_union)
        for role in RADII:
            raw_role = cv2.imread(str(checked(role_rows[frame]["role_masks"][role])), cv2.IMREAD_GRAYSCALE) > 0
            original_union |= raw_role
            expanded = spatial[role][frame].copy()
            if 0 < frame < FRAME_COUNT-1:
                expanded |= spatial[role][frame-1] & spatial[role][frame+1]
            expanded_union |= expanded
        object_mask = cv2.imread(str(checked(object_rows[frame]["mask"])), cv2.IMREAD_GRAYSCALE) > 0
        removal = expanded_union & ~object_mask
        image = cv2.imread(str(raw_frames[frame]), cv2.IMREAD_COLOR)
        if image is None or image.shape != (960,1280,3):
            raise RuntimeError(f"RAW_IMAGE_DOMAIN:{frame}")
        full_removal = save(output/"removal_masks"/f"{frame:06d}.png", removal.astype(np.uint8)*255)
        protected = save(output/"protected_object_masks"/f"{frame:06d}.png", object_mask.astype(np.uint8)*255)
        save(output/"propainter_input/frames"/f"{frame:06d}.png",
             cv2.resize(image, (960,720), interpolation=cv2.INTER_AREA))
        save(output/"propainter_input/masks"/f"{frame:06d}.png",
             cv2.resize(removal.astype(np.uint8)*255, (960,720), interpolation=cv2.INTER_NEAREST))
        added = int(np.logical_and(removal, ~original_union).sum())
        prepare_rows.append({"source_frame": frame, "raw": packet["inputs"]["raw_frames"][str(frame)],
                             "removal": full_removal, "protected_object": protected,
                             "original_role_union_pixels": int(original_union.sum()),
                             "expanded_removal_pixels": int(removal.sum()), "added_edge_shadow_pixels": added,
                             "published_removal_object_overlap_pixels": int(np.logical_and(removal, object_mask).sum())})
        total_removal += int(removal.sum()); total_original += int(original_union.sum()); total_added += added
    atomic_json(output/"PREPARE.json", {"schema_version": "exact78-cards042-fresh-clean-prepare-v1",
        "frame_count": FRAME_COUNT, "expansion": {"role_radii_px": RADII, "temporal_radius_frames": 1,
        "temporal_rule": "current dilation UNION intersection of symmetric neighbor dilations"},
        "formula": "expanded role union MINUS current physical object mask", "rows": prepare_rows,
        "totals": {"original_role_pixels": total_original, "removal_pixels": total_removal, "added_edge_shadow_pixels": total_added}})

    model_root = output/"propainter_model"
    log_path = output/"PROPAINTER_RUN.log"
    command = [sys.executable, "-B", str(checked(packet["inputs"]["propainter_entry"])),
               "--video", str(output/"propainter_input/frames"), "--mask", str(output/"propainter_input/masks"),
               "--output", str(model_root), "--width", "960", "--height", "720", "--mask_dilation", "4",
               "--ref_stride", "10", "--neighbor_length", "10", "--subvideo_length", "80", "--raft_iter", "20",
               "--save_fps", "30", "--save_frames", "--fp16"]
    lease = FileGpuLease(
        lease_path=REPO_ROOT/"_run/current/GPU_LEASE.json", lock_path=REPO_ROOT/"_run/current/GPU_LEASE.lock",
        task_id=packet["task_id"], attempt_id="attempt_0001", gpu_id=0,
        executor_epoch=packet["writer"]["executor_epoch"], fencing_token=packet["writer"]["fencing_token"])
    lease.__enter__()
    try:
        with log_path.open("xb") as log:
            completed = subprocess.run(command, cwd=REPO_ROOT/"vendor/ProPainter", stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, check=False)
    finally:
        lease.__exit__(*sys.exc_info())
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    known_writer_only = completed.returncode != 0 and "PyAVPlugin.write() got an unexpected keyword argument 'quality'" in log_text
    model_frames = sorted((model_root/"frames/frames").glob("*.png"))
    if [path.name for path in model_frames] != [f"{frame:04d}.png" for frame in range(FRAME_COUNT)]:
        raise RuntimeError(f"PROPAINTER_FRAME_SET:{completed.returncode}")
    if completed.returncode and not known_writer_only:
        raise RuntimeError(f"PROPAINTER_RETURN_CODE:{completed.returncode}")

    clean_video = cv2.VideoWriter(str(output/"clean.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280,960))
    review_video = cv2.VideoWriter(str(output/"review_3panel.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920,480))
    if not clean_video.isOpened() or not review_video.isOpened():
        raise RuntimeError("VIDEO_WRITER_OPEN")
    clean_rows = []
    outside_changed = protected_changed = 0
    for frame, prepared in enumerate(prepare_rows):
        raw = cv2.imread(str(raw_frames[frame]), cv2.IMREAD_COLOR)
        removal = cv2.imread(str(checked(prepared["removal"])), cv2.IMREAD_GRAYSCALE) > 0
        protected = cv2.imread(str(checked(prepared["protected_object"])), cv2.IMREAD_GRAYSCALE) > 0
        generated = cv2.imread(str(model_frames[frame]), cv2.IMREAD_COLOR)
        if generated is None or generated.shape != (720,960,3):
            raise RuntimeError(f"PROPAINTER_OUTPUT_DOMAIN:{frame}")
        generated = cv2.resize(generated, (1280,960), interpolation=cv2.INTER_CUBIC)
        clean = raw.copy(); clean[removal] = generated[removal]
        outside = int(np.count_nonzero(np.any(clean[~removal] != raw[~removal], axis=1)))
        protected_delta = int(np.count_nonzero(np.any(clean[protected] != raw[protected], axis=1)))
        outside_changed += outside; protected_changed += protected_delta
        clean_ref = save(output/"clean_frames"/f"{frame:06d}.png", clean)
        source = np.zeros((960,1280), np.uint8); source[protected] = 2; source[removal] = 3
        source_ref = save(output/"pixel_source"/f"{frame:06d}.png", source)
        clean_video.write(clean)
        overlay = raw.copy(); overlay[removal] = (.35*overlay[removal]+.65*np.asarray([255,0,255])).astype(np.uint8)
        panels = [cv2.resize(value, (640,480), interpolation=cv2.INTER_AREA) for value in (raw, overlay, clean)]
        review_video.write(np.concatenate(panels, axis=1))
        clean_rows.append({"source_frame": frame, "clean": clean_ref, "pixel_source": source_ref,
                           "outside_changed_pixels": outside, "protected_changed_pixels": protected_delta,
                           "synthetic_pixels": int(removal.sum())})
    clean_video.release(); review_video.release()
    hard = {"frame_count_exact": len(clean_rows) == FRAME_COUNT,
            "all_removal_masks_nonempty": all(row["expanded_removal_pixels"] > 0 for row in prepare_rows),
            "removal_object_overlap_zero": all(row["published_removal_object_overlap_pixels"] == 0 for row in prepare_rows),
            "outside_changed_zero": outside_changed == 0, "protected_changed_zero": protected_changed == 0,
            "propainter_frames_exact": len(model_frames) == FRAME_COUNT,
            "clean_video_full_decode": decode_video(output/"clean.mp4", FRAME_COUNT, (1280,960)),
            "review_video_full_decode": decode_video(output/"review_3panel.mp4", FRAME_COUNT, (1920,480))}
    atomic_json(output/"FRAME_MANIFEST.json", {"schema_version": "exact78-cards042-fresh-clean-manifest-v1",
                "session": "play_cards_0902_042", "rows": clean_rows})
    passed = all(hard.values())
    result = {"schema_version": "exact78-cards042-fresh-clean-result-v1",
              "status": "PASS_EXECUTION_PENDING_VISUAL_REVIEW" if passed else "FAILED_QUALITY_C",
              "downstream_authorized": False, "human_review": "PENDING", "quality_adopted": False,
              "hard_gates": hard, "metrics": {"frame_count": len(clean_rows), "total_synthetic_pixels": total_removal,
              "total_added_edge_shadow_pixels": total_added, "outside_changed_pixels": outside_changed,
              "protected_changed_pixels": protected_changed, "vendor_returncode": completed.returncode,
              "known_preview_writer_only_failure_bypassed": known_writer_only, "wall_seconds": time.monotonic()-started},
              "artifacts": {"prepare": artifact_ref(output/"PREPARE.json"),
                            "frame_manifest": artifact_ref(output/"FRAME_MANIFEST.json"),
                            "clean_video": artifact_ref(output/"clean.mp4"),
                            "review_video": artifact_ref(output/"review_3panel.mp4"), "log": artifact_ref(log_path)},
              "source_enum": {"0": "TARGET_RAW", "2": "PROTECTED_OBJECT_TARGET_RAW", "3": "SYNTHETIC_PROPAINTER"},
              "training_eligible": False, "deletion_authorized": False,
              "claim_limit": "Fresh synthetic Clean candidate only; visual review required, no observed-background or training truth claim."}
    atomic_json(output/"RESULT.json", result)
    atomic_json(root/"RESULT.json", {"status": result["status"], "clean_result": artifact_ref(output/"RESULT.json"),
                                      "baseline_adopted": False, "deletion_authorized": False})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
