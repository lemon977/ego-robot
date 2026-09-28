#!/usr/bin/env python3
"""Prepare and execute one explicitly non-adopted R2 ProPainter candidate.

The conservative Scene mask is structurally valid but lacks trusted visible
object protection and complete device evidence.  This runner therefore keeps
the result CANDIDATE_ONLY/REJECTED_QUALITY while providing an actual model
execution for root-cause review.  It never supplies geometry or training data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, REPO, TASK, ref, write_json
from chaoyang.ops.run_v5_scene import _mask_files, load as load_json, merge_roles
from chaoyang.pipeline.v5_scene import (
    build_conservative_repair_window,
    build_object_protected_repair_window,
    composite_clean,
)


V5 = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"
IMAGE_SIZE = (960, 720)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _image(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), value):
        raise RuntimeError(f"IMAGE_WRITE:{path}")


def _gpu_guard() -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    lease_path = REPO / "_run/current/GPU_LEASE.json"
    if not visible or not lease_path.is_file():
        raise RuntimeError("GPU_LEASE_REQUIRED")
    lease = load_json(lease_path)
    if lease.get("status") != "ACQUIRED" or lease.get("task_id") != f"{TASK}:scene":
        raise RuntimeError("GPU_LEASE_OWNER")
    if lease.get("gpu_process_pid") != os.getpid() or str(lease.get("gpu_id")) != visible:
        raise RuntimeError("GPU_LEASE_PROCESS")
    if not lease.get("fencing_token"):
        raise RuntimeError("GPU_LEASE_FENCE")
    return {"lease": ref(lease_path), "fencing_token": lease["fencing_token"]}


def paths(short: str, strategy: str = "conservative") -> tuple[Path, Path]:
    specs = {
        "031": "DIAG_031.json",
        "007": "DIAG_007.json",
        "103": "POST_103.json",
        "042": "POST_042.json",
    }
    if short not in specs:
        raise ValueError("ONLY_FIXED_R2_SCENE_SESSIONS")
    wave = {"conservative": "wave4", "object_protected": "wave5"}.get(strategy)
    if wave is None:
        raise ValueError("UNKNOWN_SCENE_STRATEGY")
    root = ATTEMPT / f"lanes/lane1_scene/clean_candidate_{short}_{wave}"
    return root, V5 / "lanes/scene" / specs[short]


def prepare(short: str, strategy: str = "conservative") -> Path:
    root, spec_path = paths(short, strategy)
    result_path = root / "PREP_RESULT.json"
    if result_path.is_file():
        return root / "SCENE_PREP_MANIFEST.json"
    if root.exists():
        raise FileExistsError(f"PARTIAL_PREP:{root}")
    spec = load_json(spec_path)
    domain_path = Path(spec["domain_manifest"])
    domain = load_json(domain_path)
    mask_values = spec.get("mask_manifests") or [spec["mask_manifest"]]
    mask_paths = [Path(value) for value in mask_values]
    mask_docs = [load_json(value) for value in mask_paths]
    count = int(domain["frame_count"])
    shape = (int(domain["height"]), int(domain["width"]))
    if count != int(spec.get("frame_count", count)) or len(domain["frames"]) != count:
        raise ValueError("FULL_TIMELINE_REQUIRED")
    for folder in ("frames", "model_masks", "write", "protect", "unknown"):
        (root / "prep" / folder).mkdir(parents=True, exist_ok=False)
    cache: dict[int, object] = {}
    rows = []
    object_overlap = 0
    device_frames = 0
    for frame in range(count):
        start, stop = max(0, frame - 2), min(count, frame + 3)
        for index in range(start, stop):
            if index not in cache:
                cache[index] = merge_roles(_mask_files(mask_docs, index, shape), shape)
        for index in list(cache):
            if index < start:
                del cache[index]
        roles = cache[frame]
        builder = (build_conservative_repair_window if strategy == "conservative"
                   else build_object_protected_repair_window)
        masks = builder([cache[index] for index in range(start, stop)],
                        frame - start, frame, support_margin=4)
        raw_path = Path(domain["frames"][frame]["rgb"])
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None or raw.shape[:2] != shape:
            raise ValueError(f"RAW_DOMAIN:{frame}")
        resized = cv2.resize(raw, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        # Model-context support may be slightly wider than M_write.  Expand by
        # one source pixel before a non-integer resize so a one-pixel cable or
        # fingertip cannot disappear; final write authority is unchanged.
        model_context = cv2.dilate(masks["context_exclude"].astype(np.uint8),
                                   np.ones((3, 3), np.uint8))
        small = cv2.resize(model_context, IMAGE_SIZE,
                           interpolation=cv2.INTER_NEAREST)
        small = cv2.dilate(small, np.ones((3, 3), np.uint8))
        restored = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if np.any(masks["context_exclude"] & ~restored):
            raise ValueError(f"CONTEXT_LOST_AT_MODEL_SCALE:{frame}")
        _image(root / "prep/frames" / f"{frame:06d}.png", resized)
        _image(root / "prep/model_masks" / f"{frame:06d}.png", small * 255)
        _image(root / "prep/write" / f"{frame:06d}.png", masks["write"].astype(np.uint8) * 255)
        _image(root / "prep/protect" / f"{frame:06d}.png", masks["protect"].astype(np.uint8) * 255)
        _image(root / "prep/unknown" / f"{frame:06d}.png", masks["unknown"].astype(np.uint8) * 255)
        object_overlap += int(masks["stats"]["object_overlap_px"])
        device_frames += int(roles.device.any())
        rows.append({
            "frame_id": frame,
            "raw": str(raw_path.resolve()),
            "write": str((root / "prep/write" / f"{frame:06d}.png").resolve()),
            "protect": str((root / "prep/protect" / f"{frame:06d}.png").resolve()),
            "unknown": str((root / "prep/unknown" / f"{frame:06d}.png").resolve()),
            "capture_time": domain["frames"][frame].get("capture_time"),
            "stats": masks["stats"],
        })
    manifest = {
        "schema_version": "HUMAN_TO_ROBOT_R2_SCENE_CANDIDATE_PREP_V1",
        "task_id": TASK,
        "session_id": spec["session_id"],
        "frame_count": count,
        "image_domain": domain["image_domain"],
        "output_size": [shape[1], shape[0]],
        "inpaint_size": list(IMAGE_SIZE),
        "source_domain": ref(domain_path),
        "source_masks": [ref(path) for path in mask_paths],
        "rows": rows,
        "strategy": strategy,
        "M_remove": "conservative semantic base plus bounded local repair",
        "M_write": "M_remove minus current-frame eroded visible-object interior" if strategy == "object_protected" else "same as M_remove for diagnostic candidate",
        "M_protect": "CURRENT_FRAME_ERODED_VISIBLE_OBJECT_INTERIOR" if strategy == "object_protected" else "EMPTY_UNTRUSTED_NOT_PROMOTED",
        "M_context_exclude": "M_write resized nearest plus one model pixel support",
        "unknown_policy": "task-object overlap and untrusted visible object remain UNKNOWN",
        "semantic_ready_for_product": False,
        "candidate_execution_allowed": True,
        "known_quality_blockers": (["VISIBLE_OBJECT_PROTECTION_INTERNAL_ONLY", "DEVICE_ROLE_INCOMPLETE"]
                                   if strategy == "object_protected" else
                                   ["VISIBLE_OBJECT_PROTECTION_UNTRUSTED", "DEVICE_ROLE_INCOMPLETE"]),
        "object_overlap_px": object_overlap,
        "device_nonempty_frames": device_frames,
        "geometry_input_allowed": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "created_at": now(),
    }
    manifest_path = root / "SCENE_PREP_MANIFEST.json"
    write_json(manifest_path, manifest)
    write_json(result_path, {
        "execution": "EXECUTED", "structure": "PASS", "quality": "REJECTED_QUALITY",
        "adoption": "CANDIDATE_ONLY", "manifest": ref(manifest_path),
        "claim_limit": "G0 candidate preparation only; not Clean or product authority.",
    })
    return manifest_path


def _encode_review(root: Path, rows: list[dict], clean_paths: list[Path]) -> dict:
    video = root / "SCENE_CLEAN_CANDIDATE_REVIEW.mp4"
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x480", "-r", "30",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index, (row, clean_path) in enumerate(zip(rows, clean_paths, strict=True)):
            raw = cv2.imread(row["raw"], cv2.IMREAD_COLOR)
            clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(row["write"], cv2.IMREAD_GRAYSCALE)
            protect = (cv2.imread(row["protect"], cv2.IMREAD_GRAYSCALE)
                       if row.get("protect") else None)
            unknown = cv2.imread(row["unknown"], cv2.IMREAD_GRAYSCALE)
            if any(value is None for value in (raw, clean, mask, unknown)):
                raise ValueError(f"REVIEW_INPUT_DECODE:{index}")
            raw_small = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
            clean_small = cv2.resize(clean, (640, 480), interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
            unknown_small = cv2.resize(unknown, (640, 480), interpolation=cv2.INTER_NEAREST)
            overlay = raw_small.copy()
            overlay[mask_small > 0] = (0, 0, 255)
            if protect is not None:
                protect_small = cv2.resize(protect, (640, 480), interpolation=cv2.INTER_NEAREST)
                overlay[protect_small > 0] = (0, 255, 0)
            overlay[unknown_small > 0] = (0, 255, 255)
            canvas = np.concatenate([raw_small, overlay, clean_small], axis=1)
            for x, title in ((8, "RAW"), (648, "RED=WRITE GREEN=PROTECT YELLOW=UNKNOWN"), (1288, "PROPAINTER CANDIDATE")):
                cv2.putText(canvas, title, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.putText(canvas, f"frame {index} | REJECTED_QUALITY / NOT ADOPTED", (8, 462),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            try:
                process.stdin.write(canvas.tobytes())
            except BrokenPipeError as error:
                stderr = process.stderr.read().decode("utf-8", errors="replace")
                code = process.wait()
                raise RuntimeError(f"FFMPEG_PIPE:{code}:{stderr[-4000:]}") from error
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
        if code:
            raise RuntimeError(f"FFMPEG:{code}:{error[-2000:]}")
    finally:
        if process.poll() is None:
            process.terminate(); process.wait()
    capture = cv2.VideoCapture(str(video)); decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != len(rows):
        raise ValueError(f"VIDEO_FRAME_COUNT:{decoded}!={len(rows)}")
    return {"video": ref(video), "decoded_frames": decoded, "expected_frames": len(rows)}


def finalize(short: str, strategy: str = "conservative") -> Path:
    """Recover only deterministic post-processing after successful inference.

    This deliberately refuses to call ProPainter.  It is valid only when the
    upstream model frames and the already composited lossless candidate frames
    are complete, which prevents a video-encoding failure from spending a
    second GPU lease or being misreported as a second model attempt.
    """
    root, _ = paths(short, strategy)
    result_path = root / "RESULT.json"
    if result_path.is_file():
        return result_path
    prep_path = root / "SCENE_PREP_MANIFEST.json"
    invocation = root / "INVOCATION.json"
    log_path = root / "PROPAINTER.log"
    prep = load_json(prep_path)
    count = int(prep["frame_count"])
    generated = sorted((root / "upstream/frames/frames").glob("*.png"))
    clean_paths = sorted((root / "clean").glob("*.png"))
    if len(generated) != count or len(clean_paths) != count:
        raise ValueError(f"INCOMPLETE_MODEL_OR_CLEAN_FRAMES:{len(generated)}:{len(clean_paths)}:{count}")
    outside_total = protected_total = changed_total = 0
    rows = []
    for index, (source, clean_path) in enumerate(zip(prep["rows"], clean_paths, strict=True)):
        raw = cv2.imread(source["raw"], cv2.IMREAD_COLOR)
        clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        write_image = cv2.imread(source["write"], cv2.IMREAD_GRAYSCALE)
        protect_image = (cv2.imread(source["protect"], cv2.IMREAD_GRAYSCALE)
                         if source.get("protect") else None)
        if raw is None or clean is None or write_image is None:
            raise ValueError(f"FINALIZE_DECODE:{index}")
        write = write_image > 0
        changed = np.any(clean != raw, axis=2)
        outside = int(changed[~write].sum())
        protected = int(changed[protect_image > 0].sum()) if protect_image is not None else 0
        if outside:
            raise ValueError(f"WRITE_BOUNDARY_VIOLATION:{index}:{outside}")
        if protected:
            raise ValueError(f"PROTECTED_PIXEL_CHANGED:{index}:{protected}")
        outside_total += outside
        protected_total += protected
        changed_total += int(changed.sum())
        rows.append({"frame_id": index, "clean": ref(clean_path),
                     "outside_write_changed_px": outside,
                     "protected_changed_px": protected,
                     "changed_px": int(changed.sum())})
    review = _encode_review(root, prep["rows"], clean_paths)
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_SCENE_CLEAN_CANDIDATE_V1",
        "task_id": TASK, "session_id": prep["session_id"],
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "REJECTED_QUALITY", "adoption": "CANDIDATE_ONLY",
        "frame_count": count, "model_returncode": 0,
        "source_kind": "SYNTHETIC_PROPAINTER_CANDIDATE",
        "outside_write_changed_px": outside_total,
        "protected_changed_px": protected_total,
        "changed_px": changed_total,
        "known_quality_blockers": prep["known_quality_blockers"],
        "scene_prep": ref(prep_path), "invocation": ref(invocation), "log": ref(log_path),
        "review": review, "rows": rows,
        "postprocess_recovery": {
            "model_rerun": False,
            "reason": "GPU model execution and lossless frame composition completed; only review encoding was recovered.",
            "failed_wrapper_receipt": ref(root / "GPU_RECEIPT.json"),
        },
        "future_frames_used": True, "geometry_input_allowed": False,
        "claim_limit": "Actual ProPainter output for offline root-cause review; visible-object protection and device roles remain untrusted, so this is not adopted Clean.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "completed_at": now(),
    }
    write_json(result_path, result)
    print(json.dumps({key: result[key] for key in ("session_id", "execution", "structure", "quality", "adoption", "frame_count", "review")}, ensure_ascii=False))
    return result_path


def inpaint(short: str, strategy: str = "conservative") -> Path:
    root, _ = paths(short, strategy)
    result_path = root / "RESULT.json"
    if result_path.is_file():
        return result_path
    prep_path = root / "SCENE_PREP_MANIFEST.json"
    prep = load_json(prep_path)
    if prep.get("semantic_ready_for_product") is not False or prep.get("candidate_execution_allowed") is not True:
        raise ValueError("CANDIDATE_SEMANTIC_STATE")
    lease = _gpu_guard()
    vendor = REPO / "vendor/ProPainter"
    weights = [vendor / "weights" / name for name in
               ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    pins = [ref(path) for path in weights]
    upstream = root / "upstream"
    command = [
        sys.executable, "-B", str(vendor / "inference_propainter.py"),
        "--video", str(root / "prep/frames"), "--mask", str(root / "prep/model_masks"),
        "--output", str(upstream), "--width", "960", "--height", "720",
        "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
        "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
        "--save_frames", "--fp16",
    ]
    invocation = root / "INVOCATION.json"
    write_json(invocation, {
        "command": command, "weights": pins, "vendor_license": ref(vendor / "LICENSE"),
        "gpu_lease": lease, "started_at": now(), "source_kind": "SYNTHETIC_PROPAINTER_CANDIDATE",
    })
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    log_path = root / "PROPAINTER.log"
    with log_path.open("xb") as log:
        completed = subprocess.run(command, cwd=vendor, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"PROPAINTER_EXIT:{completed.returncode}:{log_path}")
    generated = sorted((upstream / "frames/frames").glob("*.png"))
    if len(generated) != prep["frame_count"]:
        raise ValueError(f"PROPAINTER_FRAME_COUNT:{len(generated)}!={prep['frame_count']}")
    clean_root = root / "clean"
    clean_root.mkdir()
    clean_paths = []
    rows = []
    outside_total = protected_total = changed_total = 0
    for index, (source, generated_path) in enumerate(zip(prep["rows"], generated, strict=True)):
        raw = cv2.imread(source["raw"], cv2.IMREAD_COLOR)
        fake = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        write = cv2.imread(source["write"], cv2.IMREAD_GRAYSCALE) > 0
        if raw is None or fake is None:
            raise ValueError(f"CLEAN_DECODE:{index}")
        fake = cv2.resize(fake, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        protect_path = source.get("protect")
        protect_image = (cv2.imread(protect_path, cv2.IMREAD_GRAYSCALE)
                         if protect_path else None)
        protect = (protect_image > 0 if protect_image is not None
                   else np.zeros(write.shape, dtype=bool))
        clean = composite_clean(raw, fake, write, protect)
        target = clean_root / f"{index:06d}.png"
        _image(target, clean)
        clean_paths.append(target)
        changed = np.any(clean != raw, axis=2)
        outside = int(changed[~write].sum())
        protected = int(changed[protect].sum())
        if outside or protected:
            raise ValueError(f"WRITE_BOUNDARY_VIOLATION:{index}")
        outside_total += outside; protected_total += protected; changed_total += int(changed.sum())
        rows.append({"frame_id": index, "clean": ref(target), "outside_write_changed_px": outside,
                     "protected_changed_px": protected, "changed_px": int(changed.sum())})
    review = _encode_review(root, prep["rows"], clean_paths)
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_SCENE_CLEAN_CANDIDATE_V1",
        "task_id": TASK, "session_id": prep["session_id"],
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "REJECTED_QUALITY", "adoption": "CANDIDATE_ONLY",
        "frame_count": len(rows), "model_returncode": completed.returncode,
        "source_kind": "SYNTHETIC_PROPAINTER_CANDIDATE",
        "outside_write_changed_px": outside_total, "protected_changed_px": protected_total,
        "changed_px": changed_total, "known_quality_blockers": prep["known_quality_blockers"],
        "scene_prep": ref(prep_path), "invocation": ref(invocation), "log": ref(log_path),
        "review": review, "rows": rows,
        "future_frames_used": True, "geometry_input_allowed": False,
        "claim_limit": "Actual ProPainter output for offline root-cause review; visible-object protection and device roles remain untrusted, so this is not adopted Clean.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "completed_at": now(),
    }
    write_json(result_path, result)
    print(json.dumps({key: result[key] for key in ("session_id", "execution", "structure", "quality", "adoption", "frame_count", "review")}, ensure_ascii=False))
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "inpaint", "finalize"), required=True)
    parser.add_argument("--session", choices=("031", "007", "103", "042"), required=True)
    parser.add_argument("--strategy", choices=("conservative", "object_protected"),
                        default="conservative")
    args = parser.parse_args()
    if args.stage == "prepare":
        path = prepare(args.session, args.strategy)
    elif args.stage == "inpaint":
        path = inpaint(args.session, args.strategy)
    else:
        path = finalize(args.session, args.strategy)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
