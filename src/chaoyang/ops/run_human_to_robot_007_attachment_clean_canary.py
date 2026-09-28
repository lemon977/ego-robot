"""One changed-input ProPainter canary using a sealed 007 attachment track."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_source_coverage_rebind_007 import merge_support
from chaoyang.ops.run_human_to_robot_007_wearable_sam_canary import SPECS
from chaoyang.pipeline.v5_scene import composite_clean

TASK = "human_to_robot_007_attachment_clean_canary_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/attachment_clean_window_v1"
REBOUND = REPO_ROOT / "_run/current/human_to_robot_source_coverage_recovery_20260923/attempts/attempt_0001/lanes/scene/rebound_007_window_v2"
ATTACH = REPO_ROOT / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_track_canary_v1/get_potato_chips_0915_007"
OBJECT = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1/task_object"
OLD = REPO_ROOT / "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/scene/cable_007/clean_window_v1"
RAW_FULL = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw"
VENDOR = REPO_ROOT / "vendor/ProPainter"


def image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def complaint_coverage(mask: np.ndarray) -> dict[str, bool]:
    if mask.shape != (960, 1280):
        raise ValueError("COMPLAINT_DOMAIN")
    return {name: bool(mask[y, x]) for _, name, _, (x, y), _ in SPECS}


def seed_model_for_full_write(model: np.ndarray, full_write: np.ndarray) -> np.ndarray:
    """Mark the exact model pixels sampled by the 1280x960 nearest upsample."""
    if model.shape != (720, 960) or full_write.shape != (960, 1280):
        raise ValueError("IMAGE_DOMAIN_MISMATCH")
    seeded = model.copy()
    ys, xs = np.nonzero(full_write)
    seeded[ys * 720 // 960, xs * 960 // 1280] = True
    return seeded


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != TASK + ":scene"
            or lease.get("gpu_process_pid") != os.getpid()
            or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("GPU_LEASE_REQUIRED")
    if DEST.exists():
        raise FileExistsError(DEST)
    attachment_result = load_json(ATTACH / "RESULT.json")
    rebound_result = load_json(REBOUND / "RESULT.json")
    if (attachment_result.get("session_id") != "get_potato_chips_0915_007"
            or attachment_result.get("counts", {}).get("unknown") != 0
            or rebound_result.get("source_frames") != list(range(181, 197))):
        raise RuntimeError("UPSTREAM_FRAME_OR_AUTHORITY_MISMATCH")
    weights = [VENDOR / "weights" / name for name in
               ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    for weight in weights:
        if not weight.is_file():
            raise FileNotFoundError(weight)
    # Resolve and validate every upstream frame before writing or launching the model.
    inputs = []
    for local, frame in enumerate(range(181, 197)):
        raw_path = REBOUND / "frames" / f"{local:06d}.png"
        attachment_path = ATTACH / "candidate_masks" / f"{frame:06d}.png"
        model_path = REBOUND / "model_masks" / f"{local:06d}.png"
        write_path = REBOUND / "write" / f"{local:06d}.png"
        protect_path = REBOUND / "protect" / f"{local:06d}.png"
        object_path = OBJECT / f"{frame:06d}.png"
        raw = image(raw_path)
        attachment = image(attachment_path, cv2.IMREAD_GRAYSCALE) > 0
        current_model = image(model_path, cv2.IMREAD_GRAYSCALE) > 0
        write = image(write_path, cv2.IMREAD_GRAYSCALE) > 0
        protect = image(protect_path, cv2.IMREAD_GRAYSCALE) > 0
        # V5 task_object PNGs are uint16 with role labels 1/2. A grayscale
        # uint8 read shifts both labels to zero and silently hides conflicts.
        task_object = image(object_path, cv2.IMREAD_UNCHANGED) > 0
        if (raw.shape != (720, 960, 3) or any(x.shape != (960, 1280) for x in
            (attachment, write, protect, task_object)) or current_model.shape != (720, 960)
            or not attachment.any()):
            raise RuntimeError(f"UPSTREAM_DOMAIN_OR_EMPTY_ATTACHMENT:{frame}")
        # The sealed track is only a candidate. Protected pixels are never
        # reclassified as removal support; retain the dropped count for review.
        protected_attachment_pixels = int(np.count_nonzero(attachment & protect))
        admissible_attachment = attachment & ~protect
        seeded_model = seed_model_for_full_write(current_model, write | admissible_attachment)
        model, new_write = merge_support(seeded_model, admissible_attachment, write, protect)
        coverage = complaint_coverage(new_write) if frame == 184 else None
        if coverage is not None and not all(coverage.values()):
            raise RuntimeError(f"FRAME184_DEVICE_COMPLAINT_NOT_COVERED:{coverage}")
        inputs.append((frame, raw_path, attachment_path, model, new_write, protect,
                       int(np.count_nonzero(attachment & task_object)),
                       protected_attachment_pixels, coverage))
    DEST.mkdir(parents=True)
    prepared = DEST / "input"
    for folder in ("frames", "model_masks", "write", "protect"):
        (prepared / folder).mkdir(parents=True)
    input_rows = []
    for local, (frame, raw_path, attachment_path, model, new_write, protect, overlap,
                protected_attachment_pixels, coverage) in enumerate(inputs):
        raw = image(raw_path)
        targets = (("frames", raw), ("model_masks", model.astype(np.uint8) * 255),
                   ("write", new_write.astype(np.uint8) * 255),
                   ("protect", protect.astype(np.uint8) * 255))
        refs = {}
        for folder, value in targets:
            path = prepared / folder / f"{local:06d}.png"
            if not cv2.imwrite(str(path), value):
                raise RuntimeError(f"PREPARED_WRITE_FAILED:{frame}:{folder}")
            refs[folder] = artifact_ref(path)
        input_rows.append({"frame_id": frame, "source_raw": artifact_ref(raw_path),
                           "source_attachment": artifact_ref(attachment_path),
                           "prepared": refs, "write_pixels": int(new_write.sum()),
                           "attachment_task_object_overlap_pixels": overlap,
                           "attachment_protect_excluded_pixels": protected_attachment_pixels,
                           "frame184_complaint_coverage": coverage})
    input_result = prepared / "RESULT.json"
    input_result.write_text(json.dumps({
        "schema_version": "HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_INPUT_V1",
        "task_id": TASK, "session_id": "get_potato_chips_0915_007",
        "source_frames": list(range(181, 197)),
        "attachment_authority": "TRACKED_CANDIDATE_NOT_ADOPTED",
        "object_overlap_authority": "OLD_TASK_OBJECT_PROXY_NOT_TRUSTED_PROTECTION",
        "old_rebound": artifact_ref(REBOUND / "RESULT.json"),
        "attachment": artifact_ref(ATTACH / "RESULT.json"),
        "rows": input_rows,
        "claim_limit": "Changed ProPainter input only; task object conflicts and visual quality unresolved.",
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    command = [sys.executable, "-B", str(VENDOR / "inference_propainter.py"),
               "--video", str(prepared / "frames"), "--mask", str(prepared / "model_masks"),
               "--output", str(DEST / "upstream"), "--width", "960", "--height", "720",
               "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
               "--save_frames", "--fp16"]
    invocation = DEST / "INVOCATION.json"
    invocation.write_text(json.dumps({
        "schema_version": "HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_INVOCATION_V1",
        "task_id": TASK, "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": artifact_ref(input_result), "command": command,
        "weights": [artifact_ref(path) for path in weights],
        "lease_fencing_token": lease["fencing_token"],
        "claim_limit": "One changed-input 16-frame model canary; no Clean/product adoption.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    log_path = DEST / "PROPAINTER.log"
    with log_path.open("xb") as log:
        run = subprocess.run(command, cwd=VENDOR, env=env, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError(f"PROPAINTER_EXIT_{run.returncode}")
    generated = sorted((DEST / "upstream/frames/frames").glob("*.png"))
    if len(generated) != 16:
        raise RuntimeError(f"MODEL_OUTPUT_COUNT:{len(generated)}")
    (DEST / "clean").mkdir()
    review_path = DEST / "007_ATTACHMENT_REBOUND_OLD_NEW_CLEAN_REVIEW.mp4"
    writer = cv2.VideoWriter(str(review_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (2560, 480))
    if not writer.isOpened():
        raise RuntimeError("REVIEW_WRITER_NOT_OPEN")
    output_rows = []
    try:
        for local, fake_path in enumerate(generated):
            frame = 181 + local
            raw = image(RAW_FULL / f"{frame:06d}.png")
            if raw.shape != (960, 1280, 3):
                raise RuntimeError(f"RAW_FULL_DOMAIN:{frame}")
            old = image(OLD / "clean" / f"{local:06d}.png")
            fake = image(fake_path)
            fake_full = cv2.resize(fake, (1280, 960), interpolation=cv2.INTER_LINEAR)
            write = image(prepared / "write" / f"{local:06d}.png", cv2.IMREAD_GRAYSCALE) > 0
            protect = image(prepared / "protect" / f"{local:06d}.png", cv2.IMREAD_GRAYSCALE) > 0
            clean = composite_clean(raw, fake_full, write, protect)
            changed = np.any(clean != raw, axis=2)
            if np.any(changed & ~write) or np.any(changed & protect):
                raise RuntimeError(f"WRITE_PROTECT_VIOLATION:{frame}")
            target = DEST / "clean" / f"{local:06d}.png"
            if not cv2.imwrite(str(target), clean):
                raise RuntimeError(f"CLEAN_WRITE_FAILED:{frame}")
            support = raw.copy()
            support[write] = (support[write] * .55 + np.array([0, 0, 255]) * .45).astype(np.uint8)
            panels = [cv2.resize(v, (640, 480), interpolation=cv2.INTER_AREA)
                      for v in (raw, support, old, clean)]
            canvas = np.concatenate(panels, axis=1)
            for x, label in ((8, "RAW"), (648, "NEW WRITE"), (1288, "OLD REJECTED"), (1928, "NEW CANDIDATE")):
                cv2.putText(canvas, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (255, 255, 255), 2)
            cv2.putText(canvas, str(frame), (8, 460), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (255, 255, 255), 2)
            writer.write(canvas)
            output_rows.append({"frame_id": frame, "clean": artifact_ref(target),
                                "changed_pixels": int(changed.sum()),
                                "outside_write_changed_pixels": 0, "protected_changed_pixels": 0})
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(review_path))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != 16:
        raise RuntimeError(f"REVIEW_DECODE_MISMATCH:{decoded}")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_CANARY_V1",
        "task_id": TASK, "session_id": "get_potato_chips_0915_007",
        "source_frames": list(range(181, 197)),
        "execution": "EXECUTED", "structure": "PASS", "quality": "PENDING_INDEPENDENT_VISUAL_REVIEW",
        "adoption": "CANDIDATE_ONLY", "model_returncode": 0,
        "input": artifact_ref(input_result), "invocation": artifact_ref(invocation),
        "model_log": artifact_ref(log_path), "review": {**artifact_ref(review_path), "decoded_frames": decoded},
        "rows": output_rows,
        "object_overlap_frames": [r["frame_id"] for r in input_rows if r["attachment_task_object_overlap_pixels"] > 0],
        "claim_limit": "Changed-input canary only. Object overlap semantics and independent Clean quality remain unresolved.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EXECUTED_CANDIDATE", "result": str(DEST / "RESULT.json"),
                      "review": str(review_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
