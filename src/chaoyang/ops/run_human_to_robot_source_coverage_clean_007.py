"""One changed-input 007 ProPainter canary using the rebound human+cable mask."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.pipeline.v5_scene import composite_clean
from chaoyang.ops.run_human_to_robot_source_coverage_diagnostic import TASK, OUT
from chaoyang.ops.run_human_to_robot_source_coverage_rebind_007 import DEST as PREP

DEST = OUT / "lanes/scene/rebound_clean_007_window_v1"
OLD = REPO_ROOT / "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/scene/cable_007/clean_window_v1"
VENDOR = REPO_ROOT / "vendor/ProPainter"
RAW = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw"


def _image(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK or not index["task_packets"][0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    prepared = load_json(PREP / "RESULT.json")
    if prepared.get("quality") != "PROVISIONAL_INPUT_ONLY" or prepared.get("source_frames") != list(range(181, 197)):
        raise RuntimeError("CHANGED_INPUT_NOT_FROZEN")
    if DEST.exists():
        raise FileExistsError(DEST)
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != TASK + ":scene" or
            lease.get("gpu_process_pid") != os.getpid() or
            str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("GPU_LEASE_REQUIRED")
    weights = [VENDOR / "weights" / name for name in
               ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    for weight in weights:
        if not weight.is_file():
            raise FileNotFoundError(weight)
    DEST.mkdir(parents=True)
    command = [sys.executable, "-B", str(VENDOR / "inference_propainter.py"),
               "--video", str(PREP / "frames"), "--mask", str(PREP / "model_masks"),
               "--output", str(DEST / "upstream"), "--width", "960", "--height", "720",
               "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
               "--save_frames", "--fp16"]
    invocation = {"task_id": TASK, "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                  "input": artifact_ref(PREP / "RESULT.json"), "command": command,
                  "weights": [artifact_ref(path) for path in weights],
                  "lease_fencing_token": lease["fencing_token"],
                  "claim_limit": "One changed-input 16-frame model canary; no Clean or product adoption."}
    (DEST / "INVOCATION.json").write_text(json.dumps(invocation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with (DEST / "PROPAINTER.log").open("xb") as log:
        run = subprocess.run(command, cwd=VENDOR, env=env, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError(f"PROPAINTER_EXIT_{run.returncode}")
    generated = sorted((DEST / "upstream/frames/frames").glob("*.png"))
    if len(generated) != 16:
        raise ValueError(f"MODEL_OUTPUT_COUNT:{len(generated)}")
    (DEST / "clean").mkdir()
    video = DEST / "007_REBOUND_OLD_NEW_CLEAN_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 480))
    if not writer.isOpened():
        raise RuntimeError("REVIEW_WRITER_NOT_OPEN")
    rows = []
    try:
        for local, path in enumerate(generated):
            frame = 181 + local
            raw_full = _image(RAW / f"{frame:06d}.png")
            prior = _image(OLD / "clean" / f"{local:06d}.png")
            fake = _image(path)
            if raw_full.shape != (960, 1280, 3):
                raise ValueError(f"RAW_DOMAIN:{frame}")
            fake_full = cv2.resize(fake, (1280, 960), interpolation=cv2.INTER_LINEAR)
            write = cv2.imread(str(PREP / "write" / f"{local:06d}.png"), cv2.IMREAD_UNCHANGED) > 0
            protect = cv2.imread(str(PREP / "protect" / f"{local:06d}.png"), cv2.IMREAD_UNCHANGED) > 0
            clean = composite_clean(raw_full, fake_full, write, protect)
            changed = np.any(clean != raw_full, axis=2)
            if np.any(changed & ~write) or np.any(changed & protect):
                raise ValueError(f"WRITE_PROTECT_VIOLATION:{frame}")
            target = DEST / "clean" / f"{local:06d}.png"
            if not cv2.imwrite(str(target), clean):
                raise RuntimeError(f"CLEAN_WRITE:{frame}")
            panels = [cv2.resize(value, (640, 480), interpolation=cv2.INTER_AREA)
                      for value in (raw_full, prior, clean)]
            image = np.concatenate(panels, axis=1)
            for x, label in ((8, "RAW"), (648, "OLD REJECTED"), (1288, "NEW REBOUND CANDIDATE")):
                cv2.putText(image, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            cv2.putText(image, str(frame), (8, 460), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            writer.write(image)
            rows.append({"frame_id": frame, "clean": artifact_ref(target),
                         "changed_pixels": int(changed.sum()), "outside_write_changed_pixels": 0,
                         "protected_changed_pixels": 0})
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != 16:
        raise ValueError(f"REVIEW_DECODE:{decoded}")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_REBOUND_CLEAN_WINDOW_V1", "task_id": TASK,
        "session_id": "get_potato_chips_0915_007", "source_frames": list(range(181, 197)),
        "execution": "EXECUTED", "structure": "PASS", "quality": "PENDING_INDEPENDENT_REVIEW",
        "adoption": "CANDIDATE_ONLY", "model_returncode": 0,
        "input": artifact_ref(PREP / "RESULT.json"), "invocation": artifact_ref(DEST / "INVOCATION.json"),
        "model_log": artifact_ref(DEST / "PROPAINTER.log"),
        "review": {"path": str(video), "sha256": _sha(video), "decoded_frames": decoded},
        "rows": rows, "claim_limit": "Changed-input canary only. Independent visual quality and full-session adoption remain unproven.",
        "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EXECUTED_CANDIDATE", "result": str(DEST / "RESULT.json"), "review": str(video)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
