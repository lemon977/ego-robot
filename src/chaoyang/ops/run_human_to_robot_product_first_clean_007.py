"""One frozen 007 ProPainter canary; never grants adopted Clean authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.pipeline.v5_scene import composite_clean
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK, OUT as PREP, RAW, OLD_PREP


OUT = PREP.parent / "clean_window_v1"
VENDOR = ROOT / "vendor/ProPainter"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _render_review(clean_paths: list[Path]) -> dict:
    target = OUT / "007_CABLE_OLD_NEW_CLEAN_REVIEW.mp4"
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x480", "-r", "30",
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for local, new_path in enumerate(clean_paths):
            frame = 181 + local
            raw = _load_image(RAW / f"{frame:06d}.png")
            old = _load_image(OLD_PREP.parent / "clean" / f"{frame:06d}.png")
            new = _load_image(new_path)
            mask = cv2.imread(str(PREP / "write" / f"{local:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(frame)
            panels = [cv2.resize(value, (640, 480), interpolation=cv2.INTER_AREA)
                      for value in (raw, old, new)]
            support = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST) > 0
            panels[2][support] = (0.75 * panels[2][support] + 0.25 * np.array([0, 0, 255])).astype(np.uint8)
            canvas = np.concatenate(panels, axis=1)
            for x, label in ((8, "RAW"), (648, "OLD CLEAN"), (1288, "NEW CLEAN / RED WRITE")):
                cv2.putText(canvas, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            cv2.putText(canvas, f"007 source frame {frame} | CANDIDATE ONLY", (8, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        error = proc.stderr.read().decode("utf-8", errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"REVIEW_ENCODE:{error[-2000:]}")
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait()
    capture = cv2.VideoCapture(str(target))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    if count != len(clean_paths):
        raise ValueError(f"REVIEW_DECODE:{count}")
    return {"path": str(target), "sha256": _sha(target), "decoded_frames": count}


def main() -> int:
    prep = load_json(PREP / "RESULT.json")
    if prep.get("inpaint_candidate_structurally_ready") is not True or prep.get("frame_count") != 16:
        raise RuntimeError("PREP_NOT_READY")
    if OUT.exists():
        raise FileExistsError(OUT)
    lease = load_json(ROOT / "_run/current/GPU_LEASE.json")
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != f"{TASK}:scene" or
            lease.get("gpu_process_pid") != os.getpid() or
            str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("GPU_LEASE_REQUIRED")
    weights = [VENDOR / "weights" / name for name in
               ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True)
    command = [sys.executable, "-B", str(VENDOR / "inference_propainter.py"),
               "--video", str(PREP / "frames"), "--mask", str(PREP / "model_masks"),
               "--output", str(OUT / "upstream"), "--width", "960", "--height", "720",
               "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
               "--save_frames", "--fp16"]
    _save(OUT / "INVOCATION.json", {
        "task_id": TASK, "source": artifact_ref(PREP / "RESULT.json"), "command": command,
        "weights": [artifact_ref(path) for path in weights], "lease_fencing_token": lease["fencing_token"],
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "claim_limit": "Actual fixed-window model call, not quality adoption.",
    })
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with (OUT / "PROPAINTER.log").open("xb") as log:
        run = subprocess.run(command, cwd=VENDOR, env=env, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError(f"PROPAINTER_EXIT_{run.returncode}")
    generated = sorted((OUT / "upstream/frames/frames").glob("*.png"))
    if len(generated) != 16:
        raise ValueError(f"MODEL_OUTPUT_COUNT:{len(generated)}")
    (OUT / "clean").mkdir()
    rows = []
    clean_paths = []
    for local, path in enumerate(generated):
        frame = 181 + local
        raw = _load_image(RAW / f"{frame:06d}.png")
        fake = cv2.resize(_load_image(path), (1280, 960), interpolation=cv2.INTER_LINEAR)
        write = cv2.imread(str(PREP / "write" / f"{local:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        protect = cv2.imread(str(PREP / "protect" / f"{local:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        clean = composite_clean(raw, fake, write, protect)
        changed = np.any(raw != clean, axis=2)
        if np.any(changed & ~write) or np.any(changed & protect):
            raise ValueError(f"WRITE_PROTECT_VIOLATION:{frame}")
        target = OUT / "clean" / f"{local:06d}.png"
        if not cv2.imwrite(str(target), clean):
            raise RuntimeError(f"CLEAN_WRITE:{frame}")
        clean_paths.append(target)
        rows.append({"source_frame_id": frame, "changed_pixels": int(changed.sum()),
                     "outside_write_changed_pixels": 0, "protected_changed_pixels": 0,
                     "clean": artifact_ref(target)})
    review = _render_review(clean_paths)
    result = {"schema_version": "HUMAN_TO_ROBOT_007_PRODUCT_FIRST_CLEAN_WINDOW_V1",
              "task_id": TASK, "session_id": "get_potato_chips_0915_007", "source_frames": list(range(181, 197)),
              "execution": "EXECUTED", "structure": "PASS", "quality": "PENDING_INDEPENDENT_REVIEW",
              "adoption": "CANDIDATE_ONLY", "model_returncode": 0,
              "prep": artifact_ref(PREP / "RESULT.json"), "invocation": artifact_ref(OUT / "INVOCATION.json"),
              "model_log": artifact_ref(OUT / "PROPAINTER.log"), "review": review,
              "rows": rows, "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False,
              "claim_limit": "Actual 16-frame model output and byte-exact outside-write candidate; no independent Clean or product quality PASS."}
    _save(OUT / "RESULT.json", result)
    print(json.dumps({"status": "EXECUTED_CANDIDATE", "review": review, "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
