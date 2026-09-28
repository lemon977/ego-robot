"""Verify current formal 007 candidate reuses its exact signature without mutation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2

from chaoyang.governance.common import artifact_ref, atomic_json, load_json, REPO_ROOT
from chaoyang.ops.run_human_to_robot_product_first_cable import TASK
from chaoyang.ops.run_human_to_robot_baseline_v1 import pipeline_signature

ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PRODUCT = ROOT / "lanes/motion_product/formal_007_current/attempt_0001"
RECEIPT = PRODUCT.parent / "RESUME_RECEIPT.json"
BINDING = REPO_ROOT / "_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/bindings/get_potato_chips_0915_007/attempt_0002/BINDING.json"
INPUT = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007"


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    result_path = PRODUCT / "PRODUCT_RESULT.json"
    result = load_json(result_path)
    if result.get("session_id") != "get_potato_chips_0915_007" or result.get("decoded_frames") != 378:
        raise RuntimeError("PRODUCT_SESSION_OR_COVERAGE_MISMATCH")
    if result.get("pipeline_signature") != pipeline_signature():
        raise RuntimeError("FORMAL_PRODUCT_NOT_CURRENT_CODE_SIGNATURE")
    if result.get("quality") != "REJECTED_QUALITY":
        raise RuntimeError("OLD_REJECTED_CLEAN_NOT_PROPAGATED")
    video = PRODUCT / "robot.mp4"
    before_result = artifact_ref(result_path)
    before_video = artifact_ref(video)
    command = [sys.executable, "-m", "chaoyang.cli", "run", "run_human_to_robot_baseline_v1",
               "--input", str(INPUT), "--motion-source", "hawor", "--output", str(PRODUCT),
               "--config", str(BINDING), "--resume"]
    env = dict(os.environ)
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               TMPDIR=str(REPO_ROOT / ".cache/tmp"),
               XDG_CACHE_HOME=str(REPO_ROOT / ".cache/xdg"))
    process = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True,
                             text=True, timeout=300, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"FORMAL_RESUME_FAILED:{process.returncode}:{process.stderr[-1200:]}")
    payload = json.loads(process.stdout.strip().splitlines()[-1])
    if payload.get("status") != "REUSED":
        raise RuntimeError(f"FORMAL_RESUME_NOT_REUSED:{payload}")
    if before_result != artifact_ref(result_path) or before_video != artifact_ref(video):
        raise RuntimeError("FORMAL_RESUME_MUTATED_OUTPUT")
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != 378:
        raise RuntimeError(f"VIDEO_DECODE_COUNT:{decoded}")
    atomic_json(RECEIPT, {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_FORMAL_007_RESUME_V1",
        "task_id": TASK, "session_id": "get_potato_chips_0915_007",
        "execution": "FORMAL_CLI_CURRENT_SIGNATURE_REUSED", "quality": "REJECTED_QUALITY",
        "adoption": "NOT_ADOPTED", "command": command,
        "input_binding": artifact_ref(BINDING), "product_result": before_result,
        "video": before_video, "decoded_frames": decoded,
        "model_or_solver_reexecution": False, "output_mutated": False,
        "claim_limit": "Current formal CLI and exact-signature resume verified for an old rejected Clean; no 007 product quality promotion.",
    })
    print(json.dumps({"status": "PASS_REUSED_NO_MUTATION", "receipt": str(RECEIPT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
