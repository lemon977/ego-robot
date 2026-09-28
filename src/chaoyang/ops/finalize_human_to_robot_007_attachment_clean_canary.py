"""Seal one 007 changed-input Clean canary after independent video inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.ops.run_human_to_robot_007_attachment_clean_canary import TASK, ATTEMPT, DEST

INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REVISION = "HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_CANARY_20260923"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "RUNNING", "CLAIMED"}:
        raise RuntimeError("TASK_NOT_ACTIVE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    index = load_json(INDEX)
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_SOLE_INDEX_ENTRY")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if lease.get("status") != "RELEASED":
        raise RuntimeError("GPU_LEASE_NOT_RELEASED")
    path = DEST / "RESULT.json"
    result = load_json(path)
    if (result.get("task_id") != TASK or result.get("source_frames") != list(range(181, 197))
            or len(result.get("rows", [])) != 16 or result.get("model_returncode") != 0):
        raise RuntimeError("MODEL_RESULT_INCOMPLETE")
    video = Path(result["review"]["path"])
    if artifact_ref(video) != {k: result["review"][k] for k in ("path", "bytes", "sha256")}:
        raise RuntimeError("REVIEW_SHA_MISMATCH")
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != 16 or result["review"]["decoded_frames"] != 16:
        raise RuntimeError("REVIEW_DECODE_MISMATCH")
    for row in result["rows"]:
        ref = row["clean"]
        image = cv2.imread(str(ref["path"]), cv2.IMREAD_COLOR)
        if artifact_ref(Path(ref["path"])) != ref or image is None or image.shape != (960, 1280, 3):
            raise RuntimeError(f"CLEAN_FRAME_UNREADABLE_OR_CHANGED:{row['frame_id']}")
    # The first lease ended before model invocation on a preflight-domain bug.
    # The packet permits one recorded runtime repair; bind final proof to the
    # successful receipt while retaining the failed receipt unchanged.
    gpu_path = ATTEMPT / "PROPAINTER_GPU_RECEIPT_RETRY1.json"
    gpu = load_json(gpu_path)
    initial_gpu_path = ATTEMPT / "PROPAINTER_GPU_RECEIPT.json"
    initial_gpu = load_json(initial_gpu_path)
    if (gpu.get("status") != "PASSED" or gpu.get("task_id") != TASK + ":scene"
            or initial_gpu.get("status") != "FAILED_RUNTIME_FINAL"
            or initial_gpu.get("task_id") != TASK + ":scene"):
        raise RuntimeError("GPU_EXECUTION_NOT_PROVEN")
    visual_path = ATTEMPT / "AI_VISUAL_REVIEW.json"
    visual = load_json(visual_path)
    if (visual.get("task_id") != TASK or visual.get("review_authority") != "AI_REVIEW_PROXY_NOT_HUMAN_GT"
            or visual.get("frames_reviewed") != [181, 184, 190, 196]
            or visual.get("quality") not in {"REJECTED_QUALITY", "INCONCLUSIVE"}
            or visual.get("review_video_sha256") != result["review"]["sha256"]):
        raise RuntimeError("INDEPENDENT_VISUAL_REVIEW_INVALID")
    tests = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_human_to_robot_007_attachment_clean_canary.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_TESTS_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    test_path = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    test_path.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation, {
        "schema_version": "HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_FINAL_VALIDATION_V1",
        "task_id": TASK, "status": "PASS_FOR_EXECUTION_STRUCTURE_NOT_CLEAN_QUALITY",
        "targeted_tests": artifact_ref(test_path),
        "pytest": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
        "gpu_receipt": artifact_ref(gpu_path), "initial_runtime_failure": artifact_ref(initial_gpu_path),
        "model_result": artifact_ref(path),
        "visual_review": artifact_ref(visual_path),
        "claim_limit": "16-frame changed-input model, lossless outputs and review checked; no Clean/product adoption.",
    })
    lane_path = ATTEMPT / "lanes/scene/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_16_FRAME", execution="EXECUTED", structure="PASS",
                quality=visual["quality"], adoption="NOT_ADOPTED",
                evidence=[artifact_ref(path), artifact_ref(visual_path)], updated_at=now_iso())
    atomic_json(lane_path, lane)
    terminal = "REJECTED_QUALITY" if visual["quality"] == "REJECTED_QUALITY" else "FAILED_QUALITY_C"
    result_path = ATTEMPT / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1",
        "status": "TERMINAL_16_FRAME", "task_terminal_status": terminal,
        "terminal_at": now_iso(), "model_result": artifact_ref(path),
        "final_validation": artifact_ref(validation), "visual_review": artifact_ref(visual_path),
        "execution_summary": {"007_new_clean_candidate_frames": 16,
                              "007_new_full_clean_frames": 0,
                              "object_overlap_frames": result["object_overlap_frames"],
                              "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4"},
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS_FOR_16_FRAME",
                         "quality": visual["quality"], "clean_quality": visual["quality"],
                         "adoption": "NOT_ADOPTED"},
        "claim_limit": "Fixed changed-input 007 Clean canary only; no full-session or product/metric authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    task.update(status=terminal, result=artifact_ref(result_path), updated_at=now_iso(),
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": terminal, "created_at": now_iso(),
        "message": "Attachment-rebound 16-frame ProPainter canary sealed after visual review.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_TERMINAL",
        "plan_revision": REVISION, "execution_revision": REVISION,
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "Changed-input Clean task terminal, no active algorithm task.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_TERMINAL",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": terminal, "task_id": TASK,
                      "result": str(result_path),
                      "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
