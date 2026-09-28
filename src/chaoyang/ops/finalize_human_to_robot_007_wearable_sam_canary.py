"""Seal a raw one-frame wearable SAM canary; never promote Clean/product quality."""
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
from chaoyang.ops.run_human_to_robot_007_wearable_sam_canary import TASK, ATTEMPT, DEST

INDEX = REPO_ROOT / "tasks/current/INDEX.json"


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
    if result.get("task_id") != TASK or result.get("source_frame") != 184 or len(result.get("rows", [])) != 6:
        raise RuntimeError("CANARY_INCOMPLETE")
    if not result.get("runtime_signature") or artifact_ref(Path(result["runtime_signature"]["path"])) != result["runtime_signature"]:
        raise RuntimeError("RUNTIME_SIGNATURE_INVALID")
    review = Path(result["review"]["path"])
    image = cv2.imread(str(review))
    if artifact_ref(review) != result["review"] or image is None or image.shape != (960, 2560, 3):
        raise RuntimeError("REVIEW_UNREADABLE_OR_CHANGED")
    for row in result["rows"]:
        mask = Path(row["mask"]["path"])
        raw = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if artifact_ref(mask) != row["mask"] or raw is None or raw.shape != (960, 1280):
            raise RuntimeError(f"RAW_MASK_UNREADABLE_OR_CHANGED:{row['name']}")
    failed_runtime = ATTEMPT / "SAM_GPU_RECEIPT.json"
    if load_json(failed_runtime).get("status") != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("FIRST_API_RUNTIME_FAILURE_NOT_PRESERVED")
    receipt = ATTEMPT / "SAM_GPU_RUNTIME_RETRY_RECEIPT.json"
    gpu = load_json(receipt)
    if gpu.get("status") != "PASSED" or gpu.get("task_id") != TASK + ":scene":
        raise RuntimeError("GPU_EXECUTION_NOT_PROVEN")
    tests = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_human_to_robot_007_wearable_sam_canary.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_TESTS_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    test_path = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    test_path.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    passed = result.get("instance_quality") == "PASS"
    validation_path = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation_path, {
        "schema_version": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_FINAL_VALIDATION_V1",
        "task_id": TASK, "status": "PASS_FOR_SINGLE_FRAME_STRUCTURE_NOT_CLEAN_QUALITY",
        "targeted_tests": artifact_ref(test_path),
        "pytest": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
        "gpu_receipt": artifact_ref(receipt), "model_result": artifact_ref(path),
        "first_runtime_failure": artifact_ref(failed_runtime),
        "review": result["review"],
        "claim_limit": "Raw fixed-frame SAM masks and runtime checked; no temporal or Clean quality is established.",
    })
    lane_path = ATTEMPT / "lanes/scene/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_SINGLE_FRAME", execution="EXECUTED", structure="PASS",
                quality=result["instance_quality"], adoption="NOT_ADOPTED",
                evidence=[artifact_ref(path)], updated_at=now_iso())
    atomic_json(lane_path, lane)
    terminal = "PASSED" if passed else "REJECTED_QUALITY"
    result_path = ATTEMPT / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1",
        "status": "TERMINAL_SINGLE_FRAME", "task_terminal_status": terminal,
        "terminal_at": now_iso(), "model_result": artifact_ref(path),
        "final_validation": artifact_ref(validation_path),
        "execution_summary": {
            "007_raw_wearable_instances_supported": [row["name"] for row in result["rows"] if row["status"] == "SUPPORTED_SINGLE_FRAME"],
            "007_raw_wearable_instances_rejected": [row["name"] for row in result["rows"] if row["status"] != "SUPPORTED_SINGLE_FRAME"],
            "007_new_full_clean_frames": 0,
            "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
        },
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS_FOR_SINGLE_FRAME",
                         "quality": result["instance_quality"], "clean_quality": "NOT_EVALUATED",
                         "adoption": "NOT_ADOPTED"},
        "claim_limit": "Pinned-SAM raw one-frame device instances only; no accepted Clean, full product, Contact or metric authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    task.update(status=terminal, result=artifact_ref(result_path), updated_at=now_iso(),
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": terminal, "created_at": now_iso(),
        "message": "Single-frame wearable instance canary sealed; Clean remains unadopted.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_CANARY_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_CANARY_20260923",
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "One-frame SAM task terminal, no active algorithm task.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_WEARABLE_SAM_TERMINAL",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__),
                               task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": terminal, "task_id": TASK,
                      "result": str(result_path),
                      "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
