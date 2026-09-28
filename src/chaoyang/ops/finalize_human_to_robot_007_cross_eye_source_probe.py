"""Seal the fixed cross-eye source probe without promoting Clean quality."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.ops.run_human_to_robot_007_cross_eye_source_probe import TASK, ATTEMPT, DEST, FRAMES

INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REVISION = "HUMAN_TO_ROBOT_007_CROSS_EYE_SOURCE_PROBE_20260923"


def verify_result(result: dict) -> None:
    if result.get("task_id") != TASK or [r["source_frame_id"] for r in result.get("frames", [])] != list(FRAMES):
        raise RuntimeError("CROSS_EYE_FRAME_SET_DRIFT")
    if result.get("quality") != "SOURCE_DIAGNOSTIC_ONLY" or result.get("adoption") != "NOT_ADOPTED":
        raise RuntimeError("CROSS_EYE_AUTHORITY_OVERCLAIM")
    for name in ("domain", "source_stereo"):
        ref = result[name]
        if artifact_ref(Path(ref["path"])) != ref:
            raise RuntimeError(f"INPUT_SHA_DRIFT:{name}")
    for row in result["frames"]:
        for ref in row["inputs"].values():
            if artifact_ref(Path(ref["path"])) != ref:
                raise RuntimeError("TARGET_INPUT_SHA_DRIFT")
        if artifact_ref(Path(row["support"]["path"])) != row["support"]:
            raise RuntimeError("SUPPORT_SHA_DRIFT")
        mask = cv2.imread(row["support"]["path"], cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (960, 1280) or mask.dtype != np.uint8:
            raise RuntimeError("SUPPORT_DOMAIN_INVALID")
        if int(np.count_nonzero(mask)) != row["spatially_supported_eligible_pixels"]:
            raise RuntimeError("SUPPORT_COUNT_INVALID")
        if row["spatially_supported_eligible_pixels"] > row["eligible_write_pixels"]:
            raise RuntimeError("SUPPORT_EXCEEDS_ELIGIBLE")
        if row["left_binding"]["mean_abs_channel"] > 2.5 or row["left_binding"]["p99_abs_channel"] > 8:
            raise RuntimeError("LEFT_BINDING_INVALID")


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
    source = DEST / "RESULT.json"
    result = load_json(source)
    verify_result(result)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "TMPDIR": str(REPO_ROOT / ".cache/tmp"),
           "XDG_CACHE_HOME": str(REPO_ROOT / ".cache/xdg")}
    tests = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_human_to_robot_007_cross_eye_source_probe.py"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_TESTS_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    log = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    log.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation, {
        "schema_version": "HUMAN_TO_ROBOT_007_CROSS_EYE_FINAL_VALIDATION_V1", "task_id": TASK,
        "status": "PASS_FOR_CROSS_EYE_SOURCE_DIAGNOSTIC_ONLY",
        "source_result": artifact_ref(source), "targeted_tests": artifact_ref(log),
        "pytest": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
        "claim_limit": "No true hidden tabletop, Clean, metric depth or product quality claim.",
    })
    lane_path = ATTEMPT / "lanes/scene/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_4_FRAMES", execution="EXECUTED", structure="PASS",
                quality="SOURCE_DIAGNOSTIC_ONLY", adoption="NOT_ADOPTED",
                evidence=[artifact_ref(source)], updated_at=now_iso())
    atomic_json(lane_path, lane)
    output = ATTEMPT / "RESULT.json"
    atomic_json(output, {
        "schema_version": "HUMAN_TO_ROBOT_007_CROSS_EYE_RESULT_V1", "task_id": TASK,
        "route": "HUMAN_TO_ROBOT_BASELINE_V1", "status": "TERMINAL_4_FRAMES",
        "task_terminal_status": "PASSED", "terminal_at": now_iso(),
        "source_result": artifact_ref(source), "final_validation": artifact_ref(validation),
        "frames_checked": list(FRAMES),
        "supported_pixels": {str(r["source_frame_id"]): r["spatially_supported_eligible_pixels"] for r in result["frames"]},
        "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
        "claim_limit": "Cross-eye source diagnostic only; no Clean or product adoption.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    task.update(status="PASSED", result=artifact_ref(output), updated_at=now_iso(),
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PASSED", "created_at": now_iso(),
        "message": "Four-frame physical-right source support quantified without Clean promotion.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_007_CROSS_EYE_TERMINAL",
        "plan_revision": REVISION, "execution_revision": REVISION,
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "Cross-eye diagnostic terminal; no active algorithm task.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_CROSS_EYE_TERMINAL",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "PASSED", "task_id": TASK, "result": str(output),
                      "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
