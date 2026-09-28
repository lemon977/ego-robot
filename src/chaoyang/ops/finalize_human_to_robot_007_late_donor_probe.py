"""Seal the two-frame 007 donor availability search without Clean promotion."""
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
from chaoyang.ops.run_human_to_robot_007_late_donor_probe import TASK, ATTEMPT, DEST, TARGETS, DONORS

INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REVISION = "HUMAN_TO_ROBOT_007_LATE_DONOR_PROBE_20260923"


def verify_result(result: dict) -> None:
    if result.get("task_id") != TASK or result.get("targets") != list(TARGETS):
        raise RuntimeError("RESULT_IDENTITY_MISMATCH")
    rows = result.get("rows", [])
    if result.get("donors_evaluated") != len(rows) or [row["donor_frame"] for row in rows] != list(DONORS[:len(rows)]):
        raise RuntimeError("DONOR_COHORT_DRIFT")
    if result.get("full_cohort_evaluated") and len(rows) != len(DONORS):
        raise RuntimeError("FALSE_FULL_COHORT")
    if len(result.get("summaries", [])) != len(TARGETS):
        raise RuntimeError("TARGET_SUMMARIES_MISSING")
    for summary, frame in zip(result["summaries"], TARGETS):
        if summary["target_frame"] != frame or artifact_ref(Path(summary["count_image"]["path"])) != summary["count_image"]:
            raise RuntimeError(f"COUNT_REFERENCE_MISMATCH:{frame}")
        count = cv2.imread(summary["count_image"]["path"], cv2.IMREAD_UNCHANGED)
        if count is None or count.shape != (960, 1280) or count.dtype != np.uint16:
            raise RuntimeError(f"COUNT_IMAGE_INVALID:{frame}")
        refs = summary["inputs"]
        for ref in refs.values():
            if artifact_ref(Path(ref["path"])) != ref:
                raise RuntimeError(f"SOURCE_CHANGED:{frame}")
        write = cv2.imread(refs["write"]["path"], cv2.IMREAD_UNCHANGED) > 0
        protect = cv2.imread(refs["protect"]["path"], cv2.IMREAD_UNCHANGED) > 0
        object_labels = cv2.imread(refs["object_labels"]["path"], cv2.IMREAD_UNCHANGED)
        eligible = write & ~protect & (object_labels == 0)
        if summary["eligible_write_pixels"] != int(eligible.sum()):
            raise RuntimeError(f"ELIGIBLE_DENOMINATOR_MISMATCH:{frame}")
        if summary["zero_donor_pixels"] != int(np.count_nonzero(eligible & (count == 0))):
            raise RuntimeError(f"ZERO_DONOR_COUNT_MISMATCH:{frame}")
        if summary["one_or_more_donor_pixels"] != int(np.count_nonzero(eligible & (count >= 1))):
            raise RuntimeError(f"ONE_DONOR_COUNT_MISMATCH:{frame}")
        if summary["three_or_more_donor_pixels"] != int(np.count_nonzero(eligible & (count >= 3))):
            raise RuntimeError(f"THREE_DONOR_COUNT_MISMATCH:{frame}")
        if int(count[eligible].sum()) != sum(row["targets"][str(frame)]["eligible_table_pixels"] for row in rows):
            raise RuntimeError(f"PER_DONOR_SUPPORT_SUM_MISMATCH:{frame}")


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
         "tests/test_human_to_robot_007_late_donor_probe.py"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_TESTS_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    log = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    log.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation, {
        "schema_version": "HUMAN_TO_ROBOT_007_LATE_DONOR_FINAL_VALIDATION_V1", "task_id": TASK,
        "status": "PASS_FOR_SOURCE_AVAILABILITY_ONLY", "source_result": artifact_ref(source),
        "targeted_tests": artifact_ref(log), "pytest": {"exit_code": 0,
                                                   "summary": tests.stdout.strip().splitlines()[-1]},
        "claim_limit": "Two target count images and complete donor rows checked; no Clean or product quality claim.",
    })
    complete = result["full_cohort_evaluated"]
    found = any(row["three_or_more_donor_pixels"] > 0 for row in result["summaries"])
    terminal = "PASSED" if complete and found else "REJECTED_QUALITY" if complete else "NOT_EVALUATED_BUDGET"
    lane_path = ATTEMPT / "lanes/scene/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_SOURCE_SEARCH", execution=result["execution"],
                structure=result["structure"], quality="DONOR_SUPPORT_FOUND_DIAGNOSTIC" if found else "NO_THREE_DONOR_SUPPORT",
                adoption="NOT_ADOPTED", evidence=[artifact_ref(source)], updated_at=now_iso())
    atomic_json(lane_path, lane)
    output = ATTEMPT / "RESULT.json"
    atomic_json(output, {
        "schema_version": "HUMAN_TO_ROBOT_007_LATE_DONOR_RESULT_V1", "task_id": TASK,
        "route": "HUMAN_TO_ROBOT_BASELINE_V1", "status": "TERMINAL_SOURCE_SEARCH",
        "task_terminal_status": terminal, "terminal_at": now_iso(),
        "source_result": artifact_ref(source), "final_validation": artifact_ref(validation),
        "summaries": result["summaries"], "donors_evaluated": result["donors_evaluated"],
        "full_cohort_evaluated": complete,
        "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
        "claim_limit": "Donor-source availability only. Three-source pixels are not hidden-background truth or Clean approval.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    task.update(status=terminal, result=artifact_ref(output), updated_at=now_iso(),
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": terminal, "created_at": now_iso(),
        "message": "Late-window 007 donor-source search sealed without Clean promotion.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_007_LATE_DONOR_TERMINAL",
        "plan_revision": REVISION, "execution_revision": REVISION,
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "Source search terminal; no active algorithm task.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_LATE_DONOR_TERMINAL",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": terminal, "task_id": TASK,
                      "result": str(output), "governance_revision": published["governance_revision"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
