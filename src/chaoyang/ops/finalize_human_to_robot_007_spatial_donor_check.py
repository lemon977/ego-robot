"""Seal frozen donor interpolation support without promoting Clean quality."""
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
from chaoyang.ops.run_human_to_robot_007_spatial_donor_check import TASK, ATTEMPT, DEST, PAIRS

INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REVISION = "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_CHECK_20260923"


def verify_result(result: dict) -> None:
    if result.get("task_id") != TASK or len(result.get("pairs", [])) != 12:
        raise RuntimeError("SPATIAL_RESULT_INCOMPLETE")
    expected_pairs = [(frame, donor) for frame, donors in PAIRS.items() for donor in donors]
    if [(row["target_frame"], row["donor_frame"]) for row in result["pairs"]] != expected_pairs:
        raise RuntimeError("SPATIAL_PAIR_DRIFT")
    for row in result["pairs"]:
        for key in ("support", "donor_raw", "donor_human", "donor_role"):
            if artifact_ref(Path(row[key]["path"])) != row[key]:
                raise RuntimeError(f"SOURCE_OR_SUPPORT_SHA_MISMATCH:{key}")
        mask = cv2.imread(row["support"]["path"], cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (960, 1280) or int(np.count_nonzero(mask)) != row["spatially_supported_eligible_pixels"]:
            raise RuntimeError("PAIR_SUPPORT_COUNT_MISMATCH")
        if row["spatially_supported_eligible_pixels"] > row["old_unrestricted_eligible_pixels"]:
            raise RuntimeError("SPATIAL_SUPPORT_EXPANDED_OLD_SUPPORT")
    if [row["target_frame"] for row in result.get("summaries", [])] != list(PAIRS):
        raise RuntimeError("TARGET_SUMMARIES_DRIFT")
    for summary in result["summaries"]:
        frame = summary["target_frame"]
        if artifact_ref(Path(summary["count"]["path"])) != summary["count"]:
            raise RuntimeError("COUNT_SHA_MISMATCH")
        count = cv2.imread(summary["count"]["path"], cv2.IMREAD_UNCHANGED)
        refs = summary["inputs"]
        for ref in refs.values():
            if artifact_ref(Path(ref["path"])) != ref:
                raise RuntimeError("TARGET_INPUT_SHA_MISMATCH")
        write = cv2.imread(refs["write"]["path"], cv2.IMREAD_UNCHANGED) > 0
        protect = cv2.imread(refs["protect"]["path"], cv2.IMREAD_UNCHANGED) > 0
        objects = cv2.imread(refs["object_labels"]["path"], cv2.IMREAD_UNCHANGED)
        eligible = write & ~protect & (objects == 0)
        if count is None or count.shape != eligible.shape or count.dtype != np.uint8:
            raise RuntimeError("COUNT_IMAGE_INVALID")
        if (summary["eligible_write_pixels"] != int(eligible.sum())
                or summary["zero_spatial_donor_pixels"] != int(np.count_nonzero(eligible & (count == 0)))
                or summary["one_or_more_spatial_donor_pixels"] != int(np.count_nonzero(eligible & (count >= 1)))
                or summary["three_or_more_spatial_donor_pixels"] != int(np.count_nonzero(eligible & (count >= 3)))):
            raise RuntimeError("SPATIAL_COUNT_MISMATCH")
        if int(count[eligible].sum()) != sum(row["spatially_supported_eligible_pixels"]
                                             for row in result["pairs"] if row["target_frame"] == frame):
            raise RuntimeError("SPATIAL_PAIR_AGGREGATE_MISMATCH")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--sync-current-docs", action="store_true")
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if args.sync_current_docs:
        if task is None or task.get("status") != "PASSED" or state.get("next_task") is not None:
            raise RuntimeError("SPATIAL_TASK_NOT_TERMINAL_FOR_DOC_SYNC")
        for relative in ("docs/current/PLAN.md", "docs/current/AI_WORK_ENTRY_ZH.md",
                         "docs/current/V5_SCENE.md"):
            if TASK not in (REPO_ROOT / relative).read_text(encoding="utf-8"):
                raise RuntimeError(f"CURRENT_DOC_MARKER_MISSING:{relative}")
        published = publish_bundle(load_json(AUTHORITY_PATH), state,
                                   event_type="HUMAN_TO_ROBOT_007_SPATIAL_DONOR_DOC_SYNC",
                                   expected_revision=args.expected_revision,
                                   generator_path=Path(__file__))
        status = load_json(REPO_ROOT / "docs/current/STATUS.json")
        if (status.get("latest_task") != TASK
                or status.get("governance_revision") != published["governance_revision"]):
            raise RuntimeError("CURRENT_STATUS_NOT_SYNCED")
        print(json.dumps({"status": "DOC_SYNCED", "governance_revision": published["governance_revision"]}))
        return 0
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
         "tests/test_human_to_robot_007_spatial_donor_check.py"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_TESTS_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    log = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    log.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation, {
        "schema_version": "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_FINAL_VALIDATION_V1", "task_id": TASK,
        "status": "PASS_FOR_SPATIAL_INTERPOLATION_DIAGNOSTIC_ONLY",
        "source_result": artifact_ref(source), "targeted_tests": artifact_ref(log),
        "pytest": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
        "claim_limit": "Fit-inlier spatial support checked; no actual background or Clean quality claim.",
    })
    lane_path = ATTEMPT / "lanes/scene/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_12_PAIRS", execution="EXECUTED", structure="PASS",
                quality="INTERPOLATION_SUPPORT_DIAGNOSTIC_ONLY", adoption="NOT_ADOPTED",
                evidence=[artifact_ref(source)], updated_at=now_iso())
    atomic_json(lane_path, lane)
    output = ATTEMPT / "RESULT.json"
    atomic_json(output, {
        "schema_version": "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_RESULT_V1", "task_id": TASK,
        "route": "HUMAN_TO_ROBOT_BASELINE_V1", "status": "TERMINAL_12_PAIRS",
        "task_terminal_status": "PASSED", "terminal_at": now_iso(),
        "source_result": artifact_ref(source), "final_validation": artifact_ref(validation),
        "summaries": result["summaries"], "pairs_checked": len(result["pairs"]),
        "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
        "claim_limit": "Spatial interpolation source check only; no color, hidden background truth, Clean or product adoption.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    task.update(status="PASSED", result=artifact_ref(output), updated_at=now_iso(),
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PASSED", "created_at": now_iso(),
        "message": "Twelve frozen donor pairs spatially checked without Clean promotion.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_TERMINAL",
        "plan_revision": REVISION, "execution_revision": REVISION,
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "Spatial donor diagnostic terminal; no active algorithm task.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_SPATIAL_DONOR_TERMINAL",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": "PASSED", "task_id": TASK,
                      "result": str(output), "governance_revision": published["governance_revision"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
