"""Seal the fixed-grid CPU donor probe without Clean/product promotion."""
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
from chaoyang.ops.run_human_to_robot_007_table_donor_expansion import TASK, ATTEMPT, DEST, DONORS

INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REVISION = "HUMAN_TO_ROBOT_007_TABLE_DONOR_EXPANSION_20260923"


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
    model_path = DEST / "RESULT.json"
    result = load_json(model_path)
    if (result.get("task_id") != TASK or result.get("target_frame") != 184
            or result.get("donor_frames") != list(DONORS)
            or [r["donor_frame"] for r in result.get("rows", [])] != list(DONORS)):
        raise RuntimeError("FIXED_GRID_RESULT_INCOMPLETE")
    candidate = result.get("candidate")
    if candidate:
        for key, expected_shape in (("support", (960, 1280)), ("review", (960, 2560, 3))):
            ref = candidate[key]
            path = Path(ref["path"])
            decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if key == "support" else cv2.IMREAD_COLOR)
            if artifact_ref(path) != ref or decoded is None or decoded.shape != expected_shape:
                raise RuntimeError(f"CANDIDATE_IMAGE_INVALID:{key}")
        if candidate["replaced_table_pixels"] <= 0:
            raise RuntimeError("ZERO_SUPPORT_CANDIDATE")
    tests = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_human_to_robot_007_table_donor_expansion.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_TESTS_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    test_path = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    test_path.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation, {
        "schema_version": "HUMAN_TO_ROBOT_007_TABLE_DONOR_EXPANSION_FINAL_VALIDATION_V1",
        "task_id": TASK, "status": "PASS_FOR_FIXED_GRID_STRUCTURE_NOT_CLEAN_QUALITY",
        "targeted_tests": artifact_ref(test_path),
        "pytest": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
        "model_result": artifact_ref(model_path),
        "candidate_review": candidate["review"] if candidate else None,
        "claim_limit": "Fixed donor grid and any table-only diagnostic checked; not Clean quality.",
    })
    lane_path = ATTEMPT / "lanes/scene/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_FIXED_GRID", execution="EXECUTED", structure="PASS",
                quality="TABLE_ONLY_DIAGNOSTIC" if candidate else "REJECTED_QUALITY",
                adoption="NOT_ADOPTED", evidence=[artifact_ref(model_path)], updated_at=now_iso())
    atomic_json(lane_path, lane)
    terminal = "PASSED" if candidate else "REJECTED_QUALITY"
    result_path = ATTEMPT / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "HUMAN_TO_ROBOT_007_TABLE_DONOR_EXPANSION_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1",
        "status": "TERMINAL_FIXED_GRID", "task_terminal_status": terminal,
        "terminal_at": now_iso(), "model_result": artifact_ref(model_path),
        "final_validation": artifact_ref(validation),
        "execution_summary": {"fixed_donors_checked": len(DONORS),
                              "geometry_pass_donors": result["geometry_pass_count"],
                              "table_only_candidate_frame": candidate["donor_frame"] if candidate else None,
                              "table_only_replaced_pixels": candidate["replaced_table_pixels"] if candidate else 0,
                              "007_new_full_clean_frames": 0,
                              "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4"},
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS_FOR_FIXED_GRID",
                         "quality": "TABLE_ONLY_DIAGNOSTIC" if candidate else "NO_QUALIFIED_TABLE_DONOR",
                         "clean_quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
        "claim_limit": "Same-session table-only donor diagnostic; no plate/chips, full Clean, product or metric authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    task.update(status=terminal, result=artifact_ref(result_path), updated_at=now_iso(),
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": terminal, "created_at": now_iso(),
        "message": "Fixed-grid table donor result sealed; no Clean/product promotion.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_007_TABLE_DONOR_EXPANSION_TERMINAL",
        "plan_revision": REVISION, "execution_revision": REVISION,
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "Fixed donor task terminal, no active algorithm task.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_TABLE_DONOR_EXPANSION_TERMINAL",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": terminal, "task_id": TASK,
                      "result": str(result_path),
                      "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
