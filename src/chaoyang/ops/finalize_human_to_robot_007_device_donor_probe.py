"""Seal the 007 device/donor probe without promoting Clean or product quality."""
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
from chaoyang.ops.run_human_to_robot_007_device_donor_probe import TASK, ATTEMPT, DEST

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
    path = DEST / "RESULT.json"
    probe = load_json(path)
    if probe.get("task_id") != TASK or probe.get("target_frame") != 184 or len(probe.get("rows", [])) != 4:
        raise RuntimeError("PROBE_INCOMPLETE")
    overlay = Path(probe["overlay"]["path"])
    if artifact_ref(overlay) != probe["overlay"] or cv2.imread(str(overlay)) is None:
        raise RuntimeError("OVERLAY_UNREADABLE_OR_CHANGED")
    candidate = probe.get("table_only_candidate")
    if candidate is not None:
        image = Path(candidate["image"]["path"])
        if artifact_ref(image) != candidate["image"] or cv2.imread(str(image)) is None:
            raise RuntimeError("DONOR_DIAGNOSTIC_UNREADABLE_OR_CHANGED")
    tests = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_human_to_robot_007_device_donor_probe.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_TESTS_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    test_path = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    test_path.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation_path = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation_path, {
        "schema_version": "HUMAN_TO_ROBOT_007_DEVICE_DONOR_FINAL_VALIDATION_V1",
        "task_id": TASK, "status": "PASS_FOR_PROBE_STRUCTURE_NOT_PRODUCT_QUALITY",
        "targeted_tests": artifact_ref(test_path),
        "pytest": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
        "probe": artifact_ref(path), "overlay": probe["overlay"],
        "table_only_candidate": candidate,
        "claim_limit": "Fixed one-frame probe is structurally validated; no full Clean or product quality is established.",
    })
    lane_path = ATTEMPT / "lanes/scene/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_WITH_QUALITY_GAPS", execution="EXECUTED",
                structure="PASS", quality="PROBE_ONLY_NOT_CLEAN_QUALITY",
                adoption="NOT_ADOPTED", evidence=[artifact_ref(path)],
                updated_at=now_iso())
    atomic_json(lane_path, lane)
    outside = [row["name"] for row in probe["complaint_points"] if not row["rebound_write"]]
    passed = [row["donor_frame"] for row in probe["rows"] if row["geometry_pass"]]
    result_path = ATTEMPT / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "HUMAN_TO_ROBOT_007_DEVICE_DONOR_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1",
        "status": "TERMINAL_WITH_QUALITY_GAPS", "task_terminal_status": "REJECTED_QUALITY",
        "terminal_at": now_iso(), "result": artifact_ref(path),
        "final_validation": artifact_ref(validation_path),
        "execution_summary": {
            "007_device_complaint_points_outside_write": outside,
            "007_table_donor_geometry_pass_frames": passed,
            "007_table_only_candidate": candidate is not None,
            "007_new_full_clean_frames": 0,
            "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
        },
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS_FOR_PROBE",
                         "quality": "FULL_CLEAN_NOT_PASSED", "adoption": "NOT_ADOPTED"},
        "claim_limit": "007 device and table donor evidence only. No accepted Clean, product, Contact or metric authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    task.update(status="REJECTED_QUALITY", result=artifact_ref(result_path),
                updated_at=now_iso(), heartbeat_at=None, pid=None,
                proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "REJECTED_QUALITY",
        "created_at": now_iso(), "message": "Bounded 007 device/donor probe sealed; full Clean remains unadopted.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_007_DEVICE_DONOR_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_007_DEVICE_DONOR_PROBE_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_007_DEVICE_DONOR_PROBE_20260923",
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "Bounded probe terminal, no active algorithm task.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_DEVICE_DONOR_TERMINAL",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__),
                               task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": "TERMINAL_WITH_QUALITY_GAPS", "task_id": TASK,
                      "result": str(result_path),
                      "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
