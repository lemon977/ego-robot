from __future__ import annotations

"""Validate and atomically close exact78 V5.2 stage 0."""

import argparse
import json
from pathlib import Path

from tools.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, TASK_STATE_PATH, artifact_ref, atomic_json,
    freshness, load_json, now_iso, process_identity, publish_bundle,
    sha256_file, validate_artifact_ref,
)
from tools.governance.v52_contracts import validate_task_packet


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "tasks/control/runs/20260913_exact78_v52"
PACKET_ROOT = RUN_ROOT / "task_packets"
PLAN = ROOT / "docs/governance/PLAN_REVISION.json"
WAVE0 = ROOT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
EXPECTED_WAVE0_SHA = "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--fault-result", type=Path, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    errors: list[str] = []
    if receipt.get("governance_revision") != args.expected_revision:
        errors.append("expected revision differs from current receipt")
    for reference in receipt.get("files", {}).values():
        errors.extend(validate_artifact_ref(reference))
    if freshness(state).get("status") != "FRESH":
        errors.append(f"governance is not fresh: {freshness(state)}")
    if sha256_file(WAVE0) != EXPECTED_WAVE0_SHA:
        errors.append("frozen Wave0 SHA differs")
    plan = load_json(PLAN)
    if plan.get("plan_revision") != "exact78-v5.2" or plan.get("status") != "APPROVED_FOR_EXECUTION":
        errors.append("plan revision is not approved exact78-v5.2")
    fault = load_json(args.fault_result)
    if fault.get("status") != "PASS":
        errors.append("fault injection did not pass")
    index_path = PACKET_ROOT.parent / "TASK_PACKET_INDEX.json"
    index = load_json(index_path)
    for item in index.get("task_packets", []):
        packet_path = ROOT / item["packet_path"]
        if not packet_path.is_file() or sha256_file(packet_path) != item["packet_sha256"]:
            errors.append(f"task packet SHA mismatch: {packet_path}")
            continue
        errors.extend(f"{packet_path}: {error}" for error in validate_task_packet(load_json(packet_path)))
    for task in state.get("tasks", []):
        if task.get("status") not in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}:
            continue
        pid = task.get("pid")
        identity = process_identity(int(pid)) if pid else {"alive": False}
        if not identity.get("alive") or identity.get("start_ticks") != task.get("proc_start_ticks"):
            errors.append(f"ghost or mismatched executor remains: {task['task_id']}")

    output = RUN_ROOT / "stage0" / "STAGE0_RESULT.json"
    result = {
        "schema_version": "exact78-v52-stage0-result-v1", "created_at": now_iso(),
        "status": "PASS" if not errors else "FAILED_RUNTIME_FINAL",
        "source_revision": args.expected_revision, "target_revision": args.expected_revision + 1,
        "checks": {
            "receipt_refs": "PASS" if not [e for e in errors if "artifact" in e] else "FAIL",
            "freshness": freshness(state), "wave0_sha": sha256_file(WAVE0),
            "plan_revision": plan.get("plan_revision"), "task_packet_count": len(index.get("task_packets", [])),
            "fault_injection": artifact_ref(args.fault_result),
        },
        "errors": errors,
        "claim_limit": "Stage-0 governance and fencing closure only; no pipeline-stage authority.",
    }
    atomic_json(output, result)
    if errors:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    stage0 = next(task for task in state["tasks"] if task["task_id"] == "exact78_v52_stage0")
    stage0.update(
        status="PASSED", phase="stage0_complete", attempt=1, updated_at=now_iso(),
        heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None,
        result=artifact_ref(output),
    )
    for task_id in (
        "exact78_v52_lane_a_clean", "exact78_v52_lane_b_upstream_c",
        "exact78_v52_lane_c_contact_robot", "exact78_v52_lane_d_visual_aux",
        "exact78_v52_lane_e_baseline_docs", "exact78_v52_lane_f_cleanup",
    ):
        task = next(item for item in state["tasks"] if item["task_id"] == task_id)
        task.update(status="PENDING", updated_at=now_iso(), heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": "exact78_v52_stage0", "session": None, "attempt": 1,
        "status": "PASSED", "created_at": now_iso(),
        "message": "Task packets, fencing, immutable finals and fault injection passed.",
        "result": artifact_ref(output),
    }])[-100:]
    state["next_task"] = {
        "task_id": "exact78_v52_lane_a_clean", "session": None,
        "prerequisites": ["exact78_v52_stage0"],
        "expected_resource": "CPU preflight first; GPU only after 3x10s gate. Lane C fixtures, E and F may run CPU-only.",
        "stop_condition": "Wave0 SHA/signature conflict, prerequisite failure or GPU wait exceeding 30 minutes.",
    }
    summary_path = PACKET_ROOT / "exact78_v52_stage0" / "RESULT_SUMMARY.json"
    atomic_json(summary_path, {
        "schema_version": "exact78-task-result-summary-v1", "task_id": "exact78_v52_stage0",
        "status": "PASSED", "attempts_consumed": 1, "result": artifact_ref(output),
        "claim_limit": result["claim_limit"],
    })
    code_paths = [
        ROOT / "AGENTS.md", ROOT / "tools/governance/common.py",
        ROOT / "tools/governance/v52_contracts.py", ROOT / "tools/governance/recover_governance_v5.py",
        ROOT / "tools/governance/bootstrap_v52_task_packets.py", ROOT / "tools/governance/fault_injection_v52.py",
        ROOT / "tools/governance/finalize_stage0_v52.py",
    ]
    patch_receipt_path = PACKET_ROOT / "exact78_v52_stage0" / "PATCH_RECEIPT.json"
    atomic_json(patch_receipt_path, {
        "schema_version": "exact78-patch-receipt-v1", "task_id": "exact78_v52_stage0",
        "status": "PASS", "changed_paths": [str(path.relative_to(ROOT)) for path in code_paths],
        "code_closure": [artifact_ref(path) for path in code_paths],
    })
    published = publish_bundle(
        authority, state, event_type="EXACT78_V52_STAGE0_PASSED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps(published, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
