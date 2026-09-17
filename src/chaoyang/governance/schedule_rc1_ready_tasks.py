#!/usr/bin/env python3
"""Register the RC1 T1 and T3 tasks that become ready after T0."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH, artifact_ref, atomic_json, atomic_write,
    load_json, now_iso, publish_bundle,
)


RUN = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1"
OUT = RUN / "scheduler/attempts/attempt_0001"
PACKETS = RUN / "task_packets"
T0 = RUN / "t0_freeze_capacity_split/attempts/attempt_0001"


def _packet(task_id: str, phase: str, objective: str, read_set: list[Path], command: str, gates: list[str], claim: str) -> dict:
    return {
        "schema_version": "chaoyang-rc1-task-packet-v1", "task_id": task_id,
        "stage": phase, "objective": objective,
        "non_goals": ["No authority promotion", "No physical-truth claim", "No use of bidirectional artifacts as causal input"],
        "frozen_inputs": {str(i): artifact_ref(path) for i, path in enumerate(read_set)},
        "prerequisites": ["governance=FRESH", "rc1_t0_freeze_capacity_split=PASSED"],
        "read_set": [str(x) for x in read_set], "write_set": [str(RUN / phase.lower() / "attempts")],
        "commands": ["chaoyang validate-governance", command],
        "quality_gates": gates,
        "budgets": {"runtime_attempts": 2, "wall_seconds": 7200, "gpu_wait_seconds": 1800},
        "stop_conditions": ["PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"],
        "expected_outputs": ["RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json", "RESULT_SUMMARY.json"],
        "claim_limit": claim,
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"fresh scheduler attempt required: {OUT}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("scheduler CAS revision mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    t0 = next((x for x in state.get("tasks", []) if x.get("task_id") == "rc1_t0_freeze_capacity_split"), None)
    if not t0 or t0.get("status") != "PASSED":
        raise RuntimeError("T0 is not PASSED")

    status = REPO_ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json"
    contract = REPO_ROOT / "docs/governance/VISUAL_AUX_RC1_CONTRACT.json"
    t1_reads = [status, contract, T0 / "MASTER_LEDGER.json", T0 / "CAPACITY_REPORT.json",
        REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/PROMPT_ANNOTATIONS_FROZEN.json",
        REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_temporal_identity_canary_v1/attempts/attempt_0004/RESULT.json"]
    t3_reads = [status, contract, T0 / "MASTER_LEDGER.json",
        REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/robot_v77_terminal_index_r22/attempts/attempt_0003/RESULT.json",
        REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_robot_v78_prerequisite_closure_v1/attempts/attempt_0001/RESULT.json",
        REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_robot_v78_causal_metric_extractor_v1/attempts/attempt_0001/RESULT.json"]
    packets = {
        "rc1_t1_sam31_mask_bounded": _packet(
            "rc1_t1_sam31_mask_bounded", "RC1_T1_SAM31_MASK_BOUNDED",
            "Freeze independent Mask evaluation references, then run at most two SAM3.1 seed/propagation and re-detection/reseed revisions on one failure canary plus two regressions per task.",
            t1_reads, "chaoyang run run_mask_sam31_temporal_identity_real_canary_r22 --output-root archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/rc1_t1_sam31_mask_bounded/attempts/attempt_0001",
            ["evaluation_reference_precedes_candidate", "identity_switch_zero", "offscreen_empty", "reentry_identity", "no_new_segmentation_model"],
            "Development Mask evidence only; SAM3.1 remains the baseline and absence of independent pixel truth is not accuracy."),
        "rc1_t3_v77_causal_robot": _packet(
            "rc1_t3_v77_causal_robot", "RC1_T3_V77_CAUSAL_ROBOT",
            "Audit v77 dependencies and prefix-recompute current Robot states without consuming reverse-lookahead, bidirectional arm states or full-sequence placement.",
            t3_reads, "chaoyang run run_rc1_t3_v77_causal_preflight --output-root archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/rc1_t3_v77_causal_robot/attempts/attempt_0001",
            ["prefix_only_dependency_graph", "finite", "joint_limits", "digital_collision", "independent_target_alignment", "control_ground_truth_false"],
            "Digital visual trajectory only; not real action, contact truth or physical deployment authority."),
    }
    entries = []
    for task_id, packet in packets.items():
        path = PACKETS / task_id / "TASK_PACKET.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, packet)
        atomic_write(path.parent / "CONTEXT_CARD.md", (f"# {task_id}\n\n{packet['objective']}\n\n只读取TASK_PACKET.read_set。失败必须在预算内进入终态。\n").encode())
        entries.append({"task_id": task_id, "packet_path": str(path.relative_to(REPO_ROOT)), "packet_sha256": artifact_ref(path)["sha256"]})

    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    source = Path(str(pointer["index_path"])); source = source if source.is_absolute() else REPO_ROOT / source
    old_index = load_json(source)
    by_id = {str(x["task_id"]): x for x in old_index.get("task_packets", [])}
    for entry in entries: by_id[entry["task_id"]] = entry
    new_index = dict(old_index)
    new_index.update(packet_revision="RC1_FINAL_0003_READY_T1_T3", supersedes_index=artifact_ref(source), task_packets=list(by_id.values()))
    index_path = OUT / "TASK_PACKET_INDEX.json"

    existing = {str(x.get("task_id")) for x in state.get("tasks", [])}
    created = now_iso()
    for task_id, packet in packets.items():
        if task_id in existing: raise RuntimeError(f"task already registered: {task_id}")
        state["tasks"].append({"task_id": task_id, "phase": packet["stage"], "plan_execution_revision": "RC1", "attempt": 0, "status": "PENDING", "updated_at": created, "heartbeat_at": None, "session": None, "pid": None, "proc_start_ticks": None, "gpu_id": None, "task_packet": artifact_ref(PACKETS / task_id / "TASK_PACKET.json")})
    state["next_task"] = {"task_id": "rc1_t1_sam31_mask_bounded", "session": None, "prerequisites": packets["rc1_t1_sam31_mask_bounded"]["prerequisites"], "expected_resource": "CPU reference freeze then serialized GPU canary", "stop_condition": "Two bounded SAM3.1 revisions or earlier terminal."}
    result = {"schema_version": "chaoyang-rc1-ready-scheduler-result-v1", "task_id": "rc1_schedule_ready_after_t0", "status": "PASSED", "created_at": created, "registered": sorted(packets), "t0_next_action_correction": {"incorrect": "rc1_t1_sam31_bounded_repair", "canonical": "rc1_t1_sam31_mask_bounded"}}
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "RESULT.json"; atomic_json(result_path, result)
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": "rc1_schedule_ready_after_t0", "attempt": 1, "status": "PASSED", "created_at": created, "message": "Registered canonical RC1 T1 and T3 packets; corrected T0 next-action alias without modifying immutable T0 receipt.", "result": artifact_ref(result_path)}])[-100:]
    published = publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="RC1_READY_TASKS_REGISTERED", expected_revision=args.expected_revision, generator_path=Path(__file__), task_packet_index_path=index_path, task_packet_index_value=new_index)
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "registered": sorted(packets)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
