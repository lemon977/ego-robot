#!/usr/bin/env python3
"""CAS-publish RC1-FINAL and register the first bounded task packet."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import json
import os
from pathlib import Path
import socket

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH,
    AUTHORITY_PATH,
    DOC_AUTHORITY_MAP_PATH,
    PLAN_REVISION_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    load_json,
    now_iso,
    publish_bundle,
)


GOV = REPO_ROOT / "docs/governance"
PLAN = GOV / "CHAOYANG_RC1_FINAL_DELIVERY_PLAN_ZH.md"
RELEASE_SPEC = GOV / "RC1_RELEASE_SPEC.json"
CONTRACT = GOV / "VISUAL_AUX_RC1_CONTRACT.json"
TASK_DAG = GOV / "RC1_TASK_DAG.json"
PREVIOUS_PLAN = GOV / "CHAOYANG_R2_2_4H_INTEGRATION_PLAN_ZH.md"
RUN_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1"
ATTEMPT_ROOT = RUN_ROOT / "migration/attempts/attempt_0001"
PACKET_ROOT = RUN_ROOT / "task_packets"


def _write_json(path: Path, value: dict) -> None:
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError(f"prepared immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"prepared immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, value.encode("utf-8"))


def _packet(snapshot: Path) -> dict:
    return {
        "schema_version": "chaoyang-rc1-task-packet-v1",
        "task_id": "rc1_t0_freeze_capacity_split",
        "stage": "RC1-T0",
        "objective": "Freeze the 156-row denominator, prove independent source groups, freeze scheduled H50 starts and report capacity before quality.",
        "non_goals": [
            "Do not run GPU inference",
            "Do not train checkpoints",
            "Do not count clip/session fragments as independent source groups",
            "Do not modify historical authority or quality C",
        ],
        "frozen_inputs": {
            "run_start_snapshot": artifact_ref(snapshot),
            "plan": artifact_ref(PLAN),
            "release_spec": artifact_ref(RELEASE_SPEC),
            "visual_aux_contract": artifact_ref(CONTRACT),
            "task_dag": artifact_ref(TASK_DAG),
        },
        "prerequisites": ["governance=FRESH", "current_delivery_plan=RC1_FINAL"],
        "read_set": [
            str(GOV / "CURRENT_RC1_STATUS_MIN.json"),
            str(RELEASE_SPEC),
            str(CONTRACT),
            str(TASK_DAG),
            str(REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/exact78_terminal/EXACT78_FINAL_TERMINAL_MATRIX.json"),
            str(REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_visual_aux_capacity_preflight_v71/RESULT.json"),
        ],
        "write_set": [str(RUN_ROOT / "t0_freeze_capacity_split/attempts")],
        "commands": [
            "chaoyang validate-governance",
            "chaoyang run run_rc1_t0_freeze --expected-revision <current_revision>",
        ],
        "quality_gates": [
            "exactly_156_unique_sessions",
            "chips_78_poker_78",
            "source_group_evidence_bound",
            "no_source_group_cross_split",
            "scheduled_starts_frozen_before_quality",
        ],
        "budgets": {"wall_seconds": 7200, "gpu_seconds": 0, "runtime_attempts": 2},
        "stop_conditions": ["PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_REFERENCE_PROOF"],
        "expected_outputs": [
            "MASTER_LEDGER.json",
            "SOURCE_GROUP_LEDGER.json",
            "CAPACITY_REPORT.json",
            "RESULT.json",
            "ARTIFACT_MANIFEST.json",
            "METRICS.json",
            "RUN_RECEIPT.json",
            "DECISION.md",
            "NEXT_ACTION.json",
            "RESULT_SUMMARY.json",
        ],
        "claim_limit": "Capacity and split proof only; no Mask/Clean/Robot eligibility or checkpoint is produced.",
        "ai_io_limits": {
            "max_files_initial_read": 8,
            "max_search_results": 20,
            "max_log_tail_lines": 80,
            "max_directory_depth": 3,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    for path in (PLAN, RELEASE_SPEC, CONTRACT, TASK_DAG, PREVIOUS_PLAN):
        path.resolve(strict=True)

    current = load_json(RECEIPT_PATH)
    if current.get("governance_revision") != args.expected_revision:
        raise RuntimeError("RC1 migration CAS revision mismatch")
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    active = [
        task for task in state.get("tasks", [])
        if task.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    ]
    if active:
        raise RuntimeError(f"RC1 migration requires no active tasks: {[x.get('task_id') for x in active]}")

    prepared_snapshot = ATTEMPT_ROOT / "RUN_START_SNAPSHOT.json"
    created = load_json(prepared_snapshot)["created_at"] if prepared_snapshot.is_file() else now_iso()
    snapshot = load_json(prepared_snapshot) if prepared_snapshot.is_file() else {
        "schema_version": "chaoyang-rc1-run-start-snapshot-v1",
        "created_at": created,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "current_status_receipt": artifact_ref(RECEIPT_PATH),
        "governance_revision": current["governance_revision"],
        "generation_id": current["generation_id"],
        "plan": artifact_ref(PLAN),
        "release_spec": artifact_ref(RELEASE_SPEC),
        "visual_aux_contract": artifact_ref(CONTRACT),
        "task_dag": artifact_ref(TASK_DAG),
        "algorithm_contract": artifact_ref(ALGORITHM_CONTRACT_PATH),
        "doc_authority_map": artifact_ref(DOC_AUTHORITY_MAP_PATH),
        "plan_revision": artifact_ref(PLAN_REVISION_PATH),
        "worker_input_policy": "PINNED_AT_CLAIM_NO_LATEST_AUTO_SWITCH",
    }
    snapshot_path = ATTEMPT_ROOT / "RUN_START_SNAPSHOT.json"
    _write_json(snapshot_path, snapshot)

    packet_path = PACKET_ROOT / "rc1_t0_freeze_capacity_split/TASK_PACKET.json"
    packet = _packet(snapshot_path)
    _write_json(packet_path, packet)
    _write_text(
        packet_path.parent / "CONTEXT_CARD.md",
        "# RC1-T0\n\n目标：证明156条会话的真实source group和容量上限。\n\n只读TASK_PACKET.read_set；禁止把session数当独立录制数。输出终态后由aggregator选择后续任务。\n",
    )

    predecessor_pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    predecessor_path = Path(str(predecessor_pointer["index_path"]))
    if not predecessor_path.is_absolute():
        predecessor_path = REPO_ROOT / predecessor_path
    predecessor = load_json(predecessor_path)
    packet_index_path = ATTEMPT_ROOT / "TASK_PACKET_INDEX.json"
    predecessor_packets = {
        str(item["task_id"]): item for item in predecessor.get("task_packets", [])
        if str(item.get("task_id")) != "rc1_t0_freeze_capacity_split"
    }
    packet_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "RC1_FINAL_0001",
        "plan_revision": "chaoyang-rc1-final",
        "execution_revision": "RC1",
        "status": "PASS",
        "supersedes_index": artifact_ref(predecessor_path),
        "task_packets": list(predecessor_packets.values()) + [
            {
                "task_id": "rc1_t0_freeze_capacity_split",
                "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                "packet_sha256": artifact_ref(packet_path)["sha256"],
            }
        ],
        "claim_limit": "RC1 current delivery routing; registration does not prove execution.",
    }

    state_by_id = {str(row.get("task_id")): row for row in state.get("tasks", [])}
    if "rc1_t0_freeze_capacity_split" in state_by_id:
        raise RuntimeError("RC1-T0 task already registered")
    state["tasks"].append(
        {
            "task_id": "rc1_t0_freeze_capacity_split",
            "phase": "RC1_T0_CAPACITY_AND_SPLIT_FREEZE",
            "plan_execution_revision": "RC1",
            "attempt": 0,
            "status": "PENDING",
            "updated_at": created,
            "heartbeat_at": None,
            "session": None,
            "pid": None,
            "proc_start_ticks": None,
            "gpu_id": None,
            "task_packet": artifact_ref(packet_path),
        }
    )
    state["next_task"] = {
        "task_id": "rc1_t0_freeze_capacity_split",
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU-only source-group and capacity audit",
        "stop_condition": "Publish the 156-row frozen denominator and explicit per-pair capacity terminal.",
    }
    state["rc1_pair_status"] = {"chips": "NOT_EVALUATED", "poker": "NOT_EVALUATED"}
    state["rc1_release_flags"] = {
        "PIPELINE_OPERATIONALLY_CLOSED": False,
        "DATA_MINIMUM_MET_CHIPS": False,
        "DATA_MINIMUM_MET_POKER": False,
        "FOUR_CHECKPOINTS_TRAINED": False,
        "RATE_FINALIZED": False,
        "RC1_RELEASE_STATUS": "MIGRATED_NOT_EXECUTED",
    }
    state["rc1_budget"] = {
        "engineering_hours_limit": 24,
        "engineering_hours_used": 0,
        "production_gpu_hours_limit": 96,
        "production_gpu_hours_used": 0,
        "checkpoint_gpu_hours_limit": 48,
        "checkpoint_gpu_hours_used": 0,
    }

    result = {
        "schema_version": "chaoyang-rc1-migration-result-v1",
        "task_id": "rc1_migrate_00",
        "status": "PASSED",
        "created_at": created,
        "run_start_snapshot": artifact_ref(snapshot_path),
        "registered_next_task": "rc1_t0_freeze_capacity_split",
        "authority_promoted": False,
    }
    result_path = ATTEMPT_ROOT / "RESULT.json"
    _write_json(result_path, result)
    _write_json(ATTEMPT_ROOT / "METRICS.json", {"status": "PASSED", "registered_tasks": 1})
    _write_json(ATTEMPT_ROOT / "NEXT_ACTION.json", {"task_id": "rc1_t0_freeze_capacity_split"})
    _write_text(ATTEMPT_ROOT / "DECISION.md", "# RC1-MIGRATE-00\n\nRC1-FINAL is the sole current delivery plan; V7.1-R3 remains the architecture contract and R2.2 is historical.\n")
    manifest_path = ATTEMPT_ROOT / "ARTIFACT_MANIFEST.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "chaoyang-rc1-migration-artifact-manifest-v1",
            "artifacts": [artifact_ref(path) for path in (
                snapshot_path, packet_path, result_path,
                ATTEMPT_ROOT / "METRICS.json", ATTEMPT_ROOT / "NEXT_ACTION.json",
                ATTEMPT_ROOT / "DECISION.md",
            )],
        },
    )
    _write_json(
        ATTEMPT_ROOT / "RUN_RECEIPT.json",
        {
            "schema_version": "chaoyang-rc1-migration-run-receipt-v1",
            "task_id": "rc1_migrate_00",
            "status": "PASSED",
            "result": artifact_ref(result_path),
            "manifest": artifact_ref(manifest_path),
            "claim_limit": "Governance migration only; no production data or checkpoint was created.",
        },
    )
    _write_json(
        ATTEMPT_ROOT / "RESULT_SUMMARY.json",
        {
            "task_id": "rc1_migrate_00",
            "status": "PASSED",
            "changes": ["RC1_FINAL_CURRENT", "R2_2_HISTORICAL", "RC1_T0_REGISTERED"],
            "next_task": "rc1_t0_freeze_capacity_split",
        },
    )

    state["recent_events"] = (
        state.get("recent_events", [])
        + [{
            "task_id": "rc1_migrate_00",
            "session": None,
            "attempt": 1,
            "status": "PASSED",
            "created_at": created,
            "message": "RC1-FINAL migrated and bounded T0 registered.",
            "result": artifact_ref(result_path),
        }]
    )[-100:]

    plan_revision = {
        "schema_version": "chaoyang-plan-revision-v2",
        "plan_revision": "chaoyang-rc1-final",
        "execution_revision": "RC1",
        "status": "APPROVED_FOR_EXECUTION",
        "canonical_plan_path": str(PLAN.relative_to(REPO_ROOT)),
        "approved_source": artifact_ref(PLAN),
        "architecture_contract": artifact_ref(GOV / "CHAOYANG_V7_1_R3_EXECUTION_PLAN_ZH.md"),
        "training_input_policy": "CAUSAL_TRAINING_INPUT_ONLY",
        "artifact_revision_policy": "IMMUTABLE_RC1",
        "global_hard_gate": "CURRENT_STATUS_RECEIPT_PASS_FRESH",
        "local_debt_gate": "LOCAL_SCOPE_ONLY",
        "migration_id": "RC1-MIGRATE-00",
        "terminal_names": list(load_json(RELEASE_SPEC)["release_flags"]),
    }
    plan_migration = {
        "schema_version": "chaoyang-plan-migration-receipt-v1",
        "migration_id": "RC1-MIGRATE-00",
        "effective_status": "CURRENT",
        "previous_plan": artifact_ref(PREVIOUS_PLAN),
        "new_plan": artifact_ref(PLAN),
        "task_migrations": [
            {
                "from_task_id": "visual_aux_chips_pair_v1",
                "to_task_id": "rc1_t0_freeze_capacity_split",
                "migration_status": "MIGRATED",
                "reason": "RC1 adds source-group, causal-input and consumer-eligibility contracts before pair training.",
            },
            {
                "from_task_id": "visual_aux_poker_pair_v1",
                "to_task_id": "rc1_t0_freeze_capacity_split",
                "migration_status": "MIGRATED",
                "reason": "RC1 adds source-group, causal-input and consumer-eligibility contracts before pair training.",
            },
            {
                "from_task_id": "doc_r22",
                "to_task_id": "rc1_migrate_00",
                "migration_status": "TERMINATED",
                "reason": "R2.2 remains completed history and no longer routes delivery work.",
            },
        ],
        "claim_limit": "Plan migration and task routing only; execution evidence remains in immutable task receipts.",
    }

    receipt = publish_bundle(
        authority,
        state,
        event_type="RC1_FINAL_MIGRATED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        plan_revision_value=plan_revision,
        plan_migration_value=plan_migration,
        task_packet_index_path=packet_index_path,
        task_packet_index_value=packet_index,
    )
    print(json.dumps({
        "status": "PASSED",
        "revision": receipt["governance_revision"],
        "next_task": "rc1_t0_freeze_capacity_split",
        "receipt": str(RECEIPT_PATH),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
