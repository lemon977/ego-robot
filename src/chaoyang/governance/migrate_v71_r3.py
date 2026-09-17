"""Publish the V7.1-R3 governance migration through the existing CAS writer."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    GOVERNANCE_ROOT,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    canonical_bytes,
    load_json,
    now_iso,
    process_identity,
    publish_bundle,
    validate_artifact_ref,
)


RUN_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_chaoyang_v71_r3/MIGRATE-00"
ATTEMPT_ROOT = RUN_ROOT / "attempts/attempt_0001"
PLAN_PATH = GOVERNANCE_ROOT / "CHAOYANG_V7_1_R3_EXECUTION_PLAN_ZH.md"
PREVIOUS_PLAN_PATH = GOVERNANCE_ROOT / "EXACT78_V7_1_EXECUTION_PLAN_ZH.md"
H4_FORMAL_RESULT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/RESULT.json"
H4_DEV_RESULT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/attempts/attempt_0004/TERMINAL_RESULT.json"
V74_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v74"
V74_LOG = V74_ROOT / "RUNNER_ATTEMPT_0002.log"
V74_ATTEMPT1 = V74_ROOT / "ATTEMPT_0001_FAILED_RUNTIME.json"
OLD_CLEAN_AUDIT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/TERMINAL_AUDIT_V52_1.json"


def _write_once_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return
    atomic_json(path, value)


def _write_once_text(path: Path, value: str) -> None:
    encoded = value.encode("utf-8")
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return
    atomic_write(path, encoded)


def _task(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    row = next((item for item in state["tasks"] if item.get("task_id") == task_id), None)
    if row is None:
        row = {"task_id": task_id, "phase": "unspecified", "attempt": 0, "status": "PENDING", "updated_at": now_iso()}
        state["tasks"].append(row)
    return row


def _close_task(
    state: dict[str, Any], task_id: str, status: str, phase: str, result: Path, message: str
) -> None:
    row = _task(state, task_id)
    row.update(
        status=status,
        phase=phase,
        updated_at=now_iso(),
        heartbeat_at=None,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        result=artifact_ref(result),
    )
    state["recent_events"] = (
        state.get("recent_events", [])
        + [{
            "task_id": task_id,
            "session": row.get("session"),
            "attempt": row.get("attempt", 0),
            "status": status,
            "created_at": now_iso(),
            "message": message,
            "result": artifact_ref(result),
        }]
    )[-100:]


def _terminate_expected(pid: int, start_ticks: int) -> dict[str, Any]:
    before = process_identity(pid)
    outcome = {"pid": pid, "expected_start_ticks": start_ticks, "before": before, "signal_sent": False}
    if not before["alive"]:
        outcome["result"] = "ALREADY_EXITED"
        return outcome
    if before["start_ticks"] != start_ticks:
        outcome["result"] = "IDENTITY_MISMATCH_NOT_TOUCHED"
        return outcome
    os.kill(pid, signal.SIGTERM)
    outcome["signal_sent"] = True
    for _ in range(30):
        time.sleep(0.1)
        if not process_identity(pid)["alive"]:
            outcome["result"] = "TERMINATED"
            return outcome
    outcome["result"] = "SIGTERM_TIMEOUT"
    return outcome


def _preflight() -> None:
    for path in (PLAN_PATH, PREVIOUS_PLAN_PATH, H4_FORMAL_RESULT, H4_DEV_RESULT, V74_LOG, V74_ATTEMPT1, OLD_CLEAN_AUDIT):
        if not path.is_file():
            raise FileNotFoundError(path)
    failure_tail = V74_LOG.read_text(encoding="utf-8", errors="replace")[-5000:]
    if "KeyError: 'result'" not in failure_tail:
        raise RuntimeError("v74 failure signature not found in the immutable log tail")
    formal = load_json(H4_FORMAL_RESULT)
    if formal.get("status") != "BLOCKED_RESOURCE" or "No SAM3.1 inference ran" not in formal.get("claim_limit", ""):
        raise RuntimeError("formal H4 result does not support BLOCKED_RESOURCE/NOT_EVALUATED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--terminate-dependent-watchers", action="store_true")
    args = parser.parse_args()
    _preflight()
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)

    old_receipt_path = GOVERNANCE_ROOT / "CURRENT_STATUS_RECEIPT.json"
    old_receipt_ref = artifact_ref(old_receipt_path)
    task_packet = {
        "schema_version": "exact78-task-packet-v1",
        "task_id": "migrate_00_v71_r3",
        "plan_revision": "chaoyang-v7.1",
        "objective": "Publish V7.1-R3 governance contracts and close stale v74 state without hand-editing current JSON.",
        "prerequisites": ["revision_matches_CAS", "v74_failure_signature_present", "formal_H4_result_present"],
        "read_set": [
            str(old_receipt_path), str(TASK_STATE_PATH), str(AUTHORITY_PATH), str(PREVIOUS_PLAN_PATH),
            str(PLAN_PATH), str(V74_LOG), str(H4_FORMAL_RESULT), str(OLD_CLEAN_AUDIT),
        ],
        "write_set": [str(GOVERNANCE_ROOT), str(ATTEMPT_ROOT)],
        "output_contract": ["RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json"],
        "gates": ["CAS", "receipt_binding", "unique_CURRENT_document_scope", "v74_terminal", "H4_formal_dev_separation"],
        "attempt_max": 1,
        "budgets": {"cpu_seconds": 1800, "gpu_seconds": 0, "wall_seconds": 3600},
        "claim_limit": "Governance migration only; no algorithm, Robot, Mask, Contact or physical authority.",
        "stop_condition": "PASSED or explicit CAS/reference failure; never remain RUNNING.",
    }
    _write_once_json(ATTEMPT_ROOT / "TASK_PACKET.json", task_packet)
    _write_once_text(
        ATTEMPT_ROOT / "CONTEXT_CARD.md",
        "# MIGRATE-00 V7.1-R3\n\n只恢复治理、绑定当前计划/算法/文档合同，并按现有证据关闭 v74 幽灵任务。"
        "不修改 Robot 算法，不把 H4 开发 canary 覆盖到正式 H4。\n",
    )

    terminations: list[dict[str, Any]] = []
    if args.terminate_dependent_watchers:
        for pid, ticks in ((2860821, 170160193), (2861125, 170169311), (2954655, 171459652), (2954910, 171462920)):
            terminations.append(_terminate_expected(pid, ticks))

    v74_failure = {
        "schema_version": "robot-runner-failed-runtime-final-v1",
        "task_id": "robot_geometry_expansion_v74",
        "attempt_id": "attempt_0002",
        "status": "FAILED_RUNTIME_FINAL",
        "generated_at": now_iso(),
        "process_identity": {"pid": 2952773, "expected_start_ticks": 171433738, "observed": process_identity(2952773)},
        "cause": "FINALIZER_SCHEMA_KEYERROR_RESULT",
        "exception": "KeyError: 'result'",
        "log": artifact_ref(V74_LOG),
        "previous_runtime_attempt": artifact_ref(V74_ATTEMPT1),
        "partial_outputs_authorized": False,
        "successor": "robot_geometry_expansion_v75",
        "claim_limit": "Runtime final for v74 only; completed batch artifacts require separate SHA adopt proof and are not promoted here.",
    }
    _write_once_json(ATTEMPT_ROOT / "V74_FAILED_RUNTIME_FINAL.json", v74_failure)

    blocked_dependents: dict[str, Path] = {}
    for task_id in ("robot_hard_soft_watcher_v74", "visual_aux_candidate_bundle_v74", "occlusion_silver_v74"):
        path = ATTEMPT_ROOT / f"{task_id}_BLOCKED_PREREQ.json"
        _write_once_json(path, {
            "schema_version": "v74-dependent-blocked-prereq-v1",
            "task_id": task_id,
            "status": "BLOCKED_PREREQ",
            "generated_at": now_iso(),
            "upstream": artifact_ref(ATTEMPT_ROOT / "V74_FAILED_RUNTIME_FINAL.json"),
            "resume_condition": "A new immutable v75 successor publishes compatible candidate evidence.",
            "claim_limit": "Dependency closure only; no Robot, Occlusion or Visual Aux authority.",
        })
        blocked_dependents[task_id] = path

    plan_revision = {
        "schema_version": "chaoyang-plan-revision-v2",
        "plan_revision": "chaoyang-v7.1",
        "execution_revision": "R3",
        "migration_id": "MIGRATE-00-V71-R3",
        "approved_source": {**artifact_ref(PLAN_PATH), "lines": len(PLAN_PATH.read_text(encoding="utf-8").splitlines())},
        "canonical_plan_path": "docs/governance/CHAOYANG_V7_1_R3_EXECUTION_PLAN_ZH.md",
        "wave0_selection_sha256": "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1",
        "global_hard_gate": "G0_CORE_GOVERNANCE",
        "local_debt_gate": "G0_PROVENANCE_DEBT",
        "artifact_revision_policy": "IMMUTABLE_PINNED_R7",
        "training_input_policy": "CAUSAL_TRAINING_INPUT_ONLY",
        "terminal_names": ["EXACT78_TERMINAL_COMPLETE", "EXACT78_WAVE0_CLEAN_TERMINAL_COMPLETE", "EXACT78_END_TO_END_COMPLETE"],
        "status": "APPROVED_FOR_EXECUTION",
    }
    migration_value = {
        "schema_version": "chaoyang-plan-migration-receipt-v1",
        "migration_id": "MIGRATE-00-V71-R3",
        "generated_at": "FILLED_BY_AGGREGATOR",
        "generation_id": "FILLED_BY_AGGREGATOR",
        "effective_governance_revision": args.expected_revision + 1,
        "effective_status": "CURRENT",
        "previous_plan": artifact_ref(PREVIOUS_PLAN_PATH),
        "new_plan": artifact_ref(PLAN_PATH),
        "task_migrations": [
            {"from_task_id": "g0_core_governance_v71", "to_task_id": "migrate_00_v71_r3", "migration_status": "MIGRATED", "reason": "Add document/algorithm contracts and current receipt binding."},
            {"from_task_id": "robot_geometry_expansion_v74", "to_task_id": "robot_geometry_expansion_v75", "migration_status": "TERMINATED", "reason": "v74 PID died after finalizer KeyError; v75 must be immutable."},
            {"from_task_id": "successor_role_mask_v71", "to_task_id": "mask_role_r3", "migration_status": "MIGRATED", "reason": "Separate role identity from object identity."},
            {"from_task_id": "successor_object_identity_v71", "to_task_id": "mask_object_r3", "migration_status": "MIGRATED", "reason": "Keep physical object identity and re-entry gates independent."},
            {"from_task_id": "sensor_h3_stereo_v1", "to_task_id": "depth_00_10_20_r3", "migration_status": "MIGRATED", "reason": "Add coordinate, uncertainty and wrist fusion work."},
            {"from_task_id": "contact_evidence_dag_v1", "to_task_id": "contact_10_r3", "migration_status": "MIGRATED", "reason": "Add per-finger state and uncertainty while preserving evidence direction."},
            {"from_task_id": "exact78_clean_r70_v71", "to_task_id": "clean_20_21_r3", "migration_status": "MIGRATED", "reason": "Current 58 structural passes use recovery_v53 evidence; old 39/19 audit is superseded and successor work targets contact semantics."},
            {"from_task_id": "visual_aux_chips_pair_v1", "to_task_id": "visual_aux_chips_pair_v1", "migration_status": "UNCHANGED", "reason": "Causal future-2D pair remains valid."},
            {"from_task_id": "visual_aux_poker_pair_v1", "to_task_id": "visual_aux_poker_pair_v1", "migration_status": "UNCHANGED", "reason": "Causal future-2D pair remains valid."},
            {"from_task_id": "cleanup_current_only_v71", "to_task_id": "cleanup_current_only_v71", "migration_status": "UNCHANGED", "reason": "Current CLI is dry-run by default and requires explicit --commit-delete."},
        ],
        "claim_limit": "Plan/task routing migration only; no stage result or authority is promoted.",
    }

    _close_task(state, "robot_geometry_expansion_v74", "FAILED_RUNTIME_FINAL", "robot_batch_finalize_runtime_failure", ATTEMPT_ROOT / "V74_FAILED_RUNTIME_FINAL.json", "Recovered dead v74 PID from immutable log evidence; no partial promotion.")
    for task_id, result_path in blocked_dependents.items():
        _close_task(state, task_id, "BLOCKED_PREREQ", "wait_immutable_v75_successor", result_path, "v74 upstream terminated; wait for immutable v75.")
    _close_task(state, "occlusion_silver_v1", "BLOCKED_PREREQ", "contract_ready_wait_causal_candidate", REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/occlusion_silver/RESULT.json", "Restored formal blocked receipt; development watcher is not Silver authority.")
    _close_task(state, "sensor_h4_mask_v1", "BLOCKED_RESOURCE", "sensor_h4_formal_preflight_not_evaluated", H4_FORMAL_RESULT, "Formal H4 ran no SAM3.1 pixel inference; development canary remains separate evidence.")
    h4 = _task(state, "sensor_h4_mask_v1")
    h4.update(qa_status="NOT_EVALUATED", policy_status="POLICY_DEFERRED", pixel_mask_authority=False, development_canary=artifact_ref(H4_DEV_RESULT))

    result = {
        "schema_version": "migrate-00-v71-r3-result-v1",
        "task_id": "migrate_00_v71_r3",
        "status": "PASSED",
        "generated_at": now_iso(),
        "input_receipt": old_receipt_ref,
        "v74_terminal": artifact_ref(ATTEMPT_ROOT / "V74_FAILED_RUNTIME_FINAL.json"),
        "formal_h4": artifact_ref(H4_FORMAL_RESULT),
        "development_h4": artifact_ref(H4_DEV_RESULT),
        "old_clean_audit": artifact_ref(OLD_CLEAN_AUDIT),
        "clean_current_interpretation": "58 current structural passes are supported by recovery_v53 evidence; the old 39-pass/19-runtime-final audit is superseded.",
        "terminated_processes": terminations,
        "claim_limit": "Governance migration and stale-task recovery only; no algorithm authority.",
    }
    _write_once_json(ATTEMPT_ROOT / "RESULT.json", result)
    _write_once_json(ATTEMPT_ROOT / "METRICS.json", {"status": "PASSED", "closed_dead_tasks": 1, "restored_formal_statuses": 2, "blocked_v74_dependents": len(blocked_dependents), "gpu_seconds": 0})
    _write_once_json(ATTEMPT_ROOT / "RUN_RECEIPT.json", {"status": "PASSED", "expected_revision": args.expected_revision, "generator": artifact_ref(Path(__file__)), "input_receipt": old_receipt_ref, "claim_limit": "CAS publication intent; final current binding is CURRENT_STATUS_RECEIPT.json."})
    _write_once_text(ATTEMPT_ROOT / "DECISION.md", "# MIGRATE-00 决定\n\nV7.1-R3 成为当前执行合同；v74 以运行失败终态关闭；H4 正式状态与开发 canary 分离；旧 Clean 39/19 审计退出 current 口径。\n")
    _write_once_json(ATTEMPT_ROOT / "NEXT_ACTION.json", {"next_task_id": "robot_geometry_expansion_v75", "prerequisites": ["governance=PASS/FRESH", "v74=FAILED_RUNTIME_FINAL"], "stop_condition": "Immutable v75 terminal or bounded failure."})
    manifest_paths = [
        ATTEMPT_ROOT / "TASK_PACKET.json", ATTEMPT_ROOT / "CONTEXT_CARD.md",
        ATTEMPT_ROOT / "V74_FAILED_RUNTIME_FINAL.json", *blocked_dependents.values(),
        ATTEMPT_ROOT / "RESULT.json", ATTEMPT_ROOT / "METRICS.json",
        ATTEMPT_ROOT / "RUN_RECEIPT.json", ATTEMPT_ROOT / "DECISION.md", ATTEMPT_ROOT / "NEXT_ACTION.json",
        PLAN_PATH,
    ]
    _write_once_json(ATTEMPT_ROOT / "ARTIFACT_MANIFEST.json", {"schema_version": "migrate-00-artifact-manifest-v1", "artifacts": [artifact_ref(path) for path in manifest_paths], "claim_limit": "Immutable MIGRATE-00 outputs excluding self-reference."})

    _close_task(state, "migrate_00_v71_r3", "PASSED", "governance_migration", ATTEMPT_ROOT / "RESULT.json", "V7.1-R3 document and algorithm contracts published through CAS.")
    state["next_task"] = {
        "task_id": "robot_geometry_expansion_v75",
        "session": "v74_unfinished_selection",
        "prerequisites": ["governance=PASS/FRESH", "robot_geometry_expansion_v74=FAILED_RUNTIME_FINAL"],
        "expected_resource": "CPU Robot geometry successor; GPU only for bounded render when claimed",
        "stop_condition": "Immutable v75 terminals or explicit bounded failures",
    }
    receipt = publish_bundle(
        authority,
        state,
        event_type="PLAN_MIGRATED_V71_R3",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        plan_revision_value=plan_revision,
        plan_migration_value=migration_value,
    )
    errors = [error for reference in receipt["files"].values() for error in validate_artifact_ref(reference)]
    if errors:
        raise RuntimeError("post-publish current binding failed:\n" + "\n".join(errors))
    print(json.dumps({"status": "PASSED", "revision": receipt["governance_revision"], "receipt": str(GOVERNANCE_ROOT / "CURRENT_STATUS_RECEIPT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
