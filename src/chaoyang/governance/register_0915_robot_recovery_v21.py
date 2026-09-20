#!/usr/bin/env python3
"""CAS-register the single 0915 Robot recovery V2.1 parent task."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "0915_robot_quality_recovery_15h_v2"
PLAN_REVISION = "0915_ROBOT_RECOVERY_15H_V2_1"
EXECUTION_REVISION = "0915_ROBOT_RECOVERY_15H_V2_1"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
CLOCK = REPO_ROOT / "_run/current/0915_robot_quality_recovery_15h_v2/WINDOW_CLOCK.json"
DOC_ROOT = REPO_ROOT / "docs/chaoyang_robot_recovery_15h_v2"
RUN_ROOT = REPO_ROOT / "_run/current/0915_robot_quality_recovery_15h_v2"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
PREDECESSOR_FINAL_AUDIT = (
    REPO_ROOT
    / "_run/current/0915_robot15h_window_release_audit_v1/attempts/attempt_0001/FINAL_AUDIT.json"
)
PREDECESSOR_RESULT = (
    REPO_ROOT
    / "_run/current/0915_robot15h_window_release_audit_v1/attempts/attempt_0001/RESULT.json"
)


def write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def build_packet() -> dict[str, Any]:
    clock_ref = artifact_ref(CLOCK)
    input_refs = {
        "window_clock": clock_ref,
        "predecessor_terminal_result": artifact_ref(PREDECESSOR_RESULT),
        "predecessor_final_audit": artifact_ref(PREDECESSOR_FINAL_AUDIT),
        "batch_manifest": artifact_ref(
            REPO_ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
        ),
        "source_group_manifest": artifact_ref(
            REPO_ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/SOURCE_GROUP_MANIFEST.json"
        ),
        "document_bundle": [
            artifact_ref(DOC_ROOT / name)
            for name in (
                "00_START_HERE_ZH.md",
                "01_EXECUTION_PLAN_15H_ZH.md",
                "02_ALGORITHM_AUDIT_LIVE_ZH.md",
                "03_TASK_MANIFEST.template.json",
                "04_ACCEPTANCE_TESTS_ZH.md",
                "05_TECHNICAL_SOURCES_ZH.md",
                "SHA256SUMS.txt",
            )
        ],
    }
    return {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK_ID,
        "objective": (
            "Execute the fifteen-hour V2.1 recovery protocol over the frozen 0915 cohort, "
            "separating W0 development, W1-DIAG, W1-ADOPTION, H9 holdout and sealed EXTRA FINAL."
        ),
        "phase": "ROBOT_QUALITY_RECOVERY_0915_15H_V21",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "window_run_id": load_json(CLOCK)["run_id"],
        "window_clock": clock_ref,
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/chaoyang_robot_recovery_15h_v2/01_EXECUTION_PLAN_15H_ZH.md",
            "docs/chaoyang_robot_recovery_15h_v2/02_ALGORITHM_AUDIT_LIVE_ZH.md",
            "docs/chaoyang_robot_recovery_15h_v2/03_TASK_MANIFEST.template.json",
            "docs/chaoyang_robot_recovery_15h_v2/04_ACCEPTANCE_TESTS_ZH.md",
            "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json",
            "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/SOURCE_GROUP_MANIFEST.json",
            "_run/current/0915_robot_quality_recovery_15h_v2/WINDOW_CLOCK.json",
        ],
        "write_set": [
            "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001",
            "tasks/receipts/0915_ROBOT_QUALITY_RECOVERY_15H_V21_RESULT.json",
            "docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2",
            "docs/current/AI_WORK_ENTRY_ZH.md",
            "docs/current/README_ZH.md",
            "docs/current/ROBOT15H_0915_EXECUTION_V1_ZH.md",
            "docs/current/visuals/README_ZH.md",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "no_active_current_task",
            "0915_processed_root_READ_ONLY",
            "0916_downstream_FORBIDDEN",
            "egosteer_touch_MUST_NOT_BE_STOPPED",
            "single_parent_with_internal_isolated_writers",
            "single_GPU_lease_owner",
            "source_group_claim_METADATA_VERIFIED_NO_USER_ATTESTATION",
            "Clean_Removal_training_RL_real_control_FORBIDDEN",
        ],
        "required_outputs": [
            "SOURCE_GROUP_PROVENANCE_CORRECTION_V2.json",
            "COHORT_ACCESS_LEDGER_V1.json",
            "W0_FAILURE_MATRIX_V2.json",
            "CONSUMER_ADMISSION_V1.json",
            "WINDOW_VALIDITY_V1.json",
            "CONTACT_OBSERVABILITY_AUDIT_V1.json",
            "LOCAL_STEREO_METRIC_DEV_V1.json",
            "CANDIDATE_FREEZE_V1.json",
            "ADOPTION_DECISION_V1.json",
            "TECHNICAL_CONTRACT_SNAPSHOT_INDEX_V1.json",
            "ALGORITHM_AUDIT_LIVE_V2.json",
            "FINAL_AUDIT.json",
            "CLAIM.json",
            "RUN_SIGNATURE.json",
            "RESULT.json",
            "RUN_RECEIPT.json",
            "WORK_PACKAGE_LEDGER_T0.json",
            "P0_RESOLUTION_V1.json",
        ],
        "budgets": {
            "wall_clock_seconds": 54000,
            "gpu_concurrent_owners_max": 1,
            "p0_seconds": 2700,
            "r2_solver_seconds": 5400,
            "algorithm_freeze_elapsed_seconds": 32400,
            "stop_new_sessions_elapsed_seconds": 48600,
            "runtime_attempts_per_candidate": 1,
        },
        "stop_conditions": [
            "deadline_2026_09_20T14_44_12_plus08",
            "all_started_packages_terminal",
            "H13p5_no_new_sessions",
            "candidate_quality_C_no_automatic_retry",
            "external_or_scope_violation_fail_closed",
        ],
        "weights": "ABSENT",
        "child_weights_policy": "EACH_CHILD_RUN_SIGNATURE_MUST_BIND_EXACTLY_ONE_PINNED_WEIGHT_OR_ABSENT",
        "calibration_or_absent": "ABSENT",
        "external_metric_authority": False,
        "code_closure": [
            artifact_ref(REPO_ROOT / "src/chaoyang/ops/run_0915_robot_quality_recovery_15h_v2.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/governance/register_0915_robot_recovery_v21.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/governance/finalize_0915_robot_recovery_v21.py"),
        ],
        "input_manifest": input_refs,
        "dag_dependencies": ["0915_robot15h_window_release_audit_v1"],
        "dependency_refs": {
            "0915_robot15h_window_release_audit_v1": input_refs["predecessor_terminal_result"],
        },
        "expected_resource": "CPU_LANES_PLUS_SINGLE_GOVERNED_GPU_LEASE",
        "executor_epoch": 1,
        "fencing": {
            "pid_startticks_required": True,
            "immutable_final": True,
            "unique_primary_writer": True,
        },
        "conditional_output_contract": {
            "p0_lineage_resolution": {
                "one_of": ["R2_CONTRACT_SNAPSHOT_V1.json", "LINEAGE_TOMBSTONE_V1.json"],
                "both_allowed": False,
            }
        },
        "attempt_max": 1,
        "claim_limit": (
            "Development-relative offline Robot recovery only. No training, control, external metric, "
            "physical deployment or Contact ground-truth authority."
        ),
    }


def validate_clock() -> dict[str, Any]:
    clock = load_json(CLOCK)
    expected = {
        "schema_version": "0915-robot-recovery-15h-v21-window-clock-v1",
        "plan_version": "v2.1",
        "started_at": "2026-09-19T23:44:12+08:00",
        "deadline_at": "2026-09-20T14:44:12+08:00",
        "duration_seconds": 54000,
        "t0_policy": "WRITE_ONCE_FROM_FIRST_EXECUTION_WRITE_NOT_RESET_BY_RETRY_OR_SUCCESSOR",
    }
    for key, value in expected.items():
        if clock.get(key) != value:
            raise RuntimeError(f"V2.1 clock mismatch: {key}")
    if datetime.now().astimezone() >= datetime.fromisoformat(str(clock["deadline_at"])):
        raise RuntimeError("V2.1 clock has already expired")
    return clock


def validate_document_bundle() -> None:
    checksums = DOC_ROOT / "SHA256SUMS.txt"
    rows = [line.split(maxsplit=1) for line in checksums.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 6:
        raise RuntimeError("V2.1 SHA256SUMS must bind exactly six documents")
    from chaoyang.governance.common import sha256_file

    for expected_sha, raw_name in rows:
        name = raw_name.lstrip("* ")
        path = DOC_ROOT / name
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"V2.1 document checksum mismatch: {name}")


def validate_predecessor() -> None:
    audit = load_json(PREDECESSOR_FINAL_AUDIT)
    if audit.get("task_id") != "0915_robot15h_window_release_audit_v1":
        raise RuntimeError("unexpected predecessor final audit")
    if audit.get("status") != "REJECTED" or audit.get("sidecar_audit", {}).get("status") != "PASSED":
        raise RuntimeError("predecessor campaign is not terminal with a closed sidecar audit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    expected_output = (RUN_ROOT / "registration_0001").resolve()
    if output != expected_output:
        raise RuntimeError(f"fixed V2.1 registration namespace required: {expected_output}")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh registration output required: {output}")
    clock = validate_clock()
    for name in ("00_START_HERE_ZH.md", "01_EXECUTION_PLAN_15H_ZH.md", "02_ALGORITHM_AUDIT_LIVE_ZH.md", "03_TASK_MANIFEST.template.json", "04_ACCEPTANCE_TESTS_ZH.md", "05_TECHNICAL_SOURCES_ZH.md", "SHA256SUMS.txt"):
        if not (DOC_ROOT / name).is_file():
            raise RuntimeError(f"V2.1 document missing: {name}")
    validate_document_bundle()
    validate_predecessor()

    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before V2.1 registration")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))
    if receipt.get("freshness", {}).get("status") != "FRESH":
        raise RuntimeError("V2.1 registration requires a FRESH current receipt")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None or any(row.get("status") in LIVE for row in state.get("tasks", [])):
        raise RuntimeError("V2.1 registration requires no live task")
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError("V2.1 parent already registered")

    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    pointed = REPO_ROOT / str(pointer["index_path"])
    if pointed.resolve() != CURRENT_INDEX.resolve():
        raise RuntimeError("current task pointer does not resolve to tasks/current/INDEX.json")
    current_ref = artifact_ref(CURRENT_INDEX)
    if pointer.get("index_bytes") != current_ref["bytes"] or pointer.get("index_sha256") != current_ref["sha256"]:
        raise RuntimeError("current task pointer SHA/bytes mismatch")
    current = load_json(CURRENT_INDEX)
    if current.get("task_packets") != [] or current.get("status") != "PASS_NO_ACTIVE_TASKS":
        raise RuntimeError("current task index is not empty and terminal")

    packet = build_packet()
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid V2.1 parent packet: " + "; ".join(errors))
    for raw in packet["read_set"]:
        if not (REPO_ROOT / raw).exists():
            raise RuntimeError(f"read_set path missing: {raw}")
    packet_path = REPO_ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    if packet_path.parent.exists() or packet_path.parent.is_symlink():
        expected_bytes = json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
        if not packet_path.is_file() or packet_path.read_bytes() != expected_bytes:
            raise RuntimeError("existing V2.1 task packet conflicts with immutable retry")
    output.mkdir(parents=True)
    frozen_index = output / "PREDECESSOR_TASK_PACKET_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, frozen_index)
    write_once(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "0915_ROBOT_RECOVERY_15H_V21_PARENT_ROUTABLE",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "status": "PASS",
        "supersedes_index": artifact_ref(frozen_index),
        "task_packets": [{
            "task_id": TASK_ID,
            "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
            "weights": "ABSENT",
            "window_run_id": clock["run_id"],
        }],
        "claim_limit": "Exactly one V2.1 parent is routable; internal packages use isolated roots and one serial publisher.",
    }
    created = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID,
        "phase": packet["phase"],
        "plan_execution_revision": EXECUTION_REVISION,
        "attempt": 0,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": None,
        "session": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "task_packet": packet_ref,
        "window_run_id": clock["run_id"],
        "window_clock": artifact_ref(CLOCK),
    })
    state["next_task"] = {
        "task_id": TASK_ID,
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU_LANES_PLUS_SINGLE_GOVERNED_GPU_LEASE",
        "stop_condition": "Finite V2.1 terminal by the frozen 15-hour deadline without weakening quality gates.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK_ID,
        "session": None,
        "attempt": 0,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered the single V2.1 parent; no algorithm result yet.",
        "task_packet": packet_ref,
        "window_run_id": clock["run_id"],
    }])[-100:]
    result_path = output / "RESULT.json"
    write_once(result_path, {
        "schema_version": "0915-robot-recovery-v21-registration-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "registered_at": created,
        "window_clock": artifact_ref(CLOCK),
        "task_packet": packet_ref,
        "execution_started": False,
        "claim_limit": "Registration only; no algorithm, quality, Contact or Robot result.",
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="0915_ROBOT_RECOVERY_15H_V21_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=successor,
    )
    write_once(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot-recovery-v21-registration-receipt-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "result": artifact_ref(result_path),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "execution_started": False,
    })
    print(json.dumps({"status": "PASSED", "task_id": TASK_ID, "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
