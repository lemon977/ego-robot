#!/usr/bin/env python3
"""CAS-register the single routable parent for the V3.1 three-stream run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.build_three_stream_status_v31 import build_status
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


TASK_ID = "three_stream_stable_baseline_v31"
PLAN_REVISION = "THREE_STREAM_STABLE_BASELINE_V3_1"
EXECUTION_REVISION = "THREE_STREAM_STABLE_BASELINE_V3_1"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RUN_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}"
ATTEMPT_ROOT = RUN_ROOT / "attempts/attempt_0001"
REGISTRATION_ROOT = RUN_ROOT / "registration_0001"
SHALLOW_STATUS = REPO_ROOT / "docs/current/STATUS.json"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def build_packet() -> dict[str, Any]:
    return {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK_ID,
        "objective": (
            "Execute the V3.1 Exact78, AI1 and AI2 stable-baseline lanes with isolated CPU "
            "writers, one governed GPU lease and separately reported development authority."
        ),
        "phase": "THREE_STREAM_STABLE_BASELINE_V31",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/current/PLAN.md",
            "docs/current/EXACT78.md",
            "docs/current/AI1.md",
            "docs/current/AI2.md",
            "tasks/current/INDEX.json",
        ],
        "write_set": [
            f"_run/current/{TASK_ID}",
            "docs/current/STATUS.json",
            f"tasks/receipts/{TASK_ID.upper()}_RESULT.json",
            "docs/current/visuals/THREE_STREAM_STABLE_BASELINE_V31",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "no_active_current_task",
            "source_and_processed_datasets_READ_ONLY",
            "sealed_and_historical_results_READ_ONLY",
            "three_lane_writer_roots_ISOLATED",
            "single_GPU_lease_owner",
            "egosteer_touch_MUST_NOT_BE_STOPPED",
            "diagnostics_only_gate_direct_consumers",
        ],
        "required_outputs": [
            "attempts/attempt_0001/lanes/exact78/STATE.json",
            "attempts/attempt_0001/lanes/ai1/STATE.json",
            "attempts/attempt_0001/lanes/ai2/STATE.json",
            "attempts/attempt_0001/LANE_LEDGER.json",
            "attempts/attempt_0001/GPU_QUEUE_LEDGER.json",
            "attempts/attempt_0001/FINAL_AUDIT.json",
            "attempts/attempt_0001/RESULT.json",
            "attempts/attempt_0001/RUN_RECEIPT.json",
        ],
        "budgets": {
            "cpu_lane_count_max": 3,
            "gpu_concurrent_owners_max": 1,
            "e1_canary_frames": 12,
            "e1_runtime_attempts_max": 2,
            "e1_quality_retries_max": 0,
            "e2_models": 4,
            "e2_optimizer_updates_per_model": 10000,
            "e2_epochs_per_model_max": 100,
            "e2_gpu_seconds_per_model_max": 43200,
            "e2_checkpoint_interval_updates": 1000,
        },
        "stop_conditions": [
            "each_lane_reaches_its_frozen_success_blocked_rejected_or_budget_paused_state",
            "E1_quality_failure_has_no_automatic_retry",
            "E2_budget_expiry_saves_resumable_state",
            "any_scope_writer_or_GPU_lease_violation_fails_closed",
            "external_or_physical_authority_remains_false_without_independent_evidence",
        ],
        "weights": "ABSENT",
        "child_weights_policy": "EACH_CHILD_BINDS_EXACTLY_ONE_PINNED_WEIGHT_OR_ABSENT",
        "calibration_or_absent": "PER_LANE_PINNED_EVIDENCE_OR_ABSENT",
        "external_metric_authority": False,
        "code_closure": [
            artifact_ref(REPO_ROOT / "src/chaoyang/governance/register_three_stream_stable_baseline_v31.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/governance/build_three_stream_status_v31.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/ops/run_three_stream_stable_baseline_v31.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/ops/build_wiyh_wrist_dual_representation_v1.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/pipeline/hand_observability_v1.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/pipeline/temporal_authority_v1.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/pipeline/hawor_bounded_comparison_v31.py"),
            artifact_ref(REPO_ROOT / "src/chaoyang/pipeline/kai22_r0_tiered_admission_v1.py"),
        ],
        "dag_dependencies": [],
        "expected_resource": "THREE_ISOLATED_CPU_LANES_PLUS_ONE_GOVERNED_GPU_LEASE",
        "executor_epoch": 1,
        "fencing": {
            "pid_startticks_required": True,
            "immutable_final": True,
            "unique_primary_writer": True,
            "lane_writer_roots": {
                "exact78": f"_run/current/{TASK_ID}/attempts/attempt_0001/lanes/exact78",
                "ai1": f"_run/current/{TASK_ID}/attempts/attempt_0001/lanes/ai1",
                "ai2": f"_run/current/{TASK_ID}/attempts/attempt_0001/lanes/ai2",
            },
        },
        "attempt_max": 1,
        "claim_limit": (
            "Development baselines and diagnostics only. Training completion, numeric quality, "
            "visual review, training eligibility, control truth and physical deployment are separate claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output != REGISTRATION_ROOT.resolve():
        raise RuntimeError(f"fixed registration namespace required: {REGISTRATION_ROOT}")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh registration output required: {output}")

    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before V3.1 registration")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))
    if receipt.get("freshness", {}).get("status") != "FRESH":
        raise RuntimeError("V3.1 registration requires a FRESH current receipt")

    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None or any(
        row.get("status") in LIVE for row in state.get("tasks", [])
    ):
        raise RuntimeError("V3.1 registration requires no live task")
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError("V3.1 parent already registered")

    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    pointed = REPO_ROOT / str(pointer["index_path"])
    if pointed.resolve() != CURRENT_INDEX.resolve():
        raise RuntimeError("current task pointer does not resolve to tasks/current/INDEX.json")
    current_ref = artifact_ref(CURRENT_INDEX)
    if (
        pointer.get("index_bytes") != current_ref["bytes"]
        or pointer.get("index_sha256") != current_ref["sha256"]
    ):
        raise RuntimeError("current task pointer SHA/bytes mismatch")
    current = load_json(CURRENT_INDEX)
    if current.get("task_packets") != [] or current.get("status") != "PASS_NO_ACTIVE_TASKS":
        raise RuntimeError("current task index is not empty and terminal")

    packet = build_packet()
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid V3.1 parent packet: " + "; ".join(errors))
    for raw in packet["read_set"]:
        if not (REPO_ROOT / raw).exists():
            raise RuntimeError(f"read_set path missing: {raw}")

    output.mkdir(parents=True)
    frozen_index = output / "PREDECESSOR_TASK_PACKET_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, frozen_index)
    packet_path = REPO_ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    _write_once(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "THREE_STREAM_STABLE_BASELINE_V31_PARENT_ROUTABLE",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "status": "PASS",
        "supersedes_index": artifact_ref(frozen_index),
        "task_packets": [
            {
                "task_id": TASK_ID,
                "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                "packet_sha256": packet_ref["sha256"],
                "execution_class": "CURRENT_LEDGER_ROUTABLE",
                "execution_allowed": True,
                "weights": "ABSENT",
            }
        ],
        "claim_limit": (
            "Exactly one parent is routable. Its three internal CPU lanes have isolated writers; "
            "GPU work remains serialized by the central lease."
        ),
    }

    created = now_iso()
    state["tasks"].append(
        {
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
        }
    )
    state["next_task"] = {
        "task_id": TASK_ID,
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": packet["expected_resource"],
        "stop_condition": "Finite per-lane terminal or resumable budget state without cross-lane authority coupling.",
    }
    state["recent_events"] = (
        state.get("recent_events", [])
        + [
            {
                "task_id": TASK_ID,
                "session": None,
                "attempt": 0,
                "status": "PENDING",
                "created_at": created,
                "message": "Registered the V3.1 parent; no lane algorithm result yet.",
                "task_packet": packet_ref,
            }
        ]
    )[-100:]

    projected_state = output / "PROJECTED_TASK_STATE.json"
    _write_once(projected_state, state)
    atomic_json(
        SHALLOW_STATUS,
        build_status(
            task_state=state,
            parent_task_id=TASK_ID,
            run_root=ATTEMPT_ROOT,
            generated_at=created,
            source_task_state_path=projected_state,
        ),
    )
    result_path = output / "RESULT.json"
    _write_once(
        result_path,
        {
            "schema_version": "chaoyang-three-stream-v31-registration-result-v1",
            "task_id": TASK_ID,
            "status": "PASSED",
            "registered_at": created,
            "task_packet": packet_ref,
            "execution_started": False,
            "claim_limit": "Registration only; no lane algorithm, training or quality result.",
        },
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="THREE_STREAM_STABLE_BASELINE_V31_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=successor,
    )
    _write_once(
        output / "RUN_RECEIPT.json",
        {
            "schema_version": "chaoyang-three-stream-v31-registration-receipt-v1",
            "task_id": TASK_ID,
            "status": "PASSED",
            "result": artifact_ref(result_path),
            "shallow_status": artifact_ref(SHALLOW_STATUS),
            "governance_revision": published["governance_revision"],
            "generation_id": published["generation_id"],
            "execution_started": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASSED",
                "task_id": TASK_ID,
                "governance_revision": published["governance_revision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
