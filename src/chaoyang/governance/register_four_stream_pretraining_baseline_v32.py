#!/usr/bin/env python3
"""CAS-register the single routable parent for the V3.2 four-stream run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.build_four_stream_status_v32 import build_status
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


TASK_ID = "four_stream_pretraining_baseline_v32"
PLAN_REVISION = "FOUR_STREAM_STABLE_BASELINE_V3_2"
EXECUTION_REVISION = "FOUR_STREAM_PRETRAINING_BASELINE_V3_2"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
REGISTRATION_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/registration_0001"
STATUS_PATH = REPO_ROOT / "docs/current/STATUS.json"
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
    code_paths = (
        "src/chaoyang/governance/register_four_stream_pretraining_baseline_v32.py",
        "src/chaoyang/governance/build_four_stream_status_v32.py",
        "src/chaoyang/ops/run_four_stream_pretraining_baseline_v32.py",
        "src/chaoyang/ops/record_four_stream_lane_result_v32.py",
        "src/chaoyang/ops/run_huro_derived_hand_only_v1.py",
        "src/chaoyang/ops/build_exact78_same_session_hand_initializer_v32.py",
        "src/chaoyang/ops/render_exact78_hand_only_review_v32.py",
        "src/chaoyang/ops/run_wiyh_ai1_static_wrist_candidate_v32.py",
        "src/chaoyang/pipeline/huro_hand_only_retarget_v1.py",
        "src/chaoyang/research/world_in_your_hands/ai1_static_wrist_candidate_v32.py",
        "src/chaoyang/human_ego/exact78_v32.py",
        "src/chaoyang/human_ego/tools/build_exact78_pair_ledgers_v32.py",
        "src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py",
        "src/chaoyang/pipeline/ai2_independent_observability_v32.py",
        "src/chaoyang/pipeline/kai22_full_fk_sidecar_v1.py",
        "src/chaoyang/pipeline/kai22_full_fk_review_renderer_v1.py",
    )
    return {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK_ID,
        "objective": (
            "Produce the V3.2 Exact78, AI1, AI2 and HuRo-derived first milestones with isolated "
            "writers, one publisher, one GPU owner and separately bounded development claims."
        ),
        "phase": "FOUR_STREAM_PRETRAINING_BASELINE_V32",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/governance/ALGORITHM_CONTRACT.json",
            "docs/current/PLAN.md",
            "docs/current/EXACT78.md",
            "docs/current/AI1.md",
            "docs/current/AI2.md",
            "docs/current/AI4_HURO.md",
            "tasks/current/INDEX.json",
        ],
        "write_set": [
            f"_run/current/{TASK_ID}",
            "docs/current/STATUS.json",
            f"tasks/receipts/{TASK_ID.upper()}_RESULT.json",
            "docs/current/visuals/FOUR_STREAM_PRETRAINING_BASELINE_V32",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "no_active_current_task",
            "source_processed_archive_sealed_READ_ONLY",
            "four_lane_writer_roots_ISOLATED",
            "single_publisher",
            "single_GPU_lease_owner",
            "noncommercial_research_evaluation",
            "diagnostics_only_gate_direct_consumers",
        ],
        "required_outputs": [
            *[f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in ("exact78", "ai1", "ai2", "ai4_huro")],
            "attempts/attempt_0001/LANE_LEDGER.json",
            "attempts/attempt_0001/GPU_QUEUE_LEDGER.json",
            "attempts/attempt_0001/RUN_SIGNATURE.json",
        ],
        "budgets": {
            "cpu_threads_soft_cap": 8,
            "gpu_concurrent_owners_max": 1,
            "first_cycle_gpu_seconds_max": 72900,
            "first_cycle_exact78_gpu_seconds_max": 43200,
            "first_cycle_observation_gpu_seconds_max": 14400,
            "first_cycle_huro_gpu_seconds_max": 14400,
            "first_cycle_stereo_gpu_seconds_max": 900,
            "per_model_lifetime_gpu_seconds_max": 43200,
            "progress_receipts_hours": [2, 4, 8, 24],
        },
        "stop_conditions": [
            "each_lane_reaches_frozen_milestone_terminal_or_resumable_budget_state",
            "quality_failure_has_no_same_signature_automatic_retry",
            "budget_expiry_saves_resumable_state_before_successor_budget",
            "writer_or_GPU_lease_violation_fails_closed",
            "external_physical_control_authority_remains_false",
        ],
        "weights": "ABSENT",
        "child_weights_policy": "EACH_CHILD_BINDS_EXACTLY_ONE_PINNED_WEIGHT_OR_ABSENT",
        "calibration_or_absent": "PER_LANE_PINNED_EVIDENCE_OR_ABSENT",
        "external_metric_authority": False,
        "code_closure": [artifact_ref(REPO_ROOT / path) for path in code_paths],
        "dag_dependencies": [],
        "expected_resource": "FOUR_ISOLATED_CPU_LANES_PLUS_ONE_GOVERNED_GPU_LEASE",
        "executor_epoch": 1,
        "fencing": {
            "pid_startticks_required": True,
            "immutable_final": True,
            "unique_primary_writer": True,
            "lane_writer_roots": {
                lane: f"_run/current/{TASK_ID}/attempts/attempt_0001/lanes/{lane}"
                for lane in ("exact78", "ai1", "ai2", "ai4_huro")
            },
        },
        "attempt_max": 1,
        "claim_limit": (
            "Development baselines only. Pipeline, numeric quality, visual review, training, "
            "training eligibility, control truth and deployment remain separate claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output != REGISTRATION_ROOT.resolve():
        raise RuntimeError(f"fixed registration namespace required: {REGISTRATION_ROOT}")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh registration output required: {output}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before V3.2 registration")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))
    if receipt.get("freshness", {}).get("status") != "FRESH":
        raise RuntimeError("V3.2 registration requires FRESH governance")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None or any(row.get("status") in LIVE for row in state.get("tasks", [])):
        raise RuntimeError("V3.2 registration requires no live current task")
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError("V3.2 parent already registered")
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    if (REPO_ROOT / str(pointer["index_path"])).resolve() != CURRENT_INDEX.resolve():
        raise RuntimeError("current task pointer does not resolve to current index")
    current_ref = artifact_ref(CURRENT_INDEX)
    if pointer.get("index_bytes") != current_ref["bytes"] or pointer.get("index_sha256") != current_ref["sha256"]:
        raise RuntimeError("current task pointer SHA/bytes mismatch")
    current = load_json(CURRENT_INDEX)
    if current.get("task_packets") != [] or current.get("status") != "PASS_NO_ACTIVE_TASKS":
        raise RuntimeError("current task index is not empty and terminal")
    packet = build_packet()
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid V3.2 parent packet: " + "; ".join(errors))
    output.mkdir(parents=True)
    frozen_index = output / "PREDECESSOR_TASK_PACKET_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, frozen_index)
    packet_path = REPO_ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    _write_once(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "FOUR_STREAM_PRETRAINING_BASELINE_V32_PARENT_ROUTABLE",
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
        }],
        "claim_limit": "Exactly one parent is routable; four internal writers are isolated and GPU work is serialized.",
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
    })
    state["next_task"] = {
        "task_id": TASK_ID,
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": packet["expected_resource"],
        "stop_condition": "Finite per-lane terminal or resumable budget state without cross-lane coupling.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK_ID,
        "session": None,
        "attempt": 0,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered the V3.2 four-stream parent; no algorithm result yet.",
        "task_packet": packet_ref,
    }])[-100:]
    projected = output / "PROJECTED_TASK_STATE.json"
    _write_once(projected, state)
    atomic_json(
        STATUS_PATH,
        build_status(task_state=state, generated_at=created, source_task_state_path=projected),
    )
    result_path = output / "RESULT.json"
    _write_once(result_path, {
        "schema_version": "chaoyang-four-stream-registration-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "registered_at": created,
        "task_packet": packet_ref,
        "execution_started": False,
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="FOUR_STREAM_PRETRAINING_BASELINE_V32_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=successor,
    )
    _write_once(output / "RUN_RECEIPT.json", {
        "schema_version": "chaoyang-four-stream-registration-receipt-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "result": artifact_ref(result_path),
        "shallow_status": artifact_ref(STATUS_PATH),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
    })
    print(json.dumps({"status": "PASSED", "task_id": TASK_ID, "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
