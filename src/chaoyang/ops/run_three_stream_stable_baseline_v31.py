#!/usr/bin/env python3
"""Initialize and coordinate the isolated V3.1 three-stream parent attempt.

This coordinator owns shared state only.  Lane algorithms may write solely to
their assigned lane root, and GPU work must be dispatched through the central
lease after readiness is recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chaoyang.governance.build_three_stream_status_v31 import build_status
from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    sha256_file,
)


TASK_ID = "three_stream_stable_baseline_v31"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TASK_PACKET = REPO_ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
ATTEMPT_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
SHALLOW_STATUS = REPO_ROOT / "docs/current/STATUS.json"
LANES = ("exact78", "ai1", "ai2")


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _validate_route() -> dict[str, Any]:
    index = load_json(CURRENT_INDEX)
    entries = index.get("task_packets", [])
    if len(entries) != 1:
        raise RuntimeError("V3.1 requires exactly one current routed parent")
    entry = entries[0]
    if entry.get("task_id") != TASK_ID or entry.get("execution_allowed") is not True:
        raise RuntimeError("V3.1 parent is not execution_allowed")
    if not TASK_PACKET.is_file() or sha256_file(TASK_PACKET) != entry.get("packet_sha256"):
        raise RuntimeError("V3.1 task packet SHA mismatch")
    state = load_json(TASK_STATE_PATH)
    next_task = state.get("next_task")
    if not isinstance(next_task, dict) or next_task.get("task_id") != TASK_ID:
        raise RuntimeError("V3.1 parent is not the current ledger successor")
    return state


def _initial_lane_state(lane: str, created: str) -> dict[str, Any]:
    return {
        "schema_version": "chaoyang-three-stream-lane-state-v1",
        "parent_task_id": TASK_ID,
        "lane": lane,
        "status": "READY_CPU_PREFLIGHT",
        "current_action": "CPU_PREFLIGHT_NOT_STARTED",
        "stable_baseline": None,
        "latest_artifacts": [],
        "blocker": None,
        "next_step": "RUN_FROZEN_CPU_PREFLIGHT",
        "authority": "DEVELOPMENT_ONLY_NON_CONTROL_NON_DEPLOYABLE",
        "writer_root": str((ATTEMPT_ROOT / "lanes" / lane).resolve()),
        "updated_at": created,
        "claims": {
            "PIPELINE_COMPLETE": False,
            "NUMERIC_QUALITY_PASS": False,
            "VISUAL_REVIEW_STATUS": "NOT_REVIEWED",
            "TRAINING_COMPLETE": False,
            "TRAINING_ELIGIBLE": False,
            "CONTROL_GROUND_TRUTH": False,
            "PHYSICAL_DEPLOYABLE": False,
        },
    }


def initialize(expected_revision: int) -> dict[str, Any]:
    state = _validate_route()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != expected_revision:
        raise RuntimeError("CAS revision mismatch before V3.1 initialization")
    if ATTEMPT_ROOT.exists() or ATTEMPT_ROOT.is_symlink():
        raise RuntimeError(f"fresh V3.1 attempt required: {ATTEMPT_ROOT}")

    created = now_iso()
    ATTEMPT_ROOT.mkdir(parents=True)
    signature = {
        "schema_version": "chaoyang-three-stream-run-signature-v1",
        "task_id": TASK_ID,
        "attempt": 1,
        "created_at": created,
        "task_packet": artifact_ref(TASK_PACKET),
        "plan": artifact_ref(REPO_ROOT / "docs/current/PLAN.md"),
        "lane_documents": {
            lane: artifact_ref(REPO_ROOT / f"docs/current/{lane.upper() if lane == 'exact78' else lane.upper()}.md")
            for lane in LANES
        },
        "weights": "ABSENT_AT_PARENT_EACH_CHILD_MUST_PIN_ONE_OR_ABSENT",
        "source_write_authorized": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    _write_once(ATTEMPT_ROOT / "RUN_SIGNATURE.json", signature)

    lane_refs: dict[str, Any] = {}
    for lane in LANES:
        lane_root = ATTEMPT_ROOT / "lanes" / lane
        lane_root.mkdir(parents=True)
        state_path = lane_root / "STATE.json"
        _write_once(state_path, _initial_lane_state(lane, created))
        lane_refs[lane] = artifact_ref(state_path)
    _write_once(
        ATTEMPT_ROOT / "LANE_LEDGER.json",
        {
            "schema_version": "chaoyang-three-stream-lane-ledger-v1",
            "task_id": TASK_ID,
            "created_at": created,
            "cpu_lane_count_max": 3,
            "single_publisher": True,
            "lanes": lane_refs,
            "cross_lane_writes_allowed": False,
        },
    )
    _write_once(
        ATTEMPT_ROOT / "GPU_QUEUE_LEDGER.json",
        {
            "schema_version": "chaoyang-three-stream-gpu-queue-v1",
            "task_id": TASK_ID,
            "created_at": created,
            "concurrent_owners_max": 1,
            "current_owner": None,
            "queue_policy": [
                "EXACT78_E2_READY_HAS_MAIN_PRIORITY",
                "EXACT78_E1_GETS_ONE_12_FRAME_BOUNDED_CANARY_SLOT",
                "AI1_AI2_FROZEN_VALIDATION_ONLY_AT_E2_1000_UPDATE_CHECKPOINTS",
            ],
            "lease_required": True,
        },
    )

    task = next(row for row in state["tasks"] if row.get("task_id") == TASK_ID)
    task.update(
        attempt=1,
        status="PENDING",
        updated_at=created,
        heartbeat_at=None,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        attempt_root=str(ATTEMPT_ROOT),
    )
    state["recent_events"] = (
        state.get("recent_events", [])
        + [
            {
                "task_id": TASK_ID,
                "session": None,
                "attempt": 1,
                "status": "PENDING",
                "created_at": created,
                "message": "Initialized three isolated CPU lane roots; no algorithm or GPU execution yet.",
                "run_signature": artifact_ref(ATTEMPT_ROOT / "RUN_SIGNATURE.json"),
            }
        ]
    )[-100:]
    projected = ATTEMPT_ROOT / "TASK_STATE_INITIALIZED.json"
    _write_once(projected, state)
    atomic_json(
        SHALLOW_STATUS,
        build_status(
            task_state=state,
            parent_task_id=TASK_ID,
            run_root=ATTEMPT_ROOT,
            generated_at=created,
            source_task_state_path=projected,
        ),
    )
    result = {
        "schema_version": "chaoyang-three-stream-initialization-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "created_at": created,
        "lane_ledger": artifact_ref(ATTEMPT_ROOT / "LANE_LEDGER.json"),
        "gpu_queue": artifact_ref(ATTEMPT_ROOT / "GPU_QUEUE_LEDGER.json"),
        "algorithm_execution_started": False,
        "claim_limit": "Writer and resource initialization only; no algorithm quality or training result.",
    }
    _write_once(ATTEMPT_ROOT / "INITIALIZATION_RESULT.json", result)
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="THREE_STREAM_STABLE_BASELINE_V31_INITIALIZED",
        expected_revision=expected_revision,
        generator_path=Path(__file__),
    )
    return {
        "status": "PASSED",
        "task_id": TASK_ID,
        "attempt_root": str(ATTEMPT_ROOT),
        "governance_revision": published["governance_revision"],
        "algorithm_execution_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("initialize",), required=True)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if args.mode != "initialize":
        raise RuntimeError(f"unsupported mode: {args.mode}")
    print(json.dumps(initialize(args.expected_revision), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
