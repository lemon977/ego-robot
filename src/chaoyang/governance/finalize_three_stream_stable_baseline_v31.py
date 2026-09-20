#!/usr/bin/env python3
"""CAS-finalize the finite V3.1 parent after all three CPU lanes terminate."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.common import (
    REPO_ROOT,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)


TASK_ID = "three_stream_stable_baseline_v31"
PLAN_REVISION = "THREE_STREAM_STABLE_BASELINE_V3_1"
EXECUTION_REVISION = "THREE_STREAM_STABLE_BASELINE_V3_1"
LANES = ("exact78", "ai1", "ai2")
LANE_TERMINALS = {
    "CPU_PREFLIGHT_COMPLETE",
    "BLOCKED_CPU_PREFLIGHT",
    "REJECTED_CPU_PREFLIGHT",
}
PARENT_LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
OTHER_NONTERMINAL = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
READY_BOOLEAN_KEYS = {
    "gpu_ready",
    "training_ready",
    "ready_for_gpu",
    "ready_for_training",
    "e2_ready",
    "launch_authorized",
    "training_eligible",
}
READY_STRING_STATUSES = {
    "READY_GPU",
    "READY_FOR_GPU",
    "READY_TRAINING",
    "TRAINING_READY",
    "WAIT_GPU_RESOURCE",
    "TRAINING_PAUSED_BUDGET",
}


class FinalizeError(RuntimeError):
    """Raised when finite closure cannot be proven from current ledgers."""


def _paths(root: Path) -> dict[str, Path]:
    attempt = root / f"_run/current/{TASK_ID}/attempts/attempt_0001"
    return {
        "authority": root / "docs/governance/CURRENT_AUTHORITY_INDEX.json",
        "receipt": root / "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "task_state": root / "docs/governance/LONG_HORIZON_TASK_STATE.json",
        "index": root / "tasks/current/INDEX.json",
        "task_packet": root / f"tasks/current/{TASK_ID}/TASK_PACKET.json",
        "attempt": attempt,
        "gpu_queue": attempt / "GPU_QUEUE_LEDGER.json",
        "shallow_status": root / "docs/current/STATUS.json",
        "terminal_receipt": root
        / "tasks/receipts/THREE_STREAM_STABLE_BASELINE_V31_RESULT.json",
    }


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise FinalizeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, dict(value))


def _single_artifact_payload(state: Mapping[str, Any], lane: str) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = state.get("latest_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise FinalizeError(f"{lane} must bind exactly one latest machine result")
    reference = artifacts[0]
    if not isinstance(reference, Mapping):
        raise FinalizeError(f"{lane} latest artifact reference is not an object")
    errors = validate_artifact_ref(reference)
    if errors:
        raise FinalizeError(f"{lane} latest artifact invalid: {'; '.join(errors)}")
    result = load_json(Path(str(reference["path"])))
    if result.get("status") != state.get("machine_result_status"):
        raise FinalizeError(f"{lane} machine result status differs from lane STATE")
    return dict(reference), result


def _ready_signals(value: Any, path: str = "$") -> list[str]:
    signals: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = str(key).casefold()
            if normalized in READY_BOOLEAN_KEYS and child is True:
                signals.append(f"{child_path}=true")
            if normalized == "status" and isinstance(child, str):
                if child.upper() in READY_STRING_STATUSES:
                    signals.append(f"{child_path}={child}")
            signals.extend(_ready_signals(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            signals.extend(_ready_signals(child, f"{path}[{index}]"))
    return signals


def _audit_lanes(attempt: Path) -> tuple[dict[str, Any], list[str]]:
    rows: dict[str, Any] = {}
    ready: list[str] = []
    for lane in LANES:
        state_path = attempt / "lanes" / lane / "STATE.json"
        state = load_json(state_path)
        if state.get("parent_task_id") != TASK_ID or state.get("lane") != lane:
            raise FinalizeError(f"{lane} lane identity mismatch")
        status = state.get("status")
        if status not in LANE_TERMINALS:
            raise FinalizeError(f"{lane} lane is not terminal CPU preflight: {status}")
        result_ref, machine_result = _single_artifact_payload(state, lane)
        signals = _ready_signals(machine_result)
        if status == "CPU_PREFLIGHT_COMPLETE" and not signals:
            explicit_false = any(
                machine_result.get(key) is False for key in READY_BOOLEAN_KEYS
            )
            if not explicit_false:
                signals.append("CPU_PREFLIGHT_COMPLETE_WITHOUT_EXPLICIT_NO_READY_PROOF")
        ready.extend(f"{lane}:{signal}" for signal in signals)
        blocker = state.get("blocker")
        rows[lane] = {
            "status": status,
            "machine_result_status": state.get("machine_result_status"),
            "blocker": blocker,
            "state": artifact_ref(state_path),
            "machine_result": result_ref,
            "ready_signals": signals,
            "claims": state.get("claims", {}),
        }
    return rows, ready


def _audit_gpu_queue(path: Path) -> tuple[dict[str, Any], list[str]]:
    queue = load_json(path)
    if queue.get("task_id") != TASK_ID:
        raise FinalizeError("GPU queue task identity mismatch")
    signals = _ready_signals(queue)
    if queue.get("current_owner") is not None:
        signals.append("$.current_owner=occupied")
    for key in ("queue", "ready_jobs", "pending_jobs", "jobs"):
        value = queue.get(key)
        if isinstance(value, list) and value:
            signals.append(f"$.{key}=nonempty")
    return queue, sorted(set(signals))


def _derive_parent_status(lanes: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = {str(row["status"]) for row in lanes.values()}
    if "REJECTED_CPU_PREFLIGHT" in statuses:
        return "REJECTED_QUALITY"
    if "BLOCKED_CPU_PREFLIGHT" in statuses:
        return "BLOCKED_PREREQ"
    return "PASSED"


def _terminal_index(predecessor: Path) -> dict[str, Any]:
    return {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "THREE_STREAM_STABLE_BASELINE_V31_TERMINAL",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(predecessor),
        "task_packets": [],
        "claim_limit": (
            "The finite V3.1 parent is terminal. Lane blockers remain separate and no GPU, "
            "training, control or deployment successor is authorized."
        ),
    }


def finalize(
    *,
    expected_revision: int,
    root: Path = REPO_ROOT,
    publisher: Callable[..., dict[str, Any]] = publish_bundle,
    status_builder: Callable[..., dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Publish one bounded terminal after all current-only closure gates pass."""

    root = root.resolve(strict=True)
    paths = _paths(root)
    receipt = load_json(paths["receipt"])
    if int(receipt.get("governance_revision", -1)) != expected_revision:
        raise FinalizeError("CAS revision mismatch before V3.1 finalization")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise FinalizeError("current receipt conflict: " + "; ".join(errors))

    state = load_json(paths["task_state"])
    parent_rows = [row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID]
    if len(parent_rows) != 1 or parent_rows[0].get("status") not in PARENT_LIVE:
        raise FinalizeError("V3.1 parent is not the unique live task")
    next_task = state.get("next_task")
    if not isinstance(next_task, Mapping) or next_task.get("task_id") != TASK_ID:
        raise FinalizeError("V3.1 parent is not current next_task")
    other_live = [
        str(row.get("task_id"))
        for row in state.get("tasks", [])
        if row.get("task_id") != TASK_ID and row.get("status") in OTHER_NONTERMINAL
    ]
    if other_live:
        raise FinalizeError(f"other live tasks prevent closure: {other_live}")

    current = load_json(paths["index"])
    routes = current.get("task_packets", [])
    if len(routes) != 1 or routes[0].get("task_id") != TASK_ID:
        raise FinalizeError("V3.1 parent is not the sole current route")
    route = routes[0]
    if route.get("execution_allowed") is not True:
        raise FinalizeError("V3.1 current route is not execution_allowed")
    packet_path = root / str(route.get("packet_path"))
    if packet_path.resolve() != paths["task_packet"].resolve():
        raise FinalizeError("V3.1 routed packet path mismatch")
    packet_ref = artifact_ref(packet_path)
    if route.get("packet_sha256") != packet_ref["sha256"]:
        raise FinalizeError("V3.1 task packet SHA mismatch")

    attempt = paths["attempt"].resolve(strict=True)
    final_paths = [
        attempt / "FINAL_AUDIT.json",
        attempt / "RESULT.json",
        attempt / "RUN_RECEIPT.json",
        paths["terminal_receipt"],
    ]
    if any(path.exists() or path.is_symlink() for path in final_paths):
        raise FinalizeError("fresh V3.1 final artifacts required")

    lanes, lane_ready = _audit_lanes(attempt)
    gpu_queue, queue_ready = _audit_gpu_queue(paths["gpu_queue"])
    ready_signals = sorted(set(lane_ready + queue_ready))
    if ready_signals:
        raise FinalizeError("GPU/training-ready work remains: " + "; ".join(ready_signals))

    created = created_at or now_iso()
    predecessor = attempt / "PREDECESSOR_ACTIVE_INDEX.json"
    if predecessor.exists() or predecessor.is_symlink():
        raise FinalizeError(f"fresh predecessor snapshot required: {predecessor}")
    shutil.copyfile(paths["index"], predecessor)
    parent_status = _derive_parent_status(lanes)
    blockers = {
        lane: row["blocker"]
        for lane, row in lanes.items()
        if row.get("blocker") is not None
    }
    final_audit = {
        "schema_version": "chaoyang-three-stream-v31-final-audit-v1",
        "task_id": TASK_ID,
        "status": parent_status,
        "created_at": created,
        "lanes": lanes,
        "lane_blockers": blockers,
        "gpu_queue": artifact_ref(paths["gpu_queue"]),
        "gpu_queue_current_owner": gpu_queue.get("current_owner"),
        "gpu_or_training_ready_signals": [],
        "finite_blocked_closure_allowed": parent_status == "BLOCKED_PREREQ",
        "all_lanes_terminal_cpu_preflight": True,
        "pipeline_complete": False,
        "training_complete": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "Finite CPU-preflight closure only. Each lane blocker remains local; this audit "
            "does not claim pipeline, training, numeric, control or deployment completion."
        ),
    }
    final_audit_path = attempt / "FINAL_AUDIT.json"
    _write_once(final_audit_path, final_audit)
    result = {
        "schema_version": "chaoyang-three-stream-v31-result-v1",
        "task_id": TASK_ID,
        "status": parent_status,
        "created_at": created,
        "weights": "ABSENT",
        "final_audit": artifact_ref(final_audit_path),
        "lane_statuses": {lane: row["status"] for lane, row in lanes.items()},
        "lane_blockers": blockers,
        "pipeline_complete": False,
        "numeric_quality_pass": False,
        "visual_review_status": "NOT_REVIEWED",
        "training_complete": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": "Finite parent terminal; blocked lanes are not converted into success.",
    }
    result_path = attempt / "RESULT.json"
    _write_once(result_path, result)
    terminal_receipt = {
        "schema_version": "chaoyang-three-stream-v31-terminal-result-receipt-v1",
        "task_id": TASK_ID,
        "status": parent_status,
        "created_at": created,
        "result": artifact_ref(result_path),
        "final_audit": artifact_ref(final_audit_path),
        "lane_results": {
            lane: row["machine_result"] for lane, row in lanes.items()
        },
        "pipeline_complete": False,
        "training_complete": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    _write_once(paths["terminal_receipt"], terminal_receipt)

    parent = parent_rows[0]
    parent.update(
        status=parent_status,
        updated_at=created,
        heartbeat_at=None,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        result=artifact_ref(result_path),
    )
    state["next_task"] = None
    state["recent_events"] = (
        state.get("recent_events", [])
        + [
            {
                "task_id": TASK_ID,
                "session": None,
                "attempt": parent.get("attempt", 1),
                "status": parent_status,
                "created_at": created,
                "message": "Finite V3.1 CPU lanes closed; lane blockers remain separate.",
                "result": artifact_ref(result_path),
            }
        ]
    )[-100:]
    projected = attempt / "PROJECTED_TERMINAL_TASK_STATE.json"
    _write_once(projected, state)

    if status_builder is None:
        from chaoyang.governance.build_three_stream_status_v31 import build_status

        status_builder = build_status
    shallow = status_builder(
        task_state=state,
        parent_task_id=TASK_ID,
        run_root=attempt,
        generated_at=created,
        source_task_state_path=projected,
    )
    atomic_json(paths["shallow_status"], shallow)
    terminal_index = _terminal_index(predecessor)
    published = publisher(
        load_json(paths["authority"]),
        state,
        event_type=f"THREE_STREAM_STABLE_BASELINE_V31_{parent_status}",
        expected_revision=expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=paths["index"],
        task_packet_index_value=terminal_index,
    )
    run_receipt = {
        "schema_version": "chaoyang-three-stream-v31-run-receipt-v1",
        "task_id": TASK_ID,
        "status": parent_status,
        "finalization_status": "PASSED_CAS_PUBLISHED",
        "created_at": created,
        "result": artifact_ref(result_path),
        "final_audit": artifact_ref(final_audit_path),
        "terminal_receipt": artifact_ref(paths["terminal_receipt"]),
        "shallow_status": artifact_ref(paths["shallow_status"]),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "current_index_status": "PASS_NO_ACTIVE_TASKS",
        "pipeline_complete": False,
    }
    _write_once(attempt / "RUN_RECEIPT.json", run_receipt)
    return run_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    result = finalize(expected_revision=args.expected_revision)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
