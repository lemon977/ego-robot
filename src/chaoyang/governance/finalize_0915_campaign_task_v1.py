#!/usr/bin/env python3
"""CAS-finalize one registered 0915 campaign task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from chaoyang.governance.campaign_0915_task_specs_v1 import PLAN_REVISION, TASK_ORDER
from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)


CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TERMINAL = {
    "PASSED",
    "REJECTED_QUALITY",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "CANCELLED",
}
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def resolve_active_index_for_finalization(
    current: dict,
    task_id: str,
    *,
    expected_revision: int,
) -> tuple[dict, Path]:
    """Return the active index, including recovery from a pre-receipt partial publish.

    ``publish_bundle`` historically materialized the next packet index before it
    validated the task state.  A validation error could therefore leave the
    terminal index on disk while the authoritative receipt and task state still
    described the live predecessor.  Recovery is permitted only for that exact
    one-revision-ahead shape and only through its SHA-bound predecessor.
    """
    route = next(
        (row for row in current.get("task_packets", []) if row.get("task_id") == task_id),
        None,
    )
    if route is not None:
        return current, CURRENT_INDEX

    expected_packet_revision = f"{task_id.upper()}_TERMINAL"
    if (
        current.get("status") != "PASS_NO_ACTIVE_TASKS"
        or current.get("packet_revision") != expected_packet_revision
        or current.get("governance_revision") != expected_revision + 1
        or current.get("task_packets") != []
    ):
        raise RuntimeError("task is not current routable packet")
    predecessor_ref = current.get("supersedes_index")
    if not isinstance(predecessor_ref, dict):
        raise RuntimeError("partial terminal index has no predecessor reference")
    ref_errors = validate_artifact_ref(predecessor_ref)
    if ref_errors:
        raise RuntimeError("partial terminal predecessor invalid: " + "; ".join(ref_errors))
    predecessor_path = Path(str(predecessor_ref["path"]))
    predecessor = load_json(predecessor_path)
    predecessor_route = next(
        (
            row
            for row in predecessor.get("task_packets", [])
            if row.get("task_id") == task_id
        ),
        None,
    )
    if predecessor_route is None or predecessor_route.get("execution_allowed") is not True:
        raise RuntimeError("partial terminal predecessor is not the active task index")
    return predecessor, predecessor_path


def validate_terminal_bundle(
    result_path: Path,
    result: dict,
    packet: dict,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Validate a successful terminal before it can unlock a successor."""
    errors: list[str] = []
    if packet.get("task_id") != result.get("task_id"):
        errors.append("task packet/result identity mismatch")
    write_set = packet.get("write_set", [])
    if not write_set:
        errors.append("task packet has no write_set")
    else:
        expected = Path(str(write_set[0]))
        expected_root = expected if expected.is_absolute() else repo_root / expected
        if result_path.parent.resolve() != expected_root.resolve():
            errors.append("result is outside the primary packet write root")
    if result.get("status") == "PASSED":
        if result.get("weights") != packet.get("weights"):
            errors.append("successful result weight identity differs from task packet")
        for raw in packet.get("required_outputs", []):
            candidate = result_path.parent / str(raw)
            if not candidate.is_file():
                errors.append(f"required output is missing: {raw}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=TASK_ORDER)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh finalization output required: {output}")
    result_path = args.result.resolve(strict=True)
    result = load_json(result_path)
    if result.get("task_id") != args.task_id:
        raise RuntimeError("result task identity mismatch")
    status = str(result.get("status"))
    if status not in TERMINAL:
        raise RuntimeError(f"unsupported terminal status: {status}")
    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before finalization")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state.get("tasks", []) if row.get("task_id") == args.task_id), None)
    if task is None or task.get("status") not in LIVE:
        raise RuntimeError("task is not live")
    if state.get("next_task", {}).get("task_id") != args.task_id:
        raise RuntimeError("task is not current next_task")
    current = load_json(CURRENT_INDEX)
    active_index, active_index_path = resolve_active_index_for_finalization(
        current,
        args.task_id,
        expected_revision=args.expected_revision,
    )
    route = next((row for row in current.get("task_packets", []) if row.get("task_id") == args.task_id), None)
    if route is None:
        route = next(
            row
            for row in active_index.get("task_packets", [])
            if row.get("task_id") == args.task_id
        )
    if route.get("execution_allowed") is not True:
        raise RuntimeError("task is not current routable packet")
    packet_path = REPO_ROOT / str(route.get("packet_path"))
    packet = load_json(packet_path)
    bundle_errors = validate_terminal_bundle(result_path, result, packet)
    if bundle_errors:
        raise RuntimeError("terminal bundle invalid: " + "; ".join(bundle_errors))

    output.mkdir(parents=True)
    predecessor = output / "PREDECESSOR_ACTIVE_INDEX.json"
    shutil.copyfile(active_index_path, predecessor)
    terminal_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{args.task_id.upper()}_TERMINAL",
        "plan_revision": PLAN_REVISION,
        "execution_revision": "0915_FULL_FUNNEL_V1",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(predecessor),
        "task_packets": [],
        "claim_limit": (
            "This 0915 task is terminal. No successor is executable until a separate "
            "CAS registration publishes exactly one new packet."
        ),
    }
    created = now_iso()
    task.update({
        "status": status,
        "updated_at": created,
        "heartbeat_at": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "result": artifact_ref(result_path),
    })
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": args.task_id,
        "session": None,
        "attempt": task.get("attempt", 1),
        "status": status,
        "created_at": created,
        "message": "Finite 0915 campaign task reached a recorded terminal.",
        "result": artifact_ref(result_path),
    }])[-100:]
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type=f"{args.task_id.upper()}_{status}",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=terminal_index,
    )
    final = {
        "schema_version": "finalize-0915-campaign-task-v1",
        "task_id": args.task_id,
        "status": status,
        "result": artifact_ref(result_path),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "next_task": None,
    }
    (output / "FINALIZATION.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
