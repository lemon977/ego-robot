#!/usr/bin/env python3
"""CAS-register one immutable task packet in the current packet index."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
from chaoyang.governance.v52_contracts import validate_task_packet


def _validate_packet(packet: dict) -> list[str]:
    """Accept the current R2.2 bounded packet or the stricter V5.2 packet."""
    if packet.get("schema_version") == "exact78-task-packet-v1":
        return validate_task_packet(packet)
    if packet.get("schema_version") != "chaoyang-r22-task-packet-v1":
        return ["unsupported task packet schema"]
    required = {
        "task_id", "objective", "phase", "plan_revision", "read_set",
        "write_set", "prerequisites", "required_outputs", "budgets",
        "stop_conditions", "claim_limit",
    }
    errors: list[str] = []
    missing = sorted(required - set(packet))
    if missing:
        errors.append(f"missing fields: {missing}")
    if len(packet.get("read_set", [])) > 8:
        errors.append("initial read_set exceeds 8 files")
    if not packet.get("write_set"):
        errors.append("write_set must be non-empty")
    if not packet.get("stop_conditions"):
        errors.append("stop_conditions must be non-empty")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--supersede-existing", action="store_true")
    args = parser.parse_args()

    packet_path = args.packet.resolve()
    packet = load_json(packet_path)
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid task packet: " + "; ".join(errors))
    task_id = str(packet["task_id"])

    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before packet registration")
    current_ref = receipt.get("files", {}).get("task_packet_index_current")
    if not isinstance(current_ref, dict):
        raise RuntimeError("current receipt does not bind task packet index")
    ref_errors = validate_artifact_ref(current_ref)
    if ref_errors:
        raise RuntimeError("current task packet index conflict: " + "; ".join(ref_errors))
    predecessor_path = Path(current_ref["path"])
    predecessor = load_json(predecessor_path)
    entries = copy.deepcopy(predecessor.get("task_packets", []))
    existing = next((row for row in entries if str(row.get("task_id")) == task_id), None)
    packet_ref = artifact_ref(packet_path)
    try:
        packet_relative = str(packet_path.relative_to(REPO_ROOT))
    except ValueError as error:
        raise RuntimeError("task packet must be inside repository") from error
    new_entry = {
        "task_id": task_id,
        "packet_path": packet_relative,
        "packet_sha256": packet_ref["sha256"],
        "routing_revision": "R2.2_DIAGNOSTIC",
        "routing_status": "CURRENT_BOUNDED_DIAGNOSTIC",
        "authority_promoted": False,
    }
    if existing is not None:
        if existing.get("packet_sha256") == new_entry["packet_sha256"]:
            pass
        elif not args.supersede_existing:
            raise RuntimeError(f"task packet conflict for {task_id}")
        else:
            new_entry["supersedes_packet"] = copy.deepcopy(existing)
            entries[entries.index(existing)] = new_entry
    else:
        entries.append(new_entry)

    successor = {
        **{key: value for key, value in predecessor.items() if key not in {"task_packets", "governance_revision", "generation_id", "generated_at"}},
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "R2.2_DIAGNOSTIC",
        "plan_revision": "R2.2",
        "execution_revision": "R2.2",
        "status": "PASS",
        "supersedes_index": artifact_ref(predecessor_path),
        "task_packets": entries,
        "claim_limit": "Execution routing only; registering a bounded diagnostic does not promote authority.",
    }

    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state.get("tasks", []) if row.get("task_id") == task_id), None)
    if task is None:
        raise RuntimeError(f"task state missing for {task_id}")
    task["task_packet"] = packet_ref
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": task_id,
        "session": task.get("session"),
        "attempt": task.get("attempt", 0),
        "status": task.get("status", "PENDING"),
        "created_at": now_iso(),
        "message": "Registered bounded immutable task packet; no authority promotion.",
    }])[-100:]
    published = publish_bundle(
        authority,
        state,
        event_type="TASK_PACKET_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=args.output_index.resolve(),
        task_packet_index_value=successor,
    )
    print(json.dumps({"status": "PASSED", "task_id": task_id, "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
