#!/usr/bin/env python3
"""CAS-register the immutable Robot v7.6 successor without launching it."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chaoyang.governance.common import (  # noqa: E402
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    load_json,
    now_iso,
    publish_bundle,
    sha256_file,
    validate_artifact_ref,
)

TASK_ID = "robot_geometry_expansion_v76"
PACKET = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v76/attempts/attempt_0005_execution/TASK_PACKET.json"
ROW_CANDIDATE = PACKET.parent / "TASK_STATE_ROW_CANDIDATE.json"
V75_FAILURE = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v75/attempts/attempt_0016_execution/FAILED_RUNTIME_FINAL.json"
POINTER = REPO_ROOT / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json"


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


def _validate_ref(reference: dict[str, Any]) -> Path:
    errors = validate_artifact_ref(reference)
    if errors:
        raise RuntimeError("artifact reference mismatch:\n" + "\n".join(errors))
    return Path(reference["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable output root required: {output}")

    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError(
            f"CAS revision mismatch: expected={args.expected_revision} current={receipt.get('governance_revision')}"
        )
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict:\n" + "\n".join(errors))

    state = load_json(TASK_STATE_PATH)
    v75 = next(
        (row for row in state.get("tasks", []) if row.get("task_id") == "robot_geometry_expansion_v75"),
        None,
    )
    if v75 is None or v75.get("status") != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("v75 must be present as FAILED_RUNTIME_FINAL")
    if v75.get("result") != artifact_ref(V75_FAILURE):
        raise RuntimeError("v75 task-state terminal does not bind the exact failure receipt")
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError("v76 already exists in task state")

    task_candidate = load_json(ROW_CANDIDATE)
    task = dict(task_candidate.get("task", {}))
    if task.get("task_id") != TASK_ID or _validate_ref(task["task_packet"]) != PACKET:
        raise RuntimeError("v76 task-state candidate does not bind the exact Task Packet")
    task["plan_execution_revision"] = "R3"
    state["tasks"].append(task)

    pointer = load_json(POINTER)
    predecessor_path = Path(pointer["index_path"])
    if not predecessor_path.is_absolute():
        predecessor_path = REPO_ROOT / predecessor_path
    if sha256_file(predecessor_path) != pointer.get("index_sha256"):
        raise RuntimeError("current Task Packet pointer/index SHA mismatch")
    predecessor = load_json(predecessor_path)
    entries = list(predecessor.get("task_packets", []))
    ids = [str(item.get("task_id")) for item in entries]
    if len(ids) != len(set(ids)) or TASK_ID in ids:
        raise RuntimeError("current Task Packet index is duplicated or already contains v76")
    entries.append(
        {
            "task_id": TASK_ID,
            "packet_path": str(PACKET.relative_to(REPO_ROOT)),
            "packet_sha256": sha256_file(PACKET),
        }
    )
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "R7_3",
        "plan_revision": "chaoyang-v7.1",
        "execution_revision": "R3",
        "status": "PASS",
        "supersedes_index": artifact_ref(predecessor_path),
        "task_packets": entries,
        "claim_limit": "Execution routing only; v76 is registered PENDING and has not run.",
    }

    created = now_iso()
    state["recent_events"] = (
        state.get("recent_events", [])
        + [
            {
                "task_id": TASK_ID,
                "session": task.get("session"),
                "attempt": 0,
                "status": "PENDING",
                "created_at": created,
                "message": "Registered bounded v76 matrix-snapshot-layout successor; execution not started.",
                "task_packet": task["task_packet"],
            }
        ]
    )[-100:]
    packet_value = load_json(PACKET)
    state["next_task"] = {
        "task_id": TASK_ID,
        "session": task.get("session"),
        "prerequisites": list(packet_value["prerequisites"]),
        "expected_resource": "CPU Robot successor; no algorithm or GPU change",
        "stop_condition": packet_value["stop_condition"],
    }

    output.mkdir(parents=True)
    index_path = output / "TASK_PACKET_INDEX.json"
    result_path = output / "RESULT.json"
    packet_path = output / "TASK_PACKET.json"
    metrics_path = output / "METRICS.json"
    decision_path = output / "DECISION.md"
    next_path = output / "NEXT_ACTION.json"
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    receipt_path = output / "RUN_RECEIPT.json"
    result = {
        "schema_version": "register-robot-v76-governance-result-v1",
        "task_id": "register_robot_v76_governance",
        "status": "PASSED",
        "generated_at": created,
        "input_receipt": artifact_ref(RECEIPT_PATH),
        "predecessor_index": artifact_ref(predecessor_path),
        "v75_failure": artifact_ref(V75_FAILURE),
        "exact_task_packet": artifact_ref(PACKET),
        "predecessor_entry_count": len(entries) - 1,
        "successor_entry_count": len(entries),
        "execution_started": False,
        "claim_limit": "Governance registration only; no Robot execution or authority.",
    }
    _write_once_json(index_path, successor_index)
    _write_once_json(result_path, result)
    _write_once_json(
        packet_path,
        {
            "task_id": "register_robot_v76_governance",
            "objective": "CAS-register the exact v76 Task Packet after v75 runtime terminal.",
            "read_set": [str(RECEIPT_PATH), str(ROW_CANDIDATE), str(PACKET), str(V75_FAILURE)],
            "write_set": [str(output), str(RECEIPT_PATH.parent)],
            "stop_condition": "PASSED or CAS/reference failure",
            "claim_limit": "Governance only; no Robot execution.",
        },
    )
    _write_once_json(
        metrics_path,
        {
            "status": "PASSED",
            "predecessor_entries": len(entries) - 1,
            "successor_entries": len(entries),
            "robot_processes_started": 0,
        },
    )
    _write_once_text(
        decision_path,
        "# 决定\n\n原子注册 `robot_geometry_expansion_v76` 为 `PENDING`；v75 保持运行失败终态；未启动 Robot。\n",
    )
    _write_once_json(
        next_path,
        {
            "task_id": TASK_ID,
            "action": "EXPLICIT_CLAIM_REQUIRED",
            "execution_started": False,
        },
    )
    _write_once_json(
        manifest_path,
        {
            "schema_version": "register-robot-v76-artifact-manifest-v1",
            "artifacts": [
                artifact_ref(path)
                for path in (result_path, packet_path, metrics_path, decision_path, next_path)
            ],
        },
    )
    _write_once_json(
        receipt_path,
        {
            "schema_version": "register-robot-v76-run-receipt-v1",
            "task_id": "register_robot_v76_governance",
            "status": "PASSED",
            "generated_at": created,
            "result": artifact_ref(result_path),
            "artifact_manifest": artifact_ref(manifest_path),
            "execution_started": False,
            "claim_limit": "Governance registration only; no Robot authority.",
        },
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="ROBOT_V76_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=index_path,
        task_packet_index_value=successor_index,
    )
    print(
        json.dumps(
            {
                "status": "PASSED",
                "governance_revision": published["governance_revision"],
                "generation_id": published["generation_id"],
                "task_packet_index": artifact_ref(index_path),
                "execution_started": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
