"""Atomically register the immutable Robot v75 task and packet index."""

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

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    GOVERNANCE_ROOT,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    canonical_bytes,
    load_json,
    now_iso,
    publish_bundle,
    sha256_file,
    validate_artifact_ref,
)


CANDIDATE_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v75/attempts/attempt_0016_execution"
ROW_CANDIDATE = CANDIDATE_ROOT / "TASK_STATE_ROW_CANDIDATE.json"
INDEX_CANDIDATE = CANDIDATE_ROOT / "TASK_PACKET_INDEX_CANDIDATE.json"
ATTEMPT_ROOT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_chaoyang_v71_r3/MIGRATE-00/attempts/attempt_0002_register_v75"
SUCCESSOR_INDEX = ATTEMPT_ROOT / "TASK_PACKET_INDEX.json"


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return
    atomic_json(path, value)


def _write_text_once(path: Path, value: str) -> None:
    encoded = value.encode("utf-8")
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return
    atomic_write(path, encoded)


def _validate_ref(reference: dict[str, Any]) -> Path:
    errors = validate_artifact_ref(reference)
    if errors:
        raise RuntimeError("candidate artifact mismatch:\n" + "\n".join(errors))
    return Path(reference["path"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()

    receipt = load_json(RECEIPT_PATH)
    if receipt["governance_revision"] != args.expected_revision:
        raise RuntimeError(f"CAS revision mismatch before registration: expected={args.expected_revision} current={receipt['governance_revision']}")
    for reference in receipt["files"].values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict:\n" + "\n".join(errors))

    row_candidate = load_json(ROW_CANDIDATE)
    index_candidate = load_json(INDEX_CANDIDATE)
    task = dict(row_candidate["task"])
    if row_candidate.get("action") != "REGISTER_IF_ABSENT" or task.get("task_id") != "robot_geometry_expansion_v75":
        raise RuntimeError("unexpected task-state candidate")
    packet_path = _validate_ref(task["task_packet"])
    if packet_path != CANDIDATE_ROOT / "TASK_PACKET.json":
        raise RuntimeError("v75 task row does not carry the exact expected packet ref")
    _validate_ref(index_candidate["predecessor_pointer"])
    predecessor_path = _validate_ref(index_candidate["predecessor_index"])
    predecessor = load_json(predecessor_path)
    predecessor_entries = list(predecessor.get("task_packets", []))
    predecessor_ids = [str(item.get("task_id")) for item in predecessor_entries]
    if len(predecessor_ids) != len(set(predecessor_ids)):
        raise RuntimeError("predecessor task packet index contains duplicate task ids")
    new_entry = dict(index_candidate["task_packet_entry"])
    expected_entry_sha = sha256_file(REPO_ROOT / new_entry["packet_path"])
    if new_entry.get("packet_sha256") != expected_entry_sha:
        raise RuntimeError("v75 packet index candidate SHA mismatch")
    if new_entry["task_id"] in predecessor_ids:
        raise RuntimeError("v75 already exists in predecessor packet index")
    successor_entries = predecessor_entries + [new_entry]
    if successor_entries[: len(predecessor_entries)] != predecessor_entries:
        raise RuntimeError("predecessor entries were not preserved byte-for-structure")

    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    if any(item.get("task_id") == task["task_id"] for item in state.get("tasks", [])):
        raise RuntimeError("robot_geometry_expansion_v75 already exists in task state")
    task["plan_execution_revision"] = "R3"
    state["tasks"].append(task)

    clean_stage = next(item for item in authority.get("stages", []) if item.get("stage") == "Clean")
    waves = authority.get("waves", {})
    clean_closed = (
        clean_stage.get("passed") == 58
        and clean_stage.get("running") == 0
        and waves.get("wave0_clean_passed") == 58
        and waves.get("wave0_clean_pending") == 0
    )
    removed_blockers: list[dict[str, Any]] = []
    if clean_closed:
        keep = []
        for blocker in state.get("blockers", []):
            if blocker.get("name") == "Shared H20 occupied by external egotouch training" and blocker.get("scope") == "Wave0 ProPainter Clean GPU batch":
                removed_blockers.append(blocker)
            else:
                keep.append(blocker)
        state["blockers"] = keep

    created = now_iso()
    state["recent_events"] = (
        state.get("recent_events", [])
        + [{
            "task_id": task["task_id"],
            "session": task.get("session"),
            "attempt": task.get("attempt", 0),
            "status": "PENDING",
            "created_at": created,
            "message": "Registered exact immutable v75 packet; execution not started.",
            "task_packet": task["task_packet"],
        }]
    )[-100:]
    state["next_task"] = {
        "task_id": "robot_geometry_expansion_v75",
        "session": task["session"],
        "prerequisites": list(load_json(packet_path)["prerequisites"]),
        "expected_resource": "CPU Robot geometry successor; GPU only for bounded render after an explicit claim",
        "stop_condition": load_json(packet_path)["stop_condition"],
    }

    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "R7_3",
        "plan_revision": "chaoyang-v7.1",
        "execution_revision": "R3",
        "status": "PASS",
        "supersedes_index": artifact_ref(predecessor_path),
        "task_packets": successor_entries,
        "claim_limit": "Execution routing only; no algorithm authority. v75 is registered PENDING and has not run.",
    }
    result = {
        "schema_version": "register-robot-v75-governance-result-v1",
        "task_id": "register_robot_v75_governance",
        "status": "PASSED",
        "generated_at": created,
        "input_receipt": artifact_ref(RECEIPT_PATH),
        "task_state_candidate": artifact_ref(ROW_CANDIDATE),
        "task_packet_index_candidate": artifact_ref(INDEX_CANDIDATE),
        "exact_task_packet": artifact_ref(packet_path),
        "predecessor_entry_count": len(predecessor_entries),
        "successor_entry_count": len(successor_entries),
        "removed_completed_clean_blockers": removed_blockers,
        "execution_started": False,
        "claim_limit": "Governance registration only; no Robot process, output or authority.",
    }
    result_path = ATTEMPT_ROOT / "RESULT.json"
    task_packet_path = ATTEMPT_ROOT / "TASK_PACKET.json"
    metrics_path = ATTEMPT_ROOT / "METRICS.json"
    decision_path = ATTEMPT_ROOT / "DECISION.md"
    next_action_path = ATTEMPT_ROOT / "NEXT_ACTION.json"
    manifest_path = ATTEMPT_ROOT / "ARTIFACT_MANIFEST.json"
    run_receipt_path = ATTEMPT_ROOT / "RUN_RECEIPT.json"
    _write_once(result_path, result)
    _write_once(task_packet_path, {
        "task_id": "register_robot_v75_governance",
        "objective": "Atomically register v75 in task state/current packet index and bind current regression refs without launching it.",
        "read_set": [str(RECEIPT_PATH), str(ROW_CANDIDATE), str(INDEX_CANDIDATE)],
        "write_set": [str(GOVERNANCE_ROOT), str(ATTEMPT_ROOT)],
        "stop_condition": "PASSED or CAS/reference failure",
        "claim_limit": "Governance only; no Robot execution.",
    })
    _write_once(metrics_path, {"status": "PASSED", "predecessor_entries": len(predecessor_entries), "successor_entries": len(successor_entries), "removed_clean_blockers": len(removed_blockers), "robot_processes_started": 0})
    _write_text_once(
        decision_path,
        "# 决定\n\n原子注册 `robot_geometry_expansion_v75` 为 `PENDING`；未启动 Robot。"
        " 保留全部前序任务包，只删除已有 58/58 Clean 终态明确证明为过期的 Wave0 H20 阻塞。\n",
    )
    _write_once(next_action_path, {
        "task_id": "robot_geometry_expansion_v75",
        "action": "EXPLICIT_CLAIM_REQUIRED",
        "execution_started": False,
        "claim_limit": "Registration does not authorize automatic Robot launch.",
    })
    _write_once(manifest_path, {
        "schema_version": "register-robot-v75-artifact-manifest-v1",
        "artifacts": [artifact_ref(path) for path in (result_path, task_packet_path, metrics_path, decision_path, next_action_path)],
    })
    _write_once(run_receipt_path, {
        "schema_version": "register-robot-v75-run-receipt-v1",
        "task_id": "register_robot_v75_governance",
        "status": "PASSED",
        "generated_at": created,
        "result": artifact_ref(result_path),
        "artifact_manifest": artifact_ref(manifest_path),
        "metrics": artifact_ref(metrics_path),
        "execution_started": False,
        "claim_limit": "Governance registration only; no Robot authority.",
    })

    published = publish_bundle(
        authority,
        state,
        event_type="ROBOT_V75_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=SUCCESSOR_INDEX,
        task_packet_index_value=successor_index,
    )
    print(json.dumps({
        "status": "PASSED",
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "task_packet_index": artifact_ref(SUCCESSOR_INDEX),
        "execution_started": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
