from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    DOC_AUTHORITY_MAP_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.current_r3_contracts import build_doc_authority_map


ROUTABLE_STATUSES = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
MATRIX_PATH = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_three_line_successor_v1/TASK_MATRIX_REV_0014.json"
INDEX_PATH = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260917_governance_consistency_repair_v1/attempts/attempt_0001/TASK_PACKET_INDEX.json"


def validate_matrix() -> None:
    matrix = load_json(MATRIX_PATH)
    if matrix.get("revision") != 14 or matrix.get("formal_rc1_changed") is not False:
        raise RuntimeError("research task matrix rev14 semantics are invalid")
    references = [matrix["supersedes"]] + [item["result"] for item in matrix["new_results"]]
    errors = [error for reference in references for error in validate_artifact_ref(reference)]
    if errors:
        raise RuntimeError("research task matrix closure failed:\n" + "\n".join(errors))


def classify_index(
    source_index: dict[str, object],
    task_state: dict[str, object],
    doc_authority_map: dict[str, object],
    source_path: Path,
) -> dict[str, object]:
    document_status = {
        str(Path(str(item["path"])).resolve()): str(item["status"])
        for item in doc_authority_map.get("documents", [])
        if isinstance(item, dict)
    }
    task_status = {
        str(item.get("task_id")): str(item.get("status"))
        for item in task_state.get("tasks", [])
        if isinstance(item, dict)
    }
    next_task = task_state.get("next_task")
    next_task_id = str(next_task.get("task_id")) if isinstance(next_task, dict) else None
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for raw_row in source_index.get("task_packets", []):
        if not isinstance(raw_row, dict):
            raise RuntimeError("source task index contains a non-object row")
        row = dict(raw_row)
        packet_path = Path(str(row.get("packet_path", "")))
        if not packet_path.is_absolute():
            packet_path = REPO_ROOT / packet_path
        packet = load_json(packet_path)
        superseded_paths: list[str] = []
        for raw_path in packet.get("read_set", []):
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            resolved = str(candidate.resolve())
            if document_status.get(resolved) == "SUPERSEDED":
                superseded_paths.append(resolved)
        task_id = str(row.get("task_id"))
        status = task_status.get(task_id)
        if superseded_paths:
            execution_class = "HISTORICAL_SUPERSEDED_READ_SET"
            reason = "Packet read_set includes a document registered as SUPERSEDED."
            execution_allowed = False
        elif status is not None and status not in ROUTABLE_STATUSES:
            execution_class = "TERMINAL_LEDGER_EVIDENCE"
            reason = f"Current ledger status is terminal/non-routable: {status}."
            execution_allowed = False
        elif status in ROUTABLE_STATUSES:
            execution_class = "CURRENT_LEDGER_ROUTABLE"
            execution_allowed = task_id == next_task_id
            reason = (
                "Ledger state is routable and this is the explicit next_task."
                if execution_allowed
                else "Ledger state may be routable, but this is not the explicit next_task."
            )
        else:
            execution_class = "CATALOG_REFERENCE"
            reason = "Packet is retained for lookup but has no current ledger dispatch state."
            execution_allowed = False
        row.update(
            execution_class=execution_class,
            execution_allowed=execution_allowed,
            execution_reason=reason,
            superseded_read_set_paths=sorted(set(superseded_paths)),
        )
        counts[execution_class] += 1
        rows.append(row)
    return {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "RC1_FINAL_0007_GOVERNANCE_CLASSIFIED",
        "plan_revision": source_index["plan_revision"],
        "execution_revision": source_index.get("execution_revision", "RC1"),
        "status": "PASS",
        "claim_limit": "Receipt-bound packet catalog. Presence does not authorize execution; only execution_allowed=true plus current ledger next_task may dispatch.",
        "execution_semantics": "SUPERSEDED read-sets and terminal packets are retained as immutable history and cannot be restarted from this index.",
        "revision_semantics": {
            "source_index_created_governance_revision": source_index.get("governance_revision"),
            "plan_revision_is_frozen_identifier": True,
            "this_index_governance_revision": "set atomically by publish_bundle",
        },
        "classification_counts": dict(sorted(counts.items())),
        "supersedes_index": artifact_ref(source_path),
        "task_packets": rows,
    }


def main() -> int:
    validate_matrix()
    receipt = load_json(RECEIPT_PATH)
    authority = load_json(AUTHORITY_PATH)
    task_state = load_json(TASK_STATE_PATH)
    active = [
        str(item.get("task_id"))
        for item in task_state.get("tasks", [])
        if isinstance(item, dict) and item.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    ]
    if active:
        raise RuntimeError(f"refusing governance repair while tasks are active: {active}")
    pointer = load_json(REPO_ROOT / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json")
    source_path = Path(str(pointer["index_path"]))
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    source_index = load_json(source_path)
    preview_doc_map = build_doc_authority_map(authority, DOC_AUTHORITY_MAP_PATH.parent, REPO_ROOT)
    successor_index = classify_index(source_index, task_state, preview_doc_map, source_path)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=False)
    published = publish_bundle(
        authority,
        task_state,
        event_type="GOVERNANCE_CONSISTENCY_REPAIR_V1",
        expected_revision=int(receipt["governance_revision"]),
        generator_path=Path(__file__),
        task_packet_index_path=INDEX_PATH,
        task_packet_index_value=successor_index,
    )
    print(json.dumps({
        "status": "PUBLISHED",
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "task_index": str(INDEX_PATH),
        "classification_counts": successor_index["classification_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
