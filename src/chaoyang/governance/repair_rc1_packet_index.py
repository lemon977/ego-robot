#!/usr/bin/env python3
"""Repair the one-time RC1 migration packet-index duplicate by CAS publish."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import json
from pathlib import Path

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
)


OUT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/governance_repair/attempts/attempt_0001"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("repair CAS revision mismatch")
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    source = Path(str(pointer["index_path"]))
    if not source.is_absolute():
        source = REPO_ROOT / source
    index = load_json(source)
    deduped = {}
    for item in index.get("task_packets", []):
        deduped[str(item["task_id"])] = item
    if len(deduped) == len(index.get("task_packets", [])):
        raise RuntimeError("packet index has no duplicate to repair")
    repaired_path = OUT / "TASK_PACKET_INDEX.json"
    repaired = dict(index)
    repaired["packet_revision"] = "RC1_FINAL_0002_DEDUPED"
    repaired["supersedes_index"] = artifact_ref(source)
    repaired["task_packets"] = list(deduped.values())
    repaired["repair_reason"] = "First failed migration publish left a pointer that was reused as predecessor; duplicate T0 registration removed without changing its packet SHA."
    repaired_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "chaoyang-rc1-governance-repair-result-v1",
        "task_id": "rc1_packet_index_dedupe_repair",
        "status": "PASSED",
        "created_at": now_iso(),
        "source_index": artifact_ref(source),
        "repaired_index_path": str(repaired_path),
        "removed_duplicate_count": len(index["task_packets"]) - len(deduped),
    }
    result_path = OUT / "RESULT.json"
    atomic_json(result_path, result)
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": "rc1_packet_index_dedupe_repair", "attempt": 1,
        "status": "PASSED", "created_at": result["created_at"],
        "message": "Removed duplicate RC1-T0 packet registration.",
        "result": artifact_ref(result_path),
    }])[-100:]
    published = publish_bundle(
        authority, state, event_type="RC1_PACKET_INDEX_REPAIRED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
        task_packet_index_path=repaired_path, task_packet_index_value=repaired,
    )
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "result": str(result_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
