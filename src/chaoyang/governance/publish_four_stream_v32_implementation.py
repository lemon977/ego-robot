#!/usr/bin/env python3
"""Publish V3.2 code/docs/contracts while preserving an empty route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.build_four_stream_status_v32 import build_status
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
    validate_artifact_ref,
)


TASK_ID = "four_stream_pretraining_baseline_v32"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
OUTPUT_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/implementation_publication_0001"
STATUS_PATH = REPO_ROOT / "docs/current/STATUS.json"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def _write_once(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise RuntimeError(f"fresh publication root required: {OUTPUT_ROOT}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != args.expected_revision:
        raise RuntimeError("CAS revision mismatch")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None or any(row.get("status") in LIVE for row in state.get("tasks", [])):
        raise RuntimeError("implementation publication requires no live task")
    index = load_json(INDEX)
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("implementation publication requires empty current index")
    created = now_iso()
    OUTPUT_ROOT.mkdir(parents=True)
    projected = OUTPUT_ROOT / "PROJECTED_TASK_STATE.json"
    _write_once(projected, state)
    atomic_json(
        STATUS_PATH,
        build_status(task_state=state, generated_at=created, source_task_state_path=projected),
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="FOUR_STREAM_PRETRAINING_BASELINE_V32_IMPLEMENTATION_PUBLISHED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )
    result_path = OUTPUT_ROOT / "RESULT.json"
    _write_once(result_path, {
        "schema_version": "chaoyang-four-stream-implementation-publication-result-v1",
        "status": "PASSED",
        "task_id": TASK_ID,
        "published_at": created,
        "governance_revision": published["governance_revision"],
        "shallow_status": artifact_ref(STATUS_PATH),
        "execution_registered": False,
    })
    print(json.dumps({"status": "PASSED", "governance_revision": published["governance_revision"], "result": str(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

