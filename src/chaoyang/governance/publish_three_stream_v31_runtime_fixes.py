#!/usr/bin/env python3
"""Publish bounded V3.1 runtime fixes while no current task is active."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
OUTPUT_ROOT = REPO_ROOT / "_run/current/three_stream_v31_runtime_fixes/publication_0001"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def _write_once(path: Path, value: dict[str, object]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise RuntimeError(f"fresh runtime-fix publication required: {OUTPUT_ROOT}")

    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before runtime-fix publication")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None or any(
        row.get("status") in LIVE for row in state.get("tasks", [])
    ):
        raise RuntimeError("runtime-fix publication requires no live task")
    index = load_json(CURRENT_INDEX)
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("runtime-fix publication requires an empty current index")

    created = now_iso()
    OUTPUT_ROOT.mkdir(parents=True)
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="THREE_STREAM_V31_RUNTIME_FIXES_PUBLISHED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=index,
    )
    result = {
        "schema_version": "chaoyang-three-stream-v31-runtime-fixes-publication-v1",
        "status": "PASSED",
        "published_at": created,
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "execution_registered": False,
        "claim_limit": (
            "Code publication only. The prior failed/rejected attempt remains immutable; "
            "successor execution requires a fresh routed task packet."
        ),
    }
    result_path = OUTPUT_ROOT / "RESULT.json"
    _write_once(result_path, result)
    print(json.dumps({**result, "result": artifact_ref(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
