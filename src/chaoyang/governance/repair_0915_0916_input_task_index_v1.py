#!/usr/bin/env python3
"""CAS-repair the missing execution classification on the new task index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    publish_bundle,
)


TASK_ID = "0915_0916_input_audit_clean_v1"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh repair output required: {output}")
    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch")
    index = load_json(CURRENT_INDEX)
    entries = index.get("task_packets", [])
    if len(entries) != 1 or entries[0].get("task_id") != TASK_ID:
        raise RuntimeError("unexpected current task index")
    if entries[0].get("execution_allowed") is not True:
        raise RuntimeError("repair only applies to the live routable task")
    output.mkdir(parents=True)
    predecessor = output / "PREDECESSOR_INVALID_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, predecessor)
    entries[0]["execution_class"] = "CURRENT_LEDGER_ROUTABLE"
    successor = {
        **{key: value for key, value in index.items()
           if key not in {"governance_revision", "generation_id", "generated_at",
                          "supersedes_index", "task_packets"}},
        "packet_revision": "0915_0916_INPUT_AUDIT_CLEAN_V1_CLASSIFIED",
        "supersedes_index": artifact_ref(predecessor),
        "task_packets": entries,
        "claim_limit": "Classified current routable weightless task; no algorithm execution or authority change.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH), load_json(TASK_STATE_PATH),
        event_type="0915_0916_INPUT_TASK_INDEX_CLASSIFICATION_REPAIRED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=successor,
    )
    result = {
        "schema_version": "repair-0915-0916-input-task-index-v1",
        "status": "PASSED",
        "task_id": TASK_ID,
        "execution_class": "CURRENT_LEDGER_ROUTABLE",
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "execution_started": False,
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
