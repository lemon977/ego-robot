#!/usr/bin/env python3
"""Publish a content-addressed repair bundle for the first R2.2 G0 snapshot."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import (  # noqa: E402
    ALGORITHM_CONTRACT_PATH,
    RECEIPT_PATH,
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_write,
    load_json,
    now_iso,
)


RUN_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h"
ATTEMPT = RUN_ROOT / "g0/attempts/attempt_0002_content_addressed_repair"
ORIGINAL = RUN_ROOT / "g0/attempts/attempt_0001/RUN_START_SNAPSHOT.json"
HARD_SOFT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_v76_r3/ROBOT_HARD_SOFT_CANDIDATE_INDEX.json"


def freeze(source: Path) -> dict:
    source_ref = artifact_ref(source)
    target = ATTEMPT / "input_snapshot" / source_ref["sha256"] / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    frozen_ref = artifact_ref(target)
    if (frozen_ref["bytes"], frozen_ref["sha256"]) != (source_ref["bytes"], source_ref["sha256"]):
        raise RuntimeError(f"freeze verification failed: {source}")
    return {"source": source_ref, "frozen": frozen_ref}


def write_once(path: Path, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable conflict: {path}")
        return
    atomic_write(path, data)


def main() -> int:
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    original = load_json(ORIGINAL)
    current_receipt = load_json(RECEIPT_PATH)
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    packet_index = ROOT / pointer["index_path"]
    files = {
        "current_status_receipt_at_repair": freeze(RECEIPT_PATH),
        "task_packet_index_at_repair": freeze(packet_index),
        "algorithm_contract_at_repair": freeze(ALGORITHM_CONTRACT_PATH),
        "robot_hard_soft_candidate_index": freeze(HARD_SOFT),
    }
    original_algorithm = original["algorithm_contract"]
    current_algorithm = files["algorithm_contract_at_repair"]["source"]
    original_replayable = (
        original_algorithm["bytes"] == current_algorithm["bytes"]
        and original_algorithm["sha256"] == current_algorithm["sha256"]
    )
    payload = {
        "schema_version": "chaoyang-r22-g0-content-repair-v1",
        "created_at": now_iso(),
        "status": "PASSED_WITH_PROVENANCE_DEBT" if not original_replayable else "PASSED",
        "original_snapshot": artifact_ref(ORIGINAL),
        "original_governance_revision": original["governance_revision"],
        "repair_governance_revision": current_receipt["governance_revision"],
        "files": files,
        "original_algorithm_contract_replayable": original_replayable,
        "provenance_debt": [] if original_replayable else [
            {
                "code": "G0_MUTABLE_REFERENCE_BYTES_NOT_FROZEN",
                "scope": "R2.2 algorithm-contract snapshot at revision 10345",
                "impact": "The old reference SHA is retained but its bytes cannot be reconstructed from the first bundle.",
                "claim_limit": "Robot v77 may use the immutable v76 RUN_CODE_CLOSURE; no worker may silently substitute the repaired latest contract for the missing old bytes.",
            }
        ],
        "worker_policy": "Use immutable v76 code closure for v77; use this bundle only for tasks explicitly started after the repair.",
        "authority_promoted": False,
    }
    result = ATTEMPT / "G0_SNAPSHOT_REPAIR_RECEIPT.json"
    write_once(result, payload)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
