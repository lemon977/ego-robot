"""CAS-publish T5's receipt-bound release flags into the current RC1 projection."""

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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    TASK_STATE_PATH,
    freshness,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.rc1_t5_consistency import t5_release_flags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if receipt["governance_revision"] != args.expected_revision:
        raise RuntimeError("current revision changed before T5 reconciliation")
    for key in ("authority_index", "task_state"):
        errors = validate_artifact_ref(receipt["files"][key])
        if errors:
            raise RuntimeError("current receipt closure failed: " + "; ".join(errors))
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    if any(value.get("governance_revision") != args.expected_revision for value in (authority, state)):
        raise RuntimeError("current authority/task revision mismatch")
    if freshness(state)["status"] in {"STALE", "SUSPECTED_DEAD_WORKER"}:
        raise RuntimeError("current task state is stale; recover it first")
    flags, errors = t5_release_flags(state)
    if errors or flags is None:
        raise RuntimeError("T5 receipt is not publishable: " + "; ".join(errors or ["no PASSED T5 result"]))
    old = dict(state.get("rc1_release_flags", {}))
    if all(old.get(key) == value for key, value in flags.items()):
        print(json.dumps({"status": "ALREADY_SYNCED", "revision": args.expected_revision}, ensure_ascii=False))
        return 0
    if old.get("RC1_RELEASE_STATUS") not in {"T0_COMPLETE_CAPACITY_LIMITED", flags["RC1_RELEASE_STATUS"]}:
        raise RuntimeError("current RC1 status is not the known T0 projection; refusing blind replacement")
    state["rc1_release_flags"] = {**old, **flags}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": "rc1_t5_batch_conversion",
        "attempt": 1,
        "status": "PASSED",
        "created_at": now_iso(),
        "message": "Synchronized receipt-bound T5 release flags; no authority or task terminal changed.",
    }])[-100:]
    published = publish_bundle(
        authority,
        state,
        event_type="RC1_T5_RELEASE_FLAGS_SYNC",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(json.dumps({
        "status": "PASSED",
        "revision": published["governance_revision"],
        "previous_release_status": old.get("RC1_RELEASE_STATUS"),
        "release_flags": flags,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
