from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    TASK_STATE_PATH,
    load_json,
    publish_bundle,
)


def main() -> int:
    receipt = load_json(RECEIPT_PATH)
    authority = load_json(AUTHORITY_PATH)
    task_state = load_json(TASK_STATE_PATH)
    active = [
        str(item.get("task_id"))
        for item in task_state.get("tasks", [])
        if isinstance(item, dict) and item.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    ]
    if active:
        raise RuntimeError(f"refusing governance refresh while tasks are active: {active}")
    published = publish_bundle(
        authority,
        task_state,
        event_type="GOVERNANCE_TEST_FIXTURE_REFRESH_V1",
        expected_revision=int(receipt["governance_revision"]),
        generator_path=Path(__file__),
    )
    print(json.dumps({
        "status": "PUBLISHED",
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "semantic_state_changed": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
