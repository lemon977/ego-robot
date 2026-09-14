from __future__ import annotations

import argparse
from datetime import datetime
from tools.governance.common import AUTHORITY_PATH, GOVERNANCE_ROOT, RECEIPT_PATH, STATUS_PATH, TASK_STATE_PATH, artifact_ref, atomic_json, atomic_write, load_json, validate_artifact_ref, validate_schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    source_receipt = load_json(RECEIPT_PATH)
    errors = [error for item in source_receipt["files"].values() for error in validate_artifact_ref(item)]
    if errors:
        raise SystemExit("STATUS_CONFLICT:\n" + "\n".join(errors))
    stamp = args.timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    destination = GOVERNANCE_ROOT / "snapshots"
    outputs = {
        "status": destination / f"{stamp}_MEETING_STATUS_ZH.md",
        "authority": destination / f"{stamp}_AUTHORITY_INDEX.json",
        "task_state": destination / f"{stamp}_TASK_STATE.json",
    }
    atomic_write(outputs["status"], STATUS_PATH.read_bytes())
    atomic_write(outputs["authority"], AUTHORITY_PATH.read_bytes())
    atomic_write(outputs["task_state"], TASK_STATE_PATH.read_bytes())
    receipt = {
        "schema_version": "chaoyang-project-status-snapshot-receipt-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_receipt": artifact_ref(RECEIPT_PATH),
        "files": {key: artifact_ref(path) for key, path in outputs.items()},
    }
    validate_schema("project_status_snapshot_receipt.schema.json", receipt)
    receipt_path = destination / f"{stamp}_STATUS_RECEIPT.json"
    atomic_json(receipt_path, receipt)
    current_index = {
        "schema_version": "chaoyang-current-meeting-snapshot-v1",
        "governance_revision": source_receipt["governance_revision"],
        "generation_id": source_receipt["generation_id"],
        "snapshot_receipt": artifact_ref(receipt_path),
        "claim_limit": "Immutable meeting snapshot index; current execution still requires CURRENT_STATUS_RECEIPT.json validation.",
    }
    atomic_json(GOVERNANCE_ROOT / "CURRENT_MEETING_SNAPSHOT.json", current_index)
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
