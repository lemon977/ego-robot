"""Publish append-only cleanup progress without changing product authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "human_to_robot_product_first_cleanup_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OLD = ROOT / "PROGRESS_H2.json"
NEW = ROOT / "PROGRESS_CLEANUP_B1B.json"
SUMMARY = ROOT / "cleanup/DELETE_RECEIPT.json"
B1 = ROOT / "cleanup/batch1/DELETE_RECEIPT.json"
B1B = ROOT / "cleanup/batch1b/DELETE_RECEIPT.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    if NEW.exists():
        raise FileExistsError(NEW)
    old, batch1, batch1b = load_json(OLD), load_json(B1), load_json(B1B)
    if batch1.get("execution") != "ACTUALLY_PURGED" or batch1b.get("execution") != "ACTUALLY_PURGED":
        raise RuntimeError("CLEANUP_NOT_ACTUALLY_PURGED")
    total = int(batch1["logical_deleted_bytes"]) + int(batch1b["logical_deleted_bytes"])
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    atomic_json(SUMMARY, {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_DELETE_SUMMARY_V2",
        "task_id": TASK, "execution": "TWO_FIRST_BATCHES_ACTUALLY_PURGED",
        "batch1": artifact_ref(B1), "batch1b": artifact_ref(B1B),
        "batch2": "NOT_YET_FROZEN_OR_PURGED",
        "logical_deleted_bytes": total,
        "allocated_deleted_bytes_estimate": int(batch1["allocated_deleted_bytes_estimate"])
                                            + int(batch1b["allocated_deleted_bytes_estimate"]),
        "physical_reclaimed_bytes": "UNKNOWN_CPFS_SHARED_ALLOCATION",
        "project_net_occupancy_change": "NOT_MEASURED_CONCURRENT_WRITES",
        "historical_receipts_modified": False,
        "claim_limit": "Only two audited first-batch targets purged. Large historical payloads and environments remain pending batch2 proof.",
    })
    counts = {**old["counts"], "cleanup_logical_deleted_bytes_total": total,
              "cleanup_batch1b_aborted_stereo_logical_deleted_bytes": int(batch1b["logical_deleted_bytes"])}
    atomic_json(NEW, {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_PROGRESS_V1",
        "task_id": TASK, "created_at": now_iso(), "checkpoint": "FIRST_BATCH_CLEANUP_EXTENDED_NOT_TERMINAL",
        "previous": artifact_ref(OLD), "counts": counts,
        "cleanup": artifact_ref(SUMMARY),
        "claim_limit": "Actual audited deletion separate from 007 product quality, which remains rejected.",
        "authority": {"training_eligible": False, "control_ground_truth": False,
                      "physical_deployable": False, "external_metric_authority": False},
    })
    task = next(row for row in state["tasks"] if row.get("task_id") == TASK)
    task.update(status="PENDING", attempt=1, updated_at=now_iso(), heartbeat_at=None,
                pid=None, proc_start_ticks=None)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PENDING", "created_at": now_iso(),
        "message": f"Second audited first-batch purge removed abandoned Stereo staging; cumulative logical deletion {total} bytes; product quality unchanged.",
    }])[-100:]
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_B1B_PUBLISHED",
        expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps({"status": "PASS", "revision": published["governance_revision"],
                      "logical_deleted_bytes": total, "progress": str(NEW)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
