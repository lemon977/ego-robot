"""Publish audited second-batch trim while preserving prior receipts and quality failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, load_json, now_iso, publish_bundle,
)

TASK = "human_to_robot_product_first_cleanup_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OLD = ROOT / "PROGRESS_CLEANUP_B1B.json"
NEW = ROOT / "PROGRESS_CLEANUP_B2.json"
SUMMARY = ROOT / "cleanup/DELETE_SUMMARY_B2.json"
B1 = ROOT / "cleanup/batch1/DELETE_RECEIPT.json"
B1B = ROOT / "cleanup/batch1b/DELETE_RECEIPT.json"
B2 = ROOT / "cleanup/batch2/DELETE_RECEIPT.json"
VALIDATION = ROOT / "validation_after_cleanup_batch2/RESULT.json"


def _new_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    if NEW.exists() or SUMMARY.exists():
        raise RuntimeError("BATCH2_PROGRESS_ALREADY_PUBLISHED")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    old, receipts = load_json(OLD), [load_json(p) for p in (B1, B1B, B2)]
    validation = load_json(VALIDATION)
    if any(row.get("execution") != "ACTUALLY_PURGED" for row in receipts):
        raise RuntimeError("BATCH_NOT_PURGED")
    if validation.get("status") != "PASS_FOR_CHECKED_SCOPE" or len(validation.get("slots", [])) != 15:
        raise RuntimeError("POST_BATCH2_VALIDATION_MISSING")
    total = sum(int(row["logical_deleted_bytes"]) for row in receipts)
    allocated = sum(int(row["allocated_deleted_bytes_estimate"]) for row in receipts)
    _new_json(SUMMARY, {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_DELETE_SUMMARY_V3",
        "task_id": TASK, "execution": "THREE_AUDITED_BATCHES_ACTUALLY_PURGED",
        "batch1": artifact_ref(B1), "batch1b": artifact_ref(B1B), "batch2": artifact_ref(B2),
        "historical_trim_notice": receipts[2]["trim_notice"],
        "post_delete_validation": artifact_ref(VALIDATION),
        "logical_deleted_bytes": total, "allocated_deleted_bytes_estimate": allocated,
        "physical_reclaimed_bytes": "UNKNOWN_CPFS_SHARED_ALLOCATION",
        "project_net_occupancy_change": "NOT_MEASURED_CONCURRENT_WRITES",
        "isolated_remaining_bytes": 0, "historical_receipts_modified": False,
        "claim_limit": "Only the exact audited intermediate/temporary/historical-derived frame trees were purged; 056/068 old full Depth arrays are no longer direct inputs. No product-quality or metric promotion.",
    })
    counts = {**old["counts"], "cleanup_logical_deleted_bytes_total": total,
              "cleanup_batch2_historical_stereo_logical_deleted_bytes": int(receipts[2]["logical_deleted_bytes"])}
    _new_json(NEW, {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_PROGRESS_V1",
        "task_id": TASK, "created_at": now_iso(), "checkpoint": "BATCH2_PURGED_AND_VALIDATED_NOT_TERMINAL",
        "previous": artifact_ref(OLD), "counts": counts, "cleanup": artifact_ref(SUMMARY),
        "post_cleanup_validation": artifact_ref(VALIDATION),
        "claim_limit": "Audited large-payload deletion is separate from 007 product quality, which remains rejected; historical 056/068 full Depth consumption revoked by appended notice.",
        "authority": {"training_eligible": False, "control_ground_truth": False,
                      "physical_deployable": False, "external_metric_authority": False},
    })
    task = next(row for row in state["tasks"] if row.get("task_id") == TASK)
    task.update(status="PENDING", attempt=1, updated_at=now_iso(), heartbeat_at=None,
                pid=None, proc_start_ticks=None)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PENDING", "created_at": now_iso(),
        "message": f"Two historical Stereo frame trees trimmed; cumulative logical deletion {total} bytes; 15 current slots revalidated; product quality unchanged.",
    }])[-100:]
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_B2_PUBLISHED",
        expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps({"status": "PASS", "revision": published["governance_revision"],
                      "logical_deleted_bytes": total, "progress": str(NEW)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
