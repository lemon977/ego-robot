"""Validate current consumers and retained forensic evidence after batch-2 trim."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np

from chaoyang.governance.common import artifact_ref, load_json, validate_artifact_ref
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK
from chaoyang.ops.validate_human_to_robot_product_first_after_cleanup import decode, _environment

ATTEMPT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OUT = ATTEMPT / "validation_after_cleanup_batch2/RESULT.json"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    old = load_json(ROOT / "_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001/delivery/DELIVERY_MANIFEST.json")
    formal = load_json(ATTEMPT / "lanes/motion_product/formal_007_current/attempt_0001/PRODUCT_RESULT.json")
    if old.get("slot_count") != 15 or len(old.get("slots", [])) != 15:
        raise RuntimeError("SLOT_COUNT")
    slots = []
    for row in old["slots"]:
        ref = formal["product_video"] if row["slot_id"] == "PRODUCT_007" else row["video"]
        if validate_artifact_ref(ref) or decode(Path(ref["path"])) != int(row["expected_frames"]):
            raise RuntimeError(f"SLOT_CHANGED:{row['slot_id']}")
        slots.append({"slot_id": row["slot_id"], "session_id": row["session_id"],
                      "decoded_frames": int(row["expected_frames"]), "video": ref})
    fixed = []
    for relative in (
        "lanes/scene/cable_007/clean_window_v1/007_CABLE_OLD_NEW_CLEAN_REVIEW.mp4",
        "lanes/motion_product/product_007/window_v1/007_OLD_NEW_CLEAN_ROBOT_WINDOW.mp4",
    ):
        path = ATTEMPT / relative
        if decode(path) != 16:
            raise RuntimeError(f"FIXED_WINDOW_CHANGED:{relative}")
        fixed.append(artifact_ref(path))
    batch2 = load_json(ATTEMPT / "cleanup/batch2/DELETE_RECEIPT.json")
    selection = load_json(ATTEMPT / "cleanup/batch2/SELECTED_TARGETS.json")
    notice = ATTEMPT / "cleanup/batch2/HISTORICAL_PAYLOAD_TRIM_NOTICE.json"
    if batch2.get("execution") != "ACTUALLY_PURGED" or validate_artifact_ref(batch2["trim_notice"]):
        raise RuntimeError("BATCH2_RECEIPT_INVALID")
    if batch2["trim_notice"]["path"] != str(notice):
        raise RuntimeError("TRIM_NOTICE_NOT_BOUND")
    retained = []
    for row in selection["targets"]:
        if Path(row["source"]).exists():
            raise RuntimeError(f"PURGED_PAYLOAD_REAPPEARED:{row['session_id']}")
        for key in ("historical_result", "historical_depth_summary", "retained_review"):
            ref = row[key]
            if validate_artifact_ref(ref):
                raise RuntimeError(f"HISTORICAL_REF_CHANGED:{row['session_id']}:{key}")
        if decode(Path(row["retained_review"]["path"])) != int(row["frame_count"]):
            raise RuntimeError(f"HISTORICAL_REVIEW_INCOMPLETE:{row['session_id']}")
        for ref in row["retained_samples"].values():
            if validate_artifact_ref(ref):
                raise RuntimeError(f"HISTORICAL_SAMPLE_CHANGED:{row['session_id']}")
            with np.load(ref["path"], allow_pickle=False) as archive:
                if not archive.files:
                    raise RuntimeError(f"HISTORICAL_SAMPLE_EMPTY:{row['session_id']}")
        retained.append({"session_id": row["session_id"], "review_decoded": row["frame_count"],
                         "samples_loaded": len(row["retained_samples"])})
    environment = _environment("foundationstereo-py311-v1")
    command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "validate-governance"]
    check = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(ROOT / ".cache/tmp"),
        "XDG_CACHE_HOME": str(ROOT / ".cache/xdg")}, capture_output=True, text=True, check=False)
    if check.returncode or '"status": "PASS"' not in check.stdout:
        raise RuntimeError(f"GOVERNANCE_FAILED:{check.stdout[-700:]}:{check.stderr[-700:]}")
    result = {"schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_POST_BATCH2_VALIDATION_V1",
        "task_id": TASK, "status": "PASS_FOR_CHECKED_SCOPE", "slots": slots,
        "new_fixed_windows": fixed, "historical_sparse_retention": retained,
        "environment": environment, "cleanup_batch2": artifact_ref(ATTEMPT / "cleanup/batch2/DELETE_RECEIPT.json"),
        "governance": "PASS", "historical_full_depth_consumption": "REVOKED_BY_TRIM_NOTICE",
        "protected_root_snapshot_before_after": "NOT_ESTABLISHED_AT_T0",
        "claim_limit": "Validation covers retained current/forensic files, not old complete Stereo arrays, product quality, or physical CPFS reclaim.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False}
    OUT.parent.mkdir(parents=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "slots": len(slots),
                      "historical_sparse_retention": retained, "result": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
