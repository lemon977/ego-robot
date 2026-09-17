#!/usr/bin/env python3
"""ROBOT-REACH-20: classify motion retention separately from pose imitation."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.robot_target_reach_v75 import artifact_ref, atomic_new_json, load_json, now_iso  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = load_json(args.target_audit.resolve(strict=True))
    if target.get("schema_version") != "robot-target-10-audit-v1":
        raise SystemExit("ROBOT-TARGET-10 input required")
    rows = []
    for row in target.get("sides", []):
        metrics = row.get("metrics", {})
        retention = metrics.get("actual_target_motion_retention")
        tracking_error = metrics.get("target_tracking_large_error_ratio")
        away = metrics.get("persistent_away_frame_ratio")
        observed = metrics.get("status") == "MEASURED"
        eligibility = {
            "observed": observed,
            "motion_retention_ge_0p5": retention is not None and retention >= 0.5,
            "target_tracking_large_error_ratio_le_0p2": (
                tracking_error is not None and tracking_error <= 0.2
            ),
            "persistent_away_le_0p2": away is not None and away <= 0.2,
        }
        rows.append({"side": row["side"], "eligibility": eligibility, "passed": all(eligibility.values())})
    hard_target = target.get("hard_target_contract_pass") is True
    visual_eligible = bool(hard_target and rows and all(row["passed"] for row in rows))
    payload = {
        "schema_version": "robot-reach-20-audit-v1",
        "artifact_revision": "R7_ROBOT_5_REACH_20",
        "created_at": now_iso(),
        "task_id": "ROBOT-REACH-20",
        "task": target.get("task"),
        "session": target.get("session"),
        "evidence_class": "DIAGNOSTIC_PROXY",
        "object6d_consumed": False,
        "workspace_bounds_consumed": False,
        "terminal_status": "PASSED" if hard_target else "FAILED_QUALITY_C",
        "robot_tier": "NONE",
        "candidate_robot_tier": "POSE_ONLY_VISUAL" if visual_eligible else "NONE",
        "hard_geometry_pass": False,
        "hard_target_contract_pass": hard_target,
        "soft_pose_similarity_pass": target.get("soft_pose_similarity_pass") is True,
        "visual_train_eligible": visual_eligible,
        "metric_contact_eligible": False,
        "side_gates": rows,
        "input": artifact_ref(args.target_audit),
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "Motion-retention and target-tracking eligibility only. The >10 mm ratio is not "
            "a workspace-bound or clipping measurement. Hard geometry remains false until "
            "collision QA; no authority."
        ),
    }
    atomic_new_json(args.output.resolve(), payload)
    return 0 if hard_target else 2


if __name__ == "__main__":
    raise SystemExit(main())
