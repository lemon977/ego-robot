#!/usr/bin/env python3
"""Build the truthful 156-row R1 Robot Geometry terminal matrix.

Legacy v5.2 videos are indexed as candidates only.  They do not satisfy the
new V7.1 one-scene-z-buffer/self-collision audit and are never silently
promoted to Robot authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONVERSION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"
FAILURES = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_robot_failure_clusters_v1/ROBOT_FAILURE_CLUSTERS.json"
POSE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/pose_only_visual_robot_v1/CURRENT_CANDIDATE_INDEX.json"
STRICT_RESULTS = (
    ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/robot_visual_candidates_v1/batch_023_singleton_patch_v2/RESULT.json",
    ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/robot_visual_candidates_v1/batch_039_current_v2/RESULT.json",
    ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/robot_visual_candidates_v1/ready_three_v1/RESULT.json",
)
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def legacy_candidates(pose_path: Path, strict_results: tuple[Path, ...]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    for row in pose["rows"]:
        result[row["session_id"]] = {
            "mode": "POSE_ONLY_VISUAL_ROBOT",
            "evidence": row["result"],
            "legacy_status": row["status"],
        }
    for path in strict_results:
        value = json.loads(path.read_text(encoding="utf-8"))
        for row in value["sessions"]:
            if row.get("status") == "PASS_RENDER_READY_FOR_HUMAN_REVIEW":
                result[row["session"]] = {
                    "mode": "LEGACY_STRICT_NUMERIC_REVIEW_CANDIDATE",
                    "evidence": row["result"],
                    "legacy_status": row["status"],
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", type=Path, default=CONVERSION)
    parser.add_argument("--failures", type=Path, default=FAILURES)
    parser.add_argument("--pose-index", type=Path, default=POSE)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    conversion_path = args.conversion.resolve(strict=True)
    failures_path = args.failures.resolve(strict=True)
    pose_path = args.pose_index.resolve(strict=True)
    strict_results = tuple(path.resolve(strict=True) for path in STRICT_RESULTS)
    output_root = args.output_root.resolve()
    targets = (
        output_root / "ROBOT_GEOMETRY_TERMINAL_MATRIX.json",
        output_root / "ROBOT_TERMINAL_MATRIX.json",
        output_root / "ROBOT_GEOMETRY_TERMINAL_MATRIX.csv",
        output_root / "RESULT.json",
    )
    existing = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise SystemExit(f"no-clobber: output artifacts already exist: {existing}")
    cohort = json.loads(conversion_path.read_text(encoding="utf-8"))["rows"]
    failures = json.loads(failures_path.read_text(encoding="utf-8"))["rows"]
    failed = {row["session"]: row for row in failures}
    legacy = legacy_candidates(pose_path, strict_results)
    rows = []
    for source in cohort:
        session = source["session_id"]
        if session in failed:
            state = "FAILED_QUALITY_C"
            blocker = failed[session]["reason"]
            evidence = failed[session]["result"]
        else:
            state = "BLOCKED_PREREQ"
            blocker = "V71_UNIFIED_ZBUFFER_AND_SELF_COLLISION_AUDIT_NOT_RUN"
            evidence = legacy.get(session, {}).get("evidence")
        rows.append({
            "session_id": session,
            "task": source["task"],
            "status": state,
            "robot_mode": None,
            "primary_blocker": blocker,
            "legacy_candidate_available": session in legacy,
            "legacy_candidate_mode": legacy.get(session, {}).get("mode"),
            "evidence": evidence,
            "v71_unified_zbuffer_pass": False,
            "v71_self_collision_pass": False,
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
        })
    if len(rows) != 156 or len({r["session_id"] for r in rows}) != 156:
        raise RuntimeError("Robot terminal denominator must be 156 unique sessions")
    counts = {key: sum(r["status"] == key for r in rows) for key in (
        "METRIC_CONTACT_ROBOT", "POSE_ONLY_VISUAL_ROBOT", "FAILED_QUALITY_C",
        "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_EXTERNAL"
    )}
    payload = {
        "schema_version": "ROBOT_GEOMETRY_TERMINAL_MATRIX_V71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "TERMINAL_MATRIX_COMPLETE_NO_ROBOT_AUTHORITY",
        "counts": counts,
        "legacy_candidate_count": sum(r["legacy_candidate_available"] for r in rows),
        "rows": rows,
        "inputs": [ref(conversion_path), ref(failures_path), ref(pose_path), *(ref(p) for p in strict_results)],
        "authority": False,
        "claim_limit": "156-row finite terminal accounting. Legacy review candidates are not V7.1 Robot Geometry passes, control truth, metric contact, or deployment authority.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    matrix_data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (output_root / "ROBOT_GEOMETRY_TERMINAL_MATRIX.json").write_text(matrix_data, encoding="utf-8")
    # V7.1 final-name alias. It is byte-identical and retains the explicit
    # zero-authority claim; this is a matrix, not a promotion.
    (output_root / "ROBOT_TERMINAL_MATRIX.json").write_text(matrix_data, encoding="utf-8")
    fields = ["session_id", "task", "status", "robot_mode", "primary_blocker", "legacy_candidate_available", "legacy_candidate_mode", "control_ground_truth"]
    with (output_root / "ROBOT_GEOMETRY_TERMINAL_MATRIX.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in rows)
    result = {k: payload[k] for k in ("schema_version", "plan_revision", "artifact_revision", "generated_at", "status", "counts", "legacy_candidate_count", "authority", "claim_limit")}
    result["schema_version"] = "ROBOT_GEOMETRY_TERMINAL_MATRIX_V71_RESULT"
    result["matrix"] = ref(output_root / "ROBOT_GEOMETRY_TERMINAL_MATRIX.json")
    (output_root / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "legacy_candidate_count": payload["legacy_candidate_count"]}))


if __name__ == "__main__":
    main()
