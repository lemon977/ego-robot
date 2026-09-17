#!/usr/bin/env python3
"""Publish the 156-row V7.1 exact78 terminal matrix after Clean closes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CURRENT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
STATUS_MIN = ROOT / "docs/governance/CURRENT_PROJECT_STATUS_MIN.json"
CONVERSION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"
ROBOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/ROBOT_TERMINAL_MATRIX.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/exact78_terminal"
CLEAN_PASS_STATES = {"CLEAN_B", "PASSED_GRADE_B", "PASSED"}
CLEAN_TERMINAL_STATES = CLEAN_PASS_STATES | {
    "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"
}


def ref(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def assert_clean_fact_ledger_closed(status: dict) -> None:
    waves = status.get("waves", {})
    if int(waves.get("wave0_clean_pending", -1)) != 0:
        raise RuntimeError("fact ledger says Wave0 Clean is still pending")
    active = [
        row for row in status.get("active_tasks", [])
        if "clean" in str(row.get("task_id", "")).lower()
    ]
    if active:
        raise RuntimeError("fact ledger still contains an active Clean worker")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-matrix", type=Path, default=CURRENT)
    parser.add_argument("--status-min", type=Path, default=STATUS_MIN)
    parser.add_argument("--conversion", type=Path, default=CONVERSION)
    parser.add_argument("--robot-matrix", type=Path, default=ROBOT)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    current_path = args.current_matrix.resolve(strict=True)
    status_path = args.status_min.resolve(strict=True)
    conversion_path = args.conversion.resolve(strict=True)
    robot_path = args.robot_matrix.resolve(strict=True)
    output_root = args.output_root.resolve()
    targets = (
        output_root / "EXACT78_FINAL_TERMINAL_MATRIX.json",
        output_root / "EXACT78_FINAL_TERMINAL_MATRIX.csv",
        output_root / "RESULT.json",
    )
    existing = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise SystemExit(f"no-clobber: output artifacts already exist: {existing}")
    assert_clean_fact_ledger_closed(json.loads(status_path.read_text(encoding="utf-8")))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    conversion = {row["session_id"]: row for row in json.loads(conversion_path.read_text(encoding="utf-8"))["rows"]}
    robot = {row["session_id"]: row for row in json.loads(robot_path.read_text(encoding="utf-8"))["rows"]}
    rows = []
    for row in current["rows"]:
        sid = row["session_id"]
        conv = conversion[sid]
        clean = row["clean_state"]
        if row["metric_ready_wave0"] and clean not in CLEAN_TERMINAL_STATES:
            raise RuntimeError(f"Wave0 Clean remains non-terminal: {sid}={clean}")
        clean_result = row.get("clean_result")
        primary = conv["first_blocker"] if not row["metric_ready_wave0"] else (None if clean in CLEAN_PASS_STATES else clean)
        rows.append({
            "session_id": sid,
            "task": row["task"],
            "raw_status": "PASSED",
            "hawor_status": row["hawor"]["grade"],
            "role_mask_status": row["role_mask"]["grade"],
            "object_mask_status": row["object_mask"]["grade"],
            "calibration_status": "PASSED" if row["metric_ready_wave0"] else ("MISSING" if conv["first_blocker"] == "CALIBRATION_MISSING" else "NOT_REACHED"),
            "depth_status": "B" if row["metric_ready_wave0"] else "BLOCKED_PREREQ",
            "object6d_status": "B" if row["metric_ready_wave0"] else "BLOCKED_PREREQ",
            "clean_status": clean,
            "robot_status": robot[sid]["status"],
            "current_wave": "Wave0" if row["metric_ready_wave0"] else "none",
            "primary_blocker": primary,
            "downstream_scopes": (["DEPTH", "OBJECT6D", "CLEAN_JOIN_READY"] if row["metric_ready_wave0"] and clean in CLEAN_PASS_STATES else ["RAW_ONLY"]),
            "result_path": clean_result.get("path") if isinstance(clean_result, dict) else None,
            "result_sha": clean_result.get("sha256") if isinstance(clean_result, dict) else None,
            "superseded_history": [],
        })
    chips = sum(r["task"] == "chips" for r in rows)
    poker = sum(r["task"] == "poker" for r in rows)
    if len(rows) != 156 or len({r["session_id"] for r in rows}) != 156 or (chips, poker) != (78, 78):
        raise RuntimeError("exact78 denominator is not 78+78 unique")
    payload = {
        "schema_version": "EXACT78_FINAL_TERMINAL_MATRIX_V71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "EXACT78_TERMINAL_COMPLETE",
        "counts": {"total": 156, "chips": chips, "poker": poker, "wave0": sum(r["current_wave"] == "Wave0" for r in rows)},
        "rows": rows,
        "inputs": [ref(current_path), ref(conversion_path), ref(robot_path)],
        "claim_limit": "Cross-stage finite terminal accounting only; it does not mean every stage passed or exact78 end-to-end completion.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "EXACT78_FINAL_TERMINAL_MATRIX.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = list(rows[0].keys())
    with (output_root / "EXACT78_FINAL_TERMINAL_MATRIX.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            copy = dict(row); copy["downstream_scopes"] = ";".join(copy["downstream_scopes"]); copy["superseded_history"] = ""
            writer.writerow(copy)
    result = {
        "schema_version": "EXACT78_FINAL_TERMINAL_MATRIX_V71_RESULT",
        "status": payload["status"],
        "counts": payload["counts"],
        "matrix": ref(target),
        "claim_limit": payload["claim_limit"],
    }
    (output_root / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
