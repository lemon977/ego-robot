#!/usr/bin/env python3
"""Wait for the Robot hard/soft audit and publish a bounded conversion report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
STATUS_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def exact(item: dict[str, Any]) -> Path:
    path = Path(str(item["path"])).resolve(strict=True)
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError("hard/soft index reference mismatch")
    return path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def classify(row: dict[str, Any]) -> str:
    if row.get("strict_pose_match") is True:
        return "STRICT_POSE_MATCH"
    if row.get("hard_geometry_pass") is True:
        arm = row.get("soft_gates", {}).get("arm_pose_all_observed") is True
        hand = row.get("soft_gates", {}).get("hand_anatomy_all_observed") is True
        if not arm and not hand:
            return "HARD_PASS_SOFT_ARM_AND_HAND_MISMATCH"
        if not arm:
            return "HARD_PASS_SOFT_ARM_MISMATCH"
        if not hand:
            return "HARD_PASS_SOFT_HAND_MISMATCH"
        return "HARD_PASS_OTHER_SOFT_DIAGNOSTIC"
    return "HARD_GEOMETRY_FAIL"


def summarize(index: dict[str, Any], matrix: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_rows = {row["session_id"]: row for row in matrix.get("rows", [])}
    output_rows = []
    for row in index.get("rows", []):
        session = str(row["session"])
        source = matrix_rows.get(session)
        if source is None:
            raise RuntimeError(f"session absent from matrix: {session}")
        output_rows.append({
            "session_id": session,
            "task": source["task"],
            "split": source["split"],
            "strict_pose_match": bool(row.get("strict_pose_match")),
            "hard_geometry_pass": bool(row.get("hard_geometry_pass")),
            "conversion_diagnosis": classify(row),
            "visual_aux_candidate": bool(row.get("hard_geometry_pass")),
            "authority": False,
        })
    counts = Counter(row["conversion_diagnosis"] for row in output_rows)
    by_task = {
        task: dict(Counter(row["conversion_diagnosis"] for row in output_rows if row["task"] == task))
        for task in ("chips", "poker")
    }
    strict_failures = sum(not row["strict_pose_match"] for row in output_rows)
    hard_recovered = sum(
        not row["strict_pose_match"] and row["hard_geometry_pass"] for row in output_rows
    )
    summary = {
        "sessions": len(output_rows),
        "strict_pose_match": sum(row["strict_pose_match"] for row in output_rows),
        "hard_geometry_pass": sum(row["hard_geometry_pass"] for row in output_rows),
        "strict_failure_hard_feasible": hard_recovered,
        "strict_failure_count": strict_failures,
        "strict_failure_hard_feasible_ratio": hard_recovered / strict_failures if strict_failures else None,
        "by_diagnosis": dict(counts),
        "by_task": by_task,
    }
    return output_rows, summary


def register_claim(result: Path) -> None:
    for _ in range(12):
        revision = int(load(STATUS_RECEIPT)["governance_revision"])
        completed = subprocess.run(
            [
                sys.executable, "-m", "chaoyang.governance.register_claim",
                "--claim", "Robot strict-pose failures have a separately measured hard-geometry outcome",
                "--status", "DEVELOPMENT_EVIDENCE",
                "--scope", "exact78/robot/hard_soft_conversion_R7_3",
                "--claim-limit", "Digital URDF hard/soft conversion diagnosis only; no Robot, Contact, control or physical authority.",
                "--evidence", str(result.resolve()),
                "--expected-revision", str(revision),
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if completed.returncode == 0:
            return
        if "revision" not in (completed.stdout + completed.stderr).lower():
            raise RuntimeError((completed.stdout + completed.stderr)[-3000:])
        time.sleep(1)
    raise RuntimeError("governance CAS retry exhausted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-soft-state", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "AUTOMATION_STATE.json"
    if (output / "RESULT.json").is_file():
        return 0
    started = time.monotonic()
    while True:
        upstream = load(args.hard_soft_state) if args.hard_soft_state.is_file() else {}
        state = {
            "schema_version": "robot-conversion-diagnosis-watcher-v71-v1",
            "status": "WAITING_UPSTREAM",
            "pid": os.getpid(),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "upstream_status": upstream.get("status", "MISSING"),
            "claim_limit": "Conversion diagnosis only; no Robot, Contact, visual, control or physical authority.",
        }
        atomic_json(state_path, state)
        if upstream.get("status") == "TERMINAL":
            index_path = exact(upstream["result"])
            rows, summary = summarize(load(index_path), load(args.matrix))
            result = {
                "schema_version": "robot-hard-soft-conversion-report-v71-v1",
                "status": "PASSED_DEVELOPMENT_CONVERSION_DIAGNOSIS",
                "created_at": state["updated_at"],
                "summary": summary,
                "rows": rows,
                "inputs": {"hard_soft_index": ref(index_path), "matrix": ref(args.matrix)},
                "authority": False,
                "claim_limit": state["claim_limit"],
            }
            result_path = output / "RESULT.json"
            atomic_json(result_path, result)
            csv_path = output / "ROBOT_HARD_SOFT_CONVERSION_REPORT.csv"
            with csv_path.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["session_id"])
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            register_claim(result_path)
            state.update(status="TERMINAL", result=ref(result_path), csv=ref(csv_path))
            atomic_json(state_path, state)
            return 0
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="UPSTREAM_WAIT_BUDGET_EXHAUSTED")
            atomic_json(state_path, state)
            return 3
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
