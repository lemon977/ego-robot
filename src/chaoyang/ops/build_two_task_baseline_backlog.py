#!/usr/bin/env python3
"""Build a read-only compatibility/backlog index for the other exact78 sessions.

Usage from the repository root::

    python -m tools.build_two_task_baseline_backlog

This does not decode, process, publish, or mark any session training eligible.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260917_two_task_visual_baseline_v1/backlog/attempt_0001"
PICKED = {"get_potato_chips_0902_103", "play_cards_0902_042"}
FIELDS = ["position", "task", "session_id", "frame_count", "date", "split", "stereo_calibration_state", "compatibility", "next_action", "reason_code", "immutable_input_identity_sha256"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"immutable output mismatch: {path}")
        return
    with path.open("xb") as f:
        f.write(data)


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("counts", {}).get("sessions") != 156 or len(source.get("sessions", [])) != 156:
        raise ValueError("exact78 cohort cardinality changed")
    ids = {row["session_id"] for row in source["sessions"]}
    if not PICKED <= ids:
        raise ValueError("two frozen sessions not in cohort")
    rows = []
    for row in source["sessions"]:
        if row["session_id"] in PICKED:
            continue
        calibration = row.get("stereo_calibration_state") or "UNKNOWN"
        allowed = set(row.get("allowed_stages") or [])
        if "DEPTH" in allowed and calibration == "CALIBRATED_METRIC_STEREO":
            compatibility = "CONTRACT_COMPATIBLE_UNVERIFIED"
            next_action = "VERIFY_INPUT_SHA_AND_STAGE_RECEIPTS_BEFORE_BATCH"
            reason = "NOT_RECOMPUTED_THIS_RUN"
        else:
            compatibility = "PARTIAL_OR_BLOCKED_REVIEW"
            next_action = "RESOLVE_FROZEN_CALIBRATION_OR_STAGE_GATE"
            reason = row.get("terminal_code") or "DEPTH_NOT_ALLOWED_OR_CALIBRATION_UNKNOWN"
        rows.append({
            "position": row["position"],
            "task": row["task"],
            "session_id": row["session_id"],
            "frame_count": row["frame_count"],
            "date": row.get("date", ""),
            "split": row.get("split", ""),
            "stereo_calibration_state": calibration,
            "compatibility": compatibility,
            "next_action": next_action,
            "reason_code": reason,
            "immutable_input_identity_sha256": row.get("immutable_input_identity_sha256", ""),
        })
    if len(rows) != 154 or len({r["session_id"] for r in rows}) != 154:
        raise ValueError("backlog cardinality/identity changed")
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_once(OUT / "BACKLOG_154.csv", buffer.getvalue().encode("utf-8"))
    summary = {
        "schema_version": "two-task-visual-backlog-v1",
        "route_id": "chips_poker_exact78_offline_visual_v1",
        "source_manifest": {"path": str(SOURCE), "sha256": digest(SOURCE), "bytes": SOURCE.stat().st_size},
        "excluded_baseline_sessions": sorted(PICKED),
        "pending_count": len(rows),
        "by_task": {task: sum(r["task"] == task for r in rows) for task in ("chips", "poker")},
        "by_compatibility": {state: sum(r["compatibility"] == state for r in rows) for state in sorted({r["compatibility"] for r in rows})},
        "csv": {"path": str(OUT / "BACKLOG_154.csv"), "sha256": digest(OUT / "BACKLOG_154.csv")},
        "claim_limit": "Manifest-only compatibility screen; no frame, stage, calibration quality, causal, training, or Robot pass verified here.",
    }
    write_once(OUT / "RESULT.json", (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
