#!/usr/bin/env python3
"""ROBOT-PROFILE-30: summarize immutable Robot phase timings and throughput."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.robot_target_reach_v75 import (  # noqa: E402
    artifact_ref,
    atomic_new_json,
    load_json,
    now_iso,
    percentile,
)


PHASES = ("arm_round1", "arm_placement", "arm_round2", "hand_round1", "hand_round2", "render")


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    phase_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in PHASES}
    inputs = []
    for batch in sorted(path for path in root.glob("batch_*") if path.is_dir()):
        for phase in PHASES:
            result_path = batch / phase / "RESULT.json"
            if not result_path.is_file():
                continue
            inputs.append(artifact_ref(result_path))
            result = load_json(result_path)
            for row in result.get("sessions", []):
                start, finish = row.get("started_at"), row.get("finished_at")
                duration = None
                if isinstance(start, str) and isinstance(finish, str):
                    duration = max(0.0, timestamp(finish) - timestamp(start))
                phase_rows[phase].append(
                    {
                        "batch": batch.name,
                        "session": row.get("session"),
                        "status": row.get("status"),
                        "returncode": row.get("returncode"),
                        "duration_s": duration,
                        "frame_count": row.get("frame_count"),
                    }
                )
    summaries = {}
    for phase, rows in phase_rows.items():
        durations = [row["duration_s"] for row in rows if row["duration_s"] is not None]
        frames = sum(int(row["frame_count"]) for row in rows if isinstance(row.get("frame_count"), int))
        worker_seconds = sum(durations)
        summaries[phase] = {
            "rows": len(rows),
            "duration_s_p50": percentile(durations, 50),
            "duration_s_p95": percentile(durations, 95),
            "aggregate_worker_seconds": worker_seconds,
            "frames": frames,
            "frames_per_worker_hour": None if not worker_seconds else frames * 3600.0 / worker_seconds,
            "statuses": {status: sum(row.get("status") == status for row in rows) for status in sorted({str(row.get("status")) for row in rows})},
        }
    payload = {
        "schema_version": "robot-profile-30-v1",
        "artifact_revision": "R7_ROBOT_5_PROFILE_30",
        "created_at": now_iso(),
        "task_id": "ROBOT-PROFILE-30",
        "terminal_status": "PASSED" if inputs else "BLOCKED_PREREQ",
        "robot_tier": "NONE",
        "phases": summaries,
        "inputs": inputs,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Recorded worker timing and diagnostic throughput only; not a wall-clock completion promise or Robot authority.",
    }
    atomic_new_json(args.output.resolve(), payload)
    return 0 if inputs else 2


if __name__ == "__main__":
    raise SystemExit(main())
