#!/usr/bin/env python3
"""Close four Visual Aux branches truthfully when frozen data minima fail."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_eligibility_index_v56/ELIGIBILITY_INDEX.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the bounded Visual Aux checkpoint eligibility index."
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    eligibility = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    required = {"train_sessions": 16, "validation_sessions": 3, "train_windows": 256, "validation_windows": 48}
    for task in ("chips", "poker"):
        summary = eligibility["summaries"][task]
        observed = {
            "train_sessions": summary["train"]["ready_sessions"],
            "validation_sessions": summary["validation"]["ready_sessions"],
            "train_windows": summary["train"]["eligible_h50_windows"],
            "validation_windows": summary["validation"]["eligible_h50_windows"],
        }
        for visual_input in ("HUMAN_RAW_RGB", "ROBOTIZED_RGB"):
            rows.append({
                "task": task,
                "visual_input": visual_input,
                "pair_id": f"CHECKPOINT_PAIR_{task.upper()}",
                "status": "BLOCKED_DATA_VOLUME",
                "observed": observed,
                "required": required,
                "checkpoint": None,
                "loss_curve": None,
                "metrics": None,
                "control_ground_truth": False,
                "primary_blocker": "FROZEN_TRAIN_VALIDATION_SESSION_AND_H50_MINIMA_NOT_MET",
            })
    payload = {
        "schema_version": "VISUAL_AUX_CHECKPOINT_INDEX_V71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "FULL_BLOCKED",
        "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": sha(source)},
        "pair_status": {"chips": "BLOCKED_DATA_VOLUME", "poker": "BLOCKED_DATA_VOLUME"},
        "rows": rows,
        "counts": {"requested_checkpoints": 4, "published_checkpoints": 0, "blocked_data_volume": 4},
        "split_movement_allowed": False,
        "future_information_allowed": False,
        "resume_condition": "Add causal paired Robotized candidates within the already-frozen split until each task independently reaches all four minima.",
        "claim_limit": "Eligibility closure only. No checkpoint, loss curve, Robot action, statistical comparison, or policy authority exists.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "VISUAL_AUX_CHECKPOINT_INDEX.json"
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != data:
        raise RuntimeError(f"no-clobber conflict: {target}")
    if not target.exists():
        target.write_text(data, encoding="utf-8")
    with (output_root / "VISUAL_AUX_CHECKPOINT_INDEX.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "visual_input", "pair_id", "status", "primary_blocker"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    result = dict(payload)
    result["schema_version"] = "VISUAL_AUX_CHECKPOINT_INDEX_V71_RESULT"
    (output_root / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"]))


if __name__ == "__main__":
    main()
