#!/usr/bin/env python3
"""Build a split-aware H50 eligibility index from one frozen Robot matrix."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))


import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.human_ego.tools.preflight_visual_aux_h50_candidates_v55 import assess  # noqa: E402
from chaoyang.ops import render_poker_symmetric_chirality_flange_successor as robot  # noqa: E402


MIN_TRAIN_SESSIONS = 16
MIN_VALIDATION_SESSIONS = 3
MIN_TRAIN_WINDOWS = 256
MIN_VALIDATION_WINDOWS = 48


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize(rows: list[dict[str, Any]], task: str, split: str) -> dict[str, int]:
    selected = [row for row in rows if row["task"] == task and row["split"] == split]
    ready = [row for row in selected if row["status"] == "READY"]
    return {
        "candidates": len(selected),
        "ready_sessions": len(ready),
        "eligible_h50_windows": sum(int(row["eligible_h50_window_count"]) for row in ready),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--h50-eligibility-mode",
        choices=("BOTH_ENDPOINTS_40_OF_50", "ANY_ENDPOINT_40_OF_50"),
        default="BOTH_ENDPOINTS_40_OF_50",
    )
    args = parser.parse_args()
    matrix_path = args.matrix.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"immutable output exists: {output}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assets = robot.load_pinned_robot_assets(PROJECT)
    rows = []
    for source_row in matrix["rows"]:
        reference = source_row.get("robot_candidate_result")
        if not isinstance(reference, dict) or source_row.get("robot_current_state") != "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY":
            continue
        result_path = Path(reference["path"]).resolve(strict=True)
        if result_path.stat().st_size != reference["bytes"] or sha256(result_path) != reference["sha256"]:
            raise RuntimeError(f"Robot result reference mismatch: {result_path}")
        row = assess(result_path, assets, args.h50_eligibility_mode)
        row["split"] = source_row["split"]
        rows.append(row)
    summaries = {
        task: {split: summarize(rows, task, split) for split in ("train", "validation", "test", "heldout")}
        for task in ("chips", "poker")
    }
    readiness = {}
    for task in ("chips", "poker"):
        train = summaries[task]["train"]
        validation = summaries[task]["validation"]
        readiness[task] = {
            "ready": (
                train["ready_sessions"] >= MIN_TRAIN_SESSIONS
                and validation["ready_sessions"] >= MIN_VALIDATION_SESSIONS
                and train["eligible_h50_windows"] >= MIN_TRAIN_WINDOWS
                and validation["eligible_h50_windows"] >= MIN_VALIDATION_WINDOWS
            ),
            "required": {
                "train_sessions": MIN_TRAIN_SESSIONS,
                "validation_sessions": MIN_VALIDATION_SESSIONS,
                "train_windows": MIN_TRAIN_WINDOWS,
                "validation_windows": MIN_VALIDATION_WINDOWS,
            },
        }
    atomic_json(
        output,
        {
            "schema_version": "exact78-visual-aux-eligibility-index-v56-v1",
            "status": "PASS_ELIGIBILITY_INDEX_BUILT",
            "source_matrix": artifact_ref(matrix_path),
            "rows": rows,
            "summaries": summaries,
            "training_readiness": readiness,
            "h50_eligibility_mode": args.h50_eligibility_mode,
            "claim_limit": "Projection and fixed-split eligibility only; no paired bundle, checkpoint, Robot action, contact, or physical authority.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
