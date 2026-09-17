#!/usr/bin/env python3
"""Audit frozen PICO c2w bindings without claiming the producer is causal.

The first physical line of trackingData is a merged-session header, not a
pose sample.  ``tracking_index`` addresses the subsequent data rows.  A
metadata timestamp identical to a tracker timestamp is not independent proof
of the RGB sensor exposure time.  This audit never grants Robot eligibility.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


def inspect_frame(*, metadata: dict[str, Any], tracking: dict[str, Any], c2w: np.ndarray) -> dict[str, Any]:
    index = int(metadata["tracking_index"])
    frame_ns = int(metadata["ts"])
    tracking_ns = int(tracking["timeStampNs"])
    pose = np.asarray(metadata["c2w"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("invalid metadata c2w")
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError("invalid archive c2w")
    return {
        "frame_id": int(metadata["idx"]),
        "tracking_index": index,
        "frame_timestamp_ns": frame_ns,
        "indexed_tracking_timestamp_ns": tracking_ns,
        "indexed_tracking_minus_frame_ms": (tracking_ns - frame_ns) / 1e6,
        "indexed_tracking_is_past": tracking_ns <= frame_ns,
        "metadata_timestamp_equals_indexed_tracking": tracking_ns == frame_ns,
        "metadata_vs_archive_c2w_max_abs": float(np.max(np.abs(pose - c2w))),
        "stored_sync_error_ms": float(metadata["tracking_sync_error_ms"]),
        "stored_sync_error_minus_indexed_delta_ms": float(metadata["tracking_sync_error_ms"]) - (tracking_ns - frame_ns) / 1e6,
        "tracker_state": str(tracking.get("TrackerState", "UNKNOWN")),
        "head_status": tracking.get("Head", {}).get("status"),
        "pose_segment_id": tracking.get("_merge", {}).get("poseSegmentId"),
        "source_session": tracking.get("_merge", {}).get("sourceSession"),
        "source_record_index": tracking.get("_merge", {}).get("sourceRecordIndex"),
    }


def pose_tracking_rows(physical_lines: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove merge headers and reject unrecognized data rows."""
    rows = [line for line in physical_lines if isinstance(line.get("Head"), dict) and "pose" in line["Head"]]
    skipped = len(physical_lines) - len(rows)
    if not rows or skipped != 1:
        raise ValueError(f"expected one merged header and pose samples, got skipped={skipped}")
    return rows, skipped


def nominal_grid_delta_ms(timestamp_ns: int, first_timestamp_ns: int, frame_id: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return (timestamp_ns - first_timestamp_ns - frame_id * 1e9 / fps) / 1e6


def audit(*, raw_root: Path, hawor_npz: Path) -> dict[str, Any]:
    manifest = json.loads((raw_root / "clip_manifest.json").read_text(encoding="utf-8"))
    tracking_path = raw_root / manifest["files"]["tracking"]
    with tracking_path.open(encoding="utf-8") as handle:
        physical_lines = [json.loads(line) for line in handle if line.strip()]
    tracking, skipped_headers = pose_tracking_rows(physical_lines)
    with np.load(hawor_npz, allow_pickle=False) as archive:
        poses = np.asarray(archive["c2w"], dtype=np.float64)
    frames = int(manifest["video"]["frame_count"])
    if len(poses) != frames:
        raise ValueError("archive/video frame count mismatch")

    rows = []
    for frame in range(frames):
        path = raw_root / "preprocess/all_data" / f"{frame:05d}" / "training_data.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        if int(metadata["idx"]) != frame:
            raise ValueError("metadata frame identity mismatch")
        index = int(metadata["tracking_index"])
        if not 0 <= index < len(tracking):
            raise ValueError("tracking index out of bounds")
        rows.append(inspect_frame(metadata=metadata, tracking=tracking[index], c2w=poses[frame]))

    deltas = np.asarray([row["indexed_tracking_minus_frame_ms"] for row in rows])
    max_closure = max(row["metadata_vs_archive_c2w_max_abs"] for row in rows)
    future = [row["frame_id"] for row in rows if not row["indexed_tracking_is_past"]]
    equal = sum(row["metadata_timestamp_equals_indexed_tracking"] for row in rows)
    sync_mismatch = [abs(row["stored_sync_error_minus_indexed_delta_ms"]) for row in rows]
    nominal_mismatch = [
        abs(row["stored_sync_error_ms"] - nominal_grid_delta_ms(
            row["frame_timestamp_ns"], rows[0]["frame_timestamp_ns"], row["frame_id"], float(manifest["video"]["fps"])
        )) for row in rows
    ]
    state_counts = {value: sum(row["tracker_state"] == value for row in rows) for value in sorted({row["tracker_state"] for row in rows})}
    head_status_counts = {str(value): sum(row["head_status"] == value for row in rows) for value in sorted({row["head_status"] for row in rows}, key=str)}
    return {
        "schema_version": "chaoyang-rc1-pico-c2w-causality-audit-v4",
        "created_at": now_iso(),
        "status": "PASSED_TIMESTAMP_DIAGNOSTIC" if max_closure <= 1e-9 else "FAILED_METADATA_CLOSURE",
        "session": manifest["clip_name"],
        "frames": frames,
        "tracking_file_header_rows_excluded": skipped_headers,
        "metadata_timestamp_equals_indexed_tracking_frames": equal,
        "stored_sync_error_minus_indexed_delta_abs_max_ms": max(sync_mismatch),
        "stored_sync_error_minus_nominal_grid_abs_max_ms": max(nominal_mismatch),
        "stored_sync_error_observed_semantics": (
            "TRACKER_TIMESTAMP_MINUS_NOMINAL_FPS_GRID"
            if max(nominal_mismatch) <= 0.001 else "UNKNOWN"
        ),
        "mapped_tracker_state_counts": state_counts,
        "mapped_head_status_counts": head_status_counts,
        "mapped_not_accurate_frames": sum(row["tracker_state"] != "accurate" for row in rows),
        "original_rgb_exposure_timestamp_proven": False,
        "indexed_future_frame_count": len(future),
        "indexed_future_frame_ids": future[:20],
        "indexed_tracking_minus_frame_ms": {
            "min": float(deltas.min()), "median": float(np.median(deltas)), "max": float(deltas.max()),
        },
        "metadata_vs_archive_c2w_max_abs": max_closure,
        "camera_pose_producer_causality": "UNKNOWN_VERIFICATION_REQUIRED",
        "robot_causal_eligible": False,
        "training_eligible": False,
        "authority_promoted": False,
        "claim_limit": "Indexed pose-row timestamps and stored metadata timestamps are compared after excluding the merge header. The stored sync error numerically matches tracker timestamp minus a nominal FPS grid, not independently measured RGB exposure-to-tracker skew. Head.status=3 does not override TrackerState=notAccurate. Producer suffix-invariance, interpolation and original RGB-to-tracker alignment remain unproven.",
        "inputs": {
            "clip_manifest": artifact_ref(raw_root / "clip_manifest.json"),
            "raw_tracking": artifact_ref(tracking_path),
            "hawor_npz": artifact_ref(hawor_npz),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--hawor-npz", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output required: {args.output_root}")
    result = audit(raw_root=args.raw_root.resolve(strict=True), hawor_npz=args.hawor_npz.resolve(strict=True))
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in (
        "status", "session", "frames", "indexed_future_frame_count",
        "metadata_vs_archive_c2w_max_abs", "camera_pose_producer_causality",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
