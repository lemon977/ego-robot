#!/usr/bin/env python3
"""Adapt an immutable arm-only quality-C receipt for a watermarked review renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_ref(path: Path) -> dict:
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_compatibility(source: dict, source_ref: dict) -> dict:
    if source.get("status") != "FAILED_QUALITY_C":
        raise ValueError("only an arm-only quality-C source can enter this review adapter")
    session = source.get("session_id")
    n = source.get("frame_count")
    rows = source.get("pose_rows", [])
    if not isinstance(session, str) or not isinstance(n, int) or len(rows) != 2 * n:
        raise ValueError("session, frame count or per-side pose rows incomplete")
    failed_frames = sorted({int(row["frame"]) for row in rows if not row["target_pose_gate"]})
    if not failed_frames or failed_frames[0] < 0 or failed_frames[-1] >= n:
        raise ValueError("quality-C source must have valid failed-target frames")
    return {
        "schema_version": "robot-arm-only-research-render-compat-v1",
        "status": "HOLD_NUMERIC_CANARY",
        "research_source_status": "FAILED_QUALITY_C",
        "session": session,
        "frame_count": n,
        "metrics": source["metrics"],
        "failed_target_frame_ids": failed_frames,
        "source_result": source_ref,
        "compatibility_only": True,
        "authority": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Only a truthful renderer interface for a quality-C arm-only result; never a numeric pass or full Robot collision receipt.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.arm_result.read_text(encoding="utf-8"))
    payload = build_compatibility(source, file_ref(args.arm_result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failed_frames": len(payload["failed_target_frame_ids"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
