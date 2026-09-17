#!/usr/bin/env python3
"""Audit a full CPU-frame SAM3.1 old-B diagnostic without claiming Gold quality."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, validate_artifact_ref


def closed(ref: dict) -> Path:
    errors = validate_artifact_ref(ref)
    if errors:
        raise ValueError("artifact closure failed: " + "; ".join(errors))
    return Path(ref["path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("fresh audit output required")
    worker = json.loads(args.worker_result.read_text(encoding="utf-8"))
    session = worker["session_id"]
    frames = worker["frames"]
    if session not in {"play_cards_0901_001", "play_cards_0901_005"} or frames not in {645, 520}:
        raise ValueError("unexpected frozen session/frame identity")
    if worker.get("resource_mode", {}).get("offload_video_to_cpu") is not True:
        raise ValueError("not a CPU-frame storage experiment")
    if worker.get("quality_gate_status") != "NOT_EVALUATED":
        raise ValueError("worker incorrectly claimed quality pass")
    for ref in worker["inputs"].values():
        closed(ref)
    closed(worker["code"])
    video = closed(worker["outputs"]["review_video"])
    packed = closed(worker["outputs"]["packed_masks"])
    with np.load(packed) as value:
        ids = value["frame_ids"]
        bits = value["packed"]
    if ids.shape != (frames,) or not np.array_equal(ids, np.arange(frames)):
        raise ValueError("frame IDs are not complete and ordered")
    if bits.shape[0] != frames or bits.shape[1] != (960 * 1280 // 8):
        raise ValueError("packed mask shape mismatch")
    areas = np.unpackbits(bits, axis=1).sum(axis=1).astype(int)
    observed = worker["comparisons"]
    if len(observed) != worker["frozen_B_observed_frames"]:
        raise ValueError("old-B comparison denominator mismatch")
    missed = [row["frame_id"] for row in observed if not row["native_present"]]
    jumps = sorted(
        ((float(round(max(a, b) / min(a, b), 4)), int(i)) for i, (a, b) in
         enumerate(zip(areas[:-1], areas[1:]), start=1) if a and b),
        reverse=True,
    )[:10]
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(video),
    ], text=True, timeout=90))["streams"][0]
    if int(probe["nb_read_frames"]) != frames:
        raise ValueError("review video frame count mismatch")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(video),
                    "-f", "null", "-"], check=True, timeout=300)
    status = "FAILED_QUALITY_C_MISSED_OLD_B" if missed else "HOLD_VISUAL_IDENTITY_AND_FACE_REVIEW"
    output = {
        "schema_version": "rc1-sam31-cpu-frame-full-b-audit-v1",
        "session_id": session,
        "frame_count": frames,
        "execution_status": "COMPLETED_DIAGNOSTIC",
        "quality_status": status,
        "native_present_frames": worker["native_present_frames"],
        "frozen_B_observed_frames": len(observed),
        "native_missed_old_B_observed_frames": missed,
        "internal_iou_median": worker["internal_iou_median"],
        "internal_iou_p05": worker["internal_iou_p05"],
        "top_area_jumps_ratio_and_frame": jumps,
        "uniform_review_frames": np.linspace(0, frames - 1, 12, dtype=int).astype(int).tolist(),
        "worker_result": artifact_ref(args.worker_result),
        "full_review_video": artifact_ref(video),
        "packed_masks": artifact_ref(packed),
        "authority_promoted": False,
        "training_eligible": False,
        "same_physical_instance_proven": False,
        "face_id_proven": False,
        "pixel_accuracy_proven": False,
        "claim_limit": "Exact full-frame execution and old-B internal comparison only. Card identity, front/back face, hand contamination and true pixel accuracy need independent visual review; old B is not Gold.",
    }
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / "RESULT.json", output)
    print(json.dumps({"session_id": session, "quality_status": status, "missed": len(missed),
                      "output": str(args.output_root / "RESULT.json")}))


if __name__ == "__main__":
    main()
