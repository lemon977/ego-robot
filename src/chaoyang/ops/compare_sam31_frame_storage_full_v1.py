#!/usr/bin/env python3
"""Compare two pinned full SAM3.1 runs differing only in frame-storage mode."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, validate_artifact_ref


def check_ref(ref: dict) -> Path:
    errors = validate_artifact_ref(ref)
    if errors:
        raise ValueError("artifact closure failed: " + "; ".join(errors))
    return Path(ref["path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-result", type=Path, required=True)
    parser.add_argument("--gpu-result", type=Path, required=True)
    parser.add_argument("--cpu-samples", type=Path)
    parser.add_argument("--gpu-samples", type=Path)
    parser.add_argument("--session-id", choices=("play_cards_0901_001", "play_cards_0901_005"),
                        default="play_cards_0901_005")
    parser.add_argument("--frame-count", type=int, choices=(645, 520), default=520)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("fresh comparison output required")
    cpu = json.loads(args.cpu_result.read_text(encoding="utf-8"))
    gpu = json.loads(args.gpu_result.read_text(encoding="utf-8"))
    if (cpu["session_id"], gpu["session_id"], cpu["frames"], gpu["frames"]) != (
        args.session_id, args.session_id, args.frame_count, args.frame_count
    ):
        raise ValueError("session/frame identity mismatch")
    if cpu["resource_mode"]["offload_video_to_cpu"] is not True or (
        gpu["resource_mode"]["offload_video_to_cpu"] is not False
    ):
        raise ValueError("only frame storage mode may differ")
    for label in ("old_B_result", "old_B_manifest", "old_B_seed_mask", "source_rgb_frame0",
                  "source_rgb_last_frame", "checkpoint"):
        if cpu["inputs"][label]["sha256"] != gpu["inputs"][label]["sha256"]:
            raise ValueError(f"frozen input changed: {label}")
    if cpu["code"]["sha256"] != gpu["code"]["sha256"]:
        raise ValueError("worker code changed between modes")
    paths = [check_ref(item["outputs"]["packed_masks"]) for item in (cpu, gpu)]
    with np.load(paths[0]) as data:
        cpu_ids, cpu_bits = data["frame_ids"], data["packed"]
    with np.load(paths[1]) as data:
        gpu_ids, gpu_bits = data["frame_ids"], data["packed"]
    if not np.array_equal(cpu_ids, np.arange(args.frame_count)) or not np.array_equal(gpu_ids, cpu_ids):
        raise ValueError("frame identity/order mismatch")
    if cpu_bits.shape != gpu_bits.shape or cpu_bits.shape[0] != args.frame_count:
        raise ValueError("packed mask shapes mismatch")
    equal_frames = np.all(cpu_bits == gpu_bits, axis=1)
    differences = np.flatnonzero(~equal_frames).astype(int).tolist()
    same_id = cpu["selected_detector_object_id"] == gpu["selected_detector_object_id"]
    cpu_monitor = json.loads(args.cpu_samples.read_text(encoding="utf-8")) if args.cpu_samples else None
    gpu_monitor = json.loads(args.gpu_samples.read_text(encoding="utf-8")) if args.gpu_samples else None
    for monitor in (cpu_monitor, gpu_monitor):
        if monitor is not None and monitor["sample_count"] < 30:
            raise ValueError("physical memory sampling too sparse")
    status = (f"PASS_EXACT_{args.frame_count}_MASK_EQUIVALENCE" if not differences and same_id
              else "CHANGED_OUTPUT_NOT_EQUIVALENT")
    payload = {
        "schema_version": "rc1-sam31-frame-storage-full-equivalence-v1",
        "session_id": args.session_id,
        "frame_count": args.frame_count,
        "status": status,
        "quality_status": "NOT_EVALUATED_PHYSICAL_INSTANCE_AND_FACE",
        "equal_mask_frames": int(equal_frames.sum()),
        "different_mask_frame_ids": differences,
        "selected_object_id_equal": same_id,
        "per_frame_score_equivalence": "NOT_MEASURABLE_NOT_SERIALIZED",
        "physical_gpu_total_peak_mib": {"cpu_frame_storage": cpu_monitor["peak_gpu_used_mib"] if cpu_monitor else None,
                                        "gpu_frame_storage": gpu_monitor["peak_gpu_used_mib"] if gpu_monitor else None},
        "observed_total_peak_saving_mib": (gpu_monitor["peak_gpu_used_mib"] - cpu_monitor["peak_gpu_used_mib"]
                                          if cpu_monitor and gpu_monitor else None),
        "monitor_sample_counts": {"cpu_frame_storage": cpu_monitor["sample_count"] if cpu_monitor else None,
                                  "gpu_frame_storage": gpu_monitor["sample_count"] if gpu_monitor else None},
        "inputs": {"cpu_result": artifact_ref(args.cpu_result), "gpu_result": artifact_ref(args.gpu_result),
                   "cpu_samples": artifact_ref(args.cpu_samples) if args.cpu_samples else None,
                   "gpu_samples": artifact_ref(args.gpu_samples) if args.gpu_samples else None},
        "code": artifact_ref(Path(__file__)),
        "outputs": {"cpu_packed_masks": artifact_ref(paths[0]), "gpu_packed_masks": artifact_ref(paths[1])},
        "authority_promoted": False,
        "training_eligible": False,
        "claim_limit": "Exact full-frame mask equality tests this one clip and frame-storage factor only. Sampled whole-GPU peak includes other processes and may miss transients. No physical card identity, face, segmentation accuracy or training authority follows.",
    }
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / "RESULT.json", payload)
    print(json.dumps({"status": status, "equal_mask_frames": payload["equal_mask_frames"],
                      "observed_total_peak_saving_mib": payload["observed_total_peak_saving_mib"]}))


if __name__ == "__main__":
    main()
