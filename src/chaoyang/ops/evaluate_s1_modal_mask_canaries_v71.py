#!/usr/bin/env python3
from __future__ import annotations

"""Build a bounded, non-Gold comparison of S1 modal-mask GPU canaries.

The frozen re-entry rows are predecessor-derived audit locations, not human
ground truth.  Consequently this evaluator reports coverage, prompt fidelity,
temporal stability and cross-backend agreement but deliberately never reports
accuracy.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT = ROOT / (
    "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/"
    "s1_reentry_audit/FROZEN_REENTRY_AUDIT_FRAMES_V71.json"
)
VISIBLE = {"VISIBLE", "PARTIAL"}
UNKNOWN = {"FULLY_OCCLUDED_UNKNOWN", "OUT_OF_FRAME_UNKNOWN", "TRACK_LOST_UNKNOWN"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_ref(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} path missing/not absolute: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{label} bytes mismatch: {path}")
    if sha256(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA mismatch: {path}")
    return path


def longest_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def read_binary_mask(record: Mapping[str, Any], label: str) -> np.ndarray:
    path = checked_ref(record, label)
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{label} must be lossless NPY: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{label} is not HxW: {array.shape}")
    return np.asarray(array > 0)


def iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError(f"mask shape mismatch: {first.shape} != {second.shape}")
    union = int(np.count_nonzero(first | second))
    return 1.0 if union == 0 else int(np.count_nonzero(first & second)) / union


def receipt_output(receipt_path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    receipt = load_json(receipt_path)
    payload = receipt.get("payload", receipt)
    backend = str(payload.get("backend", "UNKNOWN"))
    if payload.get("status") != "PASSED":
        return backend, payload, {}
    output_ref = payload.get("output")
    if not isinstance(output_ref, Mapping):
        raise ValueError(f"PASSED receipt has no output: {receipt_path}")
    output = load_json(checked_ref(output_ref, f"{backend}.output"))
    if output.get("execution_mode") != "CAUSAL_PROCESSING":
        raise ValueError(f"{backend} output is not causal")
    if output.get("mask_semantics") != "VISIBLE_MODAL_SURFACE_ONLY":
        raise ValueError(f"{backend} output semantics mismatch")
    if output.get("allow_instance_union") is not False:
        raise ValueError(f"{backend} permits instance union")
    return backend, payload, output


def summarize_output(output: Mapping[str, Any]) -> dict[str, Any]:
    frames = output.get("frames", [])
    if not frames:
        raise ValueError("mask output has no frames")
    instance_ids = list(output.get("identity_qa", {}).get("instance_ids", []))
    by_id: dict[str, list[dict[str, Any]]] = {item: [] for item in instance_ids}
    total_known = total_unknown = 0
    all_areas: dict[str, list[int | None]] = {item: [] for item in instance_ids}
    for frame in frames:
        records = {item["instance_id"]: item for item in frame.get("instances", [])}
        if set(records) != set(instance_ids):
            raise ValueError(f"identity set drift at frame {frame.get('frame_id')}")
        for instance_id in instance_ids:
            item = records[instance_id]
            visibility = item.get("visibility")
            if visibility in VISIBLE:
                mask = read_binary_mask(item["mask"], f"frame {frame['frame_id']} {instance_id}")
                area = int(np.count_nonzero(mask))
                if area <= 0:
                    raise ValueError("visible mask is empty")
                total_known += 1
                all_areas[instance_id].append(area)
            elif visibility in UNKNOWN:
                if item.get("mask") is not None:
                    raise ValueError("UNKNOWN record carries pseudo-mask")
                total_unknown += 1
                all_areas[instance_id].append(None)
            else:
                raise ValueError(f"invalid visibility: {visibility}")
            by_id[instance_id].append(item)
    instance_metrics: dict[str, Any] = {}
    for instance_id, values in by_id.items():
        unknown_flags = [item["visibility"] in UNKNOWN for item in values]
        areas = all_areas[instance_id]
        known_areas = [float(value) for value in areas if value is not None]
        step_changes = []
        for before, after in zip(areas, areas[1:]):
            if before is not None and after is not None:
                step_changes.append(abs(after - before) / max(before, after, 1))
        instance_metrics[instance_id] = {
            "known_decisions": len(values) - sum(unknown_flags),
            "unknown_decisions": sum(unknown_flags),
            "unknown_ratio": sum(unknown_flags) / len(values),
            "max_unknown_run_frames": longest_run(unknown_flags),
            "area_pixels_median_known": float(np.median(known_areas)) if known_areas else None,
            "area_step_change_p95_known_pairs": float(np.percentile(step_changes, 95)) if step_changes else None,
        }
    decisions = total_known + total_unknown
    return {
        "frames": len(frames),
        "instances": len(instance_ids),
        "known_decision_coverage": total_known / decisions,
        "unknown_decision_ratio": total_unknown / decisions,
        "identity_warning_count": int(output.get("identity_qa", {}).get("warning_count", 0)),
        "per_instance": instance_metrics,
    }


def audit_event_coverage(output: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    frame_map = {
        int(frame["frame_id"]): {item["instance_id"]: item for item in frame["instances"]}
        for frame in output["frames"]
    }
    instance_ids = list(output.get("identity_qa", {}).get("instance_ids", []))
    if len(instance_ids) != 1:
        return {"status": "NOT_APPLICABLE_MULTI_INSTANCE", "events": []}
    instance_id = instance_ids[0]
    rows = [
        row for row in audit.get("rows", [])
        if row.get("session_id") == output.get("session_id") and row.get("event")
    ]
    events = []
    for row in rows:
        event = row["event"]
        start, end = int(event["empty_or_unobserved_start"]), int(event["empty_or_unobserved_end"])
        region = [frame_map.get(index, {}).get(instance_id) for index in range(start, end + 1)]
        known = sum(bool(item and item.get("visibility") in VISIBLE) for item in region)
        audit_frames = []
        for frame_id in event.get("audit_frames", []):
            item = frame_map.get(int(frame_id), {}).get(instance_id)
            audit_frames.append({
                "frame_id": int(frame_id),
                "candidate_visibility": None if item is None else item.get("visibility"),
            })
        events.append({
            "event_id": event["event_id"],
            "predecessor_unobserved_length": len(region),
            "candidate_known_coverage_in_predecessor_unobserved_region": known / len(region) if region else None,
            "audit_frames": audit_frames,
        })
    return {
        "status": "DEVELOPMENT_PREDECESSOR_DERIVED_NOT_GOLD",
        "events": events,
        "claim_limit": "Coverage over predecessor-unobserved intervals is not correctness; a candidate can be confidently wrong.",
    }


def cross_backend(outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(outputs) != 2:
        return {"status": "NOT_AVAILABLE", "common_known_masks": 0, "mask_iou_mean": None}
    names = sorted(outputs)
    first, second = outputs[names[0]], outputs[names[1]]
    first_frames = {int(frame["frame_id"]): frame for frame in first["frames"]}
    second_frames = {int(frame["frame_id"]): frame for frame in second["frames"]}
    values = []
    for frame_id in sorted(set(first_frames) & set(second_frames)):
        left = {item["instance_id"]: item for item in first_frames[frame_id]["instances"]}
        right = {item["instance_id"]: item for item in second_frames[frame_id]["instances"]}
        for instance_id in sorted(set(left) & set(right)):
            a, b = left[instance_id], right[instance_id]
            if a["visibility"] in VISIBLE and b["visibility"] in VISIBLE:
                values.append(iou(
                    read_binary_mask(a["mask"], f"{names[0]} {frame_id} {instance_id}"),
                    read_binary_mask(b["mask"], f"{names[1]} {frame_id} {instance_id}"),
                ))
    return {
        "status": "DEVELOPMENT_AGREEMENT_NOT_ACCURACY",
        "backends": names,
        "common_known_masks": len(values),
        "mask_iou_mean": float(np.mean(values)) if values else None,
        "mask_iou_p05": float(np.percentile(values, 5)) if values else None,
    }


def build_comparison(receipt_paths: list[Path], audit_path: Path) -> dict[str, Any]:
    audit = load_json(audit_path)
    outputs: dict[str, dict[str, Any]] = {}
    backends: dict[str, Any] = {}
    receipt_refs = []
    for receipt_path in receipt_paths:
        receipt_path = receipt_path.resolve()
        receipt_refs.append(file_ref(receipt_path))
        backend, payload, output = receipt_output(receipt_path)
        row: dict[str, Any] = {
            "attempt_status": payload.get("status"),
            "execution_performed": payload.get("execution_performed"),
            "runtime_error": payload.get("runtime_error"),
        }
        if output:
            outputs[backend] = output
            row["internal_metrics"] = summarize_output(output)
            row["reentry_development_coverage"] = audit_event_coverage(output, audit)
        backends[backend] = row
    execution_passed = all(row.get("attempt_status") == "PASSED" for row in backends.values())
    return {
        "schema_version": "s1-modal-mask-development-comparison-v71",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "COMPARISON_COMPLETE_REVIEW_REQUIRED" if execution_passed else "NO_GO_BACKEND_EXECUTION",
        "authority": False,
        "gold_accuracy_computed": False,
        "accuracy": None,
        "input_receipts": receipt_refs,
        "audit_selection": file_ref(audit_path),
        "backends": backends,
        "cross_backend": cross_backend(outputs),
        "next_action": "REVIEW_THEN_RUN_ONE_FAILURE_CANARY_PLUS_TWO_AB_REGRESSIONS" if execution_passed else "FIX_EXECUTION_WITHIN_ATTEMPT_BUDGET",
        "claim_limit": "Development-only internal coverage/stability/agreement. No metric is Gold accuracy or Mask/Contact/Object6D/physical authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_comparison(args.receipt, args.audit.resolve())
    if args.output.exists():
        raise FileExistsError(f"immutable comparison exists: {args.output}")
    atomic_json(args.output.resolve(), value)
    print(json.dumps({"status": value["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if value["status"] == "COMPARISON_COMPLETE_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
