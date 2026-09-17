#!/usr/bin/env python3
"""Build the H0 admission ledger for the glove/Controller/MANUS data line.

Only published metadata below ``processed`` is read.  Raw HDF5, images and
videos are never decoded or hashed.  In particular, tactile availability is
reported from the immutable source snapshot recorded by the converter; it is
not promoted to a materialized processed-frame authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = ("chips_cards_handle_0909", "chips_cards_handle_0910")
TASK_PREFIX = {
    "potato_chips": "get_potato_chips",
    "playing_cards": "play_cards",
}
ADMISSION_STATES = {
    "SENSOR_GEOMETRY_READY",
    "VISUAL_ONLY",
    "BLOCKED_SOURCE",
}


class AdmissionError(RuntimeError):
    """Raised when the dataset-level input contract is ambiguous."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AdmissionError(f"expected JSON object: {path}")
    return value


def _canonical_session(task: str, date_tag: str, source_session_id: str) -> str:
    try:
        prefix = TASK_PREFIX[task]
    except KeyError as exc:
        raise AdmissionError(f"unsupported task: {task}") from exc
    return f"{prefix}_{date_tag}_{int(source_session_id):03d}"


def _source_snapshot_has(conversion: dict[str, Any] | None, relative: str) -> bool:
    if conversion is None:
        return False
    item = conversion.get("source_snapshot", {}).get(relative)
    return bool(
        isinstance(item, dict)
        and isinstance(item.get("bytes"), int)
        and item["bytes"] > 0
        and isinstance(item.get("sha256"), str)
        and len(item["sha256"]) == 64
    )


def _artifact(path: Path | None, *, expected_bytes: int | None = None) -> dict[str, Any]:
    exists = bool(path is not None and path.is_file())
    size = path.stat().st_size if exists and path is not None else None
    return {
        "path": str(path.resolve()) if path is not None else None,
        "exists": exists,
        "bytes": size,
        "nonempty": bool(exists and size and size > 0),
        "expected_bytes": expected_bytes,
    }


def _build_row(
    dataset_id: str,
    date_tag: str,
    source_result: dict[str, Any],
) -> dict[str, Any]:
    task = str(source_result.get("task", ""))
    source_session_id = str(source_result.get("session_id", ""))
    canonical = _canonical_session(task, date_tag, source_session_id)
    classification = str(source_result.get("classification", "UNKNOWN"))
    target_text = source_result.get("target")
    target = Path(target_text) if isinstance(target_text, str) else None

    clip_path = target / "clip_manifest.json" if target else None
    sensor_path = target / "preprocess/pico_humanego_manifest.json" if target else None
    conversion_path = target / "CONVERSION_RESULT.json" if target else None
    camera_path = target / "camera_params.json" if target else None
    clip = _load_json(clip_path) if clip_path else None
    sensor = _load_json(sensor_path) if sensor_path else None
    conversion = _load_json(conversion_path) if conversion_path else None

    frame_count = source_result.get("frames")
    if frame_count is None and clip is not None:
        frame_count = clip.get("video", {}).get("frame_count")
    frame_count = int(frame_count) if isinstance(frame_count, (int, float)) else None

    files = clip.get("files", {}) if clip else {}
    mono_path = target / files["video"] if target and isinstance(files.get("video"), str) else None
    stereo_path = (
        target / files["source_stereo_video"]
        if target and isinstance(files.get("source_stereo_video"), str)
        else None
    )
    controller_path = (
        target / files["controller_poses"]
        if target and isinstance(files.get("controller_poses"), str)
        else None
    )

    validation = conversion.get("validation", {}) if conversion else {}
    stereo_probe = validation.get("stereo_probe", {})
    stereo_frames = stereo_probe.get("nb_read_frames")
    try:
        stereo_frames = int(stereo_frames)
    except (TypeError, ValueError):
        stereo_frames = None
    controller_lines = validation.get("controller_lines")
    controller_lines = int(controller_lines) if isinstance(controller_lines, (int, float)) else None

    active = sensor.get("active_frame_counts", {}) if sensor else {}
    both_active = active.get("both")
    both_active = int(both_active) if isinstance(both_active, (int, float)) else None
    hand_model = sensor.get("hand_model", {}) if sensor else {}
    manus_source_snapshot = _source_snapshot_has(conversion, "raw/manus.jsonl")
    manus_meta_snapshot = _source_snapshot_has(conversion, "raw/manus.meta.json")
    manus_eligible = bool(
        frame_count
        and both_active == frame_count
        and "MANUS25" in str(hand_model.get("name", ""))
        and manus_source_snapshot
        and manus_meta_snapshot
    )
    controller_eligible = bool(
        frame_count
        and controller_path is not None
        and controller_path.is_file()
        and controller_lines == frame_count
        and sensor
        and sensor.get("controller6d", {}).get("source")
    )
    stereo_eligible = bool(
        frame_count
        and stereo_path is not None
        and stereo_path.is_file()
        and stereo_frames == frame_count
        and stereo_probe.get("width") == 4096
        and stereo_probe.get("height") == 1536
    )
    camera_eligible = bool(
        camera_path is not None
        and camera_path.is_file()
        and clip
        and clip.get("calibration_consistent") is True
        and clip.get("source_calibration", {}).get("sha256")
    )
    visual_eligible = bool(
        classification == "CLEANED"
        and frame_count
        and mono_path is not None
        and mono_path.is_file()
        and clip
        and sensor
    )
    tactile_source_snapshot = bool(
        _source_snapshot_has(conversion, "raw/tactile.jsonl")
        and _source_snapshot_has(conversion, "raw/tactile.meta.json")
    )

    cadence = clip.get("timeline_resampling", {}) if clip else {}
    cadence_eligible = bool(
        cadence
        and cadence.get("negative_transitions") == 0
        and cadence.get("hdf5_rows") == frame_count
    )
    content_status = (
        sensor.get("content_admission", {}).get("status")
        if sensor
        else source_result.get("content_status")
    )
    geometry_components = {
        "manus": manus_eligible,
        "controller_wrist": controller_eligible,
        "source_stereo": stereo_eligible,
        "camera_contract": camera_eligible,
        "cadence": cadence_eligible,
    }
    if classification != "CLEANED" or not visual_eligible:
        admission = "BLOCKED_SOURCE"
    elif content_status == "SESSION_CONTENT_ADMISSIBLE" and all(geometry_components.values()):
        admission = "SENSOR_GEOMETRY_READY"
    else:
        admission = "VISUAL_ONLY"
    assert admission in ADMISSION_STATES

    blockers: list[str] = []
    if classification != "CLEANED":
        blockers.append("SOURCE_SESSION_REJECTED")
    if not visual_eligible:
        blockers.append("PUBLISHED_VISUAL_CONTRACT_INCOMPLETE")
    if content_status != "SESSION_CONTENT_ADMISSIBLE":
        blockers.append(str(content_status or "CONTENT_ADMISSION_UNKNOWN"))
    blockers.extend(f"{name.upper()}_NOT_ELIGIBLE" for name, ok in geometry_components.items() if not ok)

    input_evidence = []
    for path in (clip_path, sensor_path, conversion_path, camera_path):
        if path is not None and path.is_file():
            input_evidence.append(_evidence(path))

    return {
        "dataset_id": dataset_id,
        "date_tag": date_tag,
        "task": task,
        "source_session_id": source_session_id,
        "session_id": canonical,
        "source": source_result.get("source"),
        "target": str(target.resolve()) if target else None,
        "source_terminal": {
            "classification": classification,
            "status": source_result.get("status"),
            "error": source_result.get("error"),
        },
        "frame_count": frame_count,
        "content_admission": content_status or "UNKNOWN",
        "admission": admission,
        "primary_blockers": sorted(set(blockers)),
        "eligibility": {
            "visual_rgb": {
                "eligible": visual_eligible,
                "artifact": _artifact(mono_path),
                "claim": "Published 1280x960 visual-domain input only.",
            },
            "manus25": {
                "eligible": manus_eligible,
                "both_active_frames": both_active,
                "joint_count": len(hand_model.get("joint_names", [])),
                "source_snapshot_present": manus_source_snapshot and manus_meta_snapshot,
                "claim": "Wrist-local MANUS25 sensor geometry; not HaWoR, MANO mesh, or external truth.",
            },
            "controller_wrist": {
                "eligible": controller_eligible,
                "controller_lines": controller_lines,
                "artifact": _artifact(controller_path),
                "derivation": (sensor or {}).get("controller6d", {}).get("wrist_derivation"),
                "claim": "Controller plus recorded controller-to-wrist calibration; not external wrist truth.",
            },
            "tactile": {
                "eligible_for_successor_extraction": tactile_source_snapshot,
                "materialized_in_processed_frame_contract": False,
                "status": (
                    "SOURCE_SNAPSHOT_PRESENT_NOT_MATERIALIZED"
                    if tactile_source_snapshot
                    else "SOURCE_SNAPSHOT_NOT_PROVEN"
                ),
                "claim": "Source provenance only; no processed tactile/contact authority is granted.",
            },
            "source_stereo": {
                "eligible_for_depth_preflight": stereo_eligible and camera_eligible,
                "artifact": _artifact(stereo_path),
                "probe": stereo_probe or None,
                "claim": "SBS availability only; rectification, registration, and metric depth gates remain required.",
            },
            "camera_contract": {
                "eligible": camera_eligible,
                "artifact": _artifact(camera_path),
                "intrinsics_authority": (sensor or {}).get("intrinsics_authority"),
                "image_domain_mode": (sensor or {}).get("image_domain_mode"),
                "rectified": (sensor or {}).get("rectified_video_written"),
                "claim": "Same-session algebra/provenance; not external calibration accuracy.",
            },
            "cadence": {
                "eligible": cadence_eligible,
                "hdf5_rows": cadence.get("hdf5_rows"),
                "source_video_frames": cadence.get("source_video_frames"),
                "unique_source_indices": cadence.get("unique_source_indices"),
                "repeated_transitions": cadence.get("repeated_transitions"),
                "skipped_transitions": cadence.get("skipped_transitions"),
                "max_forward_step": cadence.get("max_forward_step"),
                "negative_transitions": cadence.get("negative_transitions"),
                "segment_counts": cadence.get("segment_counts"),
                "claim": "Repeat/skip events are retained quality flags, not silently repaired frames.",
            },
        },
        "downstream": {
            "canonical_manus_controller_adapter": manus_eligible and controller_eligible and camera_eligible,
            "stereo_depth_preflight": stereo_eligible and camera_eligible and cadence_eligible,
            "geometry_prompted_mask_preflight": manus_eligible and controller_eligible and camera_eligible,
            "tactile_sidecar_extraction": tactile_source_snapshot,
            "clean": "NOT_EVALUATED_REQUIRES_NEW_ROLE_AND_OBJECT_MASKS",
            "contact": "NOT_EVALUATED_REQUIRES_OBJECT6D_AND_MATERIALIZED_TACTILE",
        },
        "input_evidence": input_evidence,
        "claim_limit": (
            "H0 admission from published processed metadata only. SENSOR_GEOMETRY_READY does not grant "
            "Depth, Object6D, Mask, Clean, Contact, Robot, physical accuracy, or deployment authority."
        ),
    }


def build_ledger(processed_root: Path) -> dict[str, Any]:
    processed_root = processed_root.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    dataset_evidence: list[dict[str, Any]] = []
    for dataset_id in DATASETS:
        result_path = processed_root / dataset_id / "DATASET_RESULT.json"
        result = _load_json(result_path)
        if result is None:
            raise AdmissionError(f"dataset result missing: {result_path}")
        date_tag = str(result.get("date_tag") or dataset_id.rsplit("_", 1)[-1])
        dataset_evidence.append(_evidence(result_path))
        values = result.get("results")
        if not isinstance(values, list):
            raise AdmissionError(f"results must be an array: {result_path}")
        for value in values:
            if not isinstance(value, dict):
                raise AdmissionError(f"invalid result row: {result_path}")
            rows.append(_build_row(dataset_id, date_tag, value))

    identities = [row["session_id"] for row in rows]
    duplicates = sorted(name for name, count in Counter(identities).items() if count > 1)
    if duplicates:
        raise AdmissionError(f"duplicate canonical session identities: {duplicates}")
    rows.sort(key=lambda row: (row["date_tag"], row["task"], row["source_session_id"]))

    by_admission = Counter(row["admission"] for row in rows)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_dataset[row["dataset_id"]][row["admission"]] += 1
        by_task[row["task"]][row["admission"]] += 1
    return {
        "schema_version": "handle-sensor-pipeline-admission-ledger-v7.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed_root": str(processed_root),
        "row_count": len(rows),
        "dataset_result_evidence": dataset_evidence,
        "summary": {
            "by_admission": dict(sorted(by_admission.items())),
            "by_dataset": {k: dict(sorted(v.items())) for k, v in sorted(by_dataset.items())},
            "by_task": {k: dict(sorted(v.items())) for k, v in sorted(by_task.items())},
            "total_frames_declared": sum(row["frame_count"] or 0 for row in rows),
        },
        "rows": rows,
        "claim_limit": (
            "Dynamic H0 routing ledger. It reads processed metadata only and grants no downstream "
            "algorithm, physical truth, contact truth, Robot-control, or deployment authority."
        ),
    }


def write_ledger(output_dir: Path, ledger: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "SENSOR_PIPELINE_ADMISSION_LEDGER.json"
    csv_path = output_dir / "SENSOR_PIPELINE_ADMISSION_LEDGER.csv"
    json_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "dataset_id", "date_tag", "task", "source_session_id", "session_id",
        "frame_count", "source_classification", "content_admission", "admission",
        "manus25_eligible", "controller_wrist_eligible", "tactile_successor_eligible",
        "source_stereo_depth_preflight", "camera_contract_eligible", "cadence_eligible",
        "repeated_transitions", "skipped_transitions", "max_forward_step",
        "primary_blockers", "target", "claim_limit",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ledger["rows"]:
            e = row["eligibility"]
            writer.writerow({
                "dataset_id": row["dataset_id"],
                "date_tag": row["date_tag"],
                "task": row["task"],
                "source_session_id": row["source_session_id"],
                "session_id": row["session_id"],
                "frame_count": row["frame_count"],
                "source_classification": row["source_terminal"]["classification"],
                "content_admission": row["content_admission"],
                "admission": row["admission"],
                "manus25_eligible": e["manus25"]["eligible"],
                "controller_wrist_eligible": e["controller_wrist"]["eligible"],
                "tactile_successor_eligible": e["tactile"]["eligible_for_successor_extraction"],
                "source_stereo_depth_preflight": e["source_stereo"]["eligible_for_depth_preflight"],
                "camera_contract_eligible": e["camera_contract"]["eligible"],
                "cadence_eligible": e["cadence"]["eligible"],
                "repeated_transitions": e["cadence"]["repeated_transitions"],
                "skipped_transitions": e["cadence"]["skipped_transitions"],
                "max_forward_step": e["cadence"]["max_forward_step"],
                "primary_blockers": "|".join(row["primary_blockers"]),
                "target": row["target"],
                "claim_limit": row["claim_limit"],
            })
    receipt = {
        "schema_version": "handle-h0-admission-result-v7.1",
        "status": "PASSED",
        "row_count": ledger["row_count"],
        "summary": ledger["summary"],
        "files": {name: _evidence(output_dir / name) for name in (
            json_path.name, csv_path.name
        )},
        "claim_limit": ledger["claim_limit"],
    }
    (output_dir / "RESULT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/mnt/data/egodata/datasets/ego/processed"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_ledger(args.processed_root)
    write_ledger(args.output_dir.resolve(), ledger)
    print(json.dumps({"status": "PASSED", **ledger["summary"], "rows": ledger["row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
