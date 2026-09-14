#!/usr/bin/env python3
"""Assemble one immutable exact78 HumanEgo bundle from graded production.

Only sessions whose HaWoR, Mask, Clean and Robot receipts are all terminal A/B
and whose Object/ICT staging receipt passed are admitted.  Canonical split
roles remain the frozen 60/8/5/5 cohort; heldout is never copied into training
artifacts.  Robot adapter symlinks are dereferenced into ordinary bundle-local
files so the strict selector root check cannot escape to raw storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / "tasks/control/runs/20260908_nine_hour_156_pipeline_v1"
COHORT = CONTROL / "COHORT_EXACT78_FROZEN.json"
COHORT_SHA256 = "d482ae4b627efa78653b69af26a357cec66381e3416dd78dd278504579e7790e"
MIN_SESSIONS = {"train": 16, "validation": 4, "test": 1}
MIN_WINDOWS = {"train": 256, "validation": 64, "test": 16}
STAGES = ("HAWOR", "MASK", "CLEAN", "ROBOT")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str((published_path or path).absolute()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def verify_ref(reference: dict[str, Any], label: str) -> Path:
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if not path.is_file() or path.stat().st_size != reference.get("bytes") or sha256(path) != reference.get("sha256"):
        raise RuntimeError(f"{label} reference drift")
    return path


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def robot_candidate(row: dict[str, Any]) -> dict[str, Any]:
    stages = {stage: row.get(stage, {}) for stage in STAGES}
    if any(stage.get("status") != "TERMINAL" or stage.get("grade") not in {"A", "B"} for stage in stages.values()):
        raise RuntimeError("upstream stage is not terminal A/B")
    robot_result_path = verify_ref(stages["ROBOT"]["result"], "Robot result")
    robot = read_json(robot_result_path)
    if not robot.get("terminal") or robot.get("grade") not in {"A", "B"} or not robot.get("may_train") or not robot.get("consumption_authorized"):
        raise RuntimeError("Robot receipt does not authorize weighted training")
    sidecar = verify_ref(robot.get("artifacts", {}).get("humanego_sidecar", {}), "Robot sidecar")
    selector_manifest_path = verify_ref(
        robot.get("artifacts", {}).get("selector_input_manifest", {}),
        "Robot selector input manifest",
    )
    selector = read_json(selector_manifest_path)
    adapter_root = Path(str(selector.get("adapter_root", ""))).resolve(strict=True)
    if selector.get("session") != row["session"]:
        raise RuntimeError("Robot selector session mismatch")
    with np.load(sidecar, allow_pickle=False) as archive:
        required = {"frame_names", "q", "wrist_T_camera", "valid", "confidence", "ik_warm_start_used"}
        if not required.issubset(archive.files):
            raise RuntimeError(f"Robot sidecar fields missing: {sorted(required-set(archive.files))}")
        names = [str(value) for value in archive["frame_names"]]
        q = np.asarray(archive["q"], dtype=np.float32)
        wrist = np.asarray(archive["wrist_T_camera"], dtype=np.float64)
        valid = np.asarray(archive["valid"], dtype=bool)
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
        warm = bool(np.asarray(archive["ik_warm_start_used"]).item())
    count = len(names)
    if q.shape != (count, 2, 22) or wrist.shape != (count, 2, 4, 4) or valid.shape != (count, 2) or confidence.shape != (count, 2):
        raise RuntimeError("Robot sidecar shape contract failed")
    if not warm or len(set(names)) != count or longest_true_run(valid.all(axis=1)) < 64:
        raise RuntimeError("Robot sidecar continuity/warm-start contract failed")
    indices = {int(name): index for index, name in enumerate(names)}
    starts = []
    for start in sorted(indices):
        selected = [indices.get(start + offset) for offset in range(50)]
        # The loader also requires one future JSON after the last action frame.
        if None not in selected and (adapter_root / "preprocess/all_data" / f"{start+50:05d}" / "training_data.json").is_file():
            if bool(valid[np.asarray(selected, dtype=np.int64)].all()):
                starts.append(start)
    if not starts:
        raise RuntimeError("Robot sidecar has zero consumable H50 windows")
    frame_names = sorted({f"{value:05d}" for start in starts for value in range(start, start + 51)})
    for name in frame_names:
        source = adapter_root / "preprocess/all_data" / name
        if not (source / "training_data.json").is_file() or not (source / "rgb.png").is_file():
            raise RuntimeError(f"Robot adapter missing frame {name}")
    return {
        "robot_result": robot_result_path,
        "robot": robot,
        "sidecar": sidecar,
        "adapter_root": adapter_root,
        "frame_names": frame_names,
        "window_starts": starts,
        "longest_dual_valid": longest_true_run(valid.all(axis=1)),
    }


def inspect(task: str, downstream: Path, training_inputs: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if sha256(COHORT) != COHORT_SHA256:
        raise RuntimeError("exact78 cohort SHA drift")
    cohort = read_json(COHORT)
    checkpoint = read_json(downstream)
    if checkpoint.get("cohort", {}).get("sha256") != COHORT_SHA256:
        raise RuntimeError("downstream checkpoint cohort mismatch")
    canonical = {
        role: [row["session_id"] for row in cohort["sessions"] if row["task"] == task and row["split"] == role]
        for role in ("train", "validation", "test", "heldout")
    }
    admitted: dict[str, list[dict[str, Any]]] = {role: [] for role in MIN_SESSIONS}
    rejected: dict[str, str] = {}
    sessions = checkpoint.get("sessions", {})
    for role in MIN_SESSIONS:
        for session in canonical[role]:
            row = sessions.get(session)
            if not isinstance(row, dict):
                rejected[session] = "DOWNSTREAM_ROW_MISSING"
                continue
            row = {**row, "session": session}
            receipt_path = training_inputs / task / "receipts" / f"{session}.json"
            if not receipt_path.is_file() or read_json(receipt_path).get("status") != "PASS_EPOCH0_ICT_INPUT":
                rejected[session] = "OBJECT_ICT_STAGING_NOT_PASS"
                continue
            try:
                candidate = robot_candidate(row)
            except Exception as error:  # per-session fail-forward audit
                rejected[session] = str(error)
                continue
            admitted[role].append({"session": session, "row": row, **candidate})
    return admitted, rejected


def publish_path(temporary: Path, bundle: Path, path: Path) -> Path:
    return bundle / path.relative_to(temporary)


def copy_ordinary(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)


def materialize(task: str, admitted: dict[str, list[dict[str, Any]]], bundle: Path, training_inputs: Path, downstream: Path, rejected: dict[str, str]) -> None:
    counts = {role: len(rows) for role, rows in admitted.items()}
    windows = {role: sum(len(row["window_starts"]) for row in rows) for role, rows in admitted.items()}
    for role, minimum in MIN_SESSIONS.items():
        if counts[role] < minimum:
            raise RuntimeError(f"HOLD_{role.upper()}_SESSIONS:{counts[role]}<{minimum}")
    for role, minimum in MIN_WINDOWS.items():
        if windows[role] < minimum:
            raise RuntimeError(f"HOLD_{role.upper()}_WINDOWS:{windows[role]}<{minimum}")
    if bundle.exists():
        raise RuntimeError(f"fresh bundle already exists: {bundle}")
    temporary = bundle.with_name(f".{bundle.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"temporary bundle already exists: {temporary}")
    temporary.mkdir(parents=True)

    cohort = read_json(COHORT)
    canonical = {
        role: [row["session_id"] for row in cohort["sessions"] if row["task"] == task and row["split"] == role]
        for role in ("train", "validation", "test", "heldout")
    }
    selector_sessions: dict[str, Any] = {}
    paired_windows: dict[str, Any] = {}
    robot_digests: dict[str, str] = {}
    hawor_digests: dict[str, str] = {}
    object_npz_digests: dict[str, str] = {}
    object_json_digests: dict[str, str] = {}
    grades: dict[str, str] = {}
    weights: dict[str, float] = {}
    session_meta: dict[str, Any] = {}
    thresholds: dict[str, float] | None = None

    for role in ("train", "validation", "test"):
        for candidate in admitted[role]:
            session = candidate["session"]
            adapter_destination = temporary / "production" / session / "09_humanego_adapter"
            records = {}
            for name in candidate["frame_names"]:
                source = candidate["adapter_root"] / "preprocess/all_data" / name
                destination = adapter_destination / "preprocess/all_data" / name
                json_destination = destination / "training_data.json"
                rgb_destination = destination / "rgb.png"
                copy_ordinary((source / "training_data.json").resolve(strict=True), json_destination)
                copy_ordinary((source / "rgb.png").resolve(strict=True), rgb_destination)
                records[name] = {
                    "metadata": file_ref(json_destination, published_path=publish_path(temporary, bundle, json_destination)),
                    "image": file_ref(rgb_destination, published_path=publish_path(temporary, bundle, rgb_destination)),
                    "unresolved": False,
                }
            selector_sessions[session] = {"frames": records}
            paired_windows[session] = {
                "frames": [int(value) for value in candidate["frame_names"]],
                "window_starts": candidate["window_starts"],
            }

            robot_destination = temporary / "sidecars/kai22" / session / "sidecar.npz"
            copy_ordinary(candidate["sidecar"], robot_destination)
            robot_digests[session] = sha256(robot_destination)
            hawor_source = training_inputs / task / "hawor_v3_sidecars" / session / "entities_hawor_v3.npz"
            hawor_destination = temporary / "hawor_v3_sidecars" / session / "entities_hawor_v3.npz"
            copy_ordinary(hawor_source.resolve(strict=True), hawor_destination)
            hawor_digests[session] = sha256(hawor_destination)
            object_source = training_inputs / task / "object_state_sidecars" / session
            object_destination = temporary / "object_state_sidecars" / session
            for filename in ("AUTO_ESTIMATED_OBJECT_STATE.npz", "AUTO_ESTIMATED_OBJECT_STATE.json"):
                copy_ordinary((object_source / filename).resolve(strict=True), object_destination / filename)
            object_npz_digests[session] = sha256(object_destination / "AUTO_ESTIMATED_OBJECT_STATE.npz")
            object_json_digests[session] = sha256(object_destination / "AUTO_ESTIMATED_OBJECT_STATE.json")
            manifest = read_json(object_destination / "AUTO_ESTIMATED_OBJECT_STATE.json")
            grades[session] = str(manifest["quality_grade"])
            weights[session] = float(manifest["training_weight"])
            if thresholds is None:
                thresholds = {str(k): float(v) for k, v in manifest["confidence_thresholds"].items()}
            elif thresholds != {str(k): float(v) for k, v in manifest["confidence_thresholds"].items()}:
                raise RuntimeError("object confidence thresholds differ between sessions")
            session_meta[session] = {
                "role": role,
                "robot_result": file_ref(candidate["robot_result"]),
                "robot_sidecar_sha256": robot_digests[session],
                "frames": len(candidate["frame_names"]),
                "h50_windows": len(candidate["window_starts"]),
                "longest_consecutive_dual_valid": candidate["longest_dual_valid"],
            }

    admitted_splits = {role: [row["session"] for row in admitted[role]] for role in MIN_SESSIONS}
    admitted_all = [session for role in ("train", "validation", "test") for session in admitted_splits[role]]
    production_root = bundle / "production"
    sidecar_root = bundle / "sidecars"
    hawor_root = bundle / "hawor_v3_sidecars"
    object_root = bundle / "object_state_sidecars"
    split = {
        "schema_version": "humanego-newtask-robot-split-v3-exact78-object-ict",
        "seed": 7, "split_unit": "session", "task": task,
        **canonical,
        "admitted_splits": admitted_splits,
        "selector_sessions": admitted_all,
        "sessions": session_meta,
        "production_root": str(production_root.absolute()),
        "sidecar_root": str(sidecar_root.absolute()),
        "hawor_v3_sidecar_root": str(hawor_root.absolute()),
        "ict_contract": {
            "hand_tracking_method": "hawor_v3", "single_hand": False,
            "ict_dim": 29, "frame_mode": "camera_frame",
            "translation_unit": "metre",
            "camera_coordinate_system": "x_right_y_down_z_forward",
            "object_token_policy": "AUTO_ESTIMATED_GRADE_B_FROZEN_PER_SESSION",
        },
        "object_state_contract": {
            "consumption_mode": "training_estimated_grade_b",
            "sidecar_root": str(object_root.absolute()),
            "npz_sha256_by_session": object_npz_digests,
            "json_sha256_by_session": object_json_digests,
            "quality_grade_by_session": grades,
            "training_weight_by_session": weights,
            "confidence_thresholds": thresholds or {},
        },
    }
    selector = {
        "schema_version": "humanego-newtask-robot-selector-v1",
        "product_line": "RAW_RGB_WITH_GRADED_ROBOT_ACTION",
        "image_name": "rgb.png",
        "artifact_root": str(production_root.absolute()),
        "selector_root": str(production_root.absolute()),
        "sessions": selector_sessions,
    }
    audit = {
        "schema_version": "exact78-fresh-bundle-audit-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": task,
        "cohort": file_ref(COHORT),
        "downstream_checkpoint": file_ref(downstream),
        "admitted_counts": counts,
        "valid_h50_windows": windows,
        "heldout_in_training": False,
        "ordinary_bundle_local_rgb_json": True,
        "rejected_sessions": rejected,
    }
    payloads = {
        "split.json": split,
        "selector_records.json": selector,
        "paired_windows.json": paired_windows,
        "sidecars.json": robot_digests,
        "hawor_v3_sidecars.json": hawor_digests,
        "AUDIT.json": audit,
    }
    for filename, payload in payloads.items():
        (temporary / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "schema_version": "humanego-newtask-robot-freeze-v3-exact78-object-ict",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": task, "training_started": False,
        "bundle_files": {
            filename: file_ref(temporary / filename, published_path=bundle / filename)
            for filename in payloads
        },
    }
    (temporary / "freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, bundle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--downstream-checkpoint", type=Path, required=True)
    parser.add_argument("--training-inputs", type=Path, default=CONTROL / "training_inputs")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    admitted, rejected = inspect(args.task, args.downstream_checkpoint.resolve(strict=True), args.training_inputs.resolve(strict=True))
    report = {
        "task": args.task,
        "admitted_counts": {role: len(rows) for role, rows in admitted.items()},
        "valid_h50_windows": {role: sum(len(row["window_starts"]) for row in rows) for role, rows in admitted.items()},
        "rejected_count": len(rejected),
        "ready": all(len(admitted[role]) >= minimum for role, minimum in MIN_SESSIONS.items())
        and all(sum(len(row["window_starts"]) for row in admitted[role]) >= minimum for role, minimum in MIN_WINDOWS.items()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.report_only:
        materialize(args.task, admitted, args.bundle.absolute(), args.training_inputs.resolve(strict=True), args.downstream_checkpoint.resolve(strict=True), rejected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
