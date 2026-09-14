#!/usr/bin/env python3
"""Build the time-bounded V3 bundle from scoped Grade-B motion labels.

The canonical 60/8/5/5 assignment is preserved.  For the nine-hour
fail-forward checkpoint only a quality-admitted >=16 train / >=4 validation
subset is materialized; test may be empty, and all five heldout sessions are
strictly absent from training artifacts.  This builder never consumes a
truthful-C Robot receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / "tasks/control/runs/20260908_nine_hour_156_pipeline_v1"
COHORT = CONTROL / "COHORT_EXACT78_FROZEN.json"
COHORT_SHA256 = "d482ae4b627efa78653b69af26a357cec66381e3416dd78dd278504579e7790e"
MOTION_ROOT = CONTROL / "training_inputs"
MIN_SESSIONS = {"train": 16, "validation": 4, "test": 0}
MIN_WINDOWS = {"train": 256, "validation": 64, "test": 0}
SPAN_FRAMES = 80
HORIZON = 50


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ref(path: Path, *, published: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str((published or path).absolute()), "bytes": path.stat().st_size, "sha256": digest(path)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_ordinary(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Frozen-contract consumers reject hard-linked inputs even when their
    # content digest is correct: an alias could mutate both paths.  Always
    # materialize an independent inode (nlink=1).
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)


def longest_span(valid: np.ndarray, count: int = SPAN_FRAMES) -> tuple[int, int]:
    dual = np.asarray(valid, dtype=bool).all(axis=1)
    best_start = best_stop = start = 0
    for index in range(len(dual) + 1):
        if index < len(dual) and dual[index]:
            continue
        if index - start > best_stop - best_start:
            best_start, best_stop = start, index
        start = index + 1
    if best_stop - best_start < count:
        raise RuntimeError(f"dual-valid run {best_stop-best_start}<{count}")
    return best_start, best_start + count


def motion_receipt(task: str, session: str) -> Path:
    candidates = []
    # Historical Poker q is preferred when it passed the same empirical gate.
    if task == "poker":
        candidates.append(MOTION_ROOT / task / "motion_training_label_grade_b" / session / "RESULT.json")
    candidates.append(MOTION_ROOT / task / "motion_training_label_grade_b" / session / "RESULT.json")
    for path in candidates:
        if path.is_file():
            payload = read_json(path)
            if payload.get("status") == "PASS_MOTION_TRAINING_LABEL_GRADE_B":
                return path
    raise RuntimeError("motion Grade-B receipt missing")


def inspect_row(row: dict[str, Any]) -> dict[str, Any]:
    task, session = row["task"], row["session_id"]
    motion_result_path = motion_receipt(task, session)
    motion = read_json(motion_result_path)
    if motion.get("cohort_sha256") != COHORT_SHA256 or motion.get("split") != row["split"]:
        raise RuntimeError("motion cohort/split binding mismatch")
    artifact = motion.get("artifact", {})
    sidecar = Path(str(artifact.get("path", ""))).resolve(strict=True)
    if digest(sidecar) != artifact.get("sha256"):
        raise RuntimeError("motion artifact digest drift")
    with np.load(sidecar, allow_pickle=False) as archive:
        names = np.asarray(archive["frame_names"]).astype(str)
        q = np.asarray(archive["q"], dtype=np.float32)
        wrist = np.asarray(archive["wrist_T_camera"], dtype=np.float64)
        valid = np.asarray(archive["valid"], dtype=bool)
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
        provenance = str(np.asarray(archive["provenance_class"]).item())
        continuity = bool(np.asarray(archive["empirical_continuity_verified"]).item())
        achieved_fk = bool(np.asarray(archive["achieved_robot_fk"]).item())
    count = len(names)
    if q.shape != (count, 2, 22) or wrist.shape != (count, 2, 4, 4) or valid.shape != (count, 2):
        raise RuntimeError("motion shape contract failed")
    if confidence.shape != (count, 2) or provenance != "MOTION_TRAINING_LABEL_GRADE_B" or not continuity or achieved_fk:
        raise RuntimeError("motion scoped provenance contract failed")
    start, stop = longest_span(valid)
    selected_names = names[start:stop].tolist()
    selected_int = [int(value) for value in selected_names]
    if selected_int != list(range(selected_int[0], selected_int[0] + SPAN_FRAMES)):
        raise RuntimeError("selected frame names are not consecutive")
    historical_adapter = (
        PROJECT / "HumanEgo/artifacts/newtask_robot_bundles/poker/production"
        / session / "09_humanego_adapter/preprocess/all_data"
    )
    # Reuse ordinary digest-frozen local RGB/JSON for historical Poker rows;
    # other sessions bind directly to the canonical raw root during the copy.
    raw_all_data = historical_adapter if historical_adapter.is_dir() else Path(row["raw_path"]) / "preprocess/all_data"
    # H50 needs the following frame's metadata.  Materialize 81 frames so all
    # 31 declared starts in the selected 80-frame span are actually readable.
    first = selected_int[0]
    required = [f"{value:05d}" for value in range(first, first + SPAN_FRAMES + 1)]
    staging_receipt_path = MOTION_ROOT / task / "receipts" / f"{session}.json"
    staging = read_json(staging_receipt_path)
    if staging.get("status") != "PASS_EPOCH0_ICT_INPUT" or staging.get("split") != row["split"]:
        raise RuntimeError("Object/ICT staging not passed")
    return {
        "row": row, "motion": motion, "motion_result": motion_result_path,
        "sidecar": sidecar, "required_names": required, "first": first,
        "window_starts": list(range(first, first + SPAN_FRAMES - HORIZON + 1)),
        "raw_all_data": raw_all_data, "staging": staging,
    }


def inspect(task: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if digest(COHORT) != COHORT_SHA256:
        raise RuntimeError("cohort digest drift")
    cohort = read_json(COHORT)
    admitted = {role: [] for role in MIN_SESSIONS}
    rejected: dict[str, str] = {}
    # Minimal deterministic subset avoids copying tens of thousands of RGBs.
    target = {"train": 16, "validation": 4, "test": 0}
    for role in ("train", "validation", "test"):
        rows = [row for row in cohort["sessions"] if row["task"] == task and row["split"] == role]
        for row in rows:
            if len(admitted[role]) >= target[role]:
                break
            try:
                admitted[role].append(inspect_row(row))
            except Exception as error:
                rejected[row["session_id"]] = str(error)
    return admitted, rejected


def materialize(task: str, admitted: dict[str, list[dict[str, Any]]], rejected: dict[str, str], bundle: Path) -> None:
    counts = {role: len(rows) for role, rows in admitted.items()}
    windows = {role: sum(len(row["window_starts"]) for row in rows) for role, rows in admitted.items()}
    for role, minimum in MIN_SESSIONS.items():
        if counts[role] < minimum:
            raise RuntimeError(f"HOLD_{role.upper()}_SESSIONS:{counts[role]}<{minimum}")
    for role, minimum in MIN_WINDOWS.items():
        if windows[role] < minimum:
            raise RuntimeError(f"HOLD_{role.upper()}_WINDOWS:{windows[role]}<{minimum}")
    if bundle.exists():
        raise RuntimeError(f"no-clobber bundle exists:{bundle}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.", dir=bundle.parent))
    try:
        canonical_rows = [row for row in read_json(COHORT)["sessions"] if row["task"] == task]
        canonical = {role: [row["session_id"] for row in canonical_rows if row["split"] == role]
                     for role in ("train", "validation", "test", "heldout")}
        selector_sessions: dict[str, Any] = {}; paired_windows: dict[str, Any] = {}
        sidecar_digests: dict[str, str] = {}; hawor_digests: dict[str, str] = {}
        object_npz_digests: dict[str, str] = {}; object_json_digests: dict[str, str] = {}
        grades: dict[str, str] = {}; weights: dict[str, float] = {}; motion_receipt_digests: dict[str, str] = {}
        motion_weights: dict[str, float] = {}; contact_weights: dict[str, float] = {}
        sessions_meta: dict[str, Any] = {}; thresholds: dict[str, float] | None = None
        for role in ("train", "validation", "test"):
            for candidate in admitted[role]:
                session = candidate["row"]["session_id"]
                destination_all = temporary / "production" / session / "09_humanego_adapter/preprocess/all_data"
                frame_records = {}
                for name in candidate["required_names"]:
                    source = candidate["raw_all_data"] / name
                    destination = destination_all / name
                    copy_ordinary(source / "training_data.json", destination / "training_data.json")
                    copy_ordinary(source / "rgb.png", destination / "rgb.png")
                    frame_records[name] = {
                        "metadata": ref(destination / "training_data.json", published=bundle / (destination / "training_data.json").relative_to(temporary)),
                        "image": ref(destination / "rgb.png", published=bundle / (destination / "rgb.png").relative_to(temporary)),
                        "unresolved": False,
                    }
                selector_sessions[session] = {"frames": frame_records}
                paired_windows[session] = {"frames": [int(x) for x in candidate["required_names"]],
                                           "window_starts": candidate["window_starts"]}
                side_destination = temporary / "sidecars/kai22" / session / "sidecar.npz"
                copy_ordinary(candidate["sidecar"], side_destination)
                sidecar_digests[session] = digest(side_destination)
                hawor_source = MOTION_ROOT / task / "hawor_v3_sidecars" / session / "entities_hawor_v3.npz"
                hawor_destination = temporary / "hawor_v3_sidecars" / session / "entities_hawor_v3.npz"
                copy_ordinary(hawor_source, hawor_destination); hawor_digests[session] = digest(hawor_destination)
                object_source = MOTION_ROOT / task / "object_state_sidecars" / session
                object_destination = temporary / "object_state_sidecars" / session
                copy_ordinary(object_source / "AUTO_ESTIMATED_OBJECT_STATE.npz", object_destination / "AUTO_ESTIMATED_OBJECT_STATE.npz")
                copy_ordinary(object_source / "AUTO_ESTIMATED_OBJECT_STATE.json", object_destination / "AUTO_ESTIMATED_OBJECT_STATE.json")
                object_npz_digests[session] = digest(object_destination / "AUTO_ESTIMATED_OBJECT_STATE.npz")
                object_json_digests[session] = digest(object_destination / "AUTO_ESTIMATED_OBJECT_STATE.json")
                object_manifest = read_json(object_destination / "AUTO_ESTIMATED_OBJECT_STATE.json")
                grades[session] = str(object_manifest["quality_grade"]); weights[session] = float(object_manifest["training_weight"])
                current_thresholds = {str(k): float(v) for k, v in object_manifest["confidence_thresholds"].items()}
                if thresholds is None: thresholds = current_thresholds
                elif thresholds != current_thresholds: raise RuntimeError("object confidence threshold drift")
                motion_receipt_digests[session] = digest(candidate["motion_result"])
                motion_weights[session] = float(candidate["motion"]["weights"]["motion"])
                contact_weights[session] = float(candidate["motion"]["weights"]["contact_aux"])
                sessions_meta[session] = {
                    "role": role, "motion_label_grade": "B", "motion_weight": 0.5,
                    "contact_aux_weight": 0.0, "object_aux_weight": 0.25,
                    "motion_result": ref(candidate["motion_result"]),
                    "frames_materialized": len(candidate["required_names"]),
                    "h50_windows": len(candidate["window_starts"]),
                }
        admitted_splits = {role: [x["row"]["session_id"] for x in rows] for role, rows in admitted.items()}
        admitted_all = [x for role in ("train", "validation", "test") for x in admitted_splits[role]]
        split = {
            "schema_version": "humanego-newtask-robot-split-v3-exact78-object-ict-motion-grade-b",
            "seed": 7, "split_unit": "session", "task": task, **canonical,
            "admitted_splits": admitted_splits, "selector_sessions": admitted_all,
            "sessions": sessions_meta, "production_root": str((bundle / "production").absolute()),
            "sidecar_root": str((bundle / "sidecars").absolute()),
            "hawor_v3_sidecar_root": str((bundle / "hawor_v3_sidecars").absolute()),
            "checkpoint_scope": "NINE_HOUR_FAIL_FORWARD_NOT_FINAL_FULL_COHORT",
            "heldout_contract": {"count": 5, "in_training": False, "purpose": "FINAL_UNSEEN_INFERENCE_ONLY"},
            "motion_label_contract": {
                "consumption_mode": "MOTION_TRAINING_LABEL_GRADE_B",
                "provenance_class": "MOTION_TRAINING_LABEL_GRADE_B",
                "target": "fresh_wrist_T_camera_plus_q_hand",
                "arm_consumed": False, "achieved_robot_fk_required": False,
                "sidecar_sha256_by_session": sidecar_digests,
                "receipt_sha256_by_session": motion_receipt_digests,
                "motion_weight_by_session": motion_weights,
                "contact_aux_weight_by_session": contact_weights,
            },
            "ict_contract": {"hand_tracking_method": "hawor_v3", "single_hand": False, "ict_dim": 29,
                             "frame_mode": "camera_frame", "translation_unit": "metre",
                             "camera_coordinate_system": "x_right_y_down_z_forward",
                             "object_token_policy": "AUTO_ESTIMATED_GRADE_B_FROZEN_PER_SESSION"},
            "object_state_contract": {
                "consumption_mode": "training_estimated_grade_b", "sidecar_root": str((bundle / "object_state_sidecars").absolute()),
                "npz_sha256_by_session": object_npz_digests, "json_sha256_by_session": object_json_digests,
                "quality_grade_by_session": grades, "training_weight_by_session": weights,
                "confidence_thresholds": thresholds or {},
            },
        }
        selector = {"schema_version": "humanego-newtask-robot-selector-v1", "product_line": "RAW_RGB_WITH_SCOPED_MOTION_GRADE_B",
                    "image_name": "rgb.png", "artifact_root": str((bundle / "production").absolute()),
                    "selector_root": str((bundle / "production").absolute()), "sessions": selector_sessions}
        audit = {"schema_version": "exact78-motion-grade-b-bundle-audit-v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "task": task, "cohort": ref(COHORT), "admitted_counts": counts, "valid_h50_windows": windows,
                 "heldout_in_training": False, "robot_c_consumed": False, "test_is_optional_zero": True,
                 "rejected_sessions": rejected}
        payloads = {"split.json": split, "selector_records.json": selector, "paired_windows.json": paired_windows,
                    "sidecars.json": sidecar_digests, "hawor_v3_sidecars.json": hawor_digests, "AUDIT.json": audit}
        for name, payload in payloads.items(): write_json(temporary / name, payload)
        freeze = {"schema_version": "humanego-newtask-robot-freeze-v3-exact78-object-ict-motion-grade-b",
                  "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "task": task, "training_started": False,
                  "bundle_files": {name: ref(temporary / name, published=bundle / name) for name in payloads}}
        write_json(temporary / "freeze.json", freeze)
        os.replace(temporary, bundle)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--bundle", type=Path, required=True); parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(); admitted, rejected = inspect(args.task)
    report = {"task": args.task, "admitted_counts": {k: len(v) for k, v in admitted.items()},
              "valid_h50_windows": {k: sum(len(x["window_starts"]) for x in v) for k, v in admitted.items()},
              "rejected": rejected}
    report["ready"] = all(len(admitted[k]) >= v for k, v in MIN_SESSIONS.items()) and all(
        report["valid_h50_windows"][k] >= v for k, v in MIN_WINDOWS.items())
    print(json.dumps(report, ensure_ascii=False, indent=2));
    if not args.report_only: materialize(args.task, admitted, rejected, args.bundle.absolute())
    return 0


if __name__ == "__main__": raise SystemExit(main())
