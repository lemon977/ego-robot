#!/usr/bin/env python3
"""Promote digest-bound historical hand motion against fresh HaWoR wrists.

This is deliberately *not* a Robot success receipt.  It reuses only historical
KaiHand joint labels after finite/limit/continuity checks, discards the legacy
wrist coordinate frame, and binds the output wrist pose to the fresh HaWoR V2
production sidecar.  Object/contact supervision remains independently gated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / "tasks/control/runs/20260908_nine_hour_156_pipeline_v1"
COHORT = CONTROL / "COHORT_EXACT78_FROZEN.json"
COHORT_SHA256 = "d482ae4b627efa78653b69af26a357cec66381e3416dd78dd278504579e7790e"
OLD_BUNDLE = PROJECT / "HumanEgo/artifacts/newtask_robot_bundles/poker"
FRESH_ROOT = CONTROL / "training_inputs/poker/hawor_v3_sidecars"
OUTPUT_ROOT = CONTROL / "training_inputs/poker/motion_training_label_grade_b"
MAX_Q_STEP_RAD = 0.20
MAX_WRIST_STEP_M = 0.060
MAX_WRIST_STEP_DEG = 20.0
MIN_DUAL_RUN = 64


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def rotation_angle_deg(relative: np.ndarray) -> np.ndarray:
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def transition_valid(wrist: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    delta_m = np.linalg.norm(np.diff(wrist[..., :3, 3], axis=0), axis=-1)
    r0 = wrist[:-1, ..., :3, :3]
    r1 = wrist[1:, ..., :3, :3]
    relative = np.swapaxes(r0, -1, -2) @ r1
    delta_deg = rotation_angle_deg(relative)
    passed = (delta_m <= MAX_WRIST_STEP_M) & (delta_deg <= MAX_WRIST_STEP_DEG)
    return passed, {
        "wrist_translation_step_max_m": float(delta_m.max(initial=0.0)),
        "wrist_translation_step_p99_m": float(np.percentile(delta_m, 99)) if delta_m.size else 0.0,
        "wrist_rotation_step_max_deg": float(delta_deg.max(initial=0.0)),
        "wrist_rotation_step_p99_deg": float(np.percentile(delta_deg, 99)) if delta_deg.size else 0.0,
    }


def promote(session: str, role: str, raw_sha256: str, expected_old_digest: str) -> dict[str, Any]:
    old_path = OLD_BUNDLE / "sidecars/kai22" / session / "sidecar.npz"
    fresh_path = FRESH_ROOT / session / "entities_hawor_v3.npz"
    if not old_path.is_file() or not fresh_path.is_file():
        raise RuntimeError("required old/fresh sidecar missing")
    old_digest = digest(old_path)
    if old_digest != expected_old_digest:
        raise RuntimeError("historical sidecar digest drift")
    with np.load(old_path, allow_pickle=False) as old, np.load(fresh_path, allow_pickle=False) as fresh:
        old_names = np.asarray(old["frame_names"]).astype(str)
        fresh_names = np.asarray(fresh["frame_names"]).astype(str)
        if not np.array_equal(old_names, fresh_names):
            raise RuntimeError("old/fresh frame identity mismatch")
        q = np.asarray(old["q"], dtype=np.float32)
        lower = np.asarray(old["joint_lower"], dtype=np.float32)
        upper = np.asarray(old["joint_upper"], dtype=np.float32)
        old_valid = np.asarray(old["valid"], dtype=bool)
        old_confidence = np.asarray(old["confidence"], dtype=np.float32)
        # The q rows are physical KaiHand left/right.  The established
        # embodiment contract is crossed: human right -> physical left and
        # human left -> physical right.  Cross every fresh human-side signal
        # together; never pair physical-left q with human-left wrist by
        # accident.
        wrist = np.asarray(fresh["T_hand_to_camera"], dtype=np.float64)[:, ::-1]
        fresh_valid = np.asarray(fresh["valid"], dtype=bool)[:, ::-1]
        fresh_confidence = np.asarray(fresh["confidence"], dtype=np.float32)[:, ::-1]
        timestamps = np.asarray(old["timestamps_ns"], dtype=np.int64)
        joint_names = np.asarray(old["joint_names"])
        grasp = np.asarray(fresh["grasp"], dtype=np.float32)[:, ::-1]
        fresh_source_hawor = str(np.asarray(fresh["source_hawor_sha256"]).item())
        fresh_source_result = str(np.asarray(fresh["source_result_sha256"]).item())
    count = len(old_names)
    if q.shape != (count, 2, 22) or lower.shape != (2, 22) or upper.shape != (2, 22):
        raise RuntimeError("historical q shape contract failed")
    if wrist.shape != (count, 2, 4, 4) or old_valid.shape != fresh_valid.shape != (count, 2):
        raise RuntimeError("fresh wrist shape contract failed")
    if not np.isfinite(q).all() or not np.isfinite(wrist).all():
        raise RuntimeError("non-finite label")
    in_limits = (q >= lower[None] - 1e-6) & (q <= upper[None] + 1e-6)
    if not bool(in_limits.all()):
        raise RuntimeError("historical q violates pinned limits")
    q_step = np.max(np.abs(np.diff(q, axis=0)), axis=-1)
    q_transition = q_step <= MAX_Q_STEP_RAD
    wrist_transition, wrist_metrics = transition_valid(wrist)
    valid = old_valid & fresh_valid
    transition = q_transition & wrist_transition
    if count > 1:
        failed_transition = ~transition
        valid[:-1] &= ~failed_transition
        valid[1:] &= ~failed_transition
    confidence = np.minimum(old_confidence, fresh_confidence)
    confidence[~valid] = 0.0
    dual_run = longest_run(valid.all(axis=1))
    if dual_run < MIN_DUAL_RUN:
        raise RuntimeError(f"dual-valid continuity {dual_run}<{MIN_DUAL_RUN}")

    output_dir = OUTPUT_ROOT / session
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sidecar.npz"
    fd, temporary_name = tempfile.mkstemp(prefix=".sidecar.", suffix=".npz", dir=output_dir)
    os.close(fd)
    try:
        np.savez_compressed(
            temporary_name,
            schema_version=np.asarray("humanego-motion-label-grade-b-v1"),
            provenance_class=np.asarray("MOTION_TRAINING_LABEL_GRADE_B"),
            session_id=np.asarray(session),
            frame_names=old_names,
            timestamps_ns=timestamps,
            q=q,
            wrist_T_camera=wrist,
            valid=valid,
            confidence=confidence,
            grasp=grasp,
            joint_names=joint_names,
            joint_lower=lower,
            joint_upper=upper,
            translation_unit=np.asarray("metre"),
            q_unit=np.asarray("radian"),
            legacy_wrist_discarded=np.asarray(True),
            empirical_continuity_verified=np.asarray(True),
            ik_warm_start_used=np.asarray(False),
            achieved_robot_fk=np.asarray(False),
        )
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    receipt = {
        "schema_version": "humanego-motion-label-grade-b-receipt-v1",
        "status": "PASS_MOTION_TRAINING_LABEL_GRADE_B",
        "grade": "B",
        "may_train_motion": True,
        "consumption_authorized_for_final_robot": False,
        "session_id": session,
        "split": role,
        "cohort_sha256": COHORT_SHA256,
        "frame_identity": {
            "session_exact_match": True,
            "frame_names_old_equal_fresh": True,
            "frame_count": count,
            "canonical_raw_video_sha256": raw_sha256,
        },
        "sources": {
            "historical_q_hand": {"path": str(old_path), "sha256": old_digest},
            "fresh_hawor_wrist": {"path": str(fresh_path), "sha256": digest(fresh_path)},
            "fresh_hawor_production_sha256": fresh_source_hawor,
            "fresh_hawor_result_sha256": fresh_source_result,
            "side_mapping": "human_right_to_physical_left;human_left_to_physical_right",
            "legacy_wrist_T_camera": "DISCARDED_COORDINATE_FRAME_MISMATCH",
            "arm_label": "NOT_CONSUMED_BY_CURRENT_HUMANEGO_Q22_TARGET",
        },
        "gates": {
            "finite": True,
            "joint_limits": True,
            "empirical_branch_continuity": True,
            "max_q_step_rad": float(q_step.max(initial=0.0)),
            "max_allowed_q_step_rad": MAX_Q_STEP_RAD,
            "longest_consecutive_dual_valid": dual_run,
            "minimum_consecutive_dual_valid": MIN_DUAL_RUN,
            **wrist_metrics,
            "max_allowed_wrist_step_m": MAX_WRIST_STEP_M,
            "max_allowed_wrist_step_deg": MAX_WRIST_STEP_DEG,
        },
        "weights": {"motion": 0.50, "contact_aux": 0.0, "object_aux": 0.25},
        "limitations": [
            "NOT_A_FINAL_ROBOT_PRODUCT",
            "NO_CLAIM_OF_HAND_OBJECT_CONTACT_SUCCESS",
            "NO_CLAIM_OF_ARM_IK_SUCCESS",
            "IK_WARM_START_NOT_PROVEN_FOR_HISTORICAL_LABEL",
        ],
        "artifact": {"path": str(output_path), "sha256": digest(output_path), "bytes": output_path.stat().st_size},
    }
    atomic_json(output_dir / "RESULT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if digest(COHORT) != COHORT_SHA256:
        raise RuntimeError("cohort drift")
    cohort = read_json(COHORT)
    old_digests = read_json(OLD_BUNDLE / "sidecars.json")
    requested = set(args.session)
    rows = [row for row in cohort["sessions"] if row["task"] == "poker" and row["session_id"] in old_digests]
    if requested:
        rows = [row for row in rows if row["session_id"] in requested]
    report: dict[str, Any] = {"schema_version": "motion-grade-b-batch-v1", "passed": {}, "failed": {}}
    for row in rows:
        session = row["session_id"]
        try:
            receipt = promote(session, row["split"], row["raw_video_sha256"], old_digests[session])
            report["passed"][session] = {
                "split": row["split"],
                "longest_consecutive_dual_valid": receipt["gates"]["longest_consecutive_dual_valid"],
                "artifact_sha256": receipt["artifact"]["sha256"],
            }
        except Exception as error:
            report["failed"][session] = {"split": row["split"], "reason": str(error)}
    report["counts"] = {
        role: {
            "passed": sum(value["split"] == role for value in report["passed"].values()),
            "failed": sum(value["split"] == role for value in report["failed"].values()),
        }
        for role in ("train", "validation", "test", "heldout")
    }
    batch_path = OUTPUT_ROOT / "BATCH_REPORT.json"
    atomic_json(batch_path, report)
    print(json.dumps({"report": str(batch_path), **report["counts"]}, ensure_ascii=False, indent=2))
    return 0 if not report["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
