#!/usr/bin/env python3
"""Frozen analytic HaWoR MANO21 -> KaiHand q22 motion-label retarget.

This fail-forward producer intentionally performs no arm IK and makes no Robot
or contact-success claim.  It uses the already reviewed analytic closure map,
the historical KaiHand joint identity/limits as a digest-pinned contract, and
fresh world-consistent HaWoR wrists.  Its output may supervise the HumanEgo
q22 motion target only after the explicit Grade-B training gate accepts it.
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
REFERENCE = PROJECT / "HumanEgo/artifacts/newtask_robot_bundles/poker/sidecars/kai22/play_cards_0901_001/sidecar.npz"
REFERENCE_SHA256 = "c6464fb5c870e7324a784191816eeaf23d4d9c03c966bd34e572ec033670705e"
MAX_Q_STEP_RAD = 0.08
MAX_WRIST_STEP_M = 0.060
MAX_WRIST_STEP_DEG = 20.0
MIN_DUAL_RUN = 64
SIDE_NAMES = ("left", "right")
# Human anatomical side -> physical KaiHand side.
HUMAN_TO_PHYSICAL = (1, 0)
CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
GROUPS = {
    "thumb": tuple(range(0, 6)),
    "index": tuple(range(6, 10)),
    "middle": tuple(range(10, 14)),
    "ring": tuple(range(14, 18)),
    "pinky": tuple(range(18, 22)),
}


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
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def mapping_contract() -> dict[str, Any]:
    return {
        "schema_version": "hawor-kaihand-frozen-analytic-closure-v1",
        "formula": "closure=clip(mean(consecutive_bone_bend_rad)/(0.55*pi),0,1)",
        "initial_state": "joint_limit_midpoint",
        "temporal_projection": "previous_plus_clip(desired-previous,+/-0.08rad)",
        "human_to_physical": {"left": "right", "right": "left"},
        "chains": {key: list(value) for key, value in CHAINS.items()},
        "q_groups": {key: list(value) for key, value in GROUPS.items()},
        "joint_contract_source_sha256": REFERENCE_SHA256,
    }


def contract_digest() -> str:
    payload = json.dumps(mapping_contract(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def derive_q(joints: np.ndarray, observed: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise RuntimeError("MANO21 joint shape contract failed")
    count = joints.shape[1]
    previous = 0.5 * (lower + upper)
    output = np.empty((count, 2, 22), dtype=np.float32)
    for frame in range(count):
        current = previous.copy()
        for human, physical in enumerate(HUMAN_TO_PHYSICAL):
            points = joints[human, frame]
            # HaWoR preserves an explicit missing observation instead of
            # inventing coordinates.  Hold the last bounded q on those frames;
            # the corresponding output validity remains false below.
            if not observed[human, frame] or not np.isfinite(points).all():
                continue
            for finger, chain in CHAINS.items():
                bone = np.diff(points[np.asarray(chain)], axis=0)
                length = np.linalg.norm(bone, axis=1, keepdims=True)
                unit = bone / np.maximum(length, 1e-8)
                bend = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0))
                closure = float(np.clip(np.mean(bend) / (0.55 * np.pi), 0.0, 1.0))
                group = np.asarray(GROUPS[finger], dtype=np.int64)
                desired = lower[physical, group] + closure * (upper[physical, group] - lower[physical, group])
                delta = np.clip(desired - previous[physical, group], -MAX_Q_STEP_RAD, MAX_Q_STEP_RAD)
                current[physical, group] = np.clip(previous[physical, group] + delta, lower[physical, group], upper[physical, group])
        output[frame] = current
        previous = current
    return output


def promote(row: dict[str, Any], *, force: bool) -> dict[str, Any]:
    task, session, role = row["task"], row["session_id"], row["split"]
    output_dir = CONTROL / "training_inputs" / task / "motion_training_label_grade_b" / session
    existing = output_dir / "RESULT.json"
    if existing.is_file() and not force:
        receipt = read_json(existing)
        if receipt.get("status") == "PASS_MOTION_TRAINING_LABEL_GRADE_B":
            return receipt
    hawor_result_path = Path(row["processed_root"]) / "hawor/20260908_hawor_world_consistent_prod_v2/RESULT.json"
    hawor = read_json(hawor_result_path)
    if hawor.get("status") not in {"TERMINAL_GRADE_A", "TERMINAL_GRADE_B"} or not hawor.get("consumption_authorized"):
        raise RuntimeError("fresh HaWoR not consumption-authorized")
    if hawor.get("raw_video_sha256_authority") != row["raw_video_sha256"]:
        raise RuntimeError("fresh HaWoR/raw video identity mismatch")
    source_path = Path(hawor["source_npz"]["path"])
    if digest(source_path) != hawor["source_npz"]["sha256"]:
        raise RuntimeError("fresh HaWoR source digest drift")
    staged_path = CONTROL / "training_inputs" / task / "hawor_v3_sidecars" / session / "entities_hawor_v3.npz"
    if not staged_path.is_file():
        raise RuntimeError("fresh staged wrist missing")
    with np.load(REFERENCE, allow_pickle=False) as reference:
        lower = np.asarray(reference["joint_lower"], dtype=np.float32)
        upper = np.asarray(reference["joint_upper"], dtype=np.float32)
        joint_names = np.asarray(reference["joint_names"])
    with np.load(source_path, allow_pickle=False) as source:
        joints = np.asarray(source["joints_3d_camera"], dtype=np.float64)
        observed = np.asarray(source["observed"], dtype=bool)
        original_frames = np.asarray(source["original_frame_indices"], dtype=np.int64)
        fps = float(np.asarray(source["fps"]).item())
    with np.load(staged_path, allow_pickle=False) as staged:
        names = np.asarray(staged["frame_names"]).astype(str)
        wrist = np.asarray(staged["T_hand_to_camera"], dtype=np.float64)[:, ::-1]
        valid = np.asarray(staged["valid"], dtype=bool)[:, ::-1]
        confidence = np.asarray(staged["confidence"], dtype=np.float32)[:, ::-1]
        grasp = np.asarray(staged["grasp"], dtype=np.float32)[:, ::-1]
        source_result_sha = str(np.asarray(staged["source_result_sha256"]).item())
    expected_names = np.asarray([f"{int(value):05d}" for value in original_frames])
    if not np.array_equal(names, expected_names):
        raise RuntimeError("staged/source frame identity mismatch")
    q = derive_q(joints, observed, lower, upper)
    count = len(names)
    if q.shape != (count, 2, 22) or wrist.shape != (count, 2, 4, 4):
        raise RuntimeError("retarget output shape contract failed")
    q_step = np.max(np.abs(np.diff(q, axis=0)), axis=-1)
    wrist_delta = np.linalg.norm(np.diff(wrist[..., :3, 3], axis=0), axis=-1)
    r0, r1 = wrist[:-1, ..., :3, :3], wrist[1:, ..., :3, :3]
    wrist_angle = rotation_angle_deg(np.swapaxes(r0, -1, -2) @ r1)
    transition = (q_step <= MAX_Q_STEP_RAD + 1e-6) & (wrist_delta <= MAX_WRIST_STEP_M) & (wrist_angle <= MAX_WRIST_STEP_DEG)
    failed = ~transition
    if count > 1:
        valid[:-1] &= ~failed
        valid[1:] &= ~failed
    confidence[~valid] = 0.0
    dual_run = longest_run(valid.all(axis=1))
    if dual_run < MIN_DUAL_RUN:
        raise RuntimeError(f"dual-valid continuity {dual_run}<{MIN_DUAL_RUN}")
    in_limits = (q >= lower[None] - 1e-6) & (q <= upper[None] + 1e-6)
    if not np.isfinite(q).all() or not np.isfinite(wrist).all() or not bool(in_limits.all()):
        raise RuntimeError("finite/joint-limit gate failed")
    # A full-session sequence with L valid consecutive frames contains L-49 H50 windows.
    h50 = max(0, dual_run - 49)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sidecar.npz"
    fd, temporary_name = tempfile.mkstemp(prefix=".sidecar.", suffix=".npz", dir=output_dir)
    os.close(fd)
    try:
        np.savez_compressed(
            temporary_name,
            schema_version=np.asarray("humanego-motion-label-grade-b-v1"),
            provenance_class=np.asarray("MOTION_TRAINING_LABEL_GRADE_B"),
            session_id=np.asarray(session), frame_names=names,
            timestamps_ns=np.rint(original_frames / fps * 1e9).astype(np.int64),
            q=q, wrist_T_camera=wrist, valid=valid, confidence=confidence, grasp=grasp,
            joint_names=joint_names, joint_lower=lower, joint_upper=upper,
            translation_unit=np.asarray("metre"), q_unit=np.asarray("radian"),
            legacy_wrist_discarded=np.asarray(True), empirical_continuity_verified=np.asarray(True),
            ik_warm_start_used=np.asarray(False), achieved_robot_fk=np.asarray(False),
            retarget_contract_sha256=np.asarray(contract_digest()),
        )
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    receipt = {
        "schema_version": "humanego-motion-label-grade-b-receipt-v1",
        "status": "PASS_MOTION_TRAINING_LABEL_GRADE_B", "grade": "B",
        "may_train_motion": True, "consumption_authorized_for_final_robot": False,
        "task": task, "session_id": session, "split": role, "cohort_sha256": COHORT_SHA256,
        "method": "FROZEN_ANALYTIC_HAWOR_MANO21_TO_KAIHAND_Q22",
        "frame_identity": {"frame_names_equal_fresh_source": True, "frame_count": count,
                           "canonical_raw_video_sha256": row["raw_video_sha256"]},
        "sources": {
            "fresh_hawor": {"path": str(source_path), "sha256": digest(source_path)},
            "fresh_hawor_result": {"path": str(hawor_result_path), "sha256": digest(hawor_result_path)},
            "fresh_staged_wrist": {"path": str(staged_path), "sha256": digest(staged_path)},
            "fresh_staged_source_result_sha256": source_result_sha,
            "kaihand_joint_contract": {"path": str(REFERENCE), "sha256": REFERENCE_SHA256},
            "arm_label": "NOT_CONSUMED_BY_CURRENT_HUMANEGO_Q22_TARGET",
            "side_mapping": "human_right_to_physical_left;human_left_to_physical_right",
        },
        "retarget_contract": {**mapping_contract(), "sha256": contract_digest()},
        "gates": {
            "finite": True, "joint_limits": True, "empirical_branch_continuity": True,
            "max_q_step_rad": float(q_step.max(initial=0.0)), "max_allowed_q_step_rad": MAX_Q_STEP_RAD,
            "longest_consecutive_dual_valid": dual_run, "h50_windows_lower_bound": h50,
            "wrist_translation_step_max_m": float(wrist_delta.max(initial=0.0)),
            "wrist_rotation_step_max_deg": float(wrist_angle.max(initial=0.0)),
        },
        "weights": {"motion": 0.50, "contact_aux": 0.0, "object_aux": 0.25},
        "limitations": ["NOT_A_FINAL_ROBOT_PRODUCT", "NO_ARM_IK_LABEL", "NO_CONTACT_SUCCESS_CLAIM"],
        "artifact": {"path": str(output_path), "sha256": digest(output_path), "bytes": output_path.stat().st_size},
    }
    atomic_json(existing, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), action="append", default=[])
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if digest(COHORT) != COHORT_SHA256 or digest(REFERENCE) != REFERENCE_SHA256:
        raise RuntimeError("frozen cohort/joint contract digest drift")
    tasks = set(args.task or ("chips", "poker")); sessions = set(args.session)
    rows = [row for row in read_json(COHORT)["sessions"]
            if row["task"] in tasks and row["split"] != "heldout" and (not sessions or row["session_id"] in sessions)]
    report: dict[str, Any] = {"schema_version": "fresh-retarget-motion-grade-b-batch-v1", "passed": {}, "waiting": {}, "failed": {}}
    for row in rows:
        session = row["session_id"]
        try:
            result = promote(row, force=args.force)
            report["passed"][session] = {"task": row["task"], "split": row["split"],
                                         "dual_run": result["gates"]["longest_consecutive_dual_valid"]}
        except (FileNotFoundError, KeyError) as error:
            report["waiting"][session] = {"task": row["task"], "split": row["split"], "reason": str(error)}
        except Exception as error:
            reason = str(error)
            target = "waiting" if "missing" in reason.lower() or "not consumption-authorized" in reason else "failed"
            report[target][session] = {"task": row["task"], "split": row["split"], "reason": reason}
    report["counts"] = {}
    for task in sorted(tasks):
        report["counts"][task] = {role: sum(v["task"] == task and v["split"] == role for v in report["passed"].values())
                                  for role in ("train", "validation", "test")}
    batch = CONTROL / "training_inputs/motion_training_label_grade_b_fresh_retarget_BATCH_REPORT.json"
    atomic_json(batch, report)
    print(json.dumps({"report": str(batch), "counts": report["counts"], "waiting": len(report["waiting"]),
                      "failed": len(report["failed"])}, ensure_ascii=False, indent=2))
    return 2 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
