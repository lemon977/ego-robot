#!/usr/bin/env python3
"""Contract-bound heldout RTC inference for the exact78 chips/poker runs.

The formal output is deliberately prediction-only.  Heldout Robot/GT q is
neither loaded nor written.  The first robot state uses a neutral hand command
and the observed HaWoR wrist; later states are fed back only from already
executed checkpoint predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
HUMANEGO = PROJECT / "HumanEgo"
CONTROL = PROJECT / "tasks/control/runs/20260908_nine_hour_156_pipeline_v1"
CONTRACT = CONTROL / "HELDOUT_CHECKPOINT_PREDICTION_CONTRACT_V3.json"
COHORT = CONTROL / "COHORT_EXACT78_FROZEN.json"
COHORT_SHA256 = "d482ae4b627efa78653b69af26a357cec66381e3416dd78dd278504579e7790e"
HAWOR_INDEX = CONTROL / "HAWOR_TERMINAL_INDEX_V2.json"
STAGER = HUMANEGO / "tools/stage_exact78_grade_b_ict_session.py"
LEGACY_HELPER = PROJECT / "NOW/daemon/tools/visualize_newtask_unseen.py"
HORIZON, EXECUTION_HORIZON, INFERENCE_DELAY = 50, 10, 4
ATTENTION_SCHEDULE = "exp"
PHYSICAL_TO_HUMAN_SIDE = ("right", "left")
Q_ENSEMBLE_DECAY = 0.5
MAX_Q_STEP_RAD = float(np.deg2rad(10.0))
MAX_WRIST_STEP_M = 0.010
MAX_WRIST_ROTATION_STEP_RAD = float(np.deg2rad(8.0))
WRIST_EMA_ALPHA = 0.65
ACCEPTED_TRAINING_TERMINAL_STATUSES = {
    "COMPLETE", "TIME_BOUNDED_COMPLETE", "NATURAL_EARLY_STOP_VERIFIED",
}

sys.path.insert(0, str(HUMANEGO))
from inference.embodiment_policy import sample_receding_h50  # noqa: E402
from inference.receding_horizon import (  # noqa: E402
    CausalPoseRateLimiter,
    RecedingH50QController,
    reexpress_absolute_camera_action_chunk,
)
from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from utils.utils_math import normalize_o6d, normalize_pos, o6d_to_rotmat, rotmat_to_o6d  # noqa: E402


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def reference(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_sha_bound_local_checkpoint(
    checkpoint: Path, expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deserialize only bytes whose already-open descriptor matches authority."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError("HOLD_TRUSTED_CHECKPOINT_SHA_MISSING")
    resolved = checkpoint.resolve(strict=True)
    if resolved != checkpoint.absolute():
        raise RuntimeError("HOLD_TRUSTED_CHECKPOINT_SYMLINK_OR_PATH_ALIAS")
    with checkpoint.open("rb") as stream:
        state = os.fstat(stream.fileno())
        if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
            raise RuntimeError("HOLD_TRUSTED_CHECKPOINT_NOT_ORDINARY_NLINK1")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError("HOLD_TRUSTED_CHECKPOINT_SHA_MISMATCH")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("HOLD_CHECKPOINT_PAYLOAD_NOT_MAPPING")
    return payload, {
        "path": str(resolved), "bytes": state.st_size,
        "sha256": observed_sha256,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def cohort_rows(task: str) -> list[dict[str, Any]]:
    if sha256(COHORT) != COHORT_SHA256:
        raise RuntimeError("HOLD_EXACT78_COHORT_SHA_DRIFT")
    rows = [row for row in read_json(COHORT)["sessions"] if row["task"] == task]
    counts = {role: sum(row["split"] == role for row in rows)
              for role in ("train", "validation", "test", "heldout")}
    if counts != {"train": 60, "validation": 8, "test": 5, "heldout": 5}:
        raise RuntimeError(f"HOLD_CANONICAL_SPLIT_COUNTS:{counts}")
    return rows


def selected_session(task: str, requested: str | None, contract: dict) -> str:
    eligible = list(contract["tasks"][task]["eligible_heldout_sessions"])
    canonical = [row["session_id"] for row in cohort_rows(task)
                 if row["split"] == "heldout"]
    if eligible != canonical:
        raise RuntimeError("HOLD_CONTRACT_ELIGIBLE_HELDOUT_DRIFT")
    selected = requested or eligible[0]
    if selected not in eligible:
        raise RuntimeError(f"HOLD_NOT_CANONICAL_HELDOUT:{selected}")
    return selected


def training_leakage_audit(task: str, session: str) -> dict[str, Any]:
    checked: list[str] = []
    for split_path in sorted((HUMANEGO / "artifacts/newtask_robot_bundles").glob("*/split.json")):
        try:
            split = read_json(split_path)
        except Exception:
            continue
        if split.get("task") != task:
            continue
        admitted = split.get("admitted_splits", {})
        materialized = set(split.get("selector_sessions", []))
        for role in ("train", "validation", "test"):
            materialized.update(admitted.get(role, []))
        checked.append(str(split_path))
        if session in materialized:
            raise RuntimeError(f"HOLD_HELDOUT_IN_TRAINING_BUNDLE:{split_path}")
        for name in ("selector_records.json", "paired_windows.json", "sidecars.json"):
            path = split_path.parent / name
            if not path.is_file():
                continue
            payload = read_json(path)
            sessions = payload.get("sessions", payload)
            if isinstance(sessions, dict) and session in sessions:
                raise RuntimeError(f"HOLD_HELDOUT_MATERIALIZED_IN_TRAINING_BUNDLE:{path}")
    return {"heldout_in_training": False, "bundle_splits_checked": checked}


def validate_checkpoint_files(task: str, checkpoint: Path, session: str) -> dict[str, Any]:
    complete_path = checkpoint.parent / "training_complete.json"
    manifest_path = checkpoint.parent / "run_manifest.json"
    stats_path = checkpoint.parent / "dataset_stats.json"
    if checkpoint.name != "best.pt":
        raise RuntimeError("HOLD_CHECKPOINT_NOT_VALIDATION_SELECTED_BEST")
    if not all(path.is_file() for path in (
        checkpoint, complete_path, manifest_path, stats_path,
    )):
        return {
            "ready": False,
            "reason": "WAIT_TASK_CHECKPOINT_COMPLETE",
            "checkpoint_exists": checkpoint.is_file(),
            "training_complete_exists": complete_path.is_file(),
            "run_manifest_exists": manifest_path.is_file(),
            "dataset_stats_exists": stats_path.is_file(),
        }
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "humanego-newtask-robot-run-v3-exact78-object-ict":
        raise RuntimeError("HOLD_CHECKPOINT_NOT_EXACT78_V3")
    if manifest.get("task") != task:
        raise RuntimeError("HOLD_CHECKPOINT_TASK_MISMATCH")
    audit = manifest.get("exact78_split_audit", {})
    if audit.get("cohort_sha256") != COHORT_SHA256 or audit.get("heldout_in_training") is not False:
        raise RuntimeError("HOLD_CHECKPOINT_COHORT_BINDING")
    if session not in manifest.get("heldout_sessions", []):
        raise RuntimeError("HOLD_SELECTED_SESSION_NOT_CHECKPOINT_HELDOUT")
    consumed = (
        set(manifest.get("train_sessions", []))
        | set(manifest.get("validation_sessions", []))
        | set(manifest.get("test_sessions", []))
    )
    if session in consumed:
        raise RuntimeError("HOLD_SELECTED_SESSION_CONSUMED_BY_CHECKPOINT")
    complete = read_json(complete_path)
    completion_status = complete.get("status")
    if completion_status not in ACCEPTED_TRAINING_TERMINAL_STATUSES:
        raise RuntimeError(
            f"HOLD_TRAINING_COMPLETE_STATUS:{completion_status!r}; expected "
            "COMPLETE, TIME_BOUNDED_COMPLETE, or NATURAL_EARLY_STOP_VERIFIED"
        )
    stats_sha256 = sha256(stats_path)
    if manifest.get("dataset_stats_sha256") != stats_sha256:
        raise RuntimeError("HOLD_RUN_MANIFEST_DATASET_STATS_MISMATCH")
    verification = complete.get("verification", {})
    manifest_sha256 = sha256(manifest_path)
    if verification.get("run_manifest_sha256") != manifest_sha256:
        raise RuntimeError("HOLD_COMPLETION_RUN_MANIFEST_SHA_MISMATCH")
    if verification.get("dataset_stats_sha256") != stats_sha256:
        raise RuntimeError("HOLD_COMPLETION_DATASET_STATS_SHA_MISMATCH")
    # The watcher performs the same strict payload binding as formal inference;
    # the mere presence of best.pt and a completion marker is insufficient.
    checkpoint_payload, checkpoint_ref = load_sha_bound_local_checkpoint(
        checkpoint, verification.get("best_sha256"),
    )
    if checkpoint_payload.get("run_manifest") != manifest:
        raise RuntimeError("HOLD_CHECKPOINT_PAYLOAD_RUN_MANIFEST_MISMATCH")
    if checkpoint_payload.get("dataset_stats_sha256") != stats_sha256:
        raise RuntimeError("HOLD_CHECKPOINT_DATASET_STATS_MISMATCH")
    return {
        "ready": True,
        "completion_status": completion_status,
        "completion_scope": (
            "TIME_BOUNDED_AT_180_OR_EARLIER_POLICY"
            if completion_status == "TIME_BOUNDED_COMPLETE"
            else "NATURAL_EARLY_STOP"
        ),
        "checkpoint": checkpoint_ref,
        "training_complete": reference(complete_path),
        "run_manifest": reference(manifest_path),
        "dataset_stats": reference(stats_path),
        "manifest_payload": manifest,
    }


def preflight(task: str, requested: str | None, checkpoint_override: Path | None = None) -> dict[str, Any]:
    contract = read_json(CONTRACT)
    if contract.get("schema_version") != "exact78-heldout-checkpoint-prediction-v3":
        raise RuntimeError("HOLD_HELDOUT_CONTRACT_SCHEMA")
    if contract.get("cohort_sha256") != COHORT_SHA256:
        raise RuntimeError("HOLD_HELDOUT_CONTRACT_COHORT")
    rtc = contract.get("rtc", {})
    if rtc != {
        "pred_horizon": HORIZON,
        "execution_horizon": EXECUTION_HORIZON,
        "inference_delay": INFERENCE_DELAY,
        "attention_schedule": ATTENTION_SCHEDULE,
        "required": True,
    }:
        raise RuntimeError(f"HOLD_RTC_CONTRACT_DRIFT:{rtc}")
    session = selected_session(task, requested, contract)
    expected_checkpoint = Path(contract["tasks"][task]["checkpoint"]).absolute()
    checkpoint = (checkpoint_override or expected_checkpoint).absolute()
    if checkpoint != expected_checkpoint:
        raise RuntimeError("HOLD_CHECKPOINT_PATH_DIFFERS_FROM_V3_CONTRACT")
    leakage = training_leakage_audit(task, session)
    checkpoint_gate = validate_checkpoint_files(task, checkpoint, session)
    return {
        "schema_version": "exact78-heldout-rtc-preflight-v1",
        "status": "READY" if checkpoint_gate["ready"] else "WAIT_CHECKPOINT",
        "task": task,
        "selected_session": session,
        "canonical_role": "heldout",
        "checkpoint_path": str(checkpoint),
        "cohort_sha256": COHORT_SHA256,
        "rtc": rtc,
        "training_isolation": leakage,
        "checkpoint_gate": checkpoint_gate,
        "input_policy": {
            "rgb": "canonical raw rgb.png",
            "ict": "fresh HaWoR plus auto Object Grade-B, staged after checkpoint freeze",
            "initial_robot_state": "HaWoR anatomical wrist crossed into frozen physical Kai left/right slots plus joint-limit neutral q",
            "later_robot_state": "causal feedback from q/pose safety-filtered checkpoint prefix",
            "physical_to_human_side": list(PHYSICAL_TO_HUMAN_SIDE),
            "raw_model_plan_preserved": True,
            "heldout_robot_or_gt_q_loaded": False,
        },
    }


def find_hawor(task: str, session: str) -> tuple[Path, Path]:
    records = read_json(HAWOR_INDEX).get("records", {})
    if isinstance(records, dict):
        row = records.get(session)
    elif isinstance(records, list):
        row = next(
            (value for value in records
             if isinstance(value, dict) and value.get("session_id") == session),
            None,
        )
    else:
        row = None
    if not isinstance(row, dict):
        raise RuntimeError(f"WAIT_HELDOUT_HAWOR:{session}")
    result = Path(row.get("result", {}).get("path", ""))
    if row.get("task") != task or row.get("split") != "heldout" or not row.get("consumption_authorized"):
        raise RuntimeError(f"HOLD_HELDOUT_HAWOR_AUTHORITY:{session}")
    if not result.is_file() or sha256(result) != row["result"].get("sha256"):
        raise RuntimeError(f"HOLD_HELDOUT_HAWOR_DIGEST:{session}")
    archive = result.parent / "HAWOR_WORLD_CONSISTENT_MANO21.npz"
    return result.resolve(strict=True), archive.resolve(strict=True)


def schema_from_training_manifest(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    sidecars = manifest.get("sidecars", {})
    if not sidecars:
        raise RuntimeError("HOLD_TRAINING_SIDECAR_SCHEMA_MISSING")
    first_session = sorted(sidecars)[0]
    root = Path(manifest["bundle"]) / "sidecars/kai22" / first_session / "sidecar.npz"
    if not root.is_file() or sha256(root) != sidecars[first_session]:
        raise RuntimeError("HOLD_TRAINING_SIDECAR_SCHEMA_DIGEST")
    with np.load(root, allow_pickle=False) as archive:
        return {
            "joint_names": np.asarray(archive["joint_names"]),
            "joint_lower": np.asarray(archive["joint_lower"], dtype=np.float32),
            "joint_upper": np.asarray(archive["joint_upper"], dtype=np.float32),
        }


def build_prediction_only_inputs(task: str, session: str, raw_path: Path,
                                 hawor_result: Path, hawor_archive: Path,
                                 manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    """Stage compact, digest-bound observations without any heldout Robot q."""
    stager = _module("heldout_stager", STAGER)
    hawor_root = root / "hawor_v3_sidecars"
    hawor_row = stager.hawor_builder.build_session(
        session=session, task=task, base_bundle=root, destination=hawor_root,
        mps_path=raw_path, hawor_archive_path=hawor_archive,
        hawor_result_path=hawor_result,
    )
    object_npz, object_json = stager.build_object_sidecar(
        task=task, session=session, hawor_npz=hawor_archive,
        hawor_result=hawor_result, output_root=root,
    )
    hawor_sidecar = hawor_root / session / "entities_hawor_v3.npz"
    with np.load(hawor_sidecar, allow_pickle=False) as archive:
        names = np.asarray(archive["frame_names"]).astype(str)
        # Training supervision is indexed by physical KaiHand side.  The
        # frozen retarget contract maps human anatomical L/R to physical R/L,
        # so heldout bootstrap observations must use the same reversal.
        wrist = np.asarray(archive["T_hand_to_camera"], dtype=np.float64)[:, ::-1]
        valid = np.asarray(archive["valid"], dtype=bool)[:, ::-1]
        confidence = np.asarray(archive["confidence"], dtype=np.float32)[:, ::-1]
        grasp = np.asarray(archive["grasp"], dtype=np.float32)[:, ::-1]
    schema = schema_from_training_manifest(manifest)
    lower, upper = schema["joint_lower"], schema["joint_upper"]
    neutral = np.clip(np.zeros_like(lower), lower, upper)
    q = np.broadcast_to(neutral[None], (len(names), 2, 22)).copy()
    # This is an observation bootstrap, not a heldout motion label.  No Robot
    # or GT q path is opened anywhere in this builder.
    robot_root = root / "causal_bootstrap_sidecars/kai22" / session
    robot_root.mkdir(parents=True, exist_ok=False)
    robot_sidecar = robot_root / "sidecar.npz"
    np.savez_compressed(
        robot_sidecar,
        schema_version=np.asarray("humanego-robot-sidecar-v1"),
        embodiment=np.asarray("kai22"),
        provenance_class=np.asarray("HELDOUT_NO_GT_Q"),
        session_id=np.asarray(session), frame_names=names,
        timestamps_ns=np.arange(len(names), dtype=np.int64), q=q,
        wrist_T_camera=wrist, valid=valid, confidence=confidence, grasp=grasp,
        joint_names=schema["joint_names"], joint_lower=lower, joint_upper=upper,
        translation_unit=np.asarray("metre"), q_unit=np.asarray("radian"),
        physical_to_human_side=np.asarray(PHYSICAL_TO_HUMAN_SIDE),
        heldout_robot_or_gt_q_loaded=np.asarray(False),
    )
    raw_all_data = raw_path / "preprocess/all_data"
    adapter = root / "production" / session / "09_humanego_adapter"
    all_data = adapter / "preprocess/all_data"
    frames: dict[str, Any] = {}
    for name in names:
        source = raw_all_data / str(name)
        destination = all_data / str(name)
        destination.mkdir(parents=True, exist_ok=False)
        metadata = destination / "training_data.json"
        image = destination / "rgb.png"
        shutil.copyfile(source / "training_data.json", metadata)
        shutil.copyfile(source / "rgb.png", image)
        frames[str(name)] = {
            "metadata": reference(metadata), "image": reference(image),
            "unresolved": False,
        }
    selector = {
        "schema_version": "exact78-heldout-selector-v1", "product_line": "RAW_RGB_HELDOUT",
        "image_name": "rgb.png", "artifact_root": str(root / "production"),
        "selector_root": str(root / "production"), "sessions": {session: {"frames": frames}},
    }
    selector_path = root / "heldout_selector_records.json"
    atomic_json(selector_path, selector)
    receipt = {
        "schema_version": "exact78-heldout-prediction-input-v1",
        "created_after_checkpoint_freeze": True,
        "task": task, "session": session, "canonical_role": "heldout",
        "source": {"raw_path": str(raw_path), "hawor": reference(hawor_archive),
                   "hawor_result": reference(hawor_result)},
        "selector": reference(selector_path), "hawor_sidecar": reference(hawor_sidecar),
        "object_npz": reference(object_npz), "object_json": reference(object_json),
        "causal_bootstrap_sidecar": reference(robot_sidecar),
        "heldout_robot_or_gt_q_loaded": False,
        "prediction_target_available": False,
        "hawor_row": hawor_row,
    }
    receipt_path = root / "HELDOUT_INPUT_RECEIPT.json"
    atomic_json(receipt_path, receipt)
    return {**receipt, "receipt": reference(receipt_path), "selector_payload": selector,
            "hawor_root": str(hawor_root), "object_root": str(root / "object_state_sidecars"),
            "robot_root": str(root / "causal_bootstrap_sidecars")}


def make_dataset(cfg, raw_path: Path, session: str, stats: dict,
                 staged: dict[str, Any]) -> FlowMatchingDataloader:
    selector = staged["selector_payload"]
    frames = set(
        int(value)
        for value in list(selector["sessions"][session]["frames"])[0:-1]
    )
    return FlowMatchingDataloader(
        sessions=[MPSSessions(str(Path(staged["selector_payload"]["selector_root"]) / session / "09_humanego_adapter"))], image_size=cfg.image_size,
        pred_horizon=1, single_hand=False, max_ict=cfg.max_ict,
        img_name="rgb.png", centric_mode=cfg.centric_mode,
        frame_mode="camera_frame", action_mode="absolute",
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=staged["robot_root"],
        hawor_v3_sidecar_root=staged["hawor_root"],
        hawor_v3_sha256_by_session={session: staged["hawor_sidecar"]["sha256"]},
        object_state_sidecar_root=staged["object_root"],
        object_state_npz_sha256_by_session={session: staged["object_npz"]["sha256"]},
        object_state_json_sha256_by_session={session: staged["object_json"]["sha256"]},
        object_state_consumption_mode="training_estimated_grade_b",
        object_state_confidence_thresholds={
            "AUTO_HAWOR_FINGERTIP_TASK_GEOMETRY": 0.18,
            "AUTO_TEMPORAL_INTERPOLATED_HAWOR_TASK_GEOMETRY": 0.18,
        },
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False, enable_augmentation=False,
        hand_tracking_method="hawor_v3", use_legacy_image_loading=False,
        cache_json_in_memory=True, stats=stats, allowed_window_starts={session: frames},
        selector_records={session: selector["sessions"][session]["frames"]},
        selector_root=staged["selector_payload"]["selector_root"],
        sidecar_sha256_by_session={session: staged["causal_bootstrap_sidecar"]["sha256"]},
    )


def observation(dataset: FlowMatchingDataloader, index: int, state: np.ndarray,
                state_mask: np.ndarray, device: str) -> tuple[dict[str, torch.Tensor], dict, np.ndarray, np.ndarray]:
    path = dataset.samples[index]
    frame = dataset._read_frame(path)
    transform = dataset._get_T_w2ref(frame)
    ict, pcd, mask = dataset._build_ict(frame, transform)
    image = dataset._load_image_tensor(frame, index)
    metadata = frame["metadata"]
    c2w = np.asarray(metadata["c2w"], dtype=np.float32)
    intrinsic = np.asarray(metadata["k"], dtype=np.float32).reshape(3, 3)
    anchor_uv = np.asarray([0.5, 0.5], dtype=np.float32)
    anchor = metadata.get("anchor_key", "obj1")
    objects = frame.get("entities", {}).get("objects", {}) or {}
    if anchor in objects:
        position = np.asarray(objects[anchor]["T_obj_to_world"], dtype=np.float32)[:3, 3]
        point = np.linalg.inv(c2w) @ np.r_[position, 1.0]
        if point[2] > 1e-6:
            anchor_uv = np.asarray([
                (intrinsic[0, 0] * point[0] / point[2] + intrinsic[0, 2]) / (intrinsic[0, 2] * 2),
                (intrinsic[1, 1] * point[1] / point[2] + intrinsic[1, 2]) / (intrinsic[1, 2] * 2),
            ], dtype=np.float32)
    values = {
        "x_rgb": image, "x_ict": torch.from_numpy(ict),
        "ict_mask": torch.from_numpy(mask), "x_robot_state": torch.from_numpy(state),
        "robot_state_mask": torch.from_numpy(state_mask),
        "anchor_uv": torch.from_numpy(anchor_uv),
    }
    if dataset.use_pcd_features:
        values["x_pcd"] = torch.from_numpy(pcd)
    return ({key: value.unsqueeze(0).to(device) for key, value in values.items()}, metadata, c2w, intrinsic)


def state_from_hawor(dataset: FlowMatchingDataloader, index: int, neutral_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset._read_frame(dataset.samples[index])
    hands = frame.get("entities", {}).get("hands_hawor_v3", {}) or {}
    state = np.zeros((2, 33), dtype=np.float32)
    mask = np.zeros(2, dtype=bool)
    # State/output slots are physical KaiHand left/right, while HaWoR keys are
    # human anatomical sides.  Match the frozen training retarget mapping.
    for hand, side in enumerate(PHYSICAL_TO_HUMAN_SIDE):
        if side not in hands:
            continue
        pose = np.asarray(hands[side]["T_hand_to_world"], dtype=np.float64)
        ref = dataset._get_T_w2ref(frame) @ pose
        state[hand] = np.concatenate((
            normalize_pos(ref[:3, 3], dataset.pos_mean, dataset.pos_std),
            normalize_o6d(rotmat_to_o6d(ref[:3, :3])), neutral_q[hand], [0.5, 1.0],
        ))
        mask[hand] = True
    if not mask.any():
        raise RuntimeError("HOLD_BOOTSTRAP_NO_HAWOR_HAND")
    return state, mask


def feedback_state(executed: np.ndarray, c2w: np.ndarray, dataset: FlowMatchingDataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last = executed[-1]
    world = np.zeros((2, 4, 4), dtype=np.float64)
    state = np.zeros((2, 33), dtype=np.float32)
    for hand in range(2):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = last[hand * 3:hand * 3 + 3] * dataset.pos_std + dataset.pos_mean
        pose[:3, :3] = o6d_to_rotmat(last[6 + hand * 6:12 + hand * 6])
        world[hand] = c2w @ pose
        state[hand] = np.concatenate((last[hand * 3:hand * 3 + 3],
                                      last[6 + hand * 6:12 + hand * 6],
                                      last[18 + hand * 22:18 + (hand + 1) * 22],
                                      [1.0, 1.0]))
    return state, np.ones(2, dtype=bool), world


def initialize_pose_controller(
    controller: CausalPoseRateLimiter,
    state: np.ndarray,
    state_mask: np.ndarray,
    c2w: np.ndarray,
    dataset: FlowMatchingDataloader,
) -> None:
    """Seed the causal limiter from the current physical-side robot state."""
    initial_world = []
    for hand in range(2):
        pose = np.eye(4, dtype=np.float64)
        if bool(state_mask[hand]):
            pose[:3, 3] = state[hand, :3] * dataset.pos_std + dataset.pos_mean
            pose[:3, :3] = o6d_to_rotmat(state[hand, 3:9])
        initial_world.append(np.asarray(c2w, dtype=np.float64) @ pose)
    controller.process(np.asarray(initial_world), np.asarray(state_mask, dtype=bool))


def safety_filter_prefix(
    *,
    plan: np.ndarray,
    frame_number: int,
    current_state: np.ndarray,
    current_mask: np.ndarray,
    c2w: np.ndarray,
    dataset: FlowMatchingDataloader,
    schema: dict[str, np.ndarray],
    q_controller: RecedingH50QController,
    pose_controller: CausalPoseRateLimiter,
    pose_initialized: bool,
) -> tuple[np.ndarray, dict[str, float], bool]:
    """Apply the already-tested q/pose execution gates to one raw H50 plan."""
    if not pose_initialized:
        initialize_pose_controller(
            pose_controller, current_state, current_mask, c2w, dataset,
        )
        pose_initialized = True
    executed, diagnostics = q_controller.process(
        replan_frame=frame_number,
        plan_action=plan,
        current_q=current_state[:, 9:31],
        joint_lower=schema["joint_lower"],
        joint_upper=schema["joint_upper"],
    )
    for offset in range(EXECUTION_HORIZON):
        target_world = []
        for hand in range(2):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = (
                executed[offset, hand * 3:hand * 3 + 3]
                * dataset.pos_std + dataset.pos_mean
            )
            pose[:3, :3] = o6d_to_rotmat(
                executed[offset, 6 + hand * 6:12 + hand * 6]
            )
            target_world.append(np.asarray(c2w, dtype=np.float64) @ pose)
        smooth_world, pose_diagnostics = pose_controller.process(
            np.asarray(target_world), np.ones(2, dtype=bool),
        )
        world_to_ref = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
        for hand in range(2):
            smooth_ref = world_to_ref @ smooth_world[hand]
            executed[offset, hand * 3:hand * 3 + 3] = (
                smooth_ref[:3, 3] - dataset.pos_mean
            ) / dataset.pos_std
            executed[offset, 6 + hand * 6:12 + hand * 6] = rotmat_to_o6d(
                smooth_ref[:3, :3]
            )
        for key, value in pose_diagnostics.items():
            diagnostics[f"pose_{key}"] = diagnostics.get(
                f"pose_{key}", 0.0
            ) + float(value)
    diagnostics["pose_translation_step_m_max_contract"] = MAX_WRIST_STEP_M
    diagnostics["pose_rotation_step_deg_max_contract"] = float(
        np.degrees(MAX_WRIST_ROTATION_STEP_RAD)
    )
    return executed, diagnostics, pose_initialized


def reexpress_state(state: np.ndarray, world: np.ndarray, c2w: np.ndarray,
                    dataset: FlowMatchingDataloader) -> np.ndarray:
    result = state.copy()
    w2c = np.linalg.inv(c2w)
    for hand in range(2):
        pose = w2c @ world[hand]
        result[hand, :3] = normalize_pos(pose[:3, 3], dataset.pos_mean, dataset.pos_std)
        result[hand, 3:9] = normalize_o6d(rotmat_to_o6d(pose[:3, :3]))
    return result


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
                  Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    path = next((path for path in candidates if path.is_file()), None)
    return ImageFont.truetype(str(path), size) if path else ImageFont.load_default()


def render_panel(rgb: np.ndarray, plan: np.ndarray, pos_mean: np.ndarray,
                 pos_std: np.ndarray, intrinsic: np.ndarray, frame: int,
                 rtc: bool) -> np.ndarray:
    panel = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (960, 540))
    source_width = max(1.0, float(intrinsic[0, 2]) * 2.0)
    source_height = max(1.0, float(intrinsic[1, 2]) * 2.0)
    positions = plan[:, :6].reshape(len(plan), 2, 3) * pos_std + pos_mean
    for hand, color in ((0, (255, 120, 30)), (1, (30, 60, 255))):
        xyz = positions[:, hand]
        uv = np.stack((intrinsic[0, 0] * xyz[:, 0] / np.maximum(xyz[:, 2], 1e-6) + intrinsic[0, 2],
                       intrinsic[1, 1] * xyz[:, 1] / np.maximum(xyz[:, 2], 1e-6) + intrinsic[1, 2]), axis=-1)
        uv[:, 0] *= 960 / source_width
        uv[:, 1] *= 540 / source_height
        visible = (xyz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 960) & (uv[:, 1] >= 72) & (uv[:, 1] < 540)
        points = uv[visible].astype(np.int32)
        if len(points) > 1:
            cv2.polylines(panel, [points[:, None]], False, color, 3, cv2.LINE_AA)
        for point in points[::5]:
            cv2.circle(panel, tuple(point), 5, color, -1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (960, 72), (0, 0, 0), -1)
    pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((14, 7), "安全执行前缀 + RTC" if rtc else "安全执行前缀（首段）", font=font(25), fill=(255, 255, 255))
    draw.text((14, 40), f"帧 {frame:05d}｜蓝=Kai左(人右) 红=Kai右(人左)｜无真值", font=font(18), fill=(220, 220, 220))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def predict(task: str, pre: dict[str, Any], output: Path, max_replans: int, fps: int, device: str) -> dict[str, Any]:
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    session = pre["selected_session"]
    row = next(row for row in cohort_rows(task) if row["session_id"] == session)
    raw = Path(row["raw_path"]).resolve(strict=True)
    checkpoint = Path(pre["checkpoint_path"])
    manifest = pre["checkpoint_gate"]["manifest_payload"]
    hawor_result, hawor_archive = find_hawor(task, session)
    helper = _module("heldout_checkpoint_helper", LEGACY_HELPER)
    for key, path in (
        ("run_manifest", checkpoint.parent / "run_manifest.json"),
        ("dataset_stats", checkpoint.parent / "dataset_stats.json"),
        ("training_complete", checkpoint.parent / "training_complete.json"),
    ):
        if reference(path) != pre["checkpoint_gate"][key]:
            raise RuntimeError(f"HOLD_PREDICT_PREFLIGHT_{key.upper()}_DRIFT")
    frozen_checkpoint = pre["checkpoint_gate"]["checkpoint"]
    if str(checkpoint.resolve(strict=True)) != frozen_checkpoint.get("path"):
        raise RuntimeError("HOLD_PREDICT_PREFLIGHT_CHECKPOINT_PATH_DRIFT")
    payload, current_checkpoint = load_sha_bound_local_checkpoint(
        checkpoint, frozen_checkpoint.get("sha256"),
    )
    if current_checkpoint != frozen_checkpoint:
        raise RuntimeError("HOLD_PREDICT_PREFLIGHT_CHECKPOINT_DRIFT")
    if payload.get("run_manifest") != manifest:
        raise RuntimeError("HOLD_CHECKPOINT_PAYLOAD_RUN_MANIFEST_MISMATCH")
    stats_path = checkpoint.parent / "dataset_stats.json"
    if payload.get("dataset_stats_sha256") != sha256(stats_path):
        raise RuntimeError("HOLD_CHECKPOINT_DATASET_STATS_MISMATCH")
    stats = read_json(stats_path)
    input_root = output / "heldout_eval_input"
    input_root.mkdir()
    staged = build_prediction_only_inputs(task, session, raw, hawor_result,
                                          hawor_archive, manifest, input_root)
    cfg = helper.restore_checkpoint_config(
        payload, eval_paths=[str(raw)], sidecar_root=staged["robot_root"],
        output=output, device=device,
    )
    dataset = make_dataset(cfg, raw, session, stats, staged)
    schedule = [index for index, path in enumerate(dataset.samples)
                if int(Path(path).parent.name) % EXECUTION_HORIZON == 0]
    schedule = schedule[:max_replans]
    if len(schedule) < 2:
        raise RuntimeError("HOLD_HELDOUT_HAS_FEWER_THAN_TWO_RTC_REPLANS")
    model = helper.build_model(cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    del payload
    schema = schema_from_training_manifest(manifest)
    neutral = np.clip(np.zeros_like(schema["joint_lower"]), schema["joint_lower"], schema["joint_upper"])
    state, state_mask = state_from_hawor(dataset, schedule[0], neutral)
    q_controller = RecedingH50QController(
        22, execute_steps=EXECUTION_HORIZON,
        ensemble_decay=Q_ENSEMBLE_DECAY, max_q_step_rad=MAX_Q_STEP_RAD,
    )
    pose_controller = CausalPoseRateLimiter(
        max_translation_step_m=MAX_WRIST_STEP_M,
        max_rotation_step_rad=MAX_WRIST_ROTATION_STEP_RAD,
        ema_alpha=WRIST_EMA_ALPHA,
    )
    pose_initialized = False
    state_world = None
    previous_plan = None
    previous_c2w = None
    previous_frame = None
    plans = []
    raw_prefixes = []
    prefixes = []
    safety_diagnostics = []
    frames = []
    rtc_flags = []
    c2ws = []
    images = []
    for index in schedule:
        frame_number = int(Path(dataset.samples[index]).parent.name)
        frame_payload = dataset._read_frame(dataset.samples[index])
        c2w_now = np.asarray(frame_payload["metadata"]["c2w"], dtype=np.float32)
        if state_world is not None:
            state = reexpress_state(state, state_world, c2w_now, dataset)
        obs, metadata, c2w, intrinsic = observation(dataset, index, state, state_mask, device)
        contiguous = previous_frame is not None and frame_number == previous_frame + EXECUTION_HORIZON
        apply_rtc = previous_plan is not None and contiguous
        remainder = None
        if apply_rtc:
            remainder = reexpress_absolute_camera_action_chunk(
                previous_plan[:, EXECUTION_HORIZON:], source_c2w=previous_c2w,
                target_c2w=c2w, pos_mean=dataset.pos_mean,
                pos_std=dataset.pos_std, hand_command_dim=22,
            )
        result = sample_receding_h50(
            model, obs, seed=int(cfg.seed) + frame_number,
            execute_steps=EXECUTION_HORIZON, steps=int(cfg.num_inference_steps),
            aligned_previous_action_remainder=remainder,
            inference_delay=INFERENCE_DELAY if apply_rtc else 0,
            max_guidance_weight=10.0, rtc_attention_schedule=ATTENTION_SCHEDULE,
        )
        plan_tensor = result["plan_action"].detach()
        plan = plan_tensor[0].cpu().numpy().astype(np.float32)
        raw_prefix = result["action"][0].detach().cpu().numpy().astype(np.float32)
        executed, diagnostic, pose_initialized = safety_filter_prefix(
            plan=plan, frame_number=frame_number, current_state=state,
            current_mask=state_mask, c2w=c2w, dataset=dataset, schema=schema,
            q_controller=q_controller, pose_controller=pose_controller,
            pose_initialized=pose_initialized,
        )
        state, state_mask, state_world = feedback_state(executed, c2w, dataset)
        source = (obs["x_rgb"][0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        images.append(render_panel(source, executed, dataset.pos_mean, dataset.pos_std,
                                   intrinsic, frame_number, apply_rtc))
        plans.append(plan)
        raw_prefixes.append(raw_prefix)
        prefixes.append(executed)
        safety_diagnostics.append(diagnostic)
        frames.append(frame_number)
        rtc_flags.append(apply_rtc)
        c2ws.append(c2w)
        previous_plan, previous_c2w, previous_frame = plan_tensor, c2w, frame_number
    prediction_path = output / "heldout_checkpoint_predictions.npz"
    np.savez_compressed(
        prediction_path, schema_version=np.asarray("exact78-heldout-prediction-only-v2"),
        task=np.asarray(task), session=np.asarray(session),
        checkpoint_full_plan_action=np.stack(plans),
        raw_model_prefix=np.stack(raw_prefixes),
        causally_executed_prefix=np.stack(prefixes),
        observation_frame=np.asarray(frames, dtype=np.int64),
        observation_c2w=np.stack(c2ws), rtc_applied=np.asarray(rtc_flags, dtype=bool),
        pred_horizon=np.asarray([HORIZON]), execution_horizon=np.asarray([EXECUTION_HORIZON]),
        inference_delay=np.asarray([INFERENCE_DELAY]),
        contains_ground_truth=np.asarray(False), contains_heldout_robot_q=np.asarray(False),
        state_feedback_policy=np.asarray("safety_filtered_checkpoint_prefix_only"),
        execution_safety_applied=np.asarray(True),
        q_unit=np.asarray("radian"),
        joint_lower=schema["joint_lower"], joint_upper=schema["joint_upper"],
        physical_to_human_side=np.asarray(PHYSICAL_TO_HUMAN_SIDE),
        q_raw_joint_limit_ratio=np.asarray([
            item["raw_joint_limit_ratio"] for item in safety_diagnostics
        ], dtype=np.float32),
        q_step_clip_ratio=np.asarray([
            item["q_step_clip_ratio"] for item in safety_diagnostics
        ], dtype=np.float32),
    )
    video = output / "heldout_checkpoint_prediction.mp4"
    temporary_frames = Path(tempfile.mkdtemp(prefix=".heldout_frames.", dir=output))
    try:
        for index, image in enumerate(images):
            cv2.imwrite(str(temporary_frames / f"{index:05d}.png"), image)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
                        "-i", str(temporary_frames / "%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(video)], check=True)
    finally:
        for path in temporary_frames.glob("*.png"):
            path.unlink()
        temporary_frames.rmdir()
    result = {
        "schema_version": "exact78-heldout-checkpoint-prediction-result-v1",
        "status": "PREDICTIONS_RENDERED", "task": task, "session": session,
        "canonical_role": "heldout", "checkpoint": reference(checkpoint),
        "checkpoint_training_complete": pre["checkpoint_gate"]["training_complete"],
        "checkpoint_run_manifest": pre["checkpoint_gate"]["run_manifest"],
        "cohort": reference(COHORT), "heldout_input_receipt": staged["receipt"],
        "prediction": reference(prediction_path), "video": reference(video),
        "rtc": read_json(CONTRACT)["rtc"], "replans": len(plans),
        "rtc_guided_replans": int(sum(rtc_flags)),
        "prediction_bytes_origin": "checkpoint model output plus causal RTC",
        "execution_safety": {
            "applied_to_causally_executed_prefix": True,
            "raw_model_prefix_preserved": True,
            "q_unit": "radian",
            "q_ensemble_decay": Q_ENSEMBLE_DECAY,
            "max_q_step_deg": float(np.degrees(MAX_Q_STEP_RAD)),
            "max_wrist_step_mm": MAX_WRIST_STEP_M * 1000.0,
            "max_wrist_rotation_step_deg": float(np.degrees(MAX_WRIST_ROTATION_STEP_RAD)),
            "wrist_ema_alpha": WRIST_EMA_ALPHA,
        },
        "side_mapping": {
            "action_slots": ["kai_physical_left", "kai_physical_right"],
            "physical_to_human_anatomical": list(PHYSICAL_TO_HUMAN_SIDE),
        },
        "heldout_robot_or_gt_q_loaded": False, "ground_truth_written": False,
        "legacy_robot_or_fourlane_video_copied": False,
    }
    manifest_path = output / "heldout_prediction_manifest.json"
    atomic_json(manifest_path, result)
    return {**result, "manifest": reference(manifest_path)}


def validate_prediction_npz_schema(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        attestation = {"contains_ground_truth", "contains_heldout_robot_q"}
        forbidden = {
            key for key in keys - attestation
            if any(token in key.lower() for token in
                   ("target", "ground_truth_action", "gt_action", "y_action", "robot_q"))
        }
        required = {"checkpoint_full_plan_action", "causally_executed_prefix",
                    "observation_frame", "rtc_applied", "contains_ground_truth",
                    "contains_heldout_robot_q"}
        if forbidden or not required.issubset(keys):
            raise RuntimeError(f"HOLD_PREDICTION_SCHEMA:forbidden={sorted(forbidden)} missing={sorted(required-keys)}")
        if bool(archive["contains_ground_truth"].item()) or bool(archive["contains_heldout_robot_q"].item()):
            raise RuntimeError("HOLD_PREDICTION_CONTAINS_HELDOUT_LABEL")
        if archive["checkpoint_full_plan_action"].shape[1:] != (HORIZON, 62):
            raise RuntimeError("HOLD_PREDICTION_ACTION_SHAPE")
        schema = str(np.asarray(
            archive["schema_version"]
            if "schema_version" in keys
            else np.asarray("exact78-heldout-prediction-only-v1")
        ).item())
        if schema == "exact78-heldout-prediction-only-v2":
            safety_required = {
                "raw_model_prefix", "execution_safety_applied",
                "joint_lower", "joint_upper", "physical_to_human_side",
            }
            if not safety_required.issubset(keys):
                raise RuntimeError(
                    f"HOLD_PREDICTION_SAFETY_FIELDS:{sorted(safety_required-keys)}"
                )
            if not bool(np.asarray(archive["execution_safety_applied"]).item()):
                raise RuntimeError("HOLD_PREDICTION_SAFETY_NOT_APPLIED")
            executed = np.asarray(archive["causally_executed_prefix"])
            raw = np.asarray(archive["raw_model_prefix"])
            if executed.shape != raw.shape or executed.shape[1:] != (EXECUTION_HORIZON, 62):
                raise RuntimeError("HOLD_PREDICTION_PREFIX_SHAPE")
            lower = np.asarray(archive["joint_lower"]).reshape(1, 1, -1)
            upper = np.asarray(archive["joint_upper"]).reshape(1, 1, -1)
            q = executed[..., 18:62]
            if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
                raise RuntimeError("HOLD_PREDICTION_EXECUTED_Q_LIMIT")
            mapping = np.asarray(archive["physical_to_human_side"]).astype(str).tolist()
            if mapping != list(PHYSICAL_TO_HUMAN_SIDE):
                raise RuntimeError("HOLD_PREDICTION_SIDE_MAPPING")
    return {"status": "PASS_PREDICTION_ONLY_SCHEMA", "path": str(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("chips", "poker"))
    parser.add_argument("--session", default="")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-replans", type=int, default=32)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("--output is required for formal inference")
    if args.max_replans < 2 or args.fps < 1:
        parser.error("--max-replans must be >=2 and --fps positive")
    return args


def main() -> int:
    args = parse_args()
    pre = preflight(args.task, args.session or None, args.checkpoint)
    if args.check_only:
        print(json.dumps(pre, ensure_ascii=False, indent=2))
        return 0
    if pre["status"] != "READY":
        raise RuntimeError("WAIT_TASK_CHECKPOINT_COMPLETE")
    result = predict(args.task, pre, args.output.absolute(), args.max_replans,
                     args.fps, args.device)
    validate_prediction_npz_schema(Path(result["prediction"]["path"]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
