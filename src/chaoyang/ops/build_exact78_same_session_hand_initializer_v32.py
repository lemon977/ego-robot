#!/usr/bin/env python3
"""Build a same-session Exact78 hand seed and reloadable q22/FK evidence.

This producer deliberately does not create an arm initializer.  A HaWoR hand
trajectory and the KaiHand URDF identify a root-relative hand-shape problem,
but they do not observe the robot base, camera-to-base installation, or
tool-to-hand mount.  Those fields therefore remain absent and the arm route is
closed with a machine-readable blocker instead of an invented calibration.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from chaoyang.human_ego.exact78_v32 import validate_cohort_split
from chaoyang.ops import render_poker_same_side_outward_frame0 as shared
from chaoyang.ops import render_poker_symmetric_chirality_flange_successor as robot
from chaoyang.ops import run_newtask_robot_shared_v4_hand as handfit


INITIALIZER_SCHEMA = "EXACT78_SAME_SESSION_HAND_ACCEPTED_STATE_V32"
FK_SCHEMA = "EXACT78_HAND_ONLY_Q22_FK_V32"
PHYSICAL_SIDES = ("left", "right")
ANATOMICAL_SIDE_BY_PHYSICAL = ("left", "right")
ARM_BLOCKER = {
    "status": "BLOCKED_UNOBSERVED_INSTALLATION_GEOMETRY",
    "reason_codes": [
        "ROBOT_BASE_IN_HAWOR_WORLD_UNOBSERVED",
        "CAMERA_TO_ROBOT_BASE_UNOBSERVED",
        "TOOL_TO_HAND_MOUNT_UNOBSERVED",
    ],
    "forbidden_substitution": (
        "Do not label an identity transform, arbitrary world-axis placement, "
        "or cross-session template as measured installation calibration."
    ),
}


class SameSessionHandError(RuntimeError):
    """Raised when current same-session provenance cannot be proven."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SameSessionHandError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                dict(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_byte_alias(source: Path, destination: Path) -> None:
    """Publish a byte-identical, session-named adapter for legacy path guards."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary_name)
        if sha256(source) != sha256(Path(temporary_name)):
            raise SameSessionHandError("byte alias digest mismatch")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def reject_archive_inputs(paths: Sequence[Path]) -> None:
    for path in paths:
        resolved = path.resolve(strict=True)
        if "archive" in resolved.parts:
            raise SameSessionHandError(f"archive input forbidden: {resolved}")


def validate_session_membership(
    cohort_path: Path, session_id: str, expected_task: str
) -> dict[str, Any]:
    cohort = validate_cohort_split(load_json(cohort_path))
    row = cohort.get(session_id)
    if row is None:
        raise SameSessionHandError("session is not a frozen Exact78 V3.2 identity")
    if row["task"] != expected_task:
        raise SameSessionHandError("task/session mismatch")
    return row


def validate_hawor_closure(
    *, session_id: str, result_path: Path, npz_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reject_archive_inputs((result_path, npz_path))
    result = load_json(result_path)
    if result.get("session_id") != session_id:
        raise SameSessionHandError("HaWoR RESULT session mismatch")
    if result.get("status") != "COMPLETE_BASELINE_NO_QUALITY_GATE":
        raise SameSessionHandError("fresh development HaWoR baseline required")
    documented = result.get("outputs", {}).get("npz", result.get("npz", {}))
    if not isinstance(documented, dict) or documented.get("sha256") != sha256(npz_path):
        raise SameSessionHandError("HaWoR RESULT does not bind the exact NPZ")
    if int(documented.get("bytes", -1)) != npz_path.stat().st_size:
        raise SameSessionHandError("HaWoR RESULT/NPZ byte count mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "joints_3d_world",
        "observed",
        "c2w",
        "original_frame_indices",
        "fps",
        "anatomical_side_names",
    }
    if not required.issubset(arrays):
        raise SameSessionHandError(f"HaWoR fields missing: {sorted(required - set(arrays))}")
    count = int(result.get("frame_count", -1))
    if arrays["joints_3d_world"].shape != (2, count, 21, 3):
        raise SameSessionHandError("HaWoR MANO21 shape mismatch")
    observed = np.asarray(arrays["observed"], dtype=bool)
    if observed.shape != (2, count):
        raise SameSessionHandError("HaWoR observed shape mismatch")
    finite = np.isfinite(arrays["joints_3d_world"]).all(axis=(2, 3))
    if not np.array_equal(observed, finite):
        raise SameSessionHandError("HaWoR observed/nonfinite semantics mismatch")
    if not np.array_equal(arrays["original_frame_indices"], np.arange(count)):
        raise SameSessionHandError("HaWoR source frame axis is not contiguous 0..N-1")
    if tuple(arrays["anatomical_side_names"].astype(str).tolist()) != ("left", "right"):
        raise SameSessionHandError("HaWoR anatomical side order mismatch")
    return result, arrays


def hand_contract(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assets = robot.load_pinned_robot_assets(repo_root)
    contracts = handfit.model_contract(robot.official, shared.wrist_adapter, assets)
    rows = []
    for side, contract in enumerate(contracts):
        rows.append(
            {
                "physical_side": PHYSICAL_SIDES[side],
                "anatomical_side": ANATOMICAL_SIDE_BY_PHYSICAL[side],
                "joint_names": list(contract["names"]),
                "joint_lower_rad": np.asarray(contract["lower"], dtype=np.float64),
                "joint_upper_rad": np.asarray(contract["upper"], dtype=np.float64),
                "neutral_q22_rad": np.asarray(contract["neutral"], dtype=np.float64),
                "model": contract["model"],
            }
        )
    return rows, {
        "asset_pin": ref(repo_root / "assets/robot/ROBOT_ASSET_PIN.json"),
        "side_mapping": {
            "physical_left": "anatomical_left",
            "physical_right": "anatomical_right",
            "authority": "EXISTING_EXACT78_HAND_RUNNER_CONVENTION",
        },
    }


def initializer_arrays(rows: Sequence[Mapping[str, Any]], session_id: str) -> dict[str, np.ndarray]:
    lower = np.stack([np.asarray(row["joint_lower_rad"]) for row in rows])
    upper = np.stack([np.asarray(row["joint_upper_rad"]) for row in rows])
    neutral = np.stack([np.asarray(row["neutral_q22_rad"]) for row in rows])
    if lower.shape != (2, 22) or upper.shape != (2, 22) or neutral.shape != (2, 22):
        raise SameSessionHandError("KaiHand contract is not exact q22 per side")
    if not np.isfinite(neutral).all() or np.any(neutral < lower) or np.any(neutral > upper):
        raise SameSessionHandError("URDF-neutral q22 is nonfinite or out of limits")
    return {
        "schema_version": np.asarray(INITIALIZER_SCHEMA),
        "session_id": np.asarray(session_id),
        "physical_robot_side": np.asarray(PHYSICAL_SIDES),
        "anatomical_side": np.asarray(ANATOMICAL_SIDE_BY_PHYSICAL),
        "joint_names": np.asarray([row["joint_names"] for row in rows]),
        "joint_lower_rad": lower,
        "joint_upper_rad": upper,
        "q_hand": neutral[None],
        "q_hand_source": np.asarray("URDF_LIMIT_MIDPOINT_DEVELOPMENT_SEED"),
        "control_ground_truth": np.asarray(False),
        "physical_deployable": np.asarray(False),
    }


def validate_initializer_npz(path: Path, session_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if str(arrays.get("schema_version", np.asarray("")).item()) != INITIALIZER_SCHEMA:
        raise SameSessionHandError("initializer schema mismatch")
    if str(arrays.get("session_id", np.asarray("")).item()) != session_id:
        raise SameSessionHandError("initializer session mismatch")
    forbidden = {
        "q_arm",
        "T_world_base",
        "T_camera_base",
        "T_tool_hand_root",
        "T_target_hand_root_world",
    }
    if forbidden.intersection(arrays):
        raise SameSessionHandError("initializer fabricated arm/installation fields")
    if arrays.get("q_hand", np.empty(0)).shape != (1, 2, 22):
        raise SameSessionHandError("initializer q_hand shape mismatch")
    lower = arrays.get("joint_lower_rad")
    upper = arrays.get("joint_upper_rad")
    q = arrays["q_hand"][0]
    if lower is None or upper is None or np.any(q < lower) or np.any(q > upper):
        raise SameSessionHandError("initializer q_hand violates URDF limits")
    if bool(arrays["control_ground_truth"].item()) or bool(arrays["physical_deployable"].item()):
        raise SameSessionHandError("initializer authority boundary mismatch")
    return arrays


def build_initializer(
    *,
    repo_root: Path,
    task: str,
    session_id: str,
    cohort_path: Path,
    hawor_result_path: Path,
    hawor_npz_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"fresh output root required: {output_root}")
    reject_archive_inputs((cohort_path, hawor_result_path, hawor_npz_path))
    cohort_row = validate_session_membership(cohort_path, session_id, task)
    hawor_result, _ = validate_hawor_closure(
        session_id=session_id, result_path=hawor_result_path, npz_path=hawor_npz_path
    )
    rows, asset_evidence = hand_contract(repo_root)
    output_root.mkdir(parents=True)
    states_path = output_root / "ACCEPTED_HAND_STATES.npz"
    atomic_npz(states_path, initializer_arrays(rows, session_id))
    validate_initializer_npz(states_path, session_id)
    runner_root = output_root / f"{session_id}_maintained_runner_inputs"
    runner_hawor = runner_root / f"{session_id}_HAWOR_RAW_MANO21.npz"
    runner_result = runner_root / f"{session_id}_HAWOR_RESULT.json"
    atomic_byte_alias(hawor_npz_path, runner_hawor)
    atomic_byte_alias(hawor_result_path, runner_result)
    result = {
        "schema_version": INITIALIZER_SCHEMA,
        "created_at": now(),
        "status": "PASS_DEVELOPMENT_HAND_INITIALIZER_ARM_BLOCKED",
        "task": task,
        "session_id": session_id,
        "split": cohort_row["split"],
        "source_group_id": cohort_row["source_group_id"],
        "input_mode": "OFFLINE_NONCAUSAL_HAWOR_DERIVED_HAND_ONLY",
        "accepted_state_scope": "HAND_Q22_INITIALIZATION_ONLY",
        "same_session": True,
        "archive_inputs_consumed": False,
        "inputs": {
            "cohort": ref(cohort_path),
            "hawor_result": ref(hawor_result_path),
            "hawor_npz": ref(hawor_npz_path),
            "producer_code": ref(Path(__file__)),
            **asset_evidence,
        },
        "producer_registration_status": "PENDING_PUBLISHER_CONTRACT_REGISTRATION",
        "hawor_status": hawor_result["status"],
        "side_mapping": asset_evidence["side_mapping"],
        "initializer_policy": {
            "q22": "PER_SIDE_PINNED_URDF_LIMIT_MIDPOINT",
            "future_frames_read_for_seed": False,
            "cross_session_state_read": False,
            "arm_fields_emitted": False,
        },
        "accepted_states": ref(states_path),
        "maintained_hand_runner_inputs": {
            "hawor_npz": ref(runner_hawor),
            "hawor_result": ref(runner_result),
            "adapter_policy": "BYTE_IDENTICAL_SESSION_NAMED_PATH_ALIAS_ONLY",
            "source_npz_sha_equal": sha256(runner_hawor) == sha256(hawor_npz_path),
            "source_result_sha_equal": sha256(runner_result) == sha256(hawor_result_path),
        },
        "arm_admission": dict(ARM_BLOCKER),
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "Same-session development hand seed only. No arm/world/mount calibration, "
            "Robot action, causal-training, control, deployment or external metric authority."
        ),
    }
    atomic_json(output_root / "RESULT.json", result)
    return result


def full_fk_arrays(
    *,
    session_id: str,
    hawor: Mapping[str, np.ndarray],
    q_hand: np.ndarray,
    q_hand_raw: np.ndarray,
    valid_side_frame: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    q = np.asarray(q_hand, dtype=np.float64)
    raw = np.asarray(q_hand_raw, dtype=np.float64)
    valid_st = np.asarray(valid_side_frame, dtype=bool)
    count = len(hawor["original_frame_indices"])
    if q.shape != (count, 2, 22) or raw.shape != q.shape or valid_st.shape != (2, count):
        raise SameSessionHandError("hand trajectory axis contract mismatch")
    valid = valid_st.T
    source_observed = np.asarray(hawor["observed"], dtype=bool).T
    if not np.array_equal(valid, source_observed):
        raise SameSessionHandError("hand validity must exactly preserve HaWoR observations")
    frame_ids = np.asarray(hawor["original_frame_indices"], dtype=np.int64)
    fps = float(np.asarray(hawor["fps"]).item())
    if not np.isfinite(fps) or fps <= 0:
        raise SameSessionHandError("HaWoR fps is invalid")
    timestamp_s = frame_ids.astype(np.float64) / fps
    names = np.asarray([row["joint_names"] for row in rows])
    lower = np.stack([np.asarray(row["joint_lower_rad"]) for row in rows])
    upper = np.stack([np.asarray(row["joint_upper_rad"]) for row in rows])
    link_names = tuple(tuple(row["model"].links) for row in rows)
    if len(set(map(len, link_names))) != 1:
        raise SameSessionHandError("left/right KaiHand link counts differ")
    link_count = len(link_names[0])
    fk = np.full((count, 2, link_count, 4, 4), np.nan, dtype=np.float64)
    q_finite = np.isfinite(q).all(axis=2)
    limit_pass = np.zeros((count, 2), dtype=bool)
    max_limit_violation = np.full((count, 2), np.nan, dtype=np.float64)
    fk_finite = np.zeros((count, 2), dtype=bool)
    for frame in range(count):
        for side in range(2):
            if not valid[frame, side]:
                continue
            if not q_finite[frame, side]:
                continue
            below = np.maximum(lower[side] - q[frame, side], 0.0)
            above = np.maximum(q[frame, side] - upper[side], 0.0)
            violation = float(np.max(np.maximum(below, above)))
            max_limit_violation[frame, side] = violation
            limit_pass[frame, side] = violation <= 1e-12
            if not limit_pass[frame, side]:
                continue
            transforms = robot.official.forward_kinematics(
                rows[side]["model"],
                dict(zip(rows[side]["joint_names"], q[frame, side], strict=True)),
            )
            fk[frame, side] = np.stack([transforms[name] for name in link_names[side]])
            fk_finite[frame, side] = bool(np.isfinite(fk[frame, side]).all())
    projection_delta = q - raw
    arrays = {
        "schema_version": np.asarray(FK_SCHEMA),
        "session_id": np.asarray(session_id),
        "source_frame": frame_ids,
        "timestamp_s": timestamp_s,
        "physical_robot_side": np.asarray(PHYSICAL_SIDES),
        "anatomical_side": np.asarray(ANATOMICAL_SIDE_BY_PHYSICAL),
        "joint_names": names,
        "link_names": np.asarray(link_names),
        "q22": q,
        "q22_raw": raw,
        "q22_valid": valid,
        "source_hawor_observed": source_observed,
        "q22_inferred_not_observed": valid.copy(),
        "projection_delta": projection_delta,
        "q22_finite": q_finite,
        "joint_lower_rad": lower,
        "joint_upper_rad": upper,
        "hard_limit_pass": limit_pass,
        "max_limit_violation_rad": max_limit_violation,
        "fk_root_relative": fk,
        "fk_finite": fk_finite,
        "self_collision_diagnostic_known": np.zeros((count, 2), dtype=bool),
        "control_ground_truth": np.asarray(False),
        "physical_deployable": np.asarray(False),
    }
    metrics = {
        "frame_count": count,
        "valid_side_frames": int(valid.sum()),
        "finite_q22_valid_side_frames": int((valid & q_finite).sum()),
        "hard_limit_pass_valid_side_frames": int((valid & limit_pass).sum()),
        "full_fk_pass_valid_side_frames": int((valid & fk_finite).sum()),
        "self_collision_diagnostic_known_valid_side_frames": 0,
        "max_projection_delta_rad": float(
            np.nanmax(np.abs(projection_delta)) if np.isfinite(projection_delta).any() else 0.0
        ),
    }
    return arrays, metrics


def validate_fk_npz(path: Path, session_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if str(arrays.get("schema_version", np.asarray("")).item()) != FK_SCHEMA:
        raise SameSessionHandError("FK sidecar schema mismatch")
    if str(arrays.get("session_id", np.asarray("")).item()) != session_id:
        raise SameSessionHandError("FK sidecar session mismatch")
    required = {
        "source_frame",
        "timestamp_s",
        "joint_names",
        "link_names",
        "q22",
        "q22_valid",
        "hard_limit_pass",
        "fk_root_relative",
        "fk_finite",
    }
    if not required.issubset(arrays):
        raise SameSessionHandError(f"FK sidecar fields missing: {sorted(required - set(arrays))}")
    valid = arrays["q22_valid"].astype(bool)
    if np.any(valid & ~arrays["q22_finite"].astype(bool)):
        raise SameSessionHandError("valid q22 contains nonfinite values")
    if np.any(valid & ~arrays["hard_limit_pass"].astype(bool)):
        raise SameSessionHandError("valid q22 violates hard limits")
    if np.any(valid & ~arrays["fk_finite"].astype(bool)):
        raise SameSessionHandError("valid q22 lacks finite full FK")
    if bool(arrays["control_ground_truth"].item()) or bool(arrays["physical_deployable"].item()):
        raise SameSessionHandError("FK sidecar authority boundary mismatch")
    return arrays


def build_fk(
    *,
    repo_root: Path,
    task: str,
    session_id: str,
    cohort_path: Path,
    hawor_result_path: Path,
    hawor_npz_path: Path,
    initializer_result_path: Path,
    hand_result_path: Path,
    hand_states_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"fresh output root required: {output_root}")
    reject_archive_inputs(
        (
            cohort_path,
            hawor_result_path,
            hawor_npz_path,
            initializer_result_path,
            hand_result_path,
            hand_states_path,
        )
    )
    cohort_row = validate_session_membership(cohort_path, session_id, task)
    _, hawor = validate_hawor_closure(
        session_id=session_id, result_path=hawor_result_path, npz_path=hawor_npz_path
    )
    initializer = load_json(initializer_result_path)
    if (
        initializer.get("schema_version") != INITIALIZER_SCHEMA
        or initializer.get("session_id") != session_id
        or initializer.get("status") != "PASS_DEVELOPMENT_HAND_INITIALIZER_ARM_BLOCKED"
    ):
        raise SameSessionHandError("current same-session initializer result required")
    accepted_path = Path(initializer["accepted_states"]["path"])
    if initializer["accepted_states"] != ref(accepted_path):
        raise SameSessionHandError("initializer accepted-state reference drift")
    validate_initializer_npz(accepted_path, session_id)
    hand_result = load_json(hand_result_path)
    if hand_result.get("session") != session_id or hand_result.get("task") != task:
        raise SameSessionHandError("hand result identity mismatch")
    if hand_result.get("status") not in {"PASS_NUMERIC_CANARY_NO_AUTHORITY", "HOLD_NUMERIC_CANARY"}:
        raise SameSessionHandError("numeric hand result required")
    lineage = hand_result.get("lineage", {})
    if lineage.get("hawor", {}).get("sha256") != sha256(hawor_npz_path):
        raise SameSessionHandError("hand result/HaWoR lineage mismatch")
    if lineage.get("accepted_states", {}).get("sha256") != sha256(accepted_path):
        raise SameSessionHandError("hand result/initializer lineage mismatch")
    if hand_result.get("output_states") != ref(hand_states_path):
        raise SameSessionHandError("hand result/state exact reference mismatch")
    with np.load(hand_states_path, allow_pickle=False) as archive:
        hand = {name: np.asarray(archive[name]) for name in archive.files}
    rows, asset_evidence = hand_contract(repo_root)
    arrays, metrics = full_fk_arrays(
        session_id=session_id,
        hawor=hawor,
        q_hand=hand["q_hand"],
        q_hand_raw=hand["q_hand_raw"],
        valid_side_frame=hand["valid_side_frame"],
        rows=rows,
    )
    output_root.mkdir(parents=True)
    arrays_path = output_root / "HAND_ONLY_Q22_FULL_FK.npz"
    atomic_npz(arrays_path, arrays)
    validate_fk_npz(arrays_path, session_id)
    result = {
        "schema_version": FK_SCHEMA,
        "created_at": now(),
        "status": "COMPLETED_HAND_ONLY_Q22_FK_ARM_BLOCKED",
        "task": task,
        "session_id": session_id,
        "split": cohort_row["split"],
        "source_group_id": cohort_row["source_group_id"],
        "input_mode": "OFFLINE_NONCAUSAL_HAWOR_DERIVED_HAND_ONLY",
        "temporal_authority": "OFFLINE_NONCAUSAL",
        "same_session": True,
        "archive_inputs_consumed": False,
        "inputs": {
            "cohort": ref(cohort_path),
            "hawor_result": ref(hawor_result_path),
            "hawor_npz": ref(hawor_npz_path),
            "initializer_result": ref(initializer_result_path),
            "accepted_states": ref(accepted_path),
            "hand_result": ref(hand_result_path),
            "hand_states": ref(hand_states_path),
            "producer_code": ref(Path(__file__)),
            **asset_evidence,
        },
        "producer_registration_status": "PENDING_PUBLISHER_CONTRACT_REGISTRATION",
        "q22_fk": ref(arrays_path),
        "metrics": metrics,
        "side_mapping": asset_evidence["side_mapping"],
        "q22_semantics": {
            "observed_or_inferred": "INFERRED_FROM_HAWOR_MANO21",
            "unit": "radian",
            "missing": "NaN_WITH_VALID_FALSE_NO_FORWARD_FILL",
            "fk_frame": "KAIHAND_ROOT_RELATIVE",
            "projection_delta_saved": True,
            "self_collision_diagnostic": "UNKNOWN_NOT_RUN",
        },
        "arm_admission": dict(ARM_BLOCKER),
        "visual_aux_robot_review_eligible": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "Reloadable current same-session hand-only q22/full-FK development evidence. "
            "Arm/world/mount geometry is absent; no full Robotized render, causal training, "
            "Robot action, control, deployment, collision, Contact or external metric authority."
        ),
    }
    atomic_json(output_root / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initializer", "fk"), required=True)
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--initializer-result", type=Path)
    parser.add_argument("--hand-result", type=Path)
    parser.add_argument("--hand-states", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = (
        args.repo_root.resolve(strict=True)
        if args.repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    common = {
        "repo_root": repo_root,
        "task": args.task,
        "session_id": args.session,
        "cohort_path": args.cohort_manifest.resolve(strict=True),
        "hawor_result_path": args.hawor_result.resolve(strict=True),
        "hawor_npz_path": args.hawor_npz.resolve(strict=True),
        "output_root": args.output_root.resolve(),
    }
    if args.mode == "initializer":
        result = build_initializer(**common)
    else:
        if args.initializer_result is None or args.hand_result is None or args.hand_states is None:
            parser.error("fk mode requires --initializer-result, --hand-result and --hand-states")
        result = build_fk(
            **common,
            initializer_result_path=args.initializer_result.resolve(strict=True),
            hand_result_path=args.hand_result.resolve(strict=True),
            hand_states_path=args.hand_states.resolve(strict=True),
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "session_id": result["session_id"],
                "result": str((args.output_root / "RESULT.json").resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
