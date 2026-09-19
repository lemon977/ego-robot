#!/usr/bin/env python3
"""Publish W0 Kai22 R1-E terminals and one rejected R1-H counterfactual.

This runner is deliberately downstream of the formal Contact ledgers.  It
cannot manufacture a Contact window.  R1-E therefore remains blocked when the
Contact evidence ledger authorizes zero strict windows.  If (and only if) the
separate hypothesis ledger contains the frozen seven-frame
``119/right/index/card02`` hypothesis plus independently selected no-contact
controls, a deterministic hand-only H1 counterfactual is exported for review.

H1 is an optimization diagnostic, not adoption evidence.  It never changes a
wrist pose, never estimates object-relative placement, and is never training,
control, Contact, or deployment authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import cv2
import jsonschema
import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet
from chaoyang.pipeline.interaction_contact_robot_dev_v1 import timestamp_motion_diagnostics
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_robot_relative_refinement_v1"
PHASE = "ROBOT15H_KAI22_R1_E_AND_R1_H_WAVE0"
AUTHORITY = "DEVELOPMENT_RELATIVE_NON_CONTROL_NON_DEPLOYABLE"
CONTACT_TASK_ID = "0915_robot15h_contact_dual_evidence_v1"
CONTACT_ROOT = ROOT / f"_run/current/{CONTACT_TASK_ID}/attempts/attempt_0001"
R0_ROOT = ROOT / "_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_R1_V1"
RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_ROBOT_RELATIVE_REFINEMENT_V1_RESULT.json"
CONTRACT = ROOT / "contracts/kai22_r1_hypothesis_v1.schema.json"

EXPECTED_SESSIONS = {
    "play_cards_0915_031": "playing_cards",
    "play_cards_0915_119": "playing_cards",
    "get_potato_chips_0915_007": "potato_chips",
    "get_potato_chips_0915_042": "potato_chips",
}
HYPOTHESIS_SESSION = "play_cards_0915_119"
HYPOTHESIS_IDENTITY = ("right", "index", "playing_card_02")
CORE_FRAMES = tuple(range(92, 99))
LEFT_TAPER_FRAMES = tuple(range(86, 92))
RIGHT_TAPER_FRAMES = tuple(range(99, 105))
PHYSICAL_SIDE = 0  # anatomical right -> physical kaihand_left
INDEX_SLICE = slice(6, 10)
INDEX_FLEXION_DELTA_RAD = np.asarray((0.0, 0.06, 0.09, 0.04), np.float64)
MAX_ABS_DELTA_RAD = 0.12
MAX_TAPER_SECONDS = 0.2 + 1e-9
MOTION_TOLERANCE = 1e-10
LEFT_HAND_URDF = ROOT / (
    "assets/robot/kaihand/packages/"
    "KaiBot-Dexhand shell-URDF-L-260624(1620)/urdf/"
    "KaiBot-Dexhand shell-URDF-L-260624(1620).urdf"
)
CLAIM_LIMIT = (
    "Development-only Kai22 hypothesis counterfactual. R1-E has no admitted local "
    "metric evidence. H1 is rejected because object-independent metric wrist placement "
    "is absent; it is not Contact, adoption, training, control, deployment, force, "
    "probability, or physical-collision authority."
)


class R1HypothesisError(RuntimeError):
    """Fail-closed R1-H contract violation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def projected_ref(existing: Path, destination: Path) -> dict[str, Any]:
    source = existing.resolve(strict=True)
    return {
        "path": str(destination.resolve()),
        "bytes": source.stat().st_size,
        "sha256": sha256(source),
    }


def verify_ref(value: Mapping[str, Any]) -> Path:
    path = Path(str(value["path"])).resolve(strict=True)
    if path.stat().st_size != int(value["bytes"]) or sha256(path) != value["sha256"]:
        raise R1HypothesisError(f"artifact reference drift: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R1HypothesisError(f"JSON object required: {path}")
    return value


def validate_session_contract(value: Mapping[str, Any]) -> None:
    schema = load_json(CONTRACT)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(dict(value))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def arrays_bit_exact(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare dtype, shape, and every stored bit, including NaN payloads."""

    a, b = np.asarray(first), np.asarray(second)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise R1HypothesisError("task packet differs from frozen weights-ABSENT specification")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise R1HypothesisError(f"{TASK_ID} is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None,
    )
    if (
        route is None or route.get("execution_allowed") is not True
        or route.get("packet_sha256") != sha256(packet_path)
    ):
        raise R1HypothesisError(f"{TASK_ID} is not SHA-bound routable")
    return packet, packet_path


def heartbeat() -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(os.getpid()),
            "--status", "RUNNING", "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise R1HypothesisError(
            "governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:]
        )


def validate_contact_ledgers(
    evidence: Mapping[str, Any], hypothesis: Mapping[str, Any], batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive eligibility exclusively from the three formal Contact outputs."""

    if (
        evidence.get("schema_version") != "0915-robot15h-contact-evidence-ledger-v1"
        or evidence.get("task_id") != CONTACT_TASK_ID
        or evidence.get("window_run_id") != WINDOW_RUN_ID
        or evidence.get("strict_metric_contact_authorized") is not False
        or evidence.get("r1_e_authorized") is not False
        or evidence.get("training_eligible") is not False
    ):
        raise R1HypothesisError("formal Contact evidence ledger authority drift")
    if (
        hypothesis.get("schema_version") != "0915-robot15h-contact-hypothesis-ledger-v1"
        or hypothesis.get("task_id") != CONTACT_TASK_ID
        or hypothesis.get("window_run_id") != WINDOW_RUN_ID
        or hypothesis.get("strict_contact_authorized") is not False
        or hypothesis.get("r1_e_authorized") is not False
        or hypothesis.get("training_eligible") is not False
        or hypothesis.get("contact_ground_truth") is not False
    ):
        raise R1HypothesisError("formal Contact hypothesis ledger authority drift")
    rows = batch.get("sessions")
    if (
        batch.get("schema_version") != "0915-robot15h-contact-wave0-batch-v1"
        or batch.get("task_id") != CONTACT_TASK_ID
        or batch.get("window_run_id") != WINDOW_RUN_ID
        or batch.get("status") != "COMPLETED_ALL_TERMINAL"
        or batch.get("strict_metric_contact_authorized") is not False
        or batch.get("r1_e_authorized") is not False
        or not isinstance(rows, list)
        or {str(row.get("session_id")) for row in rows} != set(EXPECTED_SESSIONS)
        or len(rows) != len(EXPECTED_SESSIONS)
    ):
        raise R1HypothesisError("formal Contact batch identity/authority drift")
    counts = batch.get("counts")
    if not isinstance(counts, Mapping) or int(counts.get("r1_e_windows", -1)) != 0:
        raise R1HypothesisError("R1-E must remain closed at zero admitted windows")

    hypotheses = hypothesis.get("hypotheses")
    controls = hypothesis.get("no_contact_controls")
    if not isinstance(hypotheses, list) or not isinstance(controls, list):
        raise R1HypothesisError("Contact hypothesis/control rows must be explicit lists")
    keys = {
        (
            str(row.get("session_id")), int(row.get("frame_id", -1)),
            str(row.get("hand_id")), str(row.get("finger_id")),
            str(row.get("object_id")),
        )
        for row in hypotheses if isinstance(row, Mapping)
    }
    expected_keys = {
        (HYPOTHESIS_SESSION, frame, *HYPOTHESIS_IDENTITY) for frame in CORE_FRAMES
    }
    rows_are_hypothesis_only = all(
        isinstance(row, Mapping)
        and row.get("status") == "HYPOTHESIS_ONLY"
        and row.get("metric_contact") is False
        and row.get("strict_contact_admitted") is False
        and row.get("training_eligible") is False
        and row.get("contact_ground_truth") is False
        for row in hypotheses
    )
    control_keys = {
        (
            str(row.get("session_id")), str(row.get("hand_id")),
            str(row.get("finger_id")), str(row.get("object_id")),
        )
        for row in controls if isinstance(row, Mapping)
    }
    expected_control_key = {(HYPOTHESIS_SESSION, *HYPOTHESIS_IDENTITY)}
    controls_valid = bool(controls) and control_keys == expected_control_key and all(
        isinstance(row, Mapping)
        and row.get("status") == "NO_CONTACT"
        and int(row.get("frame_id", -1)) not in CORE_FRAMES
        and row.get("metric_contact") is False
        and row.get("strict_contact_admitted") is False
        and row.get("training_eligible") is False
        and row.get("contact_ground_truth") is False
        for row in controls
    )
    h1_eligible = (
        keys == expected_keys and len(hypotheses) == len(CORE_FRAMES)
        and rows_are_hypothesis_only and controls_valid
        and int(hypothesis.get("verified_hypothesis_count", -1)) == len(hypotheses)
        and int(hypothesis.get("no_contact_control_count", -1)) == len(controls)
    )
    blocker = None
    if keys != expected_keys or len(hypotheses) != len(CORE_FRAMES) or not rows_are_hypothesis_only:
        blocker = "NO_EXACT_FORMAL_SEVEN_FRAME_HYPOTHESIS"
    elif not controls_valid:
        blocker = "NO_FORMAL_NO_CONTACT_CONTROL"
    return {
        "r1_e_authorized": False,
        "r1_e_windows": 0,
        "h1_counterfactual_eligible": h1_eligible,
        "h1_eligibility_blocker": blocker,
        "hypothesis_frame_ids": sorted(key[1] for key in keys if key[0] == HYPOTHESIS_SESSION),
        "no_contact_control_frame_ids": sorted(
            int(row["frame_id"]) for row in controls if isinstance(row, Mapping)
        ),
    }


def validate_r0_batch(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = batch.get("results")
    if (
        batch.get("schema_version") != "0915-robot15h-kai22-r0-wave0-batch-v1"
        or batch.get("window_run_id") != WINDOW_RUN_ID
        or batch.get("status") != "COMPLETED_ALL_TERMINAL"
        or batch.get("session_count") != 4
        or batch.get("r0_exported") != 4
        or batch.get("failed_runtime") != 0
        or not isinstance(rows, list) or len(rows) != 4
    ):
        raise R1HypothesisError("R0 batch closure drift")
    identities = [str(row.get("session_id")) for row in rows if isinstance(row, Mapping)]
    if len(identities) != 4 or set(identities) != set(EXPECTED_SESSIONS):
        raise R1HypothesisError("R0 batch session identity/uniqueness drift")
    if any(row.get("r0_exported") is not True for row in rows):
        raise R1HypothesisError("all W0 R0 exports are required")
    return [dict(row) for row in rows]


def load_r0_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "q22_init", "relative_wrist_T", "valid_side_frame", "timestamps_s", "frame_id",
        }
        if not required.issubset(archive.files):
            raise R1HypothesisError(f"R0 NPZ lacks required fields: {required - set(archive.files)}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    q22 = arrays["q22_init"]
    wrist = arrays["relative_wrist_T"]
    valid = arrays["valid_side_frame"]
    timestamps = arrays["timestamps_s"]
    frame_id = arrays["frame_id"]
    if q22.dtype != np.dtype("float64") or q22.ndim != 3 or q22.shape[1:] != (2, 22):
        raise R1HypothesisError("R0 q22_init must be float64 [T,2,22]")
    frame_count = q22.shape[0]
    if wrist.dtype != np.dtype("float64") or wrist.shape != (frame_count, 2, 4, 4):
        raise R1HypothesisError("R0 relative_wrist_T must be float64 [T,2,4,4]")
    if valid.dtype != np.dtype("bool") or valid.shape != (frame_count, 2):
        raise R1HypothesisError("R0 valid_side_frame must be bool [T,2]")
    if timestamps.dtype != np.dtype("float64") or timestamps.shape != (frame_count,):
        raise R1HypothesisError("R0 timestamps_s must be float64 [T]")
    if frame_id.dtype != np.dtype("int64") or not np.array_equal(frame_id, np.arange(frame_count)):
        raise R1HypothesisError("R0 frame_id must be contiguous int64 source frames")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise R1HypothesisError("R0 timestamps must be finite and strictly increasing")
    if not np.isfinite(q22[valid]).all() or not np.isnan(q22[~valid]).all():
        raise R1HypothesisError("R0 q22 validity/NaN contract drift")
    if not np.isfinite(wrist[valid]).all() or not np.isnan(wrist[~valid]).all():
        raise R1HypothesisError("R0 wrist validity/NaN contract drift")
    return arrays


def taper_envelope(timestamps_s: Sequence[float]) -> np.ndarray:
    timestamps = np.asarray(timestamps_s, np.float64)
    if timestamps.ndim != 1 or timestamps.size <= RIGHT_TAPER_FRAMES[-1]:
        raise R1HypothesisError("timeline does not contain frozen H1 scope")
    if (
        timestamps[CORE_FRAMES[0]] - timestamps[LEFT_TAPER_FRAMES[0]] > MAX_TAPER_SECONDS
        or timestamps[RIGHT_TAPER_FRAMES[-1]] - timestamps[CORE_FRAMES[-1]] > MAX_TAPER_SECONDS
    ):
        raise R1HypothesisError("frozen taper exceeds 0.2 seconds")
    envelope = np.zeros(timestamps.shape, np.float64)
    left_t0, left_t1 = timestamps[LEFT_TAPER_FRAMES[0]], timestamps[CORE_FRAMES[0]]
    right_t0, right_t1 = timestamps[CORE_FRAMES[-1]], timestamps[RIGHT_TAPER_FRAMES[-1]]
    envelope[list(LEFT_TAPER_FRAMES)] = (
        timestamps[list(LEFT_TAPER_FRAMES)] - left_t0
    ) / (left_t1 - left_t0)
    envelope[list(CORE_FRAMES)] = 1.0
    envelope[list(RIGHT_TAPER_FRAMES)] = 1.0 - (
        timestamps[list(RIGHT_TAPER_FRAMES)] - right_t0
    ) / (right_t1 - right_t0)
    envelope[np.abs(envelope) < 1e-14] = 0.0
    if envelope.min() < 0 or envelope.max() > 1 or not np.all(envelope[list(CORE_FRAMES)] == 1):
        raise R1HypothesisError("invalid frozen taper envelope")
    return envelope


def build_counterfactual(
    arrays: Mapping[str, np.ndarray], *, eligible: bool,
    lower_rad: Sequence[float], upper_rad: Sequence[float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    q0 = np.asarray(arrays["q22_init"])
    wrist0 = np.asarray(arrays["relative_wrist_T"])
    valid = np.asarray(arrays["valid_side_frame"])
    timestamps = np.asarray(arrays["timestamps_s"])
    frame_id = np.asarray(arrays["frame_id"])
    q1 = q0.copy()
    wrist1 = wrist0.copy()
    envelope = np.zeros((q0.shape[0],), np.float64)
    applied = np.zeros(valid.shape, bool)
    if eligible:
        envelope = taper_envelope(timestamps)
        if not valid[list(CORE_FRAMES), PHYSICAL_SIDE].all():
            raise R1HypothesisError("formal H1 core lacks direct-observed R0 physical-left hand")
        candidate = envelope[:, None] * INDEX_FLEXION_DELTA_RAD[None, :]
        applied[:, PHYSICAL_SIDE] = (envelope > 0) & valid[:, PHYSICAL_SIDE]
        q1[applied[:, PHYSICAL_SIDE], PHYSICAL_SIDE, INDEX_SLICE] += candidate[
            applied[:, PHYSICAL_SIDE]
        ]
    delta = q1 - q0
    finite_delta = np.abs(delta[np.isfinite(delta)])
    maximum = float(finite_delta.max()) if finite_delta.size else 0.0
    if maximum > MAX_ABS_DELTA_RAD + 1e-12:
        raise R1HypothesisError("counterfactual exceeds 0.12 rad bound")
    lower, upper = np.asarray(lower_rad, np.float64), np.asarray(upper_rad, np.float64)
    if lower.shape != (22,) or upper.shape != (22,):
        raise R1HypothesisError("Kai22 limit arrays must contain 22 joints")
    if np.any(q1[valid] < np.tile(lower, (int(valid.sum()), 1)) - 1e-12) or np.any(
        q1[valid] > np.tile(upper, (int(valid.sum()), 1)) + 1e-12
    ):
        raise R1HypothesisError("counterfactual violates pinned Kai22 limits")

    allowed = np.zeros(q0.shape, bool)
    allowed[:, PHYSICAL_SIDE, INDEX_SLICE] = True
    outside_scope_exact = arrays_bit_exact(q0[~allowed], q1[~allowed])
    invalid_exact = arrays_bit_exact(q0[~valid], q1[~valid])
    wrist_exact = arrays_bit_exact(wrist0, wrist1)
    if not outside_scope_exact or not invalid_exact or not wrist_exact:
        raise R1HypothesisError("counterfactual changed data outside frozen hand-only scope")
    motion_before = timestamp_motion_diagnostics(q0[:, PHYSICAL_SIDE], timestamps, valid[:, PHYSICAL_SIDE])
    motion_after = timestamp_motion_diagnostics(q1[:, PHYSICAL_SIDE], timestamps, valid[:, PHYSICAL_SIDE])
    motion_keys = (
        "max_abs_velocity_rad_s", "p95_abs_velocity_rad_s",
        "max_abs_acceleration_rad_s2", "p95_abs_acceleration_rad_s2",
    )
    motion_not_degraded = all(
        motion_before[key] is None
        or (
            motion_after[key] is not None
            and float(motion_after[key]) <= float(motion_before[key]) + MOTION_TOLERANCE
        )
        for key in motion_keys
    )
    if eligible and not motion_not_degraded:
        raise R1HypothesisError("timestamp-based velocity/acceleration regressed")
    outputs = {
        "q22_h0": q0.copy(),
        "q22_h1_rejected": q1,
        "relative_wrist_T_h0": wrist0.copy(),
        "relative_wrist_T_h1_rejected": wrist1,
        "valid_side_frame": valid.copy(),
        "timestamps_s": timestamps.copy(),
        "frame_id": frame_id.copy(),
        "counterfactual_delta_q22": delta,
        "counterfactual_applied_mask": applied,
        "taper_envelope": envelope,
    }
    diagnostics = {
        "h0_r0_q22_bit_exact": arrays_bit_exact(outputs["q22_h0"], q0),
        "h0_r0_wrist_bit_exact": arrays_bit_exact(outputs["relative_wrist_T_h0"], wrist0),
        "wrist_h1_frozen_bit_exact": wrist_exact,
        "outside_allowed_q22_scope_bit_exact": outside_scope_exact,
        "invalid_rows_bit_exact": invalid_exact,
        "max_abs_delta_q_rad": maximum,
        "modified_side_frame_count": int(applied.sum()),
        "modified_frame_ids": np.flatnonzero(applied[:, PHYSICAL_SIDE]).astype(int).tolist(),
        "core_frame_ids": list(CORE_FRAMES),
        "evaluation_frame_ids_frozen": list(CORE_FRAMES),
        "evaluation_coverage_before": len(CORE_FRAMES) if eligible else 0,
        "evaluation_coverage_after": len(CORE_FRAMES) if eligible else 0,
        "evaluation_coverage_not_reduced": True,
        "motion_before": motion_before,
        "motion_after": motion_after,
        "timestamp_motion_not_degraded": motion_not_degraded,
        "transition_duration_left_s": (
            float(timestamps[CORE_FRAMES[0]] - timestamps[LEFT_TAPER_FRAMES[0]])
            if eligible else 0.0
        ),
        "transition_duration_right_s": (
            float(timestamps[RIGHT_TAPER_FRAMES[-1]] - timestamps[CORE_FRAMES[-1]])
            if eligible else 0.0
        ),
        "transition_zero_at_outer_boundaries": bool(
            not eligible or (
                envelope[LEFT_TAPER_FRAMES[0]] == 0
                and envelope[RIGHT_TAPER_FRAMES[-1]] == 0
            )
        ),
    }
    return outputs, diagnostics


def hand_only_self_collision(
    q22: np.ndarray, modified_mask: np.ndarray, expected_joint_order: Sequence[str],
) -> dict[str, Any]:
    """Audit every modified physical-left side-frame with pinned hand geometry."""

    frames = np.flatnonzero(np.asarray(modified_mask, bool)).astype(int).tolist()
    if not frames:
        return {
            "status": "NOT_APPLICABLE_NO_MODIFIED_FRAMES",
            "engine": None,
            "checked_frame_ids": [],
            "all_modified_side_frames_checked": True,
            "illegal_contact_count": 0,
            "adoption_authority": False,
        }
    try:
        import pybullet as bullet

        client = bullet.connect(bullet.DIRECT)
        try:
            body = bullet.loadURDF(
                str(LEFT_HAND_URDF.resolve(strict=True)), useFixedBase=True,
                flags=bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                physicsClientId=client,
            )
            joints = []
            collision_shapes = 0
            for index in range(-1, bullet.getNumJoints(body, physicsClientId=client)):
                collision_shapes += int(bool(
                    bullet.getCollisionShapeData(body, index, physicsClientId=client)
                ))
            for index in range(bullet.getNumJoints(body, physicsClientId=client)):
                info = bullet.getJointInfo(body, index, physicsClientId=client)
                if info[2] != bullet.JOINT_FIXED:
                    joints.append((index, info[1].decode("utf-8")))
            if [name for _, name in joints] != list(expected_joint_order):
                raise R1HypothesisError("PyBullet/R0 physical-left joint order mismatch")
            if collision_shapes == 0:
                raise R1HypothesisError("pinned KaiHand left URDF has no collision geometry")
            rows: list[dict[str, Any]] = []
            illegal_total = 0
            for frame in frames:
                for (joint_index, _), value in zip(joints, q22[frame, PHYSICAL_SIDE], strict=True):
                    bullet.resetJointState(
                        body, joint_index, float(value), physicsClientId=client,
                    )
                bullet.performCollisionDetection(physicsClientId=client)
                contacts = [
                    {
                        "link_a": int(point[3]),
                        "link_b": int(point[4]),
                        "contact_distance_m": float(point[8]),
                        "penetration_depth_m": float(-point[8]),
                    }
                    for point in bullet.getContactPoints(body, body, physicsClientId=client)
                    if float(point[8]) < -1e-4
                ]
                illegal_total += len(contacts)
                rows.append({
                    "frame_id": frame,
                    "illegal_contact_count": len(contacts),
                    "contacts": contacts,
                })
        finally:
            bullet.disconnect(client)
    except Exception as error:
        return {
            "status": "UNVERIFIED_FAIL_CLOSED",
            "engine": "PYBULLET_PINNED_KAIHAND_LEFT_URDF",
            "checked_frame_ids": [],
            "all_modified_side_frames_checked": False,
            "illegal_contact_count": None,
            "error": f"{type(error).__name__}:{error}",
            "adoption_authority": False,
        }
    return {
        "status": "PASS_HAND_ONLY_SELF_COLLISION" if illegal_total == 0 else "FAILED_HAND_ONLY_SELF_COLLISION",
        "engine": "PYBULLET_PINNED_KAIHAND_LEFT_URDF",
        "urdf": ref(LEFT_HAND_URDF),
        "collision_shape_count": collision_shapes,
        "penetration_tolerance_m": 1e-4,
        "checked_frame_ids": frames,
        "all_modified_side_frames_checked": len(rows) == len(frames),
        "illegal_contact_count": illegal_total,
        "frames": rows,
        "scope": "KAIHAND_LEFT_HAND_ONLY_ALL_MODIFIED_SIDE_FRAMES",
        "full_robot_object_environment": "UNVERIFIED",
        "adoption_authority": False,
    }


def _draw_hand(
    panel: np.ndarray, model: Any, q: np.ndarray, center: tuple[int, int], color: tuple[int, int, int],
) -> None:
    moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
    transforms = forward_kinematics(
        model, {joint.name: float(q[index]) for index, joint in enumerate(moving)},
    )
    positions = {name: matrix[:3, 3] for name, matrix in transforms.items()}
    origin = positions[model.root_link]
    scale = 1500.0
    projected = {
        name: np.asarray([
            center[0] + scale * (point[0] - origin[0]),
            center[1] - scale * (point[1] - origin[1]),
        ], np.int32)
        for name, point in positions.items()
    }
    for joint in model.joints:
        cv2.line(
            panel, tuple(projected[joint.parent]), tuple(projected[joint.child]),
            color, 3, cv2.LINE_AA,
        )
    for point in projected.values():
        cv2.circle(panel, tuple(point), 3, (240, 240, 240), -1, cv2.LINE_AA)


def decode_video(path: Path, expected_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    count = width = height = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        count += 1
    capture.release()
    result = {
        "frames": count, "expected_frames": expected_frames,
        "width": width, "height": height,
        "status": "PASS_FULL_DECODE" if count == expected_frames else "FAILED_DECODE",
    }
    if count != expected_frames:
        raise R1HypothesisError(f"R1 review decode mismatch: {count} != {expected_frames}")
    return result


def render_review(
    path: Path, *, session_id: str, arrays: Mapping[str, np.ndarray], attempted: bool,
    assets: Any,
) -> dict[str, Any]:
    q0 = np.asarray(arrays["q22_h0"])
    q1 = np.asarray(arrays["q22_h1_rejected"])
    valid = np.asarray(arrays["valid_side_frame"], bool)
    applied = np.asarray(arrays["counterfactual_applied_mask"], bool)
    frame_count = q0.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 480),
    )
    if not writer.isOpened():
        raise R1HypothesisError("failed to open R1 review writer")
    try:
        for frame in range(frame_count):
            image = np.full((480, 1280, 3), 24, np.uint8)
            for panel_id, (q, title, color) in enumerate((
                (q0, "R0 BASELINE", (255, 160, 60)),
                (q1, "R1-H REJECTED COUNTERFACTUAL", (60, 190, 255)),
            )):
                x0 = panel_id * 640
                cv2.putText(
                    image, title, (x0 + 20, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.66, (245, 245, 245), 2, cv2.LINE_AA,
                )
                for side in range(2):
                    if valid[frame, side]:
                        model = assets.left_hand if side == 0 else assets.right_hand
                        _draw_hand(image, model, q[frame, side], (x0 + 180 + side * 285, 250), color)
                    else:
                        cv2.putText(
                            image, f"side {side}: UNKNOWN", (x0 + 45 + side * 280, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA,
                        )
                marker = "H1 MODIFIED" if applied[frame].any() else "BIT-EXACT R0"
                cv2.putText(
                    image, marker, (x0 + 20, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (80, 190, 255) if applied[frame].any() else (190, 190, 190),
                    1, cv2.LINE_AA,
                )
            cv2.line(image, (640, 0), (640, 480), (90, 90, 90), 2)
            cv2.putText(
                image, f"{session_id} | frame {frame:04d} | attempted={attempted}",
                (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA,
            )
            cv2.putText(
                image, "OPTIMIZATION ONLY - NOT ADOPTION EVIDENCE - NON CONTROL / NON DEPLOYABLE",
                (660, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (90, 130, 255), 1, cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()
    return {"video": ref(path), "decode": decode_video(path, frame_count)}


def run_session(
    row: Mapping[str, Any], *, contact_eligibility: Mapping[str, Any],
    output: Path, visual: Path, assets: Any,
) -> dict[str, Any]:
    session_id = str(row["session_id"])
    task = EXPECTED_SESSIONS[session_id]
    r0_result_path = verify_ref(row["result"])
    r0_result = load_json(r0_result_path)
    if (
        r0_result.get("session_id") != session_id
        or r0_result.get("r0_exported") is not True
        or r0_result.get("control_ground_truth") is not False
        or r0_result.get("training_eligible") is not False
        or r0_result.get("physical_deployment_authorized") is not False
    ):
        raise R1HypothesisError(f"R0 session contract drift: {session_id}")
    r0_npz = verify_ref(r0_result["arrays"])
    arrays = load_r0_arrays(r0_npz)
    frame_count = arrays["q22_init"].shape[0]
    if frame_count != int(r0_result["frame_count"]):
        raise R1HypothesisError("R0 result/NPZ frame count mismatch")
    eligible = bool(
        session_id == HYPOTHESIS_SESSION
        and contact_eligibility["h1_counterfactual_eligible"]
    )
    limits = r0_result.get("q22_limits", {}).get("left", {})
    outputs, diagnostics = build_counterfactual(
        arrays, eligible=eligible,
        lower_rad=limits.get("lower_rad", []), upper_rad=limits.get("upper_rad", []),
    )
    joint_order = r0_result.get("q22_joint_order", {}).get("left", [])
    collision = hand_only_self_collision(
        outputs["q22_h1_rejected"],
        outputs["counterfactual_applied_mask"][:, PHYSICAL_SIDE],
        joint_order,
    )
    h1_blocker = (
        "NO_OBJECT_INDEPENDENT_METRIC_WRIST_PLACEMENT" if eligible
        else (
            contact_eligibility["h1_eligibility_blocker"]
            if session_id == HYPOTHESIS_SESSION
            else "NO_FORMAL_SESSION_HYPOTHESIS"
        )
    )
    final = output / "sessions" / task / session_id
    if final.exists() or final.is_symlink():
        raise R1HypothesisError(f"fresh session output required: {final}")
    stage = output / f".session-staging-{session_id}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    npz_stage = stage / "KAI22_R1_HYPOTHESIS_V1.npz"
    atomic_npz(npz_stage, **outputs)
    review_stage = stage / "KAI22_R0_VS_R1_H_REJECTED_REVIEW.mp4"
    review = render_review(
        review_stage, session_id=session_id, arrays=outputs, attempted=eligible, assets=assets,
    )
    result = {
        "schema_version": "kai22-r1-hypothesis-session-v1",
        "task_id": TASK_ID,
        "session_id": session_id,
        "task": task,
        "source_group": r0_result["source_group"],
        "frame_count": frame_count,
        "status": "COMPLETED_DUAL_TERMINAL",
        "r1_e": {
            "status": "BLOCKED_LOCAL_EVIDENCE",
            "attempted": False,
            "exported": False,
            "adopted": False,
            "first_blocker": "NO_STRICT_METRIC_CONTACT_WINDOW",
            "strict_metric_contact_windows": 0,
        },
        "r1_h": {
            "status": "REJECTED_COUNTERFACTUAL_NOT_ADOPTED" if eligible else "BLOCKED_NO_FORMAL_HYPOTHESIS",
            "attempted": eligible,
            "exported": eligible,
            "adopted": False,
            "first_blocker": h1_blocker,
            "scope": {
                "anatomical_hand": "right",
                "physical_hand": "kaihand_left",
                "physical_side_index": PHYSICAL_SIDE,
                "joint_indices_half_open": [6, 10],
                "joint_role": "index_flexion_only",
                "core_frames_inclusive": [92, 98],
                "left_taper_frames_inclusive": [86, 91],
                "right_taper_frames_inclusive": [99, 104],
                "max_taper_seconds": 0.2,
                "max_abs_delta_q_rad": MAX_ABS_DELTA_RAD,
            },
            "optimization_semantics": "OPTIMIZATION_ONLY_NOT_ADOPTION_EVIDENCE",
            "pad_or_object_distance_used_as_adoption_evidence": False,
            "object_independent_metric_wrist_placement": "ABSENT",
            "wrist_translation_orientation": "FROZEN_BIT_EXACT_R0",
        },
        "diagnostics": diagnostics,
        "robot_self_collision": collision,
        "collision_scope": "KAIHAND_LEFT_HAND_ONLY_ALL_MODIFIED_SIDE_FRAMES",
        "full_robot_object_environment_collision": "UNVERIFIED",
        "arrays": projected_ref(npz_stage, final / npz_stage.name),
        "review": {
            **review,
            "video": projected_ref(review_stage, final / review_stage.name),
        },
        "r0_input": ref(r0_npz),
        "r0_result": ref(r0_result_path),
        "authority": AUTHORITY,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "contact_ground_truth": False,
        "source_mutated": False,
        "claim_limit": CLAIM_LIMIT,
    }
    validate_session_contract(result)
    atomic_json(stage / "RESULT.json", result)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    published = load_json(final / "RESULT.json")
    verify_ref(published["arrays"])
    verify_ref(published["review"]["video"])
    shallow = visual / f"{session_id}_KAI22_R0_VS_R1_H_REJECTED_REVIEW.mp4"
    try:
        os.link(final / "KAI22_R0_VS_R1_H_REJECTED_REVIEW.mp4", shallow)
    except OSError:
        shutil.copy2(final / "KAI22_R0_VS_R1_H_REJECTED_REVIEW.mp4", shallow)
    if sha256(shallow) != published["review"]["video"]["sha256"]:
        raise R1HypothesisError("shallow R1 review differs from atomic session review")
    return {
        "session_id": session_id,
        "task": task,
        "source_group": r0_result["source_group"],
        "frame_count": frame_count,
        "status": "COMPLETED_DUAL_TERMINAL",
        "r1_e_status": "BLOCKED_LOCAL_EVIDENCE",
        "r1_e_attempted": False,
        "r1_e_exported": False,
        "r1_e_adopted": False,
        "r1_h_status": published["r1_h"]["status"],
        "r1_h_attempted": eligible,
        "r1_h_exported": eligible,
        "r1_h_adopted": False,
        "first_blocker": h1_blocker,
        "result": ref(final / "RESULT.json"),
        "arrays": ref(final / "KAI22_R1_HYPOTHESIS_V1.npz"),
        "review": {**published["review"], "video": ref(shallow)},
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "sessions_total": len(rows),
        "sessions_terminal": sum(row.get("status") == "COMPLETED_DUAL_TERMINAL" for row in rows),
        "failed_runtime": sum(row.get("status") == "FAILED_RUNTIME" for row in rows),
        "r1_e_attempted": sum(bool(row.get("r1_e_attempted")) for row in rows),
        "r1_e_exported": sum(bool(row.get("r1_e_exported")) for row in rows),
        "r1_e_adopted": sum(bool(row.get("r1_e_adopted")) for row in rows),
        "r1_e_blocked_local_evidence": sum(
            row.get("r1_e_status") == "BLOCKED_LOCAL_EVIDENCE" for row in rows
        ),
        "r1_h_attempted": sum(bool(row.get("r1_h_attempted")) for row in rows),
        "r1_h_exported": sum(bool(row.get("r1_h_exported")) for row in rows),
        "r1_h_adopted": sum(bool(row.get("r1_h_adopted")) for row in rows),
    }


def validate_fixed_paths(output: Path, visual: Path, receipt: Path) -> None:
    if output.resolve() != OUTPUT.resolve():
        raise R1HypothesisError(f"output root must equal {OUTPUT}")
    if visual.resolve() != VISUAL.resolve():
        raise R1HypothesisError(f"visual root must equal {VISUAL}")
    if receipt.resolve() != RECEIPT.resolve():
        raise R1HypothesisError(f"receipt must equal {RECEIPT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    output, visual, receipt = (
        args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve(),
    )
    validate_fixed_paths(output, visual, receipt)
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise R1HypothesisError("positive epoch and fencing token >=16 characters required")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise R1HypothesisError(f"fresh output required: {path}")
    packet, packet_path = validate_route()
    input_paths = [
        CONTACT_ROOT / "CONTACT_EVIDENCE_LEDGER.json",
        CONTACT_ROOT / "CONTACT_HYPOTHESIS_LEDGER.json",
        CONTACT_ROOT / "BATCH_RESULT.json",
        R0_ROOT / "BATCH_RESULT.json",
        CONTRACT,
    ]
    for path in input_paths:
        if not path.is_file():
            raise R1HypothesisError(f"required upstream artifact missing: {path}")
    contact_eligibility = validate_contact_ledgers(
        load_json(input_paths[0]), load_json(input_paths[1]), load_json(input_paths[2]),
    )
    r0_rows = validate_r0_batch(load_json(input_paths[3]))
    signature_payload = {
        "schema_version": "0915-robot15h-r1-hypothesis-wave0-run-signature-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": "ABSENT",
        "gpu_used": False,
        "task_packet": ref(packet_path),
        "inputs": [ref(path) for path in input_paths],
        "code": [ref(Path(__file__))],
        "frozen_scope": {
            "session": HYPOTHESIS_SESSION,
            "anatomical_to_physical": "right_to_kaihand_left",
            "joint_indices_half_open": [6, 10],
            "core_frames_inclusive": [92, 98],
            "taper_frames_inclusive": [[86, 91], [99, 104]],
            "max_abs_delta_q_rad": MAX_ABS_DELTA_RAD,
        },
        "contact_ledger_is_only_eligibility_source": True,
        "self_certifying_object_or_pad_distance_forbidden": True,
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    fencing_sha = hashlib.sha256(args.fencing_token.encode()).hexdigest()
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-r1-hypothesis-writer-claim-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "gpu_used": False,
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": fencing_sha,
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(output),
    })
    heartbeat()
    assets = load_pinned_robot_assets(ROOT)
    rows: list[dict[str, Any]] = []
    for row in r0_rows:
        try:
            rows.append(run_session(
                row, contact_eligibility=contact_eligibility,
                output=output, visual=visual, assets=assets,
            ))
        except Exception as error:
            rows.append({
                "session_id": row.get("session_id"),
                "task": row.get("task"),
                "status": "FAILED_RUNTIME",
                "r1_e_status": "BLOCKED_LOCAL_EVIDENCE",
                "r1_e_attempted": False,
                "r1_e_exported": False,
                "r1_e_adopted": False,
                "r1_h_status": "FAILED_RUNTIME",
                "r1_h_attempted": False,
                "r1_h_exported": False,
                "r1_h_adopted": False,
                "first_blocker": f"{type(error).__name__}:{error}",
            })
        heartbeat()
    counts = summarize(rows)
    ledger = {
        "schema_version": "0915-robot15h-kai22-r1-eligibility-ledger-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL" if counts["failed_runtime"] == 0 else "PARTIAL_RUNTIME_FAILURE",
        "sessions": rows,
        "counts": counts,
        "contact_eligibility": contact_eligibility,
        "r1_e_global_status": "BLOCKED_LOCAL_EVIDENCE",
        "r1_h_global_status": "REJECTED_COUNTERFACTUAL_NOT_ADOPTED",
        "r1_h_success": 0,
        "adopted": 0,
        "first_blocker": "NO_OBJECT_INDEPENDENT_METRIC_WRIST_PLACEMENT",
        "optimization_semantics": "OPTIMIZATION_ONLY_NOT_ADOPTION_EVIDENCE",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
    }
    atomic_json(output / "R1_ELIGIBILITY_LEDGER.json", ledger)
    batch = {
        "schema_version": "0915-robot15h-kai22-r1-wave0-batch-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": ledger["status"],
        "sessions": rows,
        "counts": counts,
        "r1_e_success": 0,
        "r1_h_success": 0,
        "r1_h_adopted": 0,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
    }
    atomic_json(output / "BATCH_RESULT.json", batch)
    atomic_json(output / "METRICS.json", {
        "schema_version": "0915-robot15h-kai22-r1-wave0-metrics-v1",
        "task_id": TASK_ID,
        "counts": counts,
        "r1_e_success": 0,
        "r1_h_success": 0,
        "r1_h_adopted": 0,
        "counterfactual_object_or_pad_distance_metric": "NOT_COMPUTED_NOT_ADOPTION_EVIDENCE",
        "evaluation_frames_frozen": list(CORE_FRAMES),
        "coverage_deletion_allowed": False,
    })
    atomic_json(visual / "INDEX.json", {
        "schema_version": "0915-robot15h-kai22-r1-wave0-visual-index-v1",
        "task_id": TASK_ID,
        "status": "COMPLETE" if counts["failed_runtime"] == 0 else "PARTIAL",
        "videos": [row["review"]["video"] for row in rows if "review" in row],
        "visual_acceptance": "PENDING",
    })
    top_status = "PASSED" if counts["failed_runtime"] == 0 else "FAILED_RUNTIME_FINAL"
    result = {
        "schema_version": "0915-robot15h-kai22-r1-wave0-result-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": top_status,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": counts,
        "r1_e_success": 0,
        "r1_h_success": 0,
        "r1_h_adopted": 0,
        "first_blocker": "NO_OBJECT_INDEPENDENT_METRIC_WRIST_PLACEMENT",
        "weights": "ABSENT",
        "gpu_used": False,
        "authority": AUTHORITY,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "r1_eligibility_ledger": ref(output / "R1_ELIGIBILITY_LEDGER.json"),
        "batch_result": ref(output / "BATCH_RESULT.json"),
        "metrics": ref(output / "METRICS.json"),
        "visual_index": ref(visual / "INDEX.json"),
        "run_signature": ref(output / "RUN_SIGNATURE.json"),
        "writer_claim": ref(output / "CLAIM.json"),
        "source_mutated": False,
        "claim_limit": packet.get("claim_limit", CLAIM_LIMIT),
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-kai22-r1-wave0-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": top_status,
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
        "gpu_used": False,
    })
    print(json.dumps({"status": top_status, "counts": counts, "result": str(output / "RESULT.json")}))
    return 0 if top_status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
