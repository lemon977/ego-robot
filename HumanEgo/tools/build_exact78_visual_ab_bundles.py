#!/usr/bin/env python3
"""Build paired exact78 HUMAN_RAW_RGB/ROBOT_VIEW_RGB HumanEgo bundles.

This is the only builder for the visual-input ablation.  A session is admitted
only from a current per-session handoff and both branches are materialised in
one transaction from the same metadata, Robot action, HaWoR and Object6D
payloads.  The sole branch-dependent payload is ``rgb.png``.

``--validate-only`` is CPU/read-only.  It succeeds with a WAIT receipt while
the current batch handoff is absent or incomplete; it never searches legacy
directories and never substitutes historical Robot-C/fallback artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
COHORT = (
    PROJECT
    / "tasks/control/runs/20260908_nine_hour_156_pipeline_v1/COHORT_EXACT78_FROZEN.json"
)
COHORT_SHA256 = "d482ae4b627efa78653b69af26a357cec66381e3416dd78dd278504579e7790e"
RECIPE = CONTROL / "TRAINING_RECIPE_REBIND_CONTRACT.json"
AB_CONTRACT = CONTROL / "HUMANEGO_VISUAL_INPUT_AB_COMPARISON_CONTRACT_V1.json"
DEFAULT_HANDOFF = CONTROL / "batch_training_handoff_v1/BATCH_TRAINING_HANDOFF.json"
DEFAULT_PREPARE = CONTROL / "training_exact78_visual_ab_prepare_v1"
BRANCHES = ("HUMAN_RAW_RGB", "ROBOT_VIEW_RGB")
ROLES = ("train", "validation", "test", "heldout")
TRAINING_ROLES = ("train", "validation", "test")
FORBIDDEN_PATH_PARTS = (
    "/archive/",
    "/legacy_runs/",
    "fail_forward_prod",
    "fallback",
    "robot_background_agnostic_quick_preview",
    "motion_training_label_grade_b",
)


class BuildHold(RuntimeError):
    """A fail-closed admission error."""


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BuildHold(f"JSON object required: {path}")
    return value


def file_ref(path: Path, *, published: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str((published or path).absolute()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_ref(value: Mapping[str, Any], label: str) -> Path:
    try:
        path = Path(str(value["path"])).resolve(strict=True)
    except Exception as error:
        raise BuildHold(f"{label}: missing file reference") from error
    if not path.is_file():
        raise BuildHold(f"{label}: not a file")
    if path.stat().st_size != value.get("bytes") or sha256(path) != value.get("sha256"):
        raise BuildHold(f"{label}: SHA/size drift")
    lowered = str(path).lower()
    if any(marker in lowered for marker in FORBIDDEN_PATH_PARTS):
        raise BuildHold(f"{label}: forbidden old/fallback lineage")
    return path


def atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    if not replace and (path.exists() or path.is_symlink()):
        raise BuildHold(f"no-clobber: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def cohort_rows(task: str) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    if sha256(COHORT) != COHORT_SHA256:
        raise BuildHold("exact78 cohort SHA drift")
    payload = read_json(COHORT)
    rows = {
        str(row["session_id"]): row
        for row in payload["sessions"]
        if row.get("task") == task
    }
    roles = {
        role: [
            str(row["session_id"])
            for row in payload["sessions"]
            if row.get("task") == task and row.get("split") == role
        ]
        for role in ROLES
    }
    expected = {"train": 60, "validation": 8, "test": 5, "heldout": 5}
    if {key: len(value) for key, value in roles.items()} != expected:
        raise BuildHold(f"invalid exact78 split for {task}")
    return roles, rows


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _root(row: Mapping[str, Any], key: str) -> Path:
    path = Path(str(row.get(key, ""))).resolve(strict=True)
    lowered = str(path).lower()
    if any(marker in lowered for marker in FORBIDDEN_PATH_PARTS):
        raise BuildHold(f"{key}: forbidden old/fallback lineage")
    if not path.is_dir():
        raise BuildHold(f"{key}: directory missing")
    return path


def inspect_session(
    session: str, expected_role: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    if row.get("session") != session or row.get("split") != expected_role:
        raise BuildHold("handoff session/split mismatch")
    if row.get("status") != "READY" or row.get("current_authority") is not True:
        raise BuildHold(f"handoff not current READY: {row.get('status')}")
    if row.get("robot_grade") not in {"A", "B"}:
        raise BuildHold("Robot grade is not A/B")
    if row.get("downstream_authorized") is not True:
        raise BuildHold("Robot downstream not authorized")

    receipt = verify_ref(row.get("robot_result", {}), "Robot result")
    result = read_json(receipt)
    if (
        result.get("session") != session
        or result.get("downstream_authorized") is not True
        or result.get("grade") not in {"A", "B"}
    ):
        raise BuildHold("Robot RESULT does not authorize training")

    action = verify_ref(row.get("robot_action_sidecar", {}), "Robot action sidecar")
    hawor = verify_ref(row.get("hawor_sidecar", {}), "HaWoR sidecar")
    object_npz = verify_ref(row.get("object_state_npz", {}), "Object6D NPZ")
    object_json = verify_ref(row.get("object_state_json", {}), "Object6D JSON")
    metadata_root = _root(row, "metadata_adapter_root")
    human_root = _root(row, "human_raw_rgb_adapter_root")
    robot_root = _root(row, "robot_view_rgb_adapter_root")

    with np.load(action, allow_pickle=False) as archive:
        required = {
            "frame_names",
            "q",
            "wrist_T_camera",
            "valid",
            "confidence",
            "ik_warm_start_used",
            "joint_lower",
            "joint_upper",
            "translation_unit",
            "q_unit",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise BuildHold(f"Robot action fields missing: {missing}")
        names = [str(value) for value in archive["frame_names"]]
        q = np.asarray(archive["q"], dtype=np.float32)
        wrist = np.asarray(archive["wrist_T_camera"], dtype=np.float64)
        valid = np.asarray(archive["valid"], dtype=bool)
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
        warm = bool(np.asarray(archive["ik_warm_start_used"]).item())
        lower = np.asarray(archive["joint_lower"], dtype=np.float32)
        upper = np.asarray(archive["joint_upper"], dtype=np.float32)
        translation_unit = str(np.asarray(archive["translation_unit"]).item())
        q_unit = str(np.asarray(archive["q_unit"]).item())
        arm_q = (
            np.asarray(archive["arm_q"], dtype=np.float32)
            if "arm_q" in archive.files else None
        )
    count = len(names)
    if (
        q.shape != (count, 2, 22)
        or wrist.shape != (count, 2, 4, 4)
        or valid.shape != (count, 2)
        or confidence.shape != (count, 2)
        or len(set(names)) != count
    ):
        raise BuildHold("Robot action shape/frame contract failed")
    if not warm or not np.isfinite(q).all() or not np.isfinite(wrist).all():
        raise BuildHold("Robot action warm-start/finite contract failed")
    if (
        lower.shape != (2, 22)
        or upper.shape != (2, 22)
        or translation_unit != "metre"
        or q_unit != "radian"
        or np.any(q < lower[None] - 1e-6)
        or np.any(q > upper[None] + 1e-6)
    ):
        raise BuildHold("Robot action unit/joint-limit contract failed")
    consecutive = valid[1:] & valid[:-1]
    if len(q) > 1 and np.any(
        np.max(np.abs(np.diff(q, axis=0)), axis=-1)[consecutive] > 0.080001
    ):
        raise BuildHold("Robot KaiHand step exceeds 0.08 rad/frame")
    if arm_q is not None:
        if arm_q.shape[:2] != (count, 2) or not np.isfinite(arm_q).all():
            raise BuildHold("Robot arm_q shape/finite contract failed")
        if len(arm_q) > 1 and float(np.max(np.abs(np.diff(arm_q, axis=0)))) > 0.120001:
            raise BuildHold("Robot arm step exceeds 0.12 rad/frame")
    if longest_true_run(valid.all(axis=1)) < 51:
        raise BuildHold("Robot action has no H50+future support")

    frame_to_index = {int(value): index for index, value in enumerate(names)}
    starts: list[int] = []
    for start in sorted(frame_to_index):
        indices = [frame_to_index.get(start + offset) for offset in range(50)]
        if None in indices or not bool(valid[np.asarray(indices, dtype=np.int64)].all()):
            continue
        names51 = [f"{start + offset:05d}" for offset in range(51)]
        if all(
            (metadata_root / "preprocess/all_data" / name / "training_data.json").is_file()
            and (human_root / "preprocess/all_data" / name / "rgb.png").is_file()
            and (robot_root / "preprocess/all_data" / name / "rgb.png").is_file()
            for name in names51
        ):
            starts.append(start)
    if not starts:
        raise BuildHold("no strictly paired H50 window")
    frames = sorted({start + offset for start in starts for offset in range(51)})
    object_manifest = read_json(object_json)
    grade = str(object_manifest.get("quality_grade", row.get("object_grade", "B")))
    if grade not in {"A", "B"}:
        raise BuildHold("Object6D grade is not A/B")
    weight = float(object_manifest.get("training_weight", 1.0 if grade == "A" else 0.25))
    if not 0.0 < weight <= 1.0:
        raise BuildHold("invalid Object6D training weight")
    thresholds = object_manifest.get("confidence_thresholds") or row.get(
        "object_confidence_thresholds"
    )
    if not isinstance(thresholds, dict) or not thresholds:
        raise BuildHold("Object6D confidence thresholds missing")
    return {
        "session": session,
        "role": expected_role,
        "robot_result": receipt,
        "action": action,
        "hawor": hawor,
        "object_npz": object_npz,
        "object_json": object_json,
        "metadata_root": metadata_root,
        "human_root": human_root,
        "robot_root": robot_root,
        "frames": frames,
        "window_starts": starts,
        "object_grade": grade,
        "object_weight": weight,
        "object_thresholds": {str(k): float(v) for k, v in thresholds.items()},
        "longest_dual_valid": longest_true_run(valid.all(axis=1)),
        "arm_q_present": arm_q is not None,
    }


def inspect(task: str, handoff_path: Path) -> dict[str, Any]:
    roles, cohort = cohort_rows(task)
    if not handoff_path.is_file():
        return {
            "task": task,
            "roles": roles,
            "cohort": cohort,
            "handoff": None,
            "admitted": {role: [] for role in TRAINING_ROLES},
            "rejected": {
                session: "WAIT_ROBOT_ACTION_SIDECAR"
                for role in TRAINING_ROLES
                for session in roles[role]
            },
        }
    handoff = read_json(handoff_path)
    if (
        handoff.get("schema_version")
        != "exact78-current-visual-ab-training-handoff-v1"
        or handoff.get("cohort_sha256") != COHORT_SHA256
    ):
        raise BuildHold("batch training handoff schema/cohort mismatch")
    rows = handoff.get("sessions")
    if not isinstance(rows, dict):
        raise BuildHold("handoff sessions object missing")
    admitted: dict[str, list[dict[str, Any]]] = {role: [] for role in TRAINING_ROLES}
    rejected: dict[str, str] = {}
    for role in TRAINING_ROLES:
        for session in roles[role]:
            row = rows.get(session)
            if not isinstance(row, dict):
                rejected[session] = "WAIT_ROBOT_ACTION_SIDECAR"
                continue
            try:
                admitted[role].append(inspect_session(session, role, row))
            except Exception as error:
                rejected[session] = f"{type(error).__name__}:{error}"
    return {
        "task": task,
        "roles": roles,
        "cohort": cohort,
        "handoff": handoff_path,
        "admitted": admitted,
        "rejected": rejected,
    }


def readiness(audit: Mapping[str, Any]) -> dict[str, Any]:
    counts = {role: len(audit["admitted"][role]) for role in TRAINING_ROLES}
    windows = {
        role: sum(len(row["window_starts"]) for row in audit["admitted"][role])
        for role in TRAINING_ROLES
    }
    minimum_sessions = {"train": 16, "validation": 3, "test": 0}
    minimum_windows = {"train": 256, "validation": 48, "test": 0}
    batch_handoff_complete = bool(
        isinstance(audit.get("handoff"), Path)
        and audit.get("handoff")
        and read_json(audit["handoff"]).get("status") == "COMPLETE"
    )
    minimum_met = all(counts[k] >= v for k, v in minimum_sessions.items()) and all(
        windows[k] >= v for k, v in minimum_windows.items()
    )
    ready = batch_handoff_complete and minimum_met
    return {
        "ready_for_optimizer": ready,
        "batch_handoff_complete": batch_handoff_complete,
        "minimum_admission_met": minimum_met,
        "admitted_counts": counts,
        "paired_h50_windows": windows,
        "minimum_sessions": minimum_sessions,
        "minimum_windows": minimum_windows,
    }


def preparation_payload(audits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    split = {
        task: {role: audits[task]["roles"][role] for role in ROLES}
        for task in ("chips", "poker")
    }
    session_plan: dict[str, Any] = {}
    for task in ("chips", "poker"):
        for role in ROLES:
            for session in audits[task]["roles"][role]:
                cohort_row = audits[task]["cohort"][session]
                session_plan[session] = {
                    "task": task,
                    "split": role,
                    "date": cohort_row.get("date"),
                    "raw_path": cohort_row.get("raw_path"),
                    "raw_video_sha256": cohort_row.get("raw_video_sha256"),
                    "expected_handoff": str(DEFAULT_HANDOFF),
                    "required_pair": {
                        "shared": [
                            "robot_action_sidecar",
                            "hawor_sidecar",
                            "object_state_npz",
                            "object_state_json",
                            "metadata_adapter_root",
                            "frame_set",
                        ],
                        "branch_only": {
                            "HUMAN_RAW_RGB": "human_raw_rgb_adapter_root/*/rgb.png",
                            "ROBOT_VIEW_RGB": "robot_view_rgb_adapter_root/*/rgb.png",
                        },
                    },
                    "calibration_expected": cohort_row.get("date") != "0901",
                }
    return {
        "schema_version": "exact78-visual-ab-preparation-v1",
        "created_at": now(),
        "status": "WAIT_ROBOT_ACTION_SIDECAR",
        "cohort": file_ref(COHORT),
        "contracts": {
            "visual_ab": file_ref(AB_CONTRACT),
            "recipe": file_ref(RECIPE),
        },
        "split": split,
        "split_sha256": json_digest(split),
        "session_plan": session_plan,
        "pairing": {
            "branches": list(BRANCHES),
            "seed": 7,
            "pred_horizon": 50,
            "same_action_sidecar": True,
            "same_split": True,
            "same_frame_set": True,
            "same_metadata": True,
            "same_hawor_object_ict": True,
            "only_difference": "rgb.png bytes",
        },
        "claim_limit": (
            "Preparation covers 78 sessions per task and freezes 60/8/5/5. "
            "Heldout is never optimizer input. Missing-calibration 0901 rows remain "
            "honest C_CALIBRATION_MISSING and cannot receive fabricated Robot data."
        ),
    }


def write_preparation(prepare_root: Path, audits: Mapping[str, Mapping[str, Any]]) -> None:
    payload = preparation_payload(audits)
    immutable = {
        "SPLIT_AND_PAIRING_PLAN.json": payload,
        "HUMAN_RAW_RGB_SELECTOR_PLAN.json": {
            "schema_version": "exact78-visual-selector-plan-v1",
            "branch": "HUMAN_RAW_RGB",
            "split_sha256": payload["split_sha256"],
            "sessions": payload["session_plan"],
        },
        "ROBOT_VIEW_RGB_SELECTOR_PLAN.json": {
            "schema_version": "exact78-visual-selector-plan-v1",
            "branch": "ROBOT_VIEW_RGB",
            "split_sha256": payload["split_sha256"],
            "sessions": payload["session_plan"],
        },
    }
    for name, value in immutable.items():
        target = prepare_root / name
        if target.is_file():
            existing = read_json(target)
            # created_at is evidence, not part of the immutable comparison.
            left = {k: v for k, v in existing.items() if k != "created_at"}
            right = {k: v for k, v in value.items() if k != "created_at"}
            if left != right:
                raise BuildHold(f"immutable preparation drift: {target}")
        else:
            atomic_json(target, value, replace=False)
    status = {
        "schema_version": "exact78-visual-ab-readiness-v1",
        "updated_at": now(),
        "status": (
            "READY_FOR_PAIRED_BUNDLE_BUILD"
            if all(readiness(audits[t])["ready_for_optimizer"] for t in audits)
            else "WAIT_ROBOT_ACTION_SIDECAR"
        ),
        "handoff": str(DEFAULT_HANDOFF),
        "tasks": {task: readiness(audits[task]) for task in audits},
        "rejection_counts": {
            task: dict(Counter(audits[task]["rejected"].values())) for task in audits
        },
        "training_started": False,
        "gpu_used": False,
    }
    atomic_json(prepare_root / "READINESS.json", status, replace=True)


def copy_ordinary(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
    if destination.stat().st_nlink != 1:
        raise BuildHold(f"ordinary-file nlink gate failed: {destination}")


def _published(temporary: Path, bundle: Path, path: Path) -> Path:
    return bundle / path.relative_to(temporary)


def materialize_branch(
    *, branch: str, task: str, audit: Mapping[str, Any], bundle: Path,
    published_bundle: Path, pairing_id: str,
) -> None:
    if bundle.exists() or bundle.is_symlink():
        raise BuildHold(f"fresh bundle already exists: {bundle}")
    temporary = bundle.with_name(f".{bundle.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise BuildHold(f"temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    selector_sessions: dict[str, Any] = {}
    paired_windows: dict[str, Any] = {}
    sidecars: dict[str, str] = {}
    hawor_sidecars: dict[str, str] = {}
    object_npz: dict[str, str] = {}
    object_json: dict[str, str] = {}
    grades: dict[str, str] = {}
    weights: dict[str, float] = {}
    receipts: dict[str, str] = {}
    thresholds: dict[str, float] | None = None
    session_meta: dict[str, Any] = {}
    admitted = {
        role: [row["session"] for row in audit["admitted"][role]]
        for role in TRAINING_ROLES
    }
    for role in TRAINING_ROLES:
        for row in audit["admitted"][role]:
            session = row["session"]
            records = {}
            for frame in row["frames"]:
                name = f"{frame:05d}"
                destination = (
                    temporary / "production" / session / "09_humanego_adapter"
                    / "preprocess/all_data" / name
                )
                metadata = destination / "training_data.json"
                image = destination / "rgb.png"
                copy_ordinary(
                    row["metadata_root"] / "preprocess/all_data" / name
                    / "training_data.json",
                    metadata,
                )
                image_root = row["human_root"] if branch == "HUMAN_RAW_RGB" else row["robot_root"]
                copy_ordinary(image_root / "preprocess/all_data" / name / "rgb.png", image)
                records[name] = {
                    "metadata": file_ref(
                        metadata, published=_published(temporary, published_bundle, metadata)
                    ),
                    "image": file_ref(
                        image, published=_published(temporary, published_bundle, image)
                    ),
                    "unresolved": False,
                }
            selector_sessions[session] = {"frames": records}
            paired_windows[session] = {
                "frames": row["frames"],
                "window_starts": row["window_starts"],
            }
            action_dst = temporary / "sidecars/kai22" / session / "sidecar.npz"
            hawor_dst = temporary / "hawor_v3_sidecars" / session / "entities_hawor_v3.npz"
            obj_dir = temporary / "object_state_sidecars" / session
            obj_npz_dst = obj_dir / "AUTO_ESTIMATED_OBJECT_STATE.npz"
            obj_json_dst = obj_dir / "AUTO_ESTIMATED_OBJECT_STATE.json"
            for source, destination in (
                (row["action"], action_dst),
                (row["hawor"], hawor_dst),
                (row["object_npz"], obj_npz_dst),
                (row["object_json"], obj_json_dst),
            ):
                copy_ordinary(source, destination)
            sidecars[session] = sha256(action_dst)
            hawor_sidecars[session] = sha256(hawor_dst)
            object_npz[session] = sha256(obj_npz_dst)
            object_json[session] = sha256(obj_json_dst)
            grades[session] = row["object_grade"]
            weights[session] = row["object_weight"]
            receipts[session] = sha256(row["robot_result"])
            if thresholds is None:
                thresholds = row["object_thresholds"]
            elif thresholds != row["object_thresholds"]:
                raise BuildHold("Object6D thresholds differ across admitted sessions")
            session_meta[session] = {
                "role": role,
                "frames": len(row["frames"]),
                "h50_windows": len(row["window_starts"]),
                "longest_consecutive_dual_valid": row["longest_dual_valid"],
                "robot_result": file_ref(row["robot_result"]),
            }

    roles = audit["roles"]
    production = published_bundle / "production"
    split = {
        "schema_version": "humanego-newtask-robot-split-v4-exact78-visual-ab-formal-robot",
        "seed": 7,
        "split_unit": "session",
        "task": task,
        "visual_branch": branch,
        "pairing_id": pairing_id,
        **{role: roles[role] for role in ROLES},
        "admitted_splits": admitted,
        "selector_sessions": [s for role in TRAINING_ROLES for s in admitted[role]],
        "sessions": session_meta,
        "production_root": str(production),
        "sidecar_root": str(published_bundle / "sidecars"),
        "hawor_v3_sidecar_root": str(published_bundle / "hawor_v3_sidecars"),
        "ict_contract": {
            "hand_tracking_method": "hawor_v3",
            "single_hand": False,
            "ict_dim": 29,
            "frame_mode": "camera_frame",
            "translation_unit": "metre",
            "camera_coordinate_system": "x_right_y_down_z_forward",
            "object_token_policy": "FORMAL_OBJECT6D_FROZEN_PER_SESSION",
        },
        "motion_label_contract": {
            "consumption_mode": "FORMAL_ROBOT_ACTION_A_OR_B",
            "provenance_class": "FORMAL_ROBOT_ACTION_A_OR_B",
            "target": "accepted_tianji_dual_kaihand_action",
            "arm_consumed": False,
            "achieved_robot_fk_required": True,
            "sidecar_sha256_by_session": sidecars,
            "receipt_sha256_by_session": receipts,
            "motion_weight_by_session": {session: 1.0 for session in sidecars},
            "contact_aux_weight_by_session": {session: 0.0 for session in sidecars},
        },
        "object_state_contract": {
            "consumption_mode": (
                "training_admitted" if set(grades.values()) == {"A"}
                else "training_estimated_grade_b"
            ),
            "sidecar_root": str(published_bundle / "object_state_sidecars"),
            "npz_sha256_by_session": object_npz,
            "json_sha256_by_session": object_json,
            "quality_grade_by_session": grades,
            "training_weight_by_session": weights,
            "confidence_thresholds": thresholds or {},
        },
    }
    selector = {
        "schema_version": "humanego-newtask-robot-selector-v1",
        "product_line": branch,
        "image_name": "rgb.png",
        "artifact_root": str(production),
        "selector_root": str(production),
        "sessions": selector_sessions,
    }
    audit_payload = {
        "schema_version": "exact78-visual-ab-branch-audit-v1",
        "created_at": now(),
        "task": task,
        "branch": branch,
        "pairing_id": pairing_id,
        "cohort": file_ref(COHORT),
        "admitted_counts": {role: len(value) for role, value in admitted.items()},
        "heldout_in_optimizer": False,
        "old_robot_c_or_fallback_consumed": False,
        "only_branch_dependent_payload": "production/*/*/rgb.png",
    }
    payloads = {
        "split.json": split,
        "selector_records.json": selector,
        "paired_windows.json": paired_windows,
        "sidecars.json": sidecars,
        "hawor_v3_sidecars.json": hawor_sidecars,
        "object_state_sidecars.json": {
            session: {"npz_sha256": object_npz[session], "json_sha256": object_json[session]}
            for session in object_npz
        },
        "AUDIT.json": audit_payload,
    }
    for name, payload in payloads.items():
        atomic_json(temporary / name, payload, replace=False)
    freeze = {
        "schema_version": "humanego-newtask-robot-freeze-v4-exact78-visual-ab-formal-robot",
        "frozen_at": now(),
        "task": task,
        "visual_branch": branch,
        "pairing_id": pairing_id,
        "training_started": False,
        "bundle_files": {
            name: file_ref(
                temporary / name, published=published_bundle / name
            )
            for name in payloads
        },
    }
    atomic_json(temporary / "freeze.json", freeze, replace=False)
    os.replace(temporary, bundle)


def materialize_pair(task: str, audit: Mapping[str, Any], root: Path) -> dict[str, Any]:
    state = readiness(audit)
    if not state["ready_for_optimizer"]:
        raise BuildHold(f"WAIT_ROBOT_ACTION_SIDECAR:{state}")
    if root.exists() or root.is_symlink():
        raise BuildHold(f"pair root already exists: {root}")
    pairing_basis = {
        "task": task,
        "cohort_sha256": COHORT_SHA256,
        "seed": 7,
        "admitted": {
            role: [row["session"] for row in audit["admitted"][role]]
            for role in TRAINING_ROLES
        },
        "actions": {
            row["session"]: sha256(row["action"])
            for role in TRAINING_ROLES
            for row in audit["admitted"][role]
        },
        "frames": {
            row["session"]: row["frames"]
            for role in TRAINING_ROLES
            for row in audit["admitted"][role]
        },
        "windows": {
            row["session"]: row["window_starts"]
            for role in TRAINING_ROLES
            for row in audit["admitted"][role]
        },
    }
    pairing_id = json_digest(pairing_basis)
    temporary = root.with_name(f".{root.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise BuildHold(f"pair temporary exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for branch in BRANCHES:
            materialize_branch(
                branch=branch,
                task=task,
                audit=audit,
                bundle=temporary / branch,
                published_bundle=root / branch,
                pairing_id=pairing_id,
            )
        left = read_json(temporary / BRANCHES[0] / "split.json")
        right = read_json(temporary / BRANCHES[1] / "split.json")
        for key in ("visual_branch", "production_root", "sidecar_root", "hawor_v3_sidecar_root"):
            left.pop(key, None)
            right.pop(key, None)
        left["object_state_contract"].pop("sidecar_root", None)
        right["object_state_contract"].pop("sidecar_root", None)
        if left != right:
            raise BuildHold("paired branch split/control mismatch")
        pairing = {
            "schema_version": "exact78-visual-ab-pairing-freeze-v1",
            "created_at": now(),
            "task": task,
            "pairing_id": pairing_id,
            "controlled_payload_sha256": json_digest(pairing_basis),
            "branches": {
                branch: file_ref(
                    temporary / branch / "freeze.json",
                    published=root / branch / "freeze.json",
                )
                for branch in BRANCHES
            },
            "only_intended_difference": "rgb.png bytes",
            "shared_action_sidecar_sha256": pairing_basis["actions"],
            "shared_frames": pairing_basis["frames"],
            "shared_windows": pairing_basis["windows"],
            "seed": 7,
            "optimizer_started": False,
        }
        atomic_json(temporary / "PAIRING_MANIFEST.json", pairing, replace=False)
        os.replace(temporary, root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_json(root / "PAIRING_MANIFEST.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker", "all"), default="all")
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--prepare-root", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    tasks = ("chips", "poker") if args.task == "all" else (args.task,)
    audits = {task: inspect(task, args.handoff) for task in tasks}
    # Keep a full two-task immutable plan even when validating one task.
    plan_audits = audits
    if set(plan_audits) != {"chips", "poker"}:
        plan_audits = {
            task: inspect(task, args.handoff) for task in ("chips", "poker")
        }
    write_preparation(args.prepare_root.absolute(), plan_audits)
    report = {
        "schema_version": "exact78-visual-ab-builder-result-v1",
        "created_at": now(),
        "status": (
            "READY_FOR_PAIRED_BUNDLE_BUILD"
            if all(readiness(audits[t])["ready_for_optimizer"] for t in tasks)
            else "WAIT_ROBOT_ACTION_SIDECAR"
        ),
        "mode": "VALIDATE_ONLY" if args.validate_only else "MATERIALIZE",
        "cohort": file_ref(COHORT),
        "handoff": str(args.handoff.absolute()),
        "tasks": {task: readiness(audits[task]) for task in tasks},
        "rejected": {task: audits[task]["rejected"] for task in tasks},
        "training_started": False,
        "gpu_used": False,
        "old_robot_c_or_fallback_consumed": False,
    }
    if not args.validate_only:
        if args.bundle_root is None:
            raise BuildHold("--bundle-root required for materialization")
        report["bundles"] = {
            task: materialize_pair(
                task, audits[task], args.bundle_root.absolute() / task
            )
            for task in tasks
        }
        report["status"] = "PAIRED_BUNDLES_CREATED"
    if args.receipt:
        atomic_json(args.receipt.absolute(), report, replace=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
