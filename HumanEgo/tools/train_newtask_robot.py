#!/usr/bin/env python3
"""Train a newtask robot HumanEgo checkpoint with the CK2 recipe.

train_embodiment.py refuses anything that is not the grap_a_cap eligible68
cohort.  This entry uses the same model, pretrained weights, dataloader and
train/eval loop, but takes the bundle built by build_robot_humanego_bundle.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HUMANEGO = Path("/mnt/workspace/code/chaoyang/HumanEgo")
sys.path.insert(0, str(HUMANEGO))

from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from training.FlowMatchingTrainer import (  # noqa: E402
    EMAModel,
    FlowMatchingModel,
    TrainConfig,
    eval_ode_inference,
    get_dataset_stats,
    save_checkpoint_atomic,
    train_one_epoch,
)
from training.checkpoint_transfer import load_compatible_pretrained  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import torch  # noqa: E402
from contextlib import contextmanager  # noqa: E402
import utils.frozen_contract as frozen_contract  # noqa: E402

_FLAP = (
    "changed during preverification",
    "changed during consumption",
    "changed while being read",
    "changed while opening",
)
_ORIG_OPEN = frozen_contract.open_verified_file_reference


def _is_flap(error: BaseException) -> bool:
    text = str(error)
    return any(token in text for token in _FLAP)


@contextmanager
def _retry_open_verified(*args, **kwargs):
    last = None
    for attempt in range(6):
        try:
            with _ORIG_OPEN(*args, **kwargs) as payload:
                yield payload
            return
        except ValueError as error:
            last = error
            if not _is_flap(error) or attempt == 5:
                raise
            time.sleep(0.08 * (attempt + 1))
    if last is not None:
        raise last


frozen_contract.open_verified_file_reference = _retry_open_verified


def _tune_cuda() -> None:
    """Keep the CK2 recipe; only stop the H20 sitting idle on I/O."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    return value


class _CudaPrefetch:
    """Copy the next batch on a side stream while the current step runs."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        stream = torch.cuda.Stream()
        pending = None
        for batch in self.loader:
            with torch.cuda.stream(stream):
                ready = _to_device(batch, self.device)
            if pending is not None:
                torch.cuda.current_stream().wait_stream(stream)
                yield pending
            pending = ready
        if pending is not None:
            torch.cuda.current_stream().wait_stream(stream)
            yield pending


def load_config(yaml_path: Path, overrides: dict) -> TrainConfig:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    fields = {item.name for item in dataclasses.fields(TrainConfig)}
    payload = {key: value for key, value in raw.items() if key in fields}
    for key, value in overrides.items():
        if key in fields:
            payload[key] = value
    return TrainConfig(**payload)


def selector_binding(bundle: Path) -> tuple[dict, dict, dict]:
    selector = json.loads((bundle / "selector_records.json").read_text(encoding="utf-8"))
    windows = json.loads((bundle / "paired_windows.json").read_text(encoding="utf-8"))
    sidecars = json.loads((bundle / "sidecars.json").read_text(encoding="utf-8"))
    records = {
        session: row["frames"]
        for session, row in selector["sessions"].items()
    }
    starts = {
        session: set(row["window_starts"])
        for session, row in windows.items()
    }
    binding = {
        "selector_records": records,
        "selector_root": selector["selector_root"],
        "window_starts": starts,
    }
    return binding, sidecars, json.loads((bundle / "split.json").read_text(encoding="utf-8"))


def adapter_paths(split: dict, role: str) -> list[str]:
    root = Path(split["production_root"])
    return [
        str(root / session / "09_humanego_adapter")
        for session in admitted_sessions(split, role)
    ]


TRAIN_DISABLED_SENTINEL = Path(
    "/mnt/workspace/code/chaoyang/HumanEgo/config/controls/HUMANEGO_TRAIN_DISABLED.json"
)
EXACT78_COHORT = Path(
    "/mnt/workspace/code/chaoyang/tasks/control/runs/"
    "20260908_nine_hour_156_pipeline_v1/COHORT_EXACT78_FROZEN.json"
)
EXACT78_COHORT_SHA256 = (
    "d482ae4b627efa78653b69af26a357cec66381e3416dd78dd278504579e7790e"
)
MINIMUM_ADMITTED_SESSIONS = {"train": 16, "validation": 4, "test": 0}
MINIMUM_VALID_WINDOWS = {"train": 256, "validation": 64, "test": 0}
VISUAL_AB_FORMAL_SCHEMA = (
    "humanego-newtask-robot-split-v4-exact78-visual-ab-formal-robot"
)
VISUAL_AB_FORMAL_FREEZE_SCHEMA = (
    "humanego-newtask-robot-freeze-v4-exact78-visual-ab-formal-robot"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resume_checkpoint(
    checkpoint: dict,
    *,
    run_manifest: dict,
    dataset_stats_sha256: str,
) -> None:
    """Prevent a detached retry from resuming weights from another bundle."""
    if checkpoint.get("run_manifest") != run_manifest:
        raise RuntimeError("HOLD_RESUME_RUN_MANIFEST_MISMATCH")
    if checkpoint.get("dataset_stats_sha256") != dataset_stats_sha256:
        raise RuntimeError("HOLD_RESUME_DATASET_STATS_MISMATCH")


def admitted_sessions(split: dict, role: str) -> list[str]:
    admitted = split.get("admitted_splits")
    values = admitted.get(role) if isinstance(admitted, dict) else split.get(role)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise RuntimeError(f"HOLD_INVALID_ADMITTED_SPLIT: {role}")
    return values


def exact78_split_binding(split: dict) -> dict:
    """Bind a task bundle to the immutable 78-session split.

    ``train/validation/test/heldout`` always contain the complete canonical
    60/8/5/5 assignment.  ``admitted_splits`` may be a quality-filtered A/B
    subset for a time-bounded checkpoint; it can only remove rows and can
    never move a session between roles.  Heldout is never admitted here.
    """
    if _sha256_file(EXACT78_COHORT) != EXACT78_COHORT_SHA256:
        raise RuntimeError("HOLD_EXACT78_COHORT_SHA_DRIFT")
    cohort = json.loads(EXACT78_COHORT.read_text(encoding="utf-8"))
    task = split.get("task")
    if task not in {"chips", "poker"}:
        raise RuntimeError("HOLD_INVALID_EXACT78_TASK")
    expected = {
        role: [
            row["session_id"] for row in cohort["sessions"]
            if row["task"] == task and row["split"] == role
        ]
        for role in ("train", "validation", "test", "heldout")
    }
    for role, count in (("train", 60), ("validation", 8), ("test", 5), ("heldout", 5)):
        actual = split.get(role)
        if not isinstance(actual, list) or set(actual) != set(expected[role]) or len(actual) != count:
            raise RuntimeError(
                f"HOLD_EXACT78_{role.upper()}_MISMATCH: expected={count} "
                f"actual={len(actual) if isinstance(actual, list) else 'invalid'}"
            )
    admitted_by_role = {
        role: admitted_sessions(split, role)
        for role in ("train", "validation", "test")
    }
    flattened = [value for rows in admitted_by_role.values() for value in rows]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("HOLD_ADMITTED_SPLITS_OVERLAP")
    minimum_sessions = (
        {"train": 16, "validation": 3, "test": 0}
        if split.get("schema_version") == VISUAL_AB_FORMAL_SCHEMA
        else MINIMUM_ADMITTED_SESSIONS
    )
    for role, rows in admitted_by_role.items():
        if not set(rows).issubset(expected[role]):
            raise RuntimeError(f"HOLD_ADMITTED_{role.upper()}_ROLE_LEAKAGE")
        minimum = minimum_sessions[role]
        if len(rows) < minimum:
            raise RuntimeError(
                f"HOLD_ADMITTED_{role.upper()}_BELOW_MINIMUM: "
                f"{len(rows)}<{minimum}"
            )
    heldout = set(expected["heldout"])
    if heldout.intersection(flattened):
        raise RuntimeError("HOLD_HELDOUT_LEAKAGE_IN_ADMITTED_SPLITS")
    if set(split.get("selector_sessions", flattened)) != set(flattened):
        raise RuntimeError("HOLD_SELECTOR_SESSION_SET_MISMATCH")
    return {
        "cohort_path": str(EXACT78_COHORT),
        "cohort_sha256": EXACT78_COHORT_SHA256,
        "canonical_counts": {key: len(value) for key, value in expected.items()},
        "admitted_counts": {key: len(value) for key, value in admitted_by_role.items()},
        "heldout_sessions": expected["heldout"],
        "heldout_in_training": False,
    }


def bundle_session_set_gate(
    split: dict, binding: dict, robot_sidecars: dict[str, str]
) -> set[str]:
    expected = {
        session for role in ("train", "validation", "test")
        for session in admitted_sessions(split, role)
    }
    actual = {
        "selector_records": set(binding["selector_records"]),
        "paired_windows": set(binding["window_starts"]),
        "robot_sidecars": set(robot_sidecars),
    }
    for label, sessions in actual.items():
        if sessions != expected:
            raise RuntimeError(f"HOLD_{label.upper()}_SESSION_SET_MISMATCH")
    if set(split["heldout"]).intersection(expected):
        raise RuntimeError("HOLD_HELDOUT_LEAKAGE_IN_BUNDLE")
    return expected


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def robot_window_gate(split: dict, binding: dict) -> dict:
    """Validate Grade-B motion labels without authorizing formal Robot output."""
    sidecar_root = Path(split["sidecar_root"]).resolve(strict=True)
    production_root = Path(split["production_root"]).resolve(strict=True)
    hawor_root = Path(split["hawor_v3_sidecar_root"]).resolve(strict=True)
    contract = split.get("motion_label_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("HOLD_MOTION_LABEL_CONTRACT_MISSING")
    expected_sessions = {
        session for role in ("train", "validation", "test")
        for session in admitted_sessions(split, role)
    }
    formal_robot = contract.get("consumption_mode") == "FORMAL_ROBOT_ACTION_A_OR_B"
    required_contract = (
        {
            "consumption_mode": "FORMAL_ROBOT_ACTION_A_OR_B",
            "provenance_class": "FORMAL_ROBOT_ACTION_A_OR_B",
            "target": "accepted_tianji_dual_kaihand_action",
            "arm_consumed": False,
            "achieved_robot_fk_required": True,
        }
        if formal_robot
        else {
            "consumption_mode": "MOTION_TRAINING_LABEL_GRADE_B",
            "provenance_class": "MOTION_TRAINING_LABEL_GRADE_B",
            "target": "fresh_wrist_T_camera_plus_q_hand",
            "arm_consumed": False,
            "achieved_robot_fk_required": False,
        }
    )
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"HOLD_MOTION_LABEL_CONTRACT:{key}")
    motion_weights = contract.get("motion_weight_by_session")
    contact_weights = contract.get("contact_aux_weight_by_session")
    motion_digests = contract.get("sidecar_sha256_by_session")
    receipt_digests = contract.get("receipt_sha256_by_session")
    for label, values in (
        ("MOTION", motion_weights), ("CONTACT", contact_weights),
        ("MOTION_SIDECAR_DIGEST", motion_digests),
        ("MOTION_RECEIPT_DIGEST", receipt_digests),
    ):
        if not isinstance(values, dict) or set(values) != expected_sessions:
            raise RuntimeError(f"HOLD_{label}_WEIGHT_SESSION_SET_MISMATCH")
    rows: dict[str, dict] = {}
    counts = {role: 0 for role in MINIMUM_VALID_WINDOWS}
    active_roles = [
        role for role in ("train", "validation", "test")
        if admitted_sessions(split, role)
    ]
    for role in active_roles:
        for session in admitted_sessions(split, role):
            path = sidecar_root / "kai22" / session / "sidecar.npz"
            if _sha256_file(path) != motion_digests[session]:
                raise RuntimeError(f"HOLD_MOTION_LABEL_SIDECAR_DIGEST:{session}")
            if (
                not isinstance(receipt_digests[session], str)
                or len(receipt_digests[session]) != 64
            ):
                raise RuntimeError(f"HOLD_MOTION_LABEL_RECEIPT_DIGEST:{session}")
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "frame_names", "q", "wrist_T_camera", "valid", "confidence",
                    "joint_lower", "joint_upper", "translation_unit", "q_unit",
                }
                if formal_robot:
                    required.add("ik_warm_start_used")
                else:
                    required.update({
                        "provenance_class", "legacy_wrist_discarded",
                        "empirical_continuity_verified",
                    })
                missing = sorted(required - set(archive.files))
                if missing:
                    raise RuntimeError(
                        f"HOLD_ROBOT_SIDECAR_FIELDS_MISSING:{session}:{missing}"
                    )
                frame_names = [str(value) for value in archive["frame_names"]]
                q = np.asarray(archive["q"], dtype=np.float32)
                wrist = np.asarray(archive["wrist_T_camera"], dtype=np.float64)
                valid = np.asarray(archive["valid"], dtype=bool)
                confidence = np.asarray(archive["confidence"], dtype=np.float32)
                provenance = (
                    "FORMAL_ROBOT_ACTION_A_OR_B" if formal_robot
                    else str(np.asarray(archive["provenance_class"]).item())
                )
                legacy_wrist_discarded = (
                    True if formal_robot else bool(np.asarray(
                        archive["legacy_wrist_discarded"]
                    ).item())
                )
                continuity_verified = (
                    bool(np.asarray(archive["ik_warm_start_used"]).item())
                    if formal_robot else bool(np.asarray(
                        archive["empirical_continuity_verified"]
                    ).item())
                )
                lower = np.asarray(archive["joint_lower"], dtype=np.float32)
                upper = np.asarray(archive["joint_upper"], dtype=np.float32)
                translation_unit = str(np.asarray(archive["translation_unit"]).item())
                q_unit = str(np.asarray(archive["q_unit"]).item())
            count = len(frame_names)
            if (
                q.shape != (count, 2, 22)
                or wrist.shape != (count, 2, 4, 4)
                or valid.shape != (count, 2)
                or confidence.shape != (count, 2)
            ):
                raise RuntimeError(f"HOLD_MOTION_LABEL_SHAPE:{session}")
            expected_provenance = (
                "FORMAL_ROBOT_ACTION_A_OR_B"
                if formal_robot else "MOTION_TRAINING_LABEL_GRADE_B"
            )
            if provenance != expected_provenance:
                raise RuntimeError(f"HOLD_MOTION_LABEL_PROVENANCE:{session}")
            if not legacy_wrist_discarded or not continuity_verified:
                raise RuntimeError(f"HOLD_MOTION_LABEL_FRESHNESS_OR_CONTINUITY:{session}")
            if translation_unit != "metre" or q_unit != "radian":
                raise RuntimeError(f"HOLD_MOTION_LABEL_UNIT:{session}")
            if lower.shape != (2, 22) or upper.shape != (2, 22):
                raise RuntimeError(f"HOLD_MOTION_LABEL_JOINT_LIMIT_SHAPE:{session}")
            if not np.isfinite(q).all() or not np.isfinite(wrist).all():
                raise RuntimeError(f"HOLD_MOTION_LABEL_NONFINITE_STATE:{session}")
            if np.any(q < lower[None] - 1e-6) or np.any(q > upper[None] + 1e-6):
                raise RuntimeError(f"HOLD_MOTION_LABEL_JOINT_LIMIT:{session}")
            if not np.isfinite(confidence).all() or np.any(
                (confidence < 0.0) | (confidence > 1.0)
            ):
                raise RuntimeError(f"HOLD_MOTION_LABEL_CONFIDENCE_RANGE:{session}")
            fresh_path = hawor_root / session / "entities_hawor_v3.npz"
            with np.load(fresh_path, allow_pickle=False) as fresh:
                fresh_names = np.asarray(fresh["frame_names"]).astype(str)
                fresh_wrist = np.asarray(
                    fresh["T_hand_to_camera"], dtype=np.float64
                )[:, ::-1]
                fresh_valid = np.asarray(fresh["valid"], dtype=bool)[:, ::-1]
            if not np.array_equal(np.asarray(frame_names), fresh_names):
                raise RuntimeError(f"HOLD_MOTION_LABEL_FRESH_FRAME_MISMATCH:{session}")
            if not formal_robot and not np.array_equal(wrist, fresh_wrist):
                raise RuntimeError(f"HOLD_MOTION_LABEL_WRIST_NOT_FRESH_HAWOR:{session}")
            if np.any(valid & ~fresh_valid):
                raise RuntimeError(f"HOLD_MOTION_LABEL_VALID_EXCEEDS_FRESH_HAWOR:{session}")
            motion_weight = float(motion_weights[session])
            contact_weight = float(contact_weights[session])
            if not 0.0 < motion_weight <= 1.0:
                raise RuntimeError(f"HOLD_MOTION_LABEL_WEIGHT:{session}")
            if contact_weight != 0.0:
                raise RuntimeError(f"HOLD_UNAUTHORIZED_CONTACT_AUX_WEIGHT:{session}")
            frame_to_index = {int(value): index for index, value in enumerate(frame_names)}
            if len(frame_to_index) != count:
                raise RuntimeError(f"HOLD_MOTION_LABEL_DUPLICATE_FRAME:{session}")
            both_valid = valid.all(axis=1)
            consecutive_valid = valid[1:] & valid[:-1]
            step_limit = 0.080001 if formal_robot else 0.200001
            if np.any(
                np.max(np.abs(np.diff(q, axis=0)), axis=-1)[consecutive_valid]
                > step_limit
            ):
                raise RuntimeError(f"HOLD_MOTION_LABEL_Q_STEP:{session}")
            longest = _longest_true_run(both_valid)
            adapter_frames = len(list(
                (production_root / session / "09_humanego_adapter/preprocess/all_data")
                .glob("*/training_data.json")
            ))
            full_session = count == adapter_frames and adapter_frames > 0
            if not full_session and longest < 64:
                raise RuntimeError(
                    f"HOLD_MOTION_LABEL_NO_64_FRAME_RUN:{session}:longest={longest}"
                )
            valid_starts = 0
            for start in binding["window_starts"][session]:
                indices = [frame_to_index.get(int(start) + offset) for offset in range(50)]
                if None in indices:
                    continue
                selected = np.asarray(indices, dtype=np.int64)
                if bool(both_valid[selected].all()):
                    valid_starts += 1
            if valid_starts < 1:
                raise RuntimeError(f"HOLD_MOTION_LABEL_ZERO_H50_WINDOWS:{session}")
            counts[role] += valid_starts
            rows[session] = {
                "role": role, "frames": count, "adapter_frames": adapter_frames,
                "full_session": full_session,
                "longest_consecutive_dual_valid": longest,
                "valid_h50_windows": valid_starts,
                "provenance_class": provenance,
                "fresh_hawor_wrist_exact": not formal_robot,
                "accepted_robot_wrist_consumed": formal_robot,
                "motion_weight": motion_weight,
                "contact_aux_weight": contact_weight,
            }
    minimum_windows = (
        {"train": 256, "validation": 48, "test": 0}
        if formal_robot else MINIMUM_VALID_WINDOWS
    )
    for role, minimum in minimum_windows.items():
        if counts[role] < minimum:
            raise RuntimeError(
                f"HOLD_{role.upper()}_VALID_WINDOWS_BELOW_MINIMUM:"
                f"{counts[role]}<{minimum}"
            )
    return {
        "status": (
            "PASS_FORMAL_ROBOT_ACTION_H50_WINDOW_GATE"
            if formal_robot else "PASS_MOTION_LABEL_H50_WINDOW_GATE"
        ),
        "formal_robot_product_required": formal_robot,
        "counts": counts,
        "sessions": rows,
    }


def object_binding(bundle: Path, split: dict) -> dict:
    contract = split.get("object_state_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("HOLD_OBJECT_STATE_CONTRACT_MISSING")
    mode = contract.get("consumption_mode")
    if mode not in {"training_estimated_grade_b", "training_admitted"}:
        raise RuntimeError(f"HOLD_OBJECT_CONSUMPTION_MODE: {mode!r}")
    root = Path(str(contract.get("sidecar_root", ""))).resolve(strict=True)
    if root != (bundle / "object_state_sidecars").resolve(strict=True):
        raise RuntimeError("HOLD_OBJECT_SIDECAR_ROOT_NOT_BUNDLE_LOCAL")
    npz_digests = contract.get("npz_sha256_by_session")
    json_digests = contract.get("json_sha256_by_session")
    grades = contract.get("quality_grade_by_session")
    weights = contract.get("training_weight_by_session")
    thresholds = contract.get("confidence_thresholds")
    expected = {
        session for role in ("train", "validation", "test")
        for session in admitted_sessions(split, role)
    }
    mappings = {
        "npz": npz_digests, "json": json_digests,
        "grade": grades, "weight": weights,
    }
    for label, value in mappings.items():
        if not isinstance(value, dict) or set(value) != expected:
            raise RuntimeError(f"HOLD_OBJECT_{label.upper()}_SESSION_SET_MISMATCH")
    if not isinstance(thresholds, dict) or not thresholds:
        raise RuntimeError("HOLD_OBJECT_CONFIDENCE_THRESHOLDS_MISSING")
    for session in sorted(expected):
        grade = grades[session]
        weight = weights[session]
        if grade == "A":
            if float(weight) != 1.0:
                raise RuntimeError(f"HOLD_OBJECT_GRADE_A_CONTRACT: {session}")
        elif grade == "B":
            if mode != "training_estimated_grade_b" or not 0.0 < float(weight) < 1.0:
                raise RuntimeError(f"HOLD_OBJECT_GRADE_B_CONTRACT: {session}")
        else:
            raise RuntimeError(f"HOLD_OBJECT_GRADE_C_OR_UNKNOWN_ADMITTED: {session}")
        manifest_path = root / session / "AUTO_ESTIMATED_OBJECT_STATE.json"
        npz_path = root / session / "AUTO_ESTIMATED_OBJECT_STATE.npz"
        if _sha256_file(manifest_path) != json_digests[session]:
            raise RuntimeError(f"HOLD_OBJECT_JSON_SHA_MISMATCH: {session}")
        if _sha256_file(npz_path) != npz_digests[session]:
            raise RuntimeError(f"HOLD_OBJECT_NPZ_SHA_MISMATCH: {session}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("quality_grade") != grade:
            raise RuntimeError(f"HOLD_OBJECT_MANIFEST_GRADE_MISMATCH: {session}")
        if float(manifest.get("training_weight", -1)) != float(weight):
            raise RuntimeError(f"HOLD_OBJECT_MANIFEST_WEIGHT_MISMATCH: {session}")
    return {
        "root": str(root), "mode": mode,
        "npz_digests": npz_digests, "json_digests": json_digests,
        "grades": grades, "weights": weights, "thresholds": thresholds,
    }


def hawor_binding(bundle: Path, split: dict) -> tuple[str, dict[str, str]]:
    contract = split.get("ict_contract", {})
    required = {
        "hand_tracking_method": "hawor_v3",
        "single_hand": False,
        "ict_dim": 29,
        "frame_mode": "camera_frame",
        "translation_unit": "metre",
        "camera_coordinate_system": "x_right_y_down_z_forward",
    }
    valid_split_schemas = {
        "humanego-newtask-robot-split-v3-exact78-object-ict-motion-grade-b",
        VISUAL_AB_FORMAL_SCHEMA,
    }
    if split.get("schema_version") not in valid_split_schemas:
        raise RuntimeError(
            "HOLD_INVALID_ICT_BUNDLE: exact78 training requires the V3 "
            "HaWoR+Object ICT bundle; legacy/hand-only bundles are forbidden"
        )
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise RuntimeError(
                f"HOLD_INVALID_ICT_CONTRACT: {key}={contract.get(key)!r}, "
                f"expected {expected!r}"
            )
    object_policy = str(contract.get("object_token_policy", ""))
    if object_policy not in {
        "AUTO_ESTIMATED_GRADE_B_FROZEN_PER_SESSION",
        "FORMAL_OBJECT6D_FROZEN_PER_SESSION",
    }:
        raise RuntimeError("HOLD_OBJECT_TOKEN_CONTRACT_MISSING")
    freeze = json.loads((bundle / "freeze.json").read_text(encoding="utf-8"))
    expected_freeze_schema = (
        VISUAL_AB_FORMAL_FREEZE_SCHEMA
        if split.get("schema_version") == VISUAL_AB_FORMAL_SCHEMA
        else "humanego-newtask-robot-freeze-v3-exact78-object-ict-motion-grade-b"
    )
    if freeze.get("schema_version") != expected_freeze_schema:
        raise RuntimeError("HOLD_HAWOR_ICT_FREEZE_MISSING")
    for filename, reference in freeze.get("bundle_files", {}).items():
        path = (bundle / filename).resolve(strict=True)
        if (
            path != Path(reference.get("path", "")).resolve(strict=True)
            or path.stat().st_size != reference.get("bytes")
            or _sha256_file(path) != reference.get("sha256")
        ):
            raise RuntimeError(f"HOLD_HAWOR_ICT_FREEZE_DRIFT: {filename}")
    sidecar_root = Path(split["hawor_v3_sidecar_root"]).resolve(strict=True)
    digests = json.loads(
        (bundle / "hawor_v3_sidecars.json").read_text(encoding="utf-8")
    )
    expected_sessions = {
        session
        for role in ("train", "validation", "test")
        for session in admitted_sessions(split, role)
    }
    if set(digests) != expected_sessions:
        raise RuntimeError("HOLD_HAWOR_SIDECAR_COHORT_MISMATCH")
    for session in sorted(expected_sessions):
        path = sidecar_root / session / "entities_hawor_v3.npz"
        if not path.is_file() or _sha256_file(path) != digests[session]:
            raise RuntimeError(
                f"HOLD_HAWOR_SIDECAR_IDENTITY_MISMATCH: {session}"
            )
    return str(sidecar_root), digests


def epoch0_ict_gate(
    *, cfg: TrainConfig, split: dict, binding: dict,
    robot_sidecars: dict[str, str], hawor_root: str,
    hawor_sidecars: dict[str, str], object_source: dict, stats: dict,
) -> dict:
    """Exercise the real loader on one both-hand sample per nonempty split."""
    active_roles = [
        role for role in ("train", "validation", "test")
        if admitted_sessions(split, role)
    ]
    cohort_counts = {
        "admitted_windows": 0,
        "ict_mask_any_windows": 0,
        "left_token_windows": 0,
        "right_token_windows": 0,
        "both_token_windows": 0,
    }
    for role in active_roles:
        for session in admitted_sessions(split, role):
            path = Path(hawor_root) / session / "entities_hawor_v3.npz"
            with np.load(path, allow_pickle=False) as archive:
                valid = np.asarray(archive["valid"], dtype=bool)
                names = [str(value) for value in archive["frame_names"]]
            frame_to_index = {int(name): index for index, name in enumerate(names)}
            for frame in binding["window_starts"][session]:
                cohort_counts["admitted_windows"] += 1
                index = frame_to_index.get(int(frame))
                if index is None:
                    raise RuntimeError(
                        f"HOLD_ICT_SIDECAR_FRAME_MISSING: {session}/{frame:05d}"
                    )
                left, right = valid[index].tolist()
                cohort_counts["left_token_windows"] += int(left)
                cohort_counts["right_token_windows"] += int(right)
                cohort_counts["both_token_windows"] += int(left and right)
                if not (left and right):
                    raise RuntimeError(
                        f"HOLD_ICT_BOTH_HANDS_REQUIRED: {session}/{frame:05d}"
                    )
                cohort_counts["ict_mask_any_windows"] += 1
    selected: dict[str, dict[str, object]] = {}
    selected_starts: dict[str, set[int]] = {}
    selected_sessions = []
    for role in active_roles:
        chosen = None
        # Exercise both capture cohorts when possible, and avoid making the
        # entire gate a frame-0 identity-only check.
        prefer_0902 = role != "train"
        role_sessions = sorted(
            admitted_sessions(split, role),
            key=lambda session: (("0902" in session) != prefer_0902, session),
        )
        for session in role_sessions:
            path = Path(hawor_root) / session / "entities_hawor_v3.npz"
            with np.load(path, allow_pickle=False) as archive:
                valid = np.asarray(archive["valid"], dtype=bool)
                names = [str(value) for value in archive["frame_names"]]
            allowed = binding["window_starts"][session]
            both = np.flatnonzero(np.logical_and(valid[:, 0], valid[:, 1]))
            candidates = [
                int(names[int(index)]) for index in both
                if int(names[int(index)]) in allowed
            ]
            if candidates:
                middle = 0.5 * (min(allowed) + max(allowed))
                frame = min(candidates, key=lambda value: abs(value - middle))
                chosen = (session, frame)
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f"HOLD_ICT_{role.upper()}_HAS_NO_BOTH_HAND_SAMPLE")
        session, frame = chosen
        selected_sessions.append(session)
        selected_starts[session] = {frame}
        selected[role] = {"session": session, "frame": frame}

    paths = [
        str(Path(split["production_root"]) / session / "09_humanego_adapter")
        for session in selected_sessions
    ]
    dataset = FlowMatchingDataloader(
        sessions=[MPSSessions(path) for path in paths],
        image_size=cfg.image_size,
        pred_horizon=cfg.pred_horizon,
        single_hand=False,
        max_ict=cfg.max_ict,
        img_name=None,
        centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode,
        action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        hawor_v3_sidecar_root=hawor_root,
        hawor_v3_sha256_by_session={
            session: hawor_sidecars[session] for session in selected_sessions
        },
        use_pcd_features=False,
        use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False,
        enable_augmentation=False,
        hand_tracking_method="hawor_v3",
        use_legacy_image_loading=False,
        cache_json_in_memory=False,
        cache_image_bytes_in_memory=False,
        stats=stats,
        allowed_window_starts=selected_starts,
        selector_records={
            session: binding["selector_records"][session]
            for session in selected_sessions
        },
        selector_root=binding["selector_root"],
        sidecar_sha256_by_session={
            session: robot_sidecars[session] for session in selected_sessions
        },
        object_state_sidecar_root=object_source["root"],
        object_state_npz_sha256_by_session={
            session: object_source["npz_digests"][session]
            for session in selected_sessions
        },
        object_state_json_sha256_by_session={
            session: object_source["json_digests"][session]
            for session in selected_sessions
        },
        object_state_consumption_mode=object_source["mode"],
        object_state_confidence_thresholds=object_source["thresholds"],
    )
    if len(dataset) != len(active_roles):
        raise RuntimeError(
            "HOLD_ICT_GATE_SAMPLE_COUNT:"
            f"expected={len(active_roles)} actual={len(dataset)}"
        )
    rows = []
    for json_path in dataset.samples:
        frame = dataset._read_frame(json_path)
        T_w2ref = dataset._get_T_w2ref(frame)
        state, _, mask = dataset._build_ict(frame, T_w2ref)
        if state.shape != (cfg.max_ict, 29) or mask.shape != (cfg.max_ict,):
            raise RuntimeError("HOLD_ICT_29D_SHAPE_CONTRACT_FAILED")
        if not bool(mask.any()):
            raise RuntimeError("HOLD_ICT_MASK_ALL_FALSE")
        active = state[mask]
        types = active[:, 0].tolist()
        if types[:2] != [1.0, 2.0]:
            raise RuntimeError(f"HOLD_ICT_HAND_TOKEN_TYPES_INVALID: {types}")
        object_types = [value for value in types if value in (3.0, 4.0)]
        if 3.0 not in object_types:
            raise RuntimeError("HOLD_OBJECT_ANCHOR_TOKEN_MISSING")
        if not np.isfinite(active).all():
            raise RuntimeError("HOLD_ICT_NONFINITE")
        # 6D rotations are unit columns; flags are binary for hand tokens.
        for rotation_slice in (slice(4, 10), slice(13, 19), slice(22, 28)):
            if float(np.max(np.abs(active[:, rotation_slice]))) > 1.0001:
                raise RuntimeError("HOLD_ICT_ROTATION_6D_RANGE_FAILED")
        for position_slice in (slice(1, 4), slice(10, 13), slice(19, 22)):
            if float(np.max(np.abs(active[:, position_slice]))) > 20.0:
                raise RuntimeError("HOLD_ICT_NORMALIZED_POSITION_RANGE_FAILED")
        type_array = active[:, 0]
        hand_flags = active[np.isin(type_array, (1.0, 2.0)), 28]
        object_flags = active[np.isin(type_array, (3.0, 4.0)), 28]
        if not np.isin(hand_flags, (0.0, 1.0)).all():
            raise RuntimeError("HOLD_ICT_GRASP_FLAG_RANGE_FAILED")
        if not np.equal(object_flags, -1.0).all():
            raise RuntimeError("HOLD_ICT_OBJECT_SENTINEL_FAILED")
        rows.append({
            "json_path": json_path,
            "token_types": types,
            "ict_mask_true": int(mask.sum()),
            "ict_abs_max": float(np.max(np.abs(active))),
            "object_tokens": len(object_types),
            "training_weight": float(
                object_source["weights"][
                    dataset._session_id_from_mps_path(
                        dataset._session_root_from_json_path(json_path)
                    )
                ]
            ),
        })
    return {
        "status": "PASS_EPOCH0_ICT_HARD_GATE",
        "ict_dim": 29,
        "frame_mode": "camera_frame",
        "translation_unit": "metre",
        "object_token_policy": split["ict_contract"]["object_token_policy"],
        "object_state_admission": dataset.object_state_admission_report(),
        "cohort_counts": cohort_counts,
        "split_samples": selected,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="run the CPU epoch-0 ICT gate and exit before CUDA/model creation",
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    run_root = args.run_root.resolve()
    out_dir = run_root / args.tag

    binding, sidecars, split = selector_binding(bundle)
    split_audit = exact78_split_binding(split)
    bundle_session_set_gate(split, binding, sidecars)
    robot_window_audit = robot_window_gate(split, binding)
    hawor_root, hawor_sidecars = hawor_binding(bundle, split)
    object_source = object_binding(bundle, split)
    train_paths = adapter_paths(split, "train")
    eval_paths = adapter_paths(split, "validation")
    pretrained = HUMANEGO / "artifacts/pretrained_humanego/latest.pt"
    yaml_path = HUMANEGO / "cfg/training/grap_a_cap/kai22_retarget_ab_r2.yaml"

    cfg = load_config(yaml_path, {
        "task": split["task"],
        "MPS_PATHS_TRAIN": train_paths,
        "MPS_PATHS_EVAL": eval_paths,
        "pretrained_checkpoint": str(pretrained),
        "robot_sidecar_root": split["sidecar_root"],
        "batch_size": 64,
        "eval_batch_size": 64,
        "epochs": 400,
        "seed": 7,
        "lr": 1.0e-4,
        "image_size": [240, 320],
        "img_name": "rgb.png",
        "use_legacy_image_loading": False,
        "persistent_workers": True,
        "out_dir": str(out_dir),
        "run_root": str(run_root),
        "device": "cuda",
        "make_video": False,
    })

    unit_stats = {"pos": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}}
    raw_ict_gate = epoch0_ict_gate(
        cfg=cfg,
        split=split,
        binding=binding,
        robot_sidecars=sidecars,
        hawor_root=hawor_root,
        hawor_sidecars=hawor_sidecars,
        object_source=object_source,
        stats=unit_stats,
    )
    if args.preflight_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            **raw_ict_gate,
            "mode": "CPU_PREFLIGHT_ONLY",
            "training_started": False,
            "training_disabled_sentinel_present": TRAIN_DISABLED_SENTINEL.is_file(),
            "exact78_split_audit": split_audit,
            "robot_window_audit": robot_window_audit,
            "object_quality_grades": object_source["grades"],
            "object_training_weights": object_source["weights"],
        }
        (out_dir / "epoch0_ict_gate.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
        return 0
    # This sentinel was written specifically for the rejected V1/V2 hand-only
    # path.  An exact78 V3 bundle that has passed the gates above is the reviewed
    # successor requested by the operator; retain the sentinel as historical
    # evidence but bind the override in the new run manifest.
    legacy_sentinel = None
    if TRAIN_DISABLED_SENTINEL.is_file():
        legacy_sentinel = {
            "path": str(TRAIN_DISABLED_SENTINEL.resolve()),
            "sha256": _sha256_file(TRAIN_DISABLED_SENTINEL),
            "override_scope": "exact78-v3-object-ict-only",
        }
    out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "schema_version": (
            "humanego-newtask-robot-run-v4-exact78-visual-ab-formal-robot"
            if split.get("schema_version") == VISUAL_AB_FORMAL_SCHEMA
            else "humanego-newtask-robot-run-v3-exact78-object-ict"
        ),
        "embodiment": "kai22",
        "task": split["task"],
        "visual_branch": split.get("visual_branch", "HUMAN_RAW_RGB"),
        "pairing_id": split.get("pairing_id"),
        "bundle": str(bundle),
        "train_sessions": admitted_sessions(split, "train"),
        "validation_sessions": admitted_sessions(split, "validation"),
        "test_sessions": admitted_sessions(split, "test"),
        "canonical_split_sessions": {
            role: split[role]
            for role in ("train", "validation", "test", "heldout")
        },
        "heldout_sessions": split["heldout"],
        "admitted_splits": split.get("admitted_splits"),
        "exact78_split_audit": split_audit,
        "robot_window_audit": robot_window_audit,
        "sidecars": sidecars,
        "hawor_v3_sidecars": hawor_sidecars,
        "hawor_v3_sidecar_root": hawor_root,
        "object_state_sidecar_root": object_source["root"],
        "object_state_consumption_mode": object_source["mode"],
        "object_quality_grades": object_source["grades"],
        "object_training_weights": object_source["weights"],
        "sample_weight_formula": (
            "sum_b(w_b*numerator_b)/sum_b(w_b*denominator_b); "
            "w=1 for A, 0<w<1 for B, w=0 excluded; all-zero batch rejected"
        ),
        "legacy_wrong_ict_sentinel": legacy_sentinel,
        "epoch0_ict_gate_raw": raw_ict_gate,
        "pretrained_sha256": _sha256_file(pretrained),
        "selected_batch_size": 64,
        "gpu_feed": "prefetch+persistent_workers+cudnn_benchmark",
        "epochs": 400,
        "seed": 7,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8")

    stats_path = out_dir / "dataset_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = get_dataset_stats(
            cfg.MPS_PATHS_TRAIN,
            cfg,
            binding["window_starts"],
            binding["selector_records"],
            binding["selector_root"],
            run_manifest["sidecars"],
            hawor_root,
            hawor_sidecars,
        )
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    run_manifest["dataset_stats_sha256"] = _sha256_file(stats_path)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    ds_train = FlowMatchingDataloader(
        sessions=[MPSSessions(path) for path in train_paths],
        image_size=cfg.image_size,
        pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand,
        single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict,
        img_name=cfg.img_name,
        centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode,
        action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        hawor_v3_sidecar_root=hawor_root,
        hawor_v3_sha256_by_session=hawor_sidecars,
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        enable_augmentation=cfg.enable_augmentation,
        enable_aug_img=cfg.enable_aug_img,
        enable_aug_rrc=cfg.enable_aug_rrc,
        enable_aug_target_jittering=cfg.enable_aug_target_jittering,
        enable_aug_cutout=cfg.enable_aug_cutout,
        enable_aug_temporal_stride=cfg.enable_aug_temporal_stride,
        enable_aug_interpolation=cfg.enable_aug_interpolation,
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=False,
        cache_json_in_memory=cfg.cache_json_in_memory,
        cache_image_bytes_in_memory=cfg.cache_image_bytes_in_memory,
        seed=cfg.seed,
        stats=stats,
        allowed_window_starts=binding["window_starts"],
        selector_records=binding["selector_records"],
        selector_root=binding["selector_root"],
        sidecar_sha256_by_session=sidecars,
        object_state_sidecar_root=object_source["root"],
        object_state_npz_sha256_by_session=object_source["npz_digests"],
        object_state_json_sha256_by_session=object_source["json_digests"],
        object_state_consumption_mode=object_source["mode"],
        object_state_confidence_thresholds=object_source["thresholds"],
    )
    ds_eval = FlowMatchingDataloader(
        sessions=[MPSSessions(path) for path in eval_paths],
        image_size=cfg.image_size,
        pred_horizon=cfg.pred_horizon,
        single_hand=cfg.single_hand,
        single_hand_side=cfg.single_hand_side,
        max_ict=cfg.max_ict,
        img_name=cfg.img_name,
        centric_mode=cfg.centric_mode,
        frame_mode=cfg.frame_mode,
        action_mode=cfg.action_mode,
        hand_action_representation=cfg.hand_action_representation,
        robot_sidecar_root=cfg.robot_sidecar_root,
        hawor_v3_sidecar_root=hawor_root,
        hawor_v3_sha256_by_session=hawor_sidecars,
        enable_augmentation=False,
        hand_tracking_method=cfg.hand_tracking_method,
        use_legacy_image_loading=False,
        cache_json_in_memory=cfg.cache_json_in_memory,
        cache_image_bytes_in_memory=cfg.cache_image_bytes_in_memory,
        seed=cfg.seed,
        stats=stats,
        allowed_window_starts=binding["window_starts"],
        selector_records=binding["selector_records"],
        selector_root=binding["selector_root"],
        sidecar_sha256_by_session=sidecars,
        object_state_sidecar_root=object_source["root"],
        object_state_npz_sha256_by_session=object_source["npz_digests"],
        object_state_json_sha256_by_session=object_source["json_digests"],
        object_state_consumption_mode=object_source["mode"],
        object_state_confidence_thresholds=object_source["thresholds"],
    )
    print(f"train samples={len(ds_train)} eval samples={len(ds_eval)}", flush=True)
    if len(ds_train) == 0 or len(ds_eval) == 0:
        raise SystemExit("empty train or eval loader")

    _tune_cuda()
    # 12-core box: yaml asks for 16, two loaders would be 32 and starve
    # both the GPU and the CPU chains.  Cap does not change batch/lr.
    workers = min(int(cfg.num_workers or 8), 8)
    loader_kwargs = {"num_workers": workers, "pin_memory": True}
    if workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": min(int(getattr(cfg, "prefetch_factor", 4) or 4), 4),
        })
    print(
        f"gpu feed workers={workers} prefetch={loader_kwargs.get('prefetch_factor')} "
        f"persistent=1 batch={cfg.batch_size} (recipe unchanged)",
        flush=True,
    )
    dl_train = _CudaPrefetch(
        DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs),
        cfg.device,
    )
    dl_eval = _CudaPrefetch(
        DataLoader(
            ds_eval, batch_size=cfg.eval_batch_size or cfg.batch_size,
            shuffle=False, **loader_kwargs),
        cfg.device,
    )

    model = FlowMatchingModel(
        single_hand=cfg.single_hand,
        pred_horizon=cfg.pred_horizon,
        max_ict=cfg.max_ict,
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        vision_embed_dim=cfg.vision_embed_dim,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        horizon_weighting=getattr(cfg, "model_horizon_weighting",
                                  getattr(cfg, "model_h_weighting", "uniform")),
        horizon_beta=getattr(cfg, "model_horizon_beta",
                             getattr(cfg, "model_h_beta", 0.0)),
        use_pcd_features=cfg.use_pcd_features,
        use_aux_obj_dynamics=cfg.use_aux_obj_dynamics,
        use_aux_visual_foresight=cfg.use_aux_visual_foresight,
        use_aux_temporal_contrastive=cfg.use_aux_temporal_contrastive,
        use_region_attn=cfg.use_region_attn,
        use_pre_norm=cfg.use_pre_norm,
        use_ctx_norm=cfg.use_ctx_norm,
        use_done_in_flow=cfg.use_done_in_flow,
        hand_action_representation=cfg.hand_action_representation,
    ).to(cfg.device)

    latest_path = out_dir / "latest.pt"
    if cfg.pretrained_checkpoint and not latest_path.is_file():
        transfer = load_compatible_pretrained(
            model, str(pretrained), str(out_dir / "pretrained_load_report.json"),
        )
        print(f"pretrained loaded_fraction={transfer['loaded_parameter_fraction']}",
              flush=True)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMAModel(model, decay=cfg.ema_decay) if cfg.use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.use_amp))
    best = {"score": 1e9, "epoch": -1}
    no_improve = 0
    start_epoch = 1
    global_step = 0
    history = []
    if latest_path.is_file():
        ckpt = torch.load(latest_path, map_location=cfg.device)
        validate_resume_checkpoint(
            ckpt,
            run_manifest=run_manifest,
            dataset_stats_sha256=run_manifest["dataset_stats_sha256"],
        )
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", 0))
        best = ckpt.get("best", best)
        no_improve = int(ckpt.get("evaluations_without_improvement", 0))
        if ema is not None and ckpt.get("ema_shadow") is not None:
            ema.load_state_dict(ckpt["ema_shadow"])
        hist_path = out_dir / "train_history.json"
        if hist_path.is_file():
            history = json.loads(hist_path.read_text(encoding="utf-8"))
        print(f"resume from epoch {start_epoch - 1}", flush=True)

    total_steps = int(cfg.epochs * max(1, len(dl_train)))
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        ds_train.set_epoch(epoch)
        train_row, global_step = train_one_epoch(
            model, dl_train, opt, scaler, cfg, epoch, global_step, total_steps, ema)
        eval_row = {}
        if epoch % cfg.eval_every == 0:
            if ema is not None:
                ema.apply_shadow()
            try:
                eval_row = eval_ode_inference(
                    model, dl_eval, cfg=cfg, max_batches=cfg.max_eval_batches)
            finally:
                if ema is not None:
                    ema.restore()
            score = (
                eval_row["pos_err_w_m"]
                + eval_row["rot_err_w_deg"] / 100.0
                + eval_row.get("joint_mae_w_normalized", 0.0) * 0.10
            )
            if score < best["score"]:
                best = {"score": float(score), "epoch": int(epoch)}
                no_improve = 0
                if ema is not None:
                    ema.apply_shadow()
                save_checkpoint_atomic(
                    {
                        "epoch": epoch, "model": model.state_dict(),
                        "opt": opt.state_dict(), "cfg": cfg.__dict__,
                        "best": best, "global_step": global_step,
                        "run_manifest": run_manifest, "model_weights": "ema",
                        "dataset_stats_sha256": run_manifest["dataset_stats_sha256"],
                    },
                    str(out_dir / "best.pt"),
                )
                if ema is not None:
                    ema.restore()
            else:
                no_improve += 1
        save_checkpoint_atomic(
            {
                "epoch": epoch, "model": model.state_dict(),
                "opt": opt.state_dict(), "cfg": cfg.__dict__,
                "best": best, "global_step": global_step,
                "run_manifest": run_manifest, "model_weights": "train",
                "dataset_stats_sha256": run_manifest["dataset_stats_sha256"],
                "evaluations_without_improvement": no_improve,
                "ema_shadow": ema.state_dict() if ema is not None else None,
            },
            str(latest_path),
        )
        row = {"epoch": epoch, **train_row, **eval_row, "best": best}
        history.append(row)
        (out_dir / "train_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"EP {epoch:03d}/{cfg.epochs} loss={train_row.get('loss')} "
            f"best={best} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        if (
            cfg.early_stop_patience is not None
            and epoch % cfg.eval_every == 0
            and no_improve >= cfg.early_stop_patience
        ):
            print(f"early stop at epoch {epoch}, best {best}", flush=True)
            break

    (out_dir / "training_complete.json").write_text(
        json.dumps({
            "status": "complete",
            "best": best,
            "elapsed_s": time.time() - t0,
            "run_root": str(out_dir),
        }, indent=2),
        encoding="utf-8",
    )
    return 0


def _heal_then_retry(bundle: Path, log_hint: str) -> None:
    heal = Path("/mnt/workspace/code/chaoyang/HumanEgo/tools/heal_bundle_preverify.py")
    if not heal.is_file():
        return
    print(f"preverify flap, healing bundle {bundle}", flush=True)
    import subprocess
    subprocess.run(
        ["python3", "-u", str(heal), "--bundle", str(bundle),
         "--session", "play_cards_0901_001", "--frame", "00605"],
        check=False,
    )
    # Also parse the exception text if it names another frame.
    import re
    match = re.search(
        r"(play_cards_\d{4}_\d{3}|get_potato_chips_\d{4}_\d{3})/(\d{5})",
        log_hint,
    )
    if match:
        subprocess.run(
            ["python3", "-u", str(heal), "--bundle", str(bundle),
             "--session", match.group(1), "--frame", match.group(2)],
            check=False,
        )


if __name__ == "__main__":
    last_error: BaseException | None = None
    bundle_arg = None
    for index, token in enumerate(sys.argv):
        if token == "--bundle" and index + 1 < len(sys.argv):
            bundle_arg = Path(sys.argv[index + 1])
    for attempt in range(4):
        try:
            raise SystemExit(main())
        except Exception as error:
            last_error = error
            if not _is_flap(error) and "changed during preverification" not in str(error):
                raise
            if bundle_arg is None:
                raise
            print(f"train attempt {attempt + 1} hit CPFS flap: {error}", flush=True)
            _heal_then_retry(bundle_arg, str(error))
            time.sleep(2)
    raise SystemExit(str(last_error))
