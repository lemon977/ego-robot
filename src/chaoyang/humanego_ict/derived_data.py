"""Independent derived-label preparation and strict causal training admission.

Preparation verifies and recomputes numeric labels, not physical qualification.
Changing supervision to DERIVED_RETARGET never admits offline/future-donor RGB.
No source arrays, labels, receipts or models are written/loaded automatically.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .data import (
    ACTION_LAYOUT, BRANCHES, TARGET, TrainingDataError, VerifiedInputCache,
    _check_rgb_causal, _checked_calibration, _npz, _object, _ordinary,
    _pose9, _pose_clock_declaration, _proper_se3, _receipt, _shape, reference_bytes,
)

SCHEMA = "pick-place-humanego-ict-revo2-derived-paired-v1"
SUPERVISION_CLASS = "DERIVED_RETARGET"
SOURCE_KIND = "PICO_CONTROLLER_INSTALLATION_PRIOR_AND_MANUS"
CHANNEL_ORDER = ["thumb_bend", "thumb_rotation", "index", "middle", "ring", "pinky"]
SOURCE_KEYS = {
    "controller_pose_world", "controller_to_wrist", "controller_valid",
    "hand_joints_wrist", "hand_joints_valid", "qpc_ns", "source_frame_id",
    "segment_id", "controller_source_qpc_ns", "hand_source_qpc_ns",
    "reference_camera_to_world",
}
LABEL_KEYS = {
    "x_ict", "ict_mask", "y_action", "action_valid_mask", "anchor_qpc_ns",
    "conditioning_source_qpc_ns", "target_qpc_ns", "target_source_frame_ids",
    "target_source_rows", "current_source_row", "reference_camera_to_world",
}


def _hand_contract(reference, allow_synthetic_fixture, cache):
    contract = _object(reference_bytes(reference, cache=cache))
    if contract.get("synthetic_fixture_only") is True and not allow_synthetic_fixture:
        raise TrainingDataError("synthetic hand contract is not production evidence")
    if (contract.get("schema") != "derived-revo2-hand-v1"
            or contract.get("supervision_kind") != SUPERVISION_CLASS
            or contract.get("channel_order") != CHANNEL_ORDER
            or contract.get("hand_command_unit") != "NORMALIZED_0_1"):
        raise TrainingDataError("explicit canonical derived six-channel contract required")
    for name in ("feature_provider", "layout", "mapping"):
        component = contract.get(name)
        # Missing component is reported by retarget as UNKNOWN/masked; an
        # existing component cannot substitute a free-form 64-hex claim for
        # actual frozen source evidence.
        if not isinstance(component, Mapping):
            continue
        ref = component.get("source_ref")
        if not isinstance(ref, Mapping):
            raise TrainingDataError(f"derived {name} requires immutable source_ref")
        reference_bytes(ref, cache=cache)
        if ref["sha256"] != component.get("source_sha256"):
            raise TrainingDataError(f"derived {name} source_ref/source_sha256 mismatch")
        if (contract.get("synthetic_fixture_only") is not True
                and name in {"feature_provider", "mapping"}
                and Path(ref["path"]) != Path(__file__).with_name("retarget.py")):
            raise TrainingDataError(f"derived {name} must bind the executed explicit retarget module")
    return contract


def _source(reference, cache):
    source = _npz(reference, cache=cache)
    if not SOURCE_KEYS.issubset(source):
        raise TrainingDataError(f"missing human source arrays: {sorted(SOURCE_KEYS - set(source))}")
    count = source["qpc_ns"].size
    joints = source["hand_joints_wrist"]
    if joints.ndim != 4 or joints.shape[:2] != (count, 2) or joints.shape[-1] != 3 or joints.shape[2] < 1:
        raise TrainingDataError("hand_joints_wrist requires [T,2,J,3] explicit layout")
    for key, shape, kind in (
        ("controller_pose_world", (count, 2, 4, 4), "float"),
        ("controller_to_wrist", (2, 4, 4), "float"),
        ("controller_valid", (count, 2), "bool"),
        ("hand_joints_wrist", joints.shape, "float"),
        ("hand_joints_valid", (count, 2), "bool"),
        ("qpc_ns", (count,), "int"), ("source_frame_id", (count,), "int"),
        ("segment_id", (count,), "int"),
        ("controller_source_qpc_ns", (count, 2), "int"),
        ("hand_source_qpc_ns", (count, 2), "int"),
        ("reference_camera_to_world", (4, 4), "float"),
    ):
        _shape(source[key], shape, key, kind)
    for key in ("qpc_ns", "source_frame_id", "segment_id", "controller_source_qpc_ns", "hand_source_qpc_ns"):
        if source[key].dtype.kind == "u" and np.any(source[key] > np.iinfo(np.int64).max):
            raise TrainingDataError("source integer exceeds signed nanosecond/index range")
    if (count < 51 or np.any(source["qpc_ns"] < 0)
            or np.any(np.diff(source["qpc_ns"].astype(np.int64)) <= 0)
            or np.any(source["source_frame_id"] < 0)
            or np.any(np.diff(source["source_frame_id"].astype(np.int64)) <= 0)):
        raise TrainingDataError("source requires increasing native frames/QPC and at least H50+anchor")
    _proper_se3(source["controller_to_wrist"], "declared installation prior")
    _proper_se3(source["reference_camera_to_world"], "declared reference camera")
    _proper_se3(source["controller_pose_world"][source["controller_valid"]], "source controller")
    for valid_key, time_key in (("controller_valid", "controller_source_qpc_ns"),
                                ("hand_joints_valid", "hand_source_qpc_ns")):
        if np.any(source[time_key][source[valid_key]] < 0):
            raise TrainingDataError("observed source requires nonnegative source QPC")
    return source


def recompute_derived_labels(source_ref: Mapping[str, Any], retarget_contract_ref: Mapping[str, Any],
                             anchor_row: int, *, retarget_code_ref: Mapping[str, Any],
                             allow_synthetic_fixture: bool = False,
                             _input_cache: VerifiedInputCache | None = None) -> dict[str, Any]:
    """Pure NumPy H50 label preparation from SHA-bound human source arrays.

    Returns labels + missing_items; missing hand mappings yield false masks, not
    made-up commands. Source sidecars use host-QPC-aligned native frame rows,
    explicit controller installation priors and wrist-local MANUS joints.
    The declared camera transform is not certified physical calibration here.
    """
    cache = _input_cache if _input_cache is not None else VerifiedInputCache()
    reference_bytes(retarget_code_ref, cache=cache)
    if Path(retarget_code_ref["path"]) != Path(__file__).with_name("retarget.py"):
        raise TrainingDataError("retarget code ref must bind the actual new retarget module")
    contract = _hand_contract(retarget_contract_ref, allow_synthetic_fixture, cache)
    source = _source(source_ref, cache)
    if type(anchor_row) is not int or anchor_row < 0 or anchor_row + 50 >= len(source["qpc_ns"]):
        raise TrainingDataError("derived anchor row does not have a complete H50")
    rows = np.arange(anchor_row + 1, anchor_row + 51, dtype=np.int64)
    if not np.array_equal(source["source_frame_id"][rows],
                          np.arange(int(source["source_frame_id"][anchor_row]) + 1,
                                    int(source["source_frame_id"][anchor_row]) + 51)):
        raise TrainingDataError("derived H50 must be contiguous native frames")
    if np.any(source["segment_id"][rows] != source["segment_id"][anchor_row]):
        raise TrainingDataError("derived H50 crosses a source segment")
    kind = f"derived:{retarget_contract_ref['sha256']}:{retarget_code_ref['sha256']}:{anchor_row}"
    cached = cache.get(kind, source_ref)
    if cached is not None:
        return cached
    from .retarget import retarget_hand

    missing = set()
    pose_valid = source["controller_valid"][rows]
    controllers = source["controller_pose_world"][rows].copy()
    controllers[~pose_valid] = np.eye(4)
    wrists = controllers @ source["controller_to_wrist"]
    camera_from_world = np.linalg.inv(source["reference_camera_to_world"])
    poses = _pose9(camera_from_world @ wrists, np.zeros(3), np.ones(3))
    commands = np.zeros((50, 2, 6), dtype=np.float64)
    command_valid = np.zeros((50, 2, 6), dtype=bool)
    for horizon, row in enumerate(rows):
        for hand, side in enumerate(("left", "right")):
            if not source["hand_joints_valid"][row, hand]:
                missing.add("UNKNOWN_HAND_SOURCE")
                continue
            result = retarget_hand(source["hand_joints_wrist"][row, hand], side=side,
                                   hand_contract=contract,
                                   allow_synthetic_fixture=allow_synthetic_fixture)
            values, valid = np.asarray(result.normalized), np.asarray(result.valid)
            _shape(values, (6,), "derived command", "float")
            _shape(valid, (6,), "derived command valid", "bool")
            if (not np.isfinite(values[valid]).all()
                    or np.any(values[valid] < 0) or np.any(values[valid] > 1)):
                raise TrainingDataError("retarget admitted nonfinite/out-of-range command")
            commands[horizon, hand] = np.where(valid, values, 0)
            command_valid[horizon, hand] = valid
            missing.update(result.missing_items)
    mask = np.concatenate((np.repeat(pose_valid, 3, axis=1),
                           np.repeat(pose_valid, 6, axis=1), command_valid.reshape(50, 12)), axis=1)
    action = np.concatenate((poses[..., :3].reshape(50, 6), poses[..., 3:].reshape(50, 12),
                             commands.reshape(50, 12)), axis=1)
    action = np.where(mask, action, 0)
    current_valid = source["controller_valid"][anchor_row].copy()
    anchor = int(source["qpc_ns"][anchor_row])
    future = current_valid & (source["controller_source_qpc_ns"][anchor_row] > anchor)
    if future.any():
        missing.add("BLOCKED_FUTURE_CONTROLLER_CONDITIONING")
        current_valid[future] = False
    current_controllers = source["controller_pose_world"][anchor_row].copy()
    current_controllers[~current_valid] = np.eye(4)
    current = current_controllers @ source["controller_to_wrist"]
    ict = np.zeros((8, 29), dtype=np.float64)
    ict_mask = np.zeros(8, dtype=bool)
    conditioning_qpc = np.full(8, -1, dtype=np.int64)
    for hand in range(2):
        if not current_valid[hand]:
            continue
        relations = _pose9(np.linalg.inv(current[hand]) @ current, np.zeros(3), np.ones(3))
        relations[~current_valid] = 0
        ict[hand] = np.concatenate(([hand + 1], _pose9(camera_from_world @ current[hand],
                                  np.zeros(3), np.ones(3)), relations.reshape(18), [-1]))
        ict_mask[hand] = True
        conditioning_qpc[hand] = source["controller_source_qpc_ns"][anchor_row, hand]
    labels = {"x_ict": ict, "ict_mask": ict_mask, "y_action": action,
              "action_valid_mask": mask, "anchor_qpc_ns": np.array(anchor, dtype=np.int64),
              "conditioning_source_qpc_ns": conditioning_qpc,
              "target_qpc_ns": source["qpc_ns"][rows].astype(np.int64),
              "target_source_frame_ids": source["source_frame_id"][rows].astype(np.int64),
              "target_source_rows": rows, "current_source_row": np.array(anchor_row, dtype=np.int64),
              "reference_camera_to_world": source["reference_camera_to_world"].copy()}
    for value in labels.values():
        value.flags.writeable = False
    prepared = {"labels": labels, "missing_items": sorted(missing),
                "supervision_class": SUPERVISION_CLASS, "control_ground_truth": False,
                "physical_deployment_authorized": False}
    cache.put(kind, source_ref, prepared, sum(value.nbytes for value in labels.values()))
    return prepared


def _checked_sample(sample, manifest, cache, allow_synthetic_fixture):
    labels = _npz(sample["labels"], cache=cache)
    if not LABEL_KEYS.issubset(labels) or any(key in labels for key in ("q_arm", "q_hand", "digital_q", "retarget_q")):
        raise TrainingDataError("derived labels require explicit human-source H50 arrays, not digital q")
    receipt = _receipt(sample["derivation_receipt"], "humanego-revo2-derived-retarget-v1",
                       cache=cache, allow_synthetic_fixture=allow_synthetic_fixture,
                       supervision_class=SUPERVISION_CLASS, control_ground_truth=False,
                       physical_deployment_authorized=False, source_kind=SOURCE_KIND,
                       session_id=sample["session_id"], labels_sha256=sample["labels"]["sha256"],
                       source_sha256=sample["human_source_sidecar"]["sha256"],
                       retarget_contract_sha256=manifest["retarget_contract"]["sha256"],
                       retarget_code_sha256=manifest["retarget_code"]["sha256"])
    if _pose_clock_declaration(receipt) != _pose_clock_declaration(manifest["pose_clock"]):
        raise TrainingDataError("derived source frame/unit/clock declaration differs")
    if (receipt.get("hand_command_channel_order") != CHANNEL_ORDER
            or receipt.get("hand_command_unit") != "NORMALIZED_0_1"):
        raise TrainingDataError("derived receipt hand channel order/unit differs")
    _shape(labels["current_source_row"], (), "current_source_row", "int")
    prepared = recompute_derived_labels(sample["human_source_sidecar"], manifest["retarget_contract"],
        int(labels["current_source_row"]), retarget_code_ref=manifest["retarget_code"],
        allow_synthetic_fixture=allow_synthetic_fixture, _input_cache=cache)
    source = _source(sample["human_source_sidecar"], cache)
    if int(source["source_frame_id"][int(labels["current_source_row"])]) != sample["source_frame_id"]:
        raise TrainingDataError("derived sample identity does not bind source frame")
    for key, expected in prepared["labels"].items():
        _shape(labels[key], expected.shape, key,
               "bool" if expected.dtype.kind == "b" else "int" if expected.dtype.kind in "iu" else "float")
    for key, expected in prepared["labels"].items():
        if expected.dtype.kind == "f":
            valid = labels["action_valid_mask"] if key == "y_action" else np.ones(expected.shape, dtype=bool)
            if not np.isfinite(labels[key][valid]).all() or not np.allclose(labels[key][valid], expected[valid], atol=1e-5, rtol=1e-5):
                raise TrainingDataError(f"derived {key} differs from recomputed human source")
        elif not np.array_equal(labels[key], expected):
            raise TrainingDataError(f"derived {key} differs from recomputed human source")
    return labels, prepared["missing_items"]


def _read_manifest(path, cache, allow_synthetic_fixture):
    path = _ordinary(Path(path).absolute())
    before = VerifiedInputCache._stat(path)
    encoded = path.read_bytes()
    if before != VerifiedInputCache._stat(path):
        raise TrainingDataError("derived manifest changed during validation")
    manifest = _object(encoded)
    if manifest.get("fixture_only") is True and not allow_synthetic_fixture:
        raise TrainingDataError("synthetic fixture manifest is not production evidence")
    if (manifest.get("schema") != SCHEMA or manifest.get("supervision_class") != SUPERVISION_CLASS
            or manifest.get("target") != TARGET or manifest.get("control_ground_truth") is not False
            or manifest.get("physical_deployment_authorized") is not False):
        raise TrainingDataError("separate DERIVED_RETARGET non-ground-truth manifest required")
    if (not isinstance(manifest.get("pair_id"), str) or not manifest["pair_id"].strip()
            or manifest.get("action_layout") != ACTION_LAYOUT
            or manifest.get("ict_schema") != "HAND_ONLY_29D_GRASP_UNKNOWN_MINUS_ONE"
            or manifest.get("normalization") != {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}):
        raise TrainingDataError("explicit paired H50/30D identity-normalized derived contract required")
    if any(key not in manifest for key in ("retarget_contract", "retarget_code", "pose_clock")):
        raise TrainingDataError("derived retarget source/code/pose_clock declarations missing")
    if not isinstance(manifest["pose_clock"], Mapping):
        raise TrainingDataError("derived pose_clock must be an explicit object")
    _pose_clock_declaration(manifest["pose_clock"])
    _hand_contract(manifest["retarget_contract"], allow_synthetic_fixture, cache)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise TrainingDataError("derived manifest has no samples")
    ids, frames, session_groups, group_splits = set(), set(), {}, {}
    counts = {"train": 0, "validation": 0}
    valid_counts = {split: np.zeros(30, dtype=np.int64) for split in counts}
    missing = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise TrainingDataError("derived sample must be an object")
        required = {"labels", "human_source_sidecar", "derivation_receipt", "rgb"}
        if not required.issubset(sample):
            raise TrainingDataError("derived sample source/labels/receipt/RGB fields missing")
        for key in ("sample_id", "session_id", "source_group_id"):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise TrainingDataError(f"derived sample requires {key}")
        frame = sample.get("source_frame_id")
        if type(frame) is not int or frame < 0 or sample.get("split") not in counts:
            raise TrainingDataError("derived native frame/split invalid")
        identity = (sample["session_id"], frame)
        if sample["sample_id"] in ids or identity in frames:
            raise TrainingDataError("duplicate derived sample/frame")
        ids.add(sample["sample_id"]); frames.add(identity)
        group, split, session = sample["source_group_id"], sample["split"], sample["session_id"]
        if group in group_splits and group_splits[group] != split:
            raise TrainingDataError("independent source group crosses train/validation")
        if session in session_groups and session_groups[session] != group:
            raise TrainingDataError("session cannot claim multiple derived source groups")
        group_splits[group], session_groups[session] = split, group
        weight = sample.get("sample_weight", 1.0)
        if type(weight) not in (int, float) or not np.isfinite(weight) or not 0 < weight <= 1:
            raise TrainingDataError("derived sample_weight must be finite in (0,1]")
        if not isinstance(sample["rgb"], Mapping) or set(sample["rgb"]) != {"raw", "robotized"}:
            raise TrainingDataError("derived pair requires raw and robotized RGB refs")
        for ref in sample["rgb"].values():
            reference_bytes(ref)
        labels, sample_missing = _checked_sample(sample, manifest, cache, allow_synthetic_fixture)
        counts[split] += 1
        valid_counts[split] += labels["action_valid_mask"].sum(axis=0)
        missing.update(sample_missing)
    return manifest, {"dataset_sha": hashlib.sha256(encoded).hexdigest(), "manifest_metadata": before,
        "sample_counts": counts, "source_group_splits": group_splits,
        "valid_action_counts": {split: value.tolist() for split, value in valid_counts.items()},
        "missing_items": sorted(missing), "supervision_class": SUPERVISION_CLASS,
        "control_ground_truth": False, "physical_deployment_authorized": False}


def _training_blockers(manifest, summary, branch, cache, allow_synthetic_fixture):
    errors = []
    def check(code, function):
        try:
            function()
        except (TrainingDataError, KeyError) as error:
            errors.append({"code": code, "detail": str(error)})
    def calibration():
        receipt = _checked_calibration(manifest, allow_synthetic_fixture=allow_synthetic_fixture, cache=cache)
        if _pose_clock_declaration(receipt) != _pose_clock_declaration(manifest["pose_clock"]):
            raise TrainingDataError("derived source/calibration frame, TCP, unit or clock differs")
    check("BLOCKED_PHYSICAL_CALIBRATION_OR_CLOCK", calibration)
    for split, counts in summary["valid_action_counts"].items():
        missing = np.flatnonzero(np.asarray(counts) == 0).tolist()
        if not summary["sample_counts"][split] or missing:
            errors.append({"code": "BLOCKED_EFFECTIVE_SUPERVISION", "split": split,
                           "missing_output_dimensions": missing})
    for item in summary["missing_items"]:
        if item.startswith("BLOCKED_"):
            errors.append({"code": item})
    for sample in manifest["samples"]:
        if sample["source_group_id"] == "UNKNOWN_SOURCE_GROUP":
            errors.append({"code": "BLOCKED_INDEPENDENT_SOURCE_GROUP", "detail": "UNKNOWN source group is not acquisition proof"})
        check("BLOCKED_INDEPENDENT_SOURCE_GROUP", lambda sample=sample: _receipt(
            sample["source_group_receipt"], "independent-acquisition-source-group-v1",
            cache=cache, allow_synthetic_fixture=allow_synthetic_fixture,
            session_id=sample["session_id"], source_group_id=sample["source_group_id"], independent_acquisition=True))
        labels = _npz(sample["labels"], cache=cache)
        if not labels["ict_mask"].any():
            errors.append({"code": "BLOCKED_CURRENT_ICT_ABSENT", "sample_id": sample["sample_id"]})
        check("BLOCKED_RGB_CAUSAL_ADMISSION", lambda sample=sample, labels=labels: _check_rgb_causal(
            sample, branch, labels, manifest["pose_clock"]["clock_domain"],
            cache=cache, allow_synthetic_fixture=allow_synthetic_fixture))
    return errors


def validate_derived_preparation_manifest(manifest_path: str | Path, *,
        allow_synthetic_fixture: bool = False, _input_cache: VerifiedInputCache | None = None) -> dict[str, Any]:
    """Verify source-recomputed labels; report remaining gates without authorizing training."""
    cache = _input_cache if _input_cache is not None else VerifiedInputCache()
    manifest, summary = _read_manifest(manifest_path, cache, allow_synthetic_fixture)
    summary.update({"manifest": manifest, "preparation_status": (
                        "DERIVED_LABELS_VERIFIED_WITH_UNKNOWN_TARGETS" if summary["missing_items"]
                        else "DERIVED_NUMERIC_LABELS_VERIFIED"),
                    "training_start_authorized": False,
                    "training_admission": {branch: _training_blockers(manifest, summary, branch, cache,
                        allow_synthetic_fixture) for branch in ("raw", "robotized")}})
    return summary


def validate_derived_training_manifest(manifest_path: str | Path, branch: str, *,
        allow_synthetic_fixture: bool = False, _input_cache: VerifiedInputCache | None = None) -> dict[str, Any]:
    """Strict selected-branch admission; task/GPU authority is a separate external gate."""
    if branch not in BRANCHES:
        raise TrainingDataError("unknown derived RGB branch")
    summary = validate_derived_preparation_manifest(manifest_path,
        allow_synthetic_fixture=allow_synthetic_fixture, _input_cache=_input_cache)
    blockers = summary["training_admission"][BRANCHES[branch]]
    if blockers:
        raise TrainingDataError(f"derived training admission blocked: {blockers}")
    summary["admitted_branch"] = BRANCHES[branch]
    return summary


class DerivedRevo2ICTDataset:
    """Same deterministic tensor API as real-control reader; never bypasses RGB gates."""
    def __init__(self, manifest_path: str | Path, branch: str, split: str,
                 image_size=(240, 320), horizon=50, *, allow_synthetic_fixture=False):
        if split not in {"train", "validation"} or tuple(image_size) != (240, 320) or horizon != 50:
            raise TrainingDataError("derived reader fixes train/validation, RGB240x320 and H50")
        self.input_cache = VerifiedInputCache()
        admitted = validate_derived_training_manifest(manifest_path, branch,
            allow_synthetic_fixture=allow_synthetic_fixture, _input_cache=self.input_cache)
        self.manifest = admitted["manifest"]
        self.dataset_sha = admitted["dataset_sha"]
        self.manifest_path = Path(manifest_path).absolute()
        self.manifest_metadata = admitted["manifest_metadata"]
        self.branch, self.split = BRANCHES[branch], split
        self.samples = [sample for sample in self.manifest["samples"] if sample["split"] == split]
        self.allow_synthetic_fixture = allow_synthetic_fixture

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if VerifiedInputCache._stat(self.manifest_path) != self.manifest_metadata:
            raise TrainingDataError("derived immutable manifest metadata/SHA authority changed")
        sample = self.samples[index]
        labels, _ = _checked_sample(sample, self.manifest, self.input_cache, self.allow_synthetic_fixture)
        _checked_calibration(self.manifest, allow_synthetic_fixture=self.allow_synthetic_fixture, cache=self.input_cache)
        _receipt(sample["source_group_receipt"], "independent-acquisition-source-group-v1",
                 cache=self.input_cache, allow_synthetic_fixture=self.allow_synthetic_fixture,
                 session_id=sample["session_id"], source_group_id=sample["source_group_id"], independent_acquisition=True)
        _check_rgb_causal(sample, self.branch, labels, self.manifest["pose_clock"]["clock_domain"],
                         cache=self.input_cache, allow_synthetic_fixture=self.allow_synthetic_fixture)
        from PIL import Image
        import torch
        try:
            with Image.open(io.BytesIO(reference_bytes(sample["rgb"][self.branch]))) as source:
                rgb = np.asarray(source.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        except (OSError, ValueError) as error:
            raise TrainingDataError("derived reviewed RGB cannot be decoded") from error
        mask = labels["action_valid_mask"]
        return {"x_rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
                "x_ict": torch.from_numpy(labels["x_ict"].astype(np.float32)),
                "ict_mask": torch.from_numpy(labels["ict_mask"].copy()),
                "y_action": torch.from_numpy(np.where(mask, labels["y_action"], 0).astype(np.float32)),
                "action_valid_mask": torch.from_numpy(mask.copy()),
                "sample_weight": torch.tensor(sample.get("sample_weight", 1.0), dtype=torch.float32),
                "sample_id": sample["sample_id"], "session_id": sample["session_id"],
                "source_frame_id": sample["source_frame_id"], "source_group_id": sample["source_group_id"],
                "branch": self.branch, "split": self.split, "sample_index": index,
                "supervision_class": SUPERVISION_CLASS, "control_ground_truth": False,
                "physical_deployment_authorized": False}
