"""SHA-bound, NumPy-only reader for derived ICT joint26 research labels.

This is a new data contract, not an upgrade of wrist labels to measured robot
controls. Invalid/unknown fields remain masked. No torch, model, IK, source NAS
replay or device library is imported by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from .project_mount_paths import reference_equal, resolve_input_path

SCHEMA = "humanego-ict-joint26-paired-manifest-v1"
LAYOUT = "left_arm7,left_hand6,right_arm7,right_hand6"
SOURCE_POLICY = "FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY"
SUPERVISION_CLASS = "DERIVED_RETARGET_ASSUMED_MODEL_JOINTS"
STRICT_ADMISSION_POLICY = "STRICT_MODEL_NUMERICS_V1"
EGO_ADMISSION_POLICY = "EGO_DERIVED_STRUCTURAL_V1"
EGO_ADMITTED_STATUS = "ADMITTED_EGO_DERIVED_LABELS_DIAGNOSTICS_ONLY"
EGO_VALIDITY_BASIS = "SOURCE_CAUSAL_TIME_MASK_FINITE_BOUNDED_UNITS_ORDER_ONLY"
ARM_INDICES = np.array(list(range(7)) + list(range(13, 20)))
HAND_INDICES = np.array(list(range(7, 13)) + list(range(20, 26)))
FIELDS = ("x_ict", "ict_mask", "x_state26", "state_mask", "y_action", "action_valid_mask")
SHAPES = ((8, 29), (8,), (26,), (26,), (30, 26), (30, 26))


class Joint26DataError(ValueError):
    pass


def _ordinary(path, root):
    try:
        return resolve_input_path(path, root)
    except ValueError as error:
        raise Joint26DataError(str(error)) from error


def file_ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def reference_bytes(reference, root):
    if (not isinstance(reference, dict) or set(reference) != {"path", "bytes", "sha256"}
            or type(reference.get("bytes")) is not int or reference["bytes"] < 0
            or not isinstance(reference.get("sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", reference["sha256"])):
        raise Joint26DataError("EXACT_SHA_REFERENCE_REQUIRED")
    path = _ordinary(reference["path"], root)
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    fp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fp(before) != fp(after) or len(raw) != reference["bytes"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise Joint26DataError("JOINT26_SOURCE_SHA_OR_IDENTITY_DRIFT:" + str(path))
    return raw


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Joint26DataError("DUPLICATE_JSON_KEY:" + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(Joint26DataError("NONFINITE_JSON:" + v)))


def validate_manifest_header(manifest):
    """Structural contract only; does not grant label/source qualification."""
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("action_dim") != 26 or manifest.get("horizon") != 30
            or manifest.get("action_layout") != LAYOUT or manifest.get("ict_dim") != 29
            or manifest.get("max_ict") != 8 or manifest.get("image_size") != [240, 320]
            or manifest.get("source_policy") != SOURCE_POLICY
            or manifest.get("supervision_class") != SUPERVISION_CLASS
            or manifest.get("control_ground_truth") is not False
            or manifest.get("physical_deployment_authorized") is not False
            or manifest.get("parameter_mode") != "ASSUMED"):
        raise Joint26DataError("JOINT26_RESEARCH_MANIFEST_CONTRACT_REQUIRED")
    if manifest.get("normalization") != {"method": "IDENTITY_NO_FIT"}:
        raise Joint26DataError("EXPLICIT_JOINT26_IDENTITY_NORMALIZATION_REQUIRED")
    policy = manifest.get("training_admission_policy", STRICT_ADMISSION_POLICY)
    if policy not in {STRICT_ADMISSION_POLICY, EGO_ADMISSION_POLICY}:
        raise Joint26DataError("EXPLICIT_JOINT26_TRAINING_ADMISSION_POLICY_REQUIRED")
    if policy == EGO_ADMISSION_POLICY and (
            manifest.get("label_validity_basis") != EGO_VALIDITY_BASIS
            or manifest.get("robot_quality_passed") is not False):
        raise Joint26DataError("EGO_DERIVED_STRUCTURAL_NOT_ROBOT_QUALITY_REQUIRED")
    if not isinstance(manifest.get("samples"), list):
        raise Joint26DataError("EXACT_JOINT26_SAMPLE_INDEX_REQUIRED")
    return manifest


def _labels(blob):
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
    except (ValueError, OSError, KeyError) as error:
        raise Joint26DataError("JOINT26_NUMERIC_NPZ_REQUIRED") from error
    for key, shape in zip(FIELDS, SHAPES):
        value = arrays.get(key)
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise Joint26DataError("JOINT26_LABEL_SHAPE:" + key)
        if "mask" in key:
            if value.dtype != np.dtype(bool):
                raise Joint26DataError("JOINT26_BOOL_MASK_REQUIRED:" + key)
        elif value.dtype.kind != "f" or not np.isfinite(value).all():
            raise Joint26DataError("JOINT26_FINITE_FLOAT_LABEL_REQUIRED:" + key)
    if (not arrays["ict_mask"][:2].all() or arrays["ict_mask"][2:].any()
            or np.any(arrays["x_ict"][~arrays["ict_mask"]] != 0)):
        raise Joint26DataError("JOINT26_CURRENT_BILATERAL_HAND_ICT_REQUIRED")
    for side in range(2):
        if arrays["x_ict"][side, 0] != side + 1 or arrays["x_ict"][side, -1] != -1:
            raise Joint26DataError("JOINT26_ICT_TYPE_OR_UNKNOWN_GRASP_DRIFT")
    if not arrays["state_mask"][ARM_INDICES].all():
        raise Joint26DataError("JOINT26_CURRENT_DERIVED_ARM_STATE_REQUIRED")
    if arrays["state_mask"][HAND_INDICES].any():
        raise Joint26DataError("JOINT26_CURRENT_HANDS_ARE_UNKNOWN_NOT_FUTURE_LABELS")
    if np.any(arrays["x_state26"][~arrays["state_mask"]] != 0):
        raise Joint26DataError("JOINT26_UNKNOWN_STATE_MUST_BE_ZERO_AND_MASKED")
    for value, mask in ((arrays["x_state26"], arrays["state_mask"]),
                        (arrays["y_action"], arrays["action_valid_mask"])):
        hand = value[..., HAND_INDICES]; valid = mask[..., HAND_INDICES]
        if np.any(valid & ((hand < 0) | (hand > 1))):
            raise Joint26DataError("JOINT26_HAND_OUTSIDE_NORMALIZED_RANGE")
    if not arrays["action_valid_mask"].all():
        raise Joint26DataError("INCOMPLETE_JOINT26_WINDOW_CANNOT_BE_ADMITTED")
    for key in ("target_qpc_ns", "target_source_frame_ids"):
        if key not in arrays or arrays[key].shape != (30,) or arrays[key].dtype.kind not in "iu":
            raise Joint26DataError("JOINT26_EXACT_NATIVE_MAPPING_REQUIRED:" + key)
    if "anchor_qpc_ns" not in arrays or arrays["anchor_qpc_ns"].shape != () or arrays["anchor_qpc_ns"].dtype.kind not in "iu":
        raise Joint26DataError("JOINT26_INTEGER_ANCHOR_REQUIRED")
    times, ids, anchor = arrays["target_qpc_ns"], arrays["target_source_frame_ids"], int(arrays["anchor_qpc_ns"])
    if anchor < 0 or np.any(times <= anchor) or np.any(np.diff(times) <= 0) or np.any(np.diff(ids) <= 0):
        raise Joint26DataError("JOINT26_NONCAUSAL_OR_NONMONOTONIC_MAPPING")
    return arrays


class Joint26Dataset:
    """Preload a bounded branch/split; SHA rechecks only at explicit boundaries."""
    def __init__(self, manifest_path, branch, split="train", *, project_root=None, max_ram_bytes=1 << 30):
        if branch not in {"raw", "robotized"} or split not in {"train", "validation"}:
            raise Joint26DataError("EXPLICIT_JOINT26_BRANCH_AND_SPLIT_REQUIRED")
        if type(max_ram_bytes) is not int or max_ram_bytes <= 0:
            raise Joint26DataError("FINITE_RAM_ARRAY_BUDGET_REQUIRED")
        root = Path(project_root or Path(__file__).resolve().parents[3])
        self.root = _ordinary(root, root)
        path = _ordinary(manifest_path, self.root)
        self.manifest_ref = file_ref(path)
        self.manifest = validate_manifest_header(_json(reference_bytes(self.manifest_ref, self.root)))
        policy = self.manifest.get("training_admission_policy", STRICT_ADMISSION_POLICY)
        ego_derived = policy == EGO_ADMISSION_POLICY
        admitted_status = EGO_ADMITTED_STATUS if ego_derived else "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
        self.dataset_sha, self.branch, self.split = self.manifest_ref["sha256"], branch, split
        self._refs = {str(path): self.manifest_ref}; self._rows = []
        samples, groups, seen, group_split = [], {"train": set(), "validation": set()}, set(), {}
        for sample in self.manifest["samples"]:
            if (not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str)
                    or sample["sample_id"] in seen or sample.get("split") not in groups
                    or not isinstance(sample.get("source_group_id"), str)
                    or not sample["source_group_id"] or type(sample.get("source_frame_id")) is not int
                    or sample["source_frame_id"] < 0):
                raise Joint26DataError("JOINT26_EXACT_SAMPLE_IDENTITY_REQUIRED")
            seen.add(sample["sample_id"])
            group = sample["source_group_id"]
            if group in group_split and group_split[group] != sample["split"]:
                raise Joint26DataError("JOINT26_TRAIN_VALIDATION_GROUP_LEAKAGE")
            group_split[group] = sample["split"]
            entry = sample.get("branches", {}).get(branch, {})
            if entry.get("admitted") is True:
                if sample.get("conversion_status") != admitted_status:
                    raise Joint26DataError("JOINT26_ADMISSION_WITHOUT_NUMERIC_EVIDENCE")
                groups[sample["split"]].add(group)
                samples.append(sample)
        self.samples = [deepcopy(s) for s in samples if s["split"] == split]
        if not self.samples:
            raise Joint26DataError("NO_ADMITTED_JOINT26_SAMPLES:" + branch + ":" + split)
        self.preloaded_array_bytes = 0
        for index, sample in enumerate(self.samples):
            labels_ref, rgb_ref = sample["labels"], sample["branches"][branch]["rgb"]
            arrays = _labels(self._consume(labels_ref))
            diagnostic = _json(self._consume(sample["diagnostics"]))
            if (diagnostic.get("status") != admitted_status
                    or diagnostic.get("current_status") != admitted_status
                    or diagnostic.get("control_ground_truth") is not False
                    or diagnostic.get("physical_deployment_authorized") is not False
                    or diagnostic.get("robot_commands_sent") != 0
                    or diagnostic.get("frame_errors") != [[] for _ in range(30)]
                    or diagnostic.get("valid_frames") != 30):
                raise Joint26DataError("JOINT26_COMPLETE_NUMERICAL_DIAGNOSTIC_REQUIRED")
            if ego_derived:
                quality = diagnostic.get("quality_diagnostics")
                if (diagnostic.get("training_admission_policy") != policy
                        or diagnostic.get("label_validity_basis") != EGO_VALIDITY_BASIS
                        or diagnostic.get("training_quality_gates_enforced") is not False
                        or diagnostic.get("robot_quality_passed") is not False
                        or diagnostic.get("settling_steps_used") != 60
                        or not isinstance(quality, dict)
                        or not isinstance(quality.get("current"), dict)
                        or not isinstance(quality.get("frames"), list)
                        or len(quality["frames"]) != 30
                        or any(not isinstance(frame, dict) for frame in quality["frames"])):
                    raise Joint26DataError("EGO_DERIVED_QUALITY_DIAGNOSTICS_REQUIRED")
            elif diagnostic.get("training_admission_policy", STRICT_ADMISSION_POLICY) != policy:
                raise Joint26DataError("JOINT26_ADMISSION_POLICY_DRIFT")
            if np.any(arrays["target_source_frame_ids"] <= sample["source_frame_id"]):
                raise Joint26DataError("JOINT26_TARGET_MUST_FOLLOW_CURRENT_SOURCE_FRAME")
            limits = self.manifest.get("numerical_limits", {})
            try:
                lower, upper = np.asarray(limits["joint_lower_rad"]), np.asarray(limits["joint_upper_rad"])
                speed, step = np.asarray(limits["joint_speed_max_rad_s"]), np.asarray(limits["joint_step_max_rad"])
            except (TypeError, KeyError) as error:
                raise Joint26DataError("JOINT26_SIGNED_NUMERICAL_LIMITS_REQUIRED") from error
            if (any(v.shape != (14,) or not np.isfinite(v).all() for v in (lower, upper, speed, step))
                    or np.any(lower >= upper) or np.any(speed <= 0) or np.any(step <= 0)
                    or diagnostic.get("numerical_limits") != limits):
                raise Joint26DataError("JOINT26_NUMERICAL_LIMIT_SOURCE_DRIFT")
            q_now = arrays["x_state26"][ARM_INDICES]
            q_targets = arrays["y_action"][:, ARM_INDICES]
            if np.any(q_now < lower) or np.any(q_now > upper) or np.any(q_targets < lower) or np.any(q_targets > upper):
                raise Joint26DataError("JOINT26_CACHED_JOINT_LIMIT")
            dt = np.diff(np.concatenate(([int(arrays["anchor_qpc_ns"])], arrays["target_qpc_ns"]))) / 1e9
            delta = np.diff(np.concatenate((q_now[None], q_targets)), axis=0)
            if not ego_derived and (np.any(np.abs(delta) > step) or np.any(np.abs(delta) > speed[None] * dt[:, None])):
                raise Joint26DataError("JOINT26_CACHED_STEP_OR_SPEED_LIMIT")
            from PIL import Image
            try:
                with Image.open(io.BytesIO(self._consume(rgb_ref))) as image:
                    rgb = np.asarray(image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR), np.float32) / 255
            except (ValueError, OSError) as error:
                raise Joint26DataError("JOINT26_RGB_CANNOT_DECODE") from error
            row = {key: arrays[key].copy() for key in FIELDS}
            row.update(x_rgb=rgb.transpose(2, 0, 1).copy(),
                       sample_weight=np.array(sample.get("sample_weight", 1), np.float32))
            if not np.isfinite(row["sample_weight"]) or float(row["sample_weight"]) <= 0:
                raise Joint26DataError("JOINT26_POSITIVE_SAMPLE_WEIGHT_REQUIRED")
            for key in row:
                if row[key].dtype.kind == "f":
                    row[key] = row[key].astype(np.float32)
                row[key].setflags(write=False)
                self.preloaded_array_bytes += row[key].nbytes
            if self.preloaded_array_bytes > max_ram_bytes:
                raise Joint26DataError("JOINT26_RAM_ARRAY_BUDGET_EXCEEDED")
            row.update({key: deepcopy(sample[key]) for key in
                        ("sample_id", "session_id", "source_frame_id", "source_group_id", "split")})
            row.update(sample_index=index, branch=branch,
                       supervision_class=SUPERVISION_CLASS, training_admission_policy=policy,
                       control_ground_truth=False,
                       physical_deployment_authorized=False)
            self._rows.append(row)
        self._qualification = {"schema": "humanego-ict-joint26-data-qualification-v1", "branch": branch,
            "source_policy": SOURCE_POLICY, "supervision_class": SUPERVISION_CLASS,
            "training_admission_policy": policy,
            "label_validity_basis": self.manifest.get("label_validity_basis", "STRICT_ASSUMED_MODEL_NUMERICS_ONLY"),
            "robot_quality_passed": False,
            "training_scope": "DERIVED_JOINT_RESEARCH_ONLY", "parameter_mode": "ASSUMED",
            "manifest_ref": deepcopy(self.manifest_ref), "dataset_sha": self.dataset_sha,
            "sample_counts": {s: sum(x["split"] == s for x in samples) for s in groups},
            "source_group_splits": {s: sorted(values) for s, values in groups.items()},
            "admitted_sample_ids": [s["sample_id"] for s in samples],
            "normalization": deepcopy(self.manifest["normalization"]),
            "numerical_limits": deepcopy(self.manifest["numerical_limits"]),
            "native_time_support": {"frame_ids_and_qpc_preserved": True,
                "timebase_qualification": self.manifest.get("timebase_qualification", "SOURCE_DECLARED_NOT_PHYSICALLY_VERIFIED"),
                "timebase_verified": False, "no_nominal_fps_resampling": True},
            "data_gate": {"status": "PASSED_RESEARCH_LABEL_GATE" if len(groups["train"]) >= 16 and len(groups["validation"]) >= 3 else "BLOCKED_DATA_VOLUME",
                          "train_groups": len(groups["train"]), "validation_groups": len(groups["validation"]),
                          "minimum_train_groups": 16, "minimum_validation_groups": 3},
            "current_nas_content_verified": False, "control_ground_truth": False,
            "physical_deployment_authorized": False, "source_qualification_granted": False,
            "preloaded_numeric_array_bytes": self.preloaded_array_bytes}
        self.verify_all_dependencies()

    def _consume(self, reference):
        path = str(_ordinary(reference["path"], self.root))
        if path in self._refs and not reference_equal(self._refs[path], reference, self.root):
            raise Joint26DataError("JOINT26_CONFLICTING_PATH_SIGNATURE")
        self._refs.setdefault(path, deepcopy(reference))
        return reference_bytes(reference, self.root)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise Joint26DataError("JOINT26_EXACT_SAMPLE_INDEX_REQUIRED")
        return {key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
                for key, value in self._rows[index].items()}

    @property
    def qualification(self):
        return deepcopy(self._qualification)

    def verify_all_dependencies(self):
        for reference in self._refs.values():
            reference_bytes(reference, self.root)


def load_joint26_cache(manifest_path, branch, split="train", **kwargs):
    dataset = Joint26Dataset(manifest_path, branch, split, **kwargs)
    return dataset, dataset.qualification
