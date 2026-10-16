"""Fail-closed paired Revo2 ICT manifest and immutable sample reader.

The 30D target is [Lxyz,Rxyz | Lrot6,Rrot6 | Lcommand6,Rcommand6].
Only reviewed synchronized REAL robot sidecars can supervise it. Existing
pick/place digital q22/q7 files are intentionally not an accepted input format.
Torch is imported only when an admitted sample is requested.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "pick-place-humanego-ict-revo2-paired-v1"
TARGET = "REVO2_DUAL_HAND_EEF_POSE_AND_COMMAND6"
BRANCHES = {"raw": "raw", "robotized": "robotized",
            "HUMAN_RAW_RGB": "raw", "ROBOTIZED_RGB": "robotized"}
ACTION_LAYOUT = "Lxyz,Rxyz|Lrot6,Rrot6|L6hand,R6hand"
LABEL_KEYS = {
    "x_ict", "ict_mask", "y_action", "action_valid_mask", "anchor_qpc_ns",
    "conditioning_source_qpc_ns", "target_qpc_ns", "target_source_frame_ids",
    "target_robot_rows", "current_robot_row", "reference_camera_to_world",
}
SIDECAR_KEYS = {"wrist_pose_world", "hand_command", "valid", "qpc_ns",
                "source_frame_id", "segment_id"}
DEFAULT_INPUT_CACHE_BYTES = 128 * 1024 * 1024


class TrainingDataError(ValueError):
    """A missing or mismatched qualification is a hard admission failure."""


class VerifiedInputCache:
    """Process-local 128 MiB LRU for SHA-verified small evidence/numeric files.

    Cached arrays are read-only. Every reuse checks inode/size/mtime/ctime and
    rejects changed backing files; immutable refs remain part of dataset_sha.
    RGB is never put in this cache. clear() releases all retained data.
    """

    def __init__(self, max_bytes: int = DEFAULT_INPUT_CACHE_BYTES):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._entries = OrderedDict()

    @staticmethod
    def _key(kind, reference):
        if not isinstance(reference, Mapping) or set(reference) != {"path", "bytes", "sha256"}:
            raise TrainingDataError("file reference requires exactly path/bytes/sha256")
        if (not isinstance(reference["path"], str)
                or not isinstance(reference["bytes"], int)
                or isinstance(reference["bytes"], bool) or reference["bytes"] < 0
                or not isinstance(reference["sha256"], str)
                or len(reference["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in reference["sha256"])):
            raise TrainingDataError("invalid immutable file reference fields")
        return kind, str(reference["path"]), reference["bytes"], reference["sha256"]

    @staticmethod
    def _stat(path):
        stat = Path(path).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def get(self, kind, reference):
        key = self._key(kind, reference)
        entry = self._entries.get(key)
        if entry is None:
            return None
        try:
            observed = self._stat(reference["path"])
        except OSError as error:
            raise TrainingDataError("cached immutable input disappeared") from error
        if observed != entry[0]:
            raise TrainingDataError("cached immutable input metadata/SHA authority changed")
        self._entries.move_to_end(key)
        return entry[1]

    def put(self, kind, reference, value, size):
        key = self._key(kind, reference)
        if size > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous[2]
        while self._entries and self.current_bytes + size > self.max_bytes:
            _, old = self._entries.popitem(last=False)
            self.current_bytes -= old[2]
        self._entries[key] = (self._stat(reference["path"]), value, size)
        self.current_bytes += size

    def remove(self, kind, reference):
        value = self._entries.pop(self._key(kind, reference), None)
        if value is not None:
            self.current_bytes -= value[2]

    def clear(self):
        self._entries.clear()
        self.current_bytes = 0


def _ordinary(path: Path) -> Path:
    if not path.is_absolute():
        raise TrainingDataError("input references must use absolute paths")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise TrainingDataError(f"symbolic-link input forbidden: {component}")
    if not path.is_file():
        raise TrainingDataError(f"ordinary input file missing: {path}")
    return path


def reference_bytes(reference: Mapping[str, Any], *, cache: VerifiedInputCache | None = None) -> bytes:
    """Verify exact bytes before parsing; never follow historical path aliases."""
    if not isinstance(reference, Mapping) or set(reference) != {"path", "bytes", "sha256"}:
        raise TrainingDataError("file reference requires exactly path/bytes/sha256")
    VerifiedInputCache._key("bytes", reference)
    if cache is not None:
        cached = cache.get("bytes", reference)
        if cached is not None:
            return cached
    path = _ordinary(Path(str(reference["path"])))
    if not isinstance(reference["bytes"], int) or isinstance(reference["bytes"], bool):
        raise TrainingDataError("reference byte count must be an integer")
    before = VerifiedInputCache._stat(path)
    encoded = path.read_bytes()
    if VerifiedInputCache._stat(path) != before:
        raise TrainingDataError(f"input changed during SHA verification: {path}")
    if len(encoded) != reference["bytes"] or hashlib.sha256(encoded).hexdigest() != reference["sha256"]:
        raise TrainingDataError(f"input SHA/size drift: {path}")
    if cache is not None:
        cache.put("bytes", reference, encoded, len(encoded))
    return encoded


def _object(encoded: bytes) -> dict[str, Any]:
    try:
        value = json.loads(encoded)
    except (ValueError, UnicodeDecodeError) as error:
        raise TrainingDataError("invalid JSON input") from error
    if not isinstance(value, dict):
        raise TrainingDataError("JSON input must be an object")
    return value


def _npz(reference: Mapping[str, Any], *, cache: VerifiedInputCache | None = None) -> dict[str, np.ndarray]:
    if cache is not None:
        cached = cache.get("npz", reference)
        if cached is not None:
            return cached
    try:
        with np.load(io.BytesIO(reference_bytes(reference, cache=cache)), allow_pickle=False) as source:
            result = {key: source[key] for key in source.files}
    except (OSError, ValueError, KeyError) as error:
        raise TrainingDataError("invalid numeric NPZ input") from error
    if any(value.dtype.kind == "O" for value in result.values()):
        raise TrainingDataError("object/pickled arrays forbidden")
    if cache is not None:
        for value in result.values():
            value.flags.writeable = False
        cache.remove("bytes", reference)
        cache.put("npz", reference, result, sum(value.nbytes for value in result.values()))
    return result


def _receipt(reference: Mapping[str, Any], schema: str, *, allow_synthetic_fixture: bool = False,
             cache: VerifiedInputCache | None = None,
             **required: Any) -> dict[str, Any]:
    receipt = _object(reference_bytes(reference, cache=cache))
    if receipt.get("fixture_only") is True and not allow_synthetic_fixture:
        raise TrainingDataError("synthetic fixture receipt is not production evidence")
    if receipt.get("schema") != schema or receipt.get("status") != "PASSED":
        raise TrainingDataError(f"missing PASSED receipt: {schema}")
    for key, value in required.items():
        if receipt.get(key) != value or (isinstance(value, bool) and receipt.get(key) is not value):
            raise TrainingDataError(f"receipt qualification missing: {schema}/{key}")
    evidence = receipt.get("evidence_refs")
    if not isinstance(evidence, list) or not evidence:
        raise TrainingDataError(f"receipt requires underlying evidence refs: {schema}")
    for ref in evidence:
        reference_bytes(ref, cache=cache)
    return receipt


def _shape(value: np.ndarray, expected: tuple[int, ...], name: str, kind: str | None = None):
    if value.shape != expected:
        raise TrainingDataError(f"{name} shape {value.shape}, expected {expected}")
    if kind == "bool" and value.dtype.kind != "b":
        raise TrainingDataError(f"{name} must be a boolean mask")
    if kind == "int" and value.dtype.kind not in "iu":
        raise TrainingDataError(f"{name} must be an integer array")
    if kind == "float" and value.dtype.kind != "f":
        raise TrainingDataError(f"{name} must be a floating-point array")


def _proper_se3(values: np.ndarray, name: str):
    if not np.isfinite(values).all() or not np.allclose(values[..., 3, :], [0, 0, 0, 1], atol=1e-5):
        raise TrainingDataError(f"invalid {name} SE3")
    rotations = values[..., :3, :3]
    if not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise TrainingDataError(f"improper {name} rotation")


def _pose9(transforms: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # Match the existing HumanEgo rotmat_to_o6d row-major first-two-columns API.
    return np.concatenate(((transforms[..., :3, 3] - mean) / std,
                           transforms[..., :3, :2].reshape(*transforms.shape[:-2], 6)), axis=-1)


def _pose_clock_declaration(receipt: Mapping[str, Any]) -> tuple[Any, ...]:
    """Bind numerical poses/QPC to declared physical units, frames and clock.

    pose_frame_id names the source/world frame of wrist_pose_world;
    camera_frame_id names the reference camera frame. tcp_frame_ids is [L, R].
    These are reviewed evidence claims, never inferred from NPZ/directory names.
    """
    if receipt.get("position_unit") != "meters":
        raise TrainingDataError("pose receipt position_unit must be meters")
    if receipt.get("timestamp_unit") != "nanoseconds":
        raise TrainingDataError("pose receipt timestamp_unit must be nanoseconds")
    for field in ("pose_frame_id", "camera_frame_id", "clock_domain"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise TrainingDataError(f"pose receipt requires nonempty {field}")
    tcp_frames = receipt.get("tcp_frame_ids")
    if (not isinstance(tcp_frames, list) or len(tcp_frames) != 2
            or any(not isinstance(frame, str) or not frame.strip() for frame in tcp_frames)):
        raise TrainingDataError("pose receipt requires ordered left/right tcp_frame_ids")
    return (receipt["position_unit"], receipt["pose_frame_id"], receipt["camera_frame_id"],
            tuple(tcp_frames), receipt["timestamp_unit"], receipt["clock_domain"])


def _checked_calibration(manifest: Mapping[str, Any], *, allow_synthetic_fixture: bool,
                         cache: VerifiedInputCache | None) -> dict[str, Any]:
    receipt = _receipt(manifest["calibration_receipt"], "robot-camera-calibration-v1",
                       allow_synthetic_fixture=allow_synthetic_fixture, cache=cache,
                       geometry_qualified=True, physical_eye_identity_verified=True)
    _pose_clock_declaration(receipt)
    return receipt


def _check_rgb_causal(sample: dict, branch: str, labels: Mapping[str, np.ndarray],
                      clock_domain: str, *, allow_synthetic_fixture: bool,
                      cache: VerifiedInputCache | None):
    receipt = _receipt(
        sample["causal_receipts"][branch], "causal-training-input-v1",
        allow_synthetic_fixture=allow_synthetic_fixture, cache=cache,
        input_class="CAUSAL_TRAINING_INPUT", sample_id=sample["sample_id"],
        source_frame_id=sample["source_frame_id"], branch=branch,
        image_sha256=sample["rgb"][branch]["sha256"],
        labels_sha256=sample["labels"]["sha256"],
        anchor_qpc_ns=int(labels["anchor_qpc_ns"]),
        timestamp_unit="nanoseconds", clock_domain=clock_domain,
    )
    anchor = receipt.get("anchor_qpc_ns")
    if not isinstance(anchor, int) or isinstance(anchor, bool):
        raise TrainingDataError("RGB causal anchor_qpc_ns must be an integer nanosecond timestamp")
    donors = receipt.get("donor_frame_ids")
    if (not isinstance(donors, list) or not donors
            or any(not isinstance(donor, int) or isinstance(donor, bool)
                   or donor < 0 or donor > sample["source_frame_id"] for donor in donors)):
        raise TrainingDataError("RGB donor must be present or past source frame")


def _check_sample(sample: dict, normalization: dict, calibration: Mapping[str, Any],
                  *, allow_synthetic_fixture: bool = False,
                  cache: VerifiedInputCache | None = None) -> dict[str, np.ndarray]:
    labels = _npz(sample["labels"], cache=cache)
    if not LABEL_KEYS.issubset(labels):
        raise TrainingDataError(f"missing label arrays: {sorted(LABEL_KEYS - set(labels))}")
    if any(key in labels for key in ("q_arm", "q_hand", "digital_q", "retarget_q")):
        raise TrainingDataError("digital retarget q is not real Revo2 supervision")
    _shape(labels["x_ict"], (8, 29), "x_ict", "float")
    _shape(labels["ict_mask"], (8,), "ict_mask", "bool")
    _shape(labels["y_action"], (50, 30), "y_action", "float")
    _shape(labels["action_valid_mask"], (50, 30), "action_valid_mask", "bool")
    _shape(labels["anchor_qpc_ns"], (), "anchor_qpc_ns", "int")
    _shape(labels["conditioning_source_qpc_ns"], (8,), "conditioning_source_qpc_ns", "int")
    for key in ("target_qpc_ns", "target_source_frame_ids", "target_robot_rows"):
        _shape(labels[key], (50,), key, "int")
    _shape(labels["current_robot_row"], (), "current_robot_row", "int")
    _shape(labels["reference_camera_to_world"], (4, 4), "reference_camera_to_world", "float")
    _proper_se3(labels["reference_camera_to_world"], "reference camera")
    ict_mask, action_mask = labels["ict_mask"], labels["action_valid_mask"]
    if not ict_mask.any() or not action_mask.any():
        raise TrainingDataError("no valid conditioning/action supervision")
    if not np.isfinite(labels["x_ict"]).all() or not np.isfinite(labels["y_action"][action_mask]).all():
        raise TrainingDataError("nonfinite admitted state or action")
    anchor = int(labels["anchor_qpc_ns"])
    if np.any(labels["conditioning_source_qpc_ns"][ict_mask] > anchor):
        raise TrainingDataError("future sensor conditioning is forbidden")
    if np.any(labels["conditioning_source_qpc_ns"][ict_mask] < 0):
        raise TrainingDataError("valid conditioning requires real source QPC")
    target_qpc = labels["target_qpc_ns"]
    if np.any(target_qpc <= anchor) or np.any(np.diff(target_qpc) <= 0):
        raise TrainingDataError("future label QPC must strictly follow the anchor")
    if not np.array_equal(labels["target_source_frame_ids"], np.arange(sample["source_frame_id"] + 1, sample["source_frame_id"] + 51)):
        raise TrainingDataError("H50 must be contiguous native source frames")
    sidecar_ref = sample["real_robot_action_sidecar"]
    receipt = _receipt(
        sample["real_robot_action_receipt"], "real-revo2-synchronized-actions-v1",
        allow_synthetic_fixture=allow_synthetic_fixture,
        cache=cache,
        target=TARGET, control_ground_truth=True, synchronized=True,
        real_robot_action=True, session_id=sample["session_id"],
        sidecar_sha256=sidecar_ref["sha256"], labels_sha256=sample["labels"]["sha256"],
    )
    if receipt.get("source_kind") not in {"REAL_ROBOT_TELEOP", "REAL_ROBOT_EXECUTION_LOG"}:
        raise TrainingDataError("control receipt must name actual robot acquisition")
    if _pose_clock_declaration(receipt) != _pose_clock_declaration(calibration):
        raise TrainingDataError("control/calibration pose frame, TCP, unit or clock_domain differs")
    channels = receipt.get("hand_command_channel_order")
    unit = receipt.get("hand_command_unit")
    if (not isinstance(channels, list) or len(channels) != 6
            or any(not isinstance(channel, str) or not channel for channel in channels)
            or len(set(channels)) != 6 or not isinstance(unit, str) or not unit):
        raise TrainingDataError("real Revo2 receipt must declare six ordered channels and their unit")
    sidecar = _npz(sidecar_ref, cache=cache)
    if not SIDECAR_KEYS.issubset(sidecar):
        raise TrainingDataError("synchronized real Revo2 sidecar fields missing")
    count = len(sidecar["qpc_ns"])
    for key, shape, kind in (
        ("wrist_pose_world", (count, 2, 4, 4), "float"),
        ("hand_command", (count, 2, 6), "float"),
        ("valid", (count, 2), "bool"),
        ("qpc_ns", (count,), "int"),
        ("source_frame_id", (count,), "int"),
        ("segment_id", (count,), "int"),
    ):
        _shape(sidecar[key], shape, key, kind)
    if np.any(np.diff(sidecar["qpc_ns"]) <= 0):
        raise TrainingDataError("real sidecar QPC is not strictly increasing")
    rows = labels["target_robot_rows"]
    current = int(labels["current_robot_row"])
    if current < 0 or current >= count or np.any(rows < 0) or np.any(rows >= count):
        raise TrainingDataError("robot sidecar row outside recording")
    if sidecar["qpc_ns"][current] > anchor:
        raise TrainingDataError("current real robot state comes from the future")
    if not np.array_equal(sidecar["qpc_ns"][rows], target_qpc) or not np.array_equal(sidecar["source_frame_id"][rows], labels["target_source_frame_ids"]):
        raise TrainingDataError("labels do not bind exact real robot sidecar frames")
    if np.any(sidecar["segment_id"][rows] != sidecar["segment_id"][current]):
        raise TrainingDataError("H50 crosses a recording segment")
    wrist_valid = sidecar["valid"][rows]
    admissible = np.concatenate((np.repeat(wrist_valid, 3, axis=1),
                                np.repeat(wrist_valid, 6, axis=1),
                                np.repeat(wrist_valid, 6, axis=1)), axis=1)
    if np.any(action_mask & ~admissible):
        raise TrainingDataError("unknown robot actions marked as valid")
    transforms = sidecar["wrist_pose_world"][rows].copy()
    transforms[~wrist_valid] = np.eye(4)
    _proper_se3(transforms[wrist_valid], "real wrist")
    camera_from_world = np.linalg.inv(labels["reference_camera_to_world"])
    pose9 = _pose9(camera_from_world @ transforms,
                   np.asarray(normalization["position_mean"]),
                   np.asarray(normalization["position_std"]))
    commands = sidecar["hand_command"][rows]
    expected = np.concatenate((pose9[..., :3].reshape(50, 6),
                               pose9[..., 3:].reshape(50, 12),
                               commands.reshape(50, 12)), axis=1)
    if not np.isfinite(expected[action_mask]).all() or not np.allclose(labels["y_action"][action_mask], expected[action_mask], atol=1e-5, rtol=1e-5):
        raise TrainingDataError("target does not match measured Revo2 pose/command")
    # Object states are absent in this minimal hand-only ICT contract. The last
    # feature is UNKNOWN grasp (-1), not an invented binary grasp observation.
    if ict_mask[2:].any() or np.any(labels["x_ict"][~ict_mask] != 0):
        raise TrainingDataError("hand-only contract cannot admit fabricated object tokens")
    current_valid = sidecar["valid"][current]
    if np.any(ict_mask[:2] & ~current_valid):
        raise TrainingDataError("unobserved current wrist admitted as ICT")
    current_transforms = sidecar["wrist_pose_world"][current].copy()
    current_transforms[~current_valid] = np.eye(4)
    _proper_se3(current_transforms[current_valid], "current wrist")
    mean, std = np.asarray(normalization["position_mean"]), np.asarray(normalization["position_std"])
    for index in range(2):
        if not ict_mask[index]:
            continue
        transform = current_transforms[index]
        relations = _pose9(np.linalg.inv(transform) @ current_transforms, mean, std)
        relations[~current_valid] = 0
        token = np.concatenate(([index + 1], _pose9(camera_from_world @ transform, mean, std), relations.reshape(18), [-1]))
        if not np.allclose(labels["x_ict"][index], token, atol=1e-5, rtol=1e-5):
            raise TrainingDataError("current ICT differs from real observed wrist state")
        if labels["conditioning_source_qpc_ns"][index] != sidecar["qpc_ns"][current]:
            raise TrainingDataError("ICT source QPC does not bind current robot row")
    return labels


def validate_training_manifest(manifest_path: str | Path, *, allow_synthetic_fixture: bool = False,
                               _input_cache: VerifiedInputCache | None = None) -> dict[str, Any]:
    """Validate all branches, evidence and measured controls before admission."""
    path = _ordinary(Path(manifest_path).absolute())
    encoded = path.read_bytes()
    manifest = _object(encoded)
    cache = _input_cache if _input_cache is not None else VerifiedInputCache()
    if manifest.get("fixture_only") is True and not allow_synthetic_fixture:
        raise TrainingDataError("synthetic fixture manifest is not production training data")
    if manifest.get("schema") != SCHEMA or manifest.get("target") != TARGET:
        raise TrainingDataError("reviewed paired real Revo2 manifest required")
    if not isinstance(manifest.get("pair_id"), str) or not manifest["pair_id"]:
        raise TrainingDataError("nonempty pair_id required")
    if manifest.get("control_ground_truth") is not True or manifest.get("physical_deployment_authorized") is not False:
        raise TrainingDataError("real-control data required; physical deployment remains unauthorized")
    if manifest.get("action_layout") != ACTION_LAYOUT or manifest.get("ict_schema") != "HAND_ONLY_29D_GRASP_UNKNOWN_MINUS_ONE":
        raise TrainingDataError("explicit 30D action and hand-only ICT schemas required")
    if "calibration_receipt" not in manifest:
        raise TrainingDataError("reviewed calibration receipt missing")
    normalization = manifest.get("normalization", {})
    if normalization.get("method") != "IDENTITY_NO_FIT":
        raise TrainingDataError("candidate supports only explicit IDENTITY_NO_FIT normalization")
    for key in ("position_mean", "position_std"):
        value = np.asarray(normalization.get(key, []), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or (key == "position_std" and np.any(value <= 0)):
            raise TrainingDataError("finite explicit training position normalization required")
        expected = np.zeros(3) if key == "position_mean" else np.ones(3)
        if not np.array_equal(value, expected):
            raise TrainingDataError("IDENTITY_NO_FIT requires position mean zero and std one")
    calibration = _checked_calibration(manifest, allow_synthetic_fixture=allow_synthetic_fixture,
                                       cache=cache)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise TrainingDataError("manifest has no reviewed samples")
    identities, frame_identities, group_splits, session_groups = set(), set(), {}, {}
    split_counts = {"train": 0, "validation": 0}
    action_counts = {split: np.zeros(30, dtype=np.int64) for split in split_counts}
    control_declaration = None
    for sample in samples:
        if not isinstance(sample, dict):
            raise TrainingDataError("sample must be an object")
        required_fields = {"labels", "real_robot_action_sidecar", "real_robot_action_receipt",
                           "source_group_receipt", "rgb", "causal_receipts"}
        if not required_fields.issubset(sample):
            raise TrainingDataError(f"missing reviewed sample fields: {sorted(required_fields - set(sample))}")
        weight = sample.get("sample_weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or not 0 < weight <= 1:
            raise TrainingDataError("sample_weight must be finite in (0,1]")
        for key in ("sample_id", "session_id", "source_group_id"):
            if not isinstance(sample.get(key), str) or not sample[key] or sample[key] == "UNKNOWN_SOURCE_GROUP":
                raise TrainingDataError(f"missing sample identity: {key}")
        frame = sample.get("source_frame_id")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise TrainingDataError("source_frame_id must be a native integer frame")
        identity = (sample["session_id"], frame)
        if sample["sample_id"] in identities or identity in frame_identities:
            raise TrainingDataError("duplicate sample or native frame identity")
        identities.add(sample["sample_id"])
        frame_identities.add(identity)
        split = sample.get("split")
        if split not in split_counts:
            raise TrainingDataError("split must be train or validation")
        split_counts[split] += 1
        group = sample["source_group_id"]
        if group in group_splits and group_splits[group] != split:
            raise TrainingDataError("independent source group crosses train/validation")
        group_splits[group] = split
        session_id = sample["session_id"]
        if session_id in session_groups and session_groups[session_id] != group:
            raise TrainingDataError("session cannot claim multiple acquisition source groups")
        session_groups[session_id] = group
        _receipt(sample["source_group_receipt"], "independent-acquisition-source-group-v1",
                 allow_synthetic_fixture=allow_synthetic_fixture,
                 cache=cache,
                 source_group_id=group, session_id=session_id, independent_acquisition=True)
        if set(sample.get("rgb", {})) != {"raw", "robotized"} or set(sample.get("causal_receipts", {})) != {"raw", "robotized"}:
            raise TrainingDataError("both paired RGB branches and causal receipts required")
        labels = _check_sample(sample, normalization, calibration,
                               allow_synthetic_fixture=allow_synthetic_fixture, cache=cache)
        for branch in ("raw", "robotized"):
            reference_bytes(sample["rgb"][branch])
            _check_rgb_causal(sample, branch, labels, calibration["clock_domain"],
                              allow_synthetic_fixture=allow_synthetic_fixture, cache=cache)
        control_receipt = _object(reference_bytes(sample["real_robot_action_receipt"], cache=cache))
        declaration = (tuple(control_receipt["hand_command_channel_order"]),
                       control_receipt["hand_command_unit"])
        if control_declaration is not None and declaration != control_declaration:
            raise TrainingDataError("hand channel order/unit differs across paired manifest sessions")
        control_declaration = declaration
        action_counts[split] += labels["action_valid_mask"].sum(axis=0)
    if not all(split_counts.values()):
        raise TrainingDataError("both isolated train and validation samples required")
    for split, counts in action_counts.items():
        missing = np.flatnonzero(counts == 0).tolist()
        if missing:
            raise TrainingDataError(f"{split} output dimensions have no effective supervision: {missing}")
    return {"manifest": manifest, "dataset_sha": hashlib.sha256(encoded).hexdigest(),
            "sample_counts": split_counts, "source_group_splits": group_splits,
            "valid_action_counts": {split: value.tolist() for split, value in action_counts.items()},
            "hand_command_channel_order": list(control_declaration[0]),
            "hand_command_unit": control_declaration[1]}


class Revo2ICTDataset:
    """Deterministic samples from a fully verified manifest; no augmentation."""

    def __init__(self, manifest_path: str | Path, branch: str, split: str,
                 image_size: tuple[int, int] = (240, 320), horizon: int = 50,
                 *, allow_synthetic_fixture: bool = False):
        if branch not in BRANCHES or split not in {"train", "validation"}:
            raise TrainingDataError("unknown branch or split")
        if tuple(image_size) != (240, 320) or horizon != 50:
            raise TrainingDataError("candidate contract fixes RGB 240x320 and H50")
        self.input_cache = VerifiedInputCache()
        admitted = validate_training_manifest(manifest_path, allow_synthetic_fixture=allow_synthetic_fixture,
                                               _input_cache=self.input_cache)
        self.manifest = admitted["manifest"]
        self.manifest_path = Path(manifest_path).absolute()
        self.manifest_metadata = VerifiedInputCache._stat(self.manifest_path)
        self.dataset_sha = admitted["dataset_sha"]
        self.branch = BRANCHES[branch]
        self.split = split
        self.samples = [sample for sample in self.manifest["samples"] if sample["split"] == split]
        self.image_size, self.horizon = image_size, horizon
        self.allow_synthetic_fixture = allow_synthetic_fixture

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from PIL import Image

        if VerifiedInputCache._stat(self.manifest_path) != self.manifest_metadata:
            raise TrainingDataError("frozen manifest metadata/SHA authority changed")
        sample = self.samples[index]
        calibration = _checked_calibration(self.manifest,
                                           allow_synthetic_fixture=self.allow_synthetic_fixture,
                                           cache=self.input_cache)
        labels = _check_sample(sample, self.manifest["normalization"], calibration,
                               allow_synthetic_fixture=self.allow_synthetic_fixture,
                               cache=self.input_cache)
        _check_rgb_causal(sample, self.branch, labels, calibration["clock_domain"],
                          allow_synthetic_fixture=self.allow_synthetic_fixture, cache=self.input_cache)
        encoded = reference_bytes(sample["rgb"][self.branch])
        try:
            with Image.open(io.BytesIO(encoded)) as source:
                rgb = np.asarray(source.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        except (OSError, ValueError) as error:
            raise TrainingDataError("reviewed RGB input cannot be decoded") from error
        mask = labels["action_valid_mask"]
        # Missing payloads are sanitized before flow noise/target construction.
        action = np.where(mask, labels["y_action"], 0).astype(np.float32)
        weight = sample.get("sample_weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or not 0 < weight <= 1:
            raise TrainingDataError("sample_weight must be finite in (0,1]")
        return {
            "x_rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
            "x_ict": torch.from_numpy(labels["x_ict"].astype(np.float32)),
            "ict_mask": torch.from_numpy(labels["ict_mask"].copy()),
            "y_action": torch.from_numpy(action),
            "action_valid_mask": torch.from_numpy(mask.copy()),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "sample_id": sample["sample_id"], "session_id": sample["session_id"],
            "source_frame_id": sample["source_frame_id"],
            "source_group_id": sample["source_group_id"], "split": sample["split"],
            "branch": self.branch, "sample_index": index,
        }
