"""CPU-only Tianji/KaiHand scene-state producer.

The producer consumes an explicit, independently supplied, session-constant
``T_tool_hand`` for each side.  It never estimates, selects, or adjusts a
mount from wrist/IK residuals.  Frozen R2 supplies KaiHand joint states and
hand-root targets; frozen HaWoR supplies the validity intersection and camera
intrinsics.  Tianji arm joint states and one session-constant camera-to-base
transform either come from a strict independent JSON+NPZ authority or, when
that authority is absent, from the existing development-only CPU IK solver.

No renderer, GPU API, collision geometry, pixel producer, or formal training
consumer is reachable from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline.robot_mount_visual_proxy import (
    CONTACT_INFEASIBLE,
    MOUNT_PROVENANCE,
    validate_se3,
    validate_session_constant_mounts,
)
from pipeline.robot_renderer_cycles import forward_kinematics
from pipeline.robot_renderer_eevee_fullchain import (
    ARM_JOINT_NAMES,
    SCENE_SCHEMA,
    SCENE_MANIFEST_DEVELOPMENT_STATUS,
    SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS,
    SCENE_STATE_MODE_DEVELOPMENT_SOLVER,
    SCENE_STATE_MODE_EXTERNAL_AUTHORITY,
    SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER,
    FIXED_CAMERA_BASE_CANDIDATE_SCHEMA,
    FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA,
    SIDES,
    FullChainSceneState,
    PinnedRobotAssets,
    decode_scene_state,
    validate_scene_manifest,
)


MOUNT_SCHEMA = "robot-explicit-visual-mount-input-v1"
MOUNT_SIDE_EVIDENCE_SCHEMA = "robot-mount-side-evidence-v1"
EXTERNAL_BASE_Q_SCHEMA = "robot-explicit-base-q-authority-v1"
EXTERNAL_BASE_Q_ARRAY_SCHEMA = "robot-explicit-base-q-authority-arrays-v1"
EXTERNAL_BASE_EVIDENCE_SCHEMA = "robot-external-camera-base-evidence-v1"
EXTERNAL_Q_ARM_SIDE_EVIDENCE_SCHEMA = "robot-external-q-arm-side-evidence-v1"
MANIFEST_SCHEMA = "robot-fullchain-scene-manifest-v1"
SOLVER_SCHEMA = "robot-scene-state-cpu-solver-v1"
MOUNT_COORDINATE_DEFINITION = (
    "Tianji left/right tool link to corresponding KaiHand root link"
)
MOUNT_MATRIX_DIRECTION = "p_target_link = T_tool_hand @ p_source_link"
MOUNT_SOURCE_LINKS = {
    "left": "hand_l_base_link",
    "right": "hand_r_base_link",
}
MOUNT_TARGET_LINKS = {"left": "left_tool", "right": "right_tool"}
RIGHT_HANDED_AXES = "RIGHT_HANDED_XYZ"
BASE_TRANSFORM_SEMANTICS = "p_camera = T_camera_base @ p_base"
MOUNT_TIME_SCOPE = "SESSION_CONSTANT_FULL_SESSION"
EXTERNAL_AUTHORITY_EVIDENCE_KIND = "DIRECT_EXTERNAL_BASE_AND_ARM_STATE_MEASUREMENT"
EXTERNAL_AUTHORITY_METHOD = "DIRECT_CAMERA_BASE_CALIBRATION_AND_JOINT_ENCODER_STATE"
EXTERNAL_FRAME_TIMESTAMP_MAPPING = (
    "ONE_TIMESTAMP_PER_ZERO_BASED_SOURCE_FRAME_NO_INTERPOLATION"
)
EXTERNAL_BASE_EVIDENCE_KIND = "DIRECT_CAMERA_TO_ROBOT_BASE_CALIBRATION"
EXTERNAL_Q_ARM_EVIDENCE_KIND = "DIRECT_ARM_JOINT_ENCODER_STATE"
EXTERNAL_ARRAY_DIGEST_CANONICALIZATION = (
    "NUMPY_DTYPE_STR_NUL_SHAPE_JSON_NUL_C_ORDER_BYTES_V1"
)
SESSION_ID_PATTERN = re.compile(r"grap_a_cap_[0-9]{3}")
POSITION_SCALE_M = 0.005
ROTATION_SCALE_RAD = np.deg2rad(2.0)
MAX_POSITION_RESIDUAL_MM = 10.0
MAX_ROTATION_RESIDUAL_DEG = 5.0
OUTER_ITERATIONS = 6
IK_MAX_EVALUATIONS = 300
IK_TRF_FTOL = 1e-10
IK_TRF_XTOL = 1e-10
IK_TRF_GTOL = 1e-10
FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS = IK_MAX_EVALUATIONS
FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS = IK_MAX_EVALUATIONS


class SceneStateProducerError(RuntimeError):
    """One input, IK, or artifact invariant failed closed."""


@dataclass(frozen=True)
class ExplicitMountInput:
    session_id: str
    matrices: np.ndarray
    development_only: bool
    authority: Mapping[str, Any]
    evidence_records: Mapping[str, Mapping[str, Any]]
    evidence_mode: str
    descriptor_sha256: str


@dataclass(frozen=True)
class FixedCameraBaseInput:
    matrix: np.ndarray
    descriptor_record: Mapping[str, Any]
    contract: Mapping[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    frame_indices: tuple[int, ...]
    source_frame_count: int
    full_valid: np.ndarray
    frame_names: tuple[str, ...]
    q_hand: np.ndarray
    wrist_T_camera: np.ndarray
    valid: np.ndarray
    camera_intrinsics: np.ndarray
    source_resolution: tuple[int, int]
    hand_joint_names: np.ndarray
    records: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class SolverResult:
    q_arm: np.ndarray
    T_camera_base: np.ndarray
    position_residual_mm: np.ndarray
    rotation_residual_deg: np.ndarray
    evaluations: np.ndarray
    scene_state_mode: str = SCENE_STATE_MODE_DEVELOPMENT_SOLVER
    solver_used: bool = True
    residual_claimed: bool = True
    development_only: bool = True
    timestamp_ns: np.ndarray | None = None
    external_authority: Mapping[str, Any] | None = None
    fixed_camera_base_candidate: Mapping[str, Any] | None = None
    position_stage_evaluations: np.ndarray | None = None
    full_pose_stage_evaluations: np.ndarray | None = None


def _forbid_path(path: Path) -> None:
    lowered = str(path).lower()
    parts = {part.lower() for part in path.parts}
    if "grap_a_cap_025" in lowered or "blind" in lowered:
        raise SceneStateProducerError("blind/025 input is forbidden")
    if "labels" in parts or "processed" in parts:
        raise SceneStateProducerError("labels/processed input is forbidden")


def _require_session_path(path: Path, *, session_id: str, name: str) -> None:
    """Reject accidental cross-session binding before any source bytes are read."""

    session_components = tuple(
        part for part in path.parts if SESSION_ID_PATTERN.fullmatch(part)
    )
    if session_components != (session_id,):
        raise SceneStateProducerError(
            f"{name} path must contain exactly the requested session component"
        )


def _preflight_single_link_file(path: Path, *, name: str) -> os.stat_result:
    """Reject final-component symlinks and hard-link aliases before opening bytes."""

    try:
        info = path.lstat()
    except OSError as exc:
        raise SceneStateProducerError(f"cannot stat {name}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise SceneStateProducerError(f"{name} must be an ordinary non-symlink file")
    if info.st_nlink != 1:
        raise SceneStateProducerError(f"{name} hard-link aliases are forbidden")
    return info


def read_bound_bytes(strict: Any, path: Path, *, name: str) -> Any:
    """Read/hash through one FD and verify that its lexical name stayed bound."""

    _forbid_path(path)
    before = _preflight_single_link_file(path, name=name)
    record = strict.read_bytes_nofollow(path)
    try:
        after = path.lstat()
    except OSError as exc:
        raise SceneStateProducerError(f"cannot restat {name}: {exc}") from exc
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or int(before.st_dev) != int(record.device)
        or int(before.st_ino) != int(record.inode)
        or int(before.st_size) != int(record.bytes)
        or int(before.st_mtime_ns) != int(after.st_mtime_ns)
        or int(before.st_ctime_ns) != int(after.st_ctime_ns)
        or int(after.st_dev) != int(record.device)
        or int(after.st_ino) != int(record.inode)
        or int(after.st_size) != int(record.bytes)
    ):
        raise SceneStateProducerError(f"{name} path identity changed during read")
    return record


def _parse_json(payload: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneStateProducerError(f"invalid {name} JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SceneStateProducerError(f"{name} JSON root must be an object")
    return value


def _parse_npz(payload: bytes, *, name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise SceneStateProducerError(f"invalid {name} NPZ: {exc}") from exc


EVIDENCE_REF_KEYS = frozenset({"path", "bytes", "sha256", "device", "inode"})
MOUNT_AUTHORITY_KEYS = frozenset(
    {
        "provider",
        "method",
        "coordinate_definition",
        "matrix_direction",
        "axes_convention",
        "translation_units",
        "rotation_units",
        "source_links",
        "target_links",
        "independent_visual_evidence",
        "synthetic_fixture",
        "evidence_by_side",
    }
)
MOUNT_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "session_id",
        "mount_provenance",
        "contact_infeasible",
        "session_constant",
        "per_frame_mount_forbidden",
        "independent_of_r2_wrist_targets",
        "selected_by_ik_residual",
        "formal_consumer_allowed",
        "development_only",
        "authority",
        "T_tool_hand",
    }
)
FIXED_CAMERA_BASE_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "T_camera_base_semantics",
        "axes_convention",
        "translation_units",
        "fixed_across_selected_frames",
        "global_across_sessions",
        "selected_on_session",
        "not_per_session_tuned",
        "per_frame_override_forbidden",
        "per_session_override_forbidden",
        "development_only",
        "candidate_requires_human_review",
        "next_bucket_blocked",
        "advancement_authorized",
        "formal_consumer_allowed",
        "T_camera_base",
    }
)


def _record(record: Any) -> dict[str, Any]:
    def value(name: str) -> Any:
        return record[name] if isinstance(record, Mapping) else getattr(record, name)

    return {
        "path": str(value("path")),
        "bytes": int(value("bytes")),
        "sha256": str(value("sha256")),
        "device": int(value("device")),
        "inode": int(value("inode")),
    }


def _require_canonical_record_path(record: Any, *, name: str) -> Path:
    path = Path(_record(record)["path"])
    if not path.is_absolute() or str(path) != _record(record)["path"]:
        raise SceneStateProducerError(f"{name} path must be absolute and canonical")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SceneStateProducerError(f"cannot resolve {name}: {exc}") from exc
    if path != resolved:
        raise SceneStateProducerError(f"{name} path must be absolute and canonical")
    return path


def _canonical_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mount_authority_manifest(mount: ExplicitMountInput) -> dict[str, Any]:
    return {
        **dict(mount.authority),
        "session_constant": True,
        "per_frame_mount_forbidden": True,
        "independent_of_r2_wrist_targets": True,
        "selected_by_ik_residual": False,
        "evidence_mode": mount.evidence_mode,
        "independent_visual_evidence": bool(mount.evidence_records),
        "synthetic_fixture": not bool(mount.evidence_records),
        "evidence_by_side": {
            side: dict(mount.evidence_records[side])
            for side in SIDES
            if side in mount.evidence_records
        },
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneStateProducerError(f"{name} must be a non-empty string")
    return value


def _validate_evidence_ref(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != EVIDENCE_REF_KEYS:
        raise SceneStateProducerError(f"{name} must be one exact evidence ref")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise SceneStateProducerError(f"{name} path must be non-empty")
    path = Path(path_value)
    if not path.is_absolute() or path_value != str(path):
        raise SceneStateProducerError(f"{name} path must be absolute and canonical")
    if (
        type(value.get("bytes")) is not int
        or int(value["bytes"]) <= 0
        or type(value.get("device")) is not int
        or int(value["device"]) < 0
        or type(value.get("inode")) is not int
        or int(value["inode"]) <= 0
        or not _is_sha256(value.get("sha256"))
    ):
        raise SceneStateProducerError(f"{name} has invalid bytes/SHA/device/inode")
    return {
        "path": path_value,
        "bytes": int(value["bytes"]),
        "sha256": str(value["sha256"]),
        "device": int(value["device"]),
        "inode": int(value["inode"]),
    }


def _read_declared_evidence(strict: Any, value: Any, *, name: str) -> Any:
    declared = _validate_evidence_ref(value, name=name)
    path = Path(declared["path"])
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SceneStateProducerError(f"cannot resolve {name}: {exc}") from exc
    if path != resolved:
        raise SceneStateProducerError(f"{name} path must be absolute and canonical")
    record = read_bound_bytes(strict, path, name=name)
    if _record(record) != declared:
        raise SceneStateProducerError(
            f"{name} declared bytes/SHA/device/inode do not match same-FD read"
        )
    return record


def _require_record_ref(value: Any, record: Any, *, name: str) -> dict[str, Any]:
    declared = _validate_evidence_ref(value, name=name)
    if declared != _record(record):
        raise SceneStateProducerError(f"{name} does not match the pinned input record")
    return declared


def _mount_link_contract(authority: Mapping[str, Any]) -> None:
    if authority.get("coordinate_definition") != MOUNT_COORDINATE_DEFINITION:
        raise SceneStateProducerError(
            "mount coordinate_definition must bind Tianji tool to KaiHand root"
        )
    fixed = {
        "matrix_direction": MOUNT_MATRIX_DIRECTION,
        "axes_convention": RIGHT_HANDED_AXES,
        "translation_units": "metres",
        "rotation_units": "radians",
    }
    for field, expected in fixed.items():
        if authority.get(field) != expected:
            raise SceneStateProducerError(f"mount authority {field} must be {expected}")
    if authority.get("source_links") != MOUNT_SOURCE_LINKS:
        raise SceneStateProducerError("mount source links must be exact KaiHand roots")
    if authority.get("target_links") != MOUNT_TARGET_LINKS:
        raise SceneStateProducerError("mount target links must be exact Tianji tools")


def _validate_mount_side_evidence(
    payload: bytes,
    *,
    side: str,
    session_id: str,
    expected_matrix: np.ndarray,
) -> Mapping[str, Any]:
    value = _parse_json(payload, name=f"{side} mount evidence")
    required_keys = {
        "schema_version",
        "session_id",
        "side",
        "evidence_kind",
        "provider",
        "method",
        "source_link",
        "target_link",
        "matrix_direction",
        "axes_convention",
        "translation_units",
        "rotation_units",
        "timestamp_ns",
        "timestamp_clock_id",
        "time_scope",
        "device_id",
        "calibration_id",
        "calibration_sha256",
        "independent_of_r2_wrist_targets",
        "derived_from_r2_wrist_targets",
        "derived_from_ik_solver",
        "selected_by_ik_residual",
        "circular_selection",
        "synthetic_fixture",
        "identity_fixture",
        "formal_consumer_allowed",
        "T_tool_hand",
    }
    if set(value) != required_keys:
        raise SceneStateProducerError(f"{side} mount evidence keys are not exact")
    exact = {
        "schema_version": MOUNT_SIDE_EVIDENCE_SCHEMA,
        "session_id": session_id,
        "side": side,
        "evidence_kind": "DIRECT_VISUAL_MOUNT_MEASUREMENT",
        "method": "DIRECT_VISUAL_MOUNT_MEASUREMENT",
        "source_link": MOUNT_SOURCE_LINKS[side],
        "target_link": MOUNT_TARGET_LINKS[side],
        "matrix_direction": MOUNT_MATRIX_DIRECTION,
        "axes_convention": RIGHT_HANDED_AXES,
        "translation_units": "metres",
        "rotation_units": "radians",
        "time_scope": MOUNT_TIME_SCOPE,
        "independent_of_r2_wrist_targets": True,
        "derived_from_r2_wrist_targets": False,
        "derived_from_ik_solver": False,
        "selected_by_ik_residual": False,
        "circular_selection": False,
        "synthetic_fixture": False,
        "identity_fixture": False,
        "formal_consumer_allowed": False,
    }
    for field, expected in exact.items():
        if value.get(field) != expected:
            raise SceneStateProducerError(
                f"{side} mount evidence {field} must be {expected}"
            )
    if type(value.get("timestamp_ns")) is not int or int(value["timestamp_ns"]) < 0:
        raise SceneStateProducerError(f"{side} mount timestamp_ns must be non-negative")
    for field in ("timestamp_clock_id", "device_id", "calibration_id"):
        _require_nonempty_string(value.get(field), name=f"{side} mount {field}")
    _require_nonempty_string(value.get("provider"), name=f"{side} mount provider")
    if not _is_sha256(value.get("calibration_sha256")):
        raise SceneStateProducerError(
            f"{side} mount calibration_sha256 must be exact lowercase SHA-256"
        )
    try:
        observed = validate_session_constant_mounts(
            {
                side: np.asarray(value["T_tool_hand"], dtype=np.float64),
                ("right" if side == "left" else "left"): np.eye(4),
            }
        )[side]
    except (ValueError, TypeError, RuntimeError) as exc:
        raise SceneStateProducerError(f"invalid {side} evidence matrix: {exc}") from exc
    if not np.array_equal(observed, expected_matrix):
        raise SceneStateProducerError(
            f"{side} evidence matrix does not equal the mount descriptor"
        )
    return value


def load_fixed_camera_base_candidate(
    payload: bytes,
    *,
    source_record: Any,
) -> FixedCameraBaseInput:
    """Load one globally reusable development-only fixed-base candidate.

    The descriptor is deliberately not an external calibration authority.  Its
    exact same-FD record is retained so a scene state cannot silently substitute
    another matrix or a per-session/per-frame override.
    """

    record = _validate_evidence_ref(
        _record(source_record), name="fixed camera-base candidate descriptor"
    )
    if len(payload) != int(record["bytes"]) or sha256_bytes(payload) != str(
        record["sha256"]
    ):
        raise SceneStateProducerError(
            "fixed camera-base payload does not match its same-FD descriptor record"
        )
    _require_canonical_record_path(
        source_record, name="fixed camera-base candidate descriptor"
    )
    value = _parse_json(payload, name="fixed camera-base candidate")
    if set(value) != FIXED_CAMERA_BASE_ROOT_KEYS:
        raise SceneStateProducerError(
            "fixed camera-base candidate descriptor keys are not exact"
        )
    expected = {
        "schema_version": FIXED_CAMERA_BASE_CANDIDATE_SCHEMA,
        "T_camera_base_semantics": BASE_TRANSFORM_SEMANTICS,
        "axes_convention": RIGHT_HANDED_AXES,
        "translation_units": "metres",
        "fixed_across_selected_frames": True,
        "global_across_sessions": True,
        "selected_on_session": "grap_a_cap_004",
        "not_per_session_tuned": True,
        "per_frame_override_forbidden": True,
        "per_session_override_forbidden": True,
        "development_only": True,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
    }
    if any(
        type(value.get(field)) is not type(required) or value.get(field) != required
        for field, required in expected.items()
    ):
        raise SceneStateProducerError(
            "fixed camera-base candidate must remain global, development-only, and blocked"
        )
    encoded_matrix = value.get("T_camera_base")
    if (
        not isinstance(encoded_matrix, list)
        or len(encoded_matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in encoded_matrix)
        or any(type(item) not in {int, float} for row in encoded_matrix for item in row)
    ):
        raise SceneStateProducerError(
            "fixed T_camera_base must contain exact numeric JSON[4,4]"
        )
    try:
        raw_matrix = np.asarray(encoded_matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneStateProducerError(
            "fixed T_camera_base must decode as exact float64[4,4]"
        ) from exc
    if raw_matrix.dtype != np.float64 or raw_matrix.shape != (4, 4):
        raise SceneStateProducerError(
            "fixed T_camera_base must decode as exact float64[4,4]"
        )
    try:
        matrix = validate_se3(raw_matrix, name="fixed T_camera_base")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise SceneStateProducerError(
            f"fixed T_camera_base must be right-handed SE(3): {exc}"
        ) from exc
    contract = {**expected, "T_camera_base": matrix.tolist()}
    return FixedCameraBaseInput(
        matrix=np.asarray(matrix, dtype=np.float64),
        descriptor_record=record,
        contract=contract,
    )


def _fixed_camera_base_candidate_manifest(
    candidate: FixedCameraBaseInput,
) -> dict[str, Any]:
    return {**dict(candidate.contract), "descriptor": dict(candidate.descriptor_record)}


def load_explicit_mount(
    payload: bytes,
    *,
    expected_session_id: str,
    strict: Any | None = None,
    source_record: Any | None = None,
) -> ExplicitMountInput:
    _validate_session_id(expected_session_id)
    if source_record is not None and (
        len(payload) != int(_record(source_record)["bytes"])
        or sha256_bytes(payload) != str(_record(source_record)["sha256"])
    ):
        raise SceneStateProducerError(
            "mount payload does not match its same-FD descriptor record"
        )
    if source_record is not None:
        _require_canonical_record_path(source_record, name="mount descriptor")
    value = _parse_json(payload, name="mount")
    if set(value) != MOUNT_ROOT_KEYS:
        raise SceneStateProducerError("mount descriptor keys are not exact")
    required_flags = {
        "session_constant": True,
        "per_frame_mount_forbidden": True,
        "independent_of_r2_wrist_targets": True,
        "selected_by_ik_residual": False,
        "formal_consumer_allowed": False,
    }
    if value.get("schema_version") != MOUNT_SCHEMA:
        raise SceneStateProducerError("unsupported explicit mount schema")
    if value.get("session_id") != expected_session_id:
        raise SceneStateProducerError("mount session does not match requested session")
    if value.get("mount_provenance") != MOUNT_PROVENANCE:
        raise SceneStateProducerError("mount must remain provisional visual-only")
    if value.get("contact_infeasible") != CONTACT_INFEASIBLE:
        raise SceneStateProducerError("mount contact field must remain UNMEASURED")
    for field, expected in required_flags.items():
        if value.get(field) is not expected:
            raise SceneStateProducerError(
                f"mount contract field {field} must be {expected}"
            )
    if not isinstance(value.get("development_only"), bool):
        raise SceneStateProducerError("mount development_only must be explicit")
    authority = value.get("authority")
    if not isinstance(authority, Mapping):
        raise SceneStateProducerError("mount authority record is required")
    if set(authority) != MOUNT_AUTHORITY_KEYS:
        raise SceneStateProducerError("mount authority keys are not exact")
    for field in ("provider", "method", "coordinate_definition"):
        if (
            not isinstance(authority.get(field), str)
            or not str(authority[field]).strip()
        ):
            raise SceneStateProducerError(f"mount authority lacks {field}")
    _mount_link_contract(authority)
    for field in ("independent_visual_evidence", "synthetic_fixture"):
        if type(authority.get(field)) is not bool:
            raise SceneStateProducerError(f"mount authority {field} must be boolean")
    raw = value.get("T_tool_hand")
    if not isinstance(raw, Mapping) or set(raw) != set(SIDES):
        raise SceneStateProducerError(
            "mount must explicitly contain left and right matrices"
        )
    try:
        matrices_by_side = validate_session_constant_mounts(
            {side: np.asarray(raw[side], dtype=np.float64) for side in SIDES}
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        raise SceneStateProducerError(
            f"invalid explicit mount matrices: {exc}"
        ) from exc
    matrices = np.stack([matrices_by_side[side] for side in SIDES])
    development_only = bool(value["development_only"])
    independent = bool(authority["independent_visual_evidence"])
    synthetic = bool(authority["synthetic_fixture"])
    evidence_refs = authority.get("evidence_by_side")
    evidence_records: dict[str, Mapping[str, Any]] = {}
    if synthetic:
        if independent or not development_only:
            raise SceneStateProducerError(
                "synthetic/identity mount fixtures must be development_only and non-independent"
            )
        if evidence_refs not in ({}, None):
            raise SceneStateProducerError(
                "development mount fixture cannot claim real side evidence"
            )
        evidence_mode = "DEVELOPMENT_ONLY_SYNTHETIC_FIXTURE"
    else:
        if not independent:
            raise SceneStateProducerError(
                "non-fixture mount requires independent per-side visual evidence"
            )
        if authority.get("method") != "DIRECT_BILATERAL_VISUAL_MOUNT_MEASUREMENT":
            raise SceneStateProducerError(
                "real mount authority method must be direct bilateral visual measurement"
            )
        if any(np.array_equal(matrix, np.eye(4)) for matrix in matrices):
            raise SceneStateProducerError(
                "identity mount cannot impersonate independent visual authority"
            )
        if strict is None or source_record is None:
            raise SceneStateProducerError(
                "real mount evidence requires strict same-FD source verification"
            )
        if not isinstance(evidence_refs, Mapping) or set(evidence_refs) != set(SIDES):
            raise SceneStateProducerError(
                "real mount requires one independent evidence ref per side"
            )
        descriptor_identity = (
            int(_record(source_record)["device"]),
            int(_record(source_record)["inode"]),
        )
        seen: set[tuple[int, int]] = {descriptor_identity}
        for index, side in enumerate(SIDES):
            record = _read_declared_evidence(
                strict, evidence_refs[side], name=f"{side} mount evidence"
            )
            identity = (int(record.device), int(record.inode))
            if identity in seen:
                raise SceneStateProducerError(
                    "mount descriptor and bilateral evidence must be independent files"
                )
            seen.add(identity)
            evidence = _validate_mount_side_evidence(
                record.payload,
                side=side,
                session_id=expected_session_id,
                expected_matrix=matrices[index],
            )
            evidence_records[side] = {
                **_record(record),
                "session_id": evidence["session_id"],
                "side": evidence["side"],
                "provider": evidence["provider"],
                "method": evidence["method"],
                "timestamp_ns": int(evidence["timestamp_ns"]),
                "timestamp_clock_id": evidence["timestamp_clock_id"],
                "time_scope": evidence["time_scope"],
                "device_id": evidence["device_id"],
                "calibration_id": evidence["calibration_id"],
                "calibration_sha256": evidence["calibration_sha256"],
                "source_link": evidence["source_link"],
                "target_link": evidence["target_link"],
                "matrix_direction": evidence["matrix_direction"],
                "axes_convention": evidence["axes_convention"],
                "translation_units": evidence["translation_units"],
                "rotation_units": evidence["rotation_units"],
                "independent_of_r2_wrist_targets": True,
                "derived_from_r2_wrist_targets": False,
                "derived_from_ik_solver": False,
                "selected_by_ik_residual": False,
                "circular_selection": False,
                "synthetic_fixture": False,
                "identity_fixture": False,
                "formal_consumer_allowed": False,
            }
        evidence_mode = "INDEPENDENT_BILATERAL_VISUAL_EVIDENCE"
    return ExplicitMountInput(
        session_id=expected_session_id,
        matrices=matrices,
        development_only=development_only,
        authority=dict(authority),
        evidence_records=evidence_records,
        evidence_mode=evidence_mode,
        descriptor_sha256=sha256_bytes(payload),
    )


def _validate_session_id(session_id: str) -> None:
    lowered = session_id.lower()
    if (
        not SESSION_ID_PATTERN.fullmatch(session_id)
        or "025" in lowered
        or "blind" in lowered
    ):
        raise SceneStateProducerError("blank or blind/025 session is forbidden")


def _validate_frames(frame_indices: Sequence[int], count: int) -> tuple[int, ...]:
    if not frame_indices:
        raise SceneStateProducerError("at least one explicit frame index is required")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in frame_indices
    ):
        raise SceneStateProducerError("frame indices must be exact integers")
    result = tuple(frame_indices)
    if tuple(sorted(set(result))) != result:
        raise SceneStateProducerError("frame indices must be sorted and unique")
    if result[0] < 0 or result[-1] >= count:
        raise SceneStateProducerError("frame index is outside the frozen inputs")
    return result


def _raw_metadata(
    strict: Any,
    raw_root: Path,
    frame_names: Sequence[str],
    expected_intrinsics: np.ndarray,
) -> tuple[tuple[int, int], np.ndarray, list[dict[str, Any]], float]:
    _forbid_path(raw_root)
    records: list[dict[str, Any]] = []
    raw_intrinsics: list[np.ndarray] = []
    resolution: tuple[int, int] | None = None
    maximum_principal_offset = 0.0
    for offset, frame_name in enumerate(frame_names):
        path = raw_root / frame_name / "training_data.json"
        record = read_bound_bytes(strict, path, name=f"RAW metadata {frame_name}")
        item = _parse_json(record.payload, name=f"RAW metadata {frame_name}")
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            raise SceneStateProducerError(
                f"RAW metadata missing for frame {frame_name}"
            )
        k = np.asarray(metadata.get("k"), dtype=np.float64)
        if k.shape != (3, 3) or not np.isfinite(k).all():
            raise SceneStateProducerError(
                f"RAW intrinsics invalid for frame {frame_name}"
            )
        observed = np.asarray([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
        expected = expected_intrinsics[offset]
        if not np.allclose(observed[:2], expected[:2], atol=1e-4, rtol=0):
            raise SceneStateProducerError(
                f"HaWoR/RAW focal lengths disagree for frame {frame_name}"
            )
        principal_offset = float(np.max(np.abs(observed[2:] - expected[2:])))
        if principal_offset > 0.5001:
            raise SceneStateProducerError(
                f"HaWoR/RAW principal points disagree by more than half a pixel: {frame_name}"
            )
        maximum_principal_offset = max(maximum_principal_offset, principal_offset)
        raw_intrinsics.append(observed)
        current = (int(metadata.get("w", 0)), int(metadata.get("h", 0)))
        if min(current) <= 0 or (resolution is not None and current != resolution):
            raise SceneStateProducerError(
                "RAW source resolution is invalid or changes by frame"
            )
        resolution = current
        records.append(_record(record))
    assert resolution is not None
    return resolution, np.stack(raw_intrinsics), records, maximum_principal_offset


def load_sources(
    *,
    strict: Any,
    session_id: str,
    sidecar_path: Path,
    hawor_path: Path,
    raw_root: Path,
    frame_indices: Sequence[int],
    assets: PinnedRobotAssets,
) -> SourceBundle:
    _validate_session_id(session_id)
    for path in (sidecar_path, hawor_path, raw_root):
        _forbid_path(path)
    _require_session_path(sidecar_path, session_id=session_id, name="R2 sidecar")
    _require_session_path(hawor_path, session_id=session_id, name="HaWoR")
    _require_session_path(raw_root, session_id=session_id, name="RAW root")
    sidecar_record = read_bound_bytes(strict, sidecar_path, name="R2 sidecar")
    hawor_record = read_bound_bytes(strict, hawor_path, name="HaWoR")
    r2 = _parse_npz(sidecar_record.payload, name="R2 sidecar")
    hawor = _parse_npz(hawor_record.payload, name="HaWoR")
    required_r2 = {
        "schema_version",
        "frame_names",
        "q",
        "valid",
        "wrist_T_camera",
        "joint_names",
        "joint_lower",
        "joint_upper",
        "canonical_relative_path",
        "embodiment",
        "q_unit",
        "translation_unit",
    }
    required_hawor = {"valid", "camera_intrinsics"}
    if missing := sorted(required_r2 - set(r2)):
        raise SceneStateProducerError(f"R2 arrays missing: {missing}")
    if missing := sorted(required_hawor - set(hawor)):
        raise SceneStateProducerError(f"HaWoR arrays missing: {missing}")
    if str(np.asarray(r2["schema_version"]).item()) != "humanego-robot-sidecar-v1":
        raise SceneStateProducerError("unsupported R2 schema")
    scalar_contract = {
        "canonical_relative_path": f"common/{session_id}/canonical_source.npz",
        "embodiment": "kai22",
        "q_unit": "radian",
        "translation_unit": "metre",
    }
    for name, expected in scalar_contract.items():
        value = np.asarray(r2[name])
        if value.shape != () or str(value.item()) != expected:
            raise SceneStateProducerError(
                f"R2 {name} does not bind the requested session/units"
            )
    all_frames = tuple(str(value) for value in r2["frame_names"])
    count = len(all_frames)
    if all_frames != tuple(f"{index:05d}" for index in range(count)):
        raise SceneStateProducerError(
            "R2 frame identity/order must be exact contiguous indices"
        )
    frames = _validate_frames(frame_indices, count)
    q_raw = np.asarray(r2["q"])
    r2_valid_raw = np.asarray(r2["valid"])
    hawor_valid_raw = np.asarray(hawor["valid"])
    if not np.issubdtype(q_raw.dtype, np.number):
        raise SceneStateProducerError("R2 q must be a numeric radian array")
    if r2_valid_raw.dtype != np.bool_ or hawor_valid_raw.dtype != np.bool_:
        raise SceneStateProducerError("R2/HaWoR valid arrays must be exact booleans")
    q = np.asarray(q_raw, dtype=np.float64)
    r2_valid = np.asarray(r2_valid_raw, dtype=bool)
    wrists = np.asarray(r2["wrist_T_camera"], dtype=np.float64)
    hawor_valid = np.asarray(hawor_valid_raw, dtype=bool)
    intrinsics = np.asarray(hawor["camera_intrinsics"], dtype=np.float64)
    if q.shape != (count, 2, 22) or r2_valid.shape != (count, 2):
        raise SceneStateProducerError("R2 q/valid shape mismatch")
    if wrists.shape != (count, 2, 4, 4):
        raise SceneStateProducerError("R2 wrist target shape mismatch")
    if hawor_valid.shape != (2, count) or intrinsics.shape != (count, 4):
        raise SceneStateProducerError("HaWoR validity/intrinsics shape mismatch")
    selected = np.asarray(frames, dtype=np.int64)
    full_valid = r2_valid & hawor_valid.T
    valid = full_valid[selected]
    if np.any(~np.any(valid, axis=1)):
        raise SceneStateProducerError("a selected frame has no R2+HaWoR-valid side")
    if (
        not np.isfinite(q[selected][valid]).all()
        or not np.isfinite(wrists[selected][valid]).all()
    ):
        raise SceneStateProducerError("valid frozen R2 state contains NaN/Inf")
    selected_intrinsics = intrinsics[selected]
    if not np.isfinite(selected_intrinsics).all() or np.any(
        selected_intrinsics[:, :2] <= 0
    ):
        raise SceneStateProducerError("selected HaWoR intrinsics are invalid")
    hand_names = np.asarray(r2["joint_names"])
    expected_hand_names = np.asarray(
        [
            [
                joint.name
                for joint in assets.left_hand.joints
                if joint.joint_type != "fixed"
            ],
            [
                joint.name
                for joint in assets.right_hand.joints
                if joint.joint_type != "fixed"
            ],
        ]
    )
    if hand_names.shape != (2, 22) or not np.array_equal(
        hand_names, expected_hand_names
    ):
        raise SceneStateProducerError(
            "R2 q_hand identity differs from pinned KaiHand URDF"
        )
    declared_lower = np.asarray(r2["joint_lower"], dtype=np.float64)
    declared_upper = np.asarray(r2["joint_upper"], dtype=np.float64)
    expected_lower = np.asarray(
        [
            [joint.lower for joint in model.joints if joint.joint_type != "fixed"]
            for model in (assets.left_hand, assets.right_hand)
        ],
        dtype=np.float64,
    )
    expected_upper = np.asarray(
        [
            [joint.upper for joint in model.joints if joint.joint_type != "fixed"]
            for model in (assets.left_hand, assets.right_hand)
        ],
        dtype=np.float64,
    )
    if (
        declared_lower.shape != (2, 22)
        or declared_upper.shape != (2, 22)
        or not np.isfinite(declared_lower).all()
        or not np.isfinite(declared_upper).all()
        or not np.allclose(declared_lower, expected_lower, atol=1e-6, rtol=0)
        or not np.allclose(declared_upper, expected_upper, atol=1e-6, rtol=0)
    ):
        raise SceneStateProducerError(
            "R2 hand limits differ from the pinned KaiHand URDF"
        )
    selected_q = q[selected]
    if np.any(
        selected_q[valid] < declared_lower[np.nonzero(valid)[1]] - 1e-6
    ) or np.any(selected_q[valid] > declared_upper[np.nonzero(valid)[1]] + 1e-6):
        raise SceneStateProducerError("valid R2 q_hand violates pinned joint limits")
    for frame, side in zip(*np.nonzero(valid)):
        matrix = wrists[selected[frame], side]
        if (
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(
                matrix[:3, :3].T @ matrix[:3, :3],
                np.eye(3),
                atol=1e-6,
                rtol=0,
            )
            or abs(float(np.linalg.det(matrix[:3, :3])) - 1.0) > 1e-6
        ):
            raise SceneStateProducerError(
                "valid R2 wrist_T_camera is not a right-handed SE(3) transform"
            )
    selected_names = tuple(all_frames[index] for index in frames)
    source_resolution, raw_intrinsics, metadata_records, principal_offset = (
        _raw_metadata(
            strict,
            raw_root,
            selected_names,
            selected_intrinsics,
        )
    )
    return SourceBundle(
        frame_indices=frames,
        source_frame_count=count,
        full_valid=full_valid,
        frame_names=selected_names,
        q_hand=q[selected],
        wrist_T_camera=wrists[selected],
        valid=valid,
        camera_intrinsics=raw_intrinsics,
        source_resolution=source_resolution,
        hand_joint_names=hand_names,
        records={
            "r2_sidecar": _record(sidecar_record),
            "hawor": _record(hawor_record),
            "raw_metadata": {
                "records": metadata_records,
                "count": len(metadata_records),
                "camera_intrinsics_source": "RAW_training_data.metadata.k",
                "hawor_focal_lengths_exact": True,
                "hawor_principal_point_max_offset_px": principal_offset,
            },
        },
    )


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(matrix).as_rotvec()


def _rotation_matrix(vector: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(vector).as_matrix()


def _average_se3(transforms: Sequence[np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    if not transforms:
        raise SceneStateProducerError("cannot average an empty transform set")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([value[:3, 3] for value in transforms], axis=0)
    rotations = Rotation.from_matrix(np.stack([value[:3, :3] for value in transforms]))
    result[:3, :3] = rotations.mean().as_matrix()
    return result


def _interpolate_se3(
    current: np.ndarray, target: np.ndarray, amount: float
) -> np.ndarray:
    relative = np.linalg.inv(current) @ target
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = _rotation_matrix(_rotation_vector(relative[:3, :3]) * amount)
    delta[:3, 3] = relative[:3, 3] * amount
    return current @ delta


def _arm_limits(assets: PinnedRobotAssets) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((2, 7), dtype=np.float64)
    upper = np.empty((2, 7), dtype=np.float64)
    by_name = {joint.name: joint for joint in assets.tianji.joints}
    for side, names in enumerate(ARM_JOINT_NAMES):
        for offset, name in enumerate(names):
            joint = by_name.get(name)
            if joint is None or joint.lower is None or joint.upper is None:
                raise SceneStateProducerError(
                    f"pinned Tianji joint lacks finite limits: {name}"
                )
            lower[side, offset] = joint.lower
            upper[side, offset] = joint.upper
    return lower, upper


def _npz_scalar_string(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = arrays.get(name)
    if value is None or np.asarray(value).shape != ():
        raise SceneStateProducerError(f"external authority scalar {name} is missing")
    return str(np.asarray(value).item())


def _require_direct_external_flags(value: Mapping[str, Any], *, name: str) -> None:
    exact = {
        "independent_of_r2_wrist_targets": True,
        "derived_from_r2_wrist_targets": False,
        "derived_from_ik_solver": False,
        "selected_by_ik_residual": False,
        "interpolated_or_filled": False,
        "synthetic_fixture": False,
        "identity_fixture": False,
        "formal_consumer_allowed": False,
    }
    for field, expected in exact.items():
        if value.get(field) is not expected:
            raise SceneStateProducerError(f"{name} {field} must be {expected}")


def _validate_external_base_evidence(
    payload: bytes,
    *,
    session_id: str,
    descriptor: Mapping[str, Any],
    expected_base: np.ndarray,
) -> Mapping[str, Any]:
    value = _parse_json(payload, name="external camera/base evidence")
    required = {
        "schema_version",
        "session_id",
        "evidence_kind",
        "provider",
        "method",
        "T_camera_base_semantics",
        "transform_convention",
        "translation_units",
        "timestamp_ns",
        "timestamp_clock_id",
        "device_id",
        "calibration_id",
        "calibration_sha256",
        "independent_of_r2_wrist_targets",
        "derived_from_r2_wrist_targets",
        "derived_from_ik_solver",
        "selected_by_ik_residual",
        "interpolated_or_filled",
        "synthetic_fixture",
        "identity_fixture",
        "formal_consumer_allowed",
        "T_camera_base",
    }
    if set(value) != required:
        raise SceneStateProducerError(
            "external camera/base evidence keys are not exact"
        )
    fixed = {
        "schema_version": EXTERNAL_BASE_EVIDENCE_SCHEMA,
        "session_id": session_id,
        "evidence_kind": EXTERNAL_BASE_EVIDENCE_KIND,
        "method": "DIRECT_CAMERA_TO_ROBOT_BASE_CALIBRATION",
        "T_camera_base_semantics": BASE_TRANSFORM_SEMANTICS,
        "transform_convention": RIGHT_HANDED_AXES,
        "translation_units": "metres",
        "timestamp_clock_id": descriptor["timestamp_clock_id"],
        "device_id": descriptor["device_id"],
        "calibration_id": descriptor["calibration_id"],
        "calibration_sha256": descriptor["calibration_sha256"],
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise SceneStateProducerError(
                f"external camera/base evidence {field} must be {expected}"
            )
    _require_nonempty_string(
        value.get("provider"), name="external camera/base evidence provider"
    )
    _require_direct_external_flags(value, name="external camera/base evidence")
    if type(value.get("timestamp_ns")) is not int or int(value["timestamp_ns"]) < 0:
        raise SceneStateProducerError(
            "external camera/base evidence timestamp_ns must be non-negative"
        )
    try:
        observed = validate_se3(
            np.asarray(value.get("T_camera_base"), dtype=np.float64),
            name="external camera/base evidence T_camera_base",
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        raise SceneStateProducerError(
            f"external camera/base evidence transform is invalid: {exc}"
        ) from exc
    if not np.array_equal(observed, expected_base):
        raise SceneStateProducerError(
            "external camera/base evidence transform differs from authority arrays"
        )
    return value


def _validate_external_q_side_evidence(
    payload: bytes,
    *,
    side: str,
    session_id: str,
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    side_index: int,
) -> Mapping[str, Any]:
    value = _parse_json(payload, name=f"external {side} q_arm evidence")
    required = {
        "schema_version",
        "session_id",
        "side",
        "evidence_kind",
        "provider",
        "method",
        "frame_count",
        "frame_range",
        "frame_timestamp_mapping",
        "timestamp_clock_id",
        "device_id",
        "calibration_id",
        "calibration_sha256",
        "axes_convention",
        "q_arm_units",
        "array_digest_canonicalization",
        "frame_names_sha256",
        "timestamp_ns_sha256",
        "valid_sha256",
        "q_arm_sha256",
        "arm_joint_names_sha256",
        "joint_lower_sha256",
        "joint_upper_sha256",
        "independent_of_r2_wrist_targets",
        "derived_from_r2_wrist_targets",
        "derived_from_ik_solver",
        "selected_by_ik_residual",
        "interpolated_or_filled",
        "synthetic_fixture",
        "identity_fixture",
        "formal_consumer_allowed",
    }
    if set(value) != required:
        raise SceneStateProducerError(
            f"external {side} q_arm evidence keys are not exact"
        )
    frame_count = int(np.asarray(arrays["frame_names"]).shape[0])
    fixed = {
        "schema_version": EXTERNAL_Q_ARM_SIDE_EVIDENCE_SCHEMA,
        "session_id": session_id,
        "side": side,
        "evidence_kind": EXTERNAL_Q_ARM_EVIDENCE_KIND,
        "method": "DIRECT_ARM_JOINT_ENCODER_READOUT",
        "frame_count": frame_count,
        "frame_range": {"start": 0, "stop_exclusive": frame_count, "contiguous": True},
        "frame_timestamp_mapping": EXTERNAL_FRAME_TIMESTAMP_MAPPING,
        "timestamp_clock_id": descriptor["timestamp_clock_id"],
        "axes_convention": RIGHT_HANDED_AXES,
        "q_arm_units": "radians",
        "array_digest_canonicalization": EXTERNAL_ARRAY_DIGEST_CANONICALIZATION,
        "frame_names_sha256": _canonical_array_sha256(arrays["frame_names"]),
        "timestamp_ns_sha256": _canonical_array_sha256(arrays["timestamp_ns"]),
        "valid_sha256": _canonical_array_sha256(arrays["valid"][:, side_index]),
        "q_arm_sha256": _canonical_array_sha256(arrays["q_arm"][:, side_index]),
        "arm_joint_names_sha256": _canonical_array_sha256(
            arrays["arm_joint_names"][side_index]
        ),
        "joint_lower_sha256": _canonical_array_sha256(
            arrays["joint_lower"][side_index]
        ),
        "joint_upper_sha256": _canonical_array_sha256(
            arrays["joint_upper"][side_index]
        ),
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise SceneStateProducerError(
                f"external {side} q_arm evidence {field} does not match arrays"
            )
    for field in ("provider", "device_id", "calibration_id"):
        _require_nonempty_string(
            value.get(field), name=f"external {side} q_arm evidence {field}"
        )
    if not _is_sha256(value.get("calibration_sha256")):
        raise SceneStateProducerError(
            f"external {side} q_arm calibration_sha256 must be lowercase SHA-256"
        )
    _require_direct_external_flags(value, name=f"external {side} q_arm evidence")
    return value


def load_external_base_q_authority(
    *,
    strict: Any,
    descriptor_record: Any,
    expected_session_id: str,
    sources: SourceBundle,
    assets: PinnedRobotAssets,
) -> SolverResult:
    """Consume independent base/q bytes without invoking or scoring the IK solver."""

    _validate_session_id(expected_session_id)
    descriptor_path = Path(str(descriptor_record.path))
    _require_canonical_record_path(descriptor_record, name="external base/q descriptor")
    _require_session_path(
        descriptor_path,
        session_id=expected_session_id,
        name="external base/q descriptor",
    )
    value = _parse_json(descriptor_record.payload, name="external base/q authority")
    required_keys = {
        "schema_version",
        "session_id",
        "authority_kind",
        "evidence_kind",
        "provider",
        "method",
        "development_only",
        "formal_consumer_allowed",
        "direct_measurement_evidence",
        "independent_of_r2_wrist_targets",
        "derived_from_r2_wrist_targets",
        "derived_from_ik_solver",
        "selected_by_ik_residual",
        "interpolated_or_filled",
        "synthetic_fixture",
        "identity_fixture",
        "solver_not_used",
        "residual_not_claimed",
        "T_camera_base_semantics",
        "transform_convention",
        "translation_units",
        "q_arm_units",
        "side_order",
        "frame_count",
        "frame_range",
        "frame_timestamp_mapping",
        "timestamp_clock_id",
        "device_id",
        "calibration_id",
        "calibration_sha256",
        "urdf_refs",
        "source_refs",
        "base_evidence_ref",
        "q_arm_evidence_by_side",
        "arrays_ref",
    }
    if set(value) != required_keys:
        raise SceneStateProducerError("external base/q descriptor keys are not exact")
    fixed = {
        "schema_version": EXTERNAL_BASE_Q_SCHEMA,
        "session_id": expected_session_id,
        "authority_kind": "DIRECT_EXTERNAL_BASE_AND_ARM_STATE",
        "evidence_kind": EXTERNAL_AUTHORITY_EVIDENCE_KIND,
        "method": EXTERNAL_AUTHORITY_METHOD,
        "formal_consumer_allowed": False,
        "direct_measurement_evidence": True,
        "independent_of_r2_wrist_targets": True,
        "derived_from_r2_wrist_targets": False,
        "derived_from_ik_solver": False,
        "selected_by_ik_residual": False,
        "interpolated_or_filled": False,
        "synthetic_fixture": False,
        "identity_fixture": False,
        "solver_not_used": True,
        "residual_not_claimed": True,
        "T_camera_base_semantics": BASE_TRANSFORM_SEMANTICS,
        "transform_convention": RIGHT_HANDED_AXES,
        "translation_units": "metres",
        "q_arm_units": "radians",
        "side_order": list(SIDES),
        "frame_timestamp_mapping": EXTERNAL_FRAME_TIMESTAMP_MAPPING,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise SceneStateProducerError(
                f"external base/q field {field} must be {expected}"
            )
    if type(value.get("development_only")) is not bool:
        raise SceneStateProducerError(
            "external base/q development_only must be an exact boolean"
        )
    _require_nonempty_string(value.get("provider"), name="external base/q provider")
    frame_count = value.get("frame_count")
    if type(frame_count) is not int or frame_count != sources.source_frame_count:
        raise SceneStateProducerError(
            "external base/q frame_count must equal the frozen source frame count"
        )
    frame_range = value.get("frame_range")
    if frame_range != {
        "start": 0,
        "stop_exclusive": frame_count,
        "contiguous": True,
    }:
        raise SceneStateProducerError(
            "external base/q must bind exact frame range 0..N-1"
        )
    for field in ("timestamp_clock_id", "device_id", "calibration_id"):
        _require_nonempty_string(value.get(field), name=f"external base/q {field}")
    if not _is_sha256(value.get("calibration_sha256")):
        raise SceneStateProducerError(
            "external base/q calibration_sha256 must be exact lowercase SHA-256"
        )

    source_refs = value.get("source_refs")
    if not isinstance(source_refs, Mapping) or set(source_refs) != {
        "r2_sidecar",
        "hawor",
    }:
        raise SceneStateProducerError(
            "external base/q source_refs must bind exact R2 and HaWoR records"
        )
    source_records: dict[str, Mapping[str, Any]] = {}
    for source_name in ("r2_sidecar", "hawor"):
        source_record = sources.records.get(source_name)
        if not isinstance(source_record, Mapping):
            raise SceneStateProducerError(
                f"external base/q source lineage lacks {source_name}"
            )
        source_records[source_name] = _require_record_ref(
            source_refs[source_name],
            source_record,
            name=f"external base/q {source_name}",
        )

    arrays_record = _read_declared_evidence(
        strict, value.get("arrays_ref"), name="external base/q arrays"
    )
    _require_session_path(
        Path(str(arrays_record.path)),
        session_id=expected_session_id,
        name="external base/q arrays",
    )
    base_evidence_record = _read_declared_evidence(
        strict,
        value.get("base_evidence_ref"),
        name="external camera/base direct evidence",
    )
    _require_session_path(
        Path(str(base_evidence_record.path)),
        session_id=expected_session_id,
        name="external camera/base direct evidence",
    )
    q_evidence_refs = value.get("q_arm_evidence_by_side")
    if not isinstance(q_evidence_refs, Mapping) or set(q_evidence_refs) != set(SIDES):
        raise SceneStateProducerError(
            "external base/q requires exact left/right q_arm evidence refs"
        )
    q_evidence_records: dict[str, Any] = {}
    for side in SIDES:
        record = _read_declared_evidence(
            strict,
            q_evidence_refs[side],
            name=f"external {side} q_arm direct evidence",
        )
        _require_session_path(
            Path(str(record.path)),
            session_id=expected_session_id,
            name=f"external {side} q_arm direct evidence",
        )
        q_evidence_records[side] = record
    urdf_refs = value.get("urdf_refs")
    if not isinstance(urdf_refs, Mapping) or set(urdf_refs) != {
        "robot_asset_pin",
        "tianji_urdf",
    }:
        raise SceneStateProducerError(
            "external base/q must bind exact robot asset-pin and Tianji URDF refs"
        )
    asset_pin_record = _read_declared_evidence(
        strict, urdf_refs["robot_asset_pin"], name="external base/q robot asset pin"
    )
    _require_record_ref(
        urdf_refs["robot_asset_pin"],
        assets.asset_pin_record,
        name="external base/q robot asset pin",
    )
    tianji_record = _read_declared_evidence(
        strict, urdf_refs["tianji_urdf"], name="external base/q Tianji URDF"
    )
    if Path(str(tianji_record.path)) != assets.tianji.path.resolve(strict=True):
        raise SceneStateProducerError(
            "external base/q Tianji URDF ref is not the pinned model"
        )
    asset_pin = _parse_json(
        asset_pin_record.payload, name="external base/q robot asset pin"
    )
    tianji_relative = assets.tianji.path.relative_to(assets.project_root).as_posix()
    pinned_rows = [
        row
        for row in asset_pin.get("files", [])
        if isinstance(row, Mapping) and row.get("path") == tianji_relative
    ]
    if len(pinned_rows) != 1 or any(
        pinned_rows[0].get(key) != _record(tianji_record)[key]
        for key in ("bytes", "sha256")
    ):
        raise SceneStateProducerError(
            "external base/q Tianji URDF bytes/SHA differ from the bound asset pin"
        )
    identity_records = (
        descriptor_record,
        arrays_record,
        base_evidence_record,
        *(q_evidence_records[side] for side in SIDES),
        asset_pin_record,
        tianji_record,
    )
    identity_refs = [*(_record(record) for record in identity_records)]
    identity_refs.extend(source_records.values())
    identities = {
        (int(record["device"]), int(record["inode"])) for record in identity_refs
    }
    if len(identities) != len(identity_refs):
        raise SceneStateProducerError(
            "external authority/evidence/source/URDF files must not alias"
        )

    arrays = _parse_npz(arrays_record.payload, name="external base/q arrays")
    array_keys = {
        "schema_version",
        "session_id",
        "frame_names",
        "timestamp_ns",
        "valid",
        "T_camera_base",
        "q_arm",
        "arm_joint_names",
        "joint_lower",
        "joint_upper",
    }
    if set(arrays) != array_keys:
        raise SceneStateProducerError("external base/q NPZ keys are not exact")
    if _npz_scalar_string(arrays, "schema_version") != EXTERNAL_BASE_Q_ARRAY_SCHEMA:
        raise SceneStateProducerError("unsupported external base/q array schema")
    if _npz_scalar_string(arrays, "session_id") != expected_session_id:
        raise SceneStateProducerError("external base/q array session mismatch")
    names_raw = np.asarray(arrays["frame_names"])
    if names_raw.ndim != 1 or names_raw.dtype.kind not in {"U", "S"}:
        raise SceneStateProducerError(
            "external frame_names must be an exact string array"
        )
    frame_names = tuple(str(item) for item in names_raw)
    expected_names = tuple(f"{index:05d}" for index in range(frame_count))
    if frame_names != expected_names:
        raise SceneStateProducerError(
            "external base/q frame identity/order must be exact contiguous 0..N-1"
        )
    timestamp_ns = np.asarray(arrays["timestamp_ns"])
    valid = np.asarray(arrays["valid"])
    base = np.asarray(arrays["T_camera_base"])
    q_arm = np.asarray(arrays["q_arm"])
    if timestamp_ns.dtype != np.int64 or timestamp_ns.shape != (frame_count,):
        raise SceneStateProducerError("external timestamps must be exact int64[N]")
    if np.any(timestamp_ns < 0) or np.any(np.diff(timestamp_ns) <= 0):
        raise SceneStateProducerError(
            "external timestamps must be non-negative and strictly increasing"
        )
    if valid.dtype != np.bool_ or valid.shape != (frame_count, 2):
        raise SceneStateProducerError("external valid must be exact bool[N,2]")
    if base.dtype != np.float64 or base.shape != (4, 4):
        raise SceneStateProducerError(
            "external T_camera_base must be exact float64[4,4]"
        )
    try:
        base = validate_se3(base, name="external T_camera_base")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise SceneStateProducerError(
            f"external T_camera_base must be right-handed SE(3): {exc}"
        ) from exc
    if q_arm.dtype != np.float64 or q_arm.shape != (frame_count, 2, 7):
        raise SceneStateProducerError("external q_arm must be exact float64[N,2,7]")
    if not np.isfinite(q_arm[valid]).all():
        raise SceneStateProducerError("external valid q_arm contains NaN/Inf")
    if np.any(~valid) and not np.isnan(q_arm[~valid]).all():
        raise SceneStateProducerError(
            "external invalid q_arm rows must be all-NaN, never filled or interpolated"
        )
    arm_names = np.asarray(arrays["arm_joint_names"])
    if arm_names.dtype.kind not in {"U", "S"} or not np.array_equal(
        arm_names, np.asarray(ARM_JOINT_NAMES)
    ):
        raise SceneStateProducerError("external q_arm joint identity/order mismatch")
    lower, upper = _arm_limits(assets)
    declared_lower = np.asarray(arrays["joint_lower"])
    declared_upper = np.asarray(arrays["joint_upper"])
    if (
        declared_lower.dtype != np.float64
        or declared_upper.dtype != np.float64
        or declared_lower.shape != (2, 7)
        or declared_upper.shape != (2, 7)
        or not np.array_equal(declared_lower, lower)
        or not np.array_equal(declared_upper, upper)
    ):
        raise SceneStateProducerError(
            "external q_arm limits must exactly match the pinned Tianji URDF"
        )
    for side in range(2):
        side_valid = valid[:, side]
        if np.any(q_arm[side_valid, side] < lower[side] - 1e-12) or np.any(
            q_arm[side_valid, side] > upper[side] + 1e-12
        ):
            raise SceneStateProducerError(
                f"external {SIDES[side]} q_arm violates pinned joint limits"
            )
    selected = np.asarray(sources.frame_indices, dtype=np.int64)
    if tuple(frame_names[index] for index in selected) != sources.frame_names:
        raise SceneStateProducerError(
            "external base/q selected frames do not join the frozen source frames"
        )
    if not np.array_equal(valid, sources.full_valid) or not np.array_equal(
        valid[selected], sources.valid
    ):
        raise SceneStateProducerError(
            "external base/q full valid mask differs from the R2+HaWoR source intersection"
        )
    base_evidence = _validate_external_base_evidence(
        base_evidence_record.payload,
        session_id=expected_session_id,
        descriptor=value,
        expected_base=base,
    )
    q_evidence: dict[str, Mapping[str, Any]] = {}
    for side_index, side in enumerate(SIDES):
        q_evidence[side] = _validate_external_q_side_evidence(
            q_evidence_records[side].payload,
            side=side,
            session_id=expected_session_id,
            descriptor=value,
            arrays=arrays,
            side_index=side_index,
        )
    selected_count = len(sources.frame_names)
    nan_residual = np.full((selected_count, 2), np.nan, dtype=np.float64)
    return SolverResult(
        q_arm=np.asarray(q_arm[selected], dtype=np.float64),
        T_camera_base=np.asarray(base, dtype=np.float64),
        position_residual_mm=nan_residual.copy(),
        rotation_residual_deg=nan_residual.copy(),
        evaluations=np.zeros((selected_count, 2), dtype=np.int32),
        scene_state_mode=SCENE_STATE_MODE_EXTERNAL_AUTHORITY,
        solver_used=False,
        residual_claimed=False,
        development_only=bool(value["development_only"]),
        timestamp_ns=np.asarray(timestamp_ns[selected], dtype=np.int64),
        external_authority={
            "schema_version": EXTERNAL_BASE_Q_SCHEMA,
            "descriptor": _record(descriptor_record),
            "arrays": _record(arrays_record),
            "direct_evidence": {
                "camera_base": {
                    **_record(base_evidence_record),
                    **dict(base_evidence),
                },
                "q_arm_by_side": {
                    side: {
                        **_record(q_evidence_records[side]),
                        **dict(q_evidence[side]),
                    }
                    for side in SIDES
                },
            },
            "urdf_refs": {
                "robot_asset_pin": _record(asset_pin_record),
                "tianji_urdf": _record(tianji_record),
            },
            "session_id": expected_session_id,
            "frame_count": frame_count,
            "selected_frame_indices": list(sources.frame_indices),
            "selected_frame_names": list(sources.frame_names),
            "selected_timestamp_ns": [
                int(item) for item in np.asarray(timestamp_ns[selected])
            ],
            "source_refs": {
                key: dict(record) for key, record in source_records.items()
            },
            "authority_kind": "DIRECT_EXTERNAL_BASE_AND_ARM_STATE",
            "evidence_kind": EXTERNAL_AUTHORITY_EVIDENCE_KIND,
            "provider": value["provider"],
            "method": EXTERNAL_AUTHORITY_METHOD,
            "direct_measurement_evidence": True,
            "independent_of_r2_wrist_targets": True,
            "derived_from_r2_wrist_targets": False,
            "derived_from_ik_solver": False,
            "selected_by_ik_residual": False,
            "interpolated_or_filled": False,
            "synthetic_fixture": False,
            "identity_fixture": False,
            "formal_consumer_allowed": False,
            "timestamp_clock_id": value["timestamp_clock_id"],
            "frame_timestamp_mapping": EXTERNAL_FRAME_TIMESTAMP_MAPPING,
            "device_id": value["device_id"],
            "calibration_id": value["calibration_id"],
            "calibration_sha256": value["calibration_sha256"],
            "T_camera_base_semantics": BASE_TRANSFORM_SEMANTICS,
            "q_arm_units": "radians",
            "solver_not_used": True,
            "residual_not_claimed": True,
        },
    )


def _tool_fk(assets: PinnedRobotAssets, side: int, q: np.ndarray) -> np.ndarray:
    values = {name: 0.0 for row in ARM_JOINT_NAMES for name in row}
    values.update(
        {
            name: float(value)
            for name, value in zip(ARM_JOINT_NAMES[side], q, strict=True)
        }
    )
    return forward_kinematics(assets.tianji, values)[f"{SIDES[side]}_tool"]


def _pose_residual(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.linalg.inv(target) @ actual
    return np.concatenate(
        [
            delta[:3, 3] / POSITION_SCALE_M,
            _rotation_vector(delta[:3, :3]) / ROTATION_SCALE_RAD,
        ]
    )


def _position_residual(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the fixed 3D translation objective used by homotopy stage one."""

    delta = np.linalg.inv(target) @ actual
    return np.asarray(delta[:3, 3] / POSITION_SCALE_M, dtype=np.float64)


def _solve_one_arm_position_only(
    assets: PinnedRobotAssets,
    *,
    side: int,
    base: np.ndarray,
    target_tool: np.ndarray,
    initial_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Solve only tool translation for one independent fixed-base pair."""

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise SceneStateProducerError("scipy is required for CPU IK") from exc

    def residual(q: np.ndarray) -> np.ndarray:
        return _position_residual(
            base @ _tool_fk(assets, side, q),
            target_tool,
        )

    result = least_squares(
        residual,
        np.clip(initial_q, lower, upper),
        bounds=(lower, upper),
        method="trf",
        ftol=IK_TRF_FTOL,
        xtol=IK_TRF_XTOL,
        gtol=IK_TRF_GTOL,
        max_nfev=FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS,
    )
    if not np.isfinite(result.x).all():
        raise SceneStateProducerError(
            "fixed-base position-only CPU IK returned non-finite q_arm"
        )
    return np.asarray(result.x, dtype=np.float64), int(result.nfev)


def _solve_one_arm(
    assets: PinnedRobotAssets,
    *,
    side: int,
    base: np.ndarray,
    target_tool: np.ndarray,
    initial_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, int]:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise SceneStateProducerError("scipy is required for CPU IK") from exc

    def residual(q: np.ndarray) -> np.ndarray:
        return _pose_residual(base @ _tool_fk(assets, side, q), target_tool)

    result = least_squares(
        residual,
        np.clip(initial_q, lower, upper),
        bounds=(lower, upper),
        method="trf",
        ftol=IK_TRF_FTOL,
        xtol=IK_TRF_XTOL,
        gtol=IK_TRF_GTOL,
        max_nfev=IK_MAX_EVALUATIONS,
    )
    if not np.isfinite(result.x).all():
        raise SceneStateProducerError("CPU IK returned non-finite q_arm")
    return np.asarray(result.x, dtype=np.float64), int(result.nfev)


def _transform_parameters(transform: np.ndarray) -> np.ndarray:
    return np.concatenate([transform[:3, 3], _rotation_vector(transform[:3, :3])])


def _parameters_transform(parameters: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = parameters[:3]
    transform[:3, :3] = _rotation_matrix(parameters[3:6])
    return transform


def _joint_base_refinement(
    assets: PinnedRobotAssets,
    *,
    base: np.ndarray,
    q_arm: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Jointly refine one base and every valid arm without touching mounts."""

    try:
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise SceneStateProducerError("scipy is required for joint CPU IK") from exc
    pairs = tuple((int(frame), int(side)) for frame, side in zip(*np.nonzero(valid)))
    initial = [_transform_parameters(base)]
    lower_bounds = [np.full(6, -np.inf, dtype=np.float64)]
    upper_bounds = [np.full(6, np.inf, dtype=np.float64)]
    for frame, side in pairs:
        initial.append(q_arm[frame, side])
        lower_bounds.append(lower[side])
        upper_bounds.append(upper[side])
    x0 = np.concatenate(initial)
    lb = np.concatenate(lower_bounds)
    ub = np.concatenate(upper_bounds)

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate_base = _parameters_transform(parameters[:6])
        rows: list[np.ndarray] = []
        for pair_index, (frame, side) in enumerate(pairs):
            start = 6 + pair_index * 7
            rows.append(
                _pose_residual(
                    candidate_base
                    @ _tool_fk(assets, side, parameters[start : start + 7]),
                    targets[frame, side],
                )
            )
        return np.concatenate(rows)

    sparsity = lil_matrix((len(pairs) * 6, 6 + len(pairs) * 7), dtype=np.int8)
    for pair_index in range(len(pairs)):
        row = slice(pair_index * 6, (pair_index + 1) * 6)
        sparsity[row, :6] = 1
        start = 6 + pair_index * 7
        sparsity[row, start : start + 7] = 1
    result = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        method="trf",
        jac_sparsity=sparsity.tocsr(),
        tr_solver="lsmr",
        x_scale="jac",
        ftol=IK_TRF_FTOL,
        xtol=IK_TRF_XTOL,
        gtol=IK_TRF_GTOL,
        max_nfev=IK_MAX_EVALUATIONS,
    )
    if not np.isfinite(result.x).all():
        raise SceneStateProducerError("joint base/IK refinement returned NaN/Inf")
    refined_q = np.zeros_like(q_arm)
    for pair_index, (frame, side) in enumerate(pairs):
        start = 6 + pair_index * 7
        refined_q[frame, side] = result.x[start : start + 7]
    return _parameters_transform(result.x[:6]), refined_q, int(result.nfev)


def solve_scene_state(
    assets: PinnedRobotAssets,
    *,
    wrist_T_camera: np.ndarray,
    valid: np.ndarray,
    mounts: np.ndarray,
) -> SolverResult:
    count = int(wrist_T_camera.shape[0])
    if wrist_T_camera.shape != (count, 2, 4, 4) or valid.shape != (count, 2):
        raise SceneStateProducerError("solver target/valid shape mismatch")
    if mounts.shape != (2, 4, 4):
        raise SceneStateProducerError("solver requires exactly two constant mounts")
    lower, upper = _arm_limits(assets)
    targets = np.zeros_like(wrist_T_camera, dtype=np.float64)
    for frame, side in zip(*np.nonzero(valid)):
        targets[frame, side] = wrist_T_camera[frame, side] @ np.linalg.inv(mounts[side])
    q_arm = np.zeros((count, 2, 7), dtype=np.float64)
    zero = np.zeros(7, dtype=np.float64)
    base_candidates = [
        targets[frame, side] @ np.linalg.inv(_tool_fk(assets, side, zero))
        for frame, side in zip(*np.nonzero(valid))
    ]
    base = _average_se3(base_candidates)
    evaluations = np.zeros((count, 2), dtype=np.int32)
    for _ in range(OUTER_ITERATIONS):
        for side in range(2):
            previous = np.zeros(7, dtype=np.float64)
            for frame in range(count):
                if not valid[frame, side]:
                    continue
                initial = q_arm[frame, side] if np.any(q_arm[frame, side]) else previous
                solved, nfev = _solve_one_arm(
                    assets,
                    side=side,
                    base=base,
                    target_tool=targets[frame, side],
                    initial_q=initial,
                    lower=lower[side],
                    upper=upper[side],
                )
                q_arm[frame, side] = solved
                evaluations[frame, side] += nfev
                previous = solved
        implied_bases = [
            targets[frame, side]
            @ np.linalg.inv(_tool_fk(assets, side, q_arm[frame, side]))
            for frame, side in zip(*np.nonzero(valid))
        ]
        base = _interpolate_se3(base, _average_se3(implied_bases), 0.5)
    base, q_arm, joint_evaluations = _joint_base_refinement(
        assets,
        base=base,
        q_arm=q_arm,
        targets=targets,
        valid=valid,
        lower=lower,
        upper=upper,
    )
    evaluations[valid] += joint_evaluations
    position = np.full((count, 2), np.nan, dtype=np.float64)
    rotation = np.full((count, 2), np.nan, dtype=np.float64)
    for frame, side in zip(*np.nonzero(valid)):
        actual_hand = base @ _tool_fk(assets, side, q_arm[frame, side]) @ mounts[side]
        delta = np.linalg.inv(wrist_T_camera[frame, side]) @ actual_hand
        position[frame, side] = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
        rotation[frame, side] = float(
            np.linalg.norm(_rotation_vector(delta[:3, :3])) * 180.0 / np.pi
        )
    if np.max(position[valid]) > MAX_POSITION_RESIDUAL_MM:
        raise SceneStateProducerError(
            f"CPU IK position residual {np.max(position[valid]):.3f} mm exceeds "
            f"{MAX_POSITION_RESIDUAL_MM:.1f} mm"
        )
    if np.max(rotation[valid]) > MAX_ROTATION_RESIDUAL_DEG:
        raise SceneStateProducerError(
            f"CPU IK rotation residual {np.max(rotation[valid]):.3f} deg exceeds "
            f"{MAX_ROTATION_RESIDUAL_DEG:.1f} deg"
        )
    return SolverResult(q_arm, base, position, rotation, evaluations)


def solve_scene_state_fixed_base(
    assets: PinnedRobotAssets,
    *,
    wrist_T_camera: np.ndarray,
    valid: np.ndarray,
    mounts: np.ndarray,
    T_camera_base: np.ndarray,
    candidate: FixedCameraBaseInput | None = None,
) -> SolverResult:
    """Solve every valid pair with a fixed two-stage homotopy and exact base.

    Each pair independently starts its position-only TRF stage at the same
    zero 7-DoF vector.  Only that pair's position-stage result initializes its
    full 6D-pose TRF stage.  No state crosses a frame, side, candidate, or call;
    no base update/refinement occurs; and failed gates remain in the frozen
    source-valid denominator.
    """

    count = int(wrist_T_camera.shape[0])
    if wrist_T_camera.shape != (count, 2, 4, 4) or valid.shape != (count, 2):
        raise SceneStateProducerError("fixed-base solver target/valid shape mismatch")
    if valid.dtype != np.bool_:
        raise SceneStateProducerError("fixed-base valid must be exact bool[N,2]")
    if mounts.dtype != np.float64 or mounts.shape != (2, 4, 4):
        raise SceneStateProducerError(
            "fixed-base solver requires exact float64[2,4,4] constant mounts"
        )
    if not np.any(valid):
        raise SceneStateProducerError("fixed-base solver requires one valid frame/side")
    for side in range(2):
        try:
            validate_se3(mounts[side], name=f"fixed-base mount[{side}]")
        except (ValueError, TypeError, RuntimeError) as exc:
            raise SceneStateProducerError(f"invalid fixed-base mount: {exc}") from exc
    base_raw = np.asarray(T_camera_base)
    if base_raw.dtype != np.float64 or base_raw.shape != (4, 4):
        raise SceneStateProducerError("fixed T_camera_base must be exact float64[4,4]")
    try:
        base = validate_se3(base_raw, name="fixed T_camera_base")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise SceneStateProducerError(
            f"fixed T_camera_base must be right-handed SE(3): {exc}"
        ) from exc
    if candidate is not None and not np.array_equal(candidate.matrix, base):
        raise SceneStateProducerError(
            "fixed camera-base matrix differs from its same-FD candidate descriptor"
        )

    lower, upper = _arm_limits(assets)
    targets = np.zeros_like(wrist_T_camera, dtype=np.float64)
    for frame, side in zip(*np.nonzero(valid)):
        try:
            wrist = validate_se3(
                wrist_T_camera[frame, side],
                name=f"fixed-base wrist_T_camera[{frame},{side}]",
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            raise SceneStateProducerError(
                f"invalid fixed-base wrist target: {exc}"
            ) from exc
        targets[frame, side] = wrist @ np.linalg.inv(mounts[side])

    q_arm = np.zeros((count, 2, 7), dtype=np.float64)
    position_stage_evaluations = np.zeros((count, 2), dtype=np.int32)
    full_pose_stage_evaluations = np.zeros((count, 2), dtype=np.int32)
    zero = np.zeros(7, dtype=np.float64)
    for frame, side in zip(*np.nonzero(valid)):
        position_q, position_nfev = _solve_one_arm_position_only(
            assets,
            side=int(side),
            base=base,
            target_tool=targets[frame, side],
            initial_q=zero.copy(),
            lower=lower[side],
            upper=upper[side],
        )
        solved, nfev = _solve_one_arm(
            assets,
            side=int(side),
            base=base,
            target_tool=targets[frame, side],
            initial_q=position_q.copy(),
            lower=lower[side],
            upper=upper[side],
        )
        q_arm[frame, side] = solved
        position_stage_evaluations[frame, side] = position_nfev
        full_pose_stage_evaluations[frame, side] = nfev

    if (
        np.any(position_stage_evaluations[valid] <= 0)
        or np.any(
            position_stage_evaluations[valid]
            > FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS
        )
        or np.any(full_pose_stage_evaluations[valid] <= 0)
        or np.any(
            full_pose_stage_evaluations[valid]
            > FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS
        )
    ):
        raise SceneStateProducerError(
            "fixed-base two-stage IK evaluation counts violate frozen limits"
        )
    evaluations = np.asarray(
        position_stage_evaluations + full_pose_stage_evaluations,
        dtype=np.int32,
    )

    position = np.full((count, 2), np.nan, dtype=np.float64)
    rotation = np.full((count, 2), np.nan, dtype=np.float64)
    for frame, side in zip(*np.nonzero(valid)):
        actual_hand = base @ _tool_fk(assets, side, q_arm[frame, side]) @ mounts[side]
        delta = np.linalg.inv(wrist_T_camera[frame, side]) @ actual_hand
        position[frame, side] = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
        rotation[frame, side] = float(
            np.linalg.norm(_rotation_vector(delta[:3, :3])) * 180.0 / np.pi
        )
    return SolverResult(
        q_arm=q_arm,
        T_camera_base=np.asarray(base, dtype=np.float64),
        position_residual_mm=position,
        rotation_residual_deg=rotation,
        evaluations=evaluations,
        scene_state_mode=SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER,
        solver_used=True,
        residual_claimed=True,
        development_only=True,
        fixed_camera_base_candidate=(
            _fixed_camera_base_candidate_manifest(candidate)
            if candidate is not None
            else None
        ),
        position_stage_evaluations=position_stage_evaluations,
        full_pose_stage_evaluations=full_pose_stage_evaluations,
    )


def fixed_base_gate_summary(
    *,
    position_residual_mm: np.ndarray,
    rotation_residual_deg: np.ndarray,
    valid: np.ndarray,
) -> dict[str, int | float | bool | str]:
    pair_denominator = int(np.count_nonzero(valid))
    frame_has_valid = np.any(valid, axis=1)
    frame_denominator = int(np.count_nonzero(frame_has_valid))
    if pair_denominator == 0 or frame_denominator == 0:
        raise SceneStateProducerError(
            "fixed camera-base gate requires at least one valid frame/side"
        )
    pair_pass_values = (position_residual_mm[valid] <= MAX_POSITION_RESIDUAL_MM) & (
        rotation_residual_deg[valid] <= MAX_ROTATION_RESIDUAL_DEG
    )
    pair_pass = np.zeros_like(valid, dtype=np.bool_)
    pair_pass[valid] = pair_pass_values
    frame_pass = frame_has_valid & np.all((~valid) | pair_pass, axis=1)
    pair_pass_count = int(np.count_nonzero(pair_pass_values))
    frame_pass_count = int(np.count_nonzero(frame_pass))
    return {
        "fixed_camera_base_optimized_or_refined": False,
        "per_valid_pair_initial_q": (
            "POSITION_STAGE_UNIFORM_ZERO_7DOF_THEN_FULL_POSE_FROM_SAME_PAIR_"
            "POSITION_RESULT_NO_CROSS_PAIR_WARM_START"
        ),
        "ik_gate_name": "IK_GATE_PASS_RATE",
        "ik_gate_position_threshold_mm": MAX_POSITION_RESIDUAL_MM,
        "ik_gate_rotation_threshold_deg": MAX_ROTATION_RESIDUAL_DEG,
        "ik_gate_pair_pass_count": pair_pass_count,
        "ik_gate_pair_denominator": pair_denominator,
        "ik_gate_pair_pass_rate": pair_pass_count / pair_denominator,
        "ik_gate_frame_all_valid_sides_pass_count": frame_pass_count,
        "ik_gate_frame_denominator": frame_denominator,
        "ik_gate_frame_pass_rate": frame_pass_count / frame_denominator,
        "failed_valid_pairs_deleted": False,
    }


def fixed_base_two_stage_summary(
    *,
    evaluations: np.ndarray,
    position_stage_evaluations: np.ndarray | None,
    full_pose_stage_evaluations: np.ndarray | None,
    valid: np.ndarray,
) -> dict[str, int | float | bool | str]:
    """Validate and summarize the exact fixed-base two-stage solver work."""

    expected_shape = valid.shape
    if valid.dtype != np.bool_ or not np.any(valid):
        raise SceneStateProducerError(
            "fixed-base two-stage summary requires exact non-empty bool valid"
        )
    if position_stage_evaluations is None or full_pose_stage_evaluations is None:
        raise SceneStateProducerError(
            "fixed-base two-stage solver lacks per-stage evaluation arrays"
        )
    arrays = {
        "total": np.asarray(evaluations),
        "position": np.asarray(position_stage_evaluations),
        "full_pose": np.asarray(full_pose_stage_evaluations),
    }
    if any(
        array.dtype != np.int32 or array.shape != expected_shape
        for array in arrays.values()
    ):
        raise SceneStateProducerError(
            "fixed-base two-stage evaluation arrays must be exact int32[N,2]"
        )
    if any(np.any(array[~valid] != 0) for array in arrays.values()):
        raise SceneStateProducerError(
            "fixed-base two-stage invalid pairs must have zero evaluations"
        )
    if (
        np.any(arrays["position"][valid] <= 0)
        or np.any(arrays["position"][valid] > FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS)
        or np.any(arrays["full_pose"][valid] <= 0)
        or np.any(
            arrays["full_pose"][valid] > FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS
        )
        or not np.array_equal(arrays["total"], arrays["position"] + arrays["full_pose"])
    ):
        raise SceneStateProducerError(
            "fixed-base two-stage evaluation arrays violate frozen stage limits or sum"
        )
    return {
        "ik_stage_count": 2,
        "position_stage_backend": "SCIPY_CPU_TRF",
        "position_stage_objective": "TOOL_TRANSLATION_3D_SCALED",
        "position_stage_initial_q": "UNIFORM_ZERO_7DOF_PER_VALID_PAIR",
        "position_stage_position_scale_m": POSITION_SCALE_M,
        "position_stage_ftol": IK_TRF_FTOL,
        "position_stage_xtol": IK_TRF_XTOL,
        "position_stage_gtol": IK_TRF_GTOL,
        "position_stage_max_evaluations": (FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS),
        "position_stage_function_evaluations_total": int(
            np.sum(arrays["position"][valid], dtype=np.int64)
        ),
        "position_stage_function_evaluations_max": int(
            np.max(arrays["position"][valid])
        ),
        "full_pose_stage_backend": "SCIPY_CPU_TRF",
        "full_pose_stage_objective": ("TOOL_SE3_TRANSLATION_AND_ROTATION_6D_SCALED"),
        "full_pose_stage_initial_q": "SAME_VALID_PAIR_POSITION_STAGE_RESULT",
        "full_pose_stage_position_scale_m": POSITION_SCALE_M,
        "full_pose_stage_rotation_scale_rad": float(ROTATION_SCALE_RAD),
        "full_pose_stage_ftol": IK_TRF_FTOL,
        "full_pose_stage_xtol": IK_TRF_XTOL,
        "full_pose_stage_gtol": IK_TRF_GTOL,
        "full_pose_stage_max_evaluations": (FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS),
        "full_pose_stage_function_evaluations_total": int(
            np.sum(arrays["full_pose"][valid], dtype=np.int64)
        ),
        "full_pose_stage_function_evaluations_max": int(
            np.max(arrays["full_pose"][valid])
        ),
        "ik_function_evaluations_total": int(
            np.sum(arrays["total"][valid], dtype=np.int64)
        ),
        "position_stage_result_used_only_by_same_pair": True,
        "cross_frame_side_or_candidate_warm_start": False,
    }


def encode_scene_state(
    *,
    session_id: str,
    sources: SourceBundle,
    mounts: ExplicitMountInput,
    solver: SolverResult,
) -> bytes:
    if solver.scene_state_mode in {
        SCENE_STATE_MODE_DEVELOPMENT_SOLVER,
        SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER,
    }:
        if not solver.solver_used or not solver.residual_claimed:
            raise SceneStateProducerError(
                "development solver result must record solver/residual use"
            )
        if solver.external_authority is not None or solver.timestamp_ns is not None:
            raise SceneStateProducerError(
                "development solver result cannot carry external authority lineage"
            )
        if (solver.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER) is (
            solver.fixed_camera_base_candidate is None
        ):
            raise SceneStateProducerError(
                "fixed solver mode must carry exactly one camera-base candidate lineage"
            )
        if solver.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER:
            fixed_base_two_stage_summary(
                evaluations=solver.evaluations,
                position_stage_evaluations=solver.position_stage_evaluations,
                full_pose_stage_evaluations=solver.full_pose_stage_evaluations,
                valid=sources.valid,
            )
        elif (
            solver.position_stage_evaluations is not None
            or solver.full_pose_stage_evaluations is not None
        ):
            raise SceneStateProducerError(
                "alternating solver result cannot carry fixed-base stage arrays"
            )
    elif solver.scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY:
        if solver.solver_used or solver.residual_claimed:
            raise SceneStateProducerError(
                "external authority result must record solver_not_used/residual_not_claimed"
            )
        if solver.external_authority is None or solver.timestamp_ns is None:
            raise SceneStateProducerError(
                "external authority result lacks descriptor/array/timestamp lineage"
            )
        if solver.fixed_camera_base_candidate is not None:
            raise SceneStateProducerError(
                "external authority result cannot carry fixed-base candidate lineage"
            )
        if (
            solver.position_stage_evaluations is not None
            or solver.full_pose_stage_evaluations is not None
        ):
            raise SceneStateProducerError(
                "external authority result cannot carry fixed-base stage arrays"
            )
    else:
        raise SceneStateProducerError("unsupported scene-state result mode")
    output = io.BytesIO()
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCENE_SCHEMA),
        "solver_schema": np.asarray(
            FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA
            if solver.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER
            else SOLVER_SCHEMA
        ),
        "session_id": np.asarray(session_id),
        "mount_provenance": np.asarray(MOUNT_PROVENANCE),
        "contact_infeasible": np.asarray(CONTACT_INFEASIBLE),
        "scene_state_mode": np.asarray(solver.scene_state_mode),
        "solver_used": np.asarray(solver.solver_used, dtype=np.bool_),
        "residual_claimed": np.asarray(solver.residual_claimed, dtype=np.bool_),
        "mount_descriptor_sha256": np.asarray(mounts.descriptor_sha256),
        "mount_evidence_mode": np.asarray(mounts.evidence_mode),
        "mount_authority_lineage_sha256": np.asarray(
            _canonical_json_sha256(_mount_authority_manifest(mounts))
        ),
        "mount_evidence_sha256_by_side": np.asarray(
            [
                str(mounts.evidence_records.get(side, {}).get("sha256", ""))
                for side in SIDES
            ]
        ),
        "frame_names": np.asarray(sources.frame_names),
        "q_arm": solver.q_arm,
        "q_hand": sources.q_hand,
        "valid": sources.valid,
        "wrist_T_camera": sources.wrist_T_camera,
        "T_camera_base": solver.T_camera_base,
        "T_tool_hand": mounts.matrices,
        "camera_intrinsics": sources.camera_intrinsics,
        "source_resolution": np.asarray(sources.source_resolution, dtype=np.int32),
        "arm_joint_names": np.asarray(ARM_JOINT_NAMES),
        "hand_joint_names": sources.hand_joint_names,
        "position_residual_mm": solver.position_residual_mm,
        "rotation_residual_deg": solver.rotation_residual_deg,
        "ik_function_evaluations": solver.evaluations,
    }
    if solver.scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY:
        assert solver.external_authority is not None
        assert solver.timestamp_ns is not None
        descriptor = solver.external_authority.get("descriptor")
        authority_arrays = solver.external_authority.get("arrays")
        direct_evidence = solver.external_authority.get("direct_evidence")
        urdf_refs = solver.external_authority.get("urdf_refs")
        source_refs = solver.external_authority.get("source_refs")
        if not isinstance(descriptor, Mapping) or not isinstance(
            authority_arrays, Mapping
        ):
            raise SceneStateProducerError(
                "external authority result lacks descriptor/array records"
            )
        if (
            not isinstance(direct_evidence, Mapping)
            or not isinstance(direct_evidence.get("camera_base"), Mapping)
            or not isinstance(direct_evidence.get("q_arm_by_side"), Mapping)
            or not isinstance(urdf_refs, Mapping)
            or not isinstance(source_refs, Mapping)
        ):
            raise SceneStateProducerError(
                "external authority result lacks direct evidence/source/URDF lineage"
            )
        q_evidence = direct_evidence["q_arm_by_side"]
        if not isinstance(q_evidence, Mapping) or set(q_evidence) != set(SIDES):
            raise SceneStateProducerError(
                "external authority result lacks bilateral q_arm evidence"
            )
        arrays.update(
            {
                "external_authority_schema": np.asarray(EXTERNAL_BASE_Q_SCHEMA),
                "external_authority_descriptor_sha256": np.asarray(
                    str(descriptor.get("sha256"))
                ),
                "external_authority_arrays_sha256": np.asarray(
                    str(authority_arrays.get("sha256"))
                ),
                "external_authority_lineage_sha256": np.asarray(
                    _canonical_json_sha256(dict(solver.external_authority))
                ),
                "external_base_evidence_sha256": np.asarray(
                    str(direct_evidence["camera_base"].get("sha256"))
                ),
                "external_q_arm_evidence_sha256_by_side": np.asarray(
                    [str(q_evidence[side].get("sha256")) for side in SIDES]
                ),
                "external_urdf_sha256": np.asarray(
                    [
                        str(urdf_refs[name].get("sha256"))
                        for name in ("robot_asset_pin", "tianji_urdf")
                    ]
                ),
                "external_source_sha256": np.asarray(
                    [
                        str(source_refs[name].get("sha256"))
                        for name in ("r2_sidecar", "hawor")
                    ]
                ),
                "timestamp_ns": np.asarray(solver.timestamp_ns, dtype=np.int64),
            }
        )
    elif solver.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER:
        assert solver.fixed_camera_base_candidate is not None
        assert solver.position_stage_evaluations is not None
        assert solver.full_pose_stage_evaluations is not None
        descriptor = solver.fixed_camera_base_candidate.get("descriptor")
        if not isinstance(descriptor, Mapping):
            raise SceneStateProducerError(
                "fixed camera-base solver lacks candidate descriptor record"
            )
        descriptor_ref = _validate_evidence_ref(
            descriptor, name="fixed camera-base candidate descriptor"
        )
        arrays.update(
            {
                "fixed_camera_base_candidate_schema": np.asarray(
                    FIXED_CAMERA_BASE_CANDIDATE_SCHEMA
                ),
                "fixed_camera_base_candidate_descriptor_sha256": np.asarray(
                    descriptor_ref["sha256"]
                ),
                "fixed_camera_base_candidate_lineage_sha256": np.asarray(
                    _canonical_json_sha256(dict(solver.fixed_camera_base_candidate))
                ),
                "ik_position_stage_function_evaluations": np.asarray(
                    solver.position_stage_evaluations, dtype=np.int32
                ),
                "ik_full_pose_stage_function_evaluations": np.asarray(
                    solver.full_pose_stage_evaluations, dtype=np.int32
                ),
            }
        )
    np.savez_compressed(output, **arrays)
    payload = output.getvalue()
    decode_scene_state(payload)
    return payload


def build_manifest(
    *,
    session_id: str,
    state_record: Mapping[str, Any],
    state: FullChainSceneState,
    mount_record: Mapping[str, Any],
    mount: ExplicitMountInput,
    sources: SourceBundle,
    solver: SolverResult,
    assets: PinnedRobotAssets,
    command: Sequence[str],
) -> dict[str, Any]:
    if (
        mount_record.get("sha256") != mount.descriptor_sha256
        or state.mount_descriptor_sha256 != mount.descriptor_sha256
        or state.mount_evidence_mode != mount.evidence_mode
    ):
        raise SceneStateProducerError(
            "mount descriptor/evidence mode does not join the encoded scene state"
        )
    expected_mount_evidence_sha = tuple(
        str(mount.evidence_records.get(side, {}).get("sha256", "")) for side in SIDES
    )
    if state.mount_evidence_sha256_by_side != expected_mount_evidence_sha:
        raise SceneStateProducerError(
            "mount side-evidence SHA does not join the encoded scene state"
        )
    external = solver.scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY
    fixed_camera_base = (
        solver.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER
    )
    if external:
        if (
            solver.solver_used
            or solver.residual_claimed
            or solver.external_authority is None
            or solver.timestamp_ns is None
            or not np.isnan(solver.position_residual_mm).all()
            or not np.isnan(solver.rotation_residual_deg).all()
            or np.any(solver.evaluations != 0)
        ):
            raise SceneStateProducerError(
                "external authority mode must have zero solver work and no residual claim"
            )
        position_summary: dict[str, float | None] = {
            "max_position_residual_mm": None,
            "max_rotation_residual_deg": None,
            "mean_position_residual_mm": None,
            "mean_rotation_residual_deg": None,
        }
    elif solver.scene_state_mode in {
        SCENE_STATE_MODE_DEVELOPMENT_SOLVER,
        SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER,
    }:
        if not solver.solver_used or not solver.residual_claimed:
            raise SceneStateProducerError(
                "development solver mode must record solver/residual use"
            )
        if fixed_camera_base:
            if solver.fixed_camera_base_candidate is None:
                raise SceneStateProducerError(
                    "fixed solver mode lacks camera-base candidate lineage"
                )
            candidate_base = np.asarray(
                solver.fixed_camera_base_candidate.get("T_camera_base"),
                dtype=np.float64,
            )
            if not np.array_equal(candidate_base, solver.T_camera_base):
                raise SceneStateProducerError(
                    "fixed camera-base lineage does not equal solver transform"
                )
        elif solver.fixed_camera_base_candidate is not None:
            raise SceneStateProducerError(
                "alternating solver cannot carry fixed camera-base lineage"
            )
        position = solver.position_residual_mm[sources.valid]
        rotation = solver.rotation_residual_deg[sources.valid]
        position_summary = {
            "max_position_residual_mm": float(np.max(position)),
            "max_rotation_residual_deg": float(np.max(rotation)),
            "mean_position_residual_mm": float(np.mean(position)),
            "mean_rotation_residual_deg": float(np.mean(rotation)),
        }
    else:
        raise SceneStateProducerError("unsupported scene-state manifest mode")
    development_only = bool(mount.development_only or solver.development_only)
    if external:
        solver_manifest: dict[str, Any] = {
            "schema_version": SOLVER_SCHEMA,
            "backend": "NOT_USED_EXTERNAL_BASE_Q_AUTHORITY",
            "solver_used": False,
            "residual_not_claimed": True,
            **position_summary,
            "joint_limit_violations": 0,
            "base_is_one_session_constant": True,
            "invalid_sides_interpolated": False,
            "mount_optimized_or_selected": False,
            "objective_and_reported_residual_use_same_r2_wrist_target": False,
            "reported_residual_is_independent_validation": False,
            "independent_camera_base_or_arm_image_authority_consumed": True,
            "structural_parameter_count": 0,
            "structural_pose_constraint_count": 0,
            "structural_unknown_minus_constraint_count": 0,
            "q_arm_and_camera_base_unique_or_authoritative": False,
        }
    elif fixed_camera_base:
        valid_count = int(np.count_nonzero(sources.valid))
        stage_summary = fixed_base_two_stage_summary(
            evaluations=solver.evaluations,
            position_stage_evaluations=solver.position_stage_evaluations,
            full_pose_stage_evaluations=solver.full_pose_stage_evaluations,
            valid=sources.valid,
        )
        solver_manifest = {
            "schema_version": FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA,
            "backend": (
                "SCIPY_CPU_TRF_FIXED_CAMERA_BASE_POSITION_THEN_FULL_POSE_PER_SIDE_FRAME"
            ),
            "solver_used": True,
            "residual_not_claimed": False,
            "outer_iterations": 0,
            "ik_max_evaluations": IK_MAX_EVALUATIONS,
            "position_scale_m": POSITION_SCALE_M,
            "rotation_scale_rad": float(ROTATION_SCALE_RAD),
            **position_summary,
            "joint_limit_violations": 0,
            "base_is_one_session_constant": True,
            "invalid_sides_interpolated": False,
            "mount_optimized_or_selected": False,
            "objective_and_reported_residual_use_same_r2_wrist_target": True,
            "reported_residual_is_independent_validation": False,
            "independent_camera_base_or_arm_image_authority_consumed": False,
            "structural_parameter_count": 7 * valid_count,
            "structural_pose_constraint_count": 6 * valid_count,
            "structural_unknown_minus_constraint_count": valid_count,
            "q_arm_and_camera_base_unique_or_authoritative": False,
            **stage_summary,
            **fixed_base_gate_summary(
                position_residual_mm=solver.position_residual_mm,
                rotation_residual_deg=solver.rotation_residual_deg,
                valid=sources.valid,
            ),
        }
    else:
        solver_manifest = {
            "schema_version": SOLVER_SCHEMA,
            "backend": "SCIPY_CPU_TRF_ALTERNATING_SESSION_CONSTANT_BASE",
            "solver_used": True,
            "residual_not_claimed": False,
            "outer_iterations": OUTER_ITERATIONS,
            "ik_max_evaluations": IK_MAX_EVALUATIONS,
            "position_scale_m": POSITION_SCALE_M,
            "rotation_scale_rad": float(ROTATION_SCALE_RAD),
            **position_summary,
            "joint_limit_violations": 0,
            "base_is_one_session_constant": True,
            "invalid_sides_interpolated": False,
            "mount_optimized_or_selected": False,
            "objective_and_reported_residual_use_same_r2_wrist_target": True,
            "reported_residual_is_independent_validation": False,
            "independent_camera_base_or_arm_image_authority_consumed": False,
            "structural_parameter_count": int(
                6 + 7 * int(np.count_nonzero(sources.valid))
            ),
            "structural_pose_constraint_count": int(
                6 * int(np.count_nonzero(sources.valid))
            ),
            "structural_unknown_minus_constraint_count": int(
                6 + int(np.count_nonzero(sources.valid))
            ),
            "q_arm_and_camera_base_unique_or_authoritative": False,
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": SCENE_MANIFEST_DEVELOPMENT_STATUS
        if development_only
        else SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS,
        "completion_mode": "ARTIFACT_EXISTS",
        "session_id": session_id,
        "frame_count": len(sources.frame_names),
        "mount_provenance": MOUNT_PROVENANCE,
        "contact_infeasible": CONTACT_INFEASIBLE,
        "visual_only": True,
        "development_only": development_only,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "ik_residual_is_independent_accuracy_evidence": False,
        "q_arm_and_camera_base_authoritative": False,
        "q_arm_and_camera_base_authority_input_consumed": external,
        "scene_state_mode": solver.scene_state_mode,
        "baseline_frozen": False,
        "gpu_calls": 0,
        "renderer_calls": 0,
        "pixels_produced": 0,
        "scene_state": dict(state_record),
        "explicit_mount_input": dict(mount_record),
        "mount_authority": _mount_authority_manifest(mount),
        "sources": {key: dict(value) for key, value in sources.records.items()},
        "tool_definition": dict(assets.tool_definition),
        "ik_tool_definition": dict(assets.tool_definition),
        "visual_fit_tool_definition": dict(assets.tool_definition),
        "solver": solver_manifest,
        "command_manifest": {
            "argv": list(command),
            "payload_kind": "PYTHON3_CPU_SCENE_STATE",
        },
        "success_evidence": [
            {
                **dict(state_record),
                "minimum_bytes": 1024,
                "verification": "SHA256_MATCH_AND_FULL_DECODE_SCENE_STATE",
            }
        ],
        "claim_limits": [
            "NO_PHYSICAL_MOUNT_CLAIM",
            "NO_COLLISION_OR_REACHABILITY_CLAIM",
            "NO_TRAINING_GROUND_TRUTH",
            "NO_REAL_ROBOT_EXECUTION",
            "NO_RENDERED_PIXELS",
            "NO_UNIQUE_Q_ARM_OR_CAMERA_BASE_CLAIM",
            (
                "SOLVER_NOT_USED_AND_RESIDUAL_NOT_CLAIMED"
                if external
                else "IK_RESIDUAL_IS_NOT_INDEPENDENT_VALIDATION"
            ),
        ],
    }
    if external:
        assert solver.external_authority is not None
        manifest["external_base_q_authority"] = dict(solver.external_authority)
    elif fixed_camera_base:
        assert solver.fixed_camera_base_candidate is not None
        manifest["fixed_camera_base_candidate"] = dict(
            solver.fixed_camera_base_candidate
        )
        manifest["claim_limits"].append(
            "FIXED_CAMERA_BASE_IS_DEVELOPMENT_CANDIDATE_NOT_AUTHORITY"
        )
        manifest["claim_limits"].append(
            "IK_GATE_PASS_RATE_IS_NOT_PHYSICAL_REACHABILITY_OR_COLLISION_PROOF"
        )
    validate_scene_manifest(
        manifest,
        state_record=state_record,
        state=state,
        cpu4_tool_definition=assets.tool_definition,
        renderer_tool_definition=assets.tool_definition,
    )
    return manifest


def preflight_scene_manifest(
    *,
    session_id: str,
    state_path: Path,
    state_payload: bytes,
    state: FullChainSceneState,
    mount_record: Mapping[str, Any],
    mount: ExplicitMountInput,
    sources: SourceBundle,
    solver: SolverResult,
    assets: PinnedRobotAssets,
    command: Sequence[str],
) -> None:
    """Validate the complete manifest semantics before publishing scene bytes.

    The final scene-state inode does not exist yet, so this preflight uses a
    deliberately non-published identity distinct from the mount descriptor.
    The caller must build the final manifest again from a same-FD readback of
    the newly published scene state.
    """

    mount_ref = _validate_evidence_ref(mount_record, name="explicit mount input")
    occupied_identities = {(int(mount_ref["device"]), int(mount_ref["inode"]))}
    if solver.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER:
        if solver.fixed_camera_base_candidate is None:
            raise SceneStateProducerError(
                "fixed solver preflight lacks camera-base candidate lineage"
            )
        candidate_ref = _validate_evidence_ref(
            solver.fixed_camera_base_candidate.get("descriptor"),
            name="fixed camera-base candidate descriptor",
        )
        occupied_identities.add(
            (int(candidate_ref["device"]), int(candidate_ref["inode"]))
        )
    # This identity exists only long enough to exercise manifest semantics before
    # an output inode exists.  Put it on a deterministic synthetic device number
    # outside every input identity instead of guessing ``mount inode + 1``: files
    # created next to one another commonly have adjacent inodes, and that guess
    # could falsely report a legitimate fixed-base descriptor as a scene alias.
    provisional_device = max(device for device, _ in occupied_identities) + 1
    provisional_state_ref = {
        "path": str(state_path),
        "bytes": len(state_payload),
        "sha256": sha256_bytes(state_payload),
        "device": provisional_device,
        "inode": 1,
    }
    build_manifest(
        session_id=session_id,
        state_record=provisional_state_ref,
        state=state,
        mount_record=mount_ref,
        mount=mount,
        sources=sources,
        solver=solver,
        assets=assets,
        command=command,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BASE_TRANSFORM_SEMANTICS",
    "EXTERNAL_ARRAY_DIGEST_CANONICALIZATION",
    "EXTERNAL_AUTHORITY_EVIDENCE_KIND",
    "EXTERNAL_AUTHORITY_METHOD",
    "EXTERNAL_BASE_EVIDENCE_KIND",
    "EXTERNAL_BASE_EVIDENCE_SCHEMA",
    "EXTERNAL_BASE_Q_ARRAY_SCHEMA",
    "EXTERNAL_BASE_Q_SCHEMA",
    "EXTERNAL_FRAME_TIMESTAMP_MAPPING",
    "EXTERNAL_Q_ARM_EVIDENCE_KIND",
    "EXTERNAL_Q_ARM_SIDE_EVIDENCE_SCHEMA",
    "ExplicitMountInput",
    "FIXED_CAMERA_BASE_CANDIDATE_SCHEMA",
    "FIXED_BASE_FULL_POSE_STAGE_MAX_EVALUATIONS",
    "FIXED_BASE_POSITION_STAGE_MAX_EVALUATIONS",
    "FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA",
    "FixedCameraBaseInput",
    "MOUNT_SCHEMA",
    "MOUNT_COORDINATE_DEFINITION",
    "MOUNT_MATRIX_DIRECTION",
    "MOUNT_SIDE_EVIDENCE_SCHEMA",
    "MOUNT_SOURCE_LINKS",
    "MOUNT_TARGET_LINKS",
    "MOUNT_TIME_SCOPE",
    "RIGHT_HANDED_AXES",
    "SceneStateProducerError",
    "SourceBundle",
    "SolverResult",
    "build_manifest",
    "encode_scene_state",
    "fixed_base_gate_summary",
    "fixed_base_two_stage_summary",
    "load_external_base_q_authority",
    "load_explicit_mount",
    "load_fixed_camera_base_candidate",
    "load_sources",
    "preflight_scene_manifest",
    "read_bound_bytes",
    "sha256_bytes",
    "solve_scene_state",
    "solve_scene_state_fixed_base",
]
