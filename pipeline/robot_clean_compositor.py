#!/usr/bin/env python3
"""Fail-closed, task-agnostic Robot-on-Clean compositor contract.

The compositor deliberately does not estimate motion, Object6D, masks, or
contact.  It consumes independently-authorised, SHA-bound products and joins
them by the frame IDs and timestamps frozen in the session manifest.  Large
raster products are represented by one NPZ bundle per frame so validation and
composition remain bounded by one frame rather than one video.

Formal inputs produce a *candidate* master that still requires downstream
codec and human review.  Development fixtures are visibly watermarked on every
frame and can never become consumable output.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

import cv2
import jsonschema
import numpy as np

from pipeline.depth_occlusion_v3 import range_to_metric_z, supersampled_intrinsics
from pipeline.robot_pbr_palette import CONFIG_ID, validate_robot_palette


INPUT_SCHEMA_VERSION = "chaoyang-robot-clean-compositor-input-v1"
SESSION_SCHEMA_VERSION = "chaoyang-robot-clean-session-v1"
OBJECT_DEPTH_MANIFEST_VERSION = "chaoyang-object-depth-frame-manifest-v1"
ROBOT_RENDER_MANIFEST_VERSION = "chaoyang-robot-render-frame-manifest-v1"
MASK_AUTHORITY_VERSION = "chaoyang-mask-object-depth-authority-v1"
CLEAN_AUTHORITY_VERSION = "chaoyang-clean-master-authority-v1"
ROBOT_AUTHORITY_VERSION = "chaoyang-robot-compositor-authority-v1"
NONPENETRATION_VERSION = "chaoyang-robot-nonpenetration-result-v1"
RESULT_SCHEMA_VERSION = "chaoyang-robot-clean-compositor-result-v1"

FORMAL_MODE = "FORMAL_CANDIDATE"
DEV_MODE = "DEV_SYNTHETIC"
OCCLUSION_EPSILON_M = 0.003
SUPERSAMPLE = 2
MAX_WIDTH = 4096
MAX_HEIGHT = 2160
MAX_FRAMES = 200_000
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_NPZ_FRAME_BYTES = 768 * 1024 * 1024
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class CompositorContractError(RuntimeError):
    """Raised before publication when any authority or payload drifts."""


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class VideoProbe:
    codec_name: str
    pix_fmt: str
    width: int
    height: int
    fps: Fraction
    frame_count_metadata: int | None
    audio_streams: int


@dataclass(frozen=True)
class PreparedContract:
    project_root: Path
    input_manifest_path: Path
    input_manifest_sha256: str
    value: dict[str, Any]
    session: dict[str, Any]
    task: dict[str, Any]
    object6d: dict[str, np.ndarray]
    camera: dict[str, np.ndarray]
    trajectory: dict[str, np.ndarray]
    object_depth_manifest: dict[str, Any]
    robot_render_manifest: dict[str, Any]
    clean_master: Path
    clean_probe: VideoProbe
    authorities: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class FrameComposite:
    bgr: np.ndarray
    object_front_subpixels: int
    robot_visible_alpha_sum: float
    robot_input_alpha_sum: float
    clean_background_fraction_sum: float


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact_ref(value: Mapping[str, Any], label: str) -> ArtifactRef:
    if set(value) != {"path", "bytes", "sha256"}:
        raise CompositorContractError(f"{label}: artifact keys must be path/bytes/sha256")
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if not isinstance(path, str) or not path:
        raise CompositorContractError(f"{label}: invalid artifact path")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise CompositorContractError(f"{label}: invalid artifact byte count")
    if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
        raise CompositorContractError(f"{label}: invalid SHA-256")
    return ArtifactRef(path=path, bytes=size, sha256=digest)


def _resolve_ref(project_root: Path, ref: ArtifactRef, label: str) -> Path:
    relative = Path(ref.path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise CompositorContractError(f"{label}: path must be normalized project-relative")
    path = project_root / relative
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise CompositorContractError(f"{label}: missing artifact {ref.path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CompositorContractError(f"{label}: artifact must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(project_root):
        raise CompositorContractError(f"{label}: artifact escapes project root")
    return resolved


def verify_artifact(
    project_root: Path,
    value: Mapping[str, Any],
    label: str,
    *,
    max_bytes: int | None = None,
) -> Path:
    ref = _artifact_ref(value, label)
    if max_bytes is not None and ref.bytes > max_bytes:
        raise CompositorContractError(f"{label}: artifact exceeds byte ceiling")
    path = _resolve_ref(project_root, ref, label)
    before = path.stat()
    if before.st_size != ref.bytes:
        raise CompositorContractError(
            f"{label}: byte mismatch expected={ref.bytes} actual={before.st_size}"
        )
    actual = sha256_file(path)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise CompositorContractError(f"{label}: artifact changed while hashing")
    if actual != ref.sha256:
        raise CompositorContractError(
            f"{label}: SHA mismatch expected={ref.sha256} actual={actual}"
        )
    return path


def _verified_bytes(
    project_root: Path,
    value: Mapping[str, Any],
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    path = verify_artifact(project_root, value, label, max_bytes=max_bytes)
    payload = path.read_bytes()
    ref = _artifact_ref(value, label)
    if len(payload) != ref.bytes or sha256_bytes(payload) != ref.sha256:
        raise CompositorContractError(f"{label}: changed between hash and read")
    return payload


def _verified_json(
    project_root: Path, value: Mapping[str, Any], label: str
) -> dict[str, Any]:
    payload = _verified_bytes(
        project_root, value, label, max_bytes=MAX_JSON_BYTES
    )
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompositorContractError(f"{label}: invalid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise CompositorContractError(f"{label}: JSON root must be an object")
    return decoded


def _verified_npz(
    project_root: Path,
    value: Mapping[str, Any],
    label: str,
    *,
    expected_keys: set[str],
    max_bytes: int = MAX_JSON_BYTES,
) -> dict[str, np.ndarray]:
    payload = _verified_bytes(project_root, value, label, max_bytes=max_bytes)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != expected_keys:
                raise CompositorContractError(
                    f"{label}: NPZ key closure mismatch missing={sorted(expected_keys - keys)} "
                    f"extra={sorted(keys - expected_keys)}"
                )
            result = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise CompositorContractError(f"{label}: invalid non-pickle NPZ") from exc
    return result


def _parse_fraction(value: Mapping[str, Any], label: str) -> Fraction:
    if set(value) != {"numerator", "denominator"}:
        raise CompositorContractError(f"{label}: FPS must have numerator/denominator")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or numerator <= 0
        or denominator <= 0
    ):
        raise CompositorContractError(f"{label}: FPS must be positive integers")
    return Fraction(numerator, denominator)


def _fraction_from_ffprobe(raw: str, label: str) -> Fraction:
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError) as exc:
        raise CompositorContractError(f"{label}: invalid ffprobe frame rate {raw!r}") from exc
    if value <= 0:
        raise CompositorContractError(f"{label}: nonpositive ffprobe frame rate")
    return value


def probe_video(path: Path) -> VideoProbe:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise CompositorContractError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CompositorContractError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise CompositorContractError(f"ffprobe stream list missing for {path}")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise CompositorContractError(f"expected exactly one video stream in {path}")
    stream = videos[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompositorContractError(f"video dimensions absent in {path}") from exc
    raw_count = stream.get("nb_frames")
    count = None
    if isinstance(raw_count, str) and raw_count.isdigit():
        count = int(raw_count)
    return VideoProbe(
        codec_name=str(stream.get("codec_name", "")),
        pix_fmt=str(stream.get("pix_fmt", "")),
        width=width,
        height=height,
        fps=_fraction_from_ffprobe(str(stream.get("avg_frame_rate", "")), str(path)),
        frame_count_metadata=count,
        audio_streams=len(audio),
    )


def _as_unicode_vector(value: np.ndarray, label: str) -> list[str]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind != "U":
        raise CompositorContractError(f"{label}: must be a one-dimensional Unicode array")
    decoded = [str(item) for item in array.tolist()]
    if any(not item for item in decoded) or len(decoded) != len(set(decoded)):
        raise CompositorContractError(f"{label}: IDs must be nonempty and unique")
    return decoded


def _as_sha_vector(value: np.ndarray, label: str, count: int) -> list[str]:
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind != "U":
        raise CompositorContractError(f"{label}: must be a Unicode[{count}] array")
    decoded = [str(item) for item in array.tolist()]
    # Multiple object IDs may intentionally share one measured geometry (for
    # example three cards using one metric CARD_THIN_BOX template).
    if any(_SHA_RE.fullmatch(item) is None for item in decoded):
        raise CompositorContractError(f"{label}: entries must be lowercase SHA-256")
    return decoded


def _as_int64_vector(value: np.ndarray, label: str, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype != np.int64:
        raise CompositorContractError(f"{label}: must be int64[{count}]")
    return array


def _validate_se3(value: np.ndarray, label: str) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-2:] != (4, 4) or not np.isfinite(array).all():
        raise CompositorContractError(f"{label}: transforms must be finite [...,4,4]")
    if not np.allclose(array[..., 3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise CompositorContractError(f"{label}: invalid homogeneous bottom row")
    rotation = array[..., :3, :3]
    identity = np.eye(3)
    if not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, identity, atol=1e-5):
        raise CompositorContractError(f"{label}: rotation is not orthonormal")
    determinant = np.linalg.det(rotation)
    if not np.allclose(determinant, 1.0, atol=1e-5):
        raise CompositorContractError(f"{label}: rotation is not right-handed")


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CompositorContractError(
            f"{label}: key closure mismatch missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _require_identity(value: Mapping[str, Any], root: Mapping[str, Any], label: str) -> None:
    for key in ("task_id", "session_id", "frame_count"):
        if value.get(key) != root.get(key):
            raise CompositorContractError(f"{label}: {key} identity mismatch")


def _authority_state(mode: str) -> tuple[str, str, bool]:
    if mode == FORMAL_MODE:
        return "PASS_FORMAL", "FULL_SESSION", True
    if mode == DEV_MODE:
        return "DEV_SYNTHETIC_ONLY", "SYNTHETIC", False
    raise CompositorContractError(f"unsupported mode {mode!r}")


def _validate_authority_header(
    value: Mapping[str, Any],
    root: Mapping[str, Any],
    schema_version: str,
    label: str,
) -> None:
    status, scope, authorized = _authority_state(str(root["mode"]))
    if value.get("schema_version") != schema_version:
        raise CompositorContractError(f"{label}: schema_version mismatch")
    _require_identity(value, root, label)
    if value.get("status") != status or value.get("scope") != scope:
        raise CompositorContractError(f"{label}: authority status/scope mismatch for mode")
    if value.get("consumption_authorized") is not authorized:
        raise CompositorContractError(f"{label}: consumption authority mismatch for mode")


def _validate_gate_map(
    value: Mapping[str, Any], names: Sequence[str], mode: str, label: str
) -> None:
    if set(value) != set(names):
        raise CompositorContractError(f"{label}: gate key closure mismatch")
    expected = "PASS" if mode == FORMAL_MODE else "DEV_SYNTHETIC"
    if any(value.get(name) != expected for name in names):
        raise CompositorContractError(f"{label}: all gates must be {expected}")


def _validate_main_constants(value: Mapping[str, Any]) -> None:
    coordinate = value["coordinate_contract"]
    expected = {
        "length_unit": "m",
        "matrix_convention": "column_vectors_left_multiply",
        "camera_axes": "+X_RIGHT_+Y_DOWN_+Z_FORWARD",
        "object_pose_field": "T_object_to_camera",
        "camera_pose_field": "T_camera_to_world",
        "object_world_consistency": "T_world_object=T_camera_to_world@T_object_to_camera",
        "robot_depth_field": "EUCLIDEAN_CAMERA_RANGE_M",
        "object_depth_field": "OPTICAL_AXIS_CAMERA_Z_M",
        "supersample": SUPERSAMPLE,
        "occlusion_epsilon_m": OCCLUSION_EPSILON_M,
        "compositing_space": "LINEAR_REC709_FROM_SRGB",
    }
    if coordinate != expected:
        raise CompositorContractError("coordinate_contract differs from frozen convention")
    mode = value["mode"]
    expected_watermark = "DEV_SYNTHETIC" if mode == DEV_MODE else "NONE"
    if value.get("watermark") != expected_watermark:
        raise CompositorContractError("watermark/mode contract mismatch")
    width = value["resolution"]["width"]
    height = value["resolution"]["height"]
    count = value["frame_count"]
    if width > MAX_WIDTH or height > MAX_HEIGHT or count > MAX_FRAMES:
        raise CompositorContractError("declared dimensions exceed hard resource ceilings")
    if mode == DEV_MODE and count > 24:
        raise CompositorContractError("DEV_SYNTHETIC is capped at 24 frames")


def _load_and_validate_task(
    project_root: Path, root: Mapping[str, Any], task_ref: Mapping[str, Any]
) -> dict[str, Any]:
    task = _verified_json(project_root, task_ref, "artifacts.task_manifest")
    schema_path = project_root / "tasks/schema/task_manifest.schema.json"
    try:
        jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(task)
    except jsonschema.ValidationError as exc:
        raise CompositorContractError(f"task manifest schema failure: {exc.message}") from exc
    task_id = root["task_id"]
    if task.get("task_id") != task_id:
        raise CompositorContractError("task manifest/task_id mismatch")
    expected_path = f"tasks/{task_id}/task.manifest.json"
    if task_ref.get("path") != expected_path:
        raise CompositorContractError("task manifest must be the canonical task-local manifest")
    robot_ref = task["system_refs"]["robot"]
    robot_system = project_root / robot_ref["path"]
    if not robot_system.is_file() or robot_system.is_symlink():
        raise CompositorContractError("task robot system manifest is missing/non-regular")
    if sha256_file(robot_system) != robot_ref["manifest_sha256"]:
        raise CompositorContractError("task robot system manifest pin has drifted")
    return task


def _validate_session(
    session: Mapping[str, Any], root: Mapping[str, Any], task: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    expected = {
        "schema_version",
        "mode",
        "task_id",
        "session_id",
        "frame_count",
        "frame_ids",
        "timestamp_ns",
        "fps",
        "resolution",
        "object_ids",
        "operated_object_ids",
        "support_object_ids",
        "formal_session",
        "coordinate_contract",
    }
    _require_keys(session, expected, "session_manifest")
    if session.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise CompositorContractError("session manifest schema_version mismatch")
    _require_identity(session, root, "session_manifest")
    if session.get("mode") != root["mode"]:
        raise CompositorContractError("session manifest mode mismatch")
    if session.get("fps") != root["fps"] or session.get("resolution") != root["resolution"]:
        raise CompositorContractError("session timing/resolution mismatch")
    if session.get("coordinate_contract") != root["coordinate_contract"]:
        raise CompositorContractError("session coordinate contract mismatch")
    expected_formal = root["mode"] == FORMAL_MODE
    if session.get("formal_session") is not expected_formal:
        raise CompositorContractError("formal_session/mode mismatch")
    count = root["frame_count"]
    frame_ids = np.asarray(session.get("frame_ids"))
    timestamps = np.asarray(session.get("timestamp_ns"))
    if frame_ids.shape != (count,) or frame_ids.dtype.kind not in "iu":
        raise CompositorContractError("session frame_ids must be an integer frame_count vector")
    frame_ids = frame_ids.astype(np.int64)
    if len(np.unique(frame_ids)) != count or np.any(np.diff(frame_ids) <= 0):
        raise CompositorContractError("session frame_ids must be unique and strictly increasing")
    if timestamps.shape != (count,) or timestamps.dtype.kind not in "iu":
        raise CompositorContractError("session timestamp_ns must be an integer frame_count vector")
    timestamps = timestamps.astype(np.int64)
    if np.any(timestamps <= 0) or np.any(np.diff(timestamps) <= 0):
        raise CompositorContractError("session timestamps must be positive and strictly increasing")
    object_ids = session.get("object_ids")
    operated = session.get("operated_object_ids")
    support = session.get("support_object_ids")
    if not isinstance(object_ids, list) or any(not isinstance(x, str) or not x for x in object_ids):
        raise CompositorContractError("session object_ids must be nonempty strings")
    if len(object_ids) != len(set(object_ids)) or not object_ids:
        raise CompositorContractError("session object_ids must be nonempty and unique")
    if not isinstance(operated, list) or not operated or not set(operated).issubset(object_ids):
        raise CompositorContractError("operated object IDs are invalid")
    if not isinstance(support, list) or not set(support).issubset(object_ids):
        raise CompositorContractError("support object IDs are invalid")
    if set(operated) & set(support):
        raise CompositorContractError("operated/support object IDs overlap")
    task_ids = task["task_config"]["object_contract"]["object_ids"]
    if object_ids != task_ids:
        raise CompositorContractError("session object IDs/order differ from task object contract")
    session_id = str(root["session_id"]).lower()
    if root["mode"] == FORMAL_MODE and ("synthetic" in session_id or "dev" in session_id):
        raise CompositorContractError("formal mode forbids development/synthetic session IDs")
    if root["mode"] == DEV_MODE and "synthetic" not in session_id:
        raise CompositorContractError("DEV_SYNTHETIC session ID must state synthetic")
    return frame_ids, timestamps, list(object_ids)


def _validate_global_npz(
    root: Mapping[str, Any],
    session_frame_ids: np.ndarray,
    timestamps: np.ndarray,
    object_ids: list[str],
    object6d: Mapping[str, np.ndarray],
    camera: Mapping[str, np.ndarray],
    trajectory: Mapping[str, np.ndarray],
) -> None:
    count = root["frame_count"]
    obj_frame = _as_int64_vector(object6d["frame_id"], "object6d.frame_id", count)
    obj_time = _as_int64_vector(object6d["timestamp_ns"], "object6d.timestamp_ns", count)
    if not np.array_equal(obj_frame, session_frame_ids) or not np.array_equal(obj_time, timestamps):
        raise CompositorContractError("Object6D frame/timestamp identity mismatch")
    ids = _as_unicode_vector(object6d["object_ids"], "object6d.object_ids")
    if ids != object_ids:
        raise CompositorContractError("Object6D object IDs/order mismatch")
    objects = len(object_ids)
    to_camera = np.asarray(object6d["T_object_to_camera"], dtype=np.float64)
    to_world = np.asarray(object6d["T_world_object"], dtype=np.float64)
    if to_camera.shape != (count, objects, 4, 4) or to_world.shape != (count, objects, 4, 4):
        raise CompositorContractError("Object6D transform shapes mismatch")
    _validate_se3(to_camera, "object6d.T_object_to_camera")
    _validate_se3(to_world, "object6d.T_world_object")
    valid = np.asarray(object6d["valid"])
    confidence = np.asarray(object6d["confidence"], dtype=np.float64)
    provenance = np.asarray(object6d["provenance"])
    _as_sha_vector(
        object6d["geometry_sha256"], "object6d.geometry_sha256", objects
    )
    if valid.shape != (count, objects) or valid.dtype != np.bool_:
        raise CompositorContractError("Object6D valid must be bool[N,M]")
    if confidence.shape != (count, objects) or not np.isfinite(confidence).all():
        raise CompositorContractError("Object6D confidence must be finite [N,M]")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise CompositorContractError("Object6D confidence outside [0,1]")
    if provenance.shape != (count, objects) or provenance.dtype.kind != "U":
        raise CompositorContractError("Object6D provenance must be Unicode[N,M]")
    allowed = {"DIRECT_TRACKED", "MODEL_FIT", "INTERPOLATED"}
    if not set(str(x) for x in provenance.reshape(-1)).issubset(allowed):
        raise CompositorContractError("Object6D contains missing/unknown provenance")
    if not valid.all():
        raise CompositorContractError("compositor requires valid per-frame Object6D for every object")

    cam_frame = _as_int64_vector(camera["frame_id"], "camera.frame_id", count)
    cam_time = _as_int64_vector(camera["timestamp_ns"], "camera.timestamp_ns", count)
    if not np.array_equal(cam_frame, session_frame_ids) or not np.array_equal(cam_time, timestamps):
        raise CompositorContractError("camera frame/timestamp identity mismatch")
    intrinsics = np.asarray(camera["K"], dtype=np.float64)
    c2w = np.asarray(camera["T_camera_to_world"], dtype=np.float64)
    if intrinsics.shape != (count, 3, 3) or not np.isfinite(intrinsics).all():
        raise CompositorContractError("camera.K must be finite [N,3,3]")
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise CompositorContractError("camera focal lengths must be positive")
    expected_last = np.broadcast_to((0.0, 0.0, 1.0), (count, 3))
    if not np.allclose(intrinsics[:, 2], expected_last, atol=1e-9):
        raise CompositorContractError("camera.K has invalid homogeneous row")
    if c2w.shape != (count, 4, 4):
        raise CompositorContractError("camera c2w shape mismatch")
    _validate_se3(c2w, "camera.T_camera_to_world")
    predicted_world = c2w[:, None] @ to_camera
    if not np.allclose(predicted_world, to_world, atol=1e-5):
        raise CompositorContractError("Object6D/c2w world transform consistency failed")

    traj_frame = _as_int64_vector(trajectory["frame_id"], "trajectory.frame_id", count)
    traj_time = _as_int64_vector(trajectory["timestamp_ns"], "trajectory.timestamp_ns", count)
    if not np.array_equal(traj_frame, session_frame_ids) or not np.array_equal(traj_time, timestamps):
        raise CompositorContractError("Robot trajectory frame/timestamp identity mismatch")
    for key in ("q_arm", "q_hand"):
        array = np.asarray(trajectory[key])
        if array.ndim != 3 or array.shape[0] != count or array.shape[1] != 2:
            raise CompositorContractError(f"trajectory.{key} must be [N,2,D]")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise CompositorContractError(f"trajectory.{key} must be finite floating point")


def _validate_frame_manifest(
    value: Mapping[str, Any],
    root: Mapping[str, Any],
    frame_ids: np.ndarray,
    version: str,
    label: str,
) -> None:
    if value.get("schema_version") != version:
        raise CompositorContractError(f"{label}: schema_version mismatch")
    _require_identity(value, root, label)
    if value.get("mode") != root["mode"]:
        raise CompositorContractError(f"{label}: mode mismatch")
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) != root["frame_count"]:
        raise CompositorContractError(f"{label}: frame list length mismatch")
    observed: list[int] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or set(frame) != {"frame_id", "bundle"}:
            raise CompositorContractError(f"{label}.frames[{index}]: invalid frame entry")
        if not isinstance(frame.get("frame_id"), int) or isinstance(frame.get("frame_id"), bool):
            raise CompositorContractError(f"{label}.frames[{index}]: frame_id must be integer")
        _artifact_ref(frame["bundle"], f"{label}.frames[{index}].bundle")
        observed.append(frame["frame_id"])
    if observed != frame_ids.tolist():
        raise CompositorContractError(f"{label}: frame IDs/order mismatch")


def _binding(value: Mapping[str, Any], name: str, ref: Mapping[str, Any], label: str) -> None:
    expected = ref["sha256"]
    if value.get(name) != expected:
        raise CompositorContractError(f"{label}: binding {name} mismatch")


def _validate_authorities(
    root: Mapping[str, Any],
    refs: Mapping[str, Mapping[str, Any]],
    values: dict[str, dict[str, Any]],
    object_ids: list[str],
) -> None:
    mode = root["mode"]
    mask = values["mask_authority"]
    clean = values["clean_authority"]
    robot = values["robot_authority"]
    nonpen = values["nonpenetration_result"]
    cross = values["cross_stage_admission"]

    _require_keys(
        mask,
        {
            "schema_version",
            "status",
            "scope",
            "consumption_authorized",
            "task_id",
            "session_id",
            "frame_count",
            "object_ids",
            "bindings",
            "gates",
            "claim_limit",
        },
        "mask_authority",
    )
    _require_keys(
        clean,
        {
            "schema_version",
            "status",
            "scope",
            "consumption_authorized",
            "is_formal_clean",
            "task_id",
            "session_id",
            "frame_count",
            "bindings",
            "codec",
            "gates",
            "claim_limit",
        },
        "clean_authority",
    )
    _require_keys(
        robot,
        {
            "schema_version",
            "status",
            "scope",
            "consumption_authorized",
            "sidecar_consumption_authorized",
            "task_id",
            "session_id",
            "frame_count",
            "bindings",
            "gates",
            "claim_limit",
        },
        "robot_authority",
    )
    _require_keys(
        nonpen,
        {
            "schema_version",
            "status",
            "task_id",
            "session_id",
            "frame_count",
            "frame_ids",
            "all_frames_evaluated",
            "bindings",
            "geometry_sha256_by_object",
            "gates",
            "max_nondesignated_penetration_m",
            "active_pad_signed_sdf_range_m",
            "claim_limit",
        },
        "nonpenetration_result",
    )
    _require_keys(
        cross,
        {
            "schema_version",
            "task_id",
            "session_id",
            "input_contract",
            "categories",
            "routes",
            "blame",
        },
        "cross_stage_admission",
    )

    _validate_authority_header(mask, root, MASK_AUTHORITY_VERSION, "mask_authority")
    if mask.get("object_ids") != object_ids:
        raise CompositorContractError("mask authority object IDs/order mismatch")
    _require_keys(
        mask["bindings"],
        {"object6d_sha256", "camera_sha256", "object_depth_frames_sha256"},
        "mask_authority.bindings",
    )
    _binding(mask["bindings"], "object6d_sha256", refs["object6d"], "mask_authority")
    _binding(mask["bindings"], "camera_sha256", refs["camera"], "mask_authority")
    _binding(
        mask["bindings"],
        "object_depth_frames_sha256",
        refs["object_depth_frames"],
        "mask_authority",
    )
    _validate_gate_map(
        mask["gates"],
        (
            "semantic_identity",
            "temporal_identity",
            "full_session_coverage",
            "object_protection",
            "object6d_projection_boundary",
            "metric_depth",
            "manual_review",
        ),
        mode,
        "mask_authority.gates",
    )

    _validate_authority_header(clean, root, CLEAN_AUTHORITY_VERSION, "clean_authority")
    if clean.get("is_formal_clean") is not (mode == FORMAL_MODE):
        raise CompositorContractError("clean authority formal flag mismatch")
    _require_keys(
        clean["bindings"],
        {"clean_master_sha256", "mask_authority_sha256", "provenance_manifest_sha256"},
        "clean_authority.bindings",
    )
    _binding(clean["bindings"], "clean_master_sha256", refs["clean_master"], "clean_authority")
    _binding(clean["bindings"], "mask_authority_sha256", refs["mask_authority"], "clean_authority")
    _binding(
        clean["bindings"],
        "provenance_manifest_sha256",
        refs["clean_provenance_manifest"],
        "clean_authority",
    )
    codec = clean.get("codec")
    expected_codec = {
        "codec_name": "ffv1",
        "lossless": True,
        "audio_streams": 0,
        "width": root["resolution"]["width"],
        "height": root["resolution"]["height"],
        "fps": root["fps"],
        "frame_count": root["frame_count"],
    }
    if codec != expected_codec:
        raise CompositorContractError("clean authority codec contract mismatch")
    _validate_gate_map(
        clean["gates"],
        (
            "input_mask_authority",
            "donor_authority",
            "source_map",
            "illumination",
            "shadow",
            "seam",
            "temporal",
            "object_protection",
            "codec",
            "manual_review",
        ),
        mode,
        "clean_authority.gates",
    )

    _validate_authority_header(robot, root, ROBOT_AUTHORITY_VERSION, "robot_authority")
    if robot.get("sidecar_consumption_authorized") is not (mode == FORMAL_MODE):
        raise CompositorContractError("robot sidecar authority mismatch")
    _require_keys(
        robot["bindings"],
        {
            "trajectory_sha256",
            "render_frames_sha256",
            "camera_sha256",
            "object6d_sha256",
            "pbr_config_sha256",
            "nonpenetration_result_sha256",
        },
        "robot_authority.bindings",
    )
    for name, ref_name in (
        ("trajectory_sha256", "robot_trajectory"),
        ("render_frames_sha256", "robot_render_frames"),
        ("camera_sha256", "camera"),
        ("object6d_sha256", "object6d"),
        ("pbr_config_sha256", "pbr_config"),
        ("nonpenetration_result_sha256", "nonpenetration_result"),
    ):
        _binding(robot["bindings"], name, refs[ref_name], "robot_authority")
    _validate_gate_map(
        robot["gates"],
        (
            "mount_calibration",
            "session_static_base",
            "pose",
            "joint_limits",
            "true_dt_temporal",
            "functional_retarget",
            "contact",
            "nonpenetration",
            "self_collision",
            "render_binding",
        ),
        mode,
        "robot_authority.gates",
    )

    if nonpen.get("schema_version") != NONPENETRATION_VERSION:
        raise CompositorContractError("nonpenetration schema_version mismatch")
    _require_identity(nonpen, root, "nonpenetration_result")
    expected_status = "PASS_FORMAL" if mode == FORMAL_MODE else "DEV_SYNTHETIC_ONLY"
    if nonpen.get("status") != expected_status:
        raise CompositorContractError("nonpenetration status mismatch")
    if nonpen.get("frame_ids") != values["session_manifest"]["frame_ids"]:
        raise CompositorContractError("nonpenetration frame coverage mismatch")
    if nonpen.get("all_frames_evaluated") is not True:
        raise CompositorContractError("nonpenetration must evaluate every frame")
    _require_keys(
        nonpen["bindings"],
        {"trajectory_sha256", "object6d_sha256"},
        "nonpenetration_result.bindings",
    )
    for name, ref_name in (
        ("trajectory_sha256", "robot_trajectory"),
        ("object6d_sha256", "object6d"),
    ):
        _binding(nonpen["bindings"], name, refs[ref_name], "nonpenetration_result")
    if nonpen.get("geometry_sha256_by_object") != values["object6d_geometry_map"]:
        raise CompositorContractError("nonpenetration geometry/object binding mismatch")
    expected_nonpen = {
        "dense_mesh_nonpenetration": True,
        "designated_contact_signed_sdf": True,
        "distinct_finger_self_sat": True,
        "joint_limits": True,
        "true_dt_velocity_acceleration": True,
    }
    if nonpen.get("gates") != expected_nonpen:
        raise CompositorContractError("nonpenetration gates are incomplete")
    if float(nonpen.get("max_nondesignated_penetration_m", math.inf)) > 0.0:
        raise CompositorContractError("non-designated penetration exceeds zero")
    if nonpen.get("active_pad_signed_sdf_range_m") != [0.0, 0.003]:
        raise CompositorContractError("active contact SDF range differs from [0,3] mm")

    if cross.get("schema_version") != "chaoyang-cross-stage-admission-v1":
        raise CompositorContractError("cross-stage admission schema mismatch")
    if cross.get("task_id") != root["task_id"] or cross.get("session_id") != root["session_id"]:
        raise CompositorContractError("cross-stage admission identity mismatch")
    routes = cross.get("routes")
    if not isinstance(routes, Mapping):
        raise CompositorContractError("cross-stage admission routes missing")
    if mode == FORMAL_MODE:
        if cross.get("input_contract", {}).get("valid") is not True:
            raise CompositorContractError("formal cross-stage input contract is invalid")
        required_routes = (
            "mask_formal_ready",
            "clean_formal_ready",
            "robot_sidecar_ready",
            "robot_composite_execution_allowed",
        )
        if any(routes.get(name) is not True for name in required_routes):
            raise CompositorContractError("formal cross-stage prerequisites are not admitted")
    else:
        if any(
            routes.get(name) is True
            for name in (
                "mask_formal_ready",
                "clean_formal_ready",
                "robot_formal_ready",
                "humanego_train_eligible",
            )
        ):
            raise CompositorContractError("DEV_SYNTHETIC cross-stage file claims formal readiness")


def _validate_render_manifest_bindings(
    root: Mapping[str, Any], refs: Mapping[str, Mapping[str, Any]], value: Mapping[str, Any]
) -> None:
    expected = {
        "rgb_storage": "SRGB_PNG_DECODED_BGR8",
        "range_storage": "EUCLIDEAN_CAMERA_RANGE_M_FLOAT32",
        "alpha_storage": "STRAIGHT_COVERAGE_FLOAT32_0_TO_1",
        "supersample": SUPERSAMPLE,
        "pbr_config_id": CONFIG_ID,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise CompositorContractError(f"robot_render_frames: {key} mismatch")
    if value.get("resolution") != root["resolution"]:
        raise CompositorContractError("robot_render_frames resolution mismatch")
    _require_keys(
        value,
        {
            "schema_version",
            "mode",
            "task_id",
            "session_id",
            "frame_count",
            "resolution",
            "rgb_storage",
            "range_storage",
            "alpha_storage",
            "supersample",
            "pbr_config_id",
            "bindings",
            "frames",
        },
        "robot_render_frames",
    )
    _require_keys(
        value["bindings"],
        {
            "trajectory_sha256",
            "camera_sha256",
            "pbr_config_sha256",
            "nonpenetration_result_sha256",
        },
        "robot_render_frames.bindings",
    )
    for name, ref_name in (
        ("trajectory_sha256", "robot_trajectory"),
        ("camera_sha256", "camera"),
        ("pbr_config_sha256", "pbr_config"),
        ("nonpenetration_result_sha256", "nonpenetration_result"),
    ):
        _binding(value["bindings"], name, refs[ref_name], "robot_render_frames")


def _validate_depth_manifest_bindings(
    root: Mapping[str, Any],
    refs: Mapping[str, Mapping[str, Any]],
    value: Mapping[str, Any],
    object_ids: list[str],
) -> None:
    expected = {
        "depth_storage": "OPTICAL_AXIS_CAMERA_Z_M_FLOAT32",
        "visible_mask_storage": "BOOL_INDEPENDENT_MASK_AUTHORITY",
        "object_index_storage": "INT16_ZERO_BACKGROUND_POSITIVE_REGISTRY",
        "supersample": SUPERSAMPLE,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise CompositorContractError(f"object_depth_frames: {key} mismatch")
    if value.get("resolution") != root["resolution"]:
        raise CompositorContractError("object_depth_frames resolution mismatch")
    _require_keys(
        value,
        {
            "schema_version",
            "mode",
            "task_id",
            "session_id",
            "frame_count",
            "resolution",
            "depth_storage",
            "visible_mask_storage",
            "object_index_storage",
            "supersample",
            "object_index_registry",
            "bindings",
            "frames",
        },
        "object_depth_frames",
    )
    _require_keys(
        value["bindings"],
        {"object6d_sha256", "camera_sha256"},
        "object_depth_frames.bindings",
    )
    registry = value.get("object_index_registry")
    if not isinstance(registry, list) or len(registry) != len(object_ids):
        raise CompositorContractError("object depth index registry length mismatch")
    expected_registry = [
        {"index": index + 1, "object_id": object_id}
        for index, object_id in enumerate(object_ids)
    ]
    if registry != expected_registry:
        raise CompositorContractError("object depth index registry/order mismatch")
    _binding(value["bindings"], "object6d_sha256", refs["object6d"], "object_depth_frames")
    _binding(value["bindings"], "camera_sha256", refs["camera"], "object_depth_frames")


def _validate_clean_video(
    path: Path, root: Mapping[str, Any], clean_authority: Mapping[str, Any]
) -> VideoProbe:
    probe = probe_video(path)
    expected_fps = _parse_fraction(root["fps"], "fps")
    if probe.codec_name != "ffv1":
        raise CompositorContractError("Clean master must be FFV1 lossless")
    if probe.audio_streams != 0:
        raise CompositorContractError("Clean master must not contain audio")
    resolution = root["resolution"]
    if (probe.width, probe.height) != (resolution["width"], resolution["height"]):
        raise CompositorContractError("Clean master resolution mismatch")
    if probe.fps != expected_fps:
        raise CompositorContractError(
            f"Clean master FPS mismatch expected={expected_fps} actual={probe.fps}"
        )
    if probe.frame_count_metadata not in (None, root["frame_count"]):
        raise CompositorContractError("Clean master metadata frame count mismatch")
    codec = clean_authority["codec"]
    if codec["codec_name"] != probe.codec_name or codec["audio_streams"] != probe.audio_streams:
        raise CompositorContractError("Clean authority disagrees with probed codec")
    return probe


def prepare_contract(input_manifest_path: Path, project_root: Path) -> PreparedContract:
    """Fully preflight all global and per-frame artifacts without writing output."""

    root_path = project_root.resolve(strict=True)
    try:
        original_info = input_manifest_path.lstat()
    except FileNotFoundError as exc:
        raise CompositorContractError("input manifest is missing") from exc
    if stat.S_ISLNK(original_info.st_mode) or not stat.S_ISREG(original_info.st_mode):
        raise CompositorContractError("input manifest must be a regular non-symlink file")
    manifest_path = input_manifest_path.resolve(strict=True)
    if not manifest_path.is_relative_to(root_path):
        raise CompositorContractError("input manifest must be a regular file inside project root")
    payload = manifest_path.read_bytes()
    if len(payload) > MAX_JSON_BYTES:
        raise CompositorContractError("input manifest exceeds JSON byte ceiling")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompositorContractError("input manifest is invalid UTF-8 JSON") from exc
    schema_path = root_path / "contracts/robot_clean_compositor_input_v1.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(value)
    except jsonschema.ValidationError as exc:
        raise CompositorContractError(f"input schema failure: {exc.message}") from exc
    if value["schema_version"] != INPUT_SCHEMA_VERSION:
        raise CompositorContractError("input schema version mismatch")
    _validate_main_constants(value)
    refs = value["artifacts"]
    task = _load_and_validate_task(root_path, value, refs["task_manifest"])
    session = _verified_json(root_path, refs["session_manifest"], "artifacts.session_manifest")
    frame_ids, timestamps, object_ids = _validate_session(session, value, task)

    object6d = _verified_npz(
        root_path,
        refs["object6d"],
        "artifacts.object6d",
        expected_keys={
            "frame_id",
            "timestamp_ns",
            "object_ids",
            "T_object_to_camera",
            "T_world_object",
            "valid",
            "confidence",
            "provenance",
            "geometry_sha256",
        },
    )
    camera = _verified_npz(
        root_path,
        refs["camera"],
        "artifacts.camera",
        expected_keys={"frame_id", "timestamp_ns", "K", "T_camera_to_world"},
    )
    trajectory = _verified_npz(
        root_path,
        refs["robot_trajectory"],
        "artifacts.robot_trajectory",
        expected_keys={"frame_id", "timestamp_ns", "q_arm", "q_hand"},
    )
    _validate_global_npz(
        value, frame_ids, timestamps, object_ids, object6d, camera, trajectory
    )
    geometry_map = dict(
        zip(
            object_ids,
            _as_sha_vector(object6d["geometry_sha256"], "geometry", len(object_ids)),
            strict=True,
        )
    )

    object_depth_manifest = _verified_json(
        root_path, refs["object_depth_frames"], "artifacts.object_depth_frames"
    )
    robot_render_manifest = _verified_json(
        root_path, refs["robot_render_frames"], "artifacts.robot_render_frames"
    )
    _validate_frame_manifest(
        object_depth_manifest,
        value,
        frame_ids,
        OBJECT_DEPTH_MANIFEST_VERSION,
        "object_depth_frames",
    )
    _validate_frame_manifest(
        robot_render_manifest,
        value,
        frame_ids,
        ROBOT_RENDER_MANIFEST_VERSION,
        "robot_render_frames",
    )
    _validate_depth_manifest_bindings(value, refs, object_depth_manifest, object_ids)
    _validate_render_manifest_bindings(value, refs, robot_render_manifest)

    authority_names = (
        "session_manifest",
        "mask_authority",
        "clean_authority",
        "robot_authority",
        "nonpenetration_result",
        "cross_stage_admission",
        "clean_provenance_manifest",
    )
    authorities = {"session_manifest": session}
    for name in authority_names[1:]:
        authorities[name] = _verified_json(root_path, refs[name], f"artifacts.{name}")
    authorities["object6d_geometry_map"] = geometry_map
    _validate_authorities(value, refs, authorities, object_ids)

    provenance = authorities["clean_provenance_manifest"]
    _require_keys(
        provenance,
        {
            "schema_version",
            "status",
            "scope",
            "consumption_authorized",
            "task_id",
            "session_id",
            "frame_count",
            "frame_ids",
            "source_map_scope",
            "claim_limit",
        },
        "clean_provenance_manifest",
    )
    expected_provenance = (
        ("PASS_FORMAL", "FULL_SESSION", True, "FULL_SESSION_PIXEL_PROVENANCE_AUTHORITY")
        if value["mode"] == FORMAL_MODE
        else ("DEV_SYNTHETIC_ONLY", "SYNTHETIC", False, "SYNTHETIC_FIXTURE_NO_REAL_PIXEL_CLAIM")
    )
    if (
        provenance.get("schema_version") != "chaoyang-clean-provenance-manifest-v1"
        or provenance.get("status") != expected_provenance[0]
        or provenance.get("scope") != expected_provenance[1]
        or provenance.get("consumption_authorized") is not expected_provenance[2]
        or provenance.get("source_map_scope") != expected_provenance[3]
        or provenance.get("task_id") != value["task_id"]
        or provenance.get("session_id") != value["session_id"]
        or provenance.get("frame_count") != value["frame_count"]
        or provenance.get("frame_ids") != frame_ids.tolist()
    ):
        raise CompositorContractError("Clean provenance manifest identity/coverage mismatch")

    pbr = _verified_json(root_path, refs["pbr_config"], "artifacts.pbr_config")
    try:
        validate_robot_palette(pbr)
    except ValueError as exc:
        raise CompositorContractError(f"PBR authority invalid: {exc}") from exc
    canonical_pbr = "systems/robot/configs/robot_pbr_palette_004ref_v1.json"
    if refs["pbr_config"]["path"] != canonical_pbr:
        raise CompositorContractError("PBR config must be the canonical shared authority")

    clean_master = verify_artifact(root_path, refs["clean_master"], "artifacts.clean_master")
    clean_probe = _validate_clean_video(clean_master, value, authorities["clean_authority"])

    prepared = PreparedContract(
        project_root=root_path,
        input_manifest_path=manifest_path,
        input_manifest_sha256=sha256_bytes(payload),
        value=dict(value),
        session=dict(session),
        task=dict(task),
        object6d=object6d,
        camera=camera,
        trajectory=trajectory,
        object_depth_manifest=dict(object_depth_manifest),
        robot_render_manifest=dict(robot_render_manifest),
        clean_master=clean_master,
        clean_probe=clean_probe,
        authorities=authorities,
    )
    preflight_frame_payloads(prepared)
    return prepared


def _decode_frame_bundle(
    prepared: PreparedContract,
    frame_entry: Mapping[str, Any],
    label: str,
    expected_keys: set[str],
) -> dict[str, np.ndarray]:
    return _verified_npz(
        prepared.project_root,
        frame_entry["bundle"],
        label,
        expected_keys=expected_keys,
        max_bytes=MAX_NPZ_FRAME_BYTES,
    )


def load_depth_frame(prepared: PreparedContract, slot: int) -> dict[str, np.ndarray]:
    entry = prepared.object_depth_manifest["frames"][slot]
    bundle = _decode_frame_bundle(
        prepared,
        entry,
        f"object_depth_frames.frames[{slot}].bundle",
        {
            "frame_id",
            "object_depth_near_m_2x",
            "object_depth_far_m_2x",
            "object_index_2x",
            "visible_object_mask_2x",
        },
    )
    frame_id = np.asarray(bundle["frame_id"])
    if frame_id.shape != () or frame_id.dtype != np.int64 or int(frame_id) != entry["frame_id"]:
        raise CompositorContractError(f"depth frame slot {slot}: embedded frame ID mismatch")
    height = prepared.value["resolution"]["height"] * SUPERSAMPLE
    width = prepared.value["resolution"]["width"] * SUPERSAMPLE
    shape = (height, width)
    near = np.asarray(bundle["object_depth_near_m_2x"])
    far = np.asarray(bundle["object_depth_far_m_2x"])
    index = np.asarray(bundle["object_index_2x"])
    visible = np.asarray(bundle["visible_object_mask_2x"])
    if near.shape != shape or far.shape != shape or near.dtype != np.float32 or far.dtype != np.float32:
        raise CompositorContractError(f"depth frame slot {slot}: depth must be float32 fixed-2x")
    if index.shape != shape or index.dtype != np.int16:
        raise CompositorContractError(f"depth frame slot {slot}: object index must be int16 fixed-2x")
    if visible.shape != shape or visible.dtype != np.bool_:
        raise CompositorContractError(f"depth frame slot {slot}: visible mask must be bool fixed-2x")
    finite = np.isfinite(near)
    if not np.array_equal(finite, np.isfinite(far)):
        raise CompositorContractError(f"depth frame slot {slot}: near/far validity differs")
    if np.any(near[finite] <= 0.0) or np.any(far[finite] < near[finite]):
        raise CompositorContractError(f"depth frame slot {slot}: invalid positive ordered depth")
    if not np.array_equal(index > 0, finite):
        raise CompositorContractError(f"depth frame slot {slot}: object index/depth support differs")
    max_index = len(prepared.session["object_ids"])
    if np.any(index < 0) or np.any(index > max_index):
        raise CompositorContractError(f"depth frame slot {slot}: unregistered object index")
    if np.any(visible & ~finite):
        raise CompositorContractError(f"depth frame slot {slot}: visible mask escapes metric depth")
    return bundle


def load_robot_frame(prepared: PreparedContract, slot: int) -> dict[str, np.ndarray]:
    entry = prepared.robot_render_manifest["frames"][slot]
    bundle = _decode_frame_bundle(
        prepared,
        entry,
        f"robot_render_frames.frames[{slot}].bundle",
        {"frame_id", "robot_bgr_2x", "robot_range_m_2x", "robot_alpha_2x"},
    )
    frame_id = np.asarray(bundle["frame_id"])
    if frame_id.shape != () or frame_id.dtype != np.int64 or int(frame_id) != entry["frame_id"]:
        raise CompositorContractError(f"robot frame slot {slot}: embedded frame ID mismatch")
    height = prepared.value["resolution"]["height"] * SUPERSAMPLE
    width = prepared.value["resolution"]["width"] * SUPERSAMPLE
    shape = (height, width)
    bgr = np.asarray(bundle["robot_bgr_2x"])
    range_m = np.asarray(bundle["robot_range_m_2x"])
    alpha = np.asarray(bundle["robot_alpha_2x"])
    if bgr.shape != (*shape, 3) or bgr.dtype != np.uint8:
        raise CompositorContractError(f"robot frame slot {slot}: RGB must be uint8 fixed-2x")
    if range_m.shape != shape or range_m.dtype != np.float32:
        raise CompositorContractError(f"robot frame slot {slot}: Range must be float32 fixed-2x")
    if alpha.shape != shape or alpha.dtype != np.float32:
        raise CompositorContractError(f"robot frame slot {slot}: alpha must be float32 fixed-2x")
    if not np.isfinite(alpha).all() or np.any((alpha < 0.0) | (alpha > 1.0)):
        raise CompositorContractError(f"robot frame slot {slot}: alpha outside [0,1]")
    support = alpha > 0.0
    if np.any(~np.isfinite(range_m[support])) or np.any(range_m[support] <= 0.0):
        raise CompositorContractError(f"robot frame slot {slot}: invalid Range under nonzero alpha")
    return bundle


def _decode_video_frame_count(path: Path, expected: int) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise CompositorContractError(f"cannot open Clean master {path}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise CompositorContractError(f"Clean master invalid decoded frame {count}")
            count += 1
            if count > expected:
                raise CompositorContractError("Clean master contains extra frames")
    finally:
        capture.release()
    if count != expected:
        raise CompositorContractError(
            f"Clean master decoded frame count mismatch expected={expected} actual={count}"
        )


def preflight_frame_payloads(prepared: PreparedContract) -> None:
    """Hash and validate every per-frame payload before output creation."""

    for slot in range(prepared.value["frame_count"]):
        load_depth_frame(prepared, slot)
        load_robot_frame(prepared, slot)
    _decode_video_frame_count(prepared.clean_master, prepared.value["frame_count"])
    # Detect a concurrent clean-master mutation across ffprobe/decode.
    clean_ref = prepared.value["artifacts"]["clean_master"]
    verify_artifact(prepared.project_root, clean_ref, "artifacts.clean_master.post_decode")


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    normalized = np.asarray(value, dtype=np.float32) / 255.0
    return np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        ((normalized + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb8(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)


def compose_frame(
    *,
    clean_bgr: np.ndarray,
    robot_bgr_2x: np.ndarray,
    robot_range_m_2x: np.ndarray,
    robot_alpha_2x: np.ndarray,
    object_depth_near_m_2x: np.ndarray,
    object_depth_far_m_2x: np.ndarray,
    object_index_2x: np.ndarray,
    visible_object_mask_2x: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> FrameComposite:
    """Depth-test and alpha-composite one frame in linear Rec.709.

    The independently accepted object mask is a consistency gate.  Amodal
    Object6D depth decides foreground ordering, while the Clean master supplies
    real/reconstructed object pixels.  This prevents a robot link from painting
    over an object that is closer to the camera.
    """

    clean = np.asarray(clean_bgr)
    if clean.ndim != 3 or clean.shape[2] != 3 or clean.dtype != np.uint8:
        raise CompositorContractError("decoded Clean frame must be uint8 HxWx3")
    height, width = clean.shape[:2]
    high_shape = (height * SUPERSAMPLE, width * SUPERSAMPLE)
    robot = np.asarray(robot_bgr_2x)
    range_m = np.asarray(robot_range_m_2x, dtype=np.float64)
    alpha = np.asarray(robot_alpha_2x, dtype=np.float64)
    near = np.asarray(object_depth_near_m_2x, dtype=np.float64)
    far = np.asarray(object_depth_far_m_2x, dtype=np.float64)
    index = np.asarray(object_index_2x)
    visible = np.asarray(visible_object_mask_2x)
    if robot.shape != (*high_shape, 3) or robot.dtype != np.uint8:
        raise CompositorContractError("robot BGR shape/dtype mismatch")
    if any(array.shape != high_shape for array in (range_m, alpha, near, far, index, visible)):
        raise CompositorContractError("frame buffers do not share fixed 2x dimensions")
    if not np.isfinite(alpha).all() or np.any((alpha < 0.0) | (alpha > 1.0)):
        raise CompositorContractError("robot alpha must be finite in [0,1]")
    object_support = np.isfinite(near)
    if not np.array_equal(object_support, np.isfinite(far)):
        raise CompositorContractError("object near/far validity differs")
    if np.any(near[object_support] <= 0.0) or np.any(far[object_support] < near[object_support]):
        raise CompositorContractError("object depth must be positive and ordered")
    if not np.array_equal(index > 0, object_support) or np.any(visible & ~object_support):
        raise CompositorContractError("object index/mask/depth support mismatch")
    intrinsics_2x = supersampled_intrinsics(np.asarray(camera_intrinsics, dtype=np.float64))
    robot_z = range_to_metric_z(
        range_m,
        intrinsics_2x[0, 0],
        intrinsics_2x[1, 1],
        intrinsics_2x[0, 2],
        intrinsics_2x[1, 2],
    )
    robot_support = alpha > 0.0
    if np.any(~np.isfinite(robot_z[robot_support])) or np.any(robot_z[robot_support] <= 0.0):
        raise CompositorContractError("robot Range invalid wherever alpha is nonzero")
    object_front = object_support & (
        (~robot_support) | (near <= robot_z + OCCLUSION_EPSILON_M)
    )
    visible_robot_alpha = np.where(object_front, 0.0, alpha)

    clean_2x = np.repeat(np.repeat(clean, SUPERSAMPLE, axis=0), SUPERSAMPLE, axis=1)
    clean_linear = _srgb_to_linear(clean_2x)
    robot_linear = _srgb_to_linear(robot)
    mixed = (
        robot_linear * visible_robot_alpha[..., None]
        + clean_linear * (1.0 - visible_robot_alpha[..., None])
    )
    # Object-front pixels are explicitly restored from the admitted Clean
    # master.  This assignment is redundant when alpha is zeroed, but makes the
    # occlusion source decision auditable and future-proof.
    mixed[object_front] = clean_linear[object_front]
    downsampled = mixed.reshape(height, SUPERSAMPLE, width, SUPERSAMPLE, 3).mean(axis=(1, 3))
    bgr = _linear_to_srgb8(downsampled)
    clean_fraction = np.where(object_front, 1.0, 1.0 - visible_robot_alpha)
    return FrameComposite(
        bgr=bgr,
        object_front_subpixels=int(np.count_nonzero(object_front)),
        robot_visible_alpha_sum=float(visible_robot_alpha.sum()),
        robot_input_alpha_sum=float(alpha.sum()),
        clean_background_fraction_sum=float(clean_fraction.sum()),
    )


def apply_dev_watermark(frame: np.ndarray, frame_id: int) -> np.ndarray:
    output = np.asarray(frame).copy()
    height, width = output.shape[:2]
    bar = max(26, min(64, height // 5))
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width - 1, bar), (0, 0, 90), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0.0, output)
    scale = max(0.42, min(0.9, width / 700.0))
    cv2.putText(
        output,
        f"DEV_SYNTHETIC | NOT FORMAL | frame_id={frame_id}",
        (6, max(18, bar - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        max(1, int(round(scale * 2))),
        cv2.LINE_AA,
    )
    # A second mark prevents a crop of only the title bar from silently
    # converting the fixture into an apparently formal image.
    cv2.putText(
        output,
        "DEV_SYNTHETIC",
        (max(2, width // 4), max(bar + 18, height - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, scale * 1.2),
        (20, 20, 255),
        max(1, int(round(scale * 2))),
        cv2.LINE_AA,
    )
    return output


def output_codec_probe(path: Path, *, expected_codec: str, prepared: PreparedContract) -> dict[str, Any]:
    probe = probe_video(path)
    resolution = prepared.value["resolution"]
    expected_fps = _parse_fraction(prepared.value["fps"], "fps")
    if probe.codec_name != expected_codec:
        raise CompositorContractError(
            f"output codec mismatch expected={expected_codec} actual={probe.codec_name}"
        )
    if probe.audio_streams != 0:
        raise CompositorContractError("output unexpectedly contains audio")
    if (probe.width, probe.height) != (resolution["width"], resolution["height"]):
        raise CompositorContractError("output resolution mismatch")
    if probe.fps != expected_fps:
        raise CompositorContractError("output FPS mismatch")
    _decode_video_frame_count(path, prepared.value["frame_count"])
    return {
        "codec_name": probe.codec_name,
        "pix_fmt": probe.pix_fmt,
        "width": probe.width,
        "height": probe.height,
        "fps": {
            "numerator": probe.fps.numerator,
            "denominator": probe.fps.denominator,
        },
        "frame_count": prepared.value["frame_count"],
        "audio_streams": probe.audio_streams,
    }
