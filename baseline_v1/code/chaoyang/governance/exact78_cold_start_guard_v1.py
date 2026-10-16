"""Small, model-free input boundary for an Exact78 candidate submission.

This module only describes files a producer may open. Runtime open tracing is
still required for T1-B; a declared manifest is not proof of actual reads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DATA_ROOT = Path("/path/to/private_data/egodata/datasets/ego")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_SCHEMA = "exact78-raw-submit-v1"


class ColdStartInputError(ValueError):
    pass


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _source_path(path: Path, allowed_root: Path) -> Path:
    if path.is_symlink():
        raise ColdStartInputError(f"SYMLINK_SOURCE_FORBIDDEN:{path}")
    resolved = path.resolve(strict=True)
    root = allowed_root.resolve(strict=True)
    if not _inside(resolved, root):
        raise ColdStartInputError(f"SOURCE_OUTSIDE_ALLOWED_ROOT:{path}")
    if not resolved.is_file():
        raise ColdStartInputError(f"SOURCE_NOT_FILE:{path}")
    return resolved


def _relative_source(session: Path, name: str, allowed_root: Path) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ColdStartInputError(f"UNSAFE_SOURCE_NAME:{name}")
    return _source_path(session / relative, allowed_root)


def file_ref(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inspect_session(raw_session: Path, *, allowed_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Resolve only raw RGB, timestamp and calibration paths; never load motion."""
    if raw_session.is_symlink():
        raise ColdStartInputError("SYMLINK_SESSION_FORBIDDEN")
    session = raw_session.resolve(strict=True)
    root = allowed_root.resolve(strict=True)
    if not _inside(session, root) or not session.is_dir():
        raise ColdStartInputError("SESSION_OUTSIDE_ALLOWED_ROOT")

    if (session / "raw/vst.h264").is_file():
        if session.parent.name.startswith("cards_") and session.parent.name.endswith("_0915"):
            session_id = f"play_cards_0915_{session.name}"
        elif session.parent.name.startswith("chips_") and session.parent.name.endswith("_0915"):
            session_id = f"get_potato_chips_0915_{session.name}"
        else:
            raise ColdStartInputError("0915_SOURCE_IDENTITY_UNVERIFIED")
        names = ("raw/vst.h264", "raw/vst.qpc.ts.jsonl", "raw/vst.ts.jsonl",
                 "raw/camera_params.json", "raw/camera_params.meta.json")
        paths = {name: _relative_source(session, name, root) for name in names}
        camera = json.loads(paths["raw/camera_params.json"].read_text(encoding="utf-8"))
        if (camera.get("left", {}).get("sourceIndex") != 1
                or camera.get("right", {}).get("sourceIndex") != 0):
            raise ColdStartInputError("PHYSICAL_EYE_IDENTITY_UNVERIFIED")
        width, height = camera.get("width"), camera.get("height")
        if not all(isinstance(value, int) and value > 0 for value in (width, height)):
            raise ColdStartInputError("INVALID_CAMERA_SIZE")
        return {
            "schema_version": "exact78-raw-source-v1", "kind": "0915_RAW_VST",
            "session_root": str(session), "session_id": session_id,
            "files": {name: file_ref(path) for name, path in paths.items()},
            "physical_eye": "left_sourceIndex_1", "crop_xywh": [width, 0, width, height],
            "time_mode": "TIMEBASE_UNVERIFIED",
            "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
            "head_pose_source": "ABSENT_UNTIL_FIELD_PROJECTION_AND_CLOCK_PROOF",
            "metric_calibration": "SOURCE_PRESENT_QUALITY_UNVERIFIED"
            if camera.get("left", {}).get("extrinsic") and camera.get("right", {}).get("extrinsic")
            else "ABSENT",
        }

    manifest_path = _relative_source(session, "clip_manifest.json", root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("clip_name") != session.name:
        raise ColdStartInputError("TRACKER_MANIFEST_IDENTITY")
    names = {
        "video": files.get("video"),
        "source_stereo_video": files.get("source_stereo_video"),
        "camera_params": files.get("camera_params", "camera_params.json"),
        "slam": files.get("slam", f"slam_trajectory_{session.name}.jsonl"),
        "clip_manifest": "clip_manifest.json",
    }
    if any(not isinstance(name, str) for name in names.values()):
        raise ColdStartInputError("TRACKER_MANIFEST_PATH_TYPE")
    if (Path(names["video"]).suffix.lower() != ".mp4"
            or Path(names["source_stereo_video"]).suffix.lower() != ".mp4"
            or Path(names["source_stereo_video"]).parts[:1] != ("source_stereo",)
            or Path(names["camera_params"]).suffix.lower() != ".json"
            or Path(names["slam"]).suffix.lower() != ".jsonl"
            or len(set(names.values())) != len(names)):
        raise ColdStartInputError("TRACKER_RAW_INPUT_ROLE_DRIFT")
    paths = {role: _relative_source(session, name, root) for role, name in names.items()}
    camera = json.loads(paths["camera_params"].read_text(encoding="utf-8"))
    extrinsics = camera.get("extrinsics")
    metric = ("SOURCE_PRESENT_QUALITY_UNVERIFIED"
              if isinstance(extrinsics, dict) and extrinsics.get("left") and extrinsics.get("right")
              else "ABSENT")
    video = manifest.get("video", {})
    if not isinstance(video.get("frame_count"), int) or video["frame_count"] <= 0:
        raise ColdStartInputError("TRACKER_FRAME_COUNT_ABSENT")
    if not isinstance(video.get("fps"), (int, float)) or video["fps"] <= 0:
        raise ColdStartInputError("TRACKER_FPS_ABSENT")
    return {
        "schema_version": "exact78-raw-source-v1", "kind": "TRACKER_CLIP_RGB",
        "session_root": str(session), "session_id": session.name,
        "files": {role: file_ref(path) for role, path in paths.items()},
        "physical_eye": "tracker_session_left_RGB", "crop_xywh": None,
        "time_mode": "TRACKER_CLIP_METADATA_ONLY", "capture_to_decode_alignment": "CLIP_INDEX_ONLY",
        "frame_count_declared": video["frame_count"], "fps_declared": video["fps"],
        "head_pose_source": "SAME_SESSION_METADATA_PROJECTION_REQUIRED",
        "metric_calibration": metric,
    }


def validate_manifest(value: Any) -> list[Path]:
    if not isinstance(value, dict) or set(value) != {"schema_version", "sessions"}:
        raise ColdStartInputError("MANIFEST_KEYS_NOT_RAW_ONLY")
    if value["schema_version"] != MANIFEST_SCHEMA or not isinstance(value["sessions"], list):
        raise ColdStartInputError("MANIFEST_SCHEMA")
    if not value["sessions"] or len(value["sessions"]) != len(set(map(str, value["sessions"]))):
        raise ColdStartInputError("MANIFEST_EMPTY_OR_DUPLICATE")
    if any(not isinstance(item, str) for item in value["sessions"]):
        raise ColdStartInputError("MANIFEST_SESSION_PATH_TYPE")
    return [Path(item) for item in value["sessions"]]


def validate_producer_reads(source: dict[str, Any], opened: Iterable[Path]) -> list[dict[str, Any]]:
    kind = source.get("kind")
    expected_roles = ({"raw/vst.h264", "raw/vst.qpc.ts.jsonl", "raw/vst.ts.jsonl",
                       "raw/camera_params.json", "raw/camera_params.meta.json"}
                      if kind == "0915_RAW_VST" else
                      {"video", "source_stereo_video", "camera_params", "slam", "clip_manifest"}
                      if kind == "TRACKER_CLIP_RGB" else None)
    if expected_roles is None or set(source.get("files", {})) != expected_roles:
        raise ColdStartInputError("SOURCE_ROLE_SET_DRIFT")
    session = Path(source["session_root"]).resolve(strict=True)
    tracker_paths: dict[str, Path] = {}
    if kind == "TRACKER_CLIP_RGB":
        manifest_ref = source["files"]["clip_manifest"]
        manifest_path = _relative_source(session, "clip_manifest.json", session)
        if file_ref(manifest_path) != manifest_ref:
            raise ColdStartInputError("TRACKER_MANIFEST_SHA_DRIFT")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files")
        if manifest.get("clip_name") != session.name or not isinstance(files, dict):
            raise ColdStartInputError("TRACKER_MANIFEST_IDENTITY")
        names = {"video": files.get("video"),
                 "source_stereo_video": files.get("source_stereo_video"),
                 "camera_params": files.get("camera_params", "camera_params.json"),
                 "slam": files.get("slam", f"slam_trajectory_{session.name}.jsonl"),
                 "clip_manifest": "clip_manifest.json"}
        if (any(not isinstance(name, str) for name in names.values())
                or Path(names["video"]).suffix.lower() != ".mp4"
                or Path(names["source_stereo_video"]).suffix.lower() != ".mp4"
                or Path(names["source_stereo_video"]).parts[:1] != ("source_stereo",)
                or Path(names["camera_params"]).suffix.lower() != ".json"
                or Path(names["slam"]).suffix.lower() != ".jsonl"
                or len(set(names.values())) != len(names)):
            raise ColdStartInputError("TRACKER_RAW_INPUT_ROLE_DRIFT")
        tracker_paths = {role: _relative_source(session, name, session)
                         for role, name in names.items()}
    for role, ref in source["files"].items():
        actual = Path(ref["path"]).resolve(strict=True)
        if not _inside(actual, session) or actual != (session / role if kind == "0915_RAW_VST"
                                                     else tracker_paths[role]):
            raise ColdStartInputError(f"SOURCE_ROLE_PATH_DRIFT:{role}")
    allowed = {Path(item["path"]).resolve(strict=True): item for item in source["files"].values()}
    rows = []
    for raw in opened:
        path = raw.resolve(strict=True)
        if path not in allowed:
            raise ColdStartInputError(f"FORBIDDEN_PRODUCER_READ:{raw}")
        actual = file_ref(path)
        if actual != allowed[path]:
            raise ColdStartInputError(f"SOURCE_SHA_DRIFT:{raw}")
        rows.append(actual)
    return rows


def validate_output_path(path: Path, *, attempt_root: Path) -> Path:
    """Constrain a stage write to the exact Task Packet attempt directory."""
    project_root = PROJECT_ROOT
    root = project_root.resolve(strict=True)
    attempt = attempt_root.resolve(strict=False)
    if not _inside(attempt, root / "_run" / "current") or "attempts" not in attempt.parts:
        raise ColdStartInputError("ATTEMPT_OUTSIDE_PROJECT")
    resolved = path.resolve(strict=False)
    if not _inside(resolved, attempt):
        raise ColdStartInputError(f"OUTPUT_OUTSIDE_PROJECT_ATTEMPT:{path}")
    return resolved


def stage_signature(*, inputs: list[dict[str, Any]], code: list[dict[str, Any]],
                    config: dict[str, Any], weights: list[dict[str, Any]] | str,
                    calibration: dict[str, Any] | str, schema_version: str) -> str:
    payload = {"inputs": inputs, "code": code, "config": config, "weights": weights,
               "calibration": calibration, "schema_version": schema_version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
