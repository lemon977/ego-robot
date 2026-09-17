"""Project preparation, resumable annotation storage and deterministic export."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

from PIL import Image

from . import __version__
from .config import AppConfig
from .media import extract_selected_frames, probe_video, select_frames
from .raster import binary_mask, overlay_image, rasterize, validate_operations
from .schema import BY_SYMBOL, CLASSES, SCHEMA_VERSION


PROJECT_SCHEMA = "offline-mask-annotation-project-v1"
ANNOTATION_SCHEMA = "offline-mask-vector-annotation-v1"
EXPORT_SCHEMA = "offline-mask-export-v1"


class ProjectError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        image.save(name, format="PNG", optimize=False)
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProjectError(f"JSON root must be an object: {path}")
    return value


def project_path(config: AppConfig) -> Path:
    return config.output_dir / "FRAME_MANIFEST.json"


def annotation_path(config: AppConfig, frame_record: dict[str, Any]) -> Path:
    return config.output_dir / "annotations" / f"{frame_record['frame_key']}.json"


def _project_identity(config: AppConfig, source_sha: str, selected: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCHEMA,
        "label_schema_version": SCHEMA_VERSION,
        "session_id": config.session_id,
        "source_sha256": source_sha,
        "sampling_config_sha256": config.identity_sha256,
        "selected_decoded_indices": [frame.decoded_index for frame in selected],
        "selected_pts_time": [str(frame.pts_time) for frame in selected],
    }


def _assert_existing_identity(config: AppConfig, manifest: dict[str, Any], identity: dict[str, Any]) -> None:
    observed = manifest.get("identity")
    if observed != identity:
        raise ProjectError(
            "output_dir already contains a different input/session/sampling identity; "
            "choose a new output_dir instead of overwriting it"
        )
    for frame in manifest.get("frames", []):
        image = config.output_dir / frame["image_relpath"]
        if not image.is_file() or image.stat().st_size != frame["image_bytes"]:
            raise ProjectError(f"prepared frame missing or byte count changed: {image}")
        if sha256_file(image) != frame["image_sha256"]:
            raise ProjectError(f"prepared frame checksum changed: {image}")


def prepare_project(config: AppConfig) -> dict[str, Any]:
    source_sha = sha256_file(config.input_mp4)
    probe = probe_video(config.input_mp4)
    selected = select_frames(probe, config.frame_count, config.timestamps_csv)
    identity = _project_identity(config, source_sha, selected)
    manifest_file = project_path(config)
    if manifest_file.exists():
        manifest = _load_json(manifest_file)
        _assert_existing_identity(config, manifest, identity)
        return manifest

    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ProjectError(
            f"output_dir is non-empty but has no FRAME_MANIFEST.json: {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    temp_frames = config.output_dir / f".frames-{uuid.uuid4().hex}.tmp"
    frames_dir = config.output_dir / "frames"
    if frames_dir.exists():
        raise ProjectError(f"unexpected frames directory before prepare: {frames_dir}")
    temp_frames.mkdir()
    try:
        ffmpeg_version = extract_selected_frames(
            config.input_mp4, selected, temp_frames / "frame_%04d.png"
        )
        extracted = sorted(temp_frames.glob("frame_*.png"))
        if len(extracted) != len(selected):
            raise ProjectError(f"ffmpeg extracted {len(extracted)} frames, expected {len(selected)}")
        frame_records = []
        for ordinal, (path, source_frame) in enumerate(zip(extracted, selected)):
            expected_name = f"frame_{ordinal:04d}.png"
            if path.name != expected_name:
                raise ProjectError(f"unexpected extracted frame name: {path.name}")
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
            frame_key = f"frame_{ordinal:04d}"
            frame_records.append({
                "ordinal": ordinal,
                "frame_key": frame_key,
                "sample_id": f"{config.session_id}:{ordinal:04d}",
                "source_decoded_index": source_frame.decoded_index,
                "source_pts": source_frame.pts,
                "source_pts_time": str(source_frame.pts_time),
                "source_duration_time": (
                    str(source_frame.duration_time) if source_frame.duration_time is not None else None
                ),
                "source_key_frame": source_frame.key_frame,
                "image_relpath": f"frames/{expected_name}",
                "image_width": width,
                "image_height": height,
                "image_mode": mode,
                "image_bytes": path.stat().st_size,
                "image_sha256": sha256_file(path),
            })
        os.replace(temp_frames, frames_dir)
    except Exception:
        if temp_frames.exists() and temp_frames.parent == config.output_dir:
            shutil.rmtree(temp_frames)
        raise

    manifest = {
        "schema_version": PROJECT_SCHEMA,
        "application_version": __version__,
        "created_at": utc_now(),
        "status": "PREPARED",
        "identity": identity,
        "source": {
            "input_mp4": str(config.input_mp4),
            "filename": config.input_mp4.name,
            "bytes": config.input_mp4.stat().st_size,
            "sha256": source_sha,
            "ffprobe_version": probe.ffprobe_version,
            "ffmpeg_version": ffmpeg_version,
            "video_stream": probe.stream,
            "decoded_frame_count": len(probe.frames),
        },
        "sampling": {
            "method": "EXPLICIT_TIMESTAMPS_NEAREST_UNIQUE_PTS" if config.timestamps_csv else "ROUNDED_LINSPACE_DECODED_INDEX",
            "requested_frame_count": config.frame_count,
            "timestamps_csv": str(config.timestamps_csv) if config.timestamps_csv else None,
            "timestamps_csv_sha256": sha256_file(config.timestamps_csv) if config.timestamps_csv else None,
        },
        "classes": list(CLASSES),
        "frames": frame_records,
    }
    annotations_dir = config.output_dir / "annotations"
    annotations_dir.mkdir()
    draft_dir = config.output_dir / "draft_class_id"
    draft_dir.mkdir()
    for frame in frame_records:
        draft_path = draft_dir / f"{frame['frame_key']}.png"
        atomic_png(
            draft_path,
            rasterize([], (frame["image_width"], frame["image_height"])),
        )
        annotation = {
            "schema_version": ANNOTATION_SCHEMA,
            "label_schema_version": SCHEMA_VERSION,
            "session_id": config.session_id,
            "sample_id": frame["sample_id"],
            "frame_key": frame["frame_key"],
            "source_decoded_index": frame["source_decoded_index"],
            "source_pts_time": frame["source_pts_time"],
            "image_sha256": frame["image_sha256"],
            "image_size": [frame["image_width"], frame["image_height"]],
            "revision": 0,
            "complete": False,
            "operations": [],
            "draft_class_id_relpath": str(draft_path.relative_to(config.output_dir)).replace(os.sep, "/"),
            "draft_class_id_bytes": draft_path.stat().st_size,
            "draft_class_id_sha256": sha256_file(draft_path),
            "updated_at": utc_now(),
        }
        atomic_json(annotation_path(config, frame), annotation)
    atomic_json(manifest_file, manifest)
    return manifest


def load_project(config: AppConfig) -> dict[str, Any]:
    path = project_path(config)
    if not path.is_file():
        raise ProjectError(f"project is not prepared: run prepare first: {path}")
    manifest = _load_json(path)
    if manifest.get("schema_version") != PROJECT_SCHEMA:
        raise ProjectError("unsupported project schema")
    if manifest.get("identity", {}).get("session_id") != config.session_id:
        raise ProjectError("config session_id does not match prepared project")
    return manifest


def frame_by_key(manifest: dict[str, Any], frame_key: str) -> dict[str, Any]:
    for frame in manifest.get("frames", []):
        if frame.get("frame_key") == frame_key:
            return frame
    raise ProjectError(f"unknown frame key: {frame_key}")


def load_annotation(config: AppConfig, frame_record: dict[str, Any]) -> dict[str, Any]:
    path = annotation_path(config, frame_record)
    annotation = _load_json(path)
    if annotation.get("sample_id") != frame_record["sample_id"]:
        raise ProjectError(f"annotation sample identity mismatch: {path}")
    if annotation.get("image_sha256") != frame_record["image_sha256"]:
        raise ProjectError(f"annotation image identity mismatch: {path}")
    return annotation


def save_annotation(
    config: AppConfig,
    frame_record: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    current = load_annotation(config, frame_record)
    if payload.get("revision") != current.get("revision"):
        raise ProjectError(
            f"stale annotation revision: browser={payload.get('revision')} disk={current.get('revision')}"
        )
    complete = payload.get("complete")
    if not isinstance(complete, bool):
        raise ProjectError("annotation complete must be boolean")
    width, height = frame_record["image_width"], frame_record["image_height"]
    operations = validate_operations(payload.get("operations"), width, height)
    draft_path = config.output_dir / "draft_class_id" / f"{frame_record['frame_key']}.png"
    atomic_png(draft_path, rasterize(operations, (width, height)))
    saved = dict(current)
    saved.update({
        "revision": int(current["revision"]) + 1,
        "complete": complete,
        "operations": operations,
        "draft_class_id_relpath": str(draft_path.relative_to(config.output_dir)).replace(os.sep, "/"),
        "draft_class_id_bytes": draft_path.stat().st_size,
        "draft_class_id_sha256": sha256_file(draft_path),
        "updated_at": utc_now(),
    })
    atomic_json(annotation_path(config, frame_record), saved)
    return saved


def export_project(config: AppConfig) -> dict[str, Any]:
    manifest = load_project(config)
    export_root = config.output_dir / "export"
    records = []
    for frame in manifest["frames"]:
        annotation = load_annotation(config, frame)
        size = (frame["image_width"], frame["image_height"])
        class_mask = rasterize(annotation["operations"], size)
        image_path = config.output_dir / frame["image_relpath"]
        with Image.open(image_path) as source:
            source.load()
            overlay = overlay_image(source, class_mask)

        class_path = export_root / "class_id" / f"{frame['frame_key']}.png"
        overlay_path = export_root / "overlay" / f"{frame['frame_key']}.png"
        atomic_png(class_path, class_mask)
        atomic_png(overlay_path, overlay)
        binary_records = {}
        for item in CLASSES:
            path = export_root / "binary" / item["symbol"] / f"{frame['frame_key']}.png"
            atomic_png(path, binary_mask(class_mask, item["id"]))
            binary_records[item["symbol"]] = {
                "relpath": str(path.relative_to(config.output_dir)).replace(os.sep, "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        records.append({
            "frame_key": frame["frame_key"],
            "sample_id": frame["sample_id"],
            "source_decoded_index": frame["source_decoded_index"],
            "source_pts": frame["source_pts"],
            "source_pts_time": frame["source_pts_time"],
            "source_image_sha256": frame["image_sha256"],
            "annotation_relpath": str(annotation_path(config, frame).relative_to(config.output_dir)).replace(os.sep, "/"),
            "annotation_revision": annotation["revision"],
            "annotation_complete": annotation["complete"],
            "annotation_sha256": sha256_file(annotation_path(config, frame)),
            "class_id": {
                "relpath": str(class_path.relative_to(config.output_dir)).replace(os.sep, "/"),
                "bytes": class_path.stat().st_size,
                "sha256": sha256_file(class_path),
            },
            "binary": binary_records,
            "overlay": {
                "relpath": str(overlay_path.relative_to(config.output_dir)).replace(os.sep, "/"),
                "bytes": overlay_path.stat().st_size,
                "sha256": sha256_file(overlay_path),
            },
        })
    result = {
        "schema_version": EXPORT_SCHEMA,
        "application_version": __version__,
        "created_at": utc_now(),
        "status": "EXPORTED",
        "project_manifest": str(project_path(config)),
        "project_manifest_sha256": sha256_file(project_path(config)),
        "session_id": config.session_id,
        "source_mp4_sha256": manifest["source"]["sha256"],
        "label_schema_version": SCHEMA_VERSION,
        "classes": list(CLASSES),
        "frame_count": len(records),
        "frames": records,
    }
    atomic_json(config.output_dir / "EXPORT_MANIFEST.json", result)
    return result
