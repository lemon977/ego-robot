"""Strict, read-only resolution of manifest-authenticated RAW frames.

The resolver deliberately has no directory scan and no fallback path.  A caller
must provide an exact manifest digest, an admitted manifest status, and an exact
session id.  RAW files are opened relative to a trusted directory descriptor
with ``O_RDONLY`` and ``O_NOFOLLOW`` for every path component.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping


SHA256_HEX_LENGTH = 64


class RawSourceError(RuntimeError):
    """A fail-closed RAW authority or integrity error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def read_local_regular_readonly(path: str | os.PathLike[str]) -> bytes:
    """Read one local ordinary file without following a final symlink."""

    absolute = os.path.abspath(os.fspath(path))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise RawSourceError(f"cannot open ordinary file read-only: {absolute}: {error}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RawSourceError(f"not an ordinary file: {absolute}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ResolvedFrame:
    frame_index: int
    timestamp_ns: int
    video_time_s: float
    image: Mapping[str, Any]
    metadata: Mapping[str, Any]
    width: int
    height: int


class StrictRawResolver:
    """Resolve only files explicitly authenticated by a source manifest."""

    def __init__(
        self,
        *,
        manifest_path: str | os.PathLike[str],
        expected_manifest_sha256: str,
        raw_root: str | os.PathLike[str],
        session_id: str,
        allowed_manifest_statuses: Iterable[str],
    ) -> None:
        if not _is_sha256(expected_manifest_sha256):
            raise RawSourceError("expected_manifest_sha256 must be 64 lowercase hex characters")
        allowed = frozenset(allowed_manifest_statuses)
        if not allowed or any(not isinstance(item, str) or not item for item in allowed):
            raise RawSourceError("allowed_manifest_statuses must be explicit and non-empty")

        self.manifest_path = os.path.abspath(os.fspath(manifest_path))
        self.raw_root = os.path.abspath(os.fspath(raw_root))
        self.session_id = session_id
        payload = read_local_regular_readonly(self.manifest_path)
        observed_digest = sha256_bytes(payload)
        if observed_digest != expected_manifest_sha256:
            raise RawSourceError(
                f"source manifest digest mismatch: expected {expected_manifest_sha256}, "
                f"observed {observed_digest}"
            )
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RawSourceError(f"invalid source manifest JSON: {error}") from error
        if not isinstance(manifest, dict):
            raise RawSourceError("source manifest root must be an object")
        legacy_status = manifest.get("status")
        document_status = manifest.get("document_status")
        if legacy_status is not None and document_status is not None:
            raise RawSourceError("source manifest cannot declare both status and document_status")
        manifest_status = document_status if document_status is not None else legacy_status
        if manifest_status not in allowed:
            raise RawSourceError(
                f"manifest status {manifest_status!r} not in explicitly admitted statuses"
            )
        if manifest.get("raw_root") != self.raw_root:
            raise RawSourceError(
                f"manifest RAW root {manifest.get('raw_root')!r} does not match {self.raw_root!r}"
            )
        if manifest.get("no_fallback") is not True:
            raise RawSourceError("source manifest must declare no_fallback=true")

        if document_status == "VERIFIED_G0_CALIBRATION_INPUT":
            if manifest.get("raw_readiness") != "PASS_G0_SOURCE_READABLE":
                raise RawSourceError("verified calibration manifest has no G0 RAW PASS")
            if manifest.get("promotion_scope") != "G2_004_MASK_CLEAN_CALIBRATION_ONLY":
                raise RawSourceError("verified calibration manifest has unexpected promotion scope")
            if manifest.get("formal_production_allowed") is not False:
                raise RawSourceError("verified calibration manifest must not authorize formal production")
            if manifest.get("immutable") is not True:
                raise RawSourceError("verified calibration manifest must be immutable")
            if manifest.get("project_enforced_read_only") is not True:
                raise RawSourceError("verified calibration manifest must require project read-only RAW")
            authorized = manifest.get("authorized_calibration_sessions")
            if not isinstance(authorized, list) or session_id not in authorized:
                raise RawSourceError(f"session {session_id!r} is not authorized for G2 calibration")

        sessions = manifest.get("sessions")
        if not isinstance(sessions, list):
            raise RawSourceError("source manifest sessions must be an array")
        matches = [entry for entry in sessions if isinstance(entry, dict) and entry.get("session_id") == session_id]
        if len(matches) != 1:
            raise RawSourceError(
                f"session {session_id!r} must have exactly one manifest entry; found {len(matches)}"
            )

        self.manifest = manifest
        self.manifest_sha256 = observed_digest
        self.manifest_status = str(manifest_status)
        self.session = matches[0]
        self._validate_session()
        if document_status == "VERIFIED_G0_CALIBRATION_INPUT":
            digests = manifest.get("session_digests")
            expected_session_digest = digests.get(session_id) if isinstance(digests, dict) else None
            if expected_session_digest != self.session_identity_sha256:
                raise RawSourceError(
                    f"verified session digest mismatch for {session_id}: "
                    f"{expected_session_digest!r} != {self.session_identity_sha256!r}"
                )

    @property
    def frames(self) -> tuple[ResolvedFrame, ...]:
        return self._frames

    @property
    def session_identity_sha256(self) -> str:
        canonical = json.dumps(
            self.session,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return sha256_bytes(canonical)

    def _validate_session(self) -> None:
        session = self.session
        if session.get("selector_policy") != "regular_file_only_no_symlink_no_fallback":
            raise RawSourceError("session selector policy is not strict/no-fallback")
        if session.get("frame_order_policy") != "ascending_zero_based_contiguous_frame_index":
            raise RawSourceError("session frame order policy is not contiguous zero-based order")
        source_session = session.get("source_session")
        if not isinstance(source_session, str):
            raise RawSourceError("source_session is missing")
        self._relative_components(source_session)

        width = session.get("width")
        height = session.get("height")
        fps = session.get("fps")
        frame_count = session.get("frame_count")
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise RawSourceError("invalid session dimensions")
        if not isinstance(fps, (int, float)) or float(fps) <= 0:
            raise RawSourceError("invalid session FPS")
        if not isinstance(frame_count, int) or frame_count <= 0:
            raise RawSourceError("invalid session frame_count")

        records = session.get("frames")
        if not isinstance(records, list) or len(records) != frame_count:
            raise RawSourceError("frame array length does not match frame_count")
        resolved: list[ResolvedFrame] = []
        previous_timestamp: int | None = None
        previous_video_time: float | None = None
        for expected_index, record in enumerate(records):
            if not isinstance(record, dict) or record.get("frame_index") != expected_index:
                raise RawSourceError(f"non-contiguous or malformed frame at position {expected_index}")
            timestamp = record.get("timestamp_ns")
            video_time = record.get("video_time_s")
            if not isinstance(timestamp, int) or not isinstance(video_time, (int, float)):
                raise RawSourceError(f"invalid timestamp at frame {expected_index}")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise RawSourceError(f"decreasing tracking timestamp at frame {expected_index}")
            if previous_video_time is not None and float(video_time) <= previous_video_time:
                raise RawSourceError(f"non-increasing video time at frame {expected_index}")
            previous_timestamp = timestamp
            previous_video_time = float(video_time)
            if record.get("width") != width or record.get("height") != height:
                raise RawSourceError(f"frame dimensions disagree at frame {expected_index}")
            image = self._validate_artifact_record(record.get("image"), expected_index, "image")
            metadata = self._validate_artifact_record(record.get("metadata"), expected_index, "metadata")
            resolved.append(
                ResolvedFrame(
                    frame_index=expected_index,
                    timestamp_ns=timestamp,
                    video_time_s=float(video_time),
                    image=image,
                    metadata=metadata,
                    width=width,
                    height=height,
                )
            )
        self._frames = tuple(resolved)

    def _validate_artifact_record(
        self,
        record: object,
        frame_index: int,
        artifact_kind: str,
    ) -> Mapping[str, Any]:
        if not isinstance(record, dict):
            raise RawSourceError(f"missing {artifact_kind} record at frame {frame_index}")
        path = record.get("path")
        if not isinstance(path, str):
            raise RawSourceError(f"missing {artifact_kind} path at frame {frame_index}")
        self._relative_components(path)
        if record.get("kind") != "regular_file":
            raise RawSourceError(f"{artifact_kind} is not declared regular at frame {frame_index}")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            raise RawSourceError(f"invalid {artifact_kind} byte count at frame {frame_index}")
        if not _is_sha256(record.get("sha256")):
            raise RawSourceError(f"invalid {artifact_kind} digest at frame {frame_index}")
        return record

    def _relative_components(self, target: str | os.PathLike[str]) -> tuple[str, ...]:
        target_text = os.fspath(target)
        if not os.path.isabs(target_text):
            raise RawSourceError(f"RAW target must be absolute: {target_text!r}")
        normalized = os.path.abspath(target_text)
        try:
            common = os.path.commonpath([self.raw_root, normalized])
        except ValueError as error:
            raise RawSourceError(f"RAW target is on another filesystem: {target_text!r}") from error
        if common != self.raw_root or normalized == self.raw_root:
            raise RawSourceError(f"RAW target escapes or names the authority root: {target_text!r}")
        relative = os.path.relpath(normalized, self.raw_root)
        parts = tuple(Path(relative).parts)
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise RawSourceError(f"unsafe RAW relative path: {relative!r}")
        return parts

    def _read_raw_ordinary(self, target: str | os.PathLike[str]) -> bytes:
        parts = self._relative_components(target)
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        opened: list[int] = []
        try:
            current = os.open(self.raw_root, directory_flags)
            opened.append(current)
            for part in parts[:-1]:
                current = os.open(part, directory_flags, dir_fd=current)
                opened.append(current)
            descriptor = os.open(parts[-1], file_flags, dir_fd=current)
            opened.append(descriptor)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise RawSourceError(f"RAW artifact is not an ordinary file: {target}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        except OSError as error:
            raise RawSourceError(f"strict RAW read failed for {target}: {error}") from error
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    def read_frame_artifact(self, frame_index: int, artifact_kind: str) -> bytes:
        if artifact_kind not in {"image", "metadata"}:
            raise RawSourceError(f"unsupported frame artifact kind: {artifact_kind!r}")
        if frame_index < 0 or frame_index >= len(self._frames):
            raise RawSourceError(f"frame index out of range: {frame_index}")
        record = getattr(self._frames[frame_index], artifact_kind)
        payload = self._read_raw_ordinary(record["path"])
        if len(payload) != record["bytes"]:
            raise RawSourceError(
                f"byte count mismatch for frame {frame_index} {artifact_kind}: "
                f"expected {record['bytes']}, observed {len(payload)}"
            )
        observed_digest = sha256_bytes(payload)
        if observed_digest != record["sha256"]:
            raise RawSourceError(
                f"digest mismatch for frame {frame_index} {artifact_kind}: "
                f"expected {record['sha256']}, observed {observed_digest}"
            )
        return payload

    def read_metadata_json(self, frame_index: int) -> Mapping[str, Any]:
        payload = self.read_frame_artifact(frame_index, "metadata")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RawSourceError(f"invalid metadata JSON at frame {frame_index}: {error}") from error
        if not isinstance(value, dict):
            raise RawSourceError(f"metadata root is not an object at frame {frame_index}")
        return value
