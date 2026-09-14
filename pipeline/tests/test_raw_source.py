from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import pipeline.raw_source as raw_source
from pipeline.raw_source import RawSourceError, StrictRawResolver


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(tmp_path: Path, *, image_path: Path | None = None) -> tuple[Path, Path, bytes]:
    raw_root = tmp_path / "raw"
    frame_root = raw_root / "source" / "session" / "preprocess" / "all_data" / "00000"
    frame_root.mkdir(parents=True)
    image_payload = b"image-payload"
    metadata_payload = json.dumps({"metadata": {}, "entities": {"hands": {}}}).encode()
    real_image = frame_root / "rgb.png"
    real_image.write_bytes(image_payload)
    metadata = frame_root / "training_data.json"
    metadata.write_bytes(metadata_payload)
    selected_image = image_path or real_image
    manifest = {
        "schema_version": "candidate-source-manifest-v0",
        "status": "TEST_ADMITTED",
        "raw_root": str(raw_root),
        "source_root": str(raw_root / "source"),
        "no_fallback": True,
        "sessions": [
            {
                "session_id": "session_alpha",
                "source_session": str(raw_root / "source" / "session"),
                "selector": "preprocess/all_data/<frame_index:05d>/rgb.png",
                "selector_policy": "regular_file_only_no_symlink_no_fallback",
                "frame_order_policy": "ascending_zero_based_contiguous_frame_index",
                "timestamp_policy": "test clock",
                "width": 2,
                "height": 2,
                "fps": 30.0,
                "frame_count": 1,
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp_ns": 1,
                        "video_time_s": 0.0,
                        "image": {
                            "path": str(selected_image),
                            "kind": "regular_file",
                            "bytes": len(image_payload),
                            "sha256": _sha(image_payload),
                        },
                        "metadata": {
                            "path": str(metadata),
                            "kind": "regular_file",
                            "bytes": len(metadata_payload),
                            "sha256": _sha(metadata_payload),
                        },
                        "width": 2,
                        "height": 2,
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_payload = json.dumps(manifest).encode()
    manifest_path.write_bytes(manifest_payload)
    return manifest_path, raw_root, manifest_payload


def _resolver(manifest: Path, raw_root: Path, payload: bytes, session: str = "session_alpha") -> StrictRawResolver:
    return StrictRawResolver(
        manifest_path=manifest,
        expected_manifest_sha256=_sha(payload),
        raw_root=raw_root,
        session_id=session,
        allowed_manifest_statuses=("TEST_ADMITTED",),
    )


def test_exact_manifest_hash_and_no_fallback(tmp_path: Path) -> None:
    manifest, raw_root, payload = _write_manifest(tmp_path)
    resolver = _resolver(manifest, raw_root, payload)
    assert resolver.read_frame_artifact(0, "image") == b"image-payload"
    with pytest.raises(RawSourceError, match="exactly one"):
        _resolver(manifest, raw_root, payload, session="missing_session")
    with pytest.raises(RawSourceError, match="digest mismatch"):
        StrictRawResolver(
            manifest_path=manifest,
            expected_manifest_sha256="0" * 64,
            raw_root=raw_root,
            session_id="session_alpha",
            allowed_manifest_statuses=("TEST_ADMITTED",),
        )


def test_rejects_symlink_in_raw_path_component(tmp_path: Path) -> None:
    manifest, raw_root, _ = _write_manifest(tmp_path)
    manifest_data = json.loads(manifest.read_text())
    real_directory = raw_root / "source" / "session" / "preprocess" / "all_data" / "00000"
    alias = real_directory.parent / "alias"
    alias.symlink_to(real_directory, target_is_directory=True)
    manifest_data["sessions"][0]["frames"][0]["image"]["path"] = str(alias / "rgb.png")
    payload = json.dumps(manifest_data).encode()
    manifest.write_bytes(payload)
    resolver = _resolver(manifest, raw_root, payload)
    with pytest.raises(RawSourceError, match="strict RAW read failed"):
        resolver.read_frame_artifact(0, "image")


def test_raw_opens_are_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, raw_root, payload = _write_manifest(tmp_path)
    resolver = _resolver(manifest, raw_root, payload)
    observed_flags: list[int] = []
    original_open = raw_source.os.open

    def recording_open(path, flags, *args, **kwargs):
        observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(raw_source.os, "open", recording_open)
    resolver.read_frame_artifact(0, "image")
    assert observed_flags
    assert all(flags & os.O_ACCMODE == os.O_RDONLY for flags in observed_flags)


def test_rejects_manifest_escape(tmp_path: Path) -> None:
    manifest, raw_root, payload = _write_manifest(tmp_path)
    data = json.loads(payload)
    data["sessions"][0]["frames"][0]["image"]["path"] = str(tmp_path / "outside.png")
    altered = json.dumps(data).encode()
    manifest.write_bytes(altered)
    with pytest.raises(RawSourceError, match="escapes"):
        _resolver(manifest, raw_root, altered)


def test_verified_calibration_manifest_binds_document_status_and_session_digest(tmp_path: Path) -> None:
    manifest_path, raw_root, payload = _write_manifest(tmp_path)
    verified = json.loads(payload)
    verified.pop("status")
    verified.update(
        {
            "document_status": "VERIFIED_G0_CALIBRATION_INPUT",
            "raw_readiness": "PASS_G0_SOURCE_READABLE",
            "promotion_scope": "G2_004_MASK_CLEAN_CALIBRATION_ONLY",
            "formal_production_allowed": False,
            "immutable": True,
            "project_enforced_read_only": True,
            "authorized_calibration_sessions": ["session_alpha"],
        }
    )
    session_digest = _sha(
        json.dumps(
            verified["sessions"][0],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    verified["session_digests"] = {"session_alpha": session_digest}
    verified_payload = json.dumps(verified).encode("utf-8")
    manifest_path.write_bytes(verified_payload)
    resolver = StrictRawResolver(
        manifest_path=manifest_path,
        expected_manifest_sha256=_sha(verified_payload),
        raw_root=raw_root,
        session_id="session_alpha",
        allowed_manifest_statuses=("VERIFIED_G0_CALIBRATION_INPUT",),
    )
    assert resolver.manifest_status == "VERIFIED_G0_CALIBRATION_INPUT"

    verified["session_digests"]["session_alpha"] = "0" * 64
    bad_payload = json.dumps(verified).encode("utf-8")
    manifest_path.write_bytes(bad_payload)
    with pytest.raises(RawSourceError, match="session digest mismatch"):
        StrictRawResolver(
            manifest_path=manifest_path,
            expected_manifest_sha256=_sha(bad_payload),
            raw_root=raw_root,
            session_id="session_alpha",
            allowed_manifest_statuses=("VERIFIED_G0_CALIBRATION_INPUT",),
        )
