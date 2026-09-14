from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import author_g4_delivery_video_cpu as author  # noqa: E402


def _write_png(path: Path, image: np.ndarray) -> dict[str, object]:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    payload = encoded.tobytes()
    path.write_bytes(payload)
    info = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _write_json(path: Path, value: object) -> dict[str, object]:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    info = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    width, height = 64, 48
    frames: list[dict[str, object]] = []
    for index in range(3):
        raw = np.full((height, width, 3), (15 + index * 20, 30, 45), np.uint8)
        raw_ref = _write_png(tmp_path / f"raw_{index}.png", raw)
        if index < 2:
            mask = np.zeros((height, width), np.uint8)
            mask[8:36, 10 + index : 34 + index] = 255
            mask_ref = _write_png(tmp_path / f"mask_{index}.png", mask)
            mask_record: dict[str, object] = {
                "status": "READY" if index == 0 else "PARTIAL",
                "image": mask_ref,
                "source_raw_sha256": raw_ref["sha256"],
                "pending_sides": [] if index == 0 else ["left"],
                "authority_status": "ACCEPT",
                "source_mask_pixel_count": int(np.count_nonzero(mask)),
            }
            clean = raw.copy()
            clean[mask > 0] = (80, 90, 100)
            clean_ref = _write_png(tmp_path / f"clean_{index}.png", clean)
            clean_record: dict[str, object] = {
                "status": "READY",
                "image": clean_ref,
                "source_raw_sha256": raw_ref["sha256"],
                "consumed_mask_sha256": mask_ref["sha256"],
                "unresolved_pixel_count": index,
            }
            if index == 0:
                robot = clean.copy()
                robot[12:28, 20:44] = (200, 210, 220)
                robot_ref = _write_png(tmp_path / "robot_0.png", robot)
                robot_record: dict[str, object] = {
                    "status": "READY_PIXELS",
                    "mode": "FINAL_RGB",
                    "image": robot_ref,
                    "source_frame_index": index,
                    "source_clean_sha256": clean_ref["sha256"],
                }
            else:
                rgba = np.zeros((height, width, 4), np.uint8)
                rgba[12:28, 20:44, :3] = (220, 190, 160)
                rgba[12:28, 20:44, 3] = 128
                robot_ref = _write_png(tmp_path / "robot_1.png", rgba)
                robot_record = {
                    "status": "READY_PIXELS",
                    "mode": "STRAIGHT_ALPHA_OVER_CLEAN",
                    "image": robot_ref,
                    "source_frame_index": index,
                    "source_clean_sha256": clean_ref["sha256"],
                }
        else:
            mask_record = {"status": "PENDING", "reason": "OUTSIDE_FINAL_SELECTION"}
            clean_record = {"status": "PENDING", "reason": "CLEAN_NOT_PUBLISHED"}
            robot_record = {"status": "PENDING", "reason": "IK_GATE_FAILED"}
        frames.append(
            {
                "frame_index": index,
                "raw": raw_ref,
                "mask": mask_record,
                "clean": clean_record,
                "robot": robot_record,
            }
        )
    session_id = "grap_a_cap_999_fixture"
    registry = {
        "schema_version": (
            "g4-robot-seed573-ten-session-external-denominator-registry-v1"
        ),
        "status": "DEVELOPMENT_ONLY_ARTIFACT_EXISTS",
        "sessions": [
            {
                "session_id": session_id,
                "raw_frame_count": 3,
                "pass_frame_half_open_segments": [[0, 2]],
                "published_scene": {"frame_count": 2, "scene_state": None},
            }
        ],
    }
    registry_ref = _write_json(tmp_path / "ROBOT_REGISTRY.json", registry)
    manifest: dict[str, object] = {
        "schema_version": author.INPUT_SCHEMA,
        "status": author.INPUT_STATUS,
        "development_only": True,
        "session_id": session_id,
        "original_frame_count": 3,
        "fps": 30.0,
        "source_resolution": [width, height],
        "semantics": author.SEMANTICS,
        "robot_registry": registry_ref,
        "wireframe_sources": {},
        "frames": frames,
    }
    path = tmp_path / "INPUT.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path, manifest


def test_straight_alpha_over_exact_integer_endpoints() -> None:
    background = np.full((1, 3, 3), 10, np.uint8)
    foreground = np.zeros((1, 3, 4), np.uint8)
    foreground[..., :3] = 210
    foreground[0, 0, 3] = 0
    foreground[0, 1, 3] = 255
    foreground[0, 2, 3] = 128
    output = author.straight_alpha_over(background, foreground)
    assert np.array_equal(output[0, 0], [10, 10, 10])
    assert np.array_equal(output[0, 1], [210, 210, 210])
    assert np.array_equal(output[0, 2], [110, 110, 110])


def test_manifest_rejects_noncontiguous_or_dropped_frames(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    manifest["frames"][1]["frame_index"] = 2  # type: ignore[index]
    with pytest.raises(author.DeliveryAuthorError, match="frame order"):
        author.validate_input_manifest(manifest)
    manifest["frames"] = manifest["frames"][:-1]  # type: ignore[index]
    with pytest.raises(author.DeliveryAuthorError, match="denominator"):
        author.validate_input_manifest(manifest)


def test_wireframe_is_disabled_by_default_before_any_scene_read(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    raw_ref = manifest["frames"][0]["raw"]  # type: ignore[index]
    registry_path = Path(manifest["robot_registry"]["path"])  # type: ignore[index]
    registry = json.loads(registry_path.read_text())
    registry["sessions"][0]["published_scene"]["scene_state"] = raw_ref
    manifest["robot_registry"] = _write_json(
        tmp_path / "ROBOT_REGISTRY_WIREFRAME.json", registry
    )
    manifest["wireframe_sources"] = {
        "strict_scene": {
            "session_id": manifest["session_id"],
            "scene_state": raw_ref,
        }
    }
    manifest["frames"][0]["robot"] = {  # type: ignore[index]
        "status": "DIAGNOSTIC_WIREFRAME",
        "source_id": "strict_scene",
        "scene_internal_index": 0,
        "source_frame_index": 0,
    }
    validated = author.validate_input_manifest(manifest)
    with pytest.raises(author.DeliveryAuthorError, match="explicit CLI opt-in"):
        author._load_wireframe_sources(validated, allow=False)


def test_author_integration_manifest_last_full_decode_and_no_overwrite(
    tmp_path: Path,
) -> None:
    input_path, _ = _manifest(tmp_path)
    output_root = tmp_path / "fresh_delivery"
    args = argparse.Namespace(
        input_manifest=input_path,
        output_root=output_root,
        panel_width=64,
        panel_height=48,
        ffmpeg_threads=1,
        allow_cpu_wireframe_fallback=False,
    )
    result = author.run(args)
    assert result["status"] == "DEVELOPMENT_ONLY_ARTIFACT_EXISTS"
    assert result["encoded_frame_count"] == 3
    assert result["counts"] == {
        "mask_ready": 1,
        "mask_partial": 1,
        "mask_pending": 1,
        "mask_empty_source_frames": 0,
        "clean_ready": 2,
        "clean_pending": 1,
        "robot_ready_pixels": 2,
        "robot_wireframe_diagnostic": 0,
        "robot_pending": 1,
        "straight_alpha_over_calls": 1,
    }
    video = output_root / "DELIVERY_RAW_MASK_CLEAN_ROBOT.mp4"
    manifest_path = output_root / "SESSION_MANIFEST.json"
    assert video.is_file() and manifest_path.is_file()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == result
    assert on_disk["zero_operations"] == {
        "frame_drop": 0,
        "frame_duplicate": 0,
        "frame_interpolation": 0,
        "neighbor_copy": 0,
        "zero_mask_substitution": 0,
        "morphology": 0,
        "hidden_fallback": 0,
    }
    decoded = subprocess.run(
        ["/usr/bin/ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert decoded.returncode == 0, decoded.stderr.decode(errors="replace")
    assert not list(tmp_path.glob(".fresh_delivery.stage.*"))
    with pytest.raises(FileExistsError):
        author.run(args)


def test_clean_cannot_bind_a_different_mask(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    manifest["frames"][0]["clean"]["consumed_mask_sha256"] = "0" * 64  # type: ignore[index]
    validated = author.validate_input_manifest(manifest)
    with pytest.raises(author.DeliveryAuthorError, match="consumed-mask mismatch"):
        list(
            author._iter_render_frames(
                validated,
                panel_width=64,
                panel_height=48,
                wireframes={},
                counts=author._empty_counts(),
            )
        )


def test_forbidden_paths_fail_before_read(tmp_path: Path) -> None:
    forbidden = tmp_path / "processed" / "x.png"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"not an image")
    with pytest.raises(author.DeliveryAuthorError, match="forbidden"):
        author._read_path(forbidden.resolve(), label="fixture")


def test_authority_accepted_empty_source_mask_remains_ready(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    zero = np.zeros((48, 64), np.uint8)
    zero_ref = _write_png(tmp_path / "zero_mask.png", zero)
    manifest["frames"][0]["mask"]["image"] = zero_ref  # type: ignore[index]
    manifest["frames"][0]["mask"]["source_mask_pixel_count"] = 0  # type: ignore[index]
    manifest["frames"][0]["clean"]["consumed_mask_sha256"] = zero_ref[  # type: ignore[index]
        "sha256"
    ]
    validated = author.validate_input_manifest(manifest)
    counts = author._empty_counts()
    frame = next(
        author._iter_render_frames(
            validated,
            panel_width=64,
            panel_height=48,
            wireframes={},
            counts=counts,
        )
    )
    assert frame.shape == (96, 128, 3)
    assert counts["mask_ready"] == 1
    assert counts["mask_pending"] == 0
    assert counts["mask_empty_source_frames"] == 1


def test_forged_zero_mask_disagrees_with_authority_pixel_count(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    zero = np.zeros((48, 64), np.uint8)
    zero_ref = _write_png(tmp_path / "forged_zero_mask.png", zero)
    manifest["frames"][0]["mask"]["image"] = zero_ref  # type: ignore[index]
    manifest["frames"][0]["clean"]["consumed_mask_sha256"] = zero_ref[  # type: ignore[index]
        "sha256"
    ]
    validated = author.validate_input_manifest(manifest)
    with pytest.raises(author.DeliveryAuthorError, match="pixel count mismatch"):
        next(
            author._iter_render_frames(
                validated,
                panel_width=64,
                panel_height=48,
                wireframes={},
                counts=author._empty_counts(),
            )
        )


def test_missing_accepted_mask_file_fails_closed(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    missing = dict(manifest["frames"][0]["mask"]["image"])  # type: ignore[index]
    missing["path"] = str((tmp_path / "missing.png").resolve())
    manifest["frames"][0]["mask"]["image"] = missing  # type: ignore[index]
    validated = author.validate_input_manifest(manifest)
    with pytest.raises(FileNotFoundError):
        next(
            author._iter_render_frames(
                validated,
                panel_width=64,
                panel_height=48,
                wireframes={},
                counts=author._empty_counts(),
            )
        )


def test_robot_ready_pixels_must_be_inside_registry_pass_set(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path)
    manifest["frames"][2]["robot"] = dict(  # type: ignore[index]
        manifest["frames"][1]["robot"]  # type: ignore[index]
    )
    manifest["frames"][2]["robot"]["source_frame_index"] = 2  # type: ignore[index]
    validated = author.validate_input_manifest(manifest)
    with pytest.raises(author.DeliveryAuthorError, match="outside scene pass set"):
        author._load_wireframe_sources(validated, allow=False)
