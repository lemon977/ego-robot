from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "tools" / "run_raw_apriltag_session_static_table_plane.py"
SPEC = importlib.util.spec_from_file_location(
    "raw_apriltag_static_plane_tested", SOURCE
)
assert SPEC and SPEC.loader
plane_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plane_tool
SPEC.loader.exec_module(plane_tool)


SESSION = "grap_a_cap_901"
K = np.asarray([[700.0, 0.0, 319.5], [0.0, 700.0, 239.5], [0.0, 0.0, 1.0]])


def _tag_rgb() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 400)
    source = np.asarray([[0, 0], [399, 0], [399, 399], [0, 399]], dtype=np.float32)
    local = plane_tool.tag_corners_local(0.1)
    rotation_vector = np.asarray([3.0, 0.2, 0.1], dtype=np.float64)
    translation_vector = np.asarray([0.0, 0.0, 0.45], dtype=np.float64)
    target = (
        cv2.projectPoints(
            local,
            rotation_vector,
            translation_vector,
            K,
            np.zeros(5),
        )[0]
        .reshape(4, 2)
        .astype(np.float32)
    )
    homography = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(marker, homography, (640, 480), borderValue=255)
    occupied = cv2.warpPerspective(
        np.full_like(marker, 255), homography, (640, 480), borderValue=0
    )
    image = np.full((480, 640), 255, dtype=np.uint8)
    image[occupied > 0] = warped[occupied > 0]
    return np.repeat(image[:, :, None], 3, axis=2)


def _make_raw_session(
    tmp_path: Path,
    *,
    frame_count: int,
    blank_frames: frozenset[int] = frozenset(),
    c2w_mutator: Any = None,
    session_id: str = SESSION,
) -> tuple[Path, Path]:
    raw_root = tmp_path / session_id / "preprocess" / "all_data"
    raw_root.mkdir(parents=True)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    tag = _tag_rgb()
    blank = np.full_like(tag, 255)
    for index in range(frame_count):
        frame = raw_root / f"{index:05d}"
        frame.mkdir()
        Image.fromarray(blank if index in blank_frames else tag).save(frame / "rgb.png")
        c2w = np.eye(4, dtype=np.float64)
        if c2w_mutator is not None:
            c2w_mutator(index, c2w)
        (frame / "training_data.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "k": K.tolist(),
                        "c2w": c2w.tolist(),
                        "w": 640,
                        "h": 480,
                    }
                }
            ),
            encoding="utf-8",
        )
    return raw_root, output_parent


def _fixed_observer(*, reject_first_max_error: bool = False):
    calls = 0
    local = plane_tool.tag_corners_local(0.1)
    rotation_vector = np.asarray([3.0, 0.2, 0.1], dtype=np.float64)
    translation_vector = np.asarray([0.0, 0.0, 0.45], dtype=np.float64)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_vector
    corners = cv2.projectPoints(
        local, rotation_vector, translation_vector, K, np.zeros(5)
    )[0].reshape(4, 2)

    def observe(_rgb: np.ndarray, _intrinsic: np.ndarray) -> Mapping[str, Any]:
        nonlocal calls
        current = calls
        calls += 1
        errors = np.zeros(4, dtype=np.float64)
        if reject_first_max_error and current == 0:
            errors[0] = 1.2
            return {
                "accepted": False,
                "reason": "CORNER_REPROJECTION_GT_1PX",
                "detected_ids": [0],
                "tag_to_camera": transform,
                "corners_uv": corners,
                "corner_reprojection_errors_px": errors,
                "corner_reprojection_mean_px": float(np.mean(errors)),
                "corner_reprojection_max_px": float(np.max(errors)),
            }
        return {
            "accepted": True,
            "reason": None,
            "detected_ids": [0],
            "tag_to_camera": transform,
            "corners_uv": corners,
            "corner_reprojection_errors_px": errors,
            "corner_reprojection_mean_px": 0.0,
            "corner_reprojection_max_px": 0.0,
        }

    return observe


def _load_output(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_bytes())
    candidate_path = Path(manifest["candidate_ref"]["path"])
    payload = candidate_path.read_bytes()
    assert len(payload) == manifest["candidate_ref"]["bytes"]
    assert hashlib.sha256(payload).hexdigest() == manifest["candidate_ref"]["sha256"]
    return manifest, json.loads(payload)


def test_real_synthetic_apriltag_scan_selects_earliest_15_and_records_every_input(
    tmp_path: Path,
) -> None:
    raw_root, output_parent = _make_raw_session(
        tmp_path,
        frame_count=17,
        blank_frames=frozenset({0}),
    )
    manifest_path = plane_tool.run(
        raw_root=raw_root,
        session_id=SESSION,
        output_parent=output_parent,
        output_name="candidate-real",
    )
    manifest, candidate = _load_output(manifest_path)

    assert manifest["status"] == "DEVELOPMENT_ONLY_ARTIFACT_EXISTS"
    assert candidate["schema_version"] == "table-plane-static-candidate-v1"
    assert candidate["session_id"] == SESSION
    assert candidate["scan_summary"] == {
        "frames_scanned": 17,
        "accepted_by_tag_gate": 16,
        "accepted_not_selected": 1,
        "selected_anchor_count": 15,
        "selected_anchor_frames": list(range(1, 16)),
        "rejection_counts": {
            "TAG_ID_0_NOT_DETECTED": 1,
            "TAG_ID_0_DUPLICATE": 0,
            "APRILTAG_IPPE_POSE_FAILED": 0,
            "APRILTAG_POSE_BEHIND_CAMERA": 0,
            "CORNER_REPROJECTION_GT_1PX": 0,
        },
    }
    assert len(candidate["frames"]) == 17
    for index, frame in enumerate(candidate["frames"]):
        assert frame["frame_index"] == index
        assert set(frame["inputs"]) == {"rgb", "training_data"}
        for record in frame["inputs"].values():
            assert Path(record["path"]).is_absolute()
            assert record["bytes"] > 0
            assert len(record["sha256"]) == 64
            assert type(record["device"]) is int
            assert type(record["inode"]) is int
    assert [
        row["frame_index"] for row in candidate["frames"] if row["selected_anchor"]
    ] == list(range(1, 16))
    assert (
        candidate["frames"][16]["tag_detection"]["acceptance_disposition"]
        == "ACCEPTED_NOT_SELECTED_AFTER_EARLIEST_15"
    )
    assert candidate["estimation_quality"]["fit_gate_pass"] is True
    assert (
        candidate["estimation_quality"]["metrics"][
            "selected_tag_corner_reprojection_max_px"
        ]
        <= 1.0
    )
    assert candidate["plane"]["coordinate_frame"] == "FROZEN_RAW_WORLD_C2W"
    assert candidate["plane"]["cardinality"] == 1
    assert candidate["scope"]["table_support_generated"] is False
    assert candidate["scope"]["table_support_expanded"] is False
    assert candidate["scope"]["contact_generated"] is False
    assert candidate["scope"]["per_frame_reanchor"] is False
    assert candidate["scope"]["fallback_used"] is False
    assert candidate["scope"]["operation_counters"] == plane_tool.OPERATION_COUNTERS
    assert not any(candidate["scope"]["operation_counters"].values())
    assert candidate["plumbing_only"] is True
    assert candidate["formal"] is False
    assert candidate["candidate_requires_human_review"] is True
    assert candidate["human_review_status"] == "PENDING"
    assert candidate["next_bucket_blocked"] is True


def test_gate_uses_max_corner_error_not_legacy_four_corner_mean(tmp_path: Path) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=16)
    manifest_path = plane_tool.run(
        raw_root=raw_root,
        session_id=SESSION,
        output_parent=output_parent,
        output_name="candidate-max-gate",
        frame_observer=_fixed_observer(reject_first_max_error=True),
    )
    _, candidate = _load_output(manifest_path)
    first = candidate["frames"][0]["tag_detection"]
    assert first["accepted_by_tag_gate"] is False
    assert first["rejection_reason"] == "CORNER_REPROJECTION_GT_1PX"
    assert first["corner_reprojection_mean_px"] == pytest.approx(0.3)
    assert first["corner_reprojection_max_px"] == pytest.approx(1.2)
    assert candidate["scan_summary"]["selected_anchor_frames"] == list(range(1, 16))


def test_unique_id0_is_accepted_alongside_other_tags_but_duplicate_id0_fails(
    tmp_path: Path,
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    base = _fixed_observer()

    def with_other_tag(rgb: np.ndarray, intrinsic: np.ndarray) -> Mapping[str, Any]:
        result = dict(base(rgb, intrinsic))
        result["detected_ids"] = [7, 0]
        return result

    manifest_path = plane_tool.run(
        raw_root=raw_root,
        session_id=SESSION,
        output_parent=output_parent,
        output_name="unique-id0-with-other-tag",
        frame_observer=with_other_tag,
    )
    _, candidate = _load_output(manifest_path)
    assert candidate["scan_summary"]["selected_anchor_frames"] == list(range(15))
    assert all(
        row["tag_detection"]["detected_ids"] == [7, 0] for row in candidate["frames"]
    )

    duplicate_root, duplicate_output = _make_raw_session(
        tmp_path / "duplicate", frame_count=15
    )
    duplicate_base = _fixed_observer()

    def with_duplicate_id0(rgb: np.ndarray, intrinsic: np.ndarray) -> Mapping[str, Any]:
        result = dict(duplicate_base(rgb, intrinsic))
        result["detected_ids"] = [0, 8, 0]
        return result

    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="frozen gate"):
        plane_tool.run(
            raw_root=duplicate_root,
            session_id=SESSION,
            output_parent=duplicate_output,
            output_name="duplicate-id0-must-remain-absent",
            frame_observer=with_duplicate_id0,
        )
    assert list(duplicate_output.iterdir()) == []


def test_observer_cannot_forge_reprojection_or_put_tag_corners_behind_camera(
    tmp_path: Path,
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    base = _fixed_observer()

    def forged_pose(rgb: np.ndarray, intrinsic: np.ndarray) -> Mapping[str, Any]:
        result = dict(base(rgb, intrinsic))
        transform = np.asarray(result["tag_to_camera"]).copy()
        transform[0, 3] += 0.02
        result["tag_to_camera"] = transform
        return result

    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="metrics disagree"):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="forged-reprojection-must-remain-absent",
            frame_observer=forged_pose,
        )

    def crossing_camera_plane(
        _rgb: np.ndarray, intrinsic: np.ndarray
    ) -> Mapping[str, Any]:
        rotation_vector = np.asarray([np.pi / 2.0, 0.0, 0.0], dtype=np.float64)
        rotation, _ = cv2.Rodrigues(rotation_vector)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[2, 3] = 0.01
        local = plane_tool.tag_corners_local(0.1)
        camera = plane_tool.transform_points(transform, local)
        projected = camera @ intrinsic.T
        projected = projected[:, :2] / projected[:, 2, None]
        return {
            "accepted": True,
            "reason": None,
            "detected_ids": [0],
            "tag_to_camera": transform,
            "corners_uv": projected,
            "corner_reprojection_errors_px": np.zeros(4),
            "corner_reprojection_mean_px": 0.0,
            "corner_reprojection_max_px": 0.0,
        }

    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="corners behind"):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="behind-corner-must-remain-absent",
            frame_observer=crossing_camera_plane,
        )
    assert list(output_parent.iterdir()) == []


def test_fewer_than_15_accepted_frames_fails_without_creating_output(
    tmp_path: Path,
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=14)
    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="15 required"):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="must-remain-absent",
            frame_observer=_fixed_observer(),
        )
    assert not (output_parent / "must-remain-absent").exists()


def test_static_fit_quality_failure_does_not_publish_pass(tmp_path: Path) -> None:
    def move_one_frame(index: int, c2w: np.ndarray) -> None:
        if index == 14:
            c2w[2, 3] = 0.20

    raw_root, output_parent = _make_raw_session(
        tmp_path,
        frame_count=15,
        c2w_mutator=move_one_frame,
    )
    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="fit/gate failed"):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="bad-fit-must-remain-absent",
            frame_observer=_fixed_observer(),
        )
    assert not (output_parent / "bad-fit-must-remain-absent").exists()


def test_output_is_a_new_direct_child_and_never_overwrites(tmp_path: Path) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    kwargs = {
        "raw_root": raw_root,
        "session_id": SESSION,
        "output_parent": output_parent,
        "output_name": "exclusive-candidate",
        "frame_observer": _fixed_observer(),
    }
    first = plane_tool.run(**kwargs)
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        plane_tool.run(**kwargs)
    assert first.read_bytes() == before
    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="direct-child"):
        plane_tool.run(**{**kwargs, "output_name": "nested/escape"})


def test_terminal_publication_is_hidden_until_complete_and_noreplace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    destination = output_parent / "terminal-candidate"
    original = plane_tool._rename_noreplace
    observed: dict[str, Any] = {}

    def inspect_then_publish(**kwargs: Any) -> None:
        assert not destination.exists()
        source_fd = kwargs["source_fd"]
        staging = Path(f"/proc/self/fd/{source_fd}") / kwargs["source_name"]
        observed["leaf_names"] = {path.name for path in staging.iterdir()}
        original(**kwargs)

    monkeypatch.setattr(plane_tool, "_rename_noreplace", inspect_then_publish)
    manifest_path = plane_tool.run(
        raw_root=raw_root,
        session_id=SESSION,
        output_parent=output_parent,
        output_name=destination.name,
        frame_observer=_fixed_observer(),
    )
    assert observed["leaf_names"] == {
        "TABLE_PLANE_STATIC.json",
        "SESSION_MANIFEST.json",
    }
    assert manifest_path == destination / "SESSION_MANIFEST.json"
    assert set(path.name for path in destination.iterdir()) == observed["leaf_names"]
    monkeypatch.setattr(plane_tool, "_rename_noreplace", original)

    occupied = output_parent / "occupied-empty-directory"
    occupied.mkdir()
    occupied_identity = occupied.stat()
    with pytest.raises(FileExistsError):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name=occupied.name,
            frame_observer=_fixed_observer(),
        )
    assert occupied.is_dir() and not any(occupied.iterdir())
    assert occupied.stat().st_ino == occupied_identity.st_ino
    assert not any(
        path.name.startswith(".raw-apriltag-plane-stage-")
        for path in output_parent.iterdir()
    )


def test_cpfs_empty_directory_listing_view_cannot_hide_direct_bound_terminal_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    original_listdir = plane_tool.os.listdir
    expected_names = {"TABLE_PLANE_STATIC.json", "SESSION_MANIFEST.json"}
    hidden_terminal_views = 0

    def cpfs_empty_view(path: Any) -> list[str]:
        nonlocal hidden_terminal_views
        names = original_listdir(path)
        if isinstance(path, int) and set(names) == expected_names:
            hidden_terminal_views += 1
            return []
        return names

    monkeypatch.setattr(plane_tool.os, "listdir", cpfs_empty_view)
    manifest_path = plane_tool.run(
        raw_root=raw_root,
        session_id=SESSION,
        output_parent=output_parent,
        output_name="cpfs-empty-list-view-terminal",
        frame_observer=_fixed_observer(),
    )
    assert hidden_terminal_views >= 1
    assert manifest_path.is_file()
    assert set(path.name for path in manifest_path.parent.iterdir()) == expected_names
    assert not any(
        path.name.startswith(".raw-apriltag-plane-stage-")
        for path in output_parent.iterdir()
    )


def test_cpfs_unsupported_renameat2_uses_locked_same_parent_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    source_identity: tuple[int, int] | None = None

    def unsupported_renameat2(**kwargs: Any) -> None:
        nonlocal source_identity
        source = os.stat(
            kwargs["source_name"],
            dir_fd=kwargs["source_fd"],
            follow_symlinks=False,
        )
        source_identity = (source.st_dev, source.st_ino)
        with pytest.raises(FileNotFoundError):
            os.stat(
                kwargs["target_name"],
                dir_fd=kwargs["target_fd"],
                follow_symlinks=False,
            )
        raise OSError(errno.EINVAL, "injected CPFS unsupported rename flag")

    monkeypatch.setattr(
        plane_tool,
        "_renameat2_noreplace",
        unsupported_renameat2,
    )
    manifest_path = plane_tool.run(
        raw_root=raw_root,
        session_id=SESSION,
        output_parent=output_parent,
        output_name="cpfs-locked-publication",
        frame_observer=_fixed_observer(),
    )
    final = manifest_path.parent.stat()
    assert source_identity == (final.st_dev, final.st_ino)
    assert set(path.name for path in manifest_path.parent.iterdir()) == {
        "TABLE_PLANE_STATIC.json",
        "SESSION_MANIFEST.json",
    }
    assert not any(
        path.name.startswith(".raw-apriltag-plane-stage-")
        for path in output_parent.iterdir()
    )


def test_cpfs_compatibility_publication_never_overwrites_injected_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    destination = output_parent / "foreign-empty-target"
    foreign_identity: tuple[int, int] | None = None

    def unsupported_after_target_injection(**kwargs: Any) -> None:
        nonlocal foreign_identity
        os.mkdir(kwargs["target_name"], 0o711, dir_fd=kwargs["target_fd"])
        foreign = os.stat(
            kwargs["target_name"],
            dir_fd=kwargs["target_fd"],
            follow_symlinks=False,
        )
        foreign_identity = (foreign.st_dev, foreign.st_ino)
        raise OSError(errno.EINVAL, "injected CPFS unsupported rename flag")

    monkeypatch.setattr(
        plane_tool,
        "_renameat2_noreplace",
        unsupported_after_target_injection,
    )
    with pytest.raises(FileExistsError):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name=destination.name,
            frame_observer=_fixed_observer(),
        )
    current = destination.stat()
    assert foreign_identity == (current.st_dev, current.st_ino)
    assert destination.is_dir() and list(destination.iterdir()) == []
    assert not any(
        path.name.startswith(".raw-apriltag-plane-stage-")
        for path in output_parent.iterdir()
    )


def test_second_leaf_failure_rolls_back_private_staging_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    original = plane_tool._write_leaf

    def fail_manifest(
        parent_fd: int, name: str, payload: bytes, absolute_path: Path
    ) -> dict[str, Any]:
        if name == "SESSION_MANIFEST.json":
            raise OSError("injected terminal-manifest write failure")
        return original(parent_fd, name, payload, absolute_path)

    monkeypatch.setattr(plane_tool, "_write_leaf", fail_manifest)
    with pytest.raises(OSError, match="injected terminal-manifest"):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="must-have-zero-partial-output",
            frame_observer=_fixed_observer(),
        )
    assert list(output_parent.iterdir()) == []


def test_transient_enotempty_rollback_preserves_primary_and_allows_serial_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    original_rename = plane_tool._rename_noreplace
    original_rmdir = plane_tool.os.rmdir
    primary = OSError(errno.EIO, "injected primary publication failure")
    publish_calls = 0
    staging_rmdir_calls = 0

    def fail_first_publish(**kwargs: Any) -> None:
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 1:
            raise primary
        original_rename(**kwargs)

    def transient_enotempty(path: Any, *, dir_fd: int | None = None) -> None:
        nonlocal staging_rmdir_calls
        if str(path).startswith(".raw-apriltag-plane-stage-"):
            staging_rmdir_calls += 1
            if staging_rmdir_calls == 1:
                raise OSError(errno.ENOTEMPTY, "injected transient ENOTEMPTY", path)
        original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(plane_tool, "_rename_noreplace", fail_first_publish)
    monkeypatch.setattr(plane_tool.os, "rmdir", transient_enotempty)
    with pytest.raises(OSError) as caught:
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="primary-failure-must-remain-absent",
            frame_observer=_fixed_observer(),
        )
    assert caught.value is primary
    assert not getattr(caught.value, "__notes__", ())
    assert staging_rmdir_calls == 2
    assert list(output_parent.iterdir()) == []

    manifest_path = plane_tool.run(
        raw_root=raw_root,
        session_id=SESSION,
        output_parent=output_parent,
        output_name="serial-retry-terminal",
        frame_observer=_fixed_observer(),
    )
    assert manifest_path.is_file()
    assert set(path.name for path in manifest_path.parent.iterdir()) == {
        "TABLE_PLANE_STATIC.json",
        "SESSION_MANIFEST.json",
    }


def test_persistent_enotempty_is_not_allowed_to_replace_primary_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    independent_parent = tmp_path / "independent-output-parent"
    independent_parent.mkdir()
    original_rename = plane_tool._rename_noreplace
    original_rmdir = plane_tool.os.rmdir
    primary = OSError(errno.EIO, "injected durable primary failure")
    staging_rmdir_calls = 0

    def fail_publish(**_kwargs: Any) -> None:
        raise primary

    def persistent_enotempty(path: Any, *, dir_fd: int | None = None) -> None:
        nonlocal staging_rmdir_calls
        if str(path).startswith(".raw-apriltag-plane-stage-"):
            staging_rmdir_calls += 1
            raise OSError(errno.ENOTEMPTY, "injected persistent ENOTEMPTY", path)
        original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(plane_tool, "_rename_noreplace", fail_publish)
    monkeypatch.setattr(plane_tool.os, "rmdir", persistent_enotempty)
    with pytest.raises(OSError) as caught:
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="persistent-primary-failure",
            frame_observer=_fixed_observer(),
        )
    assert caught.value is primary
    notes = getattr(caught.value, "__notes__", ())
    assert len(notes) == 1
    assert "ROLLBACK_FAILED" in notes[0]
    assert "ENOTEMPTY" in notes[0]
    assert staging_rmdir_calls == plane_tool.ROLLBACK_RMDIR_ATTEMPTS
    stages = [
        path
        for path in output_parent.iterdir()
        if path.name.startswith(".raw-apriltag-plane-stage-")
    ]
    assert len(stages) == 1
    assert stages[0].is_dir() and list(stages[0].iterdir()) == []
    assert not (output_parent / "persistent-primary-failure").exists()

    monkeypatch.setattr(plane_tool, "_rename_noreplace", original_rename)
    monkeypatch.setattr(plane_tool.os, "rmdir", original_rmdir)
    manifest_path = plane_tool.run(
        raw_root=raw_root,
        session_id=SESSION,
        output_parent=independent_parent,
        output_name="independent-parent-retry",
        frame_observer=_fixed_observer(),
    )
    assert manifest_path.is_file()
    stages[0].rmdir()


def test_rollback_refuses_to_delete_foreign_staging_entry(tmp_path: Path) -> None:
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    parent_fd = os.open(output_parent, plane_tool.DIR_FLAGS)
    staging_name = ""
    staging_fd = -1
    try:
        staging_name, staging_fd, staging_identity = (
            plane_tool._create_private_staging_child(parent_fd)
        )
        foreign_fd = os.open(
            "foreign-entry",
            plane_tool.WRITE_FLAGS,
            0o440,
            dir_fd=staging_fd,
        )
        os.close(foreign_fd)
        with pytest.raises(plane_tool.RawAprilTagPlaneError, match="foreign entries"):
            plane_tool._rollback_private_staging(
                parent_fd=parent_fd,
                staging_fd=staging_fd,
                staging_name=staging_name,
                staging_identity=staging_identity,
            )
        assert os.listdir(staging_fd) == ["foreign-entry"]
        os.unlink("foreign-entry", dir_fd=staging_fd)
        os.rmdir(staging_name, dir_fd=parent_fd)
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(parent_fd)


def test_forbidden_or_false_session_identity_is_rejected_before_output(
    tmp_path: Path,
) -> None:
    forbidden_root, output_parent = _make_raw_session(
        tmp_path,
        frame_count=15,
        session_id="grap_a_cap_025",
    )
    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="forbidden"):
        plane_tool.run(
            raw_root=forbidden_root,
            session_id="grap_a_cap_025",
            output_parent=output_parent,
            output_name="forbidden",
            frame_observer=_fixed_observer(),
        )
    assert not (output_parent / "forbidden").exists()

    allowed_root, second_output = _make_raw_session(
        tmp_path / "second",
        frame_count=15,
    )
    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="does not match"):
        plane_tool.run(
            raw_root=allowed_root,
            session_id="grap_a_cap_902",
            output_parent=second_output,
            output_name="false-session",
            frame_observer=_fixed_observer(),
        )
    assert not (second_output / "false-session").exists()


def test_raw_leaf_symlink_and_nonrigid_c2w_fail_closed_before_output(
    tmp_path: Path,
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    rgb = raw_root / "00000" / "rgb.png"
    target = raw_root / "00000" / "rgb-real.png"
    rgb.rename(target)
    rgb.symlink_to(target.name)
    with pytest.raises(
        plane_tool.RawAprilTagPlaneError, match="cannot open required RAW leaf"
    ):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="symlink-must-remain-absent",
            frame_observer=_fixed_observer(),
        )
    assert not (output_parent / "symlink-must-remain-absent").exists()

    rigid_root, rigid_output = _make_raw_session(tmp_path / "rigid", frame_count=15)
    metadata_path = rigid_root / "00000" / "training_data.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["c2w"][0][0] = 2.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(plane_tool.RawAprilTagPlaneError, match=r"finite SE\(3\)"):
        plane_tool.run(
            raw_root=rigid_root,
            session_id=SESSION,
            output_parent=rigid_output,
            output_name="nonrigid-must-remain-absent",
            frame_observer=_fixed_observer(),
        )
    assert not (rigid_output / "nonrigid-must-remain-absent").exists()


def test_raw_root_rename_during_scan_cannot_publish_stale_absolute_refs(
    tmp_path: Path,
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    original_observer = _fixed_observer()
    moved_root = raw_root.with_name("all_data-moved-after-held-open")
    mutated = False

    def rename_root_then_observe(
        rgb: np.ndarray, intrinsic: np.ndarray
    ) -> Mapping[str, Any]:
        nonlocal mutated
        if not mutated:
            raw_root.rename(moved_root)
            raw_root.mkdir()
            mutated = True
        return original_observer(rgb, intrinsic)

    with pytest.raises(plane_tool.RawAprilTagPlaneError, match="RAW root identity"):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="stale-ref-must-remain-absent",
            frame_observer=rename_root_then_observe,
        )
    assert list(output_parent.iterdir()) == []


def test_raw_ancestor_symlink_swap_is_rejected_despite_same_terminal_inode(
    tmp_path: Path,
) -> None:
    raw_root, output_parent = _make_raw_session(
        tmp_path / "raw-container", frame_count=15
    )
    original_observer = _fixed_observer()
    session_path = raw_root.parent.parent
    moved_session = session_path.with_name(f"{SESSION}-moved")
    mutated = False

    def swap_ancestor_then_observe(
        rgb: np.ndarray, intrinsic: np.ndarray
    ) -> Mapping[str, Any]:
        nonlocal mutated
        if not mutated:
            session_path.rename(moved_session)
            session_path.symlink_to(moved_session.name, target_is_directory=True)
            mutated = True
        return original_observer(rgb, intrinsic)

    with pytest.raises((plane_tool.RawAprilTagPlaneError, OSError)):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="symlinked-raw-ancestor-must-remain-absent",
            frame_observer=swap_ancestor_then_observe,
        )
    assert list(output_parent.iterdir()) == []


def test_output_ancestor_symlink_swap_rolls_back_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root, _ = _make_raw_session(tmp_path / "raw-container", frame_count=15)
    output_container = tmp_path / "output-container"
    output_parent = output_container / "outputs"
    output_parent.mkdir(parents=True)
    moved_container = tmp_path / "output-container-moved"
    original = plane_tool._write_leaf
    mutated = False

    def write_then_swap_ancestor(
        parent_fd: int, name: str, payload: bytes, absolute_path: Path
    ) -> dict[str, Any]:
        nonlocal mutated
        result = original(parent_fd, name, payload, absolute_path)
        if name == "SESSION_MANIFEST.json" and not mutated:
            output_container.rename(moved_container)
            output_container.symlink_to(moved_container.name, target_is_directory=True)
            mutated = True
        return result

    monkeypatch.setattr(plane_tool, "_write_leaf", write_then_swap_ancestor)
    with pytest.raises((plane_tool.RawAprilTagPlaneError, OSError)):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="symlinked-output-ancestor-must-remain-absent",
            frame_observer=_fixed_observer(),
        )
    assert list((moved_container / "outputs").iterdir()) == []


def test_same_inode_raw_leaf_mutation_during_scan_fails_before_output(
    tmp_path: Path,
) -> None:
    raw_root, output_parent = _make_raw_session(tmp_path, frame_count=15)
    original_observer = _fixed_observer()
    mutated = False

    def touch_consumed_leaf_then_observe(
        rgb: np.ndarray, intrinsic: np.ndarray
    ) -> Mapping[str, Any]:
        nonlocal mutated
        if not mutated:
            leaf = raw_root / "00000" / "rgb.png"
            before = leaf.stat()
            os.utime(
                leaf,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            assert leaf.stat().st_ino == before.st_ino
            mutated = True
        return original_observer(rgb, intrinsic)

    with pytest.raises(
        plane_tool.RawAprilTagPlaneError, match="RAW leaf identity drift"
    ):
        plane_tool.run(
            raw_root=raw_root,
            session_id=SESSION,
            output_parent=output_parent,
            output_name="mutated-leaf-must-remain-absent",
            frame_observer=touch_consumed_leaf_then_observe,
        )
    assert list(output_parent.iterdir()) == []


def test_source_has_no_support_contact_fallback_or_forbidden_payload_route() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "cv2.erode" not in source
    assert "fillConvexPoly" not in source
    assert "connectedComponents" not in source
    assert "inpaint" not in source
    assert "Object6D" not in source
    assert "processed/" not in source
    assert "labels/" not in source
    assert "cuda" not in source.casefold()
