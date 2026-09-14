from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import launch_clean_spatial12_bounded_v4 as launcher


def capacity_fixture(project: Path, expected_bytes: int = 4096) -> dict[str, str]:
    control = project / "tasks/control"
    control.mkdir(parents=True)
    reservation = project / launcher.CAPACITY_RESERVATION_RELATIVE
    reservation.write_bytes(b"x" * expected_bytes)
    metadata = reservation.stat()
    authority_path = project / launcher.CAPACITY_AUTHORITY_RELATIVE
    authority = {
        "schema_version": "project-capacity-live-authority-v1",
        "status": "PASS_CAPACITY_HELD",
        "authorizes_execution": True,
        "project_root": str(project),
        "authority_path": launcher.CAPACITY_AUTHORITY_RELATIVE.as_posix(),
        "reservation_path": launcher.CAPACITY_RESERVATION_RELATIVE.as_posix(),
        "requested_bytes": expected_bytes,
        "creator_pid": os.getpid(),
        "creator_process_start_ticks": launcher.process_start_ticks(os.getpid()),
        "allocation": {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "logical_bytes": metadata.st_size,
        },
        "verification": {
            "same_descriptor_readback": True,
            "full_pattern_match": True,
            "st_blocks_minimum_met": True,
            "path_descriptor_identity_match": True,
        },
    }
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    return {
        "PROJECT_CAPACITY_LIVE_AUTHORITY": str(authority_path),
        "PROJECT_CAPACITY_RESERVATION": str(reservation),
        "PROJECT_CAPACITY_RESERVED_BYTES": str(expected_bytes),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"PROJECT_CAPACITY_LIVE_AUTHORITY": None},
        {"PROJECT_CAPACITY_RESERVED_BYTES": "8192"},
        {"PROJECT_CAPACITY_RESERVATION": "/wrong/reserve"},
    ],
)
def test_capacity_environment_missing_or_wrong_fails_closed(
    tmp_path: Path, mutation: dict[str, str | None]
) -> None:
    environment = capacity_fixture(tmp_path)
    for key, value in mutation.items():
        if value is None:
            environment.pop(key)
        else:
            environment[key] = value
    with pytest.raises(launcher.CleanV4LaunchHold, match="environment mismatch"):
        launcher.validate_capacity_authority(
            tmp_path, environment, expected_bytes=4096
        )


def test_capacity_authority_and_allocation_are_exact(tmp_path: Path) -> None:
    environment = capacity_fixture(tmp_path)
    evidence = launcher.validate_capacity_authority(
        tmp_path, environment, expected_bytes=4096
    )
    assert evidence["profile"] == "clean-capacity-v1"
    assert evidence["requested_bytes"] == 4096
    assert evidence["reservation"]["logical_bytes"] == 4096

    authority_path = tmp_path / launcher.CAPACITY_AUTHORITY_RELATIVE
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["reservation_path"] = "tasks/control/wrong.reserve"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(launcher.CleanV4LaunchHold, match="identity/status"):
        launcher.validate_capacity_authority(
            tmp_path, environment, expected_bytes=4096
        )


def test_runtime_python_and_opencv_are_exactly_pinned() -> None:
    evidence = launcher.validate_runtime_environment(
        executable=launcher.RUNTIME_PYTHON,
        version_info=launcher.EXPECTED_PYTHON_VERSION,
        cv2_version=launcher.EXPECTED_CV2_VERSION,
    )
    assert evidence["python_path"] == str(launcher.RUNTIME_PYTHON)
    assert evidence["python_version"] == "3.11.15"
    assert evidence["opencv_version"] == "4.11.0"
    with pytest.raises(launcher.CleanV4LaunchHold, match="runtime Python drift"):
        launcher.validate_runtime_environment(executable="/usr/local/bin/python")
    with pytest.raises(launcher.CleanV4LaunchHold, match="Python version drift"):
        launcher.validate_runtime_environment(
            executable=launcher.RUNTIME_PYTHON,
            version_info=(3, 11, 14),
            cv2_version=launcher.EXPECTED_CV2_VERSION,
        )
    with pytest.raises(launcher.CleanV4LaunchHold, match="OpenCV version drift"):
        launcher.validate_runtime_environment(
            executable=launcher.RUNTIME_PYTHON,
            version_info=launcher.EXPECTED_PYTHON_VERSION,
            cv2_version="4.10.0",
        )


def spatial_config() -> dict[str, object]:
    return {
        "review_frames": list(range(12)),
        "temporal_windows": [],
        "bounded_resource_policy": {
            "stage": "SPATIAL12_VISUAL_STOP_BEFORE_TEMPORAL",
            "spatial_frames_persisted_max": 12,
            "whole_archive_extraction_forbidden": True,
        },
    }


def test_temporal_execution_remains_forbidden() -> None:
    config = spatial_config()
    assert launcher.validate_spatial_config(config) == list(range(12))
    config["temporal_windows"] = [{"frames": list(range(12))}]
    with pytest.raises(launcher.CleanV4LaunchHold, match="temporal"):
        launcher.validate_spatial_config(config)


def test_fixed_v3_argv_uses_held_reserve_substitution() -> None:
    assert launcher.fixed_base_argv()[0] == str(launcher.RUNTIME_PYTHON)
    argv = launcher.fixed_base_argv(python="python-fixed")
    assert argv[0] == "python-fixed"
    assert argv[argv.index("--workspace-reserve-bytes") + 1] == "0"
    assert argv[argv.index("--persistent-bytes-max") + 1] == str(64 * 1024 * 1024)
    assert argv[argv.index("--output") + 1] == str(
        launcher.PROJECT / launcher.STAGING_RELATIVE
    )
    assert argv[argv.index("--donor-authority") + 1] == str(
        launcher.PROJECT
        / launcher.RUN_RELATIVE
        / "DONOR_ARCHIVE_AUTHORITY.json"
    )
    assert "--validate-only" not in argv


def make_staging(root: Path) -> tuple[Path, Path]:
    staging = root / "run/.producer.v4-staging"
    staging.mkdir(parents=True)
    review = staging / "RAW_VS_MEDOID_LIGHT_SOURCE_12.png"
    image = np.zeros((12 * 64, 800, 3), dtype=np.uint8)
    for row in range(12):
        image[row * 64 : (row + 1) * 64] = (row * 10, row * 3, row * 5)
    assert cv2.imwrite(str(review), image)
    result = {
        "status": "PASS_SPATIAL12_DIAGNOSTIC_ONLY",
        "frames": list(range(12)),
        "temporal_windows": [],
        "consumption_authorized": False,
        "bounded_orchestration": {"actual_persistent_bytes": 0},
        "assets": {
            review.name: {
                "bytes": review.stat().st_size,
                "sha256": launcher.sha256(review),
            }
        },
    }
    (staging / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    return staging, review


def test_all_twelve_rows_are_watermarked_and_result_asset_is_refreshed(
    tmp_path: Path,
) -> None:
    staging, review = make_staging(tmp_path)
    before_sha = launcher.sha256(review)
    result, actual_bytes = launcher.watermark_and_annotate_result(
        staging,
        frames=list(range(12)),
        capacity_evidence={"profile": "clean-capacity-v1", "requested_bytes": 4096},
        workspace_free_before_bytes=64 * 1024 * 1024,
    )
    image = cv2.imread(str(review), cv2.IMREAD_COLOR)
    assert image is not None
    for row in range(12):
        assert tuple(image[(row + 1) * 64 - 3, -3]) == (0, 0, 180)
    assert launcher.sha256(review) != before_sha
    asset = result["assets"][review.name]
    assert asset == {"bytes": review.stat().st_size, "sha256": launcher.sha256(review)}
    assert result["diagnostic_only_watermark"]["text"] == launcher.WATERMARK_TEXT
    assert result["diagnostic_only_watermark"]["rows_watermarked"] == 12
    assert result["consumption_authorized"] is False
    assert result["v4_held_capacity_orchestration"][
        "held_reservation_replaces_extra_v3_postpublish_reserve"
    ] is True
    assert result["v4_held_capacity_orchestration"][
        "v3_workspace_reserve_argument_bytes"
    ] == 0
    assert result["v4_held_capacity_orchestration"][
        "workspace_free_required_for_output_bytes"
    ] == 64 * 1024 * 1024
    assert result["v4_held_capacity_orchestration"][
        "final_persistent_tree_bytes"
    ] == actual_bytes
    on_disk = json.loads((staging / "RESULT.json").read_text(encoding="utf-8"))
    assert on_disk == result


def test_result_with_temporal_rows_is_rejected_before_watermark(tmp_path: Path) -> None:
    staging, _ = make_staging(tmp_path)
    result_path = staging / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["temporal_windows"] = [{"frames": [1, 2]}]
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(launcher.CleanV4LaunchHold, match="Temporal|temporal"):
        launcher.watermark_and_annotate_result(
            staging,
            frames=list(range(12)),
            capacity_evidence={},
            workspace_free_before_bytes=64 * 1024 * 1024,
        )


@pytest.mark.parametrize("existing", ["staging", "output", "internal"])
def test_existing_staging_output_or_internal_publish_holds(
    tmp_path: Path, existing: str
) -> None:
    parent = tmp_path / "run"
    parent.mkdir()
    staging = parent / ".producer.v4-staging"
    output = parent / "producer"
    target = {
        "staging": staging,
        "output": output,
        "internal": parent / f".{staging.name}.publishing",
    }[existing]
    target.mkdir()
    with pytest.raises(launcher.CleanV4LaunchHold, match="fresh"):
        launcher.validate_fresh_paths(staging, output)


def test_fresh_staging_is_atomically_published_without_copy(tmp_path: Path) -> None:
    parent = tmp_path / "run"
    parent.mkdir()
    staging = parent / ".producer.v4-staging"
    output = parent / "producer"
    launcher.validate_fresh_paths(staging, output)
    staging.mkdir()
    sentinel = staging / "sentinel"
    sentinel.write_bytes(b"owned")
    inode = sentinel.stat().st_ino
    launcher.atomic_publish_staging(staging, output)
    assert not staging.exists()
    assert (output / "sentinel").stat().st_ino == inode
    with pytest.raises(launcher.CleanV4LaunchHold, match="fresh final output"):
        other = parent / ".other-staging"
        other.mkdir()
        launcher.atomic_publish_staging(other, output)
