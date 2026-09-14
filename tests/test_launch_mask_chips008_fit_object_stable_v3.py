from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import launch_mask_chips008_fit_object_stable_v3 as launcher


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_tree_requires_exact_count_and_content(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = {
        "pin": {
            "path": "artifact.json",
            "bytes": artifact.stat().st_size,
            "sha256": digest(artifact),
        }
    }
    rows = launcher.validate_reference_tree(tmp_path, manifest, expected_count=1)
    assert rows == [manifest["pin"]]
    artifact.write_text("drift\n", encoding="utf-8")
    with pytest.raises(launcher.LaunchHold, match="pinned reference drift"):
        launcher.validate_reference_tree(tmp_path, manifest, expected_count=1)


def test_reference_tree_rejects_path_escape(tmp_path: Path) -> None:
    manifest = {
        "pin": {
            "path": "../escape",
            "bytes": 0,
            "sha256": "0" * 64,
        }
    }
    with pytest.raises(launcher.LaunchHold, match="unsafe pinned path"):
        launcher.validate_reference_tree(tmp_path, manifest, expected_count=1)


def test_capacity_authority_binds_live_creator_and_reservation(
    tmp_path: Path,
) -> None:
    project = tmp_path
    control = project / "tasks/control"
    control.mkdir(parents=True)
    reserve = control / ".mask-post-hawor-128m-v1.reserve"
    reserve.write_bytes(b"x" * 4096)
    metadata = reserve.stat()
    authority_path = control / ".mask-post-hawor-128m-v1.LIVE_AUTHORITY.json"
    authority = {
        "schema_version": "project-capacity-live-authority-v1",
        "status": "PASS_CAPACITY_HELD",
        "authorizes_execution": True,
        "project_root": str(project),
        "authority_path": str(authority_path.relative_to(project)),
        "reservation_path": str(reserve.relative_to(project)),
        "requested_bytes": 4096,
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
    environment = {
        "PROJECT_CAPACITY_LIVE_AUTHORITY": str(authority_path),
        "PROJECT_CAPACITY_RESERVATION": str(reserve),
        "PROJECT_CAPACITY_RESERVED_BYTES": "4096",
    }
    observed = launcher.validate_capacity_authority(
        project, environment, expected_bytes=4096
    )
    assert observed["creator_pid"] == os.getpid()
    authority["creator_process_start_ticks"] += 1
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(launcher.LaunchHold, match="no longer live"):
        launcher.validate_capacity_authority(
            project, environment, expected_bytes=4096
        )


def test_capacity_authority_rejects_wrapper_environment_drift(tmp_path: Path) -> None:
    with pytest.raises(launcher.LaunchHold, match="environment mismatch"):
        launcher.validate_capacity_authority(tmp_path, {}, expected_bytes=4096)


def test_fixed_runner_argv_has_no_configurable_session_or_output() -> None:
    argv = launcher.fixed_runner_argv("python-fixed")
    assert argv[0] == "python-fixed"
    assert argv[1] == str(launcher.RUNNER)
    assert argv[argv.index("--evaluation-plan") + 1] == str(launcher.PLAN)
    assert argv[argv.index("--output-root") + 1] == str(launcher.OUTPUT)
    assert argv[argv.index("--lease-holder") + 1] == launcher.LEASE_HOLDER
    assert "get_potato_chips_0901_001" not in " ".join(argv)


def test_lease_payload_is_gpu0_bounded_and_initially_empty() -> None:
    value = launcher._lease_payload([])
    assert value["status"] == "ACQUIRED"
    assert value["holder"] == launcher.LEASE_HOLDER
    assert value["scope"]["gpu_indices"] == [0]
    assert value["scope"]["max_wall_seconds"] == launcher.MAX_WALL_SECONDS
    assert value["pre_start_gpu_compute_processes"] == 0
