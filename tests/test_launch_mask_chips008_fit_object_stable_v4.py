from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import launch_mask_chips008_fit_object_stable_v3 as base
from tools import launch_mask_chips008_fit_object_stable_v4 as launcher


def test_successor_configuration_is_fixed_and_base_is_pinned() -> None:
    launcher.configure_fixed_successor()
    assert base.sha256(launcher.BASE_LAUNCHER) == launcher.BASE_LAUNCHER_SHA256
    assert base.FINAL_MANIFEST == launcher.FINAL_MANIFEST
    assert base.FINAL_MANIFEST_SHA256 == launcher.FINAL_MANIFEST_SHA256
    assert base.EXPECTED_REFERENCE_COUNT == 38
    argv = base.fixed_runner_argv("python-fixed")
    assert argv[argv.index("--output-root") + 1] == str(launcher.OUTPUT)
    assert argv[argv.index("--lease-holder") + 1] == launcher.LEASE_HOLDER


def test_v2_capacity_authority_is_independent_and_live(tmp_path: Path) -> None:
    project = tmp_path
    control = project / "tasks/control"
    control.mkdir(parents=True)
    reserve = control / ".mask-chips008-retry-128m-v2.reserve"
    reserve.write_bytes(b"x" * 4096)
    metadata = reserve.stat()
    authority_path = control / ".mask-chips008-retry-128m-v2.LIVE_AUTHORITY.json"
    authority = {
        "schema_version": "project-capacity-live-authority-v1",
        "status": "PASS_CAPACITY_HELD",
        "authorizes_execution": True,
        "project_root": str(project),
        "authority_path": str(authority_path.relative_to(project)),
        "reservation_path": str(reserve.relative_to(project)),
        "requested_bytes": 4096,
        "creator_pid": os.getpid(),
        "creator_process_start_ticks": base.process_start_ticks(os.getpid()),
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
    observed = launcher.validate_capacity_authority_v2(
        project, environment, expected_bytes=4096
    )
    assert observed["creator_pid"] == os.getpid()


def test_v2_capacity_rejects_v1_environment(tmp_path: Path) -> None:
    environment = {
        "PROJECT_CAPACITY_LIVE_AUTHORITY": str(
            tmp_path / "tasks/control/.mask-post-hawor-128m-v1.LIVE_AUTHORITY.json"
        ),
        "PROJECT_CAPACITY_RESERVATION": str(
            tmp_path / "tasks/control/.mask-post-hawor-128m-v1.reserve"
        ),
        "PROJECT_CAPACITY_RESERVED_BYTES": "4096",
    }
    with pytest.raises(base.LaunchHold, match="environment mismatch"):
        launcher.validate_capacity_authority_v2(
            tmp_path, environment, expected_bytes=4096
        )
