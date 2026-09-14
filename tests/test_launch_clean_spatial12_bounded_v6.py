from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import launch_clean_spatial12_bounded_v6 as launcher


FRAMES = [0, 79, 160, 239, 320, 399, 480, 559, 640, 719, 760, 798]


def make_candidate(
    root: Path,
    *,
    temporal: bool = False,
    status: str = "PASS_CANARY_ONLY",
) -> Path:
    staging = root / ".producer.v6-owned-held-capacity-staging"
    frame_root = staging / "frames"
    frame_root.mkdir(parents=True)
    for frame in FRAMES:
        (frame_root / f"{frame:05d}.png").write_bytes(b"frame")
    (staging / "RESULT.json").write_text(
        json.dumps(
            {
                "status": status,
                "frames": FRAMES,
                "temporal_windows": [{"start": 0}] if temporal else [],
            }
        ),
        encoding="utf-8",
    )
    return staging


def test_missing_live_capacity_authority_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(launcher.CleanV6LaunchHold, match="environment mismatch"):
        launcher.require_capacity_authority(tmp_path, {})


def test_output_over_64_mib_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = make_candidate(tmp_path)
    monkeypatch.setattr(
        launcher.v4.base,
        "tree_bytes",
        lambda _root: launcher.PERSISTENT_BYTES_MAX + 1,
    )
    with pytest.raises(launcher.CleanV6LaunchHold, match="exceeds"):
        launcher.validate_owned_candidate(staging, frames=FRAMES)
    assert not (tmp_path / "producer").exists()


def test_temporal_candidate_fails_closed(tmp_path: Path) -> None:
    staging = make_candidate(tmp_path, temporal=True)
    with pytest.raises(launcher.CleanV6LaunchHold, match="temporal"):
        launcher.validate_owned_candidate(staging, frames=FRAMES)


def test_publish_enospc_is_fail_atomic_and_preserves_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = make_candidate(tmp_path)
    output = tmp_path / "producer"

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError(errno.ENOSPC, "synthetic full filesystem")

    monkeypatch.setattr(launcher.os, "replace", fail_replace)
    with pytest.raises(OSError) as caught:
        launcher.publish_owned_staging(staging, output)
    assert caught.value.errno == errno.ENOSPC
    assert staging.is_dir()
    assert not output.exists()


def test_copy_enospc_never_creates_final_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_candidate(tmp_path / "ram")
    publishing = tmp_path / "run" / ".owned"
    publishing.parent.mkdir()
    output = publishing.parent / "producer"

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "synthetic full filesystem")

    monkeypatch.setattr(launcher.shutil, "copytree", fail_copy)
    with pytest.raises(OSError) as caught:
        launcher.copy_to_owned_publishing(source, publishing)
    assert caught.value.errno == errno.ENOSPC
    assert source.is_dir()
    assert not output.exists()


def test_df_zero_with_valid_live_authority_reaches_producer_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    ram_root = tmp_path / "ram"
    ram_root.mkdir()
    publishing = run_root / ".owned"
    output = run_root / "producer"
    called = {"producer": 0, "capacity": 0}
    valid_capacity = {
        "profile": "clean-capacity-v1",
        "requested_bytes": launcher.CAPACITY_BYTES,
        "reservation": {
            "logical_bytes": launcher.CAPACITY_BYTES,
            "allocated_bytes_min": launcher.CAPACITY_BYTES,
        },
    }

    monkeypatch.setattr(launcher, "PROJECT", tmp_path)
    monkeypatch.setattr(launcher, "OWNED_PUBLISHING_RELATIVE", Path("run/.owned"))
    monkeypatch.setattr(launcher, "FINAL_OUTPUT_RELATIVE", Path("run/producer"))
    monkeypatch.setattr(launcher, "RAM_ROOT", ram_root)
    monkeypatch.setattr(
        launcher,
        "validate_static",
        lambda _project: (
            FRAMES,
            {
                "launcher": {"path": "tools/clean_python.sh"},
                "direct_ram_producer": {"path": "tools/producer.py"},
            },
        ),
    )
    monkeypatch.setattr(
        launcher.v5,
        "validate_current_runtime_origins",
        lambda _project, _environment: {"status": "PASS"},
    )

    def capacity(_project: Path, _environment: dict[str, str]) -> dict[str, object]:
        called["capacity"] += 1
        return valid_capacity

    monkeypatch.setattr(launcher, "require_capacity_authority", capacity)
    monkeypatch.setattr(
        launcher,
        "validate_donor_runtime_before",
        lambda _project: {"status": "PASS_BEFORE"},
    )
    monkeypatch.setattr(
        launcher,
        "validate_donor_runtime_after",
        lambda _project, _before: {"status": "PASS_AFTER"},
    )
    def producer_mock(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        called["producer"] += 1
        ram_output = Path(argv[argv.index("--output") + 1])
        candidate = make_candidate(ram_output.parent)
        candidate.rename(ram_output)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", producer_mock)
    monkeypatch.setattr(launcher, "validate_owned_candidate", lambda *_a, **_k: 100)
    monkeypatch.setattr(
        launcher.v4,
        "watermark_and_annotate_result",
        lambda *_a, **_k: (
            {"status": "PASS_SPATIAL12_DIAGNOSTIC_ONLY"},
            100,
        ),
    )
    monkeypatch.setattr(
        launcher.v5.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1, used=1, free=0),
    )

    assert launcher.run_fixed({}) == 0
    assert called == {"producer": 1, "capacity": 4}
    assert output.is_dir()
    assert not publishing.exists()


def test_fixed_child_argv_is_local_ram_backed_and_spatial_only() -> None:
    ram_output = Path("/dev/shm/synthetic-clean-v6/producer")
    argv = launcher.fixed_producer_argv(ram_output=ram_output)
    assert argv[0] == str(launcher.PROJECT / launcher.v5.LOCAL_LAUNCHER_RELATIVE)
    assert argv[1] == str(launcher.PROJECT / launcher.PRODUCER_RELATIVE)
    assert argv[argv.index("--output") + 1] == str(ram_output)
    assert str(launcher.v4.BASE_RUNNER) not in argv
    assert "temporal" not in " ".join(argv).lower()


def test_real_static_pins_include_local_environment_and_exact_donor() -> None:
    frames, evidence = launcher.validate_static()
    assert frames == FRAMES
    assert evidence["status"] == "PASS_PROJECT_LOCAL_CLEAN_ENVIRONMENT_AUTHORITY"
    donor = evidence["v3_donor_authority"]
    assert donor["sha256"] == launcher.DONOR_AUTHORITY_SHA256
    assert donor["status"] == "PASS_COMPLETE_STABLE_FULL_CRC"
    assert donor["zip_content_read_by_v6"] is False
    assert evidence["direct_ram_producer"]["sha256"] == launcher.PRODUCER_SHA256


def test_launcher_source_has_no_disk_usage_or_delete_path() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "disk_usage(" not in source
    assert ".unlink(" not in source
    assert "os.remove(" not in source


def test_cpu_only_contract_records_external_gpu_without_authorizing_it() -> None:
    evidence = launcher.cpu_only_gpu_coexistence_evidence()
    assert evidence["external_gpu_present"] is True
    assert evidence["external_compute_processes"] == [
        {
            "pid": 1057261,
            "memory_mib_approx": 14550,
            "ownership": "EXTERNAL_TO_CLEAN_V6",
        }
    ]
    assert evidence["does_not_authorize_or_use_gpu"] is True
    assert evidence["cuda_visible_devices"] == ""
    assert evidence["gpu_calls"] == 0


def test_pass_canary_is_normalized_and_cpu_contract_written_before_gate(
    tmp_path: Path,
) -> None:
    staging = make_candidate(tmp_path)
    status = launcher.normalize_and_annotate_producer_result(
        staging,
        donor_runtime_before={"status": "PASS_BEFORE"},
        donor_runtime_after={"status": "PASS_AFTER"},
    )
    result = json.loads((staging / "RESULT.json").read_text(encoding="utf-8"))
    assert status == "PASS_SPATIAL12_DIAGNOSTIC_ONLY"
    assert result["status"] == "PASS_SPATIAL12_DIAGNOSTIC_ONLY"
    assert result["consumption_authorized"] is False
    evidence = result["v6_cpu_only_gpu_coexistence"]
    assert evidence["external_gpu_present"] is True
    assert evidence["does_not_authorize_or_use_gpu"] is True
    assert evidence["cuda_visible_devices"] == ""
    assert evidence["gpu_calls"] == 0
    assert result["v6_donor_runtime_authority"]["before_producer"] == {
        "status": "PASS_BEFORE"
    }


def test_donor_archive_before_after_and_critical_sha_contract(tmp_path: Path) -> None:
    archive = tmp_path / "base.zip"
    archive.write_bytes(b"archive-not-opened")
    baseline = tmp_path / launcher.v4.BASELINE_STEREO_RELATIVE
    camera = tmp_path / launcher.v4.CAMERA_PARAMS_RELATIVE
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"staged-baseline")
    camera.write_bytes(b"staged-camera")
    authority_path = tmp_path / launcher.DONOR_AUTHORITY_RELATIVE
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    archive_stat = archive.stat()
    authority_path.write_text(
        json.dumps(
            {
                "schema_version": "clean-plate-archive-authority-v1",
                "status": "PASS_COMPLETE_STABLE_FULL_CRC",
                "donor_archive_identity_verified": True,
                "donor_archive_crc_verified": True,
                "archive": {
                    "path": str(archive),
                    "bytes": archive_stat.st_size,
                    "mtime_ns": archive_stat.st_mtime_ns,
                },
                "critical_members": {
                    "base_001/source_stereo/CameraRecord_base_001_stereo.mp4": {
                        "sha256": launcher.sha256(baseline)
                    },
                    "base_001/camera_params.json": {
                        "sha256": launcher.sha256(camera)
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    authority_sha = launcher.sha256(authority_path)
    before = launcher.validate_donor_runtime_before(
        tmp_path, expected_authority_sha=authority_sha
    )
    after = launcher.validate_donor_runtime_after(
        tmp_path, before, expected_authority_sha=authority_sha
    )
    assert before["archive"]["zip_content_read_by_v6"] is False
    assert after["archive"]["unchanged_since_before"] is True
    assert all(
        row["matches_authority"] is True
        for row in before["critical_staged_members"].values()
    )

    archive.write_bytes(b"archive-changed")
    with pytest.raises(launcher.CleanV6LaunchHold, match="archive changed"):
        launcher.validate_donor_runtime_after(
            tmp_path, before, expected_authority_sha=authority_sha
        )
