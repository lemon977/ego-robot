import json
import os
from pathlib import Path

import pytest

from tools import no_clobber_hardlink_snapshot as snapshot


def _source_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    package = source / "lib/python3.10/site-packages/demo"
    binary = source / "bin"
    package.mkdir(parents=True)
    binary.mkdir()
    (source / "plain.txt").write_bytes(b"snapshot-payload\n")
    os.link(source / "plain.txt", source / "plain-alias.txt")
    (package / "module.py").write_text("VALUE = 7\n", encoding="utf-8")
    (binary / "python3.10").write_bytes(b"fixture-python")
    os.chmod(binary / "python3.10", 0o755)
    os.symlink("python3.10", binary / "python")
    os.symlink("../../../../plain.txt", package / "payload-link")
    return source


def test_create_verify_and_explicit_rollback(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    target = tmp_path / "snapshot"
    baseline_nlink = os.stat(source / "plain.txt").st_nlink

    preflight, scan = snapshot.build_preflight(source, target)
    assert preflight["status"] == (
        "BLOCKED" if preflight["risk_blockers"] else "PASS"
    )
    assert preflight["same_device"] is True
    assert preflight["fatal_blockers"] == []
    assert scan.counts == {
        "directories": 6,
        "regular_files": 4,
        "symlinks": 2,
        "special": 0,
    }
    assert scan.logical_bytes > scan.unique_regular_bytes

    # This is a tiny disposable fixture.  On CPFS, statvfs reports zero data
    # blocks even though its separate metadata pool accepts these few entries.
    created = snapshot.create_snapshot(
        source,
        target,
        acknowledge_metadata_risk=bool(preflight["risk_blockers"]),
    )
    assert created["status"] == "PASS"
    assert target.is_dir()
    assert os.stat(target / "plain.txt").st_ino == os.stat(source / "plain.txt").st_ino
    assert os.stat(target / "plain-alias.txt").st_ino == os.stat(source / "plain.txt").st_ino
    assert os.stat(source / "plain.txt").st_nlink == baseline_nlink + 2
    assert os.readlink(target / "bin/python") == "python3.10"
    assert os.readlink(target / "lib/python3.10/site-packages/demo/payload-link") == "../../../../plain.txt"
    assert (target / snapshot.CONTROL_DIR / snapshot.MANIFEST_NAME).is_file()

    metadata = snapshot.verify_snapshot(target, hash_content=False)
    content = snapshot.verify_snapshot(target, hash_content=True)
    assert metadata["status"] == "PASS"
    assert metadata["hashed_unique_regular_inodes"] == 0
    assert content["status"] == "PASS"
    assert content["tree_content_sha256"] == created["tree_content_sha256"]
    assert content["hashed_unique_regular_inodes"] == 3

    unexpected_control = target / snapshot.CONTROL_DIR / "unexpected"
    unexpected_control.write_bytes(b"do-not-delete-silently")
    with pytest.raises(snapshot.SnapshotError, match="unexpected entries"):
        snapshot.verify_snapshot(target, hash_content=True)
    unexpected_control.unlink()

    dry_run = snapshot.rollback_snapshot(
        target, created["tree_content_sha256"], execute=False
    )
    assert dry_run["status"] == "DRY_RUN_PASS"
    assert target.exists()
    rolled_back = snapshot.rollback_snapshot(
        target, created["tree_content_sha256"], execute=True
    )
    assert rolled_back["status"] == "PASS"
    assert not target.exists()
    assert source.is_dir()
    assert os.stat(source / "plain.txt").st_nlink == baseline_nlink


def test_target_conflict_is_no_clobber(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    target = tmp_path / "snapshot"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_bytes(b"do-not-touch")

    preflight, _ = snapshot.build_preflight(source, target)
    assert "TARGET_ALREADY_EXISTS_NO_CLOBBER" in preflight["fatal_blockers"]
    with pytest.raises(snapshot.SnapshotError, match="TARGET_ALREADY_EXISTS_NO_CLOBBER"):
        snapshot.create_snapshot(source, target, acknowledge_metadata_risk=False)
    assert sentinel.read_bytes() == b"do-not-touch"
    assert not list(tmp_path.glob(".snapshot.hardlink-staging.*"))


def test_symlink_and_special_file_hazards_fail_preflight(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "ok").write_bytes(b"ok")
    os.symlink("missing", source / "broken")
    os.symlink("../outside", source / "escape")
    os.symlink("/absolute", source / "absolute")
    os.mkfifo(source / "fifo")

    preflight, _ = snapshot.build_preflight(source, tmp_path / "snapshot")
    assert preflight["status"] == "BLOCKED"
    assert len(preflight["hazards"]["broken_symlinks"]) == 1
    assert len(preflight["hazards"]["external_symlinks"]) == 1
    assert len(preflight["hazards"]["absolute_symlinks"]) == 1
    assert len(preflight["hazards"]["special_files"]) == 1


def test_source_mutation_retains_failure_evidence_and_can_be_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path)
    target = tmp_path / "snapshot"
    real_link = snapshot.os.link
    mutated = False

    def link_then_mutate(src, dst, *args, **kwargs):
        nonlocal mutated
        result = real_link(src, dst, *args, **kwargs)
        if not mutated:
            mutated = True
            victim = source / "plain.txt"
            current = os.stat(victim).st_mtime_ns
            os.utime(victim, ns=(current + 1_000_000, current + 1_000_000))
        return result

    monkeypatch.setattr(snapshot.os, "link", link_then_mutate)
    with pytest.raises(snapshot.SnapshotError, match="staging retained"):
        snapshot.create_snapshot(source, target, acknowledge_metadata_risk=True)
    assert not target.exists()
    stages = list(tmp_path.glob(".snapshot.hardlink-staging.*"))
    assert len(stages) == 1
    failure_path = stages[0] / snapshot.CONTROL_DIR / snapshot.FAILURE_NAME
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED_STAGING_RETAINED"
    assert failure["created"]["regular_files"] >= 1

    dry_run = snapshot.rollback_failed(
        stages[0], failure["snapshot_id"], execute=False
    )
    assert dry_run["status"] == "DRY_RUN_PASS"
    removed = snapshot.rollback_failed(stages[0], failure["snapshot_id"], execute=True)
    assert removed["status"] == "PASS"
    assert not stages[0].exists()
    assert source.exists()


def test_metadata_capacity_risk_requires_explicit_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_fixture(tmp_path)
    real = os.statvfs(tmp_path)
    zero_capacity = os.statvfs_result(
        (
            real.f_bsize,
            real.f_frsize,
            real.f_blocks,
            real.f_bfree,
            0,
            real.f_files,
            real.f_ffree,
            real.f_favail,
            real.f_flag,
            real.f_namemax,
        )
    )
    monkeypatch.setattr(snapshot.os, "statvfs", lambda _path: zero_capacity)

    preflight, _ = snapshot.build_preflight(source, tmp_path / "snapshot")
    assert "FREE_BLOCKS_BELOW_CONSERVATIVE_METADATA_ESTIMATE" in preflight["risk_blockers"]
    with pytest.raises(snapshot.SnapshotError, match="acknowledge-metadata-risk"):
        snapshot.create_snapshot(source, tmp_path / "snapshot", acknowledge_metadata_risk=False)


def test_atomic_publish_primitive_never_replaces_existing_target(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    target = tmp_path / "target"
    staged.mkdir()
    target.mkdir()
    (target / "sentinel").write_bytes(b"preserved")
    with pytest.raises(snapshot.SnapshotError, match="no-clobber preserved"):
        snapshot._rename_noreplace(staged, target)
    assert staged.is_dir()
    assert (target / "sentinel").read_bytes() == b"preserved"
