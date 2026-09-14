import ast
import errno
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import project_capacity_reservation as capacity


def _paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "reserve.bin", root / "LIVE_AUTHORITY.json", root / "RECEIPT.json"


def test_acquire_holds_real_blocks_and_same_process_release_revokes_pass(
    tmp_path: Path,
) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    lease = capacity.acquire_capacity(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        requested_bytes=96 * 1024,
        chunk_bytes=16 * 1024,
        nonce=b"fixed-test-nonce" * 2,
    )

    assert reserve.is_file()
    assert authority.is_file()
    assert reserve.stat().st_size == 96 * 1024
    assert reserve.stat().st_blocks * 512 >= reserve.stat().st_size
    live = json.loads(authority.read_text(encoding="utf-8"))
    assert live["status"] == capacity.LIVE_STATUS
    assert live["authorizes_execution"] is True
    assert live["write"]["sequential_from_offset_zero"] is True
    assert live["write"]["truncate_used"] is False
    assert live["write"]["fallocate_used"] is False
    assert live["verification"]["full_pattern_match"] is True
    assert lease.check_held(verify_content=True)["sha256"] == live["verification"][
        "sha256"
    ]

    released = lease.release(reason="TEST_COMPLETE", receipt_path=receipt)
    assert released["status"] == capacity.RELEASED_STATUS
    assert released["authorizes_execution"] is False
    assert not reserve.exists()
    assert not authority.exists()
    persisted = json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted["live_authority_withdrawn_before_reservation_unlink"] is True
    assert persisted["reservation_path_absent"] is True
    assert persisted["live_authority_path_absent"] is True


def test_probe_is_bounded_released_and_never_an_execution_authority(
    tmp_path: Path,
) -> None:
    reserve, authority, result = _paths(tmp_path)
    code, record = capacity.run_probe(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        result_path=result,
        requested_bytes=128 * 1024,
        chunk_bytes=32 * 1024,
    )

    assert code == 0
    assert record["status"] == "PASS_PROBE_RELEASED_NOT_AUTHORITY"
    assert record["authorizes_execution"] is False
    assert record["capacity_was_proven"] is True
    assert not reserve.exists()
    assert not authority.exists()
    persisted = json.loads(result.read_text(encoding="utf-8"))
    assert persisted["status"] == "PASS_PROBE_RELEASED_NOT_AUTHORITY"
    assert persisted["authorizes_execution"] is False
    assert persisted["statvfs_before"]["used_as_capacity_gate"] is False


def test_held_command_sees_reservation_and_live_authority_until_exit(
    tmp_path: Path,
) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    marker = tmp_path / "child-observed.json"
    child = (
        "import json, os, pathlib; "
        "r=pathlib.Path(os.environ['PROJECT_CAPACITY_RESERVATION']); "
        "a=pathlib.Path(os.environ['PROJECT_CAPACITY_LIVE_AUTHORITY']); "
        f"m=pathlib.Path({str(marker)!r}); "
        "assert r.is_file() and a.is_file(); "
        "d=json.loads(a.read_text()); "
        "assert d['status']=='PASS_CAPACITY_HELD'; "
        "m.write_text(json.dumps({'blocks':r.stat().st_blocks,'bytes':r.stat().st_size}))"
    )
    code, returned = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=128 * 1024,
        confirm_bytes=128 * 1024,
        chunk_bytes=32 * 1024,
        command=[sys.executable, "-c", child],
    )

    assert code == 0
    observed = json.loads(marker.read_text(encoding="utf-8"))
    assert observed["bytes"] == 128 * 1024
    assert observed["blocks"] * 512 >= observed["bytes"]
    assert returned["context"]["reservation_held_for_entire_downstream_lifetime"]
    assert not reserve.exists()
    assert not authority.exists()
    assert json.loads(receipt.read_text(encoding="utf-8"))["authorizes_execution"] is False


def test_reservation_conflict_is_no_clobber_and_child_never_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    reserve.write_bytes(b"foreign-sentinel")
    spawned = False

    def forbidden_popen(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("child must not start")

    monkeypatch.setattr(capacity.subprocess, "Popen", forbidden_popen)
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=64 * 1024,
        confirm_bytes=64 * 1024,
        chunk_bytes=16 * 1024,
        command=[sys.executable, "-c", "raise SystemExit(0)"],
    )

    assert code == 2
    assert record["status"] == "FAIL_CLOSED_CAPACITY_NOT_PROVEN"
    assert record["downstream_started"] is False
    assert spawned is False
    assert reserve.read_bytes() == b"foreign-sentinel"
    assert not authority.exists()


@pytest.mark.parametrize("as_symlink", [False, True])
def test_live_authority_conflict_is_preserved_and_payload_is_cleaned(
    tmp_path: Path, as_symlink: bool
) -> None:
    reserve, authority, _ = _paths(tmp_path)
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"preserve-me")
    if as_symlink:
        authority.symlink_to(sentinel.name)
    else:
        authority.write_bytes(b"foreign-authority")

    with pytest.raises(capacity.CapacityReservationError, match="no-clobber"):
        capacity.acquire_capacity(
            project_root=tmp_path,
            reservation_path=reserve,
            authority_path=authority,
            requested_bytes=64 * 1024,
            chunk_bytes=16 * 1024,
            nonce=b"authority-conflict-nonce-32!!",
        )
    assert not reserve.exists()
    if as_symlink:
        assert authority.is_symlink()
        assert os.readlink(authority) == sentinel.name
    else:
        assert authority.read_bytes() == b"foreign-authority"
    assert sentinel.read_bytes() == b"preserve-me"


def test_intermediate_symlink_and_outside_project_are_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(capacity.CapacityReservationError, match="no-follow"):
        capacity.acquire_capacity(
            project_root=project,
            reservation_path=project / "escape/reserve.bin",
            authority_path=project / "escape/LIVE.json",
            requested_bytes=64 * 1024,
        )
    with pytest.raises(capacity.CapacityReservationError, match="outside project"):
        capacity.acquire_capacity(
            project_root=project,
            reservation_path=outside / "reserve.bin",
            authority_path=outside / "LIVE.json",
            requested_bytes=64 * 1024,
        )
    assert list(outside.iterdir()) == []


def test_short_writes_are_completed_and_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve, authority, _ = _paths(tmp_path)
    real_write = capacity.os.write

    def short_write(fd: int, payload: bytes) -> int:
        return real_write(fd, payload[: max(1, len(payload) // 3)])

    monkeypatch.setattr(capacity.os, "write", short_write)
    lease = capacity.acquire_capacity(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        requested_bytes=64 * 1024,
        chunk_bytes=16 * 1024,
        nonce=b"short-write-test-nonce-value!!",
    )
    assert lease.authority["write"]["write_syscall_count"] > 4
    assert lease.check_held(verify_content=True)["logical_bytes"] == 64 * 1024
    lease.release(reason="TEST_COMPLETE")


def test_readback_corruption_fails_closed_without_live_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve, authority, _ = _paths(tmp_path)
    real_pread = capacity._pread_exact
    corrupted = False

    def corrupt_once(fd: int, length: int, offset: int) -> bytes:
        nonlocal corrupted
        value = real_pread(fd, length, offset)
        if not corrupted:
            corrupted = True
            value = bytes([value[0] ^ 1]) + value[1:]
        return value

    monkeypatch.setattr(capacity, "_pread_exact", corrupt_once)
    with pytest.raises(capacity.CapacityReservationError, match="pattern mismatch"):
        capacity.acquire_capacity(
            project_root=tmp_path,
            reservation_path=reserve,
            authority_path=authority,
            requested_bytes=64 * 1024,
            chunk_bytes=16 * 1024,
            nonce=b"readback-corruption-nonce-value",
        )
    assert not reserve.exists()
    assert not authority.exists()


def test_midstream_enospc_fails_closed_and_cleans_partial_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve, authority, _ = _paths(tmp_path)
    real_write = capacity.os.write
    calls = 0

    def fail_second_write(fd: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.ENOSPC, "injected no space")
        return real_write(fd, payload)

    monkeypatch.setattr(capacity.os, "write", fail_second_write)
    with pytest.raises(capacity.CapacityReservationError, match="ENOSPC"):
        capacity.acquire_capacity(
            project_root=tmp_path,
            reservation_path=reserve,
            authority_path=authority,
            requested_bytes=64 * 1024,
            chunk_bytes=16 * 1024,
            nonce=b"midstream-enospc-test-nonce!!!!",
        )
    assert not reserve.exists()
    assert not authority.exists()


def test_payload_fsync_error_fails_closed_and_cleans_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve, authority, _ = _paths(tmp_path)
    real_fsync = capacity.os.fsync
    calls = 0

    def fail_first_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EIO, "injected fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(capacity.os, "fsync", fail_first_fsync)
    with pytest.raises(capacity.CapacityReservationError, match="EIO"):
        capacity.acquire_capacity(
            project_root=tmp_path,
            reservation_path=reserve,
            authority_path=authority,
            requested_bytes=64 * 1024,
            chunk_bytes=16 * 1024,
            nonce=b"payload-fsync-test-nonce-value",
        )
    assert not reserve.exists()
    assert not authority.exists()


def test_underallocation_gate_failure_cleans_payload_and_publishes_no_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve, authority, _ = _paths(tmp_path)

    def reject_blocks(value: os.stat_result, requested_bytes: int) -> None:
        raise capacity.CapacityReservationError(
            f"reservation is underallocated/sparse: allocated=0, requested={requested_bytes}"
        )

    monkeypatch.setattr(capacity, "_require_real_allocation", reject_blocks)
    with pytest.raises(capacity.CapacityReservationError, match="underallocated/sparse"):
        capacity.acquire_capacity(
            project_root=tmp_path,
            reservation_path=reserve,
            authority_path=authority,
            requested_bytes=64 * 1024,
            chunk_bytes=16 * 1024,
            nonce=b"underallocation-test-nonce-value",
        )
    assert not reserve.exists()
    assert not authority.exists()


def test_probe_hard_cap_refuses_large_write_without_allocating(tmp_path: Path) -> None:
    reserve, authority, result = _paths(tmp_path)
    code, record = capacity.run_probe(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        result_path=result,
        requested_bytes=capacity.MAX_PROBE_BYTES + 1,
    )
    assert code == 2
    assert record["status"] == "FAIL_CLOSED_CAPACITY_NOT_PROVEN"
    assert not reserve.exists()
    assert not authority.exists()
    assert json.loads(result.read_text(encoding="utf-8"))["authorizes_execution"] is False


def test_existing_result_is_preserved_without_running_probe(tmp_path: Path) -> None:
    reserve, authority, result = _paths(tmp_path)
    result.write_bytes(b"foreign-result")
    code, record = capacity.run_probe(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        result_path=result,
        requested_bytes=64 * 1024,
    )
    assert code == 2
    assert record["status"] == "FAIL_CLOSED_CAPACITY_NOT_PROVEN"
    assert result.read_bytes() == b"foreign-result"
    assert not reserve.exists() and not authority.exists()


def test_nonzero_child_still_revokes_authority_and_releases_reserve(
    tmp_path: Path,
) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=64 * 1024,
        confirm_bytes=64 * 1024,
        chunk_bytes=16 * 1024,
        command=[sys.executable, "-c", "raise SystemExit(7)"],
    )
    assert code == 7
    assert record["context"]["downstream_returncode"] == 7
    assert not reserve.exists() and not authority.exists()
    assert receipt.is_file()


def test_only_creator_pid_can_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reserve, authority, _ = _paths(tmp_path)
    lease = capacity.acquire_capacity(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        requested_bytes=64 * 1024,
        chunk_bytes=16 * 1024,
        nonce=b"creator-pid-test-nonce-value!!!",
    )
    creator = lease.creator_pid
    monkeypatch.setattr(capacity.os, "getpid", lambda: creator + 1)
    with pytest.raises(capacity.CapacityReservationError, match="creating process"):
        lease.release(reason="FORBIDDEN")
    assert reserve.exists() and authority.exists()
    monkeypatch.setattr(capacity.os, "getpid", lambda: creator)
    lease.release(reason="TEST_COMPLETE")


def test_live_authority_same_inode_content_mutation_is_detected(
    tmp_path: Path,
) -> None:
    reserve, authority, _ = _paths(tmp_path)
    lease = capacity.acquire_capacity(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        requested_bytes=64 * 1024,
        chunk_bytes=16 * 1024,
        nonce=b"authority-content-mutation-nonce",
    )
    original = authority.read_bytes()
    mutated = bytearray(original)
    mutated[0] ^= 1
    os.chmod(authority, 0o600)
    with authority.open("r+b", buffering=0) as handle:
        handle.write(mutated)
        handle.flush()
        os.fsync(handle.fileno())
    with pytest.raises(capacity.CapacityReservationError, match="content SHA-256"):
        lease.check_held()

    with authority.open("r+b", buffering=0) as handle:
        handle.write(original)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(authority, 0o440)
    lease.release(reason="TEST_RESTORED_AFTER_MUTATION")


def test_spawn_failure_records_not_started_and_releases_capacity(tmp_path: Path) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=64 * 1024,
        confirm_bytes=64 * 1024,
        chunk_bytes=16 * 1024,
        command=[str(tmp_path / "definitely-does-not-exist")],
    )
    assert code == 2
    assert record["context"]["downstream_started"] is False
    assert record["context"]["reservation_held_for_entire_downstream_lifetime"] is False
    assert not reserve.exists() and not authority.exists()
    assert receipt.is_file()


@pytest.mark.parametrize(
    ("confirm_delta", "command"),
    [(1, [sys.executable, "-c", "pass"]), (0, [])],
)
def test_invalid_run_contract_never_spawns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confirm_delta: int,
    command: list[str],
) -> None:
    reserve, authority, receipt = _paths(tmp_path)

    def forbidden_popen(*args, **kwargs):
        raise AssertionError("child must not start")

    monkeypatch.setattr(capacity.subprocess, "Popen", forbidden_popen)
    requested = 64 * 1024
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=requested,
        confirm_bytes=requested + confirm_delta,
        chunk_bytes=16 * 1024,
        command=command,
    )
    assert code == 2
    assert record["downstream_started"] is False
    assert not reserve.exists() and not authority.exists()


def test_held_run_hard_cap_refuses_without_allocating(tmp_path: Path) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=capacity.MAX_HELD_RUN_BYTES + 1,
        confirm_bytes=capacity.MAX_HELD_RUN_BYTES + 1,
        command=[sys.executable, "-c", "raise AssertionError('must not run')"],
    )
    assert code == 2
    assert record["downstream_started"] is False
    assert not reserve.exists() and not authority.exists()


def test_release_unlinks_live_authority_before_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve, authority, _ = _paths(tmp_path)
    lease = capacity.acquire_capacity(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        requested_bytes=64 * 1024,
        chunk_bytes=16 * 1024,
        nonce=b"release-order-test-nonce-value!!",
    )
    real_unlink = capacity.os.unlink
    unlinked: list[str] = []

    def observe_unlink(path, *args, **kwargs):
        unlinked.append(os.fspath(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(capacity.os, "unlink", observe_unlink)
    lease.release(reason="TEST_RELEASE_ORDER")
    assert unlinked.index(authority.name) < unlinked.index(reserve.name)


def test_cli_rejects_noncanonical_project_root_without_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reserve, authority, result = _paths(tmp_path)
    code = capacity.main(
        [
            "--project-root",
            "/",
            "probe",
            "--profile",
            "cpfs-probe-8m-v1",
            "--reservation",
            str(reserve),
            "--authority",
            str(authority),
            "--result",
            str(result),
            "--bytes",
            "4096",
        ]
    )
    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert code == 2
    assert failure["status"] == "FAIL_CLOSED_CAPACITY_NOT_PROVEN"
    assert not reserve.exists() and not authority.exists() and not result.exists()


@pytest.mark.parametrize("failure", ["ENOSPC", "fsync EIO", "readback mismatch"])
def test_wrapper_acquisition_failure_never_spawns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    reserve, authority, receipt = _paths(tmp_path)

    def fail_acquire(**kwargs):
        raise capacity.CapacityReservationError(failure)

    def forbidden_popen(*args, **kwargs):
        raise AssertionError("child must not start")

    monkeypatch.setattr(capacity, "acquire_capacity", fail_acquire)
    monkeypatch.setattr(capacity.subprocess, "Popen", forbidden_popen)
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=64 * 1024,
        confirm_bytes=64 * 1024,
        command=[sys.executable, "-c", "raise AssertionError('must not run')"],
    )
    assert code == 2
    assert record["downstream_started"] is False
    assert receipt.is_file()


def test_wrapper_authority_conflict_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    authority.write_bytes(b"foreign-authority")

    def forbidden_popen(*args, **kwargs):
        raise AssertionError("child must not start")

    monkeypatch.setattr(capacity.subprocess, "Popen", forbidden_popen)
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=64 * 1024,
        confirm_bytes=64 * 1024,
        command=[sys.executable, "-c", "raise AssertionError('must not run')"],
    )
    assert code == 2
    assert record["downstream_started"] is False
    assert authority.read_bytes() == b"foreign-authority"
    assert not reserve.exists()


def test_exact_profile_rejects_any_path_or_size_drift(tmp_path: Path) -> None:
    profile = capacity.CLI_PROFILES["mask-capacity-v1"]
    root = capacity.PROJECT_ROOT
    args = capacity.argparse.Namespace(
        profile="mask-capacity-v1",
        mode="run",
        reservation=root / profile["reservation"],
        authority=root / profile["authority"],
        receipt=root / profile["output"],
        result=None,
        bytes=profile["bytes"],
        confirm_bytes=profile["bytes"],
        chunk_bytes=capacity.DEFAULT_CHUNK_BYTES,
    )
    capacity._validate_cli_profile(args, root)
    args.reservation = tmp_path / "drift.reserve"
    with pytest.raises(capacity.CapacityReservationError, match="exactly match"):
        capacity._validate_cli_profile(args, root)


def test_mask_retry_v2_profile_is_independent_and_exact() -> None:
    first = capacity.CLI_PROFILES["mask-capacity-v1"]
    retry = capacity.CLI_PROFILES["mask-capacity-v2"]
    assert retry["bytes"] == first["bytes"] == 128 * 1024 * 1024
    assert retry["reservation"] != first["reservation"]
    assert retry["authority"] != first["authority"]
    assert retry["output"] != first["output"]
    root = capacity.PROJECT_ROOT
    args = capacity.argparse.Namespace(
        profile="mask-capacity-v2",
        mode="run",
        reservation=root / retry["reservation"],
        authority=root / retry["authority"],
        receipt=root / retry["output"],
        result=None,
        bytes=retry["bytes"],
        confirm_bytes=retry["bytes"],
        chunk_bytes=capacity.DEFAULT_CHUNK_BYTES,
    )
    capacity._validate_cli_profile(args, root)


def test_exact_wrapper_blocks_any_acquired_gpu_lease_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    (tmp_path / "_run").mkdir()
    (tmp_path / "_run/GPU_LEASE.json").write_text(
        json.dumps({"status": "ACQUIRED", "holder": "some-other-holder"}),
        encoding="utf-8",
    )

    def forbidden_popen(*args, **kwargs):
        raise AssertionError("child must not start")

    monkeypatch.setattr(capacity.subprocess, "Popen", forbidden_popen)
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=64 * 1024,
        confirm_bytes=64 * 1024,
        command=[sys.executable, "-c", "raise AssertionError('must not run')"],
        require_no_acquired_gpu_lease=True,
    )
    assert code == 2
    assert record["context"]["downstream_started"] is False
    assert "ACQUIRED" in record["context"]["wrapper_error"]
    assert not reserve.exists() and not authority.exists()


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM unavailable")
def test_sigterm_terminates_child_withdraws_pass_and_releases_reserve(
    tmp_path: Path,
) -> None:
    reserve, authority, receipt = _paths(tmp_path)
    child = (
        "import os, signal, time; "
        "os.kill(os.getppid(), signal.SIGTERM); "
        "time.sleep(30)"
    )
    started = time.monotonic()
    code, record = capacity.run_held_command(
        project_root=tmp_path,
        reservation_path=reserve,
        authority_path=authority,
        receipt_path=receipt,
        requested_bytes=64 * 1024,
        confirm_bytes=64 * 1024,
        command=[sys.executable, "-c", child],
    )
    assert time.monotonic() - started < 10
    assert code == 2
    assert record["release_reason"] == "HANDLED_SIGNAL_CLEANUP"
    assert record["context"]["downstream_started"] is True
    assert not reserve.exists() and not authority.exists()
    assert receipt.is_file()


def test_source_contains_no_sparse_allocation_calls() -> None:
    tree = ast.parse(Path(capacity.__file__).read_text(encoding="utf-8"))
    forbidden = {"truncate", "ftruncate", "posix_fallocate", "fallocate"}
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            called.add(node.func.id)
    assert called.isdisjoint(forbidden)
