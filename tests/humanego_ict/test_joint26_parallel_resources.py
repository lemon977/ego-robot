from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from chaoyang.governance import gpu_lease_v71
from chaoyang.governance.common import atomic_json, load_json
from chaoyang.governance.gpu_lease_v71 import build_lease
from chaoyang.humanego_ict import joint26_parallel_resources as resources
from chaoyang.ops import run_gpu_command_with_v71_lease as legacy


@pytest.fixture
def manager(tmp_path, monkeypatch):
    # No test calls a physical GPU or loads a model.
    monkeypatch.setattr(resources, "query_gpu_memory_mib", lambda gpu_id: {
        "total_mib": 100000, "used_mib": 0, "free_mib": 100000,
    })
    identity = lambda pid: {"pid": pid, "alive": True, "start_ticks": 456}
    monkeypatch.setattr(resources, "process_identity", identity)
    monkeypatch.setattr(gpu_lease_v71, "process_identity", identity)
    writer = {"pid": 123, "proc_start_ticks": 456,
              "executor_epoch": 3, "fencing_token": "REDACTED_PUBLIC_EXPORT",
              "scope": "joint26_parallel_fixture"}
    return resources.ParallelGpuLeases(
        tmp_path, writer, writer["scope"], {"raw": [0, 1], "robotized": [2, 3]})


def foreign(gpu_id=0, *, status="ACQUIRED"):
    value = build_lease(task_id="other_task", attempt_id="other_attempt", pid=os.getpid(),
                        gpu_id=gpu_id, executor_epoch=1, fencing_token="foreign0123456789abcdef",
                        priority="CHECKPOINT_TRAINING")
    value["status"] = status
    return value


@pytest.mark.parametrize("value, expected", [
    ("0", [0]), (" 0, 2,7 ", [0, 2, 7]), ([0, 7], [0, 7]), ((3, 1), [3, 1]),
])
def test_parse_gpu_ids(value, expected):
    assert resources.parse_gpu_ids(value) == expected


@pytest.mark.parametrize("value", [
    "", " ", "0,", ",0", "0,,1", "0,0", "-1", "1.5", "gpu0", [],
    [0, 0], [-1], [True], ["0"], 0, None,
])
def test_parse_gpu_ids_rejects_invalid(value):
    with pytest.raises(ValueError):
        resources.parse_gpu_ids(value)


def test_groups_normalize_and_do_not_mutate_inputs():
    value = {"raw": "0,1", "robotized": (2, 3, 4, 5)}
    assert resources.validate_gpu_groups(value) == {"raw": [0, 1], "robotized": [2, 3, 4, 5]}
    assert value["raw"] == "0,1"


@pytest.mark.parametrize("groups", [
    {}, {"raw": [0]}, {"raw": [0], "robotized": [1], "extra": [2]},
    {"raw": [0, 1, 2], "robotized": [3]}, {"raw": [0], "robotized": [0]},
    {"raw": [], "robotized": [1]}, {"raw": [0, 0], "robotized": [1]},
])
def test_groups_reject_invalid_or_overlapping_groups(groups):
    with pytest.raises(ValueError):
        resources.validate_gpu_groups(groups)


def test_acquire_publishes_standard_guard_and_scalar_card_leases(manager):
    assert manager.acquire() is None
    assert manager.acquired
    guard = load_json(manager.central_path)
    assert guard["status"] == "ACQUIRED"
    assert guard["gpu_id"] == 0
    for gpu_id, path in manager.lease_paths.items():
        value = load_json(path)
        assert value["status"] == "ACQUIRED"
        assert value["gpu_id"] == gpu_id
        assert value["task_id"] == manager.task_id
        assert value["pid"] == manager.writer["pid"]
        assert value["process_startticks"] == manager.writer["proc_start_ticks"]
        assert value["executor_epoch"] == 3
        assert value["fencing_token"] == manager.writer["fencing_token"]
    manager.release_all("fixture_finished")


@pytest.mark.parametrize("low_gpu, memory", [(0, None), (3, {"free_mib": 40959})])
def test_every_capacity_check_precedes_any_lease_write(manager, monkeypatch, low_gpu, memory):
    queried = []
    writes = []

    def capacity(gpu_id):
        queried.append(gpu_id)
        return memory if gpu_id == low_gpu else {"free_mib": 100000}

    monkeypatch.setattr(resources, "query_gpu_memory_mib", capacity)
    monkeypatch.setattr(resources, "atomic_json", lambda *args: writes.append(args))
    with pytest.raises(RuntimeError, match="WAIT_GPU_RESOURCE_INSUFFICIENT_MEMORY"):
        manager.acquire()
    assert not writes
    assert not manager.central_path.exists()
    assert not manager.acquired
    assert queried[-1] == low_gpu


@pytest.mark.parametrize("slot", ["central", "card"])
def test_existing_acquired_lease_is_not_reclaimed_or_overwritten(manager, monkeypatch, slot):
    path = manager.central_path if slot == "central" else manager.lease_paths[2]
    atomic_json(path, foreign(0 if slot == "central" else 2))
    old_bytes = path.read_bytes()
    monkeypatch.setattr(resources, "query_gpu_memory_mib", lambda gpu_id: pytest.fail("capacity queried"))
    with pytest.raises(RuntimeError, match="WAIT_GPU_RESOURCE_LEASE_HELD"):
        manager.acquire()
    assert path.read_bytes() == old_bytes
    assert not manager.acquired


def test_released_foreign_leases_can_be_replaced(manager):
    atomic_json(manager.central_path, foreign(status="RELEASED"))
    atomic_json(manager.lease_paths[1], foreign(1, status="RELEASED"))
    manager.acquire()
    assert load_json(manager.central_path)["task_id"] == manager.task_id
    manager.release_all("fixture_finished")


@pytest.mark.parametrize("after_replacement", [False, True])
def test_acquire_failure_rolls_back_owned_records_without_deleting(manager, monkeypatch, after_replacement):
    original = resources.atomic_json
    failed = False

    def failing_write(path, value):
        nonlocal failed
        if path == manager.lease_paths[1] and value["status"] == "ACQUIRED" and not failed:
            failed = True
            if after_replacement:
                original(path, value)
            raise OSError("fixture disk failure")
        original(path, value)

    monkeypatch.setattr(resources, "atomic_json", failing_write)
    with pytest.raises(RuntimeError, match="ICT_GPU_LEASE_ACQUISITION_FAILED"):
        manager.acquire()
    assert load_json(manager.central_path)["status"] == "RELEASED"
    assert load_json(manager.lease_paths[0])["status"] == "RELEASED"
    assert manager.lease_paths[1].exists() is after_replacement
    if after_replacement:
        assert load_json(manager.lease_paths[1])["status"] == "RELEASED"
    assert not manager.lease_paths[2].exists()
    assert not manager.acquired


def test_rollback_retains_guard_when_owned_card_cannot_be_released(manager, monkeypatch):
    original = resources.atomic_json

    def failing_write(path, value):
        if path == manager.lease_paths[1] and value["status"] == "ACQUIRED":
            raise OSError("fixture acquire failure")
        if path == manager.lease_paths[0] and value["status"] == "RELEASED":
            raise OSError("fixture cleanup failure")
        original(path, value)

    monkeypatch.setattr(resources, "atomic_json", failing_write)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        manager.acquire()
    assert manager.cleanup_errors
    assert load_json(manager.central_path)["status"] == "ACQUIRED"
    assert load_json(manager.lease_paths[0])["status"] == "ACQUIRED"
    monkeypatch.setattr(resources, "atomic_json", original)
    manager.release_all("fixture_recovery")


def test_rollback_retains_guard_when_committed_slot_cannot_be_read(manager, monkeypatch):
    original_write, original_read = resources.atomic_json, resources.load_json
    failed_path = manager.lease_paths[1]

    def failing_write(path, value):
        original_write(path, value)
        if path == failed_path and value["status"] == "ACQUIRED":
            raise OSError("fixture post-replacement failure")

    def failing_read(path):
        if path == failed_path:
            raise OSError("fixture slot cannot be read")
        return original_read(path)

    monkeypatch.setattr(resources, "atomic_json", failing_write)
    monkeypatch.setattr(resources, "load_json", failing_read)
    with pytest.raises(RuntimeError, match="ICT_GPU_LEASE_ACQUISITION_FAILED"):
        manager.acquire()
    assert manager.acquired
    assert manager.cleanup_errors
    assert load_json(manager.central_path)["status"] == "ACQUIRED"
    assert load_json(failed_path)["status"] == "ACQUIRED"
    monkeypatch.setattr(resources, "atomic_json", original_write)
    monkeypatch.setattr(resources, "load_json", original_read)
    manager.release_all("fixture_recovery")


def test_heartbeat_binds_every_card_to_its_lane(manager):
    manager.acquire()
    manager.beat({"raw": 10001, "robotized": 10002})
    for branch, ids in manager.groups.items():
        for gpu_id in ids:
            value = load_json(manager.lease_paths[gpu_id])
            assert value["gpu_process_pid"] == {"raw": 10001, "robotized": 10002}[branch]
            assert value["status"] == "ACQUIRED"
    assert load_json(manager.central_path)["status"] == "ACQUIRED"
    manager.release_all("fixture_finished")


def test_heartbeat_clears_finished_lane_worker_pid_while_lease_remains_acquired(manager):
    manager.acquire()
    manager.beat({"raw": 10001, "robotized": 10002})
    manager.beat({"robotized": 10002})
    for gpu_id in manager.groups["raw"]:
        value = load_json(manager.lease_paths[gpu_id])
        assert value["gpu_process_pid"] is None
        assert value["status"] == "ACQUIRED"
    assert load_json(manager.lease_paths[2])["gpu_process_pid"] == 10002
    assert load_json(manager.central_path)["status"] == "ACQUIRED"
    manager.release_all("fixture_finished")


def test_release_lane_preserves_other_lane_and_guard_then_heartbeat_skips_released_cards(manager):
    manager.acquire()
    other_bytes = {gpu_id: manager.lease_paths[gpu_id].read_bytes() for gpu_id in (2, 3)}
    guard_bytes = manager.central_path.read_bytes()
    manager.release_lane("raw", "raw_finished")
    assert manager.central_path.read_bytes() == guard_bytes
    assert all(manager.lease_paths[gpu_id].read_bytes() == value for gpu_id, value in other_bytes.items())
    released_bytes = {gpu_id: manager.lease_paths[gpu_id].read_bytes() for gpu_id in (0, 1)}
    manager.beat({"robotized": 10002})
    assert all(manager.lease_paths[gpu_id].read_bytes() == value for gpu_id, value in released_bytes.items())
    assert load_json(manager.lease_paths[2])["gpu_process_pid"] == 10002
    manager.release_lane("robotized", "robotized_finished")
    assert load_json(manager.central_path)["status"] == "ACQUIRED"
    manager.release_all("coordinator_finished")
    assert all(load_json(path)["status"] == "RELEASED"
               for path in (manager.central_path, *manager.lease_paths.values()))


def test_legacy_wrapper_cannot_acquire_until_whole_coordinator_releases(manager, monkeypatch):
    monkeypatch.setattr(legacy, "query_gpu_pids", lambda: set())
    monkeypatch.setattr(legacy, "physical_gpu_capacity", lambda *args: {"free_mib": 100000})
    args = SimpleNamespace(task_id="legacy_fixture", attempt_id="attempt_0001", executor_epoch=1,
                           priority="CHECKPOINT_TRAINING", gpu_id=0, min_free_mib=40960,
                           lease=manager.central_path, lock=manager.lock_path)
    manager.acquire()
    assert legacy.try_acquire(args, "legacy0123456789abcdef") is None
    manager.release_lane("raw", "raw_finished")
    assert legacy.try_acquire(args, "legacy0123456789abcdef") is None
    manager.release_lane("robotized", "robotized_finished")
    assert legacy.try_acquire(args, "legacy0123456789abcdef") is None
    manager.release_all("coordinator_finished")
    assert legacy.try_acquire(args, "legacy0123456789abcdef")["task_id"] == "legacy_fixture"


def test_finite_guard_prevents_legacy_reclaim_after_owner_dies_and_keeps_card_ttl(manager, monkeypatch):
    started = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(resources, "now", lambda: started)
    monkeypatch.setattr(gpu_lease_v71, "now", lambda: started)
    deadline = started + timedelta(hours=2)
    guarded = resources.ParallelGpuLeases(
        manager.root, manager.writer, manager.task_id, manager.groups, guard_expires_at=deadline)
    guarded.acquire()
    dead = lambda pid: {"alive": False, "start_ticks": None}
    guard = load_json(guarded.central_path)
    at = started + timedelta(seconds=121)
    audit = gpu_lease_v71.audit_reclaim(guard, at=at, identity_probe=dead, gpu_pids=set())
    assert not audit["heartbeat_expired"]
    assert not audit["safe_to_reclaim"]
    assert gpu_lease_v71.parse_time(guard["expires_at"]) == deadline
    card = load_json(guarded.lease_paths[0])
    assert gpu_lease_v71.parse_time(card["expires_at"]) == started + timedelta(seconds=120)
    assert gpu_lease_v71.audit_reclaim(card, at=at, identity_probe=dead, gpu_pids=set())["safe_to_reclaim"]
    monkeypatch.setattr(gpu_lease_v71, "now", lambda: started + timedelta(seconds=30))
    guarded.beat({"robotized": 10002})
    guard = load_json(guarded.central_path)
    assert gpu_lease_v71.parse_time(guard["expires_at"]) == deadline
    assert guard["gpu_process_pid"] == 10002
    card = load_json(guarded.lease_paths[0])
    assert gpu_lease_v71.parse_time(card["expires_at"]) == started + timedelta(seconds=150)
    guarded.release_all("normal_completion")
    assert load_json(guarded.central_path)["status"] == "RELEASED"
    assert all(load_json(path)["status"] == "RELEASED" for path in guarded.lease_paths.values())


@pytest.mark.parametrize("expiry", [
    datetime(2026, 10, 14, 13),
    datetime(2026, 10, 14, 11, 59, 59, tzinfo=timezone.utc),
    datetime(2026, 10, 14, 12, tzinfo=timezone.utc),
    datetime(2026, 10, 15, 0, 2, 1, tzinfo=timezone.utc),
    "2026-10-14T13:00:00+00:00",
])
def test_guard_expiry_requires_timezone_future_and_finite_budget(manager, monkeypatch, expiry):
    monkeypatch.setattr(resources, "now", lambda: datetime(2026, 10, 14, 12, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="FINITE_FUTURE_TIMEZONE_AWARE_GUARD_EXPIRY_REQUIRED"):
        resources.ParallelGpuLeases(
            manager.root, manager.writer, manager.task_id, manager.groups, guard_expires_at=expiry)


@pytest.mark.parametrize("field, value", [
    ("task_id", "foreign_task"), ("pid", 99999999), ("process_startticks", 99999999),
    ("executor_epoch", 4), ("fencing_token", "foreign0123456789abcdef"),
    ("gpu_id", 99), ("boot_id", "foreign_boot"), ("hostname", "foreign_host"),
])
def test_heartbeat_preflights_all_fences_before_writing(manager, monkeypatch, field, value):
    manager.acquire()
    path = manager.lease_paths[3]
    current = load_json(path)
    current[field] = value
    atomic_json(path, current)
    before = {slot: slot.read_bytes() for slot in (manager.central_path, *manager.lease_paths.values())}
    monkeypatch.setattr(resources, "atomic_json", lambda *args: pytest.fail("heartbeat wrote before full fence check"))
    with pytest.raises(RuntimeError, match="ICT_GPU_LEASE_FENCE_DRIFT"):
        manager.beat({"raw": 10001, "robotized": 10002})
    assert all(slot.read_bytes() == value for slot, value in before.items())


def test_release_never_overwrites_foreign_card_and_retains_guard(manager):
    manager.acquire()
    path = manager.lease_paths[2]
    atomic_json(path, foreign(2))
    old_bytes = path.read_bytes()
    with pytest.raises(RuntimeError, match="ICT_GPU_ALL_RELEASE_FAILED"):
        manager.release_all("fixture_cleanup")
    assert path.read_bytes() == old_bytes
    assert load_json(manager.central_path)["status"] == "ACQUIRED"
    assert all(load_json(manager.lease_paths[gpu_id])["status"] == "RELEASED" for gpu_id in (0, 1, 3))


def test_release_never_overwrites_foreign_guard_but_releases_owned_cards(manager):
    manager.acquire()
    atomic_json(manager.central_path, foreign())
    old_bytes = manager.central_path.read_bytes()
    with pytest.raises(RuntimeError, match="ICT_GPU_ALL_RELEASE_FAILED"):
        manager.release_all("fixture_cleanup")
    assert manager.central_path.read_bytes() == old_bytes
    assert all(load_json(path)["status"] == "RELEASED" for path in manager.lease_paths.values())


def test_owner_startticks_drift_prevents_heartbeat_or_release(manager, monkeypatch):
    manager.acquire()
    before = manager.central_path.read_bytes()
    monkeypatch.setattr(resources, "process_identity", lambda pid: {"alive": True, "start_ticks": 1})
    with pytest.raises(RuntimeError, match="PROCESS_IDENTITY_DRIFT"):
        manager.beat({})
    with pytest.raises(RuntimeError, match="PROCESS_IDENTITY_DRIFT"):
        manager.release_all("invalid_owner")
    assert manager.central_path.read_bytes() == before


def test_repeated_acquire_rejected_and_repeated_release_all_idempotent(manager):
    manager.acquire()
    with pytest.raises(RuntimeError, match="INSTANCE_ALREADY_USED"):
        manager.acquire()
    manager.release_all("finished")
    before = manager.central_path.read_bytes()
    manager.release_all("finished_again")
    assert manager.central_path.read_bytes() == before
    assert not manager.acquired


@pytest.mark.parametrize("pids", [{"unknown": 1}, {"raw": 0}, {"raw": True}, {"raw": None}])
def test_heartbeat_rejects_invalid_worker_bindings(manager, pids):
    with pytest.raises(ValueError):
        manager.beat(pids)


def test_path_symlink_escape_is_rejected_before_any_lease_write(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "_run").symlink_to(tmp_path)
    writer = {"pid": 123, "proc_start_ticks": 456,
              "executor_epoch": 1, "fencing_token": "0123456789abcdef"}
    with pytest.raises(ValueError, match="GPU_LEASE_SYMLINK_PATH_FORBIDDEN"):
        resources.ParallelGpuLeases(root, writer, "task", {"raw": [0], "robotized": [1]})


@pytest.mark.parametrize("target_outside_root", [False, True])
@pytest.mark.parametrize("ancestor", [False, True])
def test_root_input_or_ancestor_symlinks_are_rejected_before_resolve(
        tmp_path, target_outside_root, ancestor):
    target = tmp_path / "actual_project"
    target.mkdir()
    if ancestor:
        (target / "project").mkdir()
    alias = tmp_path / "project_alias"
    alias.symlink_to("/tmp" if target_outside_root else target)
    root = alias / "project" if ancestor else alias
    writer = {"pid": 123, "proc_start_ticks": 456,
              "executor_epoch": 1, "fencing_token": "0123456789abcdef"}
    with pytest.raises(ValueError, match="GPU_LEASE_SYMLINK_PATH_FORBIDDEN"):
        resources.ParallelGpuLeases(root, writer, "task", {"raw": [0], "robotized": [1]})


@pytest.mark.parametrize("slot", ["central", "lock", "lease", "current_dir", "lease_dir"])
@pytest.mark.parametrize("target_outside_root", [False, True])
def test_runtime_slot_and_parent_symlinks_are_rejected_even_inside_project(
        manager, slot, target_outside_root):
    paths = {
        "central": manager.central_path,
        "lock": manager.lock_path,
        "lease": manager.lease_paths[0],
        "current_dir": manager.lock_path.parent,
        "lease_dir": manager.lease_paths[0].parent,
    }
    alias = paths[slot]
    alias.parent.mkdir(parents=True, exist_ok=True)
    target = manager.root / ("actual_dir" if slot.endswith("_dir") else "actual_slot.json")
    if target_outside_root:
        target = "/tmp"
    elif slot.endswith("_dir"):
        target.mkdir()
    else:
        atomic_json(target, foreign())
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="GPU_LEASE_SYMLINK_PATH_FORBIDDEN"):
        manager.acquire()
    assert not manager.acquired
