import json
import os
import fcntl

import pytest

from chaoyang.inpainting_candidate.resources import DeviceMapping, GpuCardLease, ResourceError, proc_start_ticks, validate_lanes
from chaoyang.inpainting_candidate import resources as resources_module


@pytest.fixture(autouse=True)
def isolated_legacy_state(tmp_path, monkeypatch):
    lock, lease = tmp_path / "legacy.lock", tmp_path / "legacy.json"
    lock.write_bytes(b"")
    lease.write_text('{"status":"RELEASED"}')
    monkeypatch.setattr(resources_module, "LEGACY_LOCK_PATH", lock)
    monkeypatch.setattr(resources_module, "LEGACY_LEASE_PATH", lease)
    monkeypatch.setattr(resources_module, "CANDIDATE_CARD_LOCK_ROOT", tmp_path)
    return lock, lease


def mapping(index=0):
    return DeviceMapping.from_inventory(index, [{"physical_index": 0, "uuid": "GPU-a"}, {"physical_index": 3, "uuid": "GPU-b"}])


def test_explicit_physical_to_worker_logical_mapping():
    selected = mapping(3)
    env = selected.worker_environment({"CUDA_VISIBLE_DEVICES": "0", "KEEP": "yes"})
    assert selected.physical_index == 3 and selected.logical_index == 0
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-b" and env["KEEP"] == "yes"
    assert env["OMP_NUM_THREADS"] == "2"


def test_card_missing_from_inventory_rejected():
    with pytest.raises(ResourceError, match="NOT_IN_INVENTORY"):
        mapping(1)


def test_lane_limit_duplicate_and_invalid_uuid_rejected():
    validate_lanes([mapping(0), mapping(3)])
    for values in ([mapping(0), mapping(0)], [mapping(0)] * 3, [], [mapping(0), DeviceMapping(0, "GPU-foreign")]):
        with pytest.raises(ResourceError, match="LIMIT_OR_DUPLICATE"):
            validate_lanes(values)
    with pytest.raises(ResourceError, match="MAPPING_INVALID"):
        validate_lanes([DeviceMapping(0, "../../foreign")])


def test_own_lease_acquire_release_and_same_card_conflict(tmp_path):
    kwargs = {"task_id": "synthetic", "run_signature": "a" * 64, "fencing_token": "own"}
    first, second = GpuCardLease(tmp_path, mapping(), **kwargs), GpuCardLease(tmp_path, mapping(), **kwargs)
    first.acquire()
    with pytest.raises(ResourceError, match="WAIT_GPU_RESOURCE"):
        second.acquire()
    first.release()
    assert not first.path.exists()


def test_different_cards_can_hold_independent_leases(tmp_path):
    kwargs = {"task_id": "synthetic", "run_signature": "b" * 64, "fencing_token": "own"}
    with GpuCardLease(tmp_path, mapping(0), **kwargs) as first:
        with GpuCardLease(tmp_path, mapping(3), **kwargs) as second:
            assert first.path != second.path


def test_never_release_foreign_owner(tmp_path):
    lease = GpuCardLease(tmp_path, mapping(), task_id="synthetic", run_signature="c" * 64, fencing_token="own")
    lease.acquire()
    foreign = dict(lease.owner, fencing_token="foreign")
    lease.path.write_text(json.dumps(foreign))
    with pytest.raises(ResourceError, match="OWNER_MISMATCH"):
        lease.release()
    assert json.loads(lease.path.read_text())["fencing_token"] == "foreign"


def test_start_ticks_fence_and_project_write_boundary(tmp_path):
    assert proc_start_ticks(os.getpid()) > 0
    kwargs = {"task_id": "synthetic", "run_signature": "d" * 64, "fencing_token": "own"}
    with pytest.raises(ResourceError, match="START_TICKS_MISMATCH"):
        GpuCardLease(tmp_path, mapping(), writer_start_ticks=1, **kwargs)
    with pytest.raises(ResourceError, match="OUTSIDE_PROJECT"):
        GpuCardLease("/tmp/foreign", mapping(), **kwargs)


def test_active_or_ambiguous_legacy_lease_not_overridden(tmp_path, isolated_legacy_state):
    _, old_lease = isolated_legacy_state
    for value in ({"status": "ACQUIRED", "gpu_id": 0}, {"status": "UNKNOWN"}):
        old_lease.write_text(json.dumps(value))
        lease = GpuCardLease(tmp_path, mapping(), task_id="synthetic", run_signature="e" * 64, fencing_token="own")
        with pytest.raises(ResourceError, match="LEGACY_GLOBAL_LEASE"):
            lease.acquire()
        assert json.loads(old_lease.read_text()) == value
        assert not lease.path.exists()


def test_legacy_read_only_guard_closes_check_start_race(tmp_path, isolated_legacy_state):
    old_lock, old_lease = isolated_legacy_state
    original_bytes = old_lease.read_bytes()
    with GpuCardLease(tmp_path, mapping(), task_id="synthetic", run_signature="f" * 64, fencing_token="own"):
        with old_lock.open("rb") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with old_lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert old_lease.read_bytes() == original_bytes


def test_same_card_cannot_use_a_second_project_lock_root(tmp_path):
    other = tmp_path / "second_lock_root"
    with pytest.raises(ResourceError, match="LOCK_ROOT_MUST_BE_CANONICAL"):
        GpuCardLease(other, mapping(), task_id="synthetic", run_signature="a" * 64, fencing_token="own")
