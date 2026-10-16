"""Fenced GPU groups for one finite Joint26 raw/robotized coordinator.

The legacy central lease remains acquired until the whole coordinator stops.
Each physical card also has a standard scalar V7.1 lease. All changes use the
legacy central GPU lock, so older project wrappers observe the guard.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime, timedelta
import fcntl
from pathlib import Path
from typing import Any, Iterator, Mapping

from chaoyang.governance.common import (
    atomic_json, load_json, process_identity, validate_schema,
)
from chaoyang.governance.gpu_lease_v71 import (
    build_lease, heartbeat, now, parse_time, query_gpu_memory_mib, release,
)


BRANCHES = ("raw", "robotized")
STEP_GUARD_SCHEMA = "chaoyang-joint26-step-budget-gpu-guard-v1"
V71_LEASE_SCHEMA = "chaoyang-gpu-lease-v71"
IDENTITY_FIELDS = (
    "schema_version", "task_id", "attempt_id", "hostname", "boot_id", "pid",
    "process_startticks", "gpu_id", "executor_epoch", "fencing_token", "priority",
)


def parse_gpu_ids(value: Any) -> list[int]:
    """Parse a nonempty comma list or integer sequence of physical GPU IDs."""
    if isinstance(value, str):
        pieces = value.split(",")
        if any(not piece.strip() or not piece.strip().isdecimal() for piece in pieces):
            raise ValueError("NONEMPTY_NONNEGATIVE_GPU_IDS_REQUIRED")
        ids = [int(piece.strip()) for piece in pieces]
    elif isinstance(value, (list, tuple)):
        if any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in value):
            raise ValueError("NONNEGATIVE_INTEGER_GPU_IDS_REQUIRED")
        ids = list(value)
    else:
        raise ValueError("GPU_ID_LIST_REQUIRED")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("NONEMPTY_UNIQUE_GPU_IDS_REQUIRED")
    return ids


def validate_gpu_groups(groups: Mapping[str, Any]) -> dict[str, list[int]]:
    """Keep both lanes disjoint and divide the frozen global batch of 64."""
    if not isinstance(groups, Mapping) or set(groups) != set(BRANCHES):
        raise ValueError("RAW_AND_ROBOTIZED_GPU_GROUPS_REQUIRED")
    result = {branch: parse_gpu_ids(groups[branch]) for branch in BRANCHES}
    if any(64 % len(ids) for ids in result.values()):
        raise ValueError("GPU_GROUP_SIZE_MUST_DIVIDE_GLOBAL_BATCH_64")
    if set(result["raw"]) & set(result["robotized"]):
        raise ValueError("DISJOINT_PHYSICAL_GPU_GROUPS_REQUIRED")
    return result


def ordinary_guard_lease(value: Mapping[str, Any]) -> dict[str, Any]:
    """A V7.1 copy for shared heartbeat/release, never an active disk guard."""
    result = copy.deepcopy(dict(value))
    if result.get("schema_version") == STEP_GUARD_SCHEMA:
        result.pop("gpu_groups", None)
        result["schema_version"] = V71_LEASE_SCHEMA
    return result


def validate_step_guard(value: Mapping[str, Any]) -> dict[str, list[int]]:
    """Reserve the whole pool in a format scalar legacy reclaimers reject."""
    if not isinstance(value, Mapping) or value.get("schema_version") != STEP_GUARD_SCHEMA:
        raise ValueError("ICT_STEP_BUDGET_GUARD_SCHEMA_REQUIRED")
    groups = validate_gpu_groups(value.get("gpu_groups"))
    if value.get("gpu_groups") != groups or value.get("gpu_id") != groups["raw"][0]:
        raise ValueError("ICT_STEP_BUDGET_GUARD_GPU_GROUP_DRIFT")
    validate_schema("gpu_lease_v71.schema.json", ordinary_guard_lease(value))
    if value.get("status") != "ACQUIRED":
        raise ValueError("ICT_STEP_BUDGET_ACTIVE_GUARD_REQUIRED")
    return groups


class ParallelGpuLeases:
    """Manage only this coordinator's leases; never reclaim another owner."""

    def __init__(self, root: Path, writer: Mapping[str, Any], task_id: str,
                 groups: Mapping[str, Any], min_free_mib: int = 40960,
                 guard_expires_at: datetime | None = None,
                 step_only_guard: bool = False):
        root_input = Path(root).absolute()
        self._reject_symlinks(root_input)
        self.root = root_input.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("PROJECT_ROOT_DIRECTORY_REQUIRED")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("TASK_ID_REQUIRED")
        self.task_id = task_id
        self.writer = copy.deepcopy(dict(writer))
        for field in ("pid", "proc_start_ticks", "executor_epoch"):
            if type(self.writer.get(field)) is not int or self.writer[field] < 1:
                raise ValueError("WRITER_IDENTITY_REQUIRED:" + field)
        token = self.writer.get("fencing_token")
        if not isinstance(token, str) or len(token) < 16:
            raise ValueError("WRITER_FENCING_TOKEN_REQUIRED")
        if self.writer.get("scope", task_id) != task_id:
            raise ValueError("WRITER_TASK_SCOPE_MISMATCH")
        if type(min_free_mib) is not int or min_free_mib < 1:
            raise ValueError("POSITIVE_MINIMUM_FREE_GPU_MEMORY_REQUIRED")
        self.min_free_mib = min_free_mib
        current = now()
        if guard_expires_at is not None and (
                not isinstance(guard_expires_at, datetime)
                or guard_expires_at.utcoffset() is None
                or not current < guard_expires_at <= current + timedelta(seconds=43200 + 120)):
            raise ValueError("FINITE_FUTURE_TIMEZONE_AWARE_GUARD_EXPIRY_REQUIRED")
        self.guard_expires_at = guard_expires_at
        if type(step_only_guard) is not bool or step_only_guard and guard_expires_at is not None:
            raise ValueError("ICT_STEP_BUDGET_GUARD_WITHOUT_TASK_DEADLINE_REQUIRED")
        self.step_only_guard = step_only_guard
        self.groups = validate_gpu_groups(groups)
        self.gpu_ids = [gpu_id for branch in BRANCHES for gpu_id in self.groups[branch]]
        self.central_path = self.root / "_run/current/GPU_LEASE.json"
        self.lock_path = self.root / "_run/current/GPU_LEASE.lock"
        self.lease_paths = {
            gpu_id: self.root / "_run/current/ICT_GPU_LEASES" / f"gpu_{gpu_id}.json"
            for gpu_id in self.gpu_ids
        }
        self._templates: dict[Path, dict[str, Any]] = {}
        self._managed: set[Path] = set()
        self._acquired = False
        self.cleanup_errors: list[str] = []
        for path in (self.central_path, self.lock_path, *self.lease_paths.values()):
            self._check_path(path)

    @property
    def acquired(self) -> bool:
        return self._acquired

    @staticmethod
    def _reject_symlinks(path: Path) -> None:
        for component in (path, *path.parents):
            if component.is_symlink():
                raise ValueError("GPU_LEASE_SYMLINK_PATH_FORBIDDEN:" + str(component))

    def _check_path(self, path: Path) -> None:
        self._reject_symlinks(path.absolute())
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("PROJECT_ONLY_GPU_LEASE_PATH_REQUIRED")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self._check_path(self.lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _check_writer(self) -> None:
        identity = process_identity(self.writer["pid"])
        if (not identity.get("alive")
                or identity.get("start_ticks") != self.writer["proc_start_ticks"]):
            raise RuntimeError("ICT_COORDINATOR_PROCESS_IDENTITY_DRIFT")

    def _read(self, path: Path) -> dict[str, Any] | None:
        self._check_path(path)
        if not path.exists():
            return None
        value = load_json(path)
        if value.get("schema_version") == STEP_GUARD_SCHEMA:
            validate_step_guard(value)
            if path != self.central_path:
                raise RuntimeError("ICT_STEP_BUDGET_GUARD_CENTRAL_PATH_REQUIRED")
        else:
            validate_schema("gpu_lease_v71.schema.json", value)
        return value

    def _owned(self, path: Path, value: Mapping[str, Any]) -> bool:
        template = self._templates.get(path)
        if template is None:
            return False
        fields = IDENTITY_FIELDS
        if (template.get("schema_version") == STEP_GUARD_SCHEMA
                and value.get("schema_version") == V71_LEASE_SCHEMA
                and value.get("status") == "RELEASED"):
            fields = tuple(field for field in fields if field != "schema_version")
        if not all(value.get(field) == template[field] for field in fields):
            return False
        return (value.get("schema_version") != STEP_GUARD_SCHEMA
                or value.get("gpu_groups") == template.get("gpu_groups"))

    def _require_owned(self, path: Path, *, acquired: bool = False) -> dict[str, Any]:
        value = self._read(path)
        if value is None or not self._owned(path, value):
            raise RuntimeError("ICT_GPU_LEASE_FENCE_DRIFT:" + str(path))
        if value["status"] not in {"ACQUIRED", "RELEASED"} or (
                acquired and value["status"] != "ACQUIRED"):
            raise RuntimeError("ICT_GPU_LEASE_STATUS_DRIFT:" + str(path))
        return value

    def _new_lease(self, gpu_id: int) -> dict[str, Any]:
        value = build_lease(
            task_id=self.task_id,
            attempt_id=f"attempt_{self.writer['executor_epoch']:04d}",
            pid=self.writer["pid"], gpu_id=gpu_id,
            executor_epoch=self.writer["executor_epoch"],
            fencing_token=self.writer["fencing_token"], priority="CHECKPOINT_TRAINING",
        )
        if value["process_startticks"] != self.writer["proc_start_ticks"]:
            raise RuntimeError("ICT_COORDINATOR_PROCESS_IDENTITY_DRIFT")
        validate_schema("gpu_lease_v71.schema.json", value)
        return value

    def _guard_lease(self, value: dict[str, Any]) -> dict[str, Any]:
        # A scalar legacy GPU PID cannot prove every rank stopped. Step-only
        # guards use a dedicated format rejected by legacy scalar reclaimers;
        # their TTL is still renewed normally and checked by our admission.
        if self.step_only_guard:
            value["schema_version"] = STEP_GUARD_SCHEMA
            value["gpu_groups"] = copy.deepcopy(self.groups)
            validate_step_guard(value)
            return value
        # Explicit-wall tasks retain the existing finite deadline reservation.
        if self.guard_expires_at is not None:
            value["expires_at"] = max(parse_time(value["expires_at"]),
                                      self.guard_expires_at).isoformat(timespec="seconds")
        validate_schema("gpu_lease_v71.schema.json", value)
        return value

    def acquire(self) -> None:
        """Check every slot and card before publishing any acquired lease."""
        with self._lock():
            self._check_writer()
            if self._managed:
                raise RuntimeError("ICT_GPU_LEASE_INSTANCE_ALREADY_USED")
            paths = (self.central_path, *self.lease_paths.values())
            previous = {}
            for path in paths:
                current = self._read(path)
                if current is not None and current["status"] != "RELEASED":
                    raise RuntimeError("WAIT_GPU_RESOURCE_LEASE_HELD:" + str(path))
                previous[path] = current
            for gpu_id in self.gpu_ids:
                memory = query_gpu_memory_mib(gpu_id)
                if (memory is None or type(memory.get("free_mib")) is not int
                        or memory["free_mib"] < self.min_free_mib):
                    raise RuntimeError("WAIT_GPU_RESOURCE_INSUFFICIENT_MEMORY:" + str(gpu_id))
            # Templates are fully constructed before the first write as well.
            self._templates = {
                self.central_path: self._guard_lease(self._new_lease(self.gpu_ids[0]))}
            self._templates.update({self.lease_paths[gpu_id]: self._new_lease(gpu_id)
                                    for gpu_id in self.gpu_ids})
            try:
                for path in paths:
                    self._check_path(path)
                    self._managed.add(path)
                    self._acquired = True
                    try:
                        atomic_json(path, self._templates[path])
                    except BaseException:
                        # An atomic replacement can succeed before a later
                        # directory fsync fails. An unreadable attempted slot
                        # keeps the guard held until ownership can be proved.
                        try:
                            if self._read(path) == previous[path]:
                                self._managed.discard(path)
                        except BaseException:
                            pass
                        raise
                self._acquired = True
            except BaseException as error:
                try:
                    self._release_all_locked("ICT_GPU_ACQUIRE_ROLLBACK")
                except BaseException as cleanup_error:
                    self.cleanup_errors.append(str(cleanup_error))
                raise RuntimeError("ICT_GPU_LEASE_ACQUISITION_FAILED:" + str(error)
                                   + ";cleanup=" + repr(self.cleanup_errors)) from error

    def beat(self, worker_pids: Mapping[str, int]) -> None:
        """Heartbeat the guard and live cards, binding each card to its lane."""
        if not isinstance(worker_pids, Mapping) or set(worker_pids) - set(BRANCHES):
            raise ValueError("VALID_LANE_WORKER_PIDS_REQUIRED")
        if any(type(pid) is not int or pid < 1 for pid in worker_pids.values()):
            raise ValueError("POSITIVE_LANE_WORKER_PIDS_REQUIRED")
        with self._lock():
            self._check_writer()
            if not self._acquired:
                raise RuntimeError("ICT_GPU_LEASE_NOT_ACQUIRED")
            guard = self._require_owned(self.central_path, acquired=True)
            pending = []
            for branch in BRANCHES:
                for gpu_id in self.groups[branch]:
                    path = self.lease_paths[gpu_id]
                    current = self._require_owned(path)
                    if current["status"] == "ACQUIRED":
                        updated = heartbeat(current, pid=self.writer["pid"],
                                            fencing_token=self.writer["fencing_token"])
                        updated["gpu_process_pid"] = worker_pids.get(branch)
                        validate_schema("gpu_lease_v71.schema.json", updated)
                        pending.append((path, updated))
            updated_guard = heartbeat(ordinary_guard_lease(guard), pid=self.writer["pid"],
                                      fencing_token=self.writer["fencing_token"])
            updated_guard["gpu_process_pid"] = next(iter(worker_pids.values()), None)
            self._guard_lease(updated_guard)
            # The guard belongs to the coordinator. Its scalar GPU field is
            # a compatibility reservation, not the whole pool description.
            for path, value in (*pending, (self.central_path, updated_guard)):
                self._check_path(path)
                atomic_json(path, value)

    def _release_cards_locked(self, gpu_ids: list[int], reason: str) -> list[str]:
        errors = []
        for gpu_id in gpu_ids:
            path = self.lease_paths[gpu_id]
            if path not in self._managed:
                continue
            try:
                current = self._require_owned(path)
                if current["status"] == "ACQUIRED":
                    updated = release(current, pid=self.writer["pid"],
                                      fencing_token=self.writer["fencing_token"], reason=reason)
                    self._check_path(path)
                    atomic_json(path, updated)
            except BaseException as error:
                errors.append(str(error))
        return errors

    def release_lane(self, branch: str, reason: str) -> None:
        """Release one finished lane while the legacy guard stays acquired."""
        if branch not in BRANCHES:
            raise ValueError("RAW_OR_ROBOTIZED_LANE_REQUIRED")
        with self._lock():
            self._check_writer()
            self._require_owned(self.central_path, acquired=True)
            errors = self._release_cards_locked(self.groups[branch], reason)
            if errors:
                self.cleanup_errors.extend(errors)
                raise RuntimeError("ICT_GPU_LANE_RELEASE_FAILED:" + repr(errors))

    def _release_all_locked(self, reason: str) -> None:
        errors = self._release_cards_locked(self.gpu_ids, reason)
        # Keep the legacy guard when any card cannot be safely released.
        if not errors and self.central_path in self._managed:
            try:
                current = self._require_owned(self.central_path)
                if current["status"] == "ACQUIRED":
                    updated = release(ordinary_guard_lease(current), pid=self.writer["pid"],
                                      fencing_token=self.writer["fencing_token"], reason=reason)
                    self._check_path(self.central_path)
                    atomic_json(self.central_path, updated)
            except BaseException as error:
                errors.append(str(error))
        if errors:
            self.cleanup_errors.extend(errors)
            raise RuntimeError("ICT_GPU_ALL_RELEASE_FAILED:" + repr(errors))
        self._acquired = False

    def release_all(self, reason: str) -> None:
        """Best-effort owned-card cleanup followed by the legacy guard."""
        with self._lock():
            self._check_writer()
            self._release_all_locked(reason)
