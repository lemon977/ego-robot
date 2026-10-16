"""Per-card candidate GPU isolation; not a queue or governance publisher."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_LEASE_PATH = PROJECT_ROOT / "_run/current/GPU_LEASE.json"
LEGACY_LOCK_PATH = PROJECT_ROOT / "_run/current/GPU_LEASE.lock"
CANDIDATE_CARD_LOCK_ROOT = PROJECT_ROOT / "_run/current/locks/inpainting_candidate_cards"


class ResourceError(RuntimeError):
    pass


def proc_start_ticks(pid: int) -> int:
    """Parse /proc correctly when process comm includes spaces or parentheses."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            raise ResourceError("WRITER_ZOMBIE")
        return int(fields[19])
    except (OSError, ValueError, IndexError) as exc:
        raise ResourceError("WRITER_PROCESS_UNAVAILABLE") from exc


def read_gpu_inventory() -> list[dict]:
    """Read-only physical inventory; no reliance on parent CUDA_VISIBLE_DEVICES."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceError("GPU_INVENTORY_UNAVAILABLE") from exc
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise ResourceError("GPU_INVENTORY_INVALID")
        rows.append({"physical_index": int(fields[0]), "uuid": fields[1]})
    if not rows or len({row["uuid"] for row in rows}) != len(rows):
        raise ResourceError("GPU_INVENTORY_EMPTY_OR_DUPLICATED")
    return rows


@dataclass(frozen=True)
class DeviceMapping:
    physical_index: int
    uuid: str
    logical_index: int = 0

    @classmethod
    def from_inventory(cls, physical_index: int, inventory: list[dict]) -> "DeviceMapping":
        rows = [row for row in inventory if row.get("physical_index") == physical_index]
        if type(physical_index) is not int or len(rows) != 1:
            raise ResourceError("GPU_PHYSICAL_ID_NOT_IN_INVENTORY")
        value = rows[0].get("uuid")
        if not isinstance(value, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", value):
            raise ResourceError("GPU_UUID_INVALID")
        return cls(physical_index, value)

    def worker_environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": self.uuid,
                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        return env


def validate_lanes(mappings: list[DeviceMapping]) -> None:
    if (not 1 <= len(mappings) <= 2 or len({mapping.uuid for mapping in mappings}) != len(mappings)
            or len({mapping.physical_index for mapping in mappings}) != len(mappings)):
        raise ResourceError("GPU_LANES_LIMIT_OR_DUPLICATE_CARD")
    if any(mapping.logical_index != 0 for mapping in mappings):
        raise ResourceError("GPU_SINGLE_CARD_WORKER_MUST_USE_LOGICAL_ZERO")
    if any(type(mapping.physical_index) is not int or mapping.physical_index < 0
           or not isinstance(mapping.uuid, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", mapping.uuid)
           for mapping in mappings):
        raise ResourceError("GPU_MAPPING_INVALID")


class GpuCardLease:
    """Fail closed on occupied card, including stale locks; never steals a lease.

    The caller must still hold the existing Task Packet governance fence. This
    child lease adds card-level isolation rather than replacing that authority.
    """
    def __init__(self, lock_root: str | Path, mapping: DeviceMapping, *, task_id: str,
                 run_signature: str, fencing_token: str, writer_pid: int | None = None,
                 writer_start_ticks: int | None = None):
        root = Path(lock_root).resolve()
        if not root.is_relative_to(PROJECT_ROOT) or root == PROJECT_ROOT:
            raise ResourceError("GPU_LOCK_OUTSIDE_PROJECT")
        if root != CANDIDATE_CARD_LOCK_ROOT.resolve():
            raise ResourceError("GPU_CARD_LOCK_ROOT_MUST_BE_CANONICAL")
        validate_lanes([mapping])
        if not task_id or not re.fullmatch(r"[0-9a-f]{64}", run_signature) or not fencing_token:
            raise ResourceError("GPU_TASK_IDENTITY_MISSING")
        pid = os.getpid() if writer_pid is None else writer_pid
        ticks = proc_start_ticks(pid) if writer_start_ticks is None else writer_start_ticks
        if proc_start_ticks(pid) != ticks:
            raise ResourceError("GPU_WRITER_START_TICKS_MISMATCH")
        self.root, self.mapping = root, mapping
        self.path = root / (mapping.uuid + ".json")
        self.owner = {"schema": "inpainting-candidate-card-lease-v1", "task_id": task_id,
                      "run_signature": run_signature, "fencing_token": fencing_token,
                      "pid": pid, "proc_start_ticks": ticks,
                      "physical_index": mapping.physical_index, "uuid": mapping.uuid,
                      "worker_logical_index": 0}
        self._held = False
        self._legacy_guard = None

    def _acquire_legacy_guard(self) -> None:
        """Read-only shared guard interoperates with old LOCK_EX publishers.

        Candidate cards can share this guard. Any acquired/ambiguous old global
        lease blocks all candidate cards conservatively. No old JSON is changed
        and no missing old lock is created. Holding the guard avoids check/start
        races with FileGpuLease's exclusive acquisition transaction.
        """
        try:
            guard = LEGACY_LOCK_PATH.open("rb")
            try:
                fcntl.flock(guard.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                current = json.loads(LEGACY_LEASE_PATH.read_text())
                if not isinstance(current, dict) or current.get("status") != "RELEASED":
                    raise ResourceError("WAIT_GPU_RESOURCE_LEGACY_GLOBAL_LEASE")
            except Exception:
                guard.close()
                raise
        except (OSError, ValueError) as exc:
            raise ResourceError("WAIT_GPU_RESOURCE_LEGACY_STATE_UNAVAILABLE") from exc
        self._legacy_guard = guard

    def _release_legacy_guard(self) -> None:
        if self._legacy_guard is not None:
            self._legacy_guard.close()
            self._legacy_guard = None

    def acquire(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        if proc_start_ticks(self.owner["pid"]) != self.owner["proc_start_ticks"]:
            raise ResourceError("GPU_WRITER_FENCE_CHANGED")
        self._acquire_legacy_guard()
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            self._release_legacy_guard()
            raise ResourceError("WAIT_GPU_RESOURCE") from exc
        except OSError:
            self._release_legacy_guard()
            raise
        with os.fdopen(descriptor, "w") as stream:
            json.dump(self.owner, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        self._held = True
        return dict(self.owner)

    def assert_owned(self) -> None:
        if not self._held:
            raise ResourceError("GPU_LEASE_NOT_ACQUIRED")
        try:
            actual = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise ResourceError("GPU_LEASE_OWNER_UNREADABLE") from exc
        if actual != self.owner or proc_start_ticks(self.owner["pid"]) != self.owner["proc_start_ticks"]:
            raise ResourceError("GPU_LEASE_OWNER_MISMATCH")

    def release(self) -> None:
        self.assert_owned()
        # Only this generated, still-owned lock is removed, never another task's.
        self.path.unlink()
        self._held = False
        self._release_legacy_guard()

    def __enter__(self) -> "GpuCardLease":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
