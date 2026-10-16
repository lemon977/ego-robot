"""Optional training wall budgets and owned, rolling step-only GPU guards."""
from __future__ import annotations

import fcntl
from pathlib import Path


STEP_GUARD_SCHEMA = 'chaoyang-joint26-single-step-budget-gpu-guard-v1'
V71_SCHEMA = 'chaoyang-gpu-lease-v71'


def normalize_wall_seconds(value, *, prepare=False):
    """Training is update-bounded by default; CPU preparation stays bounded."""
    if value is None or (type(value) is int and value == 0):
        return 43200 if prepare else None
    if type(value) is not int or not 600 < value <= 43200:
        raise ValueError('OPTIONAL_600_TO_43200_SECOND_BUDGET_REQUIRED')
    return value


def step_bounded_lease_update(packet, *, lease_path: Path, lock_path: Path,
                              worker_pid=None, release_reason=None):
    """Renew finite lease TTL without assigning any fixed task deadline.

    The distinct acquired schema makes old scalar-PID reclaimers hold this
    guard. Only this fenced owner releases it, restoring ordinary V7.1 bytes.
    A crashed owner requires explicit whole-process recovery, not takeover.
    """
    from chaoyang.governance.common import atomic_json
    from chaoyang.governance.gpu_lease_v71 import heartbeat, release
    from chaoyang.ops.run_gpu_command_with_v71_lease import load

    budget = packet.get('budgets', {})
    if (packet.get('termination_policy') != 'OPTIMIZER_UPDATES_ONLY'
            or 'deadline_at' not in packet or packet['deadline_at'] is not None
            or 'wall_seconds_max' not in budget or budget['wall_seconds_max'] is not None
            or type(budget.get('optimizer_updates_max')) is not int
            or budget['optimizer_updates_max'] < 1):
        raise RuntimeError('STEP_BOUNDED_SIGNED_TASK_REQUIRED')
    writer = packet['writer']
    with Path(lock_path).open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load(Path(lease_path))
        if (current.get('schema_version') not in {V71_SCHEMA, STEP_GUARD_SCHEMA}
                or current.get('task_id') != packet['task_id']
                or current.get('pid') != writer['pid']
                or current.get('fencing_token') != writer['fencing_token']
                or current.get('process_startticks') != writer['proc_start_ticks']
                or current.get('executor_epoch') != writer['executor_epoch']):
            raise RuntimeError('CENTRAL_GPU_LEASE_FENCE_DRIFT')
        ordinary = dict(current, schema_version=V71_SCHEMA)
        if release_reason is not None:
            updated = release(ordinary, pid=writer['pid'],
                              fencing_token=writer['fencing_token'], reason=release_reason)
        else:
            updated = heartbeat(ordinary, pid=writer['pid'], fencing_token=writer['fencing_token'])
            if worker_pid is not None:
                updated['gpu_process_pid'] = worker_pid
            updated['schema_version'] = STEP_GUARD_SCHEMA
        atomic_json(Path(lease_path), updated)
