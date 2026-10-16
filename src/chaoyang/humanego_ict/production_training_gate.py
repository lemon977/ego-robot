"""Independent, fenced source-replay production and derived training admission.

No torch/model/SDK import occurs here. Old training routes remain unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket

from .derived_runtime import _json, _path
from .runtime_gate import (file_ref, verify_ref, _is_descendant,
                          bind_cuda_visible_devices, runtime_cache_paths, storage_ready as _statvfs_ready)

TRAINING_SCOPE = "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING"
SCHEMA = "pick-place-humanego-ict-production-paired-v1"
PREPARE = "prepare_pick_place_humanego_ict_production_v1"
OPERATION = "train_pick_place_humanego_ict_production_v1"
PREPARE_ROUTE = "pick_place_humanego_ict_production_prepare_v1"
ROUTE = "pick_place_humanego_ict_production_training_v1"
PREPARE_MODE = "HUMANEGO_ICT_PRODUCTION_PREPARE_CPU"
EXECUTION_MODE = "HUMANEGO_ICT_PRODUCTION_DERIVED_TRAINING"


def production_storage_ready(output: Path, minimum_bytes: int, *, packet: dict, root: Path) -> dict:
    """New-route-only measured fallback for this mount's zero statvfs reporting.

    This does not invent free space or guarantee later capacity. Any checkpoint
    ENOSPC propagates from the protected atomic writer; old admission is unchanged.
    """
    if type(minimum_bytes) is not int or minimum_bytes < 4 * 1024**3:
        raise ValueError("MINIMUM_CHECKPOINT_CAPACITY_NOT_LOWERED")
    output = _path(output, root.resolve(strict=True))
    parent = output
    while not parent.exists():
        parent = parent.parent
    stat = os.statvfs(parent)
    free = stat.f_bavail * stat.f_frsize
    if free >= minimum_bytes:
        _statvfs_ready(output, minimum_bytes)
        return {"mode": "STATVFS_REPORTED_CAPACITY", "minimum_bytes": minimum_bytes,
                "statvfs_reported_free_bytes": free}
    if free != 0:
        raise ValueError("BLOCKED_OUTPUT_STORAGE_CAPACITY")
    if packet.get("execution_mode") not in {PREPARE_MODE, EXECUTION_MODE}:
        raise ValueError("MEASURED_CAPACITY_FALLBACK_ONLY_NEW_PRODUCTION_ROUTE")
    reference = packet.get("inputs", {}).get("storage_capacity_observation")
    if reference is None:
        raise ValueError("ZERO_STATVFS_REQUIRES_BOUND_MEASURED_CAPACITY_EVIDENCE")
    _path(Path(reference["path"]), root.resolve(strict=True), existing=True)
    observation = _json(reference)
    now = datetime.now(timezone.utc).timestamp()
    completed, started = _utc(observation["completed_at"]), _utc(observation["started_at"])
    probe = _path(Path(observation["probe_path"]), root.resolve(strict=True))
    probe_parent = probe.parent
    while not probe_parent.exists():
        probe_parent = probe_parent.parent
    digest = observation.get("write_sha256")
    if (observation.get("schema") != "task-local-write-capacity-observation-v1"
            or observation.get("fixture_only") is True
            or observation.get("status") != "PASSED_MEASURED_WRITE_READBACK"
            or observation.get("measured_capacity_not_general_free_space_guarantee") is not True
            or observation.get("source_data_or_existing_files_removed") is not False
            or observation.get("task_owned_probe_released") is not True or probe.exists()
            or type(observation.get("device_id")) is not int or observation["device_id"] != os.stat(parent).st_dev
            or os.stat(probe_parent).st_dev != os.stat(parent).st_dev
            or observation.get("statvfs_free_before") != 0 or observation.get("statvfs_free_after") != 0
            or type(observation.get("required_bytes")) is not int or observation["required_bytes"] < minimum_bytes
            or type(observation.get("written_bytes")) is not int or observation["written_bytes"] < minimum_bytes
            or observation.get("fsynced_bytes") != observation["written_bytes"]
            or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or observation.get("read_sha256") != digest or not started <= completed <= now
            or now - completed > 3600):
        raise ValueError("MEASURED_PROJECT_CAPACITY_EVIDENCE_UNQUALIFIED")
    return {"mode": "MEASURED_WRITE_READBACK", "minimum_bytes": minimum_bytes,
        "statvfs_reported_free_bytes": 0, "measured_written_fsynced_readback_bytes": observation["written_bytes"],
        "observation_ref": reference, "measured_capacity_not_general_free_space_guarantee": True,
        "later_enospc_policy": "FAIL_CLOSED_PRESERVE_LAST_COMPLETE_ATOMIC_CHECKPOINT"}


def _utc(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("EXPLICIT_UTC_OFFSET_REQUIRED")
    return stamp.astimezone(timezone.utc).timestamp()


def _sealed_producer(root, packet, state, index):
    """Consume only the sealed real CPU producer, never an aligned file alone."""
    manifest_ref = packet["inputs"]["production_manifest"]
    manifest = _json(manifest_ref)
    producer_ref = manifest.get("producer_packet")
    if not isinstance(producer_ref, dict):
        raise ValueError("SEALED_PRODUCTION_PACKET_REFERENCE_REQUIRED")
    path = _path(Path(producer_ref["path"]), root, existing=True)
    if not path.is_relative_to(root / "tasks/current"):
        raise ValueError("REGISTERED_SOURCE_PRODUCER_PACKET_REQUIRED")
    producer = _json(producer_ref)
    task_id = producer.get("task_id")
    if (task_id == packet["task_id"] or producer.get("execution_mode") != PREPARE_MODE
            or producer.get("training_scope") != TRAINING_SCOPE):
        raise ValueError("SEPARATE_CPU_SOURCE_PRODUCER_REQUIRED")
    rows = [row for row in state["tasks"] if row.get("task_id") == task_id]
    pointers = [row for row in index["task_packets"] if row.get("task_id") == task_id]
    if (len(rows) != 1 or rows[0].get("status") != "PASSED"
            or rows[0].get("task_packet") != producer_ref
            or rows[0].get("run_signature") != producer.get("run_signature")
            or len(pointers) != 1 or pointers[0].get("execution_allowed") is not False
            or pointers[0].get("packet_sha256") != producer_ref["sha256"]
            or (root / pointers[0]["packet_path"]).resolve() != path):
        raise ValueError("SEALED_PASSED_PRODUCER_AUTHORITY_REQUIRED")
    parent = _json(rows[0]["result"])
    if (parent.get("task_id") != task_id or parent.get("run_signature") != producer.get("run_signature")
            or parent.get("task_terminal_status") != "PASSED" or parent.get("child_exit_code") != 0
            or parent.get("control_ground_truth") is not False
            or parent.get("physical_deployment_authorized") is not False):
        raise ValueError("SEALED_PRODUCER_RESULT_BINDING_DRIFT")
    stages = parent.get("stage_results")
    if not isinstance(stages, list) or not stages:
        raise ValueError("SEALED_PRODUCER_INNER_RESULT_REQUIRED")
    matched = False
    for ref in stages:
        stage = _json(ref)
        if (stage.get("task_id") == task_id and stage.get("run_signature") == producer.get("run_signature")
                and stage.get("production_manifest") == manifest_ref
                and stage.get("execution_completed") is True
                and stage.get("branch_qualification", {}).get(packet.get("branch"), {}).get("status")
                    == "PASSED_NON_DEPLOYABLE_DERIVED_SOURCE"
                and stage.get("control_ground_truth") is False and stage.get("physical_deployment_authorized") is False):
            matched = True
    if not matched:
        raise ValueError("SELECTED_MANIFEST_BRANCH_NOT_PASSED_BY_SEALED_PRODUCER")


def _authority_once(root, packet_path, operation, args, output, *, receipt):
    root = root.resolve(strict=True)
    packet_path = _path(packet_path, root, existing=True)
    if not packet_path.is_relative_to(root / "tasks/current"):
        raise ValueError("REGISTERED_PROJECT_TASK_PACKET_REQUIRED")
    contract = _json(receipt["files"]["algorithm_contract"])
    route = contract.get("special_status_contracts", {}).get(PREPARE_ROUTE if operation == PREPARE else ROUTE)
    if (not isinstance(route, dict) or route.get("candidate_operation") != operation
            or route.get("task_packet_required") is not True or route.get("training_scope") != TRAINING_SCOPE
            or route.get("supervision_class") != "DERIVED_RETARGET"
            or route.get("control_ground_truth") is not False
            or route.get("physical_deployment_authorized") is not False):
        raise ValueError("PRODUCTION_CANDIDATE_CONTRACT_NOT_REGISTERED")
    ref = file_ref(packet_path)
    packet = _json(ref)
    mode = PREPARE_MODE if operation == PREPARE else EXECUTION_MODE
    if packet.get("execution_mode") != mode or packet.get("training_scope") != TRAINING_SCOPE:
        raise ValueError("SUCCESSOR_PRODUCTION_TASK_REQUIRED_NOT_DEVELOPMENT_OR_OLD_PACKET")
    now = datetime.now(timezone.utc).timestamp()
    deadline, started = _utc(packet["deadline_at"]), _utc(packet["execution_started_at"])
    wall = packet.get("budgets", {}).get("wall_seconds_max")
    if type(wall) not in {int, float} or not 0 < wall < float("inf"):
        raise ValueError("FINITE_TASK_WALL_BUDGET_REQUIRED")
    if started > now or min(deadline, started + wall) <= now:
        raise ValueError("FIXED_TASK_WALL_DEADLINE_EXPIRED")
    index = _json(receipt["files"]["task_packet_index_current"])
    allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(allowed) != 1 or allowed[0]["task_id"] != packet["task_id"]
            or allowed[0]["packet_sha256"] != ref["sha256"]
            or (root / allowed[0]["packet_path"]).resolve() != packet_path):
        raise ValueError("UNIQUE_ACTIVE_PRODUCTION_PACKET_REQUIRED")
    state = _json(receipt["files"]["task_state"])
    active = [row for row in state["tasks"] if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    writer = packet["writer"]
    if (len(active) != 1 or active[0]["task_id"] != packet["task_id"] or active[0]["status"] != "RUNNING"
            or active[0].get("run_signature") != packet["run_signature"]
            or writer.get("scope") != packet["task_id"] or active[0].get("pid") != writer.get("pid")
            or active[0].get("proc_start_ticks") != writer.get("proc_start_ticks") or not _is_descendant(writer)):
        raise ValueError("LIVE_WRITER_ANCESTRY_REQUIRED")
    if _json(file_ref(_path(Path(packet["writer_file"]), root, existing=True))) != writer:
        raise ValueError("WRITER_FENCING_TOKEN_DRIFT")
    expected, matched = [operation, *args], False
    for command in packet.get("commands", []):
        tokens = shlex.split(command)
        for i in range(len(tokens) - 2):
            if tokens[i:i + 3] == ["-m", "chaoyang.cli", "run"]:
                matched |= tokens[i + 3:] == expected
    if not matched:
        raise ValueError("EXACT_PRODUCTION_COMMAND_NOT_PINNED")
    refs, closure = packet.get("runtime_code", []), route.get("code_closure", [])
    if not refs or not closure or any(ref not in refs for ref in closure):
        raise ValueError("TASK_ROUTE_CODE_CLOSURE_MISMATCH")
    for ref in refs:
        verify_ref(ref)
    if operation == OPERATION:
        _sealed_producer(root, packet, state, index)
    if _path(output, root) != _path(Path(packet["output_dir"]), root):
        raise ValueError("TASK_OUTPUT_BINDING_DRIFT")
    packet["admitted_authorization_expires_at"] = min(deadline, started + wall)
    return packet


def _authority(root, packet_path, operation, args, output):
    """Validate one snapshot under the publisher's existing shared read lock."""
    root = root.resolve(strict=True)
    lock_path = _path(root / "_run/current/locks/governance.lock", root, existing=True)
    # No mkdir/open-create and no drift retry. publish_bundle holds LOCK_EX on
    # this same inode while replacing all receipt-bound governance documents.
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            receipt = json.loads((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_bytes())
            return _authority_once(root, packet_path, operation, args, output, receipt=receipt)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def authorize_production_prepare(root: Path, packet_path: Path, args: list[str], *, output: Path) -> dict:
    packet = _authority(root, packet_path, PREPARE, args, output)
    budget = packet["budgets"]
    if any(budget.get(name) != 0 for name in ("gpu_count", "model_calls", "training_steps")):
        raise ValueError("CPU_SOURCE_PREPARATION_ONLY")
    for role in ("resolved_input_manifest", "recipe"):
        _json(packet["inputs"][role])
    contract = _json(packet["inputs"]["source_contract"])
    if contract.get("input_manifest_ref") != packet["inputs"]["resolved_input_manifest"]:
        raise ValueError("SOURCE_CONTRACT_ORIGINAL_MANIFEST_BINDING_DRIFT")
    refs = packet["inputs"].get("legacy_mapper_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("FROZEN_LEGACY_MAPPER_REFS_REQUIRED")
    for ref in refs:
        verify_ref(ref)
    return packet


def authorize_production_training(root: Path, packet_path: Path, args: list[str], *, manifest: Path,
        config: Path, pretrained: Path, output: Path, branch: str, updates: int, world_size: int) -> dict:
    packet = _authority(root, packet_path, OPERATION, args, output)
    inputs = packet["inputs"]
    for role, path in (("production_manifest", manifest), ("training_config", config), ("pretrained", pretrained)):
        if verify_ref(inputs[role]) != path.resolve(strict=True):
            raise ValueError("TRAINING_INPUT_BINDING_DRIFT:" + role)
    _json(inputs["pretrained_config"])
    envelope = _json(inputs["production_manifest"])
    if (envelope.get("schema") != SCHEMA or envelope.get("training_scope") != TRAINING_SCOPE
            or envelope.get("supervision_class") != "DERIVED_RETARGET" or envelope.get("control_ground_truth") is not False
            or envelope.get("physical_deployment_authorized") is not False or envelope.get("fixture_only") is True):
        raise ValueError("FULL_RAW_REPLAY_PRODUCTION_SOURCE_REQUIRED")
    if inputs.get("prepared_pair") != inputs["production_manifest"]:
        raise ValueError("PREPARED_PAIR_SOURCE_BINDING_DRIFT")
    verify_ref(inputs["prepared_pair"])
    if branch not in {"raw", "robotized"} or packet.get("branch") != branch:
        raise ValueError("ONE_BRANCH_PER_TASK_REQUIRED")
    budget = packet["budgets"]
    if (type(updates) is not int or not 1 <= updates <= 20000 or type(world_size) is not int or world_size not in {1, 8}
            or type(budget.get("optimizer_updates_max")) is not int or not updates <= budget["optimizer_updates_max"] <= 20000
            or budget.get("gpu_count") != world_size or packet.get("gpu_lease_verified") is not True):
        raise ValueError("TRAINING_BUDGET_OR_GPU_LEASE_NOT_BOUND")
    lease = _json(file_ref(_path(Path(packet["gpu_lease_path"]), root, existing=True)))
    ids, writer = packet.get("allocated_gpu_ids", []), packet["writer"]
    expiry = _utc(lease["expires_at"])
    if (len(ids) != world_size or len(set(ids)) != world_size or any(type(i) is not int or i < 0 for i in ids)
            or lease.get("status") != "ACQUIRED" or lease.get("task_id") != packet["task_id"]
            or lease.get("hostname") != socket.gethostname() or lease.get("pid") != writer["pid"]
            or lease.get("process_startticks") != writer["proc_start_ticks"]
            or lease.get("fencing_token") != writer["fencing_token"] or lease.get("executor_epoch") != writer["executor_epoch"]
            or lease.get("gpu_id") != ids[0] or expiry <= datetime.now(timezone.utc).timestamp()):
        raise ValueError("LIVE_GPU_LEASE_REQUIRED")
    if world_size > 1 and lease.get("allocated_gpu_ids") != ids:
        raise ValueError("ALL_GPUS_MUST_BE_COVERED_BY_LEASE")
    packet["admitted_authorization_expires_at"] = min(packet["admitted_authorization_expires_at"], expiry)
    return packet
