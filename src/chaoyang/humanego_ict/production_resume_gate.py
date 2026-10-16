"""Independent admission for complete, sealed derived ICT state successors.

No model, torch or SDK is imported during admission.  This route cannot change
old training routes, numerical gates, source qualification or physical scope.
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
from .production_training_gate import (
    SCHEMA, TRAINING_SCOPE, EXECUTION_MODE as PARENT_EXECUTION_MODE,
    _sealed_producer, _utc,
)
from .runtime_gate import file_ref, verify_ref, _is_descendant, storage_ready as _statvfs_ready

OPERATION = "train_pick_place_humanego_ict_production_resume_v1"
ROUTE = "humanego_ict_production_resume_v1"
EXECUTION_MODE = "HUMANEGO_ICT_PRODUCTION_DERIVED_RESUME"
CONFIG_SCHEMA = "pick-place-humanego-ict-production-resume-config-v1"
INITIALIZATION = "RESUME_COMPLETE_TRAINING_STATE"
PARENT_INPUT_ROLES = {
    "task_packet": "parent_task_packet",
    "outer_result": "parent_task_result",
    "inner_result": "parent_inner_result",
    "training_config": "parent_training_config",
    "checkpoint": "resume_checkpoint",
    "checkpoint_sidecar": "resume_checkpoint_sidecar",
}
ENGINEERING_CODE_PATHS = (
    "src/chaoyang/humanego_ict/production_resume.py",
    "src/chaoyang/humanego_ict/production_resume_gate.py",
    "src/chaoyang/humanego_ict/production_resume_cli.py",
    "src/chaoyang/humanego_ict/production_resume_contract.py",
    "src/chaoyang/ops/train_pick_place_humanego_ict_production_resume_v1.py",
    "scripts/humanego_ict/production_resume_runner_v1.py",
)


def parent_refs_from_packet(packet: dict) -> dict:
    inputs = packet.get("inputs", {})
    if any(not isinstance(inputs.get(role), dict) for role in PARENT_INPUT_ROLES.values()):
        raise ValueError("COMPLETE_PARENT_RESUME_REFERENCES_REQUIRED")
    return {role: inputs[name] for role, name in PARENT_INPUT_ROLES.items()}


def validate_resume_config(config: dict, world: int) -> None:
    """Project to the unchanged old numerical validator, not the old task gate."""
    from .production_training_cli import validate_config as original_validate
    if (config.get("schema_version") != CONFIG_SCHEMA
            or config.get("initialization") != INITIALIZATION
            or not isinstance(config.get("resume_parent_bindings"), dict)
            or set(config["resume_parent_bindings"]) != set(PARENT_INPUT_ROLES)):
        raise ValueError("EXPLICIT_COMPLETE_STATE_RESUME_CONFIG_REQUIRED")
    projected = dict(config, schema_version="pick-place-humanego-ict-production-training-config-v1",
                     initialization="FRESH_SHARED_PRETRAINED_NEW_IO")
    projected.pop("resume_parent_bindings")
    original_validate(projected, world)


def _recipe(config: dict) -> dict:
    value = json.loads(json.dumps(config))
    for field in ("schema_version", "initialization", "resume_parent_bindings"):
        value.pop(field, None)
    value.get("training", {}).pop("wall_seconds_max", None)
    return value


def _sealed_engineering(root: Path, packet: dict, state: dict, index: dict) -> dict:
    """Only the sealed CPU engineering package may enable the new operation."""
    root = root.resolve(strict=True)
    inputs = packet.get("inputs", {})
    packet_ref, result_ref = inputs.get("engineering_task_packet"), inputs.get("engineering_task_result")
    if not isinstance(packet_ref, dict) or not isinstance(result_ref, dict):
        raise ValueError("SEALED_RESUME_ENGINEERING_REFERENCES_REQUIRED")
    path = _path(Path(packet_ref["path"]), root, existing=True)
    _path(Path(result_ref["path"]), root, existing=True)
    engineering, result = _json(packet_ref), _json(result_ref)
    task_id = engineering.get("task_id")
    budgets = engineering.get("budgets", {})
    if (task_id == packet.get("task_id") or not path.is_relative_to(root / "tasks/current")
            or engineering.get("execution_mode") != "CPU_DEVELOPMENT_ONLY"
            or any(budgets.get(role) != 0 for role in ("gpu_count", "model_calls", "training_steps"))):
        raise ValueError("SEPARATE_CPU_ONLY_RESUME_ENGINEERING_REQUIRED")
    rows = [row for row in state["tasks"] if row.get("task_id") == task_id]
    pointers = [row for row in index["task_packets"] if row.get("task_id") == task_id]
    if (len(rows) != 1 or rows[0].get("status") != "PASSED"
            or rows[0].get("task_packet") != packet_ref or rows[0].get("result") != result_ref
            or rows[0].get("run_signature") != engineering.get("run_signature")
            or len(pointers) != 1 or pointers[0].get("execution_allowed") is not False
            or pointers[0].get("packet_sha256") != packet_ref["sha256"]
            or (root / pointers[0]["packet_path"]).resolve() != path):
        raise ValueError("SEALED_PASSED_RESUME_ENGINEERING_AUTHORITY_REQUIRED")
    if (result.get("schema") != "humanego-ict-checkpoint-continuation-cpu-result-v1"
            or result.get("task_id") != task_id or result.get("run_signature") != engineering.get("run_signature")
            or result.get("task_terminal_status") != "PASSED"):
        raise ValueError("RESUME_ENGINEERING_RESULT_BINDING_DRIFT")
    refs = result.get("resume_code_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("SEALED_RESUME_CODE_AND_TEST_REFS_REQUIRED")
    by_path = {}
    for ref in refs:
        resolved = _path(verify_ref(ref), root, existing=True)
        if str(resolved) in by_path:
            raise ValueError("DUPLICATE_RESUME_ENGINEERING_CODE_REFERENCE")
        by_path[str(resolved)] = ref
    for relative in ENGINEERING_CODE_PATHS:
        ref = by_path.get(str(root / relative))
        if ref is None or ref not in packet.get("runtime_code", []):
            raise ValueError("RESUME_ENGINEERING_CODE_NOT_IN_RUNTIME_CLOSURE:" + relative)
    if not any(Path(path).is_relative_to(root / "tests") for path in by_path):
        raise ValueError("SEALED_RESUME_TEST_FILE_SHA_REQUIRED")
    for role in ("cpu_test_result", "protected_inputs_check"):
        _path(Path(result[role]["path"]), root, existing=True)
    tests = _json(result["cpu_test_result"])
    protected = _json(result["protected_inputs_check"])
    if (tests.get("status") != "PASSED" or tests.get("exit_code") != 0
            or type(tests.get("passed_count")) is not int or tests["passed_count"] < 1):
        raise ValueError("RESUME_CPU_TESTS_NOT_PASSED")
    if (protected.get("status") != "PASSED" or protected.get("sha_unchanged") is not True
            or not isinstance(protected.get("checked_refs"), list) or not protected["checked_refs"]):
        raise ValueError("RESUME_PROTECTED_SOURCE_PROOF_NOT_PASSED")
    for ref in protected["checked_refs"]:
        verify_ref(ref)
    return result


def _sealed_parent(root: Path, packet: dict, state: dict, index: dict, config: dict,
                   updates: int, world: int) -> dict:
    # The adapter's JSON helper has no torch/NumPy import; the later tensor
    # adapter verifies complete optimizer/EMA/RNG and actual runtime environment.
    from .production_resume import validate_parent_evidence
    refs = parent_refs_from_packet(packet)
    for ref in refs.values():
        _path(Path(ref["path"]), root, existing=True)
    evidence = validate_parent_evidence(refs, root=root)
    parent = evidence["packet"]
    parent_path = _path(Path(refs["task_packet"]["path"]), root, existing=True)
    task_id = parent.get("task_id")
    if (task_id == packet["task_id"] or not parent_path.is_relative_to(root / "tasks/current")
            or parent.get("execution_mode") not in {PARENT_EXECUTION_MODE, EXECUTION_MODE}
            or evidence["branch"] != packet.get("branch")):
        raise ValueError("SEPARATE_SAME_BRANCH_PRODUCTION_PARENT_REQUIRED")
    rows = [row for row in state["tasks"] if row.get("task_id") == task_id]
    pointers = [row for row in index["task_packets"] if row.get("task_id") == task_id]
    if (len(rows) != 1 or rows[0].get("status") != "PASSED"
            or rows[0].get("task_packet") != refs["task_packet"]
            or rows[0].get("result") != refs["outer_result"]
            or rows[0].get("run_signature") != parent.get("run_signature")
            or len(pointers) != 1 or pointers[0].get("execution_allowed") is not False
            or pointers[0].get("packet_sha256") != refs["task_packet"]["sha256"]
            or (root / pointers[0]["packet_path"]).resolve() != parent_path):
        raise ValueError("SEALED_PASSED_RESUME_PARENT_AUTHORITY_REQUIRED")
    old_inputs, inputs = parent.get("inputs", {}), packet["inputs"]
    for role in ("production_manifest", "prepared_pair", "pretrained", "pretrained_config"):
        if inputs.get(role) != old_inputs.get(role) or inputs.get(role) is None:
            raise ValueError("RESUME_PARENT_SOURCE_BINDING_DRIFT:" + role)
        verify_ref(inputs[role])
    if config.get("resume_parent_bindings") != refs:
        raise ValueError("RESUME_CONFIG_PARENT_SHA_BINDINGS_DRIFT")
    if _recipe(config) != _recipe(evidence["training_config"]):
        raise ValueError("RESUME_RECIPE_MAY_ONLY_RESET_TASK_WALL_SECONDS")
    old_code, new_code = parent.get("runtime_code", []), packet.get("runtime_code", [])
    if not old_code or any(ref not in new_code for ref in old_code):
        raise ValueError("RESUME_PARENT_CODE_CLOSURE_NOT_PRESERVED")
    for ref in old_code:
        verify_ref(ref)
    inner = evidence["inner_result"]
    step = inner.get("optimizer_updates")
    if (type(step) is not int or not 0 < step < updates
            or inner.get("world_size") != world
            or inner.get("cursor") != step * config["training"]["global_batch"]):
        raise ValueError("RESUME_PARENT_STEP_CURSOR_OR_TOPOLOGY_DRIFT")
    # A sealed normal wall-clock stop is a complete checkpoint even when its
    # update budget was not reached; no unverified partial is accepted.
    packet["admitted_resume_parent"] = {
        "task_id": task_id, "optimizer_updates": step, "cursor": inner["cursor"],
        "checkpoint_ref": refs["checkpoint"], "world_size": world,
    }
    return evidence


def _authority_once(root, packet_path, args, output, *, receipt):
    root = root.resolve(strict=True)
    packet_path = _path(packet_path, root, existing=True)
    if not packet_path.is_relative_to(root / "tasks/current"):
        raise ValueError("REGISTERED_PROJECT_TASK_PACKET_REQUIRED")
    contract = _json(receipt["files"]["algorithm_contract"])
    route = contract.get("special_status_contracts", {}).get(ROUTE)
    if (not isinstance(route, dict) or route.get("candidate_operation") != OPERATION
            or route.get("task_packet_required") is not True or route.get("training_scope") != TRAINING_SCOPE
            or route.get("supervision_class") != "DERIVED_RETARGET"
            or route.get("control_ground_truth") is not False
            or route.get("physical_deployment_authorized") is not False
            or route.get("resume_support") is not True
            or route.get("parent_sealed_checkpoint_required") is not True):
        raise ValueError("PRODUCTION_RESUME_CANDIDATE_CONTRACT_NOT_REGISTERED")
    ref = file_ref(packet_path)
    packet = _json(ref)
    if packet.get("execution_mode") != EXECUTION_MODE or packet.get("training_scope") != TRAINING_SCOPE:
        raise ValueError("SUCCESSOR_RESUME_TASK_REQUIRED_NOT_OLD_PACKET")
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
        raise ValueError("UNIQUE_ACTIVE_PRODUCTION_RESUME_PACKET_REQUIRED")
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
    expected, matched = [OPERATION, *args], False
    for command in packet.get("commands", []):
        tokens = shlex.split(command)
        for i in range(len(tokens) - 2):
            if tokens[i:i + 3] == ["-m", "chaoyang.cli", "run"]:
                matched |= tokens[i + 3:] == expected
    if not matched:
        raise ValueError("EXACT_PRODUCTION_RESUME_COMMAND_NOT_PINNED")
    refs, closure = packet.get("runtime_code", []), route.get("code_closure", [])
    if not refs or not closure or any(ref not in refs for ref in closure):
        raise ValueError("TASK_ROUTE_CODE_CLOSURE_MISMATCH")
    for ref in refs:
        verify_ref(ref)
    _sealed_engineering(root, packet, state, index)
    _sealed_producer(root, packet, state, index)
    if _path(output, root) != _path(Path(packet["output_dir"]), root):
        raise ValueError("TASK_OUTPUT_BINDING_DRIFT")
    packet["admitted_authorization_expires_at"] = min(deadline, started + wall)
    return packet, state, index


def _authority(root, packet_path, args, output):
    root = root.resolve(strict=True)
    lock_path = _path(root / "_run/current/locks/governance.lock", root, existing=True)
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            receipt = json.loads((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_bytes())
            return _authority_once(root, packet_path, args, output, receipt=receipt)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def authorize_production_resume(root: Path, packet_path: Path, args: list[str], *, manifest: Path,
        config: Path, checkpoint: Path, output: Path, branch: str, updates: int, world_size: int) -> dict:
    packet, state, index = _authority(root, packet_path, args, output)
    inputs = packet["inputs"]
    for role, path in (("production_manifest", manifest), ("training_config", config), ("resume_checkpoint", checkpoint)):
        if verify_ref(inputs[role]) != path.resolve(strict=True):
            raise ValueError("RESUME_TRAINING_INPUT_BINDING_DRIFT:" + role)
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
    value = _json(inputs["training_config"])
    validate_resume_config(value, world_size)
    _sealed_parent(root.resolve(strict=True), packet, state, index, value, updates, world_size)
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


def production_resume_storage_ready(output: Path, minimum_bytes: int, *, packet: dict, root: Path) -> dict:
    """Same four-GiB, one-hour, same-device policy; only the new mode is admitted."""
    if type(minimum_bytes) is not int or minimum_bytes < 4 * 1024**3:
        raise ValueError("MINIMUM_CHECKPOINT_CAPACITY_NOT_LOWERED")
    if packet.get("execution_mode") != EXECUTION_MODE:
        raise ValueError("MEASURED_CAPACITY_FALLBACK_ONLY_NEW_RESUME_ROUTE")
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
