"""CPU-only admission for two lanes owned by one fenced Joint26 coordinator."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import socket

from chaoyang.governance.common import canonical_bytes

from .joint26_contract import PARALLEL_MODE, PARALLEL_ROUTE, TRAIN
from .joint26_parallel_resources import STEP_GUARD_SCHEMA, validate_step_guard
from . import joint26_runtime as runtime


def lane_signature(lane: dict) -> str:
    """Sign all immutable lane fields; renewable lease bytes are not inputs."""
    return hashlib.sha256(canonical_bytes({
        key: value for key, value in lane.items() if key != "lane_signature"
    })).hexdigest()


def _expiry(value, code):
    runtime._require(isinstance(value, str), code)
    expiry = datetime.fromisoformat(value)
    runtime._require(expiry.tzinfo is not None, code)
    return expiry.astimezone(timezone.utc).timestamp()


def _lease_deadline(lease, packet, gpu_id):
    writer = packet["writer"]
    runtime._require(
        lease.get("status") == "ACQUIRED"
        and lease.get("task_id") == packet["task_id"]
        and lease.get("hostname") == socket.gethostname()
        and lease.get("pid") == writer["pid"]
        and lease.get("process_startticks") == writer["proc_start_ticks"]
        and lease.get("executor_epoch") == writer["executor_epoch"]
        and lease.get("fencing_token") == writer["fencing_token"]
        and type(lease.get("gpu_id")) is int and lease["gpu_id"] == gpu_id,
        "JOINT26_PARALLEL_LIVE_GPU_LEASE_REQUIRED",
    )
    return _expiry(lease.get("expires_at"), "JOINT26_PARALLEL_GPU_LEASE_EXPIRY_REQUIRED")


def _snapshot(packet_path):
    """Copy the receipt, ownership and lease metadata in one short read lock."""
    root = runtime.ROOT
    lock_path = root / "_run/current/locks/governance.lock"
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            packet = json.loads(packet_path.read_text())
            receipt = json.loads((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_text())
            contract = runtime._load_ref(receipt["files"]["algorithm_contract"])
            state = runtime._load_ref(receipt["files"]["task_state"])
            index = runtime._load_ref(receipt["files"]["task_packet_index_current"])
            allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
            runtime._require(
                len(allowed) == 1 and allowed[0]["task_id"] == packet["task_id"]
                and allowed[0]["packet_sha256"] == runtime.file_ref(packet_path)["sha256"]
                and (root / allowed[0]["packet_path"]).resolve() == packet_path,
                "JOINT26_PARALLEL_UNIQUE_COORDINATOR_PACKET_REQUIRED",
            )
            active = [row for row in state["tasks"]
                      if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
            writer = packet["writer"]
            runtime._require(
                len(active) == 1 and active[0].get("status") == "RUNNING"
                and active[0]["task_id"] == packet["task_id"]
                and active[0].get("run_signature") == packet["run_signature"]
                and active[0].get("pid") == writer["pid"]
                and active[0].get("proc_start_ticks") == writer["proc_start_ticks"],
                "JOINT26_PARALLEL_LIVE_COORDINATOR_STATE_REQUIRED",
            )
            runtime._require(
                json.loads(runtime._ordinary(packet["writer_file"]).read_text()) == writer,
                "WRITER_FENCING_TOKEN_DRIFT",
            )
            runtime._require(packet.get("execution_mode") == PARALLEL_MODE
                             and packet.get("route_name") == PARALLEL_ROUTE,
                             "JOINT26_PARALLEL_TRAINING_TASK_REQUIRED")
            lanes = packet.get("lanes")
            runtime._require(isinstance(lanes, dict) and set(lanes) == {"raw", "robotized"},
                             "JOINT26_PARALLEL_EXACT_TWO_LANES_REQUIRED")
            guard_path = runtime._ordinary(packet["gpu_lease_path"])
            runtime._require(guard_path == root / "_run/current/GPU_LEASE.json",
                             "JOINT26_PARALLEL_CENTRAL_GUARD_REQUIRED")
            guard = json.loads(guard_path.read_text())
            leases = {}
            for branch, lane in lanes.items():
                runtime._require(isinstance(lane, dict) and lane.get("branch") == branch,
                                 "JOINT26_PARALLEL_LANE_BRANCH_DRIFT")
                gpu_ids, paths = lane.get("gpu_ids"), lane.get("gpu_lease_paths")
                runtime._require(
                    isinstance(gpu_ids, list) and bool(gpu_ids)
                    and all(type(gpu_id) is int and gpu_id >= 0 for gpu_id in gpu_ids)
                    and len(set(gpu_ids)) == len(gpu_ids)
                    and isinstance(paths, list) and len(paths) == len(gpu_ids),
                    "JOINT26_PARALLEL_EXACT_GPU_GROUP_REQUIRED",
                )
                leases[branch] = []
                for gpu_id, value in zip(gpu_ids, paths):
                    path = runtime._ordinary(value)
                    runtime._require(path == root / f"_run/current/ICT_GPU_LEASES/gpu_{gpu_id}.json",
                                     "JOINT26_PARALLEL_CARD_LEASE_PATH_REQUIRED")
                    leases[branch].append(json.loads(path.read_text()))
            return packet, contract, guard, leases
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def authorize_parallel_training(task_packet, manifest_path, config_path, output_dir,
                                branch, world_size, rank=0) -> dict:
    """Admit one lane while preserving the single global task/writer invariant."""
    runtime._require(
        branch in {"raw", "robotized"} and type(world_size) is int and world_size > 0
        and type(rank) is int and 0 <= rank < world_size,
        "EXPLICIT_DDP_TOPOLOGY_REQUIRED",
    )
    packet_path = runtime._ordinary(task_packet)
    runtime._require(packet_path.is_relative_to(runtime.ROOT / "tasks/current"),
                     "REGISTERED_TASK_PACKET_REQUIRED")
    packet, contract, guard, leases = _snapshot(packet_path)
    runtime._require(
        packet.get("run_signature") == hashlib.sha256(canonical_bytes({
            key: value for key, value in packet.items() if key != "run_signature"
        })).hexdigest(), "JOINT26_PARALLEL_COORDINATOR_SIGNATURE_DRIFT",
    )
    route = contract.get("special_status_contracts", {}).get(PARALLEL_ROUTE)
    runtime._require(
        isinstance(route, dict) and route.get("candidate_operation") == TRAIN
        and route.get("task_packet_required") is True
        and route.get("physical_deployment_authorized") is False,
        "JOINT26_PARALLEL_ROUTE_NOT_REGISTERED",
    )
    runtime._require(packet.get("control_ground_truth") is False
                     and packet.get("physical_deployment_authorized") is False,
                     "DERIVED_RESEARCH_FLAGS_REQUIRED")
    runtime._require(runtime._is_descendant(packet["writer"]), "LIVE_WRITER_ANCESTRY_REQUIRED")
    refs, closure = packet.get("runtime_code", []), route.get("code_closure", [])
    runtime._require(bool(refs) and bool(closure), "JOINT26_FROZEN_CLOSURE_REQUIRED")
    def identity(ref):
        local = runtime.normalize_ref(ref, runtime.ROOT)
        return tuple(local[key] for key in ("path", "bytes", "sha256"))
    identities = {identity(ref) for ref in refs}
    runtime._require(all(identity(ref) in identities for ref in closure),
                     "JOINT26_FROZEN_CLOSURE_REQUIRED")
    for ref in refs:
        runtime._ordinary(ref["path"])
        runtime._verify_project_ref(ref)

    lanes = packet["lanes"]
    runtime._require(not set(lanes["raw"]["gpu_ids"]) & set(lanes["robotized"]["gpu_ids"]),
                     "JOINT26_PARALLEL_GPU_GROUP_OVERLAP")
    outputs, expected_commands = {}, []
    deadlines = []
    step_only = packet.get("deadline_at") is None
    if step_only:
        parent_budget = packet.get("budgets", {})
        runtime._require(
            "deadline_at" in packet and packet.get("termination_policy") == "OPTIMIZER_UPDATES_ONLY"
            and "wall_seconds_max" in parent_budget and parent_budget["wall_seconds_max"] is None
            and type(parent_budget.get("optimizer_updates_max")) is int
            and parent_budget["optimizer_updates_max"] > 0
            and type(parent_budget.get("model_calls")) is int and parent_budget["model_calls"] > 0,
            "JOINT26_STEP_BOUNDED_SIGNED_TASK_REQUIRED",
        )
        runtime._require(guard.get("schema_version") == STEP_GUARD_SCHEMA,
                         "JOINT26_PARALLEL_STEP_GUARD_REQUIRED")
        guard_groups = validate_step_guard(guard)
        runtime._require(guard_groups == {name: lane["gpu_ids"] for name, lane in lanes.items()},
                         "JOINT26_PARALLEL_STEP_GUARD_GPU_GROUP_DRIFT")
    else:
        runtime._require(packet.get("termination_policy") in {None, "OPTIMIZER_UPDATES_OR_DEADLINE"},
                         "JOINT26_TERMINATION_POLICY_DEADLINE_DRIFT")
        deadlines.append(_expiry(packet.get("deadline_at"), "JOINT26_TASK_DEADLINE_REQUIRED"))
    deadlines.append(_lease_deadline(guard, packet, lanes["raw"]["gpu_ids"][0]))
    for name, lane in lanes.items():
        runtime._require(lane.get("lane_signature") == lane_signature(lane),
                         "JOINT26_PARALLEL_LANE_SIGNATURE_DRIFT")
        inputs = lane.get("inputs", {})
        runtime._require(isinstance(inputs, dict)
                         and {"paired_manifest", "training_config"} <= set(inputs),
                         "JOINT26_PARALLEL_LANE_INPUTS_REQUIRED")
        for ref in inputs.values():
            runtime._ordinary(ref["path"])
            runtime._verify_project_ref(ref)
        configuration = runtime._load_ref(inputs["training_config"])
        requested = configuration.get("training", {}).get("max_updates")
        budgets = lane.get("budgets", {})
        runtime._require(
            configuration.get("schema") == "humanego-ict-joint26-training-config-v1"
            and type(requested) is int and requested > 0
            and type(budgets.get("optimizer_updates_max")) is int
            and budgets["optimizer_updates_max"] == requested
            and type(budgets.get("gpu_count")) is int
            and budgets["gpu_count"] == len(lane["gpu_ids"]),
            "JOINT26_UPDATE_OR_DDP_BUDGET_DRIFT",
        )
        if step_only:
            training = configuration.get("training", {})
            runtime._require(
                "wall_seconds_max" in budgets and budgets["wall_seconds_max"] is None
                and "wall_seconds_max" in training and training["wall_seconds_max"] is None
                and type(budgets.get("model_calls")) is int and budgets["model_calls"] > 0,
                "JOINT26_STEP_BOUNDED_LANE_WALL_BUDGET_DRIFT",
            )
        warmstart = configuration.get("warmstart_ref")
        if warmstart is not None:
            runtime._require(inputs.get("warmstart") is not None
                             and runtime.reference_equal(warmstart, inputs["warmstart"], runtime.ROOT),
                             "WARMSTART_NOT_TASK_BOUND")
        outputs[name] = runtime._ordinary(lane["output_dir"], existing=False)
        arguments = ["train", "--task-packet", str(packet_path),
                     "--manifest", str(runtime._ordinary(inputs["paired_manifest"]["path"])),
                     "--config", str(runtime._ordinary(inputs["training_config"]["path"])),
                     "--output", str(outputs[name]), "--branch", name]
        commands = lane.get("commands", [])
        runtime._require(isinstance(commands, list) and len(commands) == 1
                         and runtime._command_is_pinned(commands, TRAIN, arguments),
                         "EXACT_JOINT26_COMMAND_NOT_PINNED")
        expected_commands.extend(commands)
        for gpu_id, lease in zip(lane["gpu_ids"], leases[name]):
            if step_only:
                runtime._require(lease.get("schema_version") == "chaoyang-gpu-lease-v71",
                                 "JOINT26_PARALLEL_V71_CARD_LEASE_REQUIRED")
            deadlines.append(_lease_deadline(lease, packet, gpu_id))
    if step_only:
        runtime._require(packet["budgets"]["optimizer_updates_max"]
                         == sum(lane["budgets"]["optimizer_updates_max"] for lane in lanes.values()),
                         "JOINT26_STEP_BOUNDED_PARENT_UPDATE_BUDGET_DRIFT")
    runtime._require(
        not outputs["raw"].is_relative_to(outputs["robotized"])
        and not outputs["robotized"].is_relative_to(outputs["raw"]),
        "JOINT26_PARALLEL_OUTPUT_OVERLAP",
    )
    runtime._require(packet.get("commands") == expected_commands,
                     "JOINT26_PARALLEL_PARENT_COMMANDS_DRIFT")
    lane = lanes[branch]
    runtime._require(world_size == len(lane["gpu_ids"]), "JOINT26_UPDATE_OR_DDP_BUDGET_DRIFT")
    runtime._require(runtime._ordinary(manifest_path) == runtime._ordinary(lane["inputs"]["paired_manifest"]["path"])
                     and runtime._ordinary(config_path) == runtime._ordinary(lane["inputs"]["training_config"]["path"]),
                     "JOINT26_INPUT_BINDING_DRIFT")
    runtime._require(runtime._ordinary(output_dir, existing=False) == outputs[branch],
                     "JOINT26_OUTPUT_BINDING_DRIFT")
    runtime._require(min(deadlines) > datetime.now(timezone.utc).timestamp(),
                     "JOINT26_PARALLEL_AUTHORIZATION_EXPIRED")
    return dict(packet, inputs=lane["inputs"], output_dir=str(outputs[branch]),
                budgets=lane["budgets"], branch=branch, lane_signature=lane["lane_signature"],
                allocated_gpu_ids=lane["gpu_ids"], gpu_lease_paths=lane["gpu_lease_paths"],
                coordinator_gpu_lease_path=packet["gpu_lease_path"],
                gpu_lease_path=lane["gpu_lease_paths"][0], gpu_lease_verified=True,
                admitted_authorization_expires_at=min(deadlines))
