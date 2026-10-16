"""Separately fenced derived training; no change to REAL_CONTROL admission."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import socket
from datetime import datetime, timezone

from .runtime_gate import (
    file_ref, verify_ref, project_output, _is_descendant,
    bind_cuda_visible_devices, runtime_cache_paths, storage_ready,
)

OPERATION = "train_pick_place_humanego_ict_derived_v1"
ROUTE = "pick_place_humanego_ict_derived_training_v1"


def authorize_derived_training(root: Path, packet_path: Path, operation_args: list[str],
                       *, manifest: Path, config: Path, pretrained: Path,
                       output: Path, updates: int, world_size: int) -> dict:
    """No default bypass, no auto-registration, and no implicit model import."""
    root = root.resolve(strict=True)
    packet_path = packet_path.resolve(strict=True)
    if not packet_path.is_relative_to(root / "tasks/current"):
        raise ValueError("REGISTERED_PROJECT_TASK_PACKET_REQUIRED")
    receipt = json.loads((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_text())
    contract = json.loads(verify_ref(receipt["files"]["algorithm_contract"]).read_text())
    route = contract.get("special_status_contracts", {}).get(ROUTE)
    if not isinstance(route, dict) or route.get("candidate_operation") != OPERATION:
        raise ValueError("ICT_CANDIDATE_OPERATION_NOT_REGISTERED")
    if (route.get("task_packet_required") is not True or route.get("physical_deployment_authorized") is not False
            or route.get("supervision_class") != "DERIVED_RETARGET"
            or route.get("control_ground_truth") is not False):
        raise ValueError("ICT_CANDIDATE_CONTRACT_INCOMPLETE")
    packet = json.loads(packet_path.read_text())
    if packet.get("execution_mode") != "HUMANEGO_ICT_DERIVED_RETARGET_TRAINING":
        raise ValueError("TRAINING_TASK_REQUIRED_NOT_DEVELOPMENT_PACKET")
    if (datetime.fromisoformat(packet["deadline_at"]).tzinfo is None
            or datetime.fromisoformat(packet["deadline_at"]).astimezone(timezone.utc) <= datetime.now(timezone.utc)):
        raise ValueError("TRAINING_DEADLINE_EXPIRED")
    index = json.loads(verify_ref(receipt["files"]["task_packet_index_current"]).read_text())
    allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(allowed) != 1 or allowed[0]["task_id"] != packet["task_id"]
            or allowed[0]["packet_sha256"] != file_ref(packet_path)["sha256"]
            or (root / allowed[0]["packet_path"]).resolve() != packet_path):
        raise ValueError("UNIQUE_ACTIVE_TRAINING_PACKET_REQUIRED")
    state = json.loads(verify_ref(receipt["files"]["task_state"]).read_text())
    active = [row for row in state["tasks"] if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    if (len(active) != 1 or active[0]["task_id"] != packet["task_id"]
            or active[0]["status"] != "RUNNING" or active[0].get("run_signature") != packet["run_signature"]):
        raise ValueError("LIVE_TRAINING_STATE_REQUIRED")
    writer = packet["writer"]
    if (writer.get("scope") != packet["task_id"]
            or active[0].get("pid") != writer.get("pid")
            or active[0].get("proc_start_ticks") != writer.get("proc_start_ticks")
            or not _is_descendant(writer)):
        raise ValueError("LIVE_WRITER_ANCESTRY_REQUIRED")
    writer_path = project_output(Path(packet["writer_file"]), root)
    if json.loads(writer_path.read_text()) != writer:
        raise ValueError("WRITER_FENCING_TOKEN_DRIFT")
    expected = [OPERATION, *operation_args]
    commands = packet.get("commands", [])
    matched = False
    for command in commands:
        tokens = shlex.split(command)
        for index_ in range(len(tokens) - 2):
            if tokens[index_:index_ + 3] == ["-m", "chaoyang.cli", "run"]:
                matched |= tokens[index_ + 3:] == expected
    if not matched:
        raise ValueError("EXACT_TRAINING_COMMAND_NOT_PINNED")
    refs = packet.get("runtime_code", [])
    closure = route.get("code_closure", [])
    if not refs or not closure:
        raise ValueError("FROZEN_CODE_CLOSURE_REQUIRED")
    for ref in refs:
        verify_ref(ref)
    if any(ref not in refs for ref in closure):
        raise ValueError("TASK_ROUTE_CODE_CLOSURE_MISMATCH")
    inputs = packet["inputs"]
    for role, path in (("paired_manifest", manifest), ("training_config", config), ("pretrained", pretrained)):
        if verify_ref(inputs[role]) != path.resolve(strict=True):
            raise ValueError("TRAINING_INPUT_BINDING_DRIFT:" + role)
    if output.resolve() != project_output(Path(packet["output_dir"]), root):
        raise ValueError("TASK_OUTPUT_BINDING_DRIFT")
    if (type(updates) is not int or updates < 1
            or type(world_size) is not int or world_size not in {1, 8}
            or type(packet["budgets"]["optimizer_updates_max"]) is not int
            or updates > packet["budgets"]["optimizer_updates_max"]
            or world_size != packet["budgets"]["gpu_count"]
            or packet.get("gpu_lease_verified") is not True):
        raise ValueError("TRAINING_BUDGET_OR_GPU_LEASE_NOT_BOUND")
    lease = json.loads(project_output(Path(packet["gpu_lease_path"]), root).read_text())
    gpu_ids = packet.get("allocated_gpu_ids", [])
    if (len(gpu_ids) != world_size or len(set(gpu_ids)) != world_size
            or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids)
            or lease.get("status") != "ACQUIRED" or lease.get("task_id") != packet["task_id"]
            or lease.get("hostname") != socket.gethostname()
            or lease.get("pid") != writer["pid"]
            or lease.get("process_startticks") != writer["proc_start_ticks"]
            or lease.get("fencing_token") != writer["fencing_token"]
            or lease.get("executor_epoch") != writer["executor_epoch"]
            or lease.get("gpu_id") != gpu_ids[0]
            or datetime.fromisoformat(lease["expires_at"]).tzinfo is None
            or datetime.fromisoformat(lease["expires_at"]).astimezone(timezone.utc) <= datetime.now(timezone.utc)):
        raise ValueError("LIVE_GPU_LEASE_REQUIRED")
    if world_size > 1 and lease.get("allocated_gpu_ids") != gpu_ids:
        raise ValueError("ALL_GPUS_MUST_BE_COVERED_BY_LEASE")
    wall_budget = packet.get("budgets", {}).get("wall_seconds_max")
    if type(wall_budget) not in {int, float} or not 0 < wall_budget < float("inf"):
        raise ValueError("FINITE_TASK_WALL_BUDGET_REQUIRED")
    started = datetime.fromisoformat(packet["execution_started_at"])
    if started.tzinfo is None or started.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise ValueError("FROZEN_EXECUTION_START_TIME_REQUIRED")
    packet["admitted_authorization_expires_at"] = min(
        datetime.fromisoformat(packet["deadline_at"]).timestamp(),
        datetime.fromisoformat(lease["expires_at"]).timestamp(),
        started.timestamp() + wall_budget,
    )
    if packet["admitted_authorization_expires_at"] <= datetime.now(timezone.utc).timestamp():
        raise ValueError("FIXED_TASK_WALL_DEADLINE_EXPIRED")
    return packet
