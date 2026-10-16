"""Fail-closed candidate training admission before importing torch/models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
from datetime import datetime, timezone
import socket

OPERATION = "train_pick_place_humanego_ict_revo2_candidate_v1"
ROUTE = "pick_place_humanego_ict_revo2_candidate_v1"


def file_ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_ref(ref: dict) -> Path:
    if not isinstance(ref, dict) or set(("path", "bytes", "sha256")) - ref.keys():
        raise ValueError("ARTIFACT_REFERENCE_INCOMPLETE")
    path = Path(ref["path"])
    if not path.is_absolute() or file_ref(path) != {key: ref[key] for key in ("path", "bytes", "sha256")}:
        raise ValueError("ARTIFACT_SHA_DRIFT:" + str(path))
    return path.resolve(strict=True)


def project_output(path: Path, root: Path) -> Path:
    target, root = path.resolve(), root.resolve(strict=True)
    if target == root or not target.is_relative_to(root):
        raise ValueError("OUTPUT_MUST_BE_A_PROJECT_SUBDIRECTORY")
    return target


def _identity(pid: int) -> tuple[int, int]:
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    return int(fields[1]), int(fields[19])


def _is_descendant(writer: dict) -> bool:
    if _identity(int(writer["pid"]))[1] != int(writer["proc_start_ticks"]):
        return False
    # procfs may be mounted from the host while os.getpid() is container-local.
    pid = int(Path("/proc/self/stat").read_text().split(" ", 1)[0])
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == writer["pid"]:
            return True
        seen.add(pid)
        pid, _ = _identity(pid)
    return False


def authorize_training(root: Path, packet_path: Path, operation_args: list[str],
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
    if route.get("task_packet_required") is not True or route.get("physical_deployment_authorized") is not False:
        raise ValueError("ICT_CANDIDATE_CONTRACT_INCOMPLETE")
    packet = json.loads(packet_path.read_text())
    if packet.get("execution_mode") != "HUMANEGO_ICT_REVO2_CANDIDATE_TRAINING":
        raise ValueError("TRAINING_TASK_REQUIRED_NOT_DEVELOPMENT_PACKET")
    if datetime.fromisoformat(packet["deadline_at"]).astimezone(timezone.utc) <= datetime.now(timezone.utc):
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
    if not _is_descendant(writer):
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
    if (updates > packet["budgets"]["optimizer_updates_max"]
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
            or datetime.fromisoformat(lease["expires_at"]).astimezone(timezone.utc) <= datetime.now(timezone.utc)):
        raise ValueError("LIVE_GPU_LEASE_REQUIRED")
    packet["admitted_authorization_expires_at"] = min(
        datetime.fromisoformat(packet["deadline_at"]).timestamp(),
        datetime.fromisoformat(lease["expires_at"]).timestamp(),
    )
    return packet


def bind_cuda_visible_devices(packet: dict) -> str:
    """Map LOCAL_RANK to exactly the leased host GPU indices before torch import."""
    values = packet["allocated_gpu_ids"]
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("LEASED_GPU_INDEX_MAPPING_REQUIRED")
    expected = ",".join(map(str, values))
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None and configured != expected:
        raise ValueError("CUDA_VISIBLE_DEVICES_DIFFERS_FROM_GPU_LEASE")
    os.environ["CUDA_VISIBLE_DEVICES"] = expected
    return expected


def runtime_cache_paths(output: Path, root: Path) -> list[Path]:
    output = project_output(output, root)
    paths = [output / "runtime_cache" / name for name in ("tmp", "torch", "inductor", "triton", "xdg")]
    if any(not path.resolve().is_relative_to(output) for path in paths):
        raise ValueError("RUNTIME_CACHE_ESCAPES_TASK_OUTPUT")
    return paths


def storage_ready(path: Path, minimum_bytes: int) -> None:
    parent = path.resolve()
    while not parent.exists():
        parent = parent.parent
    stat = os.statvfs(parent)
    if stat.f_bavail * stat.f_frsize < minimum_bytes:
        raise ValueError("BLOCKED_OUTPUT_STORAGE_CAPACITY")
