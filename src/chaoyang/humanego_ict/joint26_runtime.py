"""Standard-library admission for the additive joint26 operations.

No Torch, geometry, hardware or dataset imports before an exact signed task,
live writer, immutable inputs and current route closure have been verified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import socket

from .joint26_contract import (EXECUTION_MODE, PREPARE, PREPARE_MODE,
                               PREPARE_ROUTE, ROUTE, TRAIN)
from .runtime_gate import (_is_descendant, file_ref, project_output, verify_ref)
from .project_mount_paths import (normalize_ref, reference_equal,
                                  resolve_input_path, resolve_output_path)

ROOT = Path(__file__).resolve().parents[3]


def _require(value, code):
    if not value:
        raise ValueError(code)


def preparation_options(profile):
    """Explicit, signature-bound solver/representation; no physical claim."""
    _require(isinstance(profile, dict), "EXPLICIT_PREPARATION_PROFILE_REQUIRED")
    policy = profile.get("current_initialization", {})
    _require(isinstance(policy, dict), "EXPLICIT_STATIC_INITIALIZATION_POLICY_REQUIRED")
    mode = policy.get("mode", "STATIC_NUMERICAL_CURRENT_TARGET_ONLY")
    _require(isinstance(mode, str) and mode in {"STATIC_NUMERICAL_CURRENT_TARGET_ONLY",
                      "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY"},
             "CURRENT_ONLY_STATIC_INITIALIZATION_REQUIRED")
    constants = {"static_position_cost": 100.0,
                 "static_orientation_cost": 11.459155902616466,
                 "static_qp_damping": 1e-6}
    if mode == "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY":
        _require(all(type(policy.get(k)) in {int, float} and policy[k] == v
                     for k, v in constants.items()), "FROZEN_STATIC_TASK_PARAMETERS_REQUIRED")
    else:
        _require(not set(constants) & set(policy), "STATIC_PARAMETERS_REQUIRE_STATIC_FRAME_TASK_MODE")
    collision = profile.get("collision_validation", "ORIGINAL_CAPSULE_V1")
    _require(isinstance(collision, str) and collision in {"ORIGINAL_CAPSULE_V1", "FROZEN_STL_MESH_AUDIT_V1"},
             "EXPLICIT_COLLISION_REPRESENTATION_REQUIRED")
    admission = profile.get("training_admission_policy", "STRICT_MODEL_NUMERICS_V1")
    _require(isinstance(admission, str)
             and admission in {"STRICT_MODEL_NUMERICS_V1", "EGO_DERIVED_STRUCTURAL_V1"},
             "EXPLICIT_TRAINING_ADMISSION_POLICY_REQUIRED")
    if admission == "EGO_DERIVED_STRUCTURAL_V1":
        _require(mode == "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY"
                 and collision == "ORIGINAL_CAPSULE_V1",
                 "EGO_FIXED_RETARGET_RECIPE_NO_MESH_PREREQUISITE")
    return {"current_initialization_mode": mode, "collision_validation": collision,
            "training_admission_policy": admission}


def preparation_budgets(profile):
    """Static optimizer effort, never a relaxation of action-quality gates.

    The original 60/1000 defaults remain unchanged. Expanded effort must be
    explicitly frozen in the input profile and mirrored in its task packet.
    """
    options = preparation_options(profile)
    policy = profile.get("current_initialization", {})
    _require(isinstance(policy, dict)
             and not set(policy) - {"mode", "settle_steps_max", "geometry_calls_per_window_max",
                                    "static_position_cost", "static_orientation_cost", "static_qp_damping"},
             "EXPLICIT_STATIC_INITIALIZATION_POLICY_REQUIRED")
    steps = policy.get("settle_steps_max", 60)
    calls = policy.get("geometry_calls_per_window_max", 1000)
    _require(type(steps) is int and 1 <= steps <= 300,
             "FINITE_STATIC_INITIALIZATION_STEPS_REQUIRED")
    _require(type(calls) is int and 1 <= calls <= 5000,
             "FINITE_STATIC_GEOMETRY_CALL_BUDGET_REQUIRED")
    _require(calls >= 11 * steps + 2, "STATIC_INITIALIZATION_GEOMETRY_BUDGET_INSUFFICIENT")
    if options["training_admission_policy"] == "EGO_DERIVED_STRUCTURAL_V1":
        _require(steps == 60, "EGO_FIXED_CURRENT_COMPUTATION_RECIPE_REQUIRED")
    return {"current_settle_steps_max": steps, "geometry_calls_per_window_max": calls}


def _ordinary(value, *, existing=True):
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts,
             "ABSOLUTE_CANONICAL_PROJECT_PATH_REQUIRED")
    resolver = resolve_input_path if existing else resolve_output_path
    path = resolver(path, ROOT)
    path = project_output(path, ROOT)
    _require(not existing or path.is_file(), "PROJECT_ORDINARY_INPUT_REQUIRED:" + str(path))
    return path


def _verify_project_ref(ref):
    return verify_ref(normalize_ref(ref, ROOT))


def _load_ref(ref):
    path = _ordinary(ref["path"])
    _require(_verify_project_ref(ref) == path, "JOINT26_REFERENCE_PATH_DRIFT")
    return json.loads(path.read_text())


def _command_is_pinned(commands, operation, arguments):
    if operation not in {PREPARE, TRAIN}:
        return False
    expected = [operation, *arguments]
    for command in commands:
        tokens = shlex.split(command)
        for index in range(len(tokens) - 2):
            if (tokens[index] == "-m"
                    and tokens[index + 1] in {"chaoyang.cli", "chaoyang.humanego_ict.joint26_cli"}
                    and tokens[index + 2] == "run"):
                if tokens[index + 3:] == expected:
                    return True
    return False


def _task_deadline(packet, *, phase):
    """Only explicitly signed step-bounded training can omit its wall limit."""
    value = packet.get("deadline_at")
    if value is None:
        budgets = packet.get("budgets", {})
        _require(phase == "train" and "deadline_at" in packet and isinstance(budgets, dict)
                 and packet.get("termination_policy") == "OPTIMIZER_UPDATES_ONLY"
                 and "wall_seconds_max" in budgets and budgets["wall_seconds_max"] is None
                 and type(budgets.get("optimizer_updates_max")) is int
                 and budgets["optimizer_updates_max"] > 0
                 and type(budgets.get("model_calls")) is int and budgets["model_calls"] > 0,
                 "JOINT26_FINITE_STEP_ONLY_TASK_REQUIRED")
        return None
    deadline = datetime.fromisoformat(value).astimezone(timezone.utc)
    _require(deadline > datetime.now(timezone.utc), "JOINT26_TASK_DEADLINE_EXPIRED")
    return deadline.timestamp()


def _authorize_joint26_task(task_packet, *, phase, input_paths, output_dir,
                           operation_args=None):
    """Authorize a real producer/trainer, never a CPU engineering packet."""
    _require(phase in {"prepare", "train"}, "EXPLICIT_JOINT26_PHASE_REQUIRED")
    packet_path = _ordinary(task_packet)
    _require(packet_path.is_relative_to(ROOT / "tasks/current"),
             "REGISTERED_TASK_PACKET_REQUIRED")
    packet = json.loads(packet_path.read_text())
    mode, route_name, operation = ((PREPARE_MODE, PREPARE_ROUTE, PREPARE) if phase == "prepare"
                                  else (EXECUTION_MODE, ROUTE, TRAIN))
    _require(packet.get("execution_mode") == mode,
             "JOINT26_REAL_TASK_REQUIRED_NOT_ENGINEERING")
    deadline = _task_deadline(packet, phase=phase)
    receipt = json.loads((ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_text())
    route = _load_ref(receipt["files"]["algorithm_contract"]).get("special_status_contracts", {}).get(route_name)
    _require(isinstance(route, dict) and route.get("candidate_operation") == operation
             and route.get("task_packet_required") is True
             and route.get("physical_deployment_authorized") is False,
             "JOINT26_ROUTE_NOT_REGISTERED")
    index = _load_ref(receipt["files"]["task_packet_index_current"])
    allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    _require(len(allowed) == 1 and allowed[0]["task_id"] == packet["task_id"]
             and allowed[0]["packet_sha256"] == file_ref(packet_path)["sha256"]
             and (ROOT / allowed[0]["packet_path"]).resolve() == packet_path,
             "UNIQUE_EXECUTION_ALLOWED_PACKET_REQUIRED")
    state = _load_ref(receipt["files"]["task_state"])
    active = [row for row in state["tasks"] if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    writer = packet["writer"]
    _require(len(active) == 1 and active[0].get("status") == "RUNNING"
             and active[0]["task_id"] == packet["task_id"]
             and active[0].get("run_signature") == packet["run_signature"]
             and active[0].get("pid") == writer["pid"]
             and active[0].get("proc_start_ticks") == writer["proc_start_ticks"],
             "LIVE_OWNED_JOINT26_STATE_REQUIRED")
    _require(_is_descendant(writer), "LIVE_WRITER_ANCESTRY_REQUIRED")
    _require(json.loads(_ordinary(packet["writer_file"]).read_text()) == writer,
             "WRITER_FENCING_TOKEN_DRIFT")
    closure = route.get("code_closure", [])
    refs = packet.get("runtime_code", [])
    def identity(ref):
        local = normalize_ref(ref, ROOT)
        return tuple(local[key] for key in ("path", "bytes", "sha256"))
    # Resolve each reference once, not once for every pair in the closure.
    # Full byte/SHA validation of every pinned file still follows below.
    pinned_identities = {identity(ref) for ref in refs}
    _require(bool(closure) and all(identity(ref) in pinned_identities for ref in closure),
             "JOINT26_FROZEN_CLOSURE_REQUIRED")
    for ref in refs:
        _ordinary(ref["path"]); _verify_project_ref(ref)
    _require(packet.get("control_ground_truth") is False
             and packet.get("physical_deployment_authorized") is False,
             "DERIVED_RESEARCH_FLAGS_REQUIRED")
    for role, path in input_paths.items():
        path = _ordinary(path)
        _require(_verify_project_ref(packet["inputs"][role]) == path, "JOINT26_INPUT_BINDING_DRIFT:" + role)
    output = _ordinary(output_dir, existing=False)
    _require(output == _ordinary(packet["output_dir"], existing=False), "JOINT26_OUTPUT_BINDING_DRIFT")
    if operation_args is None:
        _require(phase == "prepare", "EXACT_TRAINING_ARGUMENTS_REQUIRED")
        operation_args = ["--task-packet", str(packet_path)]
        for role, flag in (("source_manifest", "--source-manifest"), ("source_map", "--source-map"),
                           ("profile", "--profile"), ("geometry_index", "--geometry-index"),
                           ("client_source", "--client-source")):
            operation_args += [flag, str(_ordinary(input_paths[role]))]
        operation_args += ["--output", str(output)]
    _require(_command_is_pinned(packet.get("commands", []), operation, operation_args),
             "EXACT_JOINT26_COMMAND_NOT_PINNED")
    packet["admitted_authorization_expires_at"] = deadline
    if phase == "prepare":
        budget = packet["budgets"]
        profile = _load_ref(packet["inputs"]["profile"])
        expected = preparation_budgets(profile)
        options = preparation_options(profile)
        legacy = {"current_initialization_mode": "STATIC_NUMERICAL_CURRENT_TARGET_ONLY",
                  "collision_validation": "ORIGINAL_CAPSULE_V1",
                  "training_admission_policy": "STRICT_MODEL_NUMERICS_V1"}
        signed_options = packet.get("preparation_options", legacy)
        _require(isinstance(signed_options, dict), "SIGNED_PREPARATION_OPTIONS_DRIFT")
        signed_options = dict(signed_options)
        signed_options.setdefault("training_admission_policy", "STRICT_MODEL_NUMERICS_V1")
        _require(signed_options == options,
                 "SIGNED_PREPARATION_OPTIONS_DRIFT")
        signed_steps = budget.get("current_settle_steps_max", 60)
        signed_calls = budget.get("geometry_calls_per_window_max")
        _require(budget.get("gpu_count") == 0 and type(budget.get("samples_max")) is int
                 and 1 <= budget["samples_max"] <= 693
                 and type(signed_steps) is int and signed_steps == expected["current_settle_steps_max"]
                 and type(signed_calls) is int and signed_calls == expected["geometry_calls_per_window_max"],
                 "FINITE_CPU_LABEL_BUDGET_REQUIRED")
    return packet


def authorize_joint26_task(task_packet, *, phase, input_paths, output_dir,
                          operation_args=None):
    """Read the current receipt bundle under its existing publisher lock."""
    lock_path = ROOT / "_run/current/locks/governance.lock"
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            return _authorize_joint26_task(task_packet, phase=phase, input_paths=input_paths,
                output_dir=output_dir, operation_args=operation_args)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def authorize_joint26_training(packet_path, manifest_path, config_path, output_dir,
                               branch, world_size, rank=0):
    from .joint26_contract import PARALLEL_MODE
    if json.loads(_ordinary(packet_path).read_text()).get("execution_mode") == PARALLEL_MODE:
        from .joint26_parallel_admission import authorize_parallel_training
        return authorize_parallel_training(packet_path, manifest_path, config_path, output_dir,
                                           branch, world_size, rank=rank)
    _require(branch in {"raw", "robotized"} and type(world_size) is int and world_size > 0
             and type(rank) is int and 0 <= rank < world_size, "EXPLICIT_DDP_TOPOLOGY_REQUIRED")
    arguments = ["train", "--task-packet", str(_ordinary(packet_path)),
                 "--manifest", str(_ordinary(manifest_path)), "--config", str(_ordinary(config_path)),
                 "--output", str(_ordinary(output_dir, existing=False)), "--branch", branch]
    packet = authorize_joint26_task(packet_path, phase="train", input_paths={
        "paired_manifest": manifest_path, "training_config": config_path},
        output_dir=output_dir, operation_args=arguments)
    config = _load_ref(packet["inputs"]["training_config"])
    training = config.get("training", {})
    requested_updates = training.get("max_updates")
    signed_updates = packet["budgets"].get("optimizer_updates_max")
    _require(config.get("schema") == "humanego-ict-joint26-training-config-v1"
             and type(requested_updates) is int and requested_updates > 0
             and type(signed_updates) is int and signed_updates == requested_updates
             and packet["budgets"].get("gpu_count") == world_size
             and packet.get("branch") == branch, "JOINT26_UPDATE_OR_DDP_BUDGET_DRIFT")
    if packet.get("deadline_at") is None:
        _require(training.get("wall_seconds_max") is None, "JOINT26_WALL_BUDGET_DRIFT")
    if config.get("warmstart_ref") is not None:
        _require(packet["inputs"].get("warmstart") is not None
                 and reference_equal(config["warmstart_ref"], packet["inputs"]["warmstart"], ROOT),
                 "WARMSTART_NOT_TASK_BOUND")
        _ordinary(config["warmstart_ref"]["path"]); _verify_project_ref(config["warmstart_ref"])
    lease = json.loads(_ordinary(packet["gpu_lease_path"]).read_text())
    writer, gpu_ids = packet["writer"], packet.get("allocated_gpu_ids", [])
    _require(len(gpu_ids) == world_size and len(set(gpu_ids)) == world_size
             and all(type(i) is int and i >= 0 for i in gpu_ids)
             and packet.get("gpu_lease_verified") is True
             and lease.get("status") == "ACQUIRED" and lease.get("task_id") == packet["task_id"]
             and lease.get("hostname") == socket.gethostname() and lease.get("pid") == writer["pid"]
             and lease.get("process_startticks") == writer["proc_start_ticks"]
             and lease.get("fencing_token") == writer["fencing_token"]
             and lease.get("executor_epoch") == writer["executor_epoch"]
             and lease.get("gpu_id") == gpu_ids[0], "LIVE_ALLOCATED_GPU_LEASE_REQUIRED")
    expiry = datetime.fromisoformat(lease["expires_at"]).astimezone(timezone.utc).timestamp()
    _require(expiry > datetime.now(timezone.utc).timestamp(), "JOINT26_GPU_LEASE_EXPIRED")
    deadline = packet["admitted_authorization_expires_at"]
    packet["admitted_authorization_expires_at"] = expiry if deadline is None else min(deadline, expiry)
    return packet
