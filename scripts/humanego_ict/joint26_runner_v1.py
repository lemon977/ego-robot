#!/usr/bin/env python3
"""One new finite joint26 task using the existing CAS ledger and GPU lease.

Preparation and training are separate immutable tasks. No expired producer,
queue, implicit NAS replay, device execution or model download is revived.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from chaoyang.humanego_ict.governance_mount_compat import (
    install_governance_mount_compat, bind_publisher_module,
)
from chaoyang.humanego_ict.project_mount_paths import resolve_input_path, reference_equal
MOUNT_COMPAT = install_governance_mount_compat(ROOT)
CLI_MODULE = "chaoyang.humanego_ict.joint26_cli"
from chaoyang.governance.common import artifact_ref, atomic_json, canonical_bytes, load_json, now_iso, process_identity
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.humanego_ict.joint26_contract import (PREPARE, TRAIN, PREPARE_MODE,
    EXECUTION_MODE, PREPARE_ROUTE, ROUTE, candidate_contract)
from chaoyang.humanego_ict.joint26_runtime import _ordinary, preparation_budgets, preparation_options
from chaoyang.humanego_ict.joint26_limits import normalize_wall_seconds, step_bounded_lease_update

INFRA_SHA = "d3fb4d710501c1b2da4c943f84094fe8e0abd03cbdb9bc9e4d0e06c5ba2980dd"
SOURCE = "_run/current/pick_place_humanego_ict_production_prepare_a05_20261011/attempts/attempt_0001/output/PRODUCTION_MANIFEST.json"
MAP = "_run/maintenance/humanego_ict_transport_prepare_20261012/attempt_0001/payload_runtime/actual_audit_001/raw_train_map.json"
GEOMETRY = "_run/handoffs/revo2_ict_deploy_completion_v1/codex_20261010T152841Z_2cf21b18/lane_GEOM/SOURCE_INDEX.json"
CLIENT = "_run/handoffs/revo2_ict_deploy_completion_v1/codex_20261010T152841Z_2cf21b18/lane_CLIENT/sources/client/tianji_revo2_inference.py"
PROFILE = "docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/EGO_LABEL_PROFILE_V1.json"
OLD_MODEL = "_run/current/pick_place_humanego_ict_frozen_payload_raw_a02_20261012/attempts/attempt_0001/output/inference_tensors/model_config.json"
GPU_LEASE = ROOT / "_run/current/GPU_LEASE.json"
GPU_LOCK = ROOT / "_run/current/GPU_LEASE.lock"


def require(value, code):
    if not value:
        raise ValueError(code)


def default_config(world_size, wall_seconds, max_updates=20000):
    wall_seconds = normalize_wall_seconds(wall_seconds)
    require(type(world_size) is int and world_size > 0 and 64 % world_size == 0,
            "NPROC_MUST_DIVIDE_GLOBAL_BATCH_64")
    require(type(max_updates) is int and max_updates > 0, "POSITIVE_INTEGER_STEPS_REQUIRED")
    old = ROOT / OLD_MODEL
    require(artifact_ref(old)["sha256"] == "7ae7984c4d90db216636a6dffa396a9a39735dc0f358c62b1cf848b5ab5c778b",
            "FROZEN_ARCHITECTURE_SOURCE_DRIFT")
    return {"schema": "humanego-ict-joint26-training-config-v1", "training": {
        "global_batch": 64, "micro_batch": 64 // world_size, "max_updates": max_updates,
        "precision": "float32", "seed": 7, "learning_rate": 1e-4,
        "log_every": 20, "checkpoint_every": 1000, "validation_every": 1000,
        "wall_seconds_max": None if wall_seconds is None else wall_seconds - 300}, "model": {
        "horizon": 30, "action_dim": 26, "ict_dim": 29, "image_size": [240, 320],
        "max_ict": 8, "architecture": load_json(old)["architecture"]}}


def resolve_training_config(args):
    """Validate CLI/config budgets on CPU before data, storage or GPU work."""
    from chaoyang.humanego_ict.joint26_training import Joint26TrainingConfig, read_config
    args.wall_seconds = normalize_wall_seconds(args.wall_seconds)
    if args.steps is not None:
        require(type(args.steps) is int and args.steps > 0, "POSITIVE_INTEGER_STEPS_REQUIRED")
    if args.config:
        cfg, document = read_config(_ordinary(Path(args.config).absolute()))
        require(args.steps is None or args.steps == cfg.max_updates,
                "CLI_STEPS_CONFIG_BUDGET_MISMATCH")
        if args.wall_seconds is None or cfg.wall_seconds_max is None:
            document = dict(document, training=dict(document["training"],
                wall_seconds_max=None if args.wall_seconds is None else args.wall_seconds - 300))
            cfg = Joint26TrainingConfig(**document["training"])
    else:
        document = default_config(args.nproc, args.wall_seconds,
                                  20000 if args.steps is None else args.steps)
        cfg = Joint26TrainingConfig(**document["training"])
    overrides = {}
    for name in ("global_batch", "micro_batch"):
        value = getattr(args, name, None)
        if value is not None:
            require(type(value) is int and value > 0, "POSITIVE_INTEGER_" + name.upper() + "_REQUIRED")
            overrides[name] = value
    if overrides:
        document = dict(document, training=dict(document["training"], **overrides))
        cfg = Joint26TrainingConfig(**document["training"])
    try:
        cfg.validate(args.nproc, production=True)
    except ValueError as error:
        if "GLOBAL_BATCH_TOPOLOGY_MISMATCH" in str(error):
            raise ValueError(
                f"GLOBAL_BATCH_TOPOLOGY_MISMATCH: 全局batch {cfg.global_batch} 必须是 "
                f"GPU数 {args.nproc} × 每卡batch {cfg.micro_batch} 的正整数倍；"
                "请调整 --micro-batch（例如4卡配8，全局64，累积2）。") from error
        raise
    require(args.wall_seconds is None or cfg.wall_seconds_max is not None
            and cfg.wall_seconds_max <= args.wall_seconds - 300, "TRAINING_EXPORT_RESERVE_REQUIRED")
    document = dict(document, training=dict(document["training"], max_updates=cfg.max_updates))
    return cfg, document


def _infra():
    path = ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    require(artifact_ref(path)["sha256"] == INFRA_SHA, "EXISTING_INFRASTRUCTURE_SHA_DRIFT")
    spec = importlib.util.spec_from_file_location("_joint26_cas_infra", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return bind_publisher_module(module)


def _preflight(command, environment=None):
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=300)
    require(result.returncode == 0, "PREFLIGHT_FAILED:" + result.stdout[-2000:] + result.stderr[-2000:])
    return result.stdout


def _prepare_capacity(directory):
    """Bounded 64 MiB observation, not a full-volume free-space promise.

    Only our exclusively-created scratch is released after fsync/SHA readback.
    CPFS can report zero statvfs while a quota-bound write still succeeds.
    Training separately uses the unchanged 4 GiB infrastructure probe.
    """
    path = directory / "OWNED_CPU_CAPACITY_PROBE.bin"
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    identity = os.fstat(descriptor)
    try:
        block = bytes(1024 * 1024)
        for _ in range(64):
            pending = memoryview(block)
            while pending:
                count = os.write(descriptor, pending)
                require(count > 0, "CAPACITY_PROBE_ZERO_WRITE")
                digest.update(pending[:count]); pending = pending[count:]
        os.fsync(descriptor)
        os.close(descriptor); descriptor = None
        readback = artifact_ref(path)
        require(readback["bytes"] == 64 * 1024**2 and readback["sha256"] == digest.hexdigest(),
                "CPU_CAPACITY_READBACK_FAILED")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        observed = path.stat()
        require((observed.st_dev, observed.st_ino) == (identity.st_dev, identity.st_ino),
                "CAPACITY_PROBE_IDENTITY_CHANGED")
        path.unlink()  # Never selects existing data or any unresolved path.
    return {"status": "PASSED_MEASURED_WRITE_FSYNC_SHA_READBACK", "bytes": 64 * 1024**2,
            "sha256": digest.hexdigest(), "same_filesystem": directory.stat().st_dev,
            "scratch_released": True, "existing_files_deleted": False,
            "completed_at": now_iso(), "not_full_volume_free_space_guarantee": True}


def execute(args):
    from chaoyang.humanego_ict.joint26_console import ConsoleReporter
    reporter = ConsoleReporter(json_mode=getattr(args, 'json', False) or args.preflight_only)
    prepare = args.prepare
    console_branch = 'prepare' if prepare else args.branch
    wall = normalize_wall_seconds(args.wall_seconds, prepare=prepare)
    args.wall_seconds = wall
    require(prepare or args.branch in {"raw", "robotized"}, "EXPLICIT_BRANCH_REQUIRED")
    require(not prepare or not args.preflight_only, "PREPARE_IS_SEPARATE_FINITE_TASK")
    require(not prepare or args.steps is None, "TRAINING_STEPS_NOT_APPLICABLE_TO_PREPARATION")
    prepare_effort = (preparation_budgets(load_json(_ordinary(Path(args.profile).absolute())))
                      if prepare else None)
    prepare_options = (preparation_options(load_json(_ordinary(Path(args.profile).absolute())))
                       if prepare else None)
    if not prepare:
        require(args.manifest is not None, "JOINT26_MANIFEST_REQUIRED")
        require(args.nproc > 0 and 64 % args.nproc == 0, "NPROC_MUST_DIVIDE_GLOBAL_BATCH_64")
        cfg, config = resolve_training_config(args)
        manifest = _ordinary(Path(args.manifest).absolute())
        from chaoyang.humanego_ict.joint26_training import preflight, read_config
        reporter.stage(args.branch, '开始数据校验：逐项检查文件与SHA，首次可能需要几十秒。')
        _, validation, qualification = preflight(manifest, args.branch, world_size=args.nproc)
        qualification["requested_optimizer_updates"] = cfg.max_updates
        qualification["training_config_validated"] = True
        qualification["training_topology"] = {
            "world_size": args.nproc, "global_batch": cfg.global_batch,
            "micro_batch": cfg.micro_batch,
            "gradient_accumulation": cfg.global_batch // (args.nproc * cfg.micro_batch)}
        qualification["python"] = sys.executable
        reporter.preflight(args.branch, qualification)
        if args.preflight_only:
            return 0
    require(args.output is not None, "NEW_PROJECT_OUTPUT_DIRECTORY_REQUIRED")
    directory = _ordinary(Path(args.output).absolute(), existing=False)
    require(not directory.exists(), "OUTPUT_ALREADY_EXISTS_DUPLICATE_SUBMISSION_REJECTED")
    reporter.stage(console_branch, '检查运行资格、输出磁盘和GPU资源。')
    _preflight(["flock", "-s", str(ROOT / "_run/current/locks/governance.lock"),
                sys.executable, "-B", "-m", CLI_MODULE, "validate-governance"])
    infra = _infra()
    receipt, state, index = infra._snapshot(); infra._no_active_writer(state, index)
    mode, route, operation = ((PREPARE_MODE, PREPARE_ROUTE, PREPARE) if prepare else
                              (EXECUTION_MODE, ROUTE, TRAIN))
    closure = candidate_contract(ROOT).get(route, {}).get("code_closure", [])
    require(bool(closure), "JOINT26_OPERATION_CLOSURE_INCOMPLETE")
    admitted = load_json(resolve_input_path(receipt["files"]["algorithm_contract"]["path"], ROOT))
    admitted_closure = admitted.get("special_status_contracts", {}).get(route, {}).get("code_closure", [])
    require(len(admitted_closure) == len(closure)
            and all(reference_equal(a, b, ROOT) for a, b in zip(admitted_closure, closure)),
            "JOINT26_ROUTE_CLOSURE_NOT_CURRENT")
    inputs = {}
    if prepare:
        require(type(args.samples_max) is int and 1 <= args.samples_max <= 693, "FINITE_SAMPLE_BUDGET_REQUIRED")
        for role, value in (("source_manifest", args.source_manifest), ("source_map", args.source_map),
                            ("profile", args.profile), ("geometry_index", args.geometry_index),
                            ("client_source", args.client_source)):
            inputs[role] = artifact_ref(_ordinary(Path(value).absolute()))
    else:
        inputs["paired_manifest"] = artifact_ref(manifest)
        if args.config:
            inputs["configuration_source"] = artifact_ref(_ordinary(Path(args.config).absolute()))
        if args.warmstart:
            require(not config.get("warmstart_ref"), "AMBIGUOUS_DOUBLE_WARMSTART_REQUIRED")
            config["warmstart_ref"] = artifact_ref(_ordinary(Path(args.warmstart).absolute()))
        if config.get("warmstart_ref"):
            infra._verify(config["warmstart_ref"]); inputs["warmstart"] = config["warmstart_ref"]
    task_id = "pick_place_humanego_ict_joint26_" + ("prepare" if prepare else args.branch) + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower() + "_" + secrets.token_hex(3)
    packet_path = ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    output = directory / "output"
    writer = infra._writer_identity(task_id, 1)
    observed_identity = process_identity(writer["pid"])
    require(observed_identity["alive"] and observed_identity["start_ticks"] == writer["proc_start_ticks"],
            "PROCFS_IDENTITY_VIEW_CONFLICT_RUN_ON_NORMAL_HOST_NOT_VIRTUAL_PID_SANDBOX")
    started = datetime.now(timezone.utc)
    deadline = None if wall is None else started + timedelta(seconds=wall)
    directory.mkdir(parents=True, exist_ok=False)
    if not prepare:
        infra._json_once(directory / 'PRECHECK.json', qualification)
    if prepare:
        infra._json_once(directory / "STORAGE_CAPACITY_OBSERVATION.json", _prepare_capacity(directory))
    else:
        measure_storage, _ = infra._existing_helpers()
        inputs["storage_capacity_observation"] = measure_storage(directory, task_id)
    if not prepare:
        inputs["training_config"] = infra._json_once(directory / "TRAINING_CONFIG.json", config)
        cfg, _ = read_config(directory / "TRAINING_CONFIG.json"); cfg.validate(args.nproc, production=True)
        require(wall is None or cfg.wall_seconds_max is not None
                and cfg.wall_seconds_max <= wall - 300, "TRAINING_EXPORT_RESERVE_REQUIRED")
    command_args = ["--task-packet", str(packet_path)]
    if prepare:
        for role, flag in (("source_manifest", "--source-manifest"), ("source_map", "--source-map"),
                           ("profile", "--profile"), ("geometry_index", "--geometry-index"),
                           ("client_source", "--client-source")):
            command_args += [flag, inputs[role]["path"]]
        command_args += ["--output", str(output)]
    else:
        command_args = ["train", *command_args, "--manifest", str(manifest), "--config", inputs["training_config"]["path"],
                        "--output", str(output), "--branch", args.branch]
    command = [sys.executable, "-B", "-m", CLI_MODULE, "run", operation, *command_args]
    gpu_ids = [] if prepare else ([int(i) for i in args.gpu_ids.split(",")] if args.gpu_ids else list(range(args.nproc)))
    require(prepare or len(gpu_ids) == args.nproc and len(set(gpu_ids)) == args.nproc
            and all(i >= 0 for i in gpu_ids), "EXPLICIT_UNIQUE_GPU_IDS_REQUIRED")
    budgets = {"wall_seconds_max": wall, "gpu_count": len(gpu_ids), "cpu_threads_per_worker": 2,
               "optimizer_updates_max": 0 if prepare else cfg.max_updates,
               "samples_max": args.samples_max if prepare else 0,
               "geometry_calls_per_window_max": prepare_effort["geometry_calls_per_window_max"] if prepare else 0,
               "current_settle_steps_max": prepare_effort["current_settle_steps_max"] if prepare else 0,
               "model_calls": 0 if prepare else (cfg.max_updates * (cfg.global_batch // cfg.micro_batch)
                   + ((cfg.max_updates + cfg.validation_every - 1) // cfg.validation_every)
                     * (len(validation) + args.nproc)),
               "robot_commands_max": 0}
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": "Prepare derived joint26 research supervision" if prepare else "Train HumanEgo ICT H30x26 from admitted research supervision: " + args.branch,
        "phase": mode, "plan_revision": "HUMANEGO_ICT_JOINT26_V1", "execution_revision": task_id.upper(),
        "execution_mode": mode, "branch": args.branch if not prepare else None,
        "read_set": ["AGENTS.md", "docs/governance/CURRENT_STATUS_RECEIPT.json", "docs/governance/ALGORITHM_CONTRACT.json", "tasks/current/INDEX.json"],
        "read_set_expansions": ["Only explicit signed inputs, their frozen dependency closures and route runtime sources"],
        "write_set": [str(directory.relative_to(ROOT)), str(packet_path.relative_to(ROOT)),
            "tasks/current/INDEX.json", "docs/governance", "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock",
            "docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/STATUS.json"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "EXACT_CANDIDATE_ROUTE", "SHA_BOUND_INPUTS"],
        "required_outputs": ["RESULT.json", "POST_VALIDATION.json"], "budgets": budgets,
        "stop_conditions": ["requested optimizer updates completed", "explicit deadline if set", "double writer", "SHA drift", "input or numerical admission failure"],
        "claim_limit": "Derived assumed-model supervision only; not measured controls, pi0.5 weights, real-device permission or visual quality",
        "inputs": inputs, "runtime_code": closure, "writer": writer, "writer_file": str(directory / "WRITER.json"),
        "packet_path": str(packet_path), "output_dir": str(output), "commands": [shlex.join(command)],
        "execution_started_at": started.isoformat(), "deadline_at": None if deadline is None else deadline.isoformat(),
        "termination_policy": "OPTIMIZER_UPDATES_ONLY" if wall is None else "OPTIMIZER_UPDATES_OR_DEADLINE",
        "allocated_gpu_ids": gpu_ids, "gpu_lease_verified": bool(gpu_ids), "gpu_lease_path": str(GPU_LEASE),
        "control_ground_truth": False, "physical_deployment_authorized": False}
    if prepare:
        packet["preparation_options"] = prepare_options
    if prepare and args.sample_id:
        packet["selected_sample_ids"] = args.sample_id
    packet["run_signature"] = hashlib.sha256(canonical_bytes(packet)).hexdigest()
    require(not _validate_packet(packet), "INVALID_FINITE_TASK_PACKET")
    env = infra._environment(directory)
    # The CAS publisher may import cached legacy contract helpers too. Bind its
    # caches to this owned output, just like the launched workers; no /tmp or
    # historical cache is required merely to rebuild governance metadata.
    for key in ("TMPDIR", "TMP", "TEMP", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "CUDA_CACHE_PATH",
                "TORCH_EXTENSIONS_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "MPLCONFIGDIR"):
        os.environ[key] = env[key]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    env["MUJOCO_GL"] = "disable"
    child, registered, leased, errors = None, False, False, []
    stopped = []
    old_handlers = {n: signal.signal(n, lambda n, f: stopped.append(n)) for n in (signal.SIGINT, signal.SIGTERM)}
    _, lease_update = infra._existing_helpers()
    if wall is None:
        from functools import partial
        lease_update = partial(step_bounded_lease_update, lease_path=GPU_LEASE, lock_path=GPU_LOCK)
    try:
        receipt, state, index = infra._snapshot(); infra._no_active_writer(state, index)
        if gpu_ids:
            from chaoyang.ops.run_gpu_command_with_v71_lease import try_acquire, physical_gpu_capacity
            for gpu_id in gpu_ids:
                capacity = physical_gpu_capacity(gpu_id, 40960)
                require(capacity is not None, "WAIT_GPU_RESOURCE_NO_ALGORITHM_STARTED")
            acquired = try_acquire(SimpleNamespace(task_id=task_id, attempt_id="attempt_0001", executor_epoch=1,
                priority="CHECKPOINT_TRAINING", gpu_id=gpu_ids[0], min_free_mib=40960,
                lease=GPU_LEASE, lock=GPU_LOCK), writer["fencing_token"])
            require(acquired is not None, "WAIT_GPU_RESOURCE_NO_ALGORITHM_STARTED")
            leased = True; lease_update(packet)
            infra._json_once(directory / "GPU_LEASE_ACQUIRED.json", load_json(GPU_LEASE))
        packet_path.parent.mkdir(parents=True, exist_ok=False)
        packet_ref = infra._json_once(packet_path, packet)
        infra._json_once(directory / "WRITER.json", writer)
        infra._json_once(directory / "FROZEN_INPUTS.json", inputs)
        state["tasks"].append({"task_id": task_id, "phase": mode, "attempt": 1, "status": "RUNNING",
            "updated_at": now_iso(), "heartbeat_at": now_iso(), "pid": writer["pid"],
            "proc_start_ticks": writer["proc_start_ticks"], "task_packet": packet_ref, "run_signature": packet["run_signature"]})
        state["next_task"] = {"task_id": task_id, "expected_resource": "GPU" if gpu_ids else "CPU"}
        index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(ROOT)),
            "packet_sha256": packet_ref["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
        infra._publish(state, index, "ICT_JOINT26_REGISTERED", expected_revision=receipt["governance_revision"])
        registered = True
        reporter.registered(console_branch, gpu_ids, None if prepare else cfg.max_updates, output,
            {"status": "REGISTERED", "task_id": task_id, "output": str(output), "deadline_at": packet["deadline_at"]})
        launch = command if prepare else [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc_per_node=" + str(args.nproc), "--module", CLI_MODULE, "run", TRAIN, *command_args]
        with (directory / "EXECUTION.log").open("xb") as log:
            child = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if leased:
                lease_update(packet, worker_pid=child.pid)
            last = time.monotonic()
            if not prepare:
                reporter.stage(args.branch, '训练进程已启动，正在校验缓存和加载模型。')
            while child.poll() is None:
                require(not stopped, "OWNED_PUBLISHER_INTERRUPTED")
                require(deadline is None or datetime.now(timezone.utc) < deadline, "FINITE_TASK_DEADLINE_EXCEEDED")
                if time.monotonic() - last >= 60:
                    if leased:
                        lease_update(packet, worker_pid=child.pid)
                    receipt, state, index = infra._snapshot(); row, _ = infra._own_row(packet, state, index)
                    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                    atomic_json(directory / "HEARTBEAT.json", {"task_id": task_id, "child_pid": child.pid,
                        "deadline_at": packet["deadline_at"], "output": str(output), "next_action": "Continue owned finite stage; never consume unverified partials"})
                    infra._publish(state, index, "ICT_JOINT26_HEARTBEAT", expected_revision=receipt["governance_revision"])
                    last = time.monotonic()
                if not prepare:
                    reporter.progress(args.branch, output, cfg.max_updates)
                time.sleep(1)
            if not prepare:
                reporter.progress(args.branch, output, cfg.max_updates, final=True)
        require(child.returncode == 0, "JOINT26_CHILD_EXIT_" + str(child.returncode))
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
    finally:
        for number in old_handlers:
            signal.signal(number, signal.SIG_IGN)
        try:
            infra._stop_child(child)
        except BaseException as error:
            errors.append("CHILD_CLEANUP:" + str(error))
        if leased:
            try:
                lease_update(packet, release_reason="ICT_JOINT26_FINITE_EXIT")
            except BaseException as error:
                errors.append("LEASE_RELEASE:" + str(error))
    if not registered:
        infra._json_once(directory / "REGISTRATION_FAILURE.json", {"task_id": task_id, "errors": errors, "algorithm_started": False})
        if not reporter.json_mode:
            reporter.finished(console_branch, 'FAILED_RUNTIME_FINAL', {'errors': errors}, output,
                              None if prepare else cfg.max_updates)
        return 2
    if not errors:
        reporter.stage(console_branch, '训练步骤结束，正在校验权重并封存结果。' if not prepare else '正在校验并封存准备结果。')
    stage = None
    refs = []
    try:
        name = "PREPARATION_RESULT.json" if prepare else "TRAINING_RESULT.json"
        stage_path = output / name
        require(stage_path.is_file(), "COMPLETE_STAGE_RESULT_ABSENT")
        refs.append(artifact_ref(stage_path)); stage = load_json(stage_path)
        if prepare:
            ref = stage["manifest"]; infra._verify(ref); refs.append(ref)
        else:
            require(stage["optimizer_updates"] == cfg.max_updates
                    and stage.get("requested_optimizer_updates") == cfg.max_updates
                    and stage["completed_update_budget"] is True,
                    "EXACT_REQUESTED_UPDATES_NOT_COMPLETE")
            export = output / "inference_tensors/EXPORT_MANIFEST.json"
            refs.append(artifact_ref(export))
            exported = load_json(export)
            require(exported.get("requested_optimizer_updates") == cfg.max_updates
                    and exported.get("optimizer_updates") == cfg.max_updates,
                    "EXPORTED_UPDATE_BUDGET_DRIFT")
            for ref in exported["files"].values():
                infra._verify(ref); refs.append(ref)
    except BaseException as error:
        errors.append("RESULT_READBACK:" + str(error))
    result = {"schema": "humanego-ict-joint26-finite-result-v1", "task_id": task_id,
        "run_signature": packet["run_signature"], "status": "PASSED" if not errors else "FAILED_RUNTIME_FINAL",
        "branch": packet["branch"], "artifacts": refs, "errors": errors,
        "requested_optimizer_updates": None if prepare else cfg.max_updates,
        "control_ground_truth": False, "physical_deployment_authorized": False, "completed_at": now_iso()}
    receipt, state, index = infra._snapshot(); row, pointer = infra._own_row(packet, state, index)
    result_ref = infra._json_once(directory / "RESULT.json", result)
    row.update(status=result["status"], result=result_ref, pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    status_path = ROOT / "docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/STATUS.json"
    handoff = load_json(status_path)
    handoff.update(last_task_result=result_ref, last_task_id=task_id, last_task_status=result["status"])
    if not prepare:
        handoff.setdefault("requested_optimizer_updates", {})[args.branch] = cfg.max_updates
    if prepare and not errors:
        handoff.update(real_joint_label_preparation_status="EXECUTED_NOT_MODEL_QUALITY",
            joint26_training_manifest=stage["manifest"], converted_windows=stage["converted_windows"],
            joint26_data_gates=stage["data_gates"], failure_clusters=stage["failure_clusters"],
            branch_joint26_counts=stage["branches"])
        handoff["next_action"] = ("Both research data gates passed; colleague can run two independent finite GPU training tasks."
            if all(value == "PASSED_RESEARCH_LABEL_GATE" for value in stage["data_gates"].values())
            else "Do not train; inspect joint-label failure clusters or expand a frozen pilot under a new finite task. Do not lower numerical or group gates.")
    elif not prepare and not errors:
        handoff[args.branch + "_joint26_checkpoint"] = artifact_ref(Path(stage["last_checkpoint"]))
        handoff["optimizer_updates"][args.branch] = stage["optimizer_updates"]
        handoff["next_action"] = "Train remaining branch if absent, then offline ICT input/output replay; physical deployment remains unauthorized."
    else:
        handoff["next_action"] = "Inspect finite task failure; never consume partials or revive its packet."
    atomic_json(status_path, handoff)
    infra._publish(state, index, "ICT_JOINT26_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", CLI_MODULE, "validate-governance"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    infra._json_once(directory / "POST_VALIDATION.json", {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr})
    for number, handler in old_handlers.items():
        signal.signal(number, handler)
    reporter.finished(console_branch, result['status'], result, output,
        None if prepare else cfg.max_updates, stage=stage, governance_exit=check.returncode)
    return 0 if not errors and check.returncode == 0 else 2


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--parallel" in arguments:
        # One fenced publisher, two independent compute lanes. The ordinary
        # single-branch entry and its frozen training recipe are unchanged.
        from joint26_parallel_runner_v1 import main as parallel_main
        try:
            return parallel_main(arguments)
        except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
            print("未启动或并行任务已停止：" + str(error), file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Separate CPU source-derived IK preparation, no training")
    parser.add_argument("--branch", choices=("raw", "robotized"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--steps", type=int,
                        help="Positive optimizer-update target; default 20000 for compatibility, not a required count")
    parser.add_argument("--gpu-ids", help="Physical GPU IDs, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--global-batch", type=int, help="每分支全局batch；当前生产配置须为64")
    parser.add_argument("--micro-batch", "--batch-size", dest="micro_batch", type=int,
                        help="每卡每次前向batch；须整除全局batch/GPU数，显式覆盖config")
    parser.add_argument("--warmstart", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print original machine JSON instead of compact Chinese progress")
    parser.add_argument("--wall-seconds", type=int, default=None,
                        help="Optional wall budget; omitted or 0 trains to requested steps without a time limit")
    for name, default in (("source-manifest", SOURCE), ("source-map", MAP), ("geometry-index", GEOMETRY), ("client-source", CLIENT), ("profile", PROFILE)):
        parser.add_argument("--" + name, type=Path, default=ROOT / default)
    parser.add_argument("--samples-max", type=int, default=693)
    parser.add_argument("--sample-id", action="append", help="Freeze selected pilot anchors; no implicit good-result selection")
    try:
        return execute(parser.parse_args(arguments))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print("未启动或有限任务停止：" + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
