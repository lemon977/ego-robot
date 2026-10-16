"""Explicit frozen, previously admitted payload research continuation.

This is NOT the original per-sample source-stat route. No original NAS replay
or current raw-source equivalence is claimed. Adoption, historical admission,
actual project-byte validation and complete-state resume are separate gates.
The original model/trainer/loss/configuration and 121 frozen files are reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import sys

from chaoyang.governance.common import REPO_ROOT
from .production_training_gate import _sealed_producer, _utc
from .runtime_gate import _is_descendant, bind_cuda_visible_devices, runtime_cache_paths
from .production_resume import _project_path, _read_json, _config_invariants, validate_parent_evidence

OPERATION = "train_pick_place_humanego_ict_frozen_payload_resume_v1"
ROUTE = "humanego_ict_frozen_payload_resume_v1"
EXECUTION_MODE = "HUMANEGO_ICT_FROZEN_PAYLOAD_RESEARCH_RESUME"
TRAINING_SCOPE = "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING"
SOURCE_POLICY = "FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY"
PERIODIC_CONFIG_SCHEMA = "pick-place-humanego-ict-frozen-payload-periodic-config-v1"
PERIODIC_INITIALIZATION = "RESUME_VERIFIED_PERIODIC_COMPLETE_STATE"
NORMAL_ROLES = {"task_packet": "parent_task_packet", "outer_result": "parent_task_result",
    "inner_result": "parent_inner_result", "training_config": "parent_training_config",
    "checkpoint": "resume_checkpoint", "checkpoint_sidecar": "resume_checkpoint_sidecar"}
PERIODIC_ROLES = {k: v for k, v in NORMAL_ROLES.items() if k != "inner_result"}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def _ref(path, root):
    from chaoyang.human_ego.utils.frozen_contract import file_reference
    return file_reference(_project_path(path, root, existing=True))


def _json(reference, root):
    return _read_json(reference, root, "frozen-payload entry input")


def parent_refs(packet):
    mode = packet.get("resume_mode")
    require(mode in {"normal", "periodic"}, "EXPLICIT_RESUME_MODE_REQUIRED")
    roles = NORMAL_ROLES if mode == "normal" else PERIODIC_ROLES
    inputs = packet.get("inputs", {})
    require(all(isinstance(inputs.get(v), dict) for v in roles.values()), "COMPLETE_PARENT_REFS_REQUIRED")
    return {k: inputs[v] for k, v in roles.items()}


def validate_config(config, mode, world=1):
    require(mode in {"normal", "periodic"} and world == 1, "ONE_H20_EXACT_PARENT_TOPOLOGY_REQUIRED")
    if mode == "normal":
        from .production_resume_gate import validate_resume_config
        validate_resume_config(config, world)
    else:
        require(config.get("schema_version") == PERIODIC_CONFIG_SCHEMA
            and config.get("initialization") == PERIODIC_INITIALIZATION
            and isinstance(config.get("resume_parent_bindings"), dict)
            and set(config["resume_parent_bindings"]) == set(PERIODIC_ROLES),
            "EXPLICIT_PERIODIC_CONFIG_REQUIRED")
        # Numeric validator projection only, not an old admission/parent gate.
        from .production_training_cli import validate_config as numeric_validate
        view = dict(config, schema_version="pick-place-humanego-ict-production-training-config-v1",
                    initialization="FRESH_SHARED_PRETRAINED_NEW_IO")
        view.pop("resume_parent_bindings")
        numeric_validate(view, world)
    require(config["training"]["max_updates"] == 20000, "UNCHANGED_20000_TARGET_REQUIRED")


def _task_terminal(root, state, index, packet_ref, result_ref, statuses):
    packet, result = _json(packet_ref, root), _json(result_ref, root)
    task_id = packet["task_id"]
    rows = [r for r in state["tasks"] if r.get("task_id") == task_id]
    entries = [r for r in index["task_packets"] if r.get("task_id") == task_id]
    path = _project_path(packet_ref["path"], root, existing=True)
    require(path.is_relative_to(root / "tasks/current") and len(rows) == len(entries) == 1,
            "REGISTERED_SEALED_TASK_REQUIRED")
    row, item = rows[0], entries[0]
    require(row.get("status") in statuses and row.get("task_packet") == packet_ref
        and row.get("result") == result_ref and row.get("run_signature") == packet.get("run_signature")
        and item.get("execution_allowed") is False and item.get("packet_sha256") == packet_ref["sha256"]
        and _project_path(root / item["packet_path"], root, existing=True) == path,
        "SEALED_TASK_LEDGER_BINDING_DRIFT")
    require(result.get("task_id") == task_id and result.get("run_signature") == packet.get("run_signature"),
            "SEALED_TASK_RESULT_BINDING_DRIFT")
    return packet, result


def _engineering(root, packet, route, state, index):
    inputs = packet["inputs"]
    ep, er = _task_terminal(root, state, index, inputs["entry_engineering_task_packet"],
                           inputs["entry_engineering_task_result"], {"PASSED"})
    require(ep.get("execution_mode") == "CPU_DEVELOPMENT_ONLY"
        and all(ep.get("budgets", {}).get(k) == 0 for k in ("gpu_count", "model_calls", "training_steps")),
        "SEPARATE_CPU_ADOPTION_TASK_REQUIRED")
    require(er.get("schema") == "frozen-payload-entry-adoption-result-v1"
        and er.get("task_terminal_status") == "PASSED" and er.get("source_policy") == SOURCE_POLICY
        and er.get("physical_deployment_authorized") is False and er.get("control_ground_truth") is False,
        "EXPLICIT_RESEARCH_SNAPSHOT_ADOPTION_RESULT_REQUIRED")
    refs = er.get("code_refs", [])
    require(bool(refs) and all(r in packet["runtime_code"] for r in refs)
            and all(r in refs or r in er.get("protected_original_code_refs", []) for r in route["code_closure"]),
            "CPU_ADOPTION_CODE_CLOSURE_DRIFT")
    for r in refs:
        _ref(r["path"], root) == r or require(False, "ADOPTED_CODE_SHA_DRIFT")
    tests = _json(er["cpu_test_result"], root)
    require(tests.get("status") == "PASSED" and tests.get("exit_code") == 0
            and type(tests.get("passed_count")) is int and tests["passed_count"] > 0,
            "ACTUAL_CPU_TEST_RESULT_REQUIRED")


def _parent(root, packet, config, state, index):
    refs = parent_refs(packet)
    statuses = {"PASSED"} if packet["resume_mode"] == "normal" else {"FAILED", "FAILED_RUNTIME_FINAL"}
    parent, outer = _task_terminal(root, state, index, refs["task_packet"], refs["outer_result"], statuses)
    require(parent["task_id"] != packet["task_id"] and parent.get("branch") == packet.get("branch"),
            "SAME_BRANCH_SEPARATE_PARENT_REQUIRED")
    for role in ("production_manifest", "prepared_pair", "pretrained", "pretrained_config"):
        require(parent["inputs"][role] == packet["inputs"][role], "PARENT_SOURCE_BINDING_DRIFT:" + role)
    for ref in parent["runtime_code"]:
        require(ref in packet["runtime_code"] and _ref(ref["path"], root) == ref, "PARENT_CODE_NOT_PRESERVED")
    old_config = _json(refs["training_config"], root)
    require(config["resume_parent_bindings"] == refs
        and _config_invariants(old_config) == _config_invariants(config), "UNCHANGED_FULL_RECIPE_REQUIRED")
    if packet["resume_mode"] == "normal":
        if old_config.get('schema_version') == PERIODIC_CONFIG_SCHEMA:
            from .frozen_payload_state import validate_frozen_normal_parent
            evidence = validate_frozen_normal_parent(refs, root=root)
        else:
            evidence = validate_parent_evidence(refs, root=root)
        step = evidence["inner_result"]["optimizer_updates"]
    else:
        require(outer.get("task_terminal_status") in {"FAILED", "FAILED_RUNTIME_FINAL"}, "FAILED_PARENT_REQUIRED")
        from .frozen_payload_state import validate_periodic_evidence
        evidence = validate_periodic_evidence(refs, packet["periodic_evidence"], root=root)
        step = evidence["optimizer_updates"]
    require(type(step) is int and 0 < step < 20000, "ACTUAL_PARENT_STEP_AND_REMAINING_BUDGET_REQUIRED")
    packet["admitted_parent_step"] = step


def _authority_once(root, packet_path, args, output, *, receipt):
    root = root.resolve(strict=True)
    path = _project_path(packet_path, root, existing=True)
    require(path.is_relative_to(root / "tasks/current"), "REGISTERED_PROJECT_TASK_PACKET_REQUIRED")
    packet_ref = _ref(path, root)
    packet = _json(packet_ref, root)
    contract = _json(receipt["files"]["algorithm_contract"], root)
    route = contract.get("special_status_contracts", {}).get(ROUTE)
    require(isinstance(route, dict) and route.get("candidate_operation") == OPERATION
        and route.get("source_policy") == SOURCE_POLICY and route.get("task_packet_required") is True
        and route.get("full_checkpoint_required") is True and route.get("frozen_payload_map_required") is True
        and route.get("control_ground_truth") is False and route.get("physical_deployment_authorized") is False
        and route.get("training_scope") == TRAINING_SCOPE, "NEW_FROZEN_PAYLOAD_ROUTE_NOT_ADOPTED")
    require(packet.get("execution_mode") == EXECUTION_MODE and packet.get("source_policy") == SOURCE_POLICY
        and packet.get("training_scope") == TRAINING_SCOPE and packet.get("branch") in {"raw", "robotized"},
        "NEW_RESEARCH_SNAPSHOT_TASK_REQUIRED_NOT_OLD_PACKET")
    now = datetime.now(timezone.utc).timestamp()
    started, deadline = _utc(packet["execution_started_at"]), _utc(packet["deadline_at"])
    wall = packet["budgets"]["wall_seconds_max"]
    require(type(wall) in (int, float) and math.isfinite(wall) and 1500 < wall <= 43200
            and started <= now < min(deadline, started + wall), "FINITE_AUTHORIZATION_REQUIRED")
    index = _json(receipt["files"]["task_packet_index_current"], root)
    state = _json(receipt["files"]["task_state"], root)
    allowed = [r for r in index["task_packets"] if r.get("execution_allowed") is True]
    active = [r for r in state["tasks"] if r.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    writer = packet["writer"]
    require(len(allowed) == len(active) == 1 and allowed[0]["task_id"] == active[0]["task_id"] == packet["task_id"]
        and allowed[0]["packet_sha256"] == packet_ref["sha256"]
        and _project_path(root / allowed[0]["packet_path"], root, existing=True) == path
        and active[0]["status"] == "RUNNING" and active[0]["run_signature"] == packet["run_signature"]
        and active[0]["pid"] == writer["pid"] and active[0]["proc_start_ticks"] == writer["proc_start_ticks"]
        and writer["scope"] == packet["task_id"] and _is_descendant(writer), "UNIQUE_LIVE_WRITER_REQUIRED")
    require(_json(_ref(packet["writer_file"], root), root) == writer, "WRITER_FENCE_DRIFT")
    commands = [shlex.split(c) for c in packet.get("commands", [])]
    require(any(c[i:i + 3] == ["-m", "chaoyang.cli", "run"] and c[i + 3:] == [OPERATION, *args]
        for c in commands for i in range(max(0, len(c) - 2))), "EXACT_NEW_OPERATION_COMMAND_REQUIRED")
    closure = route.get("code_closure", [])
    require(closure and packet.get("runtime_code") and all(r in packet["runtime_code"] for r in closure),
            "TASK_ROUTE_CODE_CLOSURE_DRIFT")
    for ref in packet["runtime_code"]:
        require(_ref(ref["path"], root) == ref, "FROZEN_CODE_SHA_DRIFT")
    require(_project_path(output, root) == _project_path(packet["output_dir"], root), "OUTPUT_BINDING_DRIFT")
    budget = packet["budgets"]
    require(budget.get("gpu_count") == 1 and budget.get("optimizer_updates_max") == 20000
            and packet.get("gpu_lease_verified") is True and packet.get("allocated_gpu_ids") == [0],
            "UNCHANGED_20K_SINGLE_GPU_BUDGET_REQUIRED")
    lease = _json(_ref(packet["gpu_lease_path"], root), root)
    require(lease.get("status") == "ACQUIRED" and lease.get("task_id") == packet["task_id"]
        and lease.get("hostname") == socket.gethostname() and lease.get("pid") == writer["pid"]
        and lease.get("process_startticks") == writer["proc_start_ticks"]
        and lease.get("fencing_token") == writer["fencing_token"] and lease.get("executor_epoch") == writer["executor_epoch"]
        and lease.get("gpu_id") == 0 and _utc(lease["expires_at"]) > now, "CURRENT_GPU_LEASE_REQUIRED")
    inputs = packet["inputs"]
    config = _json(inputs["training_config"], root)
    validate_config(config, packet.get("resume_mode"))
    _engineering(root, packet, route, state, index)
    _sealed_producer(root, packet, state, index)  # Historical full raw-replay qualification, not current NAS replay.
    _parent(root, packet, config, state, index)
    manifest = _json(inputs["production_manifest"], root)
    require(inputs["prepared_pair"] == inputs["production_manifest"]
        and manifest.get("schema") == "pick-place-humanego-ict-production-paired-v1"
        and manifest.get("fixture_only") is not True and manifest.get("control_ground_truth") is False
        and manifest.get("physical_deployment_authorized") is False, "HISTORICALLY_ADMITTED_MANIFEST_REQUIRED")
    snapshot = _json(inputs["snapshot_map"], root)
    require(snapshot.get("production_manifest_ref") == inputs["production_manifest"]
        or snapshot.get("manifest_ref") == inputs["production_manifest"], "SNAPSHOT_ORIGINAL_MANIFEST_BINDING_DRIFT")
    packet["admitted_authorization_expires_at"] = min(deadline, started + wall, _utc(lease["expires_at"]))
    return packet


def authorize(root, packet_path, args, *, manifest, config, checkpoint, output, branch):
    root = Path(root).resolve(strict=True)
    lock_path = _project_path(root / "_run/current/locks/governance.lock", root, existing=True)
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            receipt = json.loads((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_bytes())
            packet = _authority_once(root, packet_path, args, output, receipt=receipt)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    for role, supplied in (("production_manifest", manifest), ("training_config", config), ("resume_checkpoint", checkpoint)):
        require(_ref(supplied, root) == packet["inputs"][role], "ARGUMENT_INPUT_SHA_BINDING_DRIFT:" + role)
    require(branch == packet["branch"], "ARGUMENT_BRANCH_DRIFT")
    return packet


def candidate_contract(root, original_closure):
    names = ("src/chaoyang/humanego_ict/frozen_payload_entry_cli.py",
        "src/chaoyang/humanego_ict/frozen_payload_dataset.py", "src/chaoyang/humanego_ict/frozen_payload_state.py",
        "src/chaoyang/ops/" + OPERATION + ".py", "scripts/humanego_ict/frozen_payload_entry_runner_v1.py")
    adoption_path = root / 'tasks/receipts/HUMANEGO_ICT_FROZEN_PAYLOAD_ENTRY_ADOPTION_V1.json'
    if not original_closure or not adoption_path.is_file() or any(not (root / n).is_file() for n in names):
        return {}
    adoption = _json(_ref(adoption_path, root), root)
    require(adoption.get('status') == 'PASSED' and adoption.get('source_policy') == SOURCE_POLICY,
            'EXPLICIT_ADOPTION_RECEIPT_REQUIRED')
    adoption_refs = {k: adoption[k] for k in ('entry_engineering_task_packet', 'entry_engineering_task_result')}
    from .frozen_payload_dataset import DEFAULT_READER_REF
    require(_ref(DEFAULT_READER_REF['path'], root) == DEFAULT_READER_REF, 'SEALED_RAM_READER_DEPENDENCY_DRIFT')
    closure = {r["path"]: r for r in [*original_closure, DEFAULT_READER_REF,
                                    *[_ref(root / n, root) for n in names]]}
    return {ROUTE: {"candidate_operation": OPERATION, "current_runnable_operation": None,
        "execution_mode": EXECUTION_MODE, "training_scope": TRAINING_SCOPE, "source_policy": SOURCE_POLICY,
        "task_packet_required": True, "full_checkpoint_required": True, "frozen_payload_map_required": True,
        "periodic_recovery_support": True, "control_ground_truth": False, "physical_deployment_authorized": False,
        "supervision_class": "DERIVED_RETARGET", "code_closure": list(closure.values()),
        'sealed_adoption_task_refs': adoption_refs,
        "claim_limit": "Explicit frozen admitted project snapshot research; no current raw-source stat/replay equivalence or physical control qualification."}}


def _run_training(packet, config, encoded, args, reauthorize):
    # All imports which can construct/call models are below full admission.
    from .frozen_payload_dataset import FrozenPayloadDataset
    from .frozen_payload_state import prepare_frozen_payload_checkpoint
    from .production_resume_cli import lineage_signatures
    from .production_training_cli import validate_admitted_inputs, export_dataset_envelope
    from .production_resume_gate import production_resume_storage_ready, EXECUTION_MODE as STORAGE_MODE
    from .derived_training_cli import _check_caches
    from .derived_runtime import write_once
    from .runtime_gate import file_ref
    root, output = REPO_ROOT, args.output_dir
    dataset = FrozenPayloadDataset(packet["inputs"]["production_manifest"], packet["inputs"]["snapshot_map"],
                                  project_root=root, expected_branch=args.branch)
    qualification = dataset.qualification
    validate_admitted_inputs(config, encoded, packet, qualification)
    # Project only the old storage numeric observation checker, NOT an old
    # operation/task gate or a parent status. The actual Task retains new mode.
    storage = production_resume_storage_ready(output, config["minimum_free_output_bytes"],
                                              packet=dict(packet, execution_mode=STORAGE_MODE), root=root)
    bind_cuda_visible_devices(packet)
    _check_caches(output, root)
    for p in runtime_cache_paths(output, root):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
    os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
    import torch
    torch.set_num_threads(2)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "WAIT_GPU_RESOURCE")
    torch.cuda.set_device(0)
    from .training import TrainingConfig, train_candidate
    signatures = lineage_signatures(packet, qualification)
    tc = TrainingConfig(**config["training"])
    reauthorize()
    resume, adaptation = prepare_frozen_payload_checkpoint(parent_refs(packet), mode=packet["resume_mode"],
        root=root, full_config_ref=packet["inputs"]["training_config"], training_config=tc, signatures=signatures,
        output_dir=output, dataset_length=len(dataset), world_size=1, device_type="cuda",
        periodic_evidence=packet.get("periodic_evidence"))
    dataset.verify_all_dependencies()
    reauthorize()
    from .model import Revo2ICTModel
    model = Revo2ICTModel(**config["model"])
    result = train_candidate(model, dataset, tc, output, signatures, resume=resume, device="cuda:0", initializer=None,
        authorization_expires_at=packet["admitted_authorization_expires_at"] - config["minimum_export_margin_seconds"])
    dataset.verify_all_dependencies()
    reauthorize()
    require(type(result.get("optimizer_updates")) is int
        and packet["admitted_parent_step"] < result["optimizer_updates"] <= 20000,
        "REAL_NEW_OPTIMIZER_UPDATES_REQUIRED")
    result.update(task_id=packet["task_id"], run_signature=packet["run_signature"], branch=args.branch,
        supervision_class="DERIVED_RETARGET", control_ground_truth=False, physical_deployment_authorized=False,
        training_scope=TRAINING_SCOPE, source_policy=SOURCE_POLICY, snapshot_map_ref=packet["inputs"]["snapshot_map"],
        parent_optimizer_updates=packet["admitted_parent_step"],
        new_optimizer_updates=result["optimizer_updates"] - packet["admitted_parent_step"],
        resume_mode=packet["resume_mode"], resume_adaptation=adaptation, storage_admission=storage,
        current_raw_source_replay_or_stat_equivalence=False)
    rp = output / f"RESULT_{result['optimizer_updates']:06d}.json"
    write_once(rp, result, root)
    pair = qualification["prepared_pair"]
    normal, envelope = output / "inference_normalization.json", output / "export_dataset.json"
    write_once(normal, pair["normalization"], root)
    write_once(envelope, export_dataset_envelope(qualification, packet["inputs"]["production_manifest"]), root)
    provenance = {"task_id": packet["task_id"], "run_signature": packet["run_signature"], "branch": args.branch,
        "supervision_class": "DERIVED_RETARGET", "state_origin": "LIVE_MODEL_STATE",
        "optimizer_updates": result["optimizer_updates"], "dataset_ref": file_ref(envelope),
        "production_manifest_ref": packet["inputs"]["production_manifest"], "training_scope": TRAINING_SCOPE,
        "training_config_ref": packet["inputs"]["training_config"], "training_result_ref": file_ref(rp),
        "code_refs": packet["runtime_code"], "source_policy": SOURCE_POLICY,
        "snapshot_map_ref": packet["inputs"]["snapshot_map"], "resume_mode": packet["resume_mode"]}
    def export_authorizer(value):
        require(value == provenance, "EXPORT_PRODUCER_BINDING_DRIFT")
        fresh = reauthorize()
        return {"status": "PASSED", "export_authorized": True, "task_id": fresh["task_id"],
            "run_signature": fresh["run_signature"], "execution_mode": "TRAINING_OWNED_LIVE_STATE",
            "gpu_lease_verified": True, "output_dir": str(output / "inference_tensors")}
    from .checkpoint_export import export_tensor_candidate
    frame = {"position_unit": "meters", "tcp_frame_ids": pair["pose_clock"]["tcp_frame_ids"],
        "pose_semantics": "DERIVED_RETARGET_ANCHOR", "action_reference_frame": "REFERENCE_CAMERA"}
    exported = export_tensor_candidate(model.state_dict(), model_config_ref=packet["inputs"]["training_config"],
        normalization_ref=file_ref(normal), provenance=provenance, output_dir=output / "inference_tensors",
        runtime_authorizer=export_authorizer, target_frame_contract=frame, state_dict_key="model")
    write_once(output / "INFERENCE_EXPORT.json", exported, root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    for name in ("task-packet", "manifest", "config", "checkpoint", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--branch", choices=("raw", "robotized"), required=True)
    args = parser.parse_args(argv)
    try:
        require(int(os.environ.get("WORLD_SIZE", "1")) == 1
            and int(os.environ.get("RANK", "0")) == int(os.environ.get("LOCAL_RANK", "0")) == 0,
            "EXACT_SINGLE_H20_PARENT_TOPOLOGY_REQUIRED")
        def reauthorize():
            return authorize(REPO_ROOT, args.task_packet, argv, manifest=args.manifest, config=args.config,
                             checkpoint=args.checkpoint, output=args.output_dir, branch=args.branch)
        packet = reauthorize()
        config_ref = packet["inputs"]["training_config"]
        config = _json(config_ref, REPO_ROOT)
        encoded = _project_path(config_ref["path"], REPO_ROOT, existing=True).read_bytes()
        require(hashlib.sha256(encoded).hexdigest() == config_ref["sha256"], "CONFIG_BYTES_DRIFT")
        return _run_training(packet, config, encoded, args, reauthorize)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error), "source_policy": SOURCE_POLICY,
            "control_ground_truth": False, "physical_deployment_authorized": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
