"""Sealed full-state successor adapter, not a standalone training entry point.

JSON admission is deliberately Torch-free. Tensor deserialization happens only
after all signed parent evidence passes; it uses a verified stable FD and a
fixed, minimal weights-only allowlist for the trainer's NumPy RNG state.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

from chaoyang.human_ego.utils.frozen_contract import (
    file_reference, open_verified_file_reference,
)

PARENT_ROLES = frozenset({"task_packet", "outer_result", "inner_result", "training_config",
                          "checkpoint", "checkpoint_sidecar"})
CONFIG_SCHEMA = "pick-place-humanego-ict-production-resume-config-v1"
INITIAL_CONFIG_SCHEMA = "pick-place-humanego-ict-production-training-config-v1"
TRAINING_SCOPE = "NON_DEPLOYABLE_DERIVED_MOTION_TRAINING"
RECEIPT_SCHEMA = "pick-place-humanego-ict-successor-resume-adaptation-v1"
CHECKPOINT_SCHEMA = "humanego-ict-revo2-resume-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MUTABLE_SIGNATURES = frozenset({"run_signature", "config", "code", "storage_capacity_observation"})


def _project_path(value, root, *, existing=False):
    path = Path(value).absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError("RESUME_PATH_MUST_BE_PROJECT_LOCAL")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("RESUME_SYMLINK_PATH_FORBIDDEN")
    resolved = path.resolve(strict=existing)
    if resolved != path:
        raise ValueError("RESUME_CANONICAL_PATH_REQUIRED")
    return path


def _read_json(reference, root, label):
    with open_verified_file_reference(reference, allowed_roots=[root], label=label) as (_, stream):
        value = json.loads(stream.read(), parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("NONFINITE_JSON_FORBIDDEN:" + value)))
    if not isinstance(value, dict):
        raise ValueError("RESUME_JSON_OBJECT_REQUIRED:" + label)
    return value


def _nondeployable(value, label, *, scope=False):
    if (value.get("control_ground_truth") is not False
            or value.get("physical_deployment_authorized") is not False
            or (scope and value.get("training_scope") != TRAINING_SCOPE)):
        raise ValueError("RESUME_SCOPE_ESCALATION_FORBIDDEN:" + label)


def validate_parent_evidence(parent_refs, *, root):
    """Verify signed sealed JSON provenance without importing Torch or NumPy.

The operation gate additionally proves PASSED/non-executable status in the
current governance ledger. This helper never infers live authorization from
historical files alone.
"""
    root = Path(root).resolve(strict=True)
    if not isinstance(parent_refs, dict) or set(parent_refs) != PARENT_ROLES:
        raise ValueError("EXACT_SEALED_PARENT_REFERENCE_ROLES_REQUIRED")
    packet = _read_json(parent_refs["task_packet"], root, "parent task packet")
    outer = _read_json(parent_refs["outer_result"], root, "parent sealed result")
    inner = _read_json(parent_refs["inner_result"], root, "parent training result")
    config = _read_json(parent_refs["training_config"], root, "parent training config")
    sidecar = _read_json(parent_refs["checkpoint_sidecar"], root, "parent checkpoint sidecar")
    task_id, run = packet.get("task_id"), packet.get("run_signature")
    if (not isinstance(task_id, str) or not task_id or not isinstance(run, str) or not _SHA.fullmatch(run)
            or any(item.get("task_id") != task_id or item.get("run_signature") != run for item in (outer, inner))):
        raise ValueError("SEALED_PARENT_TASK_SIGNATURE_MISMATCH")
    if outer.get("task_terminal_status") != "PASSED" or type(outer.get("child_exit_code")) is not int or outer["child_exit_code"] != 0:
        raise ValueError("PASSED_COMPLETE_PARENT_REQUIRED")
    branch = packet.get("branch")
    if branch not in {"raw", "robotized"} or inner.get("branch") != branch:
        raise ValueError("SEALED_PARENT_BRANCH_MISMATCH")
    for item, label, scope in ((outer, "outer result", False), (inner, "training result", True),
                               (config, "training config", True)):
        _nondeployable(item, label, scope=scope)
    if (packet.get("training_scope") != TRAINING_SCOPE
            or inner.get("supervision_class") != "DERIVED_RETARGET"
            or config.get("supervision_class") != "DERIVED_RETARGET"):
        raise ValueError("SEALED_PARENT_DERIVED_SCOPE_REQUIRED")
    if (config.get("schema_version"), config.get("initialization")) not in {
            (INITIAL_CONFIG_SCHEMA, "FRESH_SHARED_PRETRAINED_NEW_IO"),
            (CONFIG_SCHEMA, "RESUME_COMPLETE_TRAINING_STATE")}:
        raise ValueError("UNKNOWN_PARENT_PRODUCTION_CONFIG_SCHEMA")
    checkpoint = parent_refs["checkpoint"]
    path = _project_path(checkpoint["path"], root, existing=True)
    output = _project_path(packet["output_dir"], root, existing=True)
    if (path.parent != output or Path(parent_refs["checkpoint_sidecar"]["path"]) != Path(str(path) + ".ref.json")
            or Path(parent_refs["inner_result"]["path"]).parent != output
            or Path(parent_refs["outer_result"]["path"]) != output.parent / "RESULT.json"):
        raise ValueError("SEALED_PARENT_OUTPUT_IDENTITY_MISMATCH")
    if (parent_refs["training_config"] != packet.get("inputs", {}).get("training_config")
            or parent_refs["inner_result"] not in outer.get("stage_results", [])
            or checkpoint not in outer.get("complete_checkpoints", [])):
        raise ValueError("SEALED_PARENT_REFERENCE_NOT_PUBLISHED")
    step = inner.get("optimizer_updates")
    if (type(step) is not int or step <= 0 or path.name != f"checkpoint_{step:06d}.pt"
            or inner.get("last_checkpoint") != str(path)
            or type(inner.get("cursor")) is not int
            or inner["cursor"] != step * config["training"]["global_batch"]):
        raise ValueError("FINAL_SEALED_PARENT_PROGRESS_REQUIRED")
    if sidecar != {"bytes": checkpoint["bytes"], "sha256": checkpoint["sha256"]}:
        raise ValueError("SEALED_PARENT_CHECKPOINT_SIDECAR_MISMATCH")
    for item in outer["complete_checkpoints"]:
        name = Path(item["path"]).name
        if not re.fullmatch(r"checkpoint_[0-9]+\.pt", name) or int(name[11:-3]) > step:
            raise ValueError("SEALED_PARENT_FINAL_CHECKPOINT_MISMATCH")
    export_ref = outer.get("inference_export")
    if not isinstance(export_ref, dict) or Path(export_ref["path"]) != output / "INFERENCE_EXPORT.json":
        raise ValueError("SEALED_PARENT_SAFE_INFERENCE_EXPORT_REQUIRED")
    exported = _read_json(export_ref, root, "parent inference export")
    for name in ("checkpoint_ref", "export_receipt_ref", "inference_contract_ref", "model_config_ref", "normalization_ref"):
        reference = exported.get(name)
        if not isinstance(reference, dict) or Path(reference["path"]).parent != output / "inference_tensors":
            raise ValueError("PARENT_INFERENCE_EXPORT_REFERENCE_REQUIRED:" + name)
        with open_verified_file_reference(reference, allowed_roots=[root], label="parent inference export:" + name):
            pass
    export_receipt = _read_json(exported["export_receipt_ref"], root, "parent tensor export receipt")
    _nondeployable(export_receipt, "parent inference export")
    provenance = export_receipt.get("training_provenance", {})
    if (export_receipt.get("status") != "TENSOR_EXPORT_SAFE_ROUNDTRIP_VERIFIED_NOT_MODEL_QUALITY"
            or export_receipt.get("fixture_only") is not False
            or export_receipt.get("branch") != branch or export_receipt.get("optimizer_updates") != step
            or provenance.get("task_id") != task_id or provenance.get("run_signature") != run
            or provenance.get("branch") != branch or provenance.get("optimizer_updates") != step
            or provenance.get("training_scope") != TRAINING_SCOPE
            or provenance.get("training_config_ref") != parent_refs["training_config"]
            or provenance.get("training_result_ref") != parent_refs["inner_result"]):
        raise ValueError("SEALED_PARENT_INFERENCE_EXPORT_PROVENANCE_MISMATCH")
    return {"packet": packet, "outer_result": outer, "inner_result": inner,
            "training_config": config, "checkpoint_ref": checkpoint,
            "sidecar_ref": parent_refs["checkpoint_sidecar"], "branch": branch}


def _safe_load_checkpoint(reference, root):
    import numpy as np
    import torch
    # No user-defined globals and no weights_only=False fallback. The five
    # trusted classes/functions are required by the existing trainer's RNG and
    # torch version string; they are not selected by checkpoint contents.
    safe = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype("uint32")), torch.torch_version.TorchVersion]
    try:
        with open_verified_file_reference(reference, allowed_roots=[root],
                label="sealed successor full-state checkpoint") as (_, stream):
            with torch.serialization.safe_globals(safe):
                value = torch.load(stream, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("SAFE_FULL_STATE_CHECKPOINT_READ_FAILED:" + str(error)) from error
    if not isinstance(value, dict):
        raise ValueError("FULL_STATE_CHECKPOINT_MAPPING_REQUIRED")
    return value


def _config_invariants(config):
    value = json.loads(json.dumps(config, allow_nan=False))
    for key in ("schema_version", "initialization", "resume_parent_bindings"):
        value.pop(key, None)
    value["training"].pop("wall_seconds_max", None)
    return value


def _check_code_closure(parent, successor, root):
    if not isinstance(parent, list) or not parent or not isinstance(successor, list) or not successor:
        raise ValueError("NONEMPTY_RESUME_CODE_CLOSURE_REQUIRED")
    old = {ref["path"]: ref for ref in parent}
    new = {ref["path"]: ref for ref in successor}
    if len(old) != len(parent) or len(new) != len(successor) or any(new.get(path) != ref for path, ref in old.items()):
        raise ValueError("SEALED_PARENT_CODE_CLOSURE_MUST_BE_PRESERVED")
    for ref in successor:
        with open_verified_file_reference(ref, allowed_roots=[root], label="successor frozen code"):
            pass


def _check_state(state, evidence, training_config, dataset_length, world_size, device_type):
    import numpy as np
    import torch
    required = {"schema", "binding", "model", "optimizer", "ema", "optimizer_updates", "cursor",
                "rank_rng", "timing", "training_wall_seconds", "writer", "authorization_expires_at",
                "control_ground_truth", "physical_deployment_authorized"}
    if set(state) != required or state["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("EXACT_FULL_TRAINING_STATE_REQUIRED")
    _nondeployable(state, "full-state checkpoint")
    packet, inner, config = evidence["packet"], evidence["inner_result"], evidence["training_config"]
    binding = state["binding"]
    if (not isinstance(binding, dict) or set(binding) != {"config", "signatures", "topology", "dataset_length", "output_root", "environment"}
            or binding["config"] != config["training"] or binding["output_root"] != packet["output_dir"]):
        raise ValueError("PARENT_FULL_STATE_CONFIG_OUTPUT_BINDING_MISMATCH")
    signatures = binding["signatures"]
    expected = {"run_signature": packet["run_signature"], "config": packet["inputs"]["training_config"],
                "data": packet["inputs"]["production_manifest"], "prepared_pair": packet["inputs"]["prepared_pair"],
                "pretrained": packet["inputs"]["pretrained"], "pretrained_config": packet["inputs"]["pretrained_config"],
                "code": packet["runtime_code"], "branch": packet["branch"], "supervision_class": "DERIVED_RETARGET",
                "control_ground_truth": False, "training_scope": TRAINING_SCOPE,
                "storage_capacity_observation": packet["inputs"].get("storage_capacity_observation")}
    if signatures != expected:
        raise ValueError("PARENT_FULL_STATE_SOURCE_SIGNATURE_MISMATCH")
    topology = {"world_size": world_size, "micro_batch": training_config.micro_batch,
                "accumulation": training_config.global_batch // (world_size * training_config.micro_batch),
                "device_type": device_type}
    environment = {"torch": torch.__version__, "numpy": np.__version__, "cuda_build": torch.version.cuda,
                   "deterministic_algorithms": True, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "torch_threads": torch.get_num_threads()}
    if binding["topology"] != topology or inner.get("world_size") != world_size:
        raise ValueError("SUCCESSOR_TOPOLOGY_MUST_MATCH_SEALED_PARENT")
    if binding["dataset_length"] != dataset_length or type(dataset_length) is not int or dataset_length < 1:
        raise ValueError("SUCCESSOR_DATASET_LENGTH_MUST_MATCH_SEALED_PARENT")
    if binding["environment"] != environment:
        raise ValueError("SUCCESSOR_ENVIRONMENT_MUST_MATCH_SEALED_PARENT")
    step = state["optimizer_updates"]
    if (type(step) is not int or step != inner["optimizer_updates"] or not 0 < step < training_config.max_updates
            or type(state["cursor"]) is not int or state["cursor"] != inner["cursor"]
            or state["cursor"] != step * training_config.global_batch):
        raise ValueError("FULL_STATE_PROGRESS_MISMATCH_OR_ALREADY_COMPLETE")
    if (not isinstance(state["training_wall_seconds"], (int, float))
            or not math.isfinite(state["training_wall_seconds"]) or state["training_wall_seconds"] < 0
            or state["timing"] != inner.get("timing")):
        raise ValueError("INVALID_PARENT_WALL_OR_TIMING_STATE")
    model, ema, optimizer = state["model"], state["ema"], state["optimizer"]
    if not isinstance(model, dict) or not model or not isinstance(ema, dict) or set(model) != set(ema):
        raise ValueError("COMPLETE_MATCHED_MODEL_EMA_REQUIRED")
    for key in model:
        a, b = model[key], ema[key]
        if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                or a.shape != b.shape or a.dtype != b.dtype or a.device.type != "cpu" or b.device.type != "cpu"
                or (a.is_floating_point() and (not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item()))):
            raise ValueError("INVALID_MODEL_EMA_TENSOR:" + key)
    if (not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}
            or not isinstance(optimizer["state"], dict) or not optimizer["state"]
            or not isinstance(optimizer["param_groups"], list) or not optimizer["param_groups"]):
        raise ValueError("COMPLETE_ADAMW_STATE_REQUIRED")
    for entry in optimizer["state"].values():
        if not isinstance(entry, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(entry):
            raise ValueError("COMPLETE_ADAMW_MOMENTS_REQUIRED")
        if float(entry["step"]) != step:
            raise ValueError("ADAMW_UPDATE_COUNT_MISMATCH")
        for item in entry.values():
            if isinstance(item, torch.Tensor) and (item.device.type != "cpu" or not torch.isfinite(item).all().item()):
                raise ValueError("NONFINITE_OR_NONCPU_ADAMW_STATE")
    if not isinstance(state["rank_rng"], list) or len(state["rank_rng"]) != world_size:
        raise ValueError("COMPLETE_PER_RANK_RNG_REQUIRED")
    for rng in state["rank_rng"]:
        if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError("COMPLETE_PER_RANK_RNG_REQUIRED")
        try:
            random.Random().setstate(rng["python"])
            numpy_state = rng["numpy"]
            if (not isinstance(numpy_state, tuple) or len(numpy_state) != 5 or numpy_state[0] != "MT19937"
                    or not isinstance(numpy_state[1], np.ndarray) or numpy_state[1].dtype != np.dtype("uint32")
                    or numpy_state[1].shape != (624,)):
                raise ValueError("INVALID_NUMPY_MT19937_STATE")
            np.random.RandomState().set_state(numpy_state)
            torch.Generator(device="cpu").set_state(rng["torch"])
            cuda = rng["cuda"]
            if device_type == "cuda" and (not isinstance(cuda, torch.Tensor) or cuda.dtype != torch.uint8 or cuda.ndim != 1 or cuda.numel() == 0):
                raise ValueError("COMPLETE_CUDA_RNG_STATE_REQUIRED")
            if device_type == "cpu" and cuda is not None:
                raise ValueError("CPU_PARENT_MUST_NOT_CONTAIN_CUDA_RNG")
        except (TypeError, ValueError, RuntimeError, KeyError) as error:
            raise ValueError("INVALID_PARENT_RANK_RNG_STATE:" + str(error)) from error
    return binding


def _identical(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    return a == b


def prepare_successor_checkpoint(parent_refs, *, root, full_config_ref, training_config,
        signatures, output_dir, dataset_length, world_size=1, device_type="cuda"):
    """Adapt only task/path/wall binding, preserving every learned/random state.

Caller must first pass its live operation gate and source qualification. Rank
zero calls once, then the unchanged trainer resumes this locally written file.
"""
    root = Path(root).resolve(strict=True)
    evidence = validate_parent_evidence(parent_refs, root=root)
    successor_config = _read_json(full_config_ref, root, "successor training config")
    if (successor_config.get("schema_version") != CONFIG_SCHEMA
            or successor_config.get("initialization") != "RESUME_COMPLETE_TRAINING_STATE"
            or successor_config.get("resume_parent_bindings") != parent_refs):
        raise ValueError("EXPLICIT_SUCCESSOR_RESUME_CONFIG_REQUIRED")
    if _config_invariants(successor_config) != _config_invariants(evidence["training_config"]):
        raise ValueError("ONLY_FINITE_TASK_WALL_BUDGET_MAY_CHANGE_CONFIG")
    if successor_config["training"] != asdict(training_config):
        raise ValueError("SUCCESSOR_CORE_CONFIG_BINDING_MISMATCH")
    if type(world_size) is not int or world_size < 1 or device_type not in {"cpu", "cuda"}:
        raise ValueError("VALID_SUCCESSOR_TOPOLOGY_REQUIRED")
    training_config.validate(world_size)
    output = _project_path(output_dir, root)
    parent_output = Path(evidence["packet"]["output_dir"])
    if output == parent_output or output.is_relative_to(parent_output) or parent_output.is_relative_to(output):
        raise ValueError("NEW_INDEPENDENT_SUCCESSOR_OUTPUT_REQUIRED")
    output.mkdir(parents=True, exist_ok=True)
    if ((output / ".writer.lock").exists() or (output / "updates.jsonl").exists()
            or (output / "SUCCESSOR_RESUME_ADAPTATION.json").exists()
            or any(output.glob("checkpoint_*.pt*"))):
        raise ValueError("FRESH_SUCCESSOR_OUTPUT_WITHOUT_WRITER_REQUIRED")
    state = _safe_load_checkpoint(evidence["checkpoint_ref"], root)
    old_binding = _check_state(state, evidence, training_config, dataset_length, world_size, device_type)
    old_signatures = old_binding["signatures"]
    if (not isinstance(signatures, dict) or set(signatures) != set(old_signatures)
            or any(signatures[key] != value for key, value in old_signatures.items() if key not in _MUTABLE_SIGNATURES)):
        raise ValueError("SUCCESSOR_MAY_ONLY_REBIND_TASK_CONFIG_CODE_STORAGE_SIGNATURES")
    run = signatures.get("run_signature")
    if not isinstance(run, str) or not _SHA.fullmatch(run) or run == old_signatures["run_signature"]:
        raise ValueError("NEW_FINITE_TASK_RUN_SIGNATURE_REQUIRED")
    if signatures["config"] != full_config_ref:
        raise ValueError("SUCCESSOR_CONFIG_SIGNATURE_MISMATCH")
    _check_code_closure(old_signatures["code"], signatures["code"], root)
    new_binding = dict(old_binding, config=asdict(training_config),
                       signatures=json.loads(json.dumps(signatures, sort_keys=True)), output_root=str(output))
    adapted = dict(state, binding=new_binding, training_wall_seconds=0.0)
    from .training import _write_checkpoint
    from .derived_runtime import write_once
    checkpoint = output / f"checkpoint_{state['optimizer_updates']:06d}.pt"
    lock = output / ".successor_adaptation.lock"
    acquired = False
    try:
        with lock.open("x") as stream:
            acquired = True
            stream.write(str(os.getpid()))
        if (output / ".writer.lock").exists() or any(output.glob("checkpoint_*.pt*")):
            raise ValueError("SUCCESSOR_WRITER_OR_OUTPUT_APPEARED")
        _write_checkpoint(checkpoint, adapted)
        checkpoint_ref = file_reference(checkpoint)
        restored = _safe_load_checkpoint(checkpoint_ref, root)
        if not _identical(adapted, restored):
            raise ValueError("SUCCESSOR_FULL_STATE_SAFE_ROUNDTRIP_MISMATCH")
        receipt = {"schema": RECEIPT_SCHEMA, "status": "VERIFIED_COMPLETE_STATE_SUCCESSOR_NOT_NEW_TRAINING",
            "parent_refs": parent_refs, "branch": evidence["branch"],
            "parent_task_id": evidence["packet"]["task_id"], "parent_run_signature": old_signatures["run_signature"],
            "successor_run_signature": run, "optimizer_updates": state["optimizer_updates"], "cursor": state["cursor"],
            "checkpoint_ref": checkpoint_ref, "checkpoint_sidecar_ref": file_reference(Path(str(checkpoint) + ".ref.json")),
            "unchanged_state_components": sorted(set(state) - {"binding", "training_wall_seconds"}),
            "safe_tensor_roundtrip_verified": True, "unsafe_pickle_fallback": False,
            "parent_training_wall_seconds": state["training_wall_seconds"], "successor_training_wall_seconds": 0.0,
            "wall_accounting": "NEW_FINITE_TASK_BUDGET_PARENT_ELAPSED_EXPLICITLY_RETAINED_HERE",
            "parent_wall_seconds_max": old_binding["config"]["wall_seconds_max"],
            "successor_wall_seconds_max": training_config.wall_seconds_max,
            "parent_binding_sha256": hashlib.sha256(json.dumps(old_binding, sort_keys=True).encode()).hexdigest(),
            "successor_binding_sha256": hashlib.sha256(json.dumps(new_binding, sort_keys=True).encode()).hexdigest(),
            "training_scope": TRAINING_SCOPE, "control_ground_truth": False, "physical_deployment_authorized": False}
        write_once(output / "SUCCESSOR_RESUME_ADAPTATION.json", receipt, root)
        return checkpoint, receipt
    finally:
        if acquired:
            lock.unlink(missing_ok=True)
