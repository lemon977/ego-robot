"""Bounded Joint26 ICT training; data/task admission precedes Torch imports.

This is a new H30x26 research route, not the legacy H50x30 trainer and not a
pi0.5 checkpoint loader. No source preparation or NAS read is implicit here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from .project_mount_paths import reference_equal, resolve_input_path, resolve_output_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_SCHEMA = "humanego-ict-joint26-paired-manifest-v1"
CONFIG_SCHEMA = "humanego-ict-joint26-training-config-v1"
CHECKPOINT_SCHEMA = "humanego-ict-joint26-resume-v1"
KEYS = ("x_rgb", "x_ict", "ict_mask", "x_state26", "state_mask",
        "y_action", "action_valid_mask", "sample_weight")
SHAPES = ((3, 240, 320), (8, 29), (8,), (26,), (26,), (30, 26), (30, 26), ())


def project_path(value, *, exists=False, subdirectory=False):
    """Reject traversal/symlinks before reading or creating project files."""
    path = (resolve_input_path(value, PROJECT_ROOT, exists=False)
            if exists and not subdirectory else resolve_output_path(value, PROJECT_ROOT))
    if subdirectory and path == PROJECT_ROOT:
        raise ValueError("PROJECT_SUBDIRECTORY_REQUIRED")
    if exists and not path.is_file():
        raise ValueError("PROJECT_INPUT_MISSING: " + str(path))
    return path


def file_reference(path):
    path = project_path(path, exists=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _manifest_header(manifest_path):
    path = project_path(manifest_path, exists=True)
    header = json.loads(path.read_text())
    if (header.get("schema") != MANIFEST_SCHEMA or header.get("action_dim") != 26
            or header.get("horizon") != 30):
        raise ValueError("JOINT26_CACHE_REQUIRED: 必须先生成新版30步×26维缓存，旧50步×30维缓存禁止开训")
    if (header.get("control_ground_truth") is not False
            or header.get("physical_deployment_authorized") is not False):
        raise ValueError("DERIVED_RESEARCH_LABEL_FLAGS_REQUIRED")
    return path, header


def preflight(manifest_path, branch, *, world_size=None, config_path=None):
    """CPU-only negative gates, including independent source-group admission."""
    if branch not in {"raw", "robotized"}:
        raise ValueError("EXPLICIT_RGB_BRANCH_REQUIRED")
    path, _ = _manifest_header(manifest_path)
    # The cache reader imports only NumPy/PIL, never Torch or a model.
    from .joint26_data import load_joint26_cache
    train, train_qualification = load_joint26_cache(path, branch, split="train")
    validation, validation_qualification = load_joint26_cache(path, branch, split="validation")
    for qualification in (train_qualification, validation_qualification):
        if qualification.get("data_gate", {}).get("status") != "PASSED_RESEARCH_LABEL_GATE":
            raise ValueError("BLOCKED_DATA_VOLUME: 需要至少16个训练及3个验证独立采集组，不能以会话数或补帧充数")
        if (qualification.get("control_ground_truth") is not False
                or qualification.get("physical_deployment_authorized") is not False
                or qualification.get("supervision_class") != "DERIVED_RETARGET_ASSUMED_MODEL_JOINTS"):
            raise ValueError("JOINT26_RESEARCH_SUPERVISION_REQUIRED")
    groups = train_qualification["source_group_splits"]
    train_groups, validation_groups = set(groups.get("train", [])), set(groups.get("validation", []))
    if len(train_groups) < 16 or len(validation_groups) < 3 or train_groups & validation_groups:
        raise ValueError("BLOCKED_DATA_VOLUME: 训练/验证组不足或交叉")
    if (validation_qualification.get("source_group_splits") != groups
            or train_qualification.get("normalization") != validation_qualification.get("normalization")
            or not len(train) or not len(validation)):
        raise ValueError("JOINT26_SPLIT_BINDING_MISMATCH")
    train.verify_all_dependencies(); validation.verify_all_dependencies()
    if config_path is not None:
        config, _ = read_config(config_path)
        config.validate(world_size or 1, production=True)
    summary = {"schema": "humanego-ict-joint26-preflight-v1", "status": "PASS_RESEARCH_DATA_ONLY",
               "manifest_ref": file_reference(path), "branch": branch, "action_dim": 26,
               "horizon": 30, "training_samples": len(train), "validation_samples": len(validation),
               "training_groups": len(train_groups), "validation_groups": len(validation_groups),
               "source_group_overlap": 0, "normalization": train_qualification["normalization"],
               "numerical_limits": train_qualification.get("numerical_limits"),
               "native_time_support": train_qualification.get("native_time_support"),
               "source_policy": train_qualification.get("source_policy"),
               "supervision_class": train_qualification["supervision_class"],
               "control_ground_truth": False, "physical_deployment_authorized": False,
               "torch_imported": "torch" in sys.modules}
    return train, validation, summary


@dataclass(frozen=True)
class Joint26TrainingConfig:
    global_batch: int = 64
    micro_batch: int = 8
    max_updates: int = 20_000
    precision: str = "float32"
    seed: int = 7
    learning_rate: float = 1e-4
    warmup_updates: int = 200
    ema_decay: float = 0.999
    checkpoint_every: int = 1000
    validation_every: int = 1000
    grad_clip: float = 1.0
    wall_seconds_max: float | None = None
    log_every: int = 20

    def validate(self, world_size, *, production=False):
        integers = (self.global_batch, self.micro_batch, self.max_updates,
                    self.checkpoint_every, self.validation_every, self.log_every, world_size)
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("FINITE_POSITIVE_INTEGER_TRAINING_BUDGET_REQUIRED")
        if self.global_batch % (world_size * self.micro_batch):
            raise ValueError("GLOBAL_BATCH_TOPOLOGY_MISMATCH: global_batch须整除GPU数×micro_batch")
        if production and self.global_batch != 64:
            raise ValueError("JOINT26_PRODUCTION_GLOBAL64_REQUIRED")
        if self.precision not in {"float32", "bf16"}:
            raise ValueError("EXPLICIT_FLOAT32_OR_BF16_REQUIRED")
        if (type(self.seed) is not int or type(self.warmup_updates) is not int
                or self.warmup_updates < 0 or not 0 <= self.ema_decay < 1
                or any(not math.isfinite(value) or value <= 0 for value in
                       (self.learning_rate, self.grad_clip))):
            raise ValueError("INVALID_JOINT26_TRAINING_NUMERICS")
        if self.wall_seconds_max is not None and (
                type(self.wall_seconds_max) not in {int, float}
                or not math.isfinite(self.wall_seconds_max) or self.wall_seconds_max <= 0):
            raise ValueError("INVALID_JOINT26_TRAINING_WALL_BUDGET")


def read_config(path):
    path = project_path(path, exists=True)
    document = json.loads(path.read_text())
    if document.get("schema") != CONFIG_SCHEMA:
        raise ValueError("JOINT26_TRAINING_CONFIG_SCHEMA_REQUIRED")
    training = document.get("training")
    if not isinstance(training, dict) or set(training) - {field.name for field in fields(Joint26TrainingConfig)}:
        raise ValueError("UNKNOWN_JOINT26_TRAINING_CONFIG_FIELDS")
    config = Joint26TrainingConfig(**training)
    if not isinstance(document.get("model"), dict):
        raise ValueError("FROZEN_JOINT26_MODEL_CONFIG_REQUIRED")
    model = document["model"]
    if (model.get("horizon") != 30 or model.get("action_dim") != 26
            or model.get("ict_dim") != 29 or model.get("image_size") != [240, 320]):
        raise ValueError("JOINT26_MODEL_CONFIG_SHAPE_MISMATCH")
    return config, document


def global_sample_indices(length, cursor, count, seed):
    """One deterministic global permutation, then disjoint rank shards."""
    if min(length, count) < 1 or cursor < 0:
        raise ValueError("NONEMPTY_DATASET_AND_VALID_CURSOR_REQUIRED")
    import torch
    indices = []
    while len(indices) < count:
        epoch, offset = divmod(cursor + len(indices), length)
        order = torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        indices.extend(order[offset:offset + count - len(indices)])
    return indices


def masked_joint26_flow_statistics(prediction, target, valid, sample_weight):
    """Equal-channel loss; unlike legacy modality-major 30D wrist weights."""
    import torch
    if (prediction.shape != target.shape or valid.shape != target.shape
            or target.ndim != 3 or target.shape[-2:] != (30, 26)):
        raise ValueError("JOINT26_FLOW_REQUIRES_MATCHING_B_H30_D26")
    valid = valid.bool()
    weights = sample_weight.float().reshape(-1, 1, 1)
    if weights.shape[0] != target.shape[0]:
        raise ValueError("JOINT26_SAMPLE_WEIGHT_BATCH_MISMATCH")
    if not bool(torch.isfinite(weights).all()) or bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("INVALID_JOINT26_SAMPLE_WEIGHT")
    clean_prediction, clean_target = torch.where(valid, prediction.float(), 0), torch.where(valid, target.float(), 0)
    effective = valid.float() * weights / 30
    return ((clean_prediction - clean_target).square() * effective).sum(), effective.sum()


def _json_blob(document):
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _immutable_bytes(path, blob):
    path = project_path(path)
    if path.exists():
        if path.read_bytes() == blob:
            return
        raise ValueError("IMMUTABLE_OUTPUT_CONFLICT: " + str(path))
    with path.open("xb") as stream:
        stream.write(blob); stream.flush(); os.fsync(stream.fileno())


def _checkpoint(path, payload):
    import torch
    if path.exists():
        raise ValueError("IMMUTABLE_CHECKPOINT_ALREADY_EXISTS")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication that cannot overwrite another writer.
        _immutable_bytes(Path(str(path) + ".ref.json"), _json_blob(file_reference(path)))
    finally:
        temporary.unlink(missing_ok=True)


def run_training(model, train_dataset, validation_dataset, config, output_dir, *,
                 signatures, model_config, normalization, inference_contract,
                 device="cuda", resume=None, stop_after_updates=None,
                 authorization_expires_at=None, refresh_authorization=None, initializer=None):
    """Called only after the owner gate. CPU is reserved for synthetic unit tests.

    Update counts are real optimizer updates. Resumes bind topology, data,
    configuration and sampler cursor; they cannot silently switch GPU count.
    """
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
    from contextlib import nullcontext

    device = torch.device(device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    owned_group = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        owned_group = True
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    config.validate(world)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    output = project_path(output_dir, subdirectory=True)
    topology = {"world_size": world, "micro_batch": config.micro_batch,
                "accumulation": config.global_batch // (world * config.micro_batch), "device": device.type}
    binding = {"training_config": asdict(config), "model_config": model_config,
               "normalization": normalization, "inference_contract": inference_contract,
               "signatures": signatures, "topology": topology, "train_length": len(train_dataset),
               "validation_length": len(validation_dataset), "output": str(output),
               "environment": {"torch": torch.__version__, "numpy": np.__version__,
                    "cuda": torch.version.cuda, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "deterministic_algorithms": True, "torch_threads": torch.get_num_threads()}}

    def synchronize_error(error):
        errors = [str(error) if error else None]
        if world > 1:
            errors = [None] * world; dist.all_gather_object(errors, str(error) if error else None)
        if any(errors):
            raise RuntimeError("; ".join(value for value in errors if value))

    def checked(action):
        value, error = None, None
        try:
            value = action()
        except Exception as exc:
            error = exc
        synchronize_error(error)
        return value

    def rng_state():
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}

    def restore_rng(value):
        random.setstate(value["python"]); np.random.set_state(value["numpy"])
        torch.set_rng_state(value["torch"].cpu())
        if value["cuda"] is not None:
            torch.cuda.set_rng_state(value["cuda"].cpu(), device)

    def batch(indices, dataset):
        rows = [dataset[index] for index in indices]
        for row in rows:
            for key, shape in zip(KEYS, SHAPES):
                if np.shape(row[key]) != shape:
                    raise ValueError("JOINT26_DATASET_SHAPE_MISMATCH: " + key)
        result = {key: torch.stack([torch.as_tensor(row[key]).clone() for row in rows]) for key in KEYS}
        for key in ("x_rgb", "x_ict", "x_state26"):
            if not bool(torch.isfinite(result[key]).all()):
                raise ValueError("NONFINITE_JOINT26_CONDITIONING: " + key)
        valid = result["action_valid_mask"].bool()
        if not bool(torch.isfinite(result["y_action"][valid]).all()):
            raise ValueError("NONFINITE_ADMITTED_JOINT26_ACTION")
        result["sample_weight"] = result["sample_weight"].float().reshape(-1)
        return result

    class FlowLoss(nn.Module):
        def __init__(self, network):
            super().__init__(); self.model = network

        def forward(self, values, denominator):
            valid = values["action_valid_mask"].bool()
            action = torch.where(valid, values["y_action"], 0.0)
            noise = torch.where(valid, torch.randn_like(action), 0.0)
            t = torch.rand((action.shape[0],), device=device)
            mix = t.view(-1, 1, 1)
            prediction = self.model(**{key: values[key] for key in KEYS[:5]},
                x_t=(1 - mix) * noise + mix * action, t=t)["v_pred"]
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError("NONFINITE_JOINT26_FLOW_PREDICTION")
            numerator, _ = masked_joint26_flow_statistics(prediction, action - noise,
                                                        valid, values["sample_weight"])
            return numerator * world / denominator, numerator.detach()

    lock = None
    try:
        if world > 1:
            bindings = [None] * world; dist.all_gather_object(bindings, binding)
            synchronize_error(None if all(item == binding for item in bindings) else "RANK_BINDING_MISMATCH")
        if config.precision == "bf16" and device.type == "cuda":
            synchronize_error(None if torch.cuda.is_bf16_supported() else "REQUESTED_BF16_NOT_SUPPORTED_NO_FALLBACK")
        if not signatures or not len(train_dataset) or not len(validation_dataset):
            raise ValueError("SIGNED_NONEMPTY_TRAIN_AND_VALIDATION_REQUIRED")
        writer = None

        def acquire():
            nonlocal lock, writer
            if rank != 0:
                return
            output.mkdir(parents=True, exist_ok=True)
            lock_path = output / ".writer.lock"
            with lock_path.open("x") as stream:
                lock = lock_path
                if resume is None and ((output / "updates.jsonl").exists() or any(output.glob("checkpoint_*.pt"))):
                    raise ValueError("EXISTING_TRAINING_PROGRESS_REQUIRES_EXPLICIT_RESUME")
                proc = Path("/proc/self/stat").read_text()
                writer = {"pid": int(proc.split()[0]), "proc_start_ticks": int(proc.rsplit(")", 1)[1].split()[19]),
                          "executor_epoch": time.time_ns(), "fencing_token": os.urandom(16).hex(),
                          "writer_rank": 0, "run_signature": hashlib.sha256(_json_blob(binding)).hexdigest()}
                stream.write(json.dumps(writer))
            for name, document in (("model_config.json", model_config), ("training_config.json", asdict(config)),
                    ("normalization.json", normalization), ("inference_contract.json", inference_contract),
                    ("run_binding.json", binding)):
                _immutable_bytes(output / name, _json_blob(document))

        checked(acquire)
        random.seed(config.seed + rank); np.random.seed(config.seed + rank); torch.manual_seed(config.seed + rank)
        model = model.to(device)
        initialization = None
        if initializer is not None and resume is None:
            initialization = checked(lambda: initializer(model) if rank == 0 else None)
        loss_module = FlowLoss(model).to(device)
        if world > 1:
            loss_module = DistributedDataParallel(loss_module,
                device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                      lr=config.learning_rate)
        ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
        step, cursor, elapsed_before = 0, 0, 0.0
        last_checkpoint = None
        validation_loss = None
        steady_seconds, steady_count = 0.0, 0
        if resume is not None:
            def load_resume():
                path = project_path(resume, exists=True)
                if path.parent != output:
                    raise ValueError("SAME_OUTPUT_RESUME_REQUIRED")
                reference = json.loads(project_path(str(path) + ".ref.json", exists=True).read_text())
                if not reference_equal(file_reference(path), reference, PROJECT_ROOT):
                    raise ValueError("RESUME_SHA_DRIFT")
                state = torch.load(path, map_location=device, weights_only=False)
                if state.get("schema") != CHECKPOINT_SCHEMA or state.get("binding") != binding:
                    raise ValueError("JOINT26_RESUME_SCHEMA_OR_BINDING_MISMATCH")
                update = state["optimizer_updates"]
                if (type(update) is not int or not 0 <= update <= config.max_updates
                        or state["cursor"] != update * config.global_batch or len(state["rank_rng"]) != world):
                    raise ValueError("RESUME_PROGRESS_OR_RNG_MISMATCH")
                if any(int(path.stem.removeprefix("checkpoint_")) > update for path in output.glob("checkpoint_*.pt")):
                    raise ValueError("NEWER_CHECKPOINT_PRESENT")
                if (output / "updates.jsonl").exists():
                    with (output / "updates.jsonl").open() as stream:
                        if any(json.loads(line)["optimizer_update"] > update for line in stream):
                            raise ValueError("UPDATE_LOG_AHEAD_OF_RESUME")
                return state
            state = checked(load_resume)
            model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
            ema, step, cursor = state["ema"], state["optimizer_updates"], state["cursor"]
            elapsed_before = state["training_wall_seconds"]
            steady_seconds, steady_count = state["timing"]["seconds"], state["timing"]["updates"]
            validation_loss, initialization = state.get("validation_flow_loss"), state.get("initialization")
            restore_rng(state["rank_rng"][rank]); last_checkpoint = str(project_path(resume, exists=True))
        started = time.monotonic()
        refreshed_at = started
        deadline_epoch = authorization_expires_at

        def deadline(elapsed, *, force_refresh=False):
            nonlocal deadline_epoch, refreshed_at
            needs = torch.tensor(int(force_refresh or time.monotonic() - refreshed_at >= 60), device=device)
            if world > 1:
                dist.all_reduce(needs, op=dist.ReduceOp.MAX)
            if refresh_authorization is not None and bool(needs):
                renewed = checked(refresh_authorization)
                if renewed is not None:
                    deadline_epoch = float(renewed)
                refreshed_at = time.monotonic()
            remaining = torch.tensor([
                float("inf") if config.wall_seconds_max is None else config.wall_seconds_max - elapsed,
                float("inf") if deadline_epoch is None else deadline_epoch - time.time()],
                dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(remaining, op=dist.ReduceOp.MIN)
            if min(remaining.tolist()) <= 0:
                return "AUTHORIZATION_DEADLINE" if remaining[1] <= remaining[0] else "WALL_BUDGET"
            return None

        def seal(elapsed):
            nonlocal last_checkpoint
            path = output / f"checkpoint_{step:06d}.pt"
            if step == 0 or str(path) == last_checkpoint:
                return
            deadline(elapsed, force_refresh=True)
            states = [rng_state()]
            if world > 1:
                states = [None] * world; dist.all_gather_object(states, rng_state())
            def write():
                if rank == 0:
                    _checkpoint(path, {"schema": CHECKPOINT_SCHEMA, "binding": binding,
                        "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
                        "requested_optimizer_updates": config.max_updates,
                        "optimizer_updates": step, "cursor": cursor, "rank_rng": states,
                        "model_config": model_config, "normalization": normalization,
                        "inference_contract": inference_contract, "writer": writer,
                        "timing": {"seconds": steady_seconds, "updates": steady_count},
                        "training_wall_seconds": elapsed, "validation_flow_loss": validation_loss,
                        "initialization": initialization,
                        "control_ground_truth": False, "physical_deployment_authorized": False})
            checked(write); last_checkpoint = str(path)

        def validate():
            saved = rng_state(); model.eval()
            torch.manual_seed(config.seed + 100_000 + rank)
            totals = torch.zeros(2, device=device)
            error = None
            try:
                with torch.no_grad():
                    indices = list(range(rank, len(validation_dataset), world))
                    for begin in range(0, len(indices), config.micro_batch):
                        values = batch(indices[begin:begin + config.micro_batch], validation_dataset)
                        values = {key: value.to(device) for key, value in values.items()}
                        valid = values["action_valid_mask"].bool()
                        action = torch.where(valid, values["y_action"], 0.0)
                        noise = torch.where(valid, torch.randn_like(action), 0.0)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
                            prediction = model(**{key: values[key] for key in KEYS[:5]},
                                x_t=(noise + action) / 2, t=torch.full((len(action),), 0.5, device=device))["v_pred"]
                        if not bool(torch.isfinite(prediction).all()):
                            raise ValueError("NONFINITE_JOINT26_VALIDATION_PREDICTION")
                        numerator, denominator = masked_joint26_flow_statistics(prediction, action - noise, valid,
                                                                                values["sample_weight"])
                        totals += torch.stack((numerator, denominator))
            except Exception as exc:
                error = exc
            restore_rng(saved); model.train()
            synchronize_error(error)  # Error exchange before totals collective, never after a rank exits early.
            if world > 1:
                dist.all_reduce(totals)
            if not bool(torch.isfinite(totals).all()) or not bool(totals[1] > 0):
                raise ValueError("INVALID_JOINT26_VALIDATION_FLOW_LOSS")
            return float(totals[0] / totals[1])

        model.train()
        reason = "UPDATE_BUDGET" if step == config.max_updates else "GRACEFUL_STOP"
        while step < config.max_updates and (stop_after_updates is None or step < stop_after_updates):
            step_started = time.monotonic()
            elapsed = elapsed_before + step_started - started
            if reason_ := deadline(elapsed):
                reason = reason_; seal(elapsed); break
            indices = global_sample_indices(len(train_dataset), cursor, config.global_batch, config.seed)
            count = config.global_batch // world
            local = indices[rank * count:(rank + 1) * count]
            batches = checked(lambda: [batch(local[start:start + config.micro_batch], train_dataset)
                                       for start in range(0, count, config.micro_batch)])
            denominator = sum(masked_joint26_flow_statistics(torch.zeros_like(values["y_action"], device=device),
                values["y_action"].to(device), values["action_valid_mask"].to(device),
                values["sample_weight"].to(device))[1] for values in batches)
            if world > 1:
                dist.all_reduce(denominator)
            synchronize_error(None if bool(denominator > 0) else "NO_GLOBALLY_VALID_JOINT26_ACTION")
            before_rng = rng_state()
            before_buffers = {key: value.detach().clone() for key, value in model.named_buffers()}
            before_lr = [group["lr"] for group in optimizer.param_groups]
            lr = config.learning_rate * min(1.0, (step + 1) / max(1, config.warmup_updates))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            numerator = torch.zeros((), device=device)
            for index, values in enumerate(batches):
                sync = loss_module.no_sync() if world > 1 and index < len(batches) - 1 else nullcontext()
                with sync:
                    values = {key: value.to(device) for key, value in values.items()}
                    def forward_loss():
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
                            return loss_module(values, denominator)
                    loss, local_numerator = checked(forward_loss)
                    synchronize_error(None if bool(torch.isfinite(loss)) else "NONFINITE_JOINT26_FLOW_LOSS")
                    loss.backward(); numerator += local_numerator
            synchronize_error(None if all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
                if parameter.grad is not None) else "NONFINITE_GRADIENT_NO_OPTIMIZER_UPDATE")
            elapsed = elapsed_before + time.monotonic() - started
            if reason_ := deadline(elapsed):
                optimizer.zero_grad(set_to_none=True); restore_rng(before_rng)
                with torch.no_grad():
                    for key, value in model.named_buffers():
                        value.copy_(before_buffers[key])
                for group, value in zip(optimizer.param_groups, before_lr):
                    group["lr"] = value
                reason = reason_; seal(elapsed); break
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step(); step += 1; cursor += config.global_batch
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    if value.is_floating_point():
                        ema[key].lerp_(value, 1 - config.ema_decay)
                    else:
                        ema[key].copy_(value)
            optimizer.zero_grad(set_to_none=True)
            if world > 1:
                dist.all_reduce(numerator)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = torch.tensor(time.monotonic() - step_started, device=device)
            elapsed_tensor = torch.tensor(elapsed_before + time.monotonic() - started, device=device)
            if world > 1:
                dist.all_reduce(duration, op=dist.ReduceOp.MAX); dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor)
            if step > 20:
                steady_seconds += float(duration); steady_count += 1
            if step % config.validation_every == 0 or step == config.max_updates or step == stop_after_updates:
                validation_loss = checked(validate)
                # Validation is real elapsed work, not excluded from the time box.
                elapsed_tensor = torch.tensor(elapsed_before + time.monotonic() - started,
                                              dtype=torch.float64, device=device)
                if world > 1:
                    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                elapsed = float(elapsed_tensor)
            reason = deadline(elapsed) or ("UPDATE_BUDGET" if step == config.max_updates else "GRACEFUL_STOP")
            if (step % config.checkpoint_every == 0 or step == config.max_updates
                    or step == stop_after_updates or reason not in {"UPDATE_BUDGET", "GRACEFUL_STOP"}):
                checked(lambda: (train_dataset.verify_all_dependencies(), validation_dataset.verify_all_dependencies()))
                seal(elapsed)
            if step % config.log_every == 0 or step == config.max_updates or step == stop_after_updates:
                row = {"optimizer_update": step, "cursor": cursor, "flow_loss": float(numerator / denominator),
                       "validation_flow_loss": validation_loss, "learning_rate": lr, "step_seconds": float(duration),
                       "steady_step_seconds": steady_seconds / steady_count if steady_count else None,
                       "remaining_hours": (config.max_updates - step) * steady_seconds / steady_count / 3600 if steady_count else None,
                       "precision": config.precision, "world_size": world, "device_type": device.type}
                def log():
                    if rank != 0:
                        return
                    with (output / "updates.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    path = output / "loss.csv"
                    existed = path.exists()
                    with path.open("a", newline="") as stream:
                        writer_csv = csv.DictWriter(stream, fieldnames=list(row))
                        if not existed:
                            writer_csv.writeheader()
                        writer_csv.writerow(row)
                    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
                checked(log)
            if reason not in {"UPDATE_BUDGET", "GRACEFUL_STOP"}:
                break
        return {"schema": "humanego-ict-joint26-training-result-v1",
                "requested_optimizer_updates": config.max_updates, "optimizer_updates": step,
                "cursor": cursor, "completed_update_budget": step == config.max_updates,
                "termination_reason": reason, "last_checkpoint": last_checkpoint,
                "validation_flow_loss": validation_loss, "world_size": world,
                "control_ground_truth": False, "physical_deployment_authorized": False}
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)
        if owned_group:
            dist.destroy_process_group()


def _authorization_deadline(packet):
    if packet.get("admitted_authorization_expires_at") is not None:
        return float(packet["admitted_authorization_expires_at"])
    values = []
    if packet.get("deadline_at") is not None:
        values.append(datetime.fromisoformat(packet["deadline_at"]).timestamp())
    if packet.get("gpu_lease_path"):
        lease = json.loads(project_path(packet["gpu_lease_path"], exists=True).read_text())
        values.append(datetime.fromisoformat(lease["expires_at"]).timestamp())
    if not values:
        raise ValueError("JOINT26_AUTHORIZATION_EXPIRY_REQUIRED")
    return min(values)


def _completed_export_budget(state, expected_updates):
    """Bind export completion to the signed config, not a requested filename."""
    if type(expected_updates) is not int or expected_updates < 1:
        raise ValueError("FINITE_POSITIVE_INTEGER_EXPORT_BUDGET_REQUIRED")
    if (not isinstance(state, dict) or state.get("schema") != CHECKPOINT_SCHEMA
            or state.get("control_ground_truth") is not False
            or state.get("physical_deployment_authorized") is not False):
        raise ValueError("JOINT26_SIGNED_RESEARCH_CHECKPOINT_REQUIRED")
    binding = state.get("binding")
    configuration = binding.get("training_config") if isinstance(binding, dict) else None
    if not isinstance(configuration, dict):
        raise ValueError("SIGNED_JOINT26_TRAINING_CONFIG_REQUIRED")
    target, global_batch = configuration.get("max_updates"), configuration.get("global_batch")
    actual, cursor = state.get("optimizer_updates"), state.get("cursor")
    requested = state.get("requested_optimizer_updates", target)
    if (type(target) is not int or target < 1 or type(global_batch) is not int or global_batch != 64
            or type(actual) is not int or type(cursor) is not int or type(requested) is not int
            or target != expected_updates or actual != target or requested != target
            or cursor != target * global_batch):
        raise ValueError("EXACT_CONFIGURED_JOINT26_UPDATE_BUDGET_REQUIRED")
    return target


def export_final_candidate(checkpoint_path, export_dir, *, expected_updates):
    """Export a signed, completed MODEL, never EMA or optimizer-bearing tensors.

    Source checkpoint is trusted only after its local SHA sidecar is verified.
    Its frozen target must equal the caller's authorized configured budget.
    Export manifests remain explicit research candidates, not physical approval.
    """
    if type(expected_updates) is not int or expected_updates < 1:
        raise ValueError("FINITE_POSITIVE_INTEGER_EXPORT_BUDGET_REQUIRED")
    import torch
    source = project_path(checkpoint_path, exists=True)
    checkpoint_ref = file_reference(source)
    source_ref = json.loads(project_path(str(source) + ".ref.json", exists=True).read_text())
    if not reference_equal(source_ref, checkpoint_ref, PROJECT_ROOT):
        raise ValueError("EXPORT_SOURCE_CHECKPOINT_SHA_DRIFT")
    state = torch.load(source, map_location="cpu", weights_only=False)
    target = _completed_export_budget(state, expected_updates)
    contract = state["inference_contract"]
    if contract.get("action_dim") != 26 or contract.get("pred_horizon") != 30:
        raise ValueError("JOINT26_EXPORT_CONTRACT_SHAPE_MISMATCH")
    model_state = state["model"]
    if (not isinstance(model_state, dict) or not model_state
            or any(not isinstance(value, torch.Tensor) for value in model_state.values())
            or any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in model_state.values())):
        raise ValueError("FINITE_FLAT_MODEL_TENSORS_REQUIRED")
    output = project_path(export_dir, subdirectory=True)
    if (output / "EXPORT_MANIFEST.json").exists():
        previous = json.loads((output / "EXPORT_MANIFEST.json").read_text())
        if (previous.get("checkpoint_ref") is None
                or not reference_equal(previous["checkpoint_ref"], checkpoint_ref, PROJECT_ROOT)):
            raise ValueError("IMMUTABLE_EXPORT_SOURCE_CONFLICT")
        if (previous.get("schema") != "humanego-ict-joint26-tensor-export-manifest-v1"
                or type(previous.get("requested_optimizer_updates")) is not int
                or type(previous.get("optimizer_updates")) is not int
                or previous["requested_optimizer_updates"] != target
                or previous["optimizer_updates"] != target
                or previous.get("action_dim") != 26 or previous.get("horizon") != 30
                or previous.get("control_ground_truth") is not False
                or previous.get("physical_deployment_authorized") is not False
                or set(previous.get("files", {})) != {"tensor", "model_config", "normalization", "inference_contract"}):
            raise ValueError("IMMUTABLE_EXPORT_BUDGET_OR_CONTRACT_CONFLICT")
        for reference in previous.get("files", {}).values():
            if not reference_equal(file_reference(reference["path"]), reference, PROJECT_ROOT):
                raise ValueError("IMMUTABLE_EXPORT_BYTES_DRIFT")
        return previous
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("model_config.json", state["model_config"]),
            ("normalization.json", state["normalization"]), ("inference_contract.json", contract)):
        _immutable_bytes(output / name, _json_blob(payload))
    _checkpoint(output / "candidate_tensor.pt", {key: value.detach().cpu() for key, value in model_state.items()})
    manifest = {"schema": "humanego-ict-joint26-tensor-export-manifest-v1",
        "files": {"tensor": file_reference(output / "candidate_tensor.pt"),
                  "model_config": file_reference(output / "model_config.json"),
                  "normalization": file_reference(output / "normalization.json"),
                  "inference_contract": file_reference(output / "inference_contract.json")},
        "checkpoint_ref": checkpoint_ref, "source_state_dict_key": "model", "model_source": "MODEL_NOT_EMA",
        "requested_optimizer_updates": target, "optimizer_updates": target,
        "action_dim": 26, "horizon": 30,
        "supervision_class": "DERIVED_RETARGET_ASSUMED_MODEL_JOINTS",
        "control_ground_truth": False, "physical_deployment_authorized": False, "pi05_weight_compatible": False}
    _immutable_bytes(output / "EXPORT_MANIFEST.json", _json_blob(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="HumanEgo ICT 新版关节输出：CPU预检后方可训练")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("preflight", "train"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--manifest", required=True)
        subparser.add_argument("--branch", choices=("raw", "robotized"), required=True)
        if name == "train":
            subparser.add_argument("--task-packet", required=True)
            subparser.add_argument("--config", required=True)
            subparser.add_argument("--output", required=True)
            subparser.add_argument("--resume")
    args = parser.parse_args(argv)
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        packet = None
        if args.mode == "train":
            from .joint26_runtime import authorize_joint26_training
            packet = authorize_joint26_training(args.task_packet, args.manifest, args.config,
                args.output, args.branch, world, rank=int(os.environ.get("RANK", "0")))
        train, validation, summary = preflight(args.manifest, args.branch,
            world_size=world, config_path=args.config if args.mode == "train" else None)
        if args.mode == "preflight":
            print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
        config, document = read_config(args.config)
        config.validate(world, production=True)
        source_ref = document.get("warmstart_ref")
        if source_ref is not None and not reference_equal(file_reference(source_ref["path"]), source_ref, PROJECT_ROOT):
            raise ValueError("SHARED_LAYER_WARMSTART_SHA_DRIFT")
        # All negative data/governance gates above are standard library / CPU.
        from .runtime_gate import bind_cuda_visible_devices
        bind_cuda_visible_devices(packet)
        import torch
        from .joint26_model import build_joint26_model, warmstart_shared_layers
        from .joint26_contract import interface_contract
        torch.manual_seed(config.seed)
        model = build_joint26_model(document["model"])
        contract = dict(interface_contract(), schema="humanego-ict-joint26-inference-contract-v1",
            model_config=document["model"], normalization=summary["normalization"], state_dict_key="model",
            training_manifest_ref=summary["manifest_ref"], source_policy=summary["source_policy"],
            supervision_class=summary["supervision_class"], output_shape=[30, 26],
            input_rgb_shape=[3, 240, 320], input_ict_shape=[8, 29], input_state_shape=[26],
            hand_boundary_policy="REJECT_WITHOUT_SILENT_CLAMP", time_contract="PER_SAMPLE_TARGET_QPC_SOURCE_MAPPING",
            pi05_loader_compatible=False, current_robot_state_required=True,
            deployment_gate="OFFLINE_REPLAY_AND_HARDWARE_ADAPTATION_REQUIRED")
        contract["numerical_limits"] = train.manifest["numerical_limits"]
        contract["native_time_support"] = summary["native_time_support"]
        contract["rgb_range"] = [0.0, 1.0]
        contract["rgb_preprocessing"] = "PIL_RGB_RESIZE_320_240_DIVIDE_255_CHW_FLOAT32"
        initializer = None
        if source_ref is not None:
            def initializer(network):
                source_path = project_path(source_ref["path"], exists=True)
                state = torch.load(source_path, map_location="cpu", weights_only=True)
                if not reference_equal(file_reference(source_path), source_ref, PROJECT_ROOT):
                    raise ValueError("SHARED_LAYER_WARMSTART_CHANGED_DURING_LOAD")
                return warmstart_shared_layers(network, state, source_ref=source_ref)
        def refresh():
            renewed = authorize_joint26_training(args.task_packet, args.manifest, args.config,
                args.output, args.branch, world, rank=int(os.environ.get("RANK", "0")))
            return _authorization_deadline(renewed)
        result = run_training(model, train, validation, config, args.output,
            signatures={"manifest": summary["manifest_ref"], "config": file_reference(args.config),
                        "run_signature": packet["run_signature"]},
            model_config=document["model"], normalization=summary["normalization"], inference_contract=contract,
            authorization_expires_at=_authorization_deadline(packet), refresh_authorization=refresh,
            initializer=initializer, resume=args.resume)
        if result["completed_update_budget"] and int(os.environ.get("RANK", "0")) == 0:
            result["tensor_export"] = export_final_candidate(result["last_checkpoint"],
                project_path(args.output, subdirectory=True) / "inference_tensors",
                expected_updates=config.max_updates)
            result["inference_export_ref"] = file_reference(project_path(args.output) /
                "inference_tensors" / "EXPORT_MANIFEST.json")
        if int(os.environ.get("RANK", "0")) == 0:
            result.update(task_id=packet["task_id"], run_signature=packet["run_signature"], branch=args.branch)
            _immutable_bytes(project_path(args.output) / "TRAINING_RESULT.json", _json_blob(result))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["completed_update_budget"] else 3
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as error:
        print("训练未启动或已安全停止：" + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
