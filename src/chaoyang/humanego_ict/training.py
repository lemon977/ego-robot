"""Step-bounded candidate trainer. The caller must pass the external data/task gate."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TrainingConfig:
    global_batch: int = 64
    micro_batch: int = 8
    max_updates: int = 20_000
    precision: str = "bf16"
    seed: int = 7
    learning_rate: float = 1e-4
    warmup_updates: int = 200
    ema_decay: float = 0.999
    checkpoint_every: int = 500
    grad_clip: float = 1.0
    position_weight: float = 2.0
    rotation_weight: float = 1.0
    hand_weight: float = 5.0
    wall_seconds_max: float = 86_400.0
    timing_warmup_updates: int = 20
    log_every: int = 20

    def validate(self, world_size):
        if self.precision not in {"bf16", "float32"}:
            raise ValueError("Only bf16 and float32 supported; FP16 skip accounting is not implemented")
        if min(self.global_batch, self.micro_batch, self.max_updates, self.checkpoint_every, self.log_every) < 1:
            raise ValueError("Batch sizes, update budget and checkpoint interval must be positive")
        if self.global_batch % (world_size * self.micro_batch):
            raise ValueError("global_batch must equal world_size * micro_batch * accumulation")
        numbers = [self.learning_rate, self.grad_clip, self.position_weight,
                   self.rotation_weight, self.hand_weight]
        if any(not np.isfinite(n) or n <= 0 for n in numbers):
            raise ValueError("Learning rate, clipping and modality weights must be finite and positive")
        if (not 0 <= self.ema_decay < 1 or self.warmup_updates < 0 or self.timing_warmup_updates < 0
                or not np.isfinite(self.wall_seconds_max) or self.wall_seconds_max <= 0):
            raise ValueError("Invalid EMA decay or warmup")


def masked_flow_statistics(prediction, target, valid, sample_weight, modality_weights=None):
    """Differentiable numerator and target-only denominator, including local zero-valid shards."""
    if prediction.shape != target.shape or valid.shape != target.shape or target.shape[-1] != 30:
        raise ValueError("Flow tensors must have matching [batch,horizon,30] shapes")
    weight = sample_weight.float().reshape(-1, 1, 1)
    if weight.shape[0] != target.shape[0]:
        raise ValueError("sample_weight batch mismatch")
    valid = valid.bool()
    clean_prediction = torch.where(valid, prediction.float(), 0.0)
    clean_target = torch.where(valid, target.float(), 0.0)
    modality = torch.ones(30, device=target.device) if modality_weights is None else modality_weights
    effective = valid.float() * weight * modality / target.shape[1]
    return ((clean_prediction - clean_target).square() * effective).sum(), effective.sum()


class _FlowLoss(nn.Module):
    def __init__(self, model, config, world_size):
        super().__init__()
        self.model, self.world_size = model, world_size
        self.register_buffer("modality", torch.tensor([config.position_weight] * 6 +
                             [config.rotation_weight] * 12 + [config.hand_weight] * 12))

    def forward(self, batch, denominator):
        valid = batch["action_valid_mask"].bool()
        target = torch.where(valid, batch["y_action"], 0.0)
        noise = torch.where(valid, torch.randn_like(target), 0.0)
        t = torch.rand((target.shape[0], 1), device=target.device)
        x_t = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * target
        prediction = self.model(x_rgb=batch["x_rgb"], x_ict=batch["x_ict"],
                                ict_mask=batch["ict_mask"], x_t=x_t, t=t)["v_pred"]
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("Nonfinite flow prediction")
        numerator, _ = masked_flow_statistics(prediction, target - noise, valid,
                                               batch["sample_weight"], self.modality)
        return numerator * self.world_size / denominator, numerator.detach()


def _synchronize_error(error):
    errors = [str(error) if error else None]
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, errors[0])
        errors = gathered
    if any(errors):
        raise RuntimeError("; ".join(str(e) for e in errors if e))


def _rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _atomic(path, writer):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _write_checkpoint(path, payload):
    if path.exists():
        raise ValueError("Refusing to overwrite an immutable candidate checkpoint")
    _atomic(path, lambda stream: torch.save(payload, stream))
    reference = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    _atomic(Path(str(path) + ".ref.json"), lambda stream: stream.write(json.dumps(reference).encode()))


def _load_checkpoint(path, device):
    reference = json.loads(Path(str(path) + ".ref.json").read_text())
    if path.stat().st_size != reference["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("Resume checkpoint SHA/size mismatch")
    return torch.load(path, map_location=device, weights_only=False)


def _check_resume_progress(output, step):
    """Never append behind an existing immutable checkpoint or update-log tail."""
    for path in output.glob("checkpoint_*.pt"):
        suffix = path.stem.removeprefix("checkpoint_")
        if not suffix.isdecimal():
            raise ValueError(f"Invalid existing checkpoint name: {path.name}")
        if int(suffix) > step:
            raise ValueError(f"NEWER_CHECKPOINT_PRESENT: {path.name} exceeds resume step {step}")
    log = output / "updates.jsonl"
    if log.exists():
        with log.open() as stream:
            for line in stream:
                record = json.loads(line)
                update = record.get("optimizer_update")
                if type(update) is not int or update < 0:
                    raise ValueError("Invalid existing update-log progress")
                if update > step:
                    raise ValueError(f"UPDATE_LOG_AHEAD_OF_RESUME: {update} exceeds resume step {step}")


def _deadline_reason(elapsed, config, authorization_expires_at, device, world):
    remaining = torch.tensor([config.wall_seconds_max - elapsed,
        float("inf") if authorization_expires_at is None else authorization_expires_at - time.time()],
        dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(remaining, op=dist.ReduceOp.MIN)
    wall, authorization = remaining.tolist()
    if min(wall, authorization) > 0:
        return None
    return "authorization_expires_at" if authorization <= wall else "wall_seconds_max"


def _sample_indices(length, cursor, count, seed):
    indices = []
    while len(indices) < count:
        epoch, offset = divmod(cursor + len(indices), length)
        order = torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        indices.extend(order[offset:offset + count - len(indices)])
    return indices


def _batch(dataset, indices):
    samples = [dataset[index] for index in indices]
    batch = {key: torch.stack([torch.as_tensor(row[key]) for row in samples]) for key in
             ("x_rgb", "x_ict", "ict_mask", "y_action", "action_valid_mask", "sample_weight")}
    action, valid = batch["y_action"], batch["action_valid_mask"].bool()
    weights = batch["sample_weight"].float().reshape(-1)
    if action.ndim != 3 or action.shape[-1] != 30 or valid.shape != action.shape:
        raise ValueError("Dataset action/mask must be [horizon,30]")
    if not torch.isfinite(action[valid]).all() or not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("Nonfinite admitted action or invalid sample weights")
    if not torch.isfinite(batch["x_rgb"]).all() or not torch.isfinite(batch["x_ict"]).all():
        raise ValueError("Nonfinite conditioning")
    batch["sample_weight"] = weights
    return batch


def train_candidate(model, dataset, config, output_dir, signatures, *, resume=None,
                    device="cpu", stop_after_updates=None, initializer=None,
                    authorization_expires_at=None):
    """Train only a caller-approved candidate; no data qualification is inferred here.

    Dataset reads must be deterministic and have no hidden mutable augmentation state.
    Exact resume binds topology and configuration; changing GPU count starts a new run.
    stop_after_updates is a graceful interruption boundary, not a new update budget.
    initializer(model) runs only on fresh rank zero, under the writer lock, before DDP/EMA.
    authorization_expires_at is a renewable UTC epoch deadline, not a checkpoint binding.
    """
    device = torch.device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    owned_group = False
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", device.index or 0)))
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        owned_group = True
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    lock = None
    try:
        error = None
        try:
            config.validate(world)
            output = Path(output_dir).resolve()
            if not output.is_relative_to(PROJECT_ROOT) or not signatures or len(dataset) < 1:
                raise ValueError("Require project-local output, bound signatures and nonempty deterministic dataset")
            if resume and Path(resume).resolve().parent != output:
                raise ValueError("Resume must belong to the same candidate output directory")
            if authorization_expires_at is not None and (not np.isfinite(authorization_expires_at)
                                                        or authorization_expires_at <= 0):
                raise ValueError("authorization_expires_at must be finite positive UTC epoch seconds")
        except Exception as exc:
            error = exc
        _synchronize_error(error)
        topology = {"world_size": world, "micro_batch": config.micro_batch,
                    "accumulation": config.global_batch // (world * config.micro_batch), "device_type": device.type}
        binding = {"config": asdict(config), "signatures": json.loads(json.dumps(signatures, sort_keys=True)),
                   "topology": topology, "dataset_length": len(dataset), "output_root": str(output),
                   "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "cuda_build": torch.version.cuda, "deterministic_algorithms": True,
                        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "torch_threads": torch.get_num_threads()}}
        if world > 1:
            bindings = [None] * world
            dist.all_gather_object(bindings, binding)
            _synchronize_error(None if all(item == binding for item in bindings) else "Rank configuration/signature mismatch")
        error = None
        if rank == 0:
            acquired = False
            try:
                output.mkdir(parents=True, exist_ok=True)
                lock = output / ".writer.lock"
                with lock.open("x") as stream:
                    acquired = True
                    if resume is None and ((output / "updates.jsonl").exists() or any(output.glob("checkpoint_*.pt"))):
                        raise ValueError("Existing candidate progress requires an explicit resume checkpoint")
                    stat = Path("/proc/self/stat").read_text()
                    host_pid, ticks = int(stat.split()[0]), int(stat.rsplit(")", 1)[1].split()[19])
                    if Path(f"/proc/{host_pid}/stat").read_text().rsplit(")", 1)[1].split()[19] != str(ticks):
                        raise ValueError("Writer procfs identity mismatch")
                    writer = {"pid": host_pid, "namespace_pid": os.getpid(), "writer_rank": 0,
                              "executor_epoch": 1, "proc_start_ticks": ticks,
                              "run_signature": hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
                              "fencing_token": os.urandom(16).hex()}
                    stream.write(json.dumps(writer))
            except Exception as exc:
                if not acquired:
                    lock = None
                error = exc
        _synchronize_error(error)
        if _deadline_reason(0.0, config, authorization_expires_at, device, world):
            raise RuntimeError("Authorization expired before initialization")
        random.seed(config.seed + rank)
        np.random.seed(config.seed + rank)
        torch.manual_seed(config.seed + rank)
        model = model.to(device)
        error = None
        if rank == 0 and resume is None and initializer is not None:
            try:
                initializer(model)
            except Exception as exc:
                error = exc
        _synchronize_error(error)
        if _deadline_reason(0.0, config, authorization_expires_at, device, world):
            raise RuntimeError("Authorization expired during initialization")
        wrapped = _FlowLoss(model, config, world).to(device)
        if world > 1:
            wrapped = DDP(wrapped, device_ids=[device.index] if device.type == "cuda" else None,
                          find_unused_parameters=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
        ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
        step, cursor, elapsed_before = 0, 0, 0.0
        timing = {"samples": 0, "seconds": 0.0, "device_type": device.type,
                  "precision": config.precision, "warmup_updates": config.timing_warmup_updates}
        if resume:
            state, error = None, None
            try:
                state = _load_checkpoint(Path(resume).resolve(), device)
                if state["schema"] != "humanego-ict-revo2-resume-v1" or state["binding"] != binding:
                    raise ValueError("Resume topology/config/signature mismatch")
                if (not 0 <= state["optimizer_updates"] <= config.max_updates
                        or state["cursor"] != state["optimizer_updates"] * config.global_batch
                        or len(state["rank_rng"]) != world):
                    raise ValueError("Invalid resume update/cursor/rank RNG state")
                _check_resume_progress(output, state["optimizer_updates"])
                model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                ema, step, cursor = state["ema"], state["optimizer_updates"], state["cursor"]
                timing, elapsed_before = state["timing"], state["training_wall_seconds"]
                _restore_rng(state["rank_rng"][rank], device)
            except Exception as exc:
                error = exc
            _synchronize_error(error)
        model.train()
        last_checkpoint = str(Path(resume).resolve()) if resume else None
        started = time.monotonic()
        reason = "update_budget" if step == config.max_updates else "graceful_stop"

        def seal(elapsed):
            nonlocal last_checkpoint
            path = output / f"checkpoint_{step:06d}.pt"
            if step == 0 or last_checkpoint == str(path):
                return
            elapsed = torch.tensor(elapsed, dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            states = [_rng(device)]
            if world > 1:
                states = [None] * world
                dist.all_gather_object(states, _rng(device))
            error = None
            if rank == 0:
                try:
                    _write_checkpoint(path, {"schema": "humanego-ict-revo2-resume-v1",
                        "binding": binding, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "ema": ema, "optimizer_updates": step, "cursor": cursor, "rank_rng": states,
                        "timing": timing, "training_wall_seconds": float(elapsed), "writer": writer,
                        "authorization_expires_at": authorization_expires_at,
                        "control_ground_truth": False, "physical_deployment_authorized": False})
                except Exception as exc:
                    error = exc
            _synchronize_error(error)
            last_checkpoint = str(path)

        while step < config.max_updates and (stop_after_updates is None or step < stop_after_updates):
            step_started = time.monotonic()
            elapsed = elapsed_before + step_started - started
            deadline = _deadline_reason(elapsed, config, authorization_expires_at, device, world)
            if deadline:
                reason = deadline
                seal(elapsed)
                break
            before_update_rng = _rng(device)
            before_update_buffers = {key: value.detach().clone() for key, value in model.named_buffers()}
            before_update_lr = [group["lr"] for group in optimizer.param_groups]
            indices = _sample_indices(len(dataset), cursor, config.global_batch, config.seed)
            local_count = config.global_batch // world
            indices = indices[rank * local_count:(rank + 1) * local_count]
            batches, error = [], None
            try:
                batches = [_batch(dataset, indices[i:i + config.micro_batch]) for i in range(0, local_count, config.micro_batch)]
            except Exception as exc:
                error = exc
            _synchronize_error(error)
            modality = (wrapped.module if world > 1 else wrapped).modality
            denominator = sum(masked_flow_statistics(torch.zeros_like(batch["y_action"], device=device),
                              batch["y_action"].to(device), batch["action_valid_mask"].to(device),
                              batch["sample_weight"].to(device), modality)[1] for batch in batches)
            if world > 1:
                dist.all_reduce(denominator)
            _synchronize_error(None if bool(denominator > 0) else "No globally effective action supervision")
            lr = config.learning_rate * min(1.0, (step + 1) / max(1, config.warmup_updates))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            numerator = torch.zeros((), device=device)
            for index, cpu_batch in enumerate(batches):
                from contextlib import nullcontext
                sync = wrapped.no_sync() if world > 1 and index < len(batches) - 1 else nullcontext()
                with sync:
                    batch = {key: value.to(device) for key, value in cpu_batch.items()}
                    error, loss = None, None
                    try:
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
                            loss, local_numerator = wrapped(batch, denominator)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("Nonfinite flow loss")
                    except Exception as exc:
                        error = exc
                    _synchronize_error(error)
                    loss.backward()
                    numerator += local_numerator
            finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            _synchronize_error(None if finite else "Nonfinite gradients; no optimizer update performed")
            elapsed = elapsed_before + time.monotonic() - started
            deadline = _deadline_reason(elapsed, config, authorization_expires_at, device, world)
            if deadline:
                optimizer.zero_grad(set_to_none=True)
                _restore_rng(before_update_rng, device)
                with torch.no_grad():
                    for key, value in model.named_buffers():
                        value.copy_(before_update_buffers[key])
                for group, old_lr in zip(optimizer.param_groups, before_update_lr):
                    group["lr"] = old_lr
                reason = deadline
                seal(elapsed)
                break
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            step, cursor = step + 1, cursor + config.global_batch
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
            elapsed = torch.tensor(elapsed_before + time.monotonic() - started, device=device)
            if world > 1:
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if step > config.timing_warmup_updates:
                timing["samples"] += 1
                timing["seconds"] += float(duration)
            mean = timing["seconds"] / timing["samples"] if timing["samples"] else None
            deadline = _deadline_reason(float(elapsed), config, authorization_expires_at, device, world)
            reason = deadline or ("update_budget" if step == config.max_updates else "graceful_stop")
            if step % config.checkpoint_every == 0 or step == config.max_updates or step == stop_after_updates or deadline:
                seal(float(elapsed))
            error = None
            if rank == 0 and (step % config.log_every == 0 or step == config.max_updates or step == stop_after_updates or deadline):
                try:
                    with (output / "updates.jsonl").open("a") as stream:
                        stream.write(json.dumps({"optimizer_update": step, "cursor": cursor,
                            "flow_loss": float(numerator / denominator), "lr": lr,
                            "step_seconds": float(duration), "steady_step_seconds": mean,
                            "eta_20000_hours": mean * 20000 / 3600 if mean is not None else None,
                            "timing_device_type": device.type}) + "\n")
                except Exception as exc:
                    error = exc
            _synchronize_error(error)
            if deadline:
                break
        return {"optimizer_updates": step, "cursor": cursor, "last_checkpoint": last_checkpoint,
                "completed_update_budget": step == config.max_updates,
                "termination_reason": reason, "timing": timing,
                "world_size": world, "control_ground_truth": False, "physical_deployment_authorized": False}
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)
        if world > 1:
            dist.barrier()
        if owned_group:
            dist.destroy_process_group()
