"""CPU synthetic contracts; never instantiate HumanEgo or load real weights."""
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from chaoyang.humanego_ict.training import (
    TrainingConfig, _load_checkpoint, masked_flow_statistics, train_candidate,
)
from chaoyang.humanego_ict import training


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(30, 30)

    def forward(self, *, x_t, **conditioning):
        return {"v_pred": self.linear(x_t)}


class TinyDataset:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        valid = torch.ones(2, 30, dtype=torch.bool)
        valid[:, 18:24] = index % 2 == 0
        return {"x_rgb": torch.zeros(3, 2, 2), "x_ict": torch.zeros(2, 29),
                "ict_mask": torch.ones(2, dtype=torch.bool),
                "y_action": torch.full((2, 30), 0.1 * index),
                "action_valid_mask": valid, "sample_weight": torch.tensor(0.25 if index % 2 else 1.0)}


class FixedPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.25))

    def forward(self, target, mask, weight):
        return masked_flow_statistics(self.value.expand_as(target), target, mask, weight)


class ZeroShardDataset(TinyDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if index < 2:
            sample["action_valid_mask"][:] = False
            sample["y_action"][:] = float("nan")
            sample["sample_weight"] = torch.tensor(0.0)
        return sample


def _distributed_loss_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        target = torch.arange(120).float().reshape(4, 1, 30) / 60
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[:2] = False  # rank zero legitimately has no effective action labels
        weight = torch.tensor([0.0, 0.0, 0.25, 1.0])
        expected = torch.tensor(1.25, requires_grad=True)
        full_n, full_d = masked_flow_statistics(expected.expand_as(target), target, mask, weight)
        (full_n / full_d).backward()
        model = DistributedDataParallel(FixedPrediction())
        local = slice(rank * 2, rank * 2 + 2)
        numerator, denominator = model(target[local], mask[local], weight[local])
        dist.all_reduce(denominator)
        (2 * numerator / denominator).backward()
        assert torch.allclose(model.module.value.grad, expected.grad, atol=1e-6)
        result = train_candidate(TinyFlow(), ZeroShardDataset(),
            TrainingConfig(global_batch=4, micro_batch=1, max_updates=1, precision="float32"),
            Path(rendezvous).parent / "zero_shard_training", {"data": "synthetic-zero-shard"})
        assert result["optimizer_updates"] == 1
    finally:
        dist.destroy_process_group()


def _resume_worker(rank, world, root):
    torch.set_num_threads(1)
    root = Path(root)
    if world > 1:
        dist.init_process_group("gloo", init_method=f"file://{root / 'resume_rendezvous'}", rank=rank, world_size=world)
    try:
        config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=4,
                                precision="float32", checkpoint_every=2, warmup_updates=2,
                                timing_warmup_updates=1, log_every=1)
        torch.manual_seed(123)
        initial = TinyFlow().state_dict()
        full, interrupted = TinyFlow(), TinyFlow()
        full.load_state_dict(initial)
        interrupted.load_state_dict(initial)
        signatures = {"data": "synthetic-only-v1", "code": "unit-test-v1"}
        complete = train_candidate(full, TinyDataset(), config, root / "full", signatures)
        partial = train_candidate(interrupted, TinyDataset(), config, root / "resume", signatures, stop_after_updates=2)
        assert partial["optimizer_updates"] == 2 and partial["cursor"] == 8
        assert not (root / "resume" / ".writer.lock").exists()
        restored = TinyFlow()
        result = train_candidate(restored, TinyDataset(), config, root / "resume", signatures,
                                 resume=partial["last_checkpoint"],
                                 initializer=lambda model: (_ for _ in ()).throw(AssertionError("Resume reinitialized")))
        assert result["optimizer_updates"] == 4 and result["cursor"] == 16
        for key, tensor in full.state_dict().items():
            assert torch.equal(tensor, restored.state_dict()[key])
        if rank == 0:
            final = _load_checkpoint(Path(result["last_checkpoint"]), torch.device("cpu"))
            uninterrupted = _load_checkpoint(Path(complete["last_checkpoint"]), torch.device("cpu"))
            for key in final["ema"]:
                assert torch.equal(final["ema"][key], uninterrupted["ema"][key])
            assert len(final["rank_rng"]) == world
            assert final["timing"]["samples"] == 3 and final["timing"]["device_type"] == "cpu"
            assert final["writer"]["writer_rank"] == 0
            assert final["writer"]["pid"] > 0 and final["writer"]["proc_start_ticks"]
            assert final["control_ground_truth"] is False
            assert final["physical_deployment_authorized"] is False
            lines = (root / "resume" / "updates.jsonl").read_text().splitlines()
            assert len(lines) == 4
    finally:
        if world > 1:
            dist.destroy_process_group()


def test_fixed_global_batch_and_precision():
    config = TrainingConfig()
    assert config.global_batch == 64 and config.max_updates == 20_000
    config.validate(1)
    config.validate(8)
    with pytest.raises(ValueError, match="global_batch"):
        config.validate(3)
    with pytest.raises(ValueError, match="FP16"):
        TrainingConfig(precision="fp16").validate(1)


def test_masked_payload_has_zero_gradient():
    prediction = torch.ones(2, 2, 30, requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0] = False
    target[0] = float("nan")
    numerator, denominator = masked_flow_statistics(prediction, target, valid, torch.tensor([0.0, 0.25]))
    (numerator / denominator).backward()
    assert prediction.grad[0].count_nonzero() == 0
    assert torch.isfinite(prediction.grad).all()


def test_gloo_global_masked_loss(tmp_path):
    mp.spawn(_distributed_loss_worker, args=(str(tmp_path / "loss_rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize("world", [1, 2])
def test_exact_resume_and_rank_zero_writer(tmp_path, world):
    if world == 1:
        _resume_worker(0, 1, str(tmp_path))
    else:
        mp.spawn(_resume_worker, args=(world, str(tmp_path)), nprocs=world, join=True)
    assert sum(path.stat().st_size for path in tmp_path.rglob("*.pt")) < 1_048_576


def test_writer_and_signature_rejections(tmp_path):
    output = tmp_path / "candidate"
    output.mkdir()
    lock = output / ".writer.lock"
    lock.write_text("existing writer")
    config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=1, precision="float32")
    with pytest.raises(RuntimeError, match="writer.lock"):
        train_candidate(TinyFlow(), TinyDataset(), config, output, {"data": "v1"})
    assert lock.read_text() == "existing writer"
    lock.unlink()
    result = train_candidate(TinyFlow(), TinyDataset(), config, output, {"data": "v1"})
    with pytest.raises(RuntimeError, match="signature"):
        train_candidate(TinyFlow(), TinyDataset(), config, output, {"data": "v2"}, resume=result["last_checkpoint"])
    assert not lock.exists()
    checkpoint = Path(result["last_checkpoint"])
    with checkpoint.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA"):
        _load_checkpoint(checkpoint, torch.device("cpu"))


def test_finite_deadline(tmp_path):
    config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=20_000,
                            precision="float32", wall_seconds_max=1e-9)
    result = train_candidate(TinyFlow(), TinyDataset(), config, tmp_path / "bounded", {"data": "tiny"})
    assert result["termination_reason"] == "wall_seconds_max"
    assert result["optimizer_updates"] == 0 and result["last_checkpoint"] is None


def test_bf16_and_global_zero_rejection(tmp_path):
    config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=1, precision="bf16")
    result = train_candidate(TinyFlow(), TinyDataset(), config, tmp_path / "bf16", {"data": "tiny"})
    assert result["completed_update_budget"] is True
    class AllInvalid(TinyDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["action_valid_mask"][:] = False
            return sample
    with pytest.raises(RuntimeError, match="globally effective"):
        train_candidate(TinyFlow(), AllInvalid(), config, tmp_path / "zero", {"data": "tiny"})
    assert not list((tmp_path / "zero").glob("checkpoint_*.pt"))


def _initializer_worker(rank, root):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{root / 'initializer_rendezvous'}", rank=rank, world_size=2)
    try:
        config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=1, precision="float32")
        output = root / "initialized"
        def initialize(model):
            assert rank == 0 and (output / ".writer.lock").is_file()
            assert not any(output.glob("checkpoint_*.pt"))
            with torch.no_grad():
                model.linear.weight.fill_(0.125)
            (output / "initializer_report.json").write_text('{"writer_rank":0}')
        result = train_candidate(TinyFlow(), TinyDataset(), config, output, {"data": "tiny"}, initializer=initialize)
        state = _load_checkpoint(Path(result["last_checkpoint"]), torch.device("cpu"))
        model_weight = state["ema"]["linear.weight"]
        assert not torch.equal(state["model"]["linear.weight"], model_weight)
        assert torch.allclose(model_weight, torch.full_like(model_weight, 0.125), atol=1e-5)
        assert (output / "initializer_report.json").is_file()
        def fail(model):
            assert rank == 0 and (root / "failed" / ".writer.lock").is_file()
            raise ValueError("initializer synthetic failure")
        with pytest.raises(RuntimeError, match="initializer synthetic failure"):
            train_candidate(TinyFlow(), TinyDataset(), config, root / "failed", {"data": "tiny"}, initializer=fail)
        assert not (root / "failed" / ".writer.lock").exists()
        # One expired rank must reject every rank before the callback can write.
        with pytest.raises(RuntimeError, match="expired before initialization"):
            train_candidate(TinyFlow(), TinyDataset(), config, root / "expired", {"data": "tiny"},
                            initializer=fail, authorization_expires_at=time.time() + (60 if rank == 0 else -60))
        assert not any((root / "expired").glob("checkpoint_*.pt"))
        assert not (root / "expired" / ".writer.lock").exists()
    finally:
        dist.destroy_process_group()


def test_gloo_locked_initializer_and_synchronized_failure(tmp_path):
    mp.spawn(_initializer_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert sum(path.stat().st_size for path in tmp_path.rglob("*.pt")) < 1_048_576


def test_stale_resume_refuses_newer_checkpoint_without_log_mutation(tmp_path):
    config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=3, precision="float32",
                            checkpoint_every=1, log_every=1)
    output = tmp_path / "checkpoints"
    train_candidate(TinyFlow(), TinyDataset(), config, output, {"data": "tiny"}, stop_after_updates=2)
    log = (output / "updates.jsonl").read_bytes()
    with pytest.raises(RuntimeError, match="NEWER_CHECKPOINT_PRESENT"):
        train_candidate(TinyFlow(), TinyDataset(), config, output, {"data": "tiny"},
                        resume=output / "checkpoint_000001.pt")
    assert (output / "updates.jsonl").read_bytes() == log
    assert not (output / ".writer.lock").exists()


def test_stale_resume_refuses_update_log_ahead_without_mutation(tmp_path):
    config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=3, precision="float32")
    output = tmp_path / "log_ahead"
    partial = train_candidate(TinyFlow(), TinyDataset(), config, output, {"data": "tiny"}, stop_after_updates=1)
    log_path = output / "updates.jsonl"
    with log_path.open("a") as stream:
        stream.write('{"optimizer_update":2,"cursor":8}\n')
    log = log_path.read_bytes()
    with pytest.raises(RuntimeError, match="UPDATE_LOG_AHEAD_OF_RESUME"):
        train_candidate(TinyFlow(), TinyDataset(), config, output, {"data": "tiny"}, resume=partial["last_checkpoint"])
    assert log_path.read_bytes() == log
    assert not (output / ".writer.lock").exists()


def test_expired_authorization_prevents_initializer(tmp_path):
    called = []
    config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=1, precision="float32")
    with pytest.raises(RuntimeError, match="expired before initialization"):
        train_candidate(TinyFlow(), TinyDataset(), config, tmp_path / "expired", {"data": "tiny"},
                        initializer=lambda model: called.append(True), authorization_expires_at=time.time() - 60)
    assert not called and not (tmp_path / "expired" / ".writer.lock").exists()


def test_authorization_expiry_rolls_back_uncommitted_batch_and_renews(tmp_path, monkeypatch):
    clock = [1_000.0]
    monkeypatch.setattr(training.time, "time", lambda: clock[0])
    class ExpiringDataset(TinyDataset):
        reads = 0
        def __getitem__(self, index):
            self.reads += 1
            if self.reads > 4:
                clock[0] = 1_002.0
            return super().__getitem__(index)
    config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=3, precision="float32",
                            checkpoint_every=10, log_every=1)
    torch.manual_seed(123)
    initial = TinyFlow().state_dict()
    full, interrupted = TinyFlow(), TinyFlow()
    full.load_state_dict(initial)
    interrupted.load_state_dict(initial)
    complete = train_candidate(full, TinyDataset(), config, tmp_path / "full", {"data": "tiny"})
    partial = train_candidate(interrupted, ExpiringDataset(), config, tmp_path / "renewal", {"data": "tiny"},
                              authorization_expires_at=1_001.0)
    assert partial["termination_reason"] == "authorization_expires_at"
    assert partial["optimizer_updates"] == 1 and partial["cursor"] == 4
    state = _load_checkpoint(Path(partial["last_checkpoint"]), torch.device("cpu"))
    assert state["authorization_expires_at"] == 1_001.0
    assert "authorization_expires_at" not in state["binding"]
    assert len((tmp_path / "renewal" / "updates.jsonl").read_text().splitlines()) == 1
    restored = TinyFlow()
    resumed = train_candidate(restored, TinyDataset(), config, tmp_path / "renewal", {"data": "tiny"},
                              resume=partial["last_checkpoint"], authorization_expires_at=2_000.0)
    assert resumed["completed_update_budget"] is True
    final = _load_checkpoint(Path(resumed["last_checkpoint"]), torch.device("cpu"))
    reference = _load_checkpoint(Path(complete["last_checkpoint"]), torch.device("cpu"))
    for key in final["model"]:
        assert torch.equal(final["model"][key], reference["model"][key])
        assert torch.equal(final["ema"][key], reference["ema"][key])
    assert sum(path.stat().st_size for path in tmp_path.rglob("*.pt")) < 1_048_576


def _authorization_worker(rank, root):
    torch.set_num_threads(1)
    root = Path(root)
    clock = [1_000.0]
    original_time = training.time.time
    training.time.time = lambda: clock[0]
    dist.init_process_group("gloo", init_method=f"file://{root / 'authorization_rendezvous'}", rank=rank, world_size=2)
    try:
        class ExpiringRank(TinyDataset):
            reads = 0
            def __getitem__(self, index):
                self.reads += 1
                if rank == 1 and self.reads > 2:
                    clock[0] = 1_002.0
                return super().__getitem__(index)
        config = TrainingConfig(global_batch=4, micro_batch=1, max_updates=3, precision="float32",
                                checkpoint_every=10, log_every=1)
        torch.manual_seed(123)
        initial = TinyFlow().state_dict()
        full, interrupted = TinyFlow(), TinyFlow()
        full.load_state_dict(initial)
        interrupted.load_state_dict(initial)
        complete = train_candidate(full, TinyDataset(), config, root / "full", {"data": "tiny"})
        partial = train_candidate(interrupted, ExpiringRank(), config, root / "renewal", {"data": "tiny"},
                                  authorization_expires_at=1_001.0)
        assert partial["termination_reason"] == "authorization_expires_at"
        assert partial["optimizer_updates"] == 1 and partial["cursor"] == 4
        restored = TinyFlow()
        resumed = train_candidate(restored, TinyDataset(), config, root / "renewal", {"data": "tiny"},
                                  resume=partial["last_checkpoint"], authorization_expires_at=2_000.0)
        for key in full.state_dict():
            assert torch.equal(full.state_dict()[key], restored.state_dict()[key])
        if rank == 0:
            final = _load_checkpoint(Path(resumed["last_checkpoint"]), torch.device("cpu"))
            reference = _load_checkpoint(Path(complete["last_checkpoint"]), torch.device("cpu"))
            for key in final["ema"]:
                assert torch.equal(final["ema"][key], reference["ema"][key])
    finally:
        training.time.time = original_time
        dist.destroy_process_group()


def test_gloo_earliest_authorization_deadline_and_exact_renewal(tmp_path):
    mp.spawn(_authorization_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert sum(path.stat().st_size for path in tmp_path.rglob("*.pt")) < 1_048_576
