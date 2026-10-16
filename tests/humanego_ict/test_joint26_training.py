"""Synthetic CPU tests only; no pretrained weights, NAS or CUDA calls."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from chaoyang.humanego_ict import joint26_training as training


def test_old_30d_cache_rejected_before_torch(tmp_path):
    root = training.PROJECT_ROOT
    relative = tmp_path.relative_to(root)
    manifest = tmp_path / "old.json"
    manifest.write_text(json.dumps({"schema": "old", "action_dim": 30, "horizon": 50}))
    program = ("import sys; from chaoyang.humanego_ict.joint26_training import preflight; "
               "\ntry: preflight(sys.argv[1], 'raw')\nexcept ValueError as e: "
               "print(str(e)); print('torch_imported=' + str('torch' in sys.modules))\n")
    result = subprocess.run([sys.executable, "-B", "-c", program, str(manifest)],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")}, capture_output=True, text=True)
    assert result.returncode == 0
    assert "JOINT26_CACHE_REQUIRED" in result.stdout
    assert "torch_imported=False" in result.stdout
    assert relative.parts


def test_missing_cache_and_symlink_failclosed(tmp_path):
    with pytest.raises(ValueError, match="PROJECT_INPUT_MISSING"):
        training.preflight(tmp_path / "absent.json", "raw")
    target = tmp_path / "target"; target.mkdir()
    link = tmp_path / "link"; link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="PROJECT_SYMLINK_FORBIDDEN"):
        training.project_path(link / "output", subdirectory=True)
    with pytest.raises(ValueError, match="PROJECT_ONLY_PATH_REQUIRED"):
        training.project_path("/path/to/private_storage/output")


def test_config_topology_precision_and_configurable_budget():
    training.Joint26TrainingConfig().validate(8, production=True)
    with pytest.raises(ValueError, match="TOPOLOGY"):
        training.Joint26TrainingConfig().validate(3)
    for target in (1, 10, 20_000, 25_000):
        dataclasses.replace(training.Joint26TrainingConfig(), max_updates=target).validate(8, production=True)
    for target in (0, -1, True, False, 10.5, "10"):
        with pytest.raises(ValueError, match="POSITIVE_INTEGER"):
            dataclasses.replace(training.Joint26TrainingConfig(), max_updates=target).validate(8, production=True)
    with pytest.raises(ValueError, match="PRODUCTION_GLOBAL64"):
        dataclasses.replace(training.Joint26TrainingConfig(), global_batch=32,
                            micro_batch=4).validate(8, production=True)
    with pytest.raises(ValueError, match="FLOAT32_OR_BF16"):
        dataclasses.replace(training.Joint26TrainingConfig(), precision="fp16").validate(8)


def test_insufficient_or_overlapping_groups_reject_even_claimed_pass(tmp_path, monkeypatch):
    manifest = tmp_path / "new26.json"
    manifest.write_text(json.dumps({"schema": training.MANIFEST_SCHEMA, "action_dim": 26, "horizon": 30,
                                   "control_ground_truth": False, "physical_deployment_authorized": False}))
    qualification = {"data_gate": {"status": "PASSED_RESEARCH_LABEL_GATE"},
        "source_group_splits": {"train": [f"t{i}" for i in range(15)], "validation": ["v0", "v1", "v2"]},
        "normalization": "IDENTITY_NO_FIT", "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "supervision_class": "DERIVED_RETARGET_ASSUMED_MODEL_JOINTS"}
    class Dataset:
        def __len__(self): return 1
        def verify_all_dependencies(self): return None
    stub = types.ModuleType("chaoyang.humanego_ict.joint26_data")
    stub.load_joint26_cache = lambda *args, **kwargs: (Dataset(), qualification)
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    with pytest.raises(ValueError, match="BLOCKED_DATA_VOLUME"):
        training.preflight(manifest, "raw")
    qualification["source_group_splits"]["train"].append("v0")
    with pytest.raises(ValueError, match="BLOCKED_DATA_VOLUME"):
        training.preflight(manifest, "raw")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch not installed")
def test_joint26_masked_statistics_and_rank_shards():
    import torch
    prediction = torch.ones(2, 30, 26, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[1] = False
    target[1] = float("nan")
    numerator, denominator = training.masked_joint26_flow_statistics(prediction, target, mask, torch.ones(2))
    assert numerator.item() == pytest.approx(26)
    assert denominator.item() == pytest.approx(26)
    (numerator / denominator).backward()
    assert bool(torch.isfinite(prediction.grad).all())
    assert prediction.grad[1].abs().sum() == 0
    with pytest.raises(ValueError, match="H30_D26"):
        training.masked_joint26_flow_statistics(torch.zeros(1, 50, 30), torch.zeros(1, 50, 30),
                                               torch.ones(1, 50, 30, dtype=torch.bool), torch.ones(1))
    indices = training.global_sample_indices(128, 0, 64, 7)
    assert len(set(indices)) == 64
    assert set(indices[:32]).isdisjoint(indices[32:])
    assert indices == training.global_sample_indices(128, 0, 64, 7)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch not installed")
def test_two_real_synthetic_updates_checkpoint_and_exact_resume(tmp_path):
    import torch
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(26, 26)
        def forward(self, **values):
            return {"v_pred": self.linear(values["x_t"])}
    class Dataset:
        def __len__(self): return 4
        def verify_all_dependencies(self): return None
        def __getitem__(self, index):
            return {"x_rgb": np.zeros((3, 240, 320), np.float32),
                "x_ict": np.zeros((8, 29), np.float32), "ict_mask": np.ones(8, bool),
                "x_state26": np.zeros(26, np.float32), "state_mask": np.ones(26, bool),
                "y_action": np.full((30, 26), index / 10, np.float32),
                "action_valid_mask": np.ones((30, 26), bool), "sample_weight": np.float32(1)}
    torch.manual_seed(7)
    model = Toy(); initial = {key: value.clone() for key, value in model.state_dict().items()}
    config = training.Joint26TrainingConfig(global_batch=4, micro_batch=2, max_updates=2,
        checkpoint_every=1, validation_every=1, log_every=1, wall_seconds_max=60)
    kwargs = dict(signatures={"synthetic": "explicit"}, model_config={"synthetic": True},
        normalization="IDENTITY_NO_FIT", inference_contract={"action_dim": 26, "horizon": 30}, device="cpu")
    output = tmp_path / "resume"
    first = training.run_training(model, Dataset(), Dataset(), config, output, stop_after_updates=1, **kwargs)
    assert first["optimizer_updates"] == 1 and first["cursor"] == 4
    assert first["requested_optimizer_updates"] == 2 and not first["completed_update_budget"]
    second = training.run_training(Toy(), Dataset(), Dataset(), config, output,
                                   resume=first["last_checkpoint"], **kwargs)
    assert second["optimizer_updates"] == 2 and second["completed_update_budget"]
    assert second["requested_optimizer_updates"] == 2
    assert Path(second["last_checkpoint"]).name == "checkpoint_000002.pt"
    continuous_model = Toy(); continuous_model.load_state_dict(initial)
    continuous = training.run_training(continuous_model, Dataset(), Dataset(), config, tmp_path / "continuous", **kwargs)
    state = torch.load(second["last_checkpoint"], map_location="cpu", weights_only=False)
    uninterrupted = torch.load(continuous["last_checkpoint"], map_location="cpu", weights_only=False)
    assert state["schema"] == "humanego-ict-joint26-resume-v1"
    assert state["optimizer_updates"] == 2 and state["cursor"] == 8
    assert state["requested_optimizer_updates"] == 2
    assert len(state["rank_rng"]) == 1 and state["control_ground_truth"] is False
    assert all(torch.equal(state["model"][key], uninterrupted["model"][key]) for key in state["model"])
    assert (output / "loss.csv").read_text().count("optimizer_update") == 1
    assert not (output / ".writer.lock").exists()
    with pytest.raises(ValueError, match="EXACT_CONFIGURED_JOINT26_UPDATE_BUDGET_REQUIRED"):
        training.export_final_candidate(second["last_checkpoint"], tmp_path / "rejected_export",
                                        expected_updates=2)
    assert not (tmp_path / "rejected_export").exists()


def _synthetic_export_metadata(target=7):
    """Fabricated unit-test metadata, never an actual trained checkpoint."""
    return {"schema": training.CHECKPOINT_SCHEMA,
        "binding": {"training_config": {"max_updates": target, "global_batch": 64},
                    "signatures": {"synthetic_only": True}},
        "requested_optimizer_updates": target, "optimizer_updates": target,
        "cursor": target * 64, "control_ground_truth": False,
        "physical_deployment_authorized": False, "synthetic_only": True}


@pytest.mark.parametrize("target", [0, -1, True, False, 1.5, "7", None])
def test_export_budget_rejects_non_positive_integer_without_torch(target):
    with pytest.raises(ValueError, match="POSITIVE_INTEGER_EXPORT_BUDGET"):
        training._completed_export_budget(_synthetic_export_metadata(), target)


@pytest.mark.parametrize("field,value", [
    ("max_updates", 0), ("max_updates", True), ("max_updates", 8),
    ("global_batch", 4), ("global_batch", True), ("optimizer_updates", 6),
    ("optimizer_updates", True), ("cursor", 447), ("cursor", 448.0),
    ("requested_optimizer_updates", 6), ("requested_optimizer_updates", True),
])
def test_signed_export_budget_rejects_partial_or_config_drift(field, value):
    state = _synthetic_export_metadata()
    owner = state["binding"]["training_config"] if field in {"max_updates", "global_batch"} else state
    owner[field] = value
    with pytest.raises(ValueError, match="EXACT_CONFIGURED_JOINT26_UPDATE_BUDGET_REQUIRED"):
        training._completed_export_budget(state, 7)


def test_signed_export_budget_requires_checkpoint_and_frozen_configuration():
    state = _synthetic_export_metadata()
    assert training._completed_export_budget(state, 7) == 7
    with pytest.raises(ValueError, match="EXACT_CONFIGURED_JOINT26_UPDATE_BUDGET_REQUIRED"):
        training._completed_export_budget(state, 20_000)
    del state["binding"]
    with pytest.raises(ValueError, match="SIGNED_JOINT26_TRAINING_CONFIG_REQUIRED"):
        training._completed_export_budget(state, 7)
    state = _synthetic_export_metadata(); state["schema"] = "legacy-H50-D30"
    with pytest.raises(ValueError, match="SIGNED_RESEARCH_CHECKPOINT_REQUIRED"):
        training._completed_export_budget(state, 7)
    for flag in ("control_ground_truth", "physical_deployment_authorized"):
        state = _synthetic_export_metadata(); state[flag] = True
        with pytest.raises(ValueError, match="SIGNED_RESEARCH_CHECKPOINT_REQUIRED"):
            training._completed_export_budget(state, 7)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch not installed")
@pytest.mark.parametrize("target", [1, 7, 25_000])
def test_synthetic_export_configured_target_and_idempotence(tmp_path, target):
    """No forward or optimizer: serialize explicitly synthetic scalar weights."""
    import torch
    state = dict(_synthetic_export_metadata(target),
        model={"synthetic_only_scalar": torch.tensor([0.125])},
        model_config={"synthetic_only": True, "horizon": 30, "action_dim": 26},
        normalization={"method": "IDENTITY_NO_FIT"},
        inference_contract={"action_dim": 26, "pred_horizon": 30, "synthetic_only": True,
            "control_ground_truth": False, "physical_deployment_authorized": False})
    source = tmp_path / f"synthetic_only_checkpoint_{target:06d}.pt"
    training._checkpoint(source, state)
    output = tmp_path / "synthetic_only_export"
    manifest = training.export_final_candidate(source, output, expected_updates=target)
    assert manifest["requested_optimizer_updates"] == target
    assert manifest["optimizer_updates"] == target
    assert manifest["action_dim"] == 26 and manifest["horizon"] == 30
    assert manifest["control_ground_truth"] is False
    assert manifest["physical_deployment_authorized"] is False
    assert manifest["source_state_dict_key"] == "model"
    assert training.export_final_candidate(source, output, expected_updates=target) == manifest
    with pytest.raises(ValueError, match="EXACT_CONFIGURED_JOINT26_UPDATE_BUDGET_REQUIRED"):
        training.export_final_candidate(source, output, expected_updates=target + 1)
    # Even same-source idempotence must not return an old or misleading target.
    manifest_path = output / "EXPORT_MANIFEST.json"
    corrupted = dict(manifest); corrupted.pop("requested_optimizer_updates")
    manifest_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="IMMUTABLE_EXPORT_BUDGET_OR_CONTRACT_CONFLICT"):
        training.export_final_candidate(source, output, expected_updates=target)
