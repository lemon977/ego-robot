"""Bounded synthetic CPU proofs; never load a real model or use a GPU."""
from dataclasses import asdict, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from chaoyang.humanego_ict import production_resume as resume
from chaoyang.humanego_ict import training
from chaoyang.human_ego.utils.frozen_contract import file_reference


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(30, 30)

    def forward(self, *, x_t, **unused):
        return {"v_pred": self.linear(x_t)}


class TinyDataset:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"x_rgb": torch.zeros(3, 2, 2), "x_ict": torch.zeros(2, 29),
                "ict_mask": torch.ones(2, dtype=torch.bool), "y_action": torch.full((2, 30), index * 0.1),
                "action_valid_mask": torch.ones(2, 30, dtype=torch.bool), "sample_weight": torch.tensor(1.0)}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return file_reference(path)


def reseal(f, *, checkpoint_state=None):
    if checkpoint_state is not None:
        torch.save(checkpoint_state, f["checkpoint"])
        reference = file_reference(f["checkpoint"])
        save(Path(str(f["checkpoint"]) + ".ref.json"), {"bytes": reference["bytes"], "sha256": reference["sha256"]})
        f["outer"]["complete_checkpoints"] = [reference]
        f["refs"]["checkpoint"] = reference
        f["refs"]["checkpoint_sidecar"] = file_reference(Path(str(f["checkpoint"]) + ".ref.json"))
    f["refs"]["training_config"] = save(f["config_path"], f["full_config"])
    f["packet"]["inputs"]["training_config"] = f["refs"]["training_config"]
    f["refs"]["task_packet"] = save(f["packet_path"], f["packet"])
    f["refs"]["inner_result"] = save(f["output"] / "RESULT_000002.json", f["inner"])
    f["outer"]["stage_results"] = [f["refs"]["inner_result"]]
    f["export_receipt"]["training_provenance"]["training_config_ref"] = f["refs"]["training_config"]
    f["export_receipt"]["training_provenance"]["training_result_ref"] = f["refs"]["inner_result"]
    receipt_ref = save(f["output"] / "inference_tensors/EXPORT_RECEIPT.json", f["export_receipt"])
    exported = {"state_dict_key": "model", "export_receipt_ref": receipt_ref}
    for role, name in (("checkpoint_ref", "candidate_tensor.pt"), ("inference_contract_ref", "inference_contract.json"),
                       ("model_config_ref", "model_config.json"), ("normalization_ref", "normalization.json")):
        exported[role] = save(f["output"] / "inference_tensors" / name, {"synthetic_export_metadata_only": role})
    f["outer"]["inference_export"] = save(f["output"] / "INFERENCE_EXPORT.json", exported)
    f["refs"]["outer_result"] = save(f["output"].parent / "RESULT.json", f["outer"])


@pytest.fixture
def parent(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(training, "PROJECT_ROOT", tmp_path)
    root = tmp_path.resolve()
    output = root / "_run/current/synthetic_parent/attempts/attempt_0001/output"
    config = training.TrainingConfig(global_batch=4, micro_batch=2, max_updates=6,
        precision="float32", warmup_updates=2, checkpoint_every=2, log_every=1,
        wall_seconds_max=60, timing_warmup_updates=1)
    refs = {name: save(root / "inputs" / f"{name}.json", {"fixture": name})
            for name in ("manifest", "pretrained", "pretrained_config", "code", "storage")}
    full = {"schema_version": resume.INITIAL_CONFIG_SCHEMA,
            "initialization": "FRESH_SHARED_PRETRAINED_NEW_IO", "model": {"fixture": "TinyFlow"},
            "training": asdict(config), "training_scope": resume.TRAINING_SCOPE,
            "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
            "physical_deployment_authorized": False, "input_bindings": {"fixture": refs["manifest"]["sha256"]},
            "minimum_free_output_bytes": 4 * 1024**3, "minimum_export_margin_seconds": 1200}
    config_path = output.parent / "TRAINING_CONFIG.json"
    config_ref = save(config_path, full)
    signatures = {"run_signature": "a" * 64, "data": refs["manifest"], "prepared_pair": refs["manifest"],
                  "config": config_ref, "pretrained": refs["pretrained"], "pretrained_config": refs["pretrained_config"],
                  "branch": "raw", "code": [refs["code"]], "supervision_class": "DERIVED_RETARGET",
                  "training_scope": resume.TRAINING_SCOPE, "control_ground_truth": False,
                  "storage_capacity_observation": refs["storage"]}
    torch.manual_seed(123)
    initial = TinyFlow().state_dict()
    model = TinyFlow()
    model.load_state_dict(initial)
    result = training.train_candidate(model, TinyDataset(), config, output, signatures, stop_after_updates=2)
    result.update(task_id="synthetic_parent", run_signature=signatures["run_signature"], branch="raw",
                  training_scope=resume.TRAINING_SCOPE, supervision_class="DERIVED_RETARGET")
    checkpoint = Path(result["last_checkpoint"])
    packet_path = root / "tasks/current/synthetic_parent/TASK_PACKET.json"
    packet = {"task_id": "synthetic_parent", "run_signature": signatures["run_signature"], "branch": "raw",
              "training_scope": resume.TRAINING_SCOPE, "output_dir": str(output), "runtime_code": signatures["code"],
              "inputs": {"training_config": config_ref, "production_manifest": refs["manifest"],
                         "prepared_pair": refs["manifest"], "pretrained": refs["pretrained"],
                         "pretrained_config": refs["pretrained_config"], "storage_capacity_observation": refs["storage"]}}
    outer = {"task_id": packet["task_id"], "run_signature": packet["run_signature"], "task_terminal_status": "PASSED",
             "child_exit_code": 0, "control_ground_truth": False, "physical_deployment_authorized": False,
             "complete_checkpoints": [file_reference(checkpoint)]}
    export_receipt = {"status": "TENSOR_EXPORT_SAFE_ROUNDTRIP_VERIFIED_NOT_MODEL_QUALITY", "fixture_only": False,
        "branch": "raw", "optimizer_updates": 2, "control_ground_truth": False, "physical_deployment_authorized": False,
        "training_provenance": {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
            "branch": "raw", "optimizer_updates": 2, "training_scope": resume.TRAINING_SCOPE}}
    f = {"root": root, "config": config, "config_path": config_path, "full_config": full,
         "packet_path": packet_path, "packet": packet, "outer": outer, "inner": result,
         "checkpoint": checkpoint, "output": output, "signatures": signatures, "initial": initial,
         "export_receipt": export_receipt,
         "refs": {"checkpoint": file_reference(checkpoint), "checkpoint_sidecar": file_reference(Path(str(checkpoint) + ".ref.json"))}}
    reseal(f)
    return f


def successor(f, *, changed_config=None, changed_signatures=None, dataset_length=4, world_size=1, device_type="cpu", output=None):
    full = copy.deepcopy(f["full_config"])
    full.update(schema_version=resume.CONFIG_SCHEMA, initialization="RESUME_COMPLETE_TRAINING_STATE",
                resume_parent_bindings=copy.deepcopy(f["refs"]))
    full["training"]["wall_seconds_max"] = 120
    if changed_config:
        changed_config(full)
    output = output or f["root"] / "_run/current/synthetic_successor/attempts/attempt_0001/output"
    config_ref = save(f["root"] / "_run/current/synthetic_successor/TRAINING_CONFIG.json", full)
    signatures = copy.deepcopy(f["signatures"])
    signatures.update(run_signature="b" * 64, config=config_ref)
    if changed_signatures:
        changed_signatures(signatures)
    config = training.TrainingConfig(**full["training"])
    checkpoint, receipt = resume.prepare_successor_checkpoint(f["refs"], root=f["root"],
        full_config_ref=config_ref, training_config=config, signatures=signatures, output_dir=output,
        dataset_length=dataset_length, world_size=world_size, device_type=device_type)
    return checkpoint, receipt, config, signatures


def test_json_admission_is_torch_free():
    result = subprocess.run([sys.executable, "-c", "import sys; import chaoyang.humanego_ict.production_resume; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                            capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stderr


def test_complete_state_only_wall_and_binding_adapted(parent):
    before = resume._safe_load_checkpoint(parent["refs"]["checkpoint"], parent["root"])
    parent_sha = file_reference(parent["checkpoint"])
    checkpoint, receipt, _, _ = successor(parent)
    after = resume._safe_load_checkpoint(file_reference(checkpoint), parent["root"])
    assert checkpoint.name == "checkpoint_000002.pt" and checkpoint.parent != parent["output"]
    assert after["training_wall_seconds"] == 0.0
    for key in set(before) - {"binding", "training_wall_seconds"}:
        assert resume._identical(before[key], after[key]), key
    assert file_reference(parent["checkpoint"]) == parent_sha
    assert receipt["parent_training_wall_seconds"] == before["training_wall_seconds"]
    assert receipt["optimizer_updates"] == 2 and receipt["cursor"] == 8
    assert receipt["unsafe_pickle_fallback"] is False and receipt["safe_tensor_roundtrip_verified"] is True
    assert (checkpoint.parent / "SUCCESSOR_RESUME_ADAPTATION.json").is_file()
    assert not (checkpoint.parent / ".successor_adaptation.lock").exists()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == receipt["checkpoint_ref"]["sha256"]


def test_successor_is_exact_continuation_not_reinitialization(parent):
    uninterrupted = TinyFlow()
    uninterrupted.load_state_dict(parent["initial"])
    full = training.train_candidate(uninterrupted, TinyDataset(), parent["config"], parent["root"] / "full", parent["signatures"])
    checkpoint, _, config, signatures = successor(parent)
    restored = TinyFlow()
    final = training.train_candidate(restored, TinyDataset(), config, checkpoint.parent, signatures,
        resume=checkpoint, initializer=lambda model: (_ for _ in ()).throw(AssertionError("REINITIALIZED")))
    assert final["optimizer_updates"] == 6 and final["cursor"] == 24
    a = resume._safe_load_checkpoint(file_reference(Path(full["last_checkpoint"])), parent["root"])
    b = resume._safe_load_checkpoint(file_reference(Path(final["last_checkpoint"])), parent["root"])
    for key in ("model", "ema", "optimizer", "rank_rng", "optimizer_updates", "cursor"):
        assert resume._identical(a[key], b[key]), key


@pytest.mark.parametrize("change", [
    lambda f: f["outer"].update(task_terminal_status="FAILED"),
    lambda f: f["outer"].update(child_exit_code=1),
    lambda f: f["inner"].update(task_id="other"),
    lambda f: f["inner"].update(run_signature="c" * 64),
    lambda f: f["inner"].update(branch="robotized"),
    lambda f: f["inner"].update(optimizer_updates=1),
    lambda f: f["inner"].update(cursor=0),
    lambda f: f["inner"].update(control_ground_truth=True),
    lambda f: f["outer"].update(physical_deployment_authorized=True),
    lambda f: f["export_receipt"].update(status="UNVERIFIED"),
    lambda f: f["export_receipt"].update(fixture_only=True),
    lambda f: f["export_receipt"].update(optimizer_updates=3),
])
def test_bad_sealed_parent_rejected_before_tensor_read(parent, monkeypatch, change):
    change(parent)
    reseal(parent)
    monkeypatch.setattr(resume, "_safe_load_checkpoint", lambda *args: (_ for _ in ()).throw(AssertionError("TENSOR_LOADED")))
    with pytest.raises(ValueError):
        successor(parent)


@pytest.mark.parametrize("mutate", [
    lambda c: c["training"].update(learning_rate=0.1),
    lambda c: c["training"].update(max_updates=7),
    lambda c: c["training"].update(seed=42),
    lambda c: c["training"].update(micro_batch=1),
    lambda c: c["training"].update(ema_decay=0.9),
    lambda c: c["training"].update(position_weight=1),
    lambda c: c["model"].update(new_architecture=True),
    lambda c: c["input_bindings"].update(fixture="c" * 64),
    lambda c: c.update(minimum_export_margin_seconds=1),
    lambda c: c.update(physical_deployment_authorized=True),
])
def test_any_core_config_model_or_label_change_rejected(parent, mutate):
    with pytest.raises(ValueError, match="ONLY_FINITE_TASK_WALL"):
        successor(parent, changed_config=mutate)


@pytest.mark.parametrize("field", ["data", "pretrained", "pretrained_config", "branch", "prepared_pair", "supervision_class", "training_scope", "control_ground_truth"])
def test_semantic_signature_change_rejected(parent, field):
    def change(s):
        s[field] = "changed"
    with pytest.raises(ValueError, match="ONLY_REBIND"):
        successor(parent, changed_signatures=change)


def test_additional_signature_key_rejected(parent):
    with pytest.raises(ValueError, match="ONLY_REBIND"):
        successor(parent, changed_signatures=lambda s: s.update(unbound_role="secret"))


@pytest.mark.parametrize("code", [[], [{"path": "/other", "sha256": "c" * 64, "bytes": 1}]])
def test_parent_code_cannot_be_removed_or_replaced(parent, code):
    with pytest.raises(ValueError, match="CODE_CLOSURE"):
        successor(parent, changed_signatures=lambda s: s.update(code=code))


def test_append_verified_new_adapter_code_allowed(parent):
    ref = save(parent["root"] / "inputs/new_adapter.json", {"fixture": "new code"})
    checkpoint, _, _, _ = successor(parent, changed_signatures=lambda s: s["code"].append(ref))
    assert checkpoint.exists()


def test_parent_checkpoint_sha_drift_rejected(parent):
    with parent["checkpoint"].open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="SAFE_FULL_STATE"):
        successor(parent)


def test_parent_sidecar_hash_mismatch_rejected(parent):
    path = Path(str(parent["checkpoint"]) + ".ref.json")
    parent["refs"]["checkpoint_sidecar"] = save(path, {"bytes": 1, "sha256": "c" * 64})
    with pytest.raises(ValueError, match="SIDECAR_MISMATCH"):
        successor(parent)


def test_symlink_and_old_output_rejected(parent):
    with pytest.raises(ValueError, match="NEW_INDEPENDENT"):
        successor(parent, output=parent["output"])
    link = parent["root"] / "linked"
    link.symlink_to(parent["output"], target_is_directory=True)
    with pytest.raises(ValueError, match="SYMLINK"):
        successor(parent, output=link / "new")


def test_existing_writer_or_progress_rejected(parent):
    output = parent["root"] / "new_output"
    output.mkdir()
    (output / ".writer.lock").write_text("existing writer")
    with pytest.raises(ValueError, match="FRESH_SUCCESSOR"):
        successor(parent, output=output)
    assert (output / ".writer.lock").read_text() == "existing writer"


@pytest.mark.parametrize("kwargs", [{"dataset_length": 5}, {"world_size": 2}, {"device_type": "cuda"}])
def test_dataset_and_topology_change_rejected(parent, kwargs):
    with pytest.raises(ValueError, match="MUST_MATCH_SEALED_PARENT"):
        successor(parent, **kwargs)


def test_runtime_environment_change_rejected(parent, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="ENVIRONMENT_MUST_MATCH"):
        successor(parent)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(cursor=0),
    lambda s: s.update(rank_rng=[]),
    lambda s: s["rank_rng"][0].update(numpy=("MT19937", np.zeros(4, dtype=np.uint32), 0, 0, 0.0)),
    lambda s: s["rank_rng"][0].update(torch=torch.zeros(1, dtype=torch.uint8)),
    lambda s: s["ema"].clear(),
    lambda s: s["model"]["linear.weight"].fill_(float("nan")),
    lambda s: s["optimizer"]["state"].clear(),
    lambda s: next(iter(s["optimizer"]["state"].values()))["step"].fill_(1),
    lambda s: s.update(training_wall_seconds=float("nan")),
])
def test_incomplete_or_invalid_full_state_rejected(parent, mutate):
    state = resume._safe_load_checkpoint(parent["refs"]["checkpoint"], parent["root"])
    mutate(state)
    reseal(parent, checkpoint_state=state)
    with pytest.raises(ValueError):
        successor(parent)


def _pickle_marker(path):
    Path(path).write_text("unsafe executed")
    return {}


class UnsafePickle:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return _pickle_marker, (str(self.marker),)


def test_no_arbitrary_pickle_or_unsafe_fallback(parent):
    marker = parent["root"] / "unsafe_marker"
    reseal(parent, checkpoint_state=UnsafePickle(marker))
    with pytest.raises(ValueError, match="SAFE_FULL_STATE_CHECKPOINT_READ_FAILED"):
        successor(parent)
    assert not marker.exists()
