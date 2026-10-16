"""CPU synthetic derived training admission, never loading policy weights."""
from datetime import datetime, timedelta, timezone
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from chaoyang.humanego_ict import derived_training_gate as gate
from chaoyang.humanego_ict.derived_training_cli import validate_config, _check_caches
from chaoyang.humanego_ict.runtime_gate import file_ref, runtime_cache_paths


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return file_ref(path)


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    root, task = tmp_path, "derived-training-fixture"
    packet_path = root / "tasks/current" / task / "TASK_PACKET.json"
    output = root / "_run/out"
    manifest, config, pretrained = [root / "inputs" / n for n in ["manifest.json", "config.json", "weights.json"]]
    refs = [save(path, {"fixture": True}) for path in [manifest, config, pretrained]]
    closure = [save(root / "code.json", {"frozen": True})]
    writer = {"pid": 101, "proc_start_ticks": 123, "scope": task, "executor_epoch": 1, "fencing_token": "test-token"}
    writer_path = output / "WRITER.json"
    save(writer_path, writer)
    lease_path = output / "LEASE.json"
    expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    lease = {"status": "ACQUIRED", "task_id": task, "hostname": socket.gethostname(),
             "pid": 101, "process_startticks": 123, "fencing_token": "test-token",
             "executor_epoch": 1, "gpu_id": 0, "expires_at": expiry}
    save(lease_path, lease)
    args = ["train", "--task-packet", str(packet_path), "--manifest", str(manifest),
            "--config", str(config), "--pretrained", str(pretrained), "--output-dir", str(output), "--branch", "raw"]
    packet = {"task_id": task, "execution_mode": "HUMANEGO_ICT_DERIVED_RETARGET_TRAINING",
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "deadline_at": expiry, "writer": writer, "writer_file": str(writer_path), "run_signature": "test-signature",
        "inputs": dict(zip(["paired_manifest", "training_config", "pretrained"], refs)),
        "output_dir": str(output), "runtime_code": closure, "gpu_lease_verified": True,
        "gpu_lease_path": str(lease_path), "allocated_gpu_ids": [0],
        "budgets": {"optimizer_updates_max": 20000, "gpu_count": 1, "wall_seconds_max": 300},
        "commands": ["python -m chaoyang.cli run " + gate.OPERATION + " " + " ".join(args)]}
    contract_path = root / "docs/governance/contract.json"
    save(contract_path, {"special_status_contracts": {gate.ROUTE: {"candidate_operation": gate.OPERATION,
        "task_packet_required": True, "physical_deployment_authorized": False,
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False, "code_closure": closure}}})
    state_path = root / "docs/governance/state.json"
    save(state_path, {"tasks": [{"task_id": task, "status": "RUNNING", "run_signature": "test-signature",
                               "pid": 101, "proc_start_ticks": 123}]})
    index_path = root / "tasks/current/INDEX.json"
    def bind():
        save(packet_path, packet)
        save(index_path, {"task_packets": [{"task_id": task, "execution_allowed": True,
            "packet_path": str(packet_path.relative_to(root)), "packet_sha256": file_ref(packet_path)["sha256"]}]})
        save(root / "docs/governance/CURRENT_STATUS_RECEIPT.json", {"files": {
            "algorithm_contract": file_ref(contract_path), "task_state": file_ref(state_path),
            "task_packet_index_current": file_ref(index_path)}})
    bind()
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: True)
    return root, packet_path, manifest, config, pretrained, output, args, packet, bind, lease_path, lease


def admit(fixture, world=1):
    root, packet, manifest, config, pretrained, output, args, *_ = fixture
    return gate.authorize_derived_training(root, packet, args, manifest=manifest, config=config,
        pretrained=pretrained, output=output, updates=20000, world_size=world)


def test_valid_single_gpu_finite_admission(admitted):
    assert admit(admitted)["admitted_authorization_expires_at"] > datetime.now(timezone.utc).timestamp()


def test_reauthorization_never_resets_wall_budget(admitted):
    admitted[7]["budgets"]["wall_seconds_max"] = 100
    admitted[8]()
    first, second = admit(admitted), admit(admitted)
    assert first["admitted_authorization_expires_at"] == second["admitted_authorization_expires_at"]
    admitted[7]["execution_started_at"] = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    admitted[8]()
    with pytest.raises(ValueError, match="FIXED_TASK_WALL_DEADLINE"):
        admit(admitted)


@pytest.mark.parametrize("mode", ["CPU_DEVELOPMENT_ONLY", "HUMANEGO_ICT_REVO2_CANDIDATE_TRAINING",
    "HUMANEGO_ICT_DERIVED_LABEL_PREPARATION", "HUMANEGO_ICT_REVO2_OFFLINE_INFERENCE"])
def test_other_task_modes_not_reused(admitted, mode):
    admitted[7]["execution_mode"] = mode
    admitted[8]()
    with pytest.raises(ValueError, match="TRAINING_TASK_REQUIRED"):
        admit(admitted)


@pytest.mark.parametrize("field,value,error", [
    ("commands", [], "COMMAND_NOT_PINNED"), ("runtime_code", [], "CODE_CLOSURE"),
    ("gpu_lease_verified", False, "BUDGET_OR_GPU"),
    ("deadline_at", "2001-01-01T00:00:00+00:00", "DEADLINE"),
    ("deadline_at", "2099-01-01T00:00:00", "DEADLINE"),
])
def test_packet_drift_and_missing_permissions(admitted, field, value, error):
    admitted[7][field] = value; admitted[8]()
    with pytest.raises(ValueError, match=error):
        admit(admitted)


def test_changed_frozen_input_refused(admitted):
    admitted[2].write_text('{"changed":true}')
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        admit(admitted)


def test_writer_ancestry_required(admitted, monkeypatch):
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: False)
    with pytest.raises(ValueError, match="ANCESTRY"):
        admit(admitted)


def test_eight_gpus_require_lease_covering_all(admitted):
    admitted[7]["allocated_gpu_ids"] = list(range(8))
    admitted[7]["budgets"]["gpu_count"] = 8
    admitted[8]()
    with pytest.raises(ValueError, match="ALL_GPUS"):
        admit(admitted, 8)
    admitted[10]["allocated_gpu_ids"] = list(range(8)); save(admitted[9], admitted[10])
    assert admit(admitted, 8)["allocated_gpu_ids"] == list(range(8))


def config():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/train_derived_20000.json").read_text())


def test_same_20000_recipe_one_or_eight_gpus():
    cfg = config()
    for world in (1, 8):
        validate_config(cfg, world)
    assert cfg["training"]["max_updates"] == 20000
    assert cfg["training"]["global_batch"] * cfg["training"]["max_updates"] == 1280000


@pytest.mark.parametrize("key,value", [("max_updates", True), ("max_updates", 0),
    ("global_batch", 32), ("micro_batch", 3), ("wall_seconds_max", float("nan")),
    ("learning_rate", float("inf")), ("precision", "fp16"), ("ema_decay", 1)])
def test_bad_config_rejected_before_torch(key, value):
    cfg = config(); cfg["training"][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg, 8)


def test_fixed_horizon_and_no_old_real_config():
    cfg = config(); cfg["model"]["pred_horizon"] = 30
    with pytest.raises(ValueError, match="FIXED_H50"):
        validate_config(cfg, 1)


@pytest.mark.parametrize("key,value", [("patch_size", 0), ("num_heads", True),
    ("vision_embed_dim", 385), ("dropout", float("nan")), ("mlp_ratio", -1)])
def test_architecture_rejected_before_model(key, value):
    cfg = config(); cfg["model"][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg, 1)


def test_training_unknown_keys_and_seed_rejected_without_torch():
    for cfg in [config(), config(), config()]:
        if "unrecognized" not in cfg["training"]:
            cfg["training"]["unrecognized"] = True
        with pytest.raises(ValueError, match="UNSUPPORTED_TRAINING"):
            validate_config(cfg, 1)
    for seed in [True, -1, 2**32-1]:
        cfg = config(); cfg["training"]["seed"] = seed
        with pytest.raises(ValueError, match="SEED"):
            validate_config(cfg, 8)


def test_rank_errors_propagate_before_training_or_exit():
    from chaoyang.humanego_ict.derived_training_cli import _rank_error
    def gather(messages, local):
        messages[:] = [local, "synthetic rank-local OOM"]
    fake = SimpleNamespace(distributed=SimpleNamespace(all_gather_object=gather))
    with pytest.raises(RuntimeError, match="rank-local OOM"):
        _rank_error(None, 2, fake)
    cfg = config(); cfg["control_ground_truth"] = True
    with pytest.raises(ValueError, match="SEPARATE_DERIVED"):
        validate_config(cfg, 1)


def test_cache_environment_must_match_task_output(tmp_path, monkeypatch):
    output = tmp_path / "out"
    paths = runtime_cache_paths(output, tmp_path)
    for name, path in zip(["TMPDIR", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME"], paths):
        monkeypatch.setenv(name, str(path))
    _check_caches(output, tmp_path)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "another-task"))
    with pytest.raises(ValueError, match="TASK_LOCAL"):
        _check_caches(output, tmp_path)


@pytest.mark.parametrize("arguments", [["--help"], ["train", "--task-packet", "NO_PACKET.json",
    "--manifest", "NO_MANIFEST.json", "--branch", "raw", "--config", "NO_CONFIG.json",
    "--output-dir", "NO_OUTPUT", "--pretrained", "NO_WEIGHTS.pt"]])
def test_rejected_cli_never_imports_torch(arguments):
    script = """
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname.endswith('.FlowMatchingModel'):
            raise AssertionError('unexpected model import')
sys.meta_path.insert(0,Deny())
from chaoyang.humanego_ict.derived_training_cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, *arguments], capture_output=True, text=True)
    assert result.returncode == (0 if arguments == ["--help"] else 2), result.stderr
    assert "unexpected model import" not in result.stderr


def test_synthetic_wrapper_exports_live_state_with_bound_callback(tmp_path, monkeypatch):
    from chaoyang.humanego_ict import derived_training_cli as cli
    from chaoyang.humanego_ict import derived_data, checkpoint_export
    root, output = tmp_path, tmp_path / "out"
    cfg_path, manifest_path, weights = [tmp_path / n for n in ["config.json", "manifest.json", "weights.json"]]
    save(cfg_path, config())
    save(manifest_path, {"normalization": {"method": "IDENTITY_NO_FIT", "position_mean": [0,0,0],
        "position_std": [1,1,1]}, "pose_clock": {"tcp_frame_ids": ["LEFT_DERIVED", "RIGHT_DERIVED"]}})
    save(weights, {"fixture_only": True})
    packet = {"task_id": "synthetic-wrapper", "run_signature": "a"*64, "allocated_gpu_ids": [0],
        "runtime_code": [], "inputs": {"paired_manifest": file_ref(manifest_path),
        "training_config": file_ref(cfg_path), "pretrained": file_ref(weights)},
        "admitted_authorization_expires_at": datetime.now(timezone.utc).timestamp()+60}
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    monkeypatch.setattr(cli, "authorize_derived_training", lambda *a, **kw: dict(packet))
    monkeypatch.setattr(cli, "storage_ready", lambda *a: None)
    monkeypatch.setattr(cli, "bind_cuda_visible_devices", lambda *a: "0")
    monkeypatch.delenv("ICT_DERIVED_ADMISSION_ONLY", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    for variable, path in zip(["TMPDIR", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME"], runtime_cache_paths(output, root)):
        monkeypatch.setenv(variable, str(path))
    monkeypatch.setattr(derived_data, "validate_derived_training_manifest", lambda *a: {"dataset_sha": file_ref(manifest_path)["sha256"]})
    monkeypatch.setattr(derived_data, "DerivedRevo2ICTDataset", lambda *a: ["synthetic-sample"])
    state = {"synthetic_live_tensor": "fixture-only"}
    class Model:
        def __init__(self, **kw): pass
        def to(self, device): return self
        def state_dict(self): return state
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
        set_device=lambda i: None), manual_seed=lambda i: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "chaoyang.humanego_ict.model", SimpleNamespace(Revo2ICTModel=Model))
    def train(model, dataset, training_config, output_dir, signatures, **kw):
        kw["initializer"](model)
        return {"optimizer_updates": 1, "status": "SYNTHETIC_TEST_ONLY"}
    monkeypatch.setitem(sys.modules, "chaoyang.humanego_ict.training", SimpleNamespace(
        TrainingConfig=lambda **kw: kw, train_candidate=train))
    monkeypatch.setitem(sys.modules, "chaoyang.human_ego.training.checkpoint_transfer", SimpleNamespace(
        load_compatible_pretrained=lambda *a, **kw: None))
    exported = []
    def export(live, **kw):
        authorization = kw["runtime_authorizer"](kw["provenance"])
        assert authorization["task_id"] == "synthetic-wrapper" and authorization["export_authorized"]
        assert authorization["output_dir"] == str(output / "inference_tensors")
        assert kw["provenance"]["state_origin"] == "LIVE_MODEL_STATE" and live is state
        exported.append(kw)
        return {"fixture_only": True}
    monkeypatch.setattr(checkpoint_export, "export_tensor_candidate", export)
    args = ["train", "--task-packet", str(tmp_path/"packet.json"), "--manifest", str(manifest_path),
        "--config", str(cfg_path), "--pretrained", str(weights), "--branch", "raw", "--output-dir", str(output)]
    assert cli.main(args) == 0 and len(exported) == 1
    assert (output / "INFERENCE_EXPORT.json").is_file()
