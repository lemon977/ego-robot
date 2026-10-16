"""Bounded synthetic CPU route tests; no real model/forward/optimizer/GPU."""
from datetime import datetime, timedelta, timezone
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from chaoyang.humanego_ict import production_training_gate as gate
from chaoyang.humanego_ict import production_training_cli as cli
from chaoyang.humanego_ict.runtime_gate import file_ref, runtime_cache_paths


spec = importlib.util.spec_from_file_location("production_old_fixture", Path(__file__).with_name("test_derived_training.py"))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
save = old.save


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    f = old.admitted.__wrapped__(tmp_path, monkeypatch)
    root, path, manifest, config, weights, output, args, packet, old_bind, lease_path, lease = f
    save(root / "_run/current/locks/governance.lock", {"synthetic_existing_lock_only": True})
    source_config = save(root / "inputs/source_config.json", {})
    producer_id = "synthetic-source-producer"
    producer_path = root / "tasks/current" / producer_id / "TASK_PACKET.json"
    producer = {"task_id": producer_id, "run_signature": "a"*64,
        "execution_mode": gate.PREPARE_MODE, "training_scope": gate.TRAINING_SCOPE}
    producer_ref = save(producer_path, producer)
    save(manifest, {"schema": gate.SCHEMA, "training_scope": gate.TRAINING_SCOPE,
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "producer_packet": producer_ref})
    inner = {"task_id": producer_id, "run_signature": "a"*64, "production_manifest": file_ref(manifest),
        "execution_completed": True,
        "branch_qualification": {"raw": {"status": "PASSED_NON_DEPLOYABLE_DERIVED_SOURCE"}},
        "control_ground_truth": False, "physical_deployment_authorized": False}
    stage_ref = save(root / "_run/producer/output/RESULT.json", inner)
    parent_ref = save(root / "_run/producer/RESULT.json", {"task_id": producer_id, "run_signature": "a"*64,
        "task_terminal_status": "PASSED", "child_exit_code": 0, "stage_results": [stage_ref],
        "control_ground_truth": False, "physical_deployment_authorized": False})
    state_path = root / "docs/governance/state.json"
    state = json.loads(state_path.read_text())
    state["tasks"].append({"task_id": producer_id, "status": "PASSED", "run_signature": "a"*64,
                          "task_packet": producer_ref, "result": parent_ref})
    save(state_path, state)
    packet["inputs"] = {"production_manifest": file_ref(manifest), "prepared_pair": file_ref(manifest),
        "training_config": file_ref(config), "pretrained": file_ref(weights), "pretrained_config": source_config}
    packet.update(execution_mode=gate.EXECUTION_MODE, training_scope=gate.TRAINING_SCOPE, branch="raw")
    packet["commands"] = ["python -m chaoyang.cli run " + gate.OPERATION + " " + " ".join(args)]
    route = {"candidate_operation": gate.OPERATION, "task_packet_required": True, "training_scope": gate.TRAINING_SCOPE,
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "code_closure": packet["runtime_code"]}
    save(root / "docs/governance/contract.json", {"special_status_contracts": {gate.ROUTE: route}})
    def bind():
        old_bind()
        index_path = root / "tasks/current/INDEX.json"
        index = json.loads(index_path.read_text())
        index["task_packets"].append({"task_id": producer_id, "packet_path": str(producer_path.relative_to(root)),
            "packet_sha256": producer_ref["sha256"], "execution_allowed": False})
        save(index_path, index)
        receipt_path = root / "docs/governance/CURRENT_STATUS_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text()); receipt["files"]["task_packet_index_current"] = file_ref(index_path)
        save(receipt_path, receipt)
    bind()
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: True)
    return root, path, manifest, config, weights, output, args, packet, bind, lease_path, lease


def admit(f, world=1, branch="raw", updates=20000):
    root, packet, manifest, config, weights, output, args, *_ = f
    return gate.authorize_production_training(root, packet, args, manifest=manifest, config=config,
        pretrained=weights, output=output, branch=branch, updates=updates, world_size=world)


def config(world=1):
    ref = {"sha256": "a" * 64}
    return cli.make_training_config(60, manifest_ref=ref, pretrained_ref=ref, pretrained_config_ref=ref, world_size=world)


def test_config_global64_same_single_eight_no_mutation():
    for world in (1, 8):
        value = config(world); before = copy.deepcopy(value)
        cli.validate_config(value, world)
        assert value == before
        assert value["training"]["micro_batch"] * world == value["training"]["global_batch"] == 64
        assert value["training"]["checkpoint_every"] == 1000 and value["training"]["max_updates"] == 20000
        assert value["minimum_export_margin_seconds"] == 1200


def test_no_wrong_topology_microbatch_or_unnecessary_accumulation():
    with pytest.raises(ValueError): cli.validate_config(config(1), 8)
    with pytest.raises(ValueError): cli.validate_config(config(8), 1)


def test_config_factory_rejects_unadmitted_world_sizes():
    for world in (0, 2, 4, True):
        with pytest.raises(ValueError): config(world)


@pytest.mark.parametrize("field,value", [("global_batch", 32), ("micro_batch", 4), ("precision", "fp16"),
    ("max_updates", 1), ("checkpoint_every", 100), ("wall_seconds_max", float("nan"))])
def test_no_relaxed_or_unbounded_recipe(field, value):
    cfg = config(); cfg["training"][field] = value
    with pytest.raises(ValueError): cli.validate_config(cfg, 1)


@pytest.mark.parametrize("field,value", [("training_scope", "REAL_CONTROL"), ("physical_deployment_authorized", True),
    ("control_ground_truth", True), ("minimum_export_margin_seconds", 1199), ("initialization", "RANDOM_INIT")])
def test_explicit_scope_fresh_and_export_margin(field, value):
    cfg = config(); cfg[field] = value
    with pytest.raises(ValueError): cli.validate_config(cfg, 1)


def test_missing_config_sha_binding_refused():
    cfg = config(); cfg["input_bindings"].pop("pretrained_sha256")
    with pytest.raises(ValueError, match="SHA_BINDINGS"): cli.validate_config(cfg, 1)


def test_single_eight_all_gpu_lease(admitted):
    assert admit(admitted)["branch"] == "raw"
    admitted[7]["allocated_gpu_ids"] = list(range(8)); admitted[7]["budgets"]["gpu_count"] = 8; admitted[8]()
    with pytest.raises(ValueError, match="ALL_GPUS"): admit(admitted, 8)
    admitted[10]["allocated_gpu_ids"] = list(range(8)); save(admitted[9], admitted[10])
    assert len(admit(admitted, 8)["allocated_gpu_ids"]) == 8


@pytest.mark.parametrize("mode", ["CPU_DEVELOPMENT_ONLY", "HUMANEGO_ICT_REVO2_CANDIDATE_TRAINING",
    "HUMANEGO_ICT_RESEARCH_DERIVED_TRAINING", gate.PREPARE_MODE])
def test_successor_not_old_or_development_task(admitted, mode):
    admitted[7]["execution_mode"] = mode; admitted[8]()
    with pytest.raises(ValueError, match="SUCCESSOR_PRODUCTION_TASK"): admit(admitted)


def test_branch_not_parallel_or_reused(admitted):
    with pytest.raises(ValueError, match="ONE_BRANCH"): admit(admitted, branch="robotized")


@pytest.mark.parametrize("field,value", [("commands", []), ("runtime_code", []), ("gpu_lease_verified", False),
    ("deadline_at", "2000-01-01T00:00:00+00:00"), ("deadline_at", "2099-01-01T00:00:00")])
def test_authority_never_inferred(admitted, field, value):
    admitted[7][field] = value; admitted[8]()
    with pytest.raises(ValueError): admit(admitted)


def test_fixed_wall_and_fresh_weight_config_sha(admitted):
    assert admit(admitted)["admitted_authorization_expires_at"] == admit(admitted)["admitted_authorization_expires_at"]
    source = Path(admitted[7]["inputs"]["pretrained_config"]["path"])
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="SHA_DRIFT"): admit(admitted)


def test_duplicate_writer_state_refused(admitted):
    root = admitted[0]
    save(root / "docs/governance/state.json", {"tasks": [{"status": "RUNNING"}, {"status": "RUNNING"}]})
    admitted[8]()
    with pytest.raises(ValueError, match="ANCESTRY"): admit(admitted)


def test_live_ancestry_not_pid_only(admitted, monkeypatch):
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: False)
    with pytest.raises(ValueError, match="ANCESTRY"): admit(admitted)


@pytest.mark.parametrize("kind", ["failed_source", "bad_manifest", "branch_fail", "parent_signature", "missing_inner", "partial_execution", "child_failed"])
def test_aligned_files_not_a_sealed_legitimate_producer(admitted, kind):
    root = admitted[0]; state_path = root / "docs/governance/state.json"
    state = json.loads(state_path.read_text())
    row = next(r for r in state["tasks"] if r["task_id"] == "synthetic-source-producer")
    parent_path = Path(row["result"]["path"]); parent = json.loads(parent_path.read_text())
    inner_path = Path(parent["stage_results"][0]["path"]); inner = json.loads(inner_path.read_text())
    if kind == "failed_source": row["status"] = "FAILED_RUNTIME_FINAL"
    elif kind == "parent_signature": parent["run_signature"] = "b"*64
    elif kind == "missing_inner": parent["stage_results"] = []
    elif kind == "child_failed": parent["child_exit_code"] = 2
    else:
        if kind == "bad_manifest": inner["production_manifest"] = {"sha256": "b"*64}
        elif kind == "branch_fail": inner["branch_qualification"]["raw"]["status"] = "FAILED"
        else: inner["execution_completed"] = False
        parent["stage_results"] = [save(inner_path, inner)]
    row["result"] = save(parent_path, parent); save(state_path, state); admitted[8]()
    with pytest.raises(ValueError, match="PRODUCER|SEALED"): admit(admitted)


def test_missing_producer_packet_reference_refused(admitted):
    manifest = admitted[2]; value = json.loads(manifest.read_text()); value.pop("producer_packet")
    ref = save(manifest, value)
    admitted[7]["inputs"].update(production_manifest=ref, prepared_pair=ref); admitted[8]()
    with pytest.raises(ValueError, match="SEALED_PRODUCTION_PACKET"): admit(admitted)


def test_unregistered_sealed_producer_not_inferred(admitted):
    root = admitted[0]; path = root / "tasks/current/INDEX.json"
    index = json.loads(path.read_text())
    index["task_packets"] = [r for r in index["task_packets"] if r["task_id"] != "synthetic-source-producer"]
    ref = save(path, index)
    receipt_path = root / "docs/governance/CURRENT_STATUS_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text()); receipt["files"]["task_packet_index_current"] = ref; save(receipt_path, receipt)
    with pytest.raises(ValueError, match="SEALED_PASSED_PRODUCER_AUTHORITY"): admit(admitted)


def test_prepare_requires_registered_successor_zero_model_budget(admitted):
    root, packet_path, _, _, _, output, _, packet, bind, *_ = admitted
    packet.update(execution_mode=gate.PREPARE_MODE)
    packet["budgets"].update(gpu_count=0, model_calls=0, training_steps=0)
    args = ["--task-packet", str(packet_path), "--output-dir", str(output)]
    packet["commands"] = ["python -m chaoyang.cli run " + gate.PREPARE + " " + " ".join(args)]
    packet["inputs"].update(resolved_input_manifest=save(root / "inputs/resolved.json", {}),
        recipe=save(root / "inputs/recipe.json", {}), legacy_mapper_refs=[save(root / "inputs/mapper.txt", {"source": True})])
    packet["inputs"]["source_contract"] = save(root / "inputs/source_contract.json", {"input_manifest_ref": packet["inputs"]["resolved_input_manifest"]})
    save(root / "docs/governance/contract.json", {"special_status_contracts": {gate.PREPARE_ROUTE: {
        "candidate_operation": gate.PREPARE, "task_packet_required": True, "training_scope": gate.TRAINING_SCOPE,
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "code_closure": packet["runtime_code"]}}})
    bind()
    assert gate.authorize_production_prepare(root, packet_path, args, output=output)["budgets"]["gpu_count"] == 0
    packet["budgets"]["model_calls"] = 1; bind()
    with pytest.raises(ValueError, match="CPU_SOURCE"): gate.authorize_production_prepare(root, packet_path, args, output=output)


def test_governance_shared_snapshot_waits_for_existing_exclusive_publisher(admitted, monkeypatch):
    root = admitted[0]; path = root / "docs/governance/CURRENT_STATUS_RECEIPT.json"
    receipt = json.loads(path.read_text()); receipt.update(governance_revision=1, generation_id="old")
    save(path, receipt); calls, results, errors = [], [], []
    lock_path = root / "_run/current/locks/governance.lock"
    started, done = threading.Event(), threading.Event()
    def once(*args, **kwargs):
        calls.append(kwargs["receipt"])
        assert kwargs["receipt"]["generation_id"] == "new"
        with lock_path.open("rb") as other_writer:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other_writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {"freshly_revalidated": True}
    monkeypatch.setattr(gate, "_authority_once", once)
    def reader():
        started.set()
        try: results.append(gate._authority(root, admitted[1], gate.OPERATION, admitted[6], admitted[5]))
        except Exception as error: errors.append(error)
        finally: done.set()
    with lock_path.open("rb") as publisher:
        fcntl.flock(publisher.fileno(), fcntl.LOCK_EX)
        worker = threading.Thread(target=reader); worker.start()
        assert started.wait(1)
        # A multi-file publication may outlast the removed one-second retry.
        assert not done.wait(1.05) and not calls
        save(path, dict(receipt, governance_revision=2, generation_id="new"))
        fcntl.flock(publisher.fileno(), fcntl.LOCK_UN)
    worker.join(timeout=5)
    assert not worker.is_alive() and not errors
    assert results == [{"freshly_revalidated": True}] and len(calls) == 1
    with lock_path.open("rb") as writer:
        fcntl.flock(writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("kind", ["input", "code", "fence", "same_generation"])
def test_shared_snapshot_does_not_retry_real_drift_or_changed_receipt(admitted, monkeypatch, kind):
    root = admitted[0]; path = root / "docs/governance/CURRENT_STATUS_RECEIPT.json"
    receipt = json.loads(path.read_text()); receipt.update(governance_revision=1, generation_id="old")
    save(path, receipt); calls = []
    def once(*args, **kwargs):
        calls.append(True)
        if kind == "same_generation":
            save(path, dict(receipt, governance_revision=2))
            location = receipt["files"]["task_state"]["path"]
        else: location = str(admitted[4] if kind == "input" else root / "code.json")
        raise ValueError("WRITER_FENCING_TOKEN_DRIFT" if kind == "fence" else "ARTIFACT_SHA_DRIFT:" + location)
    monkeypatch.setattr(gate, "_authority_once", once)
    with pytest.raises(ValueError): gate._authority(root, admitted[1], gate.OPERATION, admitted[6], admitted[5])
    assert len(calls) == 1
    with (root / "_run/current/locks/governance.lock").open("rb") as writer:
        fcntl.flock(writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_missing_existing_governance_lock_rejected_before_snapshot(admitted, monkeypatch):
    root = admitted[0]
    (root / "_run/current/locks/governance.lock").unlink()
    monkeypatch.setattr(gate, "_authority_once", lambda *args, **kwargs: pytest.fail("must reject before snapshot"))
    with pytest.raises((ValueError, FileNotFoundError)):
        gate._authority(root, admitted[1], gate.OPERATION, admitted[6], admitted[5])
    assert not (root / "_run/current/locks/governance.lock").exists()


@pytest.mark.parametrize("arguments", [["--help"], ["preflight", "--manifest", "MISSING.json", "--branch", "raw"],
    ["train", "--task-packet", "NO.json", "--manifest", "NO.json", "--config", "NO.json",
     "--pretrained", "NO.pt", "--output-dir", "NO_OUTPUT", "--branch", "raw"]])
def test_rejected_cli_before_torch_import(arguments):
    script = """
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname.endswith('.FlowMatchingModel'):
            raise AssertionError('unexpected model import')
sys.meta_path.insert(0,Deny())
from chaoyang.humanego_ict.production_training_cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, *arguments], capture_output=True, text=True)
    assert result.returncode == (0 if arguments == ["--help"] else 2), result.stderr
    assert "unexpected model import" not in result.stderr


def test_shell_syntax_and_no_cache_on_rejection(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/humanego_ict/train_production_pick_place.sh"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    output = tmp_path / "never_created"
    result = subprocess.run(["bash", str(script), "raw", "1", "NO.json", "NO.json", str(output), "NO.json", "NO.pt"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 2 and not output.exists()


def tiny_weights(tmp_path):
    import torch
    model = {"vision_embed_dim": 8, "patch_size": 2, "num_decoder_layers": 1, "num_heads": 2,
             "mlp_ratio": 2.0, "pred_horizon": 50, "max_ict": 8, "use_pre_norm": True, "use_ctx_norm": True}
    tensors = {key: torch.ones(shape) for key, shape in cli.shared_schema(model).items()}
    path = tmp_path / "tiny.pt"; torch.save({"model": tensors}, path)
    source = save(tmp_path / "source_config.json", model)
    return model, tensors, path, source


def test_safe_cpu_schema_not_model_initialization(tmp_path):
    model, tensors, path, source = tiny_weights(tmp_path)
    selected, report = cli.inspect_pretrained(file_ref(path), source, model)
    assert set(selected) == set(tensors) and report["loaded_parameter_elements"] > 0
    assert all(t.device.type == "cpu" for t in selected.values())


@pytest.mark.parametrize("kind", ["empty", "missing_visual", "missing_decoder", "bad_shape", "nonfinite"])
def test_empty_or_corrupted_transfer_never_success(tmp_path, kind):
    import torch
    model, tensors, path, source = tiny_weights(tmp_path)
    if kind == "empty": tensors = {}
    elif kind == "missing_visual": tensors.pop("rgb_embed.proj.weight")
    elif kind == "missing_decoder": tensors.pop("decoder.0.norm1.weight")
    elif kind == "bad_shape": tensors["rgb_embed.proj.weight"] = torch.ones(1)
    else: tensors["rgb_embed.proj.weight"].fill_(float("nan"))
    torch.save({"model": tensors}, path)
    with pytest.raises(ValueError, match="SHARED_PRETRAINED"): cli.inspect_pretrained(file_ref(path), source, model)


def test_safe_cpu_reader_never_unsafe_fallback(tmp_path):
    model, tensors, path, source = tiny_weights(tmp_path)
    path.write_bytes(b"not a valid checkpoint")
    with pytest.raises(ValueError, match="SAFE_CPU_PRETRAINED_READ_FAILED"):
        cli.inspect_pretrained(file_ref(path), source, model)


def test_architecture_heads_semantics_bound_even_equal_shapes(tmp_path):
    model, tensors, path, source = tiny_weights(tmp_path)
    model["num_heads"] = 1
    with pytest.raises(ValueError, match="ARCHITECTURE_MISMATCH"): cli.inspect_pretrained(file_ref(path), source, model)


def test_export_envelope_not_legacy_admission():
    pair = {"schema": gate.SCHEMA, "samples": [{"label_ref": {"sha256": "a"*64}}], "normalization": {"method": "IDENTITY_NO_FIT"}}
    reference = {"sha256": "b"*64}
    result = cli.export_dataset_envelope({"prepared_pair": pair}, reference)
    assert result["samples"] == pair["samples"] and pair["schema"] == gate.SCHEMA
    assert result["production_manifest_ref"] == reference
    assert result["metadata_qualification"] == "NUMERIC_EXPORT_METADATA_NOT_OLD_READER_QUALIFICATION"
    assert result["legacy_reader_qualification"] == "NOT_CLAIMED" and not result["physical_deployment_authorized"]


@pytest.mark.parametrize("mutation", ["manifest", "pretrained", "pretrained_config", "parsed_config", "pair", "margin"])
def test_admitted_config_source_binding_and_margin_before_torch(mutation):
    import hashlib
    cfg = config(); encoded = json.dumps(cfg).encode()
    ref = {"sha256": "a"*64}
    packet = {"inputs": {"production_manifest": ref, "pretrained": ref, "pretrained_config": ref,
        "prepared_pair": ref, "training_config": {"sha256": hashlib.sha256(encoded).hexdigest()}},
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "admitted_authorization_expires_at": datetime.now(timezone.utc).timestamp()+3600}
    qualified = {"dataset_sha": "a"*64, "prepared_pair_ref": ref}
    cli.validate_admitted_inputs(cfg, encoded, packet, qualified)
    if mutation == "manifest": qualified["dataset_sha"] = "b"*64
    elif mutation == "parsed_config": encoded = b"changed"
    elif mutation == "pair": qualified["prepared_pair_ref"] = {"sha256": "b"*64}
    elif mutation == "margin": packet["admitted_authorization_expires_at"] = datetime.now(timezone.utc).timestamp()+1200
    else: packet["inputs"][mutation] = {"sha256": "b"*64}
    with pytest.raises(ValueError): cli.validate_admitted_inputs(cfg, encoded, packet, qualified)


@pytest.mark.parametrize("elapsed,accepted", [(600, True), (2950, False)])
def test_initialization_time_consumes_budget_without_resetting_upper_bound(elapsed, accepted):
    import hashlib
    cfg = config(); cfg["training"]["wall_seconds_max"] = 1800
    encoded = json.dumps(cfg).encode(); ref = {"sha256": "a"*64}
    started = datetime.now(timezone.utc) - timedelta(seconds=elapsed)
    packet = {"inputs": {"production_manifest": ref, "pretrained": ref, "pretrained_config": ref,
        "prepared_pair": ref, "training_config": {"sha256": hashlib.sha256(encoded).hexdigest()}},
        "execution_started_at": started.isoformat(),
        "admitted_authorization_expires_at": started.timestamp()+3001}
    qualified = {"dataset_sha": "a"*64, "prepared_pair_ref": ref}
    if accepted:
        # CPU admission consumed ten minutes: train max is not restarted at now.
        cli.validate_admitted_inputs(cfg, encoded, packet, qualified)
    else:
        with pytest.raises(ValueError, match="INITIALIZATION_TIME_CONSUMES"):
            cli.validate_admitted_inputs(cfg, encoded, packet, qualified)


def test_empty_transfer_guard_before_load_or_report(tmp_path, monkeypatch):
    class Model:
        def state_dict(self): raise AssertionError("empty transfer must reject before model access")
    monkeypatch.setattr(cli, "inspect_pretrained", lambda *args: ({}, {}))
    packet = {"inputs": {"pretrained": {}, "pretrained_config": {}}}
    with pytest.raises(ValueError, match="EMPTY_OR_INCOMPLETE"):
        cli.initialize_pretrained(Model(), packet, {"model": {}}, tmp_path, tmp_path)
    assert not (tmp_path / "pretrained_load_report.json").exists()


def test_zero_space_storage_gate_not_lowered(tmp_path, monkeypatch):
    from chaoyang.humanego_ict import runtime_gate
    monkeypatch.setattr(runtime_gate.os, "statvfs", lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096))
    with pytest.raises(ValueError, match="BOUND_MEASURED"):
        cli.production_storage_ready(tmp_path / "out", 4*1024**3,
            packet={"inputs": {}, "execution_mode": gate.EXECUTION_MODE}, root=tmp_path)


def test_runtime_cache_symlink_escape_rejected(tmp_path):
    output = tmp_path / "out"; output.mkdir()
    external = tmp_path / "other"; external.mkdir()
    (output / "runtime_cache").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="CACHE_ESCAPES"): runtime_cache_paths(output, tmp_path)


def capacity_observation(tmp_path):
    import os
    now = datetime.now(timezone.utc)
    return {"schema": "task-local-write-capacity-observation-v1",
        "status": "PASSED_MEASURED_WRITE_READBACK", "probe_path": str(tmp_path / "probe.bin"),
        "required_bytes": 4*1024**3, "written_bytes": 4*1024**3, "fsynced_bytes": 4*1024**3,
        "device_id": os.stat(tmp_path).st_dev, "statvfs_free_before": 0, "statvfs_free_after": 0,
        "write_sha256": "a"*64, "read_sha256": "a"*64, "task_owned_probe_released": True,
        "measured_capacity_not_general_free_space_guarantee": True, "source_data_or_existing_files_removed": False,
        "started_at": (now-timedelta(seconds=2)).isoformat(), "completed_at": (now-timedelta(seconds=1)).isoformat()}


def test_zero_statvfs_new_route_bound_measured_not_claimed_free(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.os, "statvfs", lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096))
    ref = save(tmp_path / "observation.json", capacity_observation(tmp_path))
    packet = {"inputs": {"storage_capacity_observation": ref}, "execution_mode": gate.EXECUTION_MODE}
    value = gate.production_storage_ready(tmp_path / "output", 4*1024**3, packet=packet, root=tmp_path)
    assert value["mode"] == "MEASURED_WRITE_READBACK" and value["statvfs_reported_free_bytes"] == 0
    assert value["measured_capacity_not_general_free_space_guarantee"]
    packet["execution_mode"] = "HUMANEGO_ICT_REVO2_CANDIDATE_TRAINING"
    with pytest.raises(ValueError, match="ONLY_NEW_PRODUCTION"):
        gate.production_storage_ready(tmp_path / "output", 4*1024**3, packet=packet, root=tmp_path)


@pytest.mark.parametrize("mutation", ["short", "no_fsync", "bad_read", "stale", "future", "device", "unreleased", "deleted_source", "free_claim", "fixture"])
def test_measured_capacity_cannot_lower_threshold_or_invent_proof(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(gate.os, "statvfs", lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096))
    value = capacity_observation(tmp_path)
    if mutation == "short": value["written_bytes"] -= 1
    elif mutation == "no_fsync": value["fsynced_bytes"] = 0
    elif mutation == "bad_read": value["read_sha256"] = "b"*64
    elif mutation == "stale":
        value["completed_at"] = (datetime.now(timezone.utc)-timedelta(hours=2)).isoformat()
        value["started_at"] = (datetime.now(timezone.utc)-timedelta(hours=3)).isoformat()
    elif mutation == "future": value["completed_at"] = (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    elif mutation == "device": value["device_id"] += 1
    elif mutation == "unreleased": value["task_owned_probe_released"] = False
    elif mutation == "deleted_source": value["source_data_or_existing_files_removed"] = True
    elif mutation == "free_claim": value["measured_capacity_not_general_free_space_guarantee"] = False
    else: value["fixture_only"] = True
    packet = {"inputs": {"storage_capacity_observation": save(tmp_path / "observation.json", value)}, "execution_mode": gate.EXECUTION_MODE}
    with pytest.raises(ValueError, match="EVIDENCE_UNQUALIFIED"):
        gate.production_storage_ready(tmp_path / "output", 4*1024**3, packet=packet, root=tmp_path)


def test_low_nonzero_statvfs_does_not_use_measured_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1, f_frsize=4096))
    with pytest.raises(ValueError, match="STORAGE_CAPACITY"):
        gate.production_storage_ready(tmp_path / "out", 4*1024**3, packet={}, root=tmp_path)


def test_normal_statvfs_retains_original_capacity_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1024**2, f_frsize=4096))
    result = gate.production_storage_ready(tmp_path / "out", 4*1024**3, packet={}, root=tmp_path)
    assert result["mode"] == "STATVFS_REPORTED_CAPACITY"


def test_mock_live_training_export_fresh_authorizer_and_margin(tmp_path, monkeypatch):
    """A fake consumer exercises wrapper wiring, never real training or a model."""
    from chaoyang.humanego_ict import checkpoint_export
    root, output = tmp_path, tmp_path / "out"
    manifest, cfg_path, weights, source_cfg = [tmp_path / name for name in ("manifest.json", "config.json", "weights.json", "source.json")]
    pair = {"schema": gate.SCHEMA, "normalization": {"method": "IDENTITY_NO_FIT", "position_mean": [0,0,0], "position_std": [1,1,1]},
            "pose_clock": {"tcp_frame_ids": ["SYNTHETIC_L", "SYNTHETIC_R"]}, "samples": [{"synthetic_only": True}]}
    ref = save(manifest, pair); weight_ref = save(weights, {"synthetic_only": True}); source_ref = save(source_cfg, {})
    cfg = cli.make_training_config(60, manifest_ref=ref, pretrained_ref=weight_ref, pretrained_config_ref=source_ref)
    save(cfg_path, cfg)
    packet = {"task_id": "synthetic-only", "run_signature": "a"*64,
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "runtime_code": [file_ref(Path(cli.__file__))], "allocated_gpu_ids": [0], "inputs": {
            "production_manifest": ref, "prepared_pair": ref, "pretrained": weight_ref, "pretrained_config": source_ref,
            "training_config": file_ref(cfg_path)}, "admitted_authorization_expires_at": datetime.now(timezone.utc).timestamp()+3600}
    admitted_calls = []
    def authorize(*args, **kwargs):
        admitted_calls.append(True)
        return dict(packet)
    monkeypatch.setattr(cli, "repository_root", lambda: root)
    monkeypatch.setattr(cli, "authorize_production_training", authorize)
    monkeypatch.setattr(cli, "production_storage_ready", lambda *a, **k: {"mode": "SYNTHETIC_TEST"})
    monkeypatch.setattr(cli, "bind_cuda_visible_devices", lambda *a: "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("ICT_PRODUCTION_ADMISSION_ONLY", raising=False)
    for name, path in zip(("TMPDIR", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME"), runtime_cache_paths(output, root)):
        monkeypatch.setenv(name, str(path))
    dependency_checks = []
    monkeypatch.setitem(sys.modules, "chaoyang.humanego_ict.production_data", SimpleNamespace(
        validate_production_manifest=lambda *a: {"dataset_sha": ref["sha256"], "prepared_pair_ref": ref, "prepared_pair": pair},
        ProductionDerivedICTDataset=lambda *a: SimpleNamespace(verify_all_dependencies=lambda: dependency_checks.append(True))))
    state = {"mock_marker": "not_a_tensor"}
    class Model:
        def __init__(self, **kwargs): pass
        def to(self, device): return self
        def state_dict(self): return state
    monkeypatch.setitem(sys.modules, "chaoyang.humanego_ict.model", SimpleNamespace(Revo2ICTModel=Model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
        device_count=lambda: 1, set_device=lambda _: None), manual_seed=lambda _: None))
    initializers = []
    monkeypatch.setattr(cli, "initialize_pretrained", lambda *args: initializers.append(args))
    def train(model, dataset, config, out, signatures, **kwargs):
        assert signatures["prepared_pair"] == ref and signatures["branch"] == "raw"
        assert kwargs["authorization_expires_at"] == packet["admitted_authorization_expires_at"]-1200
        kwargs["initializer"](model)
        return {"optimizer_updates": 1, "status": "SYNTHETIC_WRAPPER_TEST_NOT_REAL_TRAINING"}
    monkeypatch.setitem(sys.modules, "chaoyang.humanego_ict.training", SimpleNamespace(TrainingConfig=lambda **k: k, train_candidate=train))
    exports = []
    def export(live, **kwargs):
        assert live is state
        provenance = kwargs["provenance"]
        assert provenance["production_manifest_ref"] == ref and provenance["training_scope"] == gate.TRAINING_SCOPE
        assert kwargs["runtime_authorizer"](provenance)["export_authorized"]
        changed = dict(provenance, task_id="wrong")
        with pytest.raises(ValueError, match="EXPORT_PRODUCER_BINDING"): kwargs["runtime_authorizer"](changed)
        metadata = json.loads(Path(provenance["dataset_ref"]["path"]).read_text())
        assert metadata["legacy_reader_qualification"] == "NOT_CLAIMED" and metadata["samples"] == pair["samples"]
        exports.append(provenance)
        return {"synthetic_only": True}
    monkeypatch.setattr(checkpoint_export, "export_tensor_candidate", export)
    args = ["train", "--task-packet", str(tmp_path / "packet.json"), "--manifest", str(manifest), "--branch", "raw",
        "--config", str(cfg_path), "--output-dir", str(output), "--pretrained", str(weights)]
    assert cli.main(args) == 0 and len(initializers) == 1 and len(exports) == 1 and len(admitted_calls) >= 4
    assert len(dependency_checks) == 2
    assert (output / "INFERENCE_EXPORT.json").is_file()


def test_prepare_wrapper_storage_before_lazy_producer_and_fresh_exit(tmp_path, monkeypatch):
    events = []
    packet = {"execution_mode": gate.PREPARE_MODE}
    def authorize(*args, **kwargs):
        events.append("authorize")
        return dict(packet)
    def storage(*args, **kwargs):
        events.append("storage")
        return {"mode": "SYNTHETIC_NOT_ACTUAL_CAPACITY"}
    def execute(module):
        events.append("import_frozen_producer")
        def prepare(packet, root):
            assert packet["storage_admission"]["mode"] == "SYNTHETIC_NOT_ACTUAL_CAPACITY"
            events.append("prepare")
            return {"status": "SYNTHETIC_NOT_REAL_PRODUCTION"}
        module.prepare = prepare
    monkeypatch.setattr(cli, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "authorize_production_prepare", authorize)
    monkeypatch.setattr(cli, "production_storage_ready", storage)
    monkeypatch.setattr(cli.importlib.util, "spec_from_file_location", lambda *args: SimpleNamespace(name="synthetic_production_wrapper", loader=SimpleNamespace(exec_module=execute)))
    monkeypatch.setattr(cli.importlib.util, "module_from_spec", lambda _: SimpleNamespace())
    assert cli.prepare_main(["--task-packet", str(tmp_path / "packet.json"), "--output-dir", str(tmp_path / "out")]) == 0
    assert events == ["authorize", "storage", "import_frozen_producer", "prepare", "authorize"]
