"""Bounded synthetic admission tests; no model, optimizer or GPU operation."""
from datetime import datetime, timedelta, timezone
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaoyang.humanego_ict import production_resume_gate as gate
from chaoyang.humanego_ict import production_training_gate as original_gate
from chaoyang.humanego_ict.production_training_cli import make_training_config
from chaoyang.humanego_ict.runtime_gate import file_ref


spec = importlib.util.spec_from_file_location("resume_production_fixture", Path(__file__).with_name("test_production_training.py"))
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
save = previous.save


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    original = previous.admitted.__wrapped__(tmp_path, monkeypatch)
    root, path, manifest, config_path, pretrained, output, _, packet, old_bind, lease_path, lease = original
    old_closure = copy.deepcopy(packet["runtime_code"])
    required_code = [save(root / relative, {"synthetic_code": relative}) for relative in gate.ENGINEERING_CODE_PATHS]
    test_ref = save(root / "tests/test_synthetic_resume.py", {"synthetic_test": True})
    packet["runtime_code"] = old_closure + required_code
    cfg = make_training_config(120, manifest_ref=file_ref(manifest), pretrained_ref=file_ref(pretrained),
                               pretrained_config_ref=packet["inputs"]["pretrained_config"])
    parent_id = "synthetic-sealed-training-parent"
    parent_path = root / "tasks/current" / parent_id / "TASK_PACKET.json"
    parent_output = root / "_run/parent/attempts/attempt_0001/output"
    parent_config_ref = save(parent_output.parent / "TRAINING_CONFIG.json", cfg)
    parent_packet = copy.deepcopy(packet)
    parent_packet.update(task_id=parent_id, run_signature="b" * 64, output_dir=str(parent_output), runtime_code=old_closure)
    parent_packet["inputs"]["training_config"] = parent_config_ref
    parent_packet_ref = save(parent_path, parent_packet)
    checkpoint_ref = save(parent_output / "checkpoint_004376.pt", {"synthetic_checkpoint_not_loaded": True})
    sidecar_ref = save(Path(str(checkpoint_ref["path"]) + ".ref.json"),
                       {"sha256": checkpoint_ref["sha256"], "bytes": checkpoint_ref["bytes"]})
    inner = {"task_id": parent_id, "run_signature": "b" * 64, "branch": "raw", "optimizer_updates": 4376,
             "cursor": 4376 * 64, "world_size": 1, "last_checkpoint": checkpoint_ref["path"],
             "completed_update_budget": False, "training_scope": gate.TRAINING_SCOPE,
             "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
             "physical_deployment_authorized": False}
    inner_ref = save(parent_output / "RESULT_004376.json", inner)
    tensor_output = parent_output / "inference_tensors"
    receipt_ref = save(tensor_output / "EXPORT_RECEIPT.json", {
        "status": "TENSOR_EXPORT_SAFE_ROUNDTRIP_VERIFIED_NOT_MODEL_QUALITY", "fixture_only": False,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "branch": "raw", "optimizer_updates": 4376,
        "training_provenance": {"task_id": parent_id, "run_signature": "b" * 64, "branch": "raw",
            "optimizer_updates": 4376, "training_scope": gate.TRAINING_SCOPE,
            "training_config_ref": parent_config_ref, "training_result_ref": inner_ref}})
    export_ref = save(parent_output / "INFERENCE_EXPORT.json", {
        "checkpoint_ref": save(tensor_output / "candidate_tensor.pt", {"synthetic": True}),
        "export_receipt_ref": receipt_ref,
        "inference_contract_ref": save(tensor_output / "inference_contract.json", {}),
        "model_config_ref": save(tensor_output / "model_config.json", {}),
        "normalization_ref": save(tensor_output / "normalization.json", {})})
    outer = {"task_id": parent_id, "run_signature": "b" * 64, "task_terminal_status": "PASSED",
             "child_exit_code": 0, "stage_results": [inner_ref], "complete_checkpoints": [checkpoint_ref],
             "inference_export": export_ref, "control_ground_truth": False,
             "physical_deployment_authorized": False}
    outer_ref = save(parent_output.parent / "RESULT.json", outer)
    parents = {"task_packet": parent_packet_ref, "outer_result": outer_ref, "inner_result": inner_ref,
               "training_config": parent_config_ref, "checkpoint": checkpoint_ref, "checkpoint_sidecar": sidecar_ref}
    engineering_id = "synthetic-sealed-resume-engineering"
    engineering_path = root / "tasks/current" / engineering_id / "TASK_PACKET.json"
    engineering_packet_ref = save(engineering_path, {"task_id": engineering_id, "run_signature": "c" * 64,
        "execution_mode": "CPU_DEVELOPMENT_ONLY", "budgets": {"gpu_count": 0, "model_calls": 0, "training_steps": 0}})
    engineering = {"schema": "humanego-ict-checkpoint-continuation-cpu-result-v1", "task_id": engineering_id,
        "run_signature": "c" * 64, "task_terminal_status": "PASSED", "resume_code_refs": required_code + [test_ref],
        "cpu_test_result": save(root / "_run/engineering/CPU_TEST_RESULT.json", {"status": "PASSED", "exit_code": 0, "passed_count": 37}),
        "protected_inputs_check": save(root / "_run/engineering/PROTECTED_INPUTS_CHECK.json",
                                       {"status": "PASSED", "sha_unchanged": True, "checked_refs": old_closure})}
    engineering_ref = save(root / "_run/engineering/RESULT.json", engineering)
    state_path = root / "docs/governance/state.json"
    state = json.loads(state_path.read_text())
    state["tasks"].extend([
        {"task_id": parent_id, "status": "PASSED", "run_signature": "b" * 64, "task_packet": parent_packet_ref, "result": outer_ref},
        {"task_id": engineering_id, "status": "PASSED", "run_signature": "c" * 64,
         "task_packet": engineering_packet_ref, "result": engineering_ref}])
    save(state_path, state)
    packet.update(execution_mode=gate.EXECUTION_MODE)
    packet["inputs"].update({name: parents[role] for role, name in gate.PARENT_INPUT_ROLES.items()})
    packet["inputs"].update(engineering_task_packet=engineering_packet_ref, engineering_task_result=engineering_ref)
    cfg.update(schema_version=gate.CONFIG_SCHEMA, initialization=gate.INITIALIZATION, resume_parent_bindings=parents)
    packet["inputs"]["training_config"] = save(config_path, cfg)
    args = ["train", "--task-packet", str(path), "--manifest", str(manifest), "--config", str(config_path),
            "--checkpoint", checkpoint_ref["path"], "--output-dir", str(output), "--branch", "raw"]
    packet["commands"] = ["python -m chaoyang.cli run " + gate.OPERATION + " " + " ".join(args)]
    route = {"candidate_operation": gate.OPERATION, "task_packet_required": True, "training_scope": gate.TRAINING_SCOPE,
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False, "physical_deployment_authorized": False,
        "resume_support": True, "parent_sealed_checkpoint_required": True, "code_closure": packet["runtime_code"]}
    contract_path = root / "docs/governance/contract.json"
    save(contract_path, {"special_status_contracts": {gate.ROUTE: route}})
    def bind():
        packet["inputs"]["training_config"] = save(config_path, cfg)
        old_bind()
        index_path = root / "tasks/current/INDEX.json"
        index = json.loads(index_path.read_text())
        for task_id, pointer in ((parent_id, parents["task_packet"]), (engineering_id, engineering_packet_ref)):
            index["task_packets"].append({"task_id": task_id, "packet_path": str(Path(pointer["path"]).relative_to(root)),
                                          "packet_sha256": pointer["sha256"], "execution_allowed": False})
        save(index_path, index)
        receipt_path = root / "docs/governance/CURRENT_STATUS_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text()); receipt["files"]["task_packet_index_current"] = file_ref(index_path)
        save(receipt_path, receipt)
    bind()
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: True)
    return {"root": root, "path": path, "manifest": manifest, "config_path": config_path, "cfg": cfg,
            "output": output, "checkpoint": Path(checkpoint_ref["path"]), "parents": parents, "packet": packet,
            "args": args, "bind": bind, "lease_path": lease_path, "lease": lease, "state_path": state_path,
            "engineering": engineering, "engineering_id": engineering_id, "parent_id": parent_id,
            "contract_path": contract_path, "route": route, "old_closure": old_closure}


def admit(f, **kwargs):
    return gate.authorize_production_resume(f["root"], f["path"], f["args"], manifest=f["manifest"],
        config=f["config_path"], checkpoint=f["checkpoint"], output=f["output"],
        branch=kwargs.get("branch", "raw"), updates=kwargs.get("updates", 20000), world_size=kwargs.get("world", 1))


def test_valid_sealed_wall_stop_resume_preserves_fixed_deadline(admitted):
    first, second = admit(admitted), admit(admitted)
    assert first["admitted_authorization_expires_at"] == second["admitted_authorization_expires_at"]
    assert first["admitted_resume_parent"]["optimizer_updates"] == 4376
    assert first["admitted_resume_parent"]["cursor"] == 280064


def test_registered_source_never_old_route_mutation(admitted):
    assert original_gate.OPERATION == "train_pick_place_humanego_ict_production_v1"
    assert original_gate.EXECUTION_MODE == "HUMANEGO_ICT_PRODUCTION_DERIVED_TRAINING"
    assert gate.ROUTE == "humanego_ict_production_resume_v1"
    with pytest.raises(ValueError, match="PRODUCTION_CANDIDATE_CONTRACT"):
        original_gate.authorize_production_training(admitted["root"], admitted["path"], admitted["args"],
            manifest=admitted["manifest"], config=admitted["config_path"], pretrained=Path(admitted["packet"]["inputs"]["pretrained"]["path"]),
            output=admitted["output"], branch="raw", updates=20000, world_size=1)


@pytest.mark.parametrize("mode", ["CPU_DEVELOPMENT_ONLY", original_gate.EXECUTION_MODE, original_gate.PREPARE_MODE])
def test_old_or_development_packets_not_reused(admitted, mode):
    admitted["packet"]["execution_mode"] = mode; admitted["bind"]()
    with pytest.raises(ValueError, match="SUCCESSOR_RESUME_TASK"): admit(admitted)


@pytest.mark.parametrize("field,value", [("commands", []), ("runtime_code", []), ("gpu_lease_verified", False),
    ("deadline_at", "2000-01-01T00:00:00+00:00"), ("deadline_at", "2099-01-01T00:00:00"),
    ("deadline_at", "2000-01-01T00:00:00")])
def test_no_inferred_live_authority(admitted, field, value):
    admitted["packet"][field] = value; admitted["bind"]()
    with pytest.raises(ValueError): admit(admitted)


def test_unique_writer_and_live_ancestry(admitted, monkeypatch):
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: False)
    with pytest.raises(ValueError, match="ANCESTRY"): admit(admitted)


def test_second_active_writer_refused(admitted):
    state = json.loads(admitted["state_path"].read_text()); state["tasks"].append({"status": "RUNNING", "task_id": "rogue"})
    save(admitted["state_path"], state); admitted["bind"]()
    with pytest.raises(ValueError, match="ANCESTRY"): admit(admitted)


def test_fencing_token_not_pid_only(admitted):
    admitted["packet"]["writer"]["fencing_token"] = "different"; admitted["bind"]()
    with pytest.raises(ValueError, match="FENCING"): admit(admitted)


@pytest.mark.parametrize("role", list(gate.PARENT_INPUT_ROLES.values()))
def test_every_parent_reference_required(admitted, role):
    admitted["packet"]["inputs"].pop(role); admitted["bind"]()
    with pytest.raises((ValueError, KeyError)): admit(admitted)


@pytest.mark.parametrize("role", ["production_manifest", "training_config", "resume_checkpoint", "pretrained_config"])
def test_actual_sha_drift_not_retried(admitted, role):
    Path(admitted["packet"]["inputs"][role]["path"]).write_text('{"changed": true}')
    with pytest.raises(ValueError, match="SHA_DRIFT|digest mismatch"): admit(admitted)


@pytest.mark.parametrize("field,value", [("initialization", "FRESH_SHARED_PRETRAINED_NEW_IO"),
    ("schema_version", "old-config"), ("control_ground_truth", True), ("physical_deployment_authorized", True),
    ("minimum_export_margin_seconds", 1199), ("minimum_free_output_bytes", 1)])
def test_no_scope_or_storage_or_export_relaxation(admitted, field, value):
    admitted["cfg"][field] = value; admitted["bind"]()
    with pytest.raises(ValueError): admit(admitted)


@pytest.mark.parametrize("field,value", [("learning_rate", 0.0002), ("seed", 8), ("ema_decay", 0.998)])
def test_only_wall_seconds_can_change_inherited_recipe(admitted, field, value):
    admitted["cfg"]["training"][field] = value; admitted["bind"]()
    with pytest.raises(ValueError, match="RECIPE|recipe"): admit(admitted)


def test_schema_parent_sha_bindings_exact(admitted):
    admitted["cfg"]["resume_parent_bindings"] = dict(admitted["parents"], checkpoint={"sha256": "f" * 64})
    admitted["bind"]()
    with pytest.raises(ValueError, match="PARENT_SHA_BINDINGS"): admit(admitted)


def test_cross_branch_checkpoint_not_allowed(admitted):
    with pytest.raises(ValueError, match="ONE_BRANCH"): admit(admitted, branch="robotized")


def test_parent_code_not_silently_replaced(admitted):
    admitted["packet"]["runtime_code"] = [r for r in admitted["packet"]["runtime_code"] if r not in admitted["old_closure"]]
    admitted["route"]["code_closure"] = admitted["packet"]["runtime_code"]
    save(admitted["contract_path"], {"special_status_contracts": {gate.ROUTE: admitted["route"]}})
    admitted["bind"]()
    with pytest.raises(ValueError, match="PARENT_CODE_CLOSURE"): admit(admitted)


@pytest.mark.parametrize("task", ["parent_id", "engineering_id"])
def test_parent_or_engineering_partial_never_consumed(admitted, task):
    state = json.loads(admitted["state_path"].read_text())
    next(row for row in state["tasks"] if row.get("task_id") == admitted[task])["status"] = "FAILED_RUNTIME_FINAL"
    save(admitted["state_path"], state); admitted["bind"]()
    with pytest.raises(ValueError, match="PASSED.*AUTHORITY"): admit(admitted)


@pytest.mark.parametrize("role", ["engineering_task_packet", "engineering_task_result"])
def test_cpu_engineering_must_be_registered_and_sealed(admitted, role):
    admitted["packet"]["inputs"].pop(role); admitted["bind"]()
    with pytest.raises(ValueError, match="ENGINEERING_REFERENCES"): admit(admitted)


@pytest.mark.parametrize("field", ["resume_support", "parent_sealed_checkpoint_required", "task_packet_required"])
def test_new_contract_explicit_not_just_old_route(admitted, field):
    admitted["route"][field] = False
    save(admitted["contract_path"], {"special_status_contracts": {gate.ROUTE: admitted["route"]}})
    admitted["bind"]()
    with pytest.raises(ValueError, match="CONTRACT_NOT_REGISTERED"): admit(admitted)


def test_expired_or_changed_lease_not_renewed_by_admission(admitted):
    admitted["lease"]["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    save(admitted["lease_path"], admitted["lease"])
    with pytest.raises(ValueError, match="LIVE_GPU_LEASE"): admit(admitted)


@pytest.mark.parametrize("field,value", [("status", "RELEASED"), ("task_id", "other-task"),
    ("pid", 202), ("process_startticks", 456), ("fencing_token", "stale"), ("executor_epoch", 2), ("gpu_id", 3)])
def test_gpu_lease_all_identity_fields_bound(admitted, field, value):
    admitted["lease"][field] = value; save(admitted["lease_path"], admitted["lease"])
    with pytest.raises(ValueError, match="LIVE_GPU_LEASE"): admit(admitted)


def _replace_engineering_result(f, value):
    ref = save(Path(f["packet"]["inputs"]["engineering_task_result"]["path"]), value)
    f["packet"]["inputs"]["engineering_task_result"] = ref
    state = json.loads(f["state_path"].read_text())
    next(row for row in state["tasks"] if row.get("task_id") == f["engineering_id"])["result"] = ref
    save(f["state_path"], state); f["bind"]()


@pytest.mark.parametrize("field,value,error", [("resume_code_refs", [], "CODE_AND_TEST_REFS"),
    ("schema", "wrong", "RESULT_BINDING"), ("task_terminal_status", "PARTIAL", "RESULT_BINDING")])
def test_declared_cpu_pass_not_enough_without_bound_proof(admitted, field, value, error):
    result = copy.deepcopy(admitted["engineering"]); result[field] = value
    _replace_engineering_result(admitted, result)
    with pytest.raises(ValueError, match=error): admit(admitted)


def test_no_tests_sha_cannot_enable_route(admitted):
    result = copy.deepcopy(admitted["engineering"])
    result["resume_code_refs"] = [ref for ref in result["resume_code_refs"] if not Path(ref["path"]).is_relative_to(admitted["root"] / "tests")]
    _replace_engineering_result(admitted, result)
    with pytest.raises(ValueError, match="TEST_FILE_SHA"): admit(admitted)


@pytest.mark.parametrize("kind", ["tests_failed", "tests_zero", "protected_changed"])
def test_failed_engineering_check_cannot_enable_route(admitted, kind):
    result = copy.deepcopy(admitted["engineering"])
    role = "protected_inputs_check" if kind == "protected_changed" else "cpu_test_result"
    path = Path(result[role]["path"]); check = json.loads(path.read_text())
    if kind == "tests_failed": check["exit_code"] = 1
    elif kind == "tests_zero": check["passed_count"] = 0
    else: check["sha_unchanged"] = False
    result[role] = save(path, check); _replace_engineering_result(admitted, result)
    with pytest.raises(ValueError, match="TESTS_NOT_PASSED|PROTECTED_SOURCE_PROOF"): admit(admitted)


def test_cpu_storage_no_small_or_wrong_mode_fallback(admitted):
    for value in (1, True, 4 * 1024**3 - 1):
        with pytest.raises(ValueError, match="NOT_LOWERED"):
            gate.production_resume_storage_ready(admitted["output"], value, packet=admitted["packet"], root=admitted["root"])
    packet = dict(admitted["packet"], execution_mode=original_gate.EXECUTION_MODE)
    with pytest.raises(ValueError, match="ONLY_NEW_RESUME"):
        gate.production_resume_storage_ready(admitted["output"], 4 * 1024**3, packet=packet, root=admitted["root"])


def test_zero_statvfs_requires_bound_fresh_full_readback(admitted, monkeypatch):
    monkeypatch.setattr(gate.os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
    with pytest.raises(ValueError, match="CAPACITY_EVIDENCE"):
        gate.production_resume_storage_ready(admitted["output"], 4 * 1024**3, packet=admitted["packet"], root=admitted["root"])
    observation = {"schema": "task-local-write-capacity-observation-v1", "status": "PASSED_MEASURED_WRITE_READBACK",
        "fixture_only": False, "measured_capacity_not_general_free_space_guarantee": True,
        "source_data_or_existing_files_removed": False, "task_owned_probe_released": True,
        "device_id": admitted["root"].stat().st_dev, "probe_path": str(admitted["root"] / "released_probe"),
        "statvfs_free_before": 0, "statvfs_free_after": 0, "required_bytes": 4 * 1024**3,
        "written_bytes": 4 * 1024**3, "fsynced_bytes": 4 * 1024**3, "write_sha256": "a" * 64, "read_sha256": "a" * 64,
        "started_at": (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        "completed_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    observation_path = admitted["root"] / "observation.json"
    admitted["packet"]["inputs"]["storage_capacity_observation"] = save(observation_path, observation)
    result = gate.production_resume_storage_ready(admitted["output"], 4 * 1024**3, packet=admitted["packet"], root=admitted["root"])
    assert result["mode"] == "MEASURED_WRITE_READBACK" and result["minimum_bytes"] == 4 * 1024**3
    for field, value in (("written_bytes", 4 * 1024**3 - 1), ("read_sha256", "b" * 64),
                         ("device_id", -1), ("completed_at", (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())):
        changed = dict(observation, **{field: value})
        admitted["packet"]["inputs"]["storage_capacity_observation"] = save(observation_path, changed)
        with pytest.raises(ValueError, match="CAPACITY_EVIDENCE_UNQUALIFIED"):
            gate.production_resume_storage_ready(admitted["output"], 4 * 1024**3, packet=admitted["packet"], root=admitted["root"])
