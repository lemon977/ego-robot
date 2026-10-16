"""Synthetic fenced route tests; no actual model construction or training."""
from datetime import datetime, timedelta, timezone
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from chaoyang.humanego_ict import research_training_gate as gate
from chaoyang.humanego_ict.research_training_cli import CONFIG_SCHEMA, validate_config, _check_caches
from chaoyang.humanego_ict.research_data import SCHEMA, TRAINING_SCOPE
from chaoyang.humanego_ict.runtime_gate import file_ref, runtime_cache_paths


def helper(name):
    spec = importlib.util.spec_from_file_location("research_route_fixture_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = helper("test_derived_training")
save = old.save


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    f = old.admitted.__wrapped__(tmp_path, monkeypatch)
    root, packet_path, manifest, config_path, weights, output, args, packet, bind, *_ = f
    pair_ref = save(root / "inputs/prepared_pair.json", {"fixture_only": True, "no_training_claim": True})
    save(manifest, {"schema": SCHEMA, "training_scope": TRAINING_SCOPE, "supervision_class": "DERIVED_RETARGET",
        "control_ground_truth": False, "physical_deployment_authorized": False, "prepared_pair_ref": pair_ref})
    packet["inputs"].pop("paired_manifest")
    packet["inputs"].update(research_manifest=file_ref(manifest), prepared_pair=pair_ref)
    packet["execution_mode"], packet["training_scope"] = gate.EXECUTION_MODE, TRAINING_SCOPE
    packet["commands"] = ["python -m chaoyang.cli run " + gate.OPERATION + " " + " ".join(args)]
    save(root / "docs/governance/contract.json", {"special_status_contracts": {gate.ROUTE: {
        "candidate_operation": gate.OPERATION, "task_packet_required": True, "training_scope": TRAINING_SCOPE,
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "code_closure": packet["runtime_code"]}}})
    bind()
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: True)
    return f


def admit(fixture, world=1):
    root, packet, manifest, config_path, weights, output, args, *_ = fixture
    return gate.authorize_research_training(root, packet, args, manifest=manifest, config=config_path,
        pretrained=weights, output=output, updates=20000, world_size=world)


def config():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/quick_completion_v1/train_research_20000.json").read_text())


def test_new_recipe_retains_old_numerical_recipe_and_no_mutation():
    cfg, previous = config(), old.config()
    frozen = copy.deepcopy(cfg)
    for world in (1,8): validate_config(cfg, world)
    assert cfg == frozen
    assert cfg["model"] == previous["model"] and cfg["training"] == previous["training"]
    assert cfg["training"]["global_batch"] * cfg["training"]["max_updates"] == 1_280_000


@pytest.mark.parametrize("mutation", ["old_config", "scope", "gt", "physical", "horizon", "batch", "precision", "nan_wall"])
def test_research_config_not_an_old_route_or_weakened_recipe(mutation):
    cfg = config()
    if mutation == "old_config": cfg["schema_version"] = "pick-place-humanego-ict-derived-training-config-v1"
    elif mutation == "scope": cfg["training_scope"] = "DEPLOYABLE_CONTROL"
    elif mutation == "gt": cfg["control_ground_truth"] = True
    elif mutation == "physical": cfg["physical_deployment_authorized"] = True
    elif mutation == "horizon": cfg["model"]["pred_horizon"] = 30
    elif mutation == "batch": cfg["training"]["global_batch"] = 32
    elif mutation == "precision": cfg["training"]["precision"] = "fp16"
    else: cfg["training"]["wall_seconds_max"] = float("nan")
    with pytest.raises(ValueError): validate_config(cfg, 8)


def test_finite_single_or_eight_gpu_admission(admitted):
    assert admit(admitted)["admitted_authorization_expires_at"] > datetime.now(timezone.utc).timestamp()
    admitted[7]["allocated_gpu_ids"] = list(range(8))
    admitted[7]["budgets"]["gpu_count"] = 8
    admitted[8]()
    with pytest.raises(ValueError, match="ALL_GPUS"): admit(admitted,8)
    admitted[10]["allocated_gpu_ids"] = list(range(8))
    save(admitted[9], admitted[10])
    assert admit(admitted,8)["allocated_gpu_ids"] == list(range(8))


@pytest.mark.parametrize("mode", ["CPU_DEVELOPMENT_ONLY", "HUMANEGO_ICT_DERIVED_RETARGET_TRAINING",
    "HUMANEGO_ICT_REVO2_CANDIDATE_TRAINING", "HUMANEGO_ICT_REVO2_OFFLINE_INFERENCE"])
def test_old_or_cpu_tasks_never_authorize_research_training(admitted, mode):
    admitted[7]["execution_mode"] = mode
    admitted[8]()
    with pytest.raises(ValueError, match="TRAINING_TASK_REQUIRED"): admit(admitted)


@pytest.mark.parametrize("field,value", [("commands", []), ("runtime_code", []), ("gpu_lease_verified", False),
    ("deadline_at", "2001-01-01T00:00:00+00:00"), ("training_scope", "DEPLOYMENT")])
def test_packet_authority_cannot_be_inferred(admitted, field, value):
    admitted[7][field] = value; admitted[8]()
    with pytest.raises(ValueError): admit(admitted)


def test_prepared_pair_and_envelope_both_frozen(admitted):
    pair = Path(admitted[7]["inputs"]["prepared_pair"]["path"])
    pair.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="SHA_DRIFT"): admit(admitted)


def test_wall_deadline_never_restarts_on_reauthorization(admitted):
    first, second = admit(admitted), admit(admitted)
    assert first["admitted_authorization_expires_at"] == second["admitted_authorization_expires_at"]
    admitted[7]["execution_started_at"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    admitted[8]()
    with pytest.raises(ValueError, match="FIXED_TASK_WALL_DEADLINE"): admit(admitted)


def test_live_writer_required(admitted, monkeypatch):
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: False)
    with pytest.raises(ValueError, match="ANCESTRY"): admit(admitted)


@pytest.mark.parametrize("arguments", [["--help"], ["preflight", "--manifest", "MISSING.json", "--branch", "robotized"],
    ["train", "--task-packet", "NO_PACKET.json", "--manifest", "NO_MANIFEST.json", "--branch", "raw",
     "--config", "NO_CONFIG.json", "--output-dir", "NO_OUTPUT", "--pretrained", "NO_WEIGHTS.pt"]])
def test_rejected_cli_never_imports_torch(arguments):
    script = """
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname.endswith('.FlowMatchingModel'):
            raise AssertionError('unexpected model import')
sys.meta_path.insert(0,Deny())
from chaoyang.humanego_ict.research_training_cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run([sys.executable,"-B","-c",script,*arguments], capture_output=True,text=True)
    assert result.returncode == (0 if arguments == ["--help"] else 2), result.stderr
    assert "unexpected model import" not in result.stderr


def test_shell_syntax_and_rejected_manifest_before_cache_creation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/humanego_ict/train_research_pick_place.sh"
    assert subprocess.run(["bash","-n",str(script)], capture_output=True).returncode == 0
    output = tmp_path / "must_not_be_created"
    result = subprocess.run(["bash",str(script),"robotized","8","NO_PACKET.json","NO_MANIFEST.json",str(output)],
                            cwd=root,capture_output=True,text=True)
    assert result.returncode == 2 and not output.exists()


def test_existing_export_validates_research_recipe_pair_and_provenance(tmp_path):
    from chaoyang.humanego_ict.checkpoint_export import _provenance
    from chaoyang.humanego_ict.data import ACTION_LAYOUT
    cfg_ref = save(tmp_path / "config.json", config())
    normalizer = {"method": "IDENTITY_NO_FIT", "position_mean": [0,0,0], "position_std": [1,1,1]}
    pair_ref = save(tmp_path / "pair.json", {"schema": "pick-place-humanego-ict-revo2-derived-paired-v1",
        "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
        "physical_deployment_authorized": False, "fixture_only": True,
        "action_layout": ACTION_LAYOUT, "normalization": normalizer})
    research_ref = save(tmp_path / "research.json", {"schema": SCHEMA, "prepared_pair_ref": pair_ref,
        "training_scope": TRAINING_SCOPE, "control_ground_truth": False, "physical_deployment_authorized": False})
    result_ref = save(tmp_path / "result.json", {"optimizer_updates": 1, "control_ground_truth": False,
        "physical_deployment_authorized": False, "training_scope": TRAINING_SCOPE})
    provenance = {"task_id": "EXPLICIT_SYNTHETIC", "run_signature": "a"*64, "branch": "raw",
        "supervision_class": "DERIVED_RETARGET", "optimizer_updates": 1, "state_origin": "LIVE_MODEL_STATE",
        "dataset_ref": pair_ref, "research_manifest_ref": research_ref, "training_scope": TRAINING_SCOPE,
        "training_config_ref": cfg_ref, "training_result_ref": result_ref,
        "code_refs": [file_ref(Path(__file__).resolve().parents[2] / "src/chaoyang/humanego_ict/research_data.py")]}
    checked, pair = _provenance(provenance, cfg_ref, normalizer, "model", True)
    assert checked["research_manifest_ref"] == research_ref and checked["training_scope"] == TRAINING_SCOPE
    assert not pair["physical_deployment_authorized"]


def test_wrapper_binds_live_state_new_manifest_pair_and_scope(tmp_path, monkeypatch):
    from chaoyang.humanego_ict import research_training_cli as cli
    from chaoyang.humanego_ict import research_data, checkpoint_export
    root, output = tmp_path, tmp_path / "out"
    cfg_path, manifest_path, pair_path, weights = [tmp_path / n for n in ["config.json","research.json","pair.json","weights.json"]]
    save(cfg_path, config())
    pair = {"normalization": {"method": "IDENTITY_NO_FIT", "position_mean": [0,0,0], "position_std": [1,1,1]},
        "pose_clock": {"tcp_frame_ids": ["SYNTHETIC_LEFT","SYNTHETIC_RIGHT"]}}
    pair_ref = save(pair_path, pair)
    save(manifest_path, {"prepared_pair_ref": pair_ref, "training_scope": TRAINING_SCOPE})
    save(weights, {"fixture_only": True})
    packet = {"task_id": "synthetic-wrapper", "run_signature": "a"*64, "runtime_code": [],
        "allocated_gpu_ids": [0], "inputs": {"research_manifest": file_ref(manifest_path), "prepared_pair": pair_ref,
            "training_config": file_ref(cfg_path), "pretrained": file_ref(weights)},
        "admitted_authorization_expires_at": datetime.now(timezone.utc).timestamp()+60}
    monkeypatch.setattr(cli,"repository_root",lambda:root)
    monkeypatch.setattr(cli,"authorize_research_training",lambda *a,**kw:dict(packet))
    monkeypatch.setattr(cli,"storage_ready",lambda *a:None)
    monkeypatch.setattr(cli,"bind_cuda_visible_devices",lambda *a:"0")
    monkeypatch.delenv("ICT_RESEARCH_ADMISSION_ONLY",raising=False)
    monkeypatch.setenv("WORLD_SIZE","1")
    for name, path in zip(["TMPDIR","TORCH_HOME","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","XDG_CACHE_HOME"],runtime_cache_paths(output,root)):
        monkeypatch.setenv(name,str(path))
    monkeypatch.setattr(research_data,"validate_research_manifest",lambda *a:{
        "dataset_sha":file_ref(manifest_path)["sha256"], "prepared_pair_ref":pair_ref, "prepared_pair":pair})
    monkeypatch.setattr(research_data,"ResearchDerivedICTDataset",lambda *a:["synthetic-only"])
    state = {"not_an_actual_tensor": "mock-only"}
    class MockModel:
        def __init__(self,**kw): pass
        def to(self,device): return self
        def state_dict(self): return state
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
        device_count=lambda:1,set_device=lambda i:None),manual_seed=lambda i:None))
    monkeypatch.setitem(sys.modules,"chaoyang.humanego_ict.model",SimpleNamespace(Revo2ICTModel=MockModel))
    def mock_train(model,dataset,cfg,output_dir,signatures,**kw):
        assert signatures["prepared_pair"] == pair_ref and signatures["training_scope"] == TRAINING_SCOPE
        kw["initializer"](model)
        return {"optimizer_updates":1,"status":"SYNTHETIC_TEST_ONLY"}
    monkeypatch.setitem(sys.modules,"chaoyang.humanego_ict.training",SimpleNamespace(TrainingConfig=lambda **kw:kw,
        train_candidate=mock_train))
    monkeypatch.setitem(sys.modules,"chaoyang.human_ego.training.checkpoint_transfer",SimpleNamespace(
        load_compatible_pretrained=lambda *a,**kw:None))
    exported=[]
    def mock_export(live,**kw):
        p=kw["provenance"]
        assert live is state and p["dataset_ref"] == pair_ref
        assert p["research_manifest_ref"] == file_ref(manifest_path) and p["training_scope"] == TRAINING_SCOPE
        assert p["state_origin"] == "LIVE_MODEL_STATE"
        assert kw["runtime_authorizer"](p)["export_authorized"]
        invalid=dict(p); invalid["research_manifest_ref"]=pair_ref
        with pytest.raises(ValueError,match="EXPORT_PRODUCER_BINDING"): kw["runtime_authorizer"](invalid)
        exported.append(kw)
        return {"fixture_only":True}
    monkeypatch.setattr(checkpoint_export,"export_tensor_candidate",mock_export)
    args=["train","--task-packet",str(tmp_path/"packet.json"),"--manifest",str(manifest_path),
        "--config",str(cfg_path),"--pretrained",str(weights),"--branch","raw","--output-dir",str(output)]
    assert cli.main(args) == 0 and len(exported) == 1
    assert (output / "INFERENCE_EXPORT.json").is_file()
    result=json.loads((output / "RESULT_000001.json").read_text())
    assert result["training_scope"] == TRAINING_SCOPE and not result["physical_deployment_authorized"]
