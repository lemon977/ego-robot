"""Synthetic CPU tensors only; no model construction, training, or SDK calls."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from chaoyang.humanego_ict.checkpoint_export import (
    CheckpointExportError, PROJECT_ROOT, _authorized, export_tensor_candidate,
)
from chaoyang.humanego_ict.data import ACTION_LAYOUT, reference_bytes
from chaoyang.humanego_ict.inference import (
    TENSOR_CHECKPOINT_SCHEMA, _validate_contract, build_hand_ict, load_candidate_policy,
)

NORM = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}
FRAME = {"position_unit": "meters", "tcp_frame_ids": ["left_anchor", "right_anchor"],
         "pose_semantics": "DERIVED_RETARGET_ANCHOR", "action_reference_frame": "REFERENCE_CAMERA"}


def write_ref(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    data = path.read_bytes()
    return {"path": str(path.absolute()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def state():
    import torch
    width = 8
    shapes = {"action_proj.weight": (width, 30), "action_proj.bias": (width,),
              "head_v.weight": (30, width), "head_v.bias": (30,),
              "state_proj.weight": (width, 29), "state_proj.bias": (width,),
              "pos_embed": (50, width), "state_pos_emb": (1, 8, width)}
    return {name: torch.full(shape, .125) for name, shape in shapes.items()}


@pytest.fixture
def export_inputs(tmp_path):
    # The command selects a unique project-local --basetemp; never default /tmp.
    assert tmp_path.is_relative_to(PROJECT_ROOT)
    config = {"physical_deployment_authorized": False,
              "model": {"pred_horizon": 50, "max_ict": 8, "img_size": [240, 320], "vision_embed_dim": 8},
              "training": {"max_updates": 20}}
    config_ref = write_ref(tmp_path / "training_config.json", config)
    norm_ref = write_ref(tmp_path / "normalization.json", NORM)
    dataset = {"schema": "pick-place-humanego-ict-revo2-derived-paired-v1",
               "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
               "physical_deployment_authorized": False, "action_layout": ACTION_LAYOUT,
               "normalization": NORM, "pose_clock": {"tcp_frame_ids": FRAME["tcp_frame_ids"]},
               "fixture_only": True}
    dataset_ref = write_ref(tmp_path / "dataset.json", dataset)
    result_ref = write_ref(tmp_path / "training_result.json", {"optimizer_updates": 2,
                           "control_ground_truth": False, "physical_deployment_authorized": False})
    code_ref = write_ref(tmp_path / "frozen_fixture_code.json", {"fixture_only": True})
    provenance = {"task_id": "cpu-fixture", "run_signature": "a" * 64, "branch": "raw",
                  "supervision_class": "DERIVED_RETARGET", "state_origin": "LIVE_MODEL_STATE",
                  "optimizer_updates": 2, "dataset_ref": dataset_ref,
                  "training_config_ref": config_ref, "training_result_ref": result_ref,
                  "code_refs": [code_ref]}
    output = tmp_path / "export"

    def authorized(source):
        return {"status": "PASSED", "export_authorized": True,
                "task_id": source["task_id"], "run_signature": source["run_signature"],
                "execution_mode": "SYNTHETIC_FIXTURE_ONLY", "output_dir": str(output)}

    return {"model_config_ref": config_ref, "normalization_ref": norm_ref,
            "provenance": provenance, "output_dir": output, "runtime_authorizer": authorized,
            "target_frame_contract": FRAME, "fixture_only": True}


def test_live_cpu_state_exports_safe_payload_and_contract(export_inputs):
    import torch
    tensors = state()
    exported = export_tensor_candidate(tensors, **export_inputs)
    payload = torch.load(io.BytesIO(reference_bytes(exported["checkpoint_ref"])),
                         map_location="cpu", weights_only=True)
    assert payload["schema"] == TENSOR_CHECKPOINT_SCHEMA
    assert payload["fixture_only"] is True
    assert "optimizer" not in payload and "rank_rng" not in payload
    for name, expected in tensors.items():
        assert torch.equal(payload["model"][name], expected)
    contract = json.loads(reference_bytes(exported["inference_contract_ref"]))
    _validate_contract(contract)
    assert contract["target_frame_contract"] == FRAME
    assert contract["control_ground_truth"] is False
    assert contract["physical_deployment_authorized"] is False
    assert contract["training_provenance"]["optimizer_updates"] == 2
    model_config = json.loads(reference_bytes(exported["model_config_ref"]))
    assert model_config["architecture"] == {"vision_embed_dim": 8}
    assert model_config["horizon"] == 50


def test_export_is_same_bytes_idempotent_and_never_overwrites(export_inputs):
    import torch
    original = state()
    first = export_tensor_candidate(original, **export_inputs)
    second = export_tensor_candidate(original, **export_inputs)
    assert first == second
    original["head_v.bias"] = torch.zeros(30)
    with pytest.raises(CheckpointExportError, match="immutable"):
        export_tensor_candidate(original, **export_inputs)
    assert reference_bytes(first["checkpoint_ref"])


def test_explicit_live_ema_not_guessed_from_model(export_inputs):
    with pytest.raises(CheckpointExportError, match="live state origin"):
        export_tensor_candidate(state(), state_dict_key="ema", **export_inputs)
    export_inputs["provenance"]["state_origin"] = "LIVE_EMA_STATE"
    exported = export_tensor_candidate(state(), state_dict_key="ema", **export_inputs)
    assert exported["state_dict_key"] == "ema"


@pytest.mark.parametrize("head,shape", [("action_proj.weight", (8, 19)),
                                      ("head_v.weight", (26, 8)),
                                      ("state_proj.weight", (8, 20)),
                                      ("pos_embed", (30, 8))])
def test_legacy_or_pi05_shapes_rejected(export_inputs, head, shape):
    import torch
    tensors = state()
    tensors[head] = torch.zeros(shape)
    with pytest.raises(CheckpointExportError, match="H50/30D/ICT29"):
        export_tensor_candidate(tensors, **export_inputs)
    assert not export_inputs["output_dir"].exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_tensor_rejected(export_inputs, bad):
    tensors = state()
    tensors["head_v.bias"][0] = bad
    with pytest.raises(CheckpointExportError, match="nonfinite"):
        export_tensor_candidate(tensors, **export_inputs)
    assert not export_inputs["output_dir"].exists()


def test_untrusted_checkpoint_path_is_not_a_state_input(export_inputs):
    with pytest.raises(CheckpointExportError, match="tensor state mapping"):
        export_tensor_candidate(Path("old_untrusted.pt"), **export_inputs)
    tensors = state()
    tensors["metadata"] = {"rank_rng": "not a tensor"}
    with pytest.raises(CheckpointExportError, match="live tensor"):
        export_tensor_candidate(tensors, **export_inputs)


def test_false_provenance_or_coded_sha_drift_rejected(export_inputs):
    export_inputs["provenance"]["optimizer_updates"] = 3
    with pytest.raises(CheckpointExportError, match="update/result"):
        export_tensor_candidate(state(), **export_inputs)
    export_inputs["provenance"]["optimizer_updates"] = 2
    reference = export_inputs["provenance"]["code_refs"][0]
    Path(reference["path"]).write_text("changed source")
    with pytest.raises(ValueError, match="SHA/size drift"):
        export_tensor_candidate(state(), **export_inputs)


def test_no_physical_claim_or_real_fixture_relabel(export_inputs):
    ref = export_inputs["provenance"]["dataset_ref"]
    dataset = json.loads(reference_bytes(ref))
    dataset["control_ground_truth"] = True
    export_inputs["provenance"]["dataset_ref"] = write_ref(Path(ref["path"]), dataset)
    with pytest.raises(CheckpointExportError, match="derived non-deployment"):
        export_tensor_candidate(state(), **export_inputs)


def test_unauthorized_export_never_imports_torch_or_reads_pickle(export_inputs):
    export_inputs["runtime_authorizer"] = lambda source: {}
    with pytest.raises(CheckpointExportError, match="authorization rejected"):
        export_tensor_candidate({}, **export_inputs)
    script = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith(('torch.', 'bc_stark_sdk')):
            raise AssertionError('model/SDK imported before export authority')
sys.meta_path.insert(0, BlockTorch())
from chaoyang.humanego_ict.checkpoint_export import export_tensor_candidate
try:
    export_tensor_candidate({}, model_config_ref={}, normalization_ref={},
        provenance={'task_id':'t','run_signature':'a'*64}, output_dir=OUTPUT,
        runtime_authorizer=lambda source:{}, target_frame_contract={})
except ValueError:
    pass
else:
    raise AssertionError('export unexpectedly accepted')
assert 'torch' not in sys.modules
""".replace("OUTPUT", repr(str(export_inputs["output_dir"])))
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_expiry_while_cloning_preserves_no_candidate(export_inputs):
    count = [0]
    original = export_inputs["runtime_authorizer"]

    def verifier(source):
        count[0] += 1
        return original(source) if count[0] < 6 else {}

    export_inputs["runtime_authorizer"] = verifier
    with pytest.raises(CheckpointExportError, match="authorization rejected"):
        export_tensor_candidate(state(), **export_inputs)
    assert not export_inputs["output_dir"].exists()


def test_concurrent_live_state_mutation_rejected(export_inputs):
    tensors = state()
    count, original = [0], export_inputs["runtime_authorizer"]

    def verifier(source):
        count[0] += 1
        if count[0] == 5:
            tensors["head_v.bias"].fill_(.75)
        return original(source)

    export_inputs["runtime_authorizer"] = verifier
    with pytest.raises(CheckpointExportError, match="changed during snapshot"):
        export_tensor_candidate(tensors, **export_inputs)
    assert not export_inputs["output_dir"].exists()


def test_each_gpu_snapshot_authority_needs_fresh_lease_attestation(export_inputs):
    with pytest.raises(CheckpointExportError, match="current lease"):
        _authorized(export_inputs["runtime_authorizer"], export_inputs["provenance"],
                    export_inputs["output_dir"], True, require_gpu_lease=True)


def test_numpy_rng_or_arbitrary_objects_never_enter_payload(export_inputs):
    export_inputs["provenance"]["old_rng"] = np.array([1, 2, 3])
    with pytest.raises(CheckpointExportError, match="JSON primitives"):
        export_tensor_candidate(state(), **export_inputs)
    assert not export_inputs["output_dir"].exists()


def test_exported_contract_loads_through_inference_adapter_with_fixture_model(export_inputs):
    import torch
    exported = export_tensor_candidate(state(), **export_inputs)

    class FixtureModel:
        def load_state_dict(self, tensors, strict):
            assert strict and tensors["head_v.weight"].shape == (30, 8)

        def eval(self):
            return self

        def __call__(self, **inputs):
            return {"v_pred": np.zeros((1, 50, 30), dtype=np.float32)}

    def authorized(reference, contract):
        return {"status": "PASSED", "model_execution_allowed": True,
                "execution_mode": "SYNTHETIC_FIXTURE_ONLY", "contract_sha256": reference["sha256"]}

    policy = load_candidate_policy(exported["inference_contract_ref"],
        authorization_verifier=authorized, model_factory=lambda config: FixtureModel(),
        checkpoint_loader=lambda encoded, device: torch.load(io.BytesIO(encoded),
                                                            map_location="cpu", weights_only=True))
    current = build_hand_ict(np.broadcast_to(np.eye(4), (2, 4, 4)), np.array([True, True]),
        np.eye(4), normalization=NORM, source_qpc_ns=np.array([99, 100]), anchor_qpc_ns=100)
    result = policy.sample(x_rgb=np.zeros((1, 3, 240, 320)), x_ict=current["x_ict"][None],
        ict_mask=current["ict_mask"][None], initial_noise=np.ones((1, 50, 30)), steps=1)
    np.testing.assert_array_equal(result, np.ones((1, 50, 30)))


def test_frame_contract_cannot_change_tcp_or_units(export_inputs):
    export_inputs["target_frame_contract"] = {**FRAME, "tcp_frame_ids": ["different_l", "different_r"]}
    with pytest.raises(CheckpointExportError, match="differ from training dataset"):
        export_tensor_candidate(state(), **export_inputs)
    export_inputs["target_frame_contract"] = {**FRAME, "position_unit": "centimeters"}
    with pytest.raises(CheckpointExportError, match="reference-camera"):
        export_tensor_candidate(state(), **export_inputs)


def test_export_symlinks_or_external_outputs_forbidden(export_inputs):
    target = export_inputs["output_dir"].parent / "actual"
    target.mkdir()
    link = export_inputs["output_dir"].parent / "alias"
    link.symlink_to(target, target_is_directory=True)
    export_inputs["output_dir"] = link / "export"
    with pytest.raises(CheckpointExportError, match="symbolic-link"):
        export_tensor_candidate(state(), **export_inputs)
    export_inputs["output_dir"] = Path("/tmp/outside_export")
    with pytest.raises(CheckpointExportError, match="stay inside"):
        export_tensor_candidate(state(), **export_inputs)
