"""Bounded CPU synthetic architecture tests; no actual checkpoint is loaded."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from chaoyang.humanego_ict.joint26_contract import (
    ACTION_DIM, ACTION_LAYOUT, ARM_INDICES, HAND_INDICES, HORIZON, interface_contract,
)
from chaoyang.humanego_ict.joint26_model import (
    Joint26ICTModel, build_joint26_model, warmstart_shared_layers,
)


def _reference(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def synthetic_export(directory):
    """Synthetic fixture metadata is not an actual trained 20k checkpoint."""
    from chaoyang.humanego_ict.joint26_inference import EXPORT_SCHEMA, INFERENCE_SCHEMA

    directory.mkdir()
    config = {"horizon": 30, "action_dim": 26, "ict_dim": 29, "max_ict": 8,
              "image_size": [240, 320], "architecture": {
                  "vision_embed_dim": 16, "num_heads": 2, "num_decoder_layers": 1,
                  "mlp_ratio": 1.0, "dropout": 0.0, "patch_size": 80}}
    policy = build_joint26_model(config)
    with torch.no_grad():
        policy.head_v.weight.zero_(); policy.head_v.bias.zero_()
    tensor = directory / "candidate_tensor.pt"
    torch.save(policy.state_dict(), tensor)
    normalization = {"method": "IDENTITY_NO_FIT"}
    contract = dict(interface_contract(), schema=INFERENCE_SCHEMA, output_shape=[30, 26],
                    model_config=config, normalization=normalization,
                    state_dict_key="model", pi05_loader_compatible=False,
                    hand_boundary_policy="REJECT_WITHOUT_SILENT_CLAMP")
    names = {"tensor": "candidate_tensor.pt", "model_config": "model_config.json",
             "normalization": "normalization.json", "inference_contract": "inference_contract.json"}
    for name, value in (("model_config.json", config), ("normalization.json", normalization),
                        ("inference_contract.json", contract)):
        (directory / name).write_text(json.dumps(value))
    manifest = {"schema": EXPORT_SCHEMA,
                "files": {key: _reference(directory / name) for key, name in names.items()},
                "source_state_dict_key": "model", "model_source": "MODEL_NOT_EMA",
                "optimizer_updates": 20000, "requested_optimizer_updates": 20000,
                "synthetic_fixture_not_training": True,
                "supervision_class": "DERIVED_IK_RESEARCH", "control_ground_truth": False,
                "physical_deployment_authorized": False}
    (directory / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest))
    return directory


def model():
    return Joint26ICTModel(vision_embed_dim=16, num_heads=2, num_decoder_layers=1,
                           mlp_ratio=1.0, dropout=0.0, patch_size=80)


def inputs():
    state_mask = torch.zeros((1, 26), dtype=torch.bool)
    state_mask[:, list(ARM_INDICES)] = True
    return {
        "x_rgb": torch.zeros((1, 3, 240, 320)),
        "x_ict": torch.zeros((1, 8, 29)),
        "ict_mask": torch.zeros((1, 8), dtype=torch.bool),
        "x_state26": torch.zeros((1, 26)), "state_mask": state_mask,
        "x_t": torch.zeros((1, 30, 26)), "t": torch.tensor([0.5]),
    }


def test_fixed_joint_action_interface_is_not_pi05_or_physical_truth():
    value = interface_contract()
    assert (HORIZON, ACTION_DIM) == (30, 26)
    assert ACTION_LAYOUT == "left_arm7,left_hand6,right_arm7,right_hand6"
    assert sorted(ARM_INDICES + HAND_INDICES) == list(range(26))
    assert value["arm_unit"] == "radians"
    assert not value["control_ground_truth"]
    assert not value["physical_deployment_authorized"]
    assert not value["pi05_weight_compatible"]


def test_output_shape_and_unknown_hand_do_not_drop_causal_arm_state():
    policy = model().eval()
    captured = []
    hook = policy.robot_state_proj.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    data = inputs()
    data["x_state26"][:, 0] = 0.25
    data["x_state26"][:, list(HAND_INDICES)] = float("nan")
    with torch.no_grad():
        result = policy(**data)
    hook.remove()
    assert set(result) == {"v_pred"}
    assert result["v_pred"].shape == (1, 30, 26)
    assert torch.isfinite(result["v_pred"]).all()
    encoded = captured[0]
    assert encoded.shape == (1, 2, 26)
    assert encoded[0, 0, 0].item() == 0.25
    assert encoded[0, 0, 13:20].eq(1).all()
    assert encoded[:, :, 7:13].eq(0).all()
    assert encoded[:, :, 20:26].eq(0).all()


def test_state_mask_does_not_masquerade_unknown_values_as_measurements():
    policy = model().eval()
    data = inputs()
    with torch.no_grad():
        first = policy(**data)["v_pred"]
        data["x_state26"][:, list(HAND_INDICES)] = 100000.0
        second = policy(**data)["v_pred"]
    assert torch.equal(first, second)


def test_missing_arm_dimension_masks_that_side_token():
    policy = model().eval()
    captured = []
    hook = policy.robot_state_proj.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    data = inputs()
    data["state_mask"][:, 0] = False
    with torch.no_grad():
        policy(**data)
    hook.remove()
    assert captured[0][0, 0].eq(0).all()
    assert captured[0][0, 1, 13:20].eq(1).all()


@pytest.mark.parametrize("field,value,error", [
    ("x_t", torch.zeros((1, 50, 30)), "SHAPE"),
    ("x_ict", torch.zeros((1, 8, 26)), "SHAPE"),
    ("state_mask", torch.ones((1, 26)), "BOOLEAN"),
    ("ict_mask", torch.ones((1, 8)), "BOOLEAN"),
    ("t", torch.tensor([1.1]), "TIME_RANGE"),
])
def test_incompatible_inputs_rejected(field, value, error):
    data = inputs(); data[field] = value
    with pytest.raises(ValueError, match=error):
        model()(**data)


def test_admitted_nonfinite_state_rejected_but_masked_ict_nan_safe():
    data = inputs(); data["x_state26"][:, 0] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE_ADMITTED_CURRENT_STATE"):
        model()(**data)
    data = inputs(); data["x_ict"].fill_(float("nan"))
    with torch.no_grad():
        assert torch.isfinite(model().eval()(**data)["v_pred"]).all()


@pytest.mark.parametrize("config,error", [
    ({"pred_horizon": 50}, "FIXED_H30"),
    ({"pred_horizon": 30.0}, "FIXED_H30"),
    ({"max_ict": 9}, "FIXED_ICT8"),
    ({"max_ict": 8.0}, "FIXED_ICT8"),
    ({"action_dim": 30}, "CONFIG_MISMATCH"),
    ({"action_dim": 26.0}, "CONFIG_MISMATCH"),
    ({"single_hand": True}, "UNSUPPORTED_ARCHITECTURE"),
    ({"img_size": [224, 224]}, "FIXED_RGB"),
    ({"dropout": float("nan")}, "FINITE_NUMBER"),
    ({"vision_embed_dim": 3}, "WIDTH"),
])
def test_factory_rejects_old_action_or_bad_architecture(config, error):
    with pytest.raises(ValueError, match=error):
        build_joint26_model(config)


def test_parent_wrist_loss_cannot_silently_route_joint26_dimensions():
    with pytest.raises(ValueError, match="NOT_WRIST_LOSS"):
        model().compute_loss({}, {})


def test_nested_training_model_configuration_matches_factory():
    policy = build_joint26_model({
        "horizon": 30, "action_dim": 26, "ict_dim": 29, "max_ict": 8,
        "image_size": [240, 320],
        "architecture": {"vision_embed_dim": 16, "num_heads": 2,
                         "num_decoder_layers": 1, "mlp_ratio": 1.0,
                         "dropout": 0.0, "patch_size": 80},
    })
    assert policy.action_dim == 26 and policy.pred_horizon == 30
    assert policy.robot_state_proj.in_features == 26


@pytest.mark.parametrize("config,error", [
    ({"horizon": 30, "pred_horizon": 30}, "AMBIGUOUS_HORIZON"),
    ({"architecture": [], "horizon": 30}, "ARCHITECTURE_MAPPING"),
    ({"architecture": {}, "vision_embed_dim": 16}, "AMBIGUOUS_NESTED"),
    ({"architecture": {"single_hand": True}}, "UNSUPPORTED_ARCHITECTURE"),
    ({"architecture": {}, "horizon": 50}, "FIXED_H30"),
])
def test_nested_and_flat_configuration_ambiguity_refused(config, error):
    with pytest.raises(ValueError, match=error):
        build_joint26_model(config)


def test_shared_warmstart_never_loads_action_head_or_wrong_horizon():
    policy = model()
    source = {name: torch.ones_like(value) for name, value in policy.state_dict().items()}
    source["pos_embed"] = torch.ones((50, 16))
    before = deepcopy(policy.state_dict())
    ref = {"path": "/project/synthetic_model_tensor.pt", "sha256": "a" * 64, "bytes": 1}
    report = warmstart_shared_layers(policy, source, source_ref=ref)
    after = policy.state_dict()
    assert torch.equal(after["rgb_embed.proj.weight"], source["rgb_embed.proj.weight"])
    for name in ("action_proj.weight", "action_proj.bias", "head_v.weight", "head_v.bias",
                 "pos_embed", "robot_state_proj.weight", "robot_state_pos_emb"):
        assert torch.equal(before[name], after[name])
        assert name in report["target_initialized_keys"]
    reasons = {row["name"]: row["reason"] for row in report["skipped"]}
    assert reasons["pos_embed"] == "SHAPE_MISMATCH_REINITIALIZED"
    assert reasons["head_v.weight"] == "ACTION_SEMANTICS_CHANGED_ALWAYS_REINITIALIZED"
    assert report["source_state_dict_key"] == "model"
    assert not report["optimizer_resumed"]


def test_checkpoint_or_ema_wrapper_not_accepted_as_model_state():
    with pytest.raises(ValueError, match="FLAT_MODEL_TENSORS_ONLY"):
        warmstart_shared_layers(model(), {"ema": {"weights": torch.ones(1)}},
                                source_ref={"path": "/project/synthetic.pt", "sha256": "a" * 64})


def test_no_compatible_layer_or_nonfinite_shared_tensor_refused():
    ref = {"path": "/project/synthetic.pt", "sha256": "a" * 64}
    with pytest.raises(ValueError, match="NO_COMPATIBLE_SHARED"):
        warmstart_shared_layers(model(), {"head_v.weight": torch.ones(26, 16)}, source_ref=ref)
    with pytest.raises(ValueError, match="NONFINITE_WARMSTART"):
        warmstart_shared_layers(model(), {"state_pos_emb": torch.full((1, 8, 16), float("nan"))}, source_ref=ref)


def test_tensor_only_export_loader_and_unclipped_offline_ode_smoke(tmp_path):
    from chaoyang.humanego_ict.joint26_inference import load_joint26_candidate, Joint26PredictionRejected

    directory = synthetic_export(tmp_path / "synthetic_export")
    manifest_sha = _reference(directory / "EXPORT_MANIFEST.json")["sha256"]
    candidate = load_joint26_candidate(directory, device="cpu", expected_manifest_sha256=manifest_sha)
    data = inputs(); data.pop("x_t"); data.pop("t")
    diagnostic = candidate.sample_diagnostic(**data, steps=2, seed=7)
    assert diagnostic["actions"].shape == (1, 30, 26)
    assert torch.isfinite(diagnostic["actions"]).all()
    assert diagnostic["status"] == "REJECTED_HAND_BOUNDARY"
    assert not diagnostic["accepted_for_offline_replay"]
    assert (diagnostic["actions"][:, :, list(HAND_INDICES)] < 0).any()
    assert not diagnostic["physical_deployment_authorized"]
    with pytest.raises(Joint26PredictionRejected) as error:
        candidate.predict(**data, steps=1, seed=7)
    assert torch.equal(error.value.diagnostic["actions"], diagnostic["actions"])


def test_export_sha_old_interface_and_manifest_trust_rejected(tmp_path):
    from chaoyang.humanego_ict.joint26_inference import load_joint26_candidate

    directory = synthetic_export(tmp_path / "synthetic_export")
    with pytest.raises(ValueError, match="TRUSTED_MANIFEST_SHA_DRIFT"):
        load_joint26_candidate(directory, expected_manifest_sha256="a" * 64)
    original = (directory / "normalization.json").read_bytes()
    (directory / "normalization.json").write_text("{}")
    with pytest.raises(ValueError, match="EXPORT_SHA_DRIFT"):
        load_joint26_candidate(directory)
    (directory / "normalization.json").write_bytes(original)
    contract = json.loads((directory / "inference_contract.json").read_text())
    contract["output_shape"] = [50, 30]
    (directory / "inference_contract.json").write_text(json.dumps(contract))
    manifest = json.loads((directory / "EXPORT_MANIFEST.json").read_text())
    manifest["files"]["inference_contract"] = _reference(directory / "inference_contract.json")
    (directory / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="H30_26_REQUIRED_NOT_H50_30"):
        load_joint26_candidate(directory)


def test_unknown_current_hand_requires_zero_placeholder_and_valid_hand_range(tmp_path):
    from chaoyang.humanego_ict.joint26_inference import load_joint26_candidate

    candidate = load_joint26_candidate(synthetic_export(tmp_path / "synthetic_export"))
    data = inputs(); data.pop("x_t"); data.pop("t")
    data["x_state26"][:, 7] = 0.2
    with pytest.raises(ValueError, match="UNKNOWN_CURRENT_HAND_MUST_BE_ZERO"):
        candidate.sample_diagnostic(**data, steps=1)
    data["state_mask"][:, 7] = True
    data["x_state26"][:, 7] = 1.1
    with pytest.raises(ValueError, match="KNOWN_CURRENT_HAND_OUTSIDE"):
        candidate.sample_diagnostic(**data, steps=1)


def test_joint_numerical_gate_requires_frozen_limits_and_real_target_time_mapping():
    from chaoyang.humanego_ict.joint26_inference import Joint26Candidate

    limits = {"joint_lower_rad": [-1.0] * 14, "joint_upper_rad": [1.0] * 14,
              "joint_speed_max_rad_s": [2.0] * 14, "joint_step_max_rad": [0.2] * 14}
    candidate = Joint26Candidate(None, {"numerical_limits": limits}, {}, {}, {},
                                 "a" * 64, False, torch.device("cpu"))
    actions, state = torch.zeros((1, 30, 26)), torch.zeros((1, 26))
    actions[:, :, list(ARM_INDICES)] = 0.1
    target = torch.arange(1, 31, dtype=torch.int64).reshape(1, 30) * 100000000
    anchor = torch.zeros(1, dtype=torch.int64)
    assert candidate._joint_checks(actions, state, target, anchor) == "PASSED_ASSUMED_NUMERICAL_LIMITS"
    assert candidate._joint_checks(actions, state, None, None) == "NOT_EVALUATED_TARGET_QPC_MAPPING_ABSENT"
    bad = actions.clone(); bad[:, :, 0] = 2
    assert candidate._joint_checks(bad, state, target, anchor) == "REJECTED_ASSUMED_JOINT_BOUNDS"
    bad = actions.clone(); bad[:, :, 0] = 0.3
    assert candidate._joint_checks(bad, state, target, anchor) == "REJECTED_ASSUMED_JOINT_STEP"
    limits["joint_speed_max_rad_s"][0] = 0.01
    assert candidate._joint_checks(actions, state, target, anchor) == "REJECTED_ASSUMED_JOINT_SPEED"
    target[:, 1] = target[:, 0]
    with pytest.raises(ValueError, match="MONOTONIC"):
        candidate._joint_checks(actions, state, target, anchor)
    candidate.contract = {}
    assert candidate._joint_checks(actions, state, None, None) == "NOT_EVALUATED_JOINT_LIMITS_ABSENT"


def _pure_export_interface_fixture():
    """Metadata only: does not create a model, file or optimizer update."""
    from chaoyang.humanego_ict.joint26_inference import EXPORT_SCHEMA, INFERENCE_SCHEMA

    model_config = {"horizon": 30, "action_dim": 26, "ict_dim": 29,
                    "max_ict": 8, "image_size": [240, 320], "architecture": {}}
    normalization = {"method": "IDENTITY_NO_FIT"}
    contract = dict(interface_contract(), schema=INFERENCE_SCHEMA,
                    output_shape=[30, 26], model_config=model_config,
                    normalization=normalization, state_dict_key="model",
                    pi05_loader_compatible=False,
                    hand_boundary_policy="REJECT_WITHOUT_SILENT_CLAMP")
    manifest = {"schema": EXPORT_SCHEMA, "source_state_dict_key": "model",
                "model_source": "MODEL_NOT_EMA", "optimizer_updates": 20000,
                "requested_optimizer_updates": 20000,
                "supervision_class": "DERIVED_IK_RESEARCH",
                "control_ground_truth": False, "physical_deployment_authorized": False}
    return manifest, contract, model_config, normalization


@pytest.mark.parametrize("updates", [1, 37, 20000, 30000, 100000])
def test_flexible_export_update_budget_accepts_only_completed_positive_target(updates):
    from chaoyang.humanego_ict.joint26_inference import _check_interface

    manifest, contract, model_config, normalization = _pure_export_interface_fixture()
    manifest.update(optimizer_updates=updates, requested_optimizer_updates=updates)
    _check_interface(manifest, contract, model_config, normalization)


@pytest.mark.parametrize("requested,actual,error", [
    (None, 20000, "POSITIVE_REQUESTED"),
    (0, 0, "POSITIVE_REQUESTED"),
    (-1, -1, "POSITIVE_REQUESTED"),
    (True, 1, "POSITIVE_REQUESTED"),
    (30000.0, 30000, "POSITIVE_REQUESTED"),
    (30000, None, "COMPLETED_REQUESTED"),
    (30000, 0, "COMPLETED_REQUESTED"),
    (30000, -1, "COMPLETED_REQUESTED"),
    (1, True, "COMPLETED_REQUESTED"),
    (30000, 30000.0, "COMPLETED_REQUESTED"),
    (30000, 20000, "COMPLETED_REQUESTED"),
    (30000, 30001, "COMPLETED_REQUESTED"),
])
def test_flexible_export_update_budget_rejects_missing_invalid_or_partial(requested, actual, error):
    from chaoyang.humanego_ict.joint26_inference import _check_interface

    manifest, contract, model_config, normalization = _pure_export_interface_fixture()
    manifest.update(requested_optimizer_updates=requested, optimizer_updates=actual)
    if requested is None:
        manifest.pop("requested_optimizer_updates")
    with pytest.raises(ValueError, match=error):
        _check_interface(manifest, contract, model_config, normalization)


def test_flexible_export_explicit_incomplete_flag_rejected():
    from chaoyang.humanego_ict.joint26_inference import _check_interface

    manifest, contract, model_config, normalization = _pure_export_interface_fixture()
    manifest["completed_update_budget"] = False
    with pytest.raises(ValueError, match="COMPLETED_REQUESTED"):
        _check_interface(manifest, contract, model_config, normalization)


@pytest.mark.parametrize("field,value,error", [
    ("output_shape", [50, 30], "H30_26_REQUIRED"),
    ("physical_deployment_authorized", True, "INTERFACE_MISMATCH"),
    ("control_ground_truth", True, "INTERFACE_MISMATCH"),
    ("hand_boundary_policy", "CLAMP", "REJECT_POLICY"),
    ("pi05_loader_compatible", True, "REJECT_POLICY"),
])
def test_flexible_export_budget_does_not_relax_other_interface_gates(field, value, error):
    from chaoyang.humanego_ict.joint26_inference import _check_interface

    manifest, contract, model_config, normalization = _pure_export_interface_fixture()
    manifest.update(optimizer_updates=30000, requested_optimizer_updates=30000)
    contract[field] = value
    with pytest.raises(ValueError, match=error):
        _check_interface(manifest, contract, model_config, normalization)
