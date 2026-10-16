"""CPU numerical/callback tests; no real weights, model, GPU, SDK, or network."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from chaoyang.humanego_ict.data import ACTION_LAYOUT, _pose9
from chaoyang.humanego_ict.inference import (
    HAND_CHANNELS, HAND_UNIT, INFERENCE_SCHEMA, ROT6_LAYOUT,
    TENSOR_CHECKPOINT_SCHEMA, InferenceError, build_hand_ict, decode_actions,
    load_candidate_policy, rot6_to_matrix, sample_actions,
)


NORM = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}


def state(valid=(True, True)):
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[0, :3, 3] = [1, 2, 3]
    poses[1, :3, 3] = [2, 3, 4]
    camera = np.eye(4)
    camera[:3, 3] = [.1, .2, .3]
    return poses, camera, build_hand_ict(
        poses, np.array(valid), camera, normalization=NORM,
        source_qpc_ns=np.array([99, 100]), anchor_qpc_ns=100,
        frame_declaration={"camera_frame_id": "synthetic_camera", "fixture_only": True})


def conditioning():
    packet = state()[2]
    return {"x_rgb": np.zeros((1, 3, 240, 320), dtype=np.float32),
            "x_ict": packet["x_ict"][None], "ict_mask": packet["ict_mask"][None],
            "initial_noise": np.zeros((1, 50, 30), dtype=np.float32)}


def action():
    result = np.zeros((50, 30), dtype=np.float32)
    result[:, :6] = [1, 2, 3, 4, 5, 6]
    result[:, 6:18] = np.tile(np.eye(3)[:, :2].reshape(6), 2)
    result[:, 18:] = .5
    return result


def test_ict_matches_frozen_training_pose_semantics():
    poses, camera, packet = state()
    mean, std = np.zeros(3), np.ones(3)
    for hand in range(2):
        expected = np.concatenate(([hand + 1], _pose9(np.linalg.inv(camera) @ poses[hand], mean, std),
                                   _pose9(np.linalg.inv(poses[hand]) @ poses, mean, std).reshape(18), [-1]))
        np.testing.assert_allclose(packet["x_ict"][hand], expected, atol=1e-6)
    assert packet["ict_mask"].tolist() == [True, True, False, False, False, False, False, False]
    assert np.all(packet["x_ict"][2:] == 0)
    assert packet["x_ict"][0, -1] == packet["x_ict"][1, -1] == -1
    assert packet["conditioning_source_qpc_ns"].tolist() == [99, 100, -1, -1, -1, -1, -1, -1]
    assert packet["control_ground_truth"] is False


def test_missing_current_hand_is_masked_and_not_fabricated():
    poses, camera, _ = state()
    poses[1] = np.nan
    packet = build_hand_ict(poses, np.array([True, False]), camera, normalization=NORM,
                            source_qpc_ns=np.array([99, -1]), anchor_qpc_ns=100)
    assert not packet["ict_mask"][1]
    assert np.all(packet["x_ict"][1] == 0)
    assert np.all(packet["x_ict"][0, 19:28] == 0)  # right-hand relative pose UNKNOWN
    with pytest.raises(InferenceError, match="at least one"):
        build_hand_ict(poses, np.array([False, False]), camera, normalization=NORM,
                       source_qpc_ns=np.array([-1, -1]), anchor_qpc_ns=100)


@pytest.mark.parametrize("timestamps", [np.array([99, 101]), np.array([99., 100.]), np.array([-1, 100])])
def test_future_or_noninteger_conditioning_rejected(timestamps):
    poses, camera, _ = state()
    with pytest.raises(InferenceError, match="causal"):
        build_hand_ict(poses, np.array([True, True]), camera, normalization=NORM,
                       source_qpc_ns=timestamps, anchor_qpc_ns=100)


@pytest.mark.parametrize("method,calls", [("euler", 4), ("heun", 8)])
def test_finite_flow_integrator_callback(method, calls):
    seen = []
    target = action()[None]

    def callback(**inputs):
        seen.append(float(inputs["t"][0, 0]))
        assert inputs["x_t"].shape == (1, 50, 30)
        assert inputs["ict_mask"].dtype == bool
        return {"v_pred": target}

    result = sample_actions(callback, **conditioning(), steps=4, method=method)
    np.testing.assert_allclose(result, target)
    assert len(seen) == calls
    assert min(seen) == 0 and max(seen) <= 1


def test_sampler_rejects_wrong_velocity_and_fabricated_object():
    with pytest.raises(InferenceError, match="velocity"):
        sample_actions(lambda **kw: np.zeros((1, 50, 26)), **conditioning(), steps=1)
    inputs = conditioning()
    inputs["ict_mask"][0, 2] = True
    with pytest.raises(InferenceError, match="hand-only"):
        sample_actions(lambda **kw: np.zeros((1, 50, 30)), **inputs, steps=1)
    inputs = conditioning()
    inputs["x_ict"][0, 0, -1] = 0
    with pytest.raises(InferenceError, match="UNKNOWN grasp"):
        sample_actions(lambda **kw: np.zeros((1, 50, 30)), **inputs, steps=1)


def test_row_interleaved_rot6_round_trip_and_degenerate_failure():
    theta = .73
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    np.testing.assert_allclose(rot6_to_matrix(rotation[:, :2].reshape(6)), rotation, atol=1e-12)
    with pytest.raises(InferenceError, match="degenerate"):
        rot6_to_matrix(np.zeros(6))
    with pytest.raises(InferenceError, match="degenerate"):
        rot6_to_matrix(np.array([[1, 1], [0, 0], [0, 0]]).reshape(6))


def test_decode_preserves_camera_frame_and_reports_explicit_clipping():
    actions = action()
    camera = np.eye(4)
    camera[:3, 3] = [10, 20, 30]
    result = decode_actions(actions, camera, normalization=NORM)
    np.testing.assert_allclose(result["wrist_pose_reference_camera"][0, :, :3, 3], [[1, 2, 3], [4, 5, 6]])
    assert result["frame_id"] == "REFERENCE_CAMERA"
    assert result["hand_channel_order"] == list(HAND_CHANNELS)
    assert result["physical_deployment_authorized"] is False
    actions[0, 18] = 1.2
    with pytest.raises(InferenceError, match="outside"):
        decode_actions(actions, camera, normalization=NORM)
    clipped = decode_actions(actions, camera, normalization=NORM, command_policy="CLIP_DIAGNOSTIC")
    assert clipped["hand_command"][0, 0, 0] == 1
    assert clipped["unclipped_hand_command"][0, 0, 0] > 1
    assert clipped["out_of_range_mask"][0, 0, 0]


def test_bad_geometry_normalization_and_hand_units_rejected():
    poses, camera, _ = state()
    reflected = poses.copy()
    reflected[0, 0, 0] = -1
    with pytest.raises(InferenceError, match="proper SE3"):
        build_hand_ict(reflected, np.array([True, True]), camera, normalization=NORM,
                       source_qpc_ns=np.array([99, 100]), anchor_qpc_ns=100)
    with pytest.raises(InferenceError, match="identity"):
        decode_actions(action(), camera, normalization={**NORM, "position_std": [2, 2, 2]})
    with pytest.raises(InferenceError, match="unit/order"):
        decode_actions(action(), camera, normalization=NORM, hand_command_unit="SDK_RAW")


def _write_ref(path, value):
    path.write_text(json.dumps(value))
    encoded = path.read_bytes()
    return {"path": str(path.absolute()), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def fixtures(tmp_path):
    config_ref = _write_ref(tmp_path / "config.json", {"architecture": {}, "horizon": 50,
                            "action_dim": 30, "ict_dim": 29, "image_size": [240, 320]})
    norm_ref = _write_ref(tmp_path / "norm.json", NORM)
    checkpoint_ref = _write_ref(tmp_path / "checkpoint.json", {"fixture_only": True})
    contract = {"schema": INFERENCE_SCHEMA, "horizon": 50, "action_dim": 30,
                "ict_dim": 29, "image_size": [240, 320], "action_layout": ACTION_LAYOUT,
                "rotation_layout": ROT6_LAYOUT, "hand_command_unit": HAND_UNIT,
                "hand_channel_order": list(HAND_CHANNELS), "control_ground_truth": False,
                "physical_deployment_authorized": False, "state_dict_key": "ema",
                "model_config_ref": config_ref, "normalization_ref": norm_ref,
                "checkpoint_ref": checkpoint_ref}
    contract_ref = _write_ref(tmp_path / "contract.json", contract)
    payload = {"schema": TENSOR_CHECKPOINT_SCHEMA,
               "binding": {"config_sha256": config_ref["sha256"],
                           "normalization_sha256": norm_ref["sha256"], "action_layout": ACTION_LAYOUT},
               "ema": {"fixture": np.array([1])}, "control_ground_truth": False,
               "physical_deployment_authorized": False, "fixture_only": True}
    return contract_ref, payload


class FakeModel:
    """Injected fixture, not Revo2ICTModel/FlowMatchingModel."""
    def load_state_dict(self, state, strict):
        assert strict and set(state) == {"fixture"}

    def eval(self):
        return self

    def __call__(self, **inputs):
        return {"v_pred": action()[None]}


def authorized(reference, contract):
    return {"status": "PASSED", "model_execution_allowed": True,
            "contract_sha256": reference["sha256"], "execution_mode": "SYNTHETIC_FIXTURE_ONLY"}


def test_injected_candidate_loader_and_sampler(tmp_path):
    reference, payload = fixtures(tmp_path)
    policy = load_candidate_policy(reference, authorization_verifier=authorized,
                                   model_factory=lambda config: FakeModel(),
                                   checkpoint_loader=lambda encoded, device: payload)
    np.testing.assert_allclose(policy.sample(**conditioning(), steps=2), action()[None])
    assert policy.normalization == NORM
    assert policy.checkpoint_sha256 == json.loads((tmp_path / "contract.json").read_text())["checkpoint_ref"]["sha256"]


def test_runtime_gate_precedes_loader_factory_and_real_import(tmp_path):
    reference, payload = fixtures(tmp_path)
    calls = []
    with pytest.raises(InferenceError, match="authorization rejected"):
        load_candidate_policy(reference, authorization_verifier=lambda *args: {},
                              model_factory=lambda config: calls.append("factory"),
                              checkpoint_loader=lambda *args: calls.append("loader"))
    assert not calls
    with pytest.raises(InferenceError, match="fixture callbacks"):
        load_candidate_policy(reference, authorization_verifier=authorized)
    with pytest.raises(InferenceError, match="GPU lease"):
        load_candidate_policy(reference, authorization_verifier=lambda ref, con: {
            **authorized(ref, con), "execution_mode": "REAL_MODEL_OFFLINE", "device": "cuda:0"})


@pytest.mark.parametrize("schema", ["humanego-ict-revo2-resume-v1", "pi05", "legacy-single-hand"])
def test_resume_or_legacy_weights_not_candidate(tmp_path, schema):
    reference, payload = fixtures(tmp_path)
    payload["schema"] = schema
    with pytest.raises(InferenceError, match="tensor-only"):
        load_candidate_policy(reference, authorization_verifier=authorized,
                              model_factory=lambda config: FakeModel(),
                              checkpoint_loader=lambda *args: payload)


def test_checkpoint_sha_drift_rejected_before_callbacks(tmp_path):
    reference, payload = fixtures(tmp_path)
    (tmp_path / "checkpoint.json").write_text("changed immutable checkpoint")
    calls = []
    with pytest.raises(ValueError, match="SHA/size drift"):
        load_candidate_policy(reference, authorization_verifier=authorized,
                              model_factory=lambda config: calls.append("factory"),
                              checkpoint_loader=lambda *args: calls.append("loader"))
    assert not calls


def test_false_deployment_claim_or_head_override_rejected(tmp_path):
    reference, payload = fixtures(tmp_path)
    path = tmp_path / "contract.json"
    contract = json.loads(path.read_text())
    contract["physical_deployment_authorized"] = True
    reference = _write_ref(path, contract)
    with pytest.raises(InferenceError, match="physical_deployment_authorized"):
        load_candidate_policy(reference, authorization_verifier=authorized,
                              model_factory=lambda config: FakeModel(), checkpoint_loader=lambda *args: payload)
    contract["physical_deployment_authorized"] = False
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["architecture"] = {"single_hand": True}
    contract["model_config_ref"] = _write_ref(config_path, config)
    reference = _write_ref(path, contract)
    with pytest.raises(InferenceError, match="architecture"):
        load_candidate_policy(reference, authorization_verifier=authorized,
                              model_factory=lambda config: FakeModel(), checkpoint_loader=lambda *args: payload)


def test_authorization_expiry_rechecked_before_next_model_call(tmp_path):
    reference, payload = fixtures(tmp_path)
    count = [0]
    expired = [False]

    def verifier(ref, contract):
        count[0] += 1
        return authorized(ref, contract) if not expired[0] else {}

    policy = load_candidate_policy(reference, authorization_verifier=verifier,
                                   model_factory=lambda config: FakeModel(),
                                   checkpoint_loader=lambda *args: payload)
    count_before_prediction = count[0]
    expired[0] = True
    with pytest.raises(InferenceError, match="authorization rejected"):
        policy.sample(**conditioning(), steps=2)
    assert count[0] == count_before_prediction + 1


@pytest.mark.parametrize("expire_at,expected_calls", [(2, []), (3, ["loader"]),
                                                   (4, ["loader", "factory"])])
def test_lease_expiry_between_load_boundaries_is_fail_closed(tmp_path, expire_at, expected_calls):
    reference, payload = fixtures(tmp_path)
    count, calls = [0], []

    def verifier(ref, contract):
        count[0] += 1
        return authorized(ref, contract) if count[0] < expire_at else {}

    def loader(encoded, device):
        calls.append("loader")
        return payload

    def factory(config):
        calls.append("factory")
        return FakeModel()

    with pytest.raises(InferenceError, match="authorization rejected"):
        load_candidate_policy(reference, authorization_verifier=verifier,
                              model_factory=factory, checkpoint_loader=loader)
    assert calls == expected_calls


def test_device_mode_change_during_load_is_rejected(tmp_path):
    reference, payload = fixtures(tmp_path)
    count = [0]

    def verifier(ref, contract):
        count[0] += 1
        result = authorized(ref, contract)
        if count[0] > 1:
            result["device"] = "changed"
        return result

    with pytest.raises(InferenceError, match="mode/device changed"):
        load_candidate_policy(reference, authorization_verifier=verifier,
                              model_factory=lambda config: FakeModel(), checkpoint_loader=lambda *args: payload)


def test_import_never_loads_models_or_sdk():
    script = """
import importlib.abc, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith(('torch.', 'bc_stark_sdk')) or fullname.endswith('.FlowMatchingModel'):
            raise AssertionError('real model/SDK import forbidden')
sys.meta_path.insert(0, BlockModels())
import chaoyang.humanego_ict.inference
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
