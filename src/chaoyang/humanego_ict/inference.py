"""Offline H50/30D ICT adapter; importing it never imports Torch or an SDK.

The generic numerical functions are not data admission or deployment gates.
Actual model construction/loading is downstream of an injected runtime verifier.
Only explicitly exported tensor-only candidates are loadable; legacy HumanEgo
and pi05 checkpoints are not implicitly converted into Revo2 policies.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from chaoyang.humanego_ict.data import ACTION_LAYOUT, reference_bytes


INFERENCE_SCHEMA = "humanego-ict-revo2-offline-inference-v1"
TENSOR_CHECKPOINT_SCHEMA = "humanego-ict-revo2-inference-tensors-v1"
ROT6_LAYOUT = "ROW_INTERLEAVED_FIRST_TWO_COLUMNS"
HAND_CHANNELS = ("thumb_bend", "thumb_rotation", "index", "middle", "ring", "pinky")
HAND_UNIT = "NORMALIZED_0_1"


class InferenceError(ValueError):
    """Fail-closed conditioning, sampling, or checkpoint contract error."""


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise InferenceError(f"{name} requires finite shape {shape}")
    return result


def _se3(value, shape, name):
    value = _array(value, shape, name)
    rotation = value[..., :3, :3]
    if (not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise InferenceError(f"{name} is not a proper SE3")
    return value


def _normalization(value):
    if not isinstance(value, Mapping) or value.get("method") != "IDENTITY_NO_FIT":
        raise InferenceError("explicit IDENTITY_NO_FIT normalization required")
    mean = _array(value.get("position_mean"), (3,), "position_mean")
    std = _array(value.get("position_std"), (3,), "position_std")
    if not np.array_equal(mean, np.zeros(3)) or not np.array_equal(std, np.ones(3)):
        raise InferenceError("normalization differs from frozen identity contract")
    return mean, std


def _pose9(transforms, mean, std):
    return np.concatenate(((transforms[..., :3, 3] - mean) / std,
                           transforms[..., :3, :2].reshape(*transforms.shape[:-2], 6)), axis=-1)


def build_hand_ict(wrist_pose_world, wrist_valid, reference_camera_to_world, *,
                   normalization, source_qpc_ns, anchor_qpc_ns,
                   frame_declaration: Mapping[str, Any] | None = None):
    """Match data.py's hand-only ICT exactly, without fabricating grasp/objects.

    The caller must supply observed current wrist poses, not predicted future
    targets. Coordinate and clock qualification remains the upstream consumer's
    job; copied declarations are not calibrated measurements.
    """
    mean, std = _normalization(normalization)
    valid = np.asarray(wrist_valid)
    if valid.shape != (2,) or valid.dtype.kind != "b" or not valid.any():
        raise InferenceError("at least one explicit boolean current wrist validity required")
    poses = np.asarray(wrist_pose_world, dtype=np.float64)
    if poses.shape != (2, 4, 4):
        raise InferenceError("wrist_pose_world requires [2,4,4]")
    # Invalid upstream poses may contain NaN. They are never projected.
    poses = poses.copy()
    poses[~valid] = np.eye(4)
    _se3(poses, (2, 4, 4), "current wrists")
    camera = _se3(reference_camera_to_world, (4, 4), "reference camera")
    source = np.asarray(source_qpc_ns)
    if (source.shape != (2,) or source.dtype.kind not in "iu"
            or type(anchor_qpc_ns) is not int or anchor_qpc_ns < 0
            or np.any(source[valid] < 0) or np.any(source[valid] > anchor_qpc_ns)
            or np.any(source[valid] > np.iinfo(np.int64).max)):
        raise InferenceError("current wrist timestamps must be causal integer nanoseconds")
    tokens = np.zeros((8, 29), dtype=np.float32)
    mask = np.zeros(8, dtype=bool)
    timestamps = np.full(8, -1, dtype=np.int64)
    camera_from_world = np.linalg.inv(camera)
    for hand in range(2):
        if not valid[hand]:
            continue
        relations = _pose9(np.linalg.inv(poses[hand]) @ poses, mean, std)
        relations[~valid] = 0
        tokens[hand] = np.concatenate(([hand + 1], _pose9(camera_from_world @ poses[hand], mean, std),
                                        relations.reshape(18), [-1]))
        mask[hand] = True
        timestamps[hand] = source[hand]
    return {"x_ict": tokens, "ict_mask": mask,
            "conditioning_source_qpc_ns": timestamps,
            "anchor_qpc_ns": anchor_qpc_ns, "reference_camera_to_world": camera.copy(),
            "frame_declaration": dict(frame_declaration or {}),
            "grasp_status": "UNKNOWN", "object_status": "UNKNOWN_MASKED",
            "coordinate_qualification": "NOT_CERTIFIED_BY_NUMERICAL_ADAPTER",
            "control_ground_truth": False, "physical_deployment_authorized": False}


def _conditioning(x_rgb, x_ict, ict_mask):
    rgb = _array(x_rgb, (1, 3, 240, 320), "RGB")
    tokens = _array(x_ict, (1, 8, 29), "ICT")
    mask = np.asarray(ict_mask)
    if (mask.shape != (1, 8) or mask.dtype.kind != "b" or not mask.any()
            or mask[:, 2:].any() or np.any(tokens[~mask] != 0)
            or np.any(rgb < 0) or np.any(rgb > 1)):
        raise InferenceError("hand-only ICT/RGB conditioning contract violated")
    for hand in range(2):
        if mask[0, hand] and (tokens[0, hand, 0] != hand + 1 or tokens[0, hand, -1] != -1):
            raise InferenceError("current hand type or UNKNOWN grasp differs from training")
    return rgb.astype(np.float32), tokens.astype(np.float32), mask.copy()


def sample_actions(predict_velocity: Callable, *, x_rgb, x_ict, ict_mask,
                   initial_noise, steps=20, method="euler"):
    """Finite 0→1 Flow Matching integration with an explicit model callback.

    The callback accepts the model's five keyword inputs and returns either
    velocity ndarray or {v_pred: ndarray}. This routine draws no hidden RNG,
    loads no model, and does not reinterpret output as verified robot controls.
    """
    if type(steps) is not int or not 1 <= steps <= 1000 or method not in {"euler", "heun"}:
        raise InferenceError("finite sampler requires 1..1000 steps and euler/heun")
    rgb, tokens, mask = _conditioning(x_rgb, x_ict, ict_mask)
    action = _array(initial_noise, (1, 50, 30), "initial flow noise").astype(np.float32)

    def velocity(state, time):
        result = predict_velocity(x_rgb=rgb, x_ict=tokens, ict_mask=mask,
                                  x_t=state.copy(), t=np.array([[time]], dtype=np.float32))
        if isinstance(result, Mapping):
            result = result.get("v_pred")
        return _array(result, (1, 50, 30), "predicted flow velocity").astype(np.float32)

    dt = 1.0 / steps
    for step in range(steps):
        first = velocity(action, step * dt)
        if method == "heun":
            second = velocity(action + dt * first, (step + 1) * dt)
            action = action + (dt / 2) * (first + second)
        else:
            action = action + dt * first
        if not np.isfinite(action).all():
            raise InferenceError("nonfinite integrated actions")
    return action


def rot6_to_matrix(values):
    """Gram–Schmidt for R[:,:2].reshape(6), not column-concatenated 6D."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1:] != (6,) or not np.isfinite(values).all():
        raise InferenceError("finite row-interleaved rot6 required")
    columns = values.reshape(*values.shape[:-1], 3, 2)
    first, second = columns[..., :, 0], columns[..., :, 1]
    first_length = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_length <= 1e-8):
        raise InferenceError("degenerate first rot6 column")
    first = first / first_length
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_length = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_length <= 1e-8):
        raise InferenceError("degenerate second rot6 column")
    second = second / second_length
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def decode_actions(actions, reference_camera_to_world, *, normalization,
                   hand_command_unit=HAND_UNIT, hand_channel_order=HAND_CHANNELS,
                   command_policy="REJECT", frame_declaration=None):
    """Decode absolute candidate wrist poses in the original reference camera.

    CLIP_DIAGNOSTIC preserves the original out-of-range values and mask. It is
    an explicit offline display policy, not admission or a physical limit test.
    """
    actions = _array(actions, (50, 30), "H50 actions")
    mean, std = _normalization(normalization)
    camera = _se3(reference_camera_to_world, (4, 4), "reference camera")
    if (hand_command_unit != HAND_UNIT or tuple(hand_channel_order) != HAND_CHANNELS
            or command_policy not in {"REJECT", "CLIP_DIAGNOSTIC"}):
        raise InferenceError("six-channel unit/order or clipping policy unsupported")
    transforms = np.broadcast_to(np.eye(4), (50, 2, 4, 4)).copy()
    transforms[..., :3, 3] = actions[:, :6].reshape(50, 2, 3) * std + mean
    transforms[..., :3, :3] = rot6_to_matrix(actions[:, 6:18].reshape(50, 2, 6))
    original = actions[:, 18:].reshape(50, 2, 6).copy()
    outside = (original < 0) | (original > 1)
    if outside.any() and command_policy == "REJECT":
        raise InferenceError("predicted hand commands outside normalized range")
    return {"wrist_pose_reference_camera": transforms,
            "reference_camera_to_world": camera.copy(), "frame_id": "REFERENCE_CAMERA",
            "position_unit": "meters", "rotation_layout": ROT6_LAYOUT,
            "hand_command": np.clip(original, 0, 1) if command_policy == "CLIP_DIAGNOSTIC" else original,
            "unclipped_hand_command": original, "out_of_range_mask": outside,
            "hand_command_unit": HAND_UNIT, "hand_channel_order": list(HAND_CHANNELS),
            "display_policy": command_policy, "frame_declaration": dict(frame_declaration or {}),
            "control_ground_truth": False, "physical_deployment_authorized": False}


def _json_ref(reference):
    try:
        result = json.loads(reference_bytes(reference))
    except (TypeError, ValueError, OSError) as error:
        raise InferenceError("invalid immutable JSON reference") from error
    if not isinstance(result, dict):
        raise InferenceError("JSON reference must hold an object")
    return result


def _validate_contract(contract):
    fixed = {"schema": INFERENCE_SCHEMA, "horizon": 50, "action_dim": 30,
             "ict_dim": 29, "image_size": [240, 320], "action_layout": ACTION_LAYOUT,
             "rotation_layout": ROT6_LAYOUT, "hand_command_unit": HAND_UNIT,
             "hand_channel_order": list(HAND_CHANNELS), "control_ground_truth": False,
             "physical_deployment_authorized": False}
    for key, expected in fixed.items():
        actual = contract.get(key)
        if actual != expected or (isinstance(expected, bool) and actual is not expected):
            raise InferenceError(f"candidate inference contract mismatch: {key}")
    if contract.get("state_dict_key") not in {"model", "ema"}:
        raise InferenceError("explicit model/ema state_dict selection required")


def _authorization(verifier, reference, contract):
    if not callable(verifier):
        raise InferenceError("runtime authorization verifier required before model import")
    result = verifier(reference, contract)
    if (not isinstance(result, Mapping) or result.get("status") != "PASSED"
            or result.get("model_execution_allowed") is not True
            or result.get("contract_sha256") != reference["sha256"]):
        raise InferenceError("runtime model authorization rejected")
    mode = result.get("execution_mode")
    if mode == "REAL_MODEL_OFFLINE":
        if (result.get("gpu_lease_verified") is not True
                or not isinstance(result.get("device"), str)
                or not result["device"].startswith("cuda:")):
            raise InferenceError("real offline model requires verified GPU lease/device")
    elif mode != "SYNTHETIC_FIXTURE_ONLY":
        raise InferenceError("unsupported model execution mode")
    return dict(result)


@dataclass
class OfflinePolicy:
    """Loaded offline candidate; no sockets, robot clients, or SDK side effects."""

    predict_velocity: Callable
    normalization: Mapping[str, Any]
    contract: Mapping[str, Any]
    checkpoint_sha256: str

    def sample(self, *, x_rgb, x_ict, ict_mask, initial_noise, steps=20, method="euler"):
        return sample_actions(self.predict_velocity, x_rgb=x_rgb, x_ict=x_ict,
                              ict_mask=ict_mask, initial_noise=initial_noise,
                              steps=steps, method=method)


def load_candidate_policy(contract_ref, *, authorization_verifier,
                          model_factory=None, checkpoint_loader=None):
    """Executable loader adapter, gated before any model import/construction.

    model_factory(config) and checkpoint_loader(verified_bytes, device) may be
    injected for synthesis. Real defaults use Revo2ICTModel and torch.load with
    weights_only=True; there is deliberately no unsafe-pickle fallback. A
    future packet must freeze and authorize a tensor-only checkpoint export.
    """
    contract = _json_ref(contract_ref)
    _validate_contract(contract)
    authorization = _authorization(authorization_verifier, contract_ref, contract)

    def recheck_authorization():
        fresh = _authorization(authorization_verifier, contract_ref, contract)
        if (fresh["execution_mode"] != authorization["execution_mode"]
                or fresh.get("device") != authorization.get("device")):
            raise InferenceError("model authorization mode/device changed")
        return fresh

    synthetic = authorization["execution_mode"] == "SYNTHETIC_FIXTURE_ONLY"
    if synthetic and (model_factory is None or checkpoint_loader is None):
        raise InferenceError("synthetic loading requires both explicit fixture callbacks")
    config = _json_ref(contract["model_config_ref"])
    normalization = _json_ref(contract["normalization_ref"])
    _normalization(normalization)
    checkpoint_bytes = reference_bytes(contract["checkpoint_ref"])
    architecture = config.get("architecture")
    allowed = {"patch_size", "vision_embed_dim", "num_decoder_layers", "num_heads",
               "mlp_ratio", "dropout", "horizon_weighting", "horizon_beta",
               "use_pre_norm", "use_ctx_norm"}
    if (not isinstance(architecture, dict) or set(architecture) - allowed
            or config.get("horizon") != 50 or config.get("action_dim") != 30
            or config.get("ict_dim") != 29 or config.get("image_size") != [240, 320]):
        raise InferenceError("model architecture or fixed dimensions unsupported")
    device = "cpu" if synthetic else authorization["device"]
    if not synthetic:
        # Reading and hashing a large immutable checkpoint can outlive a lease.
        # Never carry a startup authorization across that delay into Torch.
        recheck_authorization()
        import torch
        if checkpoint_loader is None:
            checkpoint_loader = lambda encoded, device: torch.load(
                io.BytesIO(encoded), map_location=device, weights_only=True)
        if model_factory is None:
            from chaoyang.humanego_ict.model import Revo2ICTModel
            model_factory = lambda config: Revo2ICTModel(pred_horizon=50, **config["architecture"])
    recheck_authorization()
    payload = checkpoint_loader(checkpoint_bytes, device)
    expected_binding = {"config_sha256": contract["model_config_ref"]["sha256"],
                        "normalization_sha256": contract["normalization_ref"]["sha256"],
                        "action_layout": ACTION_LAYOUT}
    if (not isinstance(payload, Mapping) or payload.get("schema") != TENSOR_CHECKPOINT_SCHEMA
            or payload.get("binding") != expected_binding
            or payload.get("control_ground_truth") is not False
            or payload.get("physical_deployment_authorized") is not False
            or (payload.get("fixture_only") is True) != synthetic):
        raise InferenceError("not an explicitly bound tensor-only Revo2 candidate")
    state = payload.get(contract["state_dict_key"])
    if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) for key in state):
        raise InferenceError("candidate tensor state is missing")
    if not synthetic and any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                             for value in state.values()):
        raise InferenceError("candidate state must contain finite tensors only")
    recheck_authorization()
    model = model_factory(config)
    if not synthetic:
        recheck_authorization()
        model.to(device)
    recheck_authorization()
    model.load_state_dict(state, strict=True)
    model.eval()

    def velocity(**inputs):
        # Lease/task expiry and immutable-reference drift must be checked again
        # before every model call, not only when a policy object was constructed.
        recheck_authorization()
        if synthetic:
            return model(**inputs)
        tensors = {key: torch.as_tensor(value, device=device) for key, value in inputs.items()}
        with torch.inference_mode():
            prediction = model(**tensors)["v_pred"]
        return prediction.detach().float().cpu().numpy()

    return OfflinePolicy(velocity, normalization, contract, contract["checkpoint_ref"]["sha256"])
