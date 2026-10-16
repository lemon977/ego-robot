"""HumanEgo ICT candidate with an explicit Tianji/Revo2 H30x26 action head.

Importing defines architecture only. It does not load checkpoints, initialize
devices, download assets, send controls, or change the sealed H50x30 model.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from chaoyang.human_ego.training.FlowMatchingModel import FlowMatchingModel
from .joint26_contract import (
    ACTION_DIM, ACTION_LAYOUT, HORIZON, ICT_DIM, IMAGE_SIZE, MAX_ICT, MODEL_SCHEMA,
)


_ARCHITECTURE_KEYS = {
    "img_size", "patch_size", "vision_embed_dim", "num_decoder_layers", "num_heads",
    "mlp_ratio", "dropout", "horizon_weighting", "horizon_beta", "use_pre_norm",
    "use_ctx_norm",
}


def _validate_architecture(architecture: Mapping) -> dict:
    values = dict(architecture)
    extra = set(values) - _ARCHITECTURE_KEYS
    if extra:
        raise ValueError(f"JOINT26_UNSUPPORTED_ARCHITECTURE: {sorted(extra)}")
    if tuple(values.get("img_size", IMAGE_SIZE)) != IMAGE_SIZE:
        raise ValueError("JOINT26_FIXED_RGB_SIZE_240_320")
    values["img_size"] = IMAGE_SIZE
    integers = {
        "patch_size": 16, "vision_embed_dim": 384, "num_decoder_layers": 6,
        "num_heads": 8,
    }
    for name, default in integers.items():
        value = values.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"JOINT26_POSITIVE_INTEGER_REQUIRED: {name}")
    width = values.get("vision_embed_dim", 384)
    if width < 4 or width % 2 or width % values.get("num_heads", 8):
        raise ValueError("JOINT26_EMBED_WIDTH_HEAD_OR_TIME_EMBED_MISMATCH")
    patch = values.get("patch_size", 16)
    if any(size % patch for size in IMAGE_SIZE):
        raise ValueError("JOINT26_PATCH_MUST_DIVIDE_IMAGE")
    for name, default in (("mlp_ratio", 4.0), ("dropout", 0.05), ("horizon_beta", 0.0)):
        value = values.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"JOINT26_FINITE_NUMBER_REQUIRED: {name}")
        if name == "mlp_ratio" and value <= 0:
            raise ValueError("JOINT26_POSITIVE_MLP_RATIO_REQUIRED")
        if name == "dropout" and not 0 <= value < 1:
            raise ValueError("JOINT26_DROPOUT_RANGE")
        if name == "horizon_beta" and value < 0:
            raise ValueError("JOINT26_HORIZON_BETA_RANGE")
    for name in ("use_pre_norm", "use_ctx_norm"):
        if name in values and not isinstance(values[name], bool):
            raise ValueError(f"JOINT26_BOOLEAN_REQUIRED: {name}")
    if values.get("horizon_weighting", "uniform") != "uniform":
        raise ValueError("JOINT26_FIRST_RECIPE_UNIFORM_HORIZON_REQUIRED")
    return values


class Joint26ICTModel(FlowMatchingModel):
    """Single RGB + ICT + explicitly masked current joint state to joint flow.

    Each state token contains seven arm and six hand values, followed by the
    thirteen corresponding validity bits. Unknown values are zeroed *and*
    represented as unknown; they are never inferred from future targets.
    """

    model_schema = MODEL_SCHEMA
    action_layout = ACTION_LAYOUT
    control_ground_truth = False
    physical_deployment_authorized = False
    pi05_weight_compatible = False

    def __init__(self, *, pred_horizon: int = HORIZON, max_ict: int = MAX_ICT, **architecture):
        if isinstance(pred_horizon, bool) or not isinstance(pred_horizon, int) or pred_horizon != HORIZON:
            raise ValueError("JOINT26_FIXED_H30_REQUIRED")
        if isinstance(max_ict, bool) or not isinstance(max_ict, int) or max_ict != MAX_ICT:
            raise ValueError("JOINT26_FIXED_ICT8_REQUIRED")
        architecture = _validate_architecture(architecture)
        super().__init__(
            single_hand=False, pred_horizon=HORIZON, max_ict=MAX_ICT,
            hand_action_representation="grasp_binary", use_done_in_flow=False,
            use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
            use_aux_temporal_contrastive=False, use_pcd_features=False,
            use_region_attn=False, **architecture,
        )
        width = self.action_proj.out_features
        self.hand_action_representation = "tianji_revo2_absolute_joint26_candidate"
        self.hand_command_dim = 6
        self.base_action_dim = self.action_dim = ACTION_DIM
        self.action_proj = nn.Linear(ACTION_DIM, width)
        self.head_v = nn.Linear(width, ACTION_DIM)
        self.use_robot_state_conditioning = True
        self.robot_state_dim = 26  # Per side: 13 values and 13 validity bits.
        self.robot_state_proj = nn.Linear(self.robot_state_dim, width)
        self.robot_state_pos_emb = nn.Parameter(torch.randn(1, 2, width))
        self.head_done.requires_grad_(False)

    def forward(self, *, x_rgb, x_ict, ict_mask, x_state26, state_mask, x_t, t):
        if x_rgb.ndim != 4 or tuple(x_rgb.shape[1:]) != (3, *IMAGE_SIZE):
            raise ValueError("JOINT26_RGB_EXPECTED_B_3_240_320")
        batch = x_rgb.shape[0]
        expected = {
            "x_ict": (batch, MAX_ICT, ICT_DIM), "ict_mask": (batch, MAX_ICT),
            "x_state26": (batch, ACTION_DIM), "state_mask": (batch, ACTION_DIM),
            "x_t": (batch, HORIZON, ACTION_DIM), "t": (batch,),
        }
        actual = {"x_ict": x_ict, "ict_mask": ict_mask, "x_state26": x_state26,
                  "state_mask": state_mask, "x_t": x_t, "t": t}
        for name, shape in expected.items():
            if tuple(actual[name].shape) != shape:
                raise ValueError(f"JOINT26_SHAPE: {name}={tuple(actual[name].shape)}, expected {shape}")
            if actual[name].device != x_rgb.device:
                raise ValueError(f"JOINT26_DEVICE_MISMATCH: {name}")
        if ict_mask.dtype != torch.bool or state_mask.dtype != torch.bool:
            raise ValueError("JOINT26_MASKS_MUST_BE_BOOLEAN")
        for name, value in (("x_rgb", x_rgb), ("x_t", x_t), ("t", t)):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"JOINT26_NONFINITE_OR_NONFLOAT: {name}")
        if not x_ict.is_floating_point() or not x_state26.is_floating_point():
            raise ValueError("JOINT26_ICT_AND_STATE_MUST_BE_FLOATING_POINT")
        if not bool(torch.isfinite(x_ict[ict_mask]).all()):
            raise ValueError("JOINT26_NONFINITE_ADMITTED_ICT")
        if not bool(torch.isfinite(x_state26[state_mask]).all()):
            raise ValueError("JOINT26_NONFINITE_ADMITTED_CURRENT_STATE")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("JOINT26_FLOW_TIME_RANGE")
        side_values = torch.where(state_mask, x_state26, torch.zeros_like(x_state26)).reshape(batch, 2, 13)
        side_validity = state_mask.reshape(batch, 2, 13)
        state_tokens = torch.cat((side_values, side_validity.to(side_values.dtype)), dim=-1)
        side_mask = side_validity[:, :, :7].all(dim=-1)
        safe_ict = torch.where(ict_mask.unsqueeze(-1), x_ict, torch.zeros_like(x_ict))
        result = super().forward(
            # The public Joint26 time is (B,); the frozen parent's sinusoidal
            # embedding broadcasts each sample against a frequency row.
            x_rgb=x_rgb, x_ict=safe_ict, ict_mask=ict_mask, x_t=x_t, t=t.unsqueeze(-1),
            x_robot_state=state_tokens, robot_state_mask=side_mask,
        )
        return {"v_pred": result["v_pred"]}

    def compute_loss(self, *args, **kwargs):
        # The parent routes modality-major wrist dimensions. Using it here
        # would silently apply the wrong weights/limits to side-major joints.
        raise ValueError("JOINT26_EXPLICIT_MASKED_FLOW_LOSS_REQUIRED_NOT_WRIST_LOSS")


def build_joint26_model(model_config: Mapping) -> Joint26ICTModel:
    """Validate the explicit model declaration before allocating architecture."""
    if not isinstance(model_config, Mapping):
        raise ValueError("JOINT26_MODEL_CONFIG_MAPPING_REQUIRED")
    values = dict(model_config)
    for name, expected in (("action_dim", ACTION_DIM), ("ict_dim", ICT_DIM),
                           ("action_layout", ACTION_LAYOUT), ("model_schema", MODEL_SCHEMA)):
        if name in values:
            value = values.pop(name)
            if value != expected or (isinstance(expected, int) and
                                      (isinstance(value, bool) or not isinstance(value, int))):
                raise ValueError(f"JOINT26_MODEL_CONFIG_MISMATCH: {name}")
    if "horizon" in values:
        if "pred_horizon" in values:
            raise ValueError("JOINT26_AMBIGUOUS_HORIZON_CONFIG")
        values["pred_horizon"] = values.pop("horizon")
    if "image_size" in values:
        if "img_size" in values:
            raise ValueError("JOINT26_AMBIGUOUS_IMAGE_SIZE_CONFIG")
        values["img_size"] = values.pop("image_size")
    if "architecture" in values:
        architecture = values.pop("architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError("JOINT26_NESTED_ARCHITECTURE_MAPPING_REQUIRED")
        if set(values).intersection(_ARCHITECTURE_KEYS - {"img_size"}):
            raise ValueError("JOINT26_AMBIGUOUS_NESTED_AND_FLAT_ARCHITECTURE")
        nested = dict(architecture)
        _validate_architecture(nested)
        if "img_size" in nested and "img_size" in values:
            if tuple(nested["img_size"]) != tuple(values["img_size"]):
                raise ValueError("JOINT26_AMBIGUOUS_IMAGE_SIZE_CONFIG")
            nested.pop("img_size")
        values.update(nested)
    return Joint26ICTModel(**values)


def warmstart_shared_layers(model: Joint26ICTModel, source_state: Mapping, *, source_ref: Mapping) -> dict:
    """Load only same-shape shared MODEL tensors, never an action head or EMA.

    File SHA verification and choosing the MODEL entry are the caller's gated
    responsibility. This function takes a flat tensor state dictionary only;
    checkpoint wrappers, optimizer state and normalization are not accepted.
    """
    if not isinstance(source_ref, Mapping) or not isinstance(source_ref.get("path"), str):
        raise ValueError("JOINT26_WARMSTART_SOURCE_REF_REQUIRED")
    digest = source_ref.get("sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("JOINT26_WARMSTART_SHA_REQUIRED")
    if not isinstance(source_state, Mapping) or not source_state:
        raise ValueError("JOINT26_WARMSTART_FLAT_MODEL_STATE_REQUIRED")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
           for key, value in source_state.items()):
        raise ValueError("JOINT26_WARMSTART_FLAT_MODEL_TENSORS_ONLY_NO_EMA_OR_WRAPPER")
    target = model.state_dict()
    shared_prefixes = ("rgb_embed.", "state_proj.", "ctx_norm.", "time_mlp.", "decoder.")
    shared_singletons = {"state_pos_emb", "pos_embed"}
    selected, skipped = {}, []
    for name, tensor in source_state.items():
        reason = None
        if name.startswith(("action_proj.", "head_v.")):
            reason = "ACTION_SEMANTICS_CHANGED_ALWAYS_REINITIALIZED"
        elif not (name.startswith(shared_prefixes) or name in shared_singletons):
            reason = "NOT_SAME_SEMANTICS_SHARED_LAYER"
        elif name not in target:
            reason = "NOT_IN_TARGET_MODEL"
        elif tensor.shape != target[name].shape:
            reason = "SHAPE_MISMATCH_REINITIALIZED"
        elif tensor.dtype != target[name].dtype:
            reason = "DTYPE_MISMATCH_REINITIALIZED"
        elif not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"JOINT26_NONFINITE_WARMSTART_SHARED_TENSOR: {name}")
        if reason:
            skipped.append({"name": name, "reason": reason, "source_shape": list(tensor.shape),
                            "target_shape": list(target[name].shape) if name in target else None})
        else:
            selected[name] = tensor
    if not selected:
        raise ValueError("JOINT26_WARMSTART_NO_COMPATIBLE_SHARED_LAYERS")
    loaded = model.load_state_dict(selected, strict=False)
    if loaded.unexpected_keys:
        raise ValueError("JOINT26_UNEXPECTED_WARMSTART_KEYS")
    return {
        "schema": "humanego-ict-joint26-shared-warmstart-report-v1",
        "source_ref": dict(source_ref),
        "source_state_dict_key": "model",
        "source_is_ema": False,
        "optimizer_resumed": False,
        "loaded_keys": sorted(selected),
        "loaded_tensor_count": len(selected),
        "skipped": sorted(skipped, key=lambda row: row["name"]),
        "target_initialized_keys": sorted(loaded.missing_keys),
        "action_heads_always_reinitialized": True,
        "physical_deployment_authorized": False,
    }
