from __future__ import annotations

import torch

from inference.embodiment_policy import sample_h50
from training.FlowMatchingModel import FlowMatchingModel


def tiny_model(representation: str) -> FlowMatchingModel:
    return FlowMatchingModel(
        single_hand=False, pred_horizon=50, max_ict=3, img_size=(16, 16),
        patch_size=16, vision_embed_dim=32, num_decoder_layers=1,
        num_heads=4, mlp_ratio=2, dropout=0, use_pcd_features=False,
        use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False, use_region_attn=False,
        hand_action_representation=representation,
    )


def observation(model: FlowMatchingModel, batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "x_rgb": torch.zeros(batch_size, 3, 16, 16),
        "x_ict": torch.zeros(batch_size, 3, 29),
        "ict_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "x_robot_state": torch.zeros(batch_size, 2, model.robot_state_dim),
        "robot_state_mask": torch.ones(batch_size, 2, dtype=torch.bool),
    }


def test_output_shapes_and_determinism() -> None:
    for representation, dimension, q_count in (
        ("kaihand_joint_state", 62, 22), ("wuji_joint_state", 58, 20)
    ):
        model = tiny_model(representation).eval()
        first = sample_h50(model, observation(model), seed=7, steps=2)
        second = sample_h50(model, observation(model), seed=7, steps=2)
        assert first["action"].shape == (1, 50, dimension)
        assert first["left_q"].shape == (1, 50, q_count)
        assert first["right_q"].shape == (1, 50, q_count)
        assert torch.equal(first["action"], second["action"])


def test_batched_sampling_preserves_per_block_seeds() -> None:
    model = tiny_model("kaihand_joint_state").eval()
    batched = sample_h50(model, observation(model, 2), seed=[7, 8], steps=2)
    first = sample_h50(model, observation(model), seed=7, steps=2)
    second = sample_h50(model, observation(model), seed=8, steps=2)
    assert torch.allclose(batched["action"][0], first["action"][0], atol=1e-6)
    assert torch.allclose(batched["action"][1], second["action"][0], atol=1e-6)


def test_missing_hand_has_zero_flow_gradient() -> None:
    model = tiny_model("wuji_joint_state")
    predicted = torch.ones(1, 50, 58, requires_grad=True)
    valid = torch.ones_like(predicted)
    valid[..., 3:6] = 0
    valid[..., 12:18] = 0
    valid[..., 38:58] = 0
    loss = model.compute_loss(
        {"v_pred": predicted, "done_logit": torch.zeros(1, 1)},
        {"v_target": torch.zeros_like(predicted), "action_valid_mask": valid,
         "y_done": torch.zeros(1, 1)},
    )["loss"]
    loss.backward()
    assert torch.count_nonzero(predicted.grad[..., 3:6]) == 0
    assert torch.count_nonzero(predicted.grad[..., 12:18]) == 0
    assert torch.count_nonzero(predicted.grad[..., 38:58]) == 0
    assert torch.count_nonzero(predicted.grad[..., :3]) > 0


def test_robot_q_loss_is_range_normalized_and_constrained() -> None:
    model = tiny_model("wuji_joint_state")
    predicted = torch.zeros(1, 50, 58, requires_grad=True)
    action = torch.zeros_like(predicted)
    action[..., 18:] = 2.0
    lower = torch.full((1, 2, 20), -1.0)
    upper = torch.full((1, 2, 20), 1.0)
    losses = model.compute_loss(
        {"v_pred": predicted, "action_pred": action,
         "done_logit": torch.zeros(1, 1)},
        {"v_target": torch.zeros_like(predicted),
         "action_valid_mask": torch.ones_like(predicted),
         "y_action": torch.zeros_like(predicted),
         "joint_lower": lower, "joint_upper": upper,
         "y_done": torch.zeros(1, 1)},
        loss_lambdas={"lambda_joint_limit": 1.0, "lambda_velocity": 1.0},
    )
    assert losses["loss_joint_limit"] > 0
    assert losses["loss_velocity"] == 0
