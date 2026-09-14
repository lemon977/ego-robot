from __future__ import annotations

import pytest
import torch

from training.FlowMatchingModel import FlowMatchingModel


def _weighted_flow_loss(weight: float) -> float:
    model = FlowMatchingModel(
        single_hand=True,
        pred_horizon=1,
        max_ict=2,
        img_size=(16, 16),
        patch_size=16,
        vision_embed_dim=8,
        num_decoder_layers=1,
        num_heads=1,
        mlp_ratio=1.0,
        dropout=0.0,
        use_pcd_features=False,
        use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False,
        use_region_attn=False,
        use_done_in_flow=True,
    )
    # Sample 0 has unit squared error; sample 1 has squared error 9.  Equal
    # active dimensions make the normalized authority-weighted expectation
    # exactly (1 + weight * 9) / (1 + weight).
    target = torch.zeros((2, 1, model.action_dim), dtype=torch.float32)
    predicted = torch.stack(
        (torch.ones_like(target[0]), torch.full_like(target[1], 3.0)), dim=0
    )
    result = model.compute_loss(
        {"v_pred": predicted},
        {
            "v_target": target,
            "action_valid_mask": torch.ones_like(target, dtype=torch.bool),
            "sample_weight": torch.tensor([1.0, weight]),
        },
    )
    return float(result["loss_flow"])


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_grade_weight_changes_relative_sample_contribution(weight: float) -> None:
    assert _weighted_flow_loss(weight) == pytest.approx(
        (1.0 + weight * 9.0) / (1.0 + weight), rel=1.0e-6
    )


def test_all_zero_sample_weight_is_rejected() -> None:
    model = FlowMatchingModel(
        single_hand=True,
        pred_horizon=1,
        max_ict=2,
        img_size=(16, 16),
        patch_size=16,
        vision_embed_dim=8,
        num_decoder_layers=1,
        num_heads=1,
        mlp_ratio=1.0,
        dropout=0.0,
        use_pcd_features=False,
        use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False,
        use_region_attn=False,
        use_done_in_flow=True,
    )
    values = torch.zeros((2, 1, model.action_dim), dtype=torch.float32)
    with pytest.raises(ValueError, match="sample_weight cannot be all zero"):
        model.compute_loss(
            {"v_pred": values},
            {
                "v_target": values,
                "action_valid_mask": torch.ones_like(values, dtype=torch.bool),
                "sample_weight": torch.zeros(2),
            },
        )
