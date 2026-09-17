from __future__ import annotations

import importlib.util
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from chaoyang.human_ego.inference.embodiment_policy import (
    rtc_attention_weights,
    sample_receding_h50,
)
from chaoyang.human_ego.inference.receding_horizon import reexpress_absolute_camera_action_chunk


class _ZeroVelocityModel(torch.nn.Module):
    hand_action_representation = "kaihand_joint_state"
    pred_horizon = 50
    single_hand = False
    action_dim = 62
    hand_command_dim = 22

    def __init__(self) -> None:
        super().__init__()
        self.unused_parameter = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, *, x_t: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        return {"v_pred": x_t * 0.0 + self.unused_parameter * 0.0}


def _observation(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "x_rgb": torch.zeros(batch_size, 3, 4, 4),
        "x_ict": torch.zeros(batch_size, 1, 29),
        "ict_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "x_robot_state": torch.zeros(batch_size, 2, 33),
        "robot_state_mask": torch.ones(batch_size, 2, dtype=torch.bool),
    }


def test_rtc_attention_weights_freeze_decay_and_release() -> None:
    weights = rtc_attention_weights(
        horizon=50,
        inference_delay=4,
        execution_horizon=10,
        schedule="exp",
    )
    assert torch.equal(weights[:4], torch.ones(4))
    assert torch.all(weights[4:40] < 1.0)
    assert torch.all(weights[4:40] > 0.0)
    assert torch.all(weights[4:39] > weights[5:40])
    assert torch.equal(weights[40:], torch.zeros(10))


def test_rtc_guidance_reduces_previous_chunk_overlap_error_on_cpu() -> None:
    model = _ZeroVelocityModel()
    observation = _observation()
    execution_horizon = 10
    remainder = torch.full((1, 50 - execution_horizon, model.action_dim), 0.25)

    plain = sample_receding_h50(
        model,
        observation,
        seed=7,
        execute_steps=execution_horizon,
        steps=20,
    )["plan_action"]
    rtc = sample_receding_h50(
        model,
        observation,
        seed=7,
        execute_steps=execution_horizon,
        steps=20,
        aligned_previous_action_remainder=remainder,
        inference_delay=4,
        max_guidance_weight=10.0,
        rtc_attention_schedule="exp",
    )["plan_action"]

    plain_error = (plain[:, :4] - remainder[:, :4]).abs().mean()
    rtc_error = (rtc[:, :4] - remainder[:, :4]).abs().mean()
    assert rtc_error < plain_error * 0.01
    assert model.unused_parameter.grad is None


def test_rtc_rejects_impossible_delay_execution_pair() -> None:
    with pytest.raises(ValueError, match="execution_horizon <= horizon"):
        rtc_attention_weights(
            horizon=50,
            inference_delay=30,
            execution_horizon=30,
        )


def test_camera_action_reexpression_moves_wrist_and_preserves_robot_q() -> None:
    action = torch.zeros(1, 2, 62)
    identity_o6d = torch.eye(3)[:, :2].reshape(6)
    action[..., 6:12] = identity_o6d
    action[..., 12:18] = identity_o6d
    action[..., 18:] = torch.arange(44)
    source_c2w = torch.eye(4)
    source_c2w[0, 3] = 1.0

    moved = reexpress_absolute_camera_action_chunk(
        action,
        source_c2w=source_c2w,
        target_c2w=torch.eye(4),
        pos_mean=torch.zeros(3),
        pos_std=torch.ones(3),
        hand_command_dim=22,
    )

    assert torch.allclose(moved[..., :6], torch.tensor([1.0, 0.0, 0.0] * 2))
    assert torch.allclose(moved[..., 6:18], action[..., 6:18])
    assert torch.equal(moved[..., 18:], action[..., 18:])
