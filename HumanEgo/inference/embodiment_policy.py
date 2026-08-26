"""Direct Kai22/Wuji20 flow-policy sampling; retargeting is forbidden here."""
from __future__ import annotations

from typing import Any, Sequence

import torch


@torch.no_grad()
def sample_h50(
    model: torch.nn.Module,
    observation: dict[str, torch.Tensor],
    *,
    seed: int | Sequence[int],
    steps: int = 20,
) -> dict[str, torch.Tensor]:
    """Euler-integrate one H=50 block from fixed noise and one observation."""
    if model.hand_action_representation not in {"kaihand_joint_state", "wuji_joint_state"}:
        raise ValueError("deployment policy requires a direct robot-q representation")
    if model.pred_horizon != 50 or model.single_hand:
        raise ValueError("deployment contract is dual-hand H=50")
    required = {"x_rgb", "x_ict", "ict_mask", "x_robot_state", "robot_state_mask"}
    missing = sorted(required - observation.keys())
    if missing:
        raise ValueError(f"missing observation fields: {missing}")
    device = observation["x_rgb"].device
    batch_size = observation["x_rgb"].shape[0]
    if isinstance(seed, int):
        generator = torch.Generator(device=device).manual_seed(seed)
        action = torch.randn(
            batch_size, 50, model.action_dim,
            generator=generator, device=device,
        )
    else:
        if len(seed) != batch_size:
            raise ValueError("one deterministic seed is required per batch item")
        action = torch.stack([
            torch.randn(
                50, model.action_dim,
                generator=torch.Generator(device=device).manual_seed(int(value)),
                device=device,
            )
            for value in seed
        ])
    done_logit = None
    use_cuda_amp = device.type == "cuda"
    for index in range(steps):
        time = torch.full((action.shape[0], 1), index / steps, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda_amp):
            result: dict[str, Any] = model(
                **{key: observation[key] for key in required},
                anchor_uv=observation.get("anchor_uv"), x_t=action, t=time,
            )
        action = action + result["v_pred"].float() / steps
        done_logit = result.get("done_logit")
    per_hand = 9 + model.hand_command_dim
    return {
        "action": action,
        "left_wrist9": torch.cat((action[..., 0:3], action[..., 6:12]), dim=-1),
        "right_wrist9": torch.cat((action[..., 3:6], action[..., 12:18]), dim=-1),
        "left_q": action[..., 18 : 18 + model.hand_command_dim],
        "right_q": action[..., 18 + model.hand_command_dim : 2 * per_hand],
        "done_logit": done_logit,
    }


@torch.no_grad()
def sample_receding_h50(
    model: torch.nn.Module,
    observation: dict[str, torch.Tensor],
    *,
    seed: int | Sequence[int],
    execute_steps: int = 5,
    steps: int = 20,
) -> dict[str, torch.Tensor]:
    """Predict H=50 but expose only the short prefix selected for execution."""
    if not 1 <= execute_steps <= 50:
        raise ValueError("execute_steps must lie in [1, 50]")
    plan = sample_h50(model, observation, seed=seed, steps=steps)
    return {
        **plan,
        "plan_action": plan["action"],
        "action": plan["action"][:, :execute_steps],
        "left_wrist9": plan["left_wrist9"][:, :execute_steps],
        "right_wrist9": plan["right_wrist9"][:, :execute_steps],
        "left_q": plan["left_q"][:, :execute_steps],
        "right_q": plan["right_q"][:, :execute_steps],
    }
