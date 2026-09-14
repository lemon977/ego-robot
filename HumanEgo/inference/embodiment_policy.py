"""Direct Kai22/Wuji20 flow-policy sampling; retargeting is forbidden here."""
from __future__ import annotations

from typing import Any, Literal, Sequence

import torch


_OBSERVATION_FIELDS = {
    "x_rgb", "x_ict", "ict_mask", "x_robot_state", "robot_state_mask"
}


def _validate_deployment_model(model: torch.nn.Module) -> None:
    if model.hand_action_representation not in {"kaihand_joint_state", "wuji_joint_state"}:
        raise ValueError("deployment policy requires a direct robot-q representation")
    if model.pred_horizon != 50 or model.single_hand:
        raise ValueError("deployment contract is dual-hand H=50")


def _validate_observation(observation: dict[str, torch.Tensor]) -> None:
    missing = sorted(_OBSERVATION_FIELDS - observation.keys())
    if missing:
        raise ValueError(f"missing observation fields: {missing}")


def _initial_action_noise(
    model: torch.nn.Module,
    observation: dict[str, torch.Tensor],
    seed: int | Sequence[int],
) -> torch.Tensor:
    device = observation["x_rgb"].device
    batch_size = observation["x_rgb"].shape[0]
    if isinstance(seed, int):
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(
            batch_size, 50, model.action_dim,
            generator=generator, device=device,
        )
    if len(seed) != batch_size:
        raise ValueError("one deterministic seed is required per batch item")
    return torch.stack([
        torch.randn(
            50, model.action_dim,
            generator=torch.Generator(device=device).manual_seed(int(value)),
            device=device,
        )
        for value in seed
    ])


def _format_action_result(
    model: torch.nn.Module,
    action: torch.Tensor,
    done_logit: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    per_hand = 9 + model.hand_command_dim
    return {
        "action": action,
        "left_wrist9": torch.cat((action[..., 0:3], action[..., 6:12]), dim=-1),
        "right_wrist9": torch.cat((action[..., 3:6], action[..., 12:18]), dim=-1),
        "left_q": action[..., 18 : 18 + model.hand_command_dim],
        "right_q": action[..., 18 + model.hand_command_dim : 2 * per_hand],
        "done_logit": done_logit,
    }


def rtc_attention_weights(
    *,
    horizon: int,
    inference_delay: int,
    execution_horizon: int,
    schedule: Literal["exp", "linear", "ones", "zeros"] = "exp",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the soft overlap mask from the RTC paper (Eq. 5).

    The first ``inference_delay`` actions are already committed and receive
    unit weight.  The non-overlapping suffix receives zero weight.  The
    intermediate portion follows the selected decay schedule.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0 <= inference_delay <= execution_horizon:
        raise ValueError("RTC requires 0 <= inference_delay <= execution_horizon")
    if execution_horizon > horizon - inference_delay:
        raise ValueError("RTC requires execution_horizon <= horizon - inference_delay")
    if schedule not in {"exp", "linear", "ones", "zeros"}:
        raise ValueError(f"unsupported RTC attention schedule: {schedule}")

    weights = torch.zeros(horizon, dtype=torch.float32, device=device)
    weights[:inference_delay] = 1.0
    overlap_end = horizon - execution_horizon
    if overlap_end <= inference_delay:
        return weights
    if schedule == "zeros":
        return weights
    if schedule == "ones":
        weights[inference_delay:overlap_end] = 1.0
        return weights

    positions = torch.arange(
        inference_delay, overlap_end, dtype=torch.float32, device=device
    )
    c = (horizon - execution_horizon - positions) / (
        horizon - execution_horizon - inference_delay + 1
    )
    if schedule == "linear":
        transition = c
    else:
        transition = torch.expm1(c) / torch.expm1(
            torch.ones((), dtype=torch.float32, device=device)
        )
    weights[inference_delay:overlap_end] = transition
    return weights


@torch.no_grad()
def sample_h50(
    model: torch.nn.Module,
    observation: dict[str, torch.Tensor],
    *,
    seed: int | Sequence[int],
    steps: int = 20,
) -> dict[str, torch.Tensor]:
    """Euler-integrate one H=50 block from fixed noise and one observation."""
    _validate_deployment_model(model)
    _validate_observation(observation)
    device = observation["x_rgb"].device
    action = _initial_action_noise(model, observation, seed)
    done_logit = None
    use_cuda_amp = device.type == "cuda"
    for index in range(steps):
        time = torch.full((action.shape[0], 1), index / steps, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda_amp):
            result: dict[str, Any] = model(
                **{key: observation[key] for key in _OBSERVATION_FIELDS},
                anchor_uv=observation.get("anchor_uv"),
                x_pcd=observation.get("x_pcd"), x_t=action, t=time,
            )
        action = action + result["v_pred"].float() / steps
        done_logit = result.get("done_logit")
    return _format_action_result(model, action, done_logit)


def sample_rtc_h50(
    model: torch.nn.Module,
    observation: dict[str, torch.Tensor],
    *,
    seed: int | Sequence[int],
    aligned_previous_action_remainder: torch.Tensor,
    inference_delay: int,
    execution_horizon: int,
    steps: int = 20,
    max_guidance_weight: float = 10.0,
    attention_schedule: Literal["exp", "linear", "ones", "zeros"] = "exp",
) -> dict[str, torch.Tensor]:
    """Sample an H=50 chunk using inference-time Real-Time Chunking.

    This implements the training-free PiGDM path from *Real-Time Execution of
    Action Chunking Flow Policies* (Black et al., 2025): the unexecuted tail of
    the previous chunk is right-padded, then used as a soft inpainting target
    while Euler-integrating the new flow.  The caller owns asynchronous queue
    timing and must pass the previous chunk after removing the
    ``execution_horizon`` actions already consumed.  Wrist poses in HumanEgo
    are camera-frame values, so the caller must re-express the remainder in
    the new observation's camera frame before calling this function.  Robot-q
    values are already frame invariant.
    """
    _validate_deployment_model(model)
    _validate_observation(observation)
    if steps < 1:
        raise ValueError("steps must be positive")
    if max_guidance_weight <= 0:
        raise ValueError("max_guidance_weight must be positive")

    device = observation["x_rgb"].device
    batch_size = observation["x_rgb"].shape[0]
    remainder = aligned_previous_action_remainder.to(device=device)
    if remainder.ndim != 3 or remainder.shape[0] != batch_size:
        raise ValueError("aligned_previous_action_remainder must have shape [B,L,A]")
    if remainder.shape[2] != model.action_dim:
        raise ValueError("aligned_previous_action_remainder action dimension mismatch")
    expected_remainder = 50 - execution_horizon
    if remainder.shape[1] != expected_remainder:
        raise ValueError(
            "aligned_previous_action_remainder must contain exactly "
            "H-execution_horizon "
            f"steps ({expected_remainder}), got {remainder.shape[1]}"
        )

    weights = rtc_attention_weights(
        horizon=50,
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
        schedule=attention_schedule,
        device=device,
    ).to(dtype=remainder.dtype)
    target = remainder.new_zeros((batch_size, 50, model.action_dim))
    target[:, :expected_remainder] = remainder
    action = _initial_action_noise(model, observation, seed).to(dtype=remainder.dtype)
    done_logit = None
    dt = 1.0 / steps
    use_cuda_amp = device.type == "cuda"

    # The VJP is with respect to x_t only.  Model parameters are not updated or
    # accumulated, so this path is safe for a frozen inference checkpoint.
    for index in range(steps):
        tau = index * dt
        with torch.enable_grad():
            x_t = action.detach().requires_grad_(True)
            time = torch.full(
                (batch_size, 1), tau, dtype=x_t.dtype, device=device
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda_amp
            ):
                result: dict[str, Any] = model(
                    **{key: observation[key] for key in _OBSERVATION_FIELDS},
                    anchor_uv=observation.get("anchor_uv"),
                    x_pcd=observation.get("x_pcd"), x_t=x_t, t=time,
                )
                velocity = result["v_pred"].float()
                denoised = x_t + (1.0 - tau) * velocity
            error = (target - denoised) * weights.view(1, 50, 1)
            correction = torch.autograd.grad(
                denoised,
                x_t,
                grad_outputs=error,
                create_graph=False,
                retain_graph=False,
            )[0]
            if 0.0 < tau < 1.0:
                coefficient = min(
                    max_guidance_weight,
                    (tau * tau + (1.0 - tau) ** 2)
                    / (tau * (1.0 - tau)),
                )
            else:
                coefficient = max_guidance_weight
            action = (x_t + dt * (velocity + coefficient * correction)).detach()
            value = result.get("done_logit")
            done_logit = value.detach() if value is not None else None

    output = _format_action_result(model, action, done_logit)
    output["rtc_attention_weights"] = weights
    return output


@torch.no_grad()
def sample_receding_h50(
    model: torch.nn.Module,
    observation: dict[str, torch.Tensor],
    *,
    seed: int | Sequence[int],
    execute_steps: int = 5,
    steps: int = 20,
    aligned_previous_action_remainder: torch.Tensor | None = None,
    inference_delay: int = 0,
    max_guidance_weight: float = 10.0,
    rtc_attention_schedule: Literal["exp", "linear", "ones", "zeros"] = "exp",
) -> dict[str, torch.Tensor]:
    """Predict H=50 but expose only the short prefix selected for execution."""
    if not 1 <= execute_steps <= 50:
        raise ValueError("execute_steps must lie in [1, 50]")
    if aligned_previous_action_remainder is None:
        if inference_delay:
            raise ValueError(
                "inference_delay requires aligned_previous_action_remainder"
            )
        plan = sample_h50(model, observation, seed=seed, steps=steps)
    else:
        plan = sample_rtc_h50(
            model,
            observation,
            seed=seed,
            aligned_previous_action_remainder=aligned_previous_action_remainder,
            inference_delay=inference_delay,
            execution_horizon=execute_steps,
            steps=steps,
            max_guidance_weight=max_guidance_weight,
            attention_schedule=rtc_attention_schedule,
        )
    return {
        **plan,
        "plan_action": plan["action"],
        "action": plan["action"][:, :execute_steps],
        "left_wrist9": plan["left_wrist9"][:, :execute_steps],
        "right_wrist9": plan["right_wrist9"][:, :execute_steps],
        "left_q": plan["left_q"][:, :execute_steps],
        "right_q": plan["right_q"][:, :execute_steps],
    }
