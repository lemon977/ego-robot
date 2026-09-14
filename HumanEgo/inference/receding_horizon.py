"""Causal post-processing for dual-hand H=50 action chunks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Plan:
    frame: int
    q: np.ndarray


def reexpress_absolute_camera_action_chunk(
    action: "torch.Tensor",
    *,
    source_c2w: "torch.Tensor | np.ndarray",
    target_c2w: "torch.Tensor | np.ndarray",
    pos_mean: "torch.Tensor | np.ndarray",
    pos_std: "torch.Tensor | np.ndarray",
    hand_command_dim: int,
) -> "torch.Tensor":
    """Move dual-wrist absolute actions between observation camera frames.

    HumanEgo action layout is modality-major:
    ``[left/right position 6, left/right rotation-6D 12, robot q]``.
    Camera motion changes the first 18 values while robot q remains invariant.
    ``c2w`` must be the selector-bound camera-to-world authority for each
    observation.  This function intentionally does not support delta actions.
    """
    import torch

    values = torch.as_tensor(action)
    if values.ndim != 3:
        raise ValueError(f"action must have shape [B,H,A], got {tuple(values.shape)}")
    expected = 18 + 2 * int(hand_command_dim)
    if values.shape[-1] < expected:
        raise ValueError(
            f"action dimension {values.shape[-1]} is smaller than {expected}"
        )
    if hand_command_dim < 1:
        raise ValueError("hand_command_dim must be positive")
    if not torch.is_floating_point(values) or not torch.isfinite(values).all():
        raise ValueError("action must contain finite floating-point values")

    def camera_batch(value, label: str) -> torch.Tensor:
        matrix = torch.as_tensor(value, dtype=values.dtype, device=values.device)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        if matrix.shape[-2:] != (4, 4) or matrix.shape[0] not in {1, values.shape[0]}:
            raise ValueError(f"{label} must have shape [4,4] or [B,4,4]")
        if matrix.shape[0] == 1 and values.shape[0] != 1:
            matrix = matrix.expand(values.shape[0], -1, -1)
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{label} contains non-finite values")
        return matrix

    source = camera_batch(source_c2w, "source_c2w")
    target = camera_batch(target_c2w, "target_c2w")
    try:
        target_from_source = torch.linalg.solve(target, source)
    except RuntimeError as error:
        raise ValueError("target_c2w is not invertible") from error

    mean = torch.as_tensor(pos_mean, dtype=values.dtype, device=values.device).reshape(3)
    std = torch.as_tensor(pos_std, dtype=values.dtype, device=values.device).reshape(3)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("position statistics contain non-finite values")
    if torch.any(std <= 0):
        raise ValueError("position standard deviation must be positive")

    output = values.clone()
    position = values[..., :6].reshape(*values.shape[:2], 2, 3)
    position = position * std + mean
    rotation = target_from_source[:, None, None, :3, :3]
    translation = target_from_source[:, None, None, :3, 3]
    position = torch.matmul(rotation, position.unsqueeze(-1)).squeeze(-1) + translation
    output[..., :6] = ((position - mean) / std).reshape(*values.shape[:2], 6)

    raw_o6d = values[..., 6:18].reshape(*values.shape[:2], 2, 3, 2)
    first = torch.nn.functional.normalize(raw_o6d[..., 0], dim=-1)
    raw_second = raw_o6d[..., 1]
    second = raw_second - (first * raw_second).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    source_rotation = torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)
    target_rotation = torch.matmul(rotation, source_rotation)
    output[..., 6:18] = target_rotation[..., :2].reshape(*values.shape[:2], 12)
    return output


class RecedingH50QController:
    """Execute short prefixes while causally ensembling robot-q predictions.

    Wrist poses remain on the newest plan because HumanEgo predicts them in
    the current observation's camera reference frame.  Robot q is invariant
    to that frame and can therefore be combined across overlapping plans.
    """

    def __init__(
        self,
        joint_count: int,
        *,
        execute_steps: int = 5,
        ensemble_decay: float = 0.5,
        max_q_step_rad: float,
    ) -> None:
        if joint_count < 1:
            raise ValueError("joint_count must be positive")
        if not 1 <= execute_steps <= 50:
            raise ValueError("execute_steps must lie in [1, 50]")
        if ensemble_decay < 0:
            raise ValueError("ensemble_decay must be non-negative")
        if max_q_step_rad <= 0:
            raise ValueError("max_q_step_rad must be positive")
        self.joint_count = int(joint_count)
        self.execute_steps = int(execute_steps)
        self.ensemble_decay = float(ensemble_decay)
        self.max_q_step_rad = float(max_q_step_rad)
        self._plans: list[_Plan] = []

    @property
    def q_slice(self) -> slice:
        return slice(18, 18 + 2 * self.joint_count)

    def reset(self) -> None:
        self._plans.clear()

    def _ensemble_q(self, target_frame: int, newest_frame: int) -> np.ndarray:
        values = []
        weights = []
        for plan in self._plans:
            horizon = target_frame - plan.frame - 1
            if not 0 <= horizon < 50:
                continue
            age_queries = (newest_frame - plan.frame) / self.execute_steps
            values.append(plan.q[horizon])
            weights.append(np.exp(-self.ensemble_decay * age_queries))
        if not values:
            raise RuntimeError("no causal q prediction covers target frame")
        weight = np.asarray(weights, dtype=np.float64)
        stacked = np.asarray(values, dtype=np.float64)
        return np.sum(stacked * weight[:, None], axis=0) / np.sum(weight)

    def process(
        self,
        *,
        replan_frame: int,
        plan_action: np.ndarray,
        current_q: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Register one causal plan and return its safe executable prefix."""
        plan = np.asarray(plan_action, dtype=np.float64)
        if plan.ndim != 2 or plan.shape[0] != 50:
            raise ValueError(f"plan_action must be [50,A], got {plan.shape}")
        expected_q = 2 * self.joint_count
        if plan.shape[1] < 18 + expected_q:
            raise ValueError("plan_action does not contain the expected dual-hand q")
        current = np.asarray(current_q, dtype=np.float64).reshape(expected_q)
        lower = np.asarray(joint_lower, dtype=np.float64).reshape(expected_q)
        upper = np.asarray(joint_upper, dtype=np.float64).reshape(expected_q)
        if not (
            np.isfinite(plan).all()
            and np.isfinite(current).all()
            and np.isfinite(lower).all()
            and np.isfinite(upper).all()
        ):
            raise ValueError("non-finite rolling-horizon input")
        if np.any(lower >= upper):
            raise ValueError("invalid joint limits")

        raw_q = plan[:, self.q_slice].copy()
        self._plans.append(_Plan(int(replan_frame), raw_q))
        self._plans = [
            item for item in self._plans
            if replan_frame - item.frame < 50
        ]

        output = plan[: self.execute_steps].copy()
        previous = np.clip(current, lower, upper)
        clipped_steps = 0
        for offset in range(self.execute_steps):
            target_frame = int(replan_frame) + offset + 1
            q = self._ensemble_q(target_frame, int(replan_frame))
            delta = np.clip(
                q - previous,
                -self.max_q_step_rad,
                self.max_q_step_rad,
            )
            clipped_steps += int(np.count_nonzero(np.abs(q - previous) > self.max_q_step_rad))
            q = np.clip(previous + delta, lower, upper)
            output[offset, self.q_slice] = q
            previous = q

        raw_prefix = raw_q[: self.execute_steps]
        raw_limit_ratio = float(np.mean((raw_prefix < lower) | (raw_prefix > upper)))
        return output.astype(np.float32), {
            "raw_joint_limit_ratio": raw_limit_ratio,
            "q_step_clip_ratio": float(
                clipped_steps / max(1, self.execute_steps * expected_q)
            ),
            "plans_in_ensemble": float(len(self._plans)),
        }


class CausalPoseRateLimiter:
    """Causally smooth dual-wrist targets expressed in one stable frame.

    The caller must first transform every target into a common world or robot
    base frame.  Only the previous emitted pose and the current target are
    used; no future plan or ground truth enters this controller.
    """

    def __init__(
        self,
        *,
        max_translation_step_m: float = 0.010,
        max_rotation_step_rad: float = np.deg2rad(8.0),
        ema_alpha: float = 0.65,
    ) -> None:
        if max_translation_step_m <= 0:
            raise ValueError("max_translation_step_m must be positive")
        if max_rotation_step_rad <= 0:
            raise ValueError("max_rotation_step_rad must be positive")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must lie in (0, 1]")
        self.max_translation_step_m = float(max_translation_step_m)
        self.max_rotation_step_rad = float(max_rotation_step_rad)
        self.ema_alpha = float(ema_alpha)
        self._pose: np.ndarray | None = None
        self._active: np.ndarray | None = None

    def reset(self) -> None:
        self._pose = None
        self._active = None

    @staticmethod
    def _axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
        cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        if angle < 1e-8:
            return np.array([1.0, 0.0, 0.0]), 0.0
        sine = float(np.sin(angle))
        if abs(sine) > 1e-6:
            axis = np.array([
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]) / (2.0 * sine)
        else:
            # Stable fallback close to pi.
            diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
            axis = np.sqrt(diagonal)
            axis[np.argmax(axis)] = max(axis.max(), 1e-6)
        norm = float(np.linalg.norm(axis))
        return axis / max(norm, 1e-8), angle

    @staticmethod
    def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
        x, y, z = np.asarray(axis, dtype=np.float64)
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)

    def process(
        self,
        target_pose: np.ndarray,
        active: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        target = np.asarray(target_pose, dtype=np.float64)
        valid = np.asarray(active, dtype=bool).reshape(-1)
        if target.ndim != 3 or target.shape[1:] != (4, 4):
            raise ValueError(f"target_pose must be [N,4,4], got {target.shape}")
        if target.shape[0] != valid.size:
            raise ValueError("target_pose/active entity count mismatch")
        if not np.isfinite(target).all():
            raise ValueError("non-finite target pose")
        if self._pose is None:
            self._pose = target.copy()
            self._active = valid.copy()
            return target.copy(), {
                "translation_step_m": 0.0,
                "rotation_step_deg": 0.0,
                "translation_clipped": 0.0,
                "rotation_clipped": 0.0,
            }

        output = self._pose.copy()
        translation_steps, rotation_steps = [], []
        translation_clipped = rotation_clipped = 0
        for index in range(target.shape[0]):
            if not valid[index]:
                continue
            if not bool(self._active[index]):
                output[index] = target[index]
                continue
            previous = self._pose[index]
            raw_delta = target[index, :3, 3] - previous[:3, 3]
            filtered_delta = raw_delta * self.ema_alpha
            length = float(np.linalg.norm(filtered_delta))
            if length > self.max_translation_step_m:
                filtered_delta *= self.max_translation_step_m / length
                translation_clipped += 1
            output[index, :3, 3] = previous[:3, 3] + filtered_delta
            translation_steps.append(float(np.linalg.norm(filtered_delta)))

            relative = previous[:3, :3].T @ target[index, :3, :3]
            axis, raw_angle = self._axis_angle(relative)
            step_angle = raw_angle * self.ema_alpha
            if step_angle > self.max_rotation_step_rad:
                step_angle = self.max_rotation_step_rad
                rotation_clipped += 1
            output[index, :3, :3] = previous[:3, :3] @ self._rodrigues(axis, step_angle)
            rotation_steps.append(step_angle)
            output[index, 3] = (0.0, 0.0, 0.0, 1.0)

        self._pose = output.copy()
        self._active = valid.copy()
        return output, {
            "translation_step_m": float(np.mean(translation_steps)) if translation_steps else 0.0,
            "rotation_step_deg": float(np.degrees(np.mean(rotation_steps))) if rotation_steps else 0.0,
            "translation_clipped": float(translation_clipped),
            "rotation_clipped": float(rotation_clipped),
        }
