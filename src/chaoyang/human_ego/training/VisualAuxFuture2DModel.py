"""RGB-only future-2D auxiliary model for the exact78 Visual Aux comparison.

This model deliberately has no Robot action head.  Its only predictions are
normalized future image coordinates and their validity logits.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VisualAuxFuture2DModel(nn.Module):
    def __init__(
        self,
        *,
        horizon: int = 50,
        endpoints: int = 2,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.endpoints = endpoints
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, feature_dim, 3, stride=2, padding=1),
            nn.GroupNorm(16, feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.horizon_embedding = nn.Embedding(horizon, feature_dim)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, endpoints * 3),
        )

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must be float tensor [B,3,H,W]")
        batch = rgb.shape[0]
        context = self.encoder(rgb).flatten(1)
        horizon = self.horizon_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(context.unsqueeze(1) + horizon)
        decoded = decoded.view(batch, self.horizon, self.endpoints, 3)
        return {
            "xy_normalized": decoded[..., :2].sigmoid(),
            "valid_logits": decoded[..., 2],
        }


def visual_aux_loss(
    prediction: dict[str, torch.Tensor],
    target_xy: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    validity_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    xy = prediction["xy_normalized"].float()
    logits = prediction["valid_logits"].float()
    valid = target_valid.bool()
    if xy.shape != target_xy.shape or logits.shape != valid.shape:
        raise ValueError("prediction and future-2D target shapes differ")
    finite_target = torch.nan_to_num(target_xy.float(), nan=0.0)
    coordinate_values = F.smooth_l1_loss(xy, finite_target, reduction="none").sum(-1)
    coordinate_loss = (coordinate_values * valid).sum() / valid.sum().clamp(min=1)
    validity_loss = F.binary_cross_entropy_with_logits(logits, valid.float())
    total = coordinate_loss + validity_weight * validity_loss
    return {
        "loss": total,
        "coordinate_loss": coordinate_loss.detach(),
        "validity_loss": validity_loss.detach(),
    }


@torch.no_grad()
def visual_aux_metrics(
    prediction_xy: torch.Tensor,
    target_xy: torch.Tensor,
    valid: torch.Tensor,
    image_width: torch.Tensor,
    image_height: torch.Tensor,
    *,
    pck_threshold_px: float = 20.0,
) -> dict[str, float]:
    valid = valid.bool()
    target = torch.nan_to_num(target_xy.float(), nan=0.0)
    scale = torch.stack((image_width.float() - 1, image_height.float() - 1), -1)
    scale = scale[:, None, None, :]
    errors = torch.linalg.vector_norm((prediction_xy.float() - target) * scale, dim=-1)
    valid_errors = errors[valid]
    ade = float(valid_errors.mean()) if valid_errors.numel() else float("nan")
    final_valid = valid[:, -1]
    final_errors = errors[:, -1][final_valid]
    fde = float(final_errors.mean()) if final_errors.numel() else float("nan")
    pck = (
        float((valid_errors <= pck_threshold_px).float().mean())
        if valid_errors.numel()
        else float("nan")
    )
    both = valid.all(dim=-1)
    direct = errors.sum(dim=-1)
    swapped_delta = prediction_xy[:, :, [1, 0]] - target
    swapped = torch.linalg.vector_norm(swapped_delta * scale, dim=-1).sum(dim=-1)
    identity_error = (
        float((swapped[both] < direct[both]).float().mean()) if both.any() else float("nan")
    )
    smooth_values = []
    predicted_px = prediction_xy.float() * scale
    for side in range(prediction_xy.shape[2]):
        triplet_valid = valid[:, 2:, side] & valid[:, 1:-1, side] & valid[:, :-2, side]
        acceleration = (
            predicted_px[:, 2:, side]
            - 2 * predicted_px[:, 1:-1, side]
            + predicted_px[:, :-2, side]
        )
        if triplet_valid.any():
            smooth_values.append(torch.linalg.vector_norm(acceleration, dim=-1)[triplet_valid])
    smoothness = (
        float(torch.cat(smooth_values).mean()) if smooth_values else float("nan")
    )
    return {
        "ADE_2D_px": ade,
        "FDE_2D_px": fde,
        "PCK_20px": pck,
        "left_right_identity_error": identity_error,
        "temporal_smoothness_px": smoothness,
        "valid_endpoint_steps": float(valid.sum()),
    }
