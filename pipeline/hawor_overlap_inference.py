"""Non-invasive overlapping-window inference for the pinned HaWoR model.

The upstream ``HAWOR.inference`` method evaluates disjoint 16-frame windows.
This adapter evaluates real 16-frame windows at a configurable stride and
fuses duplicate predictions.  Rotations are projected back onto SO(3), so the
published rotation matrices are valid rotations rather than element-wise
averages.  The vendored HaWoR source is intentionally left unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import default_collate
from tqdm import tqdm


WINDOW = 16


def _starts(length: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError("empty HaWoR sequence")
    if not 1 <= stride <= WINDOW:
        raise ValueError(f"stride must be in [1,{WINDOW}]")
    if length <= WINDOW:
        return [0]
    starts = list(range(0, length - WINDOW + 1, stride))
    final = length - WINDOW
    if starts[-1] != final:
        starts.append(final)
    return starts


def _project_so3(matrices: torch.Tensor) -> torch.Tensor:
    """Return the closest proper rotations using the polar/SVD projection."""
    u, _, vh = torch.linalg.svd(matrices)
    rotation = u @ vh
    determinant = torch.linalg.det(rotation)
    if torch.any(determinant < 0):
        correction = torch.ones((*rotation.shape[:-2], 3), dtype=rotation.dtype, device=rotation.device)
        correction[..., -1] = torch.where(determinant < 0, -1.0, 1.0)
        rotation = (u * correction.unsqueeze(-2)) @ vh
    return rotation


def inference_overlap(
    model: Any,
    imgfiles: Any,
    boxes: Any,
    img_focal: float,
    img_center: Any,
    device: str = "cuda",
    do_flip: bool = False,
    *,
    stride: int = 8,
) -> dict[str, torch.Tensor | float | Any]:
    """Run true repeated-window HaWoR inference and fuse overlap predictions."""
    from lib.datasets.track_dataset import TrackDatasetEval

    database = TrackDatasetEval(
        imgfiles, boxes, img_focal=img_focal, img_center=img_center,
        normalization=True, dilate=1.2, do_flip=do_flip,
    )
    length = len(database)
    starts = _starts(length, stride)
    accumulators: dict[str, torch.Tensor] = {}
    weights = torch.zeros(length, dtype=torch.float64)
    rotation_sum: torch.Tensor | None = None

    for start in tqdm(starts, desc=f"HaWoR 16/{stride} windows"):
        real_end = min(start + WINDOW, length)
        items = [database[index] for index in range(start, real_end)]
        while len(items) < WINDOW:
            items.append(items[-1])
        batch = default_collate(items)
        batch = {
            key: value.to(device).unsqueeze(0)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.no_grad():
            output = model.forward(batch)["out"]

        count = real_end - start
        # Symmetric center weighting retains a direct prediction for every
        # frame while reducing the influence of temporal-window boundaries.
        position = torch.arange(WINDOW, dtype=torch.float64)
        local_weight = torch.minimum(position + 1, WINDOW - position)[:count]
        weights[start:real_end] += local_weight

        for key in ("pred_cam", "pred_pose", "pred_shape", "trans_full"):
            values = output[key][:count].detach().cpu().to(torch.float64)
            if key not in accumulators:
                accumulators[key] = torch.zeros((length, *values.shape[1:]), dtype=torch.float64)
            reshape = (count,) + (1,) * (values.ndim - 1)
            accumulators[key][start:real_end] += values * local_weight.reshape(reshape)

        rotations = output["pred_rotmat"][:count].detach().cpu().to(torch.float64)
        if rotation_sum is None:
            rotation_sum = torch.zeros((length, *rotations.shape[1:]), dtype=torch.float64)
        rotation_sum[start:real_end] += rotations * local_weight[:, None, None, None]

    if torch.any(weights <= 0) or rotation_sum is None:
        raise RuntimeError("overlap inference left uncovered frames")
    fused: dict[str, torch.Tensor | float | Any] = {}
    for key, values in accumulators.items():
        reshape = (length,) + (1,) * (values.ndim - 1)
        fused[key] = (values / weights.reshape(reshape)).to(torch.float32)
    fused["pred_rotmat"] = _project_so3(rotation_sum / weights[:, None, None, None]).to(torch.float32)
    fused["pred_trans"] = fused.pop("trans_full")
    fused["img_focal"] = img_focal
    fused["img_center"] = img_center
    return fused


def install_overlap_inference(stride: int = 8) -> None:
    """Monkey-patch only the loaded Python class, never the vendor files."""
    from lib.models.hawor import HAWOR

    def _inference(self: Any, imgfiles: Any, boxes: Any, img_focal: float, img_center: Any,
                   device: str = "cuda", do_flip: bool = False):
        return inference_overlap(
            self, imgfiles, boxes, img_focal, img_center,
            device=device, do_flip=do_flip, stride=stride,
        )

    HAWOR.inference = _inference
