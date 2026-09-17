"""Strictly causal tail-window inference for the pinned 16-frame HaWoR model.

For target frame ``t`` the model receives only ``max(segment_start,t-15)..t``;
short prefixes are left-padded with the segment's first frame.  Only the last
prediction is published.  The helper intentionally does not construct world
camera poses: camera-space model causality and causal SLAM are separate gates.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import default_collate


WINDOW = 16


def tail_indices(target: int, *, segment_start: int = 0, window: int = WINDOW) -> list[int]:
    if target < segment_start:
        raise ValueError("target precedes segment start")
    if window < 1:
        raise ValueError("window must be positive")
    start = max(segment_start, target - window + 1)
    values = list(range(start, target + 1))
    return [values[0]] * (window - len(values)) + values


def inference_causal_tail(
    model: Any,
    imgfiles: Any,
    boxes: Any,
    img_focal: float,
    img_center: Any,
    device: str = "cuda",
    do_flip: bool = False,
    *,
    batch_windows: int = 8,
) -> dict[str, torch.Tensor | float | Any]:
    """Run one causal 16-frame window per target and return last-frame output."""
    from lib.datasets.track_dataset import TrackDatasetEval

    database = TrackDatasetEval(
        imgfiles, boxes, img_focal=img_focal, img_center=img_center,
        normalization=True, dilate=1.2, do_flip=do_flip,
    )
    length = len(database)
    if length <= 0:
        raise ValueError("empty HaWoR segment")
    if batch_windows < 1:
        raise ValueError("batch_windows must be positive")
    keys = ("pred_cam", "pred_pose", "pred_shape", "pred_rotmat", "trans_full")
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in keys}

    for batch_start in range(0, length, batch_windows):
        targets = list(range(batch_start, min(batch_start + batch_windows, length)))
        windows = []
        for target in targets:
            items = [database[index] for index in tail_indices(target)]
            windows.append(default_collate(items))
        tensor_keys = [key for key, value in windows[0].items() if isinstance(value, torch.Tensor)]
        batch = {
            key: torch.stack([window[key] for window in windows], dim=0).to(device)
            for key in tensor_keys
        }
        with torch.no_grad():
            output = model.forward(batch)["out"]
        count = len(targets)
        for key in keys:
            values = output[key]
            if values.shape[0] != count * WINDOW:
                raise RuntimeError(f"unexpected {key} leading dimension: {values.shape}")
            reshaped = values.reshape(count, WINDOW, *values.shape[1:])
            collected[key].append(reshaped[:, -1].detach().cpu())

    fused: dict[str, torch.Tensor | float | Any] = {}
    for key in keys:
        fused["pred_trans" if key == "trans_full" else key] = torch.cat(collected[key], dim=0)
    fused["img_focal"] = img_focal
    fused["img_center"] = img_center
    fused["causal_contract"] = {
        "window": WINDOW,
        "policy": "LEFT_PAD_SEGMENT_FIRST_AND_PUBLISH_LAST_ONLY",
        "max_source_index_for_target_t": "t",
    }
    return fused


def install_causal_tail_inference(batch_windows: int = 8) -> None:
    """Patch the loaded class only; never modify vendored HaWoR files."""
    from lib.models.hawor import HAWOR

    def _inference(
        self: Any, imgfiles: Any, boxes: Any, img_focal: float, img_center: Any,
        device: str = "cuda", do_flip: bool = False,
    ):
        return inference_causal_tail(
            self, imgfiles, boxes, img_focal, img_center,
            device=device, do_flip=do_flip, batch_windows=batch_windows,
        )

    HAWOR.inference = _inference
