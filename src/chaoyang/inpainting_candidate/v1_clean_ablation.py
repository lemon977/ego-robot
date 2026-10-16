"""Candidate-only ProPainter pixel-propagation ablation.

The original loaded parent and model are never modified.  Use the returned
parent with the unchanged ``ProtectedProPainter``.  RAFT, completed flow,
feature propagation, references, chunking and all settings remain unchanged.
This module does not load models, allocate devices or grant execution rights.
"""
from __future__ import annotations

from copy import copy
from typing import Any

EXPERIMENT = "V1_PIXEL_PROPAGATION_BYPASS"


class PixelPropagationBypass:
    """Delegate the frozen model except its image-pixel propagation method."""

    def __init__(self, model: Any):
        if not callable(model) or not callable(getattr(model, "img_propagation", None)):
            raise ValueError("V1_CLEAN_MODEL_INTERFACE_REQUIRED")
        self._wrapped_model = model
        self.img_propagation_calls = 0
        self.forward_calls = 0
        self.propagation_records: list[dict[str, Any]] = []

    @property
    def wrapped_model(self) -> Any:
        return self._wrapped_model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.forward_calls += 1
        return self._wrapped_model(*args, **kwargs)

    def img_propagation(self, masked_frames: Any, completed_flows: Any,
                        masks: Any, interpolation: str = "nearest") -> tuple[Any, Any]:
        # Shape checks only: no tensor writes, device transfers, thresholds,
        # optical-flow operations or calls to the original propagation model.
        if (len(masked_frames.shape) != 5 or len(masks.shape) != 5
                or masked_frames.shape[2] != 3 or masks.shape[2] != 1
                or masked_frames.shape[:2] != masks.shape[:2]
                or masked_frames.shape[3:] != masks.shape[3:]
                or not all(int(n) > 0 for n in masked_frames.shape)):
            raise ValueError("V1_CLEAN_PROPAGATION_TENSOR_DOMAIN")
        if (masked_frames.dtype != masks.dtype
                or masked_frames.device != masks.device):
            raise ValueError("V1_CLEAN_PROPAGATION_DTYPE_OR_DEVICE")
        if interpolation != "nearest":
            raise ValueError("V1_CLEAN_FROZEN_PROPAGATION_INTERPOLATION")
        batch, length, _, height, width = masked_frames.shape
        self.img_propagation_calls += 1
        self.propagation_records.append({
            "batch": int(batch), "frames_with_original_context": int(length),
            "height": int(height), "width": int(width),
            "dtype": str(masked_frames.dtype), "interpolation": interpolation,
            "original_pixel_propagation_invoked": False,
            "updated_mask_source": "ORIGINAL_UNFILLED_MASK",
        })
        # The frozen caller expects flattened B*T outputs and restores B,T
        # with .view().  Leave every hole unfilled for feature propagation.
        return (masked_frames.reshape(batch * length, 3, height, width),
                masks.reshape(batch * length, 1, height, width))

    def metadata(self) -> dict[str, Any]:
        return {
            "experiment": EXPERIMENT,
            "single_change": "BYPASS_IMAGE_PIXEL_PROPAGATION_ONLY",
            "img_propagation_calls": self.img_propagation_calls,
            "forward_calls": self.forward_calls,
            "propagation_records": [dict(row) for row in self.propagation_records],
            "original_pixel_propagation_invoked": False,
            "model_forward_delegated_unchanged": True,
            "updated_mask_source": "ORIGINAL_UNFILLED_MASK",
            "settings_unchanged": True, "weights_unchanged": True,
            "original_parent_modified": False,
            "visual_quality_pass": False, "human_review": "PENDING",
            "training_eligible": False, "control_ground_truth": False,
        }


def make_pixel_propagation_ablation(parent: Any) -> tuple[Any, dict[str, Any]]:
    """Return a shallow, isolated parent and initial metadata; no model load.

    The caller must already hold a registered algorithm task and GPU lease.
    Shared loaded weights/RAFT/flow are read-only; do not run parent and
    candidate concurrently or call model-mutating methods through the proxy.
    Access ``candidate.model.metadata()`` after inference for actual counts.
    """
    from chaoyang.exact78_baseline.clean import SETTINGS

    if dict(parent.settings) != SETTINGS:
        raise ValueError("PROPAINTER_FROZEN_SETTINGS_DRIFT")
    proxy = PixelPropagationBypass(parent.model)
    candidate = copy(parent)
    if candidate is parent:
        raise ValueError("V1_CLEAN_PARENT_COPY_NOT_ISOLATED")
    candidate.settings = dict(parent.settings)
    candidate.model = proxy
    return candidate, proxy.metadata()
