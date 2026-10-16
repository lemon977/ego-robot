"""Matched mask-only Clean comparison on one already loaded original ProPainter.

This module loads no model and grants no execution authority. A registered
consumer supplies the authority and budget callbacks. Every call uses the same
unwrapped parent through ProtectedProPainter, including image propagation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

import numpy as np

from .v1_clean_pair import (
    RandomState,
    array_signature,
    model_identity,
    require_same_backend,
    runtime_backend,
)

PAIR_SCHEMA = "inpainting-v1-mask-clean-causal-v1"
VARIANTS = ("A", "B", "A_repeat")


def _reproducibility(original: np.ndarray, repeated: np.ndarray) -> dict:
    """Measure numerical repeatability without making it a visual quality gate."""
    counts, maximum, total = [], 0, 0
    for first, second in zip(original, repeated):
        counts.append(int(np.count_nonzero(np.any(first != second, axis=-1))))
        delta = np.abs(first.astype(np.int16) - second.astype(np.int16))
        maximum = max(maximum, int(delta.max()))
        total += int(delta.sum(dtype=np.int64))
    exact = not any(counts)
    return {
        "status": "EXACT_MATCH" if exact else "NUMERICAL_DIFFERENCE",
        "exact_equal": exact,
        "difference_pixels_per_frame": counts,
        "max_abs_channel_delta": maximum,
        "mean_abs_channel_delta": total / original.size,
        "visual_quality_gate": False,
    }


def run_mask_clean_causal(
    parent: Any,
    raw: np.ndarray,
    base_mask: np.ndarray,
    repaired_mask: np.ndarray,
    protected: np.ndarray,
    *,
    authority: Callable[[], Any],
    on_call: Callable[[str], Any],
    backend_before_load: Mapping | None = None,
) -> dict:
    """Run fresh A/base, B/repaired and A_repeat/base, at most three calls.

    Raw RGB, protection, loaded model, settings, backend and initial RNG state
    stay fixed; only the removal mask changes. Callback failures stop before
    the next inference. The return value records A-repeat drift as evidence,
    without discarding an independently observed local improvement in B.
    """
    from chaoyang.pick_place_1008.visual import ProtectedProPainter, effective_clean_masks

    if not callable(authority) or not callable(on_call):
        raise ValueError("V1_MASK_CLEAN_CAUSAL_AUTHORITY_AND_BUDGET_REQUIRED")
    if (not isinstance(raw, np.ndarray) or raw.ndim != 4 or raw.shape[-1] != 3
            or raw.dtype != np.uint8 or len(raw) < 2
            or not raw.shape[1] or not raw.shape[2]
            or raw.shape[1] % 8 or raw.shape[2] % 8):
        raise ValueError("V1_MASK_CLEAN_CAUSAL_RGB_DOMAIN")
    if any(not isinstance(value, np.ndarray) or value.shape != raw.shape[:3]
           or value.dtype != np.bool_ for value in (base_mask, repaired_mask, protected)):
        raise ValueError("V1_MASK_CLEAN_CAUSAL_MASK_DOMAIN")

    # This also checks that the original settings remain frozen.
    ProtectedProPainter(parent)
    effective_a, _ = effective_clean_masks(base_mask, protected, parent.settings)
    effective_b, _ = effective_clean_masks(repaired_mask, protected, parent.settings)
    masks = {"A": base_mask, "B": repaired_mask, "A_repeat": base_mask}
    effective = {"A": effective_a, "B": effective_b, "A_repeat": effective_a}
    fixed_inputs = {
        "raw_rgb": raw,
        "base_removal": base_mask,
        "repaired_removal": repaired_mask,
        "protected_visible": protected,
        "base_effective_removal": effective_a,
        "repaired_effective_removal": effective_b,
    }
    frozen_signatures = {name: array_signature(value) for name, value in fixed_inputs.items()}
    backend = runtime_backend(parent.torch, parent.device)
    if backend_before_load is not None:
        require_same_backend(backend_before_load, backend)
    identity = model_identity(parent)
    initial_rng = RandomState(parent.torch, parent.device)
    initial_signature = initial_rng.metadata()
    outputs, validations, observations = {}, {}, []

    def check_fixed_state() -> dict:
        actual_backend = runtime_backend(parent.torch, parent.device)
        require_same_backend(backend, actual_backend)
        if model_identity(parent) != identity:
            raise ValueError("V1_MASK_CLEAN_CAUSAL_MODEL_STATE_DRIFT")
        if {name: array_signature(value) for name, value in fixed_inputs.items()} != frozen_signatures:
            raise ValueError("V1_MASK_CLEAN_CAUSAL_ACTUAL_INPUT_DRIFT")
        return actual_backend

    for variant in VARIANTS:
        authority()
        check_fixed_state()
        # Finish all authority/budget work before restoring the inference RNG.
        on_call(variant)
        authority()
        check_fixed_state()
        initial_rng.restore()
        actual_rng = RandomState(parent.torch, parent.device).metadata()
        if actual_rng != initial_signature:
            raise ValueError("V1_MASK_CLEAN_CAUSAL_RANDOM_STATE_RESET_FAILED")
        actual_inputs = {
            "raw_rgb": array_signature(raw),
            "removal": array_signature(masks[variant]),
            "protected_visible": array_signature(protected),
            "effective_removal": array_signature(effective[variant]),
        }
        painter = ProtectedProPainter(parent)
        outputs[variant] = painter.inpaint(raw, masks[variant], protected)
        authority()
        actual_backend = check_fixed_state()
        output = outputs[variant]
        if output.shape != raw.shape or output.dtype != np.uint8:
            raise ValueError("V1_MASK_CLEAN_CAUSAL_OUTPUT_DOMAIN")
        changed = np.any(output != raw, axis=-1)
        validation = deepcopy(painter.last_validation)
        protected_changes = np.count_nonzero(changed & protected, axis=(1, 2)).tolist()
        outside_changes = np.count_nonzero(changed & ~effective[variant], axis=(1, 2)).tolist()
        if (validation.get("status") != "PASS" or any(protected_changes) or any(outside_changes)
                or validation.get("protected_changed_pixels_per_frame") != protected_changes
                or validation.get("outside_delete_changed_pixels_per_frame") != outside_changes
                or not np.array_equal(painter.effective_masks, effective[variant])):
            raise ValueError("V1_MASK_CLEAN_CAUSAL_ORIGINAL_PIXEL_PROTECTION_FAILED")
        validation.update(
            independent_protected_changed_pixels_per_frame=protected_changes,
            independent_outside_effective_changed_pixels_per_frame=outside_changes,
        )
        validations[variant] = validation
        observations.append({
            "variant": variant,
            "provider": "ORIGINAL_PROPAINTER_WITH_IMAGE_PIXEL_PROPAGATION",
            "actual_backend": actual_backend,
            "random_state_before": actual_rng,
            "random_state_after": RandomState(parent.torch, parent.device).metadata(),
            "actual_inputs": actual_inputs,
            "output": array_signature(output),
        })

    difference = np.any(outputs["A"] != outputs["B"], axis=-1)
    added, removed = effective_b & ~effective_a, effective_a & ~effective_b
    union, shared = effective_a | effective_b, effective_a & effective_b
    receipt = {
        "schema": PAIR_SCHEMA,
        "status": "PAIRED_COMPLETE",
        "model_calls": len(observations),
        "call_order": list(VARIANTS),
        "fresh_A": True,
        "fresh_B": True,
        "fresh_A_repeat": True,
        "single_loaded_model_instance": True,
        "model_loads_by_pair": 0,
        "single_change": "REMOVAL_MASK_ONLY_ORIGINAL_PIXEL_PROPAGATION",
        "same_weights_raft_flow_feature_reference_subvideo_precision": True,
        "original_image_pixel_propagation_all_calls": True,
        "sam_loaded_by_pair": False,
        "actual_inputs": frozen_signatures,
        "backend_before_load": deepcopy(dict(backend_before_load)) if backend_before_load is not None else None,
        "actual_backend": backend,
        "historical_backend_inferred": False,
        "deterministic_execution_claim": False,
        "model_identity": identity,
        "random_state_initial": initial_signature,
        "calls": observations,
        "removal_mask_difference_pixels_per_frame": np.count_nonzero(base_mask != repaired_mask, axis=(1, 2)).tolist(),
        "effective_mask_added_pixels_per_frame": np.count_nonzero(added, axis=(1, 2)).tolist(),
        "effective_mask_removed_pixels_per_frame": np.count_nonzero(removed, axis=(1, 2)).tolist(),
        "A_B_difference_pixels_per_frame": np.count_nonzero(difference, axis=(1, 2)).tolist(),
        "A_B_difference_in_added_effective_per_frame": np.count_nonzero(difference & added, axis=(1, 2)).tolist(),
        "A_B_difference_in_shared_effective_per_frame": np.count_nonzero(difference & shared, axis=(1, 2)).tolist(),
        "A_B_difference_outside_effective_union_per_frame": np.count_nonzero(difference & ~union, axis=(1, 2)).tolist(),
        "A_B_protected_difference_pixels_per_frame": np.count_nonzero(difference & protected, axis=(1, 2)).tolist(),
        "A_repeat_reproducibility": _reproducibility(outputs["A"], outputs["A_repeat"]),
        "native_frames": len(raw),
        "completed_frame_indices": list(range(len(raw))),
        "history_clean_role": "EVALUATION_ONLY_NOT_A",
        "local_improvement_requires_all_frames_improved": False,
        "local_improvement_requires_repeat_exact_match": False,
        "visual_quality_pass": False,
        "human_review": "PENDING",
        "candidate_adopted": False,
        "training_eligible": False,
        "control_ground_truth": False,
    }
    return {**outputs, "effective_removal": effective, "validations": validations,
            "receipt": receipt, "rng_arrays": initial_rng.arrays()}
