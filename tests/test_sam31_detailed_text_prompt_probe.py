from __future__ import annotations

import numpy as np
import pytest

from pipeline.sam31_detailed_text_prompt_probe import (
    ALL_PROMPTS,
    APPEARANCE_DIAGNOSTIC_PROMPTS,
    FRAME_INDICES,
    GENERAL_PROMPTS,
    OBJECT_CONTROL_PROMPT,
    DetailedPromptProbeContractError,
    prompt_group,
    prompt_slug,
    raw_instance_metrics,
    validate_prompt_inventory,
)


def _joints() -> np.ndarray:
    left = np.stack([np.linspace(10, 30, 21), np.linspace(20, 35, 21)], axis=1)
    right = np.stack([np.linspace(70, 90, 21), np.linspace(60, 75, 21)], axis=1)
    return np.stack([left, right])


def test_inventory_is_exact_disjoint_and_three_frames() -> None:
    validate_prompt_inventory()
    assert len(ALL_PROMPTS) == 16
    assert len(GENERAL_PROMPTS) == 11
    assert len(APPEARANCE_DIAGNOSTIC_PROMPTS) == 4
    assert OBJECT_CONTROL_PROMPT == "a beverage can"
    assert FRAME_INDICES == (232, 235, 242)
    assert set(GENERAL_PROMPTS).isdisjoint(APPEARANCE_DIAGNOSTIC_PROMPTS)


def test_appearance_prompts_are_never_general() -> None:
    for prompt in GENERAL_PROMPTS:
        assert prompt_group(prompt) == "GENERAL_PROJECT_SEMANTIC"
    for prompt in APPEARANCE_DIAGNOSTIC_PROMPTS:
        assert prompt_group(prompt) == "APPEARANCE_004_DIAGNOSTIC_ONLY"
    assert prompt_group(OBJECT_CONTROL_PROMPT) == "OBJECT_CONTROL"
    with pytest.raises(DetailedPromptProbeContractError):
        prompt_group("a session-specific fallback")


def test_slug_rejects_unregistered_prompt() -> None:
    assert prompt_slug("a white circular wrist device") == "a_white_circular_wrist_device"
    with pytest.raises(DetailedPromptProbeContractError):
        prompt_slug("unknown")


def test_metrics_are_read_only_and_per_instance() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 0:35] = True
    before = mask.copy()
    cad = np.zeros_like(mask)
    cad[20:30, 20:30] = True
    metrics = raw_instance_metrics(
        mask,
        score=0.9,
        instance_id=7,
        joints_by_side=_joints(),
        cad_object_mask=cad,
    )
    assert np.array_equal(mask, before)
    assert metrics["instance_id"] == 7
    assert metrics["area_pixels"] == int(mask.sum())
    assert metrics["object6d_overlap_pixels"] == 100
    assert metrics["pixels_changed_by_diagnostics"] == 0


def test_metrics_reject_shape_drift() -> None:
    with pytest.raises(DetailedPromptProbeContractError):
        raw_instance_metrics(
            np.zeros((10, 10), bool),
            score=0.0,
            instance_id=0,
            joints_by_side=np.zeros((2, 20, 2)),
            cad_object_mask=np.zeros((10, 10), bool),
        )
