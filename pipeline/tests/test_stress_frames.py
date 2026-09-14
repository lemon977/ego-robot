from __future__ import annotations

import numpy as np

from pipeline.stress_frames import select_stress_frames


def _features(length: int) -> dict[str, np.ndarray]:
    time = np.linspace(0, 1, length)
    contact = np.zeros(length)
    contact[int(0.25 * length) : int(0.45 * length)] = 0.9
    contact[int(0.65 * length) : int(0.85 * length)] = 0.95
    return {
        "arm_area_ratio_proxy": 0.1 + 0.05 * np.sin(2 * np.pi * time),
        "contact_score": contact,
        "object_occlusion_proxy": np.exp(-((time - 0.72) / 0.08) ** 2),
        "motion": np.exp(-((time - 0.55) / 0.03) ** 2),
        "sharpness": 1.0 - 0.8 * np.exp(-((time - 0.58) / 0.04) ** 2),
        "object_confidence": 0.8 - 0.6 * np.exp(-((time - 0.35) / 0.04) ** 2),
        "object_valid": np.ones(length),
        "donor_coverage": 0.9 - 0.8 * np.exp(-((time - 0.78) / 0.05) ** 2),
    }


def _role_frame(selection, role: str) -> int:
    return next(frame for frame, item in selection.evidence.items() if role in item["roles"])


def test_selects_contract_strata_and_count() -> None:
    selection = select_stress_frames(
        _features(180),
        fps=30,
        count_min=12,
        count_max=24,
        target_count=16,
        min_gap_seconds=0.1,
        stable_contact_seconds=0.2,
        pre_contact_seconds=0.2,
        contact_on_threshold=0.5,
    )
    assert 12 <= len(selection.frames) <= 24
    assert len(selection.frames) == len(set(selection.frames))
    for role in (
        "arm_area_low",
        "arm_area_high",
        "pre_contact",
        "first_contact",
        "stable_grasp",
        "maximum_occlusion",
        "release",
        "high_motion",
        "high_blur",
        "object6d_invalid_or_low_confidence",
        "donor_sparse",
    ):
        assert role in selection.covered_strata


def test_time_parameters_scale_with_fps() -> None:
    low = select_stress_frames(
        _features(120),
        fps=30,
        count_min=12,
        count_max=24,
        target_count=14,
        min_gap_seconds=0.1,
        stable_contact_seconds=0.2,
        pre_contact_seconds=0.2,
        contact_on_threshold=0.5,
    )
    high_features = {name: np.repeat(values, 2) for name, values in _features(120).items()}
    high = select_stress_frames(
        high_features,
        fps=60,
        count_min=12,
        count_max=24,
        target_count=14,
        min_gap_seconds=0.1,
        stable_contact_seconds=0.2,
        pre_contact_seconds=0.2,
        contact_on_threshold=0.5,
    )
    assert abs(_role_frame(low, "first_contact") / 30 - _role_frame(high, "first_contact") / 60) < 1 / 30
    assert low.normalized_parameters["pre_contact_frames"] == 6
    assert high.normalized_parameters["pre_contact_frames"] == 12


def test_missing_donor_is_not_claimed() -> None:
    features = _features(120)
    del features["donor_coverage"]
    selection = select_stress_frames(
        features,
        fps=30,
        count_min=12,
        count_max=24,
        target_count=12,
        min_gap_seconds=0.1,
        stable_contact_seconds=0.2,
        pre_contact_seconds=0.2,
        contact_on_threshold=0.5,
    )
    assert "donor_sparse" not in selection.covered_strata
