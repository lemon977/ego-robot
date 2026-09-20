from __future__ import annotations

import numpy as np
import pytest

from chaoyang.human_ego.exact78_v31 import (
    FrozenDevelopmentTrainingConfig,
    TemporalAuthority,
    common_completed_checkpoint,
    development_capacity_report,
    future_2d_error_metrics,
    planned_common_checkpoints,
    simple_future_2d_predictions,
    suffix_invariance_report,
    training_terminal_status,
)


def test_development_volume_targets_do_not_block_a_legal_small_baseline() -> None:
    report = development_capacity_report(
        {
            "train": {"source_groups": 2, "windows": 17},
            "validation": {"source_groups": 1, "windows": 4},
        }
    )
    assert report["status"] == "PASS_LIMITED_DEVELOPMENT_CAPACITY"
    assert report["training_start_allowed"] is True
    assert report["rc1_release_eligibility"] is False
    assert report["splits"]["train"]["target_met"] is False


def test_frozen_training_config_and_terminal_rules() -> None:
    config = FrozenDevelopmentTrainingConfig()
    config.validate()
    assert training_terminal_status(
        global_update=10_000,
        completed_epochs=30,
        elapsed_seconds=1.0,
        config=config,
    ) == "TRAINING_COMPLETE_TARGET_UPDATES"
    assert training_terminal_status(
        global_update=900,
        completed_epochs=100,
        elapsed_seconds=1.0,
        config=config,
    ) == "TRAINING_COMPLETE_MAX_EPOCHS"
    assert training_terminal_status(
        global_update=900,
        completed_epochs=2,
        elapsed_seconds=43_200.0,
        config=config,
    ) == "TRAINING_PAUSED_BUDGET"
    with pytest.raises(ValueError, match="config is frozen"):
        FrozenDevelopmentTrainingConfig(seed=8).validate()


def test_fair_comparison_uses_only_common_1000_update_nodes() -> None:
    assert planned_common_checkpoints() == list(range(1_000, 10_001, 1_000))
    assert common_completed_checkpoint(
        [1_000, 2_000, 2_500, 3_000], [1_000, 2_000, 4_000]
    ) == 2_000
    assert common_completed_checkpoint([500], [500]) is None


def test_suffix_invariance_demotes_only_suffix_dependent_current_fields() -> None:
    full = {
        "rgb": np.zeros((2, 2, 3), np.uint8),
        "state": np.asarray([1.0, 2.0], np.float32),
        "offline_label": np.asarray([3.0], np.float32),
    }
    prefix = {
        "rgb": full["rgb"].copy(),
        "state": np.asarray([1.0, 2.0 + 2e-4], np.float32),
        "offline_label": np.asarray([9.0], np.float32),
    }
    report = suffix_invariance_report(
        full,
        prefix,
        {
            "rgb": TemporalAuthority.CAUSAL_CURRENT,
            "state": TemporalAuthority.CAUSAL_CURRENT,
            "offline_label": TemporalAuthority.OFFLINE_NONCAUSAL,
        },
    )
    assert report["status"] == "FAIL_CAUSAL_CURRENT_SUFFIX_DEPENDENCE"
    assert report["fields"]["rgb"]["effective_temporal_authority"] == "CAUSAL_CURRENT"
    assert report["fields"]["state"]["effective_temporal_authority"] == "OFFLINE_NONCAUSAL"
    assert report["fields"]["offline_label"]["effective_temporal_authority"] == "OFFLINE_NONCAUSAL"


def test_hold_and_constant_velocity_baselines_use_only_current_and_previous() -> None:
    previous = np.asarray([[[0.1, 0.2], [0.6, 0.4]]], np.float32)
    current = np.asarray([[[0.2, 0.25], [0.5, 0.5]]], np.float32)
    forecasts = simple_future_2d_predictions(current, previous, horizon=3)
    assert forecasts["HOLD_POSITION"].shape == (1, 3, 2, 2)
    assert np.allclose(forecasts["HOLD_POSITION"][:, 2], current)
    assert np.allclose(
        forecasts["CONSTANT_VELOCITY"][:, 0],
        current + (current - previous),
    )
    target = forecasts["HOLD_POSITION"].copy()
    valid = np.ones((1, 3, 2), bool)
    metrics = future_2d_error_metrics(
        forecasts["HOLD_POSITION"], target, valid, width=101, height=51
    )
    assert metrics == {
        "ADE_2D_px": 0.0,
        "FDE_2D_px": 0.0,
        "PCK_20px": 1.0,
        "valid_endpoint_steps": 6,
    }
