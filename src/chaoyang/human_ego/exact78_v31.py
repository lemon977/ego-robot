"""Exact78 V3.1 development contracts shared by training and audits.

This module is intentionally free of CUDA and file-system side effects.  It
defines the frozen DEVELOPMENT training budget, prefix/suffix causality audit,
simple future-2D baselines, and fair Raw/Robotized comparison rules.  RC1 data
volume remains a separate release gate; the DEVELOPMENT targets reported here
must never be used to relax it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

import numpy as np


class TemporalAuthority(StrEnum):
    CAUSAL_CURRENT = "CAUSAL_CURRENT"
    OFFLINE_NONCAUSAL = "OFFLINE_NONCAUSAL"
    INFERRED = "INFERRED"
    UNKNOWN_TEMPORAL_AUTHORITY = "UNKNOWN_TEMPORAL_AUTHORITY"


@dataclass(frozen=True)
class FrozenDevelopmentTrainingConfig:
    seed: int = 7
    batch_size: int = 16
    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    target_optimizer_updates: int = 10_000
    maximum_epochs: int = 100
    checkpoint_every_updates: int = 1_000
    per_model_gpu_cap_seconds: int = 12 * 60 * 60

    def validate(self) -> None:
        expected = FrozenDevelopmentTrainingConfig()
        if self != expected:
            raise ValueError(
                "Exact78 V3.1 DEVELOPMENT config is frozen: "
                f"expected {asdict(expected)}, got {asdict(self)}"
            )

    def as_receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "EXACT78_FROZEN_TRAINING_CONFIG_V31",
            **asdict(self),
            "comparison_rule": (
                "Raw and Robotized twins use the same manifest, seed, sampler, "
                "initialization and optimizer-update budget"
            ),
        }


DEVELOPMENT_TARGETS = {
    "train": {"source_groups": 16, "windows": 256},
    "validation": {"source_groups": 3, "windows": 48},
}


def development_capacity_report(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Report DEVELOPMENT capacity without turning targets into start gates."""

    splits: dict[str, Any] = {}
    for split, target in DEVELOPMENT_TARGETS.items():
        actual = counts.get(split, {})
        groups = int(actual.get("source_groups", 0))
        windows = int(actual.get("windows", 0))
        usable = groups >= 1 and windows >= 1
        target_met = (
            groups >= int(target["source_groups"])
            and windows >= int(target["windows"])
        )
        splits[split] = {
            "source_groups": groups,
            "windows": windows,
            "target": dict(target),
            "usable_for_limited_development": usable,
            "target_met": target_met,
        }
    eligible = all(value["usable_for_limited_development"] for value in splits.values())
    return {
        "schema_version": "EXACT78_DEVELOPMENT_CAPACITY_V31",
        "status": (
            "PASS_DEVELOPMENT_TARGET_CAPACITY"
            if eligible and all(value["target_met"] for value in splits.values())
            else "PASS_LIMITED_DEVELOPMENT_CAPACITY"
            if eligible
            else "BLOCKED_NO_LEGAL_DEVELOPMENT_WINDOWS"
        ),
        "training_start_allowed": eligible,
        "rc1_release_eligibility": False,
        "splits": splits,
    }


def planned_common_checkpoints(
    *, target_updates: int = 10_000, every_updates: int = 1_000
) -> list[int]:
    if target_updates < 1 or every_updates < 1:
        raise ValueError("update counts must be positive")
    nodes = list(range(every_updates, target_updates + 1, every_updates))
    if not nodes or nodes[-1] != target_updates:
        nodes.append(target_updates)
    return nodes


def common_completed_checkpoint(
    first_updates: list[int] | tuple[int, ...],
    second_updates: list[int] | tuple[int, ...],
    *,
    every_updates: int = 1_000,
) -> int | None:
    """Return the largest legal node completed by both paired branches."""

    if every_updates < 1:
        raise ValueError("every_updates must be positive")
    first = {int(value) for value in first_updates if int(value) % every_updates == 0}
    second = {int(value) for value in second_updates if int(value) % every_updates == 0}
    shared = sorted(first & second)
    return shared[-1] if shared else None


def training_terminal_status(
    *,
    global_update: int,
    completed_epochs: int,
    elapsed_seconds: float,
    config: FrozenDevelopmentTrainingConfig | None = None,
) -> str:
    frozen = config or FrozenDevelopmentTrainingConfig()
    frozen.validate()
    if global_update >= frozen.target_optimizer_updates:
        return "TRAINING_COMPLETE_TARGET_UPDATES"
    if completed_epochs >= frozen.maximum_epochs:
        return "TRAINING_COMPLETE_MAX_EPOCHS"
    if elapsed_seconds >= frozen.per_model_gpu_cap_seconds:
        return "TRAINING_PAUSED_BUDGET"
    return "TRAINING_IN_PROGRESS"


def _field_equal(first: np.ndarray, second: np.ndarray) -> tuple[bool, float | None]:
    if first.shape != second.shape or first.dtype != second.dtype:
        return False, None
    if first.dtype.kind in "buOSUV":
        return bool(np.array_equal(first, second)), None
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    finite = difference[np.isfinite(difference)]
    maximum = float(finite.max()) if len(finite) else 0.0
    equal = bool(np.allclose(first, second, atol=1e-6, rtol=1e-5, equal_nan=True))
    return equal, maximum


def suffix_invariance_report(
    full_inputs: Mapping[str, np.ndarray],
    prefix_inputs: Mapping[str, np.ndarray],
    declared_authority: Mapping[str, str | TemporalAuthority],
) -> dict[str, Any]:
    """Compare the same time-t input made with and without a future suffix.

    A differing field is never silently accepted as current-causal: the report
    assigns OFFLINE_NONCAUSAL regardless of the producer's declaration.
    """

    fields = sorted(set(full_inputs) | set(prefix_inputs) | set(declared_authority))
    rows: dict[str, Any] = {}
    causal_pass = True
    for field in fields:
        declared = str(declared_authority.get(field, TemporalAuthority.UNKNOWN_TEMPORAL_AUTHORITY))
        if field not in full_inputs or field not in prefix_inputs:
            equal, maximum, reason = False, None, "MISSING_COMPARISON_FIELD"
        else:
            equal, maximum = _field_equal(
                np.asarray(full_inputs[field]), np.asarray(prefix_inputs[field])
            )
            reason = "IDENTICAL_WITHIN_FROZEN_TOLERANCE" if equal else "SUFFIX_DEPENDENT"
        effective = declared if equal else TemporalAuthority.OFFLINE_NONCAUSAL.value
        if declared == TemporalAuthority.CAUSAL_CURRENT.value and not equal:
            causal_pass = False
        rows[field] = {
            "declared_temporal_authority": declared,
            "effective_temporal_authority": effective,
            "suffix_invariant": equal,
            "maximum_absolute_float_difference": maximum,
            "reason": reason,
        }
    return {
        "schema_version": "EXACT78_SUFFIX_INVARIANCE_V31",
        "status": "PASS" if causal_pass else "FAIL_CAUSAL_CURRENT_SUFFIX_DEPENDENCE",
        "causal_current_inputs_authorized": causal_pass,
        "tolerances": {
            "discrete": "BYTE_EXACT",
            "float_atol": 1e-6,
            "float_rtol": 1e-5,
        },
        "fields": rows,
    }


def simple_future_2d_predictions(
    current_xy: np.ndarray,
    previous_xy: np.ndarray,
    *,
    horizon: int = 50,
) -> dict[str, np.ndarray]:
    """Build hold-position and constant-velocity forecasts without future data."""

    current = np.asarray(current_xy, dtype=np.float32)
    previous = np.asarray(previous_xy, dtype=np.float32)
    if current.shape != previous.shape or current.shape[-1] != 2:
        raise ValueError("current/previous endpoint coordinates must match [...,2]")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    hold = np.repeat(current[..., None, :, :], horizon, axis=-3)
    velocity = current - previous
    steps = np.arange(1, horizon + 1, dtype=np.float32)
    reshape = (1,) * (current.ndim - 2) + (horizon, 1, 1)
    constant = current[..., None, :, :] + steps.reshape(reshape) * velocity[..., None, :, :]
    return {
        "HOLD_POSITION": np.clip(hold, 0.0, 1.0),
        "CONSTANT_VELOCITY": np.clip(constant, 0.0, 1.0),
    }


def future_2d_error_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, float | int | None]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if predicted.shape != truth.shape or predicted.shape[:-1] != mask.shape:
        raise ValueError("prediction/target/valid shapes differ")
    scale = np.asarray((width - 1, height - 1), dtype=np.float64)
    error = np.linalg.norm((predicted - truth) * scale, axis=-1)
    values = error[mask & np.isfinite(error)]
    final_mask = mask[..., -1, :] if mask.ndim >= 2 else mask
    final_error = error[..., -1, :]
    final_values = final_error[final_mask & np.isfinite(final_error)]
    return {
        "ADE_2D_px": float(values.mean()) if len(values) else None,
        "FDE_2D_px": float(final_values.mean()) if len(final_values) else None,
        "PCK_20px": float(np.mean(values <= 20.0)) if len(values) else None,
        "valid_endpoint_steps": int(len(values)),
    }
