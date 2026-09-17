"""Deterministic, session-agnostic stress-frame selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


STRATUM_ORDER = (
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
)


@dataclass(frozen=True)
class StressSelection:
    frames: tuple[int, ...]
    covered_strata: tuple[str, ...]
    evidence: Mapping[int, Mapping[str, object]]
    normalized_parameters: Mapping[str, float | int]


def _finite_arg(values: np.ndarray, *, largest: bool) -> int | None:
    finite = np.flatnonzero(np.isfinite(values))
    if not finite.size:
        return None
    ranked = finite[np.argsort(values[finite], kind="stable")]
    return int(ranked[-1] if largest else ranked[0])


def _runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            if previous - start + 1 >= minimum_length:
                runs.append((start, previous))
            start = current
        previous = current
    if previous - start + 1 >= minimum_length:
        runs.append((start, previous))
    return runs


def select_stress_frames(
    features: Mapping[str, Sequence[float]],
    *,
    fps: float,
    count_min: int = 12,
    count_max: int = 24,
    target_count: int = 16,
    min_gap_seconds: float = 0.12,
    stable_contact_seconds: float = 0.20,
    pre_contact_seconds: float = 0.20,
    contact_on_threshold: float = 0.50,
) -> StressSelection:
    """Select stress frames from normalized measurements, never identifiers.

    All time controls are seconds converted using measured FPS.  Spatial inputs
    are ratios or dimensionless confidence scores computed by the profiler.
    Missing optional evidence remains uncovered rather than being fabricated.
    """

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not 12 <= count_min <= count_max <= 24:
        raise ValueError("stress frame bounds must satisfy 12 <= min <= max <= 24")
    if not count_min <= target_count <= count_max:
        raise ValueError("target_count must be within count bounds")
    for name, value in {
        "min_gap_seconds": min_gap_seconds,
        "stable_contact_seconds": stable_contact_seconds,
        "pre_contact_seconds": pre_contact_seconds,
    }.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0 <= contact_on_threshold <= 1:
        raise ValueError("contact_on_threshold must be in [0, 1]")

    required = (
        "arm_area_ratio_proxy",
        "contact_score",
        "object_occlusion_proxy",
        "motion",
        "sharpness",
        "object_confidence",
    )
    arrays: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for name in required:
        if name not in features:
            raise ValueError(f"missing frame feature: {name}")
        array = np.asarray(features[name], dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"frame feature {name} must be one-dimensional")
        arrays[name] = array
        lengths.add(len(array))
    for name in ("donor_coverage", "object_valid"):
        if name in features:
            array = np.asarray(features[name], dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"frame feature {name} must be one-dimensional")
            arrays[name] = array
            lengths.add(len(array))
    if len(lengths) != 1:
        raise ValueError("all frame features must have equal length")
    frame_count = lengths.pop()
    if frame_count < count_min:
        raise ValueError("session has fewer frames than the minimum review count")

    candidates: list[tuple[str, int | None]] = [
        ("arm_area_low", _finite_arg(arrays["arm_area_ratio_proxy"], largest=False)),
        ("arm_area_high", _finite_arg(arrays["arm_area_ratio_proxy"], largest=True)),
        ("maximum_occlusion", _finite_arg(arrays["object_occlusion_proxy"], largest=True)),
        ("high_motion", _finite_arg(arrays["motion"], largest=True)),
        ("high_blur", _finite_arg(arrays["sharpness"], largest=False)),
    ]

    object_valid = arrays.get("object_valid")
    if object_valid is not None and np.any(np.isfinite(object_valid) & (object_valid < 0.5)):
        invalid = np.flatnonzero(np.isfinite(object_valid) & (object_valid < 0.5))
        object_stress = int(invalid[0])
    else:
        object_stress = _finite_arg(arrays["object_confidence"], largest=False)
    candidates.append(("object6d_invalid_or_low_confidence", object_stress))

    donor = arrays.get("donor_coverage")
    candidates.append(("donor_sparse", _finite_arg(donor, largest=False) if donor is not None else None))

    stable_frames = max(1, int(round(stable_contact_seconds * fps)))
    pre_contact_frames = max(1, int(round(pre_contact_seconds * fps)))
    contact = np.isfinite(arrays["contact_score"]) & (
        arrays["contact_score"] >= contact_on_threshold
    )
    contact_runs = _runs(contact, stable_frames)
    if contact_runs:
        first_start = contact_runs[0][0]
        longest = max(contact_runs, key=lambda item: (item[1] - item[0] + 1, -item[0]))
        last_end = contact_runs[-1][1]
        candidates.extend(
            [
                ("pre_contact", max(0, first_start - pre_contact_frames)),
                ("first_contact", first_start),
                ("stable_grasp", (longest[0] + longest[1]) // 2),
                ("release", min(frame_count - 1, last_end + 1)),
            ]
        )

    selected_roles: dict[int, list[str]] = {}
    for stratum, frame_index in candidates:
        if frame_index is None:
            continue
        selected_roles.setdefault(int(frame_index), []).append(stratum)

    minimum_gap = max(1, int(round(min_gap_seconds * fps)))

    def sufficiently_separated(frame_index: int) -> bool:
        return all(abs(frame_index - existing) >= minimum_gap for existing in selected_roles)

    # Uniform time coverage is a deterministic fallback, not a data fallback.
    for value in np.linspace(0, frame_count - 1, target_count, dtype=np.int64):
        index = int(value)
        if len(selected_roles) >= target_count:
            break
        if index not in selected_roles and sufficiently_separated(index):
            selected_roles[index] = ["temporal_coverage"]

    # Maximin filling guarantees the minimum count even when extrema cluster.
    while len(selected_roles) < count_min:
        remaining = [index for index in range(frame_count) if index not in selected_roles]
        if not remaining:
            break
        if selected_roles:
            index = max(
                remaining,
                key=lambda item: (min(abs(item - existing) for existing in selected_roles), -item),
            )
        else:
            index = frame_count // 2
        selected_roles[index] = ["temporal_coverage"]

    if len(selected_roles) > count_max:
        required_frames = {
            frame
            for frame, roles in selected_roles.items()
            if any(role in STRATUM_ORDER for role in roles)
        }
        removable = sorted(
            (frame for frame in selected_roles if frame not in required_frames),
            reverse=True,
        )
        for frame in removable:
            if len(selected_roles) <= count_max:
                break
            del selected_roles[frame]
    if not count_min <= len(selected_roles) <= count_max:
        raise RuntimeError("selector could not satisfy the 12–24 frame contract")

    evidence: dict[int, Mapping[str, object]] = {}
    for frame in sorted(selected_roles):
        evidence[frame] = {
            "roles": tuple(selected_roles[frame]),
            "features": {
                name: (float(values[frame]) if np.isfinite(values[frame]) else None)
                for name, values in arrays.items()
            },
        }
    covered = tuple(
        stratum
        for stratum in STRATUM_ORDER
        if any(stratum in roles for roles in selected_roles.values())
    )
    parameters: Mapping[str, float | int] = {
        "fps": float(fps),
        "count_min": count_min,
        "count_max": count_max,
        "target_count": target_count,
        "min_gap_seconds": float(min_gap_seconds),
        "minimum_gap_frames": minimum_gap,
        "stable_contact_seconds": float(stable_contact_seconds),
        "stable_contact_frames": stable_frames,
        "pre_contact_seconds": float(pre_contact_seconds),
        "pre_contact_frames": pre_contact_frames,
        "contact_on_threshold": float(contact_on_threshold),
    }
    return StressSelection(
        frames=tuple(sorted(selected_roles)),
        covered_strata=covered,
        evidence=evidence,
        normalized_parameters=parameters,
    )
