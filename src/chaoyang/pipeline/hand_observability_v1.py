"""Independent, part-level observability accounting for HaWoR evaluation.

This module deliberately does not infer visibility from HaWoR output.  It
validates an evidence dependency graph, demotes observations supported only by
HaWoR-dependent evidence to ``UNKNOWN``, and then computes both observable and
full-timeline denominators.  A visible part with no HaWoR output is therefore
an estimation failure, not a denominator exclusion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "HAND_OBSERVABILITY_V1"
SIDES = ("left", "right")
PARTS = ("wrist", "palm", "thumb", "index", "middle", "ring", "little")
STATES = frozenset(
    {
        "FULLY_VISIBLE",
        "PARTIALLY_VISIBLE_EVALUABLE",
        "OCCLUDED",
        "OUT_OF_FRAME",
        "UNKNOWN",
    }
)
OBSERVABLE_STATES = frozenset({"FULLY_VISIBLE", "PARTIALLY_VISIBLE_EVALUABLE"})
HAWOR_SENTINEL = "HAWOR_UNDER_EVALUATION"
PART_JOINT_INDICES = {
    "wrist": (0,),
    "palm": (0, 1, 5, 9, 13, 17),
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}


class HandObservabilityError(ValueError):
    """Raised when an observability ledger is structurally ambiguous."""


def _provider_map(providers: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(providers):
        provider_id = row.get("provider_id")
        if not isinstance(provider_id, str) or not provider_id:
            raise HandObservabilityError(f"providers[{index}].provider_id must be non-empty")
        if provider_id == HAWOR_SENTINEL:
            raise HandObservabilityError(f"{HAWOR_SENTINEL} is reserved")
        if provider_id in result:
            raise HandObservabilityError(f"duplicate provider_id: {provider_id}")
        if not isinstance(row.get("family"), str) or not row["family"]:
            raise HandObservabilityError(f"provider {provider_id} family must be non-empty")
        if not isinstance(row.get("independent_from_hawor"), bool):
            raise HandObservabilityError(
                f"provider {provider_id} independent_from_hawor must be boolean"
            )
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(value, str) or not value for value in dependencies
        ):
            raise HandObservabilityError(f"provider {provider_id} dependencies must be strings")
        result[provider_id] = row
    for provider_id, row in result.items():
        unknown = sorted(
            set(row.get("dependencies", [])) - set(result) - {HAWOR_SENTINEL}
        )
        if unknown:
            raise HandObservabilityError(
                f"provider {provider_id} has unknown dependencies: {unknown}"
            )
    return result


def audit_provider_independence(
    providers: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return transitive HaWoR dependency decisions and reject graph cycles."""

    rows = _provider_map(providers)
    visiting: set[str] = set()
    memo: dict[str, tuple[bool, tuple[str, ...]]] = {}

    def visit(provider_id: str) -> tuple[bool, tuple[str, ...]]:
        if provider_id in memo:
            return memo[provider_id]
        if provider_id in visiting:
            raise HandObservabilityError(f"provider dependency cycle includes {provider_id}")
        visiting.add(provider_id)
        row = rows[provider_id]
        reasons: list[str] = []
        family = str(row["family"])
        if "hawor" in family.casefold():
            reasons.append("PROVIDER_FAMILY_IS_HAWOR")
        if row["independent_from_hawor"] is not True:
            reasons.append("PROVIDER_DECLARES_NOT_INDEPENDENT")
        for dependency in row.get("dependencies", []):
            if dependency == HAWOR_SENTINEL:
                reasons.append("DEPENDS_ON_HAWOR_UNDER_EVALUATION")
                continue
            independent, dependency_reasons = visit(dependency)
            if not independent:
                reasons.append(f"TRANSITIVE_HAWOR_DEPENDENCY:{dependency}")
                reasons.extend(
                    f"VIA:{dependency}:{reason}" for reason in dependency_reasons
                )
        visiting.remove(provider_id)
        decision = (not reasons, tuple(sorted(set(reasons))))
        memo[provider_id] = decision
        return decision

    for provider_id in rows:
        visit(provider_id)
    return {
        provider_id: {
            "family": rows[provider_id]["family"],
            "dependencies": list(rows[provider_id].get("dependencies", [])),
            "declared_independent_from_hawor": rows[provider_id][
                "independent_from_hawor"
            ],
            "admissible_as_hawor_observability_evidence": memo[provider_id][0],
            "reason_codes": list(memo[provider_id][1]),
        }
        for provider_id in sorted(rows)
    }


def mano21_part_presence(joint_valid: np.ndarray) -> np.ndarray:
    """Convert ``[T,2,21]`` joint validity into strict ``[T,2,7]`` presence."""

    values = np.asarray(joint_valid, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != (2, 21):
        raise HandObservabilityError("joint_valid must have shape [T,2,21]")
    return np.stack(
        [values[..., list(PART_JOINT_INDICES[part])].all(axis=-1) for part in PARTS],
        axis=-1,
    )


def evaluate_hand_observability(
    *,
    frame_count: int,
    observations: Sequence[Mapping[str, Any]],
    providers: Sequence[Mapping[str, Any]],
    hawor_part_output_present: np.ndarray,
    pose_part_quality_pass: np.ndarray,
) -> dict[str, Any]:
    """Build a fail-closed per-part denominator and quality ledger.

    Missing observation rows are explicitly materialised as ``UNKNOWN``.  A
    non-UNKNOWN row must cite at least one independent provider.  Otherwise it
    is demoted to ``UNKNOWN`` and remains visible in the dependency audit.
    """

    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        raise HandObservabilityError("frame_count must be a positive integer")
    provider_audit = audit_provider_independence(providers)
    present = np.asarray(hawor_part_output_present, dtype=bool)
    quality = np.asarray(pose_part_quality_pass, dtype=bool)
    expected_shape = (frame_count, len(SIDES), len(PARTS))
    if present.shape != expected_shape or quality.shape != expected_shape:
        raise HandObservabilityError(
            f"HaWoR part arrays must both have shape {expected_shape}"
        )
    if np.any(quality & ~present):
        raise HandObservabilityError("pose quality cannot pass without HaWoR part output")

    raw_state = np.full(expected_shape, "UNKNOWN", dtype="U32")
    effective_state = raw_state.copy()
    provider_ids: dict[tuple[int, int, int], tuple[str, ...]] = {}
    demotions: list[dict[str, Any]] = []
    occupied: set[tuple[int, int, int]] = set()
    for index, row in enumerate(observations):
        frame_id = row.get("frame_id")
        side = row.get("side")
        part = row.get("part")
        state = row.get("state")
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or not 0 <= frame_id < frame_count:
            raise HandObservabilityError(f"observations[{index}].frame_id is invalid")
        if side not in SIDES or part not in PARTS or state not in STATES:
            raise HandObservabilityError(f"observations[{index}] side/part/state is invalid")
        key = (frame_id, SIDES.index(side), PARTS.index(part))
        if key in occupied:
            raise HandObservabilityError(f"duplicate observation for {frame_id}/{side}/{part}")
        occupied.add(key)
        evidence = row.get("evidence_provider_ids", [])
        if not isinstance(evidence, list) or any(
            not isinstance(value, str) or value not in provider_audit for value in evidence
        ):
            raise HandObservabilityError(
                f"observations[{index}].evidence_provider_ids contains unknown provider"
            )
        evidence_tuple = tuple(sorted(set(evidence)))
        provider_ids[key] = evidence_tuple
        raw_state[key] = state
        admissible = [
            value
            for value in evidence_tuple
            if provider_audit[value]["admissible_as_hawor_observability_evidence"]
        ]
        if state != "UNKNOWN" and not admissible:
            effective_state[key] = "UNKNOWN"
            demotions.append(
                {
                    "frame_id": frame_id,
                    "side": side,
                    "part": part,
                    "raw_state": state,
                    "reason": "NO_HAWOR_INDEPENDENT_EVIDENCE",
                    "evidence_provider_ids": list(evidence_tuple),
                }
            )
        else:
            effective_state[key] = state

    records: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    aggregate = {
        "timeline_total": frame_count * len(SIDES) * len(PARTS),
        "observable_total": 0,
        "observable_pass": 0,
        "observable_estimation_failure": 0,
        "visible_but_no_hawor_output": 0,
        "observable_quality_failure": 0,
        "unknown": 0,
        "out_of_frame": 0,
        "occluded": 0,
    }
    for side_index, side in enumerate(SIDES):
        summaries[side] = {}
        for part_index, part in enumerate(PARTS):
            states = effective_state[:, side_index, part_index]
            observable = np.isin(states, tuple(OBSERVABLE_STATES))
            part_present = present[:, side_index, part_index]
            part_quality = quality[:, side_index, part_index]
            passed = observable & part_present & part_quality
            absent_failure = observable & ~part_present
            quality_failure = observable & part_present & ~part_quality
            counts = {
                "timeline_total": frame_count,
                "observable_total": int(observable.sum()),
                "observable_pass": int(passed.sum()),
                "observable_estimation_failure": int((observable & ~passed).sum()),
                "visible_but_no_hawor_output": int(absent_failure.sum()),
                "observable_quality_failure": int(quality_failure.sum()),
                "unknown": int((states == "UNKNOWN").sum()),
                "out_of_frame": int((states == "OUT_OF_FRAME").sum()),
                "occluded": int((states == "OCCLUDED").sum()),
            }
            counts["observable_pass_rate"] = (
                counts["observable_pass"] / counts["observable_total"]
                if counts["observable_total"]
                else None
            )
            counts["timeline_pass_rate"] = counts["observable_pass"] / frame_count
            summaries[side][part] = counts
            for key in aggregate:
                if key != "timeline_total":
                    aggregate[key] += int(counts[key])
            for frame_id in range(frame_count):
                state = str(states[frame_id])
                is_observable = bool(observable[frame_id])
                if not is_observable:
                    outcome = "NOT_IN_POSE_QUALITY_DENOMINATOR"
                elif not part_present[frame_id]:
                    outcome = "FAIL_VISIBLE_NO_HAWOR_OUTPUT"
                elif not part_quality[frame_id]:
                    outcome = "FAIL_OBSERVABLE_POSE_QUALITY"
                else:
                    outcome = "PASS_OBSERVABLE_POSE_QUALITY"
                records.append(
                    {
                        "frame_id": frame_id,
                        "side": side,
                        "part": part,
                        "raw_state": str(raw_state[frame_id, side_index, part_index]),
                        "effective_state": state,
                        "evidence_provider_ids": list(
                            provider_ids.get((frame_id, side_index, part_index), ())
                        ),
                        "hawor_output_present": bool(part_present[frame_id]),
                        "pose_quality_pass": bool(part_quality[frame_id]),
                        "outcome": outcome,
                    }
                )
    aggregate["observable_pass_rate"] = (
        aggregate["observable_pass"] / aggregate["observable_total"]
        if aggregate["observable_total"]
        else None
    )
    aggregate["timeline_pass_rate"] = aggregate["observable_pass"] / aggregate["timeline_total"]
    aggregate["unknown_fraction"] = aggregate["unknown"] / aggregate["timeline_total"]
    aggregate["out_of_frame_fraction"] = (
        aggregate["out_of_frame"] / aggregate["timeline_total"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED_ACCOUNTING",
        "provider_audit": provider_audit,
        "dependency_demotions": demotions,
        "per_side_part": summaries,
        "aggregate": aggregate,
        "records": records,
        "quality_claim_promoted": False,
        "claim_limit": (
            "Observability denominator and HaWoR-output accounting only; it does not "
            "improve pose values or promote session-strict authority."
        ),
    }
