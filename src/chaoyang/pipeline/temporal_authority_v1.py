"""Suffix-invariance audit and temporal-authority admission helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "TEMPORAL_AUTHORITY_AUDIT_V1"
AUTHORITIES = frozenset(
    {
        "CAUSAL_CURRENT",
        "OFFLINE_NONCAUSAL",
        "INFERRED",
        "UNKNOWN_TEMPORAL_AUTHORITY",
    }
)


class TemporalAuthorityError(ValueError):
    """Raised when a temporal audit is ambiguous or internally inconsistent."""


def _compare_values(full: Any, truncated: Any, *, atol: float, rtol: float) -> tuple[bool, dict[str, Any]]:
    if isinstance(full, np.ndarray) or isinstance(truncated, np.ndarray):
        left = np.asarray(full)
        right = np.asarray(truncated)
        if left.shape != right.shape:
            return False, {"reason": "SHAPE_MISMATCH", "full_shape": list(left.shape), "truncated_shape": list(right.shape)}
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            equal = bool(np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True))
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            finite = difference[np.isfinite(difference)]
            return equal, {
                "comparison": "ALLCLOSE",
                "atol": atol,
                "rtol": rtol,
                "max_abs_difference": float(finite.max()) if finite.size else 0.0,
            }
        return bool(np.array_equal(left, right)), {"comparison": "BYTE_EXACT_ARRAY"}
    equal = full == truncated
    if isinstance(equal, np.ndarray):
        equal = bool(equal.all())
    return bool(equal), {"comparison": "EXACT_VALUE"}


def audit_suffix_invariance(
    *,
    full_current_inputs: Mapping[str, Any],
    truncated_current_inputs: Mapping[str, Any],
    declared_authority: Mapping[str, str] | None = None,
    required_fields: Sequence[str] = (),
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare the exact model inputs for a full and suffix-truncated build.

    The caller supplies the already materialised input at the same target time
    ``t``.  This keeps the audit independent of any one HaWoR or compositor
    implementation while covering RGB, crop, state, confidence, and other
    current-input fields with one contract.
    """

    if atol < 0 or rtol < 0:
        raise TemporalAuthorityError("atol and rtol must be non-negative")
    declarations = dict(declared_authority or {})
    unknown_declarations = sorted(set(declarations.values()) - AUTHORITIES)
    if unknown_declarations:
        raise TemporalAuthorityError(f"invalid temporal authorities: {unknown_declarations}")
    names = sorted(set(full_current_inputs) | set(truncated_current_inputs) | set(required_fields))
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        declaration = declarations.get(name)
        if name not in full_current_inputs or name not in truncated_current_inputs:
            rows[name] = {
                "suffix_invariant": False,
                "authority": "UNKNOWN_TEMPORAL_AUTHORITY",
                "online_current_admitted": False,
                "reason": "FIELD_MISSING_FROM_ONE_OR_BOTH_BUILDS",
            }
            continue
        invariant, diagnostic = _compare_values(
            full_current_inputs[name], truncated_current_inputs[name], atol=atol, rtol=rtol
        )
        if declaration == "INFERRED":
            authority = "INFERRED"
        elif invariant:
            authority = "CAUSAL_CURRENT"
        else:
            authority = "OFFLINE_NONCAUSAL"
        rows[name] = {
            "suffix_invariant": invariant,
            "authority": authority,
            "online_current_admitted": authority == "CAUSAL_CURRENT",
            "declared_authority": declaration,
            **diagnostic,
        }
    required_failures = [
        name for name in required_fields if rows[name]["authority"] != "CAUSAL_CURRENT"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED_CAUSAL_CURRENT" if not required_failures else "REJECTED_NONCAUSAL_CURRENT_INPUT",
        "atol": atol,
        "rtol": rtol,
        "fields": rows,
        "required_fields": list(required_fields),
        "required_field_failures": required_failures,
        "future_label_use_authorized": True,
        "future_to_current_input_flow_authorized": False,
    }


def validate_online_current_inputs(
    audit: Mapping[str, Any], required_fields: Sequence[str]
) -> list[str]:
    """Return fail-closed reasons for fields requested by an online consumer."""

    fields = audit.get("fields")
    if not isinstance(fields, Mapping):
        return ["TEMPORAL_AUTHORITY_FIELDS_MISSING"]
    failures: list[str] = []
    for name in required_fields:
        row = fields.get(name)
        if not isinstance(row, Mapping):
            failures.append(f"{name}:MISSING_TEMPORAL_AUTHORITY")
        elif row.get("authority") != "CAUSAL_CURRENT" or row.get("online_current_admitted") is not True:
            failures.append(f"{name}:{row.get('authority', 'UNKNOWN_TEMPORAL_AUTHORITY')}")
    return failures
