from __future__ import annotations

import copy

import pytest

from chaoyang.ops.audit_0915_processed_self_containment_v1 import (
    SelfContainmentError,
    _check_tactile,
)


def tactile_side() -> dict:
    return {
        "schema_version": "tactile-rgb30-frame-v1",
        "wire_values_369": [0] * 369,
        "finger_grid_5x4x8": [[([0] * 8) for _ in range(4)] for _ in range(5)],
        "active_mask_5x4x8": [[([True] * 8) for _ in range(4)] for _ in range(5)],
        "valid_mask_5x4x8": [[([True] * 8) for _ in range(4)] for _ in range(5)],
        "offline_source_valid": True,
        "offline_source_offset_ms": -3.25,
    }


def test_tactile_allowlisted_payload_passes() -> None:
    value = {"left": tactile_side(), "right": tactile_side()}
    result = _check_tactile(value, session="fixture", frame=7)
    assert result == {"valid_sides": 2, "max_abs_offset_ms": 3.25}


def test_tactile_offset_fails_closed() -> None:
    value = {"left": tactile_side(), "right": tactile_side()}
    value["right"]["offline_source_offset_ms"] = 40.01
    with pytest.raises(SelfContainmentError, match="tactile offset"):
        _check_tactile(value, session="fixture", frame=7)


def test_tactile_shape_fails_closed() -> None:
    value = {"left": tactile_side(), "right": tactile_side()}
    broken = copy.deepcopy(value)
    broken["left"]["wire_values_369"] = [0] * 368
    with pytest.raises(SelfContainmentError, match="wire shape"):
        _check_tactile(broken, session="fixture", frame=0)

