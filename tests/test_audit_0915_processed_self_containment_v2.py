from __future__ import annotations

import copy
import inspect

import pytest

from chaoyang.ops.audit_0915_processed_self_containment_v1 import SelfContainmentError
from chaoyang.ops import audit_0915_processed_self_containment_v2 as subject
from chaoyang.ops.audit_0915_processed_self_containment_v2 import _check_tactile


def tactile_side() -> dict:
    return {
        "schema_version": "tactile-acquisition-aligned-frame-v2",
        "wire_values_369": [0] * 369,
        "wire_active_mask_369": [True] * 369,
        "wire_valid_mask_369": [True] * 369,
        "finger_grid_5x4x8": [[[0] * 8 for _ in range(4)] for _ in range(5)],
        "active_mask_5x4x8": [[[True] * 8 for _ in range(4)] for _ in range(5)],
        "valid_mask_5x4x8": [[[True] * 8 for _ in range(4)] for _ in range(5)],
        "offline_source_valid": True,
        "offline_source_offset_ms": -3.25,
        "materialized_view": "nearest_host_qpc",
        "offline_payload": "source/raw/tactile.jsonl",
    }


def test_released_tactile_v2_passes() -> None:
    value = {"left": tactile_side(), "right": tactile_side()}
    assert _check_tactile(value, session="fixture", frame=7) == {
        "valid_sides": 2, "max_abs_offset_ms": 3.25,
    }


def test_old_tactile_v1_fails_closed() -> None:
    value = {"left": tactile_side(), "right": tactile_side()}
    value["left"]["schema_version"] = "tactile-rgb30-frame-v1"
    with pytest.raises(SelfContainmentError, match="schema mismatch"):
        _check_tactile(value, session="fixture", frame=0)


def test_v2_wire_validity_shape_fails_closed() -> None:
    value = {"left": tactile_side(), "right": tactile_side()}
    broken = copy.deepcopy(value)
    broken["right"]["wire_valid_mask_369"] = [True] * 368
    with pytest.raises(SelfContainmentError, match="wire_valid_mask_369"):
        _check_tactile(broken, session="fixture", frame=0)


def test_audit_publishes_explicit_allowed_access_and_denied_consumption_ledger() -> None:
    source = inspect.getsource(subject.audit_session)
    assert "allowed_access_manifest_sha256" in source
    assert '"denied_sidecar_files_opened": 0' in source
    assert '"denied_hand_or_controller_fields_consumed": 0' in source
    assert '"/entities/tactile"' in source
