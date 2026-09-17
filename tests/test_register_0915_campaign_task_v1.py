from __future__ import annotations

import copy

import pytest

from chaoyang.governance import register_0915_campaign_task_v1 as subject


def _state(predecessor: str, status: str = "PASSED") -> dict:
    return {
        "next_task": None,
        "tasks": [{
            "task_id": predecessor,
            "status": status,
            "result": {"path": "/nonexistent", "bytes": 1, "sha256": "0" * 64},
        }],
    }


def test_later_stage_requires_immediate_predecessor_pass() -> None:
    state = _state("0915_foundationstereo_full_v1")
    predecessor = subject._validate_predecessor(state, "0915_sam31_mask_full_v1")
    assert predecessor["task_id"] == "0915_foundationstereo_full_v1"
    bad = copy.deepcopy(state)
    bad["tasks"][0]["status"] = "FAILED_RUNTIME_FINAL"
    with pytest.raises(RuntimeError, match="did not pass"):
        subject._validate_predecessor(bad, "0915_sam31_mask_full_v1")


def test_registration_rejects_any_live_task() -> None:
    state = _state("0915_foundationstereo_full_v1")
    state["tasks"].append({"task_id": "other", "status": "RUNNING"})
    with pytest.raises(RuntimeError, match="no active"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")


def test_registration_rejects_duplicate_task() -> None:
    state = _state("0915_foundationstereo_full_v1")
    state["tasks"].append({"task_id": "0915_sam31_mask_full_v1", "status": "PASSED"})
    with pytest.raises(RuntimeError, match="already registered"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")
