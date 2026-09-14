from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.validate_robot_task_translation_formalization import (
    FormalizationError,
    validate,
    validate_contract_constants,
)


PROJECT = Path(__file__).resolve().parents[1]
SPEC = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/"
    "robot_task_translation_formalization_prepare_v1/poker/"
    "POKER_042_VALIDATE_ONLY_SPEC.json"
)


def test_current_poker_validate_only_spec_passes_without_writes() -> None:
    result = validate(SPEC)
    assert result["status"] == "PASS_DEVELOPMENT_CANARY_VALIDATE_ONLY"
    assert result["source_frames"] == list(range(24))
    assert result["object6d_valid_frames_by_instance"] == {"0": 24}
    assert result["writes_performed"] == 0
    assert result["current_robot_authority"] is False


def test_zbuffer_epsilon_is_not_runtime_tunable() -> None:
    spec = json.loads(SPEC.read_text())
    changed = deepcopy(spec["contracts"])
    changed["zbuffer"]["epsilon_m"] = 0.004
    with pytest.raises(FormalizationError, match="constants drift"):
        validate_contract_constants(changed)


def test_contact_denominator_requires_all_five_named_pads() -> None:
    spec = json.loads(SPEC.read_text())
    changed = deepcopy(spec["contracts"])
    changed["contact"]["pads"] = ["thumb", "index"]
    with pytest.raises(FormalizationError, match="constants drift"):
        validate_contract_constants(changed)
