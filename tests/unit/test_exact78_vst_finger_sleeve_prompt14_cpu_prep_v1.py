"""CPU-only negative checks for the unregistered three-anchor proposal."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from chaoyang.governance import exact78_vst_finger_sleeve_prompt14_cpu_prep_v1 as prep


def _sealed() -> tuple[dict, dict, dict]:
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in (
        prep.SEALED / "RESULT.json", prep.SEALED / "candidate/MASK_ANCHOR.json", prep.PACKET))


def test_one_role_only_changes_and_yellow_cable_is_frozen() -> None:
    _, anchor, packet = _sealed()
    row = next(row for row in anchor["rows"] if row["session_id"] == prep.SOURCE_SESSION)
    proposed = prep._role_prompts(row, packet["config"])
    assert {name for name in proposed if proposed[name] != row["prompts"][name]} == {
        "finger_sleeve"}
    assert proposed["yellow_cable"] == "a yellow cable"
    for role in ("human", "object", "equipment_original", "equipment_added",
                 "yellow_cable"):
        bad = copy.deepcopy(row)
        bad["prompts"][role] += " changed"
        with pytest.raises(RuntimeError, match="SEALED_ROLE_DRIFT"):
            prep._role_prompts(bad, packet["config"])


def test_threshold_drift_rejected_before_model_pin_validation() -> None:
    outer, anchor, packet = _sealed()
    packet["config"]["output_prob_thresh"] = 0.49
    values = iter((outer, anchor, packet))
    with (patch.object(prep, "_object", side_effect=lambda _path: next(values)),
          patch.object(prep, "validate_artifact_ref") as validate):
        with pytest.raises(RuntimeError, match="THRESHOLD_OR_BASE_CONFIG_DRIFT"):
            prep.frozen_inputs()
    validate.assert_not_called()


def test_wrong_sealed_mask13_result_sha_rejected() -> None:
    with patch.object(prep, "_exact_ref", side_effect=RuntimeError("PROMPT14_SHA_DRIFT")):
        with pytest.raises(RuntimeError, match="PROMPT14_SHA_DRIFT"):
            prep.frozen_inputs()


def test_finite_draft_has_no_execution_authority() -> None:
    draft = prep.packet_draft({"sealed": "synthetic"})
    assert draft["status"] == "DRAFT_NOT_REGISTERED"
    assert draft["anchor_frames"] == [0, 75, 149]
    assert draft["budgets"]["model_calls_max"] == 3
    assert draft["budgets"]["session_attempts_max"] == 1
    assert draft["budgets"]["clean_calls"] == 0
    assert draft["budgets"]["render_calls"] == 0
    assert draft["quality_hold"].startswith("YELLOW_CABLE_ZERO_DETECTION")
    assert "run_signature" not in draft
    assert "writer" not in draft
    assert "runtime_code" not in draft


def test_only_check_only_cli_is_exposed(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(prep, "packet_draft", return_value=prep.packet_draft({})):
        assert prep.main(["--check-only"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "DRAFT_NOT_REGISTERED"
    assert value["yellow_cable"] == "UNRESOLVED_QUALITY_HOLD"
    with pytest.raises(SystemExit):
        prep.main(["--execute"])


def test_sealed_0915_rows_are_three_fixed_jpgs_without_device_instances() -> None:
    _, anchor, _ = _sealed()
    rows = [row for row in anchor["rows"] if row["session_id"] == prep.SOURCE_SESSION]
    assert [row["frame"] for row in rows] == [0, 75, 149]
    for row in rows:
        assert Path(row["sam_input"]["path"]).suffix == ".jpg"
        assert all(row["instance_counts"][name] == 0 for name in (
            "finger_sleeve", "yellow_cable", "equipment_original", "equipment_added"))
