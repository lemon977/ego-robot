from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.build_contact_robot_v1_canary_closure import ClosureError, run  # noqa: E402


def test_real_canary_closure_is_fail_closed_and_never_promotes(tmp_path: Path) -> None:
    wave = PROJECT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
    if not wave.exists():
        pytest.skip("Wave0 selection is produced by governance lane")
    output = tmp_path / "RESULT.json"
    result = run(wave, output)
    assert result["status"] == "NO_GO_BLOCKED_PREREQ"
    assert result["authority_promotable"] is False
    assert result["authority_promoted"] is False
    sessions = {item["session_id"]: item for item in result["sessions"]}
    chips = sessions["get_potato_chips_0902_034"]
    assert chips["wave0_member"] is False
    assert "role_mask:TERMINAL_NOT_DOWNSTREAM_AUTHORIZED" in chips["upstream_warnings"]
    assert chips["human_contact_gate"]["status"] == "BLOCKED_PREREQ"
    poker = sessions["play_cards_0902_042"]
    assert poker["wave0_member"] is True
    assert poker["human_contact_gate"]["status"] == "READY_HYPOTHESIS_ONLY_NOT_EXECUTED"
    assert any(
        "EXCLUDES_ROBOT_CONTACT_AUTHORITY" in item
        for item in poker["human_contact_gate"]["claim_limits"]
    )
    assert poker["retarget_4_keyframe_canary"]["solver_executed"] is False
    assert json.loads(output.read_text())["claim_limit"].startswith("Independent")


def test_clean_and_render_only_block_compositor_not_upstream_modules(tmp_path: Path) -> None:
    wave = PROJECT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
    if not wave.exists():
        pytest.skip("Wave0 selection is produced by governance lane")
    result = run(wave, tmp_path / "RESULT.json")
    poker = {item["session_id"]: item for item in result["sessions"]}["play_cards_0902_042"]
    assert poker["human_contact_gate"]["status"] == "READY_HYPOTHESIS_ONLY_NOT_EXECUTED"
    assert poker["retarget_gate"]["requires_clean"] is False
    assert poker["retarget_gate"]["requires_robot_render"] is False
    assert not any(
        "CLEAN" in item or "ROBOT_RENDER" in item
        for item in poker["retarget_gate"]["failures"]
    )
    assert "CURRENT_CLEAN_AUTHORITY_MISSING" in poker["compositor_gate"]["failures"]
    assert "ROBOT_RENDER_NOT_YET_PRODUCED" in poker["compositor_gate"]["failures"]
    assert poker["compositor_gate"]["feeds_back_to_human_contact"] is False
    assert poker["compositor_gate"]["feeds_back_to_retarget"] is False


def test_canary_diagnostic_is_no_clobber(tmp_path: Path) -> None:
    wave = PROJECT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
    if not wave.exists():
        pytest.skip("Wave0 selection is produced by governance lane")
    output = tmp_path / "RESULT.json"
    run(wave, output)
    with pytest.raises(ClosureError, match="refusing to overwrite"):
        run(wave, output)
