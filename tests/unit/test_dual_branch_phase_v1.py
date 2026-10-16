"""Selection-only regression; no models or new media."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from chaoyang.governance import dual_branch_phase_v1 as phase

ROOT = Path(__file__).resolve().parents[2]


def test_live_aliases_keep_parent_versions_and_qualification_scopes():
    value = phase.validate_phase(ROOT)
    assert set(value["aliases"]) == {"EXACT78_PHASE_V1", "MANUS_PHASE_V1"}
    visual, manus = value["aliases"].values()
    assert visual["parent_implementation"] == "takeover_v13"
    assert visual["validated_scope"]["total_sessions"] == 234
    assert visual["validated_scope"]["total_frames"] == 78989
    assert visual["pure_rgb"] is False
    assert manus["parent_implementation"] == "robot100_v16"
    assert manus["validated_scope"]["frames"] == 176
    assert manus["validated_scope"]["batch_100_validated"] is False
    assert manus["image_alignment_status"] == "VISIBLE_MISMATCH_NOT_ACCEPTED"
    assert all(row["training_eligible"] is False and row["control_ground_truth"] is False for row in value["aliases"].values())


@pytest.mark.parametrize("name", ["EXACT78_PHASE_V1", "MANUS_PHASE_V1"])
def test_explicit_alias_resolution(name):
    assert phase.resolve_alias(ROOT, name)["pointer"]["sha256"] in phase.PINS.values()


@pytest.mark.parametrize("name", ["latest", "takeover_v12", "Cards042_V2", "P1_retry12", "HuRo", "RuCo", "0915", "0918"])
def test_history_cannot_be_selected_as_a_phase_alias(name):
    with pytest.raises(RuntimeError, match="NOT_A_PHASE_BASELINE"):
        phase.resolve_alias(ROOT, name)


def test_parent_sha_drift_is_rejected(tmp_path):
    name = next(iter(phase.PINS))
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="PHASE_PARENT_DRIFT"):
        phase.build_phase(tmp_path)


def test_manifest_cannot_silently_promote_quality(tmp_path, monkeypatch):
    expected = phase.build_phase(ROOT)
    modified = copy.deepcopy(expected)
    modified["aliases"]["MANUS_PHASE_V1"]["validated_scope"]["batch_100_validated"] = True
    path = tmp_path / phase.PHASE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(modified))
    monkeypatch.setattr(phase, "build_phase", lambda root: expected)
    with pytest.raises(RuntimeError, match="MANIFEST_DRIFT"):
        phase.validate_phase(tmp_path)


def test_dependencies_are_not_selections():
    value = phase.validate_phase(ROOT)
    rows = value["history_policy"]["dependency_only_examples"]
    assert len(rows) == 5
    assert all(row["selection_allowed"] is False for row in rows)
    assert sum("huro_hand" in row["artifact"]["path"] for row in rows) == 2
    assert value["execution_policy"]["alias_is_execution_authorization"] is False
    assert value["history_policy"]["legacy_autopilot_execution_allowed"] is False


@pytest.mark.parametrize("command", [
    [sys.executable, "-B", "scripts/takeover/autopilot.py", "--once"],
    [sys.executable, "-B", "scripts/takeover/autopilot.py", "--dry-run", "--once"],
    ["bash", "scripts/takeover/autopilot_start.sh"],
])
def test_legacy_autopilot_entry_refuses_before_writing(command):
    names = ["docs/plans/EXACT78_TAKEOVER_20261004/AUTOPILOT_PLAN.json",
             "docs/plans/EXACT78_TAKEOVER_20261004/AUTOPILOT_STATE.json",
             "docs/plans/EXACT78_TAKEOVER_20261004/AUTOPILOT_STATUS_ZH.md",
             "_run/caches/takeover_autopilot/autopilot.pid"]
    before = {name: (ROOT / name).read_bytes() for name in names}
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 64
    assert "LEGACY_AUTOPILOT_RETIRED" in result.stderr
    assert before == {name: (ROOT / name).read_bytes() for name in names}


def test_phase_contract_does_not_authorize_execution_or_training():
    value = phase.phase_contract(ROOT)["dual_branch_phase_baselines_v1"]
    assert value["task_packet_required"] is True
    assert value["training_steps_authorized"] == 0
    assert value["selected_alias_count"] == 2
