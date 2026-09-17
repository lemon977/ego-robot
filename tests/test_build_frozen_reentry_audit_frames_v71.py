from __future__ import annotations

import importlib.util
import json
import csv
import subprocess
import sys
from pathlib import Path

import jsonschema


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "src/chaoyang/ops/build_frozen_reentry_audit_frames_v71.py"


def _module():
    spec = importlib.util.spec_from_file_location("frozen_reentry", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_empty_events_only_accepts_bounded_runs():
    mod = _module()
    assert mod._bounded_empty_events([False, True, False, False, True, False]) == [(1, 2, 3, 4)]
    assert mod._bounded_empty_events([True, True, False, False]) == []
    assert mod._bounded_empty_events([False, False, True]) == []


def test_default_build_is_per_instance_development_only():
    mod = _module()
    artifact = mod.build(mod.DEFAULT_PLAN, mod.DEFAULT_SELECTION, mod.DEFAULT_ROLE_LEDGER, mod.DEFAULT_OBJECT_LEDGER)
    schema = json.loads(mod.SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)
    assert artifact["authority"] is False
    assert artifact["gold_accuracy_authorized"] is False
    assert artifact["transition_signal"] == "OBJECT_MASK_OBSERVED_AND_VALID_AND_NONEMPTY"
    assert "not a physical-occlusion label" in artifact["claim_limit"]
    assert {x["task"] for x in artifact["selection_groups"]} == {"chips", "poker"}
    assert any(x["status"] == "FROZEN_EVENT" for x in artifact["rows"])
    assert any(x["status"].startswith("BLOCKED_") for x in artifact["rows"])
    for row in artifact["rows"]:
        if row["task"] == "chips":
            assert row["physical_instance_id"] in {0, 1, 2}
        else:
            assert row["physical_instance_id"] == 0


def test_cli_is_idempotent_and_writes_result(tmp_path: Path):
    cmd = [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)]
    subprocess.run(cmd, check=True, cwd=REPO, capture_output=True, text=True)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    subprocess.run(cmd, check=True, cwd=REPO, capture_output=True, text=True)
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert before == after
    result = json.loads((tmp_path / "RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASSED_DEVELOPMENT_AUDIT_SELECTION"
    assert result["authority"] is False
    with (tmp_path / "FROZEN_REENTRY_AUDIT_FRAMES_V71.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["group_id"].startswith("S1_")
    assert rows[0]["session_id"]
