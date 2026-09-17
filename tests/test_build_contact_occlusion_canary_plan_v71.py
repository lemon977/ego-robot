import csv
import json
from pathlib import Path
import subprocess
import sys

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/CONTACT_OCCLUSION_CANARY_PLAN_V71.json"
SCHEMA = ROOT / "contracts/contact_occlusion_canary_plan_v71.schema.json"


def test_published_plan_matches_schema_and_has_three_canaries_per_line() -> None:
    value = json.loads(OUTPUT.read_text(encoding="utf-8"))
    jsonschema.validate(value, json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert [c["canary_id"] for c in value["canaries"]] == ["S1", "S2", "S3", "N1", "N2", "N3"]
    assert sum(c["line"] == "STANDARD_EXACT78" for c in value["canaries"]) == 3
    assert sum(c["line"] == "CONTROLLER_MANUS_SENSOR" for c in value["canaries"]) == 3


def test_missing_uninstalled_challengers_fail_closed() -> None:
    value = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert value["counts"] == {"total": 6, "standard_exact78": 3, "controller_manus_sensor": 3, "ready": 0, "blocked_prereq": 6}
    assert all(c["resource_status"] == "BLOCKED_PREREQ" for c in value["canaries"])
    dependency_statuses = {d["dependency_id"]: d["status"] for c in value["canaries"] for d in c["algorithm_prerequisites"]}
    assert dependency_statuses["CUTIE_CODE_AND_WEIGHTS"] == "MISSING_LOCAL_CODE"
    assert dependency_statuses["FOUNDATIONPOSE_PIN"] == "MISSING_LOCAL_CODE"
    assert dependency_statuses["JOINT_HO_CODE_WEIGHTS"] == "MISSING_LOCAL_CODE"
    assert dependency_statuses["SHADOW_PIPELINE_CODE_ASSETS"] == "MISSING_LOCAL_CODE"


def test_causality_and_authority_limits_are_explicit() -> None:
    value = json.loads(OUTPUT.read_text(encoding="utf-8"))
    by_id = {c["canary_id"]: c for c in value["canaries"]}
    assert by_id["S1"]["execution_mode"] == "CAUSAL_PROCESSING"
    assert by_id["S2"]["execution_mode"] == "DUAL_SEPARATED"
    assert by_id["S3"]["execution_mode"] == "OFFLINE_BIDIRECTIONAL_VISUALIZATION"
    assert "TRACKED_GAP_AS_DIRECT_OBJECT6D" in by_id["S2"]["authority_limit"]["forbidden_claims"]
    assert "CURRENT_BASELINE" in by_id["N3"]["authority_limit"]["forbidden_claims"]
    assert all(c["budget"]["max_rounds"] == 2 for c in value["canaries"])
    assert all(c["budget"]["gpu_seconds_per_round"] <= 7200 for c in value["canaries"])


def test_builder_writes_schema_valid_json_and_csv(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(ROOT / "src/chaoyang/ops/build_contact_occlusion_canary_plan_v71.py"), "--output-dir", str(tmp_path)], check=True)
    value = json.loads((tmp_path / "CONTACT_OCCLUSION_CANARY_PLAN_V71.json").read_text(encoding="utf-8"))
    jsonschema.validate(value, json.loads(SCHEMA.read_text(encoding="utf-8")))
    with (tmp_path / "CONTACT_OCCLUSION_CANARY_PLAN_V71.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["resource_status"] for row in rows} == {"BLOCKED_PREREQ"}
