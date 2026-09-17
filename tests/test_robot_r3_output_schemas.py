from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v75/attempts"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(schema_name: str, record: dict) -> None:
    schema = load(ROOT / "contracts" / schema_name)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(record)


def test_target_and_reach_diagnostic_proxy_outputs_validate() -> None:
    base = ATTEMPT / "attempt_0017_target_reach_chips025"
    target = load(base / "ROBOT_TARGET_10_RESULT.json")
    reach = load(base / "ROBOT_REACH_20_RESULT.json")
    validate("robot_target_10_r3.schema.json", target)
    validate("robot_reach_20_r3.schema.json", reach)

    bad = copy.deepcopy(target)
    bad["object6d_consumed"] = True
    errors = list(jsonschema.Draft202012Validator(load(ROOT / "contracts/robot_target_10_r3.schema.json")).iter_errors(bad))
    assert errors


def test_profile_does_not_claim_robot_authority() -> None:
    profile = load(ATTEMPT / "attempt_0005_profile/RESULT.json")
    validate("robot_profile_30_r3.schema.json", profile)
    assert profile["robot_tier"] == "NONE"
    assert profile["control_ground_truth"] is False
    assert profile["physical_deployment_authorized"] is False
