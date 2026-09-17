from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[1]


def sidecar() -> dict:
    return {
        "schema_version": "robot-visual-sidecar-v1",
        "session_id": "play_cards_0915_001",
        "lane": "ROBOT_VISUAL",
        "status": "PASS",
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
        "calibration_evidence": {
            "robot_tcp": "ABSENT",
            "tool_to_kaihand_root": "PRESENT_CANDIDATE_GEOMETRY",
            "camera_world_to_base": "ABSENT",
        },
        "metrics": {
            "arm_ik": 0.1,
            "kaihand_retarget": 0.2,
            "collision": 0.0,
            "joint_limit": 0.0,
            "velocity": 0.3,
            "acceleration": 0.4,
        },
        "claim_limit": "Development visualization only.",
    }


def validator() -> jsonschema.Draft202012Validator:
    schema = json.loads((ROOT / "contracts/robot_visual_sidecar_v1.schema.json").read_text())
    return jsonschema.Draft202012Validator(schema)


def test_development_robot_sidecar_is_explicitly_non_authoritative() -> None:
    validator().validate(sidecar())


def test_robot_sidecar_cannot_claim_control_or_deployment() -> None:
    for field in ("control_ground_truth", "physical_deployment_authorized"):
        value = copy.deepcopy(sidecar())
        value[field] = True
        with pytest.raises(jsonschema.ValidationError):
            validator().validate(value)


def test_robot_sidecar_cannot_hide_missing_calibration_with_authority() -> None:
    value = copy.deepcopy(sidecar())
    value["calibration_authority"] = "PHYSICAL"
    with pytest.raises(jsonschema.ValidationError):
        validator().validate(value)
