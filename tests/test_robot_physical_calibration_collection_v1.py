from __future__ import annotations

import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]


def test_absent_calibration_template_is_explicit_and_valid() -> None:
    schema = json.loads((ROOT / "contracts/robot_physical_calibration_collection_v1.schema.json").read_text())
    value = json.loads((ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/CALIBRATION_COLLECTION_TEMPLATE.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(value)
    assert value["authority"] == "BLOCKED_EXTERNAL"
    assert value["camera_to_base"]["matrix"] is None
    assert value["tool_to_kaihand_root"]["left"]["matrix"] is None
    assert value["robot_tcp"]["right"]["matrix"] is None
