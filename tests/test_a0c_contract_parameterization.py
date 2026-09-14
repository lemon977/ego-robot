from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from tools.validate_project_contracts import implementation_inventory


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mask_schema_has_no_session_product_or_frame_count_const() -> None:
    schema = json.loads((ROOT / "contracts/mask_evidence.schema.json").read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert "const" not in properties["session_id"]
    assert "const" not in properties["product_line"]
    assert "const" not in properties["frame_count"]
    assert "const" not in properties["model"]["properties"]["weight_path"]
    assert "source_session_digest" in schema["required"]
    assert "product_line_registry_ref" in schema["required"]
    assert "model_pin_ref" in schema["required"]


def test_sam2_local_copy_is_registered_and_byte_identical() -> None:
    profile = yaml.safe_load((ROOT / "contracts/project_profile_v1.yaml").read_text(encoding="utf-8"))
    pipeline = yaml.safe_load((ROOT / "contracts/pipeline_contract_v1.yaml").read_text(encoding="utf-8"))
    dependency = profile["dependencies"]["sam2_1_mask_evidence"]
    registry = pipeline["model_pins"]["SAM2_1_HIERA_LARGE"]
    asset_pin_path = ROOT / dependency["asset_pin_ref"]
    asset_pin = json.loads(asset_pin_path.read_text(encoding="utf-8"))
    local_weight = ROOT / dependency["weight_path"]
    source_weight = Path(asset_pin["source"]["copy_source_weight"])

    assert _sha256(asset_pin_path) == dependency["asset_pin_sha256"] == registry["asset_pin_sha256"]
    assert _sha256(local_weight) == _sha256(source_weight) == dependency["weight_sha256"]
    assert implementation_inventory(ROOT / dependency["implementation_ref"]) == (
        asset_pin["local"]["implementation_inventory_entries"],
        asset_pin["local"]["implementation_regular_bytes"],
        dependency["implementation_inventory_sha256"],
    )
    assert asset_pin["load_smoke"]["parameter_count"] == 224446642


def test_contract_has_no_live_annotation_count_snapshot_and_execution_is_closed() -> None:
    pipeline_text = (ROOT / "contracts/pipeline_contract_v1.yaml").read_text(encoding="utf-8")
    profile_text = (ROOT / "contracts/project_profile_v1.yaml").read_text(encoding="utf-8")
    assert "locked_migrated" not in pipeline_text + profile_text
    assert "remaining_labels" not in pipeline_text + profile_text
    assert "5_OF_45" not in pipeline_text + profile_text

    pipeline = yaml.safe_load(pipeline_text)
    profile = yaml.safe_load(profile_text)
    assert pipeline["calibration_execution_authorized"] is False
    assert pipeline["formal_production_allowed"] is False
    assert pipeline["execution_modes"]["G2_CALIBRATION"]["authorized"] is False
    assert pipeline["execution_modes"]["FORMAL_PRODUCTION"]["authorized"] is False
    assert profile["calibration_execution_ready"] is False
    assert profile["execution_ready"] is False
    assert profile["formal_production_allowed"] is False
    assert profile["dependencies"]["sam3_mask_evidence"]["availability"] == "UNAVAILABLE"

