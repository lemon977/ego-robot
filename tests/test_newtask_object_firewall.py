from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.newtask_object_firewall import (
    ContractSchemaError,
    NewTaskObjectContractError,
    WrongGeometryError,
    authorise_geometry,
    canonical_task,
    inspect_npz_authority,
    reject_legacy_cylinder,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "_run" / "newtask_contract_v2_2"


def load_example(name: str):
    with (CONTRACT_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_example(example, *, evidence_root=None):
    return validate_contract(
        task=example["registry"]["task"],
        measurements=example["measurement"],
        registry=example["registry"],
        timeline=example["timeline"],
        schema_dir=CONTRACT_DIR,
        validation_mode=example["validation_mode"],
        evidence_root=evidence_root,
    )


def make_formal_capture(example, evidence_root):
    value = deepcopy(example)
    value["validation_mode"] = "FORMAL_CAPTURE"
    value["dry_run_only"] = False
    value["claim_limit"] = "FORMAL_CAPTURE_METRIC_AUTHORITY"
    value["measurement"]["validation_mode"] = "FORMAL_CAPTURE"
    value["measurement"]["measurement_status"] = "CAPTURED_METRIC"
    value["registry"]["validation_mode"] = "FORMAL_CAPTURE"
    value["timeline"]["validation_mode"] = "FORMAL_CAPTURE"
    for frame in value["timeline"]["frames"]:
        for state in frame["states"]:
            state["source"] = "MANUAL_ANCHOR"
    for index, measurement in enumerate(value["measurement"]["measurements"]):
        image_format = "JPEG" if index % 2 == 0 else "PNG"
        suffix = ".jpg" if image_format == "JPEG" else ".png"
        relative = f"photos/measurement_{index}{suffix}"
        path = evidence_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (8 + index, 6 + index), (30 + index, 80, 140))
        image.save(path, format=image_format)
        data = path.read_bytes()
        measurement["evidence"] = [{
            "kind": "RULER_PHOTO",
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "provenance": {
                "capture_id": f"capture-{index}",
                "captured_at_utc": "2026-09-01T12:00:00Z",
                "device_id": "pico-field-ruler-camera",
                "measurement_method": "RULER",
                "original_file": True,
                "is_capture_authority": True,
            },
        }]
    return value


@pytest.mark.parametrize(
    ("alias", "expected"),
    [("poker", "POKER"), ("puke", "POKER"), ("chips", "CHIPS"), ("shupian", "CHIPS")],
)
def test_task_aliases(alias, expected):
    assert canonical_task(alias) == expected


def test_unknown_task_fails_closed():
    with pytest.raises(NewTaskObjectContractError):
        canonical_task("can_task")


def test_poker_cylinder_must_raise():
    with pytest.raises(WrongGeometryError):
        authorise_geometry("poker", "card_0", "CYLINDER")


def test_shupian_cylinder_must_raise():
    with pytest.raises(WrongGeometryError):
        authorise_geometry("shupian", "chip_0", "CYLINDER")


def test_nested_legacy_cylinder_field_must_raise():
    with pytest.raises(WrongGeometryError):
        reject_legacy_cylinder({"objects": [{"cylinder_radius_m": 0.024}]})


@pytest.mark.parametrize(
    ("task", "object_id", "geometry", "adapter"),
    [
        ("puke", "card_0", "CARD_THIN_BOX", "box_xyz"),
        ("poker", "rack", "RACK_BOX", "box_xyz"),
        ("shupian", "chip_2", "CHIP_SADDLE", "mask_depth"),
        ("chips", "bowl", "BOWL_REVOLVE", "mask_depth"),
        ("chips", "mat_plane", "SUPPORT_PLANE", "not_rendered_as_operated_object"),
    ],
)
def test_exact_geometry_whitelist(task, object_id, geometry, adapter):
    assert authorise_geometry(task, object_id, geometry) == adapter


def test_poker_example_passes_full_contract():
    result = validate_example(load_example("example_poker.json"))
    assert result["status"] == "PASS"
    assert result["cylinder_authority_accepted"] is False
    assert result["validation_mode"] == "DRY_RUN"
    assert result["authority_status"] == "ILLUSTRATIVE_DRY_RUN_ONLY"
    assert result["claim_limit"] == "ILLUSTRATIVE_ONLY"
    assert result["formal_capture_allowed"] is False
    assert result["decoder_authority"] is None
    assert result["decoded_evidence"] == []
    assert result["manual_ruler_and_object_same_frame_review_required"] is True
    assert result["manual_scene_review_completed_by_machine"] is False
    assert result["object_ids"] == ["card_0", "card_1", "card_2", "mat_plane", "rack", "rack_top_plane"]


def test_chips_example_passes_full_contract():
    result = validate_example(load_example("example_chips.json"))
    assert result["status"] == "PASS"
    assert result["gpu_used"] is False
    assert result["formal_capture_allowed"] is False
    assert result["object_ids"] == ["bowl", "chip_0", "chip_1", "chip_2", "mat_plane"]


def test_registry_missing_required_object_rejected():
    example = load_example("example_poker.json")
    example["registry"]["objects"].pop()
    with pytest.raises(ContractSchemaError, match="registry IDs must be exactly"):
        validate_example(example)


def test_registry_wrong_card_geometry_rejected():
    example = load_example("example_poker.json")
    example["registry"]["objects"][0]["geometry_type"] = "RACK_BOX"
    with pytest.raises(WrongGeometryError, match="requires CARD_THIN_BOX"):
        validate_example(example)


def test_registry_unknown_measurement_reference_rejected():
    example = load_example("example_chips.json")
    example["registry"]["objects"][0]["measurement_id"] = "missing"
    with pytest.raises(ContractSchemaError, match="unknown measurement_id"):
        validate_example(example)


def test_cross_task_extra_measurement_geometry_rejected():
    example = load_example("example_poker.json")
    extra = deepcopy(load_example("example_chips.json")["measurement"]["measurements"][0])
    example["measurement"]["measurements"].append(extra)
    with pytest.raises(ContractSchemaError, match="measurement geometry types must be exactly"):
        validate_example(example)


def test_registry_support_whitelist_is_exact():
    example = load_example("example_chips.json")
    example["registry"]["objects"][0]["allowed_support_ids"].append("rack_top_plane")
    with pytest.raises(ContractSchemaError, match="allowed_support_ids must be"):
        validate_example(example)


def test_duplicate_registry_id_rejected():
    example = load_example("example_chips.json")
    example["registry"]["objects"][1]["object_id"] = "chip_0"
    with pytest.raises(ContractSchemaError, match="duplicate registry object_id"):
        validate_example(example)


def test_every_frame_must_explicitly_state_every_operated_object():
    example = load_example("example_poker.json")
    example["timeline"]["frames"][1]["states"][2]["object_id"] = "card_9"
    with pytest.raises(ContractSchemaError, match="explicitly state all operated objects"):
        validate_example(example)


def test_poker_state_enum_rejects_chip_state():
    example = load_example("example_poker.json")
    example["timeline"]["frames"][1]["states"][0]["state"] = "IN_BOWL"
    with pytest.raises(ContractSchemaError, match="not allowed for POKER"):
        validate_example(example)


def test_on_rack_requires_rack_top_plane():
    example = load_example("example_poker.json")
    example["timeline"]["frames"][0]["states"][0]["support_id"] = "mat_plane"
    with pytest.raises(ContractSchemaError, match="rack_top_plane"):
        validate_example(example)


def test_in_bowl_requires_bowl_support():
    example = load_example("example_chips.json")
    example["timeline"]["frames"][0]["states"][1]["support_id"] = "mat_plane"
    with pytest.raises(ContractSchemaError, match="support_id='bowl'"):
        validate_example(example)


def test_airborne_must_not_inherit_previous_support():
    example = load_example("example_chips.json")
    example["timeline"]["frames"][1]["states"][1]["support_id"] = "bowl"
    with pytest.raises(ContractSchemaError, match="must have null support_id"):
        validate_example(example)


def test_frame_indices_must_be_contiguous():
    example = load_example("example_poker.json")
    example["timeline"]["frames"][1]["frame_index"] = 4
    with pytest.raises(ContractSchemaError, match="contiguous"):
        validate_example(example)


def test_timestamps_must_be_strictly_increasing():
    example = load_example("example_poker.json")
    example["timeline"]["frames"][1]["timestamp_s"] = 0.0
    with pytest.raises(ContractSchemaError, match="strictly increasing"):
        validate_example(example)


def test_nonfinite_timestamp_rejected():
    example = load_example("example_poker.json")
    example["timeline"]["frames"][1]["timestamp_s"] = float("nan")
    with pytest.raises(ContractSchemaError, match="non-finite number"):
        validate_example(example)


def test_frame_count_must_match():
    example = load_example("example_chips.json")
    example["timeline"]["frame_count"] = 9
    with pytest.raises(ContractSchemaError, match="frame_count"):
        validate_example(example)


def test_schema_rejects_unexpected_properties():
    example = load_example("example_poker.json")
    example["measurement"]["measurements"][0]["radius_m"] = 0.02
    with pytest.raises(ContractSchemaError, match="schema violation"):
        validate_example(example)


def test_old_cylinder_npz_rejected_without_pickle(tmp_path):
    path = tmp_path / "old_object_6dof.npz"
    np.savez(path, geometry_type=np.asarray("CYLINDER"), cylinder_radius_m=np.asarray(0.02))
    with pytest.raises(WrongGeometryError):
        inspect_npz_authority("puke", "card_0", path)


def test_bytes_encoded_cylinder_npz_rejected(tmp_path):
    path = tmp_path / "encoded_old_object.npz"
    np.savez(path, geometry_type=np.asarray(b"CYLINDER"))
    with pytest.raises(WrongGeometryError):
        inspect_npz_authority("chips", "chip_0", path)


def test_npz_without_explicit_type_is_not_guessed(tmp_path):
    path = tmp_path / "ambiguous.npz"
    np.savez(path, pose=np.eye(4))
    with pytest.raises(NewTaskObjectContractError, match="guessing is forbidden"):
        inspect_npz_authority("shupian", "chip_0", path)


def test_non_cylinder_npz_metadata_can_be_inspected(tmp_path):
    path = tmp_path / "card.npz"
    np.savez(path, geometry_type=np.asarray("CARD_THIN_BOX"), pose=np.eye(4))
    result = inspect_npz_authority("poker", "card_0", path)
    assert result["geometry_type"] == "CARD_THIN_BOX"
    assert result["object_id"] == "card_0"
    assert "pose" in result["keys"]


def test_validation_does_not_mutate_documents():
    example = load_example("example_chips.json")
    before = deepcopy(example)
    validate_example(example)
    assert example == before


def test_qa_f1_captured_metric_with_illustrative_evidence_is_rejected(tmp_path):
    example = load_example("example_poker.json")
    example["validation_mode"] = "FORMAL_CAPTURE"
    example["measurement"]["validation_mode"] = "FORMAL_CAPTURE"
    example["measurement"]["measurement_status"] = "CAPTURED_METRIC"
    example["registry"]["validation_mode"] = "FORMAL_CAPTURE"
    example["timeline"]["validation_mode"] = "FORMAL_CAPTURE"
    with pytest.raises(ContractSchemaError, match="non-authority evidence"):
        validate_example(example, evidence_root=tmp_path)


@pytest.mark.parametrize("example_name", ["example_poker.json", "example_chips.json"])
def test_formal_capture_propagates_authority_and_claim_limit(example_name, tmp_path):
    example = make_formal_capture(load_example(example_name), tmp_path)
    result = validate_example(example, evidence_root=tmp_path)
    assert result["validation_mode"] == "FORMAL_CAPTURE"
    assert result["authority_status"] == "CAPTURED_METRIC_DECODED_EVIDENCE_VALIDATED"
    assert result["claim_limit"] == "FORMAL_CAPTURE_DECODED_FILE_AUTHORITY_MANUAL_SCENE_REVIEW_REQUIRED"
    assert result["formal_capture_allowed"] is True
    assert result["decoder_authority"]["implementation"] == "Pillow"
    assert result["decoder_authority"]["pinned_version"] == "11.3.0"
    assert result["decoder_authority"]["cpu_only"] is True
    assert all(item["decoded_width_px"] > 0 for item in result["decoded_evidence"])
    assert all(item["decoded_height_px"] > 0 for item in result["decoded_evidence"])
    assert result["manual_ruler_and_object_same_frame_review_required"] is True
    assert result["manual_scene_review_completed_by_machine"] is False


def test_formal_capture_requires_evidence_root(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    with pytest.raises(ContractSchemaError, match="requires evidence_root"):
        validate_example(example)


def test_formal_capture_rejects_sha_mismatch(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    example["measurement"]["measurements"][0]["evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ContractSchemaError, match="sha256 mismatch"):
        validate_example(example, evidence_root=tmp_path)


def test_formal_capture_rejects_missing_sha_field(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    del example["measurement"]["measurements"][0]["evidence"][0]["sha256"]
    with pytest.raises(ContractSchemaError, match="schema violation"):
        validate_example(example, evidence_root=tmp_path)


def test_formal_capture_rejects_symlinked_photo(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    evidence = example["measurement"]["measurements"][0]["evidence"][0]
    path = tmp_path / evidence["path"]
    target = tmp_path / "actual.jpg"
    path.replace(target)
    path.symlink_to(target)
    with pytest.raises(ContractSchemaError, match="cannot use symlinks"):
        validate_example(example, evidence_root=tmp_path)


def test_formal_capture_rejects_invalid_utc_provenance(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    provenance = example["measurement"]["measurements"][0]["evidence"][0]["provenance"]
    provenance["captured_at_utc"] = "not-a-timestamp"
    with pytest.raises(ContractSchemaError, match="explicit UTC timestamp"):
        validate_example(example, evidence_root=tmp_path)


def test_r1_four_byte_jpeg_is_rejected_by_full_decoder(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    evidence = example["measurement"]["measurements"][0]["evidence"][0]
    data = b"\xff\xd8\xff\xd9"
    (tmp_path / evidence["path"]).write_bytes(data)
    evidence["bytes"] = len(data)
    evidence["sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(ContractSchemaError, match="cannot be fully decoded"):
        validate_example(example, evidence_root=tmp_path)


def test_truncated_png_is_rejected_by_full_decoder(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    evidence = example["measurement"]["measurements"][1]["evidence"][0]
    path = tmp_path / evidence["path"]
    data = path.read_bytes()[:24]
    path.write_bytes(data)
    evidence["bytes"] = len(data)
    evidence["sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(ContractSchemaError, match="cannot be fully decoded"):
        validate_example(example, evidence_root=tmp_path)


def test_wrong_extension_rejected_against_decoded_format(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    evidence = example["measurement"]["measurements"][0]["evidence"][0]
    buffer = BytesIO()
    Image.new("RGB", (7, 5), (10, 20, 30)).save(buffer, format="PNG")
    data = buffer.getvalue()
    (tmp_path / evidence["path"]).write_bytes(data)
    evidence["bytes"] = len(data)
    evidence["sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(ContractSchemaError, match="disagrees with decoded PNG"):
        validate_example(example, evidence_root=tmp_path)


def test_normal_jpeg_and_png_return_decoded_dimensions(tmp_path):
    example = make_formal_capture(load_example("example_poker.json"), tmp_path)
    result = validate_example(example, evidence_root=tmp_path)
    by_format = {item["decoded_format"]: item for item in result["decoded_evidence"]}
    assert set(by_format) == {"JPEG", "PNG"}
    assert (by_format["JPEG"]["decoded_width_px"], by_format["JPEG"]["decoded_height_px"]) == (8, 6)
    assert (by_format["PNG"]["decoded_width_px"], by_format["PNG"]["decoded_height_px"]) == (9, 7)


def test_validation_mode_must_match_all_documents():
    example = load_example("example_chips.json")
    example["registry"]["validation_mode"] = "FORMAL_CAPTURE"
    with pytest.raises(ContractSchemaError, match="validation_mode does not match"):
        validate_example(example)


def test_formal_timeline_rejects_dry_run_state_authority(tmp_path):
    example = make_formal_capture(load_example("example_chips.json"), tmp_path)
    example["timeline"]["frames"][0]["states"][0]["source"] = "DRY_RUN_EXAMPLE"
    with pytest.raises(ContractSchemaError, match="cannot use DRY_RUN_EXAMPLE"):
        validate_example(example, evidence_root=tmp_path)


def test_bowl_inner_rim_must_be_smaller_than_outer():
    example = load_example("example_chips.json")
    bowl = example["measurement"]["measurements"][1]
    bowl["inner_rim_diameter_m"] = bowl["outer_rim_diameter_m"]
    with pytest.raises(ContractSchemaError, match="inner rim"):
        validate_example(example)


def test_npz_object_id_and_geometry_pair_is_exact(tmp_path):
    path = tmp_path / "wrong_pair.npz"
    np.savez(path, object_id=np.asarray("card_0"), geometry_type=np.asarray("CARD_THIN_BOX"))
    with pytest.raises(WrongGeometryError, match="requires CHIP_SADDLE"):
        inspect_npz_authority("chips", "chip_0", path)


def test_npz_embedded_object_id_must_match_requested_id(tmp_path):
    path = tmp_path / "wrong_id.npz"
    np.savez(path, object_id=np.asarray("card_1"), geometry_type=np.asarray("CARD_THIN_BOX"))
    with pytest.raises(WrongGeometryError, match="does not match requested"):
        inspect_npz_authority("poker", "card_0", path)
