from copy import deepcopy
import json

import pytest

from chaoyang.inpainting_candidate.contracts import file_ref
import chaoyang.inpainting_candidate.source_route as route


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    session = tmp_path / "session"
    camera_path = session / "raw/camera_params.json"
    camera_path.parent.mkdir(parents=True)
    camera_path.write_text(json.dumps({"left": {}, "right": {}, "cameraApi": "raw"}))
    raw_ref = file_ref(camera_path)
    device = {"device_serial": "SERIAL_A", "camera_sha256": raw_ref["sha256"], "eye": "right", "sbs_index": 1,
              "status": "FROZEN_VISUAL_ESTIMATE", "physical_eye_verified": False, "metric_calibration_verified": False,
              "eye_identity_qualification": "VISUAL_ASSUMPTION_SOURCEINDEX_ABSENT", "image_mode": "encoded_crop_resize_only",
              "correction_evidence": {"path": "/NEVER_READ_OLD_FIT_EVIDENCE"},
              "installation_prior_override": {"NEVER_CONSUME": "old_fitted_motion"}, "K_override": [[999]]}
    config = {"schema": "pick_place_1008_device_camera_config_v1", "status": "FROZEN_VISUAL_ESTIMATE",
              "control_ground_truth": False, "training_eligible": False, "devices": {"SERIAL_A": device}}
    config_path = tmp_path / "FITTED_CAMERA_CONFIG.json"
    config_path.write_text(json.dumps(config))
    manifest = {"schema_version": "inpainting-raw-to-robot-adopted-baseline-v1", "baseline_id": "INPAINTING_RAW_TO_ROBOT_V1",
                "selected_baseline_count": 1, "status": "USER_ADOPTED_BATCH_OFFLINE_VISUAL_BASELINE_WITH_KNOWN_LIMITS",
                "actual_motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS", "inputs": {"camera": file_ref(config_path)},
                "NEVER_OPEN_seed_reference": "/NEVER_READ_MANUAL_SEEDS"}
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(route, "CURRENT_V1_MANIFEST", manifest_path)
    report = {"session_dir": str(session), "actual_inputs": {"camera": raw_ref},
              "camera": {"status": "BLOCKED_PHYSICAL_EYE_UNRESOLVED", "eye_routes": {"left": None, "right": None},
                         "camera_device_serial": "SERIAL_A", "physical_eye": None, "sbs_index": None, "projection_qualified": False,
                         "calibration_domain_status": "UNVERIFIED_FACTORY_TO_ENCODED_DOMAIN"},
              "frame_mapping": [{"native_frame_index": 0, "qpc_declared_ns": 123, "physical_eye": None, "sbs_index": None},
                                {"native_frame_index": 1, "qpc_declared_ns": 456, "physical_eye": None, "sbs_index": None}],
              "capabilities": {"physical_eye_crop": "BLOCKED_EYE_ROUTE_UNRESOLVED", "robot_projection": "BLOCKED_CALIBRATION_ABSENT"},
              "control_ground_truth": False, "training_eligible": False}
    return {"report": report, "ref": file_ref(manifest_path), "manifest": manifest,
            "manifest_path": manifest_path, "config": config, "config_path": config_path, "camera_path": camera_path}


def update_config(frozen):
    frozen["config_path"].write_text(json.dumps(frozen["config"]))
    frozen["manifest"]["inputs"]["camera"] = file_ref(frozen["config_path"])
    frozen["manifest_path"].write_text(json.dumps(frozen["manifest"]))
    frozen["ref"] = file_ref(frozen["manifest_path"])


def test_double_matching_adopted_visual_route_is_copy_only(frozen):
    before = deepcopy(frozen["report"])
    result = route.bind_adopted_visual_route(frozen["report"], frozen["ref"])
    assert frozen["report"] == before
    assert result["camera"]["status"] == route.ADOPTED_ROUTE
    assert result["camera"]["physical_eye"] == "right" and result["camera"]["sbs_index"] == 1
    assert result["camera"]["eye_routes"] == {"left": None, "right": None}
    assert result["camera"]["physical_eye_verified"] is False
    assert result["camera"]["projection_qualified"] is False
    assert result["camera"]["metric_calibration_verified"] is False
    assert result["camera"]["adopted_visual_route_ref"] == frozen["manifest"]["inputs"]["camera"]
    assert result["camera"]["adopted_parent_v1_ref"] == frozen["ref"]
    assert result["adopted_route_contracts"] == [frozen["ref"], frozen["manifest"]["inputs"]["camera"]]
    assert [row["native_frame_index"] for row in result["frame_mapping"]] == [0, 1]
    assert [row["qpc_declared_ns"] for row in result["frame_mapping"]] == [123, 456]
    assert all(row["sbs_index"] == 1 and not row["physical_eye_verified"] for row in result["frame_mapping"])
    assert result["capabilities"]["robot_projection"] == "VISUAL_ESTIMATE_NOT_METRIC_QUALIFIED"
    assert result["training_eligible"] is False and result["control_ground_truth"] is False
    encoded = json.dumps(result)
    assert "NEVER_READ_OLD_FIT_EVIDENCE" not in encoded and "NEVER_CONSUME" not in encoded and "K_override" not in encoded


def test_original_sourceindex_declaration_wins_even_if_old_device_matches(frozen):
    camera = frozen["report"]["camera"]
    camera.update({"status": "SOURCE_INDEX_DECLARATION", "physical_eye": "right", "sbs_index": 0,
                   "eye_routes": {"left": 1, "right": 0}})
    original = deepcopy(frozen["report"])
    assert route.bind_adopted_visual_route(original, frozen["ref"]) == dict(original, adopted_route_contracts=[])


@pytest.mark.parametrize("declaration", [{"left": 0, "right": None}, {"left": False, "right": None}, {"left": 2, "right": 2}])
def test_partial_or_invalid_declaration_not_replaced(frozen, declaration):
    frozen["report"]["camera"]["eye_routes"] = declaration
    original = deepcopy(frozen["report"])
    assert route.bind_adopted_visual_route(original, frozen["ref"]) == dict(original, adopted_route_contracts=[])


@pytest.mark.parametrize("serial", [None, "UNKNOWN_0918_DEVICE"])
def test_missing_or_unknown_serial_never_gets_old_route(frozen, serial):
    frozen["report"]["camera"]["camera_device_serial"] = serial
    result = route.bind_adopted_visual_route(frozen["report"], frozen["ref"])
    assert result["camera"]["physical_eye"] is None and result["camera"]["sbs_index"] is None
    assert result["visual_route_adoption"]["status"] == "NOT_APPLIED"


def test_new_camera_sha_same_serial_does_not_use_old_configuration(frozen):
    frozen["camera_path"].write_text(json.dumps({"new_camera_snapshot": True}))
    frozen["report"]["actual_inputs"]["camera"] = file_ref(frozen["camera_path"])
    result = route.bind_adopted_visual_route(frozen["report"], frozen["ref"])
    assert result["camera"]["sbs_index"] is None
    assert result["visual_route_adoption"]["reason"] == "RAW_CAMERA_SHA_NOT_IN_ADOPTED_CONFIG"


def test_actual_camera_drift_cannot_hide_behind_old_report_sha(frozen):
    frozen["camera_path"].write_text("changed-after-source-report")
    with pytest.raises(route.SourceRouteError, match="RAW_CAMERA_CURRENT_SHA_DRIFT"):
        route.bind_adopted_visual_route(frozen["report"], frozen["ref"])


def test_adopted_config_actual_sha_drift_is_error(frozen):
    frozen["config_path"].write_text("changed-after-manifest")
    with pytest.raises(route.SourceRouteError, match="V1_CAMERA_CONFIG_SHA_DRIFT"):
        route.bind_adopted_visual_route(frozen["report"], frozen["ref"])


def test_manifest_sha_drift_is_error(frozen):
    ref = dict(frozen["ref"], sha256="0" * 64)
    with pytest.raises(route.SourceRouteError, match="V1_MANIFEST_SHA_DRIFT"):
        route.bind_adopted_visual_route(frozen["report"], ref)


@pytest.mark.parametrize("field,value", [("baseline_id", "HISTORICAL_V13"), ("selected_baseline_count", 2),
                                         ("actual_motion_source", "OLD_SESSION_ARRAYS")])
def test_current_v1_contract_mismatch_is_error(frozen, field, value):
    frozen["manifest"][field] = value
    frozen["manifest_path"].write_text(json.dumps(frozen["manifest"]))
    with pytest.raises(route.SourceRouteError, match="CURRENT_CONTRACT_MISMATCH"):
        route.bind_adopted_visual_route(frozen["report"], file_ref(frozen["manifest_path"]))


def test_alternative_manifest_not_current_pointer_even_if_same_bytes(frozen, tmp_path):
    clone = tmp_path / "OLD_MANIFEST.json"
    clone.write_bytes(frozen["manifest_path"].read_bytes())
    with pytest.raises(route.SourceRouteError, match="NOT_CURRENT_POINTER"):
        route.bind_adopted_visual_route(frozen["report"], file_ref(clone))


@pytest.mark.parametrize("field,value", [("eye", "left"), ("sbs_index", "1"), ("physical_eye_verified", True),
                                         ("metric_calibration_verified", True)])
def test_frozen_device_contract_cannot_upgrade_claims_or_change_crop(frozen, field, value):
    frozen["config"]["devices"]["SERIAL_A"][field] = value
    update_config(frozen)
    with pytest.raises(route.SourceRouteError, match="V1_DEVICE_ROUTE_CURRENT_CONTRACT_MISMATCH"):
        route.bind_adopted_visual_route(frozen["report"], frozen["ref"])


def test_old_fit_evidence_never_opened(frozen, monkeypatch):
    actual_ref = route.file_ref
    seen = []
    def observe(path):
        seen.append(str(path))
        return actual_ref(path)
    monkeypatch.setattr(route, "file_ref", observe)
    route.bind_adopted_visual_route(frozen["report"], frozen["ref"])
    assert set(seen) == {str(frozen["manifest_path"]), str(frozen["config_path"]), str(frozen["camera_path"])}


def test_metadata_absent_route_remains_unmodified(frozen):
    frozen["report"]["camera"]["status"] = "BLOCKED_CAMERA_METADATA_ABSENT"
    original = deepcopy(frozen["report"])
    assert route.bind_adopted_visual_route(original, frozen["ref"]) == dict(original, adopted_route_contracts=[])


def test_declared_source_does_not_open_adopted_contracts(frozen, monkeypatch):
    frozen["report"]["camera"].update({"status": "SOURCE_INDEX_DECLARATION", "eye_routes": {"left": 1, "right": 0},
                                        "physical_eye": "right", "sbs_index": 0})
    monkeypatch.setattr(route, "_current_manifest", lambda *_: pytest.fail("declared route must not read V1 config"))
    result = route.bind_adopted_visual_route(frozen["report"], {"path": "/not-read"})
    assert result["adopted_route_contracts"] == []
    assert result["camera"]["sbs_index"] == 0


def test_unmatched_camera_still_records_actual_contract_reads(frozen):
    frozen["camera_path"].write_text("new-supported-source-but-not-adopted-camera")
    frozen["report"]["actual_inputs"]["camera"] = file_ref(frozen["camera_path"])
    result = route.bind_adopted_visual_route(frozen["report"], frozen["ref"])
    assert result["adopted_route_contracts"] == [frozen["ref"], frozen["manifest"]["inputs"]["camera"]]
    assert result["camera"]["sbs_index"] is None
