"""CPU synthetic native adaptation tests; fixture declarations are not receipts."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import h5py

from chaoyang.humanego_ict.data import TrainingDataError
from chaoyang.humanego_ict.derived_data import SOURCE_KEYS, recompute_derived_labels
from chaoyang.humanego_ict.source_adapter import (
    LAYOUT_ROLES, SCHEMA, SOURCE_ROLES, adapt_native_source, array_content_sha256,
    inspect_native_source_metadata,
)


def _ref(path):
    data = path.read_bytes()
    return {"path": str(path.absolute()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return _ref(path)


def _npz(path, arrays):
    encoded = io.BytesIO()
    np.savez(encoded, **arrays)
    path.write_bytes(encoded.getvalue())
    return _ref(path)


@pytest.fixture
def source_fixture(tmp_path):
    count = 55
    qpc = np.arange(count, dtype=np.int64) * 100 + 1000
    controller = np.broadcast_to(np.eye(4), (count, 2, 4, 4)).copy()
    controller[:, :, 0, 3] = np.arange(count)[:, None] / 10
    camera = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    camera[:, 0, 3] = np.arange(count)
    times = qpc.copy()
    times[3] += 5
    joints = np.broadcast_to(np.arange(75).reshape(25, 3) / 1000, (count, 2, 25, 3)).copy()
    arrays = {SOURCE_ROLES["controller"]: controller,
        SOURCE_ROLES["controller_valid"]: np.ones((count, 2), dtype=bool),
        SOURCE_ROLES["hand_joints"]: joints, SOURCE_ROLES["hand_valid"]: np.ones((count, 2), dtype=bool),
        SOURCE_ROLES["qpc"]: qpc, SOURCE_ROLES["frame_id"]: np.arange(count, dtype=np.int64),
        SOURCE_ROLES["controller_source_qpc"]: times,
        SOURCE_ROLES["left_hand_source_qpc"]: times.copy(),
        SOURCE_ROLES["right_hand_source_qpc"]: times.copy(), SOURCE_ROLES["camera"]: camera,
        LAYOUT_ROLES["source_to_canonical"]: np.arange(21, dtype=np.int64),
        LAYOUT_ROLES["canonical_joint_names"]: np.array([f"EXPLICIT_TOY_JOINT_{i}" for i in range(21)]),
        LAYOUT_ROLES["raw_joint_types"]: np.array([f"EXPLICIT_TOY_TYPE_{i}" for i in range(25)]),
        LAYOUT_ROLES["qualification"]: np.array("SYNTHETIC_TOY_NOT_MANUFACTURER_TOPOLOGY"),
        # Pickled trap proves unrelated digital-q members are not materialized.
        "q_arm": np.array([{"do_not_read": "digital_q"}], dtype=object),
    }
    source = _npz(tmp_path / "native.npz", arrays)
    prior = {"schema": "controller_to_wrist_v1", "convention": "wrist_pose = compose_pose(controller_pose, transform)",
        "teleop_wrist": False, "controller_pose_frame": "SYNTHETIC_WORLD",
        "left": {"pos": [.02, .03, .04], "quat": [0, 0, 0, 1]},
        "right": {"pos": [-.02, -.03, -.04], "quat": [0, 0, 0, 1]}}
    calibration = {"control_ground_truth": False,
        "actual_motion_source": "RAW_PICO_CONTROLLER_PLUS_RAW_MANUS_INSTALLATION_PRIOR",
        "world_frame": "SYNTHETIC_WORLD", "installation_prior_used": prior,
        "timebase_status": "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS",
        "physical_eye_identity_verified": False, "training_eligible": False,
        "right_qpc_velocity_violation": 301, "occlusion_order": "UNKNOWN"}
    hdf_path = tmp_path / "original_source.hdf5"
    with h5py.File(hdf_path, "w") as hdf:
        hdf.attrs.update(gate_ms=40.0, max_skew_ms=40.0, max_p95_skew_ms=30.0, alignment_clock="host_qpc")
    hdf_ref = _ref(hdf_path)
    session = {"schema_version": "pick-place-raw-input-v2-canonical21", "session_id": "SYNTHETIC_ONLY",
        "frame_count": count, "arrays": source, "calibration": calibration,
        "source_refs": {"dataset.hdf5": hdf_ref}}
    session_ref = _json(tmp_path / "SESSION_INPUT.json", session)
    segments = {"source_frame_id": np.arange(count, dtype=np.int64), "segment_id": np.zeros(count, dtype=np.int64)}
    segment_ref = _npz(tmp_path / "explicit_segments.npz", segments)
    evidence = _json(tmp_path / "fixture_evidence.json", {"fixture_only": True, "purpose": "EXPLICIT_TOY_SOURCE"})
    contract = {"schema": SCHEMA, "fixture_only": True, "supervision_class": "DERIVED_RETARGET",
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "native_source_sha256": source["sha256"], "session_input_sha256": session_ref["sha256"],
        "source_roles": SOURCE_ROLES, "pose_clock": {"position_unit": "meters", "pose_frame_id": "SYNTHETIC_WORLD",
            "camera_frame_id": "SYNTHETIC_CAMERA", "tcp_frame_ids": ["SYNTHETIC_L_WRIST", "SYNTHETIC_R_WRIST"],
            "timestamp_unit": "nanoseconds", "clock_domain": "SYNTHETIC_CAPTURE_QPC"},
        "source_timebase_status": calibration["timebase_status"], "camera_role": "c2w",
        "hand_joint_position_unit": "meters", "hand_joint_coordinate_frame": "wrist_local",
        "time_gate_source_ref": hdf_ref, "current_max_past_skew_ns": 40_000_000,
        "reference_camera_policy": "ANCHOR_NATIVE_C2W",
        "current_source_selection": "PAST_ONLY_SAME_SEGMENT_LATEST_SOURCE_QPC",
        "target_source_selection": "PRESERVE_NATIVE_NEAREST_WITH_TRACE",
        "installation_prior": prior,
        "hand_layout": {"native_source_sha256": source["sha256"], "roles": LAYOUT_ROLES,
            "array_sha256": {role: array_content_sha256(arrays[key]) for role, key in LAYOUT_ROLES.items()}},
        "segment_binding": {"native_source_sha256": source["sha256"], "source_ref": segment_ref,
            "rule": "EXPLICIT_SYNTHETIC_SINGLE_SEGMENT", "evidence_refs": [evidence]},
        "evidence_refs": [evidence]}
    contract_ref = _json(tmp_path / "adapter_contract.json", contract)
    return {"source": source, "arrays": arrays, "session": session, "session_ref": session_ref,
        "segments": segments, "segment_ref": segment_ref, "contract": contract, "contract_ref": contract_ref,
        "tmp_path": tmp_path}


def _adapt(fixture):
    return adapt_native_source(fixture["source"], fixture["session_ref"], fixture["contract_ref"],
                              anchor_row=3, allow_synthetic_fixture=True)


def _publish_changes(fixture, *, source=False, segments=False):
    if source:
        fixture["source"] = _npz(Path(fixture["source"]["path"]), fixture["arrays"])
        fixture["session"]["arrays"] = fixture["source"]
        fixture["contract"]["native_source_sha256"] = fixture["source"]["sha256"]
        fixture["contract"]["hand_layout"]["native_source_sha256"] = fixture["source"]["sha256"]
        fixture["contract"]["segment_binding"]["native_source_sha256"] = fixture["source"]["sha256"]
    if segments:
        fixture["segment_ref"] = _npz(Path(fixture["segment_ref"]["path"]), fixture["segments"])
        fixture["contract"]["segment_binding"]["source_ref"] = fixture["segment_ref"]
    fixture["session_ref"] = _json(Path(fixture["session_ref"]["path"]), fixture["session"])
    fixture["contract"]["session_input_sha256"] = fixture["session_ref"]["sha256"]
    fixture["contract_ref"] = _json(Path(fixture["contract_ref"]["path"]), fixture["contract"])


def test_metadata_is_json_and_not_qualification(source_fixture):
    f = source_fixture
    result = inspect_native_source_metadata(f["source"], f["session_ref"])
    json.dumps(result)
    assert result["frame_count"] == 55
    assert result["future_nearest_sources"]["controller_source_qpc"] == 1
    assert result["hand_layout"]["qualification"].startswith("SYNTHETIC_TOY")
    assert result["training_start_authorized"] is False


def test_metadata_inspection_does_not_materialize_pose_or_digital_payloads(source_fixture):
    f = source_fixture
    f["arrays"][SOURCE_ROLES["controller"]] = np.array([{"must_not_load": True}], dtype=object)
    _publish_changes(f, source=True)
    result = inspect_native_source_metadata(f["source"], f["session_ref"])
    assert result["array_metadata"][SOURCE_ROLES["controller"]]["dtype"] == "|O"
    assert result["training_start_authorized"] is False
    with pytest.raises(TrainingDataError, match="invalid bounded numeric source"):
        _adapt(f)


def test_fixture_requires_opt_in(source_fixture):
    f = source_fixture
    with pytest.raises(TrainingDataError, match="synthetic"):
        adapt_native_source(f["source"], f["session_ref"], f["contract_ref"], anchor_row=3)


def test_past_current_selection_preserves_future_labels_camera_and_failures(source_fixture):
    f = source_fixture
    result = _adapt(f)
    source, trace = result["source_arrays"], result["trace"]
    assert set(source) == SOURCE_KEYS
    assert source["controller_pose_world"].shape == (51, 2, 4, 4)
    assert source["hand_joints_wrist"].shape == (51, 2, 25, 3)
    assert source["controller_pose_world"][0, 0, 0, 3] == pytest.approx(.2)
    assert source["controller_pose_world"][1, 0, 0, 3] == pytest.approx(.4)
    assert source["reference_camera_to_world"][0, 3] == 3
    assert source["controller_to_wrist"][0, :3, 3].tolist() == [.02, .03, .04]
    assert np.all(source["controller_source_qpc_ns"][0] <= source["qpc_ns"][0])
    assert trace["current_selected_native_rows"] == {"controllers": [2, 2], "hands": [2, 2]}
    assert trace["producer_qualification"]["right_qpc_velocity_violation"] == 301
    assert trace["producer_qualification"]["occlusion_order"] == "UNKNOWN"
    assert trace["original_time_gates"]["max_p95_skew_ms"] == 30.0
    assert "UNKNOWN_SOURCE_TIMEBASE_QUALIFICATION" in result["missing_items"]
    assert not result["training_start_authorized"] and not result["control_ground_truth"]
    assert all(not value.flags.writeable for value in source.values())
    json.dumps(trace)
    repeated = _adapt(f)
    for key in source:
        assert np.array_equal(source[key], repeated["source_arrays"][key])


def test_no_past_current_is_explicit_unknown_not_future(source_fixture):
    f = source_fixture
    for role in ("controller_source_qpc", "left_hand_source_qpc", "right_hand_source_qpc"):
        f["arrays"][SOURCE_ROLES[role]][:4] = 999999
    _publish_changes(f, source=True)
    result = _adapt(f)
    assert not result["source_arrays"]["controller_valid"][0].any()
    assert not result["source_arrays"]["hand_joints_valid"][0].any()
    assert result["trace"]["current_selected_native_rows"]["controllers"] == [-1, -1]
    assert any(item.startswith("UNKNOWN_PAST_CURRENT") for item in result["missing_items"])


@pytest.mark.parametrize("mutation,match", [
    ("future_selection", "selection contract"), ("prior", "substitute"),
    ("layout", "topology role/SHA"), ("clock_unit", "nanoseconds"),
    ("world_frame", "pose/camera/clock"), ("ground_truth", "DERIVED_RETARGET"),
    ("segment_missing", "segment rule/evidence"), ("evidence_sha", "SHA/size drift"),
    ("past_gate", "original HDF numeric gate"), ("hand_unit", "pose/camera/clock"),
])
def test_explicit_contract_is_not_a_boolean_receipt(source_fixture, mutation, match):
    f = source_fixture
    c = f["contract"]
    if mutation == "future_selection": c["current_source_selection"] = "NEAREST"
    elif mutation == "prior":
        c["installation_prior"] = json.loads(json.dumps(c["installation_prior"]))
        c["installation_prior"]["left"]["pos"][0] += .1
    elif mutation == "layout": c["hand_layout"]["array_sha256"]["raw_joint_types"] = "0" * 64
    elif mutation == "clock_unit": c["pose_clock"]["timestamp_unit"] = "seconds"
    elif mutation == "world_frame": c["pose_clock"]["pose_frame_id"] = "ANOTHER_WORLD"
    elif mutation == "ground_truth": c["control_ground_truth"] = True
    elif mutation == "segment_missing": c.pop("segment_binding")
    elif mutation == "evidence_sha": c["evidence_refs"][0]["sha256"] = "0" * 64
    elif mutation == "past_gate": c["current_max_past_skew_ns"] = 100_000_000
    elif mutation == "hand_unit": c["hand_joint_position_unit"] = "millimeters"
    f["contract_ref"] = _json(Path(f["contract_ref"]["path"]), c)
    with pytest.raises(TrainingDataError, match=match):
        _adapt(f)


@pytest.mark.parametrize("mutation,match", [
    ("segment_crossing", "crosses"), ("segment_frame", "identities"),
    ("native_gap", "crosses"), ("bad_controller", "improper.*rotation"),
    ("nonfinite_hand", "nonfinite"), ("negative_source_clock", "nonnegative"),
])
def test_actual_source_structure_is_rechecked(source_fixture, mutation, match):
    f = source_fixture
    if mutation == "segment_crossing":
        f["segments"]["segment_id"][20:] = 1
        _publish_changes(f, segments=True)
    elif mutation == "segment_frame":
        f["segments"]["source_frame_id"][10] += 1
        _publish_changes(f, segments=True)
    else:
        if mutation == "native_gap":
            f["arrays"][SOURCE_ROLES["frame_id"]][20:] += 1
            f["segments"]["source_frame_id"][20:] += 1
        elif mutation == "bad_controller": f["arrays"][SOURCE_ROLES["controller"]][10, 0, 0, 0] = 3
        elif mutation == "nonfinite_hand": f["arrays"][SOURCE_ROLES["hand_joints"]][10, 0, 1, 0] = np.nan
        elif mutation == "negative_source_clock": f["arrays"][SOURCE_ROLES["left_hand_source_qpc"]][10] = -1
        _publish_changes(f, source=True, segments=mutation == "native_gap")
    with pytest.raises(TrainingDataError, match=match):
        _adapt(f)


def test_source_file_sha_drift_is_rejected(source_fixture):
    f = source_fixture
    path = Path(f["source"]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(TrainingDataError, match="SHA/size drift"):
        _adapt(f)


def test_original_hdf_attribute_source_sha_drift_is_rejected(source_fixture):
    f = source_fixture
    path = Path(f["contract"]["time_gate_source_ref"]["path"])
    with h5py.File(path, "r+") as hdf:
        hdf.attrs["gate_ms"] = 80.0
    with pytest.raises(TrainingDataError, match="SHA/size drift"):
        _adapt(f)


def test_gate_is_read_from_actual_hdf_not_contract_boolean(source_fixture):
    f = source_fixture
    path = Path(f["contract"]["time_gate_source_ref"]["path"])
    with h5py.File(path, "r+") as hdf:
        hdf.attrs["gate_ms"] = 80.0
        hdf.attrs["max_skew_ms"] = 80.0
    changed = _ref(path)
    f["contract"]["time_gate_source_ref"] = changed
    f["session"]["source_refs"]["dataset.hdf5"] = changed
    _publish_changes(f)
    with pytest.raises(TrainingDataError, match="original HDF numeric gate"):
        _adapt(f)


def test_segment_id_reuse_is_rejected(source_fixture):
    f = source_fixture
    f["segments"]["segment_id"][:2] = 1
    f["segments"]["segment_id"][3:] = 1
    _publish_changes(f, segments=True)
    with pytest.raises(TrainingDataError, match="reused across disconnected"):
        _adapt(f)


def test_adapter_import_never_imports_torch_or_model():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; import chaoyang.humanego_ict.source_adapter; "
        "assert 'torch' not in sys.modules; assert 'chaoyang.humanego_ict.model' not in sys.modules"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_old_current_source_outside_original_40ms_gate_is_unknown(source_fixture):
    f = source_fixture
    # Spread the anchor clock far beyond all earlier sources without changing
    # original nearest/source observations or claiming physical clock accuracy.
    f["arrays"][SOURCE_ROLES["qpc"]][3:] += 50_000_000
    _publish_changes(f, source=True)
    result = _adapt(f)
    assert not result["source_arrays"]["controller_valid"][0].any()
    assert result["trace"]["original_time_gates"]["gate_ms"] == 40.0
    assert "SELECTED_CURRENT_AGGREGATE_TIME_QUALITY_NOT_QUALIFIED" in result["missing_items"]


def test_current_source_never_crosses_disconnected_segment(source_fixture):
    f = source_fixture
    f["segments"]["segment_id"][:3] = 1
    _publish_changes(f, segments=True)
    result = _adapt(f)
    assert not result["source_arrays"]["controller_valid"][0].any()


def test_output_is_consumable_by_existing_numeric_derived_preparation(source_fixture):
    f = source_fixture
    prepared = _adapt(f)
    adapted_ref = _npz(f["tmp_path"] / "adapted_source.npz", prepared["source_arrays"])
    # Deliberately no layout/mapping calibration: numeric pose labels exist,
    # while all six hand outputs per side remain UNKNOWN, never fabricated.
    hand = _json(f["tmp_path"] / "blocked_hand_contract.json", {
        "schema": "derived-revo2-hand-v1", "supervision_kind": "DERIVED_RETARGET",
        "synthetic_fixture_only": True, "channel_order": ["thumb_bend", "thumb_rotation", "index", "middle", "ring", "pinky"],
        "hand_command_unit": "NORMALIZED_0_1"})
    code = _ref(Path(__file__).resolve().parents[2] / "src/chaoyang/humanego_ict/retarget.py")
    labels = recompute_derived_labels(adapted_ref, hand, 0, retarget_code_ref=code, allow_synthetic_fixture=True)
    assert labels["labels"]["y_action"].shape == (50, 30)
    assert labels["labels"]["action_valid_mask"][:, :18].all()
    assert not labels["labels"]["action_valid_mask"][:, 18:].any()
    assert labels["labels"]["ict_mask"][:2].all()
    assert labels["control_ground_truth"] is False
