"""CPU fixtures only; no actual raw replay/production/model is executed."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from chaoyang.humanego_ict.data import TrainingDataError
from chaoyang.humanego_ict.production_source import (
    ALIASES, CHAINS, PARENTS, SCHEMA, SOURCE_PROFILE, CLOCK_QUALIFICATION,
    CLOCK_TRUST_BOUNDARY, LegacyHandMapper, aggregate_group_plan,
    fixed_anchor_ids, prepare_anchor, replay_session,
)

REPO = Path(__file__).resolve().parents[2]
HAND = REPO / "_run/handoffs/revo2_ict_deploy_completion_v1/codex_20261010T152841Z_2cf21b18/lane_HAND/source/robot_repo"


def ref(path):
    value = Path(path).read_bytes()
    return {"path": str(Path(path).absolute()), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def json_file(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return ref(path)


def fixture_joints():
    value = np.zeros((25, 3))
    value[1:5] = [[-.04, .025, .01], [-.06, .03, .015], [-.07, .015, .02], [-.075, 0, .03]]
    for finger, root in enumerate([5, 10, 15, 20]):
        x = -.02 + finger * .02
        value[root:root+5] = [[x, .015, 0], [x, .045, .002], [x, .065, .018], [x, .071, .037], [x, .065, .054]]
    return value


def source_fixture(directory, sessions=2, count=55):
    records = []
    for number in range(sessions):
        sid = f"{number+1:03d}"; capture = "pick_place_1008_" + sid
        root = directory / sid; root.mkdir()
        ts = 100_000_000_000 + number * 10_000_000_000 + np.arange(count) * 16_000_000
        identity = {"pos": [0., 0., 0.], "quat": [0., 0., 0., 1.]}
        prior = {"schema": "controller_to_wrist_v1", "convention": "wrist_pose = compose_pose(controller_pose, transform)",
                 "controller_pose_frame": "pico_world_rh_x_forward_y_left_z_up", "left": identity, "right": identity}
        with h5py.File(root / "prior.h5", "w") as hdf:
            hdf.attrs["controller_to_wrist_calibration"] = json.dumps(prior)
            for k, v in {"gate_ms": 40., "max_skew_ms": 40., "max_p95_skew_ms": 30., "min_hand_coverage": .95, "min_complete_coverage": .95}.items():
                hdf.attrs[k] = v
        refs = {"hdf5_prior_only": ref(root / "prior.h5")}
        manifest = {"session": capture, "clocks": {"align_clock": "single capture-host monotonic/QPC domain; never mixed", "align_key": "recv_qpc_ns preferred; whole-session recv_wall_ns fallback"},
            "coordinates": {"training_frame": "right-handed X-forward Y-left Z-up meters", "manus": "native RH wrist-local (HandMotion_None); world wrist = pico_ctrl ∘ T_calib"}}
        refs["source_manifest"] = json_file(root / "manifest.json", manifest)
        refs["selection"] = json_file(root / "selection.json", {"schema": "capture_selection_v1", "session": capture, "captured_at": f"2026-10-08T00:00:{number:02d}Z"})
        refs["workflow"] = json_file(root / "workflow.json", {"schema": "capture_workflow_v1", "session": capture, "outcome": "exported"})
        refs["manus_meta"] = json_file(root / "manus.meta.json", {"session": capture, "calibration": {s: {"glove_id": i+1, "sdk_applied": True} for i, s in enumerate(("left", "right"))}})
        for role in ("raw_vst",):
            refs[role] = json_file(root / (role + ".json"), {"synthetic_fixture_only": True, "session": capture})
        refs["camera_meta"] = json_file(root / "camera_meta.json", {"device_serial": "FIXTURE_PICO"})
        refs["camera_params"] = json_file(root / "camera_params.json", {"width": 2048, "height": 1536,
            "right": {"fx": 800., "fy": 800., "cx": 1024., "cy": 768., "extrinsic": np.eye(4)[:3].reshape(-1).tolist()}})
        pico, manus = [], []
        for frame, tick in enumerate(ts):
            for offset in (-2_000_000, 1_000_000):
                pose = {"pos": [float(frame), float(offset > 0), 0.], "quat": [0., 0., 0., 1.], "status": 3}
                pico.append({"recv_qpc_ns": int(tick+offset), "dev": "FIXTURE_PICO", "data": {"Head": pose, "Controller": {"left": pose, "right": pose}}})
            for hand, side in enumerate(("left", "right")):
                nodes = np.column_stack((fixture_joints(), np.tile([0., 0., 0., 1.], (25, 1))))
                # Independent literal of the original capture protocol: do not
                # construct the expected25 topology from the tested constant.
                raw_parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23]
                manus.append({"type": "manus_frame", "recv_qpc_ns": int(tick-3_000_000), "side": side, "glove_id": hand+1,
                    "node_count": 25, "node_ids": list(range(25)), "parent_ids": raw_parents, "joint_types": list(ALIASES), "chain_types": list(CHAINS),
                    "calibration_applied": True, "nodes": nodes.tolist()})
        for role, values in (("pico", pico), ("manus", manus), ("video_qpc", ts.tolist()), ("video_wall", (ts+1_000_000_000_000).tolist())):
            path = root / (role + ".jsonl")
            path.write_text("\n".join(json.dumps(v) for v in values) + "\n")
            refs[role] = ref(path)
        record = {"session_id": sid, "device_serial": "FIXTURE_PICO", "raw_frame_count": count, "source_refs": refs,
            "reference_eye_size": [2048, 1536], "camera": {"right": {"K": [[800., 0., 1024.], [0., 800., 768.], [0., 0., 1.]], "extrinsic_4x4": np.eye(4).tolist()}}}
        records.append(record)
    original = {"fixture_only": True, "expected_sessions": sessions, "sessions": records,
        "time_policy": {"source_clock_precision": "recorded_nanoseconds", "timestamp_semantics": "software_stream_receipt_not_exposure_sync"}}
    input_ref = json_file(directory / "input.json", original)
    contract = {"schema": SCHEMA, "source_profile": SOURCE_PROFILE, "clock_qualification": CLOCK_QUALIFICATION,
        "clock_trust_boundary": CLOCK_TRUST_BOUNDARY, "timestamp_unit": "nanoseconds", "timestamp_semantics": "HOST_ARRIVAL_RECEIVE_TIME_NOT_EXPOSURE_SYNCHRONIZATION",
        "implementation_verification": "UNKNOWN_NOT_CLAIMED", "clock_domain": "SYNTHETIC_SINGLE_HOST_NS", "control_ground_truth": False,
        "physical_deployment_authorized": False, "input_manifest_ref": input_ref,
        "camera": {"eye": "right", "sbs_index": 1, "output_size": [640, 480], "upright_axes": "capture_sensor_from_upright", "authority": "FACTORY_METADATA_PRIOR_NOT_PHYSICAL_QUALIFICATION"}}
    return {"records": records, "contract": contract,
        "mapper_ref": ref(HAND / "src/teleop/manus_revo2_retarget.py"), "calibration_ref": ref(HAND / "config/manus_revo2_calibration.json")}


def replay_fixture(value, number=0):
    return replay_session(value["records"][number], mapper_ref=value["mapper_ref"], calibration_ref=value["calibration_ref"], source_contract=value["contract"], allow_synthetic_fixture=True)


def nested_camera_fixture(value, *, change=None):
    """Literal original008 schema; not generated from the parser constants."""
    record = value["records"][0]
    parameters = {"version": "1.1", "reference_resolution": {"w": 2048, "h": 1536},
        "right": {"intrinsics": {"fx": 800., "fy": 800., "cx": 1024., "cy": 768.}},
        "extrinsics": {"right": [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]}}
    if change:
        change(parameters)
    record["camera_schema"] = "nested_4x4"
    record["source_refs"]["camera_params"] = json_file(record["source_refs"]["camera_params"]["path"], parameters)
    record["source_refs"]["camera_meta"] = json_file(record["source_refs"]["camera_meta"]["path"],
        {"device_serial": "FIXTURE_PICO", "contains": ["intrinsics", "distortion"], "extrinsics_included": False})
    path = value["contract"]["input_manifest_ref"]["path"]
    original = json.loads(Path(path).read_text()); original["sessions"] = value["records"]
    value["contract"]["input_manifest_ref"] = json_file(path, original)
    return value


def test_flat_and_literal_nested_factory_camera_same_values_and_conflict_preserved(tmp_path):
    value = source_fixture(tmp_path)
    flat = replay_fixture(value)
    nested = replay_fixture(nested_camera_fixture(value))
    assert flat["metadata"]["camera_schema"] == "flat_3x4"
    assert nested["metadata"]["camera_schema"] == "nested_4x4"
    assert np.array_equal(flat["arrays"]["K"], nested["arrays"]["K"])
    assert np.array_equal(flat["arrays"]["head_from_camera"], nested["arrays"]["head_from_camera"])
    assert nested["metadata"]["physical_calibration_status"] == "UNKNOWN_NOT_GRANTED"
    assert prepare_anchor(nested, 0)["trace"]["camera_metadata_conflicts"] == [
        "SNAPSHOT_META_EXTRINSICS_FALSE_BUT_FACTORY_JSON_CONTAINS_MATRIX"]


@pytest.mark.parametrize("field", ["intrinsics", "extrinsics", "shape", "resolution"])
def test_nested_factory_camera_tamper_or_invalid_shape_rejected(tmp_path, field):
    def change(parameters):
        if field == "intrinsics":
            parameters["right"]["intrinsics"]["fx"] += 1
        elif field == "extrinsics":
            parameters["extrinsics"]["right"][0][3] = .1
        elif field == "shape":
            parameters["extrinsics"]["right"] = parameters["extrinsics"]["right"][:3]
        else:
            parameters["reference_resolution"]["w"] = 1024
    value = nested_camera_fixture(source_fixture(tmp_path), change=change)
    with pytest.raises(TrainingDataError, match="camera"):
        replay_fixture(value)


def test_past_head_and_hand_and_nearest_future_target_separated(tmp_path):
    replay = replay_fixture(source_fixture(tmp_path))
    assert fixed_anchor_ids(replay) == [0]
    result = prepare_anchor(replay, 0)
    assert result["eligible"] and result["labels"]["y_action"].shape == (50, 30)
    assert result["labels"]["action_valid_mask"].all()
    assert result["current"]["head_source_qpc_ns"] < result["current"]["anchor_qpc_ns"]
    assert (result["current"]["manus_source_qpc_ns"] <= result["current"]["anchor_qpc_ns"]).all()
    assert (replay["arrays"]["target_pico_qpc_ns"] > replay["arrays"]["qpc_ns"]).all()
    assert np.allclose(result["current"]["c2w"], replay["arrays"]["past_head"][0] @ replay["arrays"]["head_from_camera"])
    assert result["current"]["hands_world"].shape == (2, 21, 3)
    assert np.array_equal(result["labels"]["ict_mask"], [True, True, False, False, False, False, False, False])


def test_already_local_no_second_quaternion_canonicalization(tmp_path):
    replay = replay_fixture(source_fixture(tmp_path))
    assert np.array_equal(replay["arrays"]["past_hands"][0, 0], fixture_joints())
    assert replay["metadata"]["implementation_verification"] == "UNKNOWN_NOT_CLAIMED"


def test_group_proof_and_fixed_not_quality_selected_split(tmp_path):
    value = source_fixture(tmp_path)
    metadata = [replay_fixture(value, i)["metadata"] for i in range(2)]
    plan = aggregate_group_plan(metadata[::-1])
    assert plan["001"]["split"] == "validation" and plan["002"]["split"] == "train"
    metadata[1]["capture_interval_ns"] = metadata[0]["capture_interval_ns"]
    with pytest.raises(TrainingDataError, match="overlap"):
        aggregate_group_plan(metadata)


@pytest.mark.parametrize("field,value", [("source_profile", "Core3_Auto"), ("timestamp_unit", "ticks"), ("implementation_verification", "PASSED"), ("control_ground_truth", True)])
def test_wrong_profile_clock_or_gt_rejected(tmp_path, field, value):
    fixture = source_fixture(tmp_path); fixture["contract"][field] = value
    with pytest.raises(TrainingDataError):
        replay_fixture(fixture)


def test_synthetic_evidence_requires_explicit_flag(tmp_path):
    value = source_fixture(tmp_path)
    with pytest.raises(TrainingDataError, match="synthetic raw"):
        replay_session(value["records"][0], mapper_ref=value["mapper_ref"], calibration_ref=value["calibration_ref"], source_contract=value["contract"])


def test_degenerate_zero_fill_cannot_be_observed_hand_command(tmp_path):
    fixture = source_fixture(tmp_path)
    mapper = LegacyHandMapper(fixture["mapper_ref"], fixture["calibration_ref"])
    values, valid = mapper(np.zeros((25, 3)), "left")
    assert not valid.any() and np.array_equal(values, np.zeros(6))


def test_not_stride32_or_incomplete_h50_rejected(tmp_path):
    replay = replay_fixture(source_fixture(tmp_path))
    for frame in (1, 32, -1):
        with pytest.raises(TrainingDataError, match="fixed native stride32"):
            prepare_anchor(replay, frame)


def test_preserved_original_target_time_fail_and_mask(tmp_path):
    replay = replay_fixture(source_fixture(tmp_path))
    arrays = dict(replay["arrays"]); timestamps = arrays["target_hand_qpc_ns"].copy(); timestamps[30, 1] += 50_000_000
    arrays["target_hand_qpc_ns"] = timestamps
    invalid = dict(replay, arrays=arrays)
    result = prepare_anchor(invalid, 0)
    assert not result["eligible"] and "BLOCKED_TARGET_ORIGINAL_TIME_GATE:manus_right" in result["missing_items"]


def test_frozen_mapper_drift_rejected(tmp_path):
    fixture = source_fixture(tmp_path)
    drift = dict(fixture["mapper_ref"], sha256="0"*64)
    with pytest.raises(TrainingDataError, match="SHA"):
        LegacyHandMapper(drift, fixture["calibration_ref"])


def test_shared_original_bytes_are_not_independent_groups(tmp_path):
    fixture = source_fixture(tmp_path)
    metadata = [replay_fixture(fixture, i)["metadata"] for i in range(2)]
    metadata[1]["recording_sha256"]["raw_vst"] = metadata[0]["recording_sha256"]["raw_vst"]
    with pytest.raises(TrainingDataError, match="shared original"):
        aggregate_group_plan(metadata)


def test_actual25_parent_contract_not26_self_consistent_fixture():
    from chaoyang.humanego_ict.production_source import _topology
    parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23]
    assert len(PARENTS) == 25 and tuple(parents) == PARENTS
    row = {"node_count": 25, "node_ids": list(range(25)), "parent_ids": parents,
        "chain_types": ["Hand"] + ["Thumb"]*4 + ["Index"]*5 + ["Middle"]*5 + ["Ring"]*5 + ["Pinky"]*5,
        "joint_types": ["Invalid", "MCP", "PIP", "DIP", "TIP"] + ["MCP", "PIP", "IP", "DIP", "TIP"]*4}
    _topology(row)
    row["parent_ids"] = parents[:15] + [14] + parents[15:]
    with pytest.raises(TrainingDataError, match="TOPOLOGY"):
        _topology(row)


def test_original_wrapper_is_manus_frame_and_unknown_wrapper_rejected(tmp_path):
    value = source_fixture(tmp_path)
    record = value["records"][0]; path = record["source_refs"]["manus"]["path"]
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert all(row["type"] == "manus_frame" for row in rows)
    rows[0]["type"] = "raw_skeleton"
    Path(path).write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    record["source_refs"]["manus"] = ref(path)
    input_path = value["contract"]["input_manifest_ref"]["path"]
    original = json.loads(Path(input_path).read_text()); original["sessions"] = value["records"]
    value["contract"]["input_manifest_ref"] = json_file(input_path, original)
    with pytest.raises(TrainingDataError, match="profile/glove"):
        replay_fixture(value)
