"""Synthetic CPU-only checks; no real capture inference or retarget calls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import motion_source as m
from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref
from chaoyang.inpainting_candidate.source import RAW_FILES

ORIGINAL_INSTALLATION_READER = m.parent.read_installation
ORIGINAL_MOTION_READER = m.parent.load_raw_motion


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return file_ref(path)


@pytest.fixture
def case(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(m.PROJECT_ROOT)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "V1_MANIFEST", tmp_path / "inpainting_v1/MANIFEST.json")
    root = tmp_path / "raw_captures/chips_highview/001"
    root.mkdir(parents=True)
    refs = {}
    for role, relative in RAW_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic raw input\n")
        refs[role] = file_ref(path)
    refs["manifest"] = write_json(root / RAW_FILES["manifest"], {"session": "chips_0918_001"})
    refs["camera"] = write_json(root / RAW_FILES["camera"], {"synthetic": "camera"})
    refs["camera_meta"] = write_json(root / RAW_FILES["camera_meta"], {
        "device_serial": "UNKNOWN_DEVICE", "sha256": refs["camera"]["sha256"]})
    qpc = [1_000_000_000, 1_040_000_000, 1_080_000_000]
    for role, values in (("video_qpc", qpc), ("video_wall", [2_000_000_000 + i * 40_000_000 for i in range(3)])):
        path = root / RAW_FILES[role]
        path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")
        refs[role] = file_ref(path)
    report = {"status": "SOURCE_PREFLIGHT_CANDIDATE", "session_dir": str(root),
        "session_id": "chips_0918_001", "native_sidecar_frames": 3,
        "timebase_status": "TIMEBASE_UNVERIFIED", "actual_inputs": refs,
        "camera": {"physical_eye": "right", "sbs_index": 1},
        "frame_mapping": [{"native_frame_index": i, "qpc_declared_ns": value} for i, value in enumerate(qpc)]}
    prior = {side: {"pos": [0., 0., 0.], "quat": [0., 0., 0., 1.]} for side in ("left", "right")}
    prior.update(schema="controller_to_wrist_v1",
        convention="wrist_pose = compose_pose(controller_pose, transform)",
        controller_pose_frame="pico_world_rh_x_forward_y_left_z_up")
    installation = {"source_hdf5": refs["installation_container"],
        "attribute": "controller_to_wrist_calibration", "prior_sha256": canonical_sha(prior),
        "prior_authority": "INSTALLATION_PRIOR_NOT_MEASURED_ANATOMICAL_WRIST"}
    native = {
        "head": np.tile(np.eye(4), (3, 1, 1)), "head_display": np.tile(np.eye(4), (3, 1, 1)),
        "head_source": np.array(["RAW_VALID"] * 3), "head_valid": np.ones(3, bool),
        "controller": np.tile(np.eye(4), (3, 2, 1, 1)), "wrist": np.tile(np.eye(4), (3, 2, 1, 1)),
        "manus_local_25": np.zeros((3, 2, 25, 3)), "hands": np.zeros((2, 3, 21, 3)),
        "hand_source": np.full((2, 3), "RAW_VALID"), "controller_valid": np.ones((3, 2), bool),
        "manus_valid": np.ones((3, 2), bool), "hand_observed": np.ones((2, 3), bool),
        "hand_display_valid": np.ones((2, 3), bool), "complete_valid": np.ones(3, bool),
        "joint_observed_local": np.ones((3, 2, 21), bool),
    }
    calls = []

    def fake_reader(directory, times, *, expected_serial, prior):
        calls.append({"directory": directory, "qpc": times.tolist(),
                      "serial": expected_serial, "prior": deepcopy(prior)})
        return deepcopy(native), {"frame_id": np.arange(3), "video_qpc_ns": np.asarray(qpc)}, {
            "passed": False, "complete_coverage": 1.0}

    monkeypatch.setattr(m.parent, "read_installation", lambda path: (deepcopy(prior), deepcopy(installation)))
    monkeypatch.setattr(m.parent, "load_raw_motion", fake_reader)
    monkeypatch.setattr(m.parent, "joint_mapping_metadata", lambda value: {"synthetic_topology": True})
    normalization_calls = []

    def fake_camera(params, meta, **kwargs):
        normalization_calls.append(kwargs)
        return SimpleNamespace(width=1280, height=960, K=np.diag([1000., 1000., 1.]),
            T_head_camera=np.eye(4), T_head_camera_factory=np.eye(4), calibration={})

    monkeypatch.setattr(m, "normalize_camera", fake_camera)
    for relative in m.PARENT_CODE:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic frozen parent dependency\n")
    return SimpleNamespace(root=root, refs=refs, report=report, prior=prior,
        installation=installation, native=native, calls=calls,
        normalization_calls=normalization_calls, output=tmp_path / "candidate_output/motion")


def adopt(case, *, serial="DEVICE_A", override=False, default=False):
    case.refs["camera_meta"] = write_json(case.root / RAW_FILES["camera_meta"], {
        "device_serial": serial, "sha256": case.refs["camera"]["sha256"]})
    config = {"device_serial": serial, "camera_sha256": case.refs["camera"]["sha256"],
        "status": "FROZEN_VISUAL_ESTIMATE", "output_size": [1280, 960],
        "upright_axes": "capture_sensor_from_upright", "eye": "right", "sbs_index": 1,
        "K_override": [[1000., 0., 640.], [0., 1000., 480.], [0., 0., 1.]]}
    config["correction_evidence"] = write_json(m.PROJECT_ROOT / "adopted/K_REPORT.json", {
        "status": "PASSED", "qualification": "already adopted visual metadata"})
    if override:
        config.update(installation_prior_override=deepcopy(case.prior),
            installation_original_prior_sha256=canonical_sha(case.prior),
            installation_group_key=serial + ":" + canonical_sha(case.prior), fit_quality_status="PASSED")
        config["installation_prior_override"]["right"]["pos"][0] = .02
        config["installation_correction_evidence"] = write_json(m.PROJECT_ROOT / "adopted/FIT_REPORT.json", {
            "schema": "pick_place_fixed_encoded_k_right_installation_visual_fit_v1", "status": "PASSED",
            "groups": {config["installation_group_key"]: {"status": "PASSED", "device_serial": serial,
                "original_prior_sha256": canonical_sha(case.prior), "prior_before": deepcopy(case.prior),
                "prior_used": deepcopy(config["installation_prior_override"])}}})
    config_ref = write_json(m.PROJECT_ROOT / "adopted/FITTED_CAMERA_CONFIG.json", {
        "devices": {} if default else {serial: config}, "default_device": serial,
        "defaults": config if default else {}})
    manifest_ref = write_json(m.V1_MANIFEST, {"baseline_id": "INPAINTING_RAW_TO_ROBOT_V1",
        "inputs": {"camera": config_ref}})
    return manifest_ref, config


def prepare(case, **kwargs):
    return m.prepare_motion_source(case.report, case.output,
        component_signature="a" * 64, **kwargs)


def test_unknown_device_keeps_native_audit_and_independent_mask_clean(case):
    result = prepare(case)
    assert result["raw_session"] is None
    assert result["report"]["robot_consumer_status"] == "BLOCKED_CAMERA_AXES"
    assert result["report"]["mask_clean_consumer_status"] == "READY_INDEPENDENT_OF_CAMERA"
    assert result["receipt"]["status"] == "SEALED_COMPLETE"
    assert result["receipt"]["session_id"] == "chips_0918_001"
    assert case.calls[0]["serial"] == "UNKNOWN_DEVICE"
    assert case.calls[0]["prior"] == case.prior
    assert not case.normalization_calls
    assert all(not result["report"][key] for key in ("hdf5_datasets_read", "historical_motion_arrays_read", "hardware_retarget_fields_read"))
    assert result["receipt"]["retarget_calls"] == result["receipt"]["model_calls"] == 0
    with np.load(case.output / "NATIVE_MOTION.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["native__hands"], case.native["hands"])
        assert "K" not in arrays
    rows = json.loads((case.output / "FRAME_STATUS.json").read_text())
    assert len(rows) == 3
    assert all(row["acquisition_frame_id"] == "UNKNOWN" and not row["projection_qualified"] for row in rows)
    assert all(not row["training_eligible"] for row in rows)


def test_adopted_non_override_device_consumes_current_raw_prior_and_1280k_once(case):
    manifest, _ = adopt(case)
    result = prepare(case, v1_manifest_ref=manifest)
    session = result["raw_session"]
    assert session.session_id == "chips_0918_001"
    assert session.width == 1280 and session.height == 960
    assert session.fps == 25
    assert session.raw_dir == case.output
    assert session.video == Path(case.refs["video"]["path"])
    assert result["report"]["camera_binding"]["prior_binding_status"] == "NO_FITTED_OVERRIDE_RAW_PRIOR_CONSUMED"
    assert result["report"]["camera_binding"]["adopted_domain"]
    assert not result["report"]["metric_calibration_verified"]
    assert case.normalization_calls[0]["output_size"] == (1280, 960)
    assert case.normalization_calls[0]["config"]["K_override"][0][0] == 1000
    assert case.calls[0]["prior"] == case.prior
    assert len(result["receipt"]["actual_inputs"]) >= len(RAW_FILES) + 6


@pytest.mark.parametrize("change,reason", [
    ("serial", "DEVICE_NOT_IN_ADOPTED_V1_DOMAIN"),
    ("snapshot", "CAMERA_SNAPSHOT_NOT_IN_ADOPTED_V1_DOMAIN"),
    ("prior", "FITTED_INSTALLATION_ORIGINAL_PRIOR_SHA_MISMATCH"),
    ("group", "FITTED_INSTALLATION_DEVICE_GROUP_MISMATCH"),
    ("axes", "ADOPTED_ENCODED_CAMERA_AXES_ABSENT"),
    ("eye", "ADOPTED_CAMERA_EYE_ROUTE_INVALID"),
    ("route", "RAW_RGB_AND_ADOPTED_CAMERA_EYE_ROUTE_MISMATCH"),
])
def test_no_cross_device_snapshot_prior_or_eye_fallback(case, change, reason):
    manifest, config = adopt(case, override=True)
    if change == "serial":
        case.refs["camera_meta"] = write_json(case.root / RAW_FILES["camera_meta"], {
            "device_serial": "DEVICE_NEW", "sha256": case.refs["camera"]["sha256"]})
    elif change == "route":
        case.report["camera"]["sbs_index"] = 0
    else:
        key = {"snapshot": "camera_sha256", "prior": "installation_original_prior_sha256",
            "group": "installation_group_key", "axes": "upright_axes", "eye": "eye"}[change]
        config[key] = "wrong"
        config_ref = write_json(m.PROJECT_ROOT / "adopted/FITTED_CAMERA_CONFIG.json", {"devices": {"DEVICE_A": config}})
        manifest = write_json(m.V1_MANIFEST, {"baseline_id": "INPAINTING_RAW_TO_ROBOT_V1", "inputs": {"camera": config_ref}})
    binding = m.adopted_camera_binding(case.report, prior_sha256=canonical_sha(case.prior), v1_manifest_ref=manifest)
    assert binding["status"] == "BLOCKED_CAMERA_AXES"
    assert binding["failure_code"] == reason
    assert binding["config"] is None


def test_default_device_is_never_used(case):
    manifest, _ = adopt(case, default=True)
    result = prepare(case, v1_manifest_ref=manifest)
    assert result["raw_session"] is None
    assert result["report"]["camera_binding"]["failure_code"] == "DEVICE_NOT_IN_ADOPTED_V1_DOMAIN"


def test_adopted_override_is_metadata_only_never_reads_old_fit_inputs(case, monkeypatch):
    manifest, config = adopt(case, serial="DEVICE_B", override=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("HISTORICAL_FIT_INPUTS_MUST_NOT_BE_READ")

    monkeypatch.setattr(m.parent, "validated_installation_override", forbidden)
    monkeypatch.setattr(m.parent, "load_prepared", forbidden)
    result = prepare(case, v1_manifest_ref=manifest)
    assert case.calls[0]["prior"] == config["installation_prior_override"]
    assert result["report"]["camera_binding"]["prior_binding_status"] == "ADOPTED_FROZEN_OVERRIDE_METADATA_ONLY"
    assert not result["report"]["camera_binding"]["installation_geometry_revalidated"]
    assert not result["report"]["historical_override_validator_called"]
    assert config["installation_correction_evidence"] in result["receipt"]["actual_inputs"]
    assert config["correction_evidence"] in result["receipt"]["actual_inputs"]


def test_adopted_override_prior_value_link_mismatch_rejected(case, monkeypatch):
    manifest, config = adopt(case, override=True)
    path = Path(config["installation_correction_evidence"]["path"])
    evidence = json.loads(path.read_text())
    evidence["groups"][config["installation_group_key"]]["prior_used"]["right"]["pos"][0] = .03
    config["installation_correction_evidence"] = write_json(path, evidence)
    config_ref = write_json(m.PROJECT_ROOT / "adopted/FITTED_CAMERA_CONFIG.json", {"devices": {"DEVICE_A": config}})
    manifest = write_json(m.V1_MANIFEST, {"baseline_id": "INPAINTING_RAW_TO_ROBOT_V1", "inputs": {"camera": config_ref}})
    with pytest.raises(m.MotionSourceError, match="OVERRIDE_METADATA_PRIOR_LINK_MISMATCH"):
        prepare(case, v1_manifest_ref=manifest)
    assert not case.calls


def test_adopted_k_domain_drift_rejected_not_rescaled(case):
    manifest, config = adopt(case)
    config["output_size"] = [640, 480]
    ref = write_json(m.PROJECT_ROOT / "adopted/FITTED_CAMERA_CONFIG.json", {"devices": {"DEVICE_A": config}})
    manifest = write_json(m.V1_MANIFEST, {"baseline_id": "INPAINTING_RAW_TO_ROBOT_V1", "inputs": {"camera": ref}})
    with pytest.raises(m.MotionSourceError, match="K_NATIVE_ESTIMATE_DOMAIN_DRIFT"):
        prepare(case, v1_manifest_ref=manifest)
    assert not case.output.exists() and not case.calls


@pytest.mark.parametrize("problem,reason", [
    ("identity", "RAW_MANIFEST_IDENTITY_MISMATCH"),
    ("index", "RAW_MOTION_VIDEO_MAPPING_MISMATCH"),
    ("qpc", "RAW_MOTION_VIDEO_MAPPING_MISMATCH"),
    ("count", "RAW_MOTION_VIDEO_MAPPING_MISMATCH"),
    ("missing", "RAW_MOTION_INPUTS_ABSENT"),
    ("role", "NON_RAW_INPUT_ROLE_FORBIDDEN"),
])
def test_raw_identity_timeline_and_roles_are_not_inferred(case, problem, reason):
    if problem == "identity":
        case.report["session_id"] = "pick_place_1008_001"
    elif problem == "index":
        case.report["frame_mapping"][1]["native_frame_index"] = 0
    elif problem == "qpc":
        case.report["frame_mapping"][1]["qpc_declared_ns"] += 1
    elif problem == "count":
        case.report["native_sidecar_frames"] = 2
    elif problem == "missing":
        case.refs.pop("manus")
    else:
        case.refs["historical_motion"] = case.refs["manus"]
    with pytest.raises(m.MotionSourceError, match=reason):
        prepare(case)
    assert not case.calls and not case.output.exists()


def test_reader_paths_match_pinned_roles_not_other_current_capture_files(case):
    case.refs["manus"] = case.refs["controller_head"]
    with pytest.raises(m.MotionSourceError, match="READER_PATH_SIGNATURE_MISMATCH"):
        prepare(case)


def test_raw_source_sha_drift_fails_before_reader(case):
    Path(case.refs["manus"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="INPUT_SHA_DRIFT"):
        prepare(case)
    assert not case.calls


def test_camera_snapshot_meta_mismatch_fails_before_native_reader(case):
    case.refs["camera_meta"] = write_json(case.root / RAW_FILES["camera_meta"], {
        "device_serial": "UNKNOWN_DEVICE", "sha256": "0" * 64})
    with pytest.raises(m.MotionSourceError, match="SNAPSHOT_METADATA_SHA_MISMATCH"):
        prepare(case)
    assert not case.calls


def test_installation_container_and_attribute_sha_are_checked(case, monkeypatch):
    broken = deepcopy(case.installation)
    broken["prior_sha256"] = "0" * 64
    monkeypatch.setattr(m.parent, "read_installation", lambda directory: (case.prior, broken))
    with pytest.raises(m.MotionSourceError, match="ATTRIBUTE_SIGNATURE_DRIFT"):
        prepare(case)


def test_fresh_immutable_output_and_receipt(case):
    result = prepare(case)
    for ref in result["receipt"]["outputs"]:
        assert file_ref(ref["path"]) == ref
    with pytest.raises(m.MotionSourceError, match="FRESH_OUTPUT_REQUIRED"):
        prepare(case)


def test_only_current_v1_manifest_allowed(case):
    manifest, _ = adopt(case)
    wrong = write_json(m.PROJECT_ROOT / "other_manifest.json", json.loads(Path(manifest["path"]).read_text()))
    with pytest.raises(m.MotionSourceError, match="ONLY_CURRENT_ADOPTED_V1_MANIFEST"):
        prepare(case, v1_manifest_ref=wrong)


def test_output_outside_project_rejected(case):
    with pytest.raises(m.MotionSourceError, match="OUTPUT_OUTSIDE_PROJECT"):
        m.prepare_motion_source(case.report, Path("/tmp/not-created-motion-source"), component_signature="a" * 64)


def test_component_signature_required(case):
    with pytest.raises(m.MotionSourceError, match="COMPONENT_SIGNATURE_REQUIRED"):
        m.prepare_motion_source(case.report, case.output, component_signature="partial")


def test_object_arrays_are_not_sealed(case):
    case.native["forbidden_object"] = np.array([{}], dtype=object)
    with pytest.raises(m.MotionSourceError, match="OBJECT_ARRAYS_FORBIDDEN"):
        prepare(case)
    assert not (case.output / "RECEIPT.json").exists()


def test_partial_native_arrays_are_not_sealed(case):
    case.native["hands"] = case.native["hands"][:, :2]
    with pytest.raises(m.MotionSourceError, match="ARRAY_FRAME_CONTRACT_MISMATCH"):
        prepare(case)
    assert not case.output.exists()


def test_raw_reader_consumes_native_controller_and_manus_not_hardware_retarget(case, monkeypatch):
    pose = {"pos": [0., 0., 0.], "quat": [0., 0., 0., 1.]}
    qpc = [row["qpc_declared_ns"] for row in case.report["frame_mapping"]]
    pico = [{"recv_qpc_ns": timestamp, "dev": "UNKNOWN_DEVICE", "data": {
        "Head": {**pose, "status": 3}, "Controller": {"left": pose, "right": pose},
        "Hand": {"invalid_hardware_hand": True}, "retarget": {"must_not_be_used": True}}} for timestamp in qpc]
    path = case.root / RAW_FILES["controller_head"]
    path.write_text("\n".join(json.dumps(row) for row in pico) + "\n")
    case.refs["controller_head"] = file_ref(path)
    rows = []
    for timestamp in qpc:
        for side in ("left", "right"):
            rows.append({"recv_qpc_ns": timestamp, "side": side, "glove_id": 10 if side == "left" else 20,
                "calibration_applied": True, "node_count": 25, "node_ids": list(range(25)),
                "nodes": [[0., 0., 0., 0., 0., 0., 1.]] * 25,
                "parent_ids": list(m.parent.RAW_SDK_PARENT_IDS),
                "chain_types": list(m.parent.RAW_SDK_CHAIN_TYPES),
                "joint_types": list(m.parent.RAW_CAPTURE_JOINT_ALIASES)})
    path = case.root / RAW_FILES["manus"]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    case.refs["manus"] = file_ref(path)
    case.refs["manus_meta"] = write_json(case.root / RAW_FILES["manus_meta"], {
        "calibration": {"left": {"glove_id": 10}, "right": {"glove_id": 20}}})
    monkeypatch.setattr(m.parent, "load_raw_motion", ORIGINAL_MOTION_READER)
    result = prepare(case)
    assert result["quality"]["passed"]
    assert result["native_motion"]["hands"].shape == (2, 3, 21, 3)
    assert result["native_motion"]["hand_observed"].all()
    assert all(value == "RAW_COMPOSED" for value in result["native_motion"]["hand_source"].flat)
    assert result["report"]["hardware_retarget_fields_read"] is False
    assert result["receipt"]["retarget_calls"] == 0


def test_real_installation_reader_reads_attribute_only(case, monkeypatch):
    h5py = pytest.importorskip("h5py")
    path = case.root / "dataset.hdf5"
    with h5py.File(path, "w") as stream:
        stream.attrs["controller_to_wrist_calibration"] = json.dumps(case.prior)
        stream.create_dataset("DO_NOT_READ_HISTORICAL_HANDS", data=np.zeros((2, 3)))
    original_getitem = h5py.Group.__getitem__

    def forbidden_dataset(handle, key):
        if key == "/":
            return original_getitem(handle, key)
        raise AssertionError("HDF5_DATASET_READ_FORBIDDEN")

    monkeypatch.setattr(h5py.Group, "__getitem__", forbidden_dataset)
    prior, ref = ORIGINAL_INSTALLATION_READER(case.root)
    assert prior == case.prior
    assert ref["source_hdf5"] == file_ref(path)
    assert ref["prior_sha256"] == canonical_sha(case.prior)
