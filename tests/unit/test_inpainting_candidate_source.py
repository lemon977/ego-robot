import json

import pytest

from chaoyang.inpainting_candidate import source as s
from chaoyang.inpainting_candidate.contracts import file_ref


@pytest.fixture
def raw_session(tmp_path, monkeypatch):
    root = tmp_path / "arbitrary_task" / "001"
    (root / "raw").mkdir(parents=True)
    params = {"width": 8, "height": 6,
              "left": {"sourceIndex": 1, "fx": 4, "fy": 4, "cx": 4, "cy": 3},
              "right": {"sourceIndex": 0, "fx": 4, "fy": 4, "cx": 4, "cy": 3}}
    (root / "manifest.json").write_text(json.dumps({"session": "new-source-identity"}))
    (root / "raw/camera_params.json").write_text(json.dumps(params))
    (root / "raw/camera_params.meta.json").write_text(json.dumps({
        "device_serial": "device", "sha256": file_ref(root / "raw/camera_params.json")["sha256"]}))
    (root / "raw/manus.meta.json").write_text(json.dumps({"session": "new-source-identity"}))
    (root / "raw/vst.qpc.ts.jsonl").write_text("1000000000\n1030000000\n1060000000\n")
    (root / "raw/vst.ts.jsonl").write_text("2000000000\n2030000000\n2060000000\n")
    (root / "raw/vst.h264").write_bytes(b"synthetic video; not decoded")
    (root / "raw/pico.jsonl").write_text(json.dumps({"data": {"Head": {}, "Controller": {"left": {}, "right": {}}}}) + "\n")
    (root / "raw/manus.jsonl").write_text('{"side":"left"}\n{"side":"right"}\n')
    (root / "dataset.hdf5").write_bytes(b"synthetic installation container")
    monkeypatch.setattr(s, "read_installation_attribute", lambda path: {"status": "PRESENT_INSTALLATION_PRIOR", "hdf5_datasets_read": False})
    return root


def test_generic_original_source_identity_and_time_unknown(raw_session):
    result = s.inspect_raw_session(raw_session)
    assert result["session_id"] == "new-source-identity"
    assert result["native_sidecar_frames"] == 3
    assert result["camera"]["physical_eye"] == "right"
    assert result["camera"]["sbs_index"] == 0
    assert result["qpc_frequency"] == "ABSENT" and result["cross_clock_sync_verified"] is False
    assert result["frame_count_status"] == "NOT_DECODED"
    assert result["frame_mapping"][1]["acquisition_frame_id"] == "UNKNOWN"
    assert not result["hdf5_datasets_read"] and not result["training_eligible"]


def test_old_seeds_aligned_and_processed_are_never_inputs(raw_session):
    (raw_session / "manual_seeds.json").write_text("invalid poison must never load")
    (raw_session / "aligned.jsonl").write_text("invalid poison must never load")
    (raw_session / "video").mkdir()
    (raw_session / "video/left_eye_tactile.mp4").write_bytes(b"wrong processed source")
    result = s.inspect_raw_session(raw_session)
    assert len(result["actual_inputs"]) == len(s.RAW_FILES)
    assert not result["processed_video_read"] and not result["aligned_rows_read"]
    assert result["motion"]["controller_head"]["retarget_arrays_consumed"] is False


def test_missing_raw_files_explicit_not_replaced_with_processed(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "processed.mp4").write_bytes(b"fallback forbidden")
    result = s.inspect_raw_session(session)
    assert result["status"] == "BLOCKED_INPUT_INCOMPLETE"
    assert "raw/vst.h264" in result["missing"]


def test_backward_qpc_and_sidecar_count_rejected(raw_session):
    (raw_session / "raw/vst.qpc.ts.jsonl").write_text("2\n1\n3\n")
    with pytest.raises(s.SourceError, match="DUPLICATE_OR_BACKWARD"):
        s.inspect_raw_session(raw_session)
    (raw_session / "raw/vst.qpc.ts.jsonl").write_text("1\n2\n")
    with pytest.raises(s.SourceError, match="COUNT_OR_TYPE"):
        s.inspect_raw_session(raw_session)


def test_route_missing_source_index_never_guessed(raw_session):
    params = json.loads((raw_session / "raw/camera_params.json").read_text())
    del params["left"]["sourceIndex"]
    (raw_session / "raw/camera_params.json").write_text(json.dumps(params))
    meta = json.loads((raw_session / "raw/camera_params.meta.json").read_text())
    meta["sha256"] = file_ref(raw_session / "raw/camera_params.json")["sha256"]
    (raw_session / "raw/camera_params.meta.json").write_text(json.dumps(meta))
    result = s.inspect_raw_session(raw_session)
    assert result["camera"]["status"] == "BLOCKED_PHYSICAL_EYE_UNRESOLVED"
    assert result["camera"]["physical_eye"] is None


def test_camera_snapshot_sha_drift_rejected(raw_session):
    (raw_session / "raw/camera_params.json").write_text("{}")
    with pytest.raises(s.SourceError, match="SNAPSHOT_SHA"):
        s.inspect_raw_session(raw_session)


def test_decode_count_and_mapping_still_not_exposure_proof(raw_session, monkeypatch):
    monkeypatch.setattr(s, "probe_vst", lambda path: {"decoded_frames": 3, "width": 16, "height": 6})
    result = s.inspect_raw_session(raw_session, decode_probe=True)
    assert result["decoded_frames"] == 3 and result["camera"]["encoded_dimension_match"] is True
    assert result["frame_mapping"][2]["decoded_display_index"] == 2
    assert result["frame_mapping"][2]["model_input_frame"] is None
    assert result["timebase_status"].startswith("TIMEBASE_UNVERIFIED")
    assert result["camera"]["projection_qualified"] is False


def test_bad_decode_count_rejected(raw_session, monkeypatch):
    monkeypatch.setattr(s, "probe_vst", lambda path: {"decoded_frames": 2, "width": 16, "height": 6})
    with pytest.raises(s.SourceError, match="DECODE_QPC_FRAME_COUNT"):
        s.inspect_raw_session(raw_session, decode_probe=True)


def test_encoded_domain_mismatch_not_silent_scaled_calibration(raw_session, monkeypatch):
    monkeypatch.setattr(s, "probe_vst", lambda path: {"decoded_frames": 3, "width": 12, "height": 4})
    camera = s.inspect_raw_session(raw_session, decode_probe=True)["camera"]
    assert camera["calibration_domain_status"] == "ENCODED_CALIBRATION_DOMAIN_MISMATCH"
    assert camera["projection_qualified"] is False and camera["upright_axes"] == "UNKNOWN"


def test_installation_reads_attribute_only(tmp_path, monkeypatch):
    import h5py
    prior = {side: {"pos": [0, 0, 0], "quat": [0, 0, 0, 1]} for side in ("left", "right")}

    class AttributeOnly:
        attrs = {"controller_to_wrist_calibration": json.dumps(prior)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __getitem__(self, name):
            raise AssertionError("HDF5 dataset must not be read")

    monkeypatch.setattr(h5py, "File", lambda *args: AttributeOnly())
    result = s.read_installation_attribute(tmp_path / "installation.hdf5")
    assert result["value"] == prior and result["hdf5_datasets_read"] is False


def test_missing_installation_only_blocks_its_consumer(raw_session, monkeypatch):
    original = s.Path.is_file
    monkeypatch.setattr(s.Path, "is_file", lambda path: False if path.name == "dataset.hdf5" else original(path))
    result = s.inspect_raw_session(raw_session)
    assert result["status"] == "SOURCE_PREFLIGHT_CANDIDATE"
    assert result["installation"]["status"] == "ABSENT"
    assert result["capabilities"]["mask_from_rgb"] == "CANDIDATE_RGB_AVAILABLE_GEOMETRY_NOT_REQUIRED"
    assert result["capabilities"]["installed_wrist_motion"] == "BLOCKED_INSTALLATION_ABSENT"


def test_missing_camera_and_motion_do_not_invalidate_raw_rgb(raw_session, monkeypatch):
    original = s.Path.is_file
    absent = {"camera_params.json", "camera_params.meta.json", "pico.jsonl", "manus.jsonl", "manus.meta.json"}
    monkeypatch.setattr(s.Path, "is_file", lambda path: False if path.name in absent else original(path))
    result = s.inspect_raw_session(raw_session)
    assert result["status"] == "SOURCE_PREFLIGHT_CANDIDATE"
    assert result["capabilities"]["raw_rgb_and_time"] == "SOURCE_AVAILABLE_TIMEBASE_UNVERIFIED"
    assert result["capabilities"]["native_motion"] == "BLOCKED_NATIVE_MOTION_INPUT_ABSENT"
    assert result["camera"]["physical_eye"] is None
