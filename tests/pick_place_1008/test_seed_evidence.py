import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from chaoyang.pick_place_1008 import audit, seed_evidence as evidence, source


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(tmp_path, monkeypatch, *, override=False):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    raw_root = tmp_path / "nas"
    folder = raw_root / "001"
    raw = folder / "raw"
    raw.mkdir(parents=True)
    with av.open(str(raw / "vst.h264"), "w", format="h264") as container:
        stream = container.add_stream("libx264rgb", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 6, "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for index in range(4):
            pixels = np.zeros((6, 16, 3), np.uint8)
            pixels[:, :8] = [250, 0, 0]  # discarded SBS eye
            pixels[:, 8:] = [10 + index, 20 + index, 30 + index]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    params = {
        "width": 8, "height": 6, "format": "source_encoding_unverified",
        "right": {"fx": 4., "fy": 4., "cx": 4., "cy": 3.,
                  "extrinsic": np.eye(4)[:3].reshape(-1).tolist()},
    }
    _write(raw / "camera_params.json", params)
    camera_sha = source.ref(raw / "camera_params.json")["sha256"]
    _write(raw / "camera_params.meta.json", {"device_serial": "DEVICE_A", "sha256": camera_sha})
    (raw / "vst.qpc.ts.jsonl").write_text("\n".join(map(str, [100, 150, 210, 290])) + "\n")
    _write(folder / "manifest.json", {"session_id": "001"})
    refs = {role: source.ref(folder / filename) for role, filename in evidence.SOURCE_FILENAMES.items()}
    # Nonconsumed pose/HDF refs deliberately do not exist.
    refs["hdf5_prior_only"] = {"path": str(folder / "absent.hdf5"), "bytes": 9, "sha256": "f" * 64}
    refs["pico"] = {"path": str(raw / "absent_pico.jsonl"), "bytes": 9, "sha256": "f" * 64}
    refs["manus"] = {"path": str(raw / "absent_manus.jsonl"), "bytes": 9, "sha256": "f" * 64}
    row = {"session_id": "001", "device_serial": "DEVICE_A", "raw_frame_count": 4,
           "camera_snapshot_sha256": camera_sha, "input_status": "READY_RAW_AUDIT",
           "source_refs": refs}
    manifest = {"source_root": str(raw_root), "sessions": [row]}
    selected = {
        "device_serial": "DEVICE_A", "camera_sha256": camera_sha, "eye": "right",
        "sbs_index": 1, "output_size": [1280, 960], "image_mode": "encoded_crop_resize_only",
        "extrinsic_convention": "head_to_camera", "upright_axes": "capture_sensor_from_upright",
        "status": "DEVELOPMENT_CANDIDATE",
        "eye_identity_qualification": "VISUAL_ASSUMPTION_SOURCEINDEX_ABSENT",
    }
    if override:
        selected["K_override"] = [[900., 0., 600.], [0., 850., 470.], [0., 0., 1.]]
        correction_path = tmp_path / "CORRECTION.json"
        _write(correction_path, {"status": "PASSED", "qualification": "VISUAL_ESTIMATE_ONLY"})
        selected["correction_evidence"] = source.ref(correction_path)
    config = {"devices": {"DEVICE_A": selected}}
    manifest_path, config_path = tmp_path / "INPUT.json", tmp_path / "CAMERA.json"
    _write(manifest_path, manifest)
    _write(config_path, config)
    def no_poses(*args, **kwargs):
        raise AssertionError("pose/HDF loader must not be called by raw seed evidence")
    monkeypatch.setattr(source, "load_raw_motion", no_poses)
    monkeypatch.setattr(source, "read_installation", no_poses)
    return manifest_path, config_path, manifest, config, folder


def _run(tmp_path, inputs, frame_indices=None, sids=None):
    manifest_path, config_path = inputs[:2]
    return evidence.run_seed_evidence(tmp_path, tmp_path / "out", manifest_path, config_path,
                                      ["001"] if sids is None else sids, frame_indices)


def _report(result):
    return json.loads(Path(result["source_seed_images"]["path"]).read_text())


def test_actual_h264_right_first_frame_png_rgb_digest_qpc_and_refs_close_without_pose_loading(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    result = _run(tmp_path, inputs)
    report = _report(result)
    assert result["status"] == "PASSED" and result["sessions_complete"] == 1
    assert result["native_prefix_frames_decoded"] == 1
    assert report["sessions"][0]["decoded_native_prefix_frames"] == 1
    assert report["raw_pose_loaded"] is False and report["hdf5_data_consumed"] is False
    assert report["GPU_used"] is False and report["full_video_decode_performed"] is False
    record = report["images"][0]
    pixels = cv2.cvtColor(cv2.imread(record["image"]["path"]), cv2.COLOR_BGR2RGB)
    assert pixels.shape == (480, 640, 3)
    assert np.all(pixels == [10, 20, 30])
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == record["rgb_sha256"]
    assert record["image"] == source.ref(Path(record["image"]["path"]))
    assert record["raw_frame_index"] == 0 and record["qpc_ns"] == 100
    assert record["sbs_index"] == 1 and record["pixels_overlaid"] is False
    assert record["physical_eye_verified"] is False and record["lens_remap"] is False
    assert set(record["verified_source_refs"]) == set(evidence.CONSUMED_ROLES)
    assert report["input_manifest"] == source.ref(inputs[0])
    assert report["camera_config"] == source.ref(inputs[1])
    assert "source_seed_images" not in report  # no self/report/result SHA cycle
    assert record["K_image_used"] == [[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]]


def test_target_index_uses_native_display_order_and_decoder_stops_exactly_at_maximum(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    actual_decoder = source.crop_frames
    yielded, closed = [], []
    def watched(*args):
        decoder = actual_decoder(*args)
        try:
            for index, pixels in enumerate(decoder):
                yielded.append(index)
                yield pixels
        finally:
            decoder.close()
            closed.append(True)
    monkeypatch.setattr(source, "crop_frames", watched)
    result = _run(tmp_path, inputs, {"001": [2]})
    report = _report(result)
    record = report["images"][0]
    assert yielded == [0, 1, 2] and closed == [True]
    assert result["native_prefix_frames_decoded"] == 3
    assert record["raw_frame_index"] == 2 and record["qpc_ns"] == 210
    pixels = cv2.cvtColor(cv2.imread(record["image"]["path"]), cv2.COLOR_BGR2RGB)
    assert np.all(pixels == [12, 22, 32])


def test_existing_1280_k_override_is_scaled_once_and_base_config_remains_immutable(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch, override=True)
    before = inputs[1].read_bytes()
    record = _report(_run(tmp_path, inputs, [0, 3]))["images"][0]
    assert record["K_base_used"] == inputs[3]["devices"]["DEVICE_A"]["K_override"]
    assert record["K_image_used"] == [[450., 0., 300.], [0., 425., 235.], [0., 0., 1.]]
    assert inputs[1].read_bytes() == before


def test_same_output_cannot_overwrite_sealed_png_or_manifest(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    result = _run(tmp_path, inputs)
    report_path = Path(result["source_seed_images"]["path"])
    image_path = Path(_report(result)["images"][0]["image"]["path"])
    originals = [report_path.read_bytes(), image_path.read_bytes()]
    with pytest.raises(FileExistsError):
        _run(tmp_path, inputs)
    assert [report_path.read_bytes(), image_path.read_bytes()] == originals


@pytest.mark.parametrize("selection", [
    [], [-1], [True], [4], [0, 0], {"other": [0]}, {"001": []},
])
def test_invalid_or_outside_frozen_anchor_selection_is_rejected(tmp_path, monkeypatch, selection):
    inputs = _fixture(tmp_path, monkeypatch)
    with pytest.raises(evidence.SeedEvidenceError):
        _run(tmp_path, inputs, selection)
    assert not (tmp_path / "out/SOURCE_SEED_IMAGES.json").exists()


@pytest.mark.parametrize("mutation", ["raw_drift", "qpc_count", "qpc_backward", "wrong_sbs",
                                      "wrong_output_domain", "wrong_device"])
def test_source_identity_clock_or_camera_route_drift_is_rejected(tmp_path, monkeypatch, mutation):
    inputs = _fixture(tmp_path, monkeypatch)
    manifest_path, config_path, manifest, config, folder = inputs
    if mutation == "raw_drift":
        with (folder / "raw/vst.h264").open("ab") as handle:
            handle.write(b"drift")
    elif mutation in ("qpc_count", "qpc_backward"):
        qpc = folder / "raw/vst.qpc.ts.jsonl"
        qpc.write_text("100\n150\n" if mutation == "qpc_count" else "100\n150\n140\n290\n")
        manifest["sessions"][0]["source_refs"]["video_qpc"] = source.ref(qpc)
        _write(manifest_path, manifest)
    elif mutation == "wrong_sbs":
        config["devices"]["DEVICE_A"]["sbs_index"] = 0
        _write(config_path, config)
    elif mutation == "wrong_output_domain":
        config["devices"]["DEVICE_A"]["output_size"] = [640, 480]
        _write(config_path, config)
    elif mutation == "wrong_device":
        metadata_path = folder / "raw/camera_params.meta.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["device_serial"] = "DEVICE_B"
        _write(metadata_path, metadata)
        manifest["sessions"][0]["source_refs"]["camera_meta"] = source.ref(metadata_path)
        _write(manifest_path, manifest)
    with pytest.raises((evidence.SeedEvidenceError, audit.AuditError, source.SourceContractError, ValueError)):
        _run(tmp_path, inputs)
    assert not (tmp_path / "out/SOURCE_SEED_IMAGES.json").exists()


def test_duplicate_or_missing_session_requests_and_external_output_are_rejected(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    for sids in (["001", "001"], ["199"]):
        with pytest.raises(evidence.SeedEvidenceError):
            _run(tmp_path, inputs, sids=sids)
    with pytest.raises(audit.AuditError, match="OUTSIDE_PROJECT"):
        evidence.run_seed_evidence(tmp_path, tmp_path.parent / "outside_seed_output",
                                   inputs[0], inputs[1], ["001"])


def test_short_raw_stream_is_not_claimed_to_reach_target_and_decoder_is_closed(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path, monkeypatch)
    closed = []
    def short(*args):
        try:
            yield np.zeros((480, 640, 3), np.uint8)
        finally:
            closed.append(True)
    monkeypatch.setattr(source, "crop_frames", short)
    with pytest.raises(evidence.SeedEvidenceError, match="ENDED_BEFORE_REQUESTED"):
        _run(tmp_path, inputs, [2])
    assert closed == [True]
    assert not (tmp_path / "out/RESULT.json").exists()
