import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chaoyang.pick_place_1008 import audit


def test_checksum_manifest_accepts_bom_crlf_and_rejects_escape(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = source / "raw.json"
    data.write_bytes(b"raw")
    sha = hashlib.sha256(b"raw").hexdigest()
    checksums = source / "SHA256SUMS"
    checksums.write_bytes(("\ufeff" + sha + "  raw.json\r\n").encode("utf-8"))
    result = audit.verify_checksums(source, checksums)
    assert result["status"] == "PASSED"
    assert result["entries_verified"] == 1
    assert data.read_bytes() == b"raw"
    checksums.write_text(sha + "  ../outside.json\n")
    with pytest.raises(audit.AuditError, match="CHECKSUM_PATH_ESCAPE"):
        audit.parse_checksum_manifest(source, checksums)


def test_checksum_drift_and_worker_bound(tmp_path):
    data = tmp_path / "file"
    data.write_bytes(b"bad")
    checksums = tmp_path / "SHA256SUMS"
    checksums.write_text("0" * 64 + "  file\n")
    assert audit.verify_checksums(tmp_path, checksums)["status"] == "FAILED"
    with pytest.raises(audit.AuditError, match="HASH_WORKERS"):
        audit.verify_checksums(tmp_path, checksums, workers=3)


def test_qpc_nearest_tie_and_real_time_gate():
    mapping = audit.nearest_mapping([20_000_000, 70_000_000],
                                    [10_000_000, 30_000_000, 80_000_000])
    assert mapping["indices"] == [0, 2]
    assert mapping["offsets_ns"] == [-10_000_000, 10_000_000]
    assert mapping["stats"]["status"] == "PASSED"
    failed = audit.nearest_mapping([100_000_000], [10_000_000])
    assert failed["stats"]["status"] == "FAILED"
    assert failed["valid"] == [False]
    with pytest.raises(audit.AuditError, match="NONMONOTONIC"):
        audit.nearest_mapping([1], [1, 1])
    stats = audit.clock_stats([1_000_000_000, 1_040_000_000, 1_080_000_000])
    assert stats["median_gap_ms"] == 40
    assert stats["duration_s"] == 0.08


def test_camera_keeps_nested_extrinsics_despite_metadata_false():
    eye = {"intrinsics": {"fx": 10, "fy": 11, "cx": 2, "cy": 3}}
    raw = {"reference_resolution": {"w": 8, "h": 6},
           "left": eye, "right": eye,
           "extrinsics": {"left": np.eye(4).tolist(), "right": np.eye(4).tolist()}}
    snapshot, warnings = audit.camera_snapshot(raw, {"extrinsics_included": False})
    assert snapshot["schema"] == "nested_4x4"
    assert snapshot["left"]["extrinsic_4x4"] == np.eye(4).tolist()
    assert "CAMERA_META_EXTRINSICS_FALSE_BUT_JSON_PRESENT" in warnings
    assert snapshot["image_domain_verified"] is False


def test_action_arrays_reject_shape_and_do_not_accept_fake_qualification(tmp_path):
    arrays = tmp_path / "actions.npz"
    np.savez(arrays, q_arm=np.zeros((2, 2, 7)), q_hand=np.zeros((1, 2, 22)),
             T_world_base=np.eye(4), c2w=np.repeat(np.eye(4)[None], 2, axis=0))
    assert audit.validate_action_arrays(arrays, 2) == ["INVALID_ACTION_ARRAY:q_hand"]


def test_immutable_output_and_project_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "RESULT.json"
    audit._write_json(path, {"status": "PASSED"})
    audit._write_json(path, {"status": "PASSED"})
    with pytest.raises(audit.AuditError, match="IMMUTABLE"):
        audit._write_json(path, {"status": "FAILED"})
    with pytest.raises(audit.AuditError, match="WRITE_OUTSIDE"):
        audit._write_json(tmp_path.parent / "outside.json", {})


def _make_session(root: Path):
    session = root / "001"
    raw = session / "raw"
    raw.mkdir(parents=True)
    camera = {
        "width": 8, "height": 6,
        "left": {"fx": 10, "fy": 10, "cx": 4, "cy": 3,
                 "extrinsic": np.eye(4)[:3].reshape(-1).tolist()},
        "right": {"fx": 10, "fy": 10, "cx": 4, "cy": 3,
                  "extrinsic": np.eye(4)[:3].reshape(-1).tolist()},
    }
    (raw / "camera_params.json").write_text(json.dumps(camera))
    camera_sha = audit.sha256_file(raw / "camera_params.json")
    (raw / "camera_params.meta.json").write_text(json.dumps({
        "device_serial": "DEVICE_A", "sha256": camera_sha, "extrinsics_included": True}))
    for name in ["capture.selection.json", "capture.workflow.json", "manus.meta.json"]:
        (raw / name).write_text(json.dumps({"session": "pick_place_1008_001"}))
    (session / "manifest.json").write_text(json.dumps({"session": "pick_place_1008_001"}))
    (session / "dataset.hdf5").write_bytes(b"DO_NOT_READ_HDF5_ACTIONS")
    (raw / "vst.h264").write_bytes(b"MOCK_VIDEO")
    times = [1_000_000_000, 1_040_000_000]
    for name in ["vst.qpc.ts.jsonl", "vst.ts.jsonl"]:
        (raw / name).write_text("".join(json.dumps(t) + "\n" for t in times))
    pose = {"pos": [0, 0, 0], "quat": [0, 0, 0, 1]}
    (raw / "pico.jsonl").write_text("".join(json.dumps({
        "dev": "DEVICE_A", "recv_qpc_ns": t, "data": {
            "Head": pose, "Controller": {"left": pose, "right": pose}}}) + "\n" for t in times))
    nodes = np.zeros((25, 7))
    nodes[:, 6] = 1
    (raw / "manus.jsonl").write_text("".join(json.dumps({
        "side": side, "recv_qpc_ns": t, "nodes": nodes.tolist(), "glove_id": 1,
        "node_ids": list(range(25)), "node_count": 25, "calibration_applied": True
    }) + "\n" for t in times for side in ["left", "right"]))
    return session


def test_raw_session_audit_no_hdf5_consumption_and_count_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    session = _make_session(source)
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit, "decode_video", lambda *args: {
        "status": "PASSED", "decoded_frames": 2, "stream": {"width": 16, "height": 6}})
    shas = {str(p.relative_to(source)): audit.sha256_file(p)
            for p in session.rglob("*") if p.is_file()}
    result = audit._audit_session(source, "001", shas, tmp_path / "output", 2)
    assert result["input_status"] == "READY_RAW_AUDIT"
    assert result["raw_frame_count"] == 2
    assert result["raw_metadata"]["hdf5_consumption"].startswith("NONE")
    rows = [json.loads(line) for line in Path(result["raw_frame_mapping"]["path"]).read_text().splitlines()]
    assert [row["relative_qpc_ns"] for row in rows] == [0, 40_000_000]
    monkeypatch.setattr(audit, "decode_video", lambda *args: {
        "status": "PASSED", "decoded_frames": 1, "stream": {"width": 16, "height": 6}})
    failed = audit._audit_session(source, "001", shas, tmp_path / "mismatch", 2)
    assert "RAW_DECODE_QPC_COUNT_MISMATCH" in failed["errors"]


def test_pts_accepts_microsecond_quantization_and_rejects_cfr():
    raw = [1_000_000_000, 1_039_212_300, 1_081_654_800]
    assert audit.validate_video_timeline([0, 39_212_000, 81_655_000], raw) == []
    assert audit.validate_video_timeline([0, 40_000_000, 80_000_000], raw) == [
        "DELIVERY_PTS_QPC_MISMATCH"]
    assert audit.validate_video_timeline([0, 0, 40_000_000], raw) == [
        "DELIVERY_PTS_NOT_STRICTLY_INCREASING"]


def test_p0_manifest_uses_raw_denominator_and_keeps_invalid_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    _make_session(source)
    inspection = source / "_inspection"
    inspection.mkdir()
    paths = sorted(p for p in source.rglob("*") if p.is_file())
    (inspection / "SHA256SUMS").write_text("".join(
        audit.sha256_file(p) + "  " + str(p.relative_to(source)) + "\n" for p in paths))
    (inspection / "transfer_receipt.json").write_text(json.dumps({
        "checksum_entries_verified": len(paths), "sessions": 1}))
    monkeypatch.setattr(audit, "decode_video", lambda *args: {
        "status": "PASSED", "decoded_frames": 1, "stream": {"width": 16, "height": 6}})
    result = audit.run_p0_audit(source, tmp_path / "out", expected_sessions=1, expected_frames=2)
    assert result["status"] == "FAILED"
    assert result["raw_frames"] == 2
    assert result["decoded_frames"] == 1
    assert result["failed_sessions"] == ["001"]
    manifest = json.loads(Path(result["input_manifest"]["path"]).read_text())
    assert manifest["expected_sessions"] == 1
    assert len(manifest["sessions"]) == 1
    assert manifest["sessions"][0]["input_status"] == "FAILED_RAW_AUDIT"


def test_independent_quality_review_is_required_and_bound_to_sealed_session(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    session = tmp_path / "SESSION.json"
    session.write_text(json.dumps({"session_id": "001", "visual_status": "UNREVIEWED",
        "ai_review": {"required_fixed_frames": [0, 1], "required_candidate_windows": [],
                      "required_abnormal_windows": [], "abnormal_raw_frame_indices": []}}))
    session_ref = audit.file_ref(session)
    errors, review = audit.validate_quality_review({}, session_ref, "001", 2)
    assert errors == ["MISSING_INDEPENDENT_QUALITY_REVIEW"]
    quality = tmp_path / "AI_REVIEW.json"
    value = {
        "session_id": "001", "session_result": session_ref, "visual_status": "PASSED",
        "severe_issues_open": [], "ai_inspected_raw_frame_indices": [0, 1],
        "candidate_windows_coverage": {"total": 0, "reviewed": 0, "all_covered": True,
                                       "source": session_ref, "windows": []},
        "abnormal_windows_coverage": {"total": 0, "reviewed": 0, "all_covered": True,
                                     "source": session_ref, "windows": []},
        "accepted_defects": [], "human_review_status": "PENDING",
    }
    quality.write_text(json.dumps(value))
    delivery = {"quality_review": audit.file_ref(quality)}
    assert audit.validate_quality_review(delivery, session_ref, "001", 2)[0] == []
    delivery["quality_review"]["sha256"] = "0" * 64
    assert audit.validate_quality_review(delivery, session_ref, "001", 2)[0] == [
        "QUALITY_REVIEW_SHA_MISMATCH"]
    value["session_result"] = {**session_ref, "sha256": "0" * 64}
    quality.write_text(json.dumps(value))
    delivery["quality_review"] = audit.file_ref(quality)
    assert "QUALITY_REVIEW_SESSION_RESULT_BINDING_MISMATCH" in audit.validate_quality_review(
        delivery, session_ref, "001", 2)[0]
    assert json.loads(session.read_text())["visual_status"] == "UNREVIEWED"


def test_quality_review_rejects_uninspected_frames_and_missing_window_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    session = tmp_path / "SESSION.json"
    session.write_text("{}")
    ref = audit.file_ref(session)
    quality = tmp_path / "REVIEW.json"
    quality.write_text(json.dumps({
        "session_id": "001", "session_result": ref, "visual_status": "PASSED",
        "severe_issues_open": 0, "ai_inspected_raw_frame_indices": [9],
        "accepted_defects": [],
    }))
    errors = audit.validate_quality_review({"quality_review": audit.file_ref(quality)}, ref, "001", 2)[0]
    assert "AI_INSPECTION_FRAME_EVIDENCE_MISSING_OR_INVALID" in errors
    assert any(error.startswith("REVIEW_WINDOW_COVERAGE") for error in errors)


def test_sealed_review_anchors_and_windows_cannot_be_self_reported():
    ref = {"path": "/project/SESSION.json", "sha256": "a" * 64, "bytes": 10}
    window = {"start_frame": 2, "end_frame_inclusive": 4, "phase": "NEAR_OBJECT"}
    sealed = {"ai_review": {
        "required_fixed_frames": [0, 2, 4, 6, 8],
        "required_candidate_windows": [window], "required_abnormal_windows": [window],
        "abnormal_raw_frame_indices": [3]}}
    review = {
        "ai_inspected_raw_frame_indices": [0, 2, 3, 4, 6, 8],
        **{key: {"total": 1, "reviewed": 1, "all_covered": True, "source": ref,
                 "windows": [{**window, "checked_raw_frame_indices": [3]}]}
           for key in ["candidate_windows_coverage", "abnormal_windows_coverage"]}}
    assert audit._validate_review_requirements(review, sealed, ref, 9) == []
    review["ai_inspected_raw_frame_indices"].remove(6)
    assert "REQUIRED_FIXED_FRAMES_NOT_INSPECTED" in audit._validate_review_requirements(
        review, sealed, ref, 9)
    review["ai_inspected_raw_frame_indices"].append(6)
    review["abnormal_windows_coverage"] = {"total": 0, "reviewed": 0, "all_covered": True,
                                          "source": ref, "windows": []}
    assert any("SOURCE_COUNT_BINDING" in error for error in audit._validate_review_requirements(
        review, sealed, ref, 9))
    review["abnormal_windows_coverage"] = review["candidate_windows_coverage"].copy()
    review["candidate_windows_coverage"]["windows"] = [
        {**window, "end_frame_inclusive": 5, "checked_raw_frame_indices": [3]}]
    assert any("RANGE_OR_INSPECTION" in error for error in audit._validate_review_requirements(
        review, sealed, ref, 9))


def _boundary_p0_fixture(tmp_path, monkeypatch, last_offset_ns=49_000_000):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    session = _make_session(source)
    raw = session / "raw"
    times = [1_000_000_000 + index * 100_000_000 for index in range(100)]
    for name in ["vst.qpc.ts.jsonl", "vst.ts.jsonl"]:
        (raw / name).write_text("".join(json.dumps(value) + "\n" for value in times))
    pose = {"pos": [0, 0, 0], "quat": [0, 0, 0, 1]}
    (raw / "pico.jsonl").write_text("".join(json.dumps({
        "dev": "DEVICE_A", "recv_qpc_ns": value,
        "data": {"Head": pose, "Controller": {"left": pose, "right": pose}}
    }) + "\n" for value in times))
    nodes = np.zeros((25, 7))
    nodes[:, 6] = 1
    sensor_rows = []
    for index, value in enumerate(times):
        for side in ["left", "right"]:
            timestamp = value - last_offset_ns if side == "right" and index == 99 else value
            sensor_rows.append({"side": side, "recv_qpc_ns": timestamp, "nodes": nodes.tolist(),
                               "glove_id": 1, "node_ids": list(range(25)), "node_count": 25,
                               "calibration_applied": True})
    (raw / "manus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sensor_rows))
    inspection = source / "_inspection"
    inspection.mkdir()
    paths = sorted(path for path in source.rglob("*") if path.is_file())
    (inspection / "SHA256SUMS").write_text("".join(
        audit.sha256_file(path) + "  " + str(path.relative_to(source)) + "\n" for path in paths))
    (inspection / "transfer_receipt.json").write_text(json.dumps({
        "checksum_entries_verified": len(paths), "sessions": 1}))
    monkeypatch.setattr(audit, "decode_video", lambda *args: {
        "status": "PASSED", "decoded_frames": 100, "stream": {"width": 16, "height": 6}})
    attempt = tmp_path / "p0/attempts/attempt_0001"
    stage = attempt / "candidate"
    audit.run_p0_audit(source, stage, expected_sessions=1, expected_frames=100)
    task_id = "exact78_pick_place_test_p0"
    packet_path = tmp_path / "tasks/current" / task_id / "TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps({"task_id": task_id, "run_signature": "synthetic_signed"}))
    outer = {"task_id": task_id, "run_signature": "synthetic_signed",
             "status": "FAILED_RUNTIME_FINAL", "validation_errors": [],
             "task_packet": audit.file_ref(packet_path),
             "stage_result": audit.file_ref(stage / "RESULT.json"),
             "required_outputs": {f"candidate/{name}": audit.file_ref(stage / name)
                                  for name in ["RESULT.json", "AUDIT.json", "INPUT_MANIFEST.json",
                                               "TRANSFER_SHA_AUDIT.json"]}}
    (attempt / "RESULT.json").write_text(json.dumps(outer))
    receipt = tmp_path / "tasks/receipts" / f"{task_id.upper()}.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"result": audit.file_ref(attempt / "RESULT.json"),
                                   "status": "FAILED_RUNTIME_FINAL"}))
    return stage / "INPUT_MANIFEST.json", source


def test_resolution_preserves_failed_nearest_gate_and_all_raw_qpc(tmp_path, monkeypatch):
    manifest_path, source = _boundary_p0_fixture(tmp_path, monkeypatch)
    original_bytes = manifest_path.read_bytes()
    original = json.loads(original_bytes)
    original_mapping = Path(original["sessions"][0]["raw_frame_mapping"]["path"]).read_bytes()
    result = audit.resolve_time_boundaries(manifest_path, tmp_path / "resolution")
    resolved = json.loads(Path(result["input_manifest"]["path"]).read_text())
    row = resolved["sessions"][0]
    assert row["input_status"] == "READY_RAW_WITH_INVALID_SHORT_GAP"
    assert row["raw_audit_status"] == "FAILED_RAW_AUDIT"
    assert row["time_stats"]["mapping"]["manus_right"]["max_ms"] == 49
    assert row["time_stats"]["mapping"]["manus_right"]["status"] == "FAILED"
    assert row["time_stats"]["accepted_mapping"]["manus_right"]["status"] == "PASSED"
    assert row["time_stats"]["accepted_mapping"]["manus_right"]["coverage"] == .99
    derived = [json.loads(line) for line in Path(row["raw_frame_mapping"]["path"]).read_text().splitlines()]
    assert len(derived) == 100
    assert derived[-1]["manus_right_offset_ns"] == -49_000_000
    assert derived[-1]["manus_right_observation_status"] == "INVALID_TIME_GAP"
    assert derived[-1]["manus_right_producer_source_alias"] == "EDGE_HOLD_SHORT"
    assert manifest_path.read_bytes() == original_bytes
    assert Path(original["sessions"][0]["raw_frame_mapping"]["path"]).read_bytes() == original_mapping
    assert [frame["video_qpc_ns"] for frame in derived] == audit.read_integer_jsonl(
        source / "001/raw/vst.qpc.ts.jsonl")


def test_resolution_rejects_changed_source_and_unsealed_p0(tmp_path, monkeypatch):
    path, source = _boundary_p0_fixture(tmp_path, monkeypatch)
    (source / "001/raw/pico.jsonl").write_text("{}\n")
    with pytest.raises(audit.AuditError, match="EVIDENCE_REFERENCE_DRIFT"):
        audit.resolve_time_boundaries(path, tmp_path / "resolved")


def test_resolution_refuses_long_gap_and_non_time_failure():
    targets = [1_000_000_000, 2_000_000_000]
    sensor = {"times": [1_000_000_000], "rows": [0], "valid": [True]}
    mapping = audit.nearest_mapping(targets, sensor["times"], sensor["valid"])
    with pytest.raises(audit.AuditError, match="EXCEEDS_0_5_SECONDS"):
        audit._short_gap_records(targets, sensor, mapping, "manus_right")
    bad = {"times": [1_000_000_000], "rows": [0], "valid": [False]}
    with pytest.raises(audit.AuditError, match="INVALID_RAW_SENSOR_CONTENT"):
        audit._short_gap_records(targets, bad, audit.nearest_mapping(
            targets, bad["times"], bad["valid"]), "manus_right")


def test_resolution_final_delivery_cannot_claim_filled_frame_observed(tmp_path, monkeypatch):
    path, _ = _boundary_p0_fixture(tmp_path, monkeypatch)
    result = audit.resolve_time_boundaries(path, tmp_path / "resolution")
    manifest = json.loads(Path(result["input_manifest"]["path"]).read_text())
    source = manifest["sessions"][0]
    mapping = [json.loads(line) for line in Path(source["raw_frame_mapping"]["path"]).read_text().splitlines()]
    states = [{"hands": {"right": {"raw_observed": True}}} for _ in mapping]
    monkeypatch.setattr(audit, "_verify_resolution_producer", lambda ref: None)
    assert any("SOURCE_NOT_RECORDED" in error for error in audit.validate_resolved_frame_states(
        source, manifest, mapping, states))
    states[-1]["hands"]["right"] = {"raw_observed": False, "display_valid": True,
                                     "source": "EDGE_HOLD_SHORT", "handshape_valid": False}
    assert audit.validate_resolved_frame_states(source, manifest, mapping, states) == []
