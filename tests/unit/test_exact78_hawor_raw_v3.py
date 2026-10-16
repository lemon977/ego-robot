"""CPU-only, independent cold-start boundary tests for the HaWoR v3 adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.ops import run_exact78_hawor_raw_v3 as adapter


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _tracker_fixture(tmp_path: Path, *, fps: int = 25) -> tuple[Path, dict, dict]:
    session = tmp_path / "chips_cards_tracker_0901" / "potato_chips" / "get_potato_chips_0901_013"
    session.mkdir(parents=True)
    video = session / "CameraRecord_get_potato_chips_0901_013.mp4"
    video.write_bytes(b"test clip RGB")
    camera = _json(session / "camera_params.json", {"intrinsics": "test"})
    slam = session / "slam_trajectory_get_potato_chips_0901_013.jsonl"
    slam.write_text('{"poseRelativeToClipStart": "0,0,0,0,0,0,1"}\n', encoding="utf-8")
    manifest = _json(session / "clip_manifest.json", {
        "files": {"video": video.name, "camera_params": camera.name,
                  "slam": slam.name, "tracking": "trackingData.txt",
                  "humanego_all_data": "preprocess/all_data"},
        "video": {"width": 1280, "height": 960, "fps": fps, "frame_count": 2},
        "video_rectification": {"eye": "left", "rectified": True,
                                "resolution": [1280, 960],
                                "intrinsics": [[640, 0, 639.5], [0, 640, 479.5], [0, 0, 1]]},
    })
    session_id = session.name
    freeze = {"sessions": {session_id: {
        "source_path": str(session), "role": "development", "split_claim": "DEVELOPMENT",
        "files": {"video": adapter.ref(video), "camera": adapter.ref(camera),
                  "slam": adapter.ref(slam), "clip_manifest": adapter.ref(manifest)},
    }}}
    frame_map = {"sessions": {session_id: {
        "decoded_frames": 2,
        "frame_rows": [{"decoded_display_index": i, "sbs_physical_eye": "tracker_session_left_RGB",
                        "crop_xywh": None, "capture_to_decode_alignment": "CLIP_INDEX_ONLY",
                        "best_effort_timestamp": i * 512} for i in range(2)],
    }}}
    return session, freeze, frame_map


def _vst_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    session = tmp_path / "chips_cards_hands_0915" / "cards_120_0915" / "001"
    raw = session / "raw"
    raw.mkdir(parents=True)
    (raw / "vst.h264").write_bytes(b"raw SBS test")
    (raw / "vst.qpc.ts.jsonl").write_text("14168718802000\n14168750818500\n", encoding="utf-8")
    (raw / "vst.ts.jsonl").write_text("1789481970850515900\n1789481970882763500\n", encoding="utf-8")
    _json(raw / "camera_params.json", {
        "width": 2048, "height": 1536, "outputWidth": 4096, "outputHeight": 1536,
        "fps": 30, "left": {"sourceIndex": 1, "fx": 936, "fy": 925,
                             "cx": 1011, "cy": 773}, "right": {"sourceIndex": 0},
    })
    _json(raw / "camera_params.meta.json", {"extrinsics_included": True})
    names = ("vst.h264", "vst.qpc.ts.jsonl", "vst.ts.jsonl",
             "camera_params.json", "camera_params.meta.json")
    session_id = "play_cards_0915_001"
    freeze = {"sessions": {session_id: {
        "source_path": str(session), "role": "development", "split_claim": "DEVELOPMENT",
        "files": {name: adapter.ref(raw / name) for name in names},
    }}}
    frame_map = {"sessions": {session_id: {
        "decoded_frames": 2, "time_mode": "TIMEBASE_UNVERIFIED",
        "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
        "frame_rows": [{"decoded_display_index": i, "sbs_physical_eye": "left_sourceIndex_1",
                        "crop_xywh": [2048, 0, 2048, 1536],
                        "capture_frame_id": None, "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
                        "qpc_raw_tick": [14168718802000, 14168750818500][i],
                        "capture_timestamp_raw": [1789481970850515900, 1789481970882763500][i]}
                       for i in range(2)],
    }}}
    return session, freeze, frame_map


def test_tracker_25_fps_reads_only_rgb_camera_slam_manifest(tmp_path: Path) -> None:
    session, freeze, frame_map = _tracker_fixture(tmp_path)
    spec = adapter.build_spec(session=session, session_id=session.name,
                              source_kind="tracker", input_freeze=freeze,
                              frame_time_map=frame_map)
    assert spec["fps_nominal"] == 25
    assert spec["frame_count"] == 2
    assert set(spec["allowed_refs"]) == {"video", "camera_params", "slam", "clip_manifest"}
    assert spec["time_mode"] == "CLIP_FRAME_INDEX_ONLY"
    assert not (session / "trackingData.txt").exists()
    assert not (session / "preprocess/all_data").exists()


def test_tracker_sha_drift_fails_closed(tmp_path: Path) -> None:
    session, freeze, frame_map = _tracker_fixture(tmp_path)
    (session / f"CameraRecord_{session.name}.mp4").write_bytes(b"changed")
    with pytest.raises(adapter.InputRejected, match="SHA drift"):
        adapter.build_spec(session=session, session_id=session.name,
                           source_kind="tracker", input_freeze=freeze,
                           frame_time_map=frame_map)


def test_tracker_rejects_out_of_session_rgb(tmp_path: Path) -> None:
    session, freeze, frame_map = _tracker_fixture(tmp_path)
    manifest = session / "clip_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"]["video"] = "../foreign.mp4"
    _json(manifest, payload)
    freeze["sessions"][session.name]["files"]["clip_manifest"] = adapter.ref(manifest)
    with pytest.raises((adapter.InputRejected, FileNotFoundError)):
        adapter.build_spec(session=session, session_id=session.name,
                           source_kind="tracker", input_freeze=freeze,
                           frame_time_map=frame_map)


def test_vst_physical_left_resize_and_unverified_clock(tmp_path: Path) -> None:
    session, freeze, frame_map = _vst_fixture(tmp_path)
    spec = adapter.build_spec(session=session, session_id="play_cards_0915_001",
                              source_kind="vst0915", input_freeze=freeze,
                              frame_time_map=frame_map)
    assert spec["crop_xywh"] == [2048, 0, 2048, 1536]
    assert spec["intrinsics"][0][0] == 936 * 1280 / 2048
    assert spec["time_mode"] == "TIMEBASE_UNVERIFIED"
    row = adapter.frame_status(spec["frame_rows_t0"][0], 0, spec, True, False)
    assert row["capture_frame_id"] is None
    assert row["qpc_raw_tick"] == 14168718802000
    assert row["hands"]["left"]["motion_source"] == "HAWOR_MODEL_PREDICTION"
    assert row["hands"]["right"]["motion_source"] == "MISSING"
    assert row["hands"]["right"]["wrist_position_valid"] is False
    assert not (session / "raw/pico.jsonl").exists()


def test_vst_sidecar_mismatch_and_eye_drift_fail_closed(tmp_path: Path) -> None:
    session, freeze, frame_map = _vst_fixture(tmp_path)
    raw = session / "raw"
    (raw / "vst.ts.jsonl").write_text("1\n", encoding="utf-8")
    freeze["sessions"]["play_cards_0915_001"]["files"]["vst.ts.jsonl"] = adapter.ref(raw / "vst.ts.jsonl")
    with pytest.raises(adapter.InputRejected, match="sidecar mismatch"):
        adapter.build_spec(session=session, session_id="play_cards_0915_001",
                           source_kind="vst0915", input_freeze=freeze,
                           frame_time_map=frame_map)
    (raw / "vst.ts.jsonl").write_text("1\n2\n", encoding="utf-8")
    freeze["sessions"]["play_cards_0915_001"]["files"]["vst.ts.jsonl"] = adapter.ref(raw / "vst.ts.jsonl")
    camera = raw / "camera_params.json"
    payload = json.loads(camera.read_text(encoding="utf-8"))
    payload["left"]["sourceIndex"] = 0
    _json(camera, payload)
    freeze["sessions"]["play_cards_0915_001"]["files"]["camera_params.json"] = adapter.ref(camera)
    with pytest.raises(adapter.InputRejected, match="physical-eye"):
        adapter.build_spec(session=session, session_id="play_cards_0915_001",
                           source_kind="vst0915", input_freeze=freeze,
                           frame_time_map=frame_map)


def test_vst_frame_map_count_and_time_drift_fail_closed(tmp_path: Path) -> None:
    session, freeze, frame_map = _vst_fixture(tmp_path)
    frame_map["sessions"]["play_cards_0915_001"]["frame_rows"].pop()
    with pytest.raises(adapter.InputRejected, match="frame-count disagreement"):
        adapter.build_spec(session=session, session_id="play_cards_0915_001",
                           source_kind="vst0915", input_freeze=freeze,
                           frame_time_map=frame_map)
