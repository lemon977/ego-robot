import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from chaoyang.pick_place_1008.source import (
    SourceContractError, RAW_LH_TO_REP103, raw_pose_to_rep103, nearest,
    fill_points, load_raw_motion, read_installation, encode_qpc_frames,
    pts_from_qpc, json_rows,
)


def prior():
    return {"schema": "controller_to_wrist_v1",
            "convention": "wrist_pose = compose_pose(controller_pose, transform)",
            "controller_pose_frame": "pico_world_rh_x_forward_y_left_z_up",
            "left": {"pos": [.1, 0., 0.], "quat": [0., 0., 0., 1.]},
            "right": {"pos": [.1, 0., 0.], "quat": [0., 0., 0., 1.]}}


def write_rows(path, rows, bom=False):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8-sig" if bom else "utf-8")


def raw_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    qpc = np.array([1_000_000_000, 1_040_000_000, 1_080_000_000], np.int64)
    pose = {"pos": [0., 0., -1.], "quat": [0., 0., 0., 1.]}
    pico = [{"dev": "PICO_A", "recv_qpc_ns": int(t), "recv_wall_ns": int(t + 100),
             "data": {"Head": dict(pose, status=3), "Controller": {"left": pose, "right": pose},
                      "timeStampNs": int(t + 200)}} for t in qpc]
    write_rows(raw / "pico.jsonl", pico, bom=True)
    rows = []
    for t in qpc:
        for side, glove in [("left", 11), ("right", 22)]:
            nodes = np.zeros((25, 7))
            nodes[:, 6] = 1.
            nodes[1:, 0] = np.arange(1, 25) * .005
            rows.append({"side": side, "glove_id": glove, "node_count": 25,
                         "node_ids": list(range(25)), "nodes": nodes.tolist(),
                         "parent_ids": [0,0,1,2,3,0,5,6,7,8,0,10,11,12,13,0,15,16,17,18,0,20,21,22,23],
                         "chain_types": ["Hand"] + ["Thumb"]*4 + ["Index"]*5 + ["Middle"]*5 + ["Ring"]*5 + ["Pinky"]*5,
                         "joint_types": ["Invalid","MCP","PIP","DIP","TIP"] + ["MCP","PIP","IP","DIP","TIP"]*4,
                         "calibration_applied": True, "recv_qpc_ns": int(t),
                         "collector_recv_qpc_ns": int(t + 300)})
    write_rows(raw / "manus.jsonl", rows)
    (raw / "manus.meta.json").write_text(json.dumps({"calibration": {"left": {"glove_id": 11}, "right": {"glove_id": 22}}}))
    return qpc, pico, rows


def test_raw_coordinate_known_answer_and_quaternion():
    t, good = raw_pose_to_rep103({"pos": [1., 2., 3.], "quat": [0., 0., 0., 1.]})
    assert good
    np.testing.assert_allclose(t[:3, 3], [-3., -1., 2.])
    np.testing.assert_allclose(t[:3, :3], np.eye(3))
    assert np.linalg.det(RAW_LH_TO_REP103) == pytest.approx(1.)
    t, good = raw_pose_to_rep103({"pos": [0., 0., 0.], "quat": [0., np.sqrt(.5), 0., np.sqrt(.5)]})
    assert good
    np.testing.assert_allclose(t[:3, :3], [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], atol=1e-12)
    _, good = raw_pose_to_rep103({"pos": [0., 0., 0.], "quat": [0., 0., 0., 0.]})
    assert not good


def test_qpc_no_video_drop_and_tie_selects_earlier():
    rows, skew = nearest([100, 200, 300], [100, 150, 290, 310])
    np.testing.assert_array_equal(rows, [0, 0, 2, 2])
    np.testing.assert_array_equal(skew, [0, -50, 10, -10])
    with pytest.raises(SourceContractError, match="DUPLICATE"):
        nearest([100, 100], [100, 101])
    with pytest.raises(SourceContractError, match="INTEGER"):
        nearest([100., 200.], [100, 101])


def test_raw_composition_root_mapping_and_independent_validity(tmp_path):
    qpc, pico, rows = raw_fixture(tmp_path)
    rows[2]["calibration_applied"] = False
    write_rows(tmp_path / "raw/manus.jsonl", rows)
    native, mapping, quality = load_raw_motion(tmp_path, qpc, expected_serial="PICO_A", prior=prior())
    assert native["hands"].shape == (2, 3, 21, 3)
    np.testing.assert_allclose(native["hands"][0, 0, 0], [1.1, 0., 0.])
    np.testing.assert_allclose(native["hands"][0, 0, 4], [1.12, 0., 0.])
    assert not native["hand_observed"][0, 1]
    assert native["hand_observed"][1, 1]
    assert native["hand_source"][0, 1] == "INTERP_SHORT"
    assert not native["joint_observed_local"][:, :, 0].any()
    np.testing.assert_array_equal(mapping["pico_source_line"], [1, 2, 3])
    assert quality["complete_coverage"] == pytest.approx(2/3)
    assert not quality["passed"]


def test_wrong_device_or_glove_is_rejected(tmp_path):
    qpc, pico, rows = raw_fixture(tmp_path)
    with pytest.raises(SourceContractError, match="DEVICE_IDENTITY"):
        load_raw_motion(tmp_path, qpc, expected_serial="PICO_B", prior=prior())
    rows[0]["glove_id"] = 33
    write_rows(tmp_path / "raw/manus.jsonl", rows)
    with pytest.raises(SourceContractError, match="GLOVE_ID"):
        load_raw_motion(tmp_path, qpc, expected_serial="PICO_A", prior=prior())


def test_short_gap_uses_real_time_long_gap_diagnostic_only():
    values = np.array([[[0., 0., 0.]], [[np.nan]*3], [[1., 0., 0.]]])
    short = np.array([0, 10_000_000, 100_000_000], np.int64)
    filled, labels, valid = fill_points(values, [True, False, True], short)
    assert filled[1, 0, 0] == pytest.approx(.5 - .5 * np.cos(.1 * np.pi))
    assert labels[1] == "INTERP_SHORT" and valid[1]
    long = np.array([0, 100_000_000, 1_000_000_000], np.int64)
    _, labels, valid = fill_points(values, [True, False, True], long)
    assert labels[1] == "DIAGNOSTIC_LONG_HOLD" and not valid[1]


def test_hdf5_is_attribute_only(tmp_path):
    class AttributeOnly:
        attrs = {"controller_to_wrist_calibration": json.dumps(prior())}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, key): raise AssertionError("HDF5 dataset read forbidden")
        def __iter__(self): raise AssertionError("HDF5 dataset iteration forbidden")
    (tmp_path / "dataset.hdf5").write_bytes(b"attribute fixture")
    with patch("h5py.File", return_value=AttributeOnly()):
        saved, evidence = read_installation(tmp_path)
    assert saved == prior()
    assert evidence["attribute"] == "controller_to_wrist_calibration"


def test_pyav_qpc_pts_full_roundtrip(tmp_path):
    qpc = np.array([10_000_000_000, 10_040_000_400, 10_077_000_000, 10_120_000_700], np.int64)
    np.testing.assert_array_equal(pts_from_qpc(qpc), [0, 40000, 77000, 120001])
    frames = [np.full((48, 64, 3), i * 50, np.uint8) for i in range(4)]
    result = encode_qpc_frames(tmp_path / "vfr.mp4", frames, qpc, width=64, height=48)
    assert result["decoded_frames"] == 4
    assert result["pts_roundtrip_max_error_seconds"] <= 1e-6


def test_bom_lines_keep_original_source_line(tmp_path):
    path = tmp_path / "qpc.jsonl"
    path.write_text('100\n\n200\n', encoding="utf-8-sig")
    assert json_rows(path) == [(1, 100), (3, 200)]


def test_prepare_preserves_native_qpc_and_clean_reads_original_only(tmp_path):
    import av
    import h5py
    import cv2
    from chaoyang.pick_place_1008.source import prepare_session, load_prepared, load_rgb_frames
    from chaoyang.pick_place_1008.camera import sha256
    from chaoyang.pipeline.pico_manus_motion_v2 import REP103_HEAD_TO_OPENXR
    source = tmp_path / "001"
    source.mkdir()
    qpc, _, _ = raw_fixture(source)
    (source / "manifest.json").write_text(json.dumps({"session": "pick_place_1008_001"}))
    write_rows(source / "raw/vst.qpc.ts.jsonl", qpc.tolist(), bom=True)
    write_rows(source / "raw/vst.ts.jsonl", (qpc + 100000).tolist())
    with h5py.File(source / "dataset.hdf5", "w") as f:
        f.attrs["controller_to_wrist_calibration"] = json.dumps(prior())
    e = np.eye(4)
    e[:3, :3] = REP103_HEAD_TO_OPENXR.T
    eye = {"fx": 40., "fy": 40., "cx": 32., "cy": 24., "extrinsic": e[:3].reshape(-1).tolist()}
    cp = source / "raw/camera_params.json"
    cp.write_text(json.dumps({"width": 64, "height": 48, "left": eye, "right": eye}))
    (source / "raw/camera_params.meta.json").write_text(json.dumps({"device_serial": "PICO_A", "sha256": sha256(cp)}))
    video = source / "raw/vst.h264"
    with av.open(str(video), "w", format="h264") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 128, 48, "yuv420p"
        stream.options = {"crf": "8", "bf": "0"}
        for i in range(3):
            pixels = np.zeros((48, 128, 3), np.uint8)
            pixels[:, :64] = [10+i*10, 90, 180]
            pixels[:, 64:] = [180, 90, 10+i*10]
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    config = {"eye": "left", "sbs_index": 0, "upright_axes": "identity",
              "image_mode": "encoded_crop_resize_only", "extrinsic_convention": "head_to_camera",
              "status": "DEVELOPMENT_CANDIDATE", "output_size": [64, 48]}
    session = prepare_session(source, tmp_path / "prepared", config, expected_frames=3)
    assert session.frame_count == 3 and session.fps == pytest.approx(25.)
    np.testing.assert_array_equal(session.qpc_ns, qpc)
    replay = load_prepared(tmp_path / "prepared")
    np.testing.assert_array_equal(replay.hands, session.hands)
    assert replay.calibration["hdf5_datasets_read"] is False
    raw = load_rgb_frames(session, width=64, height=48)
    original = []
    with av.open(str(video)) as c:
        for frame in c.decode(video=0): original.append(frame.to_ndarray(format="rgb24")[:, :64])
    np.testing.assert_array_equal(raw, np.stack(original))
    # A deliberately broken prepared MP4 cannot influence the Clean RGB producer.
    session.video.write_bytes(b"review-only intermediate deliberately broken")
    np.testing.assert_array_equal(load_rgb_frames(session, 64, 48), raw)


def test_malformed_single_manus_frame_keeps_other_side(tmp_path):
    qpc, _, rows = raw_fixture(tmp_path)
    rows[2]["nodes"] = "malformed one frame"
    write_rows(tmp_path / "raw/manus.jsonl", rows)
    native, _, _ = load_raw_motion(tmp_path, qpc, expected_serial="PICO_A", prior=prior())
    assert not native["hand_observed"][0, 1]
    assert native["hand_observed"][1].all()
    assert np.isfinite(native["hands"]).all()


def test_projection_cannot_access_holdout():
    from chaoyang.pick_place_1008.calibration import run_projection_audit
    with pytest.raises(SourceContractError, match="FIXED_DEVELOPMENT_READSET"):
        run_projection_audit(Path("/not-used"), Path("/not-used"), {}, dev_sessions=["007"])
