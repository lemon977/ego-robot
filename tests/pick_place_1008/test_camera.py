import copy
import numpy as np
import pytest
from chaoyang.pick_place_1008.camera import CameraContractError, normalize_camera
from chaoyang.pipeline.pico_manus_motion_v2 import REP103_HEAD_TO_OPENXR


def fixture(nested=False):
    e = np.eye(4)
    e[:3, :3] = REP103_HEAD_TO_OPENXR.T
    e[0, 3] = -.06
    intr = {"fx": 1000., "fy": 1000., "cx": 1024., "cy": 768.}
    if nested:
        p = {"reference_resolution": {"w": 2048, "h": 1536},
             "left": {"intrinsics": intr}, "right": {"intrinsics": intr},
             "extrinsics": {"left": e.tolist(), "right": e.tolist()}}
    else:
        p = {"width": 2048, "height": 1536,
             "left": {**intr, "extrinsic": e[:3].reshape(-1).tolist()},
             "right": {**intr, "extrinsic": e[:3].reshape(-1).tolist()}}
    m = {"device_serial": "PICO_A", "sha256": "ABC", "extrinsics_included": not nested}
    c = {"eye": "left", "sbs_index": 0, "upright_axes": "identity",
         "image_mode": "encoded_crop_resize_only", "extrinsic_convention": "head_to_camera",
         "status": "DEVELOPMENT_CANDIDATE"}
    return p, m, c


def run(p, m, c):
    return normalize_camera(p, m, device_serial="PICO_A", camera_sha256="ABC", config=c)


def test_both_schemas_same_transform_and_K():
    a, b = run(*fixture()), run(*fixture(True))
    np.testing.assert_allclose(a.K, b.K)
    np.testing.assert_allclose(a.T_head_camera, b.T_head_camera)
    np.testing.assert_allclose(a.K, [[625., 0., 640.], [0., 625., 480.], [0., 0., 1.]])
    assert a.crop == (0, 0, 2048, 1536)
    assert np.linalg.norm(a.T_head_camera[:3, 3]) == pytest.approx(.06)
    assert not b.calibration["source_metadata_extrinsics_included"]
    assert b.calibration["camera_schema"] == "nested_4x4"


def test_missing_route_never_inherits_old_eye():
    p, m, c = fixture()
    del c["sbs_index"]
    with pytest.raises(CameraContractError, match="EYE_ROUTE"):
        run(p, m, c)


def test_device_snapshot_and_explicit_source_index_conflict():
    p, m, c = fixture()
    for key, value in [("device_serial", "PICO_B"), ("sha256", "OTHER")]:
        bad = dict(m, **{key: value})
        with pytest.raises(CameraContractError):
            run(p, bad, c)
    p["left"]["sourceIndex"] = 1
    with pytest.raises(CameraContractError, match="SOURCE_INDEX_CONFLICT"):
        run(p, m, c)


def test_reflection_and_nonuniform_resize_rejected():
    p, m, c = fixture(True)
    p["extrinsics"]["left"][0][:3] = [-v for v in p["extrinsics"]["left"][0][:3]]
    with pytest.raises(CameraContractError, match="NONRIGID"):
        run(p, m, c)
    p, m, c = fixture()
    with pytest.raises(CameraContractError, match="NONUNIFORM"):
        normalize_camera(p, m, device_serial="PICO_A", camera_sha256="ABC", config=c, output_size=(1280, 720))


def test_no_lens_remap_and_correction_needs_evidence():
    p, m, c = fixture(True)
    p["video_info"] = {"is_distorted": True}
    p["left"]["distortion"] = {"model": "equiDis62", "coeffs": [0.] * 8}
    a = run(p, m, c)
    assert a.calibration["lens_remap"] is False
    c["T_head_camera_override"] = np.eye(4).tolist()
    with pytest.raises(CameraContractError, match="EVIDENCE"):
        run(p, m, c)
    c["correction_evidence"] = {"development_sessions": ["001", "003"], "status": "VISUAL_ESTIMATE"}
    np.testing.assert_array_equal(run(p, m, c).T_head_camera, np.eye(4))


def test_upright_is_basis_change_not_second_pixel_rotation():
    p, m, c = fixture()
    a = run(p, m, c)
    c["upright_axes"] = "capture_sensor_from_upright"
    b = run(p, m, c)
    np.testing.assert_allclose(a.T_head_camera[:3, 3], b.T_head_camera[:3, 3])
    assert a.crop == b.crop
    np.testing.assert_allclose(b.T_head_camera[:3, :3], [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
