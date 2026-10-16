import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.pick_place_1008 import audit, calibration_fit as fit


def _known_answer():
    rng = np.random.default_rng(42)
    points = rng.uniform([-.20, -.15, .50], [.23, .20, .95], (48, 3))
    k = np.array([[300., 0., 320.], [0., 290., 240.], [0., 0., 1.]])
    parameters = np.array([.08, -.06, .04, .03, -.02, .04])
    truth = fit._transform_from_parameters(parameters, np.eye(4))
    pixels, _ = fit._project_head_points(points, k, truth)
    return points, k, parameters, truth, pixels, ["001"] * 24 + ["003"] * 24


def test_known_answer_recovers_six_dof_without_changing_k_time_or_scale():
    points, k, parameters, truth, pixels, sessions = _known_answer()
    original_k = k.copy()
    report = fit.fit_rigid_proxy_points(points, pixels, k, np.eye(4), sessions)
    assert report["status"] == "PASSED"
    assert np.allclose(report["T_head_camera_fitted"], truth, atol=1e-7)
    assert np.array_equal(k, original_k)
    assert report["after"]["p50_px"] < 1e-6
    assert report["jacobian_rank"] == 6
    assert report["boundary_hit_parameter_indices"] == []
    assert report["optimizer"]["loss"] == "soft_l1"
    assert report["optimizer"]["f_scale_px"] == 8


def test_robust_fit_records_outlier_and_bound_or_degenerate_case_requires_review():
    points, k, _, _, pixels, sessions = _known_answer()
    outlier_pixels = pixels.copy()
    outlier_pixels[0] += [100, -50]
    robust = fit.fit_rigid_proxy_points(points, outlier_pixels, k, np.eye(4), sessions)
    assert robust["after"]["p50_px"] < 3
    assert robust["after"]["max_px"] > 90
    outside = fit._transform_from_parameters([0, 0, 0, .40, 0, 0], np.eye(4))
    outside_pixels, _ = fit._project_head_points(points, k, outside)
    report = fit.fit_rigid_proxy_points(points, outside_pixels, k, np.eye(4), sessions)
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["boundary_hit_parameter_indices"]
    same = np.tile([0.1, 0.1, 0.6], (8, 1))
    same_uv, _ = fit._project_head_points(same, k, np.eye(4))
    assert fit.fit_rigid_proxy_points(same, same_uv, k, np.eye(4),
                                      ["001"] * 4 + ["003"] * 4)["status"] == "REVIEW_REQUIRED"


def test_too_few_points_and_single_session_are_rejected():
    points, k, _, _, pixels, sessions = _known_answer()
    with pytest.raises(fit.CalibrationFitError, match="TOO_FEW"):
        fit.fit_rigid_proxy_points(points[:7], pixels[:7], k, np.eye(4), sessions[:7])
    with pytest.raises(fit.CalibrationFitError, match="TWO_DEVELOPMENT"):
        fit.fit_rigid_proxy_points(points, pixels, k, np.eye(4), ["001"] * len(points))


def _annotation_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    image_path = tmp_path / "raw.png"
    image_path.write_bytes(b"synthetic evidence pin")
    image_ref = audit.file_ref(image_path)
    contexts, evidence_sessions, observations = {}, [], []
    for sid in fit.DEVELOPMENT_SESSIONS:
        serial = "DEVICE_A" if sid in ["001", "003"] else "DEVICE_B"
        head = np.repeat(np.eye(4)[None], 3, axis=0)
        hands = np.zeros((2, 3, 21, 3))
        hands[:] = [.1, .1, .6]
        contexts[sid] = {"row": {"session_id": sid}, "model": SimpleNamespace(device_serial=serial),
                         "qpc_ns": np.array([10, 20, 30]),
                         "native": {"head": head, "hands": hands,
                                    "head_valid": np.ones(3, bool),
                                    "hand_observed": np.ones((2, 3), bool)},
                         "prior_reference": {"used_prior_sha256": "fixed"}}
        evidence_sessions.append({"session_id": sid, "images": [
            {"raw_frame_index": frame, "image": image_ref} for frame in [0, 1]]})
        for frame in [0, 1]:
            for joint in [4, 8]:
                observations.append({
                    "session_id": sid, "device_serial": serial, "raw_frame_index": frame,
                    "qpc_ns": int(contexts[sid]["qpc_ns"][frame]), "image": image_ref,
                    "side": "right", "joint_index_21": joint,
                    "manus_node_id": int(fit.MAP_25_TO_21[joint]), "uv": [350., 300.],
                    "visibility": "VISIBLE", "point_type": "GLOVE_TIP",
                    "semantics": "VISUAL_PROXY", "side_confirmed": True,
                    "joint_mapping_confirmed": True})
    return {"observations": observations}, {"sessions": evidence_sessions}, contexts


def test_proxy_evidence_accepts_nonroot_and_excludes_explicit_occlusion(tmp_path, monkeypatch):
    annotations, evidence, contexts = _annotation_fixture(tmp_path, monkeypatch)
    hidden = copy.deepcopy(annotations["observations"][0])
    hidden.update(joint_index_21=12, manus_node_id=int(fit.MAP_25_TO_21[12]),
                  visibility="OCCLUDED", uv=None)
    annotations["observations"].append(hidden)
    records, excluded = fit._validated_correspondences(annotations, evidence, contexts)
    assert len(records) == 16
    assert len(excluded) == 1 and excluded[0]["reason"] == "EXPLICIT_OCCLUDED"
    assert all(record["joint_index_21"] != 0 for record in records)


@pytest.mark.parametrize("mutation", [
    {"joint_index_21": 0, "manus_node_id": 0},
    {"joint_mapping_confirmed": False},
    {"side_confirmed": False},
    {"qpc_ns": 999},
    {"session_id": "007"},
    {"point_type": "ANATOMICAL_GROUND_TRUTH"},
])
def test_wrong_root_side_clock_holdout_or_authority_is_rejected(tmp_path, monkeypatch, mutation):
    annotations, evidence, contexts = _annotation_fixture(tmp_path, monkeypatch)
    annotations["observations"][0].update(mutation)
    with pytest.raises(fit.CalibrationFitError):
        fit._validated_correspondences(annotations, evidence, contexts)


def test_each_of_four_fixed_sessions_requires_multiple_frames(tmp_path, monkeypatch):
    annotations, evidence, contexts = _annotation_fixture(tmp_path, monkeypatch)
    for record in annotations["observations"]:
        if record["session_id"] == "001":
            record["raw_frame_index"] = 0
            record["qpc_ns"] = 10
    with pytest.raises(fit.CalibrationFitError):
        fit._validated_correspondences(annotations, evidence, contexts)


def test_fit_artifacts_preserve_base_config_and_form_one_way_report_binding(tmp_path, monkeypatch):
    annotations, evidence, contexts = _annotation_fixture(tmp_path, monkeypatch)
    rng = np.random.default_rng(8)
    k_image = np.array([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]])
    truth = fit._transform_from_parameters([.06, -.04, .03, .02, -.01, .035], np.eye(4))
    for context in contexts.values():
        context["row"]["source_refs"] = {}
        context["native"]["hands"][:] = rng.uniform([-.2, -.15, .5], [.2, .15, .95],
                                                    context["native"]["hands"].shape)
        context["model"].K = np.diag([2., 2., 1.]) @ k_image
        context["model"].T_head_camera = np.eye(4)
        context["model"].T_head_camera_factory = np.eye(4)
        context["model"].camera_sha256 = "a" * 64
        context["K_image"] = k_image
    for observation in annotations["observations"]:
        context = contexts[observation["session_id"]]
        point = context["native"]["hands"][1, observation["raw_frame_index"],
                                           observation["joint_index_21"]]
        uv, _ = fit._project_head_points(point[None], k_image, truth)
        observation["uv"] = uv[0].tolist()
    config = {"devices": {serial: {"device_serial": serial, "eye": "right", "sbs_index": 1,
                                  "output_size": [1280, 960], "status": "DEVELOPMENT_CANDIDATE"}
                           for serial in ["DEVICE_A", "DEVICE_B"]}}
    manifest_path = tmp_path / "MANIFEST.json"
    config_path = tmp_path / "BASE.json"
    manifest_path.write_text('{"sessions":[]}')
    config_path.write_text(json.dumps(config))
    original_config = config_path.read_bytes()
    evidence.update(status="PASSED", holdout_used=False, image_size=[640, 480],
                    input_manifest=audit.file_ref(manifest_path),
                    base_camera_config=audit.file_ref(config_path))
    images_path = tmp_path / "SOURCE_IMAGES.json"
    images_path.write_text(json.dumps(evidence))
    annotations.update(input_manifest=audit.file_ref(manifest_path),
                       base_camera_config=audit.file_ref(config_path),
                       source_images=audit.file_ref(images_path))
    annotation_path = tmp_path / "ANNOTATIONS.json"
    annotation_path.write_text(json.dumps(annotations))
    monkeypatch.setattr(fit, "_load_inputs", lambda *args: ({}, copy.deepcopy(config), contexts))
    result = fit.run_camera_fit(manifest_path, annotation_path, config_path, tmp_path / "fit")
    assert result["status"] == "PASSED"
    report = json.loads(Path(result["calibration_report"]["path"]).read_text())
    fitted = json.loads(Path(result["fitted_camera_config"]["path"]).read_text())
    assert "fitted_camera_config" not in report
    assert report["K_fitted"] is False and report["time_fitted"] is False
    assert report["installation_prior_fitted"] is False and report["holdout_used"] is False
    assert config_path.read_bytes() == original_config
    for serial in fitted["devices"]:
        assert fitted["devices"][serial]["correction_evidence"] == result["calibration_report"]
        assert fitted["devices"][serial]["physical_eye_verified"] is False
        assert fitted["devices"][serial]["eye_identity_qualification"] == fit.EYE_QUALIFICATION
