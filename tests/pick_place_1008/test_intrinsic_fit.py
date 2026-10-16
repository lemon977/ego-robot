import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.pick_place_1008 import audit, calibration_fit as calibration, intrinsic_fit as fit


def _known_answer():
    rng = np.random.default_rng(70)
    points = rng.uniform([-.25, -.20, .55], [.25, .20, 1.15], (48, 3))
    seed = np.array([[300., 0., 320.], [0., 290., 240.], [0., 0., 1.]])
    truth = fit._k_from_parameters([1.65, 1.72, 21., -14.], seed)
    transform = calibration._transform_from_parameters([.01, -.02, .015, .02, -.01, .015],
                                                       np.eye(4))
    pixels, _ = calibration._project_head_points(points, truth, transform)
    return points, seed, truth, transform, pixels, ["001"] * 24 + ["003"] * 24


def test_known_answer_recovers_k_while_extrinsics_and_metric_points_stay_fixed():
    points, seed, truth, transform, pixels, sessions = _known_answer()
    originals = [value.copy() for value in [points, seed, transform]]
    report = fit.fit_intrinsic_proxy_points(points, pixels, seed, transform, sessions)
    assert report["status"] == "PASSED"
    assert np.allclose(report["K_image_fitted"], truth, atol=1e-7)
    assert report["jacobian_rank"] == 4
    assert report["boundary_hit_parameter_indices"] == []
    assert report["after"]["p50_px"] < 1e-6
    assert report["diagnosis"] == "ENCODED_DOMAIN_FACTORY_K_VISUAL_MISMATCH_SUPPORTED"
    assert report["optimizer"]["loss"] == "soft_l1" and report["optimizer"]["f_scale_px"] == 8
    for current, original in zip([points, seed, transform], originals):
        assert np.array_equal(current, original)


def test_robust_outlier_is_retained_in_report_and_boundary_is_not_accepted():
    points, seed, _, transform, pixels, sessions = _known_answer()
    noisy = pixels.copy()
    noisy[0] += [150., -90.]
    robust = fit.fit_intrinsic_proxy_points(points, noisy, seed, transform, sessions)
    assert robust["after"]["p50_px"] < 1
    assert robust["after"]["max_px"] > 150
    outside = fit._k_from_parameters([3.6, 1.72, 21., -14.], seed)
    uv, _ = calibration._project_head_points(points, outside, transform)
    rejected = fit.fit_intrinsic_proxy_points(points, uv, seed, transform, sessions)
    assert rejected["status"] == "REVIEW_REQUIRED"
    assert 0 in rejected["boundary_hit_parameter_indices"]


def test_rank_deficient_or_nonpositive_fixed_depth_requires_review():
    seed = np.array([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]])
    points = np.tile([.1, .1, .7], (8, 1))
    uv, _ = calibration._project_head_points(points, seed, np.eye(4))
    report = fit.fit_intrinsic_proxy_points(points, uv, seed, np.eye(4), ["001"] * 4 + ["003"] * 4)
    assert report["status"] == "REVIEW_REQUIRED" and report["jacobian_rank"] < 4
    points[:4, 2] = -.7
    pixels, _ = calibration._project_head_points(points, seed, np.eye(4))
    bad_depth = fit.fit_intrinsic_proxy_points(points, pixels, seed, np.eye(4),
                                             ["001"] * 4 + ["003"] * 4)
    assert bad_depth["status"] == "REVIEW_REQUIRED"
    assert bad_depth["after_nonpositive_depth_count"] == 4


@pytest.mark.parametrize("mutation", ["too_few", "single_session", "skew_k", "nonfinite"])
def test_invalid_numeric_inputs_are_rejected(mutation):
    points, seed, _, transform, pixels, sessions = _known_answer()
    if mutation == "too_few":
        points, pixels, sessions = points[:7], pixels[:7], sessions[:7]
    elif mutation == "single_session":
        sessions = ["001"] * len(points)
    elif mutation == "skew_k":
        seed[0, 1] = 1
    elif mutation == "nonfinite":
        points[0, 0] = np.nan
    with pytest.raises(fit.IntrinsicVerificationError):
        fit.fit_intrinsic_proxy_points(points, pixels, seed, transform, sessions)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    manifest_path, config_path = tmp_path / "MANIFEST.json", tmp_path / "BASE.json"
    config = {"devices": {serial: {"device_serial": serial, "eye": "right", "sbs_index": 1,
                                 "output_size": [1280, 960], "status": "DEVELOPMENT_CANDIDATE"}
                          for serial in ["DEVICE_A", "DEVICE_B"]}}
    _write(manifest_path, {"sessions": []})
    _write(config_path, config)
    image_path = tmp_path / "raw.png"
    image_path.write_bytes(b"fixed synthetic image reference")
    image_ref = audit.file_ref(image_path)
    rng = np.random.default_rng(17)
    k = np.array([[300., 0., 320.], [0., 290., 240.], [0., 0., 1.]])
    truth = fit._k_from_parameters([1.6, 1.7, 20., -10.], k)
    contexts, evidence_sessions, observations = {}, [], []
    for sid in calibration.DEVELOPMENT_SESSIONS:
        serial = "DEVICE_A" if sid in ["001", "003"] else "DEVICE_B"
        hands = rng.uniform([-.15, -.15, .7], [.15, .15, 1.1], (2, 2, 21, 3))
        contexts[sid] = {
            "row": {"session_id": sid, "source_refs": {}}, "qpc_ns": np.array([10, 20]),
            "native": {"head": np.repeat(np.eye(4)[None], 2, axis=0), "hands": hands,
                       "head_valid": np.ones(2, bool), "hand_observed": np.ones((2, 2), bool)},
            "prior_reference": {"used_prior_sha256": "fixed"},
            "model": SimpleNamespace(device_serial=serial, K=np.diag([2., 2., 1.]) @ k,
                                     T_head_camera=np.eye(4), T_head_camera_factory=np.eye(4),
                                     width=1280, height=960, camera_sha256="a" * 64),
            "K_image": k,
        }
        evidence_sessions.append({"session_id": sid, "images": [
            {"raw_frame_index": frame, "image": image_ref} for frame in [0, 1]]})
        for frame in [0, 1]:
            for joint in [4, 8, 12, 16]:
                uv, _ = calibration._project_head_points(hands[1, frame, joint][None],
                                                         truth, np.eye(4))
                observations.append({
                    "session_id": sid, "device_serial": serial, "raw_frame_index": frame,
                    "qpc_ns": [10, 20][frame], "image": image_ref,
                    "side": "right", "joint_index_21": joint,
                    "manus_node_id": int(calibration.MAP_25_TO_21[joint]), "uv": uv[0].tolist(),
                    "visibility": "VISIBLE", "point_type": "GLOVE_TIP",
                    "semantics": "VISUAL_PROXY", "side_confirmed": True,
                    "joint_mapping_confirmed": True,
                })
    pins = {"input_manifest": audit.file_ref(manifest_path),
            "base_camera_config": audit.file_ref(config_path)}
    evidence = {**pins, "status": "PASSED", "holdout_used": False, "image_size": [640, 480],
                "sessions": evidence_sessions}
    evidence_path = tmp_path / "SOURCE_IMAGES.json"
    _write(evidence_path, evidence)
    annotations = {**pins, "source_images": audit.file_ref(evidence_path),
                   "observations": observations}
    annotations_path = tmp_path / "ANNOTATIONS.json"
    _write(annotations_path, annotations)
    records, _ = calibration._validated_correspondences(annotations, evidence, contexts)
    report_pins = {**pins, "source_images": audit.file_ref(evidence_path),
                   "annotations": audit.file_ref(annotations_path)}
    camera_path = tmp_path / "camera/attempt_0001/candidate/CALIBRATION_REPORT.json"
    install_path = tmp_path / "installation/attempt_0001/candidate/INSTALLATION_REPORT.json"
    camera = {**report_pins, "schema": "pick_place_device_camera_rigid_visual_fit_v1",
              "status": "REVIEW_REQUIRED", "holdout_used": False,
              "time_fitted": False, "scale_fitted": False, "lens_remap": False,
              "correspondences": records}
    install = {**report_pins, "schema": fit.INSTALLATION_SCHEMA, "status": "REVIEW_REQUIRED",
               "holdout_used": False, "time_fitted": False, "scale_fitted": False,
               "lens_remap": False}
    def seal(path, report, task_id, field):
        _write(path, report)
        packet_path = tmp_path / "tasks/current" / task_id / "TASK_PACKET.json"
        packet = {"task_id": task_id, "run_signature": task_id + "_signature"}
        _write(packet_path, packet)
        stage_path = path.parent / "RESULT.json"
        _write(stage_path, {"status": "REVIEW_REQUIRED", field: audit.file_ref(path)})
        outer_path = path.parent.parent / "RESULT.json"
        _write(outer_path, {
            "task_id": task_id, "status": "FAILED_RUNTIME_FINAL", "validation_errors": [],
            "run_signature": packet["run_signature"], "task_packet": audit.file_ref(packet_path),
            "stage_result": audit.file_ref(stage_path),
            "required_outputs": {"candidate/RESULT.json": audit.file_ref(stage_path),
                                 f"candidate/{path.name}": audit.file_ref(path)},
        })
        _write(tmp_path / "tasks/receipts" / f"{task_id.upper()}.json",
               {"status": "FAILED_RUNTIME_FINAL", "result": audit.file_ref(outer_path)})
    def reseal():
        seal(camera_path, camera, "CAMERA_FAILED", "calibration_report")
        install["camera_fit_report"] = audit.file_ref(camera_path)
        seal(install_path, install, "INSTALLATION_FAILED", "installation_report")
    reseal()
    monkeypatch.setattr(calibration, "_load_inputs", lambda *args: ({}, copy.deepcopy(config), contexts))
    return SimpleNamespace(
        tmp=tmp_path, manifest=manifest_path, annotations=annotations_path, config_path=config_path,
        config=config, contexts=contexts, records=records, camera_path=camera_path,
        install_path=install_path, camera=camera, install=install, seal=seal, reseal=reseal,
    )


def _run(fixture):
    return fit.run_intrinsic_verification(fixture.manifest, fixture.annotations, fixture.config_path,
                                          fixture.camera_path, fixture.install_path, fixture.tmp / "out")


def test_full_artifacts_bind_both_failures_and_only_override_k_with_correct_resize(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    original = fixture.config_path.read_bytes()
    result = _run(fixture)
    assert result["status"] == "PASSED"
    report = json.loads(Path(result["intrinsic_report"]["path"]).read_text())
    config = json.loads(Path(result["fitted_camera_config"]["path"]).read_text())
    assert report["camera_fit_report"] == audit.file_ref(fixture.camera_path)
    assert report["installation_fit_report"] == audit.file_ref(fixture.install_path)
    assert report["preceding_failure_producers"]["camera"]["terminal_receipt"]
    assert report["preceding_failure_producers"]["installation"]["terminal_receipt"]
    assert "fitted_camera_config" not in report
    assert report["camera_extrinsics_fitted"] is False and report["installation_prior_fitted"] is False
    assert report["qpc_modified"] is False and report["metric_scale_modified"] is False
    assert report["physical_eye_verified"] is False and report["training_eligible"] is False
    assert fixture.config_path.read_bytes() == original
    for serial, selected in config["devices"].items():
        assert "T_head_camera_override" not in selected and "installation_prior_override" not in selected
        assert selected["correction_evidence"] == result["intrinsic_report"]
        assert selected["eye_identity_qualification"] == calibration.EYE_QUALIFICATION
        assert selected["metric_calibration_verified"] is False
        assert np.allclose(selected["K_override"],
                           np.diag([2., 2., 1.]) @ np.asarray(report["devices"][serial]["K_image_fitted"]))


@pytest.mark.parametrize("mutation", [
    "camera_passed", "installation_unbound", "required_output", "packet_signature",
    "receipt_drift", "correspondences_drift", "missing_installation",
])
def test_missing_or_unsealed_or_wrong_order_failed_evidence_is_rejected(tmp_path, monkeypatch, mutation):
    fixture = _fixture(tmp_path, monkeypatch)
    if mutation == "camera_passed":
        fixture.camera["status"] = "PASSED"
        fixture.reseal()
    elif mutation == "installation_unbound":
        fixture.install["camera_fit_report"] = {"path": "other", "bytes": 1, "sha256": "x"}
        fixture.seal(fixture.install_path, fixture.install, "INSTALLATION_FAILED", "installation_report")
    elif mutation == "required_output":
        outer_path = fixture.camera_path.parent.parent / "RESULT.json"
        outer = json.loads(outer_path.read_text())
        outer["required_outputs"]["candidate/CALIBRATION_REPORT.json"]["sha256"] = "0" * 64
        _write(outer_path, outer)
    elif mutation == "packet_signature":
        outer_path = fixture.camera_path.parent.parent / "RESULT.json"
        outer = json.loads(outer_path.read_text())
        outer["run_signature"] = "drift"
        _write(outer_path, outer)
    elif mutation == "receipt_drift":
        receipt_path = tmp_path / "tasks/receipts/CAMERA_FAILED.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["result"]["sha256"] = "0" * 64
        _write(receipt_path, receipt)
    elif mutation == "correspondences_drift":
        fixture.camera["correspondences"][0]["point_head_m"][0] += .1
        fixture.reseal()
    elif mutation == "missing_installation":
        fixture.install_path.unlink()
    with pytest.raises((fit.IntrinsicVerificationError, FileNotFoundError)):
        _run(fixture)
    assert not (tmp_path / "out/RESULT.json").exists()


@pytest.mark.parametrize("override", ["K_override", "T_head_camera_override", "installation_prior_override"])
def test_corrected_camera_or_installation_cannot_enter_fixed_factory_diagnosis(tmp_path, monkeypatch, override):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture.config["devices"]["DEVICE_A"][override] = "must reject"
    with pytest.raises(fit.IntrinsicVerificationError, match="MUST_BE_FIXED"):
        _run(fixture)


def test_holdout_remains_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    annotations = json.loads(fixture.annotations.read_text())
    annotations["observations"][0]["session_id"] = "007"
    _write(fixture.annotations, annotations)
    with pytest.raises(calibration.CalibrationFitError, match="FIXED_DEVELOPMENT"):
        _run(fixture)

def test_one_raw_frame_in_any_development_session_is_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    annotations = json.loads(fixture.annotations.read_text())
    annotations["observations"] = [record for record in annotations["observations"]
                                  if record["session_id"] != "001" or record["raw_frame_index"] == 0]
    _write(fixture.annotations, annotations)
    with pytest.raises(calibration.CalibrationFitError, match="MULTIPLE_RAW_FRAMES"):
        _run(fixture)
