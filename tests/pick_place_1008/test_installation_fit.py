import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chaoyang.pick_place_1008 import audit, calibration_fit as calibration, installation_fit as fit, source


def _known_answer():
    rng = np.random.default_rng(87)
    n = 48
    controller = np.repeat(np.eye(4)[None], n, axis=0)
    controller[:, :3, :3] = Rotation.from_rotvec(rng.uniform(-.5, .5, (n, 3))).as_matrix()
    controller[:, :3, 3] = rng.uniform([-.2, -.15, .65], [.2, .15, 1.2], (n, 3))
    points = rng.uniform(-.11, .11, (n, 3))
    seed = fit._transform([.12, -.08, .03, .03, -.01, .02], np.eye(4))
    truth = fit._transform([-.11, .09, .07, .04, -.03, .02], seed)
    k = np.array([[320., 0., 320.], [0., 310., 240.], [0., 0., 1.]])
    uv, _, _ = fit._project(controller, points, k, truth)
    return controller, points, k, seed, truth, uv, ["001"] * 24 + ["003"] * 24


def test_known_answer_recovers_controller_local_rigid_correction_without_mutation():
    controller, points, k, seed, truth, uv, sessions = _known_answer()
    originals = [a.copy() for a in (controller, points, k, seed, uv)]
    result = fit.fit_installation_proxy_points(controller, points, uv, k, seed, sessions,
                                               np.full(len(points), 12.))
    assert result["status"] == "PASSED"
    assert np.allclose(result["T_controller_wrist_fitted"], truth, atol=1e-7)
    assert result["jacobian_rank"] == 6 and result["boundary_hit_parameter_indices"] == []
    assert result["after"]["p50_px"] < 1e-6
    assert result["delta_applied_on_left_in_controller_frame"] is True
    for a, b in zip((controller, points, k, seed, uv), originals):
        assert np.array_equal(a, b)


def test_boundary_hit_and_rank_deficiency_fail_closed():
    controller, points, k, seed, _, _, sessions = _known_answer()
    outside = fit._transform([-.1, .1, .05, .27, -.02, .01], seed)
    uv, _, _ = fit._project(controller, points, k, outside)
    result = fit.fit_installation_proxy_points(controller, points, uv, k, seed, sessions)
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["boundary_hit_parameter_indices"]
    repeated_controller = np.repeat(controller[:1], 8, axis=0)
    repeated_points = np.repeat(points[:1], 8, axis=0)
    repeated_uv, _, _ = fit._project(repeated_controller, repeated_points, k, seed)
    deficient = fit.fit_installation_proxy_points(repeated_controller, repeated_points,
                                                  repeated_uv, k, seed, ["001"] * 4 + ["003"] * 4)
    assert deficient["status"] == "REVIEW_REQUIRED" and deficient["jacobian_rank"] < 6


def test_outlier_retained_and_positive_depth_required():
    controller, points, k, seed, _, uv, sessions = _known_answer()
    noisy = uv.copy()
    noisy[0] += [140., -80.]
    robust = fit.fit_installation_proxy_points(controller, points, noisy, k, seed, sessions)
    assert robust["after"]["p50_px"] < 1 and robust["after"]["max_px"] > 100
    controller[:24, 2, 3] *= -1
    pixels, _, _ = fit._project(controller, points, k, seed)
    behind = fit.fit_installation_proxy_points(controller, points, pixels, k, seed, sessions)
    assert behind["status"] == "REVIEW_REQUIRED" and behind["after_nonpositive_depth_count"] == 24


@pytest.mark.parametrize("mutation", ["too_few", "single_session", "nonfinite", "nonrigid", "metre_domain", "uncertainty"])
def test_invalid_numeric_domains_rejected(mutation):
    controller, points, k, seed, _, uv, sessions = _known_answer()
    sigma = np.full(len(points), 12.)
    if mutation == "too_few":
        controller, points, uv, sessions, sigma = controller[:7], points[:7], uv[:7], sessions[:7], sigma[:7]
    elif mutation == "single_session":
        sessions = ["001"] * len(points)
    elif mutation == "nonfinite":
        points[0, 0] = np.nan
    elif mutation == "nonrigid":
        controller[0, 0, 0] += .2
    elif mutation == "metre_domain":
        points[0, 0] = 50.
    elif mutation == "uncertainty":
        sigma[0] = 7.99
    with pytest.raises(fit.InstallationFitError):
        fit.fit_installation_proxy_points(controller, points, uv, k, seed, sessions, sigma)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _prior(seed_matrices):
    return {"schema": "controller_to_wrist_v1",
            "convention": "wrist_pose = compose_pose(controller_pose, transform)",
            "controller_pose_frame": "pico_world_rh_x_forward_y_left_z_up",
            **{side: {"pos": matrix[:3, 3].tolist(),
                      "quat": Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()}
               for side, matrix in zip(source.SIDES, seed_matrices)}}


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    manifest_path, config_path = tmp_path / "MANIFEST.json", tmp_path / "BASE.json"
    config = {"devices": {serial: {"device_serial": serial, "eye": "right", "sbs_index": 1,
                                   "status": "DEVELOPMENT_CANDIDATE", "output_size": [1280, 960]}
                          for serial in ["DEVICE_A", "DEVICE_B"]}}
    _write(manifest_path, {"sessions": []})
    _write(config_path, config)
    image = tmp_path / "raw.png"
    image.write_bytes(b"synthetic unmodified raw reference")
    image_ref = audit.file_ref(image)
    rng = np.random.default_rng(33)
    k = np.array([[310., 0., 320.], [0., 300., 240.], [0., 0., 1.]])
    fixed_camera = fit._transform([.01, -.01, .02, .02, 0., .01], np.eye(4))
    contexts, images, observations, truth_by_group = {}, [], [], {}
    for sid in calibration.DEVELOPMENT_SESSIONS:
        serial = "DEVICE_A" if sid in ["001", "003"] else "DEVICE_B"
        seed = np.array([fit._transform([.05, -.04, .02, .02, -.01, .03], np.eye(4)),
                         fit._transform([-.03, .02, .04, -.01, .02, .025], np.eye(4))])
        prior = _prior(seed)
        prior_sha = ("a" if serial == "DEVICE_A" else "b") * 64
        truth = np.array([fit._transform([.08, -.04, .06, .025, -.02, .015], seed[0]),
                          fit._transform([-.05, .07, -.03, -.02, .01, .03], seed[1])])
        truth_by_group[serial + ":" + prior_sha] = truth
        head = np.array([fit._transform([.02 * frame, -.02, .01, .01, .015, -.01], np.eye(4))
                         for frame in [0, 1]])
        controller = np.repeat(np.eye(4)[None, None], 4, axis=0).reshape(2, 2, 4, 4)
        controller[:, :, :3, :3] = Rotation.from_rotvec(rng.uniform(-.5, .5, (4, 3))).as_matrix().reshape(2, 2, 3, 3)
        controller[:, :, :3, 3] = rng.uniform([-.18, -.1, .7], [.18, .15, 1.1], (2, 2, 3))
        local = rng.uniform(-.1, .1, (2, 2, 25, 3))
        local[:, :, 0] = 0.
        hands = np.empty((2, 2, 21, 3))
        for side_idx in [0, 1]:
            for frame in [0, 1]:
                wrist = controller[frame, side_idx] @ seed[side_idx]
                hands[side_idx, frame] = (local[frame, side_idx, calibration.MAP_25_TO_21]
                                         @ wrist[:3, :3].T + wrist[:3, 3])
        contexts[sid] = {
            "row": {"session_id": sid, "source_refs": {}},
            "qpc_ns": np.array([10, 20]), "selected_config": config["devices"][serial],
            "prior": prior,
            "prior_reference": {"prior_sha256": prior_sha, "used_prior_sha256": prior_sha},
            "native": {"head": head, "controller": controller, "manus_local_25": local, "hands": hands,
                       "head_valid": np.ones(2, bool), "hand_observed": np.ones((2, 2), bool)},
            "model": SimpleNamespace(device_serial=serial, K=np.diag([2., 2., 1.]) @ k,
                                     T_head_camera=fixed_camera, T_head_camera_factory=fixed_camera,
                                     width=1280, height=960, camera_sha256="c" * 64),
            "K_image": k}
        images.append({"session_id": sid, "images": [
            {"raw_frame_index": frame, "image": image_ref} for frame in [0, 1]]})
        for frame in [0, 1]:
            for side_idx, side in enumerate(source.SIDES):
                matrix = np.linalg.inv(fixed_camera) @ np.linalg.inv(head[frame]) @ controller[frame, side_idx]
                for joint in [4, 8, 12, 16]:
                    node = int(calibration.MAP_25_TO_21[joint])
                    uv, _, _ = fit._project(matrix[None], local[frame, side_idx, node][None], k, truth[side_idx])
                    observations.append({
                        "session_id": sid, "device_serial": serial, "raw_frame_index": frame,
                        "qpc_ns": [10, 20][frame], "image": image_ref, "side": side,
                        "joint_index_21": joint, "manus_node_id": node, "uv": uv[0].tolist(),
                        "visibility": "VISIBLE", "point_type": "GLOVE_TIP", "semantics": "VISUAL_PROXY",
                        "side_confirmed": True, "joint_mapping_confirmed": True,
                        "annotation_uncertainty_px": 12.})
    pins = {"input_manifest": audit.file_ref(manifest_path), "base_camera_config": audit.file_ref(config_path)}
    evidence_path = tmp_path / "SOURCE_IMAGES.json"
    _write(evidence_path, {**pins, "status": "PASSED", "holdout_used": False,
                          "image_size": [640, 480], "sessions": images})
    annotations_path = tmp_path / "ANNOTATIONS.json"
    _write(annotations_path, {**pins, "source_images": audit.file_ref(evidence_path),
                             "observations": observations})
    camera_path = tmp_path / "camera/attempt_0001/candidate/CALIBRATION_REPORT.json"
    camera = {**pins, "annotations": audit.file_ref(annotations_path), "source_images": audit.file_ref(evidence_path),
              "schema": "pick_place_device_camera_rigid_visual_fit_v1", "status": "REVIEW_REQUIRED",
              "holdout_used": False, "time_fitted": False, "scale_fitted": False, "lens_remap": False,
              "K_fitted": False, "installation_prior_fitted": False}
    packet_path = tmp_path / "tasks/current/CAMERA_FAILED/TASK_PACKET.json"
    stage_path, outer_path = camera_path.parent / "RESULT.json", camera_path.parent.parent / "RESULT.json"
    receipt_path = tmp_path / "tasks/receipts/CAMERA_FAILED.json"
    def reseal():
        _write(camera_path, camera)
        _write(packet_path, {"task_id": "CAMERA_FAILED", "run_signature": "sealed_signature"})
        _write(stage_path, {"status": "REVIEW_REQUIRED", "calibration_report": audit.file_ref(camera_path)})
        _write(outer_path, {"task_id": "CAMERA_FAILED", "status": "FAILED_RUNTIME_FINAL", "validation_errors": [],
                            "run_signature": "sealed_signature", "task_packet": audit.file_ref(packet_path),
                            "stage_result": audit.file_ref(stage_path),
                            "required_outputs": {"candidate/RESULT.json": audit.file_ref(stage_path),
                                                 "candidate/CALIBRATION_REPORT.json": audit.file_ref(camera_path)}})
        _write(receipt_path, {"status": "FAILED_RUNTIME_FINAL", "result": audit.file_ref(outer_path)})
    reseal()
    monkeypatch.setattr(calibration, "_load_inputs", lambda *args: ({}, copy.deepcopy(config), contexts))
    return SimpleNamespace(tmp=tmp_path, manifest=manifest_path, annotations=annotations_path,
                           config_path=config_path, config=config, contexts=contexts, truth=truth_by_group,
                           camera_path=camera_path, camera=camera, outer=outer_path, stage=stage_path,
                           packet=packet_path, receipt=receipt_path, reseal=reseal)


def _run(fixture):
    return fit.run_installation_fit(fixture.manifest, fixture.annotations, fixture.config_path,
                                     fixture.camera_path, fixture.tmp / "out")


def test_full_producer_artifacts_only_installation_changes_with_exact_prior_group_binding(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    original = fixture.config_path.read_bytes()
    result = _run(fixture)
    assert result["status"] == "PASSED"
    report = json.loads(Path(result["installation_report"]["path"]).read_text())
    config = json.loads(Path(result["fitted_camera_config"]["path"]).read_text())
    assert result["fitted_camera_config"] == result["fitted_installation_config"]
    assert report["camera_fit_report"] == audit.file_ref(fixture.camera_path)
    assert report["preceding_failure_producer"]["terminal_receipt"] == audit.file_ref(fixture.receipt)
    assert len(report["correspondences"]) == 64 and len(report["groups"]) == 2
    for key, group in report["groups"].items():
        for side_idx, side in enumerate(source.SIDES):
            assert np.allclose(group["sides"][side]["T_controller_wrist_fitted"], fixture.truth[key][side_idx], atol=1e-6)
        selected = config["devices"][group["device_serial"]]
        assert selected["installation_original_prior_sha256"] == group["original_prior_sha256"]
        assert selected["installation_correction_evidence"] == result["installation_report"]
        assert np.allclose(source.installation_matrices(selected["installation_prior_override"]), fixture.truth[key], atol=1e-6)
        assert "K_override" not in selected and "T_head_camera_override" not in selected
        assert selected["status"] == "FROZEN_VISUAL_ESTIMATE"
    for row in report["correspondences"]:
        assert row["uv_before"] != row["uv_after"] and len(row["point_head_m_after"]) == 3
    for flag in ["K_fitted", "camera_fitted", "holdout_used", "time_fitted", "scale_fitted", "lens_remap"]:
        assert report[flag] is False
    assert fixture.config_path.read_bytes() == original


def test_failed_side_never_enters_override_but_successful_group_can(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    original_fit = fit.fit_installation_proxy_points
    def reject_one_side(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        if args[5][0] in ["001", "003"] and args[4][0, 3] > 0:
            result["status"] = "REVIEW_REQUIRED"
        return result
    monkeypatch.setattr(fit, "fit_installation_proxy_points", reject_one_side)
    result = _run(fixture)
    report = json.loads(Path(result["installation_report"]["path"]).read_text())
    config = json.loads(Path(result["fitted_camera_config"]["path"]).read_text())
    assert result["status"] == "REVIEW_REQUIRED"
    assert "installation_prior_override" not in config["devices"]["DEVICE_A"]
    assert config["devices"]["DEVICE_A"]["fit_quality_status"] == "REVIEW_REQUIRED"
    assert "installation_prior_override" in config["devices"]["DEVICE_B"]
    a_group = next(group for group in report["groups"].values() if group["device_serial"] == "DEVICE_A")
    assert a_group["prior_used"] == a_group["prior_before"]


@pytest.mark.parametrize("mutation", ["report_status", "report_K", "outer_status", "outer_binding", "signature", "receipt"])
def test_camera_failure_chain_required(tmp_path, monkeypatch, mutation):
    fixture = _fixture(tmp_path, monkeypatch)
    if mutation in ["report_status", "report_K"]:
        fixture.camera["status" if mutation == "report_status" else "K_fitted"] = "PASSED" if mutation == "report_status" else True
        fixture.reseal()
    else:
        path = {"outer_status": fixture.outer, "outer_binding": fixture.outer,
                "signature": fixture.packet, "receipt": fixture.receipt}[mutation]
        data = json.loads(path.read_text())
        if mutation == "outer_status":
            data["status"] = "RUNNING"
        elif mutation == "outer_binding":
            data["stage_result"] = {}
        elif mutation == "signature":
            data["run_signature"] = "other"
            _write(path, data)
            outer = json.loads(fixture.outer.read_text())
            outer["task_packet"] = audit.file_ref(path)
            _write(fixture.outer, outer)
            receipt = json.loads(fixture.receipt.read_text())
            receipt["result"] = audit.file_ref(fixture.outer)
            _write(fixture.receipt, receipt)
        elif mutation == "receipt":
            data["result"] = {}
        _write(path, data)
    with pytest.raises((fit.InstallationFitError, audit.AuditError)):
        _run(fixture)
    assert not (fixture.tmp / "out").exists()


def test_original_prior_and_per_side_coverage_required(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture.contexts["001"]["prior_reference"]["prior_sha256"] = "new_original"
    fixture.contexts["001"]["prior_reference"]["used_prior_sha256"] = "new_original"
    with pytest.raises(fit.InstallationFitError, match="TWO_DEVELOPMENT_SESSIONS"):
        _run(fixture)
    assert not (fixture.tmp / "out").exists()


@pytest.mark.parametrize("override", ["K_override", "installation_prior_override"])
def test_factory_K_and_original_installation_required(tmp_path, monkeypatch, override):
    fixture = _fixture(tmp_path, monkeypatch)
    if override == "K_override":
        fixture.contexts["001"]["selected_config"][override] = np.eye(3).tolist()
    else:
        fixture.contexts["001"]["prior_reference"]["used_prior_sha256"] = "changed"
    with pytest.raises(fit.InstallationFitError):
        _run(fixture)
    assert not (fixture.tmp / "out").exists()


def test_raw_precomputation_checks_full_native_composition(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture.contexts["001"]["native"]["manus_local_25"][0, 0, 4, 0] += .1
    with pytest.raises(fit.InstallationFitError, match="RAW_COMPOSITION_MISMATCH"):
        _run(fixture)
