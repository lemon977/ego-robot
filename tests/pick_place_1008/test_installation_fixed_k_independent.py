"""Independent CPU checks for the fixed-K right installation hypothesis.

Synthetic landmarks below have known rigid geometry. They test the solver and
adoption boundaries; they grant no physical qualification to real annotations.
"""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chaoyang.pick_place_1008 import audit, installation_fit
from chaoyang.pick_place_1008 import installation_fixed_k as candidate


DEVICE_B = "PA9410MGK8090485G"


def _manual_project(matrices, local, k, installation):
    controller = local @ installation[:3, :3].T + installation[:3, 3]
    camera = np.einsum("nij,nj->ni", matrices[:, :3, :3], controller)
    camera += matrices[:, :3, 3]
    homogeneous = camera @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def _data(seed, frame_groups, k, truth):
    rng = np.random.default_rng(seed)
    matrices, local, sessions, frames, joints = [], [], [], [], []
    for sid, frame_values in frame_groups:
        for frame in frame_values:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_rotvec(rng.uniform(-.25, .25, 3)).as_matrix()
            pose[:3, 3] = rng.uniform([-.12, -.12, .75], [.12, .12, 1.1])
            for joint in [4, 8, 12, 16]:
                matrices.append(pose.copy())
                local.append(rng.uniform([-.08, -.06, -.02], [.08, .10, .12]))
                sessions.append(sid)
                frames.append(frame)
                joints.append(joint)
    matrices, local = np.asarray(matrices), np.asarray(local)
    return {
        "side": "right", "device_serial": DEVICE_B,
        "controller_to_camera": matrices, "local_points": local,
        "uv": _manual_project(matrices, local, k, truth),
        "session_ids": np.asarray(sessions),
        "raw_frame_indices": np.asarray(frames),
        "joint_indices_21": np.asarray(joints),
        "annotation_uncertainty_px": np.full(len(local), 10.),
    }


def _known_answer(delta=None):
    k = np.array([[500., 0., 320.], [0., 490., 240.], [0., 0., 1.]])
    original = np.eye(4)
    original[:3, :3] = Rotation.from_rotvec([.05, -.06, .08]).as_matrix()
    original[:3, 3] = [.08, -.04, -.13]
    parameters = np.array([.025, -.035, .018, .025, -.012, .018]
                          if delta is None else delta, float)
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    correction[:3, 3] = parameters[3:]
    truth = correction @ original
    fit = _data(864, [("008", [0, 3, 6, 9]), ("100", [1, 4, 7, 10])], k, truth)
    validation = _data(1201, [("008", [11, 14]), ("100", [12, 15])], k, truth)
    return fit, validation, k, original, truth


def _spy_solver(monkeypatch):
    calls = []
    real = installation_fit.fit_installation_proxy_points

    def observe(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((copy.deepcopy(args), copy.deepcopy(kwargs), copy.deepcopy(result)))
        return result

    monkeypatch.setattr(installation_fit, "fit_installation_proxy_points", observe)
    return calls


def _assert_data_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], np.ndarray):
            assert np.array_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


def test_independent_geometry_recovers_shared_right_delta_and_only_fit_enters_solver(monkeypatch):
    fit, validation, k, original, truth = _known_answer()
    snapshots = [copy.deepcopy(fit), copy.deepcopy(validation), k.copy(), original.copy()]
    calls = _spy_solver(monkeypatch)
    result = candidate.fit_and_validate_right_installation(fit, validation, k, original)
    assert result["status"] == "PASSED"
    assert np.allclose(result["accepted_installation"], truth, atol=1e-7)
    assert len(calls) == 1
    args, _, solved = calls[0]
    assert np.array_equal(args[0], fit["controller_to_camera"])
    assert np.array_equal(args[1], fit["local_points"])
    assert np.array_equal(args[2], fit["uv"])
    assert len(args[2]) == len(fit["uv"])
    assert np.allclose(solved["T_controller_wrist_fitted"], truth, atol=1e-7)
    assert solved["jacobian_rank"] == 6 and solved["boundary_hit_parameter_indices"] == []
    _assert_data_equal(fit, snapshots[0])
    _assert_data_equal(validation, snapshots[1])
    assert np.array_equal(k, snapshots[2]) and np.array_equal(original, snapshots[3])


def test_validation_uv_changes_acceptance_without_changing_any_optimized_parameter(monkeypatch):
    fit, validation, k, original, _ = _known_answer()
    calls = _spy_solver(monkeypatch)
    accepted = candidate.fit_and_validate_right_installation(fit, validation, k, original)
    rejected_validation = copy.deepcopy(validation)
    rejected_validation["uv"] = np.clip(rejected_validation["uv"] + [120., -90.], [1., 1.], [639., 479.])
    rejected = candidate.fit_and_validate_right_installation(fit, rejected_validation, k, original)
    assert accepted["status"] == "PASSED" and rejected["status"] == "REVIEW_REQUIRED"
    assert np.array_equal(rejected["accepted_installation"], original)
    assert rejected["validation"]["status"] == "REVIEW_REQUIRED"
    assert len(calls) == 2
    first, second = calls[0][2], calls[1][2]
    for key in ["T_controller_wrist_fitted", "delta_controller_rotvec_rad",
                "delta_controller_translation_m"]:
        assert np.array_equal(first[key], second[key])
    assert np.array_equal(calls[0][0][2], calls[1][0][2])


def test_whole_native_frame_overlap_rejected_even_for_a_different_tip(monkeypatch):
    fit, validation, k, original, _ = _known_answer()
    validation["session_ids"][0] = fit["session_ids"][0]
    validation["raw_frame_indices"][0] = fit["raw_frame_indices"][0]
    validation["joint_indices_21"][0] = 20
    calls = _spy_solver(monkeypatch)
    with pytest.raises(candidate.FixedKInstallationError):
        candidate.fit_and_validate_right_installation(fit, validation, k, original)
    assert calls == []


def test_duplicate_point_identity_cannot_reweight_fit_silently(monkeypatch):
    fit, validation, k, original, _ = _known_answer()
    fit["session_ids"][1] = fit["session_ids"][0]
    fit["raw_frame_indices"][1] = fit["raw_frame_indices"][0]
    fit["joint_indices_21"][1] = fit["joint_indices_21"][0]
    calls = _spy_solver(monkeypatch)
    with pytest.raises(candidate.FixedKInstallationError):
        candidate.fit_and_validate_right_installation(fit, validation, k, original)
    assert calls == []


@pytest.mark.parametrize("mutation", ["left", "device_a", "holdout007", "holdout199",
                                      "uncertainty", "negative_frame", "non_tip"])
def test_shared_b_right_scope_and_proxy_domain_rejected_before_optimizer(monkeypatch, mutation):
    fit, validation, k, original, _ = _known_answer()
    if mutation == "left":
        fit["side"] = "left"
    elif mutation == "device_a":
        validation["device_serial"] = "PA9410MGK9050807G"
    elif mutation.startswith("holdout"):
        validation["session_ids"][0] = mutation[-3:]
    elif mutation == "uncertainty":
        validation["annotation_uncertainty_px"][0] = 7.99
    elif mutation == "negative_frame":
        validation["raw_frame_indices"][0] = -1
    elif mutation == "non_tip":
        fit["joint_indices_21"][0] = 0
    calls = _spy_solver(monkeypatch)
    with pytest.raises(candidate.FixedKInstallationError):
        candidate.fit_and_validate_right_installation(fit, validation, k, original)
    assert calls == []


@pytest.mark.parametrize("mutation", ["skew", "negative_fx", "nonfinite", "nonrigid_controller",
                                      "nonrigid_prior"])
def test_fixed_projection_inputs_must_be_valid_and_not_silently_repaired(monkeypatch, mutation):
    fit, validation, k, original, _ = _known_answer()
    if mutation == "skew":
        k[0, 1] = 2.
    elif mutation == "negative_fx":
        k[0, 0] = -1.
    elif mutation == "nonfinite":
        k[0, 0] = np.nan
    elif mutation == "nonrigid_controller":
        validation["controller_to_camera"][0, 0, 0] *= 1.3
    elif mutation == "nonrigid_prior":
        original[0, 0] *= 1.3
    calls = _spy_solver(monkeypatch)
    with pytest.raises((candidate.FixedKInstallationError, ValueError)):
        candidate.fit_and_validate_right_installation(fit, validation, k, original)
    assert calls == []


def test_outside_translation_hypothesis_keeps_original_bounds_and_cannot_pass(monkeypatch):
    fit, validation, k, original, _ = _known_answer([0., 0., 0., .20, 0., 0.])
    calls = _spy_solver(monkeypatch)
    result = candidate.fit_and_validate_right_installation(fit, validation, k, original)
    assert result["status"] == "REVIEW_REQUIRED"
    assert np.array_equal(result["accepted_installation"], original)
    assert len(calls) == 1
    solved = calls[0][2]
    assert np.max(np.abs(solved["delta_controller_rotvec_rad"])) <= .35 + 1e-12
    assert np.max(np.abs(solved["delta_controller_translation_m"])) <= .15 + 1e-12
    assert solved["after"]["p50_px"] < 20.
    assert 3 in solved["boundary_hit_parameter_indices"]
    assert solved["translation_component_bound_m"] == .15
    assert solved["rotation_vector_component_bound_rad"] == .35


def test_validation_with_zero_reprojection_error_but_negative_depth_is_not_adopted(monkeypatch):
    fit, validation, k, original, truth = _known_answer()
    validation["controller_to_camera"][:, 2, 3] = -1.
    validation["uv"] = _manual_project(validation["controller_to_camera"],
                                         validation["local_points"], k, truth)
    calls = _spy_solver(monkeypatch)
    result = candidate.fit_and_validate_right_installation(fit, validation, k, original)
    assert result["fit"]["status"] == "PASSED" and len(calls) == 1
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["validation"]["nonpositive_depth_count"] == len(validation["uv"])
    assert result["validation"]["after"]["p50_px"] < 1e-6
    assert np.array_equal(result["accepted_installation"], original)


def _base_config_and_prior(k, original):
    prior = {
        "schema": "controller_to_wrist_v1",
        "convention": "wrist_pose = compose_pose(controller_pose, transform)",
        "controller_pose_frame": "pico_world_rh_x_forward_y_left_z_up",
        "teleop_wrist": False,
        "left": {"pos": [.08, .04, -.13], "quat": [0., 0., 0., 1.]},
        "right": {"pos": original[:3, 3].tolist(),
                  "quat": Rotation.from_matrix(original[:3, :3]).as_quat().tolist()},
    }
    k_ref = {"path": "/synthetic/INTRINSIC_REPORT.json", "bytes": 1, "sha256": "1" * 64}
    base = {
        "schema": "fixed_encoded_visual_candidate",
        "qpc_policy": {"modified": False, "timeline": "ORIGINAL_NATIVE_QPC"},
        "metric_scale": 1.,
        "source_refs": {"raw": {"sha256": "2" * 64}},
        "devices": {
            "PA9410MGK9050807G": {"K_override": (k * [2., 2., 1.]).tolist(),
                                  "original_prior_sha256": "A_prior",
                                  "left_metadata": {"immutable": True}},
            DEVICE_B: {"K_override": (np.diag([2., 2., 1.]) @ k).tolist(),
                       "correction_evidence": k_ref,
                       "eye_identity_qualification": "VISUAL_ASSUMPTION_SOURCEINDEX_ABSENT",
                       "status": "FROZEN_VISUAL_ESTIMATE", "fit_quality_status": "PASSED",
                       "T_head_camera_factory": np.eye(4).tolist()},
        },
    }
    return base, prior


def test_success_adds_only_b_right_installation_and_preserves_every_existing_config_value():
    fit, validation, k, original, truth = _known_answer()
    solved = candidate.fit_and_validate_right_installation(fit, validation, k, original)
    base, prior = _base_config_and_prior(k, original)
    base_snapshot, prior_snapshot = copy.deepcopy(base), copy.deepcopy(prior)
    report_ref = {"path": "/synthetic/FIXED_K_INSTALLATION_REPORT.json",
                  "bytes": 5, "sha256": "3" * 64}
    value = candidate._config_after(base, solved, prior, report_ref)
    for key in base:
        if key != "devices":
            assert value[key] == base[key]
    assert value["devices"]["PA9410MGK9050807G"] == base["devices"]["PA9410MGK9050807G"]
    selected = value["devices"][DEVICE_B]
    for key, before in base["devices"][DEVICE_B].items():
        assert selected[key] == before
    assert selected["installation_prior_override"]["left"] == prior["left"]
    assert np.allclose(selected["installation_prior_override"]["right"]["pos"],
                       truth[:3, 3], atol=1e-7)
    assert selected["installation_correction_evidence"] == report_ref
    assert base == base_snapshot and prior == prior_snapshot


def test_failed_validation_config_is_equal_to_base_and_never_contains_new_override():
    fit, validation, k, original, _ = _known_answer()
    validation["uv"] = np.clip(validation["uv"] + [120., -90.], [1., 1.], [639., 479.])
    solved = candidate.fit_and_validate_right_installation(fit, validation, k, original)
    base, prior = _base_config_and_prior(k, original)
    value = candidate._config_after(base, solved, prior,
                                    {"path": "/unadopted/report", "bytes": 1, "sha256": "0" * 64})
    assert solved["status"] == "REVIEW_REQUIRED"
    assert value == base and value is not base
    assert "installation_prior_override" not in value["devices"][DEVICE_B]


def _write_json(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _sealed_fixture(tmp_path, monkeypatch):
    from chaoyang.pick_place_1008 import audit, source
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    task = "exact78_pick_place_independent_synthetic_fixed_k_v8"
    folder = tmp_path / "_run/current" / task / "attempts/attempt_0001/candidate"
    report_path, config_path = folder / candidate.REPORT_NAME, folder / "FITTED_CAMERA_CONFIG.json"
    stage_path, outer_path = folder / "RESULT.json", folder.parent / "RESULT.json"
    packet_path = tmp_path / "tasks/current" / task / "TASK_PACKET.json"
    ledger_path = tmp_path / "docs/governance/LONG_HORIZON_TASK_STATE.json"
    receipt_path = tmp_path / "tasks/receipts" / (task.upper() + ".json")
    signature = "synthetic_cpu_test_signature"
    report = {"schema": candidate.REPORT_SCHEMA, "status": "PASSED"}
    config = {"devices": {}}
    packet = {"task_id": task, "execution_mode": "EXACT78_CANDIDATE",
              "run_signature": signature, "required_outputs": [
                  "candidate/RESULT.json", "candidate/" + candidate.REPORT_NAME,
                  "candidate/FITTED_CAMERA_CONFIG.json"]}
    _write_json(report_path, report)
    _write_json(config_path, config)
    _write_json(packet_path, packet)
    stage = {"status": "PASSED", "fixed_k_installation_report": source.ref(report_path),
             "fitted_camera_config": source.ref(config_path)}
    _write_json(stage_path, stage)
    outer = {"task_id": task, "run_signature": signature, "status": "PASSED",
             "validation_errors": [], "task_packet": source.ref(packet_path),
             "stage_result": source.ref(stage_path), "required_outputs": {
                 "candidate/RESULT.json": source.ref(stage_path),
                 "candidate/" + candidate.REPORT_NAME: source.ref(report_path),
                 "candidate/FITTED_CAMERA_CONFIG.json": source.ref(config_path)}}

    def reseal_outer():
        _write_json(outer_path, outer)
        ledger = {"tasks": [{"task_id": task, "status": "PASSED", "pid": None,
                             "proc_start_ticks": None, "run_signature": signature,
                             "result": source.ref(outer_path)}]}
        receipt = {"status": "PASSED", "result": source.ref(outer_path)}
        _write_json(ledger_path, ledger)
        _write_json(receipt_path, receipt)
        return ledger, receipt

    ledger, receipt = reseal_outer()
    return {
        "report_path": report_path, "config_path": config_path, "stage_path": stage_path,
        "outer_path": outer_path, "packet_path": packet_path, "ledger_path": ledger_path,
        "receipt_path": receipt_path, "report_ref": source.ref(report_path),
        "report": report, "config": config, "stage": stage, "outer": outer, "packet": packet,
        "ledger": ledger, "receipt": receipt, "reseal_outer": reseal_outer,
    }


def _product(fixture):
    return candidate._sealed_product(fixture["report_ref"], candidate.REPORT_NAME,
                                     candidate.REPORT_SCHEMA, "fixed_k_installation_report")


@pytest.mark.parametrize("mutation", ["live_ledger", "live_pid", "wrong_signature",
                                      "implementation_packet", "missing_output",
                                      "stage_report_mismatch", "receipt_drift", "config_byte_drift"])
def test_pass_label_and_rehashed_links_cannot_bypass_sealed_producer_gate(tmp_path, monkeypatch, mutation):
    from chaoyang.pick_place_1008 import source
    fixture = _sealed_fixture(tmp_path, monkeypatch)
    assert _product(fixture)[0]["status"] == "PASSED"
    if mutation in ["live_ledger", "live_pid"]:
        row = fixture["ledger"]["tasks"][0]
        if mutation == "live_ledger":
            row["status"] = "RUNNING"
        else:
            row["pid"], row["proc_start_ticks"] = 42, 1234
        _write_json(fixture["ledger_path"], fixture["ledger"])
    elif mutation in ["wrong_signature", "implementation_packet"]:
        fixture["packet"]["run_signature" if mutation == "wrong_signature" else "execution_mode"] = (
            "changed_signature" if mutation == "wrong_signature" else "IMPLEMENTATION_CPU_ONLY")
        _write_json(fixture["packet_path"], fixture["packet"])
        fixture["outer"]["task_packet"] = source.ref(fixture["packet_path"])
        fixture["reseal_outer"]()
    elif mutation == "missing_output":
        del fixture["outer"]["required_outputs"]["candidate/FITTED_CAMERA_CONFIG.json"]
        fixture["reseal_outer"]()
    elif mutation == "stage_report_mismatch":
        fixture["stage"]["fixed_k_installation_report"] = source.ref(fixture["config_path"])
        _write_json(fixture["stage_path"], fixture["stage"])
        fixture["outer"]["stage_result"] = source.ref(fixture["stage_path"])
        fixture["outer"]["required_outputs"]["candidate/RESULT.json"] = source.ref(fixture["stage_path"])
        fixture["reseal_outer"]()
    elif mutation == "receipt_drift":
        fixture["receipt"]["result"]["sha256"] = "0" * 64
        _write_json(fixture["receipt_path"], fixture["receipt"])
    elif mutation == "config_byte_drift":
        _write_json(fixture["config_path"], {"devices": {"unsealed": True}})
    with pytest.raises((candidate.FixedKInstallationError, audit.AuditError, ValueError)):
        _product(fixture)


def test_consumer_rejects_even_a_sealed_pass_report_without_full_implementation_and_inputs(tmp_path, monkeypatch):
    fixture = _sealed_fixture(tmp_path, monkeypatch)
    # This report has a valid task/receipt chain, but none of the new route's
    # implementation, fixed-K, native-source or whole-frame evidence closure.
    assert _product(fixture)[0]["status"] == "PASSED"
    selected = {"installation_correction_evidence": fixture["report_ref"],
                "installation_original_prior_sha256": candidate.PRIOR_SHA,
                "installation_group_key": DEVICE_B + ":" + candidate.PRIOR_SHA,
                "fit_quality_status": "PASSED", "installation_prior_override": {}}
    with pytest.raises(candidate.FixedKInstallationError):
        candidate.validate_override_evidence(selected, {"prior_sha256": candidate.PRIOR_SHA}, DEVICE_B)


def _consumer_fixture(tmp_path, monkeypatch):
    """Keep the actual sealed producer and consumer gates; isolate raw decoding.

    This fixture supplies an already validated source-loader result. Origin gates
    are checked separately above. No runtime implementation is replaced here.
    """
    from chaoyang.pick_place_1008 import source
    fixture = _sealed_fixture(tmp_path, monkeypatch)
    fit, validation, k, original, _ = _known_answer()
    base, prior = _base_config_and_prior(k, original)
    # Production starts from the recorded quaternion-derived original prior.
    original = source.installation_matrices(prior)[1]
    solved = candidate.fit_and_validate_right_installation(fit, validation, k, original)
    pins = {}
    for name in ["input_manifest", "annotations_bundle", "camera_config", "source_index"]:
        p = tmp_path / "inputs" / (name + ".json")
        _write_json(p, base if name == "camera_config" else {"synthetic": name})
        pins[name] = source.ref(p)
    implementation = {}
    for name in ["module", "source", "core"]:
        p = tmp_path / "implementation" / (name + ".py")
        p.parent.mkdir(parents=True, exist_ok=True)
        module = {"module": candidate, "source": source, "core": installation_fit}[name]
        p.write_bytes(__import__("pathlib").Path(module.__file__).read_bytes())
        monkeypatch.setattr(module, "__file__", str(p))
        implementation[name] = source.ref(p)
    original_file = tmp_path / "original" / "dataset.hdf5"
    original_file.parent.mkdir(parents=True, exist_ok=True)
    original_file.write_bytes(b"synthetic installation attribute source")
    original_reference = {"prior_sha256": candidate.PRIOR_SHA,
                          "source_hdf5": source.ref(original_file)}
    loaded = {
        "pins": pins, "base_config": base,
        "encoded_k_producer": {"task_id": candidate.K_TASK, "synthetic": "validated_origin"},
        "contexts": {"008": {"prior": prior}, "100": {"prior": copy.deepcopy(prior)}},
        "fit_records": [{"synthetic": "FIT native observation identity"}],
        "validation_records": [{"synthetic": "VALID native observation identity"}],
        "fit_data": fit, "validation_data": validation,
        "k_report": {"devices": {DEVICE_B: {"K_image_fitted": k.tolist(),
                                           "T_head_camera_factory_fixed": np.eye(4).tolist()}}},
    }
    monkeypatch.setattr(candidate, "_validate_bundle", lambda *args: loaded)
    monkeypatch.setattr(source, "read_installation",
                        lambda folder: (copy.deepcopy(prior), copy.deepcopy(original_reference)))
    used = candidate._config_after(base, solved, prior, fixture["report_ref"])[
        "devices"][DEVICE_B]["installation_prior_override"]
    group = copy.deepcopy(solved)
    group.update(device_serial=DEVICE_B, side="right",
                 original_prior_sha256=candidate.PRIOR_SHA,
                 prior_before=copy.deepcopy(prior), prior_used=copy.deepcopy(used))
    flags = {
        "installation_prior_fitted": True, "original_native_rgb_verified": True,
        "left_installation_fitted": False, "device_A_fitted": False, "K_fitted": False,
        "camera_fitted": False, "time_fitted": False, "scale_fitted": False,
        "qpc_modified": False, "lens_remap": False, "holdout_used": False,
        "validation_used_by_optimizer": False, "training_eligible": False,
        "control_ground_truth": False, "physical_ground_truth": False,
    }
    report = fixture["report"]
    report.update(**pins, **flags, implementation=implementation,
                  encoded_k_producer=loaded["encoded_k_producer"],
                  groups={DEVICE_B + ":" + candidate.PRIOR_SHA: group},
                  fit_correspondences=loaded["fit_records"],
                  validation_correspondences=loaded["validation_records"],
                  K_image_fixed=k.tolist(), T_head_camera_factory_fixed=np.eye(4).tolist(),
                  fit_native_frames=copy.deepcopy(candidate.FIT_FRAMES),
                  validation_native_frames=copy.deepcopy(candidate.VALIDATION_FRAMES),
                  fitted_variables="ONE_SHARED_DEVICE_B_RIGHT_CONTROLLER_LOCAL_SIX_DOF")
    config = candidate._config_after(base, solved, prior, fixture["report_ref"])

    def rebind():
        fixture["packet"]["inputs"] = {"files": list(pins.values())}
        _write_json(fixture["packet_path"], fixture["packet"])
        fixture["outer"]["task_packet"] = source.ref(fixture["packet_path"])
        _write_json(fixture["report_path"], report)
        report_ref = source.ref(fixture["report_path"])
        fixture["report_ref"] = report_ref
        config["devices"][DEVICE_B]["installation_correction_evidence"] = report_ref
        _write_json(fixture["config_path"], config)
        fixture["stage"].update(fixed_k_installation_report=report_ref,
                                fitted_camera_config=source.ref(fixture["config_path"]))
        _write_json(fixture["stage_path"], fixture["stage"])
        fixture["outer"]["stage_result"] = source.ref(fixture["stage_path"])
        fixture["outer"]["required_outputs"].update({
            "candidate/RESULT.json": source.ref(fixture["stage_path"]),
            "candidate/" + candidate.REPORT_NAME: report_ref,
            "candidate/FITTED_CAMERA_CONFIG.json": source.ref(fixture["config_path"])})
        fixture["reseal_outer"]()

    rebind()
    fixture.update(loaded=loaded, config=config, original_reference=original_reference,
                   prior=prior, group=group, rebind=rebind)
    return fixture


def _consume(fixture):
    return candidate.validate_override_evidence(fixture["config"]["devices"][DEVICE_B],
                                                fixture["original_reference"], DEVICE_B)


def test_legal_nontrivial_quaternion_roundtrip_is_accepted_by_real_consumer(tmp_path, monkeypatch):
    from chaoyang.pick_place_1008 import source
    fixture = _consumer_fixture(tmp_path, monkeypatch)
    reported = np.asarray(fixture["group"]["T_controller_wrist_fitted"])
    roundtripped = source.installation_matrices(fixture["group"]["prior_used"])[1]
    assert not np.array_equal(reported, roundtripped)
    assert np.max(abs(reported - roundtripped)) < 1e-14
    accepted = _consume(fixture)
    assert accepted == fixture["group"]["prior_used"]
    assert accepted["left"] == fixture["prior"]["left"]


@pytest.mark.parametrize("mutation", ["left_changed", "fixed_K_changed", "boundary_delta",
                                      "validation_uv_changed", "time_fitted", "bad_condition", "nan_condition",
                                      "changed_bounds", "changed_split", "selected_K_changed",
                                      "implementation_sha_drift"])
def test_resealed_pass_artifact_cannot_adopt_wrong_side_domain_or_failed_validation(tmp_path, monkeypatch, mutation):
    fixture = _consumer_fixture(tmp_path, monkeypatch)
    assert _consume(fixture) == fixture["group"]["prior_used"]
    if mutation == "left_changed":
        fixture["group"]["prior_used"]["left"]["pos"][0] += .01
        fixture["config"]["devices"][DEVICE_B]["installation_prior_override"]["left"]["pos"][0] += .01
    elif mutation == "fixed_K_changed":
        fixture["report"]["K_image_fixed"][0][0] += 1.
    elif mutation == "boundary_delta":
        fixture["group"]["fit"]["delta_controller_translation_m"][0] = .15
    elif mutation == "validation_uv_changed":
        fixture["loaded"]["validation_data"]["uv"] = np.clip(
            fixture["loaded"]["validation_data"]["uv"] + [120., -90.], [1., 1.], [639., 479.])
    elif mutation == "time_fitted":
        fixture["report"]["time_fitted"] = True
    elif mutation in ["bad_condition", "nan_condition"]:
        fixture["group"]["fit"]["jacobian_condition"] = 1e12 if mutation == "bad_condition" else float("nan")
    elif mutation == "changed_bounds":
        fixture["group"]["fit"]["translation_component_bound_m"] = .25
    elif mutation == "changed_split":
        fixture["report"]["validation_native_frames"]["008"][0] = 0
    elif mutation == "selected_K_changed":
        fixture["config"]["devices"][DEVICE_B]["K_override"][0][0] += 1.
    elif mutation == "implementation_sha_drift":
        fixture["report"]["implementation"]["module"]["sha256"] = "0" * 64
    fixture["rebind"]()
    with pytest.raises((candidate.FixedKInstallationError, audit.AuditError)):
        _consume(fixture)


def test_source_consumer_dispatch_accepts_only_closed_new_route(tmp_path, monkeypatch):
    from chaoyang.pick_place_1008 import source
    fixture = _consumer_fixture(tmp_path, monkeypatch)
    assert source.validated_installation_override(
        fixture["config"]["devices"][DEVICE_B], fixture["original_reference"], DEVICE_B
    ) == fixture["group"]["prior_used"]


def test_source_consumer_does_not_fall_back_to_old_group_only_gate(tmp_path, monkeypatch):
    from chaoyang.pick_place_1008 import source
    fixture = _sealed_fixture(tmp_path, monkeypatch)
    fixture["report"]["groups"] = {DEVICE_B + ":" + candidate.PRIOR_SHA: {
        "status": "PASSED", "device_serial": DEVICE_B,
        "original_prior_sha256": candidate.PRIOR_SHA, "prior_used": {}}}
    _write_json(fixture["report_path"], fixture["report"])
    fixture["report_ref"] = source.ref(fixture["report_path"])
    selected = {"installation_correction_evidence": fixture["report_ref"],
                "installation_original_prior_sha256": candidate.PRIOR_SHA,
                "installation_group_key": DEVICE_B + ":" + candidate.PRIOR_SHA,
                "fit_quality_status": "PASSED", "installation_prior_override": {}}
    with pytest.raises((source.SourceContractError, audit.AuditError)):
        source.validated_installation_override(selected, {"prior_sha256": candidate.PRIOR_SHA}, DEVICE_B)


def test_resealed_packet_without_source_index_input_cannot_consume_candidate(tmp_path, monkeypatch):
    from chaoyang.pick_place_1008 import source
    fixture = _consumer_fixture(tmp_path, monkeypatch)
    assert _consume(fixture) == fixture["group"]["prior_used"]
    fixture["packet"]["inputs"]["files"].remove(fixture["loaded"]["pins"]["source_index"])
    _write_json(fixture["packet_path"], fixture["packet"])
    fixture["outer"]["task_packet"] = source.ref(fixture["packet_path"])
    fixture["reseal_outer"]()
    with pytest.raises(candidate.FixedKInstallationError, match="PACKET_INPUT_LINK"):
        _consume(fixture)


def test_installation_change_declaration_is_true_only_when_new_prior_is_adopted():
    import json
    _, _, k, original, truth = _known_answer()
    base, prior = _base_config_and_prior(k, original)
    base["installation_prior_changed"] = False
    base_bytes = json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
    report_ref = {"path": "/synthetic/FIXED_K_INSTALLATION_REPORT.json",
                  "bytes": 5, "sha256": "4" * 64}
    success = candidate._config_after(
        base, {"status": "PASSED", "T_controller_wrist_fitted": truth.tolist()}, prior, report_ref)
    assert success["installation_prior_changed"] is True
    assert success["installation_changed_scope"] == "DEVICE_B_RIGHT_ONLY"
    assert success["devices"]["PA9410MGK9050807G"] == base["devices"]["PA9410MGK9050807G"]
    assert success["devices"][DEVICE_B]["installation_prior_override"]["left"] == prior["left"]
    for field in ["qpc_policy", "metric_scale", "source_refs"]:
        assert success[field] == base[field]
    assert success["devices"][DEVICE_B]["K_override"] == base["devices"][DEVICE_B]["K_override"]
    failure = candidate._config_after(
        base, {"status": "REVIEW_REQUIRED", "T_controller_wrist_fitted": truth.tolist()}, prior, report_ref)
    assert failure["installation_prior_changed"] is False
    assert "installation_changed_scope" not in failure
    assert json.dumps(failure, sort_keys=True, separators=(",", ":")).encode() == base_bytes
    assert json.dumps(base, sort_keys=True, separators=(",", ":")).encode() == base_bytes


def _prepared_source_1280_fixture(tmp_path, monkeypatch, sid="008"):
    """Load a real prepared-source NPZ rather than faking a 640 K session."""
    from pathlib import Path
    from chaoyang.pick_place_1008 import source
    # The loader uses its source path as the project-root anchor. Relocate that
    # anchor to the complete synthetic repository, preserving real code bytes.
    anchored = tmp_path / "src/chaoyang/pick_place_1008/source.py"
    anchored.parent.mkdir(parents=True, exist_ok=True)
    anchored.write_bytes(Path(source.__file__).read_bytes())
    monkeypatch.setattr(source, "__file__", str(anchored))
    folder = tmp_path / "prepared" / sid / "source"
    folder.mkdir(parents=True)
    video = folder / "native.mp4"
    video.write_bytes(b"synthetic decoder-independent prepared video reference")
    # Actual accepted encoded B K at 1280x960; image annotation K is 640x480.
    k_output = np.array([[1042.4497247810964, 0., 603.279959158785],
                         [0., 1135.6193426120842, 521.9762600698414],
                         [0., 0., 1.]])
    k_image = np.diag([.5, .5, 1.]) @ k_output
    arrays = folder / "source.npz"
    controller = np.tile(np.eye(4), (1, 2, 1, 1))
    controller[:, :, 2, 3] = 1.
    local = np.zeros((1, 2, 25, 3))
    local[:, :, 9] = [.009, .02, .03]
    np.savez(arrays, K=k_output, c2w=np.eye(4)[None], c2w_editor=np.eye(4)[None],
             hands=np.zeros((2, 1, 21, 3)), hand_observed=np.ones((2, 1), bool),
             hand_source=np.full((2, 1), "RAW"), body=np.zeros((1, 22, 3)),
             body_observed=np.zeros((1, 22), bool), gravity=np.array([[0., 0., -1.]]),
             qpc_ns=np.array([123456789], np.int64),
             native__head=np.eye(4)[None], native__controller=controller,
             native__manus_local_25=local, native__head_valid=np.ones(1, bool),
             native__controller_valid=np.ones((1, 2), bool),
             native__hand_observed=np.ones((2, 1), bool),
             native__joint_observed_local=np.ones((1, 2, 21), bool),
             native__manus25_to21=source.MAP_25_TO_21)
    meta = {
        "session_id": "pick_place_1008_" + sid, "status": "COMPLETE_RAW_INPUT",
        "schema_version": source.RAW_INPUT_SCHEMA, "width": 1280, "height": 960,
        "fps": 30., "frame_count": 1,
        "video": source.ref(video), "arrays": source.ref(arrays),
        "calibration": {"device_serial": DEVICE_B,
                        "hand_joint_schema": source.JOINT_MAPPING_SCHEMA,
                        "manus25_to21": source.MAP_25_TO_21.tolist(),
                        "T_head_camera_used": np.eye(4).tolist(),
                        "manus_joint_mapping": {
                            "authority": "VERIFIED_SDK_RAW25_TO_CANONICAL_MANO21",
                            "raw_joint_types": list(source.RAW_CAPTURE_JOINT_ALIASES),
                            "raw_parent_ids": list(source.RAW_SDK_PARENT_IDS),
                            "raw_chain_types": list(source.RAW_SDK_CHAIN_TYPES)}},
    }
    _write_json(folder / "SESSION_INPUT.json", meta)
    session = source.load_prepared(folder)
    report_device = {"output_size": [1280, 960], "K_output_fitted": k_output.tolist(),
                     "K_image_fitted": k_image.tolist(),
                     "T_head_camera_factory_fixed": np.eye(4).tolist()}
    assert session.session_id == "pick_place_1008_" + sid
    assert session.width == 1280 and session.height == 960
    assert np.array_equal(session.K, k_output) and not np.array_equal(session.K, k_image)
    return session, report_device


def test_real_prepared_1280_k_is_verified_then_scaled_to_640_annotation_grid(tmp_path, monkeypatch):
    session, accepted = _prepared_source_1280_fixture(tmp_path, monkeypatch)
    before = session.K.copy()
    k640 = candidate._verified_source_K(session, accepted)
    assert np.array_equal(k640, accepted["K_image_fitted"])
    assert np.array_equal(k640, np.diag([.5, .5, 1.]) @ before)
    assert np.array_equal(session.K, before)
    assert session.width == 1280 and session.height == 960


@pytest.mark.parametrize("mutation", ["source_is_640_K", "source_K_drift", "output_K_drift",
                                      "image_focal_drift", "image_center_drift", "wrong_size",
                                      "nonuniform_grid", "noninteger_size"])
def test_true_source_and_annotation_grid_mismatches_are_rejected(tmp_path, monkeypatch, mutation):
    from dataclasses import replace
    session, accepted = _prepared_source_1280_fixture(tmp_path, monkeypatch)
    if mutation == "source_is_640_K":
        session = replace(session, K=np.asarray(accepted["K_image_fitted"]))
    elif mutation == "source_K_drift":
        changed = session.K.copy()
        changed[0, 0] += 1.
        session = replace(session, K=changed)
    elif mutation == "output_K_drift":
        accepted["K_output_fitted"][0][0] += 1.
    elif mutation == "image_focal_drift":
        accepted["K_image_fitted"][0][0] += 1.
    elif mutation == "image_center_drift":
        accepted["K_image_fitted"][0][2] += 1e-9
    elif mutation == "wrong_size":
        session = replace(session, height=720)
    elif mutation == "nonuniform_grid":
        session = replace(session, height=720)
        accepted["output_size"] = [1280, 720]
    elif mutation == "noninteger_size":
        session = replace(session, width=1280.)
    with pytest.raises(candidate.FixedKInstallationError):
        candidate._verified_source_K(session, accepted)


def _native_identity_fixture(tmp_path, monkeypatch, sid="008"):
    import cv2
    from chaoyang.pick_place_1008 import source
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    session, _ = _prepared_source_1280_fixture(tmp_path, monkeypatch, sid)
    yy, xx = np.indices((480, 640))
    rgb = np.stack([(xx % 240).astype(np.uint8), (yy % 240).astype(np.uint8),
                    np.full((480, 640), 47, np.uint8)], axis=2)
    image = tmp_path / ("original_native_" + sid + "_0000.png")
    cv2.setNumThreads(1)
    assert cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    qpc = tmp_path / ("raw_vst_" + sid + ".qpc.jsonl")
    qpc.write_text('{"qpc_ns":123456789}\n')
    source_refs = {"video": source.ref(session.video), "video_qpc": source.ref(qpc)}
    context = {"session": session, "native": session.native_motion, "rgb": rgb[None],
               "row": {"session_id": sid, "device_serial": DEVICE_B, "source_refs": source_refs}}
    observation = {
        "session_id": sid, "device_serial": DEVICE_B, "raw_frame_index": 0,
        "qpc_ns": int(session.qpc_ns[0]), "image": source.ref(image),
        "side": "right", "joint_index_21": 8, "manus_node_id": 9,
        "point_type": "GLOVE_TIP", "semantics": "VISUAL_PROXY", "visibility": "VISIBLE",
        "side_confirmed": True, "joint_mapping_confirmed": True,
        "uv": [320., 240.], "annotation_uncertainty_px": 12.,
    }
    evidence = {"session_id": sid, "device_serial": DEVICE_B, "raw_frame_index": 0,
                "qpc_ns": int(session.qpc_ns[0]), "image": source.ref(image),
                "source_vst": source_refs["video"], "source_qpc": source_refs["video_qpc"],
                "pixels_overlaid": False, "lens_remap": False}
    return observation, evidence, context


@pytest.mark.parametrize("sid", ["008", "100"])
def test_actual_prefixed_prepared_session_and_numeric_annotation_identity_are_accepted(tmp_path, monkeypatch, sid):
    observation, evidence, context = _native_identity_fixture(tmp_path, monkeypatch, sid)
    assert context["session"].session_id == "pick_place_1008_" + sid
    assert context["row"]["session_id"] == observation["session_id"] == evidence["session_id"] == sid
    result = candidate._verified_native_tip(observation, evidence, context, np.eye(4), True)
    assert result["image"] == observation["image"]
    assert np.array_equal(result["local"], [.009, .02, .03])
    assert np.array_equal(result["matrix"], context["native"]["controller"][0, 1])
    assert context["session"].width == 1280 and context["session"].height == 960


@pytest.mark.parametrize("mutation", ["annotation_other_sid", "evidence_other_sid", "manifest_other_sid",
                                      "manifest_prefixed", "manifest_integer", "raw_other_sid",
                                      "raw_wrong_prefix", "raw_short_sid", "raw_extra_suffix"])
def test_actual_prepared_native_identity_never_accepts_cross_session_or_loose_prefix(tmp_path, monkeypatch, mutation):
    from dataclasses import replace
    observation, evidence, context = _native_identity_fixture(tmp_path, monkeypatch)
    if mutation == "annotation_other_sid":
        observation["session_id"] = "003"
    elif mutation == "evidence_other_sid":
        evidence["session_id"] = "003"
    elif mutation == "manifest_other_sid":
        context["row"]["session_id"] = "003"
    elif mutation == "manifest_prefixed":
        context["row"]["session_id"] = "pick_place_1008_008"
    elif mutation == "manifest_integer":
        context["row"]["session_id"] = 8
    elif mutation == "raw_other_sid":
        context["session"] = replace(context["session"], session_id="pick_place_1008_003")
    elif mutation == "raw_wrong_prefix":
        context["session"] = replace(context["session"], session_id="other_dataset_008")
    elif mutation == "raw_short_sid":
        context["session"] = replace(context["session"], session_id="008")
    elif mutation == "raw_extra_suffix":
        context["session"] = replace(context["session"], session_id="pick_place_1008_008_extra")
    with pytest.raises(candidate.FixedKInstallationError):
        candidate._verified_native_tip(observation, evidence, context, np.eye(4), True)
