"""Synthetic Motion glue only: real IK/FK, Robot construction and NAS are forbidden."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import motion as m
from chaoyang.inpainting_candidate.contracts import SCHEMA, canonical_sha, file_ref, verify_ref, verify_sealed_stage


def json_ref(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return file_ref(path)


@pytest.fixture
def case(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    raw_dir = tmp_path / "raw_captures/001"
    raw_dir.mkdir(parents=True)
    raw = {}
    for key in ("manifest", "video", "controller_head", "manus", "manus_meta", "camera", "camera_meta",
                "installation_container", "video_qpc", "video_wall"):
        path = raw_dir / (key + ".raw")
        path.write_bytes(("synthetic " + key).encode())
        raw[key] = file_ref(path)
    config = json_ref(tmp_path / "adopted/FITTED_CAMERA_CONFIG.json", {"devices": {}})
    manifest = json_ref(tmp_path / "inpainting_v1/MANIFEST.json", {
        "baseline_id": "INPAINTING_RAW_TO_ROBOT_V1", "inputs": {"camera": config}})
    times = np.array([1_000_000_000, 1_040_000_000, 1_100_000_000], np.int64)
    frame_map = [{"native_frame_index": i, "decoded_display_index": i, "model_input_frame": i,
        "output_frame": i, "sbs_source_frame": i, "qpc_declared_ns": int(timestamp),
        "acquisition_frame_id": "UNKNOWN", "physical_eye": "right", "sbs_index": 1,
        "physical_eye_verified": False, "decode": {"decoder_pts": None}, "output_pts": None,
        "timebase_status": m.TIMEBASE} for i, timestamp in enumerate(times)]
    report = {"status": "SOURCE_PREFLIGHT_CANDIDATE", "session_id": "synthetic_001",
        "session_dir": str(raw_dir), "native_sidecar_frames": 3, "actual_inputs": raw,
        "frame_mapping": deepcopy(frame_map), "timebase_status": m.TIMEBASE,
        "adopted_route_contracts": [manifest, config], "camera": {"physical_eye": "right", "sbs_index": 1}}
    source_dir = tmp_path / "source_sealed"
    source_dir.mkdir()
    np.savez_compressed(source_dir / "RAW_RGB.npz", rgb=np.zeros((3, 2, 2, 3), np.uint8))
    outputs = [file_ref(source_dir / "RAW_RGB.npz"), json_ref(source_dir / "FRAME_MAP.json", frame_map),
               json_ref(source_dir / "SOURCE_REPORT.json", report)]
    source = {"schema": SCHEMA, "stage": "Source", "status": "SEALED_COMPLETE", "native_frames": 3,
        "session_id": report["session_id"], "component_signature": "a" * 64,
        "outputs": outputs, "actual_inputs": list(raw.values()) + [manifest, config]}
    source_ref = json_ref(source_dir / "RECEIPT.json", source)
    for relative in m.CODE_CLOSURE:
        code = tmp_path / relative
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text("# synthetic pinned implementation\n")
    session = SimpleNamespace(frame_count=3, fps=20., width=1280, height=960,
        qpc_ns=times, K=np.array([[1000., 0., 640.], [0., 1000., 480.], [0., 0., 1.]]),
        c2w=np.tile(np.eye(4), (3, 1, 1)), c2w_editor=np.tile(np.eye(4), (3, 1, 1)),
        hands=np.zeros((2, 3, 21, 3)), hand_observed=np.ones((2, 3), bool),
        hand_source=np.full((2, 3), "RAW_VALID", dtype="U32"),
        motion_valid={"head_valid": np.ones(3, bool), "controller_valid": np.ones((3, 2), bool),
            "manus_valid": np.ones((3, 2), bool), "hand_display_valid": np.ones((2, 3), bool),
            "joint_observed_local": np.ones((3, 2, 21), bool), "complete_valid": np.ones(3, bool)},
        mapping={"frame_id": np.arange(3), "pico_source_row": np.arange(3),
            "pico_source_line": np.arange(3) + 1, "pico_source_qpc_ns": times,
            "pico_skew_ns": np.zeros(3, np.int64), "manus_left_source_line": np.arange(3) + 1,
            "manus_right_source_line": np.arange(3) + 11},
        calibration={"authority": "SYNTHETIC_INSTALLATION_PRIOR"})
    robot = SimpleNamespace(arms=[SimpleNamespace(joint_names=tuple(f"q_{s}_{i}" for i in range(7))) for s in range(2)],
        hands=[SimpleNamespace(names=tuple(f"h_{s}_{i}" for i in range(22))) for s in range(2)])
    result = SimpleNamespace(q_arm=np.zeros((3, 2, 7)), q_hand=np.zeros((3, 2, 22)),
        T_world_base=np.eye(4), root_world=np.tile(np.eye(4), (2, 3, 1, 1)),
        arm_position_mm=np.zeros((2, 3)), arm_rotation_deg=np.zeros((2, 3)),
        hand_bone_deg=np.zeros((2, 3)), metrics={"gates": dict(m.PARENT_GATES)},
        base_report={"rule": m.BASE_RULE, "selected": {"back_m": .12}, "sample_frames": [0, 1, 2]})
    calls = []
    state = {"camera_blocked": False}

    def native_source(received, output, *, component_signature, v1_manifest_ref):
        calls.append("native")
        assert received["frame_mapping"] == frame_map
        assert v1_manifest_ref == manifest
        for ref in received["actual_inputs"].values():
            verify_ref(ref)
        output.mkdir(parents=True, exist_ok=False)
        native_outputs = [m.motion_source._npz(output / "NATIVE_MOTION.npz", {"qpc_ns": times}),
            m.motion_source._json(output / "FRAME_STATUS.json", [{"synthetic": True}]),
            m.motion_source._json(output / "MOTION_SOURCE_REPORT.json", {"synthetic": True})]
        receipt = {"schema": SCHEMA, "stage": "MotionSource", "status": "SEALED_COMPLETE",
            "component_signature": component_signature, "outputs": native_outputs,
            "actual_inputs": list(raw.values()) + [manifest, config]}
        m.motion_source._json(output / "RECEIPT.json", receipt)
        binding = {"failure_code": "UNKNOWN_CAMERA" if state["camera_blocked"] else None,
                   "projection_qualified": False}
        return {"raw_session": None if state["camera_blocked"] else session,
            "receipt": receipt, "report": {"camera_binding": binding}}

    monkeypatch.setattr(m.motion_source, "prepare_motion_source", native_source)
    monkeypatch.setattr(m, "_load_robot", lambda repo: calls.append("robot") or robot)
    monkeypatch.setattr(m, "_robot_asset_refs", lambda repo, model: [])
    monkeypatch.setattr(m, "_retarget_session", lambda supplied, model: calls.append("retarget") or result)
    return SimpleNamespace(root=tmp_path, raw=raw, source_ref=source_ref, source=source,
        source_dir=source_dir, manifest=manifest, config=config, frame_map=frame_map, report=report,
        session=session, result=result, state=state, calls=calls, output=tmp_path / "motion_stage")


def run(case, **change):
    options = dict(source_receipt=case.source_ref, source_signature="a" * 64,
        component_signature="b" * 64, output_dir=case.output, raw_inputs=case.raw,
        camera_contract={"v1_manifest": case.manifest, "camera_config": case.config}, repo_root=case.root)
    options.update(change)
    return m.build_motion_stage(**options)


def reseal_source(case):
    case.source_ref = json_ref(case.source_dir / "RECEIPT.json", case.source)


def test_motion_seals_exact_map_provenance_and_full_joint_shapes(case):
    receipt = run(case)
    assert case.calls == ["native", "robot", "retarget"]
    assert receipt["stage"] == "Motion" and receipt["status"] == "SEALED_COMPLETE"
    assert receipt["native_frames"] == 3 and receipt["kinematic_actions_generated"]
    verify_sealed_stage(receipt, "b" * 64, output_root=case.output)
    assert json.loads((case.output / "FRAME_MAP.json").read_text()) == case.frame_map
    rows = json.loads((case.output / "FRAME_STATUS.json").read_text())
    assert all(row["acquisition_frame_id"] == "UNKNOWN" for row in rows)
    assert rows[1]["native_motion_mapping"]["manus_right_source_line"] == 12
    assert rows[0]["raw_input_sha256"]["manus"] == case.raw["manus"]["sha256"]
    assert all(not row["training_eligible"] and not row["projection_qualified"] for row in rows)
    with np.load(case.output / "MOTION.npz", allow_pickle=False) as data:
        assert data["q_arm"].shape == (3, 2, 7) and data["q_hand"].shape == (3, 2, 22)
        np.testing.assert_array_equal(data["K"], case.session.K)
        assert data["q_arm_joint_names"].shape == (2, 7)
        assert data["q_hand_joint_names"].shape == (2, 22)
    report = json.loads((case.output / "MOTION_REPORT.json").read_text())
    assert report["base"]["rule"] == m.BASE_RULE
    assert report["qpc_frequency"] == "ABSENT" and report["qpc_sampling_position"] == "UNKNOWN"
    assert report["timebase_status"] == m.TIMEBASE and report["human_review"] == "PENDING"
    assert not report["historical_actions_read"] and not report["historical_placement_read"]
    assert not report["control_ground_truth"] and not report["training_eligible"]


def test_arm_numeric_fail_is_per_frame_not_whole_session_exit(case):
    case.result.arm_position_mm[1, 1] = 21.
    receipt = run(case)
    assert receipt["kinematic_actions_generated"]
    rows = json.loads((case.output / "FRAME_STATUS.json").read_text())
    assert rows[1]["hands"]["right"]["robot_status"] == "TRACK_GATE_FAIL"
    assert rows[0]["hands"]["right"]["robot_status"] == "CANDIDATE"
    with np.load(case.output / "MOTION.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["q_arm"], case.result.q_arm)
        assert not data["arm_tracking_valid"][1, 1]


def test_velocity_checks_use_native_nonuniform_intervals(case):
    case.result.q_arm[:, 0, 0] = [0., .16, .32]
    run(case)
    with np.load(case.output / "MOTION.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["arm_velocity_rad_s"][:, 0, 0], [4., 8. / 3.])
        assert data["qpc_velocity_violation"][1, 0]
        assert not data["qpc_velocity_violation"][2, 0]
        np.testing.assert_allclose(data["arm_acceleration_rad_s2"][0, 0, 0], -80. / 3.)


def test_completion_never_upgrades_source_evidence(case):
    case.session.hand_source[0, 1] = "INTERPOLATED_SHORT"
    case.session.hand_observed[0, 1] = False
    case.session.motion_valid["head_valid"][2] = False
    run(case)
    rows = json.loads((case.output / "FRAME_STATUS.json").read_text())
    assert rows[1]["hands"]["left"]["motion_source"] == "INTERPOLATED_SHORT"
    assert rows[1]["hands"]["left"]["robot_status"] == "SOURCE_EVIDENCE_INVALID"
    assert rows[2]["hands"]["right"]["robot_status"] == "SOURCE_EVIDENCE_INVALID"
    assert all(row["time_mode"] == "OFFLINE_NONCAUSAL_RETARGET_AND_DISPLAY_FILL" for row in rows)


@pytest.mark.parametrize("problem", ["partial", "source_stage", "signature", "identity", "count"])
def test_invalid_source_receipt_rejects_before_native_or_ik(case, problem):
    key, value = {"partial": ("status", "PARTIAL"), "source_stage": ("stage", "Clean"),
        "signature": ("component_signature", "c" * 64), "identity": ("session_id", "other"),
        "count": ("native_frames", 4)}[problem]
    case.source[key] = value
    reseal_source(case)
    with pytest.raises(ValueError):
        run(case)
    assert not case.calls and not case.output.exists()


@pytest.mark.parametrize("field", ["native_frame_index", "decoded_display_index", "model_input_frame", "output_frame", "qpc_declared_ns", "sbs_index", "physical_eye"])
def test_decoded_map_identity_changes_are_not_silently_adapted(case, field):
    case.frame_map[1][field] = "bad" if field == "physical_eye" else -1
    case.source["outputs"][1] = json_ref(case.source_dir / "FRAME_MAP.json", case.frame_map)
    reseal_source(case)
    with pytest.raises(m.MotionStageError, match="FRAME_MAPPING_CHANGED"):
        run(case)
    assert not case.calls


def test_raw_refs_must_equal_current_sealed_source(case):
    raw = deepcopy(case.raw)
    raw["robot_q"] = raw["manus"]
    with pytest.raises(m.MotionStageError, match="RAW_INPUT_BINDING_MISMATCH"):
        run(case, raw_inputs=raw)
    assert not case.calls


def test_only_exact_current_v1_camera_contract_is_accepted(case):
    other = json_ref(case.root / "old_manifest.json", {"baseline_id": "INPAINTING_RAW_TO_ROBOT_V1"})
    with pytest.raises(m.MotionStageError, match="ONLY_CURRENT_V1_CAMERA"):
        run(case, camera_contract={"v1_manifest": other, "camera_config": case.config})
    assert not case.calls


def test_camera_config_not_bound_by_v1_manifest_is_rejected(case):
    other = json_ref(case.root / "other_camera.json", {})
    with pytest.raises(m.MotionStageError, match="ADOPTED_CAMERA_CONTRACT_MISMATCH"):
        run(case, camera_contract={"v1_manifest": case.manifest, "camera_config": other})
    assert not case.calls


@pytest.mark.parametrize("blocked", ["camera", "nonfinite"])
def test_local_blockers_seal_without_fabricating_actions_or_blocking_mask_clean(case, blocked):
    if blocked == "camera":
        case.state["camera_blocked"] = True
    else:
        case.session.hands[0, 0, 0, 0] = np.nan
    receipt = run(case)
    assert receipt["status"] == "SEALED_COMPLETE"
    assert not receipt["kinematic_actions_generated"]
    assert receipt["consumer_status"]["Render"] == "BLOCKED"
    assert receipt["consumer_status"]["Mask"] == "INDEPENDENT"
    assert case.calls == ["native"]
    with np.load(case.output / "MOTION.npz", allow_pickle=False) as data:
        assert data.files == ["qpc_ns"]


@pytest.mark.parametrize("problem", ["q_arm_shape", "q_hand_nan", "gates", "base"])
def test_invalid_solver_contract_cannot_be_sealed_as_complete(case, problem):
    if problem == "q_arm_shape":
        case.result.q_arm = np.zeros((2, 2, 7))
    elif problem == "q_hand_nan":
        case.result.q_hand[0, 0, 0] = np.nan
    elif problem == "gates":
        case.result.metrics["gates"]["track_position_mm"] = 21.
    else:
        case.result.base_report["rule"] = "manual_placement"
    with pytest.raises(m.MotionStageError):
        run(case)
    assert not (case.output / "RECEIPT.json").exists()


def test_native_qpc_changed_by_parser_is_rejected_before_ik(case):
    case.session.qpc_ns = case.session.qpc_ns + 1
    with pytest.raises(m.MotionStageError, match="PARSED_NATIVE_IDENTITY_CHANGED"):
        run(case)
    assert case.calls == ["native"]


def test_input_drift_during_solver_prevents_final_receipt(case, monkeypatch):
    def changed_input(session, robot):
        Path(case.raw["manus"]["path"]).write_bytes(b"changed while solving")
        return case.result
    monkeypatch.setattr(m, "_retarget_session", changed_input)
    with pytest.raises(ValueError, match="INPUT_SHA_DRIFT"):
        run(case)
    assert not (case.output / "RECEIPT.json").exists()


def test_immutable_output_refuses_repeat_without_running_consumers(case):
    run(case)
    calls = list(case.calls)
    with pytest.raises(m.MotionStageError, match="FRESH_OUTPUT_REQUIRED"):
        run(case)
    assert case.calls == calls


@pytest.mark.parametrize("kind", ["source_signature", "component_signature"])
def test_signatures_are_required_before_any_consumer(case, kind):
    with pytest.raises(m.MotionStageError, match="SIGNATURE_REQUIRED"):
        run(case, **{kind: "not a SHA"})
    assert not case.calls and not case.output.exists()


def test_project_output_boundary_is_enforced_before_reading_or_writing(case):
    with pytest.raises(m.MotionStageError, match="OUTSIDE_PROJECT"):
        run(case, output_dir=case.root.parent / "forbidden_sibling")
    assert not case.calls


def test_asset_closure_comes_from_exact_verified_robot_pin_not_assets_scan(tmp_path):
    refs, urdfs, meshes = [], [], {}
    for index in range(66):
        relative = f"assets/robot/synthetic/{index}." + ("urdf" if index < 3 else "STL")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"pinned asset {index}".encode())
        pin = file_ref(path)
        refs.append({**pin, "path": relative})
        if index < 3:
            urdfs.append({"path": relative})
        else:
            meshes[relative] = SimpleNamespace()
    proxy = tmp_path / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json"
    proxy_ref = json_ref(proxy, {"proxy_transforms": "synthetic"})
    asset_pin = json_ref(tmp_path / "assets/robot/ROBOT_ASSET_PIN.json", {
        "status": "T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION", "files": refs, "urdfs": urdfs})
    robot = SimpleNamespace(assets=SimpleNamespace(asset_pin_record=asset_pin, mesh_records=meshes))
    actual = m._robot_asset_refs(tmp_path, robot)
    assert len(actual) == 68
    assert proxy_ref in actual and asset_pin in actual
    assert all("flangePRO" not in pin["path"] for pin in actual)
    Path(actual[-1]["path"]).write_bytes(b"SHA changed")
    with pytest.raises(ValueError, match="INPUT_SHA_DRIFT"):
        m._robot_asset_refs(tmp_path, robot)
