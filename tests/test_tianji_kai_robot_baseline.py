from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import jsonschema
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT / "tools/run_tianji_kai_robot_baseline.py"
POST_PATH = PROJECT / "pipeline/tianji_kai_robot_baseline_post.py"
SCHEMA_PATH = PROJECT / "contracts/tianji_kai_robot_baseline_session_spec_v1.schema.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load(RUNNER_PATH, "test_tianji_kai_robot_baseline_runner")


def artifact(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def repin_clean_source_manifest(spec_path: Path, manifest: dict[str, object]) -> None:
    value = json.loads(spec_path.read_text())
    manifest_path = Path(value["inputs"]["clean"]["source_map_manifest"]["path"])
    write_json(manifest_path, manifest)
    manifest_ref = artifact(manifest_path)
    value["inputs"]["clean"]["source_map_manifest"] = manifest_ref
    review_path = Path(value["inputs"]["clean"]["agent_review"]["path"])
    review = json.loads(review_path.read_text())
    review["artifacts"]["source_map_manifest"] = manifest_ref
    write_json(review_path, review)
    value["inputs"]["clean"]["agent_review"] = artifact(review_path)
    write_json(spec_path, value)


def stage(
    root: Path,
    *,
    name: str,
    session: str,
    frame_count: int,
    extras: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    result_path = root / f"{name}_RESULT.json"
    write_json(
        result_path,
        {
            "schema_version": f"synthetic-{name}-v1",
            "status": "PASS_BASELINE",
            "consumption_authorized": True,
            "task": "chips",
            "session_id": session,
            "frame_count": frame_count,
        },
    )
    review_path = root / f"{name}_AGENT_REVIEW.json"
    review_artifacts = {"result": artifact(result_path)}
    if extras and name == "clean":
        review_artifacts.update(extras)
    write_json(
        review_path,
        {
            "schema_version": "baseline-agent-stage-review-v1",
            "created_at": "2026-09-08T18:00:00+08:00",
            "run_id": "20260908_two_task_e2e_baseline_v1",
            "stage": name.upper(),
            "task": "chips",
            "session": session,
            "grade": "B",
            "downstream_authorized": True,
            "hard_gates": {"identity": "PASS"},
            "soft_defects": ["synthetic fixture"],
            "inputs": {},
            "artifacts": review_artifacts,
            "claim_limit": "synthetic CPU test",
        },
    )
    value = {"result": artifact(result_path), "agent_review": artifact(review_path)}
    if extras:
        value.update(extras)
    return value


@pytest.fixture()
def valid_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    frame_count = 24
    session = "get_potato_chips_0902_034"
    raw = tmp_path / "raw.mp4"
    clean = tmp_path / "clean.mp4"
    raw.write_bytes(b"synthetic raw video")
    clean.write_bytes(b"synthetic clean video")
    raw_root = tmp_path / "raw_root"
    raw_root.mkdir()
    for frame_id in range(frame_count):
        frame_dir = raw_root / "preprocess/all_data" / f"{frame_id:05d}"
        frame_dir.mkdir(parents=True)
        (frame_dir / "rgb.png").write_bytes(f"synthetic-{frame_id}".encode())

    c2w = np.repeat(np.eye(4)[None], frame_count, axis=0)
    k = np.repeat(np.asarray([[[320.0, 0.0, 319.5], [0.0, 320.0, 239.5], [0.0, 0.0, 1.0]]]), frame_count, axis=0)
    joints = np.zeros((2, frame_count, 21, 3), dtype=np.float32)
    joints[..., 2] = 0.8
    hawor_npz = tmp_path / "hawor.npz"
    np.savez_compressed(
        hawor_npz,
        joints_3d_world=joints,
        joints_3d_camera=joints,
        joints_2d=np.zeros((2, frame_count, 21, 2), dtype=np.float32),
        observed=np.ones((2, frame_count), dtype=bool),
        provenance=np.full((2, frame_count), "BOUNDED_PARAMETER_FIT"),
        c2w=c2w,
        intrinsics=k,
        original_frame_indices=np.arange(frame_count, dtype=np.int32),
        fps=np.asarray(30.0),
        mano_joint_names=np.asarray(runner.MANO_NAMES),
        mano_wrist_index=np.asarray(0),
        mano_tip_indices=np.asarray((4, 8, 12, 16, 20)),
    )

    object_camera = np.repeat(np.eye(4)[None], frame_count, axis=0)
    object_camera[:, 2, 3] = 0.8
    object_npz = tmp_path / "object6d.npz"
    np.savez_compressed(
        object_npz,
        frame_indices=np.arange(frame_count),
        valid=np.ones(frame_count, dtype=bool),
        observed=np.ones(frame_count, dtype=bool),
        visibility=np.ones(frame_count),
        physical_instance_id=np.zeros(frame_count, dtype=np.int32),
        T_object_to_camera=object_camera,
        T_object_to_world=object_camera,
        observed_near_far_optical_z_m=np.repeat([[0.78, 0.82]], frame_count, axis=0),
        analytic_near_far_optical_z_m=np.repeat([[0.77, 0.83]], frame_count, axis=0),
        object_size_m=np.asarray([0.05, 0.05, 0.04]),
    )

    frame_manifest = tmp_path / "FRAME_MANIFEST.json"
    write_json(
        frame_manifest,
        {"session_id": session, "frames": [{"frame_id": index} for index in range(frame_count)]},
    )
    right_stereo = tmp_path / "right_stereo.mp4"
    right_stereo.write_bytes(b"synthetic synchronized stereo video")
    source_rows = []
    source_y, source_x = np.indices((12, 16), dtype=np.int32)
    for frame_id in range(frame_count):
        source_map = tmp_path / f"source_map_{frame_id:06d}.npz"
        np.savez_compressed(
            source_map,
            frame_id=np.int32(frame_id),
            source_kind=np.zeros((12, 16), dtype=np.uint8),
            source_eye=np.zeros((12, 16), dtype=np.uint8),
            source_frame=np.full((12, 16), frame_id, dtype=np.int32),
            source_x=source_x,
            source_y=source_y,
        )
        source_rows.append(
            {
                "frame_id": frame_id,
                "pixel_source_map": artifact(source_map),
                "selected_left_raw_decoded_sha256": hashlib.sha256(
                    f"left-{frame_id}".encode()
                ).hexdigest(),
                "right_raw_decoded_sha256": hashlib.sha256(
                    f"right-{frame_id}".encode()
                ).hexdigest(),
            }
        )
    source_map_manifest = tmp_path / "SOURCE_MAP_MANIFEST.json"
    write_json(
        source_map_manifest,
        {
            "session_id": session,
            "task": "chips",
            "frame_count": frame_count,
            "source_lineage": {
                "selected_left_raw_video": artifact(raw),
                "synchronized_right_stereo_video": artifact(right_stereo),
            },
            "source_kind_codes": {
                "0": "TARGET_RAW",
                "1": "SAME_SESSION_LEFT_TEMPORAL_RAW",
                "2": "PROTECTED_OBJECT_RAW",
                "3": "UNSUPPORTED_RAW",
                "4": "SAME_SESSION_SYNCHRONIZED_RIGHT_RAW",
            },
            "frames": source_rows,
        },
    )
    canonical_assets = {
        name: artifact(path) for name, path in runner.CANONICAL_ASSETS.items()
    }

    hawor = stage(
        tmp_path, name="hawor", session=session, frame_count=frame_count,
        extras={"npz": artifact(hawor_npz)},
    )
    mask = stage(
        tmp_path, name="mask", session=session, frame_count=frame_count,
        extras={"frame_manifest": artifact(frame_manifest)},
    )
    object_stage = stage(
        tmp_path, name="object6d", session=session, frame_count=frame_count,
        extras={"npz": artifact(object_npz), "frame_manifest": artifact(frame_manifest)},
    )
    clean_stage = stage(
        tmp_path, name="clean", session=session, frame_count=frame_count,
        extras={"source_map_manifest": artifact(source_map_manifest), "video": artifact(clean)},
    )
    clean_review_path = Path(clean_stage["agent_review"]["path"])
    clean_review = json.loads(clean_review_path.read_text())
    clean_review["inputs"].update(
        {
            "selected_left_raw_video": artifact(raw),
            "synchronized_right_stereo_video": artifact(right_stereo),
        }
    )
    write_json(clean_review_path, clean_review)
    clean_stage["agent_review"] = artifact(clean_review_path)
    spec_path = tmp_path / "SESSION_SPEC.json"
    value = {
        "schema_version": "tianji-kai-robot-baseline-session-spec-v1",
        "session": {"task": "chips", "session_id": session, "frame_count": frame_count, "fps": 30.0},
        "inputs": {
            "raw_video": artifact(raw),
            "raw_frame_root": str(raw_root.resolve()),
            "raw_frame_tree_sha256": runner._raw_frame_tree_digest(
                raw_root.resolve(), frame_count
            ),
            "hawor": hawor,
            "mask": mask,
            "object6d": object_stage,
            "clean": clean_stage,
        },
        "assets": canonical_assets,
        "solver": {
            "arm_step_limit_rad": 0.12,
            "hand_step_limit_rad": 0.08,
            "later_frame_seed": "PREVIOUS_ACCEPTED_ONLY",
            "first_frame_policy": "STATIC_IK_FULL_URDF_LIMITS",
            "human_to_physical": {"left": "right", "right": "left"},
            "require_urdf_limits": True,
            "require_branch_jump_false": True,
            "collision_denominator": "FRAME_X_SIDE_X_LINK_FIXED",
            "pad_names": ["thumb", "index", "middle", "ring", "pinky"],
        },
        "review": {
            "language": "zh-CN",
            "full_session": True,
            "panels": ["raw", "clean", "robot", "contact_collision"],
            "video_name": "ROBOT_BASELINE_中文全片.mp4",
        },
        "output_root": str((tmp_path / "robot_output").resolve()),
    }
    write_json(spec_path, value)
    monkeypatch.setattr(
        runner,
        "_probe_video",
        lambda _path, _name, expected_frames, fps: {
            "frames": expected_frames, "fps": fps, "width": 16, "height": 12
        },
    )
    return spec_path


def test_schema_and_validate_only_are_closed(valid_spec: Path) -> None:
    value = json.loads(valid_spec.read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)
    before = set(sys.modules)
    _, result = runner.validate_spec(valid_spec)
    added = set(sys.modules) - before
    assert result["status"] == "PASS_VALIDATE_ONLY_READY_WAIT_UPSTREAM_EXECUTION"
    assert result["geometry"]["physical_instance_id"] == 0
    assert result["mount_authority"]["formal_consumer_allowed"] is False
    assert not any(name.startswith("pipeline.robot_") for name in added)
    assert not Path(value["output_root"]).exists()


def test_validate_only_accepts_explicit_bounded_object_occlusion(valid_spec: Path) -> None:
    value = json.loads(valid_spec.read_text())
    object_ref = value["inputs"]["object6d"]["npz"]
    object_path = Path(object_ref["path"])
    with np.load(object_path, allow_pickle=False) as values:
        payload = {name: values[name] for name in values.files}
    payload["observed"] = payload["observed"].copy()
    payload["visibility"] = payload["visibility"].copy()
    payload["observed_near_far_optical_z_m"] = payload[
        "observed_near_far_optical_z_m"
    ].copy()
    payload["observed"][7:10] = False
    payload["visibility"][7:10] = 0.0
    payload["observed_near_far_optical_z_m"][7:10] = np.nan
    np.savez_compressed(object_path, **payload)
    value["inputs"]["object6d"]["npz"] = artifact(object_path)
    write_json(valid_spec, value)
    _, result = runner.validate_spec(valid_spec)
    assert result["status"] == "PASS_VALIDATE_ONLY_READY_WAIT_UPSTREAM_EXECUTION"
    assert result["geometry"]["object6d_direct_observed_fraction"] == pytest.approx(0.875)
    assert result["geometry"]["object6d_max_propagated_gap_frames"] == 3


def test_validate_only_rejects_byte_drift(valid_spec: Path) -> None:
    value = json.loads(valid_spec.read_text())
    Path(value["inputs"]["mask"]["result"]["path"]).write_text("drift")
    with pytest.raises(runner.BaselineContractError, match="byte identity drift"):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_raw_frame_tree_drift(valid_spec: Path) -> None:
    value = json.loads(valid_spec.read_text())
    raw_root = Path(value["inputs"]["raw_frame_root"])
    (raw_root / "preprocess/all_data/00012/rgb.png").write_bytes(b"drift")
    with pytest.raises(runner.BaselineContractError, match="RAW frame tree byte identity drift"):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_grade_c_review(valid_spec: Path) -> None:
    value = json.loads(valid_spec.read_text())
    review_path = Path(value["inputs"]["mask"]["agent_review"]["path"])
    review = json.loads(review_path.read_text())
    review["grade"] = "C"
    review["downstream_authorized"] = False
    review["hard_gates"] = {"identity": "FAIL"}
    write_json(review_path, review)
    value["inputs"]["mask"]["agent_review"] = artifact(review_path)
    write_json(valid_spec, value)
    with pytest.raises(runner.BaselineContractError, match="not an admitted A/B"):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_clean_video_not_bound_by_review(valid_spec: Path) -> None:
    value = json.loads(valid_spec.read_text())
    review_path = Path(value["inputs"]["clean"]["agent_review"]["path"])
    review = json.loads(review_path.read_text())
    review["artifacts"].pop("video")
    write_json(review_path, review)
    value["inputs"]["clean"]["agent_review"] = artifact(review_path)
    write_json(valid_spec, value)
    with pytest.raises(
        runner.BaselineContractError,
        match="clean review does not bind exact video SHA",
    ):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_clean_source_map_not_bound_by_review(
    valid_spec: Path,
) -> None:
    value = json.loads(valid_spec.read_text())
    review_path = Path(value["inputs"]["clean"]["agent_review"]["path"])
    review = json.loads(review_path.read_text())
    review["artifacts"].pop("source_map_manifest")
    write_json(review_path, review)
    value["inputs"]["clean"]["agent_review"] = artifact(review_path)
    write_json(valid_spec, value)
    with pytest.raises(
        runner.BaselineContractError,
        match="clean review does not bind exact source_map_manifest SHA",
    ):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_tiled_clean_review_as_robot_background(
    valid_spec: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def probe(_path: Path, name: str, expected_frames: int, fps: float):
        width, height = (1280, 960) if name == "Clean video" else (640, 480)
        return {
            "frames": expected_frames,
            "fps": fps,
            "width": width,
            "height": height,
        }

    monkeypatch.setattr(runner, "_probe_video", probe)
    with pytest.raises(
        runner.BaselineContractError,
        match="full-resolution Clean master matching RAW",
    ):
        runner.validate_spec(valid_spec)


def test_validate_only_accepts_same_session_left_temporal_clean_source(
    valid_spec: Path,
) -> None:
    value = json.loads(valid_spec.read_text())
    manifest_path = Path(value["inputs"]["clean"]["source_map_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    row = manifest["frames"][4]
    source_path = Path(row["pixel_source_map"]["path"])
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["source_kind"] = arrays["source_kind"].copy()
    arrays["source_frame"] = arrays["source_frame"].copy()
    arrays["source_kind"][3, 5] = 1
    arrays["source_frame"][3, 5] = 1
    np.savez_compressed(source_path, **arrays)
    row["pixel_source_map"] = artifact(source_path)
    repin_clean_source_manifest(valid_spec, manifest)
    _, result = runner.validate_spec(valid_spec)
    assert result["status"] == "PASS_VALIDATE_ONLY_READY_WAIT_UPSTREAM_EXECUTION"


def test_validate_only_accepts_right_raw_coordinate_outside_left_canvas(
    valid_spec: Path,
) -> None:
    value = json.loads(valid_spec.read_text())
    manifest_path = Path(value["inputs"]["clean"]["source_map_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    row = manifest["frames"][4]
    source_path = Path(row["pixel_source_map"]["path"])
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name in ("source_kind", "source_eye", "source_x", "source_y"):
        arrays[name] = arrays[name].copy()
    arrays["source_kind"][3, 5] = 4
    arrays["source_eye"][3, 5] = 1
    arrays["source_x"][3, 5] = 1800
    arrays["source_y"][3, 5] = 1200
    np.savez_compressed(source_path, **arrays)
    row["pixel_source_map"] = artifact(source_path)
    repin_clean_source_manifest(valid_spec, manifest)
    _, result = runner.validate_spec(valid_spec)
    assert result["status"] == "PASS_VALIDATE_ONLY_READY_WAIT_UPSTREAM_EXECUTION"


def test_validate_only_rejects_right_raw_coordinate_outside_2048x1536(
    valid_spec: Path,
) -> None:
    value = json.loads(valid_spec.read_text())
    manifest_path = Path(value["inputs"]["clean"]["source_map_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    row = manifest["frames"][4]
    source_path = Path(row["pixel_source_map"]["path"])
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name in ("source_kind", "source_eye", "source_x", "source_y"):
        arrays[name] = arrays[name].copy()
    arrays["source_kind"][3, 5] = 4
    arrays["source_eye"][3, 5] = 1
    arrays["source_x"][3, 5] = 2048
    arrays["source_y"][3, 5] = 1200
    np.savez_compressed(source_path, **arrays)
    row["pixel_source_map"] = artifact(source_path)
    repin_clean_source_manifest(valid_spec, manifest)
    with pytest.raises(
        runner.BaselineContractError,
        match="right-raw source coordinate out of range",
    ):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_left_temporal_coordinate_outside_1280x960(
    valid_spec: Path,
) -> None:
    value = json.loads(valid_spec.read_text())
    manifest_path = Path(value["inputs"]["clean"]["source_map_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    row = manifest["frames"][4]
    source_path = Path(row["pixel_source_map"]["path"])
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["source_kind"] = arrays["source_kind"].copy()
    arrays["source_frame"] = arrays["source_frame"].copy()
    arrays["source_x"] = arrays["source_x"].copy()
    arrays["source_kind"][3, 5] = 1
    arrays["source_frame"][3, 5] = 1
    arrays["source_x"][3, 5] = 1280
    np.savez_compressed(source_path, **arrays)
    row["pixel_source_map"] = artifact(source_path)
    repin_clean_source_manifest(valid_spec, manifest)
    with pytest.raises(
        runner.BaselineContractError,
        match="selected-left source coordinate out of range",
    ):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_unsynchronized_right_clean_source(
    valid_spec: Path,
) -> None:
    value = json.loads(valid_spec.read_text())
    manifest_path = Path(value["inputs"]["clean"]["source_map_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    row = manifest["frames"][4]
    source_path = Path(row["pixel_source_map"]["path"])
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["source_kind"] = arrays["source_kind"].copy()
    arrays["source_eye"] = arrays["source_eye"].copy()
    arrays["source_frame"] = arrays["source_frame"].copy()
    arrays["source_kind"][3, 5] = 4
    arrays["source_eye"][3, 5] = 1
    arrays["source_frame"][3, 5] = 3
    np.savez_compressed(source_path, **arrays)
    row["pixel_source_map"] = artifact(source_path)
    repin_clean_source_manifest(valid_spec, manifest)
    with pytest.raises(
        runner.BaselineContractError,
        match="right donor is not synchronized",
    ):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_left_lineage_not_equal_to_robot_raw(
    valid_spec: Path,
) -> None:
    value = json.loads(valid_spec.read_text())
    manifest_path = Path(value["inputs"]["clean"]["source_map_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    alternate = manifest_path.parent / "alternate_left.mp4"
    alternate.write_bytes(b"not the selected Robot RAW")
    manifest["source_lineage"]["selected_left_raw_video"] = artifact(alternate)
    repin_clean_source_manifest(valid_spec, manifest)
    with pytest.raises(
        runner.BaselineContractError,
        match="does not bind the exact Robot RAW video",
    ):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_right_lineage_not_bound_by_clean_review(
    valid_spec: Path,
) -> None:
    value = json.loads(valid_spec.read_text())
    review_path = Path(value["inputs"]["clean"]["agent_review"]["path"])
    review = json.loads(review_path.read_text())
    review["inputs"].pop("synchronized_right_stereo_video")
    write_json(review_path, review)
    value["inputs"]["clean"]["agent_review"] = artifact(review_path)
    write_json(valid_spec, value)
    with pytest.raises(
        runner.BaselineContractError,
        match="review does not bind exact synchronized_right_stereo_video SHA",
    ):
        runner.validate_spec(valid_spec)


def test_validate_only_rejects_nonfresh_output(valid_spec: Path) -> None:
    value = json.loads(valid_spec.read_text())
    Path(value["output_root"]).mkdir()
    with pytest.raises(runner.BaselineContractError, match="fresh output_root"):
        runner.validate_spec(valid_spec)


def test_exact_triangle_box_sat_preserves_rows() -> None:
    post = load(POST_PATH, "test_tianji_kai_robot_baseline_post")
    triangles = np.asarray(
        [
            [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [[2.0, 2.0, 2.0], [2.1, 2.0, 2.0], [2.0, 2.1, 2.0]],
        ]
    )
    result = post.triangle_box_sat(triangles, np.asarray([0.5, 0.5, 0.5]))
    assert result.tolist() == [True, False]


def test_kinematic_runner_exposes_strict_contract_flags() -> None:
    text = (PROJECT / "tools/run_newtask_robot_kinematic_canary.py").read_text()
    for token in (
        "--first-frame-static-exception",
        "--strict-previous-accepted",
        "--arm-step-limit",
        "--hand-step-limit",
        '"branch_jump"',
    ):
        assert token in text


def test_chips_validate_only_binds_v2_method_before_robot_import(valid_spec: Path) -> None:
    spec, preflight = runner.validate_spec(valid_spec)
    method = preflight["kinematic_method"]
    reference = method["formal_method_contract"]
    assert method["name"] == "V2_ONE_STEP_STOP_VIABILITY_FRAME0_STATIC_DEVELOPMENT_PLACEMENT"
    assert method["validated_before_robot_import"] is True
    assert method["target_is_objective_not_hard_bound"] is True
    assert reference["sha256"] == runner.EXPECTED_CHIPS034_V2_METHOD_CONTRACT_SHA256
    command = runner._build_kinematic_command(
        spec, preflight, Path(spec["output_root"]) / "kinematic"
    )
    assert "--gradeb-successor-contract" in command
    assert command[command.index("--gradeb-successor-contract") + 1] == reference["path"]
    assert "--prefer-earliest-window" in command
    assert "--fast-hand-retarget" not in command


def test_chips_validate_only_fails_closed_on_v2_contract_sha_drift(
    valid_spec: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "EXPECTED_CHIPS034_V2_METHOD_CONTRACT_SHA256",
        "0" * 64,
    )
    with pytest.raises(runner.BaselineContractError, match="method contract SHA drift"):
        runner.validate_spec(valid_spec)


def test_hawor_result_nested_full_video_frame_identity_is_supported() -> None:
    runner._result_identity(
        {
            "task": "chips",
            "session_id": "get_potato_chips_0902_034",
            "validation": {"full_video": {"output_frames": 293}},
        },
        task="chips",
        session_id="get_potato_chips_0902_034",
        frame_count=293,
        name="HaWoR result",
    )


def test_branch_jump_failure_writes_central_grade_c_terminal(valid_spec: Path) -> None:
    spec, preflight = runner.validate_spec(valid_spec)
    output = Path(spec["output_root"])
    output.mkdir()
    runner._atomic_new_json(output / "PREFLIGHT_RESULT.json", preflight)
    result = runner._write_grade_c_execution_receipt(
        output,
        spec,
        preflight,
        phase="KINEMATIC_SUBPROCESS",
        reason="RuntimeError: strict previous-accepted frame gate failed at source frame 12",
    )
    review = json.loads((output / "AGENT_REVIEW.json").read_text())
    validation = json.loads((output / "AGENT_REVIEW_VALIDATION.json").read_text())
    assert result["status"] == "FAIL_GRADE_C_BRANCH_JUMP"
    assert result["branch_jump"] is True
    assert review["grade"] == "C"
    assert review["downstream_authorized"] is False
    assert review["hard_gates"]["previous_accepted_branch"] == "FAIL"
    assert validation["status"] == "PASS_BASELINE_AGENT_REVIEW"
