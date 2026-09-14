from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


HUMANEGO = Path(__file__).resolve().parents[1]
BUILDER_PATH = HUMANEGO / "tools/build_two_task_single_session_e2e_bundle.py"
ENTRY_PATH = HUMANEGO / "tools/train_two_task_single_session_baseline.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load(BUILDER_PATH, "single_session_bundle_builder_test")
entry = load(ENTRY_PATH, "single_session_training_entry_test")


def test_h50_temporal_train_validation_support_is_disjoint() -> None:
    frames = np.arange(293, dtype=np.int64)
    valid = np.ones((293, 2), dtype=bool)
    split = builder.temporal_split(valid, frames)
    train = set(range(
        split["train"]["frame_start"], split["train"]["frame_stop_exclusive"]
    ))
    validation = set(range(
        split["validation"]["frame_start"],
        split["validation"]["frame_stop_exclusive"],
    ))
    assert train.isdisjoint(validation)
    assert split["support_intersection"] == []
    assert split["train"]["window_starts"]
    assert split["validation"]["window_starts"]
    for role, support in (("train", train), ("validation", validation)):
        for start in split[role]["window_starts"]:
            assert set(range(start, start + 51)).issubset(support)


def test_temporal_split_refuses_less_than_two_h50_plus_future_blocks() -> None:
    frames = np.arange(101, dtype=np.int64)
    valid = np.ones((101, 2), dtype=bool)
    with pytest.raises(builder.BundleHold, match="TWO_DISJOINT_H50"):
        builder.temporal_split(valid, frames)


def test_temporal_split_uses_only_longest_dual_valid_run() -> None:
    frames = np.arange(220, dtype=np.int64)
    valid = np.ones((220, 2), dtype=bool)
    valid[0:40] = False
    valid[170:] = False
    split = builder.temporal_split(valid, frames)
    assert split["train"]["frame_start"] == 40
    assert split["validation"]["frame_stop_exclusive"] == 170


def test_recipe_is_byte_rebound_and_preserves_effective_fields() -> None:
    contract = builder._recipe_snapshot()
    entry._assert_recipe(contract["frozen_effective_recipe"])
    recipe = contract["frozen_effective_recipe"]
    assert recipe["batch_size"] == 64
    assert recipe["eval_batch_size"] == 64
    assert recipe["epochs_in_recipe"] == 400
    assert recipe["maximum_durable_epoch_per_task"] == 180
    assert recipe["seed"] == 7
    assert recipe["contact_aux_weight"] == 0.0


def test_robot_authority_gate_refuses_current_grade_c_attempt() -> None:
    authority = builder.read_json(builder.AUTHORITY)
    result = Path(
        authority["stage_authorities"]["robot"]["chips_attempt1"]["result"]
    )
    with pytest.raises(builder.BundleHold):
        builder._authority_robot_row(authority, task="chips", robot_result=result)


def test_upstream_gate_accepts_current_hawor_via_agent_review() -> None:
    path = (
        builder.CONTROL / "hawor_fresh_bounded_v2_v1"
        / "get_potato_chips_0902_034/RESULT.json"
    )
    review = {"inputs": {"hawor_result": builder.ref(path)}}
    assert builder._verified_stage_review_input(review, "hawor_result") == path.resolve()


def test_task_sessions_and_checkpoint_names_are_independent() -> None:
    assert builder.TASKS == {
        "chips": "get_potato_chips_0902_034",
        "poker": "play_cards_0902_042",
    }
    assert entry.TASKS == builder.TASKS
    chips_tag = "chips_humanego_ict_single_session_baseline_v1"
    poker_tag = "poker_humanego_ict_single_session_baseline_v1"
    assert chips_tag != poker_tag


def test_old_exact78_and_fallback_data_lineage_is_fail_closed() -> None:
    forbidden = [
        "/mnt/workspace/code/chaoyang/HumanEgo/artifacts/"
        "newtask_robot_bundles/chips_ict_fresh78/session/sidecar.npz"
    ]
    with pytest.raises(builder.BundleHold, match="OLD_EXACT78"):
        builder._assert_current_data_lineage(map(Path, forbidden))
    with pytest.raises(entry.TrainingHold, match="OLD_EXACT78"):
        entry._assert_current_data_lineage(forbidden)


def test_full_bundle_and_epoch0_cpu_preflight_from_synthetic_current_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = "get_potato_chips_0902_034"
    count = 120
    source = tmp_path / "current_e2e_source"
    all_data = source / "preprocess/all_data"
    identity = np.eye(4).tolist()
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0l"
        "EQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    metadata = {
        "anchor_key": "physical_object_0",
        "camera_coordinate_system": {
            "x": "right", "y": "down", "z": "forward", "units": "metres",
        },
        "k": [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0],
        "c2w": identity,
        "world_transforms": {"cam0": identity, "virtual_static_anchor": identity},
        "w": 100,
        "h": 100,
        "is_finished": 0.0,
    }
    for frame in range(count):
        frame_dir = all_data / f"{frame:05d}"
        frame_dir.mkdir(parents=True)
        frame_dir.joinpath("training_data.json").write_text(
            json.dumps({
                "metadata": metadata,
                "obs": {},
                "entities": {
                    "hands": {
                        "left": {"T_hand_to_world": identity, "grasp": 0.2},
                        "right": {"T_hand_to_world": identity, "grasp": 0.8},
                    },
                    "objects": {},
                },
            }),
            encoding="utf-8",
        )
        frame_dir.joinpath("rgb.png").write_bytes(one_pixel_png)
    raw_video = source / "CameraRecord_synthetic.mp4"
    raw_video.write_bytes(b"synthetic-current-e2e-not-decoded-in-cpu-preflight")

    joints = np.zeros((2, count, 21, 3), dtype=np.float64)
    joints[..., 2] = 0.5
    joints[:, :, 5] = [0.03, 0.05, 0.5]
    joints[:, :, 9] = [0.0, 0.06, 0.5]
    hawor_npz = tmp_path / "current_hawor.npz"
    np.savez_compressed(
        hawor_npz,
        joints_3d_camera=joints,
        observed=np.ones((2, count), dtype=bool),
        detector_confidence=np.ones((2, count), dtype=np.float32),
        c2w=np.repeat(np.eye(4)[None], count, axis=0),
        original_frame_indices=np.arange(count, dtype=np.int64),
        mano_joint_names=np.asarray(
            ["wrist", *(f"joint_{index}" for index in range(1, 21))]
        ),
        anatomical_side_names=np.asarray(["left", "right"]),
    )

    from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

    assets = load_pinned_robot_assets(builder.PROJECT)
    q_sides = []
    for model in (assets.left_hand, assets.right_hand):
        moving = [joint for joint in model.joints if joint.joint_type != "fixed"]
        q_sides.append([(joint.lower + joint.upper) / 2.0 for joint in moving])
    q_hand = np.repeat(np.asarray(q_sides, dtype=np.float32)[None], count, axis=0)
    scene = tmp_path / "current_robot_scene.npz"
    np.savez_compressed(
        scene,
        source_frames=np.arange(count, dtype=np.int64),
        q_hand=q_hand,
        human_to_physical=np.asarray([1, 0], dtype=np.int64),
    )

    transforms = np.repeat(np.eye(4)[None], count, axis=0)
    transforms[:, 2, 3] = 0.7
    object_npz = tmp_path / "current_object6d.npz"
    np.savez_compressed(
        object_npz,
        frame_indices=np.arange(count, dtype=np.int64),
        valid=np.ones(count, dtype=bool),
        observed=np.ones(count, dtype=bool),
        visibility=np.ones(count, dtype=np.float32),
        T_object_to_camera=transforms,
    )

    lineage_files = {}
    for name in (
        "robot_result", "robot_review", "preflight", "spec", "kinematic",
        "hawor_result", "mask_result", "object_result", "clean_result",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"synthetic_current_lineage": name}), encoding="utf-8")
        lineage_files[name] = path
    robot_result = lineage_files["robot_result"]
    authority = tmp_path / "BASELINE_AUTHORITY.json"
    authority.write_text(json.dumps({
        "stage_authorities": {"robot": {"chips": {
            "result": str(robot_result),
            "result_sha256": hashlib.sha256(robot_result.read_bytes()).hexdigest(),
            "grade": "B",
            "downstream_authorized": True,
        }}}
    }), encoding="utf-8")
    monkeypatch.setattr(entry, "LIVE_AUTHORITY", authority)
    bundle = tmp_path / "bundle"
    builder.materialize(
        task="chips",
        session=session,
        bundle=bundle,
        authority_path=authority,
        lineage={
            "robot_result": {"grade": "B"},
            "robot_result_path": robot_result,
            "robot_review_path": lineage_files["robot_review"],
            "preflight_path": lineage_files["preflight"],
            "spec_path": lineage_files["spec"],
            "kinematic_result_path": lineage_files["kinematic"],
            "scene_path": scene,
            "hawor_result_path": lineage_files["hawor_result"],
            "hawor_npz": hawor_npz,
            "object6d_result_path": lineage_files["object_result"],
            "object6d_npz": object_npz,
            "mask_result_path": lineage_files["mask_result"],
            "clean_result_path": lineage_files["clean_result"],
            "raw_video": raw_video,
            "mps_path": source,
            "raw_all_data": all_data,
            "fps": 29.97,
        },
    )
    report = entry.cpu_preflight(bundle, tmp_path / "CPU_EPOCH0_PREFLIGHT.json")
    from jsonschema.validators import validator_for

    schema = json.loads(
        (builder.CONTROL / "training_prepare_v1/BUNDLE_SCHEMA.json").read_text(
            encoding="utf-8"
        )
    )
    validator = validator_for(schema)
    validator.check_schema(schema)
    validator(schema).validate(builder.read_json(bundle / "freeze.json"))
    assert report["status"] == "PASS_CPU_EPOCH0_NO_TRAINING"
    assert report["support_intersection"] == []
    assert report["loader"]["train"]["windows"] == 19
    assert report["loader"]["validation"]["windows"] == 1
    assert report["legacy_exact78_training_data_consumed"] is False
    assert report["grade_c_or_fallback_consumed"] is False
    assert report["gpu_used"] is False
    assert report["training_started"] is False
    assert {path.stat().st_nlink for path in bundle.rglob("*") if path.is_file()} == {1}
    with np.load(
        bundle / "sidecars/kai22" / session / "sidecar.npz", allow_pickle=False,
    ) as sidecar:
        assert int(sidecar["timestamps_ns"][1]) == round(1e9 / 29.97)
