from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np

import pytest

from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions


BUILDER = (
    Path(__file__).resolve().parents[2]
    / "NOW/daemon/tools/build_robot_humanego_ict_bundle_v2.py"
)
SPEC = importlib.util.spec_from_file_location("hawor_ict_builder_v2", BUILDER)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_mano21_wrist_frame_uses_legacy_physical_axis_contract():
    points = np.zeros((21, 3), dtype=np.float64)
    points[9] = [0.0, 1.0, 0.0]
    points[5] = [0.0, 1.0, 1.0]
    rotation = builder.mano21_wrist_frame(points)
    np.testing.assert_allclose(rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) > 0.999999


def _bare_loader() -> FlowMatchingDataloader:
    loader = FlowMatchingDataloader.__new__(FlowMatchingDataloader)
    loader.hand_entity_key = "hands_hawor_v3"
    loader.hand_tracking_method = "hawor_v3"
    loader.single_hand = False
    loader.single_hand_side = "right"
    loader.use_object_tokens = True
    loader.max_ict = 8
    loader.ict_dim = 29
    loader.pos_mean = np.zeros(3, dtype=np.float32)
    loader.pos_std = np.ones(3, dtype=np.float32)
    loader.use_legacy_image_loading = False
    return loader


def test_entity_overlay_builds_nonzero_dual_hand_29d_ict_without_aliasing_pico():
    loader = _bare_loader()
    adapter = "/bundle/production/session/09_humanego_adapter"
    poses = np.repeat(np.eye(4)[None, None], 2, axis=0)
    poses = np.repeat(poses, 2, axis=1)
    poses[0, 0, 0, 3] = -0.1
    poses[0, 1, 0, 3] = 0.1
    loader._hawor_v3_sources = {
        adapter: {
            "frame_to_index": {"00000": 0, "00001": 1},
            "T_hand_to_world": poses,
            "T_hand_to_camera": poses,
            "valid": np.ones((2, 2), dtype=bool),
            "grasp": np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "confidence": np.ones((2, 2), dtype=np.float32),
        }
    }
    raw_hands = {"left": {"PICO_ONLY": True}, "right": {"PICO_ONLY": True}}
    frame = {
        "metadata": {
            "anchor_key": "obj1",
            "world_transforms": {"cam0": np.eye(4).tolist()},
        },
        "entities": {"hands": raw_hands, "objects": {}},
    }
    json_path = adapter + "/preprocess/all_data/00000/training_data.json"
    injected = loader._inject_hawor_v3_entity(frame, json_path)
    assert injected["entities"]["hands"] is raw_hands
    assert injected["entities"]["hands_hawor_v3"] is not raw_hands
    assert set(injected["entities"]["hands_hawor_v3"]) == {"left", "right"}

    state, _, mask = loader._build_ict(injected, np.eye(4))
    assert state.shape == (8, 29)
    assert mask.tolist() == [True, True, False, False, False, False, False, False]
    assert state[mask, 0].tolist() == [1.0, 2.0]
    assert np.isfinite(state[mask]).all()


def test_single_hand_frame_with_no_admitted_hawor_side_is_rejected():
    loader = _bare_loader()
    loader.single_hand = True
    loader.single_hand_side = "right"
    frame = {"entities": {"hands_hawor_v3": {}, "objects": {}}}
    assert loader._frame_has_required_fields(frame) is False


def _write_object_gate_fixture(tmp_path: Path):
    session = "play_cards_test"
    mps = tmp_path / session
    all_data = mps / "preprocess/all_data"
    for index in range(2):
        frame_dir = all_data / f"{index:05d}"
        frame_dir.mkdir(parents=True)
        left = np.eye(4)
        right = np.eye(4)
        left[0, 3] = -0.1
        right[0, 3] = 0.1
        c2w = np.eye(4)
        c2w[0, 3] = 0.02 * index
        payload = {
            "metadata": {
                "anchor_key": "camera_start",
                "camera_coordinate_system": {
                    "x": "right", "y": "down", "z": "forward",
                    "units": "metres",
                },
                "k": [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0],
                "c2w": c2w.tolist(),
                "world_transforms": {
                    "cam0": np.eye(4).tolist(),
                    "virtual_static_anchor": np.eye(4).tolist(),
                },
            },
            "obs": {},
            "entities": {
                "hands": {
                    "left": {"T_hand_to_world": left.tolist(), "grasp": 0.0},
                    "right": {"T_hand_to_world": right.tolist(), "grasp": 1.0},
                },
                "objects": {},
            },
        }
        (frame_dir / "training_data.json").write_text(json.dumps(payload))

    sidecar_root = tmp_path / "object_sidecars"
    sidecar = sidecar_root / session
    sidecar.mkdir(parents=True)
    transforms = np.full((2, 2, 4, 4), np.nan, dtype=np.float32)
    transforms[0, 0] = np.eye(4, dtype=np.float32)
    transforms[0, 0, :3, 3] = [0.2, 0.0, 0.6]
    transforms[0, 1] = np.eye(4, dtype=np.float32)
    transforms[0, 1, :3, 3] = [-0.2, 0.0, 0.7]
    transforms[1, 1] = np.eye(4, dtype=np.float32)
    transforms[1, 1, :3, 3] = [-0.22, 0.0, 0.7]
    valid = np.asarray([[True, True], [False, True]])
    confidence = np.asarray([[0.8, 0.5], [0.0, 0.7]], dtype=np.float32)
    provenance = np.asarray([
        ["AUTO_CV_RAY_HAWOR_FINGERTIP_Z", "AUTO_CV_STATIC_SIZE_PRIOR"],
        ["UNKNOWN", "AUTO_CV_STATIC_SIZE_PRIOR"],
    ])
    pose9 = np.full((2, 2, 9), np.nan, dtype=np.float32)
    for frame_index, object_index in np.argwhere(valid):
        pose9[frame_index, object_index, :3] = transforms[
            frame_index, object_index, :3, 3
        ]
        pose9[frame_index, object_index, 3:] = [1, 0, 0, 1, 0, 0]
    npz_path = sidecar / "AUTO_ESTIMATED_OBJECT_STATE.npz"
    np.savez_compressed(
        npz_path,
        frame_names=np.asarray(["00000", "00001"]),
        object_keys=np.asarray(["active_card_est", "card_rack_est"]),
        T_object_to_camera=transforms,
        valid=valid,
        confidence=confidence,
        provenance=provenance,
        role=np.asarray(["anchor_manipulated", "other_static_fixture"]),
        anchor_key=np.asarray("active_card_est"),
        object_type_ids=np.asarray([3, 4], dtype=np.int8),
        object_pose9_camera=pose9,
        formal_object6d_valid=np.zeros((2, 2), dtype=bool),
    )
    npz_digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "auto-estimated-object-state-review-v2",
        "consumption_authorized": False,
        "session": session,
        "array_contract": {
            "coordinate_frame": "current rectified left-camera optical frame",
            "units": {"translation": "metres", "rotation": "dimensionless 6D"},
        },
        "npz": {"path": str(npz_path), "sha256": npz_digest},
    }
    json_path = sidecar / "AUTO_ESTIMATED_OBJECT_STATE.json"
    json_path.write_text(json.dumps(manifest))
    return {
        "session": session,
        "mps": mps,
        "root": sidecar_root,
        "npz_sha256": npz_digest,
        "json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
    }


def _object_gate_loader(fixture: dict, mode: str) -> FlowMatchingDataloader:
    session = fixture["session"]
    return FlowMatchingDataloader(
        sessions=[MPSSessions(str(fixture["mps"]))],
        pred_horizon=1,
        single_hand=False,
        max_ict=8,
        img_name=None,
        centric_mode="object_centric",
        frame_mode="camera_frame",
        hand_tracking_method="aria_mps",
        object_state_sidecar_root=str(fixture["root"]),
        object_state_npz_sha256_by_session={session: fixture["npz_sha256"]},
        object_state_json_sha256_by_session={session: fixture["json_sha256"]},
        object_state_consumption_mode=mode,
        object_state_confidence_thresholds={
            "AUTO_CV_RAY_HAWOR_FINGERTIP_Z": 0.65,
            "AUTO_CV_STATIC_SIZE_PRIOR": 0.60,
        },
        use_pcd_features=False,
        use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False,
        enable_augmentation=False,
    )


def _promote_grade_b_fixture(fixture: dict, *, fill_anchor: bool = True) -> dict:
    session = fixture["session"]
    sidecar = fixture["root"] / session
    npz_path = sidecar / "AUTO_ESTIMATED_OBJECT_STATE.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if fill_anchor:
        values["T_object_to_camera"][1, 0] = np.eye(4, dtype=np.float32)
        values["T_object_to_camera"][1, 0, :3, 3] = [0.21, 0.0, 0.61]
        values["valid"][1, 0] = True
        values["confidence"][1, 0] = 0.75
        values["provenance"][1, 0] = "AUTO_CV_RAY_HAWOR_FINGERTIP_Z"
        values["object_pose9_camera"][1, 0, :3] = [0.21, 0.0, 0.61]
        values["object_pose9_camera"][1, 0, 3:] = [1, 0, 0, 1, 0, 0]
    np.savez_compressed(npz_path, **values)
    npz_digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "humanego-auto-estimated-object-grade-b-v1",
        "consumption_authorized": True,
        "claims_formal_object6d": False,
        "quality_grade": "B",
        "training_weight": 0.35,
        "session": session,
        "source_provenance": {"method": "test_rgb_hawor_geometry"},
        "array_contract": {
            "coordinate_frame": "current rectified left-camera optical frame",
            "units": {"translation": "metres", "rotation": "dimensionless 6D"},
        },
        "npz": {"path": str(npz_path), "sha256": npz_digest},
    }
    json_path = sidecar / "AUTO_ESTIMATED_OBJECT_STATE.json"
    json_path.write_text(json.dumps(manifest))
    return {
        **fixture,
        "npz_sha256": npz_digest,
        "json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
    }


def test_auto_estimated_object_gate_injects_only_high_confidence_tokens(tmp_path):
    fixture = _write_object_gate_fixture(tmp_path)
    loader = _object_gate_loader(fixture, "diagnostic_estimated")
    assert len(loader) == 2
    expected_types = ([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    for path, expected in zip(loader.samples, expected_types):
        frame = loader._read_frame(path)
        state, _, mask = loader._build_ict(frame, loader._get_T_w2ref(frame))
        assert state[mask, 0].tolist() == expected
        assert np.isfinite(state[mask]).all()
        for value in frame["entities"]["objects"].values():
            np.testing.assert_allclose(
                value["T_obj_to_world"],
                np.asarray(frame["metadata"]["c2w"])
                @ np.asarray(value["T_obj_to_camera"]),
            )
    report = loader.object_state_admission_report()["sessions"][fixture["session"]]
    assert report["admitted_by_object"] == [1, 1]
    assert report["pad_by_object"] == [1, 1]


def test_auto_estimated_object_sidecar_is_rejected_for_training(tmp_path):
    fixture = _write_object_gate_fixture(tmp_path)
    with pytest.raises(
        RuntimeError, match="HOLD_AUTO_ESTIMATED_OBJECT_NOT_TRAINING_AUTHORITY"
    ):
        _object_gate_loader(fixture, "training_admitted")


def test_grade_b_auto_estimated_object_is_low_weight_and_per_frame(tmp_path):
    fixture = _promote_grade_b_fixture(_write_object_gate_fixture(tmp_path))
    loader = _object_gate_loader(fixture, "training_estimated_grade_b")
    assert len(loader) == 2
    for path in loader.samples:
        frame = loader._read_frame(path)
        state, _, mask = loader._build_ict(frame, loader._get_T_w2ref(frame))
        assert 3.0 in state[mask, 0].tolist()
    assert loader[0]["object_training_weight"].item() == pytest.approx(0.35)
    report = loader.object_state_admission_report()["sessions"][fixture["session"]]
    assert report["admitted_by_object"][0] == 2
    assert report["training_weight"] == pytest.approx(0.35)


def test_grade_b_auto_estimated_object_rejects_missing_anchor_frame(tmp_path):
    fixture = _promote_grade_b_fixture(
        _write_object_gate_fixture(tmp_path), fill_anchor=False
    )
    with pytest.raises(RuntimeError, match="HOLD_GRADE_B_OBJECT_ANCHOR_NOT_PER_FRAME"):
        _object_gate_loader(fixture, "training_estimated_grade_b")
