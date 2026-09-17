from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import jsonschema
import numpy as np
import pytest

from chaoyang.ops import build_handle_h1_h2_sidecars_v71 as sidecars


PROJECT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, controller: bool = True) -> tuple[Path, np.ndarray]:
    target = tmp_path / "processed" / "play_cards_0910_001"
    target.mkdir(parents=True)
    source = tmp_path / "source" / "dataset.hdf5"
    source.parent.mkdir()
    n = 3
    tactile = np.arange(n * 2 * 5 * 4 * 8, dtype=np.int16).reshape(n, 2, 5, 4, 8)
    local = np.zeros((n, 2, 25, 3), dtype=np.float64)
    local[..., 0] = np.arange(25)[None, None]
    with h5py.File(source, "w") as handle:
        handle.attrs["joint_names"] = json.dumps(list(sidecars.MANUS25_NAMES))
        handle.attrs["n_joint"] = 25
        handle.attrs["tactile_gate_ms"] = 40.0
        handle.attrs["tactile_finger_mapping_schema"] = "hs13_five_fingertip_4x8_four_4x7_v1"
        handle.attrs["tactile_palm_present"] = False
        handle.create_dataset("timestamp_ns", data=np.array([100, 200, 300], np.int64))
        handle.create_dataset("source_row_idx", data=np.arange(n, dtype=np.int64))
        handle.create_dataset("video_frame_idx", data=np.arange(1, n + 1, dtype=np.int32))
        handle.create_dataset("segment_id", data=np.zeros(n, np.int32))
        for s, side in enumerate(sidecars.SIDES):
            handle.create_dataset(f"{side}_hand_joints", data=local[:, s])
            handle.create_dataset(f"{side}_hand_valid", data=np.ones(n, bool))
            handle.create_dataset(f"{side}_tactile_fingers", data=tactile[:, s])
            active = np.ones((5, 4, 8), bool)
            if side == "right":
                active[1:, :, 7] = False
            handle.create_dataset(f"{side}_tactile_fingers_active_mask", data=active)
            handle.create_dataset(f"{side}_tactile_valid", data=np.ones(n, bool))
            handle.create_dataset(f"{side}_tactile_offset_ms", data=np.array([1, -2, 3], np.float32))
            handle.create_dataset(f"{side}_tactile_record_seq", data=np.arange(10, 13, dtype=np.int64))
            handle.create_dataset(f"{side}_tactile_stream_seq", data=np.arange(20, 23, dtype=np.int64))
    controller_path = target / "controller.jsonl"
    if controller:
        with controller_path.open("w") as stream:
            for frame, timestamp in enumerate((100, 200, 300)):
                hands = {}
                for s, side in enumerate(sidecars.SIDES):
                    world = np.eye(4)
                    world[:3, 3] = [10 + s, 20, 30]
                    camera = np.eye(4)
                    camera[:3, 3] = [1 + s, 2, 3]
                    hands[side] = {"T_wrist_to_world": world.tolist(), "T_wrist_to_camera": camera.tolist()}
                stream.write(json.dumps({"frame_index": frame, "timeStampNs": timestamp, "hands": hands}) + "\n")
    (target / "clip_manifest.json").write_text(json.dumps({
        "files": {"controller_poses": "controller.jsonl"},
        "acquisition_aligned_hdf5_contract": {"path": str(source), "sha256": _sha(source)},
    }))
    h0 = tmp_path / "h0.json"
    h0.write_text(json.dumps({
        "row_count": 2,
        "rows": [
            {
                "session_id": "play_cards_0910_001", "dataset_id": "chips_cards_handle_0910",
                "task": "playing_cards", "target": str(target), "frame_count": n,
                "admission": "SENSOR_GEOMETRY_READY",
                "eligibility": {"controller_wrist": {"eligible": controller}},
            },
            {
                "session_id": "play_cards_0909_104", "dataset_id": "chips_cards_handle_0909",
                "task": "playing_cards", "target": None, "frame_count": None,
                "admission": "BLOCKED_SOURCE", "eligibility": {"controller_wrist": {"eligible": False}},
            },
        ],
    }))
    return h0, tactile


def test_h1_explicit_mapping_and_coordinate_views(tmp_path: Path) -> None:
    h0, _ = _fixture(tmp_path)
    output = tmp_path / "h1"
    ledger = sidecars.build_stage(h0, output, "h1")
    assert ledger["summary"]["by_status"] == {"BLOCKED_SOURCE": 1, "PASSED": 1}
    assert ledger["observation_contract"]["name"] == "HAND21_FROM_MANUS_CONTROLLER"
    artifact = np.load(output / "sidecars/playing_cards/play_cards_0910_001.npz")
    assert artifact["joint_xyz_wrist_local_m"].shape == (3, 2, 21, 3)
    assert artifact["joint_xyz_wrist_local_m"][0, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert artifact["joint_xyz_wrist_local_m"][0, 0, 7, 0] == 8
    assert np.allclose(artifact["joint_xyz_world_m"][0, 0, 0], [10, 20, 30])
    assert np.allclose(artifact["joint_xyz_camera_m"][0, 1, 0], [2, 2, 3])
    assert Path(ledger["rows"][0]["artifact"]["path"]).is_file()
    schema = json.loads((PROJECT / "contracts/handle_canonical_hand_ledger_v71.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(ledger)


def test_h2_preserves_raw_values_validity_age_and_is_independent_of_controller(tmp_path: Path) -> None:
    h0, tactile = _fixture(tmp_path, controller=False)
    h1 = sidecars.build_stage(h0, tmp_path / "h1", "h1")
    assert h1["summary"]["by_status"] == {"BLOCKED_PREREQ_CONTROLLER_WRIST": 1, "BLOCKED_SOURCE": 1}
    h2 = sidecars.build_stage(h0, tmp_path / "h2", "h2")
    assert h2["summary"]["by_status"] == {"BLOCKED_SOURCE": 1, "PASSED": 1}
    artifact = np.load(tmp_path / "h2/sidecars/playing_cards/play_cards_0910_001.npz")
    assert np.array_equal(artifact["finger_grid_raw_int16"], tactile)
    assert artifact["finger_grid_raw_int16"].dtype == np.int16
    assert artifact["side_valid"].all()
    assert artifact["finger_valid_mask"][:, 1, 1:, :, 7].sum() == 0
    assert "force" not in artifact.files and "object_identity" not in artifact.files
    schema = json.loads((PROJECT / "contracts/handle_tactile_sidecar_ledger_v71.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(h2)


def test_no_clobber(tmp_path: Path) -> None:
    h0, _ = _fixture(tmp_path)
    output = tmp_path / "h2"
    sidecars.build_stage(h0, output, "h2")
    with pytest.raises(FileExistsError):
        sidecars.build_stage(h0, output, "h2")
