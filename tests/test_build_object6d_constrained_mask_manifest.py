from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_object6d_constrained_mask_tested",
    PROJECT / "tools/build_object6d_constrained_mask.py",
)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": subject.sha256(path)}


def make_fixture(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    task, session, count = "chips", "unit_session", 2
    width, height = 64, 48
    session_root = tmp_path / session
    session_root.mkdir()
    video = session_root / f"CameraRecord_{session}.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    assert writer.isOpened()
    writer.write(np.full((height, width, 3), 90, np.uint8))
    writer.write(np.full((height, width, 3), 120, np.uint8))
    writer.release()

    predecessor = tmp_path / "predecessor"
    roles: dict[str, list[dict]] = {}
    for role in subject.ROLE_NAMES:
        rows = []
        role_root = predecessor / "raw_role_masks" / role
        role_root.mkdir(parents=True)
        for frame in range(count):
            path = role_root / f"{frame:05d}.png"
            assert cv2.imwrite(str(path), np.zeros((height, width), np.uint8))
            rows.append({"frame": frame, **ref(path)})
        roles[role] = rows
    predecessor_result = predecessor / "RESULT.json"
    write_json(predecessor_result, {
        "status": "HOLD_AUTOMATIC_GATE",
        "task_id": task,
        "session_id": session,
        "frame_count": count,
        "forbidden_repairs_used": [],
        "artifacts": {"raw_role_masks": str(predecessor / "raw_role_masks")},
    })
    inventory = tmp_path / "MASK_INVENTORY.json"
    write_json(inventory, {
        "schema_version": subject.INVENTORY_SCHEMA,
        "status": "COMPLETE",
        "task": task,
        "session": session,
        "frame_count": count,
        "resolution": [width, height],
        "roles": roles,
    })

    trajectory = tmp_path / "trajectory.npz"
    transforms = np.repeat(np.eye(4)[None], count, axis=0)
    transforms[:, 2, 3] = 0.5
    np.savez(
        trajectory,
        valid=np.ones(count, bool),
        T_object_to_camera=transforms,
        object_dimensions_m=np.array([0.1, 0.1, 0.1]),
        smoothed_center_2d=np.array([[30.0, 25.0], [31.0, 25.0]]),
    )
    trajectory_result = tmp_path / "OBJECT6D_RESULT.json"
    write_json(trajectory_result, {
        "schema_version": "stereo-constrained-object6d-role-trajectory-v2",
        "status": "PASS_STEREO_CONSTRAINED_ROLE_TRAJECTORY",
        "mask_canary_input_authorized": True,
        "task": task,
        "session": session,
        "frame_count": count,
        "artifacts": {"trajectory": ref(trajectory)},
    })

    hawor = tmp_path / "HAWOR_OPTIMIZED_MANO21_WORLD_CONSISTENT.npz"
    world_consistency = json.dumps({
        "schema": "hawor-camera-world-consistency-v1",
        "monocular_camera_depth_changed": False,
    })
    np.savez(
        hawor,
        intrinsics=np.repeat(np.eye(3)[None], count, axis=0),
        joints_2d=np.zeros((2, count, 21, 2)),
        joints_3d_camera=np.ones((2, count, 21, 3)),
        original_frame_indices=np.arange(count),
        anatomical_side_names=np.asarray(["left", "right"]),
        world_consistency=np.asarray(world_consistency),
    )
    hawor_result = tmp_path / "HAWOR_RESULT.json"
    write_json(hawor_result, {
        "schema": "hawor-world-consistent-derived-result-v1",
        "status": "PASS",
        "task": task,
        "session": session,
        "derived_world_vs_recomputed_mm": {"max": 0.0},
        "monocular_depth_changed": False,
        "derived": ref(hawor),
    })
    output = tmp_path / "new_output"
    manifest = {
        "schema_version": subject.INPUT_SCHEMA,
        "status": subject.INPUT_STATUS,
        "production_input": True,
        "task": task,
        "session": session,
        "frame_count": count,
        "source_resolution": [width, height],
        "object_profile": "chips_yellow",
        "predecessor_root": str(predecessor),
        "dataset_partition_contract": subject.DATASET_PARTITION_CONTRACT,
        "inputs": {
            "raw_video": ref(video),
            "predecessor_result": ref(predecessor_result),
            "predecessor_mask_inventory": ref(inventory),
            "object6d_trajectory": ref(trajectory),
            "object6d_result": ref(trajectory_result),
            "world_consistent_hawor": ref(hawor),
            "world_consistent_hawor_result": ref(hawor_result),
        },
        "output_root": str(output),
        "review_name": "UNIT_REVIEW.mp4",
        "method_contract": subject.METHOD_CONTRACT,
    }
    return manifest, {
        "output": output,
        "trajectory_result": trajectory_result,
        "inventory": inventory,
    }


def test_manifest_validates_and_binds_generic_config(tmp_path: Path) -> None:
    manifest, _ = make_fixture(tmp_path)
    config = subject.validate_input_manifest(manifest)
    assert config["task"] == "chips"
    assert config["session"] == "unit_session"
    assert config["frame_count"] == 2
    assert config["object_profile"] == "chips_yellow"


def test_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest, _ = make_fixture(tmp_path)
    manifest["inputs"]["raw_video"]["sha256"] = "0" * 64
    with pytest.raises(subject.ManifestError, match="sha256 mismatch"):
        subject.validate_input_manifest(manifest)


def test_object6d_authority_fails_closed(tmp_path: Path) -> None:
    manifest, paths = make_fixture(tmp_path)
    result = json.loads(paths["trajectory_result"].read_text())
    result["mask_canary_input_authorized"] = False
    write_json(paths["trajectory_result"], result)
    manifest["inputs"]["object6d_result"] = ref(paths["trajectory_result"])
    with pytest.raises(subject.ManifestError, match="Object6D RESULT"):
        subject.validate_input_manifest(manifest)


def test_output_no_clobber(tmp_path: Path) -> None:
    manifest, paths = make_fixture(tmp_path)
    paths["output"].mkdir()
    with pytest.raises(FileExistsError, match="no-clobber"):
        subject.validate_input_manifest(manifest)


def test_inventory_frame_mapping_fails_closed(tmp_path: Path) -> None:
    manifest, paths = make_fixture(tmp_path)
    inventory = json.loads(paths["inventory"].read_text())
    inventory["roles"]["left_human"][1]["frame"] = 7
    write_json(paths["inventory"], inventory)
    manifest["inputs"]["predecessor_mask_inventory"] = ref(paths["inventory"])
    with pytest.raises(subject.ManifestError, match="frame order"):
        subject.validate_input_manifest(manifest)


def test_tracker_evidence_names_white_grey_green_per_side_frame() -> None:
    image = np.zeros((120, 120, 3), np.uint8)
    image[60:78, 35:55] = (245, 245, 245)
    image[60:78, 55:75] = (115, 115, 115)
    image[60:78, 75:95] = (20, 150, 20)
    joints = np.zeros((21, 2), np.float64)
    joints[0] = (60, 60)
    joints[[5, 9, 13, 17]] = (60, 40)
    refined, evidence = subject.tracker_mask_from_current_frame(
        image, joints, np.zeros((120, 120), bool)
    )
    assert refined.any()
    assert evidence["colour_contract"] == ["white", "grey", "green"]
    assert set(evidence["colour_pixels"]) == {"white", "grey", "green"}
    assert all(evidence["colour_pixels"][name] > 0 for name in ("white", "grey", "green"))


def test_batch_planner_blocks_pending_upstream(tmp_path: Path) -> None:
    batch = {
        "schema_version": subject.BATCH_SCHEMA,
        "status": subject.BATCH_STATUS,
        "dataset_partition_contract": subject.DATASET_PARTITION_CONTRACT,
        "rows": [{
            "task": "chips",
            "session": "0903_inventory_pending",
            "recommended_split": "PENDING_CLASSIFICATION",
            "classification_status": "PENDING",
            "object6d_status": "PENDING",
            "input_manifest": None,
        }],
    }
    result = subject.validate_batch_manifest(batch)
    assert result["status"] == "CPU_READY_BATCH_BLOCKED_UPSTREAM"
    assert result["start_authorized"] is False
    assert result["rows"][0]["blockers"] == [
        "CLASSIFICATION_INVENTORY_PENDING", "OBJECT6D_PENDING"
    ]


def test_0903_partition_is_exact_and_trash_is_excluded(tmp_path: Path) -> None:
    manifest, _ = make_fixture(tmp_path)
    partition = manifest["dataset_partition_contract"]
    assert partition["chips"] == {"start_index": 1, "end_index": 183, "count": 183}
    assert partition["poker"] == {"start_index": 184, "end_index": 249, "count": 66}
    assert partition["excluded_entry_names"] == [".pico_editor_trash"]
    invalid = copy.deepcopy(manifest)
    invalid["dataset_partition_contract"]["chips"]["end_index"] = 182
    with pytest.raises(subject.ManifestError, match="dataset_partition_contract"):
        subject.validate_input_manifest(invalid)


def test_json_schema_accepts_fixture(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    manifest, _ = make_fixture(tmp_path)
    schema = json.loads((subject.PROJECT / "contracts/object6d_constrained_mask_input_manifest_v2.schema.json").read_text())
    jsonschema.validate(manifest, schema)
