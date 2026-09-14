import json
from pathlib import Path

import cv2
import numpy as np

from tools.run_pico_geometry_raw_point_mask_canary import (
    assign_humans,
    finite_json,
    human_side_swap,
    load_object_instance_authority,
    load_precomputed_setup_world_authority,
    sha256,
)


def test_finite_json_recurses_and_serializes_strictly():
    payload = {
        "plain": 4,
        "path": Path("evidence/result.json"),
        "metrics": [float("inf"), float("-inf"), float("nan"), np.float32(1.25)],
        "array": np.asarray([2.0, np.inf]),
    }
    sanitized = finite_json(payload)
    encoded = json.dumps(sanitized, allow_nan=False)
    restored = json.loads(encoded)
    assert restored == {
        "plain": 4,
        "path": "evidence/result.json",
        "metrics": [None, None, None, 1.25],
        "array": [2.0, None],
    }


def test_human_assignment_is_global_not_left_first_greedy():
    masks = np.zeros((2, 20, 20), bool)
    # Instance 0 is compatible with both palms; instance 1 only with left.
    masks[0, 10, 6] = True
    masks[0, 10, 14] = True
    masks[1, 10, 7] = True
    pico = {"left": {"palm": np.asarray([5.0, 10.0])}, "right": {"palm": np.asarray([15.0, 10.0])}}
    assigned, evidence = assign_humans(masks, np.asarray([101, 102]), pico, (20, 20), 5.0)
    assert evidence["left"]["chosen_raw_id"] == 102
    assert evidence["right"]["chosen_raw_id"] == 101
    assert assigned["left"][10, 7]
    assert assigned["right"][10, 14]


def test_human_side_swap_is_measured_from_pico_palms():
    humans = {"left": np.zeros((20, 20), bool), "right": np.zeros((20, 20), bool)}
    humans["left"][10, 15] = True
    humans["right"][10, 5] = True
    pico = {"left": {"palm": np.asarray([5.0, 10.0])}, "right": {"palm": np.asarray([15.0, 10.0])}}
    swapped, evidence = human_side_swap(humans, pico, 5.0)
    assert swapped is True
    assert evidence["swapped_palm_distance_sum_px"] == 0.0


def test_precomputed_setup_world_authority_is_identity_and_hash_bound(tmp_path):
    shared_path = tmp_path / "shared.json"
    task_path = tmp_path / "task.json"
    plan_path = tmp_path / "plan.json"
    provider_path = tmp_path / "provider.json"
    shared_path.write_text("{}")
    task_path.write_text("{}")
    plan_path.write_text("{}")
    provider_path.write_text("{}")
    task = {"setup_geometry_normalized_xy": {"fixture": [0.5, 0.5]}}
    plan = {"task_id": "chips", "session_id": "fixture_001", "split": "FIT"}
    report = {
        "schema_version": "pico-setup-world-anchor-authority-v2",
        "status": "PASS_PICO_STATIC_AND_CONTACT_SCOPE",
        **plan,
        "world_xyz_by_role": {"fixture": [0.0, 0.0, 1.0]},
        "rgb_frames_read": 0,
        "gpu_calls": 0,
        "manual_per_frame_points_used": False,
        "derivation": {"authority": "PICO", "contact_bindings": {}},
        "canary_projection_audit": {"all_roles_in_frame": True, "rows": []},
        "canary_moving_object_authority_preflight": {
            "status": "PASS_CONTACT_BOUNDED_SEEDS_AVAILABLE"
        },
        "pins": {
            "shared_config": {"sha256": sha256(shared_path)},
            "task_config": {"sha256": sha256(task_path)},
            "evaluation_plan": {"sha256": sha256(plan_path)},
            "anchor_provider_authority": {"sha256": sha256(provider_path)},
        },
    }
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps(report))
    anchors, evidence = load_precomputed_setup_world_authority(
        authority,
        shared_path=shared_path,
        task_path=task_path,
        plan_path=plan_path,
        provider_authority_path=provider_path,
        task=task,
        plan=plan,
    )
    assert anchors == {"fixture": [0.0, 0.0, 1.0]}
    assert evidence["mode"] == "PRECOMPUTED_PICO_ONLY_AUTHORITY"
    shared_path.write_text("{\"drift\": true}")
    try:
        load_precomputed_setup_world_authority(
            authority,
            shared_path=shared_path,
            task_path=task_path,
            plan_path=plan_path,
            provider_authority_path=provider_path,
            task=task,
            plan=plan,
        )
    except RuntimeError as error:
        assert "pin drift" in str(error)
    else:
        raise AssertionError("drifted shared config pin was accepted")


def test_object_instance_authority_requires_full_hash_bound_isolated_timelines(tmp_path):
    paths = {name: tmp_path / f"{name}.json" for name in ("shared", "task", "plan", "provider", "setup")}
    for path in paths.values():
        path.write_text("{}")
    task = {
        "role_groups": {
            "task_object": {"setup_roles": ["object_0"]},
            "fixture": {"setup_roles": ["fixture"]},
        }
    }
    plan = {"task_id": "chips", "session_id": "s1", "split": "FIT", "frame_count": 2}
    instances = {}
    for index, (role, group) in enumerate((("object_0", "task_object"), ("fixture", "fixture"))):
        artifacts = []
        for frame in range(2):
            mask_path = tmp_path / f"{role}_{frame}.png"
            mask = np.zeros((8, 8), np.uint8)
            mask[2:4, 2 + index:4 + index] = 255
            assert cv2.imwrite(str(mask_path), mask)
            artifacts.append({"frame_index": frame, "path": str(mask_path), "sha256": sha256(mask_path)})
        instances[role] = {
            "status": "PASS",
            "group": group,
            "independent_model_state": True,
            "other_object_ids_in_state": [],
            "cross_instance_masks_modified": False,
            "raw_mask_artifacts": artifacts,
        }
    report = {
        "schema_version": "contact-assisted-independent-object-instance-authority-v1",
        "status": "PASS_OBJECT_INSTANCE_AUTHORITY",
        "consumption_authorized": True,
        **plan,
        "cross_instance_identity_gate": {"pass": True},
        "task_instance_roles": ["object_0"],
        "fixture_instance_roles": ["fixture"],
        "instance_order": ["object_0", "fixture"],
        "instances": instances,
        "validation": {"pins": {
            "shared_config": {"sha256": sha256(paths["shared"])},
            "task_config": {"sha256": sha256(paths["task"])},
            "evaluation_plan": {"sha256": sha256(paths["plan"])},
            "anchor_provider_authority": {"sha256": sha256(paths["provider"])},
            "setup_world_authority": {"sha256": sha256(paths["setup"])},
        }},
    }
    authority_path = tmp_path / "object_authority.json"
    authority_path.write_text(json.dumps(report))
    loaded = load_object_instance_authority(
        authority_path,
        shared_path=paths["shared"],
        task_path=paths["task"],
        plan_path=paths["plan"],
        provider_authority_path=paths["provider"],
        setup_world_authority_path=paths["setup"],
        task=task,
        plan=plan,
    )
    assert loaded["authority_file"]["sha256"] == sha256(authority_path)
    Path(instances["object_0"]["raw_mask_artifacts"][0]["path"]).write_bytes(b"tampered")
    try:
        load_object_instance_authority(
            authority_path,
            shared_path=paths["shared"],
            task_path=paths["task"],
            plan_path=paths["plan"],
            provider_authority_path=paths["provider"],
            setup_world_authority_path=paths["setup"],
            task=task,
            plan=plan,
        )
    except RuntimeError as error:
        assert "artifact pin drift" in str(error)
    else:
        raise AssertionError("tampered object mask was accepted")
