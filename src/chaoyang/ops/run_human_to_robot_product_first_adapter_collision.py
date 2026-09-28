"""Finite connector collision queries; no whole-machine collision PASS claim."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pybullet as bullet
import trimesh

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK
from chaoyang.pipeline.v5_product import ProductRobotRenderer

OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/adapter_collision/window_v1"
R2 = ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
MOTION = R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
PREP = R2 / "lanes/lane1_scene/clean_candidate_031_wave5/SCENE_PREP_MANIFEST.json"
MOUNT = ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"


def _save(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _closest(client: int, a: int, b: int, distance: float) -> dict:
    points = bullet.getClosestPoints(a, b, distance, physicsClientId=client)
    return {"shape_pair_body_ids": [a, b], "count": len(points),
            "minimum_signed_distance_m": min((float(point[8]) for point in points), default=None),
            "reported_link_pairs": sorted({(int(point[3]), int(point[4])) for point in points})}


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    mount = load_json(MOUNT)
    mesh_ref = mount["decoded_review_mesh"]
    mesh = Path(mesh_ref["path"])
    if mesh_ref["mesh_scale_to_metre"] != .001 or mesh_ref["unit"] != "millimetre":
        raise ValueError("MESH_UNIT_DRIFT")
    if hashlib.sha256(mesh.read_bytes()).hexdigest() != mesh_ref["sha256"]:
        raise ValueError("MESH_SHA_DRIFT")
    geometry = trimesh.load(str(mesh), process=True)
    if not isinstance(geometry, trimesh.Trimesh):
        raise ValueError("MESH_TOPOLOGY_UNSUPPORTED")
    with np.load(MOTION, allow_pickle=False) as archive:
        motion = {name: np.asarray(archive[name]) for name in archive.files}
    prep = load_json(PREP)
    domain = load_json(Path(prep["source_domain"]["path"]))
    OUT.mkdir(parents=True)
    renderer = ProductRobotRenderer(ROOT, motion, domain, include_adapter=True)
    client = renderer.client
    try:
        shape = bullet.createCollisionShape(
            bullet.GEOM_MESH, fileName=str(mesh), meshScale=[.001, .001, .001],
            flags=bullet.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=client)
        if shape < 0:
            raise RuntimeError("REAL_MESH_COLLISION_SHAPE_LOAD_FAILED")
        adapter_query = [bullet.createMultiBody(baseMass=0., baseCollisionShapeIndex=shape,
                                                 physicsClientId=client) for _ in range(2)]
        sphere_shape = bullet.createCollisionShape(bullet.GEOM_SPHERE, radius=.002, physicsClientId=client)
        sphere = bullet.createMultiBody(baseMass=0., baseCollisionShapeIndex=sphere_shape,
                                        physicsClientId=client)
        identity = [0., 0., 0., 1.]
        # These positions are frozen fixture coordinates on this exact STL,
        # not fitted or selected from the 031 video.
        bullet.resetBasePositionAndOrientation(sphere, [.5, 0., 0.], identity, physicsClientId=client)
        separation = _closest(client, adapter_query[0], sphere, .005)
        vertex = (np.asarray(geometry.vertices[0], np.float64) * .001).tolist()
        bullet.resetBasePositionAndOrientation(sphere, vertex, identity, physicsClientId=client)
        intersection = _closest(client, adapter_query[0], sphere, .005)
        hole_axis_xy = [-.0077, .0268]
        hole_path = []
        for z in (.012, .02, .03, .04, .05, .06):
            bullet.resetBasePositionAndOrientation(sphere, [*hole_axis_xy, z], identity,
                                                   physicsClientId=client)
            hole_path.append({"z_m": z, **_closest(client, adapter_query[0], sphere, .005)})
        bullet.resetBasePositionAndOrientation(sphere, [hole_axis_xy[0] + .02, hole_axis_xy[1], .03],
                                               identity, physicsClientId=client)
        surrounding_ring = _closest(client, adapter_query[0], sphere, .005)
        if separation["count"] != 0 or intersection["minimum_signed_distance_m"] is None or intersection["minimum_signed_distance_m"] >= 0:
            raise RuntimeError("CONTROLLED_COLLISION_BACKEND_NOT_DISCRIMINATIVE")
        if any(row["minimum_signed_distance_m"] is not None and row["minimum_signed_distance_m"] <= 0 for row in hole_path):
            raise RuntimeError("HOLE_FIXTURE_COLLIDES")
        if surrounding_ring["minimum_signed_distance_m"] is None or surrounding_ring["minimum_signed_distance_m"] >= 0:
            raise RuntimeError("RING_SURROUND_FIXTURE_NOT_COLLIDING")
        real_rows = []
        for frame in range(66, 82):
            layer = renderer.frame_layers(frame)
            row = {"source_frame_id": frame, "per_side": []}
            for side in range(2):
                if not np.isfinite(layer.T_world_adapter[side]).all():
                    row["per_side"].append({"side": side, "status": "INVALID_SOURCE_NO_QUERY"})
                    continue
                renderer.place_hand(client, adapter_query[side], layer.T_world_adapter[side])
                arm = _closest(client, adapter_query[side], renderer.robot, .005)
                hand = _closest(client, adapter_query[side], renderer.hands[side], .005)
                row["per_side"].append({"side": side, "status": "QUERIED_NO_CONTACT_WHITELIST",
                                        "adapter_vs_robot": arm, "adapter_vs_same_side_hand": hand})
            real_rows.append(row)
    finally:
        renderer.close()
    result = {"schema_version": "HUMAN_TO_ROBOT_ADAPTER_COLLISION_QUERY_V1", "task_id": TASK,
              "session_id": "play_cards_0915_031", "source_frames": list(range(66, 82)),
              "mesh": artifact_ref(mesh), "mount": artifact_ref(MOUNT), "motion": artifact_ref(MOTION),
              "mesh_topology": {"watertight": bool(geometry.is_watertight),
                                "winding_consistent": bool(geometry.is_winding_consistent),
                                "vertices": len(geometry.vertices), "faces": len(geometry.faces),
                                "bbox_m": (geometry.bounds * .001).tolist()},
              "backend": "PYBULLET_FIXED_CONCAVE_TRIMESH_VS_SPHERE_OR_PINNED_ROBOT_URDF",
              "fixtures": {"separation": separation, "intersection": intersection,
                           "hole_axis_xy_m": hole_axis_xy, "hole_path": hole_path,
                           "surrounding_ring": surrounding_ring},
              "real_window": real_rows,
              "quality": "FINITE_QUERY_EXECUTED_FULL_ADAPTER_COLLISION_PASS_NOT_AUTHORIZED",
              "reason": "SOURCE_STL_NONWATERTIGHT_AND_MOUNTING_CONTACT_WHITELIST_NOT_APPROVED",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False,
              "claim_limit": "Actual finite collision backend calls and hole-sensitive fixture only; no continuous collision, legal mounting contact, full machine or physical safety claim."}
    _save(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["quality"], "topology": result["mesh_topology"],
                      "real_frame_count": len(real_rows), "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
