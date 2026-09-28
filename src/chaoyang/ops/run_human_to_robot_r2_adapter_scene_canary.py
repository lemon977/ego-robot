#!/usr/bin/env python3
"""Load and render the received adapter mesh at a diagnostic review pose.

The pose is intentionally unrelated to a Robot flange.  This proves asset
decodability and scene-object consumption without inventing the absent
measured installation transform.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pybullet as bullet

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, TASK, ref, write_json


MESH = Path("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/decode_v1/KAI_HAND_ADAPTER_REVIEW.stl").resolve()
DECODE = Path("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/decode_v1/RESULT.json").resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    output = ATTEMPT / "lanes/lane2_motion/assembly_scene_canary_wave11"
    result_path = output / "RESULT.json"
    if result_path.is_file():
        print(result_path)
        return 0
    if output.exists():
        raise FileExistsError(f"PARTIAL_ADAPTER_CANARY:{output}")
    output.mkdir(parents=True)
    client = bullet.connect(bullet.DIRECT)
    try:
        visual = bullet.createVisualShape(
            bullet.GEOM_MESH,
            fileName=str(MESH),
            meshScale=[0.001, 0.001, 0.001],
            rgbaColor=[0.88, 0.55, 0.18, 1.0],
            physicsClientId=client,
        )
        if visual < 0:
            raise RuntimeError("ADAPTER_VISUAL_SHAPE_LOAD_FAILED")
        body = bullet.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual,
            basePosition=[0.0, 0.0, 0.0],
            physicsClientId=client,
        )
        if body < 0:
            raise RuntimeError("ADAPTER_SCENE_OBJECT_CREATE_FAILED")
        width, height = 640, 480
        view = bullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.02, 0.025], distance=0.23,
            yaw=38.0, pitch=-24.0, roll=0.0, upAxisIndex=2,
        )
        projection = bullet.computeProjectionMatrixFOV(52.0, width / height, 0.01, 2.0)
        image = bullet.getCameraImage(
            width, height, view, projection, renderer=bullet.ER_TINY_RENDERER,
            flags=bullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            shadow=0, physicsClientId=client,
        )
        rgba = np.asarray(image[2], dtype=np.uint8).reshape(height, width, 4)
        segmentation = np.asarray(image[4], dtype=np.int32).reshape(height, width)
        object_pixels = segmentation == body
        if int(object_pixels.sum()) < 100:
            raise RuntimeError("ADAPTER_SCENE_OBJECT_NOT_VISIBLE")
        bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (width, 62), (20, 20, 20), -1)
        cv2.putText(bgr, "REAL STEP-DECODED ADAPTER | REVIEW POSE ONLY", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, "NO MEASURED MOUNT / TCP / CAMERA-BASE", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 210, 255), 1, cv2.LINE_AA)
        png = output / "ADAPTER_SCENE_OBJECT_REVIEW.png"
        if not cv2.imwrite(str(png), bgr):
            raise RuntimeError("ADAPTER_REVIEW_WRITE_FAILED")
    finally:
        bullet.disconnect(client)
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_ADAPTER_SCENE_CANARY_V1",
        "task_id": TASK,
        "created_at": now(),
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "CANDIDATE_GEOMETRY_ONLY",
        "adoption": "NOT_ADOPTED",
        "adapter_cad": "PRESENT_CANDIDATE_GEOMETRY",
        "scene_object_loaded": True,
        "object_pixels": int(object_pixels.sum()),
        "mesh_scale_m_per_source_unit": 0.001,
        "review_pose_authority": "DISPLAY_ONLY_ARBITRARY_NOT_INSTALLATION",
        "measured_installation_transform": "ABSENT",
        "robot_tcp": "ABSENT",
        "camera_world_to_base": "ABSENT",
        "mesh": ref(MESH),
        "decode": ref(DECODE),
        "review": ref(png),
        "claim_limit": "Actual received STEP-derived mesh loaded as a scene object at an arbitrary review pose; this is not a flange mount, assembly validation, or deployment evidence.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
