#!/usr/bin/env python3
"""Build the immutable explicit S2 product binding for the 031 candidate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/bindings/play_cards_0915_031"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
S1 = REPO / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001"
V3 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
V5 = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"


def ref(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    OUT.mkdir(parents=True)
    scene_root = R2 / "lanes/lane1_scene/clean_candidate_031_wave5"
    motion_path = R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
    mount_path = V3 / "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
    depth_root = S1 / "lanes/geometry_contact/depth_full_v1/play_cards_0915_031"
    object_path = S1 / "lanes/geometry_contact/object6d_visible_031_v1/RESULT.json"
    mask_root = V5 / "lanes/scene/masks_031_v1"
    motion_audit = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/h3_motion_audit_031/attempt_0002/RESULT.json"
    with np.load(motion_path, allow_pickle=False) as archive:
        mapping = np.asarray(archive["human_to_physical"], dtype=np.int64)
        anatomical_names = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
        wrist_valid = np.asarray(archive["wrist_valid"], dtype=bool)
    if not np.array_equal(mapping, [0, 1]):
        raise ValueError("S2_031_MAPPING_DRIFT")
    side_evidence = {
        "schema_version": "HUMAN_TO_ROBOT_S2_SIDE_MAPPING_EVIDENCE_V1",
        "session_id": "play_cards_0915_031",
        "anatomical_axis_names": anatomical_names,
        "human_to_physical": mapping.tolist(),
        "valid_rows_by_anatomical_side": wrist_valid.sum(axis=0).astype(int).tolist(),
        "non_symmetric_test": "RIGHT_ONLY_VALID_ROWS_REMAIN_PHYSICAL_RIGHT;NO_ARRAY_FLIP",
        "non_symmetric_test_pass": bool(wrist_valid[:, 0].sum() == 0 and wrist_valid[:, 1].sum() == 102),
        "identity_authority": "AI_VISUAL_SIDE_ANCHOR_NOT_GOLD",
        "claim_limit": "Producer/consumer semantic compatibility only, not anatomical-side ground truth.",
    }
    side_path = OUT / "SIDE_MAPPING_EVIDENCE.json"
    write(side_path, side_evidence)
    depth_paths = sorted((depth_root / "frames").glob("*.npz"))
    if len(depth_paths) != 149:
        raise ValueError("DEPTH_FRAME_COUNT")
    depth_index_path = OUT / "DEPTH_FRAME_INDEX.json"
    write(depth_index_path, {"schema_version": "S2_DEPTH_FRAME_INDEX_V1",
                             "session_id": "play_cards_0915_031",
                             "rows": [ref(path) for path in depth_paths]})
    roles = ("human_hand", "human_forearm", "capture_device", "task_object")
    mask_rows = []
    for frame in range(149):
        mask_rows.append({role: ref(mask_root / role / f"{frame:06d}.png") for role in roles})
    mask_index_path = OUT / "MASK_FRAME_INDEX.json"
    write(mask_index_path, {"schema_version": "S2_MASK_FRAME_INDEX_V1",
                            "session_id": "play_cards_0915_031", "rows": mask_rows})
    binding = {
        "schema_version": "HUMAN_TO_ROBOT_S2_PRODUCT_BINDING_V1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "execution_round": "S2", "session_id": "play_cards_0915_031",
        "product_mode": "CANDIDATE_ONLY", "allow_rejected_scene": True,
        "candidate_reason": "SCENE_REJECTED_AND_OCCLUSION_INCONCLUSIVE",
        "image_domain": "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY",
        "scene_clean_manifest": ref(scene_root / "RESULT.json"),
        "scene_prep_manifest": ref(scene_root / "SCENE_PREP_MANIFEST.json"),
        "robot_r0": ref(motion_path), "mount_contract": ref(mount_path),
        "side_mapping_evidence": ref(side_path),
        "expected_human_to_physical": mapping.tolist(),
        "renderer_interface": "RGB_ALPHA_OPTICAL_DEPTH_VALID_COMPONENT_ID_V1",
        "compositor_interface": "VISIBLE_SURFACE_OWNERSHIP_V1_NO_RGB_FALLBACK",
        "depth_result": ref(depth_root / "RESULT.json"),
        "depth_frame_index": ref(depth_index_path),
        "object6d_result": ref(object_path),
        "mask_manifest": ref(mask_root / "MASK_MANIFEST.json"),
        "mask_frame_index": ref(mask_index_path),
        "motion_audit": ref(motion_audit),
        "occlusion_policy": "VISIBLE_SURFACE_OR_UNKNOWN",
        "adapter_collision_scope": "UNVERIFIED_VISUAL_GEOMETRY_ONLY",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    binding_path = OUT / "BINDING.json"
    write(binding_path, binding)
    write(OUT / "RESULT.json", {"schema_version": "HUMAN_TO_ROBOT_S2_BINDING_BUILD_V1",
                                 "status": "PASS", "binding": ref(binding_path)})
    print(json.dumps({"status": "PASS", "binding": str(binding_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
