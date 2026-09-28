#!/usr/bin/env python3
"""Build the immutable explicit S2 product binding for the 007 candidate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/bindings/get_potato_chips_0915_007/attempt_0002"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
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
    scene_root = R2 / "lanes/lane1_scene/clean_candidate_007_wave4"
    motion_path = R2 / "lanes/lane2_motion/recovered_007_wave0/ROBOT_R0_V1.npz"
    mount_path = V3 / "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
    mask_root = V5 / "lanes/scene/masks_007_v1"
    with np.load(motion_path, allow_pickle=False) as archive:
        mapping = np.asarray(archive["human_to_physical"], dtype=np.int64)
        anatomical_names = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
        wrist_valid = np.asarray(archive["wrist_valid"], dtype=bool)
        q_arm0 = np.asarray(archive["q_arm"][0], dtype=np.float64)
    if not np.array_equal(mapping, [0, 1]):
        raise ValueError("S2_007_MAPPING_DRIFT")
    asymmetric = bool(np.isfinite(q_arm0).all() and not np.allclose(q_arm0[0], q_arm0[1]))
    side_path = OUT / "SIDE_MAPPING_EVIDENCE.json"
    write(side_path, {
        "schema_version": "HUMAN_TO_ROBOT_S2_SIDE_MAPPING_EVIDENCE_V1",
        "session_id": "get_potato_chips_0915_007",
        "anatomical_axis_names": anatomical_names, "human_to_physical": mapping.tolist(),
        "valid_rows_by_anatomical_side": wrist_valid.sum(axis=0).astype(int).tolist(),
        "non_symmetric_test": "DISTINCT_LEFT_RIGHT_Q_REMAIN_ON_MATCHING_RENDERER_SIDE_INDICES",
        "non_symmetric_test_pass": asymmetric,
        "identity_authority": "FROZEN_PRODUCER_SEMANTICS_NOT_EXTERNAL_GOLD",
        "claim_limit": "Producer/consumer semantic compatibility only, not anatomical-side ground truth.",
    })
    roles = ("human_forearm", "capture_device", "task_object")
    mask_rows = [{role: ref(mask_root / role / f"{frame:06d}.png") for role in roles}
                 for frame in range(378)]
    mask_index_path = OUT / "MASK_FRAME_INDEX.json"
    write(mask_index_path, {"schema_version": "S2_MASK_FRAME_INDEX_V1",
                            "session_id": "get_potato_chips_0915_007", "rows": mask_rows})
    binding = {
        "schema_version": "HUMAN_TO_ROBOT_S2_PRODUCT_BINDING_V1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1", "execution_round": "S2",
        "session_id": "get_potato_chips_0915_007", "product_mode": "CANDIDATE_ONLY",
        "allow_rejected_scene": True,
        "candidate_reason": "SCENE_REJECTED_AND_NO_QUALIFIED_DEPTH",
        "image_domain": "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY",
        "scene_clean_manifest": ref(scene_root / "RESULT.json"),
        "scene_prep_manifest": ref(scene_root / "SCENE_PREP_MANIFEST.json"),
        "robot_r0": ref(motion_path), "mount_contract": ref(mount_path),
        "side_mapping_evidence": ref(side_path),
        "expected_human_to_physical": mapping.tolist(),
        "renderer_interface": "RGB_ALPHA_OPTICAL_DEPTH_VALID_COMPONENT_ID_V1",
        "compositor_interface": "VISIBLE_SURFACE_OWNERSHIP_V1_NO_RGB_FALLBACK",
        "geometry_mode": "UNKNOWN_NO_QUALIFIED_DEPTH",
        "mask_roles": list(roles),
        "mask_manifest": ref(mask_root / "MASK_MANIFEST.json"),
        "mask_frame_index": ref(mask_index_path),
        "occlusion_policy": "UNKNOWN_NO_QUALIFIED_DEPTH",
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
