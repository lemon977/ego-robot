#!/usr/bin/env python3
"""Build immutable S2 bindings for the two frozen 0902 regressions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/bindings"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
V3 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
V5 = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"

SESSIONS = (
    {
        "session_id": "get_potato_chips_0902_103",
        "short": "103",
        "frames": 284,
        "clean": "clean_candidate_103_wave4",
        "motion": "recovered_0902_103_wave2",
        "mask": "masks_103_v1",
    },
    {
        "session_id": "play_cards_0902_042",
        "short": "042",
        "frames": 171,
        "clean": "clean_candidate_042_wave4",
        "motion": "recovered_0902_042_wave2",
        "mask": "masks_042_v1",
    },
)


def ref(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(row: dict[str, object]) -> dict[str, object]:
    sid = str(row["session_id"])
    count = int(row["frames"])
    out = ROOT / sid / "attempt_0001"
    if out.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{out}")
    out.mkdir(parents=True)
    scene_root = R2 / "lanes/lane1_scene" / str(row["clean"])
    motion_path = R2 / "lanes/lane2_motion" / str(row["motion"]) / "ROBOT_R0_V1.npz"
    mask_root = V5 / "lanes/scene" / str(row["mask"])
    mount_path = V3 / "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
    domain_path = V3 / "lanes/exact78/exact_domain_v1" / sid / "DOMAIN_MANIFEST.json"
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    with np.load(motion_path, allow_pickle=False) as archive:
        mapping = np.asarray(archive["human_to_physical"], dtype=np.int64)
        names = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
        valid = np.asarray(archive["wrist_valid"], dtype=bool)
        q0 = np.asarray(archive["q_arm"][0], dtype=np.float64)
    if not np.array_equal(mapping, [0, 1]):
        raise ValueError(f"S2_0902_MAPPING_DRIFT:{sid}:{mapping.tolist()}")
    roles = tuple(role for role in ("human_hand", "human_forearm", "capture_device", "task_object")
                  if (mask_root / role).is_dir())
    if "task_object" not in roles:
        raise ValueError(f"TASK_OBJECT_MASK_MISSING:{sid}")
    mask_rows = [{role: ref(mask_root / role / f"{frame:06d}.png") for role in roles}
                 for frame in range(count)]
    mask_index = out / "MASK_FRAME_INDEX.json"
    write(mask_index, {"schema_version": "S2_MASK_FRAME_INDEX_V1", "session_id": sid,
                       "rows": mask_rows})
    side_path = out / "SIDE_MAPPING_EVIDENCE.json"
    write(side_path, {
        "schema_version": "HUMAN_TO_ROBOT_S2_SIDE_MAPPING_EVIDENCE_V1",
        "session_id": sid, "anatomical_axis_names": names,
        "human_to_physical": mapping.tolist(),
        "valid_rows_by_anatomical_side": valid.sum(axis=0).astype(int).tolist(),
        "non_symmetric_test": "DISTINCT_LEFT_RIGHT_Q_REMAIN_ON_MATCHING_RENDERER_SIDE_INDICES",
        "non_symmetric_test_pass": bool(np.isfinite(q0).all() and not np.allclose(q0[0], q0[1])),
        "identity_authority": "FROZEN_PRODUCER_SEMANTICS_NOT_EXTERNAL_GOLD",
        "claim_limit": "Producer/consumer semantic compatibility only, not anatomical-side ground truth.",
    })
    binding = {
        "schema_version": "HUMAN_TO_ROBOT_S2_PRODUCT_BINDING_V1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1", "execution_round": "S2",
        "session_id": sid, "product_mode": "CANDIDATE_ONLY",
        "allow_rejected_scene": True,
        "candidate_reason": "0902_SAME_RECIPE_REGRESSION_SCENE_REJECTED_NO_QUALIFIED_DEPTH",
        "image_domain": domain["image_domain"],
        "scene_clean_manifest": ref(scene_root / "RESULT.json"),
        "scene_prep_manifest": ref(scene_root / "SCENE_PREP_MANIFEST.json"),
        "robot_r0": ref(motion_path), "mount_contract": ref(mount_path),
        "side_mapping_evidence": ref(side_path),
        "expected_human_to_physical": mapping.tolist(),
        "renderer_interface": "RGB_ALPHA_OPTICAL_DEPTH_VALID_COMPONENT_ID_V1",
        "compositor_interface": "VISIBLE_SURFACE_OWNERSHIP_V1_NO_RGB_FALLBACK",
        "geometry_mode": "UNKNOWN_NO_QUALIFIED_DEPTH",
        "mask_roles": list(roles), "mask_manifest": ref(mask_root / "MASK_MANIFEST.json"),
        "mask_frame_index": ref(mask_index),
        "occlusion_policy": "UNKNOWN_NO_QUALIFIED_DEPTH",
        "adapter_collision_scope": "UNVERIFIED_VISUAL_GEOMETRY_ONLY",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    binding_path = out / "BINDING.json"
    write(binding_path, binding)
    result = {"session_id": sid, "status": "PASS", "binding": ref(binding_path)}
    write(out / "RESULT.json", {"schema_version": "HUMAN_TO_ROBOT_S2_BINDING_BUILD_V1", **result})
    return result


def main() -> int:
    results = [build(row) for row in SESSIONS]
    print(json.dumps({"status": "PASS", "bindings": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
