#!/usr/bin/env python3
"""Trace the exact semantic origin of 031's frozen wrist-pose target."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
OUT = REPO / (
    f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/"
    "target_authority_031/attempt_0001"
)
V3 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"


def ref(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"FRESH_OUTPUT_REQUIRED:{OUT}")
    producer = (
        REPO
        / "_run/current/worktrees/full-pipeline-v3-integration/src/chaoyang/pipeline"
        / "camera_robot_motion_v3.py"
    )
    upstream = V3 / "shared/robot/camera031_epoch7_c3/CAMERA_ROBOT_MOTION_V3.npz"
    current = R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
    source = V3 / "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
    for path in (producer, upstream, current, source):
        if not path.is_file():
            raise FileNotFoundError(path)
    code = producer.read_text(encoding="utf-8")
    tokens = (
        "source_basis = palm_basis(source)",
        "robot_basis = palm_basis(neutral)",
        "rotation = source_basis @ robot_basis.T",
        "root[:3, :3] = rotation",
        "root[:3, 3] = source[0] - rotation @ neutral[0]",
        "fixed_camera_placement(target['T_target_root_cam'], valid, neutral_roots)",
    )
    missing = [token for token in tokens if token not in code]
    if missing:
        raise RuntimeError(f"TARGET_PRODUCER_LOGIC_NOT_FOUND:{missing}")
    with np.load(upstream, allow_pickle=False) as archive:
        upstream_target = np.asarray(archive["T_target_root_cam"], dtype=np.float64)
        upstream_valid = np.asarray(archive["wrist_valid"], dtype=bool)
        upstream_mapping = np.asarray(archive["human_to_physical"], dtype=np.int64)
        placement_policy = str(np.asarray(archive["placement_policy"]).item())
    with np.load(current, allow_pickle=False) as archive:
        current_target = np.asarray(archive["T_target_root_cam"], dtype=np.float64)
        current_valid = np.asarray(archive["wrist_valid"], dtype=bool)
        current_mapping = np.asarray(archive["human_to_physical"], dtype=np.int64)
    with np.load(source, allow_pickle=False) as archive:
        predicted = np.asarray(archive["predicted_valid"], dtype=bool)
        observed = np.asarray(archive["observed"], dtype=bool)
        inferred = np.asarray(archive["inferred"], dtype=bool)
    if not np.array_equal(upstream_valid, current_valid):
        raise RuntimeError("TARGET_VALIDITY_CHANGED_IN_R2_HANDOFF")
    if not np.array_equal(upstream_mapping, current_mapping):
        raise RuntimeError("TARGET_SIDE_MAPPING_CHANGED_IN_R2_HANDOFF")
    valid = current_valid
    if not np.array_equal(upstream_target[valid], current_target[valid], equal_nan=True):
        raise RuntimeError("TARGET_POSE_CHANGED_IN_R2_HANDOFF")
    if not np.array_equal(predicted.T, upstream_valid):
        raise RuntimeError("TARGET_VALIDITY_NOT_EQUAL_TO_HAWOR_PREDICTION")

    payload = {
        "schema_version": "HUMAN_TO_ROBOT_S2_031_TARGET_AUTHORITY_V1",
        "task_id": TASK,
        "session_id": "play_cards_0915_031",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "INCONCLUSIVE_TARGET_AUTHORITY",
        "adoption": "NOT_ADOPTED",
        "target_definition": {
            "position": "HaWoR MANO joint0 adjusted by robot-neutral root offset",
            "rotation": (
                "palm basis derived from HaWoR predicted 21-joint geometry, aligned to "
                "the neutral KaiHand palm basis"
            ),
            "rotation_is_direct_anatomical_wrist_measurement": False,
            "rotation_is_external_ground_truth": False,
            "camera_base_placement": placement_policy,
            "placement_authority": "VIRTUAL_FIXED_CAMERA_BASE_NOT_MEASURED_WORLD_OR_ROBOT_CALIBRATION",
        },
        "handoff_checks": {
            "target_pose_byte_semantics_unchanged": True,
            "validity_unchanged": True,
            "side_mapping_unchanged": True,
            "human_to_physical": current_mapping.astype(int).tolist(),
            "valid_left_right": current_valid.sum(axis=0).astype(int).tolist(),
            "source_predicted_left_right": predicted.sum(axis=1).astype(int).tolist(),
            "source_observed_left_right": observed.sum(axis=1).astype(int).tolist(),
            "source_inferred_left_right": inferred.sum(axis=1).astype(int).tolist(),
        },
        "interpretation": (
            "The optimizer consumes the frozen target exactly. The target rotation is a model-derived "
            "palm-frame construction with no direct wrist-orientation authority; therefore a joint-limit "
            "conflict cannot establish physical unreachability or validate the target orientation."
        ),
        "inputs": {
            "target_producer_source": ref(producer),
            "upstream_robot_motion": ref(upstream),
            "r2_robot_r0": ref(current),
            "hawor_source": ref(source),
        },
        "claim_limit": (
            "Target-lineage audit only; no external pose accuracy, reachability certification, "
            "control, training, Contact, or deployment authority."
        ),
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    OUT.mkdir(parents=True)
    path = OUT / "RESULT.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "result": ref(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
