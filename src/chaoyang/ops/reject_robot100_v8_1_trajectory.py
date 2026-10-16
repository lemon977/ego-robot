"""Withdraw V8.1 as the current 0916 trajectory baseline without deleting it."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json


ATTEMPT = REPO_ROOT / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001"
V8 = ATTEMPT / "corrections/anatomical_side_identity_rear_view_v8_1"
V9 = ATTEMPT / "corrections/head_relative_shared_axes_v9"
CURRENT = ATTEMPT / "CURRENT_0916_ROBOT100_BASELINE.json"
SNAPSHOT = V8 / "SELECTION_BEFORE_TRAJECTORY_REJECTION.json"
REJECTION = V8 / "REJECTION_TRAJECTORY.json"


def main() -> int:
    prior = load_json(CURRENT)
    if prior.get("baseline_id") != "0916_ROBOT100_ANATOMICAL_IDENTITY_REAR_VIEW_V8_1":
        raise RuntimeError("CURRENT_POINTER_IS_NOT_V8_1")
    if prior.get("baseline_selected") is not True:
        raise RuntimeError("V8_1_ALREADY_NOT_SELECTED")

    generated = datetime.now(timezone.utc).isoformat()
    atomic_json(SNAPSHOT, prior)
    rejection = {
        "schema_version": "CHAOYANG_0916_ROBOT100_TRAJECTORY_REJECTION_V1",
        "status": "REJECTED_BY_HUMAN_TRAJECTORY_REVIEW",
        "scope": "0916 Robot100 trajectory only; left/right identity correction remains valid evidence",
        "rejected_baseline_id": prior["baseline_id"],
        "rejected_pointer_snapshot": artifact_ref(SNAPSHOT),
        "human_review": {
            "status": "REJECTED",
            "statement": "The motion trajectory does not match the source human motion.",
        },
        "root_cause": [
            "V8.1 inherited inv(first_world_wrist) @ current_world_wrist, expressing translation through each wrist's different local axes.",
            "The tracked head_pose was not removed, so wearer motion contaminated arm targets.",
            "V8.1 additionally used branch interpolation and post-hoc posture projection, which can change the displayed trajectory after IK.",
        ],
        "play_cards_0916_025_evidence": {
            "head_relative_right_first_last_m": [-0.19380793, 0.07716853, 0.41904202],
            "v8_1_target_right_first_last_m": [-0.255590272, 0.0203888, -0.27646506],
            "head_relative_right_path_m": 1.05686,
            "v8_1_target_right_path_m": 0.81937,
            "head_motion_path_m": 0.15934,
            "conclusion": "The vertical direction is inverted and the path is not a shared head-frame mapping.",
        },
        "successor_requirement": {
            "candidate": "HEAD_RELATIVE_SHARED_AXES_V9",
            "formula": "T_head_wrist = inv(T_world_head) @ T_world_wrist",
            "selection_rule": "A single-session canary must pass human trajectory review before any 100-session expansion.",
            "candidate_root": str(V9.resolve()),
        },
        "data_deleted": False,
        "generated_at": generated,
    }
    atomic_json(REJECTION, rejection)

    pointer = {
        "schema_version": "CHAOYANG_CURRENT_0916_ROBOT100_BASELINE_V1",
        "status": "NO_CURRENT_TRAJECTORY_BASELINE_V8_1_REJECTED",
        "baseline_selected": False,
        "algorithm_quality_adopted": False,
        "scope": prior["scope"],
        "rejected_baseline_id": prior["baseline_id"],
        "preserved_result_root": prior["result_root"],
        "rejection": artifact_ref(REJECTION),
        "successor_candidate": {
            "baseline_id": "0916_ROBOT100_HEAD_RELATIVE_SHARED_AXES_V9",
            "status": "NOT_YET_SELECTED",
            "result_root": str(V9.resolve()),
        },
        "human_review": {
            "status": "REJECTED_TRAJECTORY",
            "evidence": "User rejected the reviewed V8.1 sample because its motion trajectory did not match the human motion.",
        },
        "generated_at": generated,
    }
    atomic_json(CURRENT, pointer)
    print(json.dumps(pointer, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
