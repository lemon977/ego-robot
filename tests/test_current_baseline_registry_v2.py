from __future__ import annotations

import json
from pathlib import Path
import time


ROOT = Path("/mnt/workspace/code/chaoyang")


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_registry_has_twelve_unique_current_stages() -> None:
    registry = _load("docs/governance/CURRENT_BASELINE_REGISTRY_V2.json")
    names = [entry["stage"] for entry in registry["entries"]]
    assert len(names) == 12
    assert len(set(names)) == 12
    assert names == [
        "Raw", "HaWoR", "Role Mask", "Object Mask", "Depth", "Object6D",
        "Clean", "Contact", "Robot Visual", "Occlusion", "HumanEgo Aux", "HumanEgo Policy",
    ]


def test_every_material_reference_has_bytes_and_sha() -> None:
    registry = _load("docs/governance/CURRENT_BASELINE_REGISTRY_V2.json")
    for entry in registry["entries"]:
        for field in ("code_closure", "current_evidence", "related_current_releases"):
            for reference in entry.get(field, []):
                assert Path(reference["path"]).is_file(), reference
                assert reference["bytes"] > 0, reference
                assert len(reference["sha256"]) == 64, reference
        weights = entry["weights"]
        if isinstance(weights, list):
            for reference in weights:
                assert Path(reference["path"]).is_file(), reference
                assert reference["bytes"] > 0, reference
                assert len(reference["sha256"]) == 64, reference


def test_robot_v52_closure_and_claim_boundary_are_explicit() -> None:
    registry = _load("docs/governance/CURRENT_BASELINE_REGISTRY_V2.json")
    robot = next(entry for entry in registry["entries"] if entry["stage"] == "Robot Visual")
    names = {Path(reference["path"]).name for reference in robot["code_closure"]}
    assert {
        "run_hawor_temporal_jerk_successor.py",
        "run_robot_motion_transfer_arm_canary_v3.py",
        "select_robot_fixed_placement_v1.py",
        "run_robot_arm_segment_bidirectional_v3.py",
        "run_robot_hand_fullsession_v2.py",
        "run_newtask_robot_shared_v4_hand.py",
        "run_robot_hand_segment_bidirectional_v3.py",
        "render_robot_motion_transfer_fullsession_v2.py",
        "adopt_exact78_pose_only_visual_robot_v52.py",
    } <= names
    assert robot["authorized_scope"].startswith("NO_CURRENT_TASK_ROBOT_AUTHORITY")
    assert any("control_ground_truth=false" in value for value in robot["known_limitations"])


def test_goldset_is_not_silently_promoted() -> None:
    result = _load(
        "tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/"
        "goldset_v1_review_pack/RESULT.json"
    )
    assert result["status"] == "BLOCKED_EXTERNAL_PENDING_TWO_INDEPENDENT_HUMAN_LABELS_AND_ROBOT_CANDIDATE"
    assert result["goldset_passed"] is False
    registry = _load("docs/governance/CURRENT_BASELINE_REGISTRY_V2.json")
    for stage in ("Contact", "Occlusion"):
        entry = next(item for item in registry["entries"] if item["stage"] == stage)
        assert entry["current_counts"]["passed"] == 0


def test_receipt_binds_registry_and_stage_page() -> None:
    # The active Clean guardian may atomically publish a new bundle between
    # two reads.  Accept only a receipt-stable read, exactly like production
    # readers, instead of treating a legitimate concurrent revision as drift.
    for _ in range(20):
        receipt = _load("docs/governance/CURRENT_STATUS_RECEIPT.json")
        registry = _load("docs/governance/CURRENT_BASELINE_REGISTRY_V2.json")
        receipt_after = _load("docs/governance/CURRENT_STATUS_RECEIPT.json")
        if receipt["generation_id"] == receipt_after["generation_id"]:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("could not obtain a receipt-stable governance read")
    assert receipt["governance_revision"] == registry["governance_revision"]
    assert receipt["files"]["baseline_registry_v2"]["path"].endswith("CURRENT_BASELINE_REGISTRY_V2.json")
    assert receipt["files"]["stage_baselines"]["path"].endswith("CURRENT_STAGE_BASELINES_ZH.md")
