"""Pure selection rules for the scoped Exact78 visual baseline route.

This module does not grant execution authority. The existing task packet and
writer gate still decides whether a selected operation may run.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any


BASELINE_ID = "EXACT78_VISUAL_BASELINE_V1"
SCOPE = "chips_cards_tracker_0901_0902"
TAKEOVER_OPERATIONS = frozenset(
    {
        "run_exact78_takeover_audit_v1",
        "run_exact78_takeover_baseline_v1",
        "run_exact78_takeover_cleanup_v1",
    }
)
OLD_SUBMIT_RECIPE = "exact78-raw-to-robot"

PICK_PLACE_ROUTE = "pick_place_1008_candidate_v1"
PICK_PLACE_OPERATIONS = frozenset({
    "run_exact78_pick_place_audit_candidate_v1",
    "run_exact78_pick_place_raw_to_robot_candidate_v1",
})


def _pick_place_candidate_valid(contract: Mapping[str, Any]) -> bool:
    special = contract.get("special_status_contracts")
    if not isinstance(special, Mapping):
        return False
    route = special.get(PICK_PLACE_ROUTE)
    if not isinstance(route, Mapping):
        return False
    operations = route.get("candidate_operations")
    closure = route.get("code_closure")
    return (
        route.get("execution_status") == "CANDIDATE_FENCED_PICK_PLACE_TASK_PACKET_ONLY"
        and route.get("scope") == "pick_place_1008"
        and route.get("plan_revision") == "PICK_PLACE_1008_RAW_TO_ROBOT_V1"
        and route.get("task_packet_required") is True
        and isinstance(operations, list) and len(operations) == 2
        and all(isinstance(name, str) for name in operations)
        and set(operations) == PICK_PLACE_OPERATIONS
        and route.get("training_steps_authorized") == 0
        and route.get("current_runnable_operation") is None
        and isinstance(closure, list) and bool(closure)
        and all(isinstance(ref, Mapping) and isinstance(ref.get("path"), str)
                and isinstance(ref.get("sha256"), str) and len(ref["sha256"]) == 64
                and all(char in "0123456789abcdefABCDEF" for char in ref["sha256"])
                and type(ref.get("bytes")) is int and ref["bytes"] >= 0
                for ref in closure)
    )




def _is_scoped_exact78_operation(name: str) -> bool:
    lowered = name.lower()
    if any(other in lowered for other in ("0916", "robot100", "manus")):
        return False
    return any(marker in lowered for marker in ("exact78", "cards042", "0902"))


def _policy(contract: Mapping[str, Any]) -> object:
    special = contract.get("special_status_contracts")
    if isinstance(special, Mapping) and "exact78_visual_baseline_v1" in special:
        branch = special["exact78_visual_baseline_v1"]
    else:
        # Compatibility for isolated synthetic callers; the published contract
        # stores this branch under special_status_contracts.
        branch = contract.get("exact78_visual_baseline_v1")
    return branch.get("execution_policy") if isinstance(branch, Mapping) else None


def _selected_policy_valid(policy: Mapping[str, Any]) -> bool:
    allowed = policy.get("allowed_operations")
    return (
        policy.get("enabled") is True
        and policy.get("baseline_id") == BASELINE_ID
        and policy.get("scope") == SCOPE
        and isinstance(allowed, (list, tuple, set, frozenset))
        and len(allowed) == len(TAKEOVER_OPERATIONS)
        and all(isinstance(name, str) for name in allowed)
        and set(allowed) == TAKEOVER_OPERATIONS
    )


def filter_selected_operations(operations: Set[str], contract: Mapping[str, Any]) -> set[str]:
    """Retain only the three takeover operations once this scoped route is on.

    A missing policy or one explicitly disabled preserves the legacy set.
    A present but malformed enabled policy closes only the Exact78 namespace.
    """

    admitted_pick_place = set(operations) & PICK_PLACE_OPERATIONS if _pick_place_candidate_valid(contract) else set()
    selected = set(operations) - PICK_PLACE_OPERATIONS
    policy = _policy(contract)
    if policy is None or (isinstance(policy, Mapping) and policy.get("enabled") is False):
        return selected | admitted_pick_place
    other = {name for name in selected if not _is_scoped_exact78_operation(name)}
    if not isinstance(policy, Mapping) or not _selected_policy_valid(policy):
        return other | admitted_pick_place
    return other | (selected & TAKEOVER_OPERATIONS) | admitted_pick_place


def assert_selected_submit_allowed(contract: Mapping[str, Any], recipe: str) -> None:
    """Keep the legacy submit recipe from masquerading as the selected V1 route."""

    if recipe != OLD_SUBMIT_RECIPE:
        return
    policy = _policy(contract)
    if policy is None or (isinstance(policy, Mapping) and policy.get("enabled") is False):
        return
    raise ValueError(
        "exact78-raw-to-robot is not the selected visual route; use "
        "EXACT78_VISUAL_BASELINE_V1 via its scoped takeover task packet. "
        "A future submit recipe requires separate authorization."
    )


def build_selected_visual_contract(repo_root: Any) -> dict[str, Any]:
    """Publish scoped selection only after the separate adoption receipt exists.

    Legacy contracts remain evidence records. This independent visual release
    does not replace the 156-session ledger or any MANUS/0916 stage counts.
    """
    from chaoyang.governance.common import artifact_ref, load_json

    pointer_path = repo_root / "docs/current/EXACT78_VISUAL_BASELINE_V1.json"
    receipt_path = repo_root / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json"
    if not pointer_path.is_file() or not receipt_path.is_file():
        return {"status": "NOT_SELECTED", "execution_policy": {"enabled": False}}
    pointer = load_json(pointer_path)
    receipt = load_json(receipt_path)
    policy = pointer.get("execution_policy")
    if (not isinstance(policy, Mapping) or not _selected_policy_valid(policy)
            or receipt.get("baseline_id") != BASELINE_ID
            or receipt.get("status") != "PASSED"
            or receipt.get("baseline_pointer") != artifact_ref(pointer_path)):
        raise RuntimeError("EXACT78_VISUAL_V1_ADOPTION_OR_SELECTION_DRIFT")
    return {
        "status": "USER_ADOPTED_BATCH_VISUAL_BASELINE_WITH_KNOWN_LIMITS",
        "baseline_id": BASELINE_ID,
        "baseline_pointer": artifact_ref(pointer_path),
        "publication_receipt": artifact_ref(receipt_path),
        "code_closure": [
            artifact_ref(repo_root / "src/chaoyang/governance/exact78_visual_selection_v1.py"),
            artifact_ref(repo_root / "src/chaoyang/cli/__init__.py"),
            artifact_ref(repo_root / "src/chaoyang/governance/exact78_operation_gate.py"),
        ],
        "current_visual_operation": pointer["selected_operation"],
        "execution_policy": dict(policy),
        "actual_motion_source": pointer["actual_motion_source"],
        "validated_scope": pointer["validated_scope"],
        "historical_exact78_routes": "RETIRED_FROM_DEFAULT_SELECTION; EVIDENCE_ONLY",
        "old_submit_recipe": "RETIRED_NOT_SELECTED",
        "algorithm_changed": False,
        "legacy_156_ledger_counts_changed": False,
        "pure_rgb_hawor_qualified": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "claim_limit": pointer["claim_limit"],
    }


__all__ = ["filter_selected_operations", "assert_selected_submit_allowed", "build_selected_visual_contract"]
