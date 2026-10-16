"""Additive selection/navigation for the user-adopted original199 pipeline.

Old producer receipts and imported algorithms retain their original bytes.
This module grants no task execution, quality, training or robot authority.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

BASELINE_ID = "INPAINTING_RAW_TO_ROBOT_V1"
OPERATION = "run_exact78_pick_place_raw_to_robot_candidate_v1"
MANIFEST = "inpainting_v1/MANIFEST.json"
RECEIPT = "tasks/receipts/INPAINTING_RAW_TO_ROBOT_V1.json"
HISTORICAL_DOCUMENT_IDS = frozenset({
    "baseline_v1_source_release", "baseline_v1_source_handoff",
    "dual_branch_phase_baselines_v1", "dual_branch_phase_handoff_v1",
    "exact78_visual_baseline_v1", "exact78_visual_baseline_v1_handoff",
    "project_branch_map", "project_branch_map_handoff",
    "exact78_runtime_hardening_handoff", "exact78_runtime_hardening_contract",
    "v5_visual_index",
})


def selected(root: Path) -> dict[str, Any] | None:
    from chaoyang.governance.common import artifact_ref, load_json

    root = Path(root)
    receipt_path, path = root / RECEIPT, root / MANIFEST
    if not receipt_path.exists():
        return None  # A draft manifest is not adoption authority.
    if not path.is_file():
        raise RuntimeError("INPAINTING_V1_MANIFEST_ABSENT")
    receipt, value = load_json(receipt_path), load_json(path)
    if (receipt.get("status") != "PASSED"
            or receipt.get("baseline_id") != BASELINE_ID
            or receipt.get("baseline_pointer") != artifact_ref(path)
            or value.get("baseline_id") != BASELINE_ID
            or value.get("selected_operation") != OPERATION
            or value.get("selected_baseline_count") != 1
            or value.get("actual_motion_source") != "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS"
            or value.get("scope", {}).get("completed_sessions") != 199
            or value.get("scope", {}).get("native_frames") != 29322
            or value.get("quality", {}).get("control_ground_truth") is not False
            or value.get("quality", {}).get("all199_visual_accepted") is not False
            or any(value.get("adoption_sideeffects", {}).get(key) != 0
                   for key in ("algorithm_calls", "training_steps", "files_deleted", "external_writes"))):
        raise RuntimeError("INPAINTING_V1_ADOPTION_DRIFT")
    return value


def contract(root: Path) -> dict[str, Any]:
    from chaoyang.governance.common import artifact_ref, load_json

    value = selected(root)
    if value is None:
        return {}
    # The immutable algorithm manifest is never rewritten for a navigation
    # compatibility repair. Only this new selection wrapper may be rebound by
    # an explicit immutable publication supplement; no algorithm/asset override.
    runtime = list(value["runtime_closure"])
    dependencies = list(value["dependencies"])
    supplement_path = root / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json"
    supplement_ref = None
    if (root / "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json").exists():
        from chaoyang.inpainting_candidate.publication import runtime_publication

        runtime, dependencies, supplement_ref = runtime_publication(root, value)
    elif supplement_path.exists():
        supplement = load_json(supplement_path)
        allowed = artifact_ref(root / "src/chaoyang/governance/inpainting_v1_selection.py")
        original = next((ref for ref in runtime if ref["path"] == allowed["path"]), None)
        if (supplement.get("baseline_id") != BASELINE_ID
                or supplement.get("baseline_pointer") != artifact_ref(root / MANIFEST)
                or supplement.get("original_wrapper") != original
                or supplement.get("current_wrapper") != allowed
                or supplement.get("algorithm_changed") is not False):
            raise RuntimeError("INPAINTING_V1_PUBLICATION_SUPPLEMENT_DRIFT")
        runtime = [allowed if ref == original else ref for ref in runtime]
        supplement_ref = artifact_ref(supplement_path)
    return {"inpainting_raw_to_robot_v1": {
        "baseline_id": BASELINE_ID,
        "status": "USER_ADOPTED_BATCH_VISUAL_BASELINE_WITH_KNOWN_LIMITS",
        "selection_scope": "UNIQUE_DEFAULT_FOR_THIS_INPAINTING_BRANCH",
        "selected_baseline_count": 1,
        "baseline_pointer": artifact_ref(root / MANIFEST),
        "publication_receipt": artifact_ref(root / RECEIPT),
        "selected_operation": OPERATION,
        "operation_name_is_compatibility_identifier": True,
        "task_packet_required": True,
        "current_runnable_operation": None,
        "code_closure": value["algorithm_core"],
        "runtime_closure": runtime,
        "governance_publication_supplement": supplement_ref,
        "dependencies": dependencies,
        "inputs": value["inputs"],
        "actual_motion_source": value["actual_motion_source"],
        "validated_scope": value["scope"],
        "quality": value["quality"],
        "history_index": artifact_ref(root / "archive/inpainting_baselines/INDEX.json"),
        "retired_entrypoints": artifact_ref(root / "docs/governance/RETIRED_OPERATION_ENTRYPOINTS.json"),
        "algorithm_changed": False,
        "pure_rgb_hawor_qualified": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "training_steps_authorized_without_new_packet": 0,
        "claim_limit": value["claim_limit"],
    }}


def annotate_legacy_contracts(special: dict[str, Any], root: Path) -> None:
    """Do not mutate old pointers or the compatibility admission contract."""
    if selected(root) is None:
        return
    for key in ("baseline_v1_source_release", "dual_branch_phase_baselines_v1", "exact78_visual_baseline_v1"):
        if key in special:
            special[key] = dict(special[key], current_selection=False,
                selection_status="HISTORICAL_DEPENDENCY_NOT_DEFAULT_INPAINTING",
                superseded_by=BASELINE_ID)
    if "exact78_visual_baseline_v1" in special:
        special["exact78_visual_baseline_v1"]["current_visual_operation"] = None
    # Candidate admission is retained byte-semantically for the existing CLI
    # fence. Adoption is a separate axis, not a bypass around Task Packets.
    if "pick_place_1008_candidate_v1" in special:
        special["pick_place_1008_candidate_v1"] = dict(special["pick_place_1008_candidate_v1"],
            selected_baseline=BASELINE_ID, operation_name_is_compatibility_identifier=True)


def overlay_registry(registry: dict[str, Any], layout: dict[str, Any], root: Path) -> None:
    from chaoyang.governance.common import artifact_ref

    value = selected(root)
    if value is None:
        return
    registry["selected_visual_baseline"] = {
        "baseline_id": BASELINE_ID, "selected_baseline_count": 1,
        "pointer": artifact_ref(root / MANIFEST), "scope": value["scope"],
        "legacy_entries_role": "STAGE_QUALIFICATION_LEDGER_NOT_DEFAULT_PIPELINE_SELECTION",
        "other_manus_branch": "OUT_OF_SCOPE_UNCHANGED",
    }
    layout.setdefault("canonical_documents", {})["inpainting_raw_to_robot_v1"] = str(root / MANIFEST)
    if "inpainting_v1" not in layout.get("retained_top_level", []):
        layout.setdefault("retained_top_level", []).append("inpainting_v1")
    layout["inpainting_default_entry"] = str(root / "inpainting_v1/README.md")
    layout["historical_visual_navigation"] = str(root / "archive/inpainting_baselines/README.md")


def publish_status(root: Path, state: dict[str, Any], generated_at: str) -> None:
    from chaoyang.governance.common import artifact_ref, atomic_json

    value = selected(root)
    if value is None:
        return
    live = [row["task_id"] for row in state.get("tasks", [])
            if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    atomic_json(root / "docs/current/STATUS.json", {
        "schema_version": "INPAINTING_V1_PROJECT_LIVE_STATUS",
        "generated_at": generated_at, "governance_revision": state.get("governance_revision"),
        "status": "RUNNING_FINITE_TASKS" if live else "QUIESCENT_NO_ACTIVE_TASKS",
        "active_tasks": live, "next_task": state.get("next_task"),
        "current_index_status": "ACTIVE_FINITE_TASKS" if live else "PASS_NO_ACTIVE_TASKS",
        "baseline_id": BASELINE_ID, "baseline_pointer": artifact_ref(root / MANIFEST),
        "counts": {"selected_inpainting_baselines": 1,
                   "original_completed_sessions": 199, "original_native_frames": 29322},
        "branches": {
            "inpainting_visual": {"baseline_id": BASELINE_ID, "pointer": artifact_ref(root / MANIFEST),
                "actual_motion_source": value["actual_motion_source"], "scope": value["scope"],
                "selected_operation": OPERATION, "quality": value["quality"]},
            "manus_0916": {"ownership": "OTHER_BRANCH_READ_ONLY", "selection_scope": "OUT_OF_SCOPE_UNCHANGED",
                "pointer": artifact_ref(root / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json")},
        },
        "training_entry_unchanged": "docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/COLLEAGUE_COMMANDS_ZH.md",
        "history_index": artifact_ref(root / "archive/inpainting_baselines/INDEX.json"),
        "training_eligible": False, "control_ground_truth": False,
        "claim_limit": value["claim_limit"],
    })
