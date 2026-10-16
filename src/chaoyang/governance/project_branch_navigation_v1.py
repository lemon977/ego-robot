"""Evidence-bound navigation; does not authorize algorithms or promote quality."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from chaoyang.governance.common import artifact_ref, load_json, validate_artifact_ref

LIVE_STATUSES = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
HISTORICAL_EXACT78_FILES = (
    "EXACT78_EXECUTION_ROUTE_V1.json", "EXACT78_ACTIVE_LINEAGE_V1.json",
    "EXACT78_CARDS042_VISUAL_BASELINE_V2.json", "EXACT78_CARDS042_VISUAL_BRANCH_LOCK_V2.json",
    "EXACT78_CARDS042_VISUAL_BASELINE_ZH.md", "EXACT78_CARDS042_VISUAL_BRANCH_LOCK_ZH.md",
)


def live_tasks(state: Mapping[str, Any]) -> list[str]:
    return [str(row["task_id"]) for row in state.get("tasks", []) if row.get("status") in LIVE_STATUSES]


def verified(ref: dict) -> dict:
    if errors := validate_artifact_ref(ref):
        raise RuntimeError("BRANCH_REFERENCE_DRIFT:" + ";".join(errors))
    return ref


def visual_selection_present(root: Path) -> bool:
    pointer_path = root / "docs/current/EXACT78_VISUAL_BASELINE_V1.json"
    receipt_path = root / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json"
    if not pointer_path.exists() or not receipt_path.exists():
        return False
    pointer, receipt = load_json(pointer_path), load_json(receipt_path)
    if (pointer.get("baseline_id") != "EXACT78_VISUAL_BASELINE_V1"
            or receipt.get("status") != "PASSED" or receipt.get("baseline_pointer") != artifact_ref(pointer_path)):
        raise RuntimeError("VISUAL_SELECTION_RECEIPT_DRIFT")
    return True


def build_branch_map(root: Path) -> dict:
    if not visual_selection_present(root):
        raise RuntimeError("CURRENT_VISUAL_SELECTION_ABSENT")
    current = root / "docs/current"
    visual = load_json(current / "EXACT78_VISUAL_BASELINE_V1.json")
    hardware = load_json(current / "ROBOT100_HARDWARE_BASELINE_V16.json")
    if hardware.get("baseline_id") != "0916_HARDWARE_WORLD_WRIST_V16" or not hardware.get("baseline_selected"):
        raise RuntimeError("MANUS_SELECTION_MISMATCH")
    branches = {
        "exact78_visual": {
            "baseline_id": visual["baseline_id"], "role": "ADOPTED_BATCH_OFFLINE_VISUAL",
            "display_name": "Exact78 PICO-assisted Visual V1",
            "actual_motion_source": visual["actual_motion_source"],
            "pointer": artifact_ref(current / "EXACT78_VISUAL_BASELINE_V1.json"),
            "adoption_receipt": artifact_ref(root / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json"),
            "result": verified(visual["v13_result"]), "scope": visual["validated_scope"],
            "operation": visual["selected_operation"], "packet_required": True,
            "pure_rgb": False, "training_eligible": False, "control_ground_truth": False,
            "full_batch_human_review": "NOT_CLAIMED", "metric_occlusion_qualified": False,
        },
        "manus_0916": {
            "baseline_id": hardware["baseline_id"], "role": hardware["selection_scope"],
            "display_name": "0916 Controller plus MANUS Visual Development V16",
            "actual_motion_source": "SAME_SESSION_CONTROLLER_HEAD_MANUS_INSTALLATION_PRIOR",
            "pointer": artifact_ref(current / "ROBOT100_HARDWARE_BASELINE_V16.json"),
            "baseline": verified(hardware["baseline"]),
            "operation": "run_robot100_v16_safe_replay", "packet_required": True,
            "scope": {"validated_session": "play_cards_0916_025", "validated_frames": 176,
                      "batch_100_validated": hardware["batch_100_validated"]},
            "image_alignment_status": hardware["image_alignment_status"],
            "physical_accuracy_verified": hardware["physical_accuracy_verified"],
            "training_eligible": False, "control_ground_truth": False,
            "ownership": "OTHER_BRANCH_READ_ONLY",
        },
    }
    return {
        "schema_version": "PROJECT_BRANCH_NAVIGATION_V1", "selected_visual_route_count": len(branches),
        "branches": branches,
        "exact78_input_adapter_experiment": {
            "sessions": ["play_cards_0915_001", "get_potato_chips_0918_001"],
            "executed_sessions": 2, "frames": 439, "new_baseline": False,
            "source_by_session": {"play_cards_0915_001": "PICO_OPENXR26_SAME_SESSION",
                                  "get_potato_chips_0918_001": "CONTROLLER_MANUS_INSTALLATION_PRIOR"},
            "result": artifact_ref(root / "_run/current/exact78_takeover_v1_two_source_review_v3_20261008/attempts/attempt_0001/RESULT.json"),
            "ai_sample_quality": "NOT_PASSED", "user_full_film_review": "PENDING",
        },
        "pure_rgb_hawor": {"role": "UNPUBLISHED_MIGRATION_TARGET", "current_runnable_operation": None,
                           "old_submit_retired": True, "legacy_branch_name_is_not_motion_provenance": True},
        "work_lanes_not_baselines": ["Scene/Clean", "Sensor", "Motion/Robot", "HuRo comparison"],
        "downstream_h50_models_not_work_lanes": ["Chips Raw", "Chips Robotized", "Poker Raw", "Poker Robotized"],
        "historical_exact78_references": [
            {"file": artifact_ref(current / name), "execution_allowed": False,
             "status": "HISTORICAL_IMMUTABLE_REFERENCE", "embedded_current_flags_authoritative": False}
            for name in HISTORICAL_EXACT78_FILES],
        "history_four_stream": {
            "status": "CLOSED_NOT_GLOBAL_ADOPTION_COUNTS",
            "result": artifact_ref(root / "_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/RESULT.json"),
        },
        "claim_limit": "Navigation only. Adoption, execution, source scope, visual quality, metric geometry and trainability are separate. No new qualification.",
    }


def build_live_status(root: Path, state: Mapping[str, Any], generated_at: str) -> dict:
    map_path = root / "docs/current/PROJECT_BRANCH_MAP.json"
    board = load_json(map_path)
    # A static snapshot must still match current adopted inputs on every refresh.
    if board != build_branch_map(root):
        raise RuntimeError("BRANCH_MAP_SOURCE_DRIFT_REQUIRES_NEW_PUBLICATION")
    active = live_tasks(state)
    return {
        "schema_version": "PROJECT_BRANCH_LIVE_STATUS_V1", "generated_at": generated_at,
        "governance_revision": state.get("governance_revision"),
        "status": "RUNNING_FINITE_TASKS" if active else "QUIESCENT_NO_ACTIVE_TASKS",
        "current_index_status": "ACTIVE_FINITE_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
        "active_tasks": active, "next_task": state.get("next_task"),
        "branch_map": artifact_ref(map_path), "branches": board["branches"],
        "counts": {"selected_visual_routes": board["selected_visual_route_count"]},
        "historical_four_stream": board["history_four_stream"],
        "training_eligible": False, "control_ground_truth": False,
        "claim_limit": "Live tasks come from the ledger; PENDING is not running. Selected visual routes are not qualified training or physical-control products.",
    }


def build_source_labels(root: Path) -> dict:
    """Bind unchanged review media to the recorded, not inferred, motion source."""
    from types import SimpleNamespace
    from chaoyang.exact78_baseline.review_labels import motion_review_title

    attempt = root / "_run/current/exact78_takeover_v1_two_source_review_v3_20261008/attempts/attempt_0001"
    expected = {
        "play_cards_0915_001": "PICO_OPENXR26_SAME_SESSION",
        "get_potato_chips_0918_001": "PICO_CONTROLLER_PLUS_MANUS_INSTALLATION_PRIOR",
    }
    rows = []
    for session_id, source in expected.items():
        path = attempt / "candidate/sessions" / session_id / "SESSION.json"
        session = load_json(path)
        calibration = session["motion"]["calibration"]
        if session["session"] != session_id or calibration.get("actual_motion_source") != source:
            raise RuntimeError("REVIEW_SOURCE_MISMATCH:" + session_id)
        rows.append({
            "session": session_id, "session_result": artifact_ref(path),
            "frame_count": session["frame_count"], "actual_motion_source": source,
            "motion": verified(session["outputs"]["motion"]),
            "media": {key: verified(session["outputs"][key]) for key in ("review", "final", "clean")},
            "legacy_review_title": "RAW + PICO",
            "future_review_title_example": motion_review_title(
                SimpleNamespace(calibration=calibration), 0, session["frame_count"]),
            "existing_media_rewritten": False,
        })
    return {
        "schema_version": "IMMUTABLE_REVIEW_SOURCE_LABELS_V1", "sessions": rows,
        "claim_limit": "Source clarification and future review text only. Existing media and motion are unchanged; no quality qualification.",
    }


def snapshot_paths(root: Path) -> list[Path]:
    """Base-Git overlay plus explicit package/runtime closure, not media/weights."""
    paths = set()
    for task in ("exact78_takeover_b2_full_v13", "exact78_takeover_v1_two_source_review_v3_20261008"):
        packet = load_json(root / f"tasks/current/{task}/TASK_PACKET.json")
        for row in packet["runtime_code"]:
            p = Path(row["path"])
            if p.is_file() and root.resolve() in p.resolve().parents:
                paths.add(p)
    for folder in ("src/chaoyang/exact78_baseline", "src/chaoyang/cli", "configs", "contracts", "manifests"):
        for p in (root / folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".toml", ".json", ".yaml", ".yml"}:
                if root.resolve() not in p.resolve().parents or p.stat().st_size > 10_000_000:
                    raise RuntimeError("SNAPSHOT_PATH_OR_SIZE_INVALID:" + str(p))
                paths.add(p)
    names = (
        "pyproject.toml", "AGENTS.md", "THIRD_PARTY_NOTICES.md", "src/chaoyang/__init__.py", "src/chaoyang/paths.py",
        "src/chaoyang/governance/project_branch_navigation_v1.py",
        "src/chaoyang/governance/publish_project_branch_convergence_v1.py",
        "src/chaoyang/governance/build_human_to_robot_r2_terminal_status.py",
        "src/chaoyang/governance/current_r3_contracts.py", "src/chaoyang/governance/exact78_visual_selection_v1.py",
        "src/chaoyang/governance/exact78_operation_gate.py",
        "docs/current/PROJECT_BRANCH_MAP.json", "docs/current/PROJECT_BRANCH_MAP_ZH.md",
        "docs/current/EXACT78_VISUAL_BASELINE_V1.json", "docs/current/EXACT78_VISUAL_BASELINE_V1_ZH.md",
        "docs/current/EXACT78_EXECUTION_ROUTE_ZH.md", "docs/current/AI_WORK_ENTRY_ZH.md",
        "docs/current/GIT_DELIVERY_ZH.md", "docs/current/README_ZH.md",
        "README.md", "docs/README.md", "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md",
        "docs/current/PLAN.md", "docs/current/V5_MOTION.md", "docs/current/V5_HURO.md",
        "review/two_sessions_20261008/README.md", "review/two_sessions_20261008/SOURCE_LABELS.json",
        "docs/plans/PROJECT_BRANCH_CONVERGENCE_20261008/00_EXECUTION_ZH.md",
        "tests/unit/test_project_branch_navigation_v1.py", "tests/unit/test_exact78_visual_selection_v1.py",
        "tests/unit/test_exact78_operation_gate.py", "tests/unit/test_exact78_native_input.py",
    )
    paths.update(root / n for n in names)
    from chaoyang.governance.source_snapshot_closure_v1 import local_import_closure
    paths.update(local_import_closure(root, [path for path in paths if path.suffix == ".py" and root / "src" in path.parents]))
    paths.update(root / name for name in (
        "src/chaoyang/ops/run_exact78_takeover_baseline_v1.py",
        "src/chaoyang/ops/run_exact78_takeover_audit_v1.py",
        "src/chaoyang/ops/run_exact78_takeover_cleanup_v1.py",
        "src/chaoyang/governance/operation_retirement.py",
        "src/chaoyang/governance/source_snapshot_closure_v1.py",
        "src/chaoyang/governance/publish_exact78_runtime_hardening_v1.py",
        "src/chaoyang/governance/publish_exact78_runtime_snapshot_repair_v1.py",
        "docs/governance/RETIRED_OPERATION_ENTRYPOINTS.json",
        "docs/current/EXACT78_RUNTIME_HARDENING_V1.json",
        "docs/current/EXACT78_RUNTIME_HARDENING_ZH.md",
        "tests/unit/test_exact78_runtime_hardening_v1.py",
        "docs/plans/EXACT78_RUNTIME_HARDENING_20261008/00_EXECUTION_ZH.md",
        "docs/plans/EXACT78_RUNTIME_SNAPSHOT_REPAIR_20261008/00_EXECUTION_ZH.md",
    ))
    # Close imports from the newly added executable sources as well.
    paths.update(local_import_closure(root, [path for path in paths if path.suffix == ".py" and root / "src" in path.parents]))
    return sorted(paths)
