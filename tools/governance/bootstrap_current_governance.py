from __future__ import annotations

import argparse
from pathlib import Path

from tools.governance.common import artifact_ref, now_iso, publish_bundle


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "tasks" / "control" / "runs"


def ref(relative: str) -> dict:
    return artifact_ref(ROOT / relative)


def build_authority() -> dict:
    raw = ref("tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json")
    hawor = ref("tasks/control/runs/20260909_exact78_current_baseline_batch_v1/HAWOR_BATCH_RESULT.json")
    role = ref("tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/mask_clean/mask_role_successor_v3_final/RESULT.json")
    object_mask = ref("tasks/control/runs/20260909_exact78_current_baseline_batch_v1/MASK_TASK_OBJECT_BATCH_RESULT.json")
    join = ref("tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/mask_clean/mask_role_successor_v3_final/MASK_JOIN_READY_STATE_V3.json")
    depth26 = ref("tasks/control/runs/20260911_exact78_depth_object6d_successor_v3_delta32/PREDECESSOR_26_TERMINALS.json")
    depth32 = ref("tasks/control/runs/20260911_exact78_depth_object6d_successor_v3_delta32/TERMINAL_AUDIT.json")
    baseline = ref("tasks/control/runs/20260908_two_task_e2e_baseline_v1/BASELINE_AUTHORITY.json")
    training = ref("tasks/control/runs/20260908_two_task_e2e_baseline_v1/training_exact78_visual_ab_prepare_v1/FOUR_CHECKPOINT_CURRENT_READINESS.json")
    depth_report = ref("docs/reports/depth_accuracy/20260911/documents/DEPTH_ACCURACY_CURRENT_STATUS_ZH.md")
    wave0 = ref("tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json")
    return {
        "schema_version": "chaoyang-current-authority-index-v1",
        "governance_revision": 1,
        "generation_id": "bootstrap",
        "generated_at": now_iso(),
        "generator_code_sha": "0" * 64,
        "repository": {"commit": "UNKNOWN", "branch": "UNKNOWN"},
        "data_root": "/mnt/data/egodata/datasets/ego",
        "cohort": "exact78-chips78-poker78-frozen",
        "waves": {
            "raw_total": 156,
            "wave0_metric_ready": 58,
            "wave0_clean_passed": 4,
            "wave0_clean_pending": 54,
            "wave1_delta": 0,
            "wave2_delta": 0,
            "raw_calibration_missing": 59,
            "join_calibration_missing": 43
        },
        "stages": [
            {"stage": "Raw", "total": 156, "passed": 156, "grade_c": 0, "running": 0, "blocked": 0, "authority_scope": "FROZEN_EXACT78_COHORT", "denominator_semantics": "78 Chips + 78 Poker frozen raw sessions; calibrated routing is a separate denominator.", "evidence": [raw]},
            {"stage": "HaWoR", "total": 156, "passed": 144, "grade_c": 12, "running": 0, "blocked": 0, "authority_scope": "BOUNDED_V2_TERMINALS", "denominator_semantics": "A/B terminals are downstream eligible; C remains terminal.", "evidence": [hawor]},
            {"stage": "Role Mask", "total": 156, "passed": 124, "grade_c": 32, "running": 0, "blocked": 0, "authority_scope": "SAM31_ROLE_SUCCESSOR_V3", "denominator_semantics": "Four roles: human left/right and tracker left/right.", "evidence": [role]},
            {"stage": "Object Mask", "total": 156, "passed": 121, "grade_c": 35, "running": 0, "blocked": 0, "authority_scope": "TASK_OBJECT_IDENTITY", "denominator_semantics": "Chips three independent physical instances; Poker action-conditioned same card.", "evidence": [object_mask]},
            {"stage": "Depth", "total": 58, "passed": 58, "grade_c": 0, "running": 0, "blocked": 0, "authority_scope": "VISUAL_OBJECT6D_CANDIDATE_INPUT", "denominator_semantics": "Only three-lane A/B sessions with verified metric calibration.", "evidence": [join, depth26, depth32]},
            {"stage": "Object6D", "total": 58, "passed": 58, "grade_c": 0, "running": 0, "blocked": 0, "authority_scope": "OBSERVED_ONLY_KEEP_INVALID", "denominator_semantics": "Visible-surface geometric estimates, not contact or external pose truth.", "evidence": [depth26, depth32]},
            {"stage": "Clean", "total": 58, "passed": 4, "grade_c": 0, "running": 0, "blocked": 54, "authority_scope": "POKER4_EXPANDED_ROLE_V3_ONLY", "denominator_semantics": "Wave0 metric-ready denominator; four existing Poker B, 54 pending.", "evidence": [baseline, wave0]},
            {"stage": "Contact", "total": 2, "passed": 0, "grade_c": 0, "running": 0, "blocked": 2, "authority_scope": "NO_CURRENT_CONTACT_AUTHORITY", "denominator_semantics": "Chips034/Poker042 canary scope only until explicit promotion.", "evidence": [baseline]},
            {"stage": "Robot Visual", "total": 156, "passed": 0, "grade_c": 0, "running": 0, "blocked": 156, "authority_scope": "NO_CURRENT_TASK_ROBOT_AUTHORITY", "denominator_semantics": "Development videos and withdrawn previews are not authority.", "evidence": [baseline]},
            {"stage": "HumanEgo Aux", "total": 4, "passed": 0, "grade_c": 0, "running": 0, "blocked": 4, "authority_scope": "NOT_STARTED", "denominator_semantics": "Four Raw/Robotized visual auxiliary checkpoints.", "evidence": [training]},
            {"stage": "HumanEgo Policy", "total": 4, "passed": 0, "grade_c": 0, "running": 0, "blocked": 4, "authority_scope": "BLOCKED_REAL_ROBOT_ACTION", "denominator_semantics": "Final policy checkpoints require synchronized real Robot actions.", "evidence": [training]}
        ],
        "claims": [
            {"claim": "FoundationStereo disparity-to-depth formula and registration chain are internally closed for the current visual Depth products.", "status": "SUPPORTED_INTERNAL_CONSISTENCY", "scope": "exact78 metric-ready Depth", "evidence": [depth26, depth32, depth_report], "claim_limit": "Not an external millimetre-accuracy measurement."},
            {"claim": "Chips034 right hand shows a persistent negative HaWoR-versus-Stereo Z discrepancy.", "status": "SUPPORTED_INTERNAL_CONSISTENCY", "scope": "get_potato_chips_0902_034/right", "evidence": [depth_report], "claim_limit": "Cross-system surface difference; it does not establish which system is physically correct."},
            {"claim": "The Chips034 discrepancy is primarily a HaWoR absolute-Z placement error.", "status": "HYPOTHESIS_ONLY", "scope": "get_potato_chips_0902_034/right", "evidence": [depth_report], "claim_limit": "Median-Z correction is diagnostic and no external hand-depth truth exists."},
            {"claim": "Current Object6D poses are physical ground truth.", "status": "UNSUPPORTED", "scope": "exact78", "evidence": [depth26, depth32], "claim_limit": "Observed visible-surface geometry only; occluded frames stay invalid."},
            {"claim": "Current visual Robot results are deployable or training-authorized.", "status": "UNSUPPORTED", "scope": "exact78", "evidence": [baseline, training], "claim_limit": "Central Robot authority and action sidecars are both absent."}
        ],
        "decisions": [
            "Wave0 is frozen at 58 metric-ready sessions; repairs publish immutable deltas.",
            "Clean pixels never feed Depth, Object6D, contact geometry, or action truth.",
            "Formal Object6D remains DIRECT_OBSERVED_ONLY and KEEP_INVALID.",
            "FoundationStereo native confidence is absent and must not be fabricated.",
            "Human-video 3D is auxiliary visual evidence, not real Robot action ground truth.",
            "Visual Robot authority and physical deployment authority are separate."
        ]
    }


def task(task_id: str, phase: str, status: str) -> dict:
    return {"task_id": task_id, "phase": phase, "attempt": 0, "status": status, "updated_at": now_iso(), "heartbeat_at": None, "session": None, "pid": None, "proc_start_ticks": None, "gpu_id": None}


def build_task_state() -> dict:
    return {
        "schema_version": "chaoyang-long-horizon-task-state-v1",
        "governance_revision": 1,
        "generation_id": "bootstrap",
        "generated_at": now_iso(),
        "generator_code_sha": "0" * 64,
        "repository": {"commit": "UNKNOWN", "branch": "UNKNOWN"},
        "host": "UNKNOWN",
        "tasks": [
            task("governance_fact_ledger_v1", "governance", "PASSED"),
            task("exact78_wave0_freeze_v1", "selection", "PASSED"),
            task("exact78_wave0_clean_v1", "clean", "PENDING"),
            task("exact78_upstream_c_successors_v1", "upstream_repair", "PENDING"),
            task("contact_occlusion_canary_034_042_v1", "contact", "PENDING"),
            task("robot_visual_canary_034_042_v1", "robot", "BLOCKED_PREREQ"),
            task("humanego_visual_aux_v1", "training", "BLOCKED_PREREQ"),
            task("workspace_cleanup_v3", "maintenance", "PENDING")
        ],
        "blockers": [
            {"name": "KaiHand adapter CAD", "status": "BLOCKED_EXTERNAL", "scope": "Physical Robot authority", "resolution": "Verified adapter CAD and measured installation transform."},
            {"name": "Robot TCP and installation calibration", "status": "BLOCKED_EXTERNAL", "scope": "Physical Robot authority", "resolution": "Measured TCP, mount and camera/world-to-base calibration."},
            {"name": "Real Robot action demonstrations", "status": "BLOCKED_EXTERNAL", "scope": "HumanEgo policy checkpoints", "resolution": "Synchronized action/state/RGB data satisfying the formal schema."}
        ],
        "recent_events": [{"task_id": "governance_fact_ledger_v1", "session": None, "status": "PASSED", "created_at": now_iso(), "message": "Initial evidence-driven governance ledger published."}],
        "next_task": {"task_id": "exact78_wave0_freeze_v1", "session": None, "prerequisites": ["governance_fact_ledger_v1"], "expected_resource": "CPU/read-only evidence hashing", "stop_condition": "Any source artifact bytes/SHA mismatch or denominator conflict."}
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, default=0)
    args = parser.parse_args()
    receipt = publish_bundle(build_authority(), build_task_state(), event_type="GOVERNANCE_BOOTSTRAP", expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
