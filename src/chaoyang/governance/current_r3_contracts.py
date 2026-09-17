"""Pure builders and validators for the V7.1-R3 governance projections."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from chaoyang.governance.common import artifact_ref, validate_artifact_ref


QUALITY_GATE_CLASSES = {
    "HARD_STRUCTURAL": "Finite/schema/frame/coordinate/decode/SHA constraints; failure closes the attempt.",
    "DOWNSTREAM_ELIGIBILITY": "Controls only the named consumer and does not erase an upstream result.",
    "SOFT_DIAGNOSTIC": "Reported for comparison or tuning and never blocks Robot geometry by itself.",
    "EXTERNAL_AUTHORITY": "Requires independent physical or human ground truth; absence blocks only that authority.",
}


STAGE_GATE_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "Raw": {
        "HARD_STRUCTURAL": ["unique_session_identity", "source_sha", "frame_identity"],
        "DOWNSTREAM_ELIGIBILITY": ["cohort_contract_matches_consumer"],
        "SOFT_DIAGNOSTIC": [],
        "EXTERNAL_AUTHORITY": [],
    },
    "HaWoR": {
        "HARD_STRUCTURAL": ["finite_MANO", "left_right_identity", "frame_identity"],
        "DOWNSTREAM_ELIGIBILITY": ["bounded_temporal_update", "bone_cv"],
        "SOFT_DIAGNOSTIC": ["stereo_surface_disagreement", "pose_similarity"],
        "EXTERNAL_AUTHORITY": ["external_wrist_and_tip_truth"],
    },
    "Role Mask": {
        "HARD_STRUCTURAL": ["exact_frame_count", "offscreen_empty", "left_right_not_swapped"],
        "DOWNSTREAM_ELIGIBILITY": ["visible_coverage", "reentry_identity", "contact_boundary_leakage"],
        "SOFT_DIAGNOSTIC": ["area_temporal_variation"],
        "EXTERNAL_AUTHORITY": ["independent_pixel_gold_accuracy"],
    },
    "Object Mask": {
        "HARD_STRUCTURAL": ["same_physical_identity", "three_chips_instances_independent", "offscreen_empty"],
        "DOWNSTREAM_ELIGIBILITY": ["reentry_latency", "visible_object_retention", "contact_boundary_leakage"],
        "SOFT_DIAGNOSTIC": ["temporal_area_variation"],
        "EXTERNAL_AUTHORITY": ["independent_instance_gold_accuracy"],
    },
    "Depth": {
        "HARD_STRUCTURAL": ["finite_disparity", "Z_equals_fB_over_d", "image_domain_identity", "same_session_calibration"],
        "DOWNSTREAM_ELIGIBILITY": ["registration_residual", "valid_range", "depth_quality_evidence"],
        "SOFT_DIAGNOSTIC": ["left_right_consistency", "static_region_temporal_drift", "edge_low_texture_motion_blur_risk"],
        "EXTERNAL_AUTHORITY": ["30_50_70_100cm_external_distance_validation"],
    },
    "Object6D": {
        "HARD_STRUCTURAL": ["finite_SE3", "proper_rotation", "coordinate_domain_closed", "KEEP_INVALID"],
        "DOWNSTREAM_ELIGIBILITY": ["direct_observed_only", "instance_identity", "selected_camera_adapter"],
        "SOFT_DIAGNOSTIC": ["plane_fit_residual", "visible_surface_stability"],
        "EXTERNAL_AUTHORITY": ["external_object_pose_truth"],
    },
    "Clean": {
        "HARD_STRUCTURAL": ["frame_count", "source_map", "master_decode", "write_domain_closed"],
        "DOWNSTREAM_ELIGIBILITY": ["visible_object_retention", "human_residual_no_regression", "nonhuman_leakage_no_regression", "causal_pixel_source"],
        "SOFT_DIAGNOSTIC": ["contact_band_removed_area", "synthetic_pixel_ratio"],
        "EXTERNAL_AUTHORITY": ["physical_background_truth"],
    },
    "Contact": {
        "HARD_STRUCTURAL": ["evidence_DAG_acyclic", "attachment_cannot_prove_contact", "instance_identity", "UNKNOWN_fail_closed"],
        "DOWNSTREAM_ELIGIBILITY": ["direct_or_independent_contact_seed", "per_finger_object_binding"],
        "SOFT_DIAGNOSTIC": ["surface_distance", "relative_velocity", "slip_score", "uncertainty"],
        "EXTERNAL_AUTHORITY": ["independent_contact_gold_or_tactile_binding"],
    },
    "Robot Visual": {
        "HARD_STRUCTURAL": ["finite", "proper_rotation", "coordinate_chain_closed", "joint_limits", "digital_mesh_collision"],
        "DOWNSTREAM_ELIGIBILITY": ["wrist_path_ratio", "workspace_clipping_ratio", "causal_render_provenance"],
        "SOFT_DIAGNOSTIC": ["strict_human_pose_similarity", "per_joint_pose_residual", "trajectory_smoothness"],
        "EXTERNAL_AUTHORITY": ["TCP_installation_calibration", "physical_collision_truth", "real_robot_action"],
    },
    "Occlusion": {
        "HARD_STRUCTURAL": ["unified_zbuffer", "pixel_source_enum", "unknown_fail_closed"],
        "DOWNSTREAM_ELIGIBILITY": ["silver_provenance_coverage", "object_pixel_preservation", "authorized_band_byte_exact"],
        "SOFT_DIAGNOSTIC": ["unknown_ratio", "temporal_consistency", "depth_tie_ratio"],
        "EXTERNAL_AUTHORITY": ["frozen_independent_goldset_accuracy"],
    },
    "HumanEgo Aux": {
        "HARD_STRUCTURAL": ["raw_robotized_pair_identity", "causal_input", "split_isolation", "future2d_schema"],
        "DOWNSTREAM_ELIGIBILITY": ["minimum_sessions", "minimum_H50_windows", "silver_compositor_QA"],
        "SOFT_DIAGNOSTIC": ["ADE", "FDE", "PCK", "identity_error", "temporal_smoothness"],
        "EXTERNAL_AUTHORITY": ["multi_seed_statistical_claim"],
    },
    "HumanEgo Policy": {
        "HARD_STRUCTURAL": ["real_robot_action_schema", "synchronization", "split_isolation"],
        "DOWNSTREAM_ELIGIBILITY": ["policy_training_data_ready"],
        "SOFT_DIAGNOSTIC": ["policy_loss", "offline_policy_metrics"],
        "EXTERNAL_AUTHORITY": ["real_robot_action", "physical_deployment_validation"],
    },
}


def build_algorithm_contract(
    baseline_registry: Mapping[str, Any], authority: Mapping[str, Any]
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    depth10_receipt = repo_root / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/artifacts/depth_10/attempts/attempt_0003_real_play_cards_0910_001/RUN_RECEIPT.json"
    depth20_receipt = repo_root / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/artifacts/depth_20/attempts/attempt_0003_real_input_preflight/RUN_RECEIPT.json"
    gap_audit_root = repo_root / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_v71_r3_execution_status_audit/attempts/attempt_0004"
    stages: list[dict[str, Any]] = []
    for entry in baseline_registry.get("entries", []):
        stage = str(entry["stage"])
        gates = STAGE_GATE_OVERRIDES.get(stage)
        if gates is None:
            raise RuntimeError(f"R3 quality-gate classification missing for stage: {stage}")
        stages.append(
            {
                "stage": stage,
                "algorithm_id": entry["algorithm_id"],
                "implementation_revision": "R7_1_R3",
                "code_closure": entry.get("code_closure", []),
                "weights": entry.get("weights"),
                "input_authority": entry.get("input_authority", "UNKNOWN_VERIFICATION_REQUIRED"),
                "output_schema": entry.get("output_schema"),
                "quality_gates": gates,
                "known_limitations": entry.get("known_limitations", []),
                "successor_requirement": entry.get("successor_requirement", "UNKNOWN_VERIFICATION_REQUIRED"),
                "superseded_implementations": entry.get("superseded_implementations", []),
            }
        )
    return {
        "schema_version": "chaoyang-algorithm-contract-v1",
        "governance_revision": authority["governance_revision"],
        "generation_id": authority["generation_id"],
        "generated_at": authority["generated_at"],
        "status": "CURRENT",
        "quality_gate_classes": QUALITY_GATE_CLASSES,
        "stages": stages,
        "special_status_contracts": {
            "sensor_h4_formal": {
                "execution_status": "BLOCKED_RESOURCE",
                "qa_status": "NOT_EVALUATED",
                "policy_status": "POLICY_DEFERRED",
                "pixel_mask_authority": False,
                "claim_limit": "The formal R7_0 preflight ran no SAM3.1 pixel inference.",
            },
            "sensor_h4_development_canary": {
                "execution_status": "FAILED_QUALITY_C",
                "qa_status": "DEVELOPMENT_CANARY_EVALUATED",
                "pixel_mask_authority": False,
                "may_override_formal_h4": False,
            },
            "sensor_h4_bounded_policy_decision": {
                "terminal_status": "FAILED_QUALITY_C",
                "decision": "REJECT_CURRENT_AUTOMATIC_GLOVE_CONTROLLER_MASK_TO_CLEAN_ROUTE",
                "receipt": artifact_ref(repo_root / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_chaoyang_v71_r3/sensor_h4_bounded_decision/attempts/attempt_0001/RUN_RECEIPT.json"),
                "formal_h4_state_changed": False,
                "pixel_mask_authority": False,
                "unaffected_branches": ["H1_CONTROLLER_MANUS_HAND_OBSERVATION", "H2_TACTILE_TIMING", "H3_STEREO_AFTER_CALIBRATION_CONTRACT_REPAIR", "RAW_VISUAL_RESEARCH"],
                "claim_limit": "Bounded development/policy decision only; it does not reject the sensor data or independently eligible H1/H2/H3 branches.",
            },
            "robot_pose_similarity": {
                "gate_class": "SOFT_DIAGNOSTIC",
                "may_block_robot_geometry": False,
            },
            "mask_baseline": {
                "current": "SAM3.1",
                "challengers": ["SAM2.1", "Cutie"],
                "challenger_is_current_authority": False,
            },
            "visual_aux_checkpoint_index_cli": {
                "code_closure": [artifact_ref(Path(__file__).resolve().parents[3] / "src/chaoyang/ops/build_visual_aux_checkpoint_index_v71.py")],
                "interface": "standard argparse/--help",
                "physical_deployment_authorized": False,
            },
            "cleanup_cli": {
                "code_closure": [artifact_ref(repo_root / "src/chaoyang/ops/cleanup_current_only_v71.py")],
                "test_closure": [
                    artifact_ref(repo_root / "tests/test_cleanup_current_only_v71.py"),
                    artifact_ref(repo_root / "tests/test_cleanup_current_only_v71_cli.py"),
                ],
                "default_mode": "DRY_RUN",
                "destructive_mode_requires": "--commit-delete",
                "any_cwd_cli_required": True,
                "pre_post_current_authority_protection_required": True,
            },
            "visual_aux_final_hardening": {
                "code_closure": [artifact_ref(repo_root / path) for path in (
                    "src/chaoyang/human_ego/tools/build_visual_aux_session_bundle_v54.py",
                    "src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py",
                    "src/chaoyang/ops/run_v71_post_robot_visual_aux.py",
                )],
                "test_closure": [artifact_ref(repo_root / path) for path in (
                    "tests/human_ego/test_visual_aux_bundle_v52.py",
                    "tests/human_ego/test_visual_aux_future2d_v53.py",
                    "tests/test_run_v71_post_robot_visual_aux.py",
                )],
                "control_ground_truth": False,
                "physical_deployment_authorized": False,
            },
            "visual_aux_rc1_final": {
                "consumer_field": "consumer_eligibility.visual_aux_rc1",
                "contract": artifact_ref(repo_root / "docs/governance/VISUAL_AUX_RC1_CONTRACT.json"),
                "eligibility_schema": artifact_ref(repo_root / "contracts/visual_aux_rc1_eligibility.schema.json"),
                "validator": artifact_ref(repo_root / "src/chaoyang/human_ego/tools/visual_aux_rc1_contract.py"),
                "input_mode": "CAUSAL_TRAINING_INPUT",
                "global_robot_authority_required": False,
                "contact_authority_required": False,
                "complete_object_atlas_required": False,
                "control_ground_truth": False,
                "physical_deployment_authorized": False,
            },
            "robot_v75_output_contracts": {
                "schema_closure": [artifact_ref(repo_root / path) for path in (
                    "contracts/robot_target_10_r3.schema.json",
                    "contracts/robot_reach_20_r3.schema.json",
                    "contracts/robot_profile_30_r3.schema.json",
                )],
                "test_closure": [artifact_ref(repo_root / path) for path in (
                    "tests/test_robot_r3_output_schemas.py",
                    "tests/test_robot_v75_recovery_target_reach_profile.py",
                    "tests/test_robot_hard_soft_watcher_v75_adopt.py",
                )],
                "authority_promoted": False,
            },
            "depth10_real_development": {
                "terminal_status": "BLOCKED_PREREQ",
                "evidence_status": "DEVELOPMENT_EVIDENCE",
                "receipt": artifact_ref(depth10_receipt),
                "authority_promoted": False,
                "claim_limit": "Same-session CPU QA exposed source-routing/rectification prerequisites; no fresh FoundationStereo inference or Depth authority.",
            },
            "depth20_real_input_preflight": {
                "terminal_status": "BLOCKED_PREREQ",
                "evidence_status": "DEVELOPMENT_EVIDENCE",
                "receipt": artifact_ref(depth20_receipt),
                "authority_promoted": False,
                "claim_limit": "Controller/MANUS input is ready, but Stereo correction remains unauthorized; no fused wrist was produced.",
            },
            "development_watchers": {
                "robot_hard_soft_v75": {
                    "code_closure": [artifact_ref(repo_root / "src/chaoyang/ops/run_robot_hard_soft_audit_watcher_v71.py")],
                    "evidence_class": "DEVELOPMENT_ONLY",
                    "may_register_authority": False,
                },
                "occlusion_visible_surface_v75": {
                    "code_closure": [artifact_ref(repo_root / "src/chaoyang/ops/run_occlusion_visible_surface_watcher_v71.py")],
                    "evidence_class": "DEVELOPMENT_ONLY",
                    "may_register_authority": False,
                },
                "claim_limit": "Running watcher outputs are immutable development candidates only and cannot alter stage authority counts.",
            },
            "r3_execution_status_gap_audit": {
                "status": "PASSED_POINT_IN_TIME_AUDIT",
                "counts": {"PASSED": 2, "BLOCKED": 7, "DEVELOPMENT": 6, "MISSING": 4},
                "result": artifact_ref(gap_audit_root / "RESULT.json"),
                "execution_status_audit": artifact_ref(gap_audit_root / "EXECUTION_STATUS_AUDIT.json"),
                "receipt": artifact_ref(gap_audit_root / "RUN_RECEIPT.json"),
                "authority_promoted": False,
                "claim_limit": "Point-in-time deliverable audit. PASSED may mean structural accounting only; missing, development and blocked stages remain non-authoritative.",
            },
        },
        "claim_limit": "Machine-classified current algorithm contract; internal residuals are not external physical accuracy.",
    }


def _doc(
    document_id: str,
    status: str,
    path: Path,
    *,
    scope: str,
    claim_limit: str,
    supersedes: list[str] | None = None,
    evidence: list[Path] | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    reference = artifact_ref(path)
    return {
        "document_id": document_id,
        "status": status,
        **reference,
        "supersedes": supersedes or [],
        "evidence_receipts": [artifact_ref(item) for item in (evidence or []) if item.is_file()],
        "authorized_scope": scope,
        "claim_limit": claim_limit,
    }


def build_doc_authority_map(
    authority: Mapping[str, Any], governance_root: Path, repo_root: Path
) -> dict[str, Any]:
    migration = governance_root / "PLAN_MIGRATION_RECEIPT.json"
    candidates = [
        _doc("repository_readme", "CURRENT", repo_root / "README.md", scope="PROJECT_NAVIGATION_PROTOCOL", claim_limit="Navigation only; current facts remain receipt-bound and current algorithms come from the algorithm contract."),
        _doc("agent_execution_protocol", "CURRENT", repo_root / "AGENTS.md", scope="AI_EXECUTION_PROTOCOL", claim_limit="Execution/reading rules only; not measurement evidence or stage authority."),
        _doc("sensor_h4_bounded_policy_decision", "CURRENT", repo_root / "docs/current/reference/SENSOR_H4_BOUNDED_POLICY_DECISION.md", scope="SENSOR_H4_DEVELOPMENT_POLICY_DECISION", claim_limit="Rejects only the current automatic glove/Controller Mask-to-Clean route; formal H4 remains BLOCKED_RESOURCE/NOT_EVALUATED/POLICY_DEFERRED and H1/H2/H3 stay independent.", evidence=[repo_root / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_chaoyang_v71_r3/sensor_h4_bounded_decision/attempts/attempt_0001/RUN_RECEIPT.json"]),
        _doc("r3_execution_status_gap_audit", "HISTORICAL", repo_root / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_v71_r3_execution_status_audit/attempts/attempt_0004/DECISION.md", scope="HISTORICAL_R3_POINT_IN_TIME_DELIVERABLE_GAP_AUDIT", claim_limit="Historical point-in-time audit; it must not drive current Robot counts or routing.", evidence=[repo_root / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_v71_r3_execution_status_audit/attempts/attempt_0004/RUN_RECEIPT.json"]),
        _doc("r3_execution_dag_semantics", "CURRENT", governance_root / "EXECUTION_DAG_SEMANTICS_R3_ZH.md", scope="CURRENT_EXECUTION_DAG_SEMANTICS", claim_limit="Dependency interpretation and scheduling rules only; not stage completion evidence or authority."),
        _doc("current_project_status", "CURRENT", governance_root / "CURRENT_PROJECT_STATUS_ZH.md", scope="PROJECT_REALTIME_STATUS", claim_limit="Generated projection; receipt-bound ledgers remain factual sources."),
        _doc("current_stage_baselines", "CURRENT", governance_root / "CURRENT_STAGE_BASELINES_ZH.md", scope="CURRENT_STAGE_BASELINES", claim_limit="Generated algorithm summary; live counts come from the task ledger."),
        _doc("robot_quality_gate_policy", "CURRENT", governance_root / "ROBOT_QUALITY_GATE_POLICY_V71_ZH.md", scope="CURRENT_ROBOT_QUALITY_GATE_POLICY", claim_limit="Defines Robot hard, downstream, soft and external gate boundaries; it does not prove a run passed."),
        _doc("raw_to_humanego_reproduction", "CURRENT", repo_root / "docs/reference/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md", scope="CURRENT_END_TO_END_REPRODUCTION_GUIDE", claim_limit="Stable reproduction interfaces and ordering only; current versions and facts remain receipt-bound."),
        _doc("current_shallow_visual_navigation", "CURRENT", repo_root / "docs/current/visuals/README_ZH.md", scope="CURRENT_SHALLOW_VISUAL_NAVIGATION", claim_limit="Navigation to bounded visual-review evidence only; videos do not confer geometry, contact, training or physical authority."),
        _doc("clean_contact_wrist_baseline_audit", "CURRENT", repo_root / "docs/research/current/reports/visualization/20260915/CLEAN_CONTACT_AND_WRIST_BASELINE_AUDIT_ZH.md", scope="CURRENT_CLEAN_CONTACT_WRIST_BOUNDARY_AUDIT", claim_limit="Bounded visual and algorithm-boundary audit; synthetic pixels and visual overlap are not physical contact truth."),
        _doc("governance_consistency_repair_handoff", "CURRENT", governance_root / "GOVERNANCE_CONSISTENCY_REPAIR_20260917_ZH.md", scope="CURRENT_GOVERNANCE_CONSISTENCY_HANDOFF", claim_limit="Explains the validator and routing repair; current facts remain bound to the receipt and machine indexes."),
        _doc("historical_0909_status_video_index", "HISTORICAL", repo_root / "archive/baseline-20260917-0aa69e9/content/history/docs/stale-linked/docs/reference/pipeline/CURRENT_STATUS_AND_VIDEO_INDEX_ZH.md", scope="HISTORICAL_0909_STATUS_VIDEO_INDEX", claim_limit="Frozen 2026-09-11 snapshot; it must not route current work or supply current counts."),
        _doc("v71_r3_execution_plan", "CURRENT", governance_root / "CHAOYANG_V7_1_R3_EXECUTION_PLAN_ZH.md", scope="CURRENT_ARCHITECTURE_CONTRACT", claim_limit="Architecture and invariant contract only; RC1-FINAL exclusively routes current delivery work.", supersedes=["v71_execution_plan", "optimization_execution_v1"], evidence=[migration]),
        _doc("rc1_final_delivery_plan", "CURRENT", governance_root / "CHAOYANG_RC1_FINAL_DELIVERY_PLAN_ZH.md", scope="CURRENT_DELIVERY_PLAN", claim_limit="Finite delivery contract; task receipts and the RC1 eligibility index prove execution and training readiness.", supersedes=["rc1_delivery_proposal", "r22_bounded_integration_plan"], evidence=[migration]),
        _doc("r22_bounded_integration_plan", "HISTORICAL", governance_root / "CHAOYANG_R2_2_4H_INTEGRATION_PLAN_ZH.md", scope="HISTORICAL_BOUNDED_INTEGRATION_PLAN", claim_limit="Completed bounded integration history; it must not route RC1 work."),
        _doc("rc1_delivery_proposal", "SUPERSEDED", repo_root / "docs/DELIVERY_ROADMAP_RC1_20260916_ZH.md", scope="SUPERSEDED_RC1_PROPOSAL", claim_limit="Pre-final proposal retained for audit; RC1-FINAL is the only delivery plan."),
        _doc("rc1_gate_proposal", "SUPERSEDED", repo_root / "docs/DELIVERY_GATE_AND_TASK_CONTRACT_PROPOSED.json", scope="SUPERSEDED_RC1_GATE_PROPOSAL", claim_limit="Pre-final proposed gate contract; use the receipt-bound RC1 release and consumer contracts."),
        _doc("v71_execution_plan", "SUPERSEDED", governance_root / "EXACT78_V7_1_EXECUTION_PLAN_ZH.md", scope="SUPERSEDED_EXECUTION_PLAN", claim_limit="Historical V7.1 plan; must not route new R3 work.", evidence=[migration]),
        _doc("optimization_execution_v1", "SUPERSEDED", repo_root / "docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_EXECUTION_TASKS_V1_ZH.md", scope="SUPERSEDED_OPTIMIZATION_PLAN", claim_limit="Historical proposal retained for audit only.", evidence=[migration]),
        _doc("optimization_exec00_bootstrap_v1", "SUPERSEDED", repo_root / "docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_EXEC00_BOOTSTRAP_TASK_PACKET_V1.json", scope="SUPERSEDED_BOOTSTRAP_PACKET", claim_limit="Cancelled bootstrap retained for audit; it must not dispatch work."),
        _doc("optimization_decision_roadmap_20260915", "HISTORICAL", repo_root / "docs/research/current/CHAOYANG_PROJECT_OPTIMIZATION_DECISION_ROADMAP_20260915_ZH.md", scope="HISTORICAL_OPTIMIZATION_ROADMAP", claim_limit="Decision history only; current execution is routed by R2.2 task packets."),
        _doc("clean_layered_sam31_exploration_v1", "SUPERSEDED", repo_root / "docs/research/current/CLEAN_LAYERED_SAM31_SUCCESSOR_EXPLORATION_V1_ZH.md", scope="SUPERSEDED_CLEAN_EXPLORATION_DESIGN", claim_limit="The GPU_CANARY_NOT_STARTED design state is obsolete; use immutable R2.2 Mask/Clean receipts."),
        _doc("docs_navigation", "CURRENT", repo_root / "docs/README.md", scope="CURRENT_DOCS_NAVIGATION", claim_limit="Unique current documentation navigation; factual claims remain subordinate to receipt-bound governance artifacts."),
        _doc("v71_automation_handoff", "HISTORICAL", governance_root / "V71_AUTOMATION_HANDOFF_ZH.md", scope="HISTORICAL_AUTOMATION_HANDOFF", claim_limit="Historical handoff; it must not select current tasks."),
        _doc("v3_implementation_handoff", "HISTORICAL", governance_root / "V3_IMPLEMENTATION_HANDOFF_ZH.md", scope="HISTORICAL_IMPLEMENTATION_HANDOFF", claim_limit="Historical handoff; it must not select current tasks."),
        _doc("exact78_completion_roadmap", "HISTORICAL", governance_root / "EXACT78_COMPLETION_EXECUTION_ROADMAP_ZH.md", scope="HISTORICAL_EXACT78_ROADMAP", claim_limit="Historical roadmap; current counts and routing come from the receipt-bound ledger."),
        _doc("exact78_v52_execution_plan", "HISTORICAL", governance_root / "EXACT78_V5_2_EXECUTION_PLAN_ZH.md", scope="HISTORICAL_EXECUTION_PLAN", claim_limit="Historical evidence only; not current routing authority."),
        _doc("long_horizon_plan", "HISTORICAL", governance_root / "LONG_HORIZON_TASK_PLAN_ZH.md", scope="HISTORICAL_LONG_HORIZON_PLAN", claim_limit="Historical design record; current work uses Task Packets and R3 plan."),
        _doc("clean_terminal_audit_v52_1", "SUPERSEDED", repo_root / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/TERMINAL_AUDIT_V52_1.json", scope="SUPERSEDED_CLEAN_TERMINAL_AUDIT", claim_limit="The 39-pass/19-runtime-final snapshot predates recovery_v53 and must not drive current Clean counts.", evidence=[migration]),
    ]
    documents = [item for item in candidates if item is not None and item.get("status") == "CURRENT" and "/archive/" not in str(item.get("path", ""))]
    return {
        "schema_version": "chaoyang-doc-authority-map-v1",
        "governance_revision": authority["governance_revision"],
        "generation_id": authority["generation_id"],
        "generated_at": authority["generated_at"],
        "status": "CURRENT",
        "documents": documents,
        "claim_limit": "Only CURRENT entries may guide current work; generated files remain subordinate to the current receipt.",
    }


def validate_doc_authority_map(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    current_scopes = [
        str(item["authorized_scope"])
        for item in value.get("documents", [])
        if item.get("status") == "CURRENT"
    ]
    for scope, count in Counter(current_scopes).items():
        if count > 1:
            errors.append(f"multiple CURRENT documents for scope {scope}: {count}")
    statuses = {str(item["document_id"]): str(item["status"]) for item in value.get("documents", [])}
    required_current = {
        "repository_readme",
        "robot_quality_gate_policy",
        "raw_to_humanego_reproduction",
        "current_shallow_visual_navigation",
        "clean_contact_wrist_baseline_audit",
        "governance_consistency_repair_handoff",
        "docs_navigation",
        "sensor_h4_bounded_policy_decision",
    }
    for document_id in sorted(required_current):
        if statuses.get(document_id) != "CURRENT":
            errors.append(f"required navigation document is not CURRENT: {document_id}")
    for item in value.get("documents", []):
        if item.get("status") == "CURRENT" and "/archive/" in str(item.get("path", "")):
            errors.append(f"CURRENT document points into archive: {item.get('document_id')}")
    for item in value.get("documents", []):
        if item.get("status") != "CURRENT":
            continue
        for target in item.get("supersedes", []):
            if statuses.get(str(target)) == "CURRENT":
                errors.append(f"CURRENT document supersedes another CURRENT document: {target}")
    for item in value.get("documents", []):
        errors.extend(validate_artifact_ref(item))
        for evidence in item.get("evidence_receipts", []):
            errors.extend(validate_artifact_ref(evidence))
    bootstrap = next(
        (item for item in value.get("documents", []) if item.get("document_id") == "optimization_exec00_bootstrap_v1"),
        None,
    )
    if bootstrap is not None:
        try:
            payload = json.loads(Path(str(bootstrap["path"])).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read superseded bootstrap: {exc}")
        else:
            if payload.get("terminal_status") != "CANCELLED":
                errors.append("superseded EXEC-00 bootstrap is not CANCELLED")
    return errors
