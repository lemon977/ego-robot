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
    data_cleaning_receipt = (
        repo_root / "tasks/receipts/HANDLE_DATA_CLEANING_V3_COMPLETION.json")
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
            "handle_data_cleaning_v3": {
                "terminal_status": "COMMITTED",
                "execution_status": "COMPLETE_NO_ACTIVE_TASK",
                "implementation_revision": "V3",
                "code_closure": [artifact_ref(repo_root / path) for path in (
                    "src/chaoyang/ops/tactile_quality_gate_v1.py",
                    "src/chaoyang/ops/convert_handle_egodex_v3.py",
                    "src/chaoyang/ops/batch_clean_handle_content_v3.py",
                    "src/chaoyang/ops/run_handle_cleaning_v3_queue.py",
                )],
                "completion_receipt": artifact_ref(data_cleaning_receipt),
                "current_document": artifact_ref(
                    repo_root / "docs/current/DATA_CLEANING_0911_0915_ZH.md"),
                "counts": {
                    "session_count": 681,
                    "completed": 681,
                    "cleaned": 561,
                    "rejected": 120,
                    "failed": 0,
                },
                "modality_contract": {
                    "0911_0914": "MANUS25_CONTROLLERS_CAMERA_TACTILE_PRESENT",
                    "0915_manus": "ABSENT_NOT_CAPTURED",
                    "0915_pico26": "PRESENT_AND_PRESERVED",
                },
                "gpu_required": False,
                "restart_authorized": False,
                "claim_limit": "Raw tactile integrity/activity and declared modality completeness only; not calibrated force, contact truth, or physical accuracy.",
            },
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
                "challengers": [],
                "challenger_is_current_authority": False,
                "selection_policy": "SAM3.1_ONLY_USER_LOCKED",
                "claim_limit": "The 0915 full-funnel campaign uses only the pinned SAM3.1 weight; SAM2.1 and Cutie are not executable candidates.",
            },
            "processed_0915_full_funnel_v1": {
                "execution_status": "PENDING_NEW_BOUNDED_TASK",
                "input_root": "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915",
                "denominator": {"sessions": 220, "frames": 58686},
                "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
                "model_policy": {"hand": "HaWoR", "mask": "SAM3.1_ONLY"},
                "code_closure": [artifact_ref(repo_root / path) for path in (
                    "src/chaoyang/ops/audit_0915_processed_self_containment_v1.py",
                    "src/chaoyang/ops/run_sensor_leftmono_hawor_v1.py",
                    "src/chaoyang/ops/run_0915_leftmono_sam31_masks_v1.py",
                )],
                "claim_limit": "Processed-only structural and development pipeline contract; no PICO26 hand input, physical calibration, policy or deployment authority.",
            },
            "vst_image_domain_ab_v1": {
                "execution_scope": "SINGLE_SESSION_CPU_VISUAL_DIAGNOSTIC",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [artifact_ref(
                    repo_root / "src/chaoyang/ops/analyze_0915_vst_image_domain_ab_v1.py"
                )],
                "candidate_domain": "PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_RESIZE_ONLY",
                "authority_promoted": False,
                "claim_limit": "Image-domain A/B only; no model, calibration, full-batch or physical-accuracy authority.",
            },
            "hawor_resize_only_canary_v1": {
                "execution_scope": "PLAY_CARDS_0915_001_ONLY",
                "weights": "ONE_LOGICAL_HAWOR_INFERENCE_BUNDLE_V1",
                "gpu_required": True,
                "code_closure": [artifact_ref(
                    repo_root / "src/chaoyang/ops/run_0915_hawor_resize_only_canary_v1.py"
                )],
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_RESIZE_ONLY",
                "authority_promoted": False,
                "claim_limit": "One-session HaWoR development canary only; no SAM, Depth, batch or physical authority.",
            },
            "robot15h_window_start_inventory_v1": {
                "execution_scope": "0915_220_SESSION_INVENTORY_AND_W0_W1_FREEZE",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_window_start_inventory_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/governance/robot15h_task_specs_v1.py"),
                ],
                "input_root": "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915",
                "source_group_authority": "USER_CONFIRMED_AND_METADATA_CORROBORATED",
                "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "0916_consumption": "FORBIDDEN",
                "claim_limit": "Inventory/split/scheduling evidence only; no model, Robot, Contact, training or deployment result.",
            },
            "robot_quality_recovery_0915_15h_v21": {
                "execution_scope": "FROZEN_0915_V21_SIXTEEN_SESSION_ACCESS_LEDGER_AND_NINE_INTERNAL_PACKAGES",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot_quality_recovery_15h_v2.py"),
                    artifact_ref(repo_root / "src/chaoyang/governance/register_0915_robot_recovery_v21.py"),
                    artifact_ref(repo_root / "src/chaoyang/governance/finalize_0915_robot_recovery_v21.py"),
                ],
                "input_root": "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915",
                "source_group_authority": "USER_CONFIRMED_AND_METADATA_CORROBORATED",
                "source_group_authority_evidence": [
                    artifact_ref(repo_root / "tasks/receipts/0915_ROBOT15H_USER_AUTHORIZATION_V1.json"),
                    artifact_ref(repo_root / "tasks/receipts/0915_ROBOT15H_WINDOW_START_INVENTORY_V1_RESULT.json"),
                ],
                "historical_v21_execution_claim": "METADATA_VERIFIED_NO_USER_ATTESTATION",
                "provenance_correction": (
                    "The V2.1 execution packet conservatively omitted the already-recorded user "
                    "attestation. Current provenance restores that attestation using immutable "
                    "authorization and inventory receipts; terminal V2.1 outputs, splits and "
                    "algorithm conclusions are unchanged."
                ),
                "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "0916_consumption": "FORBIDDEN",
                "contact_authority": {
                    "local_stereo_metric_dev": "CONDITIONAL_SAME_FRAME_SAME_ENCODED_DEPTH_ONLY",
                    "external_metric_authority": False,
                    "control_ground_truth": False,
                    "physical_deployment_authorized": False,
                },
                "authority_promoted": False,
                "claim_limit": "V2.1 development-relative recovery and offline visualization only; no training, external metric, control or deployment authority.",
            },
            "three_stream_stable_baseline_v31": {
                "execution_scope": "EXACT78_AI1_AI2_ISOLATED_STABLE_BASELINE_PARENT",
                "weights": "ABSENT_AT_PARENT_EACH_CHILD_ONE_PINNED_WEIGHT_OR_ABSENT",
                "gpu_required": "CONDITIONAL_SINGLE_CENTRAL_LEASE",
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_three_stream_stable_baseline_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/record_three_stream_lane_result_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/build_wiyh_wrist_dual_representation_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/build_exact78_wrist_comparison_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_exact78_stereo_preflight_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/governance/publish_three_stream_v31_implementation.py"),
                    artifact_ref(repo_root / "src/chaoyang/governance/register_three_stream_stable_baseline_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/governance/build_three_stream_status_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/research/world_in_your_hands/wrist_dual_representation_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/hand_observability_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/temporal_authority_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/hawor_bounded_comparison_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/kai22_r0_tiered_admission_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/human_ego/exact78_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/human_ego/tools/audit_exact78_suffix_invariance_v31.py"),
                    artifact_ref(repo_root / "src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/exact78_stereo_preflight_v31.py"),
                ],
                "schema_closure": [
                    artifact_ref(repo_root / "contracts/wrist_dual_producer_config_v1.schema.json"),
                    artifact_ref(repo_root / "contracts/wrist_dual_representation_v1.schema.json"),
                    artifact_ref(repo_root / "contracts/hand_observability_v1.schema.json"),
                    artifact_ref(repo_root / "contracts/temporal_authority_audit_v1.schema.json"),
                    artifact_ref(repo_root / "contracts/hawor_bounded_comparison_v31.schema.json"),
                    artifact_ref(repo_root / "contracts/kai22_r0_tiered_admission_v1.schema.json"),
                ],
                "lane_roots": {
                    "exact78": "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001/lanes/exact78",
                    "ai1": "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001/lanes/ai1",
                    "ai2": "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001/lanes/ai2",
                },
                "authority_separation": [
                    "PIPELINE_COMPLETE",
                    "NUMERIC_QUALITY_PASS",
                    "VISUAL_REVIEW_STATUS",
                    "TRAINING_COMPLETE",
                    "TRAINING_ELIGIBLE",
                    "CONTROL_GROUND_TRUTH",
                    "PHYSICAL_DEPLOYABLE",
                ],
                "external_metric_authority": False,
                "control_ground_truth": False,
                "physical_deployment_authorized": False,
                "claim_limit": "V3.1 coordination and development baselines only; lane diagnostics constrain only their direct consumers.",
            },
            "robot_recovery_hawor_w1_diag_v1": {
                "execution_scope": "FIXED_0915_W1_DIAG_POKER044_AND_CHIPS097_INSIDE_V21_PARENT",
                "weights": "ONE_LOGICAL_HAWOR_INFERENCE_BUNDLE_V1",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot_recovery_hawor_w1_diag_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_w1_diag_persistent_worker_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_persistent_worker_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_hawor_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_play_cards_0910_001_hawor_raw.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "cohort_role": "W1_DIAG",
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "candidate_policy": "FREEZE_BEFORE_W1_ADOPTION_AND_NO_RESULT_BASED_THRESHOLD_CHANGE",
                "numeric_thresholds_changed": False,
                "authority_promoted": False,
                "claim_limit": "Fixed W1-DIAG HaWoR evidence and per-side structural eligibility only; no R0 quality, adoption, control, training or deployment authority.",
            },
            "robot_recovery_hawor_w1_adoption_v1": {
                "execution_scope": "FIXED_0915_W1_ADOPTION_POKER106_AND_CHIPS029_INSIDE_V21_PARENT",
                "weights": "ONE_LOGICAL_HAWOR_INFERENCE_BUNDLE_V1",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot_recovery_hawor_w1_adoption_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_w1_adoption_persistent_worker_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot_recovery_hawor_w1_diag_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_w1_diag_persistent_worker_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_persistent_worker_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_hawor_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "cohort_role": "W1_ADOPTION",
                "parent_candidate_signature_sha256": "9ccf7ff5eef43158848f73206a772ae756c0387cd84de9ecc72a2ee678db781b",
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "algorithm_behavior_policy": "A1_EXECUTION_BODY_EQUIVALENT_AFTER_ENUMERATED_ROLE_AND_ACCESS_SUBSTITUTION",
                "numeric_thresholds_changed": False,
                "authority_promoted": False,
                "claim_limit": "Fixed W1-ADOPTION HaWoR evidence only; no automatic adoption, R0 quality, control, training or deployment authority.",
            },
            "robot_recovery_w1_adoption_review_v1": {
                "execution_scope": "A2_W1_ADOPTION_TWO_SESSION_FULL_TIMELINE_REVIEW",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [artifact_ref(
                    repo_root / "src/chaoyang/ops/render_0915_robot_recovery_w1_adoption_review_v1.py"
                )],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "parent_candidate_signature_sha256": "9ccf7ff5eef43158848f73206a772ae756c0387cd84de9ecc72a2ee678db781b",
                "adoption_adapter_or_run_signature_sha256": "e3ca5185a1a333f2e49600ca8649f9312c85e15ebb5133f903a945b2bd92b784",
                "authority_promoted": False,
                "claim_limit": "Full-timeline W1-ADOPTION raw HaWoR overlay only; no accuracy, adoption, R0 quality, control or deployment authority.",
            },
            "kai22_r0_quality_successor_v1": {
                "execution_scope": "FIXED_A1_W1_DIAG_AND_A2_W1_ADOPTION_CPU_OWN_GATE_DIAGNOSTIC",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_kai22_r0_quality_successor_v1.py"),
                    artifact_ref(repo_root / "contracts/robot_recovery/A3_R0_QUALITY_SUCCESSOR_A1_DIAGNOSTIC_V1.json"),
                    artifact_ref(repo_root / "contracts/robot_recovery/A3_R0_QUALITY_SUCCESSOR_A2_DIAGNOSTIC_V1.json"),
                    artifact_ref(repo_root / "contracts/robot_recovery/A3_R0_QUALITY_SUCCESSOR_CONFIG_V1.schema.json"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "run_mode": "DIAGNOSTIC_ONLY_NO_QUALITY_UPGRADE",
                "evaluation_scope": "A1_POKER044_CHIPS097_AND_A2_POKER106_CHIPS029_STRUCTURAL_SIDE_FRAMES",
                "collision_scope": "PINNED_KAIHAND_INTRA_HAND_NON_PARENT_ONLY",
                "retarget_internal_mapping_clip": "PRESENT_IN_robot_visual_relative_v1_BEFORE_Q22_RETURN",
                "posthoc_gate_clip_allowed": False,
                "known_saturation_root_cause": "A3_CHIPS_THUMB_JOINT4_UPPER_BOUND_FROM_HALF_UNSIGNED_MANO_DISTAL_BEND",
                "motion_gate_status": "BLOCKED_MISSING_FROZEN_POSITIVE_VELOCITY_ACCELERATION_AND_MINIMUM_WINDOW_THRESHOLDS",
                "numeric_thresholds_changed": False,
                "authority_promoted": False,
                "claim_limit": "CPU full-frame q22 limit/FK/intra-hand collision and motion diagnostics only; structural/static-gate success is not R0 quality admission, and no control, training, external metric or deployment authority is granted.",
            },
            "robot_quality_recovery_v21_sam31_b1": {
                "execution_scope": "FOUR_W0_CACHE_REGRESSIONS_AND_FIXED_W1_DIAG_POKER044_CHIPS097",
                "weights": "ONE_LOGICAL_SAM31_MULTIPLEX_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot_quality_recovery_v21_sam31_b1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_sam31_temporal_identity_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_sam31_task_object_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "cohort_role": "W0_CACHE_REGRESSION_PLUS_W1_DIAG",
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "hand_prompt_policy": "A1_SAME_DOMAIN_RAW_DIRECT_OBSERVED_2D_PER_SIDE",
                "object_dependency": "INDEPENDENT_OF_HAND_AND_HAWOR_TERMINAL",
                "numeric_thresholds_changed": False,
                "authority_promoted": False,
                "claim_limit": "Development diagnostic SAM Hand/Object evidence only; no Mask accuracy, physical identity, hidden-surface, Contact, training, control or deployment authority.",
            },
            "robot_quality_recovery_v21_sam31_b1r_poker044_hand": {
                "execution_scope": "POKER044_HAND_ONLY_EXACT_TRACKER_KEYERROR164_FAIL_CLOSED_RECOVERY",
                "weights": "ONE_LOGICAL_SAM31_MULTIPLEX_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot_quality_recovery_v21_sam31_b1r_poker044_hand.py"),
                    artifact_ref(repo_root / "tasks/current/0915_robot_quality_recovery_v21_sam31_b1r_poker044_hand/B1R_TASK_POLICY.json"),
                    artifact_ref(repo_root / "tasks/current/0915_robot_quality_recovery_v21_sam31_b1r_poker044_hand/B1R_INPUT_MANIFEST.json"),
                    artifact_ref(repo_root / "tasks/current/0915_robot_quality_recovery_v21_sam31_b1r_poker044_hand/B1R_ROOT_CAUSE_AUDIT_V1.json"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "parent_b1_terminal": "FAILED_RUNTIME_FINAL_IMMUTABLE",
                "bounded_exception_conversion": "EXACT_KEYERROR_164_AT_ANCHOR165_REVERSE_ZERO_YIELD_TO_UNKNOWN_DIRECTION_INCOMPLETE",
                "affected_side_terminal": "REJECTED_TRACKER_DIRECTION_INCOMPLETE",
                "prompt_threshold_weight_change": False,
                "authority_promoted": False,
                "claim_limit": "Poker044 Hand-only bounded runtime recovery; the original B1 terminal remains immutable and every unreturned frame is UNKNOWN, with no Mask accuracy, training, control or deployment authority.",
            },
            "robot_quality_recovery_v21_b3_sam31_independent_hand_equipment_canary_v1": {
                "execution_scope": "POKER044_AND_CHIPS097_TEXT_ONLY_HAND_AND_WRIST_WORN_EQUIPMENT_GEOMETRY_CANARY",
                "weights": "ONE_PINNED_SAM31_MULTIPLEX_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_b3_sam31_independent_hand_equipment_canary_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/publish_b3_shallow_reviews_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/test_b3_sam31_independent_hand_equipment_canary_v1.py"),
                    artifact_ref(repo_root / "contracts/robot_recovery/B3_MASK_SUCCESSOR_CANARY_V1.json"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"),
                    artifact_ref(repo_root / "vendor/SAM3/sam3/model/sam3_multiplex_tracking.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "fixed_sessions": ["play_cards_0915_044", "get_potato_chips_0915_097"],
                "semantic_inputs": "PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY_RGB_AND_FIXED_TEXT_PROMPTS_ONLY",
                "left_right_identity": "UNKNOWN",
                "cross_chunk_identity": "NOT_STITCHED",
                "overlap_policy": "ZERO_PIXEL_HARD_REJECT_NO_SUBTRACT_NO_UNION",
                "empty_output_semantics": "UNKNOWN_NOT_ABSENT",
                "consumer_allowed": False,
                "authority_promoted": False,
                "claim_limit": "Development geometry-only SAM3.1 Hand/worn-equipment canary; no Mask accuracy, physical tracker identity, left/right identity, Object, Clean, Contact, training, control or deployment authority.",
            },
            "robot_quality_recovery_v21_b3_sam31_independent_hand_equipment_boundary_v2": {
                "execution_scope": "ONE_BOUNDARY_ONLY_SUCCESSOR_FOR_B3_V1_FAILED_RUNTIME_FINAL",
                "weights": "ONE_PINNED_SAM31_MULTIPLEX_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_b3_sam31_independent_hand_equipment_canary_v2.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/test_b3_boundary_successor_v2.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/publish_b3_shallow_reviews_v1.py"),
                    artifact_ref(repo_root / "contracts/robot_recovery/B3_MASK_SUCCESSOR_CANARY_V2.json"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"),
                    artifact_ref(repo_root / "vendor/SAM3/sam3/model/sam3_multiplex_tracking.py"),
                    artifact_ref(repo_root / "vendor/SAM3/sam3/model/sam3_multiplex_detector.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "immutable_failed_parent": "B3_MASK_SUCCESSOR_GPU_COMMAND_RECEIPT_FAILED_RUNTIME_FINAL",
                "only_change": "CONSUME_EXACT_HALF_OPEN_WINDOW_AND_CLOSE_GENERATOR_BEFORE_POISON_BOUNDARY",
                "prompts_thresholds_windows_weights_changed": False,
                "fixed_sessions": ["play_cards_0915_044", "get_potato_chips_0915_097"],
                "left_right_identity": "UNKNOWN",
                "cross_chunk_identity": "NOT_STITCHED",
                "consumer_allowed": False,
                "authority_promoted": False,
                "claim_limit": "One runtime-boundary-only successor to the immutable B3 v1 failure; no Mask accuracy, physical tracker identity, left/right identity, Object, Clean, Contact, training, control or deployment authority.",
            },
            "robot_quality_recovery_v21_d1_clean_prep_cpu_v1": {
                "execution_scope": "POKER044_AND_CHIPS097_FAIL_CLOSED_CLEAN_INPUT_NORMALIZATION_AND_ZERO_WRITE_PREPARATION",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/materialize_d1_clean_prep_composite_input.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_d1_clean_prep_cpu_orchestrator.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/validate_d1_clean_prep_prepared_output.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/test_d1_clean_prep_cpu_orchestrator.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/render_d1_clean_prep_offline_review_v1.py"),
                    artifact_ref(repo_root / "contracts/robot_recovery/D1_CLEAN_PREP_INPUT_MANIFEST_TEMPLATE.json"),
                    artifact_ref(repo_root / "contracts/robot_recovery/B1_D1_CLEAN_PREP_HANDOFF_TEMPLATE.json"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "input_authority": "B1R_POKER_HAND_CAPABILITY_PLUS_PINNED_ORIGINAL_B1_SUBORDINATE_ARTIFACTS",
                "fixed_sessions": ["play_cards_0915_044", "get_potato_chips_0915_097"],
                "domain_contract": "M_REMOVE_SUBSET_M_WRITE_SUBSET_M_FLOW_AND_UNKNOWN_EQUALS_M_WRITE_UNTIL_FRESH_MATERIALIZATION",
                "candidate_materialized": False,
                "old_clean_or_future_donor_allowed": False,
                "authority_promoted": False,
                "claim_limit": "CPU-only Clean input normalization and zero-write preparation; Poker identity and unavailable Hand/Object evidence remain UNKNOWN, and no inpainting quality, Clean terminal, training, Contact, control or deployment authority is granted.",
            },
            "robot_quality_recovery_v21_d2_fresh_propainter_offline_v1": {
                "execution_scope": "POKER044_ZERO_WRITE_PASS_THROUGH_AND_CHIPS097_FRESH_OFFLINE_PROPAINTER_VISUAL_CHALLENGER",
                "weights": "THREE_PINNED_EXISTING_PROPAINTER_WEIGHTS_NO_MODEL_SWAP",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_d2_fresh_propainter_offline_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/publish_d2_shallow_visuals_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/test_d2_fresh_propainter_offline_v1.py"),
                    artifact_ref(repo_root / "contracts/robot_recovery/D2_FRESH_PROPAINTER_OFFLINE_V1.json"),
                    artifact_ref(repo_root / "vendor/ProPainter/inference_propainter.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "fixed_sessions": ["play_cards_0915_044", "get_potato_chips_0915_097"],
                "execution_semantics": "OFFLINE_BIDIRECTIONAL_VISUAL_ONLY",
                "input_mask_domain": "M_FLOW",
                "effective_internal_model_mask": "DILATE_RESIZED_M_FLOW_BY_4_INSIDE_PROPAINTER",
                "publish_write_domain": "M_WRITE_ONLY",
                "unknown_after_fill": "EXACTLY_M_WRITE",
                "poker_action": "RAW_PASS_THROUGH_ZERO_WRITE_NO_MODEL",
                "chips_object_slots": "THREE_SEPARATE_UNKNOWN_SLOTS_NO_UNION",
                "training_eligible": False,
                "clean_terminal": False,
                "authority_promoted": False,
                "claim_limit": "One frozen fresh ProPainter offline visual challenger; synthetic pixels remain UNKNOWN and no Clean truth, causal training, Contact, control or deployment authority is granted.",
            },
            "b2_depth_to_visible_object_poker044_v1": {
                "execution_scope": "POKER044_PINNED_DEPTH_AND_VISIBLE_MASK_CANDIDATE_CPU_REVIEW_ONLY",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_b2_depth_to_visible_object_poker044_v1.py"),
                    artifact_ref(repo_root / "contracts/robot_recovery/B2_DEPTH_TO_VISIBLE_OBJECT_POKER044_V1.json"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_object6d_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/object6d_planar_observability_v2.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/object6d_planar_observability_v1.py"),
                ],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "candidate_id": "B2_DEPTH_TO_VISIBLE_OBJECT_GEOMETRY_SUCCESSOR_V1",
                "session_id": "play_cards_0915_044",
                "depth_execution": "REUSE_PINNED_CACHE_NO_RERUN",
                "physical_card_identity": "UNKNOWN_UNBOUND",
                "face_identity": "UNKNOWN_UNBOUND",
                "review_only": True,
                "consumer_allowed": False,
                "authority_promoted": False,
                "claim_limit": "CPU-only Poker044 direct-visible finite-surface review canary from pinned cached Depth and visible-mask candidate data; physical-card/face identity remains UNKNOWN and no Contact, hidden geometry, external metric, Clean, Robot, training, control or deployment authority is granted.",
            },
            "robot_recovery_c1_cpu_internal_selfcheck_v1": {
                "execution_scope": "FROZEN_C0_V2_C1_POKER119_INTERNAL_GEOMETRY_REPLAY",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [artifact_ref(
                    repo_root / "src/chaoyang/ops/run_0915_robot_recovery_c1_cpu_internal_selfcheck_v1.py"
                )],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "authoritative_input": "C0_v2_AND_C1",
                "distance_threshold_m": 0.005,
                "thresholds_changed": False,
                "authority_promoted": False,
                "claim_limit": "CPU-only immutable-cache internal consistency replay; no external metric/registration accuracy, Contact truth, R1-E, training, control or deployment authority.",
            },
            "robot_recovery_w1_diag_review_v1": {
                "execution_scope": "A1_W1_DIAG_TWO_SESSION_FULL_TIMELINE_REVIEW",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [artifact_ref(
                    repo_root / "src/chaoyang/ops/render_0915_robot_recovery_w1_diag_review_v1.py"
                )],
                "parent_task_id": "0915_robot_quality_recovery_15h_v2",
                "candidate_signature_sha256": "9ccf7ff5eef43158848f73206a772ae756c0387cd84de9ecc72a2ee678db781b",
                "authority_promoted": False,
                "claim_limit": "Full-timeline W1-DIAG raw HaWoR overlay only; no accuracy, R0 quality, control or deployment authority.",
            },
            "robot15h_hawor_wave0_v1": {
                "execution_scope": "FOUR_FROZEN_0915_DEVELOPMENT_SESSIONS",
                "weights": "ONE_LOGICAL_HAWOR_INFERENCE_BUNDLE_V1",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_hawor_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_play_cards_0910_001_hawor_raw.py"),
                ],
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "forbidden_inputs": ["legacy_prepared_mono", "equiDis62_remap", "PICO26_hand", "trackingData_hand"],
                "claim_limit": "Development HaWoR W0 only; no Contact, control, deployment, training or generalization authority.",
            },
            "robot15h_hawor_wave0_recovery_v1": {
                "execution_scope": "FOUR_FROZEN_0915_DEVELOPMENT_SESSIONS_IMPORT_RECOVERY",
                "weights": "ONE_LOGICAL_HAWOR_INFERENCE_BUNDLE_V1",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_hawor_wave0_recovery_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_hawor_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"),
                ],
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "recovery_scope": "PINNED_ENV_PROJECT_IMPORT_PATH_ONLY",
                "claim_limit": "Recorded import-path recovery only; no Contact, control, deployment, training or generalization authority.",
            },
            "robot15h_kai22_r0_wave0_v1": {
                "execution_scope": "FOUR_FROZEN_0915_W0_KAI22_R0",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_kai22_r0_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_hawor_bounded_parameter_successor.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/robot_visual_relative_v1.py"),
                ],
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "robot_layer": "R0_WRIST_LOCAL_Q22",
                "claim_limit": "Development Kai22 R0 only; no Contact, arm IK, control, training, deployment or measured calibration authority.",
            },
            "robot15h_scale_cause_audit_v1": {
                "execution_scope": "SEALED_PLAY_CARDS_0915_001_NON_CONTACT_SURFACE_EVIDENCE",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_scale_cause_audit_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/governance/robot15h_task_specs_v1.py"),
                ],
                "fit_estimand": "MANO_VISIBLE_SURFACE_OPTICAL_Z_TO_STEREO_VISIBLE_SURFACE_OPTICAL_Z",
                "contact_or_object_fit": "FORBIDDEN",
                "scale_bound_change": "FORBIDDEN",
                "claim_limit": "Development cause classification only; no metric calibration, Contact, control, training or deployment authority.",
            },
            "robot15h_foundationstereo_wave0_v1": {
                "execution_scope": "FOUR_FROZEN_0915_DEVELOPMENT_SESSIONS_ENCODED_DOMAIN",
                "weights": "ONE_LOGICAL_FOUNDATIONSTEREO_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_foundationstereo_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_foundationstereo_encoded_domain_canary_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/stereo_encoded_domain_preflight_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/vst_encoded_video_domain.py"),
                ],
                "input_domain": "PHYSICAL_EYES_SOURCEINDEX_1_0_CROP_RESIZE_ONLY",
                "disparity_sign_adapter": "SIMULTANEOUS_HORIZONTAL_REFLECTION_NO_CAMERA_SWAP_THEN_UNFLIP",
                "encoded_projection_accuracy": "UNVERIFIED",
                "strict_metric_contact_authorized": False,
                "claim_limit": "W0 development Depth and visual Object6D candidate input only; no metric Contact, control, training or deployment authority.",
            },
            "robot15h_foundationstereo_wave0_recovery_v1": {
                "execution_scope": "ONE_INTERRUPTED_W0_SESSION_PLUS_THREE_SHA_VERIFIED_ATOMIC_PREDECESSORS",
                "weights": "ONE_LOGICAL_FOUNDATIONSTEREO_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_foundationstereo_wave0_recovery_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_foundationstereo_wave0_v1.py"),
                ],
                "input_domain": "PHYSICAL_EYES_SOURCEINDEX_1_0_CROP_RESIZE_ONLY",
                "rerun_scope": "GET_POTATO_CHIPS_0915_042_ONLY",
                "incomplete_predecessor_staging": "PRESERVED_NOT_CONSUMED",
                "encoded_projection_accuracy": "UNVERIFIED",
                "strict_metric_contact_authorized": False,
                "claim_limit": "Bounded one-session runtime recovery only; no metric Contact, control, training or deployment authority.",
            },
            "robot15h_sam31_temporal_identity_v1": {
                "execution_scope": "FOUR_FROZEN_0915_W0_HAND_TEMPORAL_IDENTITY_ONLY",
                "weights": "ONE_LOGICAL_SAM31_MULTIPLEX_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_sam31_temporal_identity_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"),
                ],
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "prompt_semantics": "MULTIPLEX_GEOMETRIC_BOX_FROM_DIRECT_OBSERVED_HAWOR",
                "raw_semantic_separation": "MANDATORY",
                "unanchored_roles": "FAIL_CLOSED_UNKNOWN",
                "object_mask_consumer_allowed": False,
                "claim_limit": "W0 hand temporal-identity development evidence only; no object-mask, Contact, control, training or deployment authority.",
            },
            "robot15h_sam31_temporal_identity_recovery_v1": {
                "execution_scope": "SAME_FOUR_W0_SESSIONS_PRE_MODEL_IMPORT_RECOVERY",
                "weights": "ONE_LOGICAL_SAM31_MULTIPLEX_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_sam31_temporal_identity_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"),
                ],
                "recovery_change": "BIND_REPOSITORY_LOCAL_VENDOR_SAM3_IMPORT_PATH_ONLY",
                "input_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
                "prompt_semantics": "MULTIPLEX_GEOMETRIC_BOX_FROM_DIRECT_OBSERVED_HAWOR",
                "unanchored_roles": "FAIL_CLOSED_UNKNOWN",
                "object_mask_consumer_allowed": False,
                "claim_limit": "Bounded pre-model-import recovery; unchanged hand-only development authority and no object-mask, Contact, control, training or deployment authority.",
            },
            "robot15h_w1_release_scope_gate_v1": {
                "execution_scope": "EIGHT_FROZEN_0915_W1_SESSIONS_SCOPE_ACCOUNTING_ONLY",
                "weights": "BOUND_BY_TASK_PACKET_OR_ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_wave_scope_gate_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/governance/robot15h_task_specs_v1.py"),
                ],
                "source_video_open_count": 0,
                "blocked_nodes": ["hawor_w1", "sam31_w1", "robot_w1"],
                "claim_limit": "Release-scope and missing-input blockers only; no model execution, export, quality success, control, training or deployment authority.",
            },
            "robot15h_foundationstereo_w1_v1": {
                "execution_scope": "EIGHT_FROZEN_0915_W1_SESSIONS_ENCODED_DOMAIN",
                "weights": "ONE_LOGICAL_FOUNDATIONSTEREO_CHECKPOINT",
                "gpu_required": True,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_foundationstereo_waves_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_foundationstereo_wave0_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_foundationstereo_encoded_domain_canary_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/pipeline/stereo_encoded_domain_preflight_v1.py"),
                ],
                "input_domain": "PHYSICAL_EYES_SOURCEINDEX_1_0_CROP_RESIZE_ONLY",
                "disparity_sign_adapter": "SIMULTANEOUS_HORIZONTAL_REFLECTION_NO_CAMERA_SWAP_THEN_UNFLIP",
                "source_binding": "COPY_FROZEN_W1_ROWS_AND_BIND_SHA_AT_SCHEDULING",
                "final_holdout_release": "H9_OR_LATER",
                "encoded_projection_accuracy": "UNVERIFIED",
                "strict_metric_contact_authorized": False,
                "claim_limit": "W1 development Depth and visual/Object6D candidate input only; no strict metric Contact, control, training or deployment authority.",
            },
            "robot15h_final_evidence_v1": {
                "execution_scope": "H14_RUNTIME_VIDEO_TEST_AND_LOCAL_COMMIT_SIDECARS",
                "weights": "ABSENT",
                "gpu_required": False,
                "code_closure": [
                    artifact_ref(repo_root / "src/chaoyang/ops/build_0915_robot15h_final_evidence_v1.py"),
                    artifact_ref(repo_root / "src/chaoyang/ops/run_0915_robot15h_release_control_v1.py"),
                ],
                "minimum_time": "H14_GPU_DRAIN",
                "remote_push": False,
                "claim_limit": "Final evidence integrity sidecars only; no algorithm, control, training, deployment or physical authority.",
            },
            "handle_data_cleaning_v4_0916": {
                "execution_status": "PENDING_NEW_BOUNDED_TASK",
                "input_root": "/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916",
                "target_root": "/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916",
                "denominator": {"sessions": 240},
                "workers_max": 2,
                "weights": "ABSENT",
                "conversion_semantics": "UNCHANGED_FROM_V3",
                "code_closure": [artifact_ref(repo_root / path) for path in (
                    "src/chaoyang/ops/convert_handle_egodex_v3.py",
                    "src/chaoyang/ops/batch_clean_handle_content_v4.py",
                    "src/chaoyang/ops/run_0915_0916_input_audit_clean_v1.py",
                )],
                "downstream_execution": "FORBIDDEN_IN_THIS_TASK",
                "claim_limit": "0916 cleaning and content admission only; no downstream algorithm authority.",
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
        _doc("current_navigation", "CURRENT", repo_root / "docs/current/README_ZH.md", scope="CURRENT_SHALLOW_ENTRY", claim_limit="Unique shallow entry only; machine facts remain receipt-bound."),
        _doc("ai_work_entry", "CURRENT", repo_root / "docs/current/AI_WORK_ENTRY_ZH.md", scope="CURRENT_AI_WORK_HANDOFF", claim_limit="Reading order, task naming, resource boundaries and candidate optimization lanes; it does not authorize execution without a current task packet."),
        _doc("three_stream_plan_v31", "CURRENT", repo_root / "docs/current/PLAN.md", scope="CURRENT_THREE_STREAM_PLAN", claim_limit="Three-stream objectives, interfaces, isolation and stop rules; execution still requires the current routed task packet."),
        _doc("three_stream_status_v31", "CURRENT", repo_root / "docs/current/STATUS.json", scope="CURRENT_THREE_STREAM_SHALLOW_STATUS", claim_limit="Generated navigation projection only; the task ledger and lane receipts remain authoritative."),
        _doc("three_stream_exact78_v31", "CURRENT", repo_root / "docs/current/EXACT78.md", scope="CURRENT_THREE_STREAM_EXACT78_HANDOFF", claim_limit="Exact78 lane navigation only; training completion and quality remain receipt-bound."),
        _doc("three_stream_ai1_v31", "CURRENT", repo_root / "docs/current/AI1.md", scope="CURRENT_THREE_STREAM_AI1_HANDOFF", claim_limit="AI1 lane navigation only; calibration, fusion and metric claims remain receipt-bound."),
        _doc("three_stream_ai2_v31", "CURRENT", repo_root / "docs/current/AI2.md", scope="CURRENT_THREE_STREAM_AI2_HANDOFF", claim_limit="AI2 lane navigation only; observability, HaWoR and Robot quality remain receipt-bound."),
        _doc("data_cleaning_0911_0915", "CURRENT", repo_root / "docs/current/DATA_CLEANING_0911_0915_ZH.md", scope="CURRENT_DATA_CLEANING_BASELINE", claim_limit="Three-dataset cleaning terminal state and reuse boundary only; tactile activity is not contact or force truth.", evidence=[repo_root / "tasks/receipts/HANDLE_DATA_CLEANING_V3_COMPLETION.json"]),
        _doc("handle_data_cleaning_v3_guide", "CURRENT", repo_root / "docs/guides/data/HANDLE_DATA_CLEANING_V3.md", scope="CURRENT_DATA_CLEANING_ALGORITHM_GUIDE", claim_limit="V3 cleaning design, modality contract and output schema; terminal counts remain bound to the completion receipt."),
        _doc("clean_baseline_v1_reference", "CURRENT", repo_root / "docs/reference/architecture/CLEAN_BASELINE_V1_ZH.md", scope="CURRENT_CLEAN_BASELINE_ARCHITECTURE", claim_limit="Repository layout, archive and restore contract; current facts remain receipt-bound."),
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
        "current_navigation",
        "ai_work_entry",
        "three_stream_plan_v31",
        "three_stream_status_v31",
        "three_stream_exact78_v31",
        "three_stream_ai1_v31",
        "three_stream_ai2_v31",
        "data_cleaning_0911_0915",
        "handle_data_cleaning_v3_guide",
        "clean_baseline_v1_reference",
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
