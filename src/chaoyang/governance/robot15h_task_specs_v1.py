"""Frozen task graph for the authorized 0915 Robot fifteen-hour window.

Only one node may be published as ``execution_allowed`` at a time.  The DAG is
therefore a scheduling plan, not permission to run every node concurrently.
Model nodes bind exactly one pinned logical weight; orchestration and CPU nodes
bind ``ABSENT``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PLAN_REVISION = "0915_ROBOT_15H_V1"
EXECUTION_REVISION = "0915_ROBOT_15H_V1"
WINDOW_RUN_ID = "robot15h-0915-20260919T000959+0800"
WINDOW_CLOCK = "_run/current/0915_robot15h_v1/WINDOW_CLOCK.json"

HAWOR_WEIGHT = "assets/models/vendor/hawor/0915_HAWOR_INFERENCE_BUNDLE_V1.json"
SAM31_WEIGHT = "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
FOUNDATION_WEIGHT = "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"

TASK_ORDER = (
    "0915_robot15h_window_start_inventory_v1",
    "0915_robot15h_scale_cause_audit_v1",
    "0915_robot15h_hawor_wave0_v1",
    "0915_robot15h_hawor_wave0_recovery_v1",
    "0915_robot15h_kai22_r0_wave0_v1",
    "0915_robot15h_foundationstereo_wave0_v1",
    "0915_robot15h_foundationstereo_wave0_recovery_v1",
    "0915_robot15h_sam31_temporal_identity_v1",
    "0915_robot15h_sam31_temporal_identity_recovery_v1",
    "0915_robot15h_sam31_temporal_identity_quality_v2",
    "0915_robot15h_sam31_task_object_wave0_v1",
    "0915_robot15h_sam31_task_object_wave0_recovery_v1",
    "0915_robot15h_geometry_object6d_wave0_v1",
    "0915_robot15h_interaction_occlusion_v1",
    "0915_robot15h_contact_dual_evidence_v1",
    "0915_robot15h_robot_relative_refinement_v1",
    "0915_robot15h_robot_virtual_arm_v1",
    "0915_robot15h_release_candidate_v1",
    "0915_robot15h_hawor_waves_v1",
    "0915_robot15h_sam31_waves_v1",
    "0915_robot15h_foundationstereo_waves_v1",
    "0915_robot15h_robot_waves_v1",
    "0915_robot15h_window_release_audit_v1",
)

DEPENDENCIES: dict[str, list[str]] = {
    "0915_robot15h_window_start_inventory_v1": ["0915_human_stereo_alignment_bound_audit_v1"],
    "0915_robot15h_scale_cause_audit_v1": ["0915_robot15h_window_start_inventory_v1"],
    "0915_robot15h_hawor_wave0_v1": ["0915_robot15h_window_start_inventory_v1"],
    "0915_robot15h_hawor_wave0_recovery_v1": ["0915_robot15h_hawor_wave0_v1"],
    "0915_robot15h_kai22_r0_wave0_v1": ["0915_robot15h_hawor_wave0_recovery_v1"],
    "0915_robot15h_foundationstereo_wave0_v1": ["0915_robot15h_window_start_inventory_v1"],
    "0915_robot15h_foundationstereo_wave0_recovery_v1": ["0915_robot15h_foundationstereo_wave0_v1"],
    "0915_robot15h_sam31_temporal_identity_v1": ["0915_robot15h_hawor_wave0_recovery_v1"],
    "0915_robot15h_sam31_temporal_identity_recovery_v1": ["0915_robot15h_sam31_temporal_identity_v1"],
    "0915_robot15h_sam31_temporal_identity_quality_v2": [
        "0915_robot15h_sam31_temporal_identity_recovery_v1"
    ],
    "0915_robot15h_sam31_task_object_wave0_v1": [
        "0915_robot15h_sam31_temporal_identity_quality_v2"
    ],
    "0915_robot15h_sam31_task_object_wave0_recovery_v1": [
        "0915_robot15h_sam31_task_object_wave0_v1"
    ],
    "0915_robot15h_geometry_object6d_wave0_v1": [
        "0915_robot15h_foundationstereo_wave0_recovery_v1",
        "0915_robot15h_sam31_task_object_wave0_recovery_v1",
    ],
    "0915_robot15h_interaction_occlusion_v1": [
        "0915_robot15h_hawor_wave0_recovery_v1",
        "0915_robot15h_sam31_task_object_wave0_recovery_v1",
        "0915_robot15h_geometry_object6d_wave0_v1",
    ],
    "0915_robot15h_contact_dual_evidence_v1": [
        "0915_robot15h_geometry_object6d_wave0_v1",
        "0915_robot15h_interaction_occlusion_v1",
    ],
    "0915_robot15h_robot_relative_refinement_v1": [
        "0915_robot15h_kai22_r0_wave0_v1",
        "0915_robot15h_contact_dual_evidence_v1",
    ],
    "0915_robot15h_robot_virtual_arm_v1": ["0915_robot15h_kai22_r0_wave0_v1"],
    "0915_robot15h_release_candidate_v1": list(
        TASK_ORDER[1:TASK_ORDER.index("0915_robot15h_release_candidate_v1")]
    ),
    "0915_robot15h_hawor_waves_v1": ["0915_robot15h_release_candidate_v1"],
    "0915_robot15h_sam31_waves_v1": ["0915_robot15h_release_candidate_v1"],
    "0915_robot15h_foundationstereo_waves_v1": ["0915_robot15h_release_candidate_v1"],
    "0915_robot15h_robot_waves_v1": [
        "0915_robot15h_hawor_waves_v1",
        "0915_robot15h_sam31_waves_v1",
        "0915_robot15h_foundationstereo_waves_v1",
    ],
    "0915_robot15h_window_release_audit_v1": list(TASK_ORDER[:-1]),
}

WEIGHTS: dict[str, str | list[str]] = {task_id: "ABSENT" for task_id in TASK_ORDER}
for task_id in (
    "0915_robot15h_hawor_wave0_v1",
    "0915_robot15h_hawor_wave0_recovery_v1",
    "0915_robot15h_hawor_waves_v1",
):
    WEIGHTS[task_id] = [HAWOR_WEIGHT]
for task_id in (
    "0915_robot15h_sam31_temporal_identity_v1",
    "0915_robot15h_sam31_temporal_identity_recovery_v1",
    "0915_robot15h_sam31_temporal_identity_quality_v2",
    "0915_robot15h_sam31_task_object_wave0_v1",
    "0915_robot15h_sam31_task_object_wave0_recovery_v1",
    "0915_robot15h_sam31_waves_v1",
):
    WEIGHTS[task_id] = [SAM31_WEIGHT]
for task_id in (
    "0915_robot15h_foundationstereo_wave0_v1",
    "0915_robot15h_foundationstereo_wave0_recovery_v1",
    "0915_robot15h_foundationstereo_waves_v1",
):
    WEIGHTS[task_id] = [FOUNDATION_WEIGHT]

FIRST_ATTEMPT = (
    "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001"
)

TASK_SPECS: dict[str, dict[str, Any]] = {
    TASK_ORDER[0]: {
        "phase": "ROBOT15H_WINDOW_START_INVENTORY",
        "objective": (
            "Freeze the immutable fifteen-hour clock and corrected DAG, inventory only "
            "the 220-session 0915 processed cohort, corroborate independent source "
            "recordings, freeze source-group splits and W0/W1 selections, and capture "
            "resource state without running a model."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "tasks/receipts/0915_ROBOT15H_USER_AUTHORIZATION_V1.json",
            "docs/chaoyang_robot_15h_v1/01_EXECUTION_PLAN_15H_ZH.md",
            "docs/chaoyang_robot_15h_v1/BUNDLE_MANIFEST.json",
            "tasks/receipts/0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json",
            "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/DATASET_RESULT.json",
            "src/chaoyang/ops/run_0915_robot15h_window_start_inventory_v1.py",
            WINDOW_CLOCK,
        ],
        "write_set": [
            FIRST_ATTEMPT,
            "tasks/receipts/0915_ROBOT15H_WINDOW_START_INVENTORY_V1_RESULT.json",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "only_0915_single_underscore_processed_root",
            "0916_highview_FORBIDDEN",
            "source_and_processed_roots_READ_ONLY",
            "physical_left_sourceIndex1_crop_resize_only",
            "lens_undistortion_FORBIDDEN",
            "play_cards_0915_001_DEVELOPMENT_ONLY",
            "gpu_FORBIDDEN",
            "weights_ABSENT",
        ],
        "required_outputs": [
            "CLAIM.json",
            "RUN_SIGNATURE.json",
            "WINDOW_CLOCK_REF.json",
            "RESOURCE_SNAPSHOT.json",
            "SOURCE_GROUP_MANIFEST.json",
            "BATCH_MANIFEST.json",
            "FROZEN_DAG.json",
            "METRICS.json",
            "RESULT.json",
            "RUN_RECEIPT.json",
        ],
        "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 0.5},
        "expected_resource": "CPU_IO_ONLY; GPU_FORBIDDEN; weights ABSENT",
        "claim_limit": (
            "Inventory, source grouping, split and scheduling authority only.  It does "
            "not create HaWoR, Depth, Contact, Robot or cross-recording generalization "
            "results and does not authorize 0916 consumption."
        ),
    },
}


def _placeholder(task_id: str) -> dict[str, Any]:
    """Return a bounded future-node packet skeleton.

    The node is inert until its real runner and required outputs are implemented,
    reviewed and CAS-registered.  Keeping the skeleton here freezes dependency and
    weight identity without pretending the node is executable now.
    """

    return {
        "phase": task_id.upper(),
        "objective": f"Execute the bounded {task_id} node within the frozen fifteen-hour window.",
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            WINDOW_CLOCK,
            "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/FROZEN_DAG.json",
        ],
        "write_set": [f"_run/current/{task_id}/attempts/attempt_0001"],
        "prerequisites": [
            *[f"{value}=TERMINAL" for value in DEPENDENCIES[task_id]],
            "window_deadline_not_exceeded",
            "source_and_processed_roots_READ_ONLY",
            "0916_highview_FORBIDDEN",
        ],
        "required_outputs": ["CLAIM.json", "RUN_SIGNATURE.json", "RESULT.json", "RUN_RECEIPT.json"],
        "budgets": {"runtime_attempts": 1, "window_deadline_bound": True},
        "expected_resource": "SERIAL_GOVERNED_NODE",
        "claim_limit": "Development-only bounded node; no control, deployment or training authority.",
    }


for _task_id in TASK_ORDER[1:]:
    TASK_SPECS[_task_id] = _placeholder(_task_id)


TASK_SPECS["0915_robot15h_scale_cause_audit_v1"] = {
    "phase": "ROBOT15H_HUMAN_STEREO_SCALE_CAUSE_AUDIT",
    "objective": (
        "Reproduce and correctly name the reported 0.783730 affine slope, test its "
        "stability across hands and temporal blocks, audit encoded-domain camera "
        "geometry assumptions, and classify rather than conceal unresolved scale causes."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_human_stereo_surface_association_canary_v1/attempts/attempt_0001/MANO_SURFACE_ASSOCIATION_V1.json",
        "_run/current/0915_human_stereo_alignment_bound_audit_v1/attempts/attempt_0001/ALIGNMENT_BOUND_SATURATION_AUDIT_V1.json",
        "_run/current/0915_foundationstereo_encoded_domain_canary_v1/attempts/attempt_0001/DEPTH_CONTRACT.json",
        "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/playing_cards/play_cards_0915_001/camera_params.json",
        "src/chaoyang/ops/run_0915_robot15h_scale_cause_audit_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_scale_cause_audit_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_SCALE_CAUSE_AUDIT_V1",
        "tasks/receipts/0915_ROBOT15H_SCALE_CAUSE_AUDIT_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_window_start_inventory_v1=PASSED",
        "sealed_001_surface_pairs_READ_ONLY",
        "contact_and_object_fit_FORBIDDEN",
        "fixed_48mm_bias_FORBIDDEN",
        "scale_bound_change_FORBIDDEN",
        "encoded_intrinsics_external_accuracy_remains_UNVERIFIED",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "SCALE_CAUSE_MATRIX.json",
        "FIT_STABILITY.json", "DEPTH_GEOMETRY_AUDIT.json", "METRICS.json",
        "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 2.5},
    "expected_resource": "CPU_ONLY_READ_ONLY_SCALE_CAUSE_AUDIT; weights ABSENT",
    "claim_limit": (
        "Development diagnosis over sealed 001 evidence only. It may reject a proposed "
        "global correction but cannot calibrate encoded stereo, authorize metric Contact, "
        "change HaWoR/Depth evidence, or grant control, training or deployment authority."
    ),
}


TASK_SPECS["0915_robot15h_hawor_wave0_v1"] = {
    "phase": "ROBOT15H_HAWOR_WAVE0_RESIZE_ONLY",
    "objective": (
        "Run the pinned HaWoR logical bundle once across the four frozen W0 "
        "development sessions in the physical-left sourceIndex=1 crop+resize-only "
        "domain, preserving missing hands and producing per-session review evidence."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        FIRST_ATTEMPT + "/SOURCE_GROUP_MANIFEST.json",
        "src/chaoyang/ops/run_0915_robot15h_hawor_wave0_v1.py",
        "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py",
        HAWOR_WEIGHT,
        "tasks/receipts/0915_HAWOR_BOUNDED_V2_CANARY_V1_RESULT.json",
    ],
    "write_set": [
        "_run/current/0915_robot15h_hawor_wave0_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_HAWOR_V1",
        "tasks/receipts/0915_ROBOT15H_HAWOR_WAVE0_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_window_start_inventory_v1=PASSED",
        "W0_four_development_sessions_frozen",
        "physical_left_sourceIndex1_crop_resize_only",
        "legacy_prepared_mono_FORBIDDEN",
        "equiDis62_lens_remap_FORBIDDEN",
        "PICO26_tracking_hand_FORBIDDEN",
        "central_GPU_single_lease",
        "missing_hands_remain_invalid",
    ],
    "required_outputs": [
        "CLAIM.json",
        "RUN_SIGNATURE.json",
        "PREPARED_MANIFEST.json",
        "GPU_COMMAND_RECEIPT.json",
        "hawor/BATCH_RESULT.json",
        "REVIEW_MANIFEST.json",
        "METRICS.json",
        "RESULT.json",
        "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 2.5, "runtime_attempts": 1, "wall_clock_hours": 3},
    "expected_resource": "ONE_SERIAL_GPU_LEASE_HAWOR_W0; pinned logical HaWoR bundle",
    "claim_limit": (
        "Four-session development HaWoR evidence in the confirmed resize-only domain. "
        "No anatomical ground truth, Contact, control, deployment, training or "
        "cross-recording generalization authority."
    ),
}


TASK_SPECS["0915_robot15h_hawor_wave0_recovery_v1"] = {
    "phase": "ROBOT15H_HAWOR_WAVE0_RESIZE_ONLY_RECOVERY",
    "objective": (
        "Recover only the activation-free project import defect recorded by the W0 "
        "terminal, without changing T0, the cohort, image domain, model, quality gates, "
        "or single-GPU lease policy."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_hawor_wave0_v1/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_hawor_wave0_recovery_v1.py",
        "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py",
        HAWOR_WEIGHT,
    ],
    "write_set": [
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_HAWOR_RECOVERY_V1",
        "tasks/receipts/0915_ROBOT15H_HAWOR_WAVE0_RECOVERY_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_hawor_wave0_v1=FAILED_RUNTIME_FINAL",
        "recorded_blocker=PINNED_ENV_PROJECT_IMPORT_PATH",
        "same_T0_and_frozen_W0",
        "same_logical_weight_and_quality_gates",
        "physical_left_sourceIndex1_crop_resize_only",
        "central_GPU_single_lease",
    ],
    "required_outputs": [
        "CLAIM.json",
        "RUN_SIGNATURE.json",
        "PREPARED_MANIFEST.json",
        "GPU_COMMAND_RECEIPT.json",
        "hawor/BATCH_RESULT.json",
        "REVIEW_MANIFEST.json",
        "METRICS.json",
        "RESULT.json",
        "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 2.5, "runtime_attempts": 1, "wall_clock_hours": 3},
    "expected_resource": "ONE_SERIAL_GPU_LEASE_HAWOR_W0_RECOVERY; same pinned logical bundle",
    "claim_limit": (
        "Bounded recovery of a recorded Python import defect only; development HaWoR "
        "evidence with no Contact, control, deployment, training or generalization authority."
    ),
}


TASK_SPECS["0915_robot15h_kai22_r0_wave0_v1"] = {
    "phase": "ROBOT15H_KAI22_R0_WAVE0",
    "objective": (
        "Apply the accepted fixed bounded-v2 MANO stabilizer to the four W0 HaWoR "
        "tracks, admit only numeric-pass candidates with raw direct-observed fallback, "
        "and export real pinned-URDF Kai22 q22/wrist-local R0 trajectories and reviews."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/hawor/BATCH_RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_kai22_r0_wave0_v1.py",
        "src/chaoyang/ops/run_hawor_bounded_parameter_successor.py",
        "tasks/receipts/0915_HAWOR_BOUNDED_V2_CANARY_V1_RESULT.json",
    ],
    "write_set": [
        "_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_KAI22_R0_V1",
        "tasks/receipts/0915_ROBOT15H_KAI22_R0_WAVE0_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_hawor_wave0_recovery_v1=PASSED",
        "bounded_v2_parameters_frozen",
        "numeric_pass_candidate_else_raw_direct_observed_fallback",
        "missing_frames_remain_invalid",
        "pinned_Kai22_URDF_and_joint_limits",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "BOUNDED_V2_CONTRACT.json",
        "bounded_v2/RESULT.json", "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 2},
    "expected_resource": "CPU_ONLY_KAI22_R0_W0; weights ABSENT; pinned URDF",
    "claim_limit": (
        "Four-session wrist-local Kai22 R0 development evidence only. Missing frames "
        "remain invalid; no Contact, arm IK, control, training, deployment or measured calibration authority."
    ),
}


TASK_SPECS["0915_robot15h_foundationstereo_wave0_v1"] = {
    "phase": "ROBOT15H_FOUNDATIONSTEREO_WAVE0_ENCODED_DOMAIN",
    "objective": (
        "Run the one pinned FoundationStereo checkpoint once across all four W0 "
        "recordings using physical left/right sourceIndex 1/0 crop+resize-only, the "
        "explicit simultaneous-reflection disparity-sign adapter, and spatial unflip "
        "back to physical-left pixels; publish each session atomically and separately."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "tasks/receipts/0915_ROBOT15H_USER_AUTHORIZATION_V1.json",
        "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json",
        "configs/systems/depth/foundationstereo_0915_encoded_domain_canary_v1.json",
        "_run/current/0915_robot15h_scale_cause_audit_v1/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_foundationstereo_wave0_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_foundationstereo_wave0_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_FOUNDATIONSTEREO_V1",
        "tasks/receipts/0915_ROBOT15H_FOUNDATIONSTEREO_WAVE0_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_window_start_inventory_v1=PASSED",
        "W0_four_development_sessions_frozen",
        "physical_left_right_sourceIndex_1_0",
        "crop_resize_only_no_lens_remap",
        "simultaneous_horizontal_reflection_no_camera_swap",
        "output_unflip_to_physical_left",
        "encoded_K_P_external_accuracy_UNVERIFIED",
        "central_GPU_single_lease",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "GPU_COMMAND_RECEIPT.json",
        "PREFLIGHT_BATCH.json", "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 3, "runtime_attempts": 1, "wall_clock_hours": 4},
    "expected_resource": "ONE_SERIAL_GPU_LEASE_FOUNDATIONSTEREO_W0; one persistent model load",
    "claim_limit": (
        "Four-session encoded-domain development Depth evidence only. Passing sessions "
        "may feed observed visual Object6D candidates; encoded metric accuracy, strict "
        "Contact, control, training and deployment remain unauthorized."
    ),
}


TASK_SPECS["0915_robot15h_sam31_temporal_identity_v1"] = {
    "phase": "ROBOT15H_SAM31_HAND_TEMPORAL_IDENTITY_WAVE0",
    "objective": (
        "Run the one pinned SAM3.1 multiplex checkpoint once across the four frozen "
        "W0 recordings, using only side-specific direct-observed HaWoR hand boxes as "
        "geometric prompts; preserve raw candidates separately from quality-admitted "
        "semantic masks and fail closed for every role without an independent visual anchor."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/PREPARED_MANIFEST.json",
        "src/chaoyang/ops/run_0915_robot15h_sam31_temporal_identity_v1.py",
        "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_sam31_temporal_identity_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_V1",
        "tasks/receipts/0915_ROBOT15H_SAM31_TEMPORAL_IDENTITY_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_window_start_inventory_v1=PASSED",
        "W0_four_development_sessions_frozen",
        "physical_left_sourceIndex1_crop_resize_only_no_lens_remap",
        "direct_observed_HaWoR_side_specific_anchor_only",
        "MULTIPLEX_GEOMETRIC_BOX_not_visual_exemplar",
        "raw_candidate_separate_from_semantic_admission",
        "quality_triggered_max_one_reseed_not_periodic",
        "roles_without_independent_visual_anchor_fail_closed",
        "pico26_controller_trackingData_hand_FORBIDDEN",
        "central_GPU_single_lease",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "GPU_COMMAND_RECEIPT.json",
        "ROLE_BLOCKER_LEDGER.json", "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 1, "runtime_attempts": 1, "wall_clock_hours": 1.5},
    "expected_resource": "ONE_SERIAL_GPU_LEASE_SAM31_W0; one persistent model load",
    "claim_limit": (
        "Four-session development hand temporal-identity evidence only. Raw candidates "
        "are not semantic truth; UNKNOWN is not absence; task-object/forearm/sleeve/cable "
        "masks remain blocked without role-specific anchors. No Object6D mask-consumer, "
        "Contact, control, training or deployment authority."
    ),
}


TASK_SPECS["0915_robot15h_foundationstereo_wave0_recovery_v1"] = {
    "phase": "ROBOT15H_FOUNDATIONSTEREO_WAVE0_BOUNDED_RECOVERY",
    "objective": (
        "Preserve and SHA-verify the three atomically completed predecessor sessions, "
        "rerun only the interrupted get_potato_chips_0915_042 session with one fresh "
        "pinned FoundationStereo model load, and close the exact four-session W0 denominator."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_foundationstereo_wave0_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_foundationstereo_wave0_v1/attempts/attempt_0001/INTERRUPTION_EVIDENCE.json",
        "_run/current/0915_robot15h_foundationstereo_wave0_v1/attempts/attempt_0001/PREFLIGHT_BATCH.json",
        "src/chaoyang/ops/run_0915_robot15h_foundationstereo_wave0_recovery_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_foundationstereo_wave0_recovery_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_FOUNDATIONSTEREO_RECOVERY_V1",
        "tasks/receipts/0915_ROBOT15H_FOUNDATIONSTEREO_WAVE0_RECOVERY_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_foundationstereo_wave0_v1=FAILED_RUNTIME_FINAL",
        "same_T0_and_frozen_W0",
        "three_atomic_predecessor_sessions_SHA_verified",
        "rerun_only_get_potato_chips_0915_042",
        "incomplete_staging_preserved_not_consumed",
        "same_encoded_domain_and_quality_gates",
        "central_GPU_single_lease",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "GPU_COMMAND_RECEIPT.json",
        "RECOVERY_SCOPE.json", "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 1.5, "runtime_attempts": 1, "wall_clock_hours": 2},
    "expected_resource": "ONE_SERIAL_GPU_LEASE_FOUNDATIONSTEREO_W0_RECOVERY; one fresh model load for one session",
    "claim_limit": (
        "Bounded runtime recovery for one interrupted W0 session plus SHA references to "
        "three predecessor atomic sessions. No metric Contact, control, training, deployment "
        "or external encoded-camera accuracy authority."
    ),
}


TASK_SPECS["0915_robot15h_sam31_temporal_identity_recovery_v1"] = deepcopy(
    TASK_SPECS["0915_robot15h_sam31_temporal_identity_v1"]
)
TASK_SPECS["0915_robot15h_sam31_temporal_identity_recovery_v1"].update({
    "phase": "ROBOT15H_SAM31_HAND_TEMPORAL_IDENTITY_WAVE0_RUNTIME_RECOVERY",
    "objective": (
        "Recover the pre-model-import SAM3.1 W0 attempt after adding the missing fixed "
        "vendor import path, with the same W0 cohort, checkpoint, anchors, gates, output "
        "semantics and immutable window clock."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/PREPARED_MANIFEST.json",
        "_run/current/0915_robot15h_sam31_temporal_identity_v1/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_sam31_temporal_identity_v1.py",
        "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_sam31_temporal_identity_recovery_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_RECOVERY_V1",
        "tasks/receipts/0915_ROBOT15H_SAM31_TEMPORAL_IDENTITY_RECOVERY_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_sam31_temporal_identity_v1=FAILED_RUNTIME_FINAL",
        "predecessor_failed_before_model_import",
        "fixed_vendor_SAM3_path_only_no_dependency_install",
        "same_T0_W0_checkpoint_anchors_and_quality_gates",
        "physical_left_sourceIndex1_crop_resize_only_no_lens_remap",
        "roles_without_independent_visual_anchor_fail_closed",
        "central_GPU_single_lease",
    ],
    "expected_resource": "ONE_SERIAL_GPU_LEASE_SAM31_W0_RUNTIME_RECOVERY; one persistent model load",
    "claim_limit": (
        "Bounded pre-model-import recovery with unchanged hand-only semantic authority. "
        "Task-object/forearm/sleeve/cable masks remain blocked without independent anchors; "
        "no Object6D mask-consumer, Contact, control, training or deployment authority."
    ),
})


TASK_SPECS["0915_robot15h_sam31_temporal_identity_quality_v2"] = {
    "phase": "ROBOT15H_SAM31_HAND_TEMPORAL_IDENTITY_WAVE0_QUALITY_V2",
    "objective": (
        "Rerun the same pinned SAM3.1 W0 hand-only experiment after three bounded, "
        "evidence-driven corrections: scale-aware seed admission, direct-observation-aware "
        "temporal QA, and final-target artifact references after atomic publication. Preserve "
        "unadmitted seed candidates as raw evidence and keep every unanchored role fail-closed."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/PREPARED_MANIFEST.json",
        "_run/current/0915_robot15h_sam31_temporal_identity_recovery_v1/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_sam31_temporal_identity_v1.py",
        "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_sam31_temporal_identity_quality_v2/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_QUALITY_V2",
        "tasks/receipts/0915_ROBOT15H_SAM31_TEMPORAL_IDENTITY_QUALITY_V2_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_sam31_temporal_identity_recovery_v1=REJECTED_QUALITY",
        "same_T0_W0_checkpoint_input_domain_and_direct_HaWoR_anchors",
        "seed_area_gate_is_scale_aware_and_bounded",
        "whole_video_true_absence_does_not_override_direct_observation_QA",
        "identity_conflict_remains_rejecting",
        "unadmitted_seed_candidates_remain_raw_nonsemantic_evidence",
        "published_artifact_refs_resolve_after_atomic_rename",
        "roles_without_independent_visual_anchor_fail_closed",
        "central_GPU_single_lease",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "GPU_COMMAND_RECEIPT.json",
        "ROLE_BLOCKER_LEDGER.json", "BATCH_RESULT.json", "PUBLICATION_REF_AUDIT.json",
        "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 1, "runtime_attempts": 1, "wall_clock_hours": 1.5},
    "expected_resource": "ONE_SERIAL_GPU_LEASE_SAM31_W0_QUALITY_V2; one persistent model load",
    "claim_limit": (
        "Bounded hand-only quality/publication correction. Task-object, forearm, sleeve "
        "and cable masks remain blocked without independent anchors; no Object6D mask "
        "consumer, Contact, control, training, deployment or generalization authority."
    ),
}


TASK_SPECS["0915_robot15h_sam31_task_object_wave0_v1"] = {
    "phase": "ROBOT15H_SAM31_TASK_OBJECT_IDENTITY_WAVE0",
    "objective": (
        "Use the pinned SAM3.1 text route on the same physical-left resize-only W0 "
        "videos to initialize separate visible task-object instances, propagate each "
        "identity bidirectionally, and admit only quality-supported card or potato-chip "
        "instances for development Object6D consumption."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_sam31_temporal_identity_quality_v2/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/PREPARED_MANIFEST.json",
        "src/chaoyang/ops/run_0915_robot15h_sam31_task_object_wave0_v1.py",
        "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_sam31_task_object_wave0_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TASK_OBJECT_V1",
        "tasks/receipts/0915_ROBOT15H_SAM31_TASK_OBJECT_WAVE0_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_sam31_temporal_identity_quality_v2=TERMINAL",
        "same_T0_W0_checkpoint_and_physical_left_resize_only_domain",
        "task_specific_short_text_only_playing_card_or_potato_chip",
        "no_manual_coordinate_or_session_specific_box",
        "cards_and_chips_remain_separate_visible_physical_instances",
        "empty_or_exited_instance_is_UNKNOWN_not_absent",
        "support_tray_or_bowl_not_merged_into_task_object",
        "raw_candidate_separate_from_semantic_admission",
        "central_GPU_single_lease",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "GPU_COMMAND_RECEIPT.json",
        "OBJECT_IDENTITY_BATCH.json", "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 1.5, "runtime_attempts": 1, "wall_clock_hours": 2},
    "expected_resource": "ONE_SERIAL_GPU_LEASE_SAM31_W0_TASK_OBJECT; one persistent model load",
    "claim_limit": (
        "Development visual task-object masks only. Text candidates are not ground "
        "truth; only admitted visible instances may feed visual Object6D candidates. "
        "No hidden shape, complete extent, Contact, control, training, deployment or "
        "cross-recording generalization authority."
    ),
}


TASK_SPECS["0915_robot15h_sam31_task_object_wave0_recovery_v1"] = {
    "phase": "ROBOT15H_SAM31_TASK_OBJECT_IDENTITY_WAVE0_EMPTY_CANDIDATE_RECOVERY",
    "objective": (
        "Repeat the frozen W0 SAM3.1 task-object run after correcting only the "
        "zero-candidate archive shape: an empty text result is serialized as raw "
        "UNKNOWN evidence instead of raising reshape(0,-1). All prompts, weights, "
        "identity gates and downstream authority remain unchanged."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_sam31_task_object_wave0_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_sam31_temporal_identity_quality_v2/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/PREPARED_MANIFEST.json",
        "src/chaoyang/ops/run_0915_robot15h_sam31_task_object_wave0_v1.py",
        "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_sam31_task_object_wave0_recovery_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TASK_OBJECT_RECOVERY_V1",
        "tasks/receipts/0915_ROBOT15H_SAM31_TASK_OBJECT_WAVE0_RECOVERY_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_sam31_task_object_wave0_v1=FAILED_RUNTIME_FINAL",
        "recorded_blocker=EMPTY_TEXT_CANDIDATE_RESHAPE_AMBIGUITY",
        "empty_text_candidate_serializes_as_raw_UNKNOWN_not_absent",
        "same_T0_W0_checkpoint_input_domain_prompts_and_quality_gates",
        "no_manual_coordinate_or_session_specific_box",
        "raw_candidate_separate_from_semantic_admission",
        "central_GPU_single_lease",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "GPU_COMMAND_RECEIPT.json",
        "OBJECT_IDENTITY_BATCH.json", "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 1.5, "runtime_attempts": 1, "wall_clock_hours": 2},
    "expected_resource": "ONE_SERIAL_GPU_LEASE_SAM31_W0_TASK_OBJECT_RECOVERY; one persistent model load",
    "claim_limit": (
        "Bounded recovery of the recorded empty-candidate serialization defect only. "
        "Development visible task-object masks remain non-ground-truth; no hidden shape, "
        "complete extent, Contact, control, training, deployment or generalization authority."
    ),
}


TASK_SPECS["0915_robot15h_geometry_object6d_wave0_v1"] = {
    "phase": "ROBOT15H_OBJECT6D_W0_VISIBLE_PATCH",
    "objective": (
        "Join the four frozen W0 Depth and task-object-mask terminals; produce "
        "directly observed finite playing-card surface geometry for independently "
        "admitted Poker sessions, and record the first upstream blocker for every "
        "other session without emitting empty or synthetic geometry."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_foundationstereo_wave0_recovery_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "_run/current/0915_robot15h_sam31_task_object_wave0_recovery_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_object6d_wave0_v1.py",
        "src/chaoyang/pipeline/object6d_planar_observability_v2.py",
        "src/chaoyang/pipeline/object6d_planar_observability_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_geometry_object6d_wave0_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_OBJECT6D_V1",
        "tasks/receipts/0915_ROBOT15H_GEOMETRY_OBJECT6D_WAVE0_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_foundationstereo_wave0_recovery_v1=TERMINAL",
        "0915_robot15h_sam31_task_object_wave0_recovery_v1=TERMINAL",
        "depth_blocker_has_precedence",
        "object_mask_must_be_consumer_admitted",
        "finite_visible_surface_only",
        "visible_centroid_is_not_object_fixed_center",
        "normal_and_pi_periodic_axis_signs_temporally_unified",
        "hidden_extent_and_thickness_remain_UNOBSERVABLE",
        "all_unknown_or_empty_Object6D_FORBIDDEN",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "OBJECT6D_ELIGIBILITY_LEDGER.json",
        "BATCH_RESULT.json", "METRICS.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 0.25},
    "expected_resource": "CPU_ONLY_OBJECT6D_VISIBLE_PATCH_OR_EXPLICIT_BLOCKER; weights ABSENT",
    "claim_limit": (
        "Development directly visible finite-surface geometry only. Visible centroids "
        "are not object-fixed centres; hidden extent, thickness and support objects stay "
        "unknown. No strict metric Contact, control, training, deployment or external "
        "accuracy authority is granted."
    ),
}


def _downstream_blocker_spec(
    *, task_id: str, phase: str, objective: str, dependency_result: str,
    dependency_ledger: str, output_ledger: str, receipt: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "objective": objective,
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            WINDOW_CLOCK,
            FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
            dependency_result,
            dependency_ledger,
            "src/chaoyang/ops/run_0915_robot15h_downstream_terminal_v1.py",
        ],
        "write_set": [
            f"_run/current/{task_id}/attempts/attempt_0001",
            f"tasks/receipts/{receipt}",
        ],
        "prerequisites": [
            *[f"{value}=TERMINAL" for value in DEPENDENCIES[task_id]],
            "upstream_blocker_propagated_per_session",
            "algorithm_artifact_not_fabricated",
            "R0_preserved_and_virtual_R2_independent",
            "weights_ABSENT",
            "gpu_FORBIDDEN",
        ],
        "required_outputs": [
            "CLAIM.json", "RUN_SIGNATURE.json", output_ledger,
            "BATCH_RESULT.json", "METRICS.json", "RESULT.json", "RUN_RECEIPT.json",
        ],
        "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 0.25},
        "expected_resource": "CPU_ONLY_UPSTREAM_BLOCKER_TERMINAL; weights ABSENT",
        "claim_limit": (
            "Per-session development blocker authority only. If upstream evidence is "
            "admitted, this blocker-only runner must refuse and a real algorithm runner "
            "is required. No control, training or deployment authority."
        ),
    }


TASK_SPECS["0915_robot15h_interaction_occlusion_v1"] = {
    "phase": "ROBOT15H_INTERACTION_W0_VISIBLE_SURFACE",
    "objective": (
        "For W0, associate only direct-observed HaWoR 2D finger projections with "
        "same-frame Stereo visible surfaces admitted by independent SAM hand roles, "
        "then measure relations to exact finite Object6D support without publishing Contact."
    ),
    "read_set": [
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/hawor/BATCH_RESULT.json",
        "_run/current/0915_robot15h_foundationstereo_wave0_recovery_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "_run/current/0915_robot15h_sam31_temporal_identity_quality_v2/attempts/attempt_0001/BATCH_RESULT.json",
        "_run/current/0915_robot15h_sam31_task_object_wave0_recovery_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "_run/current/0915_robot15h_geometry_object6d_wave0_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_interaction_wave0_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_interaction_occlusion_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_INTERACTION_V1",
        "tasks/receipts/0915_ROBOT15H_INTERACTION_OCCLUSION_V1_RESULT.json",
    ],
    "prerequisites": [
        *[f"{value}=TERMINAL" for value in DEPENDENCIES["0915_robot15h_interaction_occlusion_v1"]],
        "direct_observed_HaWoR_2D_only",
        "consumer_admitted_SAM_hand_role_required",
        "finger_associated_visible_surface_not_anatomical_fingertip",
        "same_frame_Stereo_visible_surface_only",
        "exact_finite_visible_object_support_only",
        "fragmented_support_largest_component_fraction_at_least_0_98",
        "temporal_gap_resets_relative_motion",
        "registered_valid_depth_fraction_not_a_quality_gate",
        "HaWoR_absolute_Z_48mm_bias_contact_fit_and_Removal_FORBIDDEN",
        "strict_Contact_publication_FORBIDDEN",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "INTERACTION_EVIDENCE_LEDGER.json",
        "BATCH_RESULT.json", "METRICS.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 0.5},
    "expected_resource": "CPU_ONLY_W0_INTERACTION_VISIBLE_SURFACE; weights ABSENT",
    "claim_limit": (
        "Development Interaction diagnosis only. Finger-associated visible surfaces are "
        "not anatomical fingertip truth, object geometry is directly visible finite support "
        "only, and no strict Contact, control, training, deployment or external metric "
        "accuracy authority is granted."
    ),
}

TASK_SPECS["0915_robot15h_contact_dual_evidence_v1"] = {
    "phase": "ROBOT15H_CONTACT_W0_DUAL_EVIDENCE",
    "objective": (
        "Evaluate fixed 5 mm finite-patch Contact without uncertainty expansion or "
        "authority escalation, and separately publish bounded visual-overlap plus "
        "finger-specific tactile hypotheses with explicit no-contact controls."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_interaction_occlusion_v1/attempts/attempt_0001/INTERACTION_EVIDENCE_LEDGER.json",
        "_run/current/0915_robot15h_interaction_occlusion_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "_run/current/0915_robot15h_foundationstereo_wave0_recovery_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "_run/current/0915_robot15h_geometry_object6d_wave0_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_contact_wave0_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_contact_dual_evidence_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_CONTACT_V1",
        "tasks/receipts/0915_ROBOT15H_CONTACT_DUAL_EVIDENCE_V1_RESULT.json",
    ],
    "prerequisites": [
        *[f"{value}=TERMINAL" for value in DEPENDENCIES["0915_robot15h_contact_dual_evidence_v1"]],
        "fixed_5mm_finite_visible_patch_gate",
        "uncertainty_must_not_expand_distance_gate",
        "external_metric_authority_required_for_R1_E",
        "finger_specific_processed_tactile_only",
        "tactile_offline_source_valid_and_abs_offset_at_most_40ms",
        "hypothesis_separate_from_observed_and_strict_Contact",
        "hypothesis_includes_no_contact_control",
        "R1_H_training_eligible_false",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "CONTACT_EVIDENCE_LEDGER.json",
        "CONTACT_HYPOTHESIS_LEDGER.json", "BATCH_RESULT.json", "METRICS.json",
        "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 0.5},
    "expected_resource": "CPU_ONLY_W0_CONTACT_DUAL_EVIDENCE; weights ABSENT",
    "claim_limit": (
        "Development Contact diagnosis and explicitly isolated hypothesis authority only. "
        "Hypotheses are non-causal, non-metric Contact explanations, never probability, "
        "ground truth, force, training data, control or deployment authority."
    ),
}

TASK_SPECS["0915_robot15h_robot_relative_refinement_v1"] = {
    "phase": "ROBOT15H_KAI22_R1_W0_EVIDENCE_AND_HYPOTHESIS",
    "objective": (
        "Keep strict R1-E blocked when no admitted Contact window exists, while "
        "running one bounded R1-H counterfactual against a bit-exact R0/no-contact "
        "control and adopting it only under independent, non-self-certifying gates."
    ),
    "read_set": [
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "_run/current/0915_robot15h_contact_dual_evidence_v1/attempts/attempt_0001/CONTACT_EVIDENCE_LEDGER.json",
        "_run/current/0915_robot15h_contact_dual_evidence_v1/attempts/attempt_0001/CONTACT_HYPOTHESIS_LEDGER.json",
        "_run/current/0915_robot15h_contact_dual_evidence_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "contracts/kai22_r1_hypothesis_v1.schema.json",
        "src/chaoyang/ops/run_0915_robot15h_r1_hypothesis_wave0_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_robot_relative_refinement_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_R1_V1",
        "tasks/receipts/0915_ROBOT15H_ROBOT_RELATIVE_REFINEMENT_V1_RESULT.json",
    ],
    "prerequisites": [
        *[f"{value}=TERMINAL" for value in DEPENDENCIES["0915_robot15h_robot_relative_refinement_v1"]],
        "R1_E_requires_admitted_strict_Contact",
        "R1_H_consumes_only_formal_hypothesis_ledger",
        "R1_H_has_bit_exact_R0_no_contact_control",
        "single_frozen_hand_finger_object_window",
        "non_target_sessions_hands_joints_and_frames_bit_exact",
        "wrist_translation_and_orientation_frozen_without_independent_metric_placement",
        "counterfactual_delta_q_at_most_0_12rad",
        "transition_at_most_0_2s_and_creates_no_Contact_label",
        "optimization_metric_not_adoption_evidence",
        "R1_H_training_eligible_false",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "R1_ELIGIBILITY_LEDGER.json",
        "BATCH_RESULT.json", "METRICS.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 0.75},
    "expected_resource": "CPU_ONLY_W0_KAI22_R1_COUNTERFACTUAL; weights ABSENT",
    "claim_limit": (
        "Development counterfactual Robot trajectory authority only. R1-E remains "
        "blocked without strict Contact; an exported R1-H attempt is not adopted unless "
        "independent gates pass, and never grants training, control, deployment or truth."
    ),
}


TASK_SPECS["0915_robot15h_robot_virtual_arm_v1"] = {
    "phase": "ROBOT15H_ROBOT_VIRTUAL_ARM_WAVE0",
    "objective": (
        "Consume only sealed R0 q22, relative wrist transforms, validity and timestamps "
        "to produce fixed-base Tianji plus KaiHand virtual IK, pinned-digital self-collision "
        "evidence and independent Robot-space review videos."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        "_run/current/0915_robot15h_kai22_r0_wave0_v1/attempts/attempt_0001/BATCH_RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_virtual_arm_wave0_v1.py",
        "assets/robot/ROBOT_ASSET_PIN.json",
        "contracts/robot15h_virtual_installation_r0_motion_v1.schema.json",
        "manifests/hardware/robot15h_virtual_installation_r0_motion_v1.json",
    ],
    "write_set": [
        "_run/current/0915_robot15h_robot_virtual_arm_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W0_VIRTUAL_R2_V1",
        "tasks/receipts/0915_ROBOT15H_ROBOT_VIRTUAL_ARM_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_kai22_r0_wave0_v1=PASSED",
        "dynamic_session_input_is_R0_only",
        "q22_bit_exact_pass_through",
        "relative_wrist_motion_unit_gain_no_scaling",
        "fixed_virtual_base_not_camera_world_calibration",
        "virtual_installation_contract_FROZEN_DEVELOPMENT_ONLY",
        "measured_TCP_mount_and_camera_base_calibration_ABSENT",
        "invalid_frames_remain_NaN",
        "collision_scope_ROBOT_SELF_ONLY_BILATERAL_VALID_FRAMES_PARTIAL",
        "R2_quality_cannot_exceed_R0_quality",
        "no_hardware_or_network_control",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "BATCH_RESULT.json",
        "REVIEW_MANIFEST.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 2},
    "expected_resource": "CPU_ONLY_VIRTUAL_IK_RENDER_AND_SELF_COLLISION; weights ABSENT",
    "claim_limit": (
        "Independent virtual design-frame Robot evidence only. R2 quality admission "
        "cannot exceed upstream R0 admission. Measured TCP/mount/camera-base calibration, "
        "object/environment collision, control, training and deployment remain absent."
    ),
}


TASK_SPECS["0915_robot15h_release_candidate_v1"] = {
    "phase": "ROBOT15H_H9_RELEASE_CANDIDATE",
    "objective": (
        "At the immutable H9 gate, freeze only capability/task/scope rows backed by "
        "SHA-valid terminal evidence, keeping execution, export and quality admission "
        "separate so a structural terminal can never masquerade as algorithm success."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "docs/governance/LONG_HORIZON_TASK_STATE.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/FROZEN_DAG.json",
        "src/chaoyang/ops/run_0915_robot15h_release_control_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_release_candidate_v1/attempts/attempt_0001",
    ],
    "prerequisites": [
        *[f"{value}=TERMINAL" for value in DEPENDENCIES["0915_robot15h_release_candidate_v1"]],
        "immutable_H9_freeze_clock",
        "execution_export_and_quality_admission_separate",
        "scoped_W1_expansion_only",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "CAPABILITY_MATRIX.json",
        "DEADLINE_AUDIT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "window_deadline_bound": True},
    "expected_resource": "CPU_ONLY_H9_RELEASE_CONTROL; weights ABSENT",
    "claim_limit": (
        "Release-candidate accounting and bounded W1 scope selection only. It grants no "
        "new algorithm, Contact, control, training, deployment or generalization authority."
    ),
}


TASK_SPECS["0915_robot15h_hawor_waves_v1"] = {
    "phase": "ROBOT15H_HAWOR_W1_RELEASE_SCOPE_GATE",
    "objective": (
        "Apply the frozen H9 capability matrix to the eight W1 sessions. Because W0 "
        "HaWoR has zero quality-admitted expansion scope, publish explicit per-session "
        "blockers without opening RGB, loading the weight, or claiming W1 execution."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_release_candidate_v1/attempts/attempt_0001/CAPABILITY_MATRIX.json",
        "src/chaoyang/ops/run_0915_robot15h_wave_scope_gate_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_hawor_waves_v1/attempts/attempt_0001",
        "tasks/receipts/0915_ROBOT15H_HAWOR_WAVES_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_release_candidate_v1=PASSED",
        "W1_hawor_quality_admitted_scope=ZERO",
        "W1_source_video_open_count=ZERO",
        "gpu_FORBIDDEN",
        "single_pinned_HaWoR_weight_identity_preserved",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "SCOPE_AUDIT.json",
        "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "window_deadline_bound": True},
    "expected_resource": "CPU_ONLY_RELEASE_SCOPE_GATE; GPU_FORBIDDEN; pinned HaWoR weight not loaded",
    "claim_limit": (
        "W1 HaWoR scope rejection only. It is not model execution, quality success, "
        "a Human prior, Robot input, control, training, deployment or generalization evidence."
    ),
}


TASK_SPECS["0915_robot15h_sam31_waves_v1"] = {
    "phase": "ROBOT15H_SAM31_W1_PROMPT_SOURCE_GATE",
    "objective": (
        "Preserve the frozen direct-anchored SAM3.1 algorithm contract. The W1 hand scope "
        "cannot execute without same-session release-admitted HaWoR prompts, so publish "
        "explicit blockers rather than silently switching to text-only or invented boxes."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_release_candidate_v1/attempts/attempt_0001/CAPABILITY_MATRIX.json",
        "_run/current/0915_robot15h_hawor_waves_v1/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_wave_scope_gate_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_sam31_waves_v1/attempts/attempt_0001",
        "tasks/receipts/0915_ROBOT15H_SAM31_WAVES_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_release_candidate_v1=PASSED",
        "0915_robot15h_hawor_waves_v1=TERMINAL_REJECTED_QUALITY",
        "same_session_release_admitted_HaWoR_prompt_source_REQUIRED",
        "text_only_or_fabricated_box_substitution_FORBIDDEN",
        "W1_source_video_open_count=ZERO",
        "gpu_FORBIDDEN",
        "single_pinned_SAM31_weight_identity_preserved",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "SCOPE_AUDIT.json",
        "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "window_deadline_bound": True},
    "expected_resource": "CPU_ONLY_PROMPT_SOURCE_GATE; GPU_FORBIDDEN; pinned SAM3.1 weight not loaded",
    "claim_limit": (
        "W1 SAM3.1 prompt-source blocker only. The release-admitted W0 hand-proxy scope is "
        "not evidence that unanchored W1 inference is equivalent or quality admitted."
    ),
}


TASK_SPECS["0915_robot15h_foundationstereo_waves_v1"] = {
    "phase": "ROBOT15H_FOUNDATIONSTEREO_W1_ENCODED_DOMAIN",
    "objective": (
        "Run one pinned FoundationStereo model over the eight frozen W1 sessions after "
        "their release time, using crop/resize-only encoded SBS, the explicit simultaneous "
        "horizontal-reflection disparity-sign adapter, and atomic per-session publication."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_release_candidate_v1/attempts/attempt_0001/CAPABILITY_MATRIX.json",
        "tasks/receipts/0915_ROBOT15H_USER_AUTHORIZATION_V1.json",
        "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json",
        "configs/systems/depth/foundationstereo_0915_encoded_domain_canary_v1.json",
        "_run/current/0915_robot15h_scale_cause_audit_v1/attempts/attempt_0001/RESULT.json",
    ],
    "write_set": [
        "_run/current/0915_robot15h_foundationstereo_waves_v1/attempts/attempt_0001",
        "docs/current/visuals/0915_ROBOT15H_W1_FOUNDATIONSTEREO_V1",
        "tasks/receipts/0915_ROBOT15H_FOUNDATIONSTEREO_WAVES_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_robot15h_release_candidate_v1=PASSED",
        "FoundationStereo_cards_and_chips_W1_scope=QUALITY_ADMITTED",
        "all_eight_W1_source_SHA_bound_before_scheduling",
        "final_holdout_source_open_only_at_or_after_H9",
        "new_sessions_start_before_H13_5",
        "physical_left_sourceIndex1_crop_resize_only",
        "simultaneous_horizontal_reflection_then_output_unflip",
        "lens_undistortion_FORBIDDEN",
        "external_metric_accuracy_UNVERIFIED",
        "single_GPU_lease",
        "single_pinned_FoundationStereo_weight",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "W1_INPUT_BINDING.json", "PREFLIGHT_BATCH.json", "COMMAND.json",
        "GPU_COMMAND_RECEIPT.json", "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 4.5, "runtime_attempts": 1, "window_deadline_bound": True},
    "expected_resource": "GPU0_SINGLE_V71_LEASE_SERIAL; one pinned FoundationStereo weight",
    "claim_limit": (
        "W1 encoded-domain development depth for visual/Object6D candidate use only. "
        "External metric accuracy, strict Contact, control, training and physical deployment remain unauthorized."
    ),
}


TASK_SPECS["0915_robot15h_robot_waves_v1"] = {
    "phase": "ROBOT15H_ROBOT_W1_UPSTREAM_GATE",
    "objective": (
        "Close all eight W1 Robot rows honestly after the model waves. FoundationStereo "
        "depth cannot substitute for absent W1 HaWoR/R0, so do not fabricate R0, R1 or R2 trajectories."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        WINDOW_CLOCK,
        FIRST_ATTEMPT + "/BATCH_MANIFEST.json",
        "_run/current/0915_robot15h_release_candidate_v1/attempts/attempt_0001/CAPABILITY_MATRIX.json",
        "_run/current/0915_robot15h_hawor_waves_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_sam31_waves_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/0915_robot15h_foundationstereo_waves_v1/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_robot15h_wave_scope_gate_v1.py",
    ],
    "write_set": [
        "_run/current/0915_robot15h_robot_waves_v1/attempts/attempt_0001",
        "tasks/receipts/0915_ROBOT15H_ROBOT_WAVES_V1_RESULT.json",
    ],
    "prerequisites": [
        *[f"{value}=TERMINAL" for value in DEPENDENCIES["0915_robot15h_robot_waves_v1"]],
        "W1_R0_input_REQUIRED",
        "depth_or_SAM_cannot_substitute_for_HaWoR_R0",
        "no_hardware_or_network_control",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "SCOPE_AUDIT.json",
        "BATCH_RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "window_deadline_bound": True},
    "expected_resource": "CPU_ONLY_UPSTREAM_GATE; GPU_FORBIDDEN; weights ABSENT",
    "claim_limit": (
        "W1 Robot upstream blocker accounting only. It emits no trajectory and grants no "
        "R0/R1/R2 success, control, training, deployment or generalization authority."
    ),
}


TASK_SPECS["0915_robot15h_window_release_audit_v1"] = {
    "phase": "ROBOT15H_H15_FINAL_RELEASE_AUDIT",
    "objective": (
        "After the immutable H14 drain gate, close the window using SHA-bound runtime, "
        "video, test and local-commit evidence, and publish honest executed/exported/"
        "quality-admitted counts without upgrading development artifacts."
    ),
    "read_set": [
        "docs/governance/LONG_HORIZON_TASK_STATE.json",
        WINDOW_CLOCK,
        "src/chaoyang/ops/run_0915_robot15h_release_control_v1.py",
        "src/chaoyang/ops/build_0915_robot15h_final_evidence_v1.py",
        "_run/current/0915_robot15h_v1/final_audit_inputs/RUNTIME_EVIDENCE.json",
        "_run/current/0915_robot15h_v1/final_audit_inputs/VIDEO_EVIDENCE.json",
        "_run/current/0915_robot15h_v1/final_audit_inputs/TEST_EVIDENCE.json",
        "_run/current/0915_robot15h_v1/final_audit_inputs/COMMIT_EVIDENCE.json",
    ],
    "write_set": [
        "_run/current/0915_robot15h_window_release_audit_v1/attempts/attempt_0001",
    ],
    "prerequisites": [
        *[f"{value}=TERMINAL" for value in DEPENDENCIES["0915_robot15h_window_release_audit_v1"]],
        "H14_gpu_and_writer_drain_reached",
        "H15_deadline_not_exceeded",
        "all_owned_tasks_terminal",
        "all_review_videos_full_decode",
        "regression_and_governance_evidence_SHA_bound",
        "local_commit_clean_and_not_pushed",
        "weights_ABSENT",
        "gpu_FORBIDDEN",
    ],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "CAPABILITY_MATRIX.json", "CAMPAIGN_COUNTS.json",
        "DEADLINE_AUDIT.json", "REFERENCE_AUDIT.json", "FINAL_AUDIT.json",
        "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "window_deadline_bound": True},
    "expected_resource": "CPU_ONLY_H15_FINAL_EVIDENCE_AUDIT; weights ABSENT",
    "claim_limit": (
        "Final accounting and evidence-integrity authority only. It does not turn "
        "development-relative evidence into control, training, deployment or physical truth."
    ),
}


def build_packet(task_id: str) -> dict[str, Any]:
    if task_id not in TASK_SPECS:
        raise KeyError(task_id)
    spec = deepcopy(TASK_SPECS[task_id])
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": task_id,
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "window_run_id": WINDOW_RUN_ID,
        "window_clock": WINDOW_CLOCK,
        "phase": spec.pop("phase"),
        "objective": spec.pop("objective"),
        "dag_dependencies": deepcopy(DEPENDENCIES[task_id]),
        "weights": deepcopy(WEIGHTS[task_id]),
        "stop_conditions": [
            "PASSED",
            "REJECTED_QUALITY",
            "FAILED_RUNTIME_FINAL",
            "BLOCKED_RESOURCE",
            "BLOCKED_EXTERNAL",
            "CANCELLED",
            "BUDGET_EXHAUSTED",
        ],
        **spec,
    }
    validate_packet_policy(packet)
    return packet


def validate_packet_policy(packet: dict[str, Any]) -> None:
    task_id = str(packet.get("task_id"))
    weights = packet.get("weights")
    if weights != "ABSENT" and (not isinstance(weights, list) or len(weights) != 1):
        raise ValueError(f"{task_id}: each algorithm node must bind exactly one logical weight")
    if len(packet.get("read_set", [])) > 8:
        raise ValueError(f"{task_id}: read_set exceeds eight files")
    if task_id not in DEPENDENCIES:
        raise ValueError(f"{task_id}: missing frozen DAG dependencies")
    if task_id.startswith("0915_robot15h_sam31") and weights != [SAM31_WEIGHT]:
        raise ValueError(f"{task_id}: SAM node is not pinned to SAM3.1")


def frozen_dag() -> dict[str, Any]:
    return {
        "schema_version": "0915-robot15h-frozen-dag-v1",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "window_run_id": WINDOW_RUN_ID,
        "scheduler_policy": "ONE_CURRENT_EXECUTABLE_NODE_WITH_INTERNAL_ISOLATED_CPU_LANES",
        "gpu_policy": "ONE_SERIAL_LEASE_ONE_LOGICAL_WEIGHT_PER_MODEL_NODE",
        "nodes": [
            {
                "task_id": task_id,
                "dependencies": deepcopy(DEPENDENCIES[task_id]),
                "weights": deepcopy(WEIGHTS[task_id]),
            }
            for task_id in TASK_ORDER
        ],
    }
