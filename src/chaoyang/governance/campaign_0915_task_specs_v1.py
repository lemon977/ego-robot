"""Finite, one-weight task specifications for the 0915 full funnel.

The mappings in this module are inert until a CAS registration command
materialises exactly one packet in ``tasks/current``.  In particular, the
Mask stage is user-locked to SAM3.1 and has no challenger or model-selection
branch.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PLAN_REVISION = "0915_FULL_FUNNEL_0916_CLEAN_V1"
TASK_ORDER = (
    "0915_input_prepare_cad_v1",
    "0915_input_prepare_cad_v2",
    "0915_hawor_full_v1",
    "0915_vst_image_domain_ab_v1",
    "0915_hawor_resize_only_canary_v1",
    "0915_sam31_strict_role_canary_v1",
    "0915_sam31_strict_role_canary_v2",
    "0915_sam31_strict_role_canary_v3",
    "0915_sam31_strict_role_canary_v4",
    "0915_sam31_strict_role_canary_v5",
    "0915_stereo_interaction_cpu_canary_v1",
    "0915_sam31_weak_role_canary_v1",
    "0915_removal_envelope_single_session_canary_v1",
    "0915_foundationstereo_single_session_canary_v1",
    "0915_planar_object6d_single_session_canary_v1",
    "0915_sam31_mask_full_v1",
    "0915_foundationstereo_full_v1",
    "0915_post_geometry_robot_v1",
    "0915_stereo_encoded_domain_preflight_v1",
    "0915_removal_envelope_v2_real_canary_v1",
    "0915_foundationstereo_encoded_domain_canary_v1",
    "0915_planar_object6d_observability_canary_v2",
    "0915_interaction_contact_robot_dev_v1",
    "0915_interaction_contact_robot_dev_v2",
    "0915_interaction_contact_robot_dev_v3",
)

INPUT_ATTEMPT_V1 = "_run/current/0915_input_prepare_cad_v1/attempts/attempt_0001"
INPUT_ATTEMPT = "_run/current/0915_input_prepare_cad_v2/attempts/attempt_0001"
HAWOR_ATTEMPT = "_run/current/0915_hawor_full_v1/attempts/attempt_0001"
VST_AB_ATTEMPT = "_run/current/0915_vst_image_domain_ab_v1/attempts/attempt_0001"
HAWOR_RESIZE_CANARY_ATTEMPT = "_run/current/0915_hawor_resize_only_canary_v1/attempts/attempt_0001"
MASK_STRICT_CANARY_ATTEMPT = "_run/current/0915_sam31_strict_role_canary_v1/attempts/attempt_0001"
MASK_STRICT_CANARY_V2_ATTEMPT = "_run/current/0915_sam31_strict_role_canary_v2/attempts/attempt_0001"
MASK_STRICT_CANARY_V5_ATTEMPT = "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001"
DEPTH_ATTEMPT = "_run/current/0915_foundationstereo_full_v1/attempts/attempt_0001"
MASK_ATTEMPT = "_run/current/0915_sam31_mask_full_v1/attempts/attempt_0001"
CPU_EVIDENCE_ATTEMPT = "_run/current/0915_stereo_interaction_cpu_canary_v1/attempts/attempt_0001"
WEAK_ROLE_ATTEMPT = "_run/current/0915_sam31_weak_role_canary_v1/attempts/attempt_0001"
REMOVAL_ENVELOPE_ATTEMPT = "_run/current/0915_removal_envelope_single_session_canary_v1/attempts/attempt_0001"
DEPTH_CANARY_ATTEMPT = "_run/current/0915_foundationstereo_single_session_canary_v1/attempts/attempt_0001"
OBJECT6D_CANARY_ATTEMPT = "_run/current/0915_planar_object6d_single_session_canary_v1/attempts/attempt_0001"
ENCODED_STEREO_PREFLIGHT_ATTEMPT = "_run/current/0915_stereo_encoded_domain_preflight_v1/attempts/attempt_0001"
REMOVAL_ENVELOPE_V2_ATTEMPT = "_run/current/0915_removal_envelope_v2_real_canary_v1/attempts/attempt_0001"
ENCODED_DEPTH_CANARY_ATTEMPT = "_run/current/0915_foundationstereo_encoded_domain_canary_v1/attempts/attempt_0001"
OBJECT6D_OBSERVABILITY_V2_ATTEMPT = "_run/current/0915_planar_object6d_observability_canary_v2/attempts/attempt_0001"
INTERACTION_CONTACT_ROBOT_DEV_ATTEMPT = "_run/current/0915_interaction_contact_robot_dev_v1/attempts/attempt_0001"
INTERACTION_CONTACT_ROBOT_DEV_V2_ATTEMPT = "_run/current/0915_interaction_contact_robot_dev_v2/attempts/attempt_0001"
INTERACTION_CONTACT_ROBOT_DEV_V3_ATTEMPT = "_run/current/0915_interaction_contact_robot_dev_v3/attempts/attempt_0001"


TASK_SPECS: dict[str, dict[str, Any]] = {
    "0915_input_prepare_cad_v1": {
        "phase": "0915_INPUT_AUDIT_PREPARE_AND_CAD",
        "objective": (
            "Audit the fixed 220-session processed-only cohort with tactile-v2, "
            "prepare physical-left rectified RGB, and audit the candidate KaiHand STEP."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "src/chaoyang/ops/audit_0915_processed_self_containment_v2.py",
            "src/chaoyang/ops/prepare_0915_physical_left_batch_v1.py",
            "src/chaoyang/ops/audit_kaihand_adapter_step_v1.py",
            "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915",
            "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/received_design/KAI_HAND固定件.STEP",
        ],
        "write_set": [INPUT_ATTEMPT_V1],
        "prerequisites": [
            "governance_PASS_FRESH", "no_active_task",
            "0915_processed_root_read_only", "weights_ABSENT",
        ],
        "weights": "ABSENT",
        "required_outputs": [
            "SELF_CONTAINMENT_V2.json", "prepared_physical_left/BATCH_RESULT.json",
            "kaihand_adapter_step_audit/RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
        ],
        "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
        "expected_resource": "LOW_PRIORITY_CPU_IO; weights ABSENT",
        "claim_limit": (
            "Input proof, physical-left visual preparation and candidate STEP structure only; "
            "no model inference or calibration authority."
        ),
    },
    "0915_input_prepare_cad_v2": {
        "phase": "0915_INPUT_AUDIT_PREPARE_AND_CAD_V2",
        "objective": (
            "Run the corrective tactile-v2 processed-only audit, prepare physical-left "
            "rectified RGB, and audit the candidate KaiHand STEP through direct module "
            "entry points after the v1 CLI routing failure."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "_run/current/0915_input_prepare_cad_v1/attempts/attempt_0001/RESULT.json",
            "src/chaoyang/ops/audit_0915_processed_self_containment_v2.py",
            "src/chaoyang/ops/prepare_0915_physical_left_batch_v1.py",
            "src/chaoyang/ops/audit_kaihand_adapter_step_v1.py",
            "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915",
            "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/received_design/KAI_HAND固定件.STEP",
        ],
        "write_set": [INPUT_ATTEMPT],
        "prerequisites": [
            "0915_input_prepare_cad_v1=FAILED_RUNTIME_FINAL_CLI_ROUTE_ONLY",
            "governance_PASS_FRESH", "weights_ABSENT",
            "direct_module_entrypoints", "0915_processed_root_read_only",
        ],
        "weights": "ABSENT",
        "required_outputs": [
            "SELF_CONTAINMENT_V2.json", "prepared_physical_left/BATCH_RESULT.json",
            "kaihand_adapter_step_audit/RESULT.json", "RESULT.json", "RUN_RECEIPT.json",
        ],
        "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
        "expected_resource": "LOW_PRIORITY_CPU_IO; weights ABSENT",
        "claim_limit": (
            "Corrective input proof, physical-left visual preparation and candidate STEP "
            "structure only; no model inference or calibration authority."
        ),
    },
    "0915_hawor_full_v1": {
        "phase": "0915_HAWOR_PHYSICAL_LEFT_FULL",
        "objective": (
            "Run the pinned HaWoR logical inference bundle persistently on every prepared "
            "physical-left session and publish fail-closed per-session terminals."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            f"{INPUT_ATTEMPT}/RESULT.json",
            f"{INPUT_ATTEMPT}/prepared_physical_left/BATCH_RESULT.json",
            "src/chaoyang/ops/run_0915_hawor_persistent_worker_v1.py",
            "assets/models/vendor/hawor/0915_HAWOR_INFERENCE_BUNDLE_V1.json",
        ],
        "write_set": [HAWOR_ATTEMPT],
        "prerequisites": [
            "0915_input_prepare_cad_v2=PASSED", "governance_PASS_FRESH",
            "central_GPU_lease", "physical_left_only", "PICO26_NOT_CONSUMED",
        ],
        "weights": ["assets/models/vendor/hawor/0915_HAWOR_INFERENCE_BUNDLE_V1.json"],
        "required_outputs": ["hawor/BATCH_RESULT.json", "GPU_COMMAND_RECEIPT.json", "RESULT.json"],
        "budgets": {"gpu_hours": 72, "runtime_attempts": 1},
        "expected_resource": "SERIAL_GPU_FULL_BATCH; one logical HaWoR bundle",
        "claim_limit": "Development monocular hand reconstruction; no missing-hand fill or physical 3D truth.",
    },
    "0915_vst_image_domain_ab_v1": {
        "phase": "0915_VST_IMAGE_DOMAIN_SINGLE_SESSION_AB",
        "objective": (
            "Compare one fixed 0915 session in the physical-left sourceIndex=1 "
            "passthrough domain against the held equiDis62-to-pinhole remap and the "
            "legacy processed mono, without running any model."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "tasks/receipts/0915_HAWOR_FULL_V1_USER_STOP.json",
            "src/chaoyang/ops/analyze_0915_vst_image_domain_ab_v1.py",
            "src/chaoyang/ops/prepare_0915_physical_left_batch_v1.py",
            "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915/cleaned/playing_cards/play_cards_0915_001",
            f"{INPUT_ATTEMPT}/prepared_physical_left/sessions/playing_cards/play_cards_0915_001",
        ],
        "write_set": [
            VST_AB_ATTEMPT,
            "docs/current/visuals/0915_VST_IMAGE_DOMAIN_AB_V1",
            "tasks/receipts/0915_VST_IMAGE_DOMAIN_AB_V1_RESULT.json",
        ],
        "prerequisites": [
            "0915_hawor_full_v1=CANCELLED_USER_IMAGE_DOMAIN_HOLD",
            "governance_PASS_FRESH", "weights_ABSENT", "gpu_FORBIDDEN",
            "single_session_play_cards_0915_001", "source_and_processed_read_only",
        ],
        "weights": "ABSENT",
        "required_outputs": [
            "RESULT.json", "IMAGE_DOMAIN_METRICS.json",
            "0915_VST_IMAGE_DOMAIN_AB_CONTACT_SHEET.jpg",
            "0915_VST_IMAGE_DOMAIN_WARP_FIELD.png",
            "0915_VST_IMAGE_DOMAIN_AB_REVIEW.mp4",
            "tasks/receipts/0915_VST_IMAGE_DOMAIN_AB_V1_RESULT.json",
        ],
        "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
        "expected_resource": "BOUNDED_CPU_VISUAL_DIAGNOSTIC; weights ABSENT",
        "claim_limit": (
            "Single-session VST image-domain diagnosis only; no model inference, "
            "camera-calibration promotion, full-batch baseline or source mutation."
        ),
    },
    "0915_hawor_resize_only_canary_v1": {
        "phase": "0915_HAWOR_PHYSICAL_LEFT_RESIZE_ONLY_CANARY",
        "objective": (
            "Run pinned HaWoR on only play_cards_0915_001 using the user-confirmed "
            "physical-left sourceIndex=1 passthrough pixels with resize only."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "tasks/receipts/0915_VST_IMAGE_DOMAIN_AB_V1_RESULT.json",
            "tasks/receipts/0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json",
            "src/chaoyang/ops/run_0915_hawor_resize_only_canary_v1.py",
            "src/chaoyang/ops/run_play_cards_0910_001_hawor_raw.py",
            "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915/cleaned/playing_cards/play_cards_0915_001",
            "assets/models/vendor/hawor/0915_HAWOR_INFERENCE_BUNDLE_V1.json",
        ],
        "write_set": [
            HAWOR_RESIZE_CANARY_ATTEMPT,
            "docs/current/visuals/0915_HAWOR_RESIZE_ONLY_CANARY_V1",
            "tasks/receipts/0915_HAWOR_RESIZE_ONLY_CANARY_V1_RESULT.json",
        ],
        "prerequisites": [
            "0915_vst_image_domain_ab_v1=BLOCKED_EXTERNAL_REVIEW_COMPLETE",
            "user_confirmed_A_physical_left_sourceIndex1_resize_only",
            "governance_PASS_FRESH", "central_GPU_lease",
            "single_session_play_cards_0915_001", "PICO26_NOT_CONSUMED",
        ],
        "weights": ["assets/models/vendor/hawor/0915_HAWOR_INFERENCE_BUNDLE_V1.json"],
        "required_outputs": [
            "INPUT_DOMAIN.json", "hawor/HAWOR_RAW_MANO21.npz",
            "HAWOR_METRICS.json", "GPU_COMMAND_RECEIPT.json", "RESULT.json",
            "0915_HAWOR_RESIZE_ONLY_REVIEW.mp4",
            "0915_HAWOR_RESIZE_ONLY_CONTACT_SHEET.jpg",
            "tasks/receipts/0915_HAWOR_RESIZE_ONLY_CANARY_V1_RESULT.json",
        ],
        "budgets": {"gpu_hours": 1, "runtime_attempts": 1},
        "expected_resource": "SERIAL_GPU_CANARY; one logical HaWoR bundle",
        "claim_limit": (
            "One-session development HaWoR evidence on the confirmed monocular pixel "
            "domain; no SAM, Depth, batch, physical-3D or deployment authority."
        ),
    },
    "0915_sam31_strict_role_canary_v1": {
        "phase": "0915_SAM31_STRICT_ROLE_SINGLE_SESSION_CANARY",
        "objective": (
            "Run pinned SAM3.1 on only play_cards_0915_001 in the confirmed resize-only "
            "pixel domain, with separate hand/forearm/sleeve/cable/object roles, initial "
            "visual boxes, quality-triggered finite reseeding, and explicit temporal states."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "tasks/receipts/0915_HAWOR_RESIZE_ONLY_CANARY_V1_RESULT.json",
            "src/chaoyang/ops/run_0915_sam31_strict_role_canary_v1.py",
            "src/chaoyang/pipeline/sam31_0915_strict_role_contract_v1.py",
            "contracts/visual_role_mask_v2.schema.json",
            HAWOR_RESIZE_CANARY_ATTEMPT,
            "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
        ],
        "write_set": [
            MASK_STRICT_CANARY_ATTEMPT,
            "docs/current/visuals/0915_SAM31_STRICT_ROLE_CANARY_V1",
            "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V1_RESULT.json",
        ],
        "prerequisites": [
            "0915_hawor_resize_only_canary_v1=PASSED",
            "user_accepted_resize_only_bounded_hawor_canary",
            "governance_PASS_FRESH", "central_GPU_lease",
            "single_session_play_cards_0915_001", "SAM3.1_ONLY_USER_LOCKED",
            "PICO26_NOT_CONSUMED", "no_batch_expansion",
        ],
        "weights": ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"],
        "required_outputs": [
            "PROMPT_PLAN.json", "ROLE_MANIFEST.json", "TEMPORAL_STATE_LEDGER.json",
            "QUALITY_TRIGGER_LEDGER.json", "RESULT.json", "GPU_COMMAND_RECEIPT.json",
            "0915_SAM31_STRICT_ROLE_REVIEW.mp4",
            "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V1_RESULT.json",
        ],
        "budgets": {"gpu_hours": 2, "runtime_attempts": 1},
        "expected_resource": "SERIAL_GPU_SINGLE_SESSION_CANARY; exactly one SAM3.1 checkpoint",
        "claim_limit": (
            "One-session development masks for human review. Empty masks remain unknown "
            "unless absence is evidenced; no alternate model, batch authority, tracker/controller "
            "instance, ground truth, contact truth, hidden-shape claim or deployment authority."
        ),
    },
    "0915_sam31_strict_role_canary_v2": {
        "phase": "0915_SAM31_STRICT_ROLE_SINGLE_SESSION_CANARY_V2",
        "objective": (
            "Run pinned SAM3.1 on only play_cards_0915_001 in the confirmed resize-only "
            "pixel domain, using the accepted bounded HaWoR successor, separate role "
            "instances, quality-triggered finite reseeding, and explicit temporal states."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            f"{MASK_STRICT_CANARY_ATTEMPT}/RESULT.json",
            "src/chaoyang/ops/run_0915_sam31_strict_role_canary_v1.py",
            "src/chaoyang/pipeline/sam31_0915_strict_role_contract_v1.py",
            "contracts/visual_role_mask_v2.schema.json",
            f"{HAWOR_RESIZE_CANARY_ATTEMPT}/input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
            "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
            "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
        ],
        "write_set": [
            MASK_STRICT_CANARY_V2_ATTEMPT,
            "docs/current/visuals/0915_SAM31_STRICT_ROLE_CANARY_V1",
            "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V1_RESULT.json",
        ],
        "prerequisites": [
            "0915_sam31_strict_role_canary_v1=CANCELLED_READ_SET_CORRECTION",
            "0915_hawor_resize_only_canary_v1=PASSED",
            "user_accepted_resize_only_bounded_hawor_canary",
            "governance_PASS_FRESH", "central_GPU_lease",
            "single_session_play_cards_0915_001", "SAM3.1_ONLY_USER_LOCKED",
            "PICO26_NOT_CONSUMED", "no_batch_expansion",
        ],
        "weights": ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"],
        "required_outputs": [
            "PROMPT_PLAN.json", "ROLE_MANIFEST.json", "TEMPORAL_STATE_LEDGER.json",
            "QUALITY_TRIGGER_LEDGER.json", "RESULT.json", "GPU_COMMAND_RECEIPT.json",
            "0915_SAM31_STRICT_ROLE_REVIEW.mp4",
            "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V1_RESULT.json",
        ],
        "budgets": {"gpu_hours": 2, "runtime_attempts": 1},
        "expected_resource": "SERIAL_GPU_SINGLE_SESSION_CANARY; exactly one SAM3.1 checkpoint",
        "claim_limit": (
            "One-session development masks for human review. Empty masks remain unknown "
            "unless absence is evidenced; no alternate model, batch authority, tracker/controller "
            "instance, ground truth, contact truth, hidden-shape claim or deployment authority."
        ),
    },
    "0915_foundationstereo_full_v1": {
        "phase": "0915_FOUNDATIONSTEREO_PHYSICAL_LEFT_FULL",
        "objective": (
            "Run the pinned FoundationStereo model persistently with same-session rectification "
            "and publish internally consistent optical-Z terminals for all sessions."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            f"{INPUT_ATTEMPT}/RESULT.json",
            f"{INPUT_ATTEMPT}/prepared_physical_left/BATCH_RESULT.json",
            "src/chaoyang/ops/run_0915_foundationstereo_persistent_worker_v2.py",
            "src/chaoyang/ops/foundationstereo_gpu_python.sh",
            "assets/models/checkpoints/foundationstereo/23-51-11/ASSET_PIN.json",
            "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth",
        ],
        "write_set": [DEPTH_ATTEMPT],
        "prerequisites": [
            "0915_input_prepare_cad_v2=PASSED", "governance_PASS_FRESH",
            "central_GPU_lease", "same_session_calibration_only",
        ],
        "weights": ["assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"],
        "required_outputs": ["depth/BATCH_RESULT.json", "GPU_COMMAND_RECEIPT.json", "RESULT.json"],
        "budgets": {"gpu_hours": 72, "runtime_attempts": 1},
        "expected_resource": "SERIAL_GPU_FULL_BATCH; one FoundationStereo checkpoint",
        "claim_limit": "Internal optical-Z consistency only; no external millimetre-accuracy claim.",
    },
    "0915_sam31_mask_full_v1": {
        "phase": "0915_SAM31_ONLY_MASK_FULL",
        "objective": (
            "Run only the pinned SAM3.1 model on HaWoR-PASS physical-left sessions with "
            "dynamic role/object prompts and fail-closed identity/occlusion gates."
        ),
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            f"{INPUT_ATTEMPT}/prepared_physical_left/BATCH_RESULT.json",
            f"{HAWOR_ATTEMPT}/hawor/BATCH_RESULT.json",
            "src/chaoyang/ops/run_0915_sam31_persistent_masks_v2.py",
            "src/chaoyang/pipeline/sam31_0915_prompt_contract_v2.py",
            "contracts/visual_role_mask_v1.schema.json",
            "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
        ],
        "write_set": [MASK_ATTEMPT],
        "prerequisites": [
            "0915_hawor_full_v1=PASSED", "governance_PASS_FRESH",
            "central_GPU_lease", "SAM3.1_ONLY_USER_LOCKED",
            "no_tracker_or_controller_instance",
        ],
        "weights": ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"],
        "required_outputs": ["sam31/BATCH_RESULT.json", "GPU_COMMAND_RECEIPT.json", "RESULT.json"],
        "budgets": {"gpu_hours": 72, "runtime_attempts": 1},
        "expected_resource": "SERIAL_GPU_FULL_BATCH; exactly one SAM3.1 checkpoint",
        "claim_limit": (
            "SAM3.1-only development Mask. No alternate model, challenger, winner selection, "
            "tracker/controller instance, or hidden-shape claim."
        ),
    },
    "0915_post_geometry_robot_v1": {
        "phase": "0915_OBJECT6D_CLEAN_CONTACT_ROBOT_LEDGER",
        "objective": (
            "Join terminal HaWoR, SAM3.1 and Depth evidence into observed-only Object6D, "
            "evidence-preserving Clean, bounded Contact hypotheses, Robot visual sidecars and "
            "the fixed 220-session full-funnel ledger."
        ),
        "read_set": [
            f"{INPUT_ATTEMPT}/prepared_physical_left/BATCH_RESULT.json",
            f"{HAWOR_ATTEMPT}/hawor/BATCH_RESULT.json",
            f"{DEPTH_ATTEMPT}/depth/BATCH_RESULT.json",
            f"{MASK_ATTEMPT}/sam31/BATCH_RESULT.json",
            "src/chaoyang/ops/run_0915_post_geometry_robot_v1.py",
            "src/chaoyang/pipeline/object6d_visible_surface_v1.py",
            "src/chaoyang/pipeline/robot_visual_relative_v1.py",
            "contracts/robot_visual_sidecar_v1.schema.json",
        ],
        "write_set": ["_run/current/0915_post_geometry_robot_v1/attempts/attempt_0001"],
        "prerequisites": [
            "0915_hawor_full_v1=PASSED", "0915_foundationstereo_full_v1=PASSED",
            "0915_sam31_mask_full_v1=PASSED", "governance_PASS_FRESH",
            "calibration_missing_stays_ABSENT",
        ],
        "weights": "ABSENT",
        "required_outputs": [
            "FULL_FUNNEL_LEDGER.json", "FUNNEL_SUMMARY.json", "RESULT.json", "RUN_RECEIPT.json",
        ],
        "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
        "expected_resource": "CPU_POSTPROCESS; weights ABSENT",
        "claim_limit": (
            "Development observed-only geometry, visual synthesis, tactile-supported hypotheses "
            "and Robot visualization; no physical deployment or control ground truth."
        ),
    },
}

# Bounded post-v5 evidence chain.  Routing is deliberately serial because the
# current governance contract exposes exactly one executable ``next_task``.
# ``algorithm_prerequisites`` records the real evidence dependencies so the
# routing predecessor is never misrepresented as an algorithm input.
TASK_SPECS["0915_stereo_interaction_cpu_canary_v1"] = {
    "phase": "0915_STEREO_INTERACTION_CPU_CANARY",
    "objective": (
        "Run two fenced CPU lanes for the fixed play_cards_0915_001 session: "
        "a raw-resize versus calibrated-rectified stereo-domain preflight and "
        "image-plane-only Interaction Evidence v0a."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "tasks/receipts/0915_PARALLEL_CANARIES_USER_AUTHORIZATION.json",
        "src/chaoyang/ops/run_0915_stereo_interaction_cpu_canary_v1.py",
        "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915/cleaned/playing_cards/play_cards_0915_001",
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001/ROLE_MANIFEST.json",
        "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001/TEMPORAL_STATE_LEDGER.json",
        "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001/masks",
    ],
    "write_set": [
        CPU_EVIDENCE_ATTEMPT,
        "docs/current/visuals/0915_STEREO_INTERACTION_CPU_CANARY_V1",
    ],
    "prerequisites": [
        "0915_sam31_strict_role_canary_v5=PASSED",
        "user_authorized_bounded_parallel_canaries",
        "governance_PASS_FRESH", "weights_ABSENT", "gpu_FORBIDDEN",
        "two_disjoint_lane_writer_fences", "source_read_only",
    ],
    "algorithm_prerequisites": {
        "stereo_preflight": ["same_session_SBS", "same_session_camera_params"],
        "interaction_v0a": [
            "accepted_bounded_HaWoR", "SAM3.1_v5_role_masks_and_states",
            "processed_entities_tactile_only",
        ],
    },
    "weights": "ABSENT",
    "required_outputs": [
        "lanes/stereo_preflight/STEREO_PREFLIGHT.json",
        "lanes/interaction_v0a/INTERACTION_EVIDENCE_V0A.json",
        "lanes/interaction_v0a/INTERACTION_V0A_TIMELINE.png",
        "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
    "expected_resource": "TWO_FENCED_CPU_LANES_ONE_GOVERNANCE_COORDINATOR",
    "claim_limit": (
        "CPU single-session evidence only. The outer task may pass while a later "
        "Depth task remains blocked by the inner stereo admission. Interaction v0a "
        "contains no Z, occlusion-order, contact-truth, Object6D or Robot authority."
    ),
}

TASK_SPECS["0915_sam31_weak_role_canary_v1"] = {
    "phase": "0915_SAM31_WEAK_ROLE_SINGLE_SESSION_CANARY",
    "objective": (
        "Run the pinned SAM3.1 checkpoint on weak roles only: forearms, visible "
        "per-finger sleeves, visible yellow cables and playing_card_02; retain the "
        "v5 hands and card00/card01 as read-only regression evidence."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        f"{CPU_EVIDENCE_ATTEMPT}/RESULT.json",
        "src/chaoyang/ops/run_0915_sam31_weak_role_canary_v1.py",
        "src/chaoyang/pipeline/sam31_0915_weak_role_contract_v1.py",
        f"{HAWOR_RESIZE_CANARY_ATTEMPT}/input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001",
        "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
    ],
    "write_set": [
        WEAK_ROLE_ATTEMPT,
        "docs/current/visuals/0915_SAM31_WEAK_ROLE_CANARY_V1",
        "tasks/receipts/0915_SAM31_WEAK_ROLE_CANARY_V1_RESULT.json",
    ],
    "prerequisites": [
        "routing_predecessor_0915_stereo_interaction_cpu_canary_v1=PASSED",
        "algorithm_independent_of_stereo_and_interaction_outputs",
        "governance_PASS_FRESH", "central_GPU_lease",
        "SAM3.1_ONLY_USER_LOCKED", "single_session_play_cards_0915_001",
        "quality_triggered_reseed_only", "no_batch_expansion",
    ],
    "algorithm_prerequisites": {
        "mask": ["accepted_bounded_HaWoR", "resize_only_RGB", "v5_regression_evidence"],
        "not_dependencies": ["stereo_preflight", "interaction_v0a"],
    },
    "weights": ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"],
    "required_outputs": [
        "PROMPT_PLAN.json", "WEAK_ROLE_MANIFEST.json", "TEMPORAL_STATE_LEDGER.json",
        "QUALITY_TRIGGER_LEDGER.json", "REGRESSION_GATES.json",
        "RESULT.json", "GPU_COMMAND_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 2, "runtime_attempts": 1},
    "expected_resource": "SERIAL_GPU_SINGLE_SESSION_CANARY_ONE_SAM3.1_WEIGHT",
    "claim_limit": (
        "Weak-role development masks only. Initial boxes are prompts, unknown is not "
        "absence, multiplex is an efficiency route rather than identity proof, and no "
        "batch, contact, hidden-shape or deployment authority is granted."
    ),
}

TASK_SPECS["0915_removal_envelope_single_session_canary_v1"] = {
    "phase": "0915_REMOVAL_ENVELOPE_SINGLE_SESSION_CANARY_V1",
    "objective": (
        "Build an auditable visual-only Clean removal envelope for play_cards_0915_001 "
        "from sealed admitted SAM evidence, bounded MANO geometry and a replaceable "
        "device-instance cable appearance profile, without rerunning SAM or inpainting."
    ),
    "read_set": [
        "src/chaoyang/ops/run_0915_removal_envelope_single_session_canary_v1.py",
        "src/chaoyang/pipeline/removal_envelope_v1.py",
        "contracts/removal_envelope_v1.schema.json",
        "configs/systems/clean/removal_envelope_0915_play_cards_001_v1.json",
        f"{HAWOR_RESIZE_CANARY_ATTEMPT}/input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001",
        WEAK_ROLE_ATTEMPT,
    ],
    "write_set": [
        REMOVAL_ENVELOPE_ATTEMPT,
        "docs/current/visuals/0915_REMOVAL_ENVELOPE_CANARY_V1",
        "tasks/receipts/0915_REMOVAL_ENVELOPE_SINGLE_SESSION_CANARY_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_sam31_weak_role_canary_v1=PASSED_EXECUTION_ONLY",
        "user_rejected_weak_role_as_clean_baseline",
        "user_authorized_independent_removal_envelope_v1",
        "single_session_play_cards_0915_001", "governance_PASS_FRESH",
        "weights_ABSENT", "gpu_FORBIDDEN", "no_SAM_rerun", "no_inpaint",
    ],
    "algorithm_prerequisites": {
        "removal": [
            "sealed_admitted_SAM_masks", "accepted_bounded_HaWoR",
            "resize_only_RGB", "replaceable_cable_appearance_profile",
        ],
        "forbidden_consumers": [
            "Depth", "Object6D", "Contact", "RobotGeometry", "ControlGroundTruth",
        ],
    },
    "weights": "ABSENT",
    "required_outputs": [
        "raw_candidate/STATUS.json", "semantic/SEALED_INPUTS.json",
        "removal/MANO_FINGER_CAPSULES.npz",
        "removal/PALM_WRIST_FOREARM_CORRIDORS.npz",
        "removal/CABLE_TRACKED_REGION.npz",
        "removal/PROTECTED_VISIBLE_OBJECT.npz",
        "removal/REMOVAL_ENVELOPE.npz",
        "removal/SOURCE_BITS_INVENTORY.json",
        "clean/FEATHER_ALPHA_INVENTORY.json", "clean/INPAINT_NOT_RUN.json",
        "FRAME_PROVENANCE_LEDGER.json", "REMOVAL_ENVELOPE_MANIFEST.json",
        "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
    "expected_resource": "CPU_ONLY_SINGLE_SESSION_VISUAL_CLEAN_MASK_CANARY",
    "claim_limit": (
        "Visual Clean removal inference only. Expanded or held pixels and feather alpha "
        "are not semantic truth, geometry observation, Contact evidence, Robot geometry, "
        "control truth or physical-deployment authority; no inpaint or batch is authorized."
    ),
}

TASK_SPECS["0915_foundationstereo_single_session_canary_v1"] = {
    "phase": "0915_FOUNDATIONSTEREO_SINGLE_SESSION_METRIC_CANARY",
    "objective": (
        "Historical wrong-image-domain task definition retained for immutable ledger "
        "closure only; registration is forbidden and a fresh encoded-video-domain "
        "successor must use zero lens-undistortion."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        f"{CPU_EVIDENCE_ATTEMPT}/lanes/stereo_preflight/STEREO_PREFLIGHT.json",
        "src/chaoyang/ops/run_0915_foundationstereo_single_session_canary_v1.py",
        "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/playing_cards/play_cards_0915_001",
        "tasks/receipts/0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json",
        "assets/models/checkpoints/foundationstereo/23-51-11",
        "configs/systems/depth/foundationstereo_0915_canary_v1.json",
        "tasks/receipts/FOUNDATIONSTEREO_RUNTIME_CLOSURE_V1.json",
    ],
    "write_set": [
        DEPTH_CANARY_ATTEMPT,
        "docs/current/visuals/0915_FOUNDATIONSTEREO_CANARY_V1",
    ],
    "prerequisites": [
        "HISTORICAL_TERMINAL_DO_NOT_REGISTER",
        "VST_ENCODED_VIDEO_ALREADY_UNDISTORTED",
        "EQUIDIS62_LENS_UNDISTORTION_FORBIDDEN",
        "FRESH_SUCCESSOR_TASK_REQUIRED",
    ],
    "algorithm_prerequisites": {
        "depth": ["stereo_preflight_PASS_GPU_DEPTH_ADMISSION"],
        "not_dependencies": ["Removal_Envelope", "SAM3.1_weak_role_result", "interaction_v0a"],
    },
    "weights": ["assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "CALIBRATION.json",
        "REGISTRATION_MAPS.npz", "REGISTRATION.json", "DEPTH_CONTRACT.json",
        "DEPTH_SUMMARY.json", "DEPTH_WORKER_RESULT.json", "RESULT.json",
        "GPU_COMMAND_RECEIPT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 2, "runtime_attempts": 1},
    "expected_resource": "SERIAL_GPU_SINGLE_SESSION_FOUNDATIONSTEREO",
    "claim_limit": (
        "WITHDRAWN_WRONG_IMAGE_DOMAIN. The immutable historical run used redundant "
        "equiDis62 lens-undistortion and grants no Depth or Object6D authority."
    ),
}

TASK_SPECS["0915_planar_object6d_single_session_canary_v1"] = {
    "phase": "0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY",
    "objective": (
        "Estimate observed-only planar geometry for playing_card_00 and playing_card_01 "
        "from admitted SAM masks and stereo depth, with independent observability fields."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        f"{DEPTH_CANARY_ATTEMPT}/RESULT.json",
        f"{DEPTH_CANARY_ATTEMPT}/DEPTH_CONTRACT.json",
        "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001",
        "src/chaoyang/ops/run_0915_planar_object6d_single_session_canary_v1.py",
        "src/chaoyang/governance/register_0915_campaign_task_v1.py",
        "src/chaoyang/pipeline/object6d_planar_observability_v1.py",
        "contracts/object6d_planar_observability_v1.schema.json",
    ],
    "write_set": [
        OBJECT6D_CANARY_ATTEMPT,
        "docs/current/visuals/0915_PLANAR_OBJECT6D_CANARY_V1",
    ],
    "prerequisites": [
        "0915_foundationstereo_single_session_canary_v1=PASSED",
        "depth_admission=PASS", "card00_card01_v5_masks_only",
        "governance_PASS_FRESH", "weights_ABSENT",
    ],
    "algorithm_prerequisites": {
        "object6d": ["admitted_rectified_depth", "card00_card01_valid_masks"],
        "excluded_until_successor": ["playing_card_02"],
    },
    "weights": "ABSENT",
    "required_outputs": [
        "OBJECT6D_OBSERVABILITY.json", "OBJECT6D_SUMMARY.json",
        "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
    "expected_resource": "CPU_OBSERVED_ONLY_PLANAR_GEOMETRY",
    "claim_limit": (
        "Visible planar geometry only. Translation, plane normal and in-plane rotation "
        "observability are independent; hidden pose, unified confidence, contact truth "
        "and Robot authority are forbidden."
    ),
}

# Corrective successor for the v2 pre-inference import-path failure.  It keeps
# the algorithm, inputs and gates byte-for-byte scoped to the same canary; only
# the entry-point environment closure is corrected.
TASK_SPECS["0915_sam31_strict_role_canary_v3"] = deepcopy(
    TASK_SPECS["0915_sam31_strict_role_canary_v2"]
)
TASK_SPECS["0915_sam31_strict_role_canary_v3"].update({
    "phase": "0915_SAM31_STRICT_ROLE_SINGLE_SESSION_CANARY_V3",
    "objective": (
        "Correct the repository-local vendor/SAM3 import closure and run the same "
        "pinned, single-session strict-role SAM3.1 canary specified by v2."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "_run/current/0915_sam31_strict_role_canary_v2/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_sam31_strict_role_canary_v1.py",
        "src/chaoyang/pipeline/sam31_0915_strict_role_contract_v1.py",
        "contracts/visual_role_mask_v2.schema.json",
        f"{HAWOR_RESIZE_CANARY_ATTEMPT}/input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
    ],
    "write_set": [
        "_run/current/0915_sam31_strict_role_canary_v3/attempts/attempt_0001",
        "docs/current/visuals/0915_SAM31_STRICT_ROLE_CANARY_V1",
        "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V2_RESULT.json",
    ],
    "prerequisites": [
        "0915_sam31_strict_role_canary_v2=FAILED_RUNTIME_FINAL_PRE_MODEL_IMPORT_PATH",
        "0915_hawor_resize_only_canary_v1=PASSED",
        "user_accepted_resize_only_bounded_hawor_canary",
        "governance_PASS_FRESH", "central_GPU_lease",
        "repository_local_vendor_SAM3", "single_session_play_cards_0915_001",
        "SAM3.1_ONLY_USER_LOCKED", "PICO26_NOT_CONSUMED", "no_batch_expansion",
    ],
    "required_outputs": [
        "PROMPT_PLAN.json", "ROLE_MANIFEST.json", "TEMPORAL_STATE_LEDGER.json",
        "QUALITY_TRIGGER_LEDGER.json", "RESULT.json", "GPU_COMMAND_RECEIPT.json",
        "0915_SAM31_STRICT_ROLE_REVIEW.mp4",
        "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V2_RESULT.json",
    ],
})

TASK_SPECS["0915_sam31_strict_role_canary_v4"] = deepcopy(
    TASK_SPECS["0915_sam31_strict_role_canary_v3"]
)
TASK_SPECS["0915_sam31_strict_role_canary_v4"].update({
    "phase": "0915_SAM31_STRICT_ROLE_SINGLE_SESSION_CANARY_V4",
    "objective": (
        "Run the same strict-role canary through the pinned multiplex full semantic "
        "propagation route, using points for seed selection and quality evidence after "
        "the v3 point-refinement partial route failed temporal continuity."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "_run/current/0915_sam31_strict_role_canary_v3/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_sam31_strict_role_canary_v1.py",
        "src/chaoyang/pipeline/sam31_0915_strict_role_contract_v1.py",
        "contracts/visual_role_mask_v2.schema.json",
        f"{HAWOR_RESIZE_CANARY_ATTEMPT}/input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
    ],
    "write_set": [
        "_run/current/0915_sam31_strict_role_canary_v4/attempts/attempt_0001",
        "docs/current/visuals/0915_SAM31_STRICT_ROLE_CANARY_V1",
        "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V3_RESULT.json",
    ],
    "prerequisites": [
        "0915_sam31_strict_role_canary_v3=CANCELLED_EARLY_POINT_PARTIAL_ROUTE_DIAGNOSTIC",
        "0915_hawor_resize_only_canary_v1=PASSED",
        "user_accepted_resize_only_bounded_hawor_canary",
        "governance_PASS_FRESH", "central_GPU_lease",
        "multiplex_semantic_full_propagation", "points_selection_and_quality_only",
        "single_session_play_cards_0915_001", "SAM3.1_ONLY_USER_LOCKED",
        "PICO26_NOT_CONSUMED", "no_batch_expansion",
    ],
    "required_outputs": [
        "PROMPT_PLAN.json", "ROLE_MANIFEST.json", "TEMPORAL_STATE_LEDGER.json",
        "QUALITY_TRIGGER_LEDGER.json", "RESULT.json", "GPU_COMMAND_RECEIPT.json",
        "0915_SAM31_STRICT_ROLE_REVIEW.mp4",
        "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V3_RESULT.json",
    ],
})

TASK_SPECS["0915_sam31_strict_role_canary_v5"] = deepcopy(
    TASK_SPECS["0915_sam31_strict_role_canary_v4"]
)
TASK_SPECS["0915_sam31_strict_role_canary_v5"].update({
    "phase": "0915_SAM31_STRICT_ROLE_SINGLE_SESSION_CANARY_V5",
    "objective": (
        "Continue the same strict-role canary while converting the pinned tracker's "
        "no-points reverse-direction exception into explicit direction-level unknown "
        "evidence instead of aborting the session."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "_run/current/0915_sam31_strict_role_canary_v4/attempts/attempt_0001/RESULT.json",
        "src/chaoyang/ops/run_0915_sam31_strict_role_canary_v1.py",
        "src/chaoyang/pipeline/sam31_0915_strict_role_contract_v1.py",
        "contracts/visual_role_mask_v2.schema.json",
        f"{HAWOR_RESIZE_CANARY_ATTEMPT}/input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4",
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
    ],
    "write_set": [
        "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001",
        "docs/current/visuals/0915_SAM31_STRICT_ROLE_CANARY_V1",
        "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V4_RESULT.json",
    ],
    "prerequisites": [
        "0915_sam31_strict_role_canary_v4=FAILED_RUNTIME_FINAL_DIRECTION_EXCEPTION",
        "0915_hawor_resize_only_canary_v1=PASSED",
        "user_accepted_resize_only_bounded_hawor_canary",
        "governance_PASS_FRESH", "central_GPU_lease",
        "direction_level_fail_closed_unknown", "multiplex_semantic_full_propagation",
        "single_session_play_cards_0915_001", "SAM3.1_ONLY_USER_LOCKED",
        "PICO26_NOT_CONSUMED", "no_batch_expansion",
    ],
    "required_outputs": [
        "PROMPT_PLAN.json", "ROLE_MANIFEST.json", "TEMPORAL_STATE_LEDGER.json",
        "QUALITY_TRIGGER_LEDGER.json", "RESULT.json", "GPU_COMMAND_RECEIPT.json",
        "0915_SAM31_STRICT_ROLE_REVIEW.mp4",
        "tasks/receipts/0915_SAM31_STRICT_ROLE_CANARY_V4_RESULT.json",
    ],
})


TASK_SPECS["0915_stereo_encoded_domain_preflight_v1"] = {
    "phase": "0915_STEREO_ENCODED_DOMAIN_PREFLIGHT_V1",
    "objective": (
        "Measure all 150 sourceIndex-aware encoded VST stereo frames after crop and "
        "resize only, including frames 81 and 94, and decide whether direct "
        "FoundationStereo input is admissible or encoded-domain epipolar alignment "
        "is still required."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "tasks/receipts/0915_CPU_NEXT_TASKS_USER_AUTHORIZATION.json",
        "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json",
        "tasks/receipts/0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json",
        "src/chaoyang/ops/run_0915_stereo_encoded_domain_preflight_v1.py",
        "src/chaoyang/pipeline/stereo_encoded_domain_preflight_v1.py",
        "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/playing_cards/play_cards_0915_001",
    ],
    "write_set": [
        ENCODED_STEREO_PREFLIGHT_ATTEMPT,
        "docs/current/visuals/0915_STEREO_ENCODED_DOMAIN_PREFLIGHT_V1",
    ],
    "prerequisites": [
        "wrong_domain_foundationstereo_terminal=REJECTED_QUALITY",
        "VST_ENCODED_VIDEO_ALREADY_UNDISTORTED",
        "sourceIndex_crop_then_resize_only", "zero_lens_undistortion",
        "single_session_play_cards_0915_001", "governance_PASS_FRESH",
        "weights_ABSENT", "gpu_FORBIDDEN", "source_read_only",
    ],
    "algorithm_prerequisites": {
        "stereo_preflight": ["same_session_SBS", "same_session_camera_params"],
        "not_dependencies": ["Clean", "Removal_Envelope", "old_rectified_depth"],
    },
    "weights": "ABSENT",
    "required_outputs": [
        "WRITER_CLAIM.json", "RUN_SIGNATURE.json", "PER_FRAME_METRICS.json",
        "ENCODED_STEREO_PREFLIGHT.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
    "expected_resource": "CPU_ONLY_FULL_150_FRAME_ENCODED_STEREO_DIAGNOSTIC",
    "claim_limit": (
        "Single-session encoded-pixel epipolar diagnosis only. It may authorize a "
        "separate FoundationStereo canary but creates no Depth, metric-accuracy, "
        "Object6D, Contact or Robot authority."
    ),
}


TASK_SPECS["0915_removal_envelope_v2_real_canary_v1"] = {
    "phase": "0915_REMOVAL_ENVELOPE_V2_REAL_CANARY_V1",
    "objective": (
        "Evaluate the conservative SAM-base-plus-validated-local-repairs V2 design "
        "on all 150 resize-only frames without rerunning SAM or inpainting, and "
        "compare it quantitatively against the sealed semantic baseline."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        "tasks/receipts/0915_CPU_NEXT_TASKS_USER_AUTHORIZATION.json",
        "src/chaoyang/ops/run_0915_removal_envelope_v2_real_canary_v1.py",
        "src/chaoyang/pipeline/removal_envelope_v2.py",
        "configs/systems/clean/removal_envelope_0915_play_cards_001_v2.json",
        "_run/current/0915_hawor_resize_only_canary_v1/attempts/attempt_0001",
        "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001",
        "_run/current/0915_sam31_weak_role_canary_v1/attempts/attempt_0001",
    ],
    "write_set": [
        REMOVAL_ENVELOPE_V2_ATTEMPT,
        "docs/current/visuals/0915_REMOVAL_ENVELOPE_V2_REAL_CANARY_V1",
    ],
    "prerequisites": [
        "routing_predecessor_0915_stereo_encoded_domain_preflight_v1=PASSED",
        "algorithm_independent_of_stereo_preflight_result",
        "strict_and_weak_SAM_outputs_read_only", "accepted_bounded_HaWoR",
        "single_session_play_cards_0915_001", "governance_PASS_FRESH",
        "weights_ABSENT", "gpu_FORBIDDEN", "no_SAM_rerun", "no_inpaint",
    ],
    "algorithm_prerequisites": {
        "removal": [
            "sealed_admitted_SAM_base", "real_foreground_proposals",
            "direct_observed_MANO", "stable_background_hook",
            "cable_reverse_pass_support",
        ],
        "not_dependencies": ["stereo_preflight_result", "Depth", "Object6D"],
        "forbidden_consumers": [
            "Depth", "Object6D", "Contact", "RobotGeometry", "ControlGroundTruth",
        ],
    },
    "weights": "ABSENT",
    "required_outputs": [
        "WRITER_CLAIM.json", "RUN_SIGNATURE.json", "STABLE_BACKGROUND.npz",
        "V2_MASK_LAYERS.npz", "FRAME_QUALITY_LEDGER.json",
        "REMOVAL_ENVELOPE_V2_REAL_SUMMARY.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
    "expected_resource": "CPU_ONLY_150_FRAME_VISUAL_CLEAN_MASK_CANARY",
    "claim_limit": (
        "Visual Clean-mask comparison only. PASS still requires user visual review "
        "before Clean authority. Removal and feather are never geometry, Contact, "
        "Robot control truth or physical-deployment evidence."
    ),
}


TASK_SPECS["0915_foundationstereo_encoded_domain_canary_v1"] = {
    "phase": "0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1",
    "objective": (
        "Run one FoundationStereo canary on sourceIndex-aware VST encoded pixels with "
        "crop/resize only, a simultaneous two-eye horizontal-reflection adapter, mirrored "
        "intrinsics, physical-left output unflip and pixelwise RGB-alignment proof."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        ENCODED_STEREO_PREFLIGHT_ATTEMPT,
        "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json",
        "tasks/receipts/0915_FOUNDATIONSTEREO_OBJECT6D_USER_CONFIRMATION_V1.json",
        "src/chaoyang/ops/run_0915_foundationstereo_encoded_domain_canary_v1.py",
        "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/playing_cards/play_cards_0915_001",
        "assets/models/checkpoints/foundationstereo/23-51-11",
        "configs/systems/depth/foundationstereo_0915_encoded_domain_canary_v1.json",
    ],
    "write_set": [
        ENCODED_DEPTH_CANARY_ATTEMPT,
        "docs/current/visuals/0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1",
    ],
    "prerequisites": [
        "0915_stereo_encoded_domain_preflight_v1=PASSED",
        "preflight_decision=PASS_DIRECT_FOUNDATION_INPUT",
        "VST_ENCODED_VIDEO_ALREADY_UNDISTORTED",
        "sourceIndex_crop_resize_only", "zero_lens_undistortion",
        "simultaneous_horizontal_reflection_no_camera_swap",
        "single_session_play_cards_0915_001", "governance_PASS_FRESH",
        "source_read_only", "serial_GPU_lease",
    ],
    "algorithm_prerequisites": {
        "depth": ["encoded_stereo_preflight_PASS_DIRECT_FOUNDATION_INPUT"],
        "not_dependencies": ["Clean", "Removal_Envelope", "old_rectified_depth"],
    },
    "weights": ["assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"],
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "ADAPTER_CONTRACT.json",
        "RGB_ALIGNMENT_QA.json", "DEPTH_CONTRACT.json", "DEPTH_SUMMARY.json",
        "DEPTH_WORKER_RESULT.json", "RESULT.json", "GPU_COMMAND_RECEIPT.json",
        "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 2, "runtime_attempts": 1},
    "expected_resource": "SERIAL_GPU_SINGLE_SESSION_FOUNDATIONSTEREO_ENCODED_DOMAIN",
    "claim_limit": (
        "Single-session development optical-Z candidate only. Consumption is limited "
        "to VISUAL_OBJECT6D_CANDIDATE_INPUT after all gates pass; external metric "
        "accuracy, batch, Contact, Robot and deployment authority remain absent."
    ),
}


TASK_SPECS["0915_planar_object6d_observability_canary_v2"] = {
    "phase": "0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2",
    "objective": (
        "Consume only the admitted encoded-domain FoundationStereo optical-Z and "
        "SAM3.1 task-object evidence to publish independent observability for three "
        "playing cards, while keeping the black tray separate and card_set semantic-only."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        ENCODED_DEPTH_CANARY_ATTEMPT,
        MASK_STRICT_CANARY_V5_ATTEMPT,
        "tasks/receipts/0915_FOUNDATIONSTEREO_OBJECT6D_USER_CONFIRMATION_V1.json",
        "src/chaoyang/ops/run_0915_planar_object6d_observability_canary_v2.py",
        "src/chaoyang/pipeline/object6d_planar_observability_v2.py",
        "src/chaoyang/pipeline/object6d_planar_observability_v1.py",
        "contracts/object6d_planar_observability_v2.schema.json",
    ],
    "write_set": [
        OBJECT6D_OBSERVABILITY_V2_ATTEMPT,
        "docs/current/visuals/0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2",
    ],
    "prerequisites": [
        "0915_foundationstereo_encoded_domain_canary_v1=PASSED",
        "depth_admission=PASS_VISUAL_OBJECT6D_CANDIDATE_INPUT",
        "same_physical_left_encoded_resize_only_domain",
        "three_cards_independent", "black_tray_separate_support_entity",
        "card_set_semantic_only", "measured_card_dimensions=ABSENT",
        "source_read_only", "weights_ABSENT", "gpu_FORBIDDEN",
        "single_session_play_cards_0915_001", "governance_PASS_FRESH",
    ],
    "algorithm_prerequisites": {
        "object6d": [
            "encoded_foundationstereo_PASS",
            "sam31_task_object_masks",
            "field_level_observability",
        ],
        "not_dependencies": [
            "Clean", "Removal_Envelope", "old_rectified_depth", "tray_pose_guess",
        ],
    },
    "weights": "ABSENT",
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "DEPTH_FRAME_INPUT_MANIFEST.json",
        "OBJECT6D_OBSERVABILITY_V2.json", "OBJECT6D_SUMMARY_V2.json", "RESULT.json",
        "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1},
    "expected_resource": "CPU_ONLY_150_FRAME_PLANAR_OBJECT6D_OBSERVABILITY",
    "claim_limit": (
        "Single-session development visible-surface observability only. center_xyz is "
        "not the hidden object centre; full extent remains unknown without complete "
        "boundary evidence. No external accuracy, Contact, Robot or deployment authority."
    ),
}


TASK_SPECS["0915_interaction_contact_robot_dev_v1"] = {
    "phase": "0915_INTERACTION_CONTACT_KAI22_DEVELOPMENT_V1",
    "objective": (
        "On the fixed 150-frame play_cards_0915_001 evidence bundle, publish Object6D "
        "QA, bounded non-contact Human/Stereo alignment, finite-patch Interaction and "
        "Contact candidates, an unconditional Kai22 R0 baseline, and only where locally "
        "admitted an R1 q22+wrist refinement."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        OBJECT6D_OBSERVABILITY_V2_ATTEMPT,
        ENCODED_DEPTH_CANARY_ATTEMPT,
        MASK_STRICT_CANARY_V5_ATTEMPT,
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/playing_cards/play_cards_0915_001",
        "src/chaoyang/ops/run_0915_interaction_contact_robot_dev_v1.py",
        "src/chaoyang/pipeline/interaction_contact_robot_dev_v1.py",
    ],
    "write_set": [
        INTERACTION_CONTACT_ROBOT_DEV_ATTEMPT,
        "docs/current/visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1",
        "tasks/receipts/0915_INTERACTION_CONTACT_ROBOT_DEV_V1_RESULT.json",
    ],
    "prerequisites": [
        "0915_planar_object6d_observability_canary_v2=PASSED",
        "single_session_play_cards_0915_001", "exact_150_frames",
        "HaWoR_SAM31_FoundationStereo_Object6D_v2_FROZEN_NO_RERUN",
        "Removal_Clean_FORBIDDEN", "weights_ABSENT", "gpu_FORBIDDEN",
        "development_relative_non_control_non_deployable", "governance_PASS_FRESH",
    ],
    "algorithm_prerequisites": {
        "r0": ["direct_observed_HaWoR", "pinned_Kai22_URDF"],
        "interaction": ["finite_visible_object_patch", "finger_associated_visible_surface"],
        "contact": ["fixed_5mm_proximity", "complete_bounded_internal_uncertainty"],
        "r1": ["alignment_heldout_PASS", "fixed_pair_local_contact_window"],
        "not_dependencies": ["Removal", "Clean", "archive", "short_gap_inferred"],
    },
    "weights": "ABSENT",
    "required_outputs": [
        "CLAIM.json", "RUN_SIGNATURE.json", "OBJECT6D_GEOMETRY_QA_V1.json",
        "CARD_DIMENSION_ESTIMATE_V1.json", "HUMAN_STEREO_ALIGNMENT_CHECK_V1.json",
        "INTERACTION_EVIDENCE_V1.json", "CONTACT_CANDIDATE_V1.json",
        "KAI22_R0_BASELINE_V1.npz", "KAI22_R0_BASELINE_V1.json",
        "KAI22_R1_LOCAL_REFINEMENT_V1.npz", "KAI22_R1_LOCAL_REFINEMENT_V1.json",
        "KAI22_R2_ARM_VISUAL_V1.json", "METRICS.json", "RESULT.json", "RUN_RECEIPT.json",
    ],
    "budgets": {"gpu_hours": 0, "runtime_attempts": 1, "wall_clock_hours": 6},
    "expected_resource": "CPU_ONLY_SINGLE_SESSION_150_FRAME_DEVELOPMENT_EVIDENCE_CHAIN",
    "claim_limit": (
        "DEVELOPMENT_RELATIVE / NON_CONTROL / NON_DEPLOYABLE only. Automatic gates "
        "authorize continued development, not visual acceptance or physical correctness. "
        "R0 must be delivered; R1 may be BLOCKED_LOCAL_EVIDENCE; R2 never blocks R1."
    ),
}


TASK_SPECS["0915_interaction_contact_robot_dev_v2"] = deepcopy(
    TASK_SPECS["0915_interaction_contact_robot_dev_v1"]
)
TASK_SPECS["0915_interaction_contact_robot_dev_v2"].update({
    "phase": "0915_INTERACTION_CONTACT_KAI22_DEVELOPMENT_V2",
    "objective": (
        "Correct only the V1 pre-execution HaWoR provenance enum validator, then run "
        "the unchanged fixed 150-frame Object6D QA, non-contact Human/Stereo alignment, "
        "finite-patch Contact, Kai22 R0, and fail-closed local R1 workflow."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        INTERACTION_CONTACT_ROBOT_DEV_ATTEMPT + "/RESULT.json",
        OBJECT6D_OBSERVABILITY_V2_ATTEMPT,
        ENCODED_DEPTH_CANARY_ATTEMPT,
        MASK_STRICT_CANARY_V5_ATTEMPT,
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "src/chaoyang/ops/run_0915_interaction_contact_robot_dev_v2.py",
        "src/chaoyang/pipeline/interaction_contact_robot_dev_v1.py",
    ],
    "write_set": [
        INTERACTION_CONTACT_ROBOT_DEV_V2_ATTEMPT,
        "docs/current/visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1",
        "tasks/receipts/0915_INTERACTION_CONTACT_ROBOT_DEV_V2_RESULT.json",
    ],
    "prerequisites": [
        "0915_interaction_contact_robot_dev_v1=FAILED_RUNTIME_FINAL_PRE_R0_ENUM_ONLY",
        "observed_provenance=BOUNDED_PARAMETER_FIT", "missing_provenance=MISSING",
        "boolean_observed_axis_remains_authority", "short_gap_inferred_FORBIDDEN",
        "all_v1_algorithm_thresholds_and_boundaries_unchanged",
        "weights_ABSENT", "gpu_FORBIDDEN", "governance_PASS_FRESH",
    ],
})


TASK_SPECS["0915_interaction_contact_robot_dev_v3"] = deepcopy(
    TASK_SPECS["0915_interaction_contact_robot_dev_v2"]
)
TASK_SPECS["0915_interaction_contact_robot_dev_v3"].update({
    "phase": "0915_INTERACTION_CONTACT_KAI22_DEVELOPMENT_V3",
    "objective": (
        "Correct only the V2 CARD_DIMENSION_ESTIMATE_V1 schema field list, then run "
        "the unchanged fixed 150-frame development evidence chain to terminal."
    ),
    "read_set": [
        "docs/governance/CURRENT_STATUS_RECEIPT.json",
        INTERACTION_CONTACT_ROBOT_DEV_V2_ATTEMPT + "/RESULT.json",
        OBJECT6D_OBSERVABILITY_V2_ATTEMPT,
        ENCODED_DEPTH_CANARY_ATTEMPT,
        MASK_STRICT_CANARY_V5_ATTEMPT,
        "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/bounded_output_guarded_identity_fixed/play_cards_0915_001/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
        "src/chaoyang/ops/run_0915_interaction_contact_robot_dev_v3.py",
        "src/chaoyang/pipeline/interaction_contact_robot_dev_v1.py",
    ],
    "write_set": [
        INTERACTION_CONTACT_ROBOT_DEV_V3_ATTEMPT,
        "docs/current/visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1",
        "tasks/receipts/0915_INTERACTION_CONTACT_ROBOT_DEV_V3_RESULT.json",
    ],
    "prerequisites": [
        "0915_interaction_contact_robot_dev_v2=FAILED_RUNTIME_FINAL_AFTER_R0_SCHEMA_ONLY",
        "dimension_budget_fields_added_to_schema", "R0_algorithm_unchanged",
        "all_interaction_contact_robot_thresholds_unchanged",
        "weights_ABSENT", "gpu_FORBIDDEN", "governance_PASS_FRESH",
    ],
})


def build_packet(task_id: str) -> dict[str, Any]:
    if task_id not in TASK_SPECS:
        raise KeyError(task_id)
    spec = deepcopy(TASK_SPECS[task_id])
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": task_id,
        "plan_revision": PLAN_REVISION,
        "phase": spec.pop("phase"),
        "objective": spec.pop("objective"),
        "stop_conditions": [
            "PASSED", "REJECTED_QUALITY", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE",
            "BLOCKED_EXTERNAL", "CANCELLED",
        ],
        **spec,
    }
    validate_packet_policy(packet)
    return packet


def validate_packet_policy(packet: dict[str, Any]) -> None:
    task_id = str(packet.get("task_id"))
    weights = packet.get("weights")
    if weights != "ABSENT" and (not isinstance(weights, list) or len(weights) != 1):
        raise ValueError(f"{task_id}: algorithm task must bind exactly one logical weight")
    if len(packet.get("read_set", [])) > 8:
        raise ValueError(f"{task_id}: read_set exceeds bounded packet limit")
    if task_id in {
        "0915_sam31_strict_role_canary_v1",
        "0915_sam31_strict_role_canary_v2",
        "0915_sam31_strict_role_canary_v3",
        "0915_sam31_strict_role_canary_v4",
        "0915_sam31_strict_role_canary_v5",
        "0915_sam31_weak_role_canary_v1",
        "0915_sam31_mask_full_v1",
    }:
        encoded = str(packet).lower()
        if weights != ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"]:
            raise ValueError("0915 Mask must bind exactly the pinned SAM3.1 checkpoint")
        if "sam2" in encoded or "cutie" in encoded:
            raise ValueError("0915 Mask may not register an alternate model")
        if "sam3.1_only_user_locked" not in encoded:
            raise ValueError("0915 Mask must carry the user lock")


def predecessor_task(task_id: str) -> str | None:
    if task_id == "0915_interaction_contact_robot_dev_v3":
        return "0915_interaction_contact_robot_dev_v2"
    if task_id == "0915_interaction_contact_robot_dev_v2":
        return "0915_interaction_contact_robot_dev_v1"
    if task_id == "0915_interaction_contact_robot_dev_v1":
        return "0915_planar_object6d_observability_canary_v2"
    if task_id == "0915_planar_object6d_observability_canary_v2":
        return "0915_foundationstereo_encoded_domain_canary_v1"
    if task_id == "0915_foundationstereo_encoded_domain_canary_v1":
        return "0915_stereo_encoded_domain_preflight_v1"
    if task_id == "0915_stereo_encoded_domain_preflight_v1":
        return "0915_foundationstereo_single_session_canary_v1"
    if task_id == "0915_removal_envelope_v2_real_canary_v1":
        return "0915_stereo_encoded_domain_preflight_v1"
    index = TASK_ORDER.index(task_id)
    return TASK_ORDER[index - 1] if index else "0915_0916_input_audit_clean_v1"
