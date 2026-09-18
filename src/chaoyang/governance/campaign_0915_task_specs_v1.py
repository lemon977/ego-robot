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
    "0915_sam31_mask_full_v1",
    "0915_foundationstereo_full_v1",
    "0915_post_geometry_robot_v1",
)

INPUT_ATTEMPT_V1 = "_run/current/0915_input_prepare_cad_v1/attempts/attempt_0001"
INPUT_ATTEMPT = "_run/current/0915_input_prepare_cad_v2/attempts/attempt_0001"
HAWOR_ATTEMPT = "_run/current/0915_hawor_full_v1/attempts/attempt_0001"
VST_AB_ATTEMPT = "_run/current/0915_vst_image_domain_ab_v1/attempts/attempt_0001"
HAWOR_RESIZE_CANARY_ATTEMPT = "_run/current/0915_hawor_resize_only_canary_v1/attempts/attempt_0001"
DEPTH_ATTEMPT = "_run/current/0915_foundationstereo_full_v1/attempts/attempt_0001"
MASK_ATTEMPT = "_run/current/0915_sam31_mask_full_v1/attempts/attempt_0001"


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
            "PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE",
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
    if task_id == "0915_sam31_mask_full_v1":
        encoded = str(packet).lower()
        if weights != ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"]:
            raise ValueError("0915 Mask must bind exactly the pinned SAM3.1 checkpoint")
        if "sam2" in encoded or "cutie" in encoded:
            raise ValueError("0915 Mask may not register an alternate model")
        if "sam3.1_only_user_locked" not in encoded:
            raise ValueError("0915 Mask must carry the user lock")


def predecessor_task(task_id: str) -> str | None:
    index = TASK_ORDER.index(task_id)
    return TASK_ORDER[index - 1] if index else "0915_0916_input_audit_clean_v1"
