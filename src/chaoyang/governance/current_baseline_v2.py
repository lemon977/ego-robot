"""Render the current baseline registry and bounded human-facing indexes.

This module is pure with respect to governance state: callers pass the already
validated authority/task ledgers and publish the returned bytes in the same
transaction as the current receipt.  Missing evidence stays explicit instead
of being inferred from filenames or old README text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"
STAGE_DOC_PATH = ROOT / "docs/governance/CURRENT_STAGE_BASELINES_ZH.md"
LAYOUT_PATH = ROOT / "docs/governance/CURRENT_FILE_LAYOUT.json"
REGRESSION_PATH = ROOT / "docs/governance/CURRENT_REGRESSION_MANIFEST.json"

PINNED_LARGE_ARTIFACTS: dict[str, dict[str, Any]] = {
    str(ROOT / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"): {
        "bytes": 3298527334,
        "sha256": "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1",
        "verification": "PINNED_FULL_SHA256_VERIFIED",
    },
    str(ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"): {
        "bytes": 3502755717,
        "sha256": "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "verification": "PINNED_FULL_SHA256_VERIFIED",
    },
    str(ROOT / "assets/models/cutie/cutie-base-mega.pth"): {
        "bytes": 140443788,
        "sha256": "9c05402ee36d3a356fb72715d263ba7e1ea06ad3bada48c1306491792da43023",
        "verification": "PINNED_FULL_SHA256_VERIFIED",
    },
    str(ROOT / "assets/models/cutie/torch_home/hub/checkpoints/resnet50-19c8e357.pth"): {
        "bytes": 102502400,
        "sha256": "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097",
        "verification": "PINNED_FULL_SHA256_VERIFIED",
    },
    str(ROOT / "assets/models/cutie/torch_home/hub/checkpoints/resnet18-5c106cde.pth"): {
        "bytes": 46827520,
        "sha256": "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8",
        "verification": "PINNED_FULL_SHA256_VERIFIED",
    },
    str(ROOT / "assets/models/sam2_1_hiera_large/sam2.1_hiera_large.pt"): {
        "bytes": 898083611,
        "sha256": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
        "verification": "PINNED_FULL_SHA256_VERIFIED",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ref(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        return {
            "status": "UNKNOWN_VERIFICATION_REQUIRED",
            "path": str(candidate),
            "reason": "FILE_NOT_FOUND_AT_GENERATION",
        }
    pinned = PINNED_LARGE_ARTIFACTS.get(str(candidate))
    if pinned is not None:
        if candidate.stat().st_size != pinned["bytes"]:
            return {
                "status": "UNKNOWN_VERIFICATION_REQUIRED",
                "path": str(candidate),
                "reason": "PINNED_LARGE_ARTIFACT_SIZE_CHANGED",
            }
        return {"path": str(candidate), **pinned}
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": _sha256(candidate)}


def _stage(authority: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next((item for item in authority.get("stages", []) if item.get("stage") == name), {})


def _weights_from_clean_result() -> Any:
    result = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/propainter_v1/get_potato_chips_0902_107/RESULT.json"
    if not result.is_file():
        return {"status": "UNKNOWN_VERIFICATION_REQUIRED", "reason": "PINNED_CLEAN_RESULT_MISSING"}
    try:
        value = json.loads(result.read_text(encoding="utf-8"))
        weights = value.get("vendor", {}).get("weights", {})
        if not weights:
            raise KeyError("vendor.weights")
        model_root = ROOT / "assets/models/vendor/propainter"
        current = {}
        for name in sorted(weights):
            candidate = model_root / name
            if not candidate.is_file():
                return {
                    "status": "UNKNOWN_VERIFICATION_REQUIRED",
                    "reason": f"CURRENT_CLEAN_WEIGHT_MISSING:{candidate}",
                }
            current[name] = _ref(candidate)
        return current
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"status": "UNKNOWN_VERIFICATION_REQUIRED", "reason": "CLEAN_WEIGHT_PROVENANCE_NOT_PARSEABLE"}


SPECS: tuple[dict[str, Any], ...] = (
    {
        "stage": "Raw", "algorithm_id": "exact78_frozen_cohort_v1",
        "code": ["src/chaoyang/ops/batch_convert_handle_acquisition_aligned_v2.py", "src/chaoyang/ops/convert_handle_egodex_to_tracker_session.py"],
        "weights": "NOT_APPLICABLE", "input_authority": "RAW_CAPTURE",
        "output_schema": "EXACT78_BATCH_MANIFEST", "quality_gates": ["156_unique_sessions", "frame_identity", "source_sha"],
        "limitations": ["0909/0910 acquisition-aligned v2 is a separate release and denominator."],
        "successor": "No successor changes the frozen exact78 denominator.", "superseded": ["ad-hoc directory inferred cohorts"],
        "related_evidence": ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260911_handle_acquisition_aligned_v2/FINAL_COMPLETION_AUDIT.json"],
    },
    {
        "stage": "HaWoR", "algorithm_id": "hawor_bounded_v2",
        "code": ["src/chaoyang/ops/run_hawor_bounded_parameter_successor.py", "src/chaoyang/ops/run_hawor_diagnostic_reproduction_pair_v71.py"], "weights": "UNKNOWN_VERIFICATION_REQUIRED",
        "input_authority": "FROZEN_EXACT78_COHORT", "output_schema": "exact78-current-hawor-batch-result-v1",
        "quality_gates": ["finite_MANO", "bounded_temporal_update", "bone_cv", "session_terminal"],
        "limitations": ["Monocular MANO 3D and Z are not external metric truth.", "The corrected CPU temporal diagnostic reproduction completed 6 canaries: 4 passed numeric gates for human review and 2 held on left-hand bone-length CV regression; it does not recover the original HaWoR inference weight SHA or promote successor authority."],
        "successor": "Only bounded canary plus fixed regression may replace a C cluster.", "superseded": ["pre-bounded HaWoR previews"],
        "related_evidence": ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/hawor/diagnostic_reproduction_R7_4/RESULT.json"],
    },
    {
        "stage": "Role Mask", "algorithm_id": "sam31_role_successor_v3",
        "code": [
            "src/chaoyang/ops/run_exact78_fullsession_role_mask.py",
            "src/chaoyang/ops/run_role_mask_runtime_contract_repair_v71.py",
            "src/chaoyang/ops/run_0915_sam31_strict_role_canary_v1.py",
            "src/chaoyang/ops/run_0915_sam31_weak_role_canary_v1.py",
            "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
            "src/chaoyang/pipeline/sam31_0915_strict_role_contract_v1.py",
            "src/chaoyang/pipeline/sam31_0915_weak_role_contract_v1.py",
            "src/chaoyang/pipeline/depth_mask_clean_contracts_r3.py",
        ], "weights": ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"],
        "input_authority": "HAWOR_A_B", "output_schema": "exact78-role-mask-successor-finalize-result-v3",
        "quality_gates": ["human_left_right_independent", "tracker_left_right_independent", "visible_coverage", "reentry", "offscreen_empty"],
        "limitations": ["Four exact78 role masks are independent; C sessions are not downstream-authorized.", "The 0915 strict-role result was accepted only as a bounded starting point; its forearm/sleeve/cable/card02 masks remain weak.", "The 0915 weak-role runner completed, but user visual review rejected it as a Clean baseline because masks flicker and finger sleeves/yellow cables are not reliable; execution success is not Clean authority."],
        "successor": "SAM3.1 remains the only executable semantic Mask model. Removal V1 is rejected: a successor may add only validated local repairs near admitted SAM foreground, never promote weak MANO, forearm or appearance priors into unconstrained erase regions, and must proceed without registering an alternate model route.", "superseded": ["metadata-only masks", "zero-reject full-session refresh gate", "SAM2.1/Cutie challenger selection", "Removal Envelope V1 weak-evidence union"],
        "related_evidence": ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_semantic_audit/RESULT.json", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/attempts/attempt_0005_final_contract_hardening/RUN_RECEIPT.json", "tasks/receipts/0915_SAM31_WEAK_ROLE_CANARY_V1_RESULT.json", "tasks/receipts/0915_SAM31_WEAK_ROLE_CANARY_V1_USER_VISUAL_REVIEW.json"],
    },
    {
        "stage": "Object Mask", "algorithm_id": "sam31_task_object_identity_v1",
        "code": [
            "src/chaoyang/ops/run_exact78_task_object_identity_guardian.py",
            "src/chaoyang/pipeline/causal_modal_mask_gpu_adapter_v71.py",
            "src/chaoyang/ops/evaluate_s1_modal_mask_canaries_v71.py",
            "src/chaoyang/pipeline/depth_mask_clean_contracts_r3.py",
        ], "weights": ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"],
        "input_authority": "RAW_RGB_AND_HAWOR", "output_schema": "exact78-mask-lane-batch-result-v1",
        "quality_gates": [
            "same_physical_identity", "observed_empty_when_ambiguous",
            "three_chips_instances_independent", "causal_prompt_order",
            "no_gold_accuracy_from_predecessor_audit",
        ],
        "limitations": [
            "Chips physical instances may never be unioned to pass an identity gate.",
            "Predecessor-derived re-entry intervals support development coverage and stability only, not accuracy.",
            "SAM3.1 is the only executable model for this Mask stage; no model-selection or challenger task is authorized.",
        ],
        "successor": "Improve SAM3.1 causal prompts, temporal identity, occlusion re-entry and object leakage gates on bounded canaries, then expand the same pinned SAM3.1 implementation only after those gates pass.", "superseded": ["class-only union masks", "SAM2.1/Cutie challenger selection"],
        "related_evidence": [
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/revisions/R7_3/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_input_preflight/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_semantic_audit/RESULT.json",
            "assets/models/cutie/ASSET_PIN.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/attempts/attempt_0005_final_contract_hardening/RUN_RECEIPT.json",
            "tasks/receipts/0915_REMOVAL_ENVELOPE_SINGLE_SESSION_CANARY_V1_RESULT.json",
            "tasks/receipts/0915_REMOVAL_ENVELOPE_V1_USER_VISUAL_REVIEW.json",
            "tasks/receipts/0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json",
        ],
    },
    {
        "stage": "Depth", "algorithm_id": "foundationstereo_encoded_physical_left_canary_pass_v4",
        "code": ["src/chaoyang/pipeline/vst_encoded_video_domain.py", "src/chaoyang/pipeline/stereo_encoded_domain_preflight_v1.py", "src/chaoyang/ops/run_0915_stereo_encoded_domain_preflight_v1.py", "src/chaoyang/ops/run_0915_foundationstereo_encoded_domain_canary_v1.py", "src/chaoyang/pipeline/depth_mask_clean_contracts_r3.py", "src/chaoyang/ops/validate_pipeline_contracts_r3.py"],
        "weights": ["assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"],
        "input_authority": "ENCODED_VST_SOURCEINDEX_CROP_RESIZE_ONLY", "output_schema": "0915-foundationstereo-encoded-depth-contract-v1",
        "quality_gates": ["encoded_VST_video_no_lens_undistortion", "sourceIndex_crop_then_resize_only", "simultaneous_horizontal_reflection_no_camera_swap", "mirrored_cx_width_minus_1_minus_cx", "output_unflip_to_physical_left", "pixelwise_RGB_depth_alignment_zero_error", "Z_equals_fB_over_d", "valid_range", "full_decode", "left_right_consistency", "temporal_distribution_stability"],
        "limitations": ["All decoded VST video is already undistorted; camera_params distortion fields are capture provenance and must not be applied to encoded pixels.", "The prior 0915 FoundationStereo canary used equiDis62 remapping and is WITHDRAWN_WRONG_IMAGE_DOMAIN in addition to its immutable REJECTED_QUALITY terminal.", "The fresh 150-frame encoded-domain FoundationStereo canary passed all internal gates after reflecting both physical eyes without swapping them and unflipping outputs into the original physical-left 640x480 domain.", "Pixelwise RGB roundtrip mismatch and coordinate roundtrip error are both zero; this proves domain alignment, not external metric accuracy.", "FoundationStereo exposes no native confidence, and external 30/50/70/100 cm validation remains absent; external_accuracy stays UNVERIFIED."],
        "successor": "Consume this Depth only in the bounded three-card Planar Object6D observability canary using the analytic physical-left 640x480 to SAM 1280x960 pixel-centre map. Do not expand to batch, Contact or Robot before field-level Object6D review and external depth calibration.", "superseded": ["uncorrected and preview-only depth", "0915 equiDis62-remapped FoundationStereo canary", "extra encoded-domain epipolar remap for play_cards_0915_001"],
        "related_evidence": [
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/attempts/attempt_0005_final_contract_hardening/RUN_RECEIPT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/artifacts/depth_10/attempts/attempt_0003_real_play_cards_0910_001/RUN_RECEIPT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/artifacts/depth_20/attempts/attempt_0003_real_input_preflight/RUN_RECEIPT.json",
            "configs/systems/depth/foundationstereo_0915_canary_v1.json",
            "tasks/receipts/FOUNDATIONSTEREO_RUNTIME_CLOSURE_V1.json",
            "tasks/receipts/0915_PROCESSED_ROOT_MOUNT_RELOCATION_V1.json",
            "tasks/receipts/0915_FOUNDATIONSTEREO_SINGLE_SESSION_CANARY_V1_RESULT.json",
            "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json",
            "tasks/receipts/0915_FOUNDATIONSTEREO_OBJECT6D_USER_CONFIRMATION_V1.json",
            "configs/systems/depth/foundationstereo_0915_encoded_domain_canary_v1.json",
            "_run/current/0915_stereo_encoded_domain_preflight_v1/attempts/attempt_0001/RESULT.json",
            "_run/current/0915_foundationstereo_encoded_domain_canary_v1/attempts/attempt_0001/RESULT.json",
            "docs/current/visuals/0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1/README_ZH.md",
        ],
    },
    {
        "stage": "Object6D", "algorithm_id": "planar_object6d_encoded_observability_v2_pass",
        "code": [
            "src/chaoyang/pipeline/object6d_planar_observability_v1.py",
            "contracts/object6d_planar_observability_v1.schema.json",
            "src/chaoyang/pipeline/object6d_planar_observability_v2.py",
            "contracts/object6d_planar_observability_v2.schema.json",
            "src/chaoyang/ops/run_0915_planar_object6d_observability_canary_v2.py",
        ], "weights": "NOT_APPLICABLE",
        "input_authority": "ENCODED_DOMAIN_DEPTH_PASSED_SINGLE_SESSION", "output_schema": "object6d-planar-observability-v2",
        "quality_gates": ["encoded_domain_depth_admitted", "analytic_pixel_center_2x_map", "mask_depth_valid", "KEEP_INVALID", "three_card_instance_independence", "tray_separate_unknown", "card_set_semantic_only", "field_level_observability"],
        "limitations": [
            "All 58 historical current Object6D terminals depended on FoundationStereo Depth produced after forbidden lens remapping of already-undistorted VST video; their current authority is withdrawn.",
            "Only visible-surface geometry may be measured from the admitted encoded-domain Depth; occluded frames must remain invalid.",
            "Plane residual is not external pose truth.",
            "Measured card dimensions are absent, so full physical extent must remain unobservable; no hidden centre or full 6DoF may be completed.",
            "The black tray has no independent mask evidence and remains UNKNOWN; card_set is semantic membership only and owns no rigid pose.",
            "On play_cards_0915_001, visible-surface center is observable on 143/146/95 frames and plane normal on 139/113/48 frames for card00/card01/card02 respectively; these are internal observability counts, not pose accuracy.",
        ],
        "successor": "Add a bounded per-frame geometry overlay review and measured card width/height before Interaction v1. Contact and Robot remain blocked until visible geometry is reviewed; old rectified-depth products cannot seed authority.", "superseded": ["filled occlusion pose previews", "rectified pose multiplied directly by selected-camera c2w", "58 remap-derived exact78 Object6D terminals", "card_plus_tray_single_rigid_body"],
        "related_evidence": [
            "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json",
            "tasks/receipts/0915_FOUNDATIONSTEREO_OBJECT6D_USER_CONFIRMATION_V1.json",
            "_run/current/0915_foundationstereo_encoded_domain_canary_v1/attempts/attempt_0001/RESULT.json",
            "_run/current/0915_planar_object6d_observability_canary_v2/attempts/attempt_0001/RESULT.json",
            "_run/current/0915_planar_object6d_observability_canary_v2/attempts/attempt_0001/OBJECT6D_SUMMARY_V2.json",
            "docs/current/visuals/0915_PLANAR_OBJECT6D_OBSERVABILITY_CANARY_V2/README_ZH.md",
        ],
    },
    {
        "stage": "Clean", "algorithm_id": "same_pixel_temporal_donor_then_propainter_v1",
        "code": ["src/chaoyang/ops/run_clean_synthetic_propainter_baseline.py", "src/chaoyang/ops/run_generic_same_session_real_donor_v1.py", "src/chaoyang/ops/run_0915_removal_envelope_single_session_canary_v1.py", "src/chaoyang/pipeline/removal_envelope_v1.py", "contracts/removal_envelope_v1.schema.json", "configs/systems/clean/removal_envelope_0915_play_cards_001_v1.json", "src/chaoyang/ops/run_0915_removal_envelope_v2_real_canary_v1.py", "src/chaoyang/pipeline/removal_envelope_v2.py", "contracts/removal_envelope_v2.schema.json", "configs/systems/clean/removal_envelope_0915_play_cards_001_v2.json", "src/chaoyang/pipeline/depth_mask_clean_contracts_r3.py"],
        "weights": "FROM_PINNED_CLEAN_RESULT", "input_authority": "ROLE_MASK_B_AND_OBJECT_MASK_B",
        "output_schema": "clean_frames+master+review+source_map+result", "quality_gates": ["frame_count", "visible_object_mask_byte_exact", "source_map", "master_decode", "review_decode"],
        "limitations": [
            "The current real-donor producer is temporal only: it copies the identical integer pixel coordinate from same-session frames after two-frame RGB consensus; it does not use Stereo reprojection or scene geometry.",
            "Human masks are expanded by 18-24 px and tracker masks by 60 px before removal; this fixed expansion can delete excessive boundary context.",
            "Byte-exact object protection covers only pixels present in the current visible task-object mask. It does not preserve or reconstruct object surface hidden by fingertips.",
            "A donor pixel can be byte-exact and still be semantically wrong when a moving plate/background surface occupies the same image coordinate in donor frames.",
            "A structural Grade-B terminal proves frame/provenance/decode closure, not contact-boundary or semantic inpainting correctness.",
            "Generated pixels are visual completion, not physical background truth.",
            "Clean never feeds Depth/Object6D/contact truth.",
            "Removal Envelope V1 is REJECTED_QUALITY and retained only as a failed experiment.",
            "Removal Envelope V2 real-video canary is also REJECTED_QUALITY: repairs stayed bounded (P95 contribution 0.00293, area inflation 1.00294, protected-object damage 0) and did not worsen relative flicker, but absolute temporal derivative P95 remained 0.8424 because the sealed SAM base itself flickers. The stable-background hook was also too sparse at 0.00163 of pixels.",
        ],
        "successor": "Do not tune V2 repair radii. First stabilize or temporally admit the SAM semantic base and replace the whole-video exact-stability background hook with a bounded local/background model; only then create a new finite Clean canary. Keep removal/feather forbidden to Depth/Object6D/Contact/Robot geometry. Inpaint remains a separate weighted task.",
        "superseded": ["unproven clean previews", "the inaccurate label that the current identical-coordinate donor uses Stereo geometry"],
        "related_evidence": [
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/optimization/clean_runtime/CLEAN_RUNTIME_BOTTLENECK_AUDIT_V71.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_user_visual_review_v1/CLEAN_CONTACT_BOUNDARY_USER_REVIEW.json",
            "docs/research/current/reports/visualization/20260915/CLEAN_CONTACT_AND_WRIST_BASELINE_AUDIT_ZH.md",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/attempts/attempt_0005_final_contract_hardening/RUN_RECEIPT.json",
            "tasks/receipts/0915_REMOVAL_ENVELOPE_V1_USER_VISUAL_REVIEW.json",
            "docs/current/REMOVAL_ENVELOPE_V2_ZH.md",
            "_run/current/0915_removal_envelope_v2_real_canary_v1/attempts/attempt_0001/RESULT.json",
        ],
    },
    {
        "stage": "Contact", "algorithm_id": "contact_evidence_dag_v71_geometry_v2",
        "code": ["src/chaoyang/pipeline/human_contact_hypothesis_v1.py", "src/chaoyang/pipeline/contact_geometry_v2.py", "src/chaoyang/pipeline/object_contact_evidence_v1.py", "src/chaoyang/pipeline/interaction_evidence_v0a.py", "src/chaoyang/pipeline/contact_occlusion_contracts_r3.py"], "weights": "NOT_APPLICABLE",
        "input_authority": "HAWOR_OBJECT_MASK_DIRECT_OBJECT6D", "output_schema": "contact_hypothesis_not_truth",
        "quality_gates": ["synthetic_geometry_fixture", "evidence_DAG_acyclic", "attachment_cannot_prove_contact", "instance_identity", "UNKNOWN_fail_closed"],
        "limitations": ["Poker/Chips geometry and evidence-direction fixtures pass, but no real-session Contact authority exists.", "Interaction v0a is image-plane-only adjacency/approach/co-motion with optional aligned tactile support; it explicitly contains no relative Z, occlusion order or contact truth.", "Attachment-propagated pose cannot prove Contact, Object6D, tactile support, or gold accuracy.", "Hypotheses are not physical contact truth."],
        "successor": "Use direct/tracked Object evidence to seed Contact, then attachment only in the downstream direction; freeze independent gold labels before authority.", "superseded": ["robot-render-dependent contact loop", "attachment-contact self-certification"],
        "related_evidence": ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact/RESULT.json", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/RESULT.json", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/revisions/R7_2/RESULT.json", "docs/research/current/CONTACT_OCCLUSION_METHODS_V71_ZH.md", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/attempts/attempt_0005_final_contract_hardening/RUN_RECEIPT.json"],
    },
    {
        "stage": "Robot Visual", "algorithm_id": "robot_geometry_v71_over_world_first_retarget_v5_2",
        "code": [
            "src/chaoyang/pipeline/contact_aware_robot_retarget_v1.py",
            "src/chaoyang/pipeline/robot_geometry_v1.py",
            "src/chaoyang/ops/run_tianji_kai_robot_baseline.py",
            "src/chaoyang/ops/run_hawor_temporal_jerk_successor.py",
            "src/chaoyang/ops/run_robot_motion_transfer_arm_canary_v3.py",
            "src/chaoyang/ops/select_robot_fixed_placement_v1.py",
            "src/chaoyang/ops/run_robot_arm_segment_bidirectional_v3.py",
            "src/chaoyang/ops/run_robot_hand_fullsession_v2.py",
            "src/chaoyang/ops/run_newtask_robot_shared_v4_hand.py",
            "src/chaoyang/ops/run_robot_hand_segment_bidirectional_v3.py",
            "src/chaoyang/ops/render_robot_motion_transfer_fullsession_v2.py",
            "src/chaoyang/ops/adopt_exact78_pose_only_visual_robot_v52.py",
            "src/chaoyang/ops/audit_robot_geometry_self_collision_v71.py",
            "src/chaoyang/ops/export_robot_unified_zbuffer_canary_v71.py",
            "src/chaoyang/ops/build_robot_object_ownership_canary_v71.py",
            "src/chaoyang/ops/audit_robot_hard_soft_gate_v71.py",
            "src/chaoyang/ops/run_robot_hard_soft_audit_watcher_v71.py",
            "src/chaoyang/ops/run_robot_conversion_diagnosis_watcher_v71.py",
            "src/chaoyang/ops/audit_kaihand_tip_reachability_floor_v71.py",
            "src/chaoyang/ops/audit_robot_placement_candidate_reduction_v71.py",
            "src/chaoyang/ops/run_robot_placement_reduction_watcher_v71.py",
            "src/chaoyang/ops/audit_robot_hand_round2_value_v71.py",
            "src/chaoyang/ops/run_v71_post_finalize_robot_expansion.py",
            "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py",
            "src/chaoyang/ops/run_exact78_robot_ready_batches_v75.py",
            "src/chaoyang/ops/robot_target_reach_v75.py",
            "src/chaoyang/ops/audit_robot_target_chain_v75.py",
            "src/chaoyang/ops/audit_robot_reach_v75.py",
            "src/chaoyang/ops/profile_robot_pipeline_v75.py",
            "contracts/robot_target_10_r3.schema.json",
            "contracts/robot_reach_20_r3.schema.json",
            "contracts/robot_profile_30_r3.schema.json",
        ], "weights": "NOT_APPLICABLE",
        "input_authority": "HAND_WRIST_URDF_CAMERA_OPTIONAL_OBJECT_CONTACT", "output_schema": "robot_geometry_v1+visual_robot_trajectory_sidecar+robot_target_10_r3+robot_reach_20_r3+robot_profile_30_r3",
        "quality_gates": ["hard:finite_proper_SE3", "hard:chirality_mount_root_closure", "hard:URDF_joint_limits", "hard:missing_stays_UNKNOWN", "hard:nonadjacent_self_collision", "visual_continuity:fps_bound_velocity_acceleration", "robotized:unified_zbuffer_and_provenance", "soft:human_arm_pose_similarity", "soft:human_hand_anatomy_similarity", "30s_total_frame_budget"],
        "limitations": ["R1 Robot Geometry is independent of Clean; Object6D absence permits pose-only visual results but not metric contact.", "Strict per-frame human-pose imitation is a soft target because MANO and KaiHand morphology/workspace differ; motion may differ while remaining usable for visual training.", "Velocity/acceleration are visual-continuity gates, not physical Robot safety certification.", "The bounded hard/soft audit can retain hard-feasible development candidates when only human-pose similarity misses; it does not promote strict Robot authority.", "Development candidates and quality-C diagnostics exist, but no Robot authority is currently published.", "control_ground_truth=false.", "Adapter/TCP/install/world-to-base physical calibration is absent."],
        "successor": "Continue immutable three-session batches; audit strict-C outputs with hard geometry/collision gates, then expand only hard-feasible candidates into causal Visual Aux inputs. R2 metric contact and V1 compositor remain separately gated.", "superseded": ["withdrawn wrong-mount and wrong-chirality videos", "Robot geometry coupled to Clean/compositor", "treating every soft human-pose residual as a hard geometry failure"],
        "related_evidence": [
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/pose_only_visual_robot_v1/CURRENT_CANDIDATE_INDEX.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_robot_visual_review_publication_v1/RESULT_SUMMARY.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_robot_failure_clusters_v1/ROBOT_FAILURE_CLUSTERS.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/canaries/get_potato_chips_0902_023/self_collision_4frame_R7_1/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/canaries/get_potato_chips_0902_023/self_collision_24frame_R7_1/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/canaries/get_potato_chips_0902_023/self_collision_fullsession_R7_1/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/canaries/get_potato_chips_0902_023/unified_zbuffer_4frame_R7_1/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/canaries/get_potato_chips_0902_023/unified_zbuffer_24frame_R7_2/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/canaries/get_potato_chips_0902_023/robot_object_ownership_24frame_R7_4/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_watcher_v71/batch_001/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_watcher_v71/candidates/get_potato_chips_0902_087/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_kaihand_pinky_reachability_v71/get_potato_chips_0902_087/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_placement_reduction_audit_v71/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hand_round2_value_audit_v71/RESULT.json",
            "docs/governance/ROBOT_QUALITY_GATE_POLICY_V71_ZH.md",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v75/attempts/attempt_0016_execution/FAILED_RUNTIME_FINAL.json",
        ],
    },
    {
        "stage": "Occlusion", "algorithm_id": "occlusion_silver_gold_split_v71",
        "code": ["src/chaoyang/pipeline/occlusion_compositor_v1.py", "src/chaoyang/pipeline/causal_robotized_compositor_v1.py", "src/chaoyang/pipeline/contact_occlusion_contracts_r3.py", "src/chaoyang/ops/build_visible_surface_occlusion_canary_v71.py", "src/chaoyang/ops/run_occlusion_visible_surface_watcher_v71.py", "src/chaoyang/ops/run_occlusion_silver_governance_finalizer_v71.py"], "weights": "NOT_APPLICABLE",
        "input_authority": "CLEAN_ROBOT_RENDER_DEPTH_OBJECT_APPEARANCE", "output_schema": "ownership+training_valid_mask",
        "quality_gates": ["silver_provenance_coverage", "silver_unknown_ratio", "zbuffer_consistency", "pixel_source_legality", "gold_accuracy_only_with_independent_labels"],
        "limitations": [
            "A real visible-surface z-buffer canary passes bounded numeric coverage, UNKNOWN and conditional-retention checks, but it does not reconstruct hidden object appearance or close every Silver gate.",
            "Current Clean-base Robot review videos recover renderer-changed pixels by a Raw-versus-Robot difference mask; they do not implement authorized object/Robot ownership and visibly retain wrong ordering in contact regions.",
            "Gold is blocked because the frozen independent 240-frame double-reviewed set does not exist; no accuracy is reported.",
        ],
        "successor": "First close object/contact-aware Clean inputs and legal hidden-object appearance provenance, then extend verified visible-surface ordering to a causal full-session compositor with unified Robot/object z-buffer, temporal/byte-exact/provenance gates; independently freeze gold labels before any accuracy claim.",
        "superseded": ["depth-only overlay without appearance provenance", "accuracy inferred from Silver consistency", "requiring Stereo quality where no Robot/object overlap exists", "worst-frame retention substituted for pixel-weighted retention", "treating Clean-base Robot review exports as occlusion-correct output"],
        "related_evidence": ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/occlusion_silver/RESULT.json", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/occlusion_gold/RESULT.json", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/revisions/R7_2/RESULT.json", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_occlusion_silver_canary_v71/get_potato_chips_0902_087/visible_surface_ordering_24frame_92_115_v8/RESULT.json", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_user_visual_review_v1/CLEAN_CONTACT_BOUNDARY_USER_REVIEW.json", "docs/research/current/CONTACT_OCCLUSION_METHODS_V71_ZH.md", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/attempts/attempt_0005_final_contract_hardening/RUN_RECEIPT.json"],
    },
    {
        "stage": "HumanEgo Aux", "algorithm_id": "h50_future_2d_causal_task_pairs_v71",
        "code": [
            "src/chaoyang/human_ego/tools/preflight_visual_aux_h50_candidates_v55.py",
            "src/chaoyang/human_ego/tools/build_visual_aux_eligibility_index_v56.py",
            "src/chaoyang/ops/build_visual_aux_robot_expansion_plan_v71.py",
            "src/chaoyang/human_ego/tools/build_visual_aux_session_bundle_v54.py",
            "src/chaoyang/human_ego/tools/validate_visual_aux_bundle_v52.py",
            "src/chaoyang/human_ego/tools/smoke_visual_aux_bundle_v55.py",
            "src/chaoyang/human_ego/training/VisualAuxFuture2DModel.py",
            "src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py",
            "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
            "src/chaoyang/ops/run_visual_tier_causal_clean_v71.py",
            "src/chaoyang/ops/run_visual_aux_candidate_bundle_watcher_v71.py",
            "src/chaoyang/ops/build_visual_aux_capacity_preflight_v71.py",
            "src/chaoyang/ops/run_v71_post_robot_visual_tier.py",
            "src/chaoyang/ops/run_v71_post_robot_visual_aux.py",
        ], "weights": "NOT_YET_PUBLISHED",
        "input_authority": "STRICT_RAW_ROBOTIZED_PAIRING_AND_VISUAL_2D_LABEL", "output_schema": "future_2d_xy+future_2d_valid+visual_aux_checkpoint",
        "quality_gates": ["causal_donor_proof", "paired_frame_ledger", "task_pair_independence", "H50_window_coverage", "ADE_2D", "FDE_2D", "PCK", "loss_curve", "value_gate"],
        "limitations": ["Zero current checkpoints; train/validation session and H50-window ledgers have not reached the frozen minima.", "A bounded capacity preflight finds sufficient potential routes for both task pairs, but pending Robot/Clean/bundle receipts are not counted as completed training data.", "Hard-feasible Robot development candidates can produce strictly causal paired bundles with ANY_ENDPOINT_40_OF_50 validity and a preserved per-side mask; this is development input evidence, not Robot/Occlusion authority.", "Existing bundles built from the current structural-only Clean are not checkpoint-eligible until contact-boundary preservation, semantic donor and Occlusion pixel-source gates pass; the user-rejected Chips039/Poker245 reviews are explicit negative evidence.", "Visual retarget labels are not Robot control actions."],
        "successor": "Continue immutable development prebuilds only; do not freeze a training ledger or start checkpoints until object/contact-aware Clean, legal pixel provenance and Occlusion Silver gates pass. Then freeze task-specific train/validation ledgers and train Chips/Poker pairs independently.", "superseded": ["old Kai22 checkpoints not bound to the current contract", "future-frame donor training input", "train_embodiment.py incorrectly named as the Visual Aux trainer", "bundle builders that publish zero-window sessions as training-ready", "requiring both projected endpoints valid at every H50 step", "treating structural-only Clean bundles as checkpoint-ready"],
        "related_evidence": [
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_h50_preflight_v55/PREFLIGHT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_h50_pose_only_preflight_v55/PREFLIGHT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_pose_only_batch_preflight_v55/PREFLIGHT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_eligibility_index_v56/ELIGIBILITY_INDEX.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/expansion_plan_R7_1/VISUAL_AUX_ROBOT_EXPANSION_PLAN.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_bundle_canary_v55/REAL_BUNDLE_SMOKE_RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_pose_only_bundle_canary_v55_2/get_potato_chips_0903_050/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_pose_only_bundle_canary_v55_2/REAL_BUNDLE_SMOKE_RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/causal_bundle_canary_R7_1/get_potato_chips_0903_050/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_visual_aux_bundle_canary_v71/get_potato_chips_0902_087/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_visual_aux_capacity_preflight_v71/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_user_visual_review_v1/CLEAN_CONTACT_BOUNDARY_USER_REVIEW.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/causal_training_R7_2/bundles/get_potato_chips_0902_083/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/causal_training_R7_2/bundles/get_potato_chips_0902_087/RESULT.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/causal_training_R7_2/bundles/get_potato_chips_0902_090/RESULT.json",
        ],
    },
    {
        "stage": "HumanEgo Policy", "algorithm_id": "blocked_external_real_robot_action",
        "code": ["src/chaoyang/human_ego/tools/train_embodiment.py"], "weights": "ABSENT",
        "input_authority": "REAL_ROBOT_ACTION_SIDECAR_REQUIRED", "output_schema": "policy_checkpoint",
        "quality_gates": ["real_action_schema", "synchronization", "train_validation_split", "policy_metrics"],
        "limitations": ["Real synchronized Robot action supervision is absent; visual trajectories cannot replace it."],
        "successor": "Remain BLOCKED_EXTERNAL until real actions exist.", "superseded": ["auxiliary checkpoints mislabelled as policy"],
    },
)


def build_registry(authority: Mapping[str, Any], task_state: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for spec in SPECS:
        current = _stage(authority, spec["stage"])
        if spec["weights"] == "FROM_PINNED_CLEAN_RESULT":
            weights = _weights_from_clean_result()
        elif isinstance(spec["weights"], list):
            weights = [_ref(path) for path in spec["weights"]]
        else:
            weights = spec["weights"]
        schema_files = {
            "Contact": "contracts/human_contact_hypothesis_v1.schema.json",
            "Robot Visual": "contracts/robot_geometry_v1.schema.json",
            "Occlusion": "contracts/occlusion_silver_v1.schema.json",
            "HumanEgo Aux": "contracts/visual_aux_dataset_ledger_v53.schema.json",
        }
        entry = {
            "stage": spec["stage"],
            "algorithm_id": spec["algorithm_id"],
            "code_closure": [_ref(path) for path in spec["code"]],
            "weights": weights,
            "input_authority": spec["input_authority"],
            "output_schema": spec["output_schema"],
            "schema_artifact": _ref(schema_files[spec["stage"]]) if spec["stage"] in schema_files else {
                "status": "UNKNOWN_VERIFICATION_REQUIRED",
                "reason": "SCHEMA_ID_RECORDED_BUT_NO_STANDALONE_CURRENT_SCHEMA_ARTIFACT",
            },
            "quality_gates": spec["quality_gates"],
            "current_evidence": list(current.get("evidence", [])),
            "authorized_scope": current.get("authority_scope", "NO_CURRENT_AUTHORITY"),
            "current_counts": {key: current.get(key, 0) for key in ("total", "passed", "grade_c", "running", "blocked")},
            "known_limitations": spec["limitations"],
            "successor_requirement": spec["successor"],
            "superseded_implementations": spec["superseded"],
        }
        if spec.get("related_evidence"):
            entry["related_current_releases"] = [_ref(path) for path in spec["related_evidence"]]
        entries.append(entry)
    sensor_tasks = {
        row.get("task_id"): row for row in task_state.get("tasks", [])
        if str(row.get("task_id", "")).startswith("sensor_h")
    }
    sensor_pipeline = {
        "pipeline": "handle_controller_manus_pico_sensor_v71",
        "cohort": "0909/0910 acquisition-aligned release; isolated from exact78 denominator",
        "components": {
            task_id: {
                "status": sensor_tasks.get(task_id, {}).get("status", "UNKNOWN_VERIFICATION_REQUIRED"),
                "phase": sensor_tasks.get(task_id, {}).get("phase"),
                "result": sensor_tasks.get(task_id, {}).get("result"),
            }
            for task_id in (
                "sensor_h0_admission_v1", "sensor_h1_hand_v1", "sensor_h2_tactile_v1",
                "sensor_h3_stereo_v1", "sensor_h4_mask_v1",
            )
        },
        "code_closure": [_ref(path) for path in (
            "src/chaoyang/ops/build_handle_sensor_pipeline_admission_v71.py",
            "src/chaoyang/ops/build_handle_h1_h2_sidecars_v71.py",
            "src/chaoyang/ops/build_handle_h3_h4_preflight_v71.py",
            "src/chaoyang/ops/run_sensor_h3_foundationstereo_canary_v71.py",
            "src/chaoyang/ops/run_chips001_pico_mask_temporal.py",
            "src/chaoyang/ops/run_v71_post_visual_aux_sensor_canaries.py",
            "src/chaoyang/ops/build_sensor_pipeline_terminal_matrix_v71.py",
        )],
        "weights": [_ref(path) for path in (
            "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth",
            "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt",
        )],
        "current_contract": "H0 admission branches to H1 hand, H2 tactile and H4 sensor-role Mask independently. H3 Stereo is blocked until a zero-lens-undistortion encoded-domain calibration successor is published.",
        "known_limitations": [
            "H3/H4 full GPU batches are not authorized by a bounded canary.",
            "All decoded VST video is already undistorted. H3 must not apply equiDis62 or any other lens-undistortion remap; any epipolar-only alignment needs a fresh encoded-domain calibration and review.",
            "H4 seed prompts are sensor-domain glove/Controller/object prompts; exact78 bare-hand prompts are not reusable.",
            "Stereo optical-Z is visible-surface engineering depth, not external metric or wrist truth.",
        ],
        "successor_requirement": "After the Visual Aux pair reaches a terminal, run one H3 and one H4 bounded canary; use measured quality/throughput to close or explicitly schedule a full batch.",
        "claim_limit": "Parallel sensor pipeline status; no exact78 denominator, physical accuracy, full Depth/Mask or deployment authority is implied.",
    }
    return {
        "schema_version": "chaoyang-current-baseline-registry-v2",
        "governance_revision": authority["governance_revision"],
        "generation_id": authority["generation_id"],
        "generated_at": authority["generated_at"],
        "status": "CURRENT",
        "entries": entries,
        "parallel_pipelines": [sensor_pipeline],
        "claim_limit": "Current algorithm and authority registry. UNKNOWN_VERIFICATION_REQUIRED is intentional and must not be inferred.",
    }


def build_layout(authority: Mapping[str, Any], task_state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "chaoyang-current-file-layout-v1",
        "governance_revision": authority["governance_revision"],
        "generation_id": authority["generation_id"],
        "generated_at": authority["generated_at"],
        "repository_root": str(ROOT),
        "retained_top_level": [
            "README.md", "AGENTS.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
            "assets", "configs", "contracts", "docs", "manifests", "scripts",
            "src", "tasks", "tests", "vendor", "archive", "_run",
        ],
        "protected_external": [
            "/mnt/data/egodata",
            "/nas/chenxianchi/egosteertouch",
            "/nas/chenxianchi/egoverse_piper",
            "/nas/chenxianchi/openpi",
            "/nas/chenxianchi/tactiel_pretrain_outputs",
        ],
        "canonical_documents": {
            "current_status": str(ROOT / "docs/governance/CURRENT_PROJECT_STATUS_ZH.md"),
            "baseline_registry": str(REGISTRY_PATH),
            "algorithm_contract": str(ROOT / "docs/governance/ALGORITHM_CONTRACT.json"),
            "document_authority_map": str(ROOT / "docs/governance/DOC_AUTHORITY_MAP.json"),
            "execution_plan": str(ROOT / "docs/governance/CHAOYANG_V7_1_R3_EXECUTION_PLAN_ZH.md"),
            "plan_migration_receipt": str(ROOT / "docs/governance/PLAN_MIGRATION_RECEIPT.json"),
            "stage_baselines": str(STAGE_DOC_PATH),
            "end_to_end": str(ROOT / "docs/reference/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md"),
            "depth_report": str(ROOT / "docs/research/current/reports/depth_accuracy/20260911"),
        },
        "forbidden_top_level_after_cleanup": [
            "NOW", "DEPTH_ACCURACY_PACKAGE_20260911",
            "RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md",
            "HumanEgo", "pipeline", "systems", "third_party", "tools", "data",
        ],
        "claim_limit": "Layout contract only. Historical snapshot paths are resolved through PATH_REDIRECTS.json.",
    }


def build_regression_manifest(authority: Mapping[str, Any]) -> dict[str, Any]:
    tests = (
        ("data_cleaning", "tests/test_data_cleaning_baseline_contract.py"),
        ("governance", "tests/test_governance_fact_ledger.py"),
        ("governance", "tests/test_governance_heartbeat_clean_reconcile_v52.py"),
        ("governance", "tests/test_heartbeat_task_live_guard_v1.py"),
        ("governance", "tests/test_current_baseline_registry_v2.py"),
        ("clean", "tests/test_exact78_clean_wave_attempts_v3.py"),
        ("hawor", "tests/test_hawor_bounded_parameter_successor.py"),
        ("hawor", "tests/test_run_hawor_diagnostic_reproduction_pair_v71.py"),
        ("role_mask", "tests/test_run_exact78_role_mask_successor_v3.py"),
        ("role_mask", "tests/test_0915_sam31_strict_role_canary_v1.py"),
        ("role_mask", "tests/test_0915_sam31_weak_role_canary_v1.py"),
        ("clean", "tests/pipeline/test_removal_envelope_v1.py"),
        ("clean", "tests/pipeline/test_removal_envelope_v2.py"),
        ("clean", "tests/test_removal_envelope_v2_real_canary_v1.py"),
        ("clean", "tests/test_0915_removal_envelope_single_session_canary_v1.py"),
        ("object_mask", "tests/test_mask_successor_protocol.py"),
        ("depth_object6d", "tests/test_exact78_depth_object6d_expansion_v2.py"),
        ("depth_object6d", "tests/test_0915_stereo_domain_preflight_v1.py"),
        ("depth_object6d", "tests/test_stereo_encoded_domain_preflight_v1.py"),
        ("depth_object6d", "tests/test_run_0915_stereo_interaction_cpu_canary_v1.py"),
        ("depth_object6d", "tests/test_0915_foundationstereo_single_session_canary_v1.py"),
        ("depth_object6d", "tests/test_0915_foundationstereo_encoded_domain_canary_v1.py"),
        ("depth_object6d", "tests/pipeline/test_vst_encoded_video_domain.py"),
        ("governance", "tests/test_withdraw_vst_remap_geometry_authority_v1.py"),
        ("object6d", "tests/test_0915_planar_object6d_single_session_v1.py"),
        ("object6d", "tests/pipeline/test_object6d_planar_observability_v2.py"),
        ("object6d", "tests/test_0915_planar_object6d_observability_canary_v2.py"),
        ("object6d", "tests/test_object6d_agent_baseline.py"),
        ("object6d", "tests/test_adapt_object6d_rectified_to_selected_v71.py"),
        ("object6d", "tests/test_audit_object6d_coordinate_domain_v71.py"),
        ("object6d", "tests/test_run_object6d_selected_camera_adapter_batch_v71.py"),
        ("contact", "tests/test_contact_geometry_v2_fixtures.py"),
        ("contact", "tests/test_interaction_evidence_v0a.py"),
        ("occlusion", "tests/test_contact_robot_occlusion_v1.py"),
        ("occlusion", "tests/test_build_visible_surface_occlusion_canary_v71.py"),
        ("occlusion", "tests/test_occlusion_visible_surface_watcher_v71.py"),
        ("occlusion", "tests/test_run_occlusion_silver_governance_finalizer_v71.py"),
        ("robot", "tests/test_tianji_kai_robot_baseline.py"),
        ("robot", "tests/test_robot_hand_round1_v52.py"),
        ("humanego_aux", "tests/human_ego/test_visual_aux_contract_v1.py"),
        ("humanego_aux", "tests/human_ego/test_visual_aux_bundle_v52.py"),
        ("humanego_aux", "tests/human_ego/test_visual_aux_future2d_v53.py"),
        ("humanego_aux", "tests/test_run_gpu_command_with_v71_lease.py"),
        ("humanego_aux", "tests/test_run_v71_post_robot_visual_aux.py"),
        ("humanego_aux", "tests/test_run_visual_tier_causal_clean_v71.py"),
        ("humanego_aux", "tests/test_run_v71_post_robot_visual_tier.py"),
        ("governance", "tests/test_governance_v71_contracts.py"),
        ("governance", "tests/governance/test_r3_governance_contracts.py"),
        ("pipeline_r3", "tests/test_pipeline_contracts_r3.py"),
        ("robot", "tests/test_robot_v75_recovery_target_reach_profile.py"),
        ("robot", "tests/test_robot_r3_output_schemas.py"),
        ("robot", "tests/test_audit_robot_hard_soft_gate_v71.py"),
        ("robot", "tests/test_robot_hard_soft_watcher_v75_adopt.py"),
        ("humanego_aux", "tests/test_visual_aux_checkpoint_index_cli_v71.py"),
        ("cleanup", "tests/test_cleanup_current_only_v71_cli.py"),
        ("governance", "tests/test_publish_v71_task_packet_successor.py"),
        ("governance", "tests/test_gpu_lease_v71.py"),
        ("governance", "tests/test_v71_finite_convergence_supervisor.py"),
        ("governance", "tests/test_run_post_clean_s1_automation_v71.py"),
        ("governance", "tests/test_run_v71_post_s1_finalize.py"),
        ("cleanup", "tests/test_cleanup_current_only_v71.py"),
        ("conversion", "tests/test_build_conversion_cause_ledger_v2.py"),
        ("conversion", "tests/test_build_exact78_conversion_cause_ledger_v2.py"),
        ("conversion", "tests/test_build_successor_canary_selection_v71.py"),
        ("conversion", "tests/test_robot_conversion_diagnosis_watcher_v71.py"),
        ("conversion", "tests/test_build_successor_command_packets_v71.py"),
        ("exact78", "tests/test_build_exact78_final_terminal_matrix_v71.py"),
        ("object_mask", "tests/test_audit_s1_input_semantics_v71.py"),
        ("object_mask", "tests/test_build_contact_occlusion_canary_plan_v71.py"),
        ("object_mask", "tests/test_build_contact_occlusion_canary_revision_r72.py"),
        ("object_mask", "tests/test_build_contact_occlusion_canary_revision_r73.py"),
        ("object_mask", "tests/test_build_frozen_reentry_audit_frames_v71.py"),
        ("object_mask", "tests/test_build_s1_input_preflight_index_v71.py"),
        ("object_mask", "tests/test_causal_modal_mask_challenger_v71.py"),
        ("object_mask", "tests/test_causal_modal_mask_gpu_adapter_v71.py"),
        ("object_mask", "tests/test_cutie_asset_pin_v71.py"),
        ("object_mask", "tests/test_evaluate_s1_modal_mask_canaries_v71.py"),
        ("role_mask", "tests/test_run_role_mask_runtime_contract_repair_v71.py"),
        ("role_mask", "tests/test_sam31_canonical_tree_identity_v71.py"),
        ("contact", "tests/test_object_contact_evidence_v1.py"),
        ("object6d", "tests/test_object_pose_hypothesis_v2.py"),
        ("occlusion", "tests/test_build_occlusion_blocked_receipts_v71.py"),
        ("occlusion", "tests/test_causal_robotized_compositor_v1.py"),
        ("robot", "tests/test_build_metric_contact_robot_blocked_v71.py"),
        ("robot", "tests/test_build_robot_geometry_terminal_matrix_v71.py"),
        ("robot", "tests/test_continue_exact78_robot_session_067_v1.py"),
        ("robot", "tests/test_audit_robot_geometry_self_collision_v71.py"),
        ("robot", "tests/test_export_robot_unified_zbuffer_canary_v71.py"),
        ("robot", "tests/test_robot_geometry_v1.py"),
        ("robot", "tests/test_build_robot_object_ownership_canary_v71.py"),
        ("robot", "tests/test_run_exact78_robot_ready_batches_v53.py"),
        ("robot", "tests/test_run_v71_post_finalize_robot_expansion.py"),
        ("robot", "tests/test_audit_robot_hard_soft_gate_v71.py"),
        ("robot", "tests/test_audit_kaihand_tip_reachability_floor_v71.py"),
        ("robot", "tests/test_audit_robot_placement_candidate_reduction_v71.py"),
        ("robot", "tests/test_run_robot_placement_reduction_watcher_v71.py"),
        ("robot", "tests/test_audit_robot_hand_round2_value_v71.py"),
        ("sensor", "tests/test_build_handle_sensor_pipeline_admission_v71.py"),
        ("sensor", "tests/test_build_handle_h1_h2_sidecars_v71.py"),
        ("sensor", "tests/test_build_handle_h3_h4_preflight_v71.py"),
        ("sensor", "tests/test_build_sensor_pipeline_terminal_matrix_v71.py"),
        ("sensor", "tests/test_sensor_post_visual_aux_canaries_v71.py"),
        ("humanego_aux", "tests/test_build_visual_aux_checkpoint_index_v71.py"),
        ("humanego_aux", "tests/test_build_visual_aux_robot_expansion_plan_v71.py"),
        ("humanego_aux", "tests/test_visual_aux_candidate_bundle_watcher_v71.py"),
        ("humanego_aux", "tests/test_visual_aux_capacity_preflight_v71.py"),
    )
    return {
        "schema_version": "chaoyang-current-regression-manifest-v1",
        "governance_revision": authority["governance_revision"],
        "generation_id": authority["generation_id"],
        "generated_at": authority["generated_at"],
        "tests": [{"stage": stage, **_ref(path)} for stage, path in tests],
        "command": "python -m pytest -q <tests[].path>",
        "claim_limit": "Current smoke/regression selection; full historical test inventory is not implied current.",
    }


def render_stage_doc(registry: Mapping[str, Any]) -> str:
    lines = [
        "# 当前各阶段算法与整改需求",
        "",
        "> 本页随事实账本原子生成；运行数量以 `CURRENT_PROJECT_STATUS_ZH.md` 为准。",
        "",
        "| 阶段 | 当前算法 | 当前授权 | 主要边界 | 下一整改 |",
        "|---|---|---|---|---|",
    ]
    for entry in registry["entries"]:
        limits = "；".join(entry["known_limitations"])
        lines.append(f"| {entry['stage']} | `{entry['algorithm_id']}` | `{entry['authorized_scope']}` | {limits} | {entry['successor_requirement']} |")
    lines += ["", "## 独立并行数据线", ""]
    for pipeline in registry.get("parallel_pipelines", []):
        states = "，".join(
            f"{name}={value['status']}" for name, value in pipeline["components"].items()
        )
        lines += [
            f"### `{pipeline['pipeline']}`",
            "",
            f"- Cohort：{pipeline['cohort']}",
            f"- 当前组件：{states}",
            f"- 合同：{pipeline['current_contract']}",
            f"- 下一步：{pipeline['successor_requirement']}",
            f"- 边界：{pipeline['claim_limit']}",
            "",
        ]
    lines += [
        "",
        "## 固定解释边界",
        "",
        "- HaWoR 单目三维、Stereo/Object6D 内部残差与 Robot 数字接触距离都不是物理真值。",
        "- `visual_robot_trajectory_sidecar` 始终标记 `control_ground_truth=false`，不能冒充真实 Robot action。",
        "- Contact 与 Occlusion 只有接口/几何开发证据；真实 goldset 未闭合，因此不能声称遮挡关系已经解决。",
        "- 当前基线只由本页同 revision 的机器注册表与 receipt 解释；旧 README、目录名和聊天不构成 authority。",
        "",
    ]
    return "\n".join(lines)
