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


ROOT = Path(__file__).resolve().parents[2]
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
    result = ROOT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/propainter_v1/get_potato_chips_0902_107/RESULT.json"
    if not result.is_file():
        return {"status": "UNKNOWN_VERIFICATION_REQUIRED", "reason": "PINNED_CLEAN_RESULT_MISSING"}
    try:
        value = json.loads(result.read_text(encoding="utf-8"))
        weights = value.get("vendor", {}).get("weights", {})
        if not weights:
            raise KeyError("vendor.weights")
        return {name: dict(reference) for name, reference in sorted(weights.items())}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"status": "UNKNOWN_VERIFICATION_REQUIRED", "reason": "CLEAN_WEIGHT_PROVENANCE_NOT_PARSEABLE"}


SPECS: tuple[dict[str, Any], ...] = (
    {
        "stage": "Raw", "algorithm_id": "exact78_frozen_cohort_v1",
        "code": ["tools/batch_convert_handle_acquisition_aligned_v2.py", "tools/convert_handle_egodex_to_tracker_session.py"],
        "weights": "NOT_APPLICABLE", "input_authority": "RAW_CAPTURE",
        "output_schema": "EXACT78_BATCH_MANIFEST", "quality_gates": ["156_unique_sessions", "frame_identity", "source_sha"],
        "limitations": ["0909/0910 acquisition-aligned v2 is a separate release and denominator."],
        "successor": "No successor changes the frozen exact78 denominator.", "superseded": ["ad-hoc directory inferred cohorts"],
        "related_evidence": ["tasks/control/runs/20260911_handle_acquisition_aligned_v2/FINAL_COMPLETION_AUDIT.json"],
    },
    {
        "stage": "HaWoR", "algorithm_id": "hawor_bounded_v2",
        "code": ["tools/run_hawor_bounded_parameter_successor.py"], "weights": "UNKNOWN_VERIFICATION_REQUIRED",
        "input_authority": "FROZEN_EXACT78_COHORT", "output_schema": "exact78-current-hawor-batch-result-v1",
        "quality_gates": ["finite_MANO", "bounded_temporal_update", "bone_cv", "session_terminal"],
        "limitations": ["Monocular MANO 3D and Z are not external metric truth."],
        "successor": "Only bounded canary plus fixed regression may replace a C cluster.", "superseded": ["pre-bounded HaWoR previews"],
    },
    {
        "stage": "Role Mask", "algorithm_id": "sam31_role_successor_v3",
        "code": ["tools/run_exact78_fullsession_role_mask.py"], "weights": ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"],
        "input_authority": "HAWOR_A_B", "output_schema": "exact78-role-mask-successor-finalize-result-v3",
        "quality_gates": ["human_left_right_independent", "tracker_left_right_independent", "visible_coverage", "reentry", "offscreen_empty"],
        "limitations": ["Four role masks are independent; C sessions are not downstream-authorized."],
        "successor": "Fresh representative canary and two A/B regressions.", "superseded": ["metadata-only masks", "zero-reject full-session refresh gate"],
    },
    {
        "stage": "Object Mask", "algorithm_id": "sam31_task_object_identity_v1",
        "code": ["tools/run_exact78_task_object_identity_guardian.py"], "weights": ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"],
        "input_authority": "RAW_RGB_AND_HAWOR", "output_schema": "exact78-mask-lane-batch-result-v1",
        "quality_gates": ["same_physical_identity", "observed_empty_when_ambiguous", "three_chips_instances_independent"],
        "limitations": ["Chips physical instances may never be unioned to pass an identity gate."],
        "successor": "Action-conditioned identity reacquisition with fail-closed ambiguity.", "superseded": ["class-only union masks"],
    },
    {
        "stage": "Depth", "algorithm_id": "foundationstereo_corrected_metric_v1",
        "code": ["tools/run_exact78_foundationstereo_corrected_depth_worker.py"],
        "weights": ["assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"],
        "input_authority": "REGISTERED_RECTIFIED_STEREO_AND_SAME_SESSION_CALIBRATION", "output_schema": "depth_m+depth_valid+registration",
        "quality_gates": ["Z_equals_fB_over_d_recompute", "registration", "valid_range", "full_decode"],
        "limitations": ["FoundationStereo exposes no native confidence in this contract.", "Internal metric closure is not external millimetre accuracy."],
        "successor": "External 30/50/70/100 cm validation is required for physical accuracy claims.", "superseded": ["uncorrected and preview-only depth"],
    },
    {
        "stage": "Object6D", "algorithm_id": "object6d_observed_visible_surface_v1",
        "code": ["tools/run_visual_fixed_instance_object6d.py"], "weights": "NOT_APPLICABLE",
        "input_authority": "DEPTH_B_AND_OBJECT_MASK_B", "output_schema": "camera_world_SE3_observed_only",
        "quality_gates": ["mask_depth_valid", "finite_SE3", "proper_rotation", "KEEP_INVALID", "instance_independence"],
        "limitations": ["Only visible-surface geometry is measured; occluded frames remain invalid.", "Plane residual is not external pose truth."],
        "successor": "A separate hypothesis sidecar may bridge bounded gaps but cannot overwrite formal Object6D.", "superseded": ["filled occlusion pose previews"],
    },
    {
        "stage": "Clean", "algorithm_id": "real_temporal_stereo_donor_then_propainter_v1",
        "code": ["tools/run_clean_synthetic_propainter_baseline.py", "tools/run_generic_same_session_real_donor_v1.py"],
        "weights": "FROM_PINNED_CLEAN_RESULT", "input_authority": "ROLE_MASK_B_AND_OBJECT_MASK_B",
        "output_schema": "clean_frames+master+review+source_map+result", "quality_gates": ["frame_count", "object_byte_exact", "source_map", "master_decode", "review_decode"],
        "limitations": ["Generated pixels are visual completion, not physical background truth.", "Clean never feeds Depth/Object6D/contact truth."],
        "successor": "Finish frozen Wave0 terminals without changing selection.", "superseded": ["unproven clean previews"],
    },
    {
        "stage": "Contact", "algorithm_id": "human_contact_hypothesis_v1_geometry_v2",
        "code": ["pipeline/human_contact_hypothesis_v1.py", "pipeline/contact_geometry_v2.py"], "weights": "NOT_APPLICABLE",
        "input_authority": "HAWOR_OBJECT_MASK_DIRECT_OBJECT6D", "output_schema": "contact_hypothesis_not_truth",
        "quality_gates": ["synthetic_geometry_fixture", "bounded_gap", "instance_identity", "UNKNOWN_fail_closed", "frozen_goldset_required"],
        "limitations": ["Geometry fixtures pass, but the real-session goldset pack is explicitly blocked pending two independent human labels and a Robot candidate.", "Hypotheses are not physical contact truth."],
        "successor": "Freeze and double-review the real contact/occlusion goldset before authority.", "superseded": ["robot-render-dependent contact loop"],
        "related_evidence": ["tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/goldset_v1_review_pack/RESULT.json"],
    },
    {
        "stage": "Robot Visual", "algorithm_id": "world_first_retarget_v5_2_first_observed_anchor",
        "code": [
            "pipeline/contact_aware_robot_retarget_v1.py",
            "tools/run_tianji_kai_robot_baseline.py",
            "tools/run_hawor_temporal_jerk_successor.py",
            "tools/run_robot_motion_transfer_arm_canary_v3.py",
            "tools/select_robot_fixed_placement_v1.py",
            "tools/run_robot_arm_segment_bidirectional_v3.py",
            "tools/run_robot_hand_fullsession_v2.py",
            "tools/run_newtask_robot_shared_v4_hand.py",
            "tools/run_robot_hand_segment_bidirectional_v3.py",
            "tools/render_robot_motion_transfer_fullsession_v2.py",
            "tools/adopt_exact78_pose_only_visual_robot_v52.py",
        ], "weights": "NOT_APPLICABLE",
        "input_authority": "CLEAN_JOIN_READY_OR_POSE_ONLY_VISUAL", "output_schema": "visual_robot_trajectory_sidecar",
        "quality_gates": ["proper_SE3", "chirality", "mount_closure", "joint_limits", "reachability", "full_video_review"],
        "limitations": ["No Robot authority is currently published; 23 review videos are development candidates only (3 strict-numeric-pass, 5 pose-only, 15 quality-C diagnostics).", "control_ground_truth=false.", "Adapter/TCP/install/world-to-base physical calibration is absent."],
        "successor": "Four keyframes, then 24 frames, then full session; at most two bounded successors.", "superseded": ["withdrawn wrong-mount and wrong-chirality videos"],
        "related_evidence": [
            "tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/pose_only_visual_robot_v1/CURRENT_CANDIDATE_INDEX.json",
            "tasks/control/runs/20260914_robot_visual_review_publication_v1/RESULT_SUMMARY.json",
        ],
    },
    {
        "stage": "Occlusion", "algorithm_id": "occlusion_compositor_v1_development_interface",
        "code": ["pipeline/occlusion_compositor_v1.py"], "weights": "NOT_APPLICABLE",
        "input_authority": "CLEAN_ROBOT_RENDER_DEPTH_OBJECT_APPEARANCE", "output_schema": "ownership+training_valid_mask",
        "quality_gates": ["known_accuracy", "known_coverage", "unknown_ratio", "pixel_source_legality", "object_conditional_retention"],
        "limitations": ["The review pack remains BLOCKED_EXTERNAL_PENDING_TWO_INDEPENDENT_HUMAN_LABELS_AND_ROBOT_CANDIDATE; occlusion is not yet solved."],
        "successor": "Goldset first, then task canaries; compositor never feeds back into retarget.", "superseded": ["depth-only overlay without appearance provenance"],
        "related_evidence": ["tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/goldset_v1_review_pack/RESULT.json"],
    },
    {
        "stage": "HumanEgo Aux", "algorithm_id": "h50_future_2d_visual_aux_v5_2",
        "code": ["HumanEgo/tools/validate_visual_aux_bundle_v52.py", "HumanEgo/tools/train_embodiment.py", "HumanEgo/tools/evaluate_h50.py"], "weights": "NOT_YET_PUBLISHED",
        "input_authority": "STRICT_RAW_ROBOTIZED_PAIRING_AND_VISUAL_2D_LABEL", "output_schema": "future_2d_xy+future_2d_valid+visual_aux_checkpoint",
        "quality_gates": ["paired_frame_ledger", "H50_window_coverage", "ADE_2D", "FDE_2D", "PCK", "loss_curve"],
        "limitations": ["Zero current checkpoints.", "Visual retarget labels are not Robot control actions."],
        "successor": "Train only after Robotized visual authority and eligibility gates pass.", "superseded": ["old Kai22 checkpoints not bound to the current contract"],
    },
    {
        "stage": "HumanEgo Policy", "algorithm_id": "blocked_external_real_robot_action",
        "code": ["HumanEgo/tools/train_embodiment.py"], "weights": "ABSENT",
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
            "Robot Visual": "contracts/contact_aware_robot_retarget_v1.schema.json",
            "Occlusion": "contracts/occlusion_compositor_v1.schema.json",
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
    return {
        "schema_version": "chaoyang-current-baseline-registry-v2",
        "governance_revision": authority["governance_revision"],
        "generation_id": authority["generation_id"],
        "generated_at": authority["generated_at"],
        "status": "CURRENT",
        "entries": entries,
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
            "README.md", "AGENTS.md", "THIRD_PARTY_NOTICES.md", "assets", "contracts", "data", "docs",
            "HumanEgo", "pipeline", "systems", "tasks", "tests", "third_party", "tools", "archive", "_run",
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
            "stage_baselines": str(STAGE_DOC_PATH),
            "end_to_end": str(ROOT / "docs/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md"),
            "depth_report": str(ROOT / "docs/reports/depth_accuracy/20260911"),
        },
        "forbidden_top_level_after_cleanup": ["NOW", "DEPTH_ACCURACY_PACKAGE_20260911", "RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md"],
        "claim_limit": "Layout contract only. Historical snapshot paths are resolved through PATH_REDIRECTS.json.",
    }


def build_regression_manifest(authority: Mapping[str, Any]) -> dict[str, Any]:
    tests = (
        ("governance", "tests/test_governance_fact_ledger.py"),
        ("governance", "tests/test_governance_heartbeat_clean_reconcile_v52.py"),
        ("governance", "tests/test_current_baseline_registry_v2.py"),
        ("clean", "tests/test_exact78_clean_wave_attempts_v3.py"),
        ("hawor", "tests/test_hawor_bounded_parameter_successor.py"),
        ("role_mask", "tests/test_run_exact78_role_mask_successor_v3.py"),
        ("object_mask", "tests/test_mask_successor_protocol.py"),
        ("depth_object6d", "tests/test_exact78_depth_object6d_expansion_v2.py"),
        ("object6d", "tests/test_object6d_agent_baseline.py"),
        ("contact", "tests/test_contact_geometry_v2_fixtures.py"),
        ("occlusion", "tests/test_contact_robot_occlusion_v1.py"),
        ("robot", "tests/test_tianji_kai_robot_baseline.py"),
        ("robot", "tests/test_robot_hand_round1_v52.py"),
        ("humanego_aux", "HumanEgo/tests/test_visual_aux_contract_v1.py"),
        ("humanego_aux", "HumanEgo/tests/test_visual_aux_bundle_v52.py"),
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
