from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.lr_per_side_identity import (
    MAX_OBJECT_OVERLAP_OVER_INSTANCE,
    MIN_JOINT_SUPPORT_RATIO,
    MIN_SIDE_MARGIN,
    ROUTE_B_RELEVANT_GATE,
    ROUTE_OF_SELECTOR_EVIDENCE,
    EvidenceRef,
    PerSideIdentityContractError,
    PointPromptEvidence,
    SelectorThresholds,
    SideAuthority,
    SideRawCandidate,
    evaluate_route_b_prompt_identity_gate,
    require_a_prime_gpu_admission,
    select_per_side_identities,
    select_side_identity,
    validate_cpu_preflight_governance,
)


def write_ref(root: Path, name: str, payload: object, kind: str) -> EvidenceRef:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return EvidenceRef(str(path), len(data), hashlib.sha256(data).hexdigest(), kind)


def mask_sha(mask: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(mask, dtype=np.bool_)
    header = f"{contiguous.shape[0]}x{contiguous.shape[1]}:bool:".encode("ascii")
    return hashlib.sha256(header + contiguous.tobytes()).hexdigest()


def candidate(
    root: Path,
    side: str,
    lineage: str,
    offset: int,
    *,
    joint: float = 0.8,
    non_target: float = 0.0,
    wrist: bool = True,
    boundary: bool = True,
    object_overlap: float = 0.0,
    mask: np.ndarray | None = None,
    source_name: str | None = None,
    frame_id: str = "grap_a_cap_004:00232",
    instance_id: int | None = None,
) -> SideRawCandidate:
    if mask is None:
        mask = np.zeros((24, 32), dtype=bool)
        mask[2 + offset : 10 + offset, 3:15] = True
    resolved_instance_id = 100 + offset if instance_id is None else instance_id
    source = write_ref(
        root,
        source_name or f"raw_{side}_{offset}_{resolved_instance_id}.json",
        {
            "source_kind": "SAM_RAW_INSTANCE",
            "frame_id": frame_id,
            "instance_id": resolved_instance_id,
            "mask_sha256": mask_sha(mask),
        },
        "SAM_RAW_INSTANCE",
    )
    return SideRawCandidate(
        side,  # type: ignore[arg-type]
        lineage,
        frame_id,
        source,
        offset,
        resolved_instance_id,
        mask,
        0.9 - 0.01 * offset,
        joint,
        non_target,
        wrist,
        boundary,
        object_overlap,
        float(mask.mean()),
    )


def side_authority(
    root: Path,
    side: str,
    lineage: str,
    *,
    state: str = "AVAILABLE",
    lineage_verified: bool = True,
    quality_verified: bool = True,
    label_independent: bool = True,
) -> SideAuthority:
    payload = {
        "source_kind": "HAWOR_SIDE_AUTHORITY",
        "side": side,
        "lineage_id": lineage,
        "state": state,
        "lineage_verified": lineage_verified,
        "quality_verified": quality_verified,
        "label_independent": label_independent,
    }
    evidence = write_ref(
        root,
        f"authority_{side}_{lineage}_{state}.json",
        payload,
        "HAWOR_SIDE_AUTHORITY",
    )
    return SideAuthority(
        side,  # type: ignore[arg-type]
        lineage,
        state,  # type: ignore[arg-type]
        lineage_verified,
        evidence,
        quality_verified,
        label_independent,
    )


def authorities(root: Path) -> dict[str, SideAuthority]:
    return {
        "left": side_authority(root, "left", "hawor-left-track"),
        "right": side_authority(root, "right", "hawor-right-track"),
    }


def point_prompts(root: Path) -> dict[str, PointPromptEvidence]:
    project_root = Path(__file__).resolve().parents[1]
    policy_data = (project_root / "contracts/hawor_point_quality_policy_v1.json").read_bytes()
    manifest_data = (project_root / "contracts/manifests/verified_source_manifest_v1.json").read_bytes()
    policy_path = root / "authority/hawor_point_quality_policy_v1.json"
    manifest_path = root / "authority/verified_source_manifest_v1.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_bytes(policy_data)
    manifest_path.write_bytes(manifest_data)
    policy_ref = EvidenceRef(
        str(policy_path), len(policy_data), hashlib.sha256(policy_data).hexdigest(),
        "HAWOR_POINT_QUALITY_POLICY",
    )
    manifest_ref = EvidenceRef(
        str(manifest_path), len(manifest_data), hashlib.sha256(manifest_data).hexdigest(),
        "VERIFIED_SOURCE_MANIFEST",
    )
    result: dict[str, PointPromptEvidence] = {}
    for side, xy, quality in (("left", (0.2, 0.7), 0.9), ("right", (0.8, 0.7), 0.91)):
        lineage = f"hawor-{side}-track"
        payload = {
            "source_kind": "HAWOR_POINT_PROMPT",
            "side": side,
            "lineage_id": lineage,
            "normalized_xy": list(xy),
            "quality_value": quality,
            "quality_policy_ref": "hawor-point-quality-v1",
            "quality_policy_sha256": policy_ref.sha256,
            "source_manifest_sha256": manifest_ref.sha256,
            "label_independent": True,
        }
        result[side] = PointPromptEvidence(
            side, lineage, xy, quality, "hawor-point-quality-v1",
            write_ref(root, f"point_{side}.json", payload, "HAWOR_POINT_PROMPT"),
            policy_ref,
            manifest_ref,
        )
    return result  # type: ignore[return-value]


def governance() -> dict[str, object]:
    return {
        "AUTH_TIER": "T1_CPU_PREFLIGHT_ONLY",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "gpu_started": False,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
    }


def gpu_admission() -> dict[str, object]:
    return {
        "route_of_evidence": "A_PRIME_SELECTOR_ONLY",
        "t0_status": "COMPLETE_DECISION_BRANCH_D_PER_SIDE_IDENTITY_REQUIRED_BEFORE_ROUTE_B",
        "t0_decision_branch": "d",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "labels_read_before_run_freeze": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "min_joint_support_ratio": 0.20,
        "min_side_margin": 0.05,
        "max_object_overlap_over_instance": 0.12,
    }


def gpu_refs(root: Path) -> dict[str, EvidenceRef]:
    root.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    t0_data = (project_root / "archive/audits/LR_ASYMMETRY_CAUSAL_T0.json").read_bytes()
    t0 = root / "t0.json"
    t0.write_bytes(t0_data)
    task_data = (project_root / "archive/legacy/task_cards/18_LR_PER_SIDE_IDENTITY_T1.md").read_bytes()
    module_data = (project_root / "pipeline/lr_per_side_identity.py").read_bytes()
    tests_data = (project_root / "tests/test_lr_per_side_identity.py").read_bytes()
    implementation_payload = {
        "schema_version": "lr-per-side-implementation-v1",
        "task_path": "archive/legacy/task_cards/18_LR_PER_SIDE_IDENTITY_T1.md",
        "task_sha256": hashlib.sha256(task_data).hexdigest(),
        "module_path": "pipeline/lr_per_side_identity.py",
        "module_sha256": hashlib.sha256(module_data).hexdigest(),
        "tests_path": "tests/test_lr_per_side_identity.py",
        "tests_sha256": hashlib.sha256(tests_data).hexdigest(),
    }
    implementation_ref = write_ref(
        root, "implementation.json", implementation_payload, "A_PRIME_IMPLEMENTATION"
    )
    qa_payload = {
        "schema_version": "lr-per-side-independent-qa-v3",
        "producer": "independent_cpu_qa",
        "qa_scope": "CPU_ADMISSION_ONLY",
        "route_of_evidence": "A_PRIME_SELECTOR_ONLY",
        "status": "PASS_CPU_ADMISSION_EXACT",
        "p0_findings": 0,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "gpu_started": False,
        "task_sha256": implementation_payload["task_sha256"],
        "module_sha256": implementation_payload["module_sha256"],
        "tests_sha256": implementation_payload["tests_sha256"],
        "implementation_ref_sha256": implementation_ref.sha256,
    }
    refs = {
        "t0_decision": EvidenceRef(
            str(t0), len(t0_data), hashlib.sha256(t0_data).hexdigest(),
            "LR_CAUSAL_T0_JSON",
        ),
        "independent_qa": write_ref(
            root, "INDEPENDENT_QA_LR_PER_SIDE_IDENTITY_T1_V3.json", qa_payload,
            "INDEPENDENT_QA_JSON",
        ),
        "frozen_config": write_ref(root, "config.json", {"frozen": True}, "A_PRIME_FROZEN_CONFIG"),
        "implementation": implementation_ref,
        "input_manifest": write_ref(root, "input.json", {"frames": 15}, "A_PRIME_INPUT_MANIFEST"),
    }
    return refs


def test_route_scopes_are_explicitly_separate() -> None:
    assert ROUTE_OF_SELECTOR_EVIDENCE == "A_PRIME_SELECTOR_ONLY"
    assert ROUTE_B_RELEVANT_GATE == "HAWOR_POINT_PROMPT_IDENTITY_AND_QUALITY_ONLY"


def test_three_numeric_thresholds_are_exactly_frozen() -> None:
    assert MIN_JOINT_SUPPORT_RATIO == 0.20
    assert MIN_SIDE_MARGIN == 0.05
    assert MAX_OBJECT_OVERLAP_OVER_INSTANCE == 0.12
    SelectorThresholds().validate()
    for changed in (
        SelectorThresholds(0.19, 0.05, 0.12),
        SelectorThresholds(0.20, 0.04, 0.12),
        SelectorThresholds(0.20, 0.05, 0.13),
    ):
        with pytest.raises(PerSideIdentityContractError, match="threshold drift"):
            changed.validate()


def test_one_side_outside_holds_only_that_side_without_global_abort(tmp_path: Path) -> None:
    auth = authorities(tmp_path)
    auth["left"] = side_authority(
        tmp_path, "left", "hawor-left-track", state="OUTSIDE_IMAGE"
    )
    right = candidate(tmp_path, "right", "hawor-right-track", 0)
    before = right.mask.copy()
    result = select_per_side_identities(
        auth,  # type: ignore[arg-type]
        {"left": (), "right": (right,)},  # type: ignore[arg-type]
        evidence_root=tmp_path,
    )
    assert result.left.status == "HOLD"
    assert result.left.failure_reason == "SIDE_AUTHORITY_EVIDENCE_OUTSIDE_IMAGE"
    assert result.right.status == "ACCEPT"
    assert result.frame_status == "PARTIAL_SIDE_HOLD"
    assert not result.frame_complete
    np.testing.assert_array_equal(result.right.mask, before)
    np.testing.assert_array_equal(right.mask, before)
    assert result.pixels_created_or_edited == 0
    assert result.union_operations == result.crop_operations == result.fill_operations == 0
    assert result.morphology_operations == 0


def test_left_and_right_candidate_pools_cannot_cross_identity_or_lineage(tmp_path: Path) -> None:
    wrong_side = candidate(tmp_path, "right", "hawor-left-track", 0)
    with pytest.raises(PerSideIdentityContractError, match="crossed pool"):
        select_side_identity(authorities(tmp_path)["left"], (wrong_side,), evidence_root=tmp_path)
    wrong_lineage = candidate(tmp_path, "left", "different-lineage", 0)
    with pytest.raises(PerSideIdentityContractError, match="lineage mismatch"):
        select_side_identity(authorities(tmp_path)["left"], (wrong_lineage,), evidence_root=tmp_path)


def test_aliased_authority_lineage_holds_both_identities_instead_of_guessing(tmp_path: Path) -> None:
    auth = {
        "left": side_authority(tmp_path, "left", "same-track"),
        "right": side_authority(tmp_path, "right", "same-track"),
    }
    result = select_per_side_identities(
        auth,
        {
            "left": (candidate(tmp_path, "left", "same-track", 0),),
            "right": (candidate(tmp_path, "right", "same-track", 1),),
        },
        evidence_root=tmp_path,
    )
    assert result.frame_status == "IDENTITY_HOLD"
    assert result.left.failure_reason == "SIDE_IDENTITY_LINEAGE_UNVERIFIED"
    assert result.right.failure_reason == "SIDE_IDENTITY_LINEAGE_UNVERIFIED"
    assert result.left.mask is None and result.right.mask is None


def test_alias_holds_both_even_if_only_one_side_claims_verified_lineage(tmp_path: Path) -> None:
    auth = {
        "left": side_authority(
            tmp_path, "left", "same-track", lineage_verified=False,
            quality_verified=False, label_independent=False,
        ),
        "right": side_authority(tmp_path, "right", "same-track"),
    }
    result = select_per_side_identities(
        auth,
        {
            "left": (candidate(tmp_path, "left", "same-track", 0),),
            "right": (candidate(tmp_path, "right", "same-track", 1),),
        },
        evidence_root=tmp_path,
    )
    assert result.frame_status == "IDENTITY_HOLD"
    assert result.left.status == result.right.status == "HOLD"


def test_boundary_supported_is_diagnostic_only_and_cannot_reject_valid_instance(tmp_path: Path) -> None:
    recalled = candidate(
        tmp_path, "left", "hawor-left-track", 0, joint=0.81, boundary=False
    )
    decision = select_side_identity(authorities(tmp_path)["left"], (recalled,), evidence_root=tmp_path)
    assert decision.status == "ACCEPT"
    assert decision.failure_reason is None
    assert decision.candidate_audit[0].rejection_reasons == ()
    assert decision.candidate_audit[0].boundary_supported_diagnostic is False
    assert decision.candidate_audit[0].joint_support_ratio == 0.81
    assert decision.candidate_audit[0].side_difference == 0.81


def test_object6d_is_rejection_only_and_raw_pixels_remain_unchanged(tmp_path: Path) -> None:
    raw = candidate(
        tmp_path, "left", "hawor-left-track", 0, object_overlap=0.120001
    )
    before = raw.mask.copy()
    decision = select_side_identity(authorities(tmp_path)["left"], (raw,), evidence_root=tmp_path)
    assert decision.status == "REJECT"
    assert decision.candidate_audit[0].rejection_reasons == (
        "OBJECT6D_PROTECTION_REJECTION",
    )
    assert decision.mask is None
    np.testing.assert_array_equal(raw.mask, before)


def test_prompt_recall_insufficient_is_side_local(tmp_path: Path) -> None:
    weak = candidate(tmp_path, "left", "hawor-left-track", 0, joint=0.19, wrist=False)
    result = select_per_side_identities(
        authorities(tmp_path),  # type: ignore[arg-type]
        {
            "left": (weak,),
            "right": (candidate(tmp_path, "right", "hawor-right-track", 1),),
        },  # type: ignore[arg-type]
        evidence_root=tmp_path,
    )
    assert result.left.failure_reason == "PROMPT_RECALL_INSUFFICIENT"
    assert result.right.status == "ACCEPT"
    assert result.frame_status == "INCOMPLETE"


def test_route_b_prompt_gate_uses_only_hawor_identity_quality_and_provenance(tmp_path: Path) -> None:
    result = evaluate_route_b_prompt_identity_gate(
        point_prompts(tmp_path), evidence_root=tmp_path,  # type: ignore[arg-type]
    )
    assert result.route_b_prompt_identity_ready
    assert result.left.status == result.right.status == "PASS"
    assert result.a_prime_selector_evidence_consumed is False
    assert not hasattr(result, "boundary_supported")
    assert not hasattr(result, "min_joint_support_ratio")


def test_route_b_prompt_gate_holds_alias_and_per_side_quality_without_a_selector(tmp_path: Path) -> None:
    alias = point_prompts(tmp_path)
    right = alias["right"]
    alias["right"] = PointPromptEvidence(
        "right", alias["left"].lineage_id, right.normalized_xy, right.quality_value,
        right.quality_policy_ref, right.evidence, right.quality_policy, right.source_manifest,
    )
    result = evaluate_route_b_prompt_identity_gate(alias, evidence_root=tmp_path)  # type: ignore[arg-type]
    assert not result.route_b_prompt_identity_ready
    assert result.left.hold_reason == result.right.hold_reason == "SIDE_IDENTITY_LINEAGE_ALIAS"

    quality = point_prompts(tmp_path / "quality")
    left = quality["left"]
    quality["left"] = PointPromptEvidence(
        left.identity, left.lineage_id, left.normalized_xy, 2.0,
        left.quality_policy_ref, left.evidence, left.quality_policy, left.source_manifest,
    )
    result = evaluate_route_b_prompt_identity_gate(
        quality, evidence_root=tmp_path / "quality",  # type: ignore[arg-type]
    )
    assert result.left.hold_reason == "POINT_PROMPT_EVIDENCE_PAYLOAD_MISMATCH"
    assert result.right.status == "PASS"


def test_route_b_prompt_gate_holds_only_outside_side(tmp_path: Path) -> None:
    prompts = point_prompts(tmp_path)
    right = prompts["right"]
    prompts["right"] = PointPromptEvidence(
        right.identity, right.lineage_id, (1.1, 0.7), right.quality_value,
        right.quality_policy_ref, right.evidence, right.quality_policy, right.source_manifest,
    )
    result = evaluate_route_b_prompt_identity_gate(
        prompts, evidence_root=tmp_path,  # type: ignore[arg-type]
    )
    assert result.left.status == "PASS"
    assert result.right.hold_reason == "POINT_PROMPT_EVIDENCE_PAYLOAD_MISMATCH"
    assert not result.route_b_prompt_identity_ready


def test_route_b_rejects_self_attested_or_a_prime_selector_evidence(tmp_path: Path) -> None:
    prompts = point_prompts(tmp_path)
    left = prompts["left"]
    prompts["left"] = PointPromptEvidence(
        left.identity, left.lineage_id, left.normalized_xy, left.quality_value,
        left.quality_policy_ref,
        EvidenceRef(
            left.evidence.path, left.evidence.bytes, left.evidence.sha256,
            "A_PRIME_SELECTOR_OUTPUT",
        ),
        left.quality_policy,
        left.source_manifest,
    )
    result = evaluate_route_b_prompt_identity_gate(
        prompts, evidence_root=tmp_path,  # type: ignore[arg-type]
    )
    assert not result.route_b_prompt_identity_ready
    assert result.left.hold_reason == "POINT_PROMPT_SOURCE_KIND_INVALID"


def test_route_b_rejects_zero_quality_and_unverified_policy(tmp_path: Path) -> None:
    prompts = point_prompts(tmp_path)
    left = prompts["left"]
    zero_payload = {
        "source_kind": "HAWOR_POINT_PROMPT",
        "side": "left",
        "lineage_id": left.lineage_id,
        "normalized_xy": list(left.normalized_xy),
        "quality_value": 0.0,
        "quality_policy_ref": left.quality_policy_ref,
        "quality_policy_sha256": left.quality_policy.sha256,
        "source_manifest_sha256": left.source_manifest.sha256,
        "label_independent": True,
    }
    prompts["left"] = PointPromptEvidence(
        "left", left.lineage_id, left.normalized_xy, 0.0,
        left.quality_policy_ref,
        write_ref(tmp_path, "point_left_zero.json", zero_payload, "HAWOR_POINT_PROMPT"),
        left.quality_policy,
        left.source_manifest,
    )
    result = evaluate_route_b_prompt_identity_gate(
        prompts, evidence_root=tmp_path,  # type: ignore[arg-type]
    )
    assert result.left.hold_reason == "POINT_PROMPT_QUALITY_INVALID"
    assert result.right.status == "PASS"

    prompts = point_prompts(tmp_path / "bad_policy")
    left = prompts["left"]
    arbitrary_policy = write_ref(
        tmp_path / "bad_policy", "arbitrary-policy.json",
        {"policy_id": "self-attested", "minimum_quality_exclusive": -1.0},
        "HAWOR_POINT_QUALITY_POLICY",
    )
    prompts["left"] = PointPromptEvidence(
        left.identity, left.lineage_id, left.normalized_xy, left.quality_value,
        left.quality_policy_ref, left.evidence, arbitrary_policy, left.source_manifest,
    )
    result = evaluate_route_b_prompt_identity_gate(
        prompts, evidence_root=tmp_path / "bad_policy",  # type: ignore[arg-type]
    )
    assert result.left.hold_reason == "POINT_PROMPT_QUALITY_POLICY_SHA_MISMATCH"


def test_evidence_ref_rejects_intermediate_directory_symlink(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    ref = write_ref(real_dir, "evidence.json", {"ok": True}, "TEST_EVIDENCE")
    link_dir = tmp_path / "linked"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    linked_ref = EvidenceRef(
        str(link_dir / "evidence.json"), ref.bytes, ref.sha256, ref.source_kind,
    )
    with pytest.raises(PerSideIdentityContractError, match="symlink or non-directory"):
        linked_ref.read_verified(allowed_root=tmp_path)


def test_side_authority_requires_quality_and_label_independent_provenance(tmp_path: Path) -> None:
    for quality_verified, label_independent, reason in (
        (False, True, "SIDE_AUTHORITY_QUALITY_UNVERIFIED"),
        (True, False, "SIDE_AUTHORITY_LABEL_INDEPENDENCE_UNVERIFIED"),
    ):
        authority = side_authority(
            tmp_path, "left", f"lineage-{quality_verified}-{label_independent}",
            quality_verified=quality_verified,
            label_independent=label_independent,
        )
        decision = select_side_identity(authority, (), evidence_root=tmp_path)
        assert decision.status == "HOLD"
        assert decision.failure_reason == reason


def test_same_raw_instance_cannot_be_renamed_into_both_side_pools(tmp_path: Path) -> None:
    mask = np.zeros((24, 32), dtype=bool)
    mask[4:18, 6:22] = True
    left = candidate(
        tmp_path, "left", "hawor-left-track", 0, mask=mask,
        source_name="shared.json", instance_id=701,
    )
    right = candidate(
        tmp_path, "right", "hawor-right-track", 0, mask=mask,
        source_name="shared.json", instance_id=701,
    )
    result = select_per_side_identities(
        authorities(tmp_path),  # type: ignore[arg-type]
        {"left": (left,), "right": (right,)},  # type: ignore[arg-type]
        evidence_root=tmp_path,
    )
    assert result.frame_status == "IDENTITY_HOLD"
    assert result.left.failure_reason == result.right.failure_reason == "RAW_INSTANCE_CROSS_POOL_ALIAS"
    assert result.left.mask is None and result.right.mask is None


def test_cpu_preflight_governance_forbids_advancement_and_overrides() -> None:
    record = governance()
    validate_cpu_preflight_governance(record)
    record["session_overrides"] = {"grap_a_cap_004": {}}
    with pytest.raises(PerSideIdentityContractError, match="forbidden override"):
        validate_cpu_preflight_governance(record)
    record = governance()
    record["gpu_started"] = True
    with pytest.raises(PerSideIdentityContractError, match="gpu_started"):
        validate_cpu_preflight_governance(record)
    record = governance()
    record["gpu_started"] = 0
    with pytest.raises(PerSideIdentityContractError, match="gpu_started"):
        validate_cpu_preflight_governance(record)


def test_gpu_admission_derives_zero_p0_and_really_creates_o_excl_owner(tmp_path: Path) -> None:
    run_parent = tmp_path / "_run"
    run_parent.mkdir()
    refs = gpu_refs(tmp_path / "refs")
    marker = require_a_prime_gpu_admission(
        gpu_admission(), frozen_refs=refs, evidence_root=tmp_path,
        project_run_root=run_parent, run_root=run_parent / "lr_per_side_candidate_v1",
    )
    assert Path(marker.path).is_file()
    assert marker.source_kind == "A_PRIME_RUN_OWNER"
    with pytest.raises(PerSideIdentityContractError, match="already exists"):
        require_a_prime_gpu_admission(
            gpu_admission(), frozen_refs=refs, evidence_root=tmp_path,
            project_run_root=run_parent, run_root=run_parent / "lr_per_side_candidate_v1",
        )


def test_gpu_admission_requires_exact_status_and_live_sha_bindings(tmp_path: Path) -> None:
    run_parent = tmp_path / "_run"
    run_parent.mkdir()
    refs = gpu_refs(tmp_path / "refs_minimal")
    refs["independent_qa"] = write_ref(
        tmp_path / "refs_minimal", "INDEPENDENT_QA_LR_PER_SIDE_IDENTITY_T1_V3.json",
        {"status": "PASS_CPU_ADMISSION_EXACT", "p0_findings": 0},
        "INDEPENDENT_QA_JSON",
    )
    with pytest.raises(PerSideIdentityContractError, match="identity mismatch"):
        require_a_prime_gpu_admission(
            gpu_admission(), frozen_refs=refs, evidence_root=tmp_path,
            project_run_root=run_parent, run_root=run_parent / "lr_per_side_candidate_minimal_qa",
        )

    refs = gpu_refs(tmp_path / "refs")
    qa = json.loads(Path(refs["independent_qa"].path).read_text())
    qa["status"] = "PASS_ANY_ARBITRARY_TEXT"
    refs["independent_qa"] = write_ref(
        tmp_path / "refs", "qa_arbitrary.json", qa, "INDEPENDENT_QA_JSON",
    )
    with pytest.raises(PerSideIdentityContractError, match="status is not exact PASS"):
        require_a_prime_gpu_admission(
            gpu_admission(), frozen_refs=refs, evidence_root=tmp_path,
            project_run_root=run_parent, run_root=run_parent / "lr_per_side_candidate_bad_status",
        )

    refs = gpu_refs(tmp_path / "refs_wrong_sha")
    implementation = json.loads(Path(refs["implementation"].path).read_text())
    implementation["module_sha256"] = "0" * 64
    refs["implementation"] = write_ref(
        tmp_path / "refs_wrong_sha", "implementation_wrong.json", implementation,
        "A_PRIME_IMPLEMENTATION",
    )
    qa = json.loads(Path(refs["independent_qa"].path).read_text())
    qa["implementation_ref_sha256"] = refs["implementation"].sha256
    qa["module_sha256"] = "0" * 64
    refs["independent_qa"] = write_ref(
        tmp_path / "refs_wrong_sha", "INDEPENDENT_QA_LR_PER_SIDE_IDENTITY_T1_V3.json",
        qa, "INDEPENDENT_QA_JSON",
    )
    with pytest.raises(PerSideIdentityContractError, match="implementation/live SHA mismatch"):
        require_a_prime_gpu_admission(
            gpu_admission(), frozen_refs=refs, evidence_root=tmp_path,
            project_run_root=run_parent, run_root=run_parent / "lr_per_side_candidate_wrong_sha",
        )

    bad_refs = gpu_refs(tmp_path / "bad_refs")
    bad_refs["independent_qa"] = write_ref(
        tmp_path / "bad_refs", "qa_bad.json",
        {"status": "HOLD", "p0_findings": 2}, "INDEPENDENT_QA_JSON",
    )
    with pytest.raises(PerSideIdentityContractError, match="P0 is not zero"):
        require_a_prime_gpu_admission(
            gpu_admission(), frozen_refs=bad_refs, evidence_root=tmp_path,
            project_run_root=run_parent, run_root=run_parent / "lr_per_side_candidate_v2",
        )


def test_gpu_admission_rejects_nested_overrides_and_boolean_o_excl_claim(tmp_path: Path) -> None:
    record = gpu_admission()
    record["nested"] = {"session_overrides": {"004": {}}, "hidden_fallback": True}
    record["fresh_o_excl_run"] = True
    with pytest.raises(PerSideIdentityContractError, match="forbidden override"):
        require_a_prime_gpu_admission(
            record, frozen_refs=gpu_refs(tmp_path / "refs"), evidence_root=tmp_path,
            project_run_root=tmp_path, run_root=tmp_path / "lr_per_side_bad",
        )


def test_gpu_admission_rejects_threshold_drift(tmp_path: Path) -> None:
    record = gpu_admission()
    record["min_side_margin"] = 0.049
    with pytest.raises(PerSideIdentityContractError, match="threshold drift"):
        require_a_prime_gpu_admission(
            record, frozen_refs=gpu_refs(tmp_path / "refs"), evidence_root=tmp_path,
            project_run_root=tmp_path, run_root=tmp_path / "lr_per_side_bad",
        )


def test_a_prime_gpu_admission_rejects_route_b_evidence_substitution(tmp_path: Path) -> None:
    record = gpu_admission()
    record["route_of_evidence"] = "ROUTE_B_HAWOR_POINT_PROMPT"
    with pytest.raises(PerSideIdentityContractError, match="route_of_evidence"):
        require_a_prime_gpu_admission(
            record, frozen_refs=gpu_refs(tmp_path / "refs"), evidence_root=tmp_path,
            project_run_root=tmp_path, run_root=tmp_path / "lr_per_side_bad",
        )


def test_exact_left_right_mapping_is_required(tmp_path: Path) -> None:
    with pytest.raises(PerSideIdentityContractError, match="exact left/right"):
        select_per_side_identities(
            {"left": authorities(tmp_path)["left"]},  # type: ignore[arg-type]
            {"left": ()},  # type: ignore[arg-type]
            evidence_root=tmp_path,
        )
