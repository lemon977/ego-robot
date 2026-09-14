from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from pipeline.lr_distributed_side_evidence import (
    FIXED_FRAME_INDICES,
    GPU_EXECUTION_AUTHORIZED,
    MAX_OBJECT_OVERLAP_OVER_INSTANCE,
    MIN_JOINT_SUPPORT_RATIO,
    MIN_SIDE_MARGIN,
    PRODUCER_ID,
    ROUTE_OF_EVIDENCE,
    SIDES,
    TASK26_SHA256,
    TEXT_PROMPT,
    DistributedSideEvidenceError,
    EvidenceRef,
    FrameSelection,
    JointSupportEvidence,
    RawInstance,
    SelectorThresholds,
    SideAuthority,
    SideDecision,
    canonical_json,
    check_wrong_side_acceptance,
    claim_cpu_audit_root,
    extract_wrist_review_slice,
    mask_sha256,
    measure_in_image_joint_support,
    require_gpu_execution,
    resolve_cpu_preflight_status,
    select_frame,
    summarize_frames,
    validate_cpu_config,
    validate_cpu_governance,
    validate_input_covariate_freeze,
)


def write_ref(root: Path, relative: str, payload: object, source_kind: str) -> EvidenceRef:
    data = canonical_json(payload)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return EvidenceRef(str(path), len(data), hashlib.sha256(data).hexdigest(), source_kind)


def authority_payload(
    frame_index: int,
    side: str,
    lineage: str,
    state: str,
    wrist_xy: tuple[float, float] | None,
) -> dict[str, object]:
    return {
        "identity": side,
        "lineage_id": lineage,
        "state": state,
        "wrist_xy": None if wrist_xy is None else list(wrist_xy),
    }


def authorities(
    root: Path,
    *,
    session_id: str = "grap_a_cap_005",
    frame_index: int = 120,
    left_state: str = "AVAILABLE",
    right_state: str = "AVAILABLE",
    left_wrist: tuple[float, float] | None = (6.0, 8.0),
    right_wrist: tuple[float, float] | None = (25.0, 8.0),
) -> dict[str, SideAuthority]:
    lineage = {"left": f"{session_id}:left", "right": f"{session_id}:right"}
    payload = {
        "frame_index": frame_index,
        "sides": {
            "left": {
                "authority": authority_payload(
                    frame_index, "left", lineage["left"], left_state, left_wrist
                )
            },
            "right": {
                "authority": authority_payload(
                    frame_index, "right", lineage["right"], right_state, right_wrist
                )
            },
        },
    }
    source = write_ref(
        root,
        f"source/{session_id}_{frame_index}.json",
        payload,
        "A_PRIME_V3_FRAME_SELECTION",
    )
    return {
        "left": SideAuthority(
            "left",
            session_id,
            frame_index,
            lineage["left"],
            left_state,  # type: ignore[arg-type]
            left_wrist,
            source,
        ),
        "right": SideAuthority(
            "right",
            session_id,
            frame_index,
            lineage["right"],
            right_state,  # type: ignore[arg-type]
            right_wrist,
            source,
        ),
    }


def support(
    supported: int,
    total: int = 10,
    *,
    wrist: bool = True,
    radius: float = 2.0,
) -> JointSupportEvidence:
    return JointSupportEvidence(total, supported, supported / total if total else 0.0, radius, wrist)


def raw(
    root: Path,
    offset: int,
    *,
    session_id: str = "grap_a_cap_005",
    frame_index: int = 120,
    left: JointSupportEvidence | None = None,
    right: JointSupportEvidence | None = None,
    object_overlap: float = 0.0,
    boundary_left: bool = False,
    boundary_right: bool = False,
    mask: np.ndarray | None = None,
) -> RawInstance:
    if mask is None:
        mask = np.zeros((24, 32), dtype=np.bool_)
        mask[3 + offset : 12 + offset, 4:18] = True
    instance_id = 100 + offset
    source = write_ref(
        root,
        f"raw/{session_id}_{frame_index}_{offset}.json",
        {
            "source_kind": "SAM_RAW_INSTANCE",
            "frame_id": f"{session_id}:{frame_index}",
            "instance_id": instance_id,
            "mask_sha256": mask_sha256(mask),
        },
        "SAM_RAW_INSTANCE",
    )
    return RawInstance(
        session_id,
        frame_index,
        offset,
        instance_id,
        source,
        mask,
        0.9 - 0.01 * offset,
        {
            "left": support(8) if left is None else left,
            "right": support(0) if right is None else right,
        },
        {"left": boundary_left, "right": boundary_right},
        object_overlap,
        float(mask.mean()),
    )


def cpu_config() -> dict[str, object]:
    return {
        "schema_version": "a-prime-distributed-side-cpu-config-v1",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "producer_id": PRODUCER_ID,
        "prompt": TEXT_PROMPT,
        "wrist_supported_role": "DIAGNOSTIC_ONLY",
        "authority_wrist_outside_image_role": "DIAGNOSTIC_ONLY",
        "side_routing_basis": "IN_IMAGE_HAWOR_JOINT_SUPPORT_DISTRIBUTION_ONLY",
        "min_joint_support_ratio": 0.20,
        "min_side_margin": 0.05,
        "max_object_overlap_over_instance": 0.12,
        "boundary_supported_role": "DIAGNOSTIC_ONLY",
        "object6d_role": "REJECTION_PROTECTION_ONLY",
        "raw_pixel_semantics": "COPY_ONE_UNMODIFIED_RAW_SAM_INSTANCE_PER_ACCEPTED_SIDE",
        "forbidden_pixel_operations": [
            "union",
            "fill",
            "crop",
            "hard_crop",
            "morphology",
            "object6d_subtraction",
        ],
        "fixed_frame_indices": {
            session: list(indices) for session, indices in FIXED_FRAME_INDICES.items()
        },
        "forbidden_session": "grap_a_cap_025",
    }


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
        "auditor_outputs_read": 0,
    }


def accepted_decision(side: str, offset: int, *, outside: bool = False) -> SideDecision:
    return SideDecision(
        side,  # type: ignore[arg-type]
        f"lineage-{side}",
        "OUTSIDE_IMAGE" if outside else "AVAILABLE",
        "ACCEPT",
        offset,
        100 + offset,
        np.zeros((2, 2), dtype=np.bool_),
        None,
        outside,
        (),
    )


def rejected_decision(side: str) -> SideDecision:
    return SideDecision(
        side,  # type: ignore[arg-type]
        f"lineage-{side}",
        "AVAILABLE",
        "REJECT",
        None,
        None,
        None,
        "NO_INSTANCE_SUPPORTS_SIDE",
        False,
        (),
    )


def test_exact_route_prompt_and_thresholds_are_frozen() -> None:
    assert ROUTE_OF_EVIDENCE == "A_PRIME_SELECTOR_ONLY"
    assert TEXT_PROMPT == "an arm"
    assert (MIN_JOINT_SUPPORT_RATIO, MIN_SIDE_MARGIN, MAX_OBJECT_OVERLAP_OVER_INSTANCE) == (
        0.20,
        0.05,
        0.12,
    )
    SelectorThresholds().validate()
    for changed in (
        SelectorThresholds(0.19, 0.05, 0.12),
        SelectorThresholds(0.20, 0.049, 0.12),
        SelectorThresholds(0.20, 0.05, 0.121),
    ):
        with pytest.raises(DistributedSideEvidenceError, match="threshold drift"):
            changed.validate()


def test_joint_support_uses_only_in_image_joint_distribution() -> None:
    mask = np.zeros((20, 30), dtype=np.bool_)
    joints = np.asarray([[6.0 + index % 5, 6.0 + index // 5] for index in range(21)])
    joints[0] = (6.0, 25.0)  # the authority wrist is outside the image
    for x, y in joints[1:]:
        mask[int(y), int(x)] = True
    result = measure_in_image_joint_support(mask, joints, frozen_hand_scale_px=20.0)
    assert result.in_image_joint_count == 20
    assert result.supported_in_image_joint_count == 20
    assert result.joint_support_ratio == 1.0
    assert result.wrist_supported_diagnostic is False


def test_s1_wrist_is_diagnostic_only_and_boundary_is_diagnostic_only(tmp_path: Path) -> None:
    candidate = raw(
        tmp_path,
        0,
        left=support(9, wrist=False),
        right=support(0),
        boundary_left=False,
    )
    before = candidate.mask.copy()
    result = select_frame(
        authorities(tmp_path),
        (candidate,),
        evidence_root=tmp_path,
        image_shape=(24, 32),
    )
    assert result.left.status == "ACCEPT"
    audit = next(audit for audit in result.left.candidate_audit if audit.considered_for_side)
    assert audit.wrist_supported_diagnostic is False
    assert audit.boundary_supported_diagnostic is False
    assert audit.rejection_reasons == ()
    np.testing.assert_array_equal(result.left.mask, before)
    np.testing.assert_array_equal(candidate.mask, before)


def test_s2_outside_wrist_is_diagnostic_only_and_cannot_abort_other_side(tmp_path: Path) -> None:
    auth = authorities(
        tmp_path,
        left_state="OUTSIDE_IMAGE",
        left_wrist=(6.0, 30.0),
    )
    left = raw(tmp_path, 0, left=support(9, wrist=False), right=support(0))
    right = raw(tmp_path, 1, left=support(0), right=support(8))
    result = select_frame(auth, (left, right), evidence_root=tmp_path, image_shape=(24, 32))
    assert result.left.status == result.right.status == "ACCEPT"
    assert result.left.authority_wrist_outside_image_diagnostic is True
    assert result.right.authority_wrist_outside_image_diagnostic is False
    assert result.frame_complete
    assert result.pixels_created_or_edited == 0
    assert result.union_operations == result.crop_operations == result.fill_operations == 0
    assert result.morphology_operations == result.object6d_subtraction_operations == 0


def test_missing_authority_still_holds_only_that_side(tmp_path: Path) -> None:
    auth = authorities(tmp_path, left_state="MISSING", left_wrist=None)
    right = raw(tmp_path, 0, left=support(0), right=support(8))
    result = select_frame(auth, (right,), evidence_root=tmp_path, image_shape=(24, 32))
    assert result.left.status == "HOLD"
    assert result.left.failure_reason == "SIDE_AUTHORITY_EVIDENCE_MISSING"
    assert result.right.status == "ACCEPT"
    assert result.frame_status == "PARTIAL_SIDE_HOLD"


def test_real_failure_taxonomy_does_not_reuse_prompt_recall_label(tmp_path: Path) -> None:
    no_support = select_frame(
        authorities(tmp_path),
        (raw(tmp_path, 0, left=support(0), right=support(8)),),
        evidence_root=tmp_path,
        image_shape=(24, 32),
    )
    assert no_support.left.failure_reason == "NO_INSTANCE_SUPPORTS_SIDE"

    weak_root = tmp_path / "weak"
    weak = select_frame(
        authorities(weak_root),
        (raw(weak_root, 0, left=support(4, 21), right=support(0)),),
        evidence_root=weak_root,
        image_shape=(24, 32),
    )
    assert weak.left.failure_reason == "WEAK_IN_IMAGE_JOINT_SUPPORT"

    alias_root = tmp_path / "alias"
    alias = select_frame(
        authorities(alias_root),
        (raw(alias_root, 0, left=support(7), right=support(10)),),
        evidence_root=alias_root,
        image_shape=(24, 32),
    )
    assert alias.left.failure_reason == "AUTHORITY_SIDE_ALIAS"
    assert alias.right.status == "ACCEPT"

    object_root = tmp_path / "object"
    protected = select_frame(
        authorities(object_root),
        (raw(object_root, 0, left=support(8), right=support(0), object_overlap=0.121),),
        evidence_root=object_root,
        image_shape=(24, 32),
    )
    assert protected.left.failure_reason == "OBJECT6D_PROTECTION_REJECTION"
    all_reasons = {
        no_support.left.failure_reason,
        weak.left.failure_reason,
        alias.left.failure_reason,
        protected.left.failure_reason,
    }
    assert "PROMPT_RECALL_INSUFFICIENT" not in all_reasons


def test_side_routing_ignores_wrist_diagnostic_and_uses_joint_distribution(tmp_path: Path) -> None:
    candidate = raw(
        tmp_path,
        0,
        left=support(7, wrist=True),
        right=support(9, wrist=False),
    )
    result = select_frame(
        authorities(tmp_path),
        (candidate,),
        evidence_root=tmp_path,
        image_shape=(24, 32),
    )
    assert result.routing_audit[0].assigned_pool == "right"
    assert result.left.failure_reason == "AUTHORITY_SIDE_ALIAS"
    assert result.right.status == "ACCEPT"


def test_wrong_side_acceptance_is_hard_fail_independent_of_wrist_and_oob_fields() -> None:
    available = FrameSelection(
        "grap_a_cap_005",
        220,
        accepted_decision("left", 1),
        rejected_decision("right"),
        False,
        "INCOMPLETE",
        (),
    )
    outside = replace(
        available,
        frame_index=270,
        left=accepted_decision("left", 1, outside=True),
    )
    checks = check_wrong_side_acceptance((available, outside))
    assert [check.status for check in checks] == [
        "FAIL_WRONG_SIDE_ACCEPTANCE",
        "FAIL_WRONG_SIDE_ACCEPTANCE",
    ]
    assert all(check.expected_side == "right" for check in checks)
    assert (
        resolve_cpu_preflight_status(
            wrong_side_failed=True,
            development_frame_passed=15,
            preregistered_arithmetic_matches=True,
        )
        == "FAIL_WRONG_SIDE_ACCEPTANCE"
    )


def test_005_alias_sentinels_allow_rejection_or_right_acceptance_but_never_left() -> None:
    frames = (
        FrameSelection(
            "grap_a_cap_005",
            220,
            rejected_decision("left"),
            accepted_decision("right", 1),
            False,
            "INCOMPLETE",
            (),
        ),
        FrameSelection(
            "grap_a_cap_005",
            270,
            rejected_decision("left"),
            rejected_decision("right"),
            False,
            "INCOMPLETE",
            (),
        ),
    )
    checks = check_wrong_side_acceptance(frames)
    assert all(check.status == "PASS" for check in checks)


def test_authority_lineage_alias_holds_both_sides(tmp_path: Path) -> None:
    auth = authorities(tmp_path)
    auth["right"] = replace(auth["right"], lineage_id=auth["left"].lineage_id)
    result = select_frame(auth, (), evidence_root=tmp_path, image_shape=(24, 32))
    assert result.left.status == result.right.status == "HOLD"
    assert result.left.failure_reason == result.right.failure_reason == "SIDE_IDENTITY_LINEAGE_ALIAS"
    assert result.frame_status == "IDENTITY_HOLD"


def test_raw_source_digest_and_mask_lineage_are_mandatory(tmp_path: Path) -> None:
    candidate = raw(tmp_path, 0)
    changed = candidate.mask.copy()
    changed[0, 0] = True
    with pytest.raises(DistributedSideEvidenceError, match="raw source/mask payload mismatch"):
        select_frame(
            authorities(tmp_path),
            (replace(candidate, mask=changed),),
            evidence_root=tmp_path,
            image_shape=(24, 32),
        )


def test_exact_41x41_center_and_edge_review_are_never_algorithm_input() -> None:
    mask = np.zeros((50, 60), dtype=np.bool_)
    mask[20:30, 20:30] = True
    center = extract_wrist_review_slice(mask, (25.0, 25.0), authority_state="AVAILABLE")
    edge = extract_wrist_review_slice(mask, (30.0, 70.0), authority_state="OUTSIDE_IMAGE")
    assert center.pixels.shape == edge.pixels.shape == (41, 41)
    assert center.anchor_kind == "AUTHORITY_WRIST_CENTER"
    assert edge.anchor_kind == "NEAREST_VALID_IMAGE_EDGE"
    assert edge.anchor_xy == (30, 49)
    assert center.algorithm_consumed is edge.algorithm_consumed is False
    assert center.authority_wrist_repaired is edge.authority_wrist_repaired is False


def covariate_freeze() -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for session_id, indices in FIXED_FRAME_INDICES.items():
        for frame_index in indices:
            frames.append(
                {
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "joint_source": {
                        "path": f"geometry/{session_id}.npz",
                        "bytes": 100,
                        "sha256": "a" * 64,
                    },
                    "left": {
                        "source_track_index": 0,
                        "validity_state": "VALID_IN_IMAGE_WRIST",
                        "wrist_xy": [10.0, 20.0],
                        "hand_scale_pixels": 100.0,
                    },
                    "right": {
                        "source_track_index": 1,
                        "validity_state": "VALID_IN_IMAGE_WRIST",
                        "wrist_xy": [30.0, 20.0],
                        "hand_scale_pixels": 110.0,
                    },
                    "wrist_pixel_distance": 20.0,
                    "left_over_right_hand_scale_ratio": 100.0 / 110.0,
                }
            )
    return {
        "schema_version": "a-prime-input-covariates-freeze-v1",
        "document_status": "FROZEN_PURE_INPUT_NUMBERS_NO_ANALYSIS",
        "task26_sha256": TASK26_SHA256,
        "created_before_selector_replay": True,
        "analysis_performed": False,
        "frame_count": 39,
        "frames": frames,
    }


def test_all_39_input_covariates_are_frozen_without_analysis() -> None:
    record = covariate_freeze()
    validate_input_covariate_freeze(record)
    record["analysis_performed"] = True
    with pytest.raises(DistributedSideEvidenceError, match="analysis_performed"):
        validate_input_covariate_freeze(record)


def test_input_covariate_freeze_rejects_missing_frame_or_source_sha() -> None:
    record = covariate_freeze()
    record["frames"] = record["frames"][:-1]  # type: ignore[index]
    with pytest.raises(DistributedSideEvidenceError, match="39 frames"):
        validate_input_covariate_freeze(record)
    record = covariate_freeze()
    record["frames"][0]["joint_source"]["sha256"] = "bad"  # type: ignore[index]
    with pytest.raises(DistributedSideEvidenceError, match="source ref invalid"):
        validate_input_covariate_freeze(record)


def test_cpu_config_governance_and_gpu_hold_are_fail_closed() -> None:
    validate_cpu_config(cpu_config())
    validate_cpu_governance(governance())
    changed = cpu_config()
    changed["min_side_margin"] = 0.049
    with pytest.raises(DistributedSideEvidenceError, match="exact archive/legacy/task_cards/26 freeze"):
        validate_cpu_config(changed)
    changed = governance()
    changed["gpu_started"] = True
    with pytest.raises(DistributedSideEvidenceError, match="governance drift"):
        validate_cpu_governance(changed)
    assert GPU_EXECUTION_AUTHORIZED is False
    with pytest.raises(DistributedSideEvidenceError, match="not authorized"):
        require_gpu_execution()


def test_cpu_audit_root_and_owner_are_true_o_excl_and_task26_digest_bound(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "archive/audits").mkdir(parents=True)
    task = project / "archive/legacy/task_cards" / "26_APRIME_DISTRIBUTED_SIDE_EVIDENCE_T1.md"
    task.parent.mkdir(parents=True)
    source_task = Path(__file__).resolve().parents[1] / "archive/legacy/task_cards/26_APRIME_DISTRIBUTED_SIDE_EVIDENCE_T1.md"
    task.write_bytes(source_task.read_bytes())
    output = project / "archive/audits/lr_distributed_side_evidence_cpu_test"
    marker = claim_cpu_audit_root(output, project_root=project, freeze_payload={"frozen": True})
    assert Path(marker.path).is_file()
    assert marker.source_kind == "A_PRIME_DISTRIBUTED_SIDE_CPU_OWNER"
    with pytest.raises(DistributedSideEvidenceError, match="already exists"):
        claim_cpu_audit_root(output, project_root=project, freeze_payload={"frozen": True})


def test_evidence_ref_rejects_intermediate_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    ref = write_ref(real, "value.json", {"ok": True}, "TEST")
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    link_ref = replace(ref, path=str(linked / "value.json"))
    with pytest.raises(DistributedSideEvidenceError, match="symlink or non-directory"):
        link_ref.read_verified(allowed_root=tmp_path)


def test_side_and_frame_denominators_are_both_reported() -> None:
    complete = FrameSelection(
        "s",
        1,
        accepted_decision("left", 0),
        accepted_decision("right", 1),
        True,
        "COMPLETE",
        (),
    )
    partial = FrameSelection(
        "s",
        2,
        accepted_decision("left", 0),
        rejected_decision("right"),
        False,
        "INCOMPLETE",
        (),
    )
    summary = summarize_frames((complete, partial))
    assert summary["frame_passed"] == 1 and summary["frame_total"] == 2
    assert summary["side_passed"] == 3 and summary["side_total"] == 4
    assert summary["left_passed"] == 2 and summary["right_passed"] == 1


def test_exact_side_inventory_constant_contains_no_025() -> None:
    assert tuple(SIDES) == ("left", "right")
    assert sum(len(indices) for indices in FIXED_FRAME_INDICES.values()) == 39
    assert "grap_a_cap_025" not in FIXED_FRAME_INDICES
