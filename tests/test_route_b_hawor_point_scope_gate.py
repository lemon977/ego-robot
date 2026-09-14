from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

import pipeline.route_b_hawor_point_scope_gate as gate_module
from pipeline.route_b_hawor_point_scope_gate import (
    ASSET_PIN_SHA256,
    QUALITY_POLICY_SHA256,
    ROUTE_OF_EVIDENCE,
    SOURCE_MANIFEST_SHA256,
    U_SEMANTICS,
    EvidenceRef,
    HaworPointPrompt,
    RouteBPointScopeContractError,
    ScopeObservation,
    aggregate_scope_panel,
    evaluate_hawor_point_identity_gate,
    evaluate_scope_observation,
    implementation_inventory,
    require_bytecode_disabled,
    require_route_b_gpu_admission,
    validate_cpu_preflight_governance,
    validate_hyperparameter_freeze,
    validate_input_freeze,
    verify_sam21_asset_pin,
)


PROJECT = Path(__file__).resolve().parents[1]


def bytes_ref(path: Path, payload: bytes, kind: str) -> EvidenceRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return EvidenceRef(str(path), len(payload), hashlib.sha256(payload).hexdigest(), kind)


def json_ref(path: Path, payload: object, kind: str) -> EvidenceRef:
    return bytes_ref(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        kind,
    )


def authority_refs(root: Path) -> tuple[EvidenceRef, EvidenceRef]:
    policy_path = root / "contracts/hawor_point_quality_policy_v1.json"
    source_path = root / "contracts/manifests/verified_source_manifest_v1.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT / "contracts/hawor_point_quality_policy_v1.json", policy_path)
    shutil.copyfile(PROJECT / "contracts/manifests/verified_source_manifest_v1.json", source_path)
    policy = EvidenceRef(
        str(policy_path), policy_path.stat().st_size, QUALITY_POLICY_SHA256,
        "HAWOR_POINT_QUALITY_POLICY",
    )
    source = EvidenceRef(
        str(source_path), source_path.stat().st_size, SOURCE_MANIFEST_SHA256,
        "VERIFIED_SOURCE_MANIFEST",
    )
    return policy, source


def prompt_base(
    root: Path,
    side: str,
    *,
    state: str = "AVAILABLE",
    lineage: str | None = None,
    track: str | None = None,
    xy: tuple[float, float] | None = None,
    quality: float | None = 0.9,
    frame: int = 228,
    session: str = "grap_a_cap_004",
    source_track_index: int | None = None,
    hawor_source_state: str = "observed_gated_smoothed",
    hawor_source_valid: bool = True,
    hawor_measurement_accepted: bool = True,
) -> HaworPointPrompt:
    lineage = lineage or f"hawor-{side}-lineage"
    track = track or f"hawor-{side}-track"
    source_slot = 0 if side == "left" else 1
    if source_track_index is None and state not in {"MISSING", "UNVERIFIED_IDENTITY"}:
        source_track_index = source_slot
    hawor_source = bytes_ref(
        root / f"hawor/{side}_frozen_geometry.npz", b"synthetic-frozen-hawor-geometry",
        "FROZEN_HAWOR_GEOMETRY",
    )
    if xy is None and state == "AVAILABLE":
        xy = (0.25, 0.55) if side == "left" else (0.75, 0.55)
    if state == "OUTSIDE_IMAGE" and xy is None:
        xy = (0.25, 1.05) if side == "left" else (0.75, 1.05)
    if state in {"MISSING", "UNVERIFIED_IDENTITY"}:
        xy = None
        quality = None
    placeholder = bytes_ref(
        root / "unsealed" / f"{side}_{frame}.json",
        b"UNSEALED_CPU_TEST_POINT",
        "HAWOR_ROUTE_B_POINT_PROMPT",
    )
    return HaworPointPrompt(
        side, state, session, frame, lineage, track, xy, quality,
        source_slot, source_track_index, hawor_source_state, hawor_source_valid,
        hawor_measurement_accepted, hawor_source, placeholder,
    )


def point_payload(prompt: HaworPointPrompt, authority_sha: str) -> dict[str, object]:
    return {
        "schema_version": "route-b-hawor-point-v2",
        "source_kind": "HAWOR_ROUTE_B_POINT_PROMPT",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "session_id": prompt.session_id,
        "frame_index": prompt.frame_index,
        "side": prompt.side,
        "state": prompt.state,
        "lineage_id": prompt.lineage_id,
        "track_id": prompt.track_id,
        "source_slot": prompt.source_slot,
        "source_track_index": prompt.source_track_index,
        "hawor_source_state": prompt.hawor_source_state,
        "hawor_source_valid": prompt.hawor_source_valid,
        "hawor_measurement_accepted": prompt.hawor_measurement_accepted,
        "hawor_source": {
            "bytes": prompt.hawor_source.bytes,
            "sha256": prompt.hawor_source.sha256,
        },
        "normalized_xy": None if prompt.normalized_xy is None else [
            float(prompt.normalized_xy[0]), float(prompt.normalized_xy[1])
        ],
        "quality_value": None if prompt.quality_value is None else float(prompt.quality_value),
        "quality_policy_sha256": QUALITY_POLICY_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "hawor_authority_manifest_sha256": authority_sha,
        "label_independent": True,
        "route_a_selector_evidence_consumed": False,
    }


def make_authority(
    root: Path, prompts: dict[str, HaworPointPrompt]
) -> EvidenceRef:
    entries = []
    for prompt in prompts.values():
        if prompt.state not in {"AVAILABLE", "OUTSIDE_IMAGE"}:
            continue
        entries.append(
            {
                "session_id": prompt.session_id,
                "frame_index": prompt.frame_index,
                "side": prompt.side,
                "state": prompt.state,
                "lineage_id": prompt.lineage_id,
                "track_id": prompt.track_id,
                "source_slot": prompt.source_slot,
                "source_track_index": prompt.source_track_index,
                "hawor_source_state": prompt.hawor_source_state,
                "hawor_source_valid": prompt.hawor_source_valid,
                "hawor_measurement_accepted": prompt.hawor_measurement_accepted,
                "hawor_source_ref": prompt.hawor_source.as_dict(),
                "normalized_xy": None if prompt.normalized_xy is None else [
                    float(prompt.normalized_xy[0]), float(prompt.normalized_xy[1])
                ],
                "quality_value": prompt.quality_value,
            }
        )
    payload = {
        "schema_version": "route-b-hawor-authority-manifest-v4",
        "document_status": "FROZEN_UPSTREAM_HAWOR_POINT_AUTHORITY",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "label_independent": True,
        "entries": entries,
    }
    ref = json_ref(
        root / gate_module.HAWOR_AUTHORITY_MANIFEST_PATH,
        payload,
        "HAWOR_AUTHORITY_MANIFEST",
    )
    return replace(ref, path=gate_module.HAWOR_AUTHORITY_MANIFEST_PATH)


def seal_prompts(
    root: Path,
    prompts: dict[str, HaworPointPrompt],
    authority: EvidenceRef,
    *,
    payload_extras: dict[str, dict[str, object]] | None = None,
) -> dict[str, HaworPointPrompt]:
    sealed = {}
    for side, prompt in prompts.items():
        payload = point_payload(prompt, authority.sha256)
        if payload_extras and side in payload_extras:
            payload.update(payload_extras[side])
        ref = json_ref(
            root / "points" / side / f"{side}_{prompt.frame_index}.json",
            payload,
            "HAWOR_ROUTE_B_POINT_PROMPT",
        )
        sealed[side] = replace(prompt, evidence=ref)
    return sealed


def point_bundle(
    root: Path,
    *,
    left: HaworPointPrompt | None = None,
    right: HaworPointPrompt | None = None,
    payload_extras: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, HaworPointPrompt], EvidenceRef]:
    prompts = {
        "left": left or prompt_base(root, "left"),
        "right": right or prompt_base(root, "right"),
    }
    authority = make_authority(root, prompts)
    gate_module.HAWOR_AUTHORITY_MANIFEST_SHA256 = authority.sha256
    return seal_prompts(root, prompts, authority, payload_extras=payload_extras), authority


def evaluate(
    root: Path,
    prompts: dict[str, HaworPointPrompt],
    authority: EvidenceRef,
):
    policy, source = authority_refs(root)
    return evaluate_hawor_point_identity_gate(
        prompts,  # type: ignore[arg-type]
        quality_policy_ref=policy,
        source_manifest_ref=source,
        hawor_authority_manifest_ref=authority,
        evidence_root=root,
    )


def test_exact_independent_left_right_points_pass(tmp_path: Path) -> None:
    prompts, authority = point_bundle(tmp_path)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.status == result.right.status == "PASS"
    assert result.route_b_point_identity_ready is True


@pytest.mark.parametrize("outside_side", ["left", "right"])
def test_one_side_outside_never_aborts_other_side(tmp_path: Path, outside_side: str) -> None:
    bases = {
        "left": prompt_base(tmp_path, "left"),
        "right": prompt_base(tmp_path, "right"),
    }
    bases[outside_side] = prompt_base(tmp_path, outside_side, state="OUTSIDE_IMAGE")
    prompts, authority = point_bundle(tmp_path, left=bases["left"], right=bases["right"])
    result = evaluate(tmp_path, prompts, authority)
    other = "right" if outside_side == "left" else "left"
    assert getattr(result, outside_side).reason == "POINT_OUTSIDE_IMAGE"
    assert getattr(result, other).status == "PASS"


@pytest.mark.parametrize(
    ("state", "reason"),
    [("MISSING", "POINT_MISSING"), ("UNVERIFIED_IDENTITY", "POINT_IDENTITY_UNVERIFIED")],
)
def test_one_side_missing_or_unverified_is_isolated(
    tmp_path: Path, state: str, reason: str
) -> None:
    prompts, authority = point_bundle(
        tmp_path, left=prompt_base(tmp_path, "left", state=state)
    )
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.status == "HOLD" and result.left.reason == reason
    assert result.right.status == "PASS"


def test_available_null_track_is_side_hold(tmp_path: Path) -> None:
    prompts, authority = point_bundle(tmp_path)
    left = replace(prompts["left"], source_track_index=None)
    left_ref = json_ref(
        tmp_path / "points/left/null_track.json",
        point_payload(left, authority.sha256),
        "HAWOR_ROUTE_B_POINT_PROMPT",
    )
    prompts["left"] = replace(left, evidence=left_ref)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == "POINT_SOURCE_TRACK_MISSING_OR_INVALID"
    assert result.right.status == "PASS"


def test_arbitrary_nonmanifest_hawor_source_is_side_hold(tmp_path: Path) -> None:
    prompts, authority = point_bundle(tmp_path)
    attacker = bytes_ref(
        tmp_path / "hawor/attacker.npz", b"attacker-source", "FROZEN_HAWOR_GEOMETRY"
    )
    left = replace(prompts["left"], hawor_source=attacker)
    left_ref = json_ref(
        tmp_path / "points/left/attacker.json",
        point_payload(left, authority.sha256),
        "HAWOR_ROUTE_B_POINT_PROMPT",
    )
    prompts["left"] = replace(left, evidence=left_ref)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == "POINT_PINNED_HAWOR_MEMBERSHIP_MISMATCH"
    assert result.right.status == "PASS"


def test_authority_member_source_tamper_holds_only_that_side(tmp_path: Path) -> None:
    prompts, authority = point_bundle(tmp_path)
    Path(prompts["left"].hawor_source.path).write_bytes(b"tampered")
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == "POINT_SIDE_VERIFICATION_FAILED:RouteBPointScopeContractError"
    assert result.right.status == "PASS"


def test_cross_frame_and_cross_session_pairs_are_rejected(tmp_path: Path) -> None:
    cases = [(229, "grap_a_cap_004"), (228, "grap_a_cap_005")]
    for index, (frame, session) in enumerate(cases):
        root = tmp_path / f"case_{index}"
        prompts, authority = point_bundle(
            root,
            left=prompt_base(root, "left"),
            right=prompt_base(root, "right", frame=frame, session=session),
        )
        result = evaluate(root, prompts, authority)
        assert result.left.reason == result.right.reason == "LEFT_RIGHT_POINT_PAIR_FRAME_MISMATCH"
        assert result.route_b_point_identity_ready is False


@pytest.mark.parametrize("alias_kind", ["lineage", "track", "source_track", "evidence"])
def test_left_right_alias_fails_both_sides(tmp_path: Path, alias_kind: str) -> None:
    left = prompt_base(tmp_path, "left")
    right = prompt_base(tmp_path, "right")
    if alias_kind == "lineage":
        right = replace(right, lineage_id=left.lineage_id)
    elif alias_kind == "track":
        right = replace(right, track_id=left.track_id)
    elif alias_kind == "source_track":
        right = replace(
            right,
            source_track_index=left.source_track_index,
            hawor_source=left.hawor_source,
        )
    prompts, authority = point_bundle(tmp_path, left=left, right=right)
    if alias_kind == "evidence":
        prompts["right"] = replace(prompts["right"], evidence=prompts["left"].evidence)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == result.right.reason == "LEFT_RIGHT_POINT_IDENTITY_ALIAS"


def test_source_slot_mismatch_is_side_local(tmp_path: Path) -> None:
    prompts, authority = point_bundle(tmp_path)
    prompts["left"] = replace(prompts["left"], source_slot=1)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == "POINT_SOURCE_SLOT_IDENTITY_MISMATCH"
    assert result.right.status == "PASS"


@pytest.mark.parametrize(
    "source_state",
    [
        "invalid",
        "short_optical_flow_or_interpolation",
        "mid_bidirectional_interpolation",
        "reinitialized",
    ],
)
def test_nonobserved_hawor_source_state_holds_only_that_side(
    tmp_path: Path, source_state: str
) -> None:
    left = prompt_base(
        tmp_path,
        "left",
        hawor_source_state=source_state,
        hawor_source_valid=source_state != "invalid",
        hawor_measurement_accepted=False,
    )
    prompts, authority = point_bundle(tmp_path, left=left)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == "POINT_SOURCE_STATE_NOT_ADMITTED"
    assert result.right.status == "PASS"


def test_invalid_and_unaccepted_observation_hold_only_that_side(tmp_path: Path) -> None:
    cases = [
        (
            {"hawor_source_valid": False},
            "POINT_SOURCE_INVALID",
        ),
        (
            {
                "quality": 0.40353721380233765,
                "hawor_measurement_accepted": False,
            },
            "POINT_SOURCE_MEASUREMENT_NOT_ACCEPTED",
        ),
    ]
    for index, (kwargs, reason) in enumerate(cases):
        root = tmp_path / f"case_{index}"
        left = prompt_base(root, "left", **kwargs)
        prompts, authority = point_bundle(root, left=left)
        result = evaluate(root, prompts, authority)
        assert result.left.reason == reason
        assert result.right.status == "PASS"


def test_right_source_slot_collapse_holds_right_without_aborting_left(
    tmp_path: Path,
) -> None:
    right = replace(prompt_base(tmp_path, "right"), source_slot=0)
    prompts, authority = point_bundle(tmp_path, right=right)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.status == "PASS"
    assert result.right.reason == "POINT_SOURCE_SLOT_IDENTITY_MISMATCH"


def test_interpolated_left_cannot_borrow_missing_right_track_or_xy(
    tmp_path: Path,
) -> None:
    """Reproduce the 012 failure topology without consuming Route-A decisions.

    The interpolated left record deliberately carries the nominal right track and
    coordinate while the right record is MISSING.  Each side must retain its own
    typed provenance: left cannot be admitted from the borrowed coordinate and the
    missing right cannot be repaired from left evidence.
    """

    left = prompt_base(
        tmp_path,
        "left",
        xy=(0.75, 0.55),
        source_track_index=1,
        hawor_source_state="mid_bidirectional_interpolation",
        hawor_source_valid=True,
        hawor_measurement_accepted=False,
    )
    right = prompt_base(tmp_path, "right", state="MISSING")
    prompts, authority = point_bundle(tmp_path, left=left, right=right)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == "POINT_SOURCE_STATE_NOT_ADMITTED"
    assert result.right.reason == "POINT_MISSING"
    assert result.route_b_point_identity_ready is False


def test_mapping_identity_swap_is_rejected(tmp_path: Path) -> None:
    prompts, authority = point_bundle(tmp_path)
    with pytest.raises(RouteBPointScopeContractError, match="mapping identity"):
        evaluate(tmp_path, {"left": prompts["right"], "right": prompts["left"]}, authority)


def test_route_a_selector_or_recall_evidence_is_rejected(tmp_path: Path) -> None:
    prompts, authority = point_bundle(
        tmp_path, payload_extras={"left": {"selector_decision": "ACCEPT"}}
    )
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == "POINT_SIDE_VERIFICATION_FAILED:RouteBPointScopeContractError"
    assert result.right.status == "PASS"


def test_self_signed_quality_policy_is_rejected(tmp_path: Path) -> None:
    policy, source = authority_refs(tmp_path)
    payload = json.loads(Path(policy.path).read_text())
    payload["minimum_quality_exclusive"] = -1.0
    fake = json_ref(tmp_path / "contracts/fake_policy.json", payload, "HAWOR_POINT_QUALITY_POLICY")
    prompts, authority = point_bundle(tmp_path)
    with pytest.raises(RouteBPointScopeContractError, match="policy SHA"):
        evaluate_hawor_point_identity_gate(
            prompts,
            quality_policy_ref=fake,
            source_manifest_ref=source,
            hawor_authority_manifest_ref=authority,
            evidence_root=tmp_path,
        )


@pytest.mark.parametrize("middle", [False, True])
def test_final_and_intermediate_symlink_are_rejected(tmp_path: Path, middle: bool) -> None:
    prompts, authority = point_bundle(tmp_path)
    real = Path(prompts["left"].evidence.path)
    if middle:
        moved = tmp_path / "real_points"
        real.parent.rename(moved)
        real.parent.symlink_to(moved, target_is_directory=True)
    else:
        target = real.with_name("target.json")
        real.rename(target)
        real.symlink_to(target)
    result = evaluate(tmp_path, prompts, authority)
    assert result.left.reason == "POINT_SIDE_VERIFICATION_FAILED:RouteBPointScopeContractError"
    assert result.right.status == "PASS"


def scope(root: Path, side: str, **overrides: object) -> ScopeObservation:
    frame = int(overrides.pop("frame_index", 228))
    prompt_ref = json_ref(
        root / f"scope/{side}_{frame}_point.json",
        {
            "schema_version": "route-b-hawor-point-v2",
            "route_of_evidence": ROUTE_OF_EVIDENCE,
            "session_id": "grap_a_cap_004",
            "frame_index": frame,
            "side": side,
        },
        "HAWOR_ROUTE_B_POINT_PROMPT",
    )
    mask_ref = bytes_ref(
        root / f"scope/{side}_{frame}_mask.bin",
        f"raw-mask-{side}-{frame}".encode(),
        "SAM21_RAW_DECODER_MASK",
    )
    mask_record_ref = json_ref(
        root / f"scope/{side}_{frame}_mask_record.json",
        {
            "schema_version": "route-b-raw-decoder-mask-record-v1",
            "route_of_evidence": ROUTE_OF_EVIDENCE,
            "session_id": "grap_a_cap_004",
            "frame_index": frame,
            "side": side,
            "prompt_evidence_ref": prompt_ref.as_dict(),
            "decoder_mask_ref": mask_ref.as_dict(),
            "postprocessing_applied": False,
        },
        "SAM21_RAW_DECODER_MASK_RECORD",
    )
    values: dict[str, object] = {
        "side": side,
        "session_id": "grap_a_cap_004",
        "frame_index": frame,
        "prompt_evidence_ref": prompt_ref,
        "decoder_mask_ref": mask_ref,
        "decoder_mask_record_ref": mask_record_ref,
        "hand": True,
        "wrist_cuff": True,
        "sleeve": True,
        "forearm_to_required_image_extent": True,
    }
    values.update(overrides)
    return ScopeObservation(**values)  # type: ignore[arg-type]


def test_complete_scope_requires_all_four_components(tmp_path: Path) -> None:
    assert evaluate_scope_observation(
        scope(tmp_path, "left"), evidence_root=tmp_path
    ).status == "COMPLETE_REQUIRED_SCOPE"
    for component in ("hand", "wrist_cuff", "sleeve", "forearm_to_required_image_extent"):
        result = evaluate_scope_observation(
            scope(tmp_path, "left", **{component: False}), evidence_root=tmp_path
        )
        assert result.status == "SYSTEMATIC_SCOPE_TOO_SMALL"
        assert result.missing_components == (component,)


def test_aggregate_metric_cannot_hide_one_short_side(tmp_path: Path) -> None:
    result = aggregate_scope_panel(
        [
            scope(tmp_path, "left", frame_index=228),
            scope(tmp_path, "right", frame_index=228),
            scope(tmp_path, "left", frame_index=229, sleeve=False),
            scope(tmp_path, "right", frame_index=229),
        ],
        evidence_root=tmp_path,
    )
    assert result.status == "SYSTEMATIC_SCOPE_TOO_SMALL"
    assert (result.left_total, result.left_too_small) == (2, 1)
    assert (result.right_total, result.right_too_small) == (2, 0)


def test_scope_evaluator_is_review_only_and_strictly_typed(tmp_path: Path) -> None:
    with pytest.raises(RouteBPointScopeContractError, match="may not modify"):
        evaluate_scope_observation(
            scope(tmp_path, "left", modifies_pixels=True), evidence_root=tmp_path
        )
    with pytest.raises(RouteBPointScopeContractError, match="strict Boolean"):
        evaluate_scope_observation(scope(tmp_path, "left", hand=1), evidence_root=tmp_path)
    with pytest.raises(RouteBPointScopeContractError, match="cross-route"):
        evaluate_scope_observation(
            scope(tmp_path, "left", route_of_evidence="A_PRIME_SELECTOR_ONLY"),
            evidence_root=tmp_path,
        )


def test_scope_rejects_nonhex_unbound_and_cross_side_mask(tmp_path: Path) -> None:
    observation = scope(tmp_path, "left")
    forged = replace(
        observation.prompt_evidence_ref,
        sha256="z" * 64,
    )
    with pytest.raises(RouteBPointScopeContractError, match="malformed"):
        evaluate_scope_observation(
            replace(observation, prompt_evidence_ref=forged), evidence_root=tmp_path
        )
    right = scope(tmp_path, "right")
    with pytest.raises(RouteBPointScopeContractError, match="record identity"):
        evaluate_scope_observation(
            replace(
                observation,
                decoder_mask_ref=right.decoder_mask_ref,
                decoder_mask_record_ref=right.decoder_mask_record_ref,
            ),
            evidence_root=tmp_path,
        )


def governance() -> dict[str, object]:
    return {
        "AUTH_TIER": "T1_CPU_PREFLIGHT_ONLY",
        "WHY_NOT_BLOCKED": "AUTHORIZED_ROUTE_B_CPU_GATE_NO_LABEL_GPU_OR_MASK_ACCESS",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "gpu_started": False,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "route_a_evidence_consumed": False,
    }


def test_cpu_governance_is_exact_and_strictly_typed() -> None:
    validate_cpu_preflight_governance(governance())
    value = governance()
    value["labels_read"] = False
    with pytest.raises(RouteBPointScopeContractError, match="labels_read"):
        validate_cpu_preflight_governance(value)
    value = governance()
    value["selector_pass_rate"] = 1.0
    with pytest.raises(RouteBPointScopeContractError, match="key set drift"):
        validate_cpu_preflight_governance(value)


def hyperparameters() -> dict[str, object]:
    return {
        "schema_version": "route-b-hyperparameter-freeze-v1",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "frozen_before_labels": True,
        "one_configuration_only": True,
        "no_sweep": True,
        "seed": 20260828,
        "optimizer": "AdamW",
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "loss_formula": "ASYMMETRIC_HOUB_V1",
        "u_semantics": U_SEMANTICS,
        "batch_size": 1,
        "gradient_accumulation_steps": 10,
        "max_steps": 100,
        "prompt_generation": "FROZEN_LABEL_INDEPENDENT_HAWOR_PER_SIDE_POINT",
        "augmentation": [],
        "mask_logit_threshold": 0.0,
        "checkpoint_every_steps": 10,
        "stop_rule": "FIXED_MAX_STEPS_NO_RESULT_DRIVEN_EARLY_STOP",
        "trainable_prefix": "sam_mask_decoder.*",
        "trainable_tensor_count": 131,
        "trainable_parameter_count": 4_215_109,
        "oof_folds": {
            "F1": [228, 231, 234, 237, 240],
            "F2": [229, 232, 235, 238, 241],
            "F3": [230, 233, 236, 239, 242],
        },
    }


def input_artifacts(root: Path) -> dict[str, EvidenceRef]:
    bases: dict[str, HaworPointPrompt] = {}
    for frame in range(228, 243):
        for side in ("left", "right"):
            bases[f"{frame}:{side}"] = prompt_base(root, side, frame=frame)
    authority = make_authority(root, bases)
    gate_module.HAWOR_AUTHORITY_MANIFEST_SHA256 = authority.sha256
    sealed = seal_prompts(root, bases, authority)
    prompt_manifest = json_ref(
        root / "manifests/route_b_sealed_prompt_manifest_v4.json",
        {
            "schema_version": "route-b-sealed-point-prompt-manifest-v4",
            "route_of_evidence": ROUTE_OF_EVIDENCE,
            "sealed_before_labels": True,
            "labels_read_before_seal": 0,
            "supervised_session": "grap_a_cap_004",
            "development_frames": list(range(228, 243)),
            "hawor_authority_manifest_ref": authority.as_dict(),
            "entries": [
                {
                    "session_id": prompt.session_id,
                    "frame_index": prompt.frame_index,
                    "side": prompt.side,
                    "point_evidence_ref": prompt.evidence.as_dict(),
                }
                for prompt in sealed.values()
            ],
        },
        "ROUTE_B_SEALED_PROMPT_MANIFEST",
    )
    raw_entries = []
    for frame in range(228, 243):
        raw = bytes_ref(
            root / f"raw/{frame:05d}.rgb",
            f"frozen-raw-{frame}".encode(),
            "FROZEN_RAW_RGB",
        )
        raw_entries.append(
            {
                "session_id": "grap_a_cap_004",
                "frame_index": frame,
                "raw_image_ref": raw.as_dict(),
            }
        )
    raw_manifest = json_ref(
        root / "manifests/route_b_raw_input_manifest_v2.json",
        {
            "schema_version": "route-b-raw-input-manifest-v2",
            "route_of_evidence": ROUTE_OF_EVIDENCE,
            "frozen_before_labels": True,
            "labels_read_before_freeze": 0,
            "supervised_session": "grap_a_cap_004",
            "development_frames": list(range(228, 243)),
            "entries": raw_entries,
        },
        "ROUTE_B_RAW_INPUT_MANIFEST",
    )
    return {
        "hawor_authority_manifest": authority,
        "prompt_manifest": prompt_manifest,
        "raw_input_manifest": raw_manifest,
    }


def inputs(refs: dict[str, EvidenceRef]) -> dict[str, object]:
    return {
        "schema_version": "route-b-input-freeze-v2",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "supervised_session": "grap_a_cap_004",
        "supervised_split": "development",
        "development_frames": list(range(228, 243)),
        "oof_fold_ids": ["F1", "F2", "F3"],
        "labels_read_before_freeze": 0,
        "point_prompts_sealed_before_labels": True,
        "point_identity_gate": "PASS_LEFT_RIGHT_INDEPENDENT",
        "route_a_evidence_consumed": False,
        "blind_references_in_freeze": False,
        "grap_a_cap_025_references_in_freeze": False,
        "free_visual_sessions_after_candidate": [
            "grap_a_cap_002", "grap_a_cap_005", "grap_a_cap_012"
        ],
        "prompt_manifest_ref": refs["prompt_manifest"].as_dict(),
        "raw_input_manifest_ref": refs["raw_input_manifest"].as_dict(),
        "hawor_authority_manifest_ref": refs["hawor_authority_manifest"].as_dict(),
    }


def refreeze_mutated_prompt_chain(
    root: Path,
    refs: dict[str, EvidenceRef],
    *,
    mutations: dict[str, object],
    include_qa_and_release: bool,
) -> dict[str, EvidenceRef]:
    prompt_manifest_path = Path(refs["prompt_manifest"].path)
    prompt_manifest = json.loads(prompt_manifest_path.read_text())
    victim = next(
        entry
        for entry in prompt_manifest["entries"]
        if entry["session_id"] == "grap_a_cap_004"
        and entry["frame_index"] == 228
        and entry["side"] == "left"
    )
    point_path = Path(victim["point_evidence_ref"]["path"])
    point = json.loads(point_path.read_text())
    for field, value in mutations.items():
        if field.startswith("hawor_source."):
            point["hawor_source"][field.split(".", 1)[1]] = value
        else:
            point[field] = value
    point_ref = json_ref(point_path, point, "HAWOR_ROUTE_B_POINT_PROMPT")
    victim["point_evidence_ref"] = point_ref.as_dict()
    refs["prompt_manifest"] = json_ref(
        prompt_manifest_path,
        prompt_manifest,
        "ROUTE_B_SEALED_PROMPT_MANIFEST",
    )
    if "input_freeze" in refs:
        input_path = Path(refs["input_freeze"].path)
        refs["input_freeze"] = json_ref(
            input_path, inputs(refs), "ROUTE_B_INPUT_FREEZE"
        )
    if include_qa_and_release:
        qa_path = root / gate_module.CANONICAL_QA_PATH
        qa = json.loads(qa_path.read_text())
        qa["frozen_authority_refs"]["prompt_manifest"] = refs[
            "prompt_manifest"
        ].as_dict()
        qa["frozen_authority_refs"]["input_freeze"] = refs[
            "input_freeze"
        ].as_dict()
        qa_ref = json_ref(qa_path, qa, "ROUTE_B_INDEPENDENT_QA")
        refs["independent_qa"] = replace(
            qa_ref, path=gate_module.CANONICAL_QA_PATH
        )
        release_path = root / gate_module.CANONICAL_OWNER_RELEASE_PATH
        release = json.loads(release_path.read_text())
        release["independent_qa_ref"] = refs["independent_qa"].as_dict()
        release["input_freeze_ref"] = refs["input_freeze"].as_dict()
        release["prompt_manifest_ref"] = refs["prompt_manifest"].as_dict()
        release_ref = json_ref(
            release_path, release, "ROUTE_B_GPU_RELEASE_AUTHORIZATION"
        )
        refs["owner_gpu_release"] = replace(
            release_ref, path=gate_module.CANONICAL_OWNER_RELEASE_PATH
        )
    return refs


def refreeze_authority_pair_alias_chain(
    root: Path,
    refs: dict[str, EvidenceRef],
    *,
    alias_kind: str,
    include_qa_and_release: bool,
) -> dict[str, EvidenceRef]:
    """Refreeze a digest-consistent frame-228 pair alias through every caller layer."""

    authority_path = root / refs["hawor_authority_manifest"].path
    authority = json.loads(authority_path.read_text())
    left = next(
        entry for entry in authority["entries"]
        if entry["session_id"] == "grap_a_cap_004"
        and entry["frame_index"] == 228 and entry["side"] == "left"
    )
    right = next(
        entry for entry in authority["entries"]
        if entry["session_id"] == "grap_a_cap_004"
        and entry["frame_index"] == 228 and entry["side"] == "right"
    )
    if alias_kind in {"lineage", "all"}:
        right["lineage_id"] = left["lineage_id"]
    if alias_kind in {"track", "all"}:
        right["track_id"] = left["track_id"]
    if alias_kind in {"geometry_track", "all"}:
        right["source_track_index"] = left["source_track_index"]
        right["hawor_source_ref"] = left["hawor_source_ref"]
    assert left["source_slot"] == 0 and right["source_slot"] == 1
    authority_ref = json_ref(
        authority_path, authority, "HAWOR_AUTHORITY_MANIFEST"
    )
    refs["hawor_authority_manifest"] = replace(
        authority_ref, path=gate_module.HAWOR_AUTHORITY_MANIFEST_PATH
    )

    prompt_manifest_path = Path(refs["prompt_manifest"].path)
    prompt_manifest = json.loads(prompt_manifest_path.read_text())
    prompt_manifest["hawor_authority_manifest_ref"] = refs[
        "hawor_authority_manifest"
    ].as_dict()
    for entry in prompt_manifest["entries"]:
        point_path = Path(entry["point_evidence_ref"]["path"])
        point = json.loads(point_path.read_text())
        point["hawor_authority_manifest_sha256"] = refs[
            "hawor_authority_manifest"
        ].sha256
        if (
            point["session_id"] == "grap_a_cap_004"
            and point["frame_index"] == 228
            and point["side"] == "right"
        ):
            if alias_kind in {"lineage", "all"}:
                point["lineage_id"] = left["lineage_id"]
            if alias_kind in {"track", "all"}:
                point["track_id"] = left["track_id"]
            if alias_kind in {"geometry_track", "all"}:
                point["source_track_index"] = left["source_track_index"]
                point["hawor_source"] = {
                    "bytes": left["hawor_source_ref"]["bytes"],
                    "sha256": left["hawor_source_ref"]["sha256"],
                }
            assert point["source_slot"] == 1
        point_ref = json_ref(point_path, point, "HAWOR_ROUTE_B_POINT_PROMPT")
        entry["point_evidence_ref"] = point_ref.as_dict()
    refs["prompt_manifest"] = json_ref(
        prompt_manifest_path,
        prompt_manifest,
        "ROUTE_B_SEALED_PROMPT_MANIFEST",
    )

    input_path = Path(refs["input_freeze"].path)
    refs["input_freeze"] = json_ref(
        input_path, inputs(refs), "ROUTE_B_INPUT_FREEZE"
    )
    if include_qa_and_release:
        qa_path = root / gate_module.CANONICAL_QA_PATH
        qa = json.loads(qa_path.read_text())
        for name in (
            "hawor_authority_manifest", "prompt_manifest", "input_freeze"
        ):
            qa["frozen_authority_refs"][name] = refs[name].as_dict()
        qa_ref = json_ref(qa_path, qa, "ROUTE_B_INDEPENDENT_QA")
        refs["independent_qa"] = replace(
            qa_ref, path=gate_module.CANONICAL_QA_PATH
        )
        release_path = root / gate_module.CANONICAL_OWNER_RELEASE_PATH
        release = json.loads(release_path.read_text())
        release["independent_qa_ref"] = refs["independent_qa"].as_dict()
        release["input_freeze_ref"] = refs["input_freeze"].as_dict()
        release["hawor_authority_manifest_ref"] = refs[
            "hawor_authority_manifest"
        ].as_dict()
        release["prompt_manifest_ref"] = refs["prompt_manifest"].as_dict()
        release_ref = json_ref(
            release_path, release, "ROUTE_B_GPU_RELEASE_AUTHORIZATION"
        )
        refs["owner_gpu_release"] = replace(
            release_ref, path=gate_module.CANONICAL_OWNER_RELEASE_PATH
        )
    return refs


def refreeze_authority_source_state_chain(
    root: Path,
    refs: dict[str, EvidenceRef],
    *,
    changes: dict[str, object],
    include_qa_and_release: bool,
) -> dict[str, EvidenceRef]:
    """Refreeze one left source-state mutation through all digest-bound layers."""

    authority_path = root / refs["hawor_authority_manifest"].path
    authority = json.loads(authority_path.read_text())
    victim = next(
        entry for entry in authority["entries"]
        if entry["session_id"] == "grap_a_cap_004"
        and entry["frame_index"] == 228 and entry["side"] == "left"
    )
    victim.update(changes)
    authority_ref = json_ref(
        authority_path, authority, "HAWOR_AUTHORITY_MANIFEST"
    )
    refs["hawor_authority_manifest"] = replace(
        authority_ref, path=gate_module.HAWOR_AUTHORITY_MANIFEST_PATH
    )

    prompt_manifest_path = Path(refs["prompt_manifest"].path)
    prompt_manifest = json.loads(prompt_manifest_path.read_text())
    prompt_manifest["hawor_authority_manifest_ref"] = refs[
        "hawor_authority_manifest"
    ].as_dict()
    for entry in prompt_manifest["entries"]:
        point_path = Path(entry["point_evidence_ref"]["path"])
        point = json.loads(point_path.read_text())
        point["hawor_authority_manifest_sha256"] = refs[
            "hawor_authority_manifest"
        ].sha256
        if (
            point["session_id"] == "grap_a_cap_004"
            and point["frame_index"] == 228 and point["side"] == "left"
        ):
            point.update(changes)
        point_ref = json_ref(point_path, point, "HAWOR_ROUTE_B_POINT_PROMPT")
        entry["point_evidence_ref"] = point_ref.as_dict()
    refs["prompt_manifest"] = json_ref(
        prompt_manifest_path,
        prompt_manifest,
        "ROUTE_B_SEALED_PROMPT_MANIFEST",
    )
    input_path = Path(refs["input_freeze"].path)
    refs["input_freeze"] = json_ref(
        input_path, inputs(refs), "ROUTE_B_INPUT_FREEZE"
    )
    if include_qa_and_release:
        qa_path = root / gate_module.CANONICAL_QA_PATH
        qa = json.loads(qa_path.read_text())
        for name in (
            "hawor_authority_manifest", "prompt_manifest", "input_freeze"
        ):
            qa["frozen_authority_refs"][name] = refs[name].as_dict()
        qa_ref = json_ref(qa_path, qa, "ROUTE_B_INDEPENDENT_QA")
        refs["independent_qa"] = replace(
            qa_ref, path=gate_module.CANONICAL_QA_PATH
        )
        release_path = root / gate_module.CANONICAL_OWNER_RELEASE_PATH
        release = json.loads(release_path.read_text())
        release["independent_qa_ref"] = refs["independent_qa"].as_dict()
        release["input_freeze_ref"] = refs["input_freeze"].as_dict()
        release["hawor_authority_manifest_ref"] = refs[
            "hawor_authority_manifest"
        ].as_dict()
        release["prompt_manifest_ref"] = refs["prompt_manifest"].as_dict()
        release_ref = json_ref(
            release_path, release, "ROUTE_B_GPU_RELEASE_AUTHORIZATION"
        )
        refs["owner_gpu_release"] = replace(
            release_ref, path=gate_module.CANONICAL_OWNER_RELEASE_PATH
        )
    return refs


@pytest.mark.parametrize(
    "mutations",
    [
        {"session_id": "grap_a_cap_005"},
        {"frame_index": 229},
        {"side": "right"},
        {"lineage_id": "attacker-left-lineage"},
        {"track_id": "attacker-left-track"},
        {"source_slot": 1},
        {"source_track_index": 9},
        {"hawor_source_state": "mid_bidirectional_interpolation"},
        {"hawor_source_valid": False},
        {"hawor_measurement_accepted": False},
        {"state": "OUTSIDE_IMAGE"},
        {"normalized_xy": [0.99, 0.99]},
        {"quality_value": 0.8},
        {"hawor_source.bytes": 1},
        {"hawor_source.sha256": "f" * 64},
    ],
)
def test_every_sealed_point_field_is_joined_to_exact_hawor_authority(
    tmp_path: Path, mutations: dict[str, object]
) -> None:
    refs = input_artifacts(tmp_path)
    refreeze_mutated_prompt_chain(
        tmp_path,
        refs,
        mutations=mutations,
        include_qa_and_release=False,
    )
    with pytest.raises(
        RouteBPointScopeContractError,
        match="identity drift|membership mismatch|geometry bytes/SHA mismatch",
    ):
        validate_input_freeze(
            inputs(refs),
            evidence_root=tmp_path,
            prompt_manifest_ref=refs["prompt_manifest"],
            raw_input_manifest_ref=refs["raw_input_manifest"],
            hawor_authority_manifest_ref=refs["hawor_authority_manifest"],
        )


def test_hawor_authority_requires_exact_thirty_dev_side_memberships(
    tmp_path: Path,
) -> None:
    refs = input_artifacts(tmp_path)
    authority_path = tmp_path / gate_module.HAWOR_AUTHORITY_MANIFEST_PATH
    authority = json.loads(authority_path.read_text())
    authority["entries"].pop()
    authority_ref = json_ref(
        authority_path, authority, "HAWOR_AUTHORITY_MANIFEST"
    )
    refs["hawor_authority_manifest"] = replace(
        authority_ref, path=gate_module.HAWOR_AUTHORITY_MANIFEST_PATH
    )
    gate_module.HAWOR_AUTHORITY_MANIFEST_SHA256 = authority_ref.sha256
    prompt_path = Path(refs["prompt_manifest"].path)
    prompt = json.loads(prompt_path.read_text())
    prompt["hawor_authority_manifest_ref"] = refs[
        "hawor_authority_manifest"
    ].as_dict()
    refs["prompt_manifest"] = json_ref(
        prompt_path, prompt, "ROUTE_B_SEALED_PROMPT_MANIFEST"
    )
    with pytest.raises(RouteBPointScopeContractError, match="exact dev15 sides"):
        validate_input_freeze(
            inputs(refs),
            evidence_root=tmp_path,
            prompt_manifest_ref=refs["prompt_manifest"],
            raw_input_manifest_ref=refs["raw_input_manifest"],
            hawor_authority_manifest_ref=refs["hawor_authority_manifest"],
        )


@pytest.mark.parametrize("alias_kind", ["lineage", "track", "geometry_track"])
def test_input_freeze_rejects_each_refrozen_authority_pair_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    refs = gpu_refs(tmp_path)
    refreeze_authority_pair_alias_chain(
        tmp_path,
        refs,
        alias_kind=alias_kind,
        include_qa_and_release=False,
    )
    monkeypatch.setattr(
        gate_module,
        "HAWOR_AUTHORITY_MANIFEST_SHA256",
        refs["hawor_authority_manifest"].sha256,
    )
    with pytest.raises(RouteBPointScopeContractError, match="left/right identity alias"):
        validate_input_freeze(
            inputs(refs),
            evidence_root=tmp_path,
            prompt_manifest_ref=refs["prompt_manifest"],
            raw_input_manifest_ref=refs["raw_input_manifest"],
            hawor_authority_manifest_ref=refs["hawor_authority_manifest"],
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "hawor_source_state": "mid_bidirectional_interpolation",
                "hawor_source_valid": True,
                "hawor_measurement_accepted": False,
                "quality_value": 0.40353721380233765,
            },
            "source state not admitted",
        ),
        ({"hawor_source_valid": False}, "source invalid"),
        ({"hawor_measurement_accepted": False}, "measurement not accepted"),
        ({"source_slot": 1}, "slot side-collapse or alias"),
    ],
)
def test_input_freeze_rejects_refrozen_nonadmitted_source_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    refs = gpu_refs(tmp_path)
    refreeze_authority_source_state_chain(
        tmp_path,
        refs,
        changes=changes,
        include_qa_and_release=False,
    )
    monkeypatch.setattr(
        gate_module,
        "HAWOR_AUTHORITY_MANIFEST_SHA256",
        refs["hawor_authority_manifest"].sha256,
    )
    with pytest.raises(RouteBPointScopeContractError, match=message):
        validate_input_freeze(
            inputs(refs),
            evidence_root=tmp_path,
            prompt_manifest_ref=refs["prompt_manifest"],
            raw_input_manifest_ref=refs["raw_input_manifest"],
            hawor_authority_manifest_ref=refs["hawor_authority_manifest"],
        )


def test_hyperparameter_and_input_freezes_are_complete(tmp_path: Path) -> None:
    refs = input_artifacts(tmp_path)
    validate_hyperparameter_freeze(hyperparameters())
    validate_input_freeze(
        inputs(refs),
        evidence_root=tmp_path,
        prompt_manifest_ref=refs["prompt_manifest"],
        raw_input_manifest_ref=refs["raw_input_manifest"],
        hawor_authority_manifest_ref=refs["hawor_authority_manifest"],
    )
    value = hyperparameters()
    del value["stop_rule"]
    with pytest.raises(RouteBPointScopeContractError, match="key set"):
        validate_hyperparameter_freeze(value)
    value = inputs(refs)
    value["grap_a_cap_025_references_in_freeze"] = True
    with pytest.raises(RouteBPointScopeContractError, match="025"):
        validate_input_freeze(
            value,
            evidence_root=tmp_path,
            prompt_manifest_ref=refs["prompt_manifest"],
            raw_input_manifest_ref=refs["raw_input_manifest"],
            hawor_authority_manifest_ref=refs["hawor_authority_manifest"],
        )


@pytest.mark.parametrize("forbidden", sorted(gate_module.FORBIDDEN_ROUTE_A_VALUES))
def test_every_route_a_value_is_rejected_inside_nested_list(forbidden: str) -> None:
    value = hyperparameters()
    value["augmentation"] = ["safe", ["nested", forbidden]]
    with pytest.raises(RouteBPointScopeContractError, match="forbidden Route-A"):
        gate_module.reject_route_a_or_override_evidence(value)


@pytest.mark.parametrize("forbidden", sorted(gate_module.FORBIDDEN_OVERRIDE_KEYS))
def test_every_override_key_is_rejected_recursively(forbidden: str) -> None:
    with pytest.raises(RouteBPointScopeContractError, match="override keys"):
        gate_module.reject_route_a_or_override_evidence(
            {"outer": [{"inner": {forbidden: False}}]}
        )


def test_prompt_and_raw_manifest_refs_reject_drift_and_intermediate_symlink(
    tmp_path: Path,
) -> None:
    refs = input_artifacts(tmp_path)
    Path(refs["raw_input_manifest"].path).write_bytes(b"drift")
    with pytest.raises(RouteBPointScopeContractError, match="byte count mismatch|SHA256 mismatch"):
        validate_input_freeze(
            inputs(refs),
            evidence_root=tmp_path,
            prompt_manifest_ref=refs["prompt_manifest"],
            raw_input_manifest_ref=refs["raw_input_manifest"],
            hawor_authority_manifest_ref=refs["hawor_authority_manifest"],
        )
    root = tmp_path / "symlink_case"
    refs = input_artifacts(root)
    manifests = root / "manifests"
    real = root / "real_manifests"
    manifests.rename(real)
    manifests.symlink_to(real, target_is_directory=True)
    with pytest.raises(RouteBPointScopeContractError, match="symlink|non-directory"):
        validate_input_freeze(
            inputs(refs),
            evidence_root=root,
            prompt_manifest_ref=refs["prompt_manifest"],
            raw_input_manifest_ref=refs["raw_input_manifest"],
            hawor_authority_manifest_ref=refs["hawor_authority_manifest"],
        )


def make_synthetic_asset(root: Path) -> EvidenceRef:
    implementation = root / "asset/implementation"
    config = implementation / "sam2/configs/model.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: synthetic\n", encoding="utf-8")
    (implementation / "sam2/model.py").write_text("VALUE = 1\n", encoding="utf-8")
    weight = root / "asset/model.pt"
    weight.write_bytes(b"synthetic-weight")
    inventory = implementation_inventory(implementation)
    payload = {
        "schema_version": "local-model-asset-pin-v1",
        "model_identifier": "SAM2_1_HIERA_LARGE",
        "local": {
            "implementation_ref": "asset/implementation",
            "implementation_inventory_entries": inventory[0],
            "implementation_regular_bytes": inventory[1],
            "implementation_inventory_sha256": inventory[2],
            "config_path": "sam2/configs/model.yaml",
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "weight_path": "asset/model.pt",
            "weight_bytes": weight.stat().st_size,
            "weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        },
    }
    return json_ref(root / "asset/ASSET_PIN.json", payload, "SAM21_ASSET_PIN")


def test_asset_pin_inventory_checkpoint_and_bytecode_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = make_synthetic_asset(tmp_path)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    result = verify_sam21_asset_pin(
        project_root=tmp_path, asset_pin_ref=pin, expected_asset_pin_sha256=pin.sha256
    )
    assert result["asset_pin_sha256"] == pin.sha256
    (tmp_path / "asset/implementation/sam2/__pycache__").mkdir()
    with pytest.raises(RouteBPointScopeContractError, match="bytecode/cache"):
        verify_sam21_asset_pin(
            project_root=tmp_path, asset_pin_ref=pin, expected_asset_pin_sha256=pin.sha256
        )


def test_both_bytecode_switches_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    with pytest.raises(RouteBPointScopeContractError, match="PYTHONDONTWRITEBYTECODE"):
        require_bytecode_disabled()
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    with pytest.raises(RouteBPointScopeContractError, match="sys.dont_write_bytecode"):
        require_bytecode_disabled()


def gpu_admission(round_index: int = 1) -> dict[str, object]:
    return {
        "AUTH_TIER": "T1",
        "WHY_NOT_BLOCKED": "AUTHORIZED_TASK16_AFTER_INDEPENDENT_P0_ZERO",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "round_index": round_index,
        "previous_completed_route_b_rounds": round_index - 1,
        "point_identity_gate": "PASS_LEFT_RIGHT_INDEPENDENT",
        "scope_gate": "PENDING_RUNTIME_REQUIRED_SCOPE_EVALUATION",
        "u_semantics": U_SEMANTICS,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "labels_read_before_run_freeze": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "route_a_evidence_consumed": False,
        "gpu_started_before_admission": False,
    }


def gpu_refs(root: Path) -> dict[str, EvidenceRef]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "_run").mkdir()
    relative_paths = {
        "task": Path("archive/legacy/task_cards/23_ROUTE_B_HAWOR_POINT_IDENTITY_AND_SCOPE_GATE_T1.md"),
        "identity_scope_module": Path("pipeline/route_b_hawor_point_scope_gate.py"),
        "identity_scope_tests": Path("tests/test_route_b_hawor_point_scope_gate.py"),
        "route_b_contract": Path("pipeline/sam21_route_b_contract.py"),
        "route_b_contract_tests": Path("tests/test_sam21_route_b_contract.py"),
    }
    for relative in relative_paths.values():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, target)
    lives = gate_module._live_implementation_binding(root)
    kind_map = {
        "task": "ROUTE_B_GATE_TASK",
        "identity_scope_module": "ROUTE_B_GATE_MODULE",
        "identity_scope_tests": "ROUTE_B_GATE_TESTS",
        "route_b_contract": "ROUTE_B_CONTRACT_MODULE",
        "route_b_contract_tests": "ROUTE_B_CONTRACT_TESTS",
    }
    refs = {}
    for name, relative in relative_paths.items():
        target = root / relative
        data = target.read_bytes()
        refs[name] = EvidenceRef(
            str(relative), len(data), hashlib.sha256(data).hexdigest(), kind_map[name]
        )
    implementation = {
        "schema_version": "route-b-point-scope-implementation-freeze-v4",
        "task_path": "archive/legacy/task_cards/23_ROUTE_B_HAWOR_POINT_IDENTITY_AND_SCOPE_GATE_T1.md",
        "module_path": "pipeline/route_b_hawor_point_scope_gate.py",
        "tests_path": "tests/test_route_b_hawor_point_scope_gate.py",
        "route_b_contract_path": "pipeline/sam21_route_b_contract.py",
        "route_b_contract_tests_path": "tests/test_sam21_route_b_contract.py",
        **lives,
    }
    refs["implementation_freeze"] = json_ref(
        root / "archive/audits/ROUTE_B_HAWOR_POINT_SCOPE_CPU_FREEZE_V4.json",
        implementation,
        "ROUTE_B_IMPLEMENTATION_FREEZE",
    )
    refs["hyperparameter_freeze"] = json_ref(
        root / "hyperparameters.json", hyperparameters(), "ROUTE_B_HYPERPARAMETER_FREEZE"
    )
    refs.update(input_artifacts(root))
    refs["input_freeze"] = json_ref(
        root / "archive/audits/route_b_inputs_v2.json", inputs(refs), "ROUTE_B_INPUT_FREEZE"
    )
    asset_bytes = (PROJECT / "assets/models/sam2_1_hiera_large/ASSET_PIN.json").read_bytes()
    refs["asset_pin"] = bytes_ref(root / "ASSET_PIN.json", asset_bytes, "SAM21_ASSET_PIN")
    assert refs["asset_pin"].sha256 == ASSET_PIN_SHA256
    qa_names = {
        "task", "identity_scope_module", "identity_scope_tests",
        "route_b_contract", "route_b_contract_tests", "implementation_freeze",
        "hyperparameter_freeze", "input_freeze", "asset_pin",
        "hawor_authority_manifest", "prompt_manifest", "raw_input_manifest",
    }
    qa = {
        "schema_version": "route-b-point-scope-independent-qa-v4",
        "producer": "independent_cpu_qa",
        "qa_scope": "ROUTE_B_POINT_IDENTITY_SCOPE_GPU_ADMISSION_ONLY",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "status": "PASS_CPU_ADMISSION_EXACT",
        "p0_findings": 0,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "real_hawor_prompts_read": 0,
        "gpu_started": False,
        **lives,
        "frozen_authority_refs": {
            name: refs[name].as_dict() for name in sorted(qa_names)
        },
    }
    qa_ref = json_ref(
        root / gate_module.CANONICAL_QA_PATH,
        qa,
        "ROUTE_B_INDEPENDENT_QA",
    )
    refs["independent_qa"] = replace(qa_ref, path=gate_module.CANONICAL_QA_PATH)
    release = {
        "schema_version": "route-b-gpu-release-authorization-v3",
        "authorization_scope": "TASK16_ROUTE_B_GPU_ROUND_1_OR_2_ONLY",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "gpu_release_authorized": True,
        "maximum_route_b_rounds": 2,
        **lives,
        "independent_qa_ref": refs["independent_qa"].as_dict(),
        "implementation_freeze_ref": refs["implementation_freeze"].as_dict(),
        "hyperparameter_freeze_ref": refs["hyperparameter_freeze"].as_dict(),
        "input_freeze_ref": refs["input_freeze"].as_dict(),
        "hawor_authority_manifest_ref": refs["hawor_authority_manifest"].as_dict(),
        "prompt_manifest_ref": refs["prompt_manifest"].as_dict(),
        "raw_input_manifest_ref": refs["raw_input_manifest"].as_dict(),
    }
    release_ref = json_ref(
        root / gate_module.CANONICAL_OWNER_RELEASE_PATH,
        release,
        "ROUTE_B_GPU_RELEASE_AUTHORIZATION",
    )
    refs["owner_gpu_release"] = replace(
        release_ref, path=gate_module.CANONICAL_OWNER_RELEASE_PATH
    )
    return refs


def admit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    refs: dict[str, EvidenceRef] | None = None,
    record: dict[str, object] | None = None,
    run_name: str = "route_b_round1_v1",
) -> EvidenceRef:
    refs = refs or gpu_refs(tmp_path)
    run_parent = tmp_path / "_run"
    monkeypatch.setattr(gate_module, "verify_sam21_asset_pin", lambda **_: {"status": "PASS"})
    monkeypatch.setattr(
        gate_module, "OWNER_GPU_RELEASE_AUTHORIZATION_SHA256",
        refs["owner_gpu_release"].sha256,
    )
    return require_route_b_gpu_admission(
        record or gpu_admission(),
        frozen_refs=refs,
        evidence_root=tmp_path,
        project_root=tmp_path,
        project_run_root=run_parent,
        run_root=run_parent / run_name,
    )


def test_current_owner_directive_blocks_gpu_even_before_other_admission_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate_module, "OWNER_GPU_RELEASE_AUTHORIZATION_SHA256", None)
    with pytest.raises(RouteBPointScopeContractError, match="current owner directive"):
        require_route_b_gpu_admission(
            {}, frozen_refs={}, evidence_root=tmp_path, project_root=PROJECT,
            project_run_root=tmp_path, run_root=tmp_path / "route_b_forbidden",
        )
    assert not (tmp_path / "route_b_forbidden").exists()


def test_gpu_admission_creates_real_o_excl_owner_and_reuse_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = gpu_refs(tmp_path)
    marker = admit(tmp_path, monkeypatch, refs=refs)
    assert Path(marker.path).is_file() and marker.source_kind == "ROUTE_B_RUN_OWNER"
    with pytest.raises(RouteBPointScopeContractError, match="already exists"):
        admit(tmp_path, monkeypatch, refs=refs)


def test_gpu_admission_replays_v2_refrozen_prompt_membership_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = gpu_refs(tmp_path)
    authority_sha = refs["hawor_authority_manifest"].sha256
    refreeze_mutated_prompt_chain(
        tmp_path,
        refs,
        mutations={
            "lineage_id": "attacker-left-not-in-authority",
            "normalized_xy": [0.99, 0.99],
        },
        include_qa_and_release=True,
    )
    assert refs["hawor_authority_manifest"].sha256 == authority_sha
    run_root = tmp_path / "_run/route_b_refrozen_prompt_attack"
    with pytest.raises(RouteBPointScopeContractError, match="membership mismatch"):
        admit(
            tmp_path,
            monkeypatch,
            refs=refs,
            run_name=run_root.name,
        )
    assert not run_root.exists()


def test_gpu_admission_rejects_fully_refrozen_left_right_authority_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = gpu_refs(tmp_path)
    refreeze_authority_pair_alias_chain(
        tmp_path,
        refs,
        alias_kind="all",
        include_qa_and_release=True,
    )
    monkeypatch.setattr(
        gate_module,
        "HAWOR_AUTHORITY_MANIFEST_SHA256",
        refs["hawor_authority_manifest"].sha256,
    )
    run_root = tmp_path / "_run/route_b_refrozen_pair_alias_attack"
    with pytest.raises(RouteBPointScopeContractError, match="left/right identity alias"):
        admit(
            tmp_path,
            monkeypatch,
            refs=refs,
            run_name=run_root.name,
        )
    assert not run_root.exists()


def test_gpu_admission_rejects_fully_refrozen_interpolated_source_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = gpu_refs(tmp_path)
    refreeze_authority_source_state_chain(
        tmp_path,
        refs,
        changes={
            "hawor_source_state": "mid_bidirectional_interpolation",
            "hawor_source_valid": True,
            "hawor_measurement_accepted": False,
            "quality_value": 0.40353721380233765,
        },
        include_qa_and_release=True,
    )
    monkeypatch.setattr(
        gate_module,
        "HAWOR_AUTHORITY_MANIFEST_SHA256",
        refs["hawor_authority_manifest"].sha256,
    )
    run_root = tmp_path / "_run/route_b_refrozen_interpolation_attack"
    with pytest.raises(RouteBPointScopeContractError, match="source state not admitted"):
        admit(
            tmp_path,
            monkeypatch,
            refs=refs,
            run_name=run_root.name,
        )
    assert not run_root.exists()


def test_gpu_admission_allows_only_first_two_frozen_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = gpu_refs(tmp_path)
    admit(tmp_path, monkeypatch, refs=refs, record=gpu_admission(2), run_name="route_b_round2_v1")
    with pytest.raises(RouteBPointScopeContractError, match="round 1 or 2"):
        admit(tmp_path, monkeypatch, refs=refs, record=gpu_admission(3), run_name="route_b_round3_v1")


def test_gpu_admission_rejects_false_qa_and_stale_live_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_root = tmp_path / "bad_qa"
    refs = gpu_refs(bad_root)
    qa_path = bad_root / refs["independent_qa"].path
    qa = json.loads(qa_path.read_text())
    qa["p0_findings"] = 1
    changed_qa = json_ref(qa_path, qa, "ROUTE_B_INDEPENDENT_QA")
    refs["independent_qa"] = replace(changed_qa, path=gate_module.CANONICAL_QA_PATH)
    with pytest.raises(RouteBPointScopeContractError, match="QA/binding"):
        admit(bad_root, monkeypatch, refs=refs, run_name="route_b_bad_qa")
    stale_root = tmp_path / "stale"
    refs = gpu_refs(stale_root)
    refs["task"] = EvidenceRef(refs["task"].path, refs["task"].bytes, "0" * 64, refs["task"].source_kind)
    with pytest.raises(RouteBPointScopeContractError, match="SHA256 mismatch|live ref SHA"):
        admit(stale_root, monkeypatch, refs=refs, run_name="route_b_stale")


def test_gpu_admission_rejects_route_a_override_and_sealed_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route_a_root = tmp_path / "route_a"
    refs = gpu_refs(route_a_root)
    hp = json.loads(Path(refs["hyperparameter_freeze"].path).read_text())
    hp["selector_decision"] = "ACCEPT"
    refs["hyperparameter_freeze"] = json_ref(
        Path(refs["hyperparameter_freeze"].path), hp, "ROUTE_B_HYPERPARAMETER_FREEZE"
    )
    with pytest.raises(
        RouteBPointScopeContractError,
        match="forbidden Route-A|independent QA/binding mismatch",
    ):
        admit(route_a_root, monkeypatch, refs=refs, run_name="route_b_route_a")
    sealed_root = tmp_path / "sealed"
    refs = gpu_refs(sealed_root)
    frozen_input = json.loads(Path(refs["input_freeze"].path).read_text())
    frozen_input["free_visual_sessions_after_candidate"].append("grap_a_cap_025")
    refs["input_freeze"] = json_ref(
        Path(refs["input_freeze"].path), frozen_input, "ROUTE_B_INPUT_FREEZE"
    )
    with pytest.raises(
        RouteBPointScopeContractError,
        match="sealed 025/blind|independent QA/binding mismatch",
    ):
        admit(sealed_root, monkeypatch, refs=refs, run_name="route_b_025")


def test_gpu_admission_rejects_boolean_claim_and_strict_type_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = gpu_admission()
    record["fresh_o_excl_run"] = True
    with pytest.raises(RouteBPointScopeContractError, match="key set mismatch"):
        admit(tmp_path / "boolean", monkeypatch, record=record, run_name="route_b_boolean_claim")
    record = gpu_admission()
    record["labels_read_before_run_freeze"] = False
    with pytest.raises(RouteBPointScopeContractError, match="labels_read"):
        admit(tmp_path / "false_zero", monkeypatch, record=record, run_name="route_b_false_zero")


def test_gpu_admission_rejects_noncanonical_qa_path_and_intermediate_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path_root = tmp_path / "path"
    refs = gpu_refs(path_root)
    canonical = path_root / refs["independent_qa"].path
    attacker = path_root / "attacker_controlled_not_audits" / canonical.name
    attacker.parent.mkdir()
    shutil.copyfile(canonical, attacker)
    refs["independent_qa"] = replace(
        refs["independent_qa"], path=str(attacker.relative_to(path_root))
    )
    with pytest.raises(RouteBPointScopeContractError, match="canonical project-relative"):
        admit(path_root, monkeypatch, refs=refs, run_name="route_b_bad_qa_path")

    symlink_root = tmp_path / "symlink"
    refs = gpu_refs(symlink_root)
    audits = symlink_root / "archive/audits"
    real = symlink_root / "real_audits"
    audits.rename(real)
    audits.symlink_to(real, target_is_directory=True)
    with pytest.raises(RouteBPointScopeContractError, match="symlink|non-directory"):
        admit(symlink_root, monkeypatch, refs=refs, run_name="route_b_qa_symlink")
