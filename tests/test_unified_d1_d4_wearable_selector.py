import hashlib
from io import BytesIO
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline import lr_contact_phase_object6d_evidence as d4
from pipeline import lr_distributed_side_evidence as d1
from pipeline import neutral_wearable_identity_v2 as wearable
from pipeline import unified_d1_d4_wearable_selector as unified


ROOT = Path(__file__).parents[1]
SHAPE = (12, 20)
SHA = "1" * 64


def _write(path: Path, payload: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _ref(path: Path, payload: bytes, kind: str, cls=d1.EvidenceRef):
    size, sha = _write(path, payload)
    return cls(str(path), size, sha, kind)


def _png_bytes(mask: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return stream.getvalue()


def _bound_wearable(
    root: Path,
    session: str,
    instance: wearable.RawWearableInstance,
    *,
    metadata_source_kind: str = unified.WEARABLE_METADATA_SOURCE_KIND,
    mask_source_kind: str = unified.WEARABLE_MASK_SOURCE_KIND,
    metadata_overrides: dict[str, object] | None = None,
    png_mask: np.ndarray | None = None,
) -> unified.BoundWearableRawInstance:
    source_mask = instance.mask if png_mask is None else png_mask
    stem = (
        f"{instance.prompt_id}_frame_{instance.frame_index:05d}_"
        f"offset_{instance.raw_instance_offset}_id_{instance.instance_id}"
    )
    mask_ref = _ref(
        root / "wearable_raw" / f"{stem}.png",
        _png_bytes(source_mask),
        mask_source_kind,
        wearable.EvidenceRef,
    )
    metadata = {
        "source_kind": unified.WEARABLE_METADATA_SOURCE_KIND,
        "session_id": session,
        "frame_index": instance.frame_index,
        "prompt_id": instance.prompt_id,
        "prompt_text": instance.prompt_text,
        "raw_instance_offset": instance.raw_instance_offset,
        "instance_id": instance.instance_id,
        "score": instance.score,
        "raw_mask_sha256": instance.raw_mask_sha256,
        "mask_png_ref": {
            "path": mask_ref.path,
            "bytes": mask_ref.bytes,
            "sha256": mask_ref.sha256,
            "source_kind": mask_ref.source_kind,
        },
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    metadata_payload = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode()
    metadata_ref = _ref(
        root / "wearable_raw" / f"{stem}.json",
        metadata_payload,
        metadata_source_kind,
        wearable.EvidenceRef,
    )
    return unified.BoundWearableRawInstance(instance, metadata_ref, mask_ref)


def _support(own: float) -> d1.JointSupportEvidence:
    supported = int(round(own * 10))
    return d1.JointSupportEvidence(10, supported, supported / 10.0, 1.0, True)


def _arm_raw(
    root: Path,
    session: str,
    frame: int,
    offset: int,
    instance_id: int,
    mask: np.ndarray,
    left: float,
    right: float,
    overlap: float,
) -> d1.RawInstance:
    payload = json.dumps(
        {
            "frame_id": f"{session}:{frame}",
            "instance_id": instance_id,
            "mask_sha256": d1.mask_sha256(mask),
            "source_kind": "SAM_RAW_INSTANCE",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    ref = _ref(
        root / f"frame_{frame:05d}_offset_{offset:03d}.json",
        payload,
        "SAM_RAW_INSTANCE",
    )
    return d1.RawInstance(
        session,
        frame,
        offset,
        instance_id,
        ref,
        mask,
        0.9,
        {"left": _support(left), "right": _support(right)},
        {"left": True, "right": True},
        overlap,
        float(mask.mean()),
    )


def _authority_ref(root: Path, session: str, frame: int):
    payload = {
        "frame_index": frame,
        "sides": {
            "left": {
                "authority": {
                    "identity": "left",
                    "lineage_id": "left-track",
                    "state": "AVAILABLE",
                    "wrist_xy": [3.0, 3.0],
                }
            },
            "right": {
                "authority": {
                    "identity": "right",
                    "lineage_id": "right-track",
                    "state": "AVAILABLE",
                    "wrist_xy": [16.0, 3.0],
                }
            },
        },
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _ref(root / "arm_authority.json", data, "A_PRIME_V3_FRAME_SELECTION")


def _inputs(tmp_path: Path, *, session: str = "grap_a_cap_004", overlap=0.30):
    frame = 7
    arm_left = np.zeros(SHAPE, dtype=np.bool_)
    arm_left[2:8, 2:7] = True
    arm_right = np.zeros(SHAPE, dtype=np.bool_)
    arm_right[2:8, 13:18] = True
    raws = [
        _arm_raw(tmp_path, session, frame, 0, 10, arm_left, 0.5, 0.1, overlap),
        _arm_raw(tmp_path, session, frame, 1, 11, arm_right, 0.1, 0.5, 0.0),
    ]
    authority_ref = _authority_ref(tmp_path, session, frame)
    authorities = {
        "left": d1.SideAuthority(
            "left", session, frame, "left-track", "AVAILABLE", (3.0, 3.0), authority_ref
        ),
        "right": d1.SideAuthority(
            "right",
            session,
            frame,
            "right-track",
            "AVAILABLE",
            (16.0, 3.0),
            authority_ref,
        ),
    }
    points_inside = np.zeros((21, 3), dtype=np.float64)
    points_outside = np.full((21, 3), 5.0, dtype=np.float64)
    geometry = {
        "left": d4.ContactGeometryInput(
            session,
            frame,
            "left",
            0,
            points_inside,
            np.eye(4),
            1.0,
            2.0,
            True,
            True,
            None,
            SHA,
            SHA,
            SHA,
        ),
        "right": d4.ContactGeometryInput(
            session,
            frame,
            "right",
            1,
            points_outside,
            np.eye(4),
            1.0,
            2.0,
            True,
            True,
            None,
            SHA,
            SHA,
            SHA,
        ),
    }
    contract_payload = (
        ROOT / "contracts/neutral_wearable_prompts_v1.json"
    ).read_bytes()
    contract_ref = _ref(
        tmp_path / "prompt_contract.json",
        contract_payload,
        "FROZEN_NEUTRAL_WEARABLE_PROMPT_CONTRACT",
        wearable.EvidenceRef,
    )
    optimized_ref = _ref(
        tmp_path / "hawor_optimized.npz",
        b"optimized-frozen-bytes",
        "HAWOR_OPTIMIZED_JOINTS_NPZ",
        wearable.EvidenceRef,
    )
    raw_hawor_ref = _ref(
        tmp_path / "hawor_raw.npz",
        b"raw-hawor-frozen-bytes",
        "HAWOR_RAW_PROJECTION_NPZ",
        wearable.EvidenceRef,
    )
    wrists = {
        "left": wearable.WristAuthority(
            "left",
            frame,
            (7.0, 3.0),
            8.0,
            "wear-left",
            0,
            0,
            optimized_ref,
            raw_hawor_ref,
        ),
        "right": wearable.WristAuthority(
            "right",
            frame,
            (16.0, 3.0),
            8.0,
            "wear-right",
            1,
            1,
            optimized_ref,
            raw_hawor_ref,
        ),
    }
    wearable_mask = np.zeros(SHAPE, dtype=np.bool_)
    wearable_mask[2:5, 7:9] = True
    instance = wearable.RawWearableInstance(
        "W1",
        wearable.FROZEN_PROMPTS["W1"],
        contract_ref,
        frame,
        20,
        120,
        0.9,
        wearable_mask,
        wearable.mask_sha256(wearable_mask),
    )
    batches = {prompt: [] for prompt in wearable.FROZEN_PROMPTS}
    batches["W1"] = [_bound_wearable(tmp_path, session, instance)]
    frame_input = unified.FrozenWearableFrame(
        session, frame, contract_ref, wrists, batches
    )
    return authorities, raws, geometry, frame_input, arm_left, arm_right, wearable_mask


def _wearable_instance(
    frame: unified.FrozenWearableFrame,
    prompt_id: str,
    mask: np.ndarray,
    *,
    offset: int,
    instance_id: int,
    score: float = 0.9,
    prompt_contract_ref: wearable.EvidenceRef | None = None,
) -> wearable.RawWearableInstance:
    return wearable.RawWearableInstance(
        prompt_id,
        wearable.FROZEN_PROMPTS[prompt_id],
        prompt_contract_ref or frame.prompt_contract_ref,
        frame.frame_index,
        offset,
        instance_id,
        score,
        mask,
        wearable.mask_sha256(mask),
    )


def _with_prompt_instances(
    frame: unified.FrozenWearableFrame,
    **instances_by_prompt: list[wearable.RawWearableInstance],
) -> unified.FrozenWearableFrame:
    batches = {prompt_id: [] for prompt_id in wearable.FROZEN_PROMPTS}
    root = Path(frame.prompt_contract_ref.path).parent
    batches.update(
        {
            prompt_id: [
                _bound_wearable(root, frame.session_id, instance)
                for instance in instances
            ]
            for prompt_id, instances in instances_by_prompt.items()
        }
    )
    return replace(frame, instances_by_prompt=batches)


def _bound_prompt_mapping(
    frame: unified.FrozenWearableFrame,
    instances_by_prompt: dict[str, list[wearable.RawWearableInstance]],
) -> dict[str, list[unified.BoundWearableRawInstance]]:
    root = Path(frame.prompt_contract_ref.path).parent
    return {
        prompt_id: [
            _bound_wearable(root, frame.session_id, instance) for instance in instances
        ]
        for prompt_id, instances in instances_by_prompt.items()
    }


def test_d4_flip_then_wearable_union_and_exactly_one_d1_call(monkeypatch, tmp_path):
    authorities, raws, geometry, frame, arm_left, arm_right, wearable_mask = _inputs(
        tmp_path
    )
    original = d4.base.select_frame
    calls = 0

    def once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            pytest.fail("D1 was called more than once")
        return original(*args, **kwargs)

    monkeypatch.setattr(d4.base, "select_frame", once)
    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )
    assert calls == 1
    assert result.execution_order == ("D1", "D4", "WEARABLE_UNION")
    assert result.contact.base_selection.left.status == "REJECT"
    assert result.contact.selection.left.status == "ACCEPT"
    assert result.contact.application_audits["left"].status == (
        "APPLIED_OBJECT_OVERLAP_AS_CONTACT_EVIDENCE"
    )
    assert result.sides["left"].arm_identity.session_id == "grap_a_cap_004"
    assert np.array_equal(result.sides["left"].final_mask, arm_left | wearable_mask)
    assert result.sides["left"].added_pixels == int(
        np.count_nonzero(wearable_mask & ~arm_left)
    )
    assert np.array_equal(result.sides["right"].final_mask, arm_right)
    assert tuple(result.wearable_decisions) == unified.FROZEN_WEARABLE_PROMPT_ORDER
    assert result.wearable_prompt_order == ("W1", "W2", "W3", "W4")
    assert result.wearable_combination_rule == (
        "ALL_INDEPENDENT_ACCEPTS_RAW_BOOLEAN_OR_V1"
    )
    assert result.prompt_contract_ref == frame.prompt_contract_ref
    assert result.optimized_hawor_source_ref == (
        frame.wrist_authorities["left"].optimized_source_ref
    )
    assert result.raw_hawor_source_ref == (
        frame.wrist_authorities["left"].raw_hawor_source_ref
    )
    selected = result.sides["left"].selected_wearables
    assert [(item.prompt_id, item.prompt_text) for item in selected] == [
        ("W1", wearable.FROZEN_PROMPTS["W1"])
    ]
    assert selected[0].prompt_contract_sha256 == frame.prompt_contract_ref.sha256
    assert selected[0].raw_mask_sha256 == wearable.mask_sha256(wearable_mask)
    assert selected[0].metadata_ref.source_kind == (
        unified.WEARABLE_METADATA_SOURCE_KIND
    )
    assert selected[0].mask_png_ref.source_kind == unified.WEARABLE_MASK_SOURCE_KIND
    assert result.sides["left"].final_mask_sha256 == d1.mask_sha256(
        result.sides["left"].final_mask
    )
    assert result.semantic_change_class == ("H5(b)_PIXEL_SEMANTIC_CRITERION_CHANGE")
    assert result.model_calls == 0
    assert result.fill_operations == 0
    assert result.morphology_operations == 0
    assert result.crop_operations == 0
    assert result.temporal_propagation_operations == 0
    assert result.fallback_operations == 0


def test_no_accepted_wearable_preserves_exact_accepted_arm(tmp_path):
    authorities, raws, geometry, frame, arm_left, arm_right, _ = _inputs(
        tmp_path, overlap=0.0
    )
    frame = _with_prompt_instances(frame)
    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )
    assert np.array_equal(result.sides["left"].final_mask, arm_left)
    assert np.array_equal(result.sides["right"].final_mask, arm_right)
    assert result.sides["left"].selected_wearables == ()
    assert result.sides["left"].added_pixels == 0


def test_multiple_accepted_prompts_containment_is_pure_or(tmp_path):
    authorities, raws, geometry, frame, arm_left, *_ = _inputs(tmp_path)
    outer = np.zeros(SHAPE, dtype=np.bool_)
    outer[2:6, 7:11] = True
    inner = np.zeros(SHAPE, dtype=np.bool_)
    inner[3:5, 7:9] = True
    assert np.all(outer[inner])
    first = _wearable_instance(frame, "W1", outer, offset=20, instance_id=120)
    second = _wearable_instance(frame, "W2", inner, offset=21, instance_id=121)
    # Input mapping order is deliberately different from frozen W1..W4 order.
    frame = replace(
        frame,
        instances_by_prompt=_bound_prompt_mapping(
            frame,
            {"W4": [], "W2": [second], "W1": [first], "W3": []},
        ),
    )

    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )

    left = result.sides["left"]
    assert np.array_equal(left.final_mask, arm_left | outer | inner)
    assert tuple(item.prompt_id for item in left.selected_wearables) == ("W1", "W2")
    assert left.union_operations == 2
    assert left.added_pixels == int(np.count_nonzero((outer | inner) & ~arm_left))


def test_all_w1_w4_accepts_are_retained_in_frozen_order(tmp_path):
    authorities, raws, geometry, frame, arm_left, *_ = _inputs(tmp_path)
    masks = []
    for extension in range(4):
        mask = np.zeros(SHAPE, dtype=np.bool_)
        mask[3, 7 : 8 + extension] = True
        masks.append(mask)
    instances = {
        prompt_id: [
            _wearable_instance(
                frame,
                prompt_id,
                masks[index],
                offset=20 + index,
                instance_id=120 + index,
            )
        ]
        for index, prompt_id in enumerate(unified.FROZEN_WEARABLE_PROMPT_ORDER)
    }
    frame = replace(
        frame,
        instances_by_prompt=_bound_prompt_mapping(
            frame,
            {
                "W4": instances["W4"],
                "W3": instances["W3"],
                "W2": instances["W2"],
                "W1": instances["W1"],
            },
        ),
    )

    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )

    left = result.sides["left"]
    expected = np.logical_or.reduce([arm_left, *masks])
    assert np.array_equal(left.final_mask, expected)
    assert tuple(item.prompt_id for item in left.selected_wearables) == (
        "W1",
        "W2",
        "W3",
        "W4",
    )
    assert tuple(item.raw_instance_offset for item in left.selected_wearables) == (
        20,
        21,
        22,
        23,
    )
    assert all(
        item.prompt_contract_sha256 == frame.prompt_contract_ref.sha256
        for item in left.selected_wearables
    )
    assert left.union_operations == 4


def test_multiple_accepted_prompts_partial_overlap_is_pure_or(tmp_path):
    authorities, raws, geometry, frame, arm_left, *_ = _inputs(tmp_path)
    first_mask = np.zeros(SHAPE, dtype=np.bool_)
    first_mask[2:5, 7:9] = True
    second_mask = np.zeros(SHAPE, dtype=np.bool_)
    second_mask[3:7, 7:11] = True
    assert np.logical_and(first_mask, second_mask).any()
    assert np.logical_and(first_mask, ~second_mask).any()
    assert np.logical_and(second_mask, ~first_mask).any()
    frame = _with_prompt_instances(
        frame,
        W1=[_wearable_instance(frame, "W1", first_mask, offset=20, instance_id=120)],
        W2=[_wearable_instance(frame, "W2", second_mask, offset=21, instance_id=121)],
    )

    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )

    expected = arm_left | first_mask | second_mask
    assert np.array_equal(result.sides["left"].final_mask, expected)
    assert result.sides["left"].added_pixels == int(
        np.count_nonzero(expected & ~arm_left)
    )


def test_multiple_accepted_prompts_disjoint_masks_are_supported(tmp_path):
    authorities, raws, geometry, frame, arm_left, *_ = _inputs(tmp_path)
    left_wrist = replace(frame.wrist_authorities["left"], hand_scale_px=20.0)
    frame = replace(
        frame,
        wrist_authorities={
            "left": left_wrist,
            "right": frame.wrist_authorities["right"],
        },
    )
    first_mask = np.zeros(SHAPE, dtype=np.bool_)
    first_mask[3, 5] = True
    second_mask = np.zeros(SHAPE, dtype=np.bool_)
    second_mask[3, 9] = True
    assert not np.logical_and(first_mask, second_mask).any()
    frame = _with_prompt_instances(
        frame,
        W1=[_wearable_instance(frame, "W1", first_mask, offset=20, instance_id=120)],
        W2=[_wearable_instance(frame, "W2", second_mask, offset=21, instance_id=121)],
    )

    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )

    assert np.array_equal(
        result.sides["left"].final_mask, arm_left | first_mask | second_mask
    )
    assert tuple(
        item.raw_mask_sha256 for item in result.sides["left"].selected_wearables
    ) == (wearable.mask_sha256(first_mask), wearable.mask_sha256(second_mask))


def test_duplicate_accepted_masks_are_idempotent_but_both_identities_remain(tmp_path):
    authorities, raws, geometry, frame, arm_left, *_ = _inputs(tmp_path)
    first_mask = np.zeros(SHAPE, dtype=np.bool_)
    first_mask[2:5, 7:10] = True
    second_mask = first_mask.copy()
    assert not np.shares_memory(first_mask, second_mask)
    frame = _with_prompt_instances(
        frame,
        W1=[_wearable_instance(frame, "W1", first_mask, offset=20, instance_id=120)],
        W2=[_wearable_instance(frame, "W2", second_mask, offset=21, instance_id=121)],
    )

    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )

    left = result.sides["left"]
    assert np.array_equal(left.final_mask, arm_left | first_mask)
    assert tuple(item.prompt_id for item in left.selected_wearables) == ("W1", "W2")
    assert left.union_operations == 2
    assert left.added_pixels == int(np.count_nonzero(first_mask & ~arm_left))


def test_rejected_prompt_contributes_zero_pixels_and_keeps_its_audit(tmp_path):
    authorities, raws, geometry, frame, arm_left, _, accepted_mask = _inputs(tmp_path)
    rejected_mask = np.zeros(SHAPE, dtype=np.bool_)
    rejected_mask[3, 7] = True
    rejected_mask[8:10, 10:12] = True
    frame = _with_prompt_instances(
        frame,
        W1=[
            _wearable_instance(
                frame, "W1", accepted_mask.copy(), offset=20, instance_id=120
            )
        ],
        W2=[
            _wearable_instance(
                frame,
                "W2",
                rejected_mask,
                offset=21,
                instance_id=121,
                score=0.49,
            )
        ],
    )

    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )

    left = result.sides["left"]
    assert np.array_equal(left.final_mask, arm_left | accepted_mask)
    assert tuple(item.prompt_id for item in left.selected_wearables) == ("W1",)
    rejected = result.wearable_decisions["W2"]["left"]
    assert rejected.status == "HOLD"
    assert rejected.audits[0].raw_mask_sha256 == wearable.mask_sha256(rejected_mask)
    assert rejected.audits[0].rejection_reasons == (
        "MODEL_SCORE_BELOW_FROZEN_THRESHOLD",
    )


def test_arm_hold_remains_hold_and_wearable_cannot_create_h(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    authorities = dict(authorities)
    authorities["right"] = replace(
        authorities["right"], lineage_id=authorities["left"].lineage_id
    )

    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )

    assert result.wearable_decisions["W1"]["left"].status == "ACCEPT"
    for side in d1.SIDES:
        final = result.sides[side]
        assert final.arm_status == "HOLD"
        assert final.arm_identity is None
        assert final.selected_wearables == ()
        assert final.final_mask is None
        assert final.added_pixels == 0
        assert final.union_operations == 0


def test_cross_session_wearable_inventory_is_rejected_before_d1(monkeypatch, tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    frame = replace(frame, session_id="grap_a_cap_012")
    monkeypatch.setattr(d4.base, "select_frame", lambda *a, **k: pytest.fail("D1 ran"))
    with pytest.raises(unified.UnifiedSelectorError, match="cross-session"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            frame,
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_hawor_reference_alias_is_rejected(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    left = frame.wrist_authorities["left"]
    alias = wearable.EvidenceRef(
        left.optimized_source_ref.path,
        left.optimized_source_ref.bytes,
        left.optimized_source_ref.sha256,
        "HAWOR_RAW_PROJECTION_NPZ",
    )
    wrists = {
        side: replace(authority, raw_hawor_source_ref=alias)
        for side, authority in frame.wrist_authorities.items()
    }
    with pytest.raises(unified.UnifiedSelectorError, match="reference alias"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, wrist_authorities=wrists),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_arm_wearable_mask_alias_is_rejected(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    aliased = wearable.RawWearableInstance(
        "W1",
        wearable.FROZEN_PROMPTS["W1"],
        frame.prompt_contract_ref,
        frame.frame_index,
        21,
        121,
        0.9,
        raws[0].mask,
        wearable.mask_sha256(raws[0].mask),
    )
    batches = dict(frame.instances_by_prompt)
    batches["W1"] = [_bound_wearable(tmp_path, frame.session_id, aliased)]
    with pytest.raises(unified.UnifiedSelectorError, match="mask alias"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_wearable_prompt_contract_provenance_drift_is_rejected_before_d1(
    monkeypatch, tmp_path
):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    original = frame.instances_by_prompt["W1"][0].instance
    drifted_ref = replace(original.prompt_contract_ref, sha256="0" * 64)
    drifted = replace(original, prompt_contract_ref=drifted_ref)
    frame = _with_prompt_instances(frame, W1=[drifted])
    monkeypatch.setattr(d4.base, "select_frame", lambda *a, **k: pytest.fail("D1 ran"))

    with pytest.raises(unified.UnifiedSelectorError, match="prompt reference mismatch"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            frame,
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_wearable_raw_mask_sha_drift_is_rejected(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    original = frame.instances_by_prompt["W1"][0].instance
    drifted = replace(original, raw_mask_sha256="0" * 64)
    frame = _with_prompt_instances(frame, W1=[drifted])

    with pytest.raises(wearable.WearableIdentityError, match="raw wearable mask SHA"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            frame,
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_forbidden_025_is_rejected_before_any_evidence_read(monkeypatch, tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path, session="grap_a_cap_025")
    monkeypatch.setattr(d4.base, "select_frame", lambda *a, **k: pytest.fail("D1 ran"))
    with pytest.raises(unified.UnifiedSelectorError, match="forbidden"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            frame,
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )

    for alias in ("025", "grap_a_cap_025/", "grap_a_cap_004/../grap_a_cap_025"):
        authorities, raws, geometry, frame, *_ = _inputs(
            tmp_path / alias.replace("/", "_"), session=alias
        )
        with pytest.raises(unified.UnifiedSelectorError, match="invalid session"):
            unified.select_frame(
                authorities,
                raws,
                geometry,
                frame,
                evidence_root=tmp_path / alias.replace("/", "_"),
                image_shape=SHAPE,
            )


def test_threshold_drift_is_rejected(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    with pytest.raises(unified.UnifiedSelectorError, match="thresholds changed"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            frame,
            evidence_root=tmp_path,
            image_shape=SHAPE,
            thresholds=d1.SelectorThresholds(0.20, 0.05, 0.12001),
        )


def test_exact_w1_w4_invocation_order_once_and_after_d4(monkeypatch, tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    base_mask = frame.instances_by_prompt["W1"][0].instance.mask
    instances = {
        prompt_id: [
            _wearable_instance(
                frame,
                prompt_id,
                base_mask.copy(),
                offset=index,
                instance_id=200 + index,
            )
        ]
        for index, prompt_id in enumerate(unified.FROZEN_WEARABLE_PROMPT_ORDER)
    }
    frame = replace(frame, instances_by_prompt=_bound_prompt_mapping(frame, instances))
    events: list[str] = []
    original_d4 = d4.select_frame
    original_wearable = wearable.select_wearable_per_side

    def tracked_d4(*args, **kwargs):
        events.append("D4")
        return original_d4(*args, **kwargs)

    def tracked_wearable(instances_arg, *args, **kwargs):
        prompt_ids = {instance.prompt_id for instance in instances_arg}
        assert len(prompt_ids) == 1
        events.append(next(iter(prompt_ids)))
        return original_wearable(instances_arg, *args, **kwargs)

    monkeypatch.setattr(d4, "select_frame", tracked_d4)
    monkeypatch.setattr(wearable, "select_wearable_per_side", tracked_wearable)
    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )

    assert events == ["D4", "W1", "W2", "W3", "W4"]
    assert result.wearable_prompt_invocation_counts == (
        ("W1", 1),
        ("W2", 1),
        ("W3", 1),
        ("W4", 1),
    )


@pytest.mark.parametrize("missing", unified.FROZEN_WEARABLE_PROMPT_ORDER)
def test_prompt_key_set_must_be_exact_before_d1(monkeypatch, tmp_path, missing):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    batches = dict(frame.instances_by_prompt)
    del batches[missing]
    monkeypatch.setattr(d4.base, "select_frame", lambda *a, **k: pytest.fail("D1 ran"))

    with pytest.raises(unified.UnifiedSelectorError, match="exact frozen prompt"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_extra_prompt_key_is_rejected_before_d1(monkeypatch, tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    batches = dict(frame.instances_by_prompt)
    batches["W5"] = []
    monkeypatch.setattr(d4.base, "select_frame", lambda *a, **k: pytest.fail("D1 ran"))

    with pytest.raises(unified.UnifiedSelectorError, match="exact frozen prompt"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("raw_instance_offset", -1),
        ("raw_instance_offset", True),
        ("instance_id", -1),
        ("instance_id", False),
    ),
)
def test_wearable_offset_and_id_require_nonnegative_exact_int(tmp_path, field, value):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    original = frame.instances_by_prompt["W1"][0].instance
    invalid = replace(original, **{field: value})
    frame = _with_prompt_instances(frame, W1=[invalid])

    with pytest.raises(unified.UnifiedSelectorError, match="numeric identity"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            frame,
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


@pytest.mark.parametrize(
    ("which_ref", "source_kind", "message"),
    (
        ("metadata_ref", "DERIVED_RENDER_JSON", "metadata source kind"),
        ("mask_png_ref", "RENDERED_OVERLAY_PNG", "mask PNG source kind"),
    ),
)
def test_wearable_raw_source_kinds_are_exact(tmp_path, which_ref, source_kind, message):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    bound = frame.instances_by_prompt["W1"][0]
    bad_ref = replace(getattr(bound, which_ref), source_kind=source_kind)
    bad_bound = replace(bound, **{which_ref: bad_ref})
    batches = dict(frame.instances_by_prompt)
    batches["W1"] = [bad_bound]

    with pytest.raises(unified.UnifiedSelectorError, match=message):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


@pytest.mark.parametrize(
    ("metadata_overrides", "message"),
    (
        ({"session_id": "grap_a_cap_012"}, "metadata/instance payload"),
        ({"frame_index": 8}, "metadata/instance payload"),
        ({"prompt_text": "a rendered white blob"}, "metadata/instance payload"),
        ({"score": 0.75}, "metadata/instance payload"),
        ({"raw_mask_sha256": "0" * 64}, "metadata/instance payload"),
    ),
)
def test_wearable_metadata_replay_or_payload_drift_is_rejected(
    tmp_path, metadata_overrides, message
):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    instance = frame.instances_by_prompt["W1"][0].instance
    bad_bound = _bound_wearable(
        tmp_path,
        frame.session_id,
        instance,
        metadata_overrides=metadata_overrides,
    )
    batches = dict(frame.instances_by_prompt)
    batches["W1"] = [bad_bound]

    with pytest.raises(unified.UnifiedSelectorError, match=message):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_cross_session_mask_png_ref_replay_is_rejected(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    original = frame.instances_by_prompt["W1"][0]
    replay = _bound_wearable(
        tmp_path / "other_session",
        "grap_a_cap_012",
        original.instance,
    )
    # Keep 004 metadata bytes but replace only its bound PNG reference.
    bad_bound = replace(original, mask_png_ref=replay.mask_png_ref)
    batches = dict(frame.instances_by_prompt)
    batches["W1"] = [bad_bound]

    with pytest.raises(unified.UnifiedSelectorError, match="metadata/instance payload"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_png_pixels_must_equal_bound_raw_array(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    instance = frame.instances_by_prompt["W1"][0].instance
    different = instance.mask.copy()
    different[0, 0] = True
    bad_bound = _bound_wearable(
        tmp_path,
        frame.session_id,
        instance,
        png_mask=different,
    )
    batches = dict(frame.instances_by_prompt)
    batches["W1"] = [bad_bound]

    with pytest.raises(unified.UnifiedSelectorError, match="PNG/instance mask"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_unbound_caller_claimed_raw_mask_is_rejected(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    batches = dict(frame.instances_by_prompt)
    batches["W1"] = [frame.instances_by_prompt["W1"][0].instance]

    with pytest.raises(unified.UnifiedSelectorError, match="unbound wearable"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


@pytest.mark.parametrize("failure_mode", ("malformed", "nonbinary"))
def test_mask_png_must_fully_decode_as_exact_binary(tmp_path, failure_mode):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    bound = frame.instances_by_prompt["W1"][0]
    if failure_mode == "malformed":
        payload = b"not-a-png"
        message = "full decode"
    else:
        stream = BytesIO()
        pixels = np.zeros(SHAPE, dtype=np.uint8)
        pixels[0, 0] = 1
        Image.fromarray(pixels).save(stream, format="PNG")
        payload = stream.getvalue()
        message = "not exactly binary"
    mask_ref = _ref(
        tmp_path / f"{failure_mode}.png",
        payload,
        unified.WEARABLE_MASK_SOURCE_KIND,
        wearable.EvidenceRef,
    )
    batches = dict(frame.instances_by_prompt)
    batches["W1"] = [replace(bound, mask_png_ref=mask_ref)]

    with pytest.raises(unified.UnifiedSelectorError, match=message):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


@pytest.mark.parametrize("mode", ("dtype", "shape"))
def test_raw_wearable_array_dtype_and_shape_are_exact(tmp_path, mode):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    original = frame.instances_by_prompt["W1"][0].instance
    if mode == "dtype":
        mask = original.mask.astype(np.uint8)
    else:
        mask = np.zeros((SHAPE[0] - 1, SHAPE[1]), dtype=np.bool_)
    invalid = replace(
        original,
        mask=mask,
        raw_mask_sha256=wearable.mask_sha256(mask),
    )
    frame = _with_prompt_instances(frame, W1=[invalid])

    with pytest.raises(wearable.WearableIdentityError, match="bool HxW"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            frame,
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )


def test_result_masks_and_provenance_mappings_are_deeply_immutable(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    input_wearable = frame.instances_by_prompt["W1"][0].instance.mask
    result = unified.select_frame(
        authorities, raws, geometry, frame, evidence_root=tmp_path, image_shape=SHAPE
    )
    final = result.sides["left"].final_mask
    contact_mask = result.contact.selection.left.mask
    assert final is not None and contact_mask is not None
    final_before = final.copy()
    contact_before = contact_mask.copy()

    with pytest.raises(ValueError):
        final[0, 0] = ~final[0, 0]
    with pytest.raises(ValueError):
        final.setflags(write=True)
    with pytest.raises(ValueError):
        contact_mask[0, 0] = ~contact_mask[0, 0]
    with pytest.raises(ValueError):
        contact_mask.setflags(write=True)
    with pytest.raises(TypeError):
        result.sides["left"] = result.sides["right"]
    with pytest.raises(TypeError):
        result.wearable_decisions["W1"] = result.wearable_decisions["W2"]
    with pytest.raises(TypeError):
        result.wearable_decisions["W1"]["left"] = result.wearable_decisions["W1"][
            "right"
        ]
    with pytest.raises(TypeError):
        result.contact.contact_measurements["left"] = (
            result.contact.contact_measurements["right"]
        )
    with pytest.raises(TypeError):
        result.contact.application_audits["left"] = result.contact.application_audits[
            "right"
        ]
    with pytest.raises(TypeError):
        result.wrist_authorities["left"] = result.wrist_authorities["right"]

    # Post-return mutation of caller-owned arrays cannot change the result.
    input_wearable[:] = False
    raws[0].mask[:] = False
    assert np.array_equal(final, final_before)
    assert np.array_equal(contact_mask, contact_before)
    assert result.sides["left"].final_mask_sha256 == d1.mask_sha256(final)


def test_extra_arm_or_geometry_key_is_not_silently_discarded(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    for mapping_name in ("authorities", "geometry"):
        bad_authorities = dict(authorities)
        bad_geometry = dict(geometry)
        if mapping_name == "authorities":
            bad_authorities["extra"] = authorities["left"]
        else:
            bad_geometry["extra"] = geometry["left"]
        with pytest.raises(unified.UnifiedSelectorError, match="exact left/right"):
            unified.select_frame(
                bad_authorities,
                raws,
                bad_geometry,
                frame,
                evidence_root=tmp_path,
                image_shape=SHAPE,
            )


def test_wearable_mask_evidence_sha_drift_is_rejected(tmp_path):
    authorities, raws, geometry, frame, *_ = _inputs(tmp_path)
    bound = frame.instances_by_prompt["W1"][0]
    bad_bound = replace(
        bound,
        mask_png_ref=replace(bound.mask_png_ref, sha256="0" * 64),
    )
    batches = dict(frame.instances_by_prompt)
    batches["W1"] = [bad_bound]

    with pytest.raises(wearable.WearableIdentityError, match="bytes/SHA mismatch"):
        unified.select_frame(
            authorities,
            raws,
            geometry,
            replace(frame, instances_by_prompt=batches),
            evidence_root=tmp_path,
            image_shape=SHAPE,
        )
